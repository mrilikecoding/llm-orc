"""Main CLI command implementations."""

import asyncio
import shutil
import sys
from pathlib import Path

import click
import yaml

from llm_orc.authentication import AuthenticationManager, CredentialStorage
from llm_orc.cli_auth import (
    handle_anthropic_interactive_auth,
    handle_claude_cli_auth,
    handle_claude_pro_max_oauth,
    show_auth_method_help,
    test_provider_authentication,
)
from llm_orc.cli_config import (
    check_ensemble_availability,
    display_default_models_config,
    display_global_profiles,
    display_local_profiles,
    display_providers_status,
    get_available_providers,
    show_provider_details,
)
from llm_orc.cli_visualization import (
    run_standard_execution,
    run_streaming_execution,
)
from llm_orc.config import ConfigurationManager
from llm_orc.ensemble_config import EnsembleLoader
from llm_orc.ensemble_execution import EnsembleExecutor
from llm_orc.mcp_server_runner import MCPServerRunner


def invoke_ensemble(
    ensemble_name: str,
    input_data: str | None,
    config_dir: str | None,
    input_data_option: str | None,
    output_format: str,
    streaming: bool,
    max_concurrent: int | None,
    detailed: bool,
) -> None:
    """Invoke an ensemble of agents."""
    # Initialize configuration manager
    config_manager = ConfigurationManager()

    # Determine ensemble directories
    if config_dir is None:
        # Use configuration manager to get ensemble directories
        ensemble_dirs = config_manager.get_ensembles_dirs()
        if not ensemble_dirs:
            raise click.ClickException(
                "No ensemble directories found. Run 'llm-orc config init' to set up "
                "local configuration."
            )
    else:
        # Use specified config directory
        ensemble_dirs = [Path(config_dir)]

    # Handle input data priority: positional > option > stdin > default
    final_input_data = input_data or input_data_option

    if final_input_data is None:
        if not sys.stdin.isatty():
            # Read from stdin (piped input)
            final_input_data = sys.stdin.read().strip()
        else:
            # No input provided and not piped, use default
            final_input_data = "Please analyze this."

    input_data = final_input_data

    # Find ensemble in the directories
    loader = EnsembleLoader()
    ensemble_config = None

    for ensemble_dir in ensemble_dirs:
        ensemble_config = loader.find_ensemble(str(ensemble_dir), ensemble_name)
        if ensemble_config is not None:
            break

    if ensemble_config is None:
        searched_dirs = [str(d) for d in ensemble_dirs]
        raise click.ClickException(
            f"Ensemble '{ensemble_name}' not found in: {', '.join(searched_dirs)}"
        )

    # Create standard executor
    executor = EnsembleExecutor()

    # Override concurrency settings if provided
    if max_concurrent is not None:
        # Apply concurrency limit to executor configuration
        pass  # This would be implemented as needed

    # Show performance configuration for text output
    if output_format == "text":
        try:
            performance_config = config_manager.load_performance_config()
            effective_concurrency = executor._get_effective_concurrency_limit(
                len(ensemble_config.agents)
            )
            # Determine effective streaming setting (CLI flag overrides config)
            effective_streaming = streaming or performance_config.get(
                "streaming_enabled", False
            )
            click.echo(
                f"🚀 Executing ensemble '{ensemble_name}' with "
                f"{len(ensemble_config.agents)} agents"
            )
            click.echo(
                f"⚡ Performance: max_concurrent={effective_concurrency}, "
                f"streaming={effective_streaming}"
            )
            click.echo("─" * 50)
        except Exception:
            # Fallback to original output if performance config fails
            click.echo(f"Invoking ensemble: {ensemble_name}")
            click.echo(f"Description: {ensemble_config.description}")
            click.echo(f"Agents: {len(ensemble_config.agents)}")
            click.echo(f"Input: {input_data}")
            click.echo("---")

    # Determine effective streaming setting
    performance_config = config_manager.load_performance_config()
    effective_streaming = streaming or performance_config.get(
        "streaming_enabled", False
    )

    # Execute the ensemble
    try:
        if effective_streaming:
            # Streaming execution with Rich status
            asyncio.run(
                run_streaming_execution(
                    executor, ensemble_config, input_data, output_format, detailed
                )
            )
        else:
            # Standard execution
            asyncio.run(
                run_standard_execution(
                    executor, ensemble_config, input_data, output_format, detailed
                )
            )

    except Exception as e:
        raise click.ClickException(f"Ensemble execution failed: {str(e)}") from e


def list_ensembles_command(config_dir: str | None) -> None:
    """List available ensembles."""
    # Initialize configuration manager
    config_manager = ConfigurationManager()

    if config_dir is None:
        # Use configuration manager to get ensemble directories
        ensemble_dirs = config_manager.get_ensembles_dirs()
        if not ensemble_dirs:
            click.echo("No ensemble directories found.")
            click.echo("Run 'llm-orc config init' to set up local configuration.")
            return

        # List ensembles from all directories, grouped by location
        loader = EnsembleLoader()
        local_ensembles = []
        global_ensembles = []

        for dir_path in ensemble_dirs:
            ensembles = loader.list_ensembles(str(dir_path))
            is_local = config_manager.local_config_dir and str(dir_path).startswith(
                str(config_manager.local_config_dir)
            )

            if is_local:
                local_ensembles.extend(ensembles)
            else:
                global_ensembles.extend(ensembles)

        # Check if we have any ensembles at all
        if not local_ensembles and not global_ensembles:
            click.echo("No ensembles found in any configured directories:")
            for dir_path in ensemble_dirs:
                click.echo(f"  {dir_path}")
            click.echo("  (Create .yaml files with ensemble configurations)")
            return

        click.echo("Available ensembles:")

        # Show local ensembles first
        if local_ensembles:
            click.echo("\n📁 Local Repo (.llm-orc/ensembles):")
            for ensemble in sorted(local_ensembles, key=lambda e: e.name):
                click.echo(f"  {ensemble.name}: {ensemble.description}")

        # Show global ensembles
        if global_ensembles:
            global_config_label = (
                f"Global ({config_manager.global_config_dir}/ensembles)"
            )
            click.echo(f"\n🌐 {global_config_label}:")
            for ensemble in sorted(global_ensembles, key=lambda e: e.name):
                click.echo(f"  {ensemble.name}: {ensemble.description}")
    else:
        # Use specified config directory
        loader = EnsembleLoader()
        ensembles = loader.list_ensembles(config_dir)

        if not ensembles:
            click.echo(f"No ensembles found in {config_dir}")
            click.echo("  (Create .yaml files with ensemble configurations)")
        else:
            click.echo(f"Available ensembles in {config_dir}:")
            for ensemble in ensembles:
                click.echo(f"  {ensemble.name}: {ensemble.description}")


def list_profiles_command() -> None:
    """List available model profiles with their provider/model details."""
    # Initialize configuration manager
    config_manager = ConfigurationManager()

    # Get all model profiles (merged global + local)
    all_profiles = config_manager.get_model_profiles()

    if not all_profiles:
        click.echo("No model profiles found.")
        click.echo("Run 'llm-orc config init' to create default profiles.")
        return

    # Get separate global and local profiles for grouping
    global_profiles = {}
    global_config_file = config_manager.global_config_dir / "config.yaml"
    if global_config_file.exists():
        with open(global_config_file) as f:
            global_config = yaml.safe_load(f) or {}
            global_profiles = global_config.get("model_profiles", {})

    local_profiles = {}
    if config_manager.local_config_dir:
        local_config_file = config_manager.local_config_dir / "config.yaml"
        if local_config_file.exists():
            with open(local_config_file) as f:
                local_config = yaml.safe_load(f) or {}
                local_profiles = local_config.get("model_profiles", {})

    click.echo("Available model profiles:")

    # Get available providers for status indicators
    available_providers = get_available_providers(config_manager)

    # Show local profiles first (if any)
    if local_profiles:
        display_local_profiles(local_profiles, available_providers)

    # Show global profiles
    if global_profiles:
        global_config_label = f"Global ({config_manager.global_config_dir}/config.yaml)"
        click.echo(f"\n🌐 {global_config_label}:")
        for profile_name in sorted(global_profiles.keys()):
            # Skip if this profile is overridden by local
            if profile_name in local_profiles:
                click.echo(f"  {profile_name}: (overridden by local)")
                continue

            profile = global_profiles[profile_name]

            # Handle case where profile is not a dict (malformed YAML)
            if not isinstance(profile, dict):
                click.echo(
                    f"  {profile_name}: [Invalid profile format - "
                    f"expected dict, got {type(profile).__name__}]"
                )
                continue

            model = profile.get("model", "Unknown")
            provider = profile.get("provider", "Unknown")
            cost = profile.get("cost_per_token", "Not specified")

            click.echo(f"  {profile_name}:")
            click.echo(f"    Model: {model}")
            click.echo(f"    Provider: {provider}")
            click.echo(f"    Cost per token: {cost}")


def init_local_config(project_name: str | None) -> None:
    """Initialize local .llm-orc configuration for current project."""
    config_manager = ConfigurationManager()

    try:
        config_manager.init_local_config(project_name)
        click.echo("Local configuration initialized successfully!")
        click.echo("Created .llm-orc directory with:")
        click.echo("  - ensembles/   (project-specific ensembles)")
        click.echo("  - models/      (shared model configurations)")
        click.echo("  - scripts/     (project-specific scripts)")
        click.echo("  - config.yaml  (project configuration)")
        click.echo(
            "\nYou can now create project-specific ensembles in .llm-orc/ensembles/"
        )
    except ValueError as e:
        raise click.ClickException(str(e)) from e


def reset_global_config(backup: bool, preserve_auth: bool) -> None:
    """Reset global configuration to template defaults."""
    config_manager = ConfigurationManager()
    global_config_dir = Path(config_manager.global_config_dir)

    # Create backup if requested and config exists
    if backup and global_config_dir.exists():
        backup_path = global_config_dir.with_suffix(".backup")
        if backup_path.exists():
            shutil.rmtree(backup_path)
        shutil.copytree(global_config_dir, backup_path)
        click.echo(f"📦 Backed up existing config to {backup_path}")

    # Preserve authentication files if requested
    auth_files = []
    if preserve_auth and global_config_dir.exists():
        potential_auth_files = [
            "credentials.yaml",
            ".encryption_key",
            ".credentials.yaml",  # legacy
        ]
        for auth_file in potential_auth_files:
            auth_path = global_config_dir / auth_file
            if auth_path.exists():
                # Save auth file content
                auth_files.append((auth_file, auth_path.read_bytes()))
                click.echo(f"🔐 Preserving authentication file: {auth_file}")

    # Remove existing config directory
    if global_config_dir.exists():
        shutil.rmtree(global_config_dir)

    # Create fresh config directory
    global_config_dir.mkdir(parents=True, exist_ok=True)

    # Copy template to global config
    template_path = Path(__file__).parent / "templates" / "global-config.yaml"
    global_config_path = global_config_dir / "config.yaml"

    if template_path.exists():
        shutil.copy(template_path, global_config_path)
        click.echo("📋 Installed fresh global config from template")

        # Restore authentication files
        if auth_files:
            for auth_file, auth_content in auth_files:
                auth_path = global_config_dir / auth_file
                auth_path.write_bytes(auth_content)
                click.echo(f"🔐 Restored authentication file: {auth_file}")

        click.echo(f"✅ Global config reset complete at {global_config_dir}")

        if preserve_auth and auth_files:
            click.echo("🔐 Authentication credentials preserved")
        elif not preserve_auth:
            click.echo(
                "💡 Note: You may need to reconfigure authentication "
                "with 'llm-orc auth setup'"
            )
    else:
        raise click.ClickException(f"Template not found at {template_path}")


def check_global_config() -> None:
    """Check global configuration status."""
    config_manager = ConfigurationManager()
    global_config_dir = Path(config_manager.global_config_dir)
    global_config_path = global_config_dir / "config.yaml"

    click.echo("Global Configuration Status:")
    click.echo(f"Directory: {global_config_dir}")

    if global_config_path.exists():
        click.echo("Status: configured")

        # Show basic info about the config
        try:
            # Get available providers first
            available_providers = get_available_providers(config_manager)

            # Show providers FIRST, right after status
            display_providers_status(available_providers, config_manager)

            # Read ONLY global config file, not merged profiles
            with open(global_config_path) as f:
                global_config = yaml.safe_load(f) or {}

            # Show default model profiles configuration
            display_default_models_config(config_manager, available_providers)

            # Check global ensembles SECOND
            global_ensembles_dir = global_config_dir / "ensembles"
            check_ensemble_availability(
                global_ensembles_dir, available_providers, config_manager
            )

            # Show global profiles
            display_global_profiles(global_config, available_providers)

        except Exception as e:
            click.echo(f"Error reading config: {e}")
    else:
        click.echo("Status: missing")
        click.echo("Run 'llm-orc config init' to create it")


def check_local_config() -> None:
    """Check local .llm-orc configuration status."""
    local_config_dir = Path(".llm-orc")
    local_config_path = local_config_dir / "config.yaml"

    if local_config_path.exists():
        # Show basic info about the config
        try:
            config_manager = ConfigurationManager()

            # Check project config first to get project name
            project_config = config_manager.load_project_config()
            if project_config:
                project_name = project_config.get("project", {}).get("name", "Unknown")
                click.echo(f"Local Configuration Status: {project_name}")
                click.echo(f"Directory: {local_config_dir.absolute()}")
                click.echo("Status: configured")

                # Get available providers for ensemble checking
                available_providers = get_available_providers(config_manager)

                # Check local ensembles with availability indicators
                ensembles_dir = local_config_dir / "ensembles"
                check_ensemble_availability(
                    ensembles_dir, available_providers, config_manager
                )

                # Show local model profiles
                local_profiles = project_config.get("model_profiles", {})
                if local_profiles:
                    click.echo(
                        f"\n💻 Local model profiles ({len(local_profiles)} found):"
                    )
                    for profile_name in sorted(local_profiles.keys()):
                        profile = local_profiles[profile_name]
                        model = profile.get("model", "unknown")
                        provider = profile.get("provider", "unknown")
                        cost = profile.get("cost_per_token", "not specified")
                        timeout = profile.get("timeout_seconds", "not specified")
                        has_system_prompt = "system_prompt" in profile

                        # Check if provider is available
                        provider_available = provider in available_providers
                        status_symbol = "🟢" if provider_available else "🟥"

                        timeout_display = (
                            f"{timeout}s" if timeout != "not specified" else timeout
                        )
                        click.echo(
                            f"  {status_symbol} {profile_name}: {model} ({provider})"
                        )
                        system_prompt_indicator = "✓" if has_system_prompt else "✗"
                        click.echo(
                            f"    Cost: {cost}, Timeout: {timeout_display}, "
                            f"System prompt: {system_prompt_indicator}"
                        )
            else:
                click.echo("Local Configuration Status:")
                click.echo(f"Directory: {local_config_dir.absolute()}")
                click.echo("Status: configured but no project config found")

        except Exception as e:
            click.echo("Local Configuration Status:")
            click.echo(f"Directory: {local_config_dir.absolute()}")
            click.echo(f"Error reading local config: {e}")
    else:
        click.echo("Local Configuration Status:")
        click.echo(f"Directory: {local_config_dir.absolute()}")
        click.echo("Status: missing")
        click.echo("Run 'llm-orc config init' to create it")


def reset_local_config(
    backup: bool, preserve_ensembles: bool, project_name: str | None
) -> None:
    """Reset local .llm-orc configuration to template defaults."""
    config_manager = ConfigurationManager()
    local_config_dir = Path(".llm-orc")

    if not local_config_dir.exists():
        click.echo("❌ No local .llm-orc directory found")
        click.echo("💡 Run 'llm-orc config init' to create initial local config")
        return

    # Create backup if requested
    if backup:
        backup_path = Path(".llm-orc.backup")
        if backup_path.exists():
            shutil.rmtree(backup_path)
        shutil.copytree(local_config_dir, backup_path)
        click.echo(f"📦 Backed up existing local config to {backup_path}")

    # Preserve ensembles if requested
    ensembles_backup = None
    if preserve_ensembles:
        ensembles_dir = local_config_dir / "ensembles"
        if ensembles_dir.exists():
            # Save ensembles directory content
            ensembles_backup = {}
            for ensemble_file in ensembles_dir.glob("*.yaml"):
                ensembles_backup[ensemble_file.name] = ensemble_file.read_text()
            click.echo(f"🎭 Preserving {len(ensembles_backup)} ensemble(s)")

    # Remove existing local config
    shutil.rmtree(local_config_dir)

    # Initialize fresh local config
    try:
        config_manager.init_local_config(project_name)
        click.echo("📋 Created fresh local config from template")

        # Restore ensembles if preserved
        if ensembles_backup:
            ensembles_dir = local_config_dir / "ensembles"
            for ensemble_name, ensemble_content in ensembles_backup.items():
                ensemble_path = ensembles_dir / ensemble_name
                ensemble_path.write_text(ensemble_content)
                click.echo(f"🎭 Restored ensemble: {ensemble_name}")

        click.echo(f"✅ Local config reset complete at {local_config_dir}")

        if preserve_ensembles and ensembles_backup:
            click.echo("🎭 Existing ensembles preserved")
        elif not preserve_ensembles:
            click.echo("💡 Note: All ensembles were reset to template defaults")

    except ValueError as e:
        raise click.ClickException(str(e)) from e


def serve_ensemble(ensemble_name: str, port: int) -> None:
    """Serve an ensemble as an MCP server."""
    runner = MCPServerRunner(ensemble_name, port)
    runner.run()


def add_auth_provider(
    provider: str,
    api_key: str | None,
    client_id: str | None,
    client_secret: str | None,
) -> None:
    """Add authentication for a provider (API key or OAuth)."""
    config_manager = ConfigurationManager()
    storage = CredentialStorage(config_manager)
    auth_manager = AuthenticationManager(storage)

    # Special handling for claude-cli provider
    if provider.lower() == "claude-cli":
        try:
            handle_claude_cli_auth(storage)
            return
        except Exception as e:
            raise click.ClickException(
                f"Failed to set up Claude CLI authentication: {str(e)}"
            ) from e

    # Special handling for anthropic-claude-pro-max OAuth
    if provider.lower() == "anthropic-claude-pro-max":
        try:
            handle_claude_pro_max_oauth(auth_manager, storage)
            return
        except Exception as e:
            raise click.ClickException(
                f"Failed to set up Claude Pro/Max OAuth authentication: {str(e)}"
            ) from e

    # Special interactive flow for Anthropic
    is_anthropic_interactive = (
        provider.lower() == "anthropic"
        and not api_key
        and not (client_id and client_secret)
    )
    if is_anthropic_interactive:
        try:
            handle_anthropic_interactive_auth(auth_manager, storage)
            return
        except Exception as e:
            raise click.ClickException(
                f"Failed to set up Anthropic authentication: {str(e)}"
            ) from e

    # Validate input for non-interactive flow
    if api_key and (client_id or client_secret):
        raise click.ClickException("Cannot use both API key and OAuth credentials")

    if not api_key and not (client_id and client_secret):
        raise click.ClickException(
            "Must provide either --api-key or both --client-id and --client-secret"
        )

    try:
        if api_key:
            # API key authentication
            storage.store_api_key(provider, api_key)
            click.echo(f"API key for {provider} added successfully")
        else:
            # OAuth authentication - we know these are not None due to validation above
            assert client_id is not None
            assert client_secret is not None
            if auth_manager.authenticate_oauth(provider, client_id, client_secret):
                click.echo(
                    f"OAuth authentication for {provider} completed successfully"
                )
            else:
                raise click.ClickException(
                    f"OAuth authentication for {provider} failed"
                )
    except Exception as e:
        raise click.ClickException(f"Failed to add authentication: {str(e)}") from e


def list_auth_providers(interactive: bool) -> None:
    """List configured authentication providers."""
    from llm_orc.menu_system import (
        AuthMenus,
        confirm_action,
        show_error,
        show_info,
        show_success,
        show_working,
    )

    config_manager = ConfigurationManager()
    storage = CredentialStorage(config_manager)
    auth_manager = AuthenticationManager(storage)

    try:
        providers = storage.list_providers()

        if not interactive:
            # Simple list view (original behavior)
            if not providers:
                click.echo("No authentication providers configured")
            else:
                click.echo("Configured providers:")
                for provider in providers:
                    auth_method = storage.get_auth_method(provider)
                    if auth_method == "oauth":
                        click.echo(f"  {provider}: OAuth")
                    else:
                        click.echo(f"  {provider}: API key")
            return

        # Interactive mode with action menu
        while True:
            action, selected_provider = AuthMenus.auth_list_actions(providers)

            if action == "quit":
                break
            elif action == "setup" or action == "add":
                # Run the setup wizard
                auth_setup()
                # Refresh provider list
                providers = storage.list_providers()
            elif action == "test" and selected_provider:
                show_working(f"Testing {selected_provider}...")
                try:
                    success = test_provider_authentication(
                        storage, auth_manager, selected_provider
                    )
                    if success:
                        show_success(
                            f"Authentication for {selected_provider} is working!"
                        )
                    else:
                        show_error(f"Authentication for {selected_provider} failed")
                except Exception as e:
                    show_error(f"Test failed: {str(e)}")
            elif action == "remove" and selected_provider:
                if confirm_action(f"Remove authentication for {selected_provider}?"):
                    storage.remove_provider(selected_provider)
                    show_success(f"Removed {selected_provider}")
                    providers = storage.list_providers()
            elif action == "details" and selected_provider:
                show_provider_details(storage, selected_provider)
            elif action == "refresh" and selected_provider:
                show_working(f"Refreshing tokens for {selected_provider}...")
                try:
                    auth_method = storage.get_auth_method(selected_provider)
                    if auth_method == "oauth":
                        # For now, just re-authenticate with OAuth
                        show_info("Re-authentication required for OAuth token refresh")
                        # This would typically trigger a re-auth flow
                        show_success("Token refresh would be performed here")
                    else:
                        show_error("Token refresh only available for OAuth providers")
                except Exception as e:
                    show_error(f"Refresh failed: {str(e)}")

    except Exception as e:
        raise click.ClickException(f"Failed to list providers: {str(e)}") from e


def remove_auth_provider(provider: str) -> None:
    """Remove authentication for a provider."""
    config_manager = ConfigurationManager()
    storage = CredentialStorage(config_manager)

    try:
        # Check if provider exists
        if provider not in storage.list_providers():
            raise click.ClickException(f"No authentication found for {provider}")

        storage.remove_provider(provider)
        click.echo(f"Authentication for {provider} removed")
    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(f"Failed to remove provider: {str(e)}") from e


def auth_setup() -> None:
    """Interactive setup wizard for authentication."""
    from llm_orc.menu_system import (
        AuthMenus,
        confirm_action,
        show_error,
        show_success,
        show_working,
    )
    from llm_orc.provider_registry import provider_registry

    config_manager = ConfigurationManager()
    storage = CredentialStorage(config_manager)
    auth_manager = AuthenticationManager(storage)

    click.echo("🚀 Welcome to LLM Orchestra setup!")
    click.echo("This wizard will help you configure authentication for LLM providers.")

    while True:
        # Use interactive menu for provider selection
        provider_key = AuthMenus.provider_selection()

        # Get provider info
        provider = provider_registry.get_provider(provider_key)

        if not provider:
            show_error(f"Provider '{provider_key}' not found in registry")
            continue

        if not provider.requires_auth:
            show_success(f"{provider.display_name} doesn't require authentication!")
            if not confirm_action("Add another provider?"):
                break
            continue

        # Get authentication method based on provider
        if provider_key == "anthropic-claude-pro-max":
            auth_method = "oauth"  # Claude Pro/Max only supports OAuth
        elif provider_key == "anthropic-api":
            auth_method = "api_key"  # Anthropic API only supports API key
        elif provider_key == "google-gemini":
            auth_method = "api_key"  # Google Gemini only supports API key
        else:
            # For other providers, use the menu system
            auth_method = AuthMenus.get_auth_method_for_provider(provider_key)

        # Handle authentication setup based on method
        try:
            if auth_method == "help":
                show_auth_method_help()
                continue
            elif auth_method == "oauth" and provider_key == "anthropic-claude-pro-max":
                show_working("Setting up Claude Pro/Max OAuth...")
                handle_claude_pro_max_oauth(auth_manager, storage)
                show_success("Claude Pro/Max OAuth configured!")
            elif auth_method == "api_key" and provider_key == "anthropic-api":
                api_key = click.prompt("Anthropic API key", hide_input=True)
                storage.store_api_key("anthropic-api", api_key)
                show_success("Anthropic API key configured!")
            elif auth_method == "api_key" and provider_key == "google-gemini":
                api_key = click.prompt("Google Gemini API key", hide_input=True)
                storage.store_api_key("google-gemini", api_key)
                show_success("Google Gemini API key configured!")
            elif auth_method == "api_key" or auth_method == "api-key":
                # Generic API key setup for other providers
                api_key = click.prompt(
                    f"{provider.display_name} API key", hide_input=True
                )
                storage.store_api_key(provider_key, api_key)
                show_success(f"{provider.display_name} API key configured!")
            elif auth_method == "oauth":
                # Generic OAuth setup for other providers
                client_id = click.prompt("OAuth client ID")
                client_secret = click.prompt("OAuth client secret", hide_input=True)

                if auth_manager.authenticate_oauth(
                    provider_key, client_id, client_secret
                ):
                    show_success(f"{provider.display_name} OAuth configured!")
                else:
                    show_error(
                        f"OAuth authentication for {provider.display_name} failed"
                    )
            else:
                show_error(f"Unknown authentication method: {auth_method}")

        except Exception as e:
            show_error(f"Failed to configure {provider.display_name}: {str(e)}")

        if not confirm_action("Add another provider?"):
            break

    click.echo()
    show_success(
        "Setup complete! Use 'llm-orc auth list' to see your configured providers."
    )


def logout_oauth_providers(provider: str | None, logout_all: bool) -> None:
    """Logout from OAuth providers (revokes tokens and removes credentials)."""
    config_manager = ConfigurationManager()
    storage = CredentialStorage(config_manager)
    auth_manager = AuthenticationManager(storage)

    try:
        if logout_all:
            # Logout from all OAuth providers
            results = auth_manager.logout_all_oauth_providers()

            if not results:
                click.echo("No OAuth providers found to logout")
                return

            success_count = sum(1 for success in results.values() if success)

            click.echo(f"Logged out from {success_count} OAuth providers:")
            for provider_name, success in results.items():
                status = "✅" if success else "❌"
                click.echo(f"  {provider_name}: {status}")

        elif provider:
            # Logout from specific provider
            if auth_manager.logout_oauth_provider(provider):
                click.echo(f"✅ Logged out from {provider}")
            else:
                raise click.ClickException(
                    f"Failed to logout from {provider}. "
                    f"Provider may not exist or is not an OAuth provider."
                )
        else:
            raise click.ClickException("Must specify a provider name or use --all flag")

    except Exception as e:
        raise click.ClickException(f"Failed to logout: {str(e)}") from e
