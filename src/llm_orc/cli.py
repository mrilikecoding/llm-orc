"""Command line interface for llm-orc."""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

import click

from llm_orc.authentication import AuthenticationManager, CredentialStorage
from llm_orc.config import ConfigurationManager
from llm_orc.ensemble_config import EnsembleLoader
from llm_orc.ensemble_execution import EnsembleExecutor
from llm_orc.mcp_server_runner import MCPServerRunner


@click.group()
@click.version_option(package_name="llm-orchestra")
def cli() -> None:
    """LLM Orchestra - Multi-agent LLM communication system."""
    pass


@cli.command()
@click.argument("ensemble_name")
@click.option(
    "--config-dir",
    default=None,
    help="Directory containing ensemble configurations",
)
@click.option(
    "--input-data",
    default=None,
    help="Input data for the ensemble (if not provided, reads from stdin)",
)
@click.option(
    "--output-format",
    type=click.Choice(["json", "text"]),
    default="text",
    help="Output format for results",
)
def invoke(
    ensemble_name: str, config_dir: str, input_data: str, output_format: str
) -> None:
    """Invoke an ensemble of agents."""
    # Initialize configuration manager
    config_manager = ConfigurationManager()

    # Handle migration if needed
    if config_manager.needs_migration():
        click.echo("Migrating configuration from ~/.llm-orc to new location...")
        config_manager.migrate_from_old_location()
        click.echo(f"Configuration migrated to: {config_manager.global_config_dir}")

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

    # Handle input from stdin if not provided via --input
    if input_data is None:
        if not sys.stdin.isatty():
            # Read from stdin (piped input)
            input_data = sys.stdin.read().strip()
        else:
            # No input provided and not piped, use default
            input_data = "Please analyze this."

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

    if output_format == "text":
        click.echo(f"Invoking ensemble: {ensemble_name}")
        click.echo(f"Description: {ensemble_config.description}")
        click.echo(f"Agents: {len(ensemble_config.agents)}")
        click.echo(f"Input: {input_data}")
        click.echo("---")

    # Execute the ensemble
    async def run_ensemble() -> dict[str, Any]:
        executor = EnsembleExecutor()
        return await executor.execute(ensemble_config, input_data)

    try:
        result = asyncio.run(run_ensemble())

        if output_format == "json":
            click.echo(json.dumps(result, indent=2))
        else:
            # Text format - show readable output
            click.echo(f"Status: {result['status']}")
            click.echo(f"Duration: {result['metadata']['duration']}")

            # Show usage summary
            if "usage" in result["metadata"]:
                usage = result["metadata"]["usage"]
                totals = usage.get("totals", {})
                click.echo("\nUsage Summary:")
                click.echo(f"  Total Tokens: {totals.get('total_tokens', 0):,}")
                click.echo(f"  Total Cost: ${totals.get('total_cost_usd', 0.0):.4f}")
                click.echo(f"  Agents: {totals.get('agents_count', 0)}")

                # Show per-agent usage
                agents_usage = usage.get("agents", {})
                if agents_usage:
                    click.echo("\nPer-Agent Usage:")
                    for agent_name, agent_usage in agents_usage.items():
                        tokens = agent_usage.get("total_tokens", 0)
                        cost = agent_usage.get("cost_usd", 0.0)
                        duration = agent_usage.get("duration_ms", 0)
                        model = agent_usage.get("model", "unknown")
                        click.echo(
                            f"  {agent_name} ({model}): {tokens:,} tokens, "
                            f"${cost:.4f}, {duration}ms"
                        )

                # Show synthesis usage if present
                synthesis_usage = usage.get("synthesis", {})
                if synthesis_usage:
                    tokens = synthesis_usage.get("total_tokens", 0)
                    cost = synthesis_usage.get("cost_usd", 0.0)
                    duration = synthesis_usage.get("duration_ms", 0)
                    model = synthesis_usage.get("model", "unknown")
                    click.echo(
                        f"  synthesis ({model}): {tokens:,} tokens, "
                        f"${cost:.4f}, {duration}ms"
                    )

            click.echo("\nAgent Results:")
            for agent_name, agent_result in result["results"].items():
                if agent_result["status"] == "success":
                    click.echo(f"  {agent_name}: {agent_result['response']}")
                else:
                    click.echo(f"  {agent_name}: ERROR - {agent_result['error']}")

            if result.get("synthesis"):
                click.echo(f"\nSynthesis: {result['synthesis']}")

    except Exception as e:
        raise click.ClickException(f"Ensemble execution failed: {str(e)}") from e


@cli.command("list-ensembles")
@click.option(
    "--config-dir",
    default=None,
    help="Directory containing ensemble configurations",
)
def list_ensembles(config_dir: str) -> None:
    """List available ensembles."""
    # Initialize configuration manager
    config_manager = ConfigurationManager()

    # Handle migration if needed
    if config_manager.needs_migration():
        click.echo("Migrating configuration from ~/.llm-orc to new location...")
        config_manager.migrate_from_old_location()
        click.echo(f"Configuration migrated to: {config_manager.global_config_dir}")

    if config_dir is None:
        # Use configuration manager to get ensemble directories
        ensemble_dirs = config_manager.get_ensembles_dirs()
        if not ensemble_dirs:
            click.echo("No ensemble directories found.")
            click.echo("Run 'llm-orc config init' to set up local configuration.")
            return

        # List ensembles from all directories in priority order
        loader = EnsembleLoader()
        all_ensembles = {}

        for dir_path in ensemble_dirs:
            ensembles = loader.list_ensembles(str(dir_path))
            for ensemble in ensembles:
                if ensemble.name not in all_ensembles:
                    all_ensembles[ensemble.name] = (ensemble, str(dir_path))

        if not all_ensembles:
            click.echo("No ensembles found in any configured directories:")
            for dir_path in ensemble_dirs:
                click.echo(f"  {dir_path}")
            click.echo("  (Create .yaml files with ensemble configurations)")
        else:
            click.echo("Available ensembles:")
            for _ensemble_name, (ensemble, source_dir) in sorted(all_ensembles.items()):
                click.echo(f"  {ensemble.name}: {ensemble.description}")
                if len(ensemble_dirs) > 1:
                    click.echo(f"    Source: {source_dir}")
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


@cli.group()
def config() -> None:
    """Configuration management commands."""
    pass


@config.command()
@click.option(
    "--project-name",
    default=None,
    help="Name for the project (defaults to directory name)",
)
def init(project_name: str) -> None:
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


@config.command()
def migrate() -> None:
    """Migrate configuration from old ~/.llm-orc location to new XDG-compliant
    location."""
    config_manager = ConfigurationManager()

    if not config_manager.needs_migration():
        click.echo(
            "No migration needed. Configuration is already in the correct location."
        )
        return

    try:
        config_manager.migrate_from_old_location()
        click.echo("Configuration migrated successfully!")
        click.echo("Old location: ~/.llm-orc")
        click.echo(f"New location: {config_manager.global_config_dir}")
    except ValueError as e:
        raise click.ClickException(str(e)) from e


@config.command()
def show() -> None:
    """Show current configuration information."""
    config_manager = ConfigurationManager()

    click.echo("Configuration Information:")
    click.echo(f"Global config directory: {config_manager.global_config_dir}")

    if config_manager.local_config_dir:
        click.echo(f"Local config directory: {config_manager.local_config_dir}")
    else:
        click.echo("Local config directory: Not found")

    click.echo("\nEnsemble directories (in search order):")
    ensemble_dirs = config_manager.get_ensembles_dirs()
    if ensemble_dirs:
        for i, dir_path in enumerate(ensemble_dirs, 1):
            click.echo(f"  {i}. {dir_path}")
    else:
        click.echo("  None found")

    if config_manager.needs_migration():
        click.echo(
            "\n⚠️  Migration available: Run 'llm-orc config migrate' to update "
            "configuration location"
        )

    # Show project config if available
    project_config = config_manager.load_project_config()
    if project_config:
        click.echo("\nProject Configuration:")
        project_name = project_config.get("project", {}).get("name", "Unknown")
        click.echo(f"  Project name: {project_name}")

        profiles = project_config.get("model_profiles", {})
        if profiles:
            click.echo("  Model profiles:")
            for profile_name in profiles.keys():
                click.echo(f"    - {profile_name}")


@cli.group()
def auth() -> None:
    """Authentication management commands."""
    pass


@auth.command("add")
@click.argument("provider")
@click.option("--api-key", help="API key for the provider")
@click.option("--client-id", help="OAuth client ID")
@click.option("--client-secret", help="OAuth client secret")
def auth_add(provider: str, api_key: str, client_id: str, client_secret: str) -> None:
    """Add authentication for a provider (API key or OAuth)."""
    config_manager = ConfigurationManager()
    storage = CredentialStorage(config_manager)
    auth_manager = AuthenticationManager(storage)

    # Special handling for claude-cli provider
    if provider.lower() == "claude-cli":
        try:
            _handle_claude_cli_auth(storage)
            return
        except Exception as e:
            raise click.ClickException(
                f"Failed to set up Claude CLI authentication: {str(e)}"
            ) from e

    # Special handling for anthropic-claude-pro-max OAuth
    if provider.lower() == "anthropic-claude-pro-max":
        try:
            _handle_claude_pro_max_oauth(auth_manager, storage)
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
            _handle_anthropic_interactive_auth(auth_manager, storage)
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
            # OAuth authentication
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


def _handle_anthropic_interactive_auth(
    auth_manager: AuthenticationManager, storage: CredentialStorage
) -> None:
    """Handle interactive Anthropic authentication setup."""
    click.echo("How would you like to authenticate with Anthropic?")
    click.echo("1. API Key (for Anthropic API access)")
    click.echo("2. Claude Pro/Max OAuth (for your existing Claude subscription)")
    click.echo("3. Both (set up multiple authentication methods)")
    click.echo()

    choice = click.prompt("Choice", type=click.Choice(["1", "2", "3"]), default="1")

    if choice == "1":
        # API Key only
        api_key = click.prompt("Anthropic API key", hide_input=True)
        storage.store_api_key("anthropic-api", api_key)
        click.echo("✅ API key configured as 'anthropic-api'")

    elif choice == "2":
        # OAuth only
        _setup_anthropic_oauth(auth_manager, "anthropic-claude-pro-max")
        click.echo("✅ OAuth configured as 'anthropic-claude-pro-max'")

    elif choice == "3":
        # Both methods
        click.echo()
        click.echo("🔑 Setting up API key access...")
        api_key = click.prompt("Anthropic API key", hide_input=True)
        storage.store_api_key("anthropic-api", api_key)
        click.echo("✅ API key configured as 'anthropic-api'")

        click.echo()
        click.echo("🔧 Setting up Claude Pro/Max OAuth...")
        _setup_anthropic_oauth(auth_manager, "anthropic-claude-pro-max")
        click.echo("✅ OAuth configured as 'anthropic-claude-pro-max'")


def _setup_anthropic_oauth(
    auth_manager: AuthenticationManager, provider_key: str
) -> None:
    """Set up Anthropic OAuth authentication."""
    from llm_orc.authentication import AnthropicOAuthFlow

    oauth_flow = AnthropicOAuthFlow.create_with_guidance()

    if not auth_manager.authenticate_oauth(
        provider_key, oauth_flow.client_id, oauth_flow.client_secret
    ):
        raise click.ClickException("OAuth authentication failed")


def _handle_claude_pro_max_oauth(
    auth_manager: AuthenticationManager, storage: CredentialStorage
) -> None:
    """Handle Claude Pro/Max OAuth authentication setup using hardcoded client ID."""
    import base64
    import hashlib
    import secrets
    import time
    import webbrowser
    from urllib.parse import urlencode

    click.echo("🔧 Setting up Claude Pro/Max OAuth Authentication")
    click.echo("=" * 55)
    click.echo("This will authenticate with your existing Claude Pro/Max subscription.")
    click.echo()

    # Hardcoded OAuth parameters from issue-32
    client_id = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
    redirect_uri = "https://console.anthropic.com/oauth/code/callback"
    scope = "org:create_api_key user:profile user:inference"

    # Generate PKCE parameters
    code_verifier = (
        base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("utf-8").rstrip("=")
    )
    code_challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode("utf-8")).digest())
        .decode("utf-8")
        .rstrip("=")
    )

    # Build authorization URL
    params = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": scope,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": code_verifier,
    }

    auth_url = f"https://claude.ai/oauth/authorize?{urlencode(params)}"

    click.echo("📋 OAuth Flow Details:")
    click.echo(f"   • Client ID: {client_id}")
    click.echo(f"   • Scope: {scope}")
    click.echo(f"   • Redirect URI: {redirect_uri}")
    click.echo()

    # Open browser and guide user
    click.echo("🌐 Opening authorization URL in your browser...")
    click.echo(f"   {auth_url}")
    click.echo()

    if click.confirm("Open browser automatically?", default=True):
        webbrowser.open(auth_url)
        click.echo("✅ Browser opened")
    else:
        click.echo("Please manually navigate to the URL above")

    click.echo()
    click.echo("📋 Instructions:")
    click.echo("1. Sign in to your Claude Pro/Max account")
    click.echo("2. Authorize the application")
    click.echo("3. You'll be redirected to a callback page")
    click.echo("4. Copy the full URL from the address bar")
    click.echo("5. Extract the authorization code from the URL")
    click.echo()

    # Get authorization code from user
    auth_code = click.prompt(
        "Authorization code (format: code#state)", type=str
    ).strip()

    # Parse auth code
    splits = auth_code.split("#")
    if len(splits) != 2:
        raise click.ClickException(
            f"Invalid authorization code format. Expected 'code#state', "
            f"got: {auth_code}"
        )

    code_part = splits[0]
    state_part = splits[1]

    # Verify state matches
    if state_part != code_verifier:
        click.echo("⚠️  Warning: State mismatch - proceeding anyway")

    # Exchange code for tokens
    click.echo("🔄 Exchanging authorization code for access tokens...")

    import requests

    token_url = "https://console.anthropic.com/v1/oauth/token"
    data = {
        "code": code_part,
        "state": state_part,
        "grant_type": "authorization_code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "code_verifier": code_verifier,
    }

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(token_url, json=data, headers=headers, timeout=30)

        if response.status_code == 200:
            tokens = response.json()

            # Store OAuth tokens
            storage.store_oauth_token(
                "anthropic-claude-pro-max",
                tokens["access_token"],
                tokens.get("refresh_token"),
                int(time.time()) + tokens.get("expires_in", 3600),
                client_id,
            )

            click.echo("✅ OAuth authentication successful!")
            click.echo("✅ Tokens stored as 'anthropic-claude-pro-max'")

        else:
            raise click.ClickException(
                f"Token exchange failed. Status: {response.status_code}, "
                f"Response: {response.text}"
            )

    except requests.exceptions.RequestException as e:
        raise click.ClickException(
            f"Network error during token exchange: {str(e)}"
        ) from e


def _handle_claude_cli_auth(storage: CredentialStorage) -> None:
    """Handle Claude CLI authentication setup."""
    import shutil

    # Check if claude command is available
    claude_path = shutil.which("claude")
    if not claude_path:
        raise click.ClickException(
            "Claude CLI not found. Please install the Claude CLI from: "
            "https://docs.anthropic.com/en/docs/claude-code"
        )

    # Store claude-cli as a special auth method
    # We'll store the path to the claude executable
    storage.store_api_key("claude-cli", claude_path)

    click.echo("✅ Claude CLI authentication configured")
    click.echo(f"Using local claude command at: {claude_path}")


@auth.command("list")
@click.option(
    "--interactive", "-i", is_flag=True, help="Show interactive menu with actions"
)
def auth_list(interactive: bool) -> None:
    """List configured authentication providers."""
    from .menu_system import (
        AuthMenus,
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
                    auth_method = storage.get_auth_method(selected_provider)
                    if not auth_method:
                        show_error(f"No authentication found for {selected_provider}")
                        continue

                    # Test the authentication
                    success = False
                    if auth_method == "api_key":
                        api_key = storage.get_api_key(selected_provider)
                        if api_key:
                            success = auth_manager.authenticate(
                                selected_provider, api_key
                            )
                    elif auth_method == "oauth":
                        oauth_token = storage.get_oauth_token(selected_provider)
                        if oauth_token:
                            # For OAuth, we'll consider it successful if we have a valid token
                            import time
                            if "expires_at" in oauth_token:
                                success = time.time() < oauth_token["expires_at"]
                            else:
                                success = True  # No expiration info, assume valid

                    if success:
                        show_success(
                            f"Authentication for {selected_provider} is working!"
                        )
                    else:
                        show_error(f"Authentication for {selected_provider} failed")
                except Exception as e:
                    show_error(f"Test failed: {str(e)}")
            elif action == "remove" and selected_provider:
                from .menu_system import confirm_action
                if confirm_action(f"Remove authentication for {selected_provider}?"):
                    storage.remove_provider(selected_provider)
                    show_success(f"Removed {selected_provider}")
                    providers = storage.list_providers()
            elif action == "details" and selected_provider:
                _show_provider_details(storage, selected_provider)
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


def _show_provider_details(storage: "CredentialStorage", provider: str) -> None:
    """Show detailed information about a provider."""
    from .provider_registry import provider_registry

    click.echo(f"\n📋 Provider Details: {provider}")
    click.echo("=" * 40)

    # Get registry info
    provider_info = provider_registry.get_provider(provider)
    if provider_info:
        click.echo(f"Display Name: {provider_info.display_name}")
        click.echo(f"Description: {provider_info.description}")

        auth_methods = []
        if provider_info.supports_oauth:
            auth_methods.append("OAuth")
        if provider_info.supports_api_key:
            auth_methods.append("API Key")
        if not provider_info.requires_auth:
            auth_methods.append("No authentication required")
        click.echo(f"Supported Auth: {', '.join(auth_methods)}")

    # Get stored auth info
    auth_method = storage.get_auth_method(provider)
    if auth_method:
        click.echo(f"Configured Method: {auth_method.upper()}")

        if auth_method == "oauth":
            # Try to get OAuth details if available
            try:
                # This would need to be implemented in storage
                click.echo("OAuth Status: Configured")
            except Exception:
                pass
    else:
        click.echo("Status: Not configured")

    click.echo()


@auth.command("remove")
@click.argument("provider")
def auth_remove(provider: str) -> None:
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


@auth.command("test")
@click.argument("provider")
def auth_test(provider: str) -> None:
    """Test authentication for a provider."""
    config_manager = ConfigurationManager()
    storage = CredentialStorage(config_manager)
    auth_manager = AuthenticationManager(storage)

    try:
        auth_method = storage.get_auth_method(provider)
        if not auth_method:
            raise click.ClickException(f"No authentication found for {provider}")

        if auth_method == "api_key":
            api_key = storage.get_api_key(provider)
            if not api_key:
                raise click.ClickException(f"No API key found for {provider}")

            if auth_manager.authenticate(provider, api_key):
                click.echo(f"API key authentication for {provider} is working")
            else:
                raise click.ClickException(
                    f"API key authentication for {provider} failed"
                )

        elif auth_method == "oauth":
            oauth_token = storage.get_oauth_token(provider)
            if not oauth_token:
                raise click.ClickException(f"No OAuth token found for {provider}")

            # Check if token is expired
            if "expires_at" in oauth_token:
                if time.time() > oauth_token["expires_at"]:
                    click.echo(f"OAuth token for {provider} has expired")
                    return

            click.echo(f"OAuth authentication for {provider} is working")

        else:
            raise click.ClickException(f"Unknown authentication method: {auth_method}")

    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(f"Failed to test authentication: {str(e)}") from e


@auth.command("oauth")
@click.argument("provider")
@click.option("--client-id", required=True, help="OAuth client ID")
@click.option("--client-secret", required=True, help="OAuth client secret")
def auth_oauth(provider: str, client_id: str, client_secret: str) -> None:
    """Configure OAuth authentication for a provider."""
    config_manager = ConfigurationManager()
    storage = CredentialStorage(config_manager)
    auth_manager = AuthenticationManager(storage)

    try:
        if auth_manager.authenticate_oauth(provider, client_id, client_secret):
            click.echo(f"OAuth authentication for {provider} completed successfully")
        else:
            raise click.ClickException(f"OAuth authentication for {provider} failed")
    except Exception as e:
        raise click.ClickException(
            f"Failed to complete OAuth authentication: {str(e)}"
        ) from e


@auth.command("setup")
def auth_setup() -> None:
    """Interactive setup wizard for authentication."""
    from .menu_system import (
        AuthMenus,
        confirm_action,
        show_error,
        show_success,
        show_working,
    )
    from .provider_registry import provider_registry

    config_manager = ConfigurationManager()
    storage = CredentialStorage(config_manager)
    auth_manager = AuthenticationManager(storage)

    click.echo("🚀 Welcome to LLM Orchestra setup!")
    click.echo("This wizard will help you configure authentication for LLM providers.")

    while True:
        # Use interactive menu for provider selection
        provider_key = AuthMenus.provider_selection()

        if provider_key == "custom":
            provider_key = click.prompt("Enter provider name")

        # Get provider info
        provider = provider_registry.get_provider(provider_key)

        if provider and not provider.requires_auth:
            show_success(f"{provider.display_name} doesn't require authentication!")
            if not confirm_action("Add another provider?"):
                break
            continue

        # Get authentication method
        if provider_key == "anthropic":
            auth_method = AuthMenus.anthropic_auth_method()
        elif provider:
            auth_method = AuthMenus.get_auth_method_for_provider(provider_key)
        else:
            # Custom provider - ask for method
            auth_method = click.prompt(
                "Authentication method",
                type=click.Choice(["api_key", "oauth"]),
                default="api_key",
            )

        # Handle authentication setup based on method
        try:
            if auth_method == "help":
                _show_auth_method_help()
                continue
            elif auth_method == "oauth" and provider_key == "anthropic":
                show_working("Setting up Claude Pro/Max OAuth...")
                _handle_claude_pro_max_oauth(auth_manager, storage)
                show_success("Claude Pro/Max OAuth configured!")
            elif auth_method == "api-key" and provider_key == "anthropic":
                api_key = click.prompt("Anthropic API key", hide_input=True)
                storage.store_api_key("anthropic", api_key)
                show_success("Anthropic API key configured!")
            elif auth_method == "both" and provider_key == "anthropic":
                # Set up both OAuth and API key for Anthropic
                show_working("Setting up Claude Pro/Max OAuth...")
                _handle_claude_pro_max_oauth(auth_manager, storage)
                show_success("OAuth configured!")

                api_key = click.prompt("Anthropic API key", hide_input=True)
                storage.store_api_key("anthropic", api_key)
                show_success("API key configured!")
            elif auth_method == "api_key" or auth_method == "api-key":
                api_key = click.prompt(f"{provider_key} API key", hide_input=True)
                storage.store_api_key(provider_key, api_key)
                show_success(f"{provider_key} API key configured!")
            elif auth_method == "oauth":
                client_id = click.prompt("OAuth client ID")
                client_secret = click.prompt("OAuth client secret", hide_input=True)

                if auth_manager.authenticate_oauth(provider_key, client_id, client_secret):
                    show_success(f"{provider_key} OAuth configured!")
                else:
                    show_error(f"OAuth authentication for {provider_key} failed")
            else:
                show_error(f"Unknown authentication method: {auth_method}")

        except Exception as e:
            show_error(f"Failed to configure {provider_key}: {str(e)}")

        if not confirm_action("Add another provider?"):
            break

    click.echo()
    show_success("Setup complete! Use 'llm-orc auth list' to see your configured providers.")


def _show_auth_method_help() -> None:
    """Show help for choosing authentication methods."""
    click.echo("\n📚 Authentication Method Guide")
    click.echo("=" * 30)
    click.echo()
    click.echo("🔐 Claude Pro/Max OAuth:")
    click.echo("   • Best if you have a Claude Pro or Claude Max subscription")
    click.echo("   • Uses your existing subscription (no extra API costs)")
    click.echo("   • Automatic token management and refresh")
    click.echo("   • Most convenient for regular Claude users")
    click.echo()
    click.echo("🔑 API Key:")
    click.echo("   • Best for programmatic access or if you don't have Claude Pro/Max")
    click.echo("   • Requires separate API subscription (~$20/month minimum)")
    click.echo("   • Direct API access with manual key management")
    click.echo("   • Good for production applications")
    click.echo()
    click.echo("💡 Recommendation:")
    click.echo("   Choose Claude Pro/Max if you already have a subscription.")
    click.echo("   Choose API Key if you need programmatic access or don't have Claude Pro/Max.")
    click.echo()


@auth.command("logout")
@click.argument("provider", required=False)
@click.option(
    "--all", "logout_all", is_flag=True, help="Logout from all OAuth providers"
)
def auth_logout(provider: str | None, logout_all: bool) -> None:
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


@cli.command()
@click.argument("ensemble_name")
@click.option("--port", default=3000, help="Port to serve MCP server on")
def serve(ensemble_name: str, port: int) -> None:
    """Serve an ensemble as an MCP server."""
    runner = MCPServerRunner(ensemble_name, port)
    runner.run()


if __name__ == "__main__":
    cli()
