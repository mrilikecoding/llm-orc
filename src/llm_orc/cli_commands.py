"""Main CLI command implementations."""

import asyncio
import sys
from pathlib import Path

import click
import yaml

from llm_orc.cli_modules.commands import AuthCommands, ConfigCommands
from llm_orc.cli_modules.utils.config_utils import (
    display_local_profiles,
    get_available_providers,
)
from llm_orc.cli_modules.utils.visualization import (
    run_standard_execution,
    run_streaming_execution,
)
from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.core.config.ensemble_config import EnsembleLoader
from llm_orc.core.execution.ensemble_execution import EnsembleExecutor
from llm_orc.integrations.mcp.runner import MCPServerRunner


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
    ConfigCommands.init_local_config(project_name)


def reset_global_config(backup: bool, preserve_auth: bool) -> None:
    """Reset global configuration to template defaults."""
    ConfigCommands.reset_global_config(backup, preserve_auth)


def check_global_config() -> None:
    """Check global configuration status."""
    ConfigCommands.check_global_config()


def check_local_config() -> None:
    """Check local .llm-orc configuration status."""
    ConfigCommands.check_local_config()


def reset_local_config(
    backup: bool, preserve_ensembles: bool, project_name: str | None
) -> None:
    """Reset local .llm-orc configuration to template defaults."""
    ConfigCommands.reset_local_config(backup, preserve_ensembles, project_name)


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
    AuthCommands.add_auth_provider(provider, api_key, client_id, client_secret)


def list_auth_providers(interactive: bool) -> None:
    """List configured authentication providers."""
    AuthCommands.list_auth_providers(interactive)


def remove_auth_provider(provider: str) -> None:
    """Remove authentication for a provider."""
    AuthCommands.remove_auth_provider(provider)


def test_token_refresh(provider: str) -> None:
    """Test OAuth token refresh for a specific provider."""
    AuthCommands.test_token_refresh(provider)


def auth_setup() -> None:
    """Interactive setup wizard for authentication."""
    AuthCommands.auth_setup()


def logout_oauth_providers(provider: str | None, logout_all: bool) -> None:
    """Logout from OAuth providers (revokes tokens and removes credentials)."""
    AuthCommands.logout_oauth_providers(provider, logout_all)
