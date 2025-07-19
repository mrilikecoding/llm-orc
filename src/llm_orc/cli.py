"""Command line interface for llm-orc."""

import click

from llm_orc.cli_commands import (
    add_auth_provider,
    check_global_config,
    check_local_config,
    init_local_config,
    invoke_ensemble,
    list_auth_providers,
    list_ensembles_command,
    list_profiles_command,
    logout_oauth_providers,
    remove_auth_provider,
    reset_global_config,
    reset_local_config,
    serve_ensemble,
)


@click.group()
@click.version_option(package_name="llm-orchestra")
def cli() -> None:
    """LLM Orchestra - Multi-agent LLM communication system."""
    pass


@cli.command()
@click.argument("ensemble_name")
@click.argument("input_data", required=False)
@click.option(
    "--config-dir",
    default=None,
    help="Directory containing ensemble configurations",
)
@click.option(
    "--input-data",
    "--input",
    "input_data_option",
    default=None,
    help="Input data for the ensemble (alternative to positional argument)",
)
@click.option(
    "--output-format",
    type=click.Choice(["json", "text"]),
    default="text",
    help="Output format for results",
)
@click.option(
    "--streaming",
    is_flag=True,
    help="Enable streaming execution for real-time progress updates",
)
@click.option(
    "--max-concurrent",
    type=int,
    default=None,
    help="Maximum number of concurrent agents (overrides config)",
)
@click.option(
    "--detailed",
    is_flag=True,
    help="Show detailed results and performance metrics",
)
def invoke(
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
    invoke_ensemble(
        ensemble_name,
        input_data,
        config_dir,
        input_data_option,
        output_format,
        streaming,
        max_concurrent,
        detailed,
    )


@cli.command("list-ensembles")
@click.option(
    "--config-dir",
    default=None,
    help="Directory containing ensemble configurations",
)
def list_ensembles(config_dir: str | None) -> None:
    """List available ensembles."""
    list_ensembles_command(config_dir)


@cli.command("list-profiles")
def list_profiles() -> None:
    """List available model profiles with their provider/model details."""
    list_profiles_command()


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
def init(project_name: str | None) -> None:
    """Initialize local .llm-orc configuration for current project."""
    init_local_config(project_name)


@config.command("reset-global")
@click.option(
    "--backup/--no-backup",
    default=True,
    help="Create backup of existing global config (default: True)",
)
@click.option(
    "--preserve-auth/--reset-auth",
    default=True,
    help="Preserve existing authentication credentials (default: True)",
)
@click.confirmation_option(
    prompt="This will reset your global LLM Orchestra configuration. Continue?"
)
def reset_global(backup: bool, preserve_auth: bool) -> None:
    """Reset global configuration to template defaults."""
    reset_global_config(backup, preserve_auth)


@config.command("check-global")
def check_global() -> None:
    """Check global configuration status."""
    check_global_config()


@config.command("reset-local")
@click.option(
    "--backup/--no-backup",
    default=True,
    help="Create backup of existing local config (default: True)",
)
@click.option(
    "--preserve-ensembles/--reset-ensembles",
    default=True,
    help="Preserve existing ensembles directory (default: True)",
)
@click.option(
    "--project-name",
    default=None,
    help="Name for the project (defaults to directory name)",
)
@click.confirmation_option(
    prompt="This will reset your local .llm-orc configuration. Continue?"
)
def reset_local(
    backup: bool, preserve_ensembles: bool, project_name: str | None
) -> None:
    """Reset local .llm-orc configuration to template defaults."""
    reset_local_config(backup, preserve_ensembles, project_name)


@config.command("check")
def check() -> None:
    """Check both global and local configuration status."""
    # Show legend at the top
    click.echo("Configuration Status Legend:")
    click.echo("🟢 Ready to use (provider authenticated/available)")
    click.echo("🟥 Needs setup (provider not authenticated/available)")
    click.echo("=" * 50)

    # Show global config first
    check_global_config()

    # Add separator
    click.echo("\n" + "=" * 50)

    # Show local config
    check_local_config()
    click.echo("=" * 50)


@config.command("check-local")
def check_local() -> None:
    """Check local .llm-orc configuration status."""
    check_local_config()


@cli.group()
def auth() -> None:
    """Authentication management commands."""
    pass


@auth.command("add")
@click.argument("provider")
@click.option("--api-key", help="API key for the provider")
@click.option("--client-id", help="OAuth client ID")
@click.option("--client-secret", help="OAuth client secret")
def auth_add(
    provider: str,
    api_key: str | None,
    client_id: str | None,
    client_secret: str | None,
) -> None:
    """Add authentication for a provider (API key or OAuth)."""
    add_auth_provider(provider, api_key, client_id, client_secret)


@auth.command("list")
@click.option(
    "--interactive", "-i", is_flag=True, help="Show interactive menu with actions"
)
def auth_list(interactive: bool) -> None:
    """List configured authentication providers."""
    list_auth_providers(interactive)


@auth.command("remove")
@click.argument("provider")
def auth_remove(provider: str) -> None:
    """Remove authentication for a provider."""
    remove_auth_provider(provider)


@auth.command("setup")
def auth_setup_command() -> None:
    """Interactive setup wizard for authentication."""
    from llm_orc.cli_commands import auth_setup as auth_setup_impl

    auth_setup_impl()


@auth.command("logout")
@click.argument("provider", required=False)
@click.option(
    "--all", "logout_all", is_flag=True, help="Logout from all OAuth providers"
)
def auth_logout(provider: str | None, logout_all: bool) -> None:
    """Logout from OAuth providers (revokes tokens and removes credentials)."""
    logout_oauth_providers(provider, logout_all)


@cli.command()
@click.argument("ensemble_name")
@click.option("--port", default=3000, help="Port to serve MCP server on")
def serve(ensemble_name: str, port: int) -> None:
    """Serve an ensemble as an MCP server."""
    serve_ensemble(ensemble_name, port)


# Help command that shows main help with aliases
@cli.command()
def help_command() -> None:
    """Show help for llm-orc commands."""
    ctx = click.get_current_context()
    if not ctx.parent:
        click.echo("Help not available")
        return

    # Custom help that shows aliases alongside commands
    click.echo("Usage: llm-orc [OPTIONS] COMMAND [ARGS]...")
    click.echo()
    click.echo("  LLM Orchestra - Multi-agent LLM communication system.")
    click.echo()
    click.echo("Options:")
    click.echo("  --version  Show the version and exit.")
    click.echo("  --help     Show this message and exit.")
    click.echo()
    click.echo("Commands:")

    # Command mappings with their aliases
    commands_with_aliases = [
        ("auth", "a", "Authentication management commands."),
        ("config", "c", "Configuration management commands."),
        ("help", "h", "Show help for llm-orc commands."),
        ("invoke", "i", "Invoke an ensemble of agents."),
        ("list-ensembles", "le", "List available ensembles."),
        (
            "list-profiles",
            "lp",
            "List available model profiles with their provider/model...",
        ),
        ("serve", "s", "Serve an ensemble as an MCP server."),
    ]

    for cmd, alias, desc in commands_with_aliases:
        click.echo(f"  {cmd:<15} ({alias:<2}) {desc}")

    click.echo()
    click.echo("You can use either the full command name or its alias.")
    click.echo(
        "Example: 'llm-orc invoke simple \"test\"' or 'llm-orc i simple \"test\"'"
    )


# Add command shortcuts for all top-level commands
cli.add_command(invoke, name="i")
cli.add_command(auth, name="a")
cli.add_command(config, name="c")
cli.add_command(list_ensembles, name="le")
cli.add_command(list_profiles, name="lp")
cli.add_command(serve, name="s")
cli.add_command(help_command, name="help")
cli.add_command(help_command, name="h")


if __name__ == "__main__":
    cli()
