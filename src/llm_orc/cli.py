"""Command line interface for llm-orc."""

import click

from llm_orc.cli_commands import (
    invoke_ensemble,
    list_ensembles_command,
    list_profiles_command,
)
from llm_orc.cli_completion import (
    complete_ensemble_names,
    complete_library_ensemble_paths,
    complete_providers,
)
from llm_orc.cli_library.library import (
    browse_library,
    copy_ensemble,
    list_categories,
    show_ensemble_info,
)
from llm_orc.cli_modules.commands.auth_commands import (
    add_auth_provider,
    auth_setup,
    list_auth_providers,
    logout_oauth_providers,
    remove_auth_provider,
    test_token_refresh,
)
from llm_orc.cli_modules.commands.config_commands import (
    check_global_config,
    check_local_config,
    init_local_config,
    reset_global_config,
    reset_local_config,
)
from llm_orc.cli_modules.commands.validation_commands import (
    validate_all_ensembles,
    validate_ensemble,
    validate_ensemble_category,
)


@click.group()
@click.version_option(package_name="llm-orchestra")
def cli() -> None:
    """LLM Orchestra - Multi-agent LLM communication system."""
    pass


@cli.command()
@click.option(
    "--shell",
    type=click.Choice(["bash", "zsh", "fish"], case_sensitive=False),
    help="Shell type for completion script (auto-detected if not specified)",
)
def completion(shell: str | None) -> None:
    """Generate shell completion script for llm-orc.

    To enable tab completion, run one of these commands:

    Bash:
      eval "$(_LLM_ORC_COMPLETE=bash_source llm-orc completion)"

    Zsh:
      eval "$(_LLM_ORC_COMPLETE=zsh_source llm-orc completion)"

    Fish:
      _LLM_ORC_COMPLETE=fish_source llm-orc completion | source

    You can also add the appropriate line to your shell's config file
    (~/.bashrc, ~/.zshrc, ~/.config/fish/config.fish) to enable completion permanently.
    """
    import os

    # Get shell from environment if not specified
    if shell is None:
        shell_env = os.environ.get("SHELL", "").split("/")[-1]
        if shell_env in ["bash", "zsh", "fish"]:
            shell = shell_env
        else:
            shell = "bash"  # Default to bash

    shell = shell.lower()
    complete_var = f"_LLM_ORC_COMPLETE={shell}_source"

    click.echo(f"# Tab completion for llm-orc ({shell})")
    click.echo("# Add this line to your shell config file:")

    if shell == "fish":
        click.echo(f"{complete_var} llm-orc completion | source")
    else:
        click.echo(f'eval "$({complete_var} llm-orc completion)"')


@cli.command()
@click.argument("ensemble_name", shell_complete=complete_ensemble_names)
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
    default=None,
    help="Output format for results (default: rich streaming interface)",
)
@click.option(
    "--streaming/--no-streaming",
    default=True,
    help="Enable streaming execution for real-time progress updates",
)
@click.option(
    "--max-concurrent",
    type=int,
    default=None,
    help="Maximum number of concurrent agents (overrides config)",
)
@click.option(
    "--detailed/--no-detailed",
    default=True,
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


@cli.command()
@click.option(
    "--project-name",
    default=None,
    help="Name for the project (defaults to directory name)",
)
def init(project_name: str | None) -> None:
    """Initialize llm-orc project with examples and configuration.

    Creates .llm-orc/ directory with:
      - Example ensembles demonstrating patterns
      - Configuration templates

    \b
    Examples:
      # Initialize with all defaults
      llm-orc init

      # Specify project name
      llm-orc init --project-name my-ensemble-project

    Ready! After init, try:
      llm-orc scripts list          # See installed primitives
      llm-orc list-ensembles        # See example ensembles
    """
    init_local_config(project_name)


@cli.group()
def config() -> None:
    """Configuration management commands."""
    pass


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


@cli.group()
def library() -> None:
    """Library management commands for browsing and copying ensembles."""
    pass


@cli.group()
def scripts() -> None:
    """Script management commands."""
    pass


@cli.group()
def artifacts() -> None:
    """Artifact management commands."""
    pass


@cli.group()
def validate() -> None:
    """Validation commands for testing ensembles."""
    pass


@cli.group()
def mcp() -> None:
    """MCP (Model Context Protocol) server commands."""
    pass


@cli.command()
@click.option("--port", default=8765, help="Port to run the web server on")
@click.option(
    "--host",
    default="127.0.0.1",
    help="Host to bind to (use 0.0.0.0 for network access)",
)
@click.option("--open", "open_browser", is_flag=True, help="Open browser automatically")
def web(port: int, host: str, open_browser: bool) -> None:
    """Start the web UI server for ensemble management.

    Provides a browser-based interface for:
    - Browsing and managing ensembles
    - Executing ensembles with real-time output
    - Viewing execution artifacts and metrics
    - Managing model profiles and scripts
    """
    import webbrowser

    import uvicorn

    from llm_orc.web.server import create_app

    url = f"http://{host}:{port}"

    if host == "0.0.0.0":
        click.echo(
            "WARNING: Binding to 0.0.0.0 exposes the server to your network",
            err=True,
        )

    click.echo(f"Starting llm-orc web UI at {url}", err=True)
    click.echo("Press Ctrl+C to stop", err=True)

    if open_browser:
        # Open browser after a short delay to let server start
        import threading

        def open_browser_delayed() -> None:
            import time

            time.sleep(1)
            webbrowser.open(url)

        threading.Thread(target=open_browser_delayed, daemon=True).start()

    app = create_app()
    uvicorn.run(app, host=host, port=port, log_level="warning")


@mcp.command("serve")
@click.option(
    "--transport",
    type=click.Choice(["stdio", "http"]),
    default="stdio",
    help="Transport type (stdio for MCP clients, http for debugging)",
)
@click.option("--port", default=8080, help="Port for HTTP transport")
def mcp_serve(transport: str, port: int) -> None:
    """Start the MCP server exposing all ensembles.

    Uses the MCPServer with full resource and tool support.
    Default transport is stdio for MCP client compatibility.
    """
    import signal
    import sys

    from llm_orc.mcp import MCPServer

    def handle_shutdown(_signum: int, _frame: object) -> None:
        """Handle shutdown signals gracefully."""
        click.echo("\nShutting down MCP server...", err=True)
        sys.exit(0)

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    server = MCPServer()

    if transport == "stdio":
        # Minimal output for stdio - it's typically auto-spawned by MCP clients
        # and the output goes to their logs, not a user terminal
        click.echo("llm-orc MCP server ready", err=True)
        try:
            server.run()
        except KeyboardInterrupt:
            click.echo("Shutting down", err=True)
    else:
        # Detailed output for HTTP - user is manually running for web UI or debugging
        click.echo(f"MCP server ready at http://localhost:{port}/mcp/", err=True)
        click.echo("", err=True)
        click.echo("Endpoints:", err=True)
        click.echo(f"  SSE stream: http://localhost:{port}/mcp/sse", err=True)
        click.echo(f"  Messages:   http://localhost:{port}/mcp/messages", err=True)
        click.echo("", err=True)
        click.echo("Press Ctrl+C to stop", err=True)
        try:
            server.run(transport="http", port=port)
        except KeyboardInterrupt:
            click.echo("\nShutting down MCP server...", err=True)


@scripts.command("list")
@click.option(
    "--format",
    "format_type",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format",
)
def scripts_list(format_type: str) -> None:
    """List available scripts."""
    from llm_orc.cli_commands import scripts_list_command

    scripts_list_command(format_type)


@scripts.command("show")
@click.argument("name")
def scripts_show(name: str) -> None:
    """Show script documentation."""
    from llm_orc.cli_commands import scripts_show_command

    scripts_show_command(name)


@scripts.command("test")
@click.argument("name")
@click.option("--parameters", help="JSON parameters for the script")
def scripts_test(name: str, parameters: str | None) -> None:
    """Test script with parameters."""
    from llm_orc.cli_commands import scripts_test_command

    scripts_test_command(name, parameters)


@artifacts.command("list")
@click.option(
    "--format",
    "format_type",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format",
)
def artifacts_list(format_type: str) -> None:
    """List execution artifacts."""
    from llm_orc.cli_commands import artifacts_list_command

    artifacts_list_command(format_type)


@artifacts.command("show")
@click.argument("name")
@click.option(
    "--format",
    "format_type",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format",
)
@click.option("--execution", help="Specific execution timestamp")
def artifacts_show(name: str, format_type: str, execution: str | None) -> None:
    """Show latest results."""
    from llm_orc.cli_commands import artifacts_show_command

    artifacts_show_command(name, format_type, execution)


@validate.command()
@click.argument("ensemble_name", shell_complete=complete_ensemble_names)
@click.option("--verbose", is_flag=True, help="Show detailed validation output")
@click.option(
    "--config-dir",
    default=None,
    help="Directory containing ensemble configurations",
)
def run(ensemble_name: str, verbose: bool, config_dir: str | None) -> None:
    """Validate a single ensemble."""
    validate_ensemble(ensemble_name, verbose, config_dir)


@validate.command()
@click.option(
    "--category",
    required=True,
    help="Validation category (primitive, integration, etc.)",
)
@click.option("--verbose", is_flag=True, help="Show detailed validation output")
def category(category: str, verbose: bool) -> None:
    """Validate ensembles by category."""
    validate_ensemble_category(category, verbose)


@validate.command("all")
@click.option("--verbose", is_flag=True, help="Show detailed validation output")
def validate_all_command(verbose: bool) -> None:
    """Validate all validation ensembles."""
    validate_all_ensembles(verbose)


@library.command()
@click.argument("category", required=False)
def browse(category: str | None) -> None:
    """Browse available ensembles by category."""
    browse_library(category)


@library.command()
@click.argument("ensemble_path", shell_complete=complete_library_ensemble_paths)
@click.option(
    "--global", "is_global", is_flag=True, help="Copy to global config instead of local"
)
def copy(ensemble_path: str, is_global: bool) -> None:
    """Copy an ensemble from the library to your config."""
    copy_ensemble(ensemble_path, is_global)


@library.command()
def categories() -> None:
    """List all available ensemble categories."""
    list_categories()


@library.command()
@click.argument("ensemble_path", shell_complete=complete_library_ensemble_paths)
def show(ensemble_path: str) -> None:
    """Show detailed information about an ensemble."""
    show_ensemble_info(ensemble_path)


@auth.command("add")
@click.argument("provider", shell_complete=complete_providers)
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
@click.argument("provider", shell_complete=complete_providers)
def auth_remove(provider: str) -> None:
    """Remove authentication for a provider."""
    remove_auth_provider(provider)


@auth.command("setup")
def auth_setup_command() -> None:
    """Interactive setup wizard for authentication."""
    auth_setup()


@auth.command("logout")
@click.argument("provider", required=False, shell_complete=complete_providers)
@click.option(
    "--all", "logout_all", is_flag=True, help="Logout from all OAuth providers"
)
def auth_logout(provider: str | None, logout_all: bool) -> None:
    """Logout from OAuth providers (revokes tokens and removes credentials)."""
    logout_oauth_providers(provider, logout_all)


@auth.command("test-refresh")
@click.argument("provider", shell_complete=complete_providers)
def auth_test_refresh(provider: str) -> None:
    """Test OAuth token refresh for a provider."""
    test_token_refresh(provider)


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
        ("artifacts", "ar", "Artifact management commands."),
        ("auth", "a", "Authentication management commands."),
        ("config", "c", "Configuration management commands."),
        ("help", "h", "Show help for llm-orc commands."),
        ("invoke", "i", "Invoke an ensemble of agents."),
        (
            "library",
            "l",
            "Library management commands for browsing and copying ensembles.",
        ),
        ("list-ensembles", "le", "List available ensembles."),
        (
            "list-profiles",
            "lp",
            "List available model profiles with their provider/model...",
        ),
        ("scripts", "sc", "Script management commands."),
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
cli.add_command(library, name="l")
cli.add_command(scripts, name="sc")
cli.add_command(artifacts, name="ar")
cli.add_command(list_ensembles, name="le")
cli.add_command(list_profiles, name="lp")
cli.add_command(mcp, name="m")
cli.add_command(web, name="w")
cli.add_command(help_command, name="help")
cli.add_command(help_command, name="h")


if __name__ == "__main__":
    cli()
