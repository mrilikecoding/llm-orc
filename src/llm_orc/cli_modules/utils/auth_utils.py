"""Authentication utility functions for CLI operations."""

import shutil
import time

import click

from llm_orc.core.auth.authentication import AuthenticationManager, CredentialStorage


def handle_claude_cli_auth(storage: CredentialStorage) -> None:
    """Handle Claude CLI authentication setup."""
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


def handle_anthropic_interactive_auth(
    auth_manager: AuthenticationManager, storage: CredentialStorage
) -> None:
    """Handle interactive Anthropic authentication setup."""
    api_key = click.prompt("Anthropic API key", hide_input=True)
    storage.store_api_key("anthropic-api", api_key)
    click.echo("✅ API key configured as 'anthropic-api'")


def show_auth_method_help() -> None:
    """Show help for choosing authentication methods."""
    click.echo("\n📚 Authentication Method Guide")
    click.echo("=" * 30)
    click.echo()
    click.echo("🔑 API Key:")
    click.echo("   • Direct API access with your Anthropic API key")
    click.echo("   • Get your key at console.anthropic.com")
    click.echo("   • Good for production applications")
    click.echo()
    click.echo("🔐 OAuth:")
    click.echo("   • OAuth authentication for providers that support it")
    click.echo("   • Automatic token management and refresh")
    click.echo()


def validate_provider_authentication(
    storage: CredentialStorage, auth_manager: AuthenticationManager, provider: str
) -> bool:
    """Validate authentication for a specific provider."""
    auth_method = storage.get_auth_method(provider)
    if not auth_method:
        return False

    success = False
    if auth_method == "api_key":
        api_key = storage.get_api_key(provider)
        if api_key:
            success = auth_manager.authenticate(provider, api_key)
    elif auth_method == "oauth":
        oauth_token = storage.get_oauth_token(provider)
        if oauth_token:
            # For OAuth, we'll consider it successful if we have a valid token
            if "expires_at" in oauth_token:
                success = time.time() < oauth_token["expires_at"]
            else:
                success = True  # No expiration info, assume valid

    return success
