"""CLI authentication command implementations."""

import base64
import hashlib
import secrets
import shutil
import time
import webbrowser
from urllib.parse import urlencode

import click
import requests

from llm_orc.authentication import AuthenticationManager, CredentialStorage


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
        setup_anthropic_oauth(auth_manager, "anthropic-claude-pro-max")
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
        setup_anthropic_oauth(auth_manager, "anthropic-claude-pro-max")
        click.echo("✅ OAuth configured as 'anthropic-claude-pro-max'")


def setup_anthropic_oauth(
    auth_manager: AuthenticationManager, provider_key: str
) -> None:
    """Set up Anthropic OAuth authentication."""
    from llm_orc.authentication import AnthropicOAuthFlow

    oauth_flow = AnthropicOAuthFlow.create_with_guidance()

    if not auth_manager.authenticate_oauth(
        provider_key, oauth_flow.client_id, oauth_flow.client_secret
    ):
        raise click.ClickException("OAuth authentication failed")


def handle_claude_pro_max_oauth(
    auth_manager: AuthenticationManager, storage: CredentialStorage
) -> None:
    """Handle Claude Pro/Max OAuth authentication setup using hardcoded client ID."""
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


def show_auth_method_help() -> None:
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
    click.echo(
        "   Choose API Key if you need programmatic access or don't have "
        "Claude Pro/Max."
    )
    click.echo()


def test_provider_authentication(
    storage: CredentialStorage, auth_manager: AuthenticationManager, provider: str
) -> bool:
    """Test authentication for a specific provider."""
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
