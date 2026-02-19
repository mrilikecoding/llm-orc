"""Authentication management CLI commands."""

import time
from typing import TYPE_CHECKING, Any

import click

from llm_orc.cli_modules.utils.auth_utils import (
    handle_anthropic_interactive_auth,
    handle_claude_cli_auth,
    handle_claude_pro_max_oauth,
)
from llm_orc.core.auth.authentication import AuthenticationManager, CredentialStorage
from llm_orc.core.config.config_manager import ConfigurationManager

if TYPE_CHECKING:
    from llm_orc.providers.registry import ProviderInfo


def _handle_special_provider(
    provider: str,
    api_key: str | None,
    client_id: str | None,
    client_secret: str | None,
    storage: CredentialStorage,
    auth_manager: AuthenticationManager,
) -> bool:
    """Handle special provider auth flows. Returns True if handled."""
    provider_lower = provider.lower()

    if provider_lower == "claude-cli":
        handle_claude_cli_auth(storage)
        return True

    if provider_lower == "anthropic-claude-pro-max":
        _remove_existing_provider(storage, provider)
        handle_claude_pro_max_oauth(auth_manager, storage)
        return True

    is_anthropic_interactive = (
        provider_lower == "anthropic"
        and not api_key
        and not (client_id and client_secret)
    )
    if is_anthropic_interactive:
        _remove_existing_provider(storage, provider)
        handle_anthropic_interactive_auth(auth_manager, storage)
        return True

    return False


def _store_credentials(
    provider: str,
    api_key: str | None,
    client_id: str | None,
    client_secret: str | None,
    storage: CredentialStorage,
    auth_manager: AuthenticationManager,
) -> None:
    """Store API key or OAuth credentials for a provider."""
    _remove_existing_provider(storage, provider)

    if api_key:
        storage.store_api_key(provider, api_key)
        click.echo(f"API key for {provider} added successfully")
        return

    assert client_id is not None
    assert client_secret is not None
    if auth_manager.authenticate_oauth(provider, client_id, client_secret):
        click.echo(f"OAuth authentication for {provider} completed successfully")
    else:
        raise click.ClickException(f"OAuth authentication for {provider} failed")


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

    try:
        if _handle_special_provider(
            provider, api_key, client_id, client_secret, storage, auth_manager
        ):
            return
    except Exception as e:
        raise click.ClickException(
            f"Failed to set up {provider} authentication: {str(e)}"
        ) from e

    _validate_auth_credentials(api_key, client_id, client_secret)

    try:
        _store_credentials(
            provider, api_key, client_id, client_secret, storage, auth_manager
        )
    except Exception as e:
        raise click.ClickException(f"Failed to add authentication: {str(e)}") from e


def list_auth_providers(interactive: bool) -> None:
    """List configured authentication providers."""
    from llm_orc.menu_system import AuthMenus

    config_manager = ConfigurationManager()
    storage = CredentialStorage(config_manager)
    auth_manager = AuthenticationManager(storage)

    try:
        providers = storage.list_providers()

        if not interactive:
            # Simple list view using helper method
            _display_simple_provider_list(providers, storage)
            return

        # Interactive mode with action menu
        while True:
            action, selected_provider = AuthMenus.auth_list_actions(providers)

            # Handle action using helper method
            result = _handle_interactive_action(
                action, selected_provider, storage, auth_manager
            )

            if result is None:
                break  # Quit requested
            elif result is True:
                # Refresh provider list needed
                providers = storage.list_providers()

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


def test_token_refresh(provider: str) -> None:
    """Test OAuth token refresh for a specific provider."""
    config_manager = ConfigurationManager()
    storage = CredentialStorage(config_manager)

    try:
        # Check if provider exists
        if provider not in storage.list_providers():
            raise click.ClickException(f"No authentication found for {provider}")

        # Get OAuth token info
        oauth_token = storage.get_oauth_token(provider)
        if not oauth_token:
            raise click.ClickException(f"No OAuth token found for {provider}")

        # Analyze token structure and display status
        token_analysis = _analyze_token_info(oauth_token)
        _display_token_status(provider, token_analysis)

        # Check if refresh is possible
        if not token_analysis["has_refresh_token"]:
            click.echo("\n❌ Cannot test refresh: no refresh token available")
            return

        # Resolve client ID (with provider-specific defaults)
        client_id = _resolve_client_id(provider, token_analysis, oauth_token)
        if not client_id:
            return

        # Test the refresh
        click.echo(f"\n🔄 Testing token refresh for {provider}...")

        from llm_orc.core.auth.oauth_client import OAuthClaudeClient

        client = OAuthClaudeClient(
            access_token=oauth_token["access_token"],
            refresh_token=oauth_token["refresh_token"],
        )

        if client.refresh_access_token(client_id):
            click.echo("✅ Token refresh successful!")

            # Update stored credentials
            storage.store_oauth_token(
                provider,
                client.access_token,
                client.refresh_token,
                expires_at=int(time.time()) + 3600,
                client_id=client_id,
            )
            click.echo("✅ Updated stored credentials")
        else:
            click.echo("❌ Token refresh failed!")
            click.echo("Check the error messages above for details.")

    except Exception as e:
        raise click.ClickException(f"Failed to test token refresh: {str(e)}") from e


def auth_setup() -> None:
    """Interactive setup wizard for authentication."""
    from llm_orc.menu_system import AuthMenus, show_error
    from llm_orc.providers.registry import provider_registry

    # Initialize managers and show welcome message
    config_manager, storage, auth_manager = _initialize_auth_setup_managers()

    while True:
        # Use interactive menu for provider selection
        provider_key = AuthMenus.provider_selection()

        # Get provider info
        provider = provider_registry.get_provider(provider_key)

        if not provider:
            show_error(f"Provider '{provider_key}' not found in registry")
            continue

        # Process the selected provider
        result = _process_single_provider(provider_key, provider, storage, auth_manager)

        if result == "break":
            break

    # Display completion message
    _show_setup_completion_message()


def _handle_existing_provider(
    storage: "CredentialStorage",
    provider_key: str,
    provider_display_name: str,
) -> bool | None:
    """Handle existing provider authentication replacement logic.

    Returns:
        True: Continue with provider setup
        False: Skip this provider but continue with others
        None: Exit setup entirely
    """
    from llm_orc.menu_system import confirm_action, show_success

    if provider_key not in storage.list_providers():
        return True  # Provider doesn't exist, continue with setup

    click.echo(f"\n🔄 Existing authentication found for {provider_display_name}")

    if confirm_action("Replace existing authentication?"):
        storage.remove_provider(provider_key)
        show_success(f"Removed existing authentication for {provider_display_name}")
        return True  # Continue with setup
    else:
        if confirm_action("Add another provider?"):
            return False  # Skip this provider but continue
        else:
            return None  # Exit setup entirely


def _determine_auth_method(provider_key: str) -> str:
    """Determine authentication method for a provider.

    Returns:
        str: The authentication method ('oauth', 'api_key', or 'help')
    """
    from llm_orc.menu_system import AuthMenus

    # Hard-coded auth methods for specific providers
    if provider_key == "anthropic-claude-pro-max":
        return "oauth"  # Claude Pro/Max only supports OAuth
    elif provider_key == "anthropic-api":
        return "api_key"  # Anthropic API only supports API key
    elif provider_key == "google-gemini":
        return "api_key"  # Google Gemini only supports API key
    else:
        # For other providers, use the menu system
        return AuthMenus.get_auth_method_for_provider(provider_key)


def _handle_authentication_setup(
    auth_method: str,
    provider_key: str,
    provider: "ProviderInfo",
    storage: "CredentialStorage",
    auth_manager: "AuthenticationManager",
) -> bool:
    """Handle authentication setup based on method.

    Returns:
        bool: True to continue with next iteration (help case),
              False to continue with next provider (normal case)
    """
    from llm_orc.cli_modules.utils.auth_utils import (
        handle_claude_pro_max_oauth,
    )
    from llm_orc.menu_system import show_error, show_success, show_working

    if auth_method == "help":
        from llm_orc.cli_modules.utils.auth_utils import show_auth_method_help

        show_auth_method_help()
        return True  # Continue with next iteration
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
        api_key = click.prompt(f"{provider.display_name} API key", hide_input=True)
        storage.store_api_key(provider_key, api_key)
        show_success(f"{provider.display_name} API key configured!")
    elif auth_method == "oauth":
        # Generic OAuth setup for other providers
        client_id = click.prompt("OAuth client ID")
        client_secret = click.prompt("OAuth client secret", hide_input=True)

        if auth_manager.authenticate_oauth(provider_key, client_id, client_secret):
            show_success(f"{provider.display_name} OAuth configured!")
        else:
            show_error(f"OAuth authentication for {provider.display_name} failed")
    else:
        show_error(f"Unknown authentication method: {auth_method}")

    return False  # Continue with next provider


def _display_simple_provider_list(
    providers: list[str], storage: "CredentialStorage | None" = None
) -> None:
    """Display a simple list of configured providers.

    Args:
        providers: List of provider names
        storage: Optional credential storage for auth method details
    """
    if not providers:
        click.echo("No authentication providers configured")
        return

    click.echo("Configured providers:")
    for provider in providers:
        if storage:
            auth_method = storage.get_auth_method(provider)
            if auth_method == "oauth":
                click.echo(f"  {provider}: OAuth")
            else:
                click.echo(f"  {provider}: API key")
        else:
            click.echo(f"  {provider}")


def _handle_interactive_action(
    action: str,
    selected_provider: str | None,
    storage: "CredentialStorage",
    auth_manager: "AuthenticationManager",
) -> bool | None:
    """Handle interactive action for provider management.

    Args:
        action: The action to perform
        selected_provider: The selected provider (if any)
        storage: Credential storage instance
        auth_manager: Authentication manager instance

    Returns:
        True: Refresh provider list needed
        False: No refresh needed
        None: Quit/exit requested
    """
    from llm_orc.cli_modules.utils.config_utils import show_provider_details
    from llm_orc.menu_system import (
        confirm_action,
        show_success,
    )

    if action == "quit":
        return None
    elif action == "setup" or action == "add":
        # Run the setup wizard
        auth_setup()
        return True  # Refresh provider list
    elif action == "test" and selected_provider:
        return _handle_provider_test(selected_provider, storage, auth_manager)
    elif action == "remove" and selected_provider:
        if confirm_action(f"Remove authentication for {selected_provider}?"):
            storage.remove_provider(selected_provider)
            show_success(f"Removed {selected_provider}")
            return True  # Refresh provider list
        return False
    elif action == "details" and selected_provider:
        show_provider_details(storage, selected_provider)
        return False
    elif action == "refresh" and selected_provider:
        return _handle_token_refresh(selected_provider, storage)

    return False


def _handle_provider_test(
    selected_provider: str,
    storage: "CredentialStorage",
    auth_manager: "AuthenticationManager",
) -> bool:
    """Handle provider authentication testing logic.

    Args:
        selected_provider: Provider name to test
        storage: Credential storage instance
        auth_manager: Authentication manager instance

    Returns:
        bool: False (no provider list refresh needed)
    """
    from llm_orc.cli_modules.utils.auth_utils import (
        validate_provider_authentication,
    )
    from llm_orc.menu_system import show_error, show_success, show_working

    show_working(f"Testing {selected_provider}...")
    try:
        success = validate_provider_authentication(
            storage, auth_manager, selected_provider
        )
        if success:
            show_success(f"Authentication for {selected_provider} is working!")
        else:
            show_error(f"Authentication for {selected_provider} failed")
    except Exception as e:
        show_error(f"Test failed: {str(e)}")
    return False


def _handle_token_refresh(selected_provider: str, storage: "CredentialStorage") -> bool:
    """Handle OAuth token refresh logic.

    Args:
        selected_provider: Provider name to refresh tokens for
        storage: Credential storage instance

    Returns:
        bool: False (no provider list refresh needed)
    """
    from llm_orc.menu_system import (
        show_error,
        show_info,
        show_success,
        show_working,
    )

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
    return False


def _remove_existing_provider(storage: "CredentialStorage", provider: str) -> None:
    """Remove existing provider authentication if it exists.

    Args:
        storage: Credential storage instance
        provider: Provider name to check and remove
    """
    if provider in storage.list_providers():
        click.echo(f"🔄 Existing authentication found for {provider}")
        click.echo("   Removing old authentication before setting up new...")
        storage.remove_provider(provider)
        click.echo("✅ Old authentication removed")


def _validate_auth_credentials(
    api_key: str | None,
    client_id: str | None,
    client_secret: str | None,
) -> None:
    """Validate authentication credentials for non-interactive flow.

    Args:
        api_key: API key (if provided)
        client_id: OAuth client ID (if provided)
        client_secret: OAuth client secret (if provided)

    Raises:
        click.ClickException: If validation fails
    """
    if api_key and (client_id or client_secret):
        raise click.ClickException("Cannot use both API key and OAuth credentials")

    if not api_key and not (client_id and client_secret):
        raise click.ClickException(
            "Must provide either --api-key or both --client-id and --client-secret"
        )


def logout_oauth_providers(provider: str | None, logout_all: bool) -> None:
    """Logout from OAuth providers (revokes tokens and removes credentials)."""
    config_manager = ConfigurationManager()
    storage = CredentialStorage(config_manager)
    auth_manager = AuthenticationManager(storage)

    try:
        if logout_all:
            _handle_all_providers_logout(auth_manager)
        elif provider:
            _handle_single_provider_logout(provider, auth_manager)
        else:
            raise click.ClickException("Must specify a provider name or use --all flag")

    except Exception as e:
        raise click.ClickException(f"Failed to logout: {str(e)}") from e


def _analyze_token_info(oauth_token: dict[str, Any]) -> dict[str, Any]:
    """Analyze OAuth token structure and extract useful information.

    Args:
        oauth_token: OAuth token dictionary from storage

    Returns:
        Dictionary containing analysis results with boolean flags and data
    """
    has_refresh_token = "refresh_token" in oauth_token
    has_client_id = "client_id" in oauth_token
    has_expires_at = "expires_at" in oauth_token

    token_analysis: dict[str, Any] = {
        "has_refresh_token": has_refresh_token,
        "has_client_id": has_client_id,
        "has_expires_at": has_expires_at,
    }

    if has_expires_at:
        expires_at = oauth_token["expires_at"]
        now = time.time()
        token_analysis["expires_at"] = expires_at
        token_analysis["current_time"] = now
        token_analysis["is_expired"] = expires_at <= now

        if expires_at > now:
            remaining = int(expires_at - now)
            hours, remainder = divmod(remaining, 3600)
            minutes, seconds = divmod(remainder, 60)
            token_analysis["time_remaining"] = {
                "hours": hours,
                "minutes": minutes,
                "seconds": seconds,
            }
        else:
            expired_for = int(now - expires_at)
            token_analysis["expired_for_seconds"] = expired_for

    return token_analysis


def _display_token_status(provider: str, token_analysis: dict[str, Any]) -> None:
    """Display token status information to the user.

    Args:
        provider: Provider name
        token_analysis: Token analysis results from _analyze_token_info
    """
    click.echo(f"🔍 Token info for {provider}:")
    refresh_status = "✅" if token_analysis["has_refresh_token"] else "❌"
    client_status = "✅" if token_analysis["has_client_id"] else "❌"
    expires_status = "✅" if token_analysis["has_expires_at"] else "❌"

    click.echo(f"  Has refresh token: {refresh_status}")
    click.echo(f"  Has client ID: {client_status}")
    click.echo(f"  Has expiration: {expires_status}")

    if token_analysis["has_expires_at"]:
        if "time_remaining" in token_analysis:
            time_info = token_analysis["time_remaining"]
            click.echo(
                f"  Token expires in: {time_info['hours']}h "
                f"{time_info['minutes']}m {time_info['seconds']}s"
            )
        elif "expired_for_seconds" in token_analysis:
            expired_for = token_analysis["expired_for_seconds"]
            click.echo(f"  ⚠️ Token expired {expired_for} seconds ago")


def _resolve_client_id(
    provider: str, token_analysis: dict[str, Any], oauth_token: dict[str, Any]
) -> str | None:
    """Resolve client ID for token refresh, using defaults when needed.

    Args:
        provider: Provider name
        token_analysis: Token analysis results
        oauth_token: OAuth token dictionary

    Returns:
        Client ID string if available, None if cannot be resolved
    """
    if token_analysis["has_client_id"]:
        return str(oauth_token["client_id"])

    # Use default client ID for anthropic-claude-pro-max
    if provider == "anthropic-claude-pro-max":
        default_client_id = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
        click.echo(f"\n🔧 Using default client ID: {default_client_id}")
        return default_client_id

    click.echo("\n❌ Cannot test refresh: no client ID available")
    return None


def _initialize_auth_setup_managers() -> tuple[
    "ConfigurationManager", "CredentialStorage", "AuthenticationManager"
]:
    """Initialize managers needed for auth setup.

    Returns:
        Tuple of (config_manager, storage, auth_manager)
    """
    import click

    config_manager = ConfigurationManager()
    storage = CredentialStorage(config_manager)
    auth_manager = AuthenticationManager(storage)

    click.echo("🚀 Welcome to LLM Orchestra setup!")
    click.echo("This wizard will help you configure authentication for LLM providers.")

    return config_manager, storage, auth_manager


def _process_single_provider(
    provider_key: str,
    provider: "ProviderInfo",
    storage: "CredentialStorage",
    auth_manager: "AuthenticationManager",
) -> str:
    """Process a single provider during auth setup.

    Args:
        provider_key: The provider key
        provider: Provider information
        storage: Credential storage instance
        auth_manager: Authentication manager instance

    Returns:
        "break" to exit main loop, "continue" to continue with next provider
    """
    from llm_orc.menu_system import confirm_action, show_error, show_success

    if not provider.requires_auth:
        show_success(f"{provider.display_name} doesn't require authentication!")
        if not confirm_action("Add another provider?"):
            return "break"
        return "continue"

    # Handle existing provider authentication
    existing_result = _handle_existing_provider(
        storage, provider_key, provider.display_name
    )
    if existing_result is None:
        return "break"  # User chose to exit
    elif existing_result is False:
        return "continue"  # Skip this provider but continue with others

    # Determine authentication method for provider
    auth_method = _determine_auth_method(provider_key)

    # Handle authentication setup based on method
    try:
        should_continue = _handle_authentication_setup(
            auth_method, provider_key, provider, storage, auth_manager
        )
        if should_continue:
            return "continue"  # Help case - continue with next iteration
    except Exception as e:
        show_error(f"Failed to configure {provider.display_name}: {str(e)}")

    if not confirm_action("Add another provider?"):
        return "break"

    return "continue"


def _show_setup_completion_message() -> None:
    """Display setup completion message."""
    import click

    from llm_orc.menu_system import show_success

    click.echo()
    show_success(
        "Setup complete! Use 'llm-orc auth list' to see your configured providers."
    )


def _handle_all_providers_logout(auth_manager: "AuthenticationManager") -> None:
    """Handle logout from all OAuth providers.

    Args:
        auth_manager: Authentication manager instance
    """
    import click

    results = auth_manager.logout_all_oauth_providers()

    if not results:
        click.echo("No OAuth providers found to logout")
        return

    _display_logout_results(results)


def _handle_single_provider_logout(
    provider: str, auth_manager: "AuthenticationManager"
) -> None:
    """Handle logout from a single OAuth provider.

    Args:
        provider: Provider name to logout from
        auth_manager: Authentication manager instance

    Raises:
        click.ClickException: If logout fails
    """
    import click

    if auth_manager.logout_oauth_provider(provider):
        click.echo(f"✅ Logged out from {provider}")
    else:
        raise click.ClickException(
            f"Failed to logout from {provider}. "
            f"Provider may not exist or is not an OAuth provider."
        )


def _display_logout_results(results: dict[str, bool]) -> None:
    """Display logout results for multiple providers.

    Args:
        results: Dictionary mapping provider names to success status
    """
    import click

    success_count = sum(1 for success in results.values() if success)

    click.echo(f"Logged out from {success_count} OAuth providers:")
    for provider_name, success in results.items():
        status = "✅" if success else "❌"
        click.echo(f"  {provider_name}: {status}")
