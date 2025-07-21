"""Authentication management CLI commands."""

import time

import click

from llm_orc.cli_modules.utils.auth_utils import (
    handle_anthropic_interactive_auth,
    handle_claude_cli_auth,
    handle_claude_pro_max_oauth,
    show_auth_method_help,
    test_provider_authentication,
)
from llm_orc.cli_modules.utils.config_utils import show_provider_details
from llm_orc.core.auth.authentication import AuthenticationManager, CredentialStorage
from llm_orc.core.config.config_manager import ConfigurationManager


class AuthCommands:
    """Authentication management commands."""

    @staticmethod
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
                # Check if provider already exists and remove if so
                if provider in storage.list_providers():
                    click.echo(f"🔄 Existing authentication found for {provider}")
                    click.echo(
                        "   Removing old authentication before setting up new..."
                    )
                    storage.remove_provider(provider)
                    click.echo("✅ Old authentication removed")

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
                # Check if provider already exists and remove if so
                if provider in storage.list_providers():
                    click.echo(f"🔄 Existing authentication found for {provider}")
                    click.echo(
                        "   Removing old authentication before setting up new..."
                    )
                    storage.remove_provider(provider)
                    click.echo("✅ Old authentication removed")

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
            # Check if provider already exists and remove if so
            if provider in storage.list_providers():
                click.echo(f"🔄 Existing authentication found for {provider}")
                click.echo("   Removing old authentication before setting up new...")
                storage.remove_provider(provider)
                click.echo("✅ Old authentication removed")

            if api_key:
                # API key authentication
                storage.store_api_key(provider, api_key)
                click.echo(f"API key for {provider} added successfully")
            else:
                # OAuth authentication - we know these are not None due to validation
                # above
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

    @staticmethod
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
                    AuthCommands.auth_setup()
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
                    if confirm_action(
                        f"Remove authentication for {selected_provider}?"
                    ):
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
                            show_info(
                                "Re-authentication required for OAuth token refresh"
                            )
                            # This would typically trigger a re-auth flow
                            show_success("Token refresh would be performed here")
                        else:
                            show_error(
                                "Token refresh only available for OAuth providers"
                            )
                    except Exception as e:
                        show_error(f"Refresh failed: {str(e)}")

        except Exception as e:
            raise click.ClickException(f"Failed to list providers: {str(e)}") from e

    @staticmethod
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

    @staticmethod
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

            # Check what we have
            has_refresh_token = "refresh_token" in oauth_token
            has_client_id = "client_id" in oauth_token
            has_expires_at = "expires_at" in oauth_token

            click.echo(f"🔍 Token info for {provider}:")
            click.echo(f"  Has refresh token: {'✅' if has_refresh_token else '❌'}")
            click.echo(f"  Has client ID: {'✅' if has_client_id else '❌'}")
            click.echo(f"  Has expiration: {'✅' if has_expires_at else '❌'}")

            if has_expires_at:
                expires_at = oauth_token["expires_at"]
                now = time.time()
                if expires_at > now:
                    remaining = int(expires_at - now)
                    hours, remainder = divmod(remaining, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    click.echo(f"  Token expires in: {hours}h {minutes}m {seconds}s")
                else:
                    expired_for = int(now - expires_at)
                    click.echo(f"  ⚠️ Token expired {expired_for} seconds ago")

            if not has_refresh_token:
                click.echo("\n❌ Cannot test refresh: no refresh token available")
                return

            if not has_client_id:
                # Use default client ID for anthropic-claude-pro-max
                if provider == "anthropic-claude-pro-max":
                    client_id = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
                    click.echo(f"\n🔧 Using default client ID: {client_id}")
                else:
                    click.echo("\n❌ Cannot test refresh: no client ID available")
                    return
            else:
                client_id = oauth_token["client_id"]

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

    @staticmethod
    def auth_setup() -> None:
        """Interactive setup wizard for authentication."""
        from llm_orc.menu_system import (
            AuthMenus,
            confirm_action,
            show_error,
            show_success,
            show_working,
        )
        from llm_orc.providers.registry import provider_registry

        config_manager = ConfigurationManager()
        storage = CredentialStorage(config_manager)
        auth_manager = AuthenticationManager(storage)

        click.echo("🚀 Welcome to LLM Orchestra setup!")
        click.echo(
            "This wizard will help you configure authentication for LLM providers."
        )

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

            # Check if provider already exists and offer to replace
            if provider_key in storage.list_providers():
                click.echo(
                    f"\n🔄 Existing authentication found for {provider.display_name}"
                )
                if confirm_action("Replace existing authentication?"):
                    storage.remove_provider(provider_key)
                    show_success(
                        f"Removed existing authentication for {provider.display_name}"
                    )
                else:
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
                elif (
                    auth_method == "oauth"
                    and provider_key == "anthropic-claude-pro-max"
                ):
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

    @staticmethod
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
                raise click.ClickException(
                    "Must specify a provider name or use --all flag"
                )

        except Exception as e:
            raise click.ClickException(f"Failed to logout: {str(e)}") from e


# Module-level exports for CLI imports
add_auth_provider = AuthCommands.add_auth_provider
list_auth_providers = AuthCommands.list_auth_providers
remove_auth_provider = AuthCommands.remove_auth_provider
test_token_refresh = AuthCommands.test_token_refresh
logout_oauth_providers = AuthCommands.logout_oauth_providers
