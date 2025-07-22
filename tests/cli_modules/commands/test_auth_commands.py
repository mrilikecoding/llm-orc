"""Comprehensive tests for auth commands module."""

import time
from unittest.mock import Mock, patch

import click
import pytest

from llm_orc.cli_modules.commands.auth_commands import AuthCommands


class TestAddAuthProvider:
    """Test adding authentication providers."""

    @patch("llm_orc.cli_modules.commands.auth_commands.ConfigurationManager")
    @patch("llm_orc.cli_modules.commands.auth_commands.CredentialStorage")
    @patch("llm_orc.cli_modules.commands.auth_commands.AuthenticationManager")
    def test_add_api_key_provider(
        self,
        mock_auth_manager_class: Mock,
        mock_storage_class: Mock,
        mock_config_class: Mock,
    ) -> None:
        """Test adding provider with API key."""
        # Given
        mock_config = Mock()
        mock_storage = Mock()
        mock_auth_manager = Mock()

        mock_config_class.return_value = mock_config
        mock_storage_class.return_value = mock_storage
        mock_auth_manager_class.return_value = mock_auth_manager

        mock_storage.list_providers.return_value = []

        # When
        AuthCommands.add_auth_provider(
            provider="test-provider",
            api_key="test_api_key",
            client_id=None,
            client_secret=None,
        )

        # Then
        mock_storage.store_api_key.assert_called_once_with(
            "test-provider", "test_api_key"
        )

    @patch("llm_orc.cli_modules.commands.auth_commands.ConfigurationManager")
    @patch("llm_orc.cli_modules.commands.auth_commands.CredentialStorage")
    @patch("llm_orc.cli_modules.commands.auth_commands.AuthenticationManager")
    def test_add_oauth_provider(
        self,
        mock_auth_manager_class: Mock,
        mock_storage_class: Mock,
        mock_config_class: Mock,
    ) -> None:
        """Test adding provider with OAuth credentials."""
        # Given
        mock_config = Mock()
        mock_storage = Mock()
        mock_auth_manager = Mock()

        mock_config_class.return_value = mock_config
        mock_storage_class.return_value = mock_storage
        mock_auth_manager_class.return_value = mock_auth_manager

        mock_storage.list_providers.return_value = []
        mock_auth_manager.authenticate_oauth.return_value = True

        # When
        AuthCommands.add_auth_provider(
            provider="test-provider",
            api_key=None,
            client_id="test_client_id",
            client_secret="test_client_secret",
        )

        # Then
        mock_auth_manager.authenticate_oauth.assert_called_once_with(
            "test-provider", "test_client_id", "test_client_secret"
        )

    @patch("llm_orc.cli_modules.commands.auth_commands.handle_claude_cli_auth")
    @patch("llm_orc.cli_modules.commands.auth_commands.ConfigurationManager")
    @patch("llm_orc.cli_modules.commands.auth_commands.CredentialStorage")
    def test_add_claude_cli_provider(
        self,
        mock_storage_class: Mock,
        mock_config_class: Mock,
        mock_handle_claude_cli: Mock,
    ) -> None:
        """Test adding Claude CLI provider."""
        # Given
        mock_config = Mock()
        mock_storage = Mock()

        mock_config_class.return_value = mock_config
        mock_storage_class.return_value = mock_storage

        # When
        AuthCommands.add_auth_provider(
            provider="claude-cli",
            api_key=None,
            client_id=None,
            client_secret=None,
        )

        # Then
        mock_handle_claude_cli.assert_called_once_with(mock_storage)

    def test_add_provider_validation_both_credentials(self) -> None:
        """Test validation error when both API key and OAuth provided."""
        # When/Then
        with pytest.raises(
            click.ClickException,
            match="Cannot use both API key and OAuth credentials"
        ):
            AuthCommands.add_auth_provider(
                provider="test-provider",
                api_key="test_key",
                client_id="test_id",
                client_secret="test_secret",
            )

    def test_add_provider_validation_no_credentials(self) -> None:
        """Test validation error when no credentials provided."""
        # When/Then
        with pytest.raises(
            click.ClickException,
            match=(
                "Must provide either --api-key or both --client-id and "
                "--client-secret"
            ),
        ):
            AuthCommands.add_auth_provider(
                provider="test-provider",
                api_key=None,
                client_id=None,
                client_secret=None,
            )

    def test_add_provider_validation_incomplete_oauth(self) -> None:
        """Test validation error when OAuth credentials incomplete."""
        # When/Then
        with pytest.raises(
            click.ClickException,
            match=(
                "Must provide either --api-key or both --client-id and "
                "--client-secret"
            ),
        ):
            AuthCommands.add_auth_provider(
                provider="test-provider",
                api_key=None,
                client_id="test_id",
                client_secret=None,
            )

    @patch("llm_orc.cli_modules.commands.auth_commands.ConfigurationManager")
    @patch("llm_orc.cli_modules.commands.auth_commands.CredentialStorage")
    @patch("llm_orc.cli_modules.commands.auth_commands.AuthenticationManager")
    def test_add_provider_replaces_existing(
        self,
        mock_auth_manager_class: Mock,
        mock_storage_class: Mock,
        mock_config_class: Mock,
    ) -> None:
        """Test replacing existing provider."""
        # Given
        mock_config = Mock()
        mock_storage = Mock()
        mock_auth_manager = Mock()

        mock_config_class.return_value = mock_config
        mock_storage_class.return_value = mock_storage
        mock_auth_manager_class.return_value = mock_auth_manager

        mock_storage.list_providers.return_value = ["test-provider"]

        # When
        AuthCommands.add_auth_provider(
            provider="test-provider",
            api_key="test_api_key",
            client_id=None,
            client_secret=None,
        )

        # Then
        mock_storage.remove_provider.assert_called_once_with("test-provider")
        mock_storage.store_api_key.assert_called_once_with(
            "test-provider", "test_api_key"
        )

    @patch("llm_orc.cli_modules.commands.auth_commands.ConfigurationManager")
    @patch("llm_orc.cli_modules.commands.auth_commands.CredentialStorage")
    @patch("llm_orc.cli_modules.commands.auth_commands.AuthenticationManager")
    def test_add_oauth_provider_failure(
        self,
        mock_auth_manager_class: Mock,
        mock_storage_class: Mock,
        mock_config_class: Mock,
    ) -> None:
        """Test OAuth provider addition failure."""
        # Given
        mock_config = Mock()
        mock_storage = Mock()
        mock_auth_manager = Mock()

        mock_config_class.return_value = mock_config
        mock_storage_class.return_value = mock_storage
        mock_auth_manager_class.return_value = mock_auth_manager

        mock_storage.list_providers.return_value = []
        mock_auth_manager.authenticate_oauth.return_value = False

        # When/Then
        with pytest.raises(
            click.ClickException,
            match="OAuth authentication for test-provider failed"
        ):
            AuthCommands.add_auth_provider(
                provider="test-provider",
                api_key=None,
                client_id="test_client_id",
                client_secret="test_client_secret",
            )


class TestListAuthProviders:
    """Test listing authentication providers."""

    @patch("llm_orc.cli_modules.commands.auth_commands.ConfigurationManager")
    @patch("llm_orc.cli_modules.commands.auth_commands.CredentialStorage")
    @patch("llm_orc.cli_modules.commands.auth_commands.AuthenticationManager")
    def test_list_providers_non_interactive_empty(
        self,
        mock_auth_manager_class: Mock,
        mock_storage_class: Mock,
        mock_config_class: Mock,
    ) -> None:
        """Test listing providers when none configured (non-interactive)."""
        # Given
        mock_config = Mock()
        mock_storage = Mock()
        mock_auth_manager = Mock()

        mock_config_class.return_value = mock_config
        mock_storage_class.return_value = mock_storage
        mock_auth_manager_class.return_value = mock_auth_manager

        mock_storage.list_providers.return_value = []

        # When
        with patch("click.echo") as mock_echo:
            AuthCommands.list_auth_providers(interactive=False)

        # Then
        mock_echo.assert_called_once_with("No authentication providers configured")

    @patch("llm_orc.cli_modules.commands.auth_commands.ConfigurationManager")
    @patch("llm_orc.cli_modules.commands.auth_commands.CredentialStorage")
    @patch("llm_orc.cli_modules.commands.auth_commands.AuthenticationManager")
    def test_list_providers_non_interactive_with_providers(
        self,
        mock_auth_manager_class: Mock,
        mock_storage_class: Mock,
        mock_config_class: Mock,
    ) -> None:
        """Test listing providers when configured (non-interactive)."""
        # Given
        mock_config = Mock()
        mock_storage = Mock()
        mock_auth_manager = Mock()

        mock_config_class.return_value = mock_config
        mock_storage_class.return_value = mock_storage
        mock_auth_manager_class.return_value = mock_auth_manager

        mock_storage.list_providers.return_value = ["provider1", "provider2"]
        mock_storage.get_auth_method.side_effect = ["api_key", "oauth"]

        # When
        with patch("click.echo") as mock_echo:
            AuthCommands.list_auth_providers(interactive=False)

        # Then
        assert mock_echo.call_count == 3
        mock_echo.assert_any_call("Configured providers:")
        mock_echo.assert_any_call("  provider1: API key")
        mock_echo.assert_any_call("  provider2: OAuth")


class TestRemoveAuthProvider:
    """Test removing authentication providers."""

    @patch("llm_orc.cli_modules.commands.auth_commands.ConfigurationManager")
    @patch("llm_orc.cli_modules.commands.auth_commands.CredentialStorage")
    def test_remove_existing_provider(
        self,
        mock_storage_class: Mock,
        mock_config_class: Mock,
    ) -> None:
        """Test removing existing provider."""
        # Given
        mock_config = Mock()
        mock_storage = Mock()

        mock_config_class.return_value = mock_config
        mock_storage_class.return_value = mock_storage

        mock_storage.list_providers.return_value = ["test-provider"]

        # When
        with patch("click.echo") as mock_echo:
            AuthCommands.remove_auth_provider("test-provider")

        # Then
        mock_storage.remove_provider.assert_called_once_with("test-provider")
        mock_echo.assert_called_once_with("Authentication for test-provider removed")

    @patch("llm_orc.cli_modules.commands.auth_commands.ConfigurationManager")
    @patch("llm_orc.cli_modules.commands.auth_commands.CredentialStorage")
    def test_remove_nonexistent_provider(
        self,
        mock_storage_class: Mock,
        mock_config_class: Mock,
    ) -> None:
        """Test removing non-existent provider."""
        # Given
        mock_config = Mock()
        mock_storage = Mock()

        mock_config_class.return_value = mock_config
        mock_storage_class.return_value = mock_storage

        mock_storage.list_providers.return_value = []

        # When/Then
        with pytest.raises(
            click.ClickException,
            match="No authentication found for test-provider"
        ):
            AuthCommands.remove_auth_provider("test-provider")


class TestTokenRefresh:
    """Test OAuth token refresh functionality."""

    @patch("llm_orc.cli_modules.commands.auth_commands.ConfigurationManager")
    @patch("llm_orc.cli_modules.commands.auth_commands.CredentialStorage")
    def test_refresh_nonexistent_provider(
        self,
        mock_storage_class: Mock,
        mock_config_class: Mock,
    ) -> None:
        """Test refreshing tokens for non-existent provider."""
        # Given
        mock_config = Mock()
        mock_storage = Mock()

        mock_config_class.return_value = mock_config
        mock_storage_class.return_value = mock_storage

        mock_storage.list_providers.return_value = []

        # When/Then
        with pytest.raises(
            click.ClickException,
            match="No authentication found for test-provider"
        ):
            AuthCommands.test_token_refresh("test-provider")

    @patch("llm_orc.cli_modules.commands.auth_commands.ConfigurationManager")
    @patch("llm_orc.cli_modules.commands.auth_commands.CredentialStorage")
    def test_refresh_non_oauth_provider(
        self,
        mock_storage_class: Mock,
        mock_config_class: Mock,
    ) -> None:
        """Test refreshing tokens for non-OAuth provider."""
        # Given
        mock_config = Mock()
        mock_storage = Mock()

        mock_config_class.return_value = mock_config
        mock_storage_class.return_value = mock_storage

        mock_storage.list_providers.return_value = ["test-provider"]
        mock_storage.get_oauth_token.return_value = None

        # When/Then
        with pytest.raises(
            click.ClickException,
            match="No OAuth token found for test-provider"
        ):
            AuthCommands.test_token_refresh("test-provider")

    @patch("llm_orc.cli_modules.commands.auth_commands.ConfigurationManager")
    @patch("llm_orc.cli_modules.commands.auth_commands.CredentialStorage")
    def test_refresh_token_info_display(
        self,
        mock_storage_class: Mock,
        mock_config_class: Mock,
    ) -> None:
        """Test token info display for OAuth provider."""
        # Given
        mock_config = Mock()
        mock_storage = Mock()

        mock_config_class.return_value = mock_config
        mock_storage_class.return_value = mock_storage

        mock_storage.list_providers.return_value = ["test-provider"]
        oauth_token = {
            "access_token": "test_token",
            "refresh_token": "test_refresh",
            "client_id": "test_client",
            "expires_at": time.time() + 3600,  # Expires in 1 hour
        }
        mock_storage.get_oauth_token.return_value = oauth_token

        # When
        with patch("click.echo") as mock_echo:
            AuthCommands.test_token_refresh("test-provider")

        # Then
        echo_calls = [call[0][0] for call in mock_echo.call_args_list]
        token_info_found = any(
            "Token info for test-provider" in call for call in echo_calls
        )
        assert token_info_found

    @patch("time.time")
    @patch("llm_orc.cli_modules.commands.auth_commands.ConfigurationManager")
    @patch("llm_orc.cli_modules.commands.auth_commands.CredentialStorage")
    def test_refresh_expired_token(
        self,
        mock_storage_class: Mock,
        mock_config_class: Mock,
        mock_time: Mock,
    ) -> None:
        """Test token refresh with expired token."""
        # Given
        mock_config = Mock()
        mock_storage = Mock()

        mock_config_class.return_value = mock_config
        mock_storage_class.return_value = mock_storage

        current_time = 1000000
        mock_time.return_value = current_time

        mock_storage.list_providers.return_value = ["test-provider"]
        oauth_token = {
            "access_token": "test_token",
            "refresh_token": "test_refresh",
            "client_id": "test_client",
            "expires_at": current_time - 3600,  # Expired 1 hour ago
        }
        mock_storage.get_oauth_token.return_value = oauth_token

        # When
        with patch("click.echo") as mock_echo:
            AuthCommands.test_token_refresh("test-provider")

        # Then
        echo_calls = [call[0][0] for call in mock_echo.call_args_list]
        expired_found = any("Token expired" in call for call in echo_calls)
        assert expired_found


class TestSpecialProviderHandling:
    """Test special provider handling (claude-cli, anthropic, etc)."""

    @patch("llm_orc.cli_modules.commands.auth_commands.handle_claude_cli_auth")
    @patch("llm_orc.cli_modules.commands.auth_commands.ConfigurationManager")
    @patch("llm_orc.cli_modules.commands.auth_commands.CredentialStorage")
    def test_claude_cli_auth_error(
        self,
        mock_storage_class: Mock,
        mock_config_class: Mock,
        mock_handle_claude_cli: Mock,
    ) -> None:
        """Test Claude CLI authentication error handling."""
        # Given
        mock_config = Mock()
        mock_storage = Mock()

        mock_config_class.return_value = mock_config
        mock_storage_class.return_value = mock_storage
        mock_handle_claude_cli.side_effect = Exception("Claude CLI error")

        # When/Then
        with pytest.raises(
            click.ClickException,
            match="Failed to set up Claude CLI authentication: Claude CLI error"
        ):
            AuthCommands.add_auth_provider(
                provider="claude-cli",
                api_key=None,
                client_id=None,
                client_secret=None,
            )

    @patch("llm_orc.cli_modules.commands.auth_commands.handle_claude_pro_max_oauth")
    @patch("llm_orc.cli_modules.commands.auth_commands.ConfigurationManager")
    @patch("llm_orc.cli_modules.commands.auth_commands.CredentialStorage")
    @patch("llm_orc.cli_modules.commands.auth_commands.AuthenticationManager")
    def test_claude_pro_max_oauth_with_existing(
        self,
        mock_auth_manager_class: Mock,
        mock_storage_class: Mock,
        mock_config_class: Mock,
        mock_handle_oauth: Mock,
    ) -> None:
        """Test Claude Pro/Max OAuth with existing provider."""
        # Given
        mock_config = Mock()
        mock_storage = Mock()
        mock_auth_manager = Mock()

        mock_config_class.return_value = mock_config
        mock_storage_class.return_value = mock_storage
        mock_auth_manager_class.return_value = mock_auth_manager

        mock_storage.list_providers.return_value = ["anthropic-claude-pro-max"]

        # When
        with patch("click.echo"):
            AuthCommands.add_auth_provider(
                provider="anthropic-claude-pro-max",
                api_key=None,
                client_id=None,
                client_secret=None,
            )

        # Then
        mock_storage.remove_provider.assert_called_once_with("anthropic-claude-pro-max")
        mock_handle_oauth.assert_called_once_with(mock_auth_manager, mock_storage)

    @patch("llm_orc.cli_modules.commands.auth_commands.handle_claude_pro_max_oauth")
    @patch("llm_orc.cli_modules.commands.auth_commands.ConfigurationManager")
    @patch("llm_orc.cli_modules.commands.auth_commands.CredentialStorage")
    @patch("llm_orc.cli_modules.commands.auth_commands.AuthenticationManager")
    def test_claude_pro_max_oauth_error(
        self,
        mock_auth_manager_class: Mock,
        mock_storage_class: Mock,
        mock_config_class: Mock,
        mock_handle_oauth: Mock,
    ) -> None:
        """Test Claude Pro/Max OAuth error handling."""
        # Given
        mock_config = Mock()
        mock_storage = Mock()
        mock_auth_manager = Mock()

        mock_config_class.return_value = mock_config
        mock_storage_class.return_value = mock_storage
        mock_auth_manager_class.return_value = mock_auth_manager

        mock_storage.list_providers.return_value = []
        mock_handle_oauth.side_effect = Exception("OAuth error")

        # When/Then
        with pytest.raises(
            click.ClickException,
            match="Failed to set up Claude Pro/Max OAuth authentication: OAuth error"
        ):
            AuthCommands.add_auth_provider(
                provider="anthropic-claude-pro-max",
                api_key=None,
                client_id=None,
                client_secret=None,
            )

    @patch("llm_orc.cli_modules.commands.auth_commands.handle_anthropic_interactive_auth")
    @patch("llm_orc.cli_modules.commands.auth_commands.ConfigurationManager")
    @patch("llm_orc.cli_modules.commands.auth_commands.CredentialStorage")
    @patch("llm_orc.cli_modules.commands.auth_commands.AuthenticationManager")
    def test_anthropic_interactive_auth_with_existing(
        self,
        mock_auth_manager_class: Mock,
        mock_storage_class: Mock,
        mock_config_class: Mock,
        mock_handle_interactive: Mock,
    ) -> None:
        """Test Anthropic interactive authentication with existing provider."""
        # Given
        mock_config = Mock()
        mock_storage = Mock()
        mock_auth_manager = Mock()

        mock_config_class.return_value = mock_config
        mock_storage_class.return_value = mock_storage
        mock_auth_manager_class.return_value = mock_auth_manager

        mock_storage.list_providers.return_value = ["anthropic"]

        # When
        with patch("click.echo"):
            AuthCommands.add_auth_provider(
                provider="anthropic",
                api_key=None,
                client_id=None,
                client_secret=None,
            )

        # Then
        mock_storage.remove_provider.assert_called_once_with("anthropic")
        mock_handle_interactive.assert_called_once_with(mock_auth_manager, mock_storage)

    @patch("llm_orc.cli_modules.commands.auth_commands.handle_anthropic_interactive_auth")
    @patch("llm_orc.cli_modules.commands.auth_commands.ConfigurationManager")
    @patch("llm_orc.cli_modules.commands.auth_commands.CredentialStorage")
    @patch("llm_orc.cli_modules.commands.auth_commands.AuthenticationManager")
    def test_anthropic_interactive_auth_error(
        self,
        mock_auth_manager_class: Mock,
        mock_storage_class: Mock,
        mock_config_class: Mock,
        mock_handle_interactive: Mock,
    ) -> None:
        """Test Anthropic interactive authentication error handling."""
        # Given
        mock_config = Mock()
        mock_storage = Mock()
        mock_auth_manager = Mock()

        mock_config_class.return_value = mock_config
        mock_storage_class.return_value = mock_storage
        mock_auth_manager_class.return_value = mock_auth_manager

        mock_storage.list_providers.return_value = []
        mock_handle_interactive.side_effect = Exception("Interactive auth error")

        # When/Then
        with pytest.raises(
            click.ClickException,
            match="Failed to set up Anthropic authentication: Interactive auth error"
        ):
            AuthCommands.add_auth_provider(
                provider="anthropic",
                api_key=None,
                client_id=None,
                client_secret=None,
            )


class TestTokenRefreshAdvanced:
    """Test advanced token refresh functionality."""

    @patch("llm_orc.cli_modules.commands.auth_commands.ConfigurationManager")
    @patch("llm_orc.cli_modules.commands.auth_commands.CredentialStorage")
    def test_refresh_no_refresh_token(
        self,
        mock_storage_class: Mock,
        mock_config_class: Mock,
    ) -> None:
        """Test token refresh when no refresh token available."""
        # Given
        mock_config = Mock()
        mock_storage = Mock()

        mock_config_class.return_value = mock_config
        mock_storage_class.return_value = mock_storage

        mock_storage.list_providers.return_value = ["test-provider"]
        oauth_token = {
            "access_token": "test_token",
            "client_id": "test_client",
            "expires_at": time.time() + 3600,
        }
        mock_storage.get_oauth_token.return_value = oauth_token

        # When
        with patch("click.echo") as mock_echo:
            AuthCommands.test_token_refresh("test-provider")

        # Then
        echo_calls = [call[0][0] for call in mock_echo.call_args_list]
        no_refresh_found = any(
            "Cannot test refresh: no refresh token available" in call
            for call in echo_calls
        )
        assert no_refresh_found

    @patch("llm_orc.cli_modules.commands.auth_commands.ConfigurationManager")
    @patch("llm_orc.cli_modules.commands.auth_commands.CredentialStorage")
    def test_refresh_no_client_id_non_anthropic(
        self,
        mock_storage_class: Mock,
        mock_config_class: Mock,
    ) -> None:
        """Test token refresh when no client ID for non-anthropic provider."""
        # Given
        mock_config = Mock()
        mock_storage = Mock()

        mock_config_class.return_value = mock_config
        mock_storage_class.return_value = mock_storage

        mock_storage.list_providers.return_value = ["test-provider"]
        oauth_token = {
            "access_token": "test_token",
            "refresh_token": "test_refresh",
            "expires_at": time.time() + 3600,
        }
        mock_storage.get_oauth_token.return_value = oauth_token

        # When
        with patch("click.echo") as mock_echo:
            AuthCommands.test_token_refresh("test-provider")

        # Then
        echo_calls = [call[0][0] for call in mock_echo.call_args_list]
        no_client_id_found = any(
            "Cannot test refresh: no client ID available" in call
            for call in echo_calls
        )
        assert no_client_id_found

    # NOTE: Advanced token refresh tests with OAuthClaudeClient are
    # intentionally omitted
    # because they would trigger real OAuth HTTP requests even with mocking
    # The token info display tests above provide sufficient coverage for safe testing


class TestLogoutOAuthProviders:
    """Test OAuth logout functionality."""

    @patch("llm_orc.cli_modules.commands.auth_commands.ConfigurationManager")
    @patch("llm_orc.cli_modules.commands.auth_commands.CredentialStorage")
    @patch("llm_orc.cli_modules.commands.auth_commands.AuthenticationManager")
    def test_logout_single_provider_success(
        self,
        mock_auth_manager_class: Mock,
        mock_storage_class: Mock,
        mock_config_class: Mock,
    ) -> None:
        """Test successful logout from single provider."""
        # Given
        mock_config = Mock()
        mock_storage = Mock()
        mock_auth_manager = Mock()

        mock_config_class.return_value = mock_config
        mock_storage_class.return_value = mock_storage
        mock_auth_manager_class.return_value = mock_auth_manager

        mock_auth_manager.logout_oauth_provider.return_value = True

        # When
        with patch("click.echo") as mock_echo:
            AuthCommands.logout_oauth_providers(
                provider="test-provider", logout_all=False
            )

        # Then
        mock_auth_manager.logout_oauth_provider.assert_called_once_with("test-provider")
        mock_echo.assert_called_once_with("✅ Logged out from test-provider")

    @patch("llm_orc.cli_modules.commands.auth_commands.ConfigurationManager")
    @patch("llm_orc.cli_modules.commands.auth_commands.CredentialStorage")
    @patch("llm_orc.cli_modules.commands.auth_commands.AuthenticationManager")
    def test_logout_single_provider_failure(
        self,
        mock_auth_manager_class: Mock,
        mock_storage_class: Mock,
        mock_config_class: Mock,
    ) -> None:
        """Test failed logout from single provider."""
        # Given
        mock_config = Mock()
        mock_storage = Mock()
        mock_auth_manager = Mock()

        mock_config_class.return_value = mock_config
        mock_storage_class.return_value = mock_storage
        mock_auth_manager_class.return_value = mock_auth_manager

        mock_auth_manager.logout_oauth_provider.return_value = False

        # When/Then
        with pytest.raises(
            click.ClickException,
            match="Failed to logout from test-provider"
        ):
            AuthCommands.logout_oauth_providers(
                provider="test-provider", logout_all=False
            )

    @patch("llm_orc.cli_modules.commands.auth_commands.ConfigurationManager")
    @patch("llm_orc.cli_modules.commands.auth_commands.CredentialStorage")
    @patch("llm_orc.cli_modules.commands.auth_commands.AuthenticationManager")
    def test_logout_all_providers_success(
        self,
        mock_auth_manager_class: Mock,
        mock_storage_class: Mock,
        mock_config_class: Mock,
    ) -> None:
        """Test successful logout from all providers."""
        # Given
        mock_config = Mock()
        mock_storage = Mock()
        mock_auth_manager = Mock()

        mock_config_class.return_value = mock_config
        mock_storage_class.return_value = mock_storage
        mock_auth_manager_class.return_value = mock_auth_manager

        mock_auth_manager.logout_all_oauth_providers.return_value = {
            "provider1": True,
            "provider2": False,
        }

        # When
        with patch("click.echo") as mock_echo:
            AuthCommands.logout_oauth_providers(provider=None, logout_all=True)

        # Then
        mock_auth_manager.logout_all_oauth_providers.assert_called_once()
        echo_calls = [call[0][0] for call in mock_echo.call_args_list]
        assert any("Logged out from 1 OAuth providers" in call for call in echo_calls)

    @patch("llm_orc.cli_modules.commands.auth_commands.ConfigurationManager")
    @patch("llm_orc.cli_modules.commands.auth_commands.CredentialStorage")
    @patch("llm_orc.cli_modules.commands.auth_commands.AuthenticationManager")
    def test_logout_all_providers_none_found(
        self,
        mock_auth_manager_class: Mock,
        mock_storage_class: Mock,
        mock_config_class: Mock,
    ) -> None:
        """Test logout when no OAuth providers found."""
        # Given
        mock_config = Mock()
        mock_storage = Mock()
        mock_auth_manager = Mock()

        mock_config_class.return_value = mock_config
        mock_storage_class.return_value = mock_storage
        mock_auth_manager_class.return_value = mock_auth_manager

        mock_auth_manager.logout_all_oauth_providers.return_value = {}

        # When
        with patch("click.echo") as mock_echo:
            AuthCommands.logout_oauth_providers(provider=None, logout_all=True)

        # Then
        mock_echo.assert_called_once_with("No OAuth providers found to logout")

    def test_logout_no_provider_or_all_flag(self) -> None:
        """Test error when neither provider nor all flag specified."""
        # When/Then
        with pytest.raises(
            click.ClickException,
            match="Must specify a provider name or use --all flag"
        ):
            AuthCommands.logout_oauth_providers(provider=None, logout_all=False)

