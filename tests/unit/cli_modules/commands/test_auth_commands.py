"""Comprehensive tests for auth commands module."""

import time
from typing import Any
from unittest.mock import Mock, call, patch

import click
import pytest

from llm_orc.cli_modules.commands.auth_commands import (
    AuthCommands,
    _display_logout_results,
    _handle_all_providers_logout,
    _handle_single_provider_logout,
)


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
            click.ClickException, match="Cannot use both API key and OAuth credentials"
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
                "Must provide either --api-key or both --client-id and --client-secret"
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
                "Must provide either --api-key or both --client-id and --client-secret"
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
            click.ClickException, match="OAuth authentication for test-provider failed"
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

    @patch("llm_orc.cli_modules.commands.auth_commands.ConfigurationManager")
    @patch("llm_orc.cli_modules.commands.auth_commands.CredentialStorage")
    @patch("llm_orc.cli_modules.commands.auth_commands.AuthenticationManager")
    @patch("llm_orc.menu_system.AuthMenus")
    @patch("llm_orc.menu_system.show_working")
    @patch("llm_orc.menu_system.show_success")
    @patch("llm_orc.menu_system.show_error")
    @patch("llm_orc.cli_modules.utils.auth_utils.validate_provider_authentication")
    def test_list_providers_interactive_quit_action(
        self,
        mock_validate: Mock,
        mock_show_error: Mock,
        mock_show_success: Mock,
        mock_show_working: Mock,
        mock_auth_menus: Mock,
        mock_auth_manager_class: Mock,
        mock_storage_class: Mock,
        mock_config_class: Mock,
    ) -> None:
        """Test interactive listing with quit action."""
        # Given
        mock_config = Mock()
        mock_storage = Mock()
        mock_auth_manager = Mock()

        mock_config_class.return_value = mock_config
        mock_storage_class.return_value = mock_storage
        mock_auth_manager_class.return_value = mock_auth_manager

        providers = ["provider1", "provider2"]
        mock_storage.list_providers.return_value = providers

        # Mock menu to return quit action
        mock_auth_menus.auth_list_actions.return_value = ("quit", None)

        # When
        AuthCommands.list_auth_providers(interactive=True)

        # Then
        mock_auth_menus.auth_list_actions.assert_called_once_with(providers)

    @patch("llm_orc.cli_modules.commands.auth_commands.ConfigurationManager")
    @patch("llm_orc.cli_modules.commands.auth_commands.CredentialStorage")
    @patch("llm_orc.cli_modules.commands.auth_commands.AuthenticationManager")
    @patch("llm_orc.menu_system.AuthMenus")
    @patch("llm_orc.menu_system.show_working")
    @patch("llm_orc.menu_system.show_success")
    @patch("llm_orc.menu_system.show_error")
    @patch("llm_orc.cli_modules.utils.auth_utils.validate_provider_authentication")
    def test_list_providers_interactive_test_action_success(
        self,
        mock_validate: Mock,
        mock_show_error: Mock,
        mock_show_success: Mock,
        mock_show_working: Mock,
        mock_auth_menus: Mock,
        mock_auth_manager_class: Mock,
        mock_storage_class: Mock,
        mock_config_class: Mock,
    ) -> None:
        """Test interactive listing with test action that succeeds."""
        # Given
        mock_config = Mock()
        mock_storage = Mock()
        mock_auth_manager = Mock()

        mock_config_class.return_value = mock_config
        mock_storage_class.return_value = mock_storage
        mock_auth_manager_class.return_value = mock_auth_manager

        providers = ["provider1"]
        mock_storage.list_providers.return_value = providers

        # Mock menu to return test action, then quit
        mock_auth_menus.auth_list_actions.side_effect = [
            ("test", "provider1"),
            ("quit", None),
        ]

        # Mock successful validation
        mock_validate.return_value = True

        # When
        AuthCommands.list_auth_providers(interactive=True)

        # Then
        mock_show_working.assert_called_with("Testing provider1...")
        mock_validate.assert_called_once_with(
            mock_storage, mock_auth_manager, "provider1"
        )
        mock_show_success.assert_called_with("Authentication for provider1 is working!")

    @patch("llm_orc.cli_modules.commands.auth_commands.ConfigurationManager")
    @patch("llm_orc.cli_modules.commands.auth_commands.CredentialStorage")
    @patch("llm_orc.cli_modules.commands.auth_commands.AuthenticationManager")
    @patch("llm_orc.menu_system.AuthMenus")
    @patch("llm_orc.menu_system.show_working")
    @patch("llm_orc.menu_system.show_success")
    @patch("llm_orc.menu_system.show_error")
    @patch("llm_orc.cli_modules.utils.auth_utils.validate_provider_authentication")
    def test_list_providers_interactive_test_action_failure(
        self,
        mock_validate: Mock,
        mock_show_error: Mock,
        mock_show_success: Mock,
        mock_show_working: Mock,
        mock_auth_menus: Mock,
        mock_auth_manager_class: Mock,
        mock_storage_class: Mock,
        mock_config_class: Mock,
    ) -> None:
        """Test interactive listing with test action that fails."""
        # Given
        mock_config = Mock()
        mock_storage = Mock()
        mock_auth_manager = Mock()

        mock_config_class.return_value = mock_config
        mock_storage_class.return_value = mock_storage
        mock_auth_manager_class.return_value = mock_auth_manager

        providers = ["provider1"]
        mock_storage.list_providers.return_value = providers

        # Mock menu to return test action, then quit
        mock_auth_menus.auth_list_actions.side_effect = [
            ("test", "provider1"),
            ("quit", None),
        ]

        # Mock failed validation
        mock_validate.return_value = False

        # When
        AuthCommands.list_auth_providers(interactive=True)

        # Then
        mock_show_working.assert_called_with("Testing provider1...")
        mock_validate.assert_called_once_with(
            mock_storage, mock_auth_manager, "provider1"
        )
        mock_show_error.assert_called_with("Authentication for provider1 failed")

    @patch("llm_orc.cli_modules.commands.auth_commands.ConfigurationManager")
    @patch("llm_orc.cli_modules.commands.auth_commands.CredentialStorage")
    @patch("llm_orc.cli_modules.commands.auth_commands.AuthenticationManager")
    @patch("llm_orc.menu_system.AuthMenus")
    @patch("llm_orc.menu_system.show_working")
    @patch("llm_orc.menu_system.show_success")
    @patch("llm_orc.menu_system.show_error")
    @patch("llm_orc.cli_modules.utils.auth_utils.validate_provider_authentication")
    def test_list_providers_interactive_test_action_exception(
        self,
        mock_validate: Mock,
        mock_show_error: Mock,
        mock_show_success: Mock,
        mock_show_working: Mock,
        mock_auth_menus: Mock,
        mock_auth_manager_class: Mock,
        mock_storage_class: Mock,
        mock_config_class: Mock,
    ) -> None:
        """Test interactive listing with test action that raises exception."""
        # Given
        mock_config = Mock()
        mock_storage = Mock()
        mock_auth_manager = Mock()

        mock_config_class.return_value = mock_config
        mock_storage_class.return_value = mock_storage
        mock_auth_manager_class.return_value = mock_auth_manager

        providers = ["provider1"]
        mock_storage.list_providers.return_value = providers

        # Mock menu to return test action, then quit
        mock_auth_menus.auth_list_actions.side_effect = [
            ("test", "provider1"),
            ("quit", None),
        ]

        # Mock validation raising exception
        mock_validate.side_effect = Exception("Test error")

        # When
        AuthCommands.list_auth_providers(interactive=True)

        # Then
        mock_show_working.assert_called_with("Testing provider1...")
        mock_show_error.assert_called_with("Test failed: Test error")

    @patch("llm_orc.cli_modules.commands.auth_commands.ConfigurationManager")
    @patch("llm_orc.cli_modules.commands.auth_commands.CredentialStorage")
    @patch("llm_orc.cli_modules.commands.auth_commands.AuthenticationManager")
    @patch("llm_orc.menu_system.AuthMenus")
    @patch("llm_orc.menu_system.confirm_action")
    @patch("llm_orc.menu_system.show_success")
    def test_list_providers_interactive_remove_action_confirmed(
        self,
        mock_show_success: Mock,
        mock_confirm: Mock,
        mock_auth_menus: Mock,
        mock_auth_manager_class: Mock,
        mock_storage_class: Mock,
        mock_config_class: Mock,
    ) -> None:
        """Test interactive listing with remove action that is confirmed."""
        # Given
        mock_config = Mock()
        mock_storage = Mock()
        mock_auth_manager = Mock()

        mock_config_class.return_value = mock_config
        mock_storage_class.return_value = mock_storage
        mock_auth_manager_class.return_value = mock_auth_manager

        providers = ["provider1"]
        mock_storage.list_providers.return_value = providers

        # Mock menu to return remove action, then quit
        mock_auth_menus.auth_list_actions.side_effect = [
            ("remove", "provider1"),
            ("quit", None),
        ]

        # Mock confirmation as True
        mock_confirm.return_value = True

        # When
        AuthCommands.list_auth_providers(interactive=True)

        # Then
        mock_confirm.assert_called_with("Remove authentication for provider1?")
        mock_storage.remove_provider.assert_called_once_with("provider1")
        mock_show_success.assert_called_with("Removed provider1")

    @patch("llm_orc.cli_modules.commands.auth_commands.ConfigurationManager")
    @patch("llm_orc.cli_modules.commands.auth_commands.CredentialStorage")
    @patch("llm_orc.cli_modules.commands.auth_commands.AuthenticationManager")
    @patch("llm_orc.menu_system.AuthMenus")
    @patch("llm_orc.cli_modules.commands.auth_commands.AuthCommands.auth_setup")
    def test_list_providers_interactive_setup_action(
        self,
        mock_auth_setup: Mock,
        mock_auth_menus: Mock,
        mock_auth_manager_class: Mock,
        mock_storage_class: Mock,
        mock_config_class: Mock,
    ) -> None:
        """Test interactive listing with setup action."""
        # Given
        mock_config = Mock()
        mock_storage = Mock()
        mock_auth_manager = Mock()

        mock_config_class.return_value = mock_config
        mock_storage_class.return_value = mock_storage
        mock_auth_manager_class.return_value = mock_auth_manager

        providers: list[str] = []
        mock_storage.list_providers.return_value = providers

        # Mock menu to return setup action, then quit
        mock_auth_menus.auth_list_actions.side_effect = [
            ("setup", None),
            ("quit", None),
        ]

        # When
        AuthCommands.list_auth_providers(interactive=True)

        # Then
        mock_auth_setup.assert_called_once()

    @patch("llm_orc.cli_modules.commands.auth_commands.ConfigurationManager")
    @patch("llm_orc.cli_modules.commands.auth_commands.CredentialStorage")
    @patch("llm_orc.cli_modules.commands.auth_commands.AuthenticationManager")
    @patch("llm_orc.menu_system.AuthMenus")
    @patch("llm_orc.cli_modules.utils.config_utils.show_provider_details")
    def test_list_providers_interactive_details_action(
        self,
        mock_show_details: Mock,
        mock_auth_menus: Mock,
        mock_auth_manager_class: Mock,
        mock_storage_class: Mock,
        mock_config_class: Mock,
    ) -> None:
        """Test interactive listing with details action."""
        # Given
        mock_config = Mock()
        mock_storage = Mock()
        mock_auth_manager = Mock()

        mock_config_class.return_value = mock_config
        mock_storage_class.return_value = mock_storage
        mock_auth_manager_class.return_value = mock_auth_manager

        providers = ["provider1"]
        mock_storage.list_providers.return_value = providers

        # Mock menu to return details action, then quit
        mock_auth_menus.auth_list_actions.side_effect = [
            ("details", "provider1"),
            ("quit", None),
        ]

        # When
        AuthCommands.list_auth_providers(interactive=True)

        # Then
        mock_show_details.assert_called_once_with(mock_storage, "provider1")

    @patch("llm_orc.cli_modules.commands.auth_commands.ConfigurationManager")
    @patch("llm_orc.cli_modules.commands.auth_commands.CredentialStorage")
    @patch("llm_orc.cli_modules.commands.auth_commands.AuthenticationManager")
    @patch("llm_orc.menu_system.AuthMenus")
    @patch("llm_orc.menu_system.show_working")
    @patch("llm_orc.menu_system.show_success")
    @patch("llm_orc.menu_system.show_error")
    @patch("llm_orc.menu_system.show_info")
    def test_list_providers_interactive_refresh_oauth_action(
        self,
        mock_show_info: Mock,
        mock_show_error: Mock,
        mock_show_success: Mock,
        mock_show_working: Mock,
        mock_auth_menus: Mock,
        mock_auth_manager_class: Mock,
        mock_storage_class: Mock,
        mock_config_class: Mock,
    ) -> None:
        """Test interactive listing with refresh action for OAuth provider."""
        # Given
        mock_config = Mock()
        mock_storage = Mock()
        mock_auth_manager = Mock()

        mock_config_class.return_value = mock_config
        mock_storage_class.return_value = mock_storage
        mock_auth_manager_class.return_value = mock_auth_manager

        providers = ["provider1"]
        mock_storage.list_providers.return_value = providers
        mock_storage.get_auth_method.return_value = "oauth"

        # Mock menu to return refresh action, then quit
        mock_auth_menus.auth_list_actions.side_effect = [
            ("refresh", "provider1"),
            ("quit", None),
        ]

        # When
        AuthCommands.list_auth_providers(interactive=True)

        # Then
        mock_show_working.assert_called_with("Refreshing tokens for provider1...")
        mock_storage.get_auth_method.assert_called_with("provider1")
        mock_show_info.assert_called_with(
            "Re-authentication required for OAuth token refresh"
        )
        mock_show_success.assert_called_with("Token refresh would be performed here")

    @patch("llm_orc.cli_modules.commands.auth_commands.ConfigurationManager")
    @patch("llm_orc.cli_modules.commands.auth_commands.CredentialStorage")
    @patch("llm_orc.cli_modules.commands.auth_commands.AuthenticationManager")
    @patch("llm_orc.menu_system.AuthMenus")
    @patch("llm_orc.menu_system.show_working")
    @patch("llm_orc.menu_system.show_error")
    def test_list_providers_interactive_refresh_api_key_action(
        self,
        mock_show_error: Mock,
        mock_show_working: Mock,
        mock_auth_menus: Mock,
        mock_auth_manager_class: Mock,
        mock_storage_class: Mock,
        mock_config_class: Mock,
    ) -> None:
        """Test interactive listing with refresh action for API key provider."""
        # Given
        mock_config = Mock()
        mock_storage = Mock()
        mock_auth_manager = Mock()

        mock_config_class.return_value = mock_config
        mock_storage_class.return_value = mock_storage
        mock_auth_manager_class.return_value = mock_auth_manager

        providers = ["provider1"]
        mock_storage.list_providers.return_value = providers
        mock_storage.get_auth_method.return_value = "api_key"

        # Mock menu to return refresh action, then quit
        mock_auth_menus.auth_list_actions.side_effect = [
            ("refresh", "provider1"),
            ("quit", None),
        ]

        # When
        AuthCommands.list_auth_providers(interactive=True)

        # Then
        mock_show_working.assert_called_with("Refreshing tokens for provider1...")
        mock_storage.get_auth_method.assert_called_with("provider1")
        mock_show_error.assert_called_with(
            "Token refresh only available for OAuth providers"
        )

    @patch("llm_orc.cli_modules.commands.auth_commands.ConfigurationManager")
    @patch("llm_orc.cli_modules.commands.auth_commands.CredentialStorage")
    @patch("llm_orc.cli_modules.commands.auth_commands.AuthenticationManager")
    @patch("llm_orc.menu_system.AuthMenus")
    @patch("llm_orc.menu_system.show_working")
    @patch("llm_orc.menu_system.show_error")
    def test_list_providers_interactive_refresh_exception(
        self,
        mock_show_error: Mock,
        mock_show_working: Mock,
        mock_auth_menus: Mock,
        mock_auth_manager_class: Mock,
        mock_storage_class: Mock,
        mock_config_class: Mock,
    ) -> None:
        """Test interactive listing with refresh action that raises exception."""
        # Given
        mock_config = Mock()
        mock_storage = Mock()
        mock_auth_manager = Mock()

        mock_config_class.return_value = mock_config
        mock_storage_class.return_value = mock_storage
        mock_auth_manager_class.return_value = mock_auth_manager

        providers = ["provider1"]
        mock_storage.list_providers.return_value = providers
        mock_storage.get_auth_method.side_effect = Exception("Auth method error")

        # Mock menu to return refresh action, then quit
        mock_auth_menus.auth_list_actions.side_effect = [
            ("refresh", "provider1"),
            ("quit", None),
        ]

        # When
        AuthCommands.list_auth_providers(interactive=True)

        # Then
        mock_show_working.assert_called_with("Refreshing tokens for provider1...")
        mock_show_error.assert_called_with("Refresh failed: Auth method error")

    @patch("llm_orc.cli_modules.commands.auth_commands.ConfigurationManager")
    @patch("llm_orc.cli_modules.commands.auth_commands.CredentialStorage")
    @patch("llm_orc.cli_modules.commands.auth_commands.AuthenticationManager")
    def test_list_providers_interactive_exception_handling(
        self,
        mock_auth_manager_class: Mock,
        mock_storage_class: Mock,
        mock_config_class: Mock,
    ) -> None:
        """Test interactive listing with exception in main flow."""
        # Given
        mock_config = Mock()
        mock_storage = Mock()

        mock_config_class.return_value = mock_config
        mock_storage_class.return_value = mock_storage

        # Mock storage to raise exception
        mock_storage.list_providers.side_effect = Exception("Storage error")

        # When/Then
        with pytest.raises(
            click.ClickException, match="Failed to list providers: Storage error"
        ):
            AuthCommands.list_auth_providers(interactive=True)


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
            click.ClickException, match="No authentication found for test-provider"
        ):
            AuthCommands.remove_auth_provider("test-provider")

    @patch("llm_orc.cli_modules.commands.auth_commands.ConfigurationManager")
    @patch("llm_orc.cli_modules.commands.auth_commands.CredentialStorage")
    def test_remove_provider_exception_handling(
        self,
        mock_storage_class: Mock,
        mock_config_class: Mock,
    ) -> None:
        """Test exception handling when remove operation fails (lines 227-228)."""
        # Given
        mock_config = Mock()
        mock_storage = Mock()

        mock_config_class.return_value = mock_config
        mock_storage_class.return_value = mock_storage

        mock_storage.list_providers.return_value = ["test-provider"]

        # Mock storage.remove_provider to raise a generic exception
        mock_storage.remove_provider.side_effect = Exception("Storage error")

        # When & Then
        with pytest.raises(
            click.ClickException, match="Failed to remove provider: Storage error"
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
            click.ClickException, match="No authentication found for test-provider"
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
            click.ClickException, match="No OAuth token found for test-provider"
        ):
            AuthCommands.test_token_refresh("test-provider")

    @patch("llm_orc.core.auth.oauth_client.OAuthClaudeClient")
    @patch("llm_orc.cli_modules.commands.auth_commands.ConfigurationManager")
    @patch("llm_orc.cli_modules.commands.auth_commands.CredentialStorage")
    def test_refresh_token_info_display(
        self,
        mock_storage_class: Mock,
        mock_config_class: Mock,
        mock_oauth_client_class: Mock,
    ) -> None:
        """Test token info display for OAuth provider."""
        # Given
        mock_config = Mock()
        mock_storage = Mock()
        mock_oauth_client = Mock()

        mock_config_class.return_value = mock_config
        mock_storage_class.return_value = mock_storage
        mock_oauth_client_class.return_value = mock_oauth_client

        mock_storage.list_providers.return_value = ["test-provider"]
        oauth_token = {
            "access_token": "test_token",
            "refresh_token": "test_refresh",
            "client_id": "test_client",
            "expires_at": time.time() + 3600,  # Expires in 1 hour
        }
        mock_storage.get_oauth_token.return_value = oauth_token

        # Mock successful token refresh
        mock_oauth_client.refresh_access_token.return_value = True

        # When
        with patch("click.echo") as mock_echo:
            AuthCommands.test_token_refresh("test-provider")

        # Then
        echo_calls = [call[0][0] for call in mock_echo.call_args_list]
        token_info_found = any(
            "Token info for test-provider" in call for call in echo_calls
        )
        assert token_info_found

    @patch("llm_orc.core.auth.oauth_client.OAuthClaudeClient")
    @patch("time.time")
    @patch("llm_orc.cli_modules.commands.auth_commands.ConfigurationManager")
    @patch("llm_orc.cli_modules.commands.auth_commands.CredentialStorage")
    def test_refresh_expired_token(
        self,
        mock_storage_class: Mock,
        mock_config_class: Mock,
        mock_time: Mock,
        mock_oauth_client_class: Mock,
    ) -> None:
        """Test token refresh with expired token."""
        # Given
        mock_config = Mock()
        mock_storage = Mock()

        mock_config_class.return_value = mock_config
        mock_storage_class.return_value = mock_storage

        # Mock the OAuth client to prevent actual HTTP requests
        mock_oauth_client = Mock()
        mock_oauth_client.refresh_access_token.return_value = False  # Simulate failure
        mock_oauth_client_class.return_value = mock_oauth_client

        import time

        current_time = int(time.time())  # Use current time
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
            match="Failed to set up claude-cli authentication: Claude CLI error",
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
            match="Failed to set up anthropic-claude-pro-max"
            " authentication: OAuth error",
        ):
            AuthCommands.add_auth_provider(
                provider="anthropic-claude-pro-max",
                api_key=None,
                client_id=None,
                client_secret=None,
            )

    @patch(
        "llm_orc.cli_modules.commands.auth_commands.handle_anthropic_interactive_auth"
    )
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

    @patch(
        "llm_orc.cli_modules.commands.auth_commands.handle_anthropic_interactive_auth"
    )
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
            match="Failed to set up anthropic authentication: Interactive auth error",
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
            "Cannot test refresh: no client ID available" in call for call in echo_calls
        )
        assert no_client_id_found

    @patch("llm_orc.cli_modules.commands.auth_commands.ConfigurationManager")
    @patch("llm_orc.cli_modules.commands.auth_commands.CredentialStorage")
    def test_refresh_default_client_id_anthropic_claude_pro_max(
        self,
        mock_storage_class: Mock,
        mock_config_class: Mock,
    ) -> None:
        """
        Test token refresh using default client ID for anthropic-claude-pro-max.
        (lines 275-276)
        """
        # Given
        mock_config = Mock()
        mock_storage = Mock()

        mock_config_class.return_value = mock_config
        mock_storage_class.return_value = mock_storage

        mock_storage.list_providers.return_value = ["anthropic-claude-pro-max"]
        oauth_token = {
            "access_token": "test_token",
            "refresh_token": "test_refresh",
            "expires_at": time.time() + 3600,
            # No client_id in the token - should use default
        }
        mock_storage.get_oauth_token.return_value = oauth_token

        # Mock OAuthClaudeClient to return successful refresh
        with patch(
            "llm_orc.core.auth.oauth_client.OAuthClaudeClient"
        ) as mock_client_class:
            mock_client = Mock()
            mock_client.refresh_access_token.return_value = True
            mock_client.access_token = "new_access_token"
            mock_client.refresh_token = "new_refresh_token"
            mock_client_class.return_value = mock_client

            with (
                patch("click.echo") as mock_echo,
                patch("time.time", return_value=1000),
            ):
                # When
                AuthCommands.test_token_refresh("anthropic-claude-pro-max")

                # Then
                echo_calls = [call[0][0] for call in mock_echo.call_args_list]
                default_client_id_found = any(
                    "Using default client ID: 9d1c250a-e61b-44d9-88ed-5944d1962f5e"
                    in call
                    for call in echo_calls
                )
                assert default_client_id_found

                # Verify client was created with correct tokens
                mock_client_class.assert_called_once_with(
                    access_token="test_token",
                    refresh_token="test_refresh",
                )

                # Verify refresh was called with default client ID
                mock_client.refresh_access_token.assert_called_once_with(
                    "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
                )

    @patch("llm_orc.cli_modules.commands.auth_commands.ConfigurationManager")
    @patch("llm_orc.cli_modules.commands.auth_commands.CredentialStorage")
    def test_refresh_successful_token_update(
        self,
        mock_storage_class: Mock,
        mock_config_class: Mock,
    ) -> None:
        """Test successful token refresh and credential update (lines 294-304)."""
        # Given
        mock_config = Mock()
        mock_storage = Mock()

        mock_config_class.return_value = mock_config
        mock_storage_class.return_value = mock_storage

        mock_storage.list_providers.return_value = ["test-provider"]
        oauth_token = {
            "access_token": "old_token",
            "refresh_token": "old_refresh",
            "client_id": "test_client_id",
            "expires_at": time.time() + 3600,
        }
        mock_storage.get_oauth_token.return_value = oauth_token

        # Mock OAuthClaudeClient to return successful refresh
        with patch(
            "llm_orc.core.auth.oauth_client.OAuthClaudeClient"
        ) as mock_client_class:
            mock_client = Mock()
            mock_client.refresh_access_token.return_value = True
            mock_client.access_token = "new_access_token"
            mock_client.refresh_token = "new_refresh_token"
            mock_client_class.return_value = mock_client

            with (
                patch("click.echo") as mock_echo,
                patch("time.time", return_value=1000),
            ):
                # When
                AuthCommands.test_token_refresh("test-provider")

                # Then
                echo_calls = [call[0][0] for call in mock_echo.call_args_list]
                success_messages = [
                    "Token refresh successful!",
                    "Updated stored credentials",
                ]
                for message in success_messages:
                    found = any(message in call for call in echo_calls)
                    assert found, (
                        f"Expected message '{message}' not found in {echo_calls}"
                    )

                # Verify credentials were updated
                mock_storage.store_oauth_token.assert_called_once_with(
                    "test-provider",
                    "new_access_token",
                    "new_refresh_token",
                    expires_at=1000 + 3600,  # time.time() + 3600
                    client_id="test_client_id",
                )

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
            click.ClickException, match="Failed to logout from test-provider"
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
            click.ClickException, match="Must specify a provider name or use --all flag"
        ):
            AuthCommands.logout_oauth_providers(provider=None, logout_all=False)


class TestAuthSetupHelperMethods:
    """Test helper methods extracted from auth_setup for complexity reduction."""

    def test_handle_existing_provider_replace_confirmed(self) -> None:
        """Test replacing existing provider when user confirms."""
        # Given
        mock_storage = Mock()
        mock_storage.list_providers.return_value = ["test-provider"]

        provider_display_name = "Test Provider"
        provider_key = "test-provider"

        # When
        with (
            patch("llm_orc.menu_system.confirm_action", return_value=True),
            patch("llm_orc.menu_system.show_success") as mock_success,
            patch("click.echo"),
        ):
            result = AuthCommands._handle_existing_provider(
                storage=mock_storage,
                provider_key=provider_key,
                provider_display_name=provider_display_name,
            )

        # Then
        assert result is True  # Should continue with setup
        mock_storage.remove_provider.assert_called_once_with(provider_key)
        mock_success.assert_called_once_with(
            f"Removed existing authentication for {provider_display_name}"
        )

    def test_handle_existing_provider_replace_declined_continue_confirmed(self) -> None:
        """Test declining replacement but confirming to add another provider."""
        # Given
        mock_storage = Mock()
        mock_storage.list_providers.return_value = ["test-provider"]

        provider_display_name = "Test Provider"
        provider_key = "test-provider"

        # When
        with (
            patch("llm_orc.menu_system.confirm_action", side_effect=[False, True]),
            patch("click.echo"),
        ):
            result = AuthCommands._handle_existing_provider(
                storage=mock_storage,
                provider_key=provider_key,
                provider_display_name=provider_display_name,
            )

        # Then
        assert result is False  # Should not continue with this provider
        mock_storage.remove_provider.assert_not_called()

    def test_handle_existing_provider_replace_declined_continue_declined(self) -> None:
        """Test declining replacement and declining to add another provider."""
        # Given
        mock_storage = Mock()
        mock_storage.list_providers.return_value = ["test-provider"]

        provider_display_name = "Test Provider"
        provider_key = "test-provider"

        # When
        with (
            patch("llm_orc.menu_system.confirm_action", side_effect=[False, False]),
            patch("click.echo"),
        ):
            result = AuthCommands._handle_existing_provider(
                storage=mock_storage,
                provider_key=provider_key,
                provider_display_name=provider_display_name,
            )

        # Then
        assert result is None  # Should exit setup entirely
        mock_storage.remove_provider.assert_not_called()

    def test_determine_auth_method_anthropic_claude_pro_max(self) -> None:
        """Test auth method determination for Claude Pro/Max."""
        # When/Then
        result = AuthCommands._determine_auth_method("anthropic-claude-pro-max")
        assert result == "oauth"

    def test_determine_auth_method_anthropic_api(self) -> None:
        """Test auth method determination for Anthropic API."""
        # When/Then
        assert AuthCommands._determine_auth_method("anthropic-api") == "api_key"

    def test_determine_auth_method_google_gemini(self) -> None:
        """Test auth method determination for Google Gemini."""
        # When/Then
        assert AuthCommands._determine_auth_method("google-gemini") == "api_key"

    def test_determine_auth_method_generic_provider(self) -> None:
        """Test auth method determination for generic provider using menu."""
        # Given
        provider_key = "generic-provider"

        # When
        with patch(
            "llm_orc.menu_system.AuthMenus.get_auth_method_for_provider",
            return_value="oauth",
        ) as mock_menu:
            result = AuthCommands._determine_auth_method(provider_key)

        # Then
        assert result == "oauth"
        mock_menu.assert_called_once_with(provider_key)

    def test_handle_authentication_setup_help(self) -> None:
        """Test authentication setup with help request."""
        # Given
        mock_storage = Mock()
        mock_auth_manager = Mock()
        mock_provider = Mock()
        mock_provider.display_name = "Test Provider"

        # When
        with patch(
            "llm_orc.cli_modules.utils.auth_utils.show_auth_method_help"
        ) as mock_help:
            result = AuthCommands._handle_authentication_setup(
                auth_method="help",
                provider_key="test-provider",
                provider=mock_provider,
                storage=mock_storage,
                auth_manager=mock_auth_manager,
            )

        # Then
        assert result is True  # Should continue (not break)
        mock_help.assert_called_once()

    def test_handle_authentication_setup_anthropic_api_key(self) -> None:
        """Test authentication setup for Anthropic API key."""
        # Given
        mock_storage = Mock()
        mock_auth_manager = Mock()
        mock_provider = Mock()
        mock_provider.display_name = "Anthropic API"

        # When
        with (
            patch("click.prompt", return_value="test-api-key"),
            patch("llm_orc.menu_system.show_success") as mock_success,
        ):
            result = AuthCommands._handle_authentication_setup(
                auth_method="api_key",
                provider_key="anthropic-api",
                provider=mock_provider,
                storage=mock_storage,
                auth_manager=mock_auth_manager,
            )

        # Then
        assert result is False  # Should continue with next provider
        mock_storage.store_api_key.assert_called_once_with(
            "anthropic-api", "test-api-key"
        )
        mock_success.assert_called_once_with("Anthropic API key configured!")

    def test_handle_authentication_setup_oauth_success(self) -> None:
        """Test successful OAuth authentication setup."""
        # Given
        mock_storage = Mock()
        mock_auth_manager = Mock()
        mock_auth_manager.authenticate_oauth.return_value = True
        mock_provider = Mock()
        mock_provider.display_name = "Test Provider"

        # When
        with (
            patch("click.prompt", side_effect=["client-id", "client-secret"]),
            patch("llm_orc.menu_system.show_success") as mock_success,
        ):
            result = AuthCommands._handle_authentication_setup(
                auth_method="oauth",
                provider_key="test-provider",
                provider=mock_provider,
                storage=mock_storage,
                auth_manager=mock_auth_manager,
            )

        # Then
        assert result is False  # Should continue with next provider
        mock_auth_manager.authenticate_oauth.assert_called_once_with(
            "test-provider", "client-id", "client-secret"
        )
        mock_success.assert_called_once_with("Test Provider OAuth configured!")

    def test_handle_authentication_setup_oauth_failure(self) -> None:
        """Test failed OAuth authentication setup."""
        # Given
        mock_storage = Mock()
        mock_auth_manager = Mock()
        mock_auth_manager.authenticate_oauth.return_value = False
        mock_provider = Mock()
        mock_provider.display_name = "Test Provider"

        # When
        with (
            patch("click.prompt", side_effect=["client-id", "client-secret"]),
            patch("llm_orc.menu_system.show_error") as mock_error,
        ):
            result = AuthCommands._handle_authentication_setup(
                auth_method="oauth",
                provider_key="test-provider",
                provider=mock_provider,
                storage=mock_storage,
                auth_manager=mock_auth_manager,
            )

        # Then
        assert result is False  # Should continue with next provider
        mock_error.assert_called_once_with(
            "OAuth authentication for Test Provider failed"
        )

    def test_initialize_auth_setup_managers(self) -> None:
        """Test initializing managers for auth setup."""
        from llm_orc.cli_modules.commands.auth_commands import (
            _initialize_auth_setup_managers,
        )

        # When
        with (
            patch(
                "llm_orc.cli_modules.commands.auth_commands.ConfigurationManager"
            ) as mock_config_class,
            patch(
                "llm_orc.cli_modules.commands.auth_commands.CredentialStorage"
            ) as mock_storage_class,
            patch(
                "llm_orc.cli_modules.commands.auth_commands.AuthenticationManager"
            ) as mock_auth_class,
            patch("click.echo"),
        ):
            config_manager, storage, auth_manager = _initialize_auth_setup_managers()

        # Then
        mock_config_class.assert_called_once()
        mock_storage_class.assert_called_once_with(mock_config_class.return_value)
        mock_auth_class.assert_called_once_with(mock_storage_class.return_value)
        assert config_manager == mock_config_class.return_value
        assert storage == mock_storage_class.return_value
        assert auth_manager == mock_auth_class.return_value

    def test_process_single_provider_no_auth_required(self) -> None:
        """Test processing a provider that doesn't require authentication."""
        from llm_orc.cli_modules.commands.auth_commands import _process_single_provider

        # Given
        mock_storage = Mock()
        mock_auth_manager = Mock()
        mock_provider = Mock()
        mock_provider.requires_auth = False
        mock_provider.display_name = "Test Provider"

        # When
        with (
            patch("llm_orc.menu_system.show_success") as mock_success,
            patch("llm_orc.menu_system.confirm_action", return_value=False),
        ):
            result = _process_single_provider(
                provider_key="test-provider",
                provider=mock_provider,
                storage=mock_storage,
                auth_manager=mock_auth_manager,
            )

        # Then
        assert result == "break"  # Should exit main loop
        mock_success.assert_called_once_with(
            "Test Provider doesn't require authentication!"
        )

    def test_process_single_provider_authentication_success(self) -> None:
        """Test successfully processing a provider that requires authentication."""
        from llm_orc.cli_modules.commands.auth_commands import _process_single_provider

        # Given
        mock_storage = Mock()
        mock_auth_manager = Mock()
        mock_provider = Mock()
        mock_provider.requires_auth = True
        mock_provider.display_name = "Test Provider"

        # When
        with (
            patch(
                "llm_orc.cli_modules.commands.auth_commands.AuthCommands._handle_existing_provider",
                return_value=True,
            ),
            patch(
                "llm_orc.cli_modules.commands.auth_commands.AuthCommands._determine_auth_method",
                return_value="api_key",
            ),
            patch(
                "llm_orc.cli_modules.commands.auth_commands.AuthCommands._handle_authentication_setup",
                return_value=False,
            ),
            patch("llm_orc.menu_system.confirm_action", return_value=False),
        ):
            result = _process_single_provider(
                provider_key="test-provider",
                provider=mock_provider,
                storage=mock_storage,
                auth_manager=mock_auth_manager,
            )

        # Then
        assert result == "break"  # Should exit main loop

    def test_process_single_provider_user_exit(self) -> None:
        """Test processing when user chooses to exit."""
        from llm_orc.cli_modules.commands.auth_commands import _process_single_provider

        # Given
        mock_storage = Mock()
        mock_auth_manager = Mock()
        mock_provider = Mock()
        mock_provider.requires_auth = True

        # When
        with patch(
            "llm_orc.cli_modules.commands.auth_commands.AuthCommands._handle_existing_provider",
            return_value=None,
        ):
            result = _process_single_provider(
                provider_key="test-provider",
                provider=mock_provider,
                storage=mock_storage,
                auth_manager=mock_auth_manager,
            )

        # Then
        assert result == "break"  # Should exit main loop

    def test_show_setup_completion_message(self) -> None:
        """Test displaying setup completion message."""
        from llm_orc.cli_modules.commands.auth_commands import (
            _show_setup_completion_message,
        )

        # When
        with (
            patch("click.echo") as mock_echo,
            patch("llm_orc.menu_system.show_success") as mock_success,
        ):
            _show_setup_completion_message()

        # Then
        mock_echo.assert_called_once()
        mock_success.assert_called_once_with(
            "Setup complete! Use 'llm-orc auth list' to see your configured providers."
        )


class TestListAuthProvidersHelperMethods:
    """Test helper methods extracted from list_auth_providers for complexity."""

    def test_display_simple_provider_list_empty(self) -> None:
        """Test displaying simple provider list when no providers configured."""
        # Given
        providers: list[str] = []

        # When
        with patch("click.echo") as mock_echo:
            AuthCommands._display_simple_provider_list(providers)

        # Then
        mock_echo.assert_called_once_with("No authentication providers configured")

    def test_display_simple_provider_list_with_providers(self) -> None:
        """Test displaying simple provider list with configured providers."""
        # Given
        providers = ["anthropic-api", "google-gemini"]
        mock_storage = Mock()
        mock_storage.get_auth_method.side_effect = ["api_key", "oauth"]

        # When
        with patch("click.echo") as mock_echo:
            AuthCommands._display_simple_provider_list(providers, mock_storage)

        # Then
        expected_calls = [
            call("Configured providers:"),
            call("  anthropic-api: API key"),
            call("  google-gemini: OAuth"),
        ]
        mock_echo.assert_has_calls(expected_calls)

    def test_display_simple_provider_list_oauth_method(self) -> None:
        """Test displaying provider with OAuth method."""
        # Given
        providers = ["test-oauth-provider"]
        mock_storage = Mock()
        mock_storage.get_auth_method.return_value = "oauth"

        # When
        with patch("click.echo") as mock_echo:
            AuthCommands._display_simple_provider_list(providers, mock_storage)

        # Then
        expected_calls = [
            call("Configured providers:"),
            call("  test-oauth-provider: OAuth"),
        ]
        mock_echo.assert_has_calls(expected_calls)

    def test_display_simple_provider_list_no_storage(self) -> None:
        """Test displaying provider list without storage details."""
        # Given
        providers = ["anthropic-api", "google-gemini"]

        # When
        with patch("click.echo") as mock_echo:
            AuthCommands._display_simple_provider_list(providers)

        # Then
        expected_calls = [
            call("Configured providers:"),
            call("  anthropic-api"),
            call("  google-gemini"),
        ]
        mock_echo.assert_has_calls(expected_calls)

    def test_handle_interactive_action_quit(self) -> None:
        """Test handling quit action."""
        # Given
        action = "quit"
        selected_provider = None
        storage = Mock()
        auth_manager = Mock()

        # When
        result = AuthCommands._handle_interactive_action(
            action, selected_provider, storage, auth_manager
        )

        # Then
        assert result is None  # Quit action returns None

    def test_handle_interactive_action_setup(self) -> None:
        """Test handling setup action."""
        # Given
        action = "setup"
        selected_provider = None
        storage = Mock()
        auth_manager = Mock()

        # When
        with patch.object(AuthCommands, "auth_setup") as mock_auth_setup:
            result = AuthCommands._handle_interactive_action(
                action, selected_provider, storage, auth_manager
            )

        # Then
        assert result is True  # Setup action returns True to refresh providers
        mock_auth_setup.assert_called_once()

    def test_handle_interactive_action_test_success(self) -> None:
        """Test handling test action with successful validation."""
        # Given
        action = "test"
        selected_provider = "anthropic-api"
        storage = Mock()
        auth_manager = Mock()

        # When
        with (
            patch(
                "llm_orc.cli_modules.utils.auth_utils.validate_provider_authentication",
                return_value=True,
            ),
            patch("llm_orc.menu_system.show_working") as mock_working,
            patch("llm_orc.menu_system.show_success") as mock_success,
        ):
            result = AuthCommands._handle_interactive_action(
                action, selected_provider, storage, auth_manager
            )

        # Then
        assert result is False  # Test action returns False (no provider refresh)
        mock_working.assert_called_once_with("Testing anthropic-api...")
        mock_success.assert_called_once_with(
            "Authentication for anthropic-api is working!"
        )

    def test_handle_interactive_action_test_failure(self) -> None:
        """Test handling test action with failed validation."""
        # Given
        action = "test"
        selected_provider = "anthropic-api"
        storage = Mock()
        auth_manager = Mock()

        # When
        with (
            patch(
                "llm_orc.cli_modules.utils.auth_utils.validate_provider_authentication",
                return_value=False,
            ),
            patch("llm_orc.menu_system.show_working"),
            patch("llm_orc.menu_system.show_error") as mock_error,
        ):
            result = AuthCommands._handle_interactive_action(
                action, selected_provider, storage, auth_manager
            )

        # Then
        assert result is False
        mock_error.assert_called_once_with("Authentication for anthropic-api failed")

    def test_handle_interactive_action_test_exception(self) -> None:
        """Test handling test action with exception."""
        # Given
        action = "test"
        selected_provider = "anthropic-api"
        storage = Mock()
        auth_manager = Mock()

        # When
        with (
            patch(
                "llm_orc.cli_modules.utils.auth_utils.validate_provider_authentication",
                side_effect=Exception("Test error"),
            ),
            patch("llm_orc.menu_system.show_working"),
            patch("llm_orc.menu_system.show_error") as mock_error,
        ):
            result = AuthCommands._handle_interactive_action(
                action, selected_provider, storage, auth_manager
            )

        # Then
        assert result is False
        mock_error.assert_called_once_with("Test failed: Test error")

    def test_handle_interactive_action_remove_confirmed(self) -> None:
        """Test handling remove action with user confirmation."""
        # Given
        action = "remove"
        selected_provider = "anthropic-api"
        storage = Mock()
        auth_manager = Mock()

        # When
        with (
            patch("llm_orc.menu_system.confirm_action", return_value=True),
            patch("llm_orc.menu_system.show_success") as mock_success,
        ):
            result = AuthCommands._handle_interactive_action(
                action, selected_provider, storage, auth_manager
            )

        # Then
        assert result is True  # Remove action returns True to refresh providers
        storage.remove_provider.assert_called_once_with("anthropic-api")
        mock_success.assert_called_once_with("Removed anthropic-api")

    def test_handle_interactive_action_remove_cancelled(self) -> None:
        """Test handling remove action with user cancellation."""
        # Given
        action = "remove"
        selected_provider = "anthropic-api"
        storage = Mock()
        auth_manager = Mock()

        # When
        with patch("llm_orc.menu_system.confirm_action", return_value=False):
            result = AuthCommands._handle_interactive_action(
                action, selected_provider, storage, auth_manager
            )

        # Then
        assert result is False  # Cancelled remove returns False
        storage.remove_provider.assert_not_called()

    def test_handle_interactive_action_details(self) -> None:
        """Test handling details action."""
        # Given
        action = "details"
        selected_provider = "anthropic-api"
        storage = Mock()
        auth_manager = Mock()

        # When
        with patch(
            "llm_orc.cli_modules.utils.config_utils.show_provider_details"
        ) as mock_details:
            result = AuthCommands._handle_interactive_action(
                action, selected_provider, storage, auth_manager
            )

        # Then
        assert result is False  # Details action returns False
        mock_details.assert_called_once_with(storage, "anthropic-api")


class TestAddAuthProviderHelperMethods:
    """Test helper methods extracted from add_auth_provider for complexity reduction."""

    def test_remove_existing_provider_not_exists(self) -> None:
        """Test removing existing provider when provider doesn't exist."""
        # Given
        storage = Mock()
        storage.list_providers.return_value = []  # Provider doesn't exist
        provider = "test-provider"

        # When
        with patch("click.echo") as mock_echo:
            AuthCommands._remove_existing_provider(storage, provider)

        # Then
        storage.remove_provider.assert_not_called()
        mock_echo.assert_not_called()

    def test_remove_existing_provider_exists(self) -> None:
        """Test removing existing provider when provider exists."""
        # Given
        storage = Mock()
        storage.list_providers.return_value = ["test-provider"]  # Provider exists
        provider = "test-provider"

        # When
        with patch("click.echo") as mock_echo:
            AuthCommands._remove_existing_provider(storage, provider)

        # Then
        storage.remove_provider.assert_called_once_with("test-provider")
        expected_calls = [
            call("🔄 Existing authentication found for test-provider"),
            call("   Removing old authentication before setting up new..."),
            call("✅ Old authentication removed"),
        ]
        mock_echo.assert_has_calls(expected_calls)

    def test_validate_auth_credentials_both_provided(self) -> None:
        """Test validation when both API key and OAuth credentials are provided."""
        # Given
        api_key = "test-api-key"
        client_id = "client-id"
        client_secret = "client-secret"

        # When/Then
        with pytest.raises(
            click.ClickException, match="Cannot use both API key and OAuth credentials"
        ):
            AuthCommands._validate_auth_credentials(api_key, client_id, client_secret)

    def test_validate_auth_credentials_none_provided(self) -> None:
        """Test validation when no credentials are provided."""
        # Given
        api_key = None
        client_id = None
        client_secret = None

        # When/Then
        with pytest.raises(
            click.ClickException,
            match=(
                "Must provide either --api-key or both --client-id and --client-secret"
            ),
        ):
            AuthCommands._validate_auth_credentials(api_key, client_id, client_secret)

    def test_validate_auth_credentials_incomplete_oauth(self) -> None:
        """Test validation with incomplete OAuth credentials."""
        # Given
        api_key = None
        client_id = "client-id"
        client_secret = None  # Missing client_secret

        # When/Then
        with pytest.raises(
            click.ClickException,
            match=(
                "Must provide either --api-key or both --client-id and --client-secret"
            ),
        ):
            AuthCommands._validate_auth_credentials(api_key, client_id, client_secret)

    def test_validate_auth_credentials_valid_api_key(self) -> None:
        """Test validation with valid API key."""
        # Given
        api_key = "test-api-key"
        client_id = None
        client_secret = None

        # When (should not raise exception)
        AuthCommands._validate_auth_credentials(api_key, client_id, client_secret)

        # Then - no exception raised means success

    def test_validate_auth_credentials_valid_oauth(self) -> None:
        """Test validation with valid OAuth credentials."""
        # Given
        api_key = None
        client_id = "client-id"
        client_secret = "client-secret"

        # When (should not raise exception)
        AuthCommands._validate_auth_credentials(api_key, client_id, client_secret)

        # Then - no exception raised means success

    def test_handle_provider_test_success(self) -> None:
        """Test _handle_provider_test helper method with successful validation."""
        # Given
        selected_provider = "anthropic-api"
        storage = Mock()
        auth_manager = Mock()

        # When
        with (
            patch(
                "llm_orc.cli_modules.utils.auth_utils.validate_provider_authentication",
                return_value=True,
            ),
            patch("llm_orc.menu_system.show_working") as mock_working,
            patch("llm_orc.menu_system.show_success") as mock_success,
        ):
            result = AuthCommands._handle_provider_test(
                selected_provider, storage, auth_manager
            )

        # Then
        assert result is False  # Test action returns False (no provider refresh)
        mock_working.assert_called_once_with("Testing anthropic-api...")
        mock_success.assert_called_once_with(
            "Authentication for anthropic-api is working!"
        )

    def test_handle_provider_test_failure(self) -> None:
        """Test _handle_provider_test helper method with failed validation."""
        # Given
        selected_provider = "anthropic-api"
        storage = Mock()
        auth_manager = Mock()

        # When
        with (
            patch(
                "llm_orc.cli_modules.utils.auth_utils.validate_provider_authentication",
                return_value=False,
            ),
            patch("llm_orc.menu_system.show_working"),
            patch("llm_orc.menu_system.show_error") as mock_error,
        ):
            result = AuthCommands._handle_provider_test(
                selected_provider, storage, auth_manager
            )

        # Then
        assert result is False
        mock_error.assert_called_once_with("Authentication for anthropic-api failed")

    def test_handle_provider_test_exception(self) -> None:
        """Test _handle_provider_test helper method with exception."""
        # Given
        selected_provider = "anthropic-api"
        storage = Mock()
        auth_manager = Mock()

        # When
        with (
            patch(
                "llm_orc.cli_modules.utils.auth_utils.validate_provider_authentication",
                side_effect=Exception("Test error"),
            ),
            patch("llm_orc.menu_system.show_working"),
            patch("llm_orc.menu_system.show_error") as mock_error,
        ):
            result = AuthCommands._handle_provider_test(
                selected_provider, storage, auth_manager
            )

        # Then
        assert result is False
        mock_error.assert_called_once_with("Test failed: Test error")

    def test_handle_token_refresh_oauth_success(self) -> None:
        """Test _handle_token_refresh helper method for OAuth provider."""
        # Given
        selected_provider = "anthropic-claude-pro-max"
        storage = Mock()
        storage.get_auth_method.return_value = "oauth"

        # When
        with (
            patch("llm_orc.menu_system.show_working") as mock_working,
            patch("llm_orc.menu_system.show_info") as mock_info,
            patch("llm_orc.menu_system.show_success") as mock_success,
        ):
            result = AuthCommands._handle_token_refresh(selected_provider, storage)

        # Then
        assert result is False  # Refresh action returns False
        mock_working.assert_called_once_with(
            "Refreshing tokens for anthropic-claude-pro-max..."
        )
        mock_info.assert_called_once_with(
            "Re-authentication required for OAuth token refresh"
        )
        mock_success.assert_called_once_with("Token refresh would be performed here")

    def test_handle_token_refresh_api_key_error(self) -> None:
        """Test _handle_token_refresh helper method for API key provider."""
        # Given
        selected_provider = "anthropic-api"
        storage = Mock()
        storage.get_auth_method.return_value = "api_key"

        # When
        with (
            patch("llm_orc.menu_system.show_working"),
            patch("llm_orc.menu_system.show_error") as mock_error,
        ):
            result = AuthCommands._handle_token_refresh(selected_provider, storage)

        # Then
        assert result is False
        mock_error.assert_called_once_with(
            "Token refresh only available for OAuth providers"
        )

    def test_handle_token_refresh_exception(self) -> None:
        """Test _handle_token_refresh helper method with exception."""
        # Given
        selected_provider = "test-provider"
        storage = Mock()
        storage.get_auth_method.side_effect = Exception("Auth method error")

        # When
        with (
            patch("llm_orc.menu_system.show_working"),
            patch("llm_orc.menu_system.show_error") as mock_error,
        ):
            result = AuthCommands._handle_token_refresh(selected_provider, storage)

        # Then
        assert result is False
        mock_error.assert_called_once_with("Refresh failed: Auth method error")


class TestTokenRefreshHelperMethods:
    """Test helper methods extracted from test_token_refresh for complexity."""

    def test_analyze_token_info(self) -> None:
        """Test token information analysis helper method."""
        # Given
        oauth_token = {
            "access_token": "test_token",
            "refresh_token": "test_refresh",
            "client_id": "test_client",
            "expires_at": 1234567890,
        }

        # When
        from llm_orc.cli_modules.commands.auth_commands import _analyze_token_info

        result = _analyze_token_info(oauth_token)

        # Then
        assert result["has_refresh_token"] is True
        assert result["has_client_id"] is True
        assert result["has_expires_at"] is True
        assert "expires_at" in result
        assert "current_time" in result
        assert "is_expired" in result

    def test_analyze_token_info_missing_fields(self) -> None:
        """Test token analysis with missing fields."""
        # Given
        oauth_token = {"access_token": "test_token"}

        # When
        from llm_orc.cli_modules.commands.auth_commands import _analyze_token_info

        result = _analyze_token_info(oauth_token)

        # Then
        assert result["has_refresh_token"] is False
        assert result["has_client_id"] is False
        assert result["has_expires_at"] is False

    @patch("click.echo")
    def test_display_token_status(self, mock_echo: Mock) -> None:
        """Test token status display helper method."""
        # Given
        provider = "test-provider"
        token_analysis = {
            "has_refresh_token": True,
            "has_client_id": False,
            "has_expires_at": True,
            "time_remaining": {"hours": 0, "minutes": 14, "seconds": 50},
        }

        # When
        from llm_orc.cli_modules.commands.auth_commands import _display_token_status

        _display_token_status(provider, token_analysis)

        # Then
        expected_calls = [
            "🔍 Token info for test-provider:",
            "  Has refresh token: ✅",
            "  Has client ID: ❌",
            "  Has expiration: ✅",
        ]
        actual_calls = [call.args[0] for call in mock_echo.call_args_list]
        for expected in expected_calls:
            assert expected in actual_calls

    def test_resolve_client_id_with_existing(self) -> None:
        """Test client ID resolution when already present."""
        # Given
        oauth_token = {"client_id": "existing_client_id"}
        provider = "test-provider"
        token_analysis = {"has_client_id": True}

        # When
        from llm_orc.cli_modules.commands.auth_commands import _resolve_client_id

        client_id = _resolve_client_id(provider, token_analysis, oauth_token)

        # Then
        assert client_id == "existing_client_id"

    @patch("click.echo")
    def test_resolve_client_id_anthropic_default(self, mock_echo: Mock) -> None:
        """Test client ID resolution for anthropic with default."""
        # Given
        oauth_token: dict[str, Any] = {}
        provider = "anthropic-claude-pro-max"
        token_analysis = {"has_client_id": False}

        # When
        from llm_orc.cli_modules.commands.auth_commands import _resolve_client_id

        client_id = _resolve_client_id(provider, token_analysis, oauth_token)

        # Then
        assert client_id == "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
        mock_echo.assert_any_call(
            "\n🔧 Using default client ID: 9d1c250a-e61b-44d9-88ed-5944d1962f5e"
        )

    @patch("click.echo")
    def test_resolve_client_id_missing_other_provider(self, mock_echo: Mock) -> None:
        """Test client ID resolution for other provider without client ID."""
        # Given
        oauth_token: dict[str, Any] = {}
        provider = "other-provider"
        token_analysis = {"has_client_id": False}

        # When
        from llm_orc.cli_modules.commands.auth_commands import _resolve_client_id

        client_id = _resolve_client_id(provider, token_analysis, oauth_token)

        # Then
        assert client_id is None
        mock_echo.assert_any_call("\n❌ Cannot test refresh: no client ID available")


class TestLogoutOAuthProvidersHelperMethods:
    """Test helper methods extracted from logout_oauth_providers for complexity."""

    def test_handle_all_providers_logout_with_results(self) -> None:
        """Test _handle_all_providers_logout helper method with successful results."""
        # Given
        mock_auth_manager = Mock()
        mock_auth_manager.logout_all_oauth_providers.return_value = {
            "provider1": True,
            "provider2": False,
            "provider3": True,
        }

        # When
        with patch("click.echo") as mock_echo:
            _handle_all_providers_logout(mock_auth_manager)

        # Then
        mock_auth_manager.logout_all_oauth_providers.assert_called_once()
        echo_calls = [call[0][0] for call in mock_echo.call_args_list]
        assert any("Logged out from 2 OAuth providers" in call for call in echo_calls)

    def test_handle_all_providers_logout_no_providers(self) -> None:
        """Test _handle_all_providers_logout helper method with no providers found."""
        # Given
        mock_auth_manager = Mock()
        mock_auth_manager.logout_all_oauth_providers.return_value = {}

        # When
        with patch("click.echo") as mock_echo:
            _handle_all_providers_logout(mock_auth_manager)

        # Then
        mock_echo.assert_called_once_with("No OAuth providers found to logout")

    def test_handle_single_provider_logout_success(self) -> None:
        """Test _handle_single_provider_logout helper method with successful logout."""
        # Given
        provider = "test-provider"
        mock_auth_manager = Mock()
        mock_auth_manager.logout_oauth_provider.return_value = True

        # When
        with patch("click.echo") as mock_echo:
            _handle_single_provider_logout(provider, mock_auth_manager)

        # Then
        mock_auth_manager.logout_oauth_provider.assert_called_once_with(provider)
        mock_echo.assert_called_once_with(f"✅ Logged out from {provider}")

    def test_handle_single_provider_logout_failure(self) -> None:
        """Test _handle_single_provider_logout helper method with failed logout."""
        # Given
        provider = "test-provider"
        mock_auth_manager = Mock()
        mock_auth_manager.logout_oauth_provider.return_value = False

        # When/Then
        with pytest.raises(
            click.ClickException,
            match=(
                "Failed to logout from test-provider. "
                "Provider may not exist or is not an OAuth provider."
            ),
        ):
            _handle_single_provider_logout(provider, mock_auth_manager)

    def test_display_logout_results_with_mixed_results(self) -> None:
        """Test _display_logout_results helper method with mixed success/failure."""
        # Given
        results = {
            "provider1": True,
            "provider2": False,
            "provider3": True,
            "provider4": False,
        }

        # When
        with patch("click.echo") as mock_echo:
            _display_logout_results(results)

        # Then
        echo_calls = [call[0][0] for call in mock_echo.call_args_list]
        # Should show count of successful logouts
        assert any("Logged out from 2 OAuth providers" in call for call in echo_calls)
        # Should show individual provider results
        assert any("provider1: ✅" in call for call in echo_calls)
        assert any("provider2: ❌" in call for call in echo_calls)
        assert any("provider3: ✅" in call for call in echo_calls)
        assert any("provider4: ❌" in call for call in echo_calls)
