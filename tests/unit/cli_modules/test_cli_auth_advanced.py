"""Advanced tests for CLI authentication commands covering complex scenarios."""

import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from llm_orc.cli import cli


class TestAuthCommandsAdvanced:
    """Test advanced CLI authentication command scenarios."""

    @pytest.fixture
    def temp_config_dir(self) -> Iterator[Path]:
        """Create a temporary config directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Click test runner."""
        return CliRunner()

    def test_auth_test_refresh_no_provider_found(
        self, runner: CliRunner, temp_config_dir: Path
    ) -> None:
        """Test token refresh when provider not found."""
        # Given
        provider = "nonexistent-provider"

        config_manager_path = (
            "llm_orc.cli_modules.commands.auth_commands.ConfigurationManager"
        )
        credential_storage_path = (
            "llm_orc.cli_modules.commands.auth_commands.CredentialStorage"
        )
        with patch(config_manager_path) as mock_config_manager:
            mock_instance = mock_config_manager.return_value
            mock_instance._global_config_dir = temp_config_dir
            mock_instance.ensure_global_config_dir.return_value = None
            mock_instance.get_credentials_file.return_value = (
                temp_config_dir / "credentials.yaml"
            )
            mock_instance.get_encryption_key_file.return_value = (
                temp_config_dir / ".encryption_key"
            )
            mock_instance.needs_migration.return_value = False

            with patch(credential_storage_path) as mock_storage_class:
                mock_storage = Mock()
                mock_storage_class.return_value = mock_storage
                mock_storage.list_providers.return_value = []

                # When
                result = runner.invoke(cli, ["auth", "test-refresh", provider])

                # Then
                assert result.exit_code != 0
                assert f"No authentication found for {provider}" in result.output

    def test_auth_test_refresh_no_oauth_token(
        self, runner: CliRunner, temp_config_dir: Path
    ) -> None:
        """Test token refresh when no OAuth token found."""
        # Given
        provider = "test-provider"

        config_manager_path = (
            "llm_orc.cli_modules.commands.auth_commands.ConfigurationManager"
        )
        credential_storage_path = (
            "llm_orc.cli_modules.commands.auth_commands.CredentialStorage"
        )
        with patch(config_manager_path) as mock_config_manager_class:
            with patch(credential_storage_path) as mock_storage_class:
                # Mock config manager
                mock_config_manager = Mock()
                mock_config_manager_class.return_value = mock_config_manager
                mock_config_manager._global_config_dir = temp_config_dir
                mock_config_manager.ensure_global_config_dir.return_value = None
                mock_config_manager.get_credentials_file.return_value = (
                    temp_config_dir / "credentials.yaml"
                )
                mock_config_manager.get_encryption_key_file.return_value = (
                    temp_config_dir / ".encryption_key"
                )
                mock_config_manager.needs_migration.return_value = False

                # Mock storage
                mock_storage = Mock()
                mock_storage_class.return_value = mock_storage
                mock_storage.list_providers.return_value = [provider]
                mock_storage.get_oauth_token.return_value = None

                # When
                result = runner.invoke(cli, ["auth", "test-refresh", provider])

                # Then
                assert result.exit_code != 0
                assert f"No OAuth token found for {provider}" in result.output

    def test_auth_test_refresh_token_status_display(
        self, runner: CliRunner, temp_config_dir: Path
    ) -> None:
        """Test token status display with no refresh token."""
        # Given
        provider = "test-provider"
        oauth_token = {
            "access_token": "test_access_token",
            "client_id": "test_client_id",
            "expires_at": time.time() + 3600,
        }

        config_manager_path = (
            "llm_orc.cli_modules.commands.auth_commands.ConfigurationManager"
        )
        credential_storage_path = (
            "llm_orc.cli_modules.commands.auth_commands.CredentialStorage"
        )
        with patch(config_manager_path) as mock_config_manager:
            mock_instance = mock_config_manager.return_value
            mock_instance._global_config_dir = temp_config_dir
            mock_instance.ensure_global_config_dir.return_value = None
            mock_instance.get_credentials_file.return_value = (
                temp_config_dir / "credentials.yaml"
            )
            mock_instance.get_encryption_key_file.return_value = (
                temp_config_dir / ".encryption_key"
            )
            mock_instance.needs_migration.return_value = False

            with patch(credential_storage_path) as mock_storage_class:
                mock_storage = Mock()
                mock_storage_class.return_value = mock_storage
                mock_storage.list_providers.return_value = [provider]
                mock_storage.get_oauth_token.return_value = oauth_token

                # When
                result = runner.invoke(cli, ["auth", "test-refresh", provider])

                # Then
                assert result.exit_code == 0
                assert f"Token info for {provider}:" in result.output
                assert "Has refresh token: ❌" in result.output
                assert "Has client ID: ✅" in result.output
                assert "Has expiration: ✅" in result.output

    def test_auth_test_refresh_expired_token(
        self, runner: CliRunner, temp_config_dir: Path
    ) -> None:
        """Test token status display for expired token."""
        # Given
        provider = "test-provider"
        expired_time = time.time() - 3600  # Expired 1 hour ago
        oauth_token = {
            "access_token": "test_access_token",
            "refresh_token": "test_refresh_token",
            "client_id": "existing_client_id",
            "expires_at": expired_time,
        }

        config_manager_path = (
            "llm_orc.cli_modules.commands.auth_commands.ConfigurationManager"
        )
        credential_storage_path = (
            "llm_orc.cli_modules.commands.auth_commands.CredentialStorage"
        )

        with patch(config_manager_path) as mock_config_manager:
            mock_instance = mock_config_manager.return_value
            mock_instance._global_config_dir = temp_config_dir
            mock_instance.ensure_global_config_dir.return_value = None
            mock_instance.get_credentials_file.return_value = (
                temp_config_dir / "credentials.yaml"
            )
            mock_instance.get_encryption_key_file.return_value = (
                temp_config_dir / ".encryption_key"
            )
            mock_instance.needs_migration.return_value = False

            with patch(credential_storage_path) as mock_storage_class:
                mock_storage = Mock()
                mock_storage_class.return_value = mock_storage
                mock_storage.list_providers.return_value = [provider]
                mock_storage.get_oauth_token.return_value = oauth_token

                # When
                result = runner.invoke(cli, ["auth", "test-refresh", provider])

                # Then
                assert result.exit_code == 0
                assert "Token expired" in result.output

    def test_auth_test_refresh_exception(
        self, runner: CliRunner, temp_config_dir: Path
    ) -> None:
        """Test exception handling in token refresh."""
        # Given
        provider = "test-provider"

        config_manager_path = (
            "llm_orc.cli_modules.commands.auth_commands.ConfigurationManager"
        )
        credential_storage_path = (
            "llm_orc.cli_modules.commands.auth_commands.CredentialStorage"
        )
        with patch(config_manager_path) as mock_config_manager:
            mock_instance = mock_config_manager.return_value
            mock_instance._global_config_dir = temp_config_dir
            mock_instance.ensure_global_config_dir.return_value = None
            mock_instance.get_credentials_file.return_value = (
                temp_config_dir / "credentials.yaml"
            )
            mock_instance.get_encryption_key_file.return_value = (
                temp_config_dir / ".encryption_key"
            )
            mock_instance.needs_migration.return_value = False

            with patch(credential_storage_path) as mock_storage_class:
                mock_storage = Mock()
                mock_storage_class.return_value = mock_storage
                mock_storage.list_providers.side_effect = Exception("Storage error")

                # When
                result = runner.invoke(cli, ["auth", "test-refresh", provider])

                # Then
                assert result.exit_code != 0
                assert "Failed to check token status: Storage error" in result.output

    def test_auth_add_claude_cli_success(
        self, runner: CliRunner, temp_config_dir: Path
    ) -> None:
        """Test successful claude-cli authentication setup."""
        # Given
        provider = "claude-cli"

        config_manager_path = (
            "llm_orc.cli_modules.commands.auth_commands.ConfigurationManager"
        )
        claude_cli_auth_path = (
            "llm_orc.cli_modules.utils.auth_utils.handle_claude_cli_auth"
        )
        credential_storage_path = "llm_orc.core.auth.authentication.CredentialStorage"
        with patch(config_manager_path) as mock_config_manager:
            mock_instance = mock_config_manager.return_value
            mock_instance._global_config_dir = temp_config_dir
            mock_instance.ensure_global_config_dir.return_value = None
            mock_instance.get_credentials_file.return_value = (
                temp_config_dir / "credentials.yaml"
            )
            mock_instance.get_encryption_key_file.return_value = (
                temp_config_dir / ".encryption_key"
            )
            mock_instance.needs_migration.return_value = False

            with patch(credential_storage_path) as mock_storage_class:
                mock_storage = Mock()
                mock_storage_class.return_value = mock_storage

                with patch("shutil.which") as mock_which:
                    # Mock Claude CLI is available
                    mock_which.return_value = "/usr/local/bin/claude"

                    with patch(claude_cli_auth_path) as mock_claude_cli_auth:
                        # Mock successful authentication
                        mock_claude_cli_auth.return_value = None

                        # When
                        result = runner.invoke(cli, ["auth", "add", provider])

                        # Then - test succeeds when Claude CLI is available
                        assert result.exit_code == 0
                        assert "Claude CLI authentication configured" in result.output

    def test_auth_add_anthropic_interactive_success(
        self, runner: CliRunner, temp_config_dir: Path
    ) -> None:
        """Test successful anthropic interactive authentication."""
        # Given
        provider = "anthropic"

        config_manager_path = (
            "llm_orc.cli_modules.commands.auth_commands.ConfigurationManager"
        )
        interactive_auth_path = (
            "llm_orc.cli_modules.commands.auth_commands."
            "handle_anthropic_interactive_auth"
        )
        credential_storage_path = "llm_orc.core.auth.authentication.CredentialStorage"
        with patch(config_manager_path) as mock_config_manager:
            mock_instance = mock_config_manager.return_value
            mock_instance._global_config_dir = temp_config_dir
            mock_instance.ensure_global_config_dir.return_value = None
            mock_instance.get_credentials_file.return_value = (
                temp_config_dir / "credentials.yaml"
            )
            mock_instance.get_encryption_key_file.return_value = (
                temp_config_dir / ".encryption_key"
            )
            mock_instance.needs_migration.return_value = False

            with patch(credential_storage_path) as mock_storage_class:
                mock_storage = Mock()
                mock_storage_class.return_value = mock_storage

                with patch(interactive_auth_path) as mock_interactive_auth:
                    mock_interactive_auth.return_value = None  # Mock successful auth

                    # When
                    result = runner.invoke(cli, ["auth", "add", provider])

                    # Then - test succeeds when interactive auth is successful
                    assert result.exit_code == 0

    def test_auth_list_interactive_mode_basic_flow(
        self, runner: CliRunner, temp_config_dir: Path
    ) -> None:
        """Test basic flow of interactive list mode."""
        # Given
        providers = ["anthropic-api", "google-gemini"]

        config_manager_path = (
            "llm_orc.cli_modules.commands.auth_commands.ConfigurationManager"
        )
        credential_storage_path = (
            "llm_orc.cli_modules.commands.auth_commands.CredentialStorage"
        )
        auth_menus_path = "llm_orc.menu_system.AuthMenus"

        with patch(config_manager_path) as mock_config_manager:
            mock_instance = mock_config_manager.return_value
            mock_instance._global_config_dir = temp_config_dir
            mock_instance.ensure_global_config_dir.return_value = None
            mock_instance.get_credentials_file.return_value = (
                temp_config_dir / "credentials.yaml"
            )
            mock_instance.get_encryption_key_file.return_value = (
                temp_config_dir / ".encryption_key"
            )
            mock_instance.needs_migration.return_value = False

            with patch(credential_storage_path) as mock_storage_class:
                with patch(auth_menus_path) as mock_auth_menus:
                    mock_storage = Mock()
                    mock_storage_class.return_value = mock_storage
                    mock_storage.list_providers.return_value = providers

                    # Mock the menu to quit immediately
                    mock_auth_menus.auth_list_actions.return_value = ("quit", None)

                    # When
                    result = runner.invoke(cli, ["auth", "list", "--interactive"])

                    # Then
                    assert result.exit_code == 0
                    mock_auth_menus.auth_list_actions.assert_called_once_with(providers)

    def test_auth_list_interactive_test_action(
        self, runner: CliRunner, temp_config_dir: Path
    ) -> None:
        """Test test action in interactive list mode."""
        # Given
        providers = ["test-provider"]
        selected_provider = "test-provider"

        config_manager_path = (
            "llm_orc.cli_modules.commands.auth_commands.ConfigurationManager"
        )
        credential_storage_path = (
            "llm_orc.cli_modules.commands.auth_commands.CredentialStorage"
        )
        auth_menus_path = "llm_orc.menu_system.AuthMenus"
        test_provider_auth_path = (
            "llm_orc.cli_modules.utils.auth_utils.validate_provider_authentication"
        )
        show_success_path = "llm_orc.menu_system.show_success"
        show_working_path = "llm_orc.menu_system.show_working"

        with patch(config_manager_path) as mock_config_manager:
            mock_instance = mock_config_manager.return_value
            mock_instance._global_config_dir = temp_config_dir
            mock_instance.ensure_global_config_dir.return_value = None
            mock_instance.get_credentials_file.return_value = (
                temp_config_dir / "credentials.yaml"
            )
            mock_instance.get_encryption_key_file.return_value = (
                temp_config_dir / ".encryption_key"
            )
            mock_instance.needs_migration.return_value = False

            with patch(credential_storage_path) as mock_storage_class:
                with patch(auth_menus_path) as mock_auth_menus:
                    with patch(test_provider_auth_path) as mock_test_auth:
                        with patch(show_success_path):
                            with patch(show_working_path):
                                mock_storage = Mock()
                                mock_storage_class.return_value = mock_storage
                                mock_storage.list_providers.return_value = providers

                                # First return test action, then quit
                                mock_auth_menus.auth_list_actions.side_effect = [
                                    ("test", selected_provider),
                                    ("quit", None),
                                ]
                                mock_test_auth.return_value = True

                                # When
                                result = runner.invoke(
                                    cli, ["auth", "list", "--interactive"]
                                )

                                # Then
                                assert result.exit_code == 0
                                mock_test_auth.assert_called_once()
