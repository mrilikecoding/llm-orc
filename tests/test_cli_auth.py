"""Tests for CLI authentication commands with new ConfigurationManager."""

import tempfile
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from llm_orc.cli import cli


class TestAuthCommandsNew:
    """Test CLI authentication commands with new ConfigurationManager."""

    @pytest.fixture
    def temp_config_dir(self) -> Iterator[Path]:
        """Create a temporary config directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Click test runner."""
        return CliRunner()

    def test_auth_add_command_stores_api_key(
        self, runner: CliRunner, temp_config_dir: Path
    ) -> None:
        """Test that 'auth add' command stores API key."""
        # Given
        provider = "anthropic"
        api_key = "test_key_123"

        # When
        config_manager_path = (
            "llm_orc.cli_modules.commands.auth_commands.ConfigurationManager"
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

            result = runner.invoke(
                cli,
                [
                    "auth",
                    "add",
                    provider,
                    "--api-key",
                    api_key,
                ],
            )

        # Then
        assert result.exit_code == 0
        assert f"API key for {provider} added successfully" in result.output

    def test_auth_list_command_shows_configured_providers(
        self, runner: CliRunner, temp_config_dir: Path
    ) -> None:
        """Test that 'auth list' command shows configured providers."""
        # Given - Set up some providers
        config_manager_path = (
            "llm_orc.cli_modules.commands.auth_commands.ConfigurationManager"
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

            # Add some providers first
            runner.invoke(
                cli,
                [
                    "auth",
                    "add",
                    "anthropic",
                    "--api-key",
                    "key1",
                ],
            )
            runner.invoke(
                cli,
                [
                    "auth",
                    "add",
                    "google",
                    "--api-key",
                    "key2",
                ],
            )

            # When
            result = runner.invoke(cli, ["auth", "list"])

            # Then
            assert result.exit_code == 0
            assert "anthropic" in result.output
            assert "google" in result.output
            assert "API key" in result.output

    def test_auth_list_command_shows_no_providers_message(
        self, runner: CliRunner, temp_config_dir: Path
    ) -> None:
        """Test that 'auth list' command shows message when no providers configured."""
        # Given - No providers configured
        # When
        config_manager_path = (
            "llm_orc.cli_modules.commands.auth_commands.ConfigurationManager"
        )
        credential_storage_path = (
            "llm_orc.cli_modules.commands.auth_commands.CredentialStorage"
        )
        with patch(config_manager_path) as mock_config_manager:
            with patch(credential_storage_path) as mock_storage_class:
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

                # Mock CredentialStorage to return no providers
                mock_storage = Mock()
                mock_storage.list_providers.return_value = []
                mock_storage_class.return_value = mock_storage

                result = runner.invoke(cli, ["auth", "list"])

        # Then
        assert result.exit_code == 0
        assert "No authentication providers configured" in result.output

    def test_auth_remove_command_deletes_provider(
        self, runner: CliRunner, temp_config_dir: Path
    ) -> None:
        """Test that 'auth remove' command deletes provider."""
        # Given
        provider = "anthropic"

        config_manager_path = (
            "llm_orc.cli_modules.commands.auth_commands.ConfigurationManager"
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

            # Add provider first
            runner.invoke(
                cli,
                [
                    "auth",
                    "add",
                    provider,
                    "--api-key",
                    "test_key",
                ],
            )

            # When
            result = runner.invoke(cli, ["auth", "remove", provider])

            # Then
            assert result.exit_code == 0
            assert f"Authentication for {provider} removed" in result.output

    def test_auth_remove_command_fails_for_nonexistent_provider(
        self, runner: CliRunner, temp_config_dir: Path
    ) -> None:
        """Test that 'auth remove' command fails for nonexistent provider."""
        # Given
        provider = "nonexistent"

        # When
        config_manager_path = (
            "llm_orc.cli_modules.commands.auth_commands.ConfigurationManager"
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

            result = runner.invoke(cli, ["auth", "remove", provider])

        # Then
        assert result.exit_code != 0
        assert f"No authentication found for {provider}" in result.output

    def test_auth_setup_command_interactive_wizard(
        self, runner: CliRunner, temp_config_dir: Path
    ) -> None:
        """Test that 'auth setup' command runs interactive wizard."""
        # Given
        # Mock user input - select anthropic-api provider directly
        inputs = [
            "1",  # Select first provider (anthropic-api)
            "test_key_123",  # API key
            "n",  # No more providers
        ]  # provider selection, api_key, no more

        # When
        config_manager_path = (
            "llm_orc.cli_modules.commands.auth_commands.ConfigurationManager"
        )
        credential_storage_path = (
            "llm_orc.cli_modules.commands.auth_commands.CredentialStorage"
        )
        auth_manager_path = (
            "llm_orc.cli_modules.commands.auth_commands.AuthenticationManager"
        )
        with patch(config_manager_path) as mock_config_manager:
            with patch(credential_storage_path) as mock_storage_class:
                with patch(auth_manager_path) as mock_auth_class:
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

                    # Mock CredentialStorage
                    mock_storage = Mock()
                    mock_storage.list_providers.return_value = []
                    mock_storage.store_api_key.return_value = None
                    mock_storage_class.return_value = mock_storage

                    # Mock AuthenticationManager
                    mock_auth_manager = Mock()
                    mock_auth_class.return_value = mock_auth_manager

                    result = runner.invoke(
                        cli,
                        ["auth", "setup"],
                        input="\n".join(inputs),
                    )

        # Then
        assert result.exit_code == 0
        assert "Welcome to LLM Orchestra setup!" in result.output
        assert "Anthropic API key configured!" in result.output
        assert "Setup complete!" in result.output

    def test_auth_add_anthropic_interactive_api_key(
        self, runner: CliRunner, temp_config_dir: Path
    ) -> None:
        """Test interactive Anthropic auth setup choosing API key."""
        # Given - user chooses API key option
        inputs = [
            "1",  # Choose API key option
            "sk-ant-test123",  # API key input
        ]

        # When
        config_manager_path = (
            "llm_orc.cli_modules.commands.auth_commands.ConfigurationManager"
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

            result = runner.invoke(
                cli,
                ["auth", "add", "anthropic"],
                input="\n".join(inputs),
            )

        # Then
        assert result.exit_code == 0
        assert "How would you like to authenticate with Anthropic?" in result.output
        assert "1. API Key (for Anthropic API access)" in result.output
        assert (
            "2. Claude Pro/Max OAuth (for your existing Claude subscription)"
            in result.output
        )
        assert "✅ API key configured as 'anthropic-api'" in result.output

    def test_auth_add_anthropic_interactive_oauth(
        self, runner: CliRunner, temp_config_dir: Path
    ) -> None:
        """Test interactive Anthropic auth setup choosing OAuth."""
        # Given - user chooses OAuth option
        inputs = [
            "2",  # Choose OAuth option
            # OAuth flow would be mocked in real implementation
        ]

        # When
        config_manager_path = (
            "llm_orc.cli_modules.commands.auth_commands.ConfigurationManager"
        )
        setup_oauth_path = "llm_orc.cli_modules.utils.auth_utils.setup_anthropic_oauth"
        with patch(config_manager_path) as mock_config_manager:
            with patch(setup_oauth_path) as mock_setup_oauth:
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

                # Mock successful OAuth flow - just return successfully
                mock_setup_oauth.return_value = None

                result = runner.invoke(
                    cli,
                    ["auth", "add", "anthropic"],
                    input="\n".join(inputs),
                )

        # Then
        assert result.exit_code == 0
        assert "How would you like to authenticate with Anthropic?" in result.output
        assert "2. Claude Pro/Max OAuth" in result.output
        assert "✅ OAuth configured as 'anthropic-claude-pro-max'" in result.output

    def test_auth_add_anthropic_interactive_both(
        self, runner: CliRunner, temp_config_dir: Path
    ) -> None:
        """Test interactive Anthropic auth setup choosing both methods."""
        # Given - user chooses both options
        inputs = [
            "3",  # Choose both option
            "sk-ant-test123",  # API key input
        ]

        # When
        config_manager_path = (
            "llm_orc.cli_modules.commands.auth_commands.ConfigurationManager"
        )
        setup_oauth_path = "llm_orc.cli_modules.utils.auth_utils.setup_anthropic_oauth"
        with patch(config_manager_path) as mock_config_manager:
            with patch(setup_oauth_path) as mock_setup_oauth:
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

                # Mock successful OAuth flow - just return successfully
                mock_setup_oauth.return_value = None

                result = runner.invoke(
                    cli,
                    ["auth", "add", "anthropic"],
                    input="\n".join(inputs),
                )

        # Then
        assert result.exit_code == 0
        assert "How would you like to authenticate with Anthropic?" in result.output
        assert "3. Both (set up multiple authentication methods)" in result.output
        assert "✅ API key configured as 'anthropic-api'" in result.output
        assert "✅ OAuth configured as 'anthropic-claude-pro-max'" in result.output

    def test_auth_add_claude_cli_when_available(
        self, runner: CliRunner, temp_config_dir: Path
    ) -> None:
        """Test adding claude-cli authentication when claude command is available."""
        # When - Claude CLI is available
        config_manager_path = (
            "llm_orc.cli_modules.commands.auth_commands.ConfigurationManager"
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

            # Mock claude command being available
            with patch("shutil.which", return_value="/usr/local/bin/claude"):
                result = runner.invoke(cli, ["auth", "add", "claude-cli"])

        # Then
        assert result.exit_code == 0
        assert "✅ Claude CLI authentication configured" in result.output
        assert "Using local claude command at: /usr/local/bin/claude" in result.output

    def test_auth_add_claude_cli_when_not_available(
        self, runner: CliRunner, temp_config_dir: Path
    ) -> None:
        """Test adding claude-cli auth when claude command is not available."""
        # When - Claude CLI is not available
        config_manager_path = (
            "llm_orc.cli_modules.commands.auth_commands.ConfigurationManager"
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

            # Mock claude command not available
            with patch("shutil.which", return_value=None):
                result = runner.invoke(cli, ["auth", "add", "claude-cli"])

        # Then
        assert result.exit_code != 0
        assert "Claude CLI not found" in result.output
        assert "Please install the Claude CLI" in result.output

    def test_auth_logout_oauth_provider(
        self, runner: CliRunner, temp_config_dir: Path
    ) -> None:
        """Test that 'auth logout' command logs out OAuth provider."""
        # Given - Set up OAuth provider first
        config_manager_path = (
            "llm_orc.cli_modules.commands.auth_commands.ConfigurationManager"
        )
        auth_manager_path = (
            "llm_orc.cli_modules.commands.auth_commands.AuthenticationManager"
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

            # Mock successful logout
            with patch(auth_manager_path) as mock_auth:
                mock_auth_manager = mock_auth.return_value
                mock_auth_manager.logout_oauth_provider.return_value = True

                # When
                result = runner.invoke(
                    cli, ["auth", "logout", "anthropic-claude-pro-max"]
                )

                # Then
                assert result.exit_code == 0
                assert "Logged out from anthropic-claude-pro-max" in result.output
                mock_auth_manager.logout_oauth_provider.assert_called_once_with(
                    "anthropic-claude-pro-max"
                )

    def test_auth_logout_nonexistent_provider(
        self, runner: CliRunner, temp_config_dir: Path
    ) -> None:
        """Test that 'auth logout' command fails for nonexistent provider."""
        # Given
        config_manager_path = (
            "llm_orc.cli_modules.commands.auth_commands.ConfigurationManager"
        )
        auth_manager_path = (
            "llm_orc.cli_modules.commands.auth_commands.AuthenticationManager"
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

            # Mock failed logout (provider doesn't exist)
            with patch(auth_manager_path) as mock_auth:
                mock_auth_manager = mock_auth.return_value
                mock_auth_manager.logout_oauth_provider.return_value = False

                # When
                result = runner.invoke(cli, ["auth", "logout", "nonexistent-provider"])

                # Then
                assert result.exit_code != 0
                assert "Failed to logout" in result.output

    def test_auth_add_replaces_existing_provider(
        self, runner: CliRunner, temp_config_dir: Path
    ) -> None:
        """Test that adding a provider automatically removes existing one."""
        # Given
        provider = "anthropic-api"
        api_key = "new_test_key_123"

        # When
        config_manager_path = (
            "llm_orc.cli_modules.commands.auth_commands.ConfigurationManager"
        )
        credential_storage_path = (
            "llm_orc.cli_modules.commands.auth_commands.CredentialStorage"
        )
        auth_manager_path = (
            "llm_orc.cli_modules.commands.auth_commands.AuthenticationManager"
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
                mock_storage.list_providers.return_value = [provider]  # Provider exists

                with patch(auth_manager_path):
                    result = runner.invoke(
                        cli, ["auth", "add", provider, "--api-key", api_key]
                    )

                    # Then
                    assert result.exit_code == 0
                    # Should remove existing provider then add new one
                    mock_storage.remove_provider.assert_called_once_with(provider)
                    mock_storage.store_api_key.assert_called_once_with(
                        provider, api_key
                    )
                    assert "Existing authentication found" in result.output
                    assert "Old authentication removed" in result.output
                    assert f"API key for {provider} added successfully" in result.output

    def test_auth_logout_all_command(
        self, runner: CliRunner, temp_config_dir: Path
    ) -> None:
        """Test that 'auth logout --all' command logs out all OAuth providers."""
        # Given
        config_manager_path = (
            "llm_orc.cli_modules.commands.auth_commands.ConfigurationManager"
        )
        auth_manager_path = (
            "llm_orc.cli_modules.commands.auth_commands.AuthenticationManager"
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

            # Mock successful logout of multiple providers
            with patch(auth_manager_path) as mock_auth:
                mock_auth_manager = mock_auth.return_value
                mock_auth_manager.logout_all_oauth_providers.return_value = {
                    "anthropic-claude-pro-max": True,
                    "google-oauth": True,
                }

                # When
                result = runner.invoke(cli, ["auth", "logout", "--all"])

                # Then
                assert result.exit_code == 0
                assert "Logged out from 2 OAuth providers" in result.output
                assert "anthropic-claude-pro-max: ✅" in result.output
                assert "google-oauth: ✅" in result.output
                mock_auth_manager.logout_all_oauth_providers.assert_called_once()
