"""Tests for CLI authentication commands with new ConfigurationManager."""

import tempfile
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

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
        with patch("llm_orc.cli.ConfigurationManager") as mock_config_manager:
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

        # Verify credentials were stored
        credentials_file = temp_config_dir / "credentials.yaml"
        assert credentials_file.exists()

    def test_auth_list_command_shows_configured_providers(
        self, runner: CliRunner, temp_config_dir: Path
    ) -> None:
        """Test that 'auth list' command shows configured providers."""
        # Given - Set up some providers
        with patch("llm_orc.cli.ConfigurationManager") as mock_config_manager:
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
        with patch("llm_orc.cli.ConfigurationManager") as mock_config_manager:
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

        with patch("llm_orc.cli.ConfigurationManager") as mock_config_manager:
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
        with patch("llm_orc.cli.ConfigurationManager") as mock_config_manager:
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

    def test_auth_test_command_validates_credentials(
        self, runner: CliRunner, temp_config_dir: Path
    ) -> None:
        """Test that 'auth test' command validates credentials."""
        # Given
        provider = "anthropic"
        api_key = "test_key_123"

        with patch("llm_orc.cli.ConfigurationManager") as mock_config_manager:
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
                    api_key,
                ],
            )

            # When
            result = runner.invoke(cli, ["auth", "test", provider])

            # Then
            assert result.exit_code == 0
            assert f"Authentication for {provider} is working" in result.output

    def test_auth_test_command_fails_for_invalid_credentials(
        self, runner: CliRunner, temp_config_dir: Path
    ) -> None:
        """Test that 'auth test' command fails for invalid credentials."""
        # Given
        provider = "anthropic"
        invalid_key = "invalid_key"

        with patch("llm_orc.cli.ConfigurationManager") as mock_config_manager:
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

            # Add provider with invalid key
            runner.invoke(
                cli,
                [
                    "auth",
                    "add",
                    provider,
                    "--api-key",
                    invalid_key,
                ],
            )

            # When
            result = runner.invoke(cli, ["auth", "test", provider])

            # Then
            assert result.exit_code != 0
            assert f"Authentication for {provider} failed" in result.output

    def test_auth_setup_command_interactive_wizard(
        self, runner: CliRunner, temp_config_dir: Path
    ) -> None:
        """Test that 'auth setup' command runs interactive wizard."""
        # Given
        # Mock user input
        inputs = ["anthropic", "test_key_123", "n"]  # provider, api_key, no more

        # When
        with patch("llm_orc.cli.ConfigurationManager") as mock_config_manager:
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
                ["auth", "setup"],
                input="\n".join(inputs),
            )

        # Then
        assert result.exit_code == 0
        assert "Welcome to LLM Orchestra setup!" in result.output
        assert "✓ anthropic configured successfully" in result.output
        assert "Setup complete!" in result.output
