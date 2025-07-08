"""Tests for CLI authentication commands."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from llm_orc.cli import cli


class TestAuthCommands:
    """Test CLI authentication commands."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary config directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def runner(self):
        """Click test runner."""
        return CliRunner()

    def test_auth_add_command_stores_api_key(self, runner, temp_config_dir):
        """Test that 'auth add' command stores API key."""
        # Given
        provider = "anthropic"
        api_key = "test_key_123"

        # When
        result = runner.invoke(
            cli,
            [
                "auth",
                "add",
                provider,
                "--api-key",
                api_key,
                "--config-dir",
                str(temp_config_dir),
            ],
        )

        # Then
        assert result.exit_code == 0
        assert f"API key for {provider} added successfully" in result.output

        # Verify credentials were stored
        credentials_file = temp_config_dir / "credentials.yaml"
        assert credentials_file.exists()

    def test_auth_add_command_fails_without_api_key(self, runner):
        """Test that 'auth add' command fails without API key."""
        # Given
        provider = "anthropic"

        # When
        result = runner.invoke(cli, ["auth", "add", provider])

        # Then
        assert result.exit_code != 0
        assert "Error: Missing option '--api-key'" in result.output

    def test_auth_list_command_shows_configured_providers(
        self, runner, temp_config_dir
    ):
        """Test that 'auth list' command shows configured providers."""
        # Given - Set up some providers
        # Add some providers first
        runner.invoke(
            cli,
            [
                "auth",
                "add",
                "anthropic",
                "--api-key",
                "key1",
                "--config-dir",
                str(temp_config_dir),
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
                "--config-dir",
                str(temp_config_dir),
            ],
        )

        # When
        result = runner.invoke(
            cli, ["auth", "list", "--config-dir", str(temp_config_dir)]
        )

        # Then
        assert result.exit_code == 0
        assert "anthropic" in result.output
        assert "google" in result.output
        assert "API key" in result.output

    def test_auth_list_command_shows_no_providers_message(
        self, runner, temp_config_dir
    ):
        """Test that 'auth list' command shows message when no providers configured."""
        # Given - No providers configured
        # When
        result = runner.invoke(
            cli, ["auth", "list", "--config-dir", str(temp_config_dir)]
        )

        # Then
        assert result.exit_code == 0
        assert "No authentication providers configured" in result.output

    def test_auth_remove_command_deletes_provider(self, runner, temp_config_dir):
        """Test that 'auth remove' command deletes a provider."""
        # Given - Set up a provider
        provider = "anthropic"
        runner.invoke(
            cli,
            [
                "auth",
                "add",
                provider,
                "--api-key",
                "key1",
                "--config-dir",
                str(temp_config_dir),
            ],
        )

        # When
        result = runner.invoke(
            cli, ["auth", "remove", provider, "--config-dir", str(temp_config_dir)]
        )

        # Then
        assert result.exit_code == 0
        assert f"Authentication for {provider} removed" in result.output

        # Verify provider is gone
        list_result = runner.invoke(
            cli, ["auth", "list", "--config-dir", str(temp_config_dir)]
        )
        assert provider not in list_result.output

    def test_auth_remove_command_fails_for_nonexistent_provider(
        self, runner, temp_config_dir
    ):
        """Test that 'auth remove' command fails for non-existent provider."""
        # Given - No providers configured
        provider = "nonexistent"
        # When
        result = runner.invoke(
            cli, ["auth", "remove", provider, "--config-dir", str(temp_config_dir)]
        )

        # Then
        assert result.exit_code != 0
        assert f"No authentication found for {provider}" in result.output

    def test_auth_test_command_validates_credentials(self, runner, temp_config_dir):
        """Test that 'auth test' command validates credentials."""
        # Given - Set up a provider
        provider = "anthropic"
        runner.invoke(
            cli,
            [
                "auth",
                "add",
                provider,
                "--api-key",
                "valid_key",
                "--config-dir",
                str(temp_config_dir),
            ],
        )

        # When
        with patch(
            "llm_orc.authentication.AuthenticationManager.authenticate",
            return_value=True,
        ):
            result = runner.invoke(
                cli, ["auth", "test", provider, "--config-dir", str(temp_config_dir)]
            )

        # Then
        assert result.exit_code == 0
        assert f"Authentication for {provider} is working" in result.output

    def test_auth_test_command_fails_for_invalid_credentials(
        self, runner, temp_config_dir
    ):
        """Test that 'auth test' command fails for invalid credentials."""
        # Given - Set up a provider
        provider = "anthropic"
        runner.invoke(
            cli,
            [
                "auth",
                "add",
                provider,
                "--api-key",
                "invalid_key",
                "--config-dir",
                str(temp_config_dir),
            ],
        )

        # When
        with patch(
            "llm_orc.authentication.AuthenticationManager.authenticate",
            return_value=False,
        ):
            result = runner.invoke(
                cli, ["auth", "test", provider, "--config-dir", str(temp_config_dir)]
            )

        # Then
        assert result.exit_code != 0
        assert f"Authentication for {provider} failed" in result.output

    def test_auth_setup_command_interactive_wizard(self, runner, temp_config_dir):
        """Test that 'auth setup' command runs interactive wizard."""
        # Given
        # Mock user input
        inputs = ["anthropic", "test_key_123", "n"]  # provider, api_key, no more

        # When
        result = runner.invoke(
            cli,
            ["auth", "setup", "--config-dir", str(temp_config_dir)],
            input="\n".join(inputs),
        )

        # Then
        assert result.exit_code == 0
        assert "Welcome to LLM Orchestra setup" in result.output
        assert "Setup complete" in result.output
