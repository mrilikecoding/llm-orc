"""Tests for CLI OAuth authentication commands following TDD approach."""

import tempfile
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from llm_orc.cli import cli


class TestOAuthCLI:
    """Test CLI OAuth authentication functionality."""

    @pytest.fixture
    def temp_config_dir(self) -> Iterator[Path]:
        """Create a temporary config directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Click test runner."""
        return CliRunner()

    def test_auth_add_anthropic_claude_pro_max_triggers_oauth_flow(
        self, runner: CliRunner, temp_config_dir: Path
    ) -> None:
        """Test anthropic-claude-pro-max triggers OAuth flow without credentials."""
        # Given
        provider = "anthropic-claude-pro-max"

        # Mock the configuration and storage
        with patch("llm_orc.cli.ConfigurationManager") as mock_config_manager:
            with patch("webbrowser.open") as mock_webbrowser_open:
                with patch("requests.post") as mock_requests_post:
                    # Setup config manager mocks
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

                    # Setup successful OAuth token exchange
                    mock_response = Mock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {
                        "access_token": "test_access_token",
                        "refresh_token": "test_refresh_token",
                        "expires_in": 3600,
                    }
                    mock_requests_post.return_value = mock_response

                    # When - invoke the command with simulated user input
                    result = runner.invoke(
                        cli,
                        ["auth", "add", provider],
                        input="y\ntest_code#test_state\n",  # Simulate user input
                    )

                    # Then
                    assert result.exit_code == 0
                    assert "Claude Pro/Max OAuth Authentication" in result.output
                    assert "OAuth authentication successful" in result.output
                    assert (
                        "Tokens stored as 'anthropic-claude-pro-max'" in result.output
                    )

                    # Verify OAuth flow was triggered
                    mock_webbrowser_open.assert_called_once()
                    mock_requests_post.assert_called_once()

                    # Verify the OAuth URL contains expected parameters
                    called_url = mock_webbrowser_open.call_args[0][0]
                    assert "claude.ai/oauth/authorize" in called_url
                    assert (
                        "client_id=9d1c250a-e61b-44d9-88ed-5944d1962f5e" in called_url
                    )
                    assert (
                        "scope=org%3Acreate_api_key+user%3Aprofile+user%3Ainference"
                        in called_url
                    )

    def test_auth_add_anthropic_claude_pro_max_help_mentions_oauth(
        self, runner: CliRunner
    ) -> None:
        """Test that help text mentions anthropic-claude-pro-max as an OAuth option."""
        # When
        result = runner.invoke(cli, ["auth", "add", "--help"])

        # Then
        assert result.exit_code == 0
        # This test will initially fail since we haven't documented it yet
        assert (
            "anthropic-claude-pro-max" in result.output.lower()
            or "oauth" in result.output.lower()
        )

    def test_auth_add_anthropic_claude_pro_max_without_client_credentials_succeeds(
        self, runner: CliRunner, temp_config_dir: Path
    ) -> None:
        """Test anthropic-claude-pro-max doesn't require client credentials."""
        # Given
        provider = "anthropic-claude-pro-max"

        # Mock configuration
        with patch("llm_orc.cli.ConfigurationManager") as mock_config_manager:
            with patch("webbrowser.open"):
                with patch("requests.post") as mock_requests_post:
                    # Setup mocks
                    mock_instance = mock_config_manager.return_value
                    mock_instance._global_config_dir = temp_config_dir
                    mock_instance.ensure_global_config_dir.return_value = None
                    mock_instance.get_credentials_file.return_value = (
                        temp_config_dir / "credentials.yaml"
                    )
                    mock_instance.get_encryption_key_file.return_value = (
                        temp_config_dir / ".encryption_key"
                    )

                    # Setup successful OAuth response
                    mock_response = Mock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {
                        "access_token": "test_token",
                        "refresh_token": "test_refresh",
                        "expires_in": 3600,
                    }
                    mock_requests_post.return_value = mock_response

                    # When - invoke without any OAuth credentials
                    result = runner.invoke(
                        cli,
                        ["auth", "add", provider],
                        input="y\ntest_code#test_state\n",
                    )

                    # Then - should NOT show the error about missing client credentials
                    assert result.exit_code == 0
                    assert (
                        "Must provide either --api-key or both "
                        "--client-id and --client-secret" not in result.output
                    )
