"""Tests for CLI OAuth authentication commands following TDD approach."""

import tempfile
from collections.abc import Generator, Iterator
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from llm_orc.cli import cli


@pytest.fixture(autouse=True)
def mock_expensive_dependencies() -> Generator[None, None, None]:
    """Mock expensive dependencies for all CLI OAuth tests."""
    config_manager_path = (
        "llm_orc.cli_modules.commands.auth_commands.ConfigurationManager"
    )
    auth_manager_path = (
        "llm_orc.cli_modules.commands.auth_commands.AuthenticationManager"
    )
    credential_storage_path = (
        "llm_orc.cli_modules.commands.auth_commands.CredentialStorage"
    )

    with patch(config_manager_path):
        with patch(auth_manager_path):
            with patch(credential_storage_path):
                yield


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

    def test_auth_add_help_mentions_oauth(self, runner: CliRunner) -> None:
        """Test that help text mentions OAuth as an authentication option."""
        # When
        result = runner.invoke(cli, ["auth", "add", "--help"])

        # Then
        assert result.exit_code == 0
        assert "oauth" in result.output.lower()
