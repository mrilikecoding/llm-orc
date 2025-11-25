"""Tests for script management CLI commands."""

import json
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from llm_orc.cli_modules.commands.script_commands import (
    list_scripts,
    show_script,
    test_script,
)


@pytest.fixture
def cli_runner() -> CliRunner:
    """Create CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_script_registry() -> Generator[MagicMock, None, None]:
    """Mock PrimitiveRegistry for testing."""
    with patch(
        "llm_orc.cli_modules.commands.script_commands.PrimitiveRegistry"
    ) as mock:
        registry = MagicMock()
        mock.return_value = registry
        yield registry


class TestListScripts:
    """Tests for list_scripts command."""

    def test_list_scripts_shows_discovered_primitives(
        self, cli_runner: CliRunner, mock_script_registry: MagicMock
    ) -> None:
        """List scripts command displays discovered primitive scripts."""
        # Arrange
        mock_script_registry.discover_primitives.return_value = [
            {
                "name": "read_file",
                "category": "file-ops",
                "path": "/path/to/read_file.py",
                "description": "Read file content",
            },
            {
                "name": "get_user_input",
                "category": "user-interaction",
                "path": "/path/to/get_user_input.py",
                "description": "Collect user input",
            },
        ]

        # Act
        result = cli_runner.invoke(list_scripts)

        # Assert
        assert result.exit_code == 0
        assert "read_file" in result.output
        assert "get_user_input" in result.output
        assert "file-ops" in result.output
        assert "user-interaction" in result.output

    def test_list_scripts_filters_by_category(
        self, cli_runner: CliRunner, mock_script_registry: MagicMock
    ) -> None:
        """List scripts command filters by specified category."""
        # Arrange
        mock_script_registry.discover_primitives.return_value = [
            {"name": "read_file", "category": "file-ops"},
            {"name": "get_user_input", "category": "user-interaction"},
        ]

        # Act
        result = cli_runner.invoke(list_scripts, ["--category", "file-ops"])

        # Assert
        assert result.exit_code == 0
        assert "read_file" in result.output
        assert "get_user_input" not in result.output

    def test_list_scripts_json_output(
        self, cli_runner: CliRunner, mock_script_registry: MagicMock
    ) -> None:
        """List scripts command outputs valid JSON with --json flag."""
        # Arrange
        primitives = [
            {"name": "read_file", "category": "file-ops", "path": "/path/read_file.py"}
        ]
        mock_script_registry.discover_primitives.return_value = primitives

        # Act
        result = cli_runner.invoke(list_scripts, ["--json"])

        # Assert
        assert result.exit_code == 0
        output_data = json.loads(result.output)
        assert len(output_data) == 1
        assert output_data[0]["name"] == "read_file"


class TestShowScript:
    """Tests for show_script command."""

    def test_show_script_displays_script_info(
        self, cli_runner: CliRunner, mock_script_registry: MagicMock
    ) -> None:
        """Show script command displays detailed script information."""
        # Arrange
        mock_script_registry.get_primitive_info.return_value = {
            "name": "read_file",
            "category": "file-ops",
            "description": "Read file content and output as JSON",
            "path": "/path/to/read_file.py",
            "parameters": {"path": "str", "encoding": "str"},
            "returns": {"success": "bool", "content": "str"},
        }

        # Act
        result = cli_runner.invoke(show_script, ["read_file"])

        # Assert
        assert result.exit_code == 0
        assert "read_file" in result.output
        assert "Read file content" in result.output
        assert "path" in result.output
        assert "encoding" in result.output

    def test_show_script_handles_missing_script(
        self, cli_runner: CliRunner, mock_script_registry: MagicMock
    ) -> None:
        """Show script command handles script not found."""
        # Arrange
        mock_script_registry.get_primitive_info.side_effect = KeyError("not_found")

        # Act
        result = cli_runner.invoke(show_script, ["nonexistent_script"])

        # Assert
        assert result.exit_code != 0
        assert "not found" in result.output.lower()


class TestTestScript:
    """Tests for test_script command."""

    def test_test_script_executes_with_parameters(
        self, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test script command executes script with provided parameters."""
        # Arrange - create test script
        script_path = tmp_path / "test_script.py"
        script_path.write_text(
            """
import json
import sys

input_data = json.loads(sys.stdin.read())
output = {"success": True, "data": f"Processed: {input_data['test_param']}"}
print(json.dumps(output))
"""
        )
        script_path.chmod(0o755)

        # Act
        result = cli_runner.invoke(
            test_script,
            [
                str(script_path),
                "--parameters",
                json.dumps({"test_param": "test_value"}),
            ],
        )

        # Assert
        assert result.exit_code == 0
        assert "success" in result.output.lower()

    def test_test_script_shows_validation_errors(
        self, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test script command shows validation errors for invalid output."""
        # Arrange - create script with invalid output
        script_path = tmp_path / "invalid_script.py"
        script_path.write_text(
            """
import sys
print("invalid json output")
"""
        )
        script_path.chmod(0o755)

        # Act
        result = cli_runner.invoke(test_script, [str(script_path)])

        # Assert
        assert result.exit_code != 0
        assert "error" in result.output.lower() or "invalid" in result.output.lower()
