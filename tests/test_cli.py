"""Tests for CLI interface."""

import tempfile

import yaml
from click.testing import CliRunner

from llm_orc.cli import cli


class TestCLI:
    """Test CLI interface."""

    def test_cli_help(self) -> None:
        """Test that CLI shows help message."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "llm orchestra" in result.output.lower()

    def test_cli_invoke_command_exists(self) -> None:
        """Test that invoke command exists."""
        runner = CliRunner()
        result = runner.invoke(cli, ["invoke", "--help"])
        assert result.exit_code == 0
        assert "invoke" in result.output.lower()

    def test_cli_invoke_requires_ensemble_name(self) -> None:
        """Test that invoke command requires ensemble name."""
        runner = CliRunner()
        result = runner.invoke(cli, ["invoke"])
        assert result.exit_code != 0
        assert (
            "ensemble" in result.output.lower() or "required" in result.output.lower()
        )

    def test_cli_invoke_with_ensemble_name(self) -> None:
        """Test basic ensemble invocation."""
        runner = CliRunner()
        result = runner.invoke(cli, ["invoke", "test_ensemble"])
        # Should fail because ensemble doesn't exist
        assert result.exit_code != 0
        # Either no ensemble directories found or ensemble not found in existing dirs
        assert (
            "No ensemble directories found" in result.output
            or "test_ensemble" in result.output
        )

    def test_cli_invoke_with_config_option(self) -> None:
        """Test invoke command accepts config directory option."""
        runner = CliRunner()
        result = runner.invoke(cli, ["invoke", "--config-dir", "/tmp", "test_ensemble"])
        assert result.exit_code != 0
        # Should show that it's looking in the specified config directory
        assert "test_ensemble" in result.output

    def test_cli_list_command_exists(self) -> None:
        """Test that list-ensembles command exists."""
        runner = CliRunner()
        result = runner.invoke(cli, ["list-ensembles", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output.lower() or "ensemble" in result.output.lower()

    def test_cli_list_ensembles_with_actual_configs(self) -> None:
        """Test listing ensembles when config files exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test ensemble file
            ensemble = {
                "name": "test_ensemble",
                "description": "A test ensemble for CLI testing",
                "agents": [
                    {"name": "agent1", "role": "tester", "model": "claude-3-sonnet"}
                ],  # noqa: E501
                "coordinator": {"synthesis_prompt": "Test", "output_format": "json"},
            }

            with open(f"{temp_dir}/test_ensemble.yaml", "w") as f:
                yaml.dump(ensemble, f)

            runner = CliRunner()
            result = runner.invoke(cli, ["list-ensembles", "--config-dir", temp_dir])
            assert result.exit_code == 0
            assert "test_ensemble" in result.output
            assert "A test ensemble for CLI testing" in result.output

    def test_cli_invoke_existing_ensemble(self) -> None:
        """Test invoking an ensemble that exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test ensemble file
            ensemble = {
                "name": "working_ensemble",
                "description": "A working test ensemble",
                "agents": [
                    {"name": "agent1", "role": "tester", "model": "claude-3-sonnet"},
                    {"name": "agent2", "role": "reviewer", "model": "claude-3-sonnet"},
                ],
                "coordinator": {
                    "synthesis_prompt": "Combine results",
                    "output_format": "json",
                },  # noqa: E501
            }

            with open(f"{temp_dir}/working_ensemble.yaml", "w") as f:
                yaml.dump(ensemble, f)

            runner = CliRunner()
            result = runner.invoke(
                cli, ["invoke", "--config-dir", temp_dir, "working_ensemble"]
            )  # noqa: E501
            # Should now succeed and show execution results (using JSON output)
            assert "working_ensemble" in result.output
            # Should see some execution output or JSON structure
            assert result.exit_code == 0 or "execution" in result.output.lower()
