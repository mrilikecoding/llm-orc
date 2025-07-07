"""Tests for CLI interface."""

from click.testing import CliRunner

from llm_orc.cli import cli


class TestCLI:
    """Test CLI interface."""

    def test_cli_help(self):
        """Test that CLI shows help message."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "llm orchestra" in result.output.lower()

    def test_cli_invoke_command_exists(self):
        """Test that invoke command exists."""
        runner = CliRunner()
        result = runner.invoke(cli, ["invoke", "--help"])
        assert result.exit_code == 0
        assert "invoke" in result.output.lower()

    def test_cli_invoke_requires_ensemble_name(self):
        """Test that invoke command requires ensemble name."""
        runner = CliRunner()
        result = runner.invoke(cli, ["invoke"])
        assert result.exit_code != 0
        assert (
            "ensemble" in result.output.lower() or "required" in result.output.lower()
        )

    def test_cli_invoke_with_ensemble_name(self):
        """Test basic ensemble invocation."""
        runner = CliRunner()
        result = runner.invoke(cli, ["invoke", "test_ensemble"])
        # Should fail because ensemble doesn't exist, but with proper error message
        assert result.exit_code != 0
        assert "test_ensemble" in result.output

    def test_cli_invoke_with_config_option(self):
        """Test invoke command accepts config directory option."""
        runner = CliRunner()
        result = runner.invoke(cli, ["invoke", "--config-dir", "/tmp", "test_ensemble"])
        assert result.exit_code != 0
        # Should show that it's looking in the specified config directory
        assert "test_ensemble" in result.output

    def test_cli_list_command_exists(self):
        """Test that list-ensembles command exists."""
        runner = CliRunner()
        result = runner.invoke(cli, ["list-ensembles", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output.lower() or "ensemble" in result.output.lower()
