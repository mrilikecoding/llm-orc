"""Tests for the web CLI command."""

from click.testing import CliRunner

from llm_orc.cli import cli


class TestWebCLI:
    """Tests for web CLI command."""

    def test_web_command_exists(self) -> None:
        """Test that web command is registered."""
        runner = CliRunner()
        result = runner.invoke(cli, ["web", "--help"])

        assert result.exit_code == 0
        assert "Start the web UI server" in result.output

    def test_web_command_has_port_option(self) -> None:
        """Test that web command has --port option."""
        runner = CliRunner()
        result = runner.invoke(cli, ["web", "--help"])

        assert "--port" in result.output

    def test_web_command_has_host_option(self) -> None:
        """Test that web command has --host option."""
        runner = CliRunner()
        result = runner.invoke(cli, ["web", "--help"])

        assert "--host" in result.output

    def test_web_command_has_open_option(self) -> None:
        """Test that web command has --open option."""
        runner = CliRunner()
        result = runner.invoke(cli, ["web", "--help"])

        assert "--open" in result.output
