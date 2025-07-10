"""Test suite for CLI serve command."""

from unittest.mock import Mock, patch

from click.testing import CliRunner

from llm_orc.cli import serve


class TestCLIServe:
    """Test CLI serve command functionality."""

    def test_serve_command_exists(self) -> None:
        """Should have serve command available."""
        runner = CliRunner()
        result = runner.invoke(serve, ["--help"])
        assert result.exit_code == 0
        assert "serve" in result.output.lower()

    @patch("llm_orc.cli.MCPServerRunner")
    def test_serve_command_with_ensemble_name(self, mock_runner_class: Mock) -> None:
        """Should start MCP server with specified ensemble."""
        mock_runner = Mock()
        mock_runner_class.return_value = mock_runner

        runner = CliRunner()
        result = runner.invoke(serve, ["architecture_review"])

        assert result.exit_code == 0
        mock_runner_class.assert_called_once_with("architecture_review", 3000)
        mock_runner.run.assert_called_once()

    @patch("llm_orc.cli.MCPServerRunner")
    def test_serve_command_with_custom_port(self, mock_runner_class: Mock) -> None:
        """Should start MCP server with custom port."""
        mock_runner = Mock()
        mock_runner_class.return_value = mock_runner

        runner = CliRunner()
        result = runner.invoke(serve, ["gesture_analysis", "--port", "8080"])

        assert result.exit_code == 0
        mock_runner_class.assert_called_once_with("gesture_analysis", 8080)
        mock_runner.run.assert_called_once()

    @patch("llm_orc.cli.MCPServerRunner")
    def test_serve_command_default_port(self, mock_runner_class: Mock) -> None:
        """Should use default port 3000 when not specified."""
        mock_runner = Mock()
        mock_runner_class.return_value = mock_runner

        runner = CliRunner()
        result = runner.invoke(serve, ["research_validation"])

        assert result.exit_code == 0
        mock_runner_class.assert_called_once_with("research_validation", 3000)
