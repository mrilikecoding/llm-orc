"""Tests for the web CLI command."""

from unittest.mock import patch

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


class TestServeCLI:
    """``llm-orc serve`` shares the ``web`` server start path.

    The commands expose the same FastAPI app with distinct CLI framing
    for the browser-UI versus agentic-serving audiences. Verify the
    command exists, has the expected options, and that invoking it
    actually starts the app via the shared helper.
    """

    def test_serve_command_exists(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["serve", "--help"])

        assert result.exit_code == 0
        assert "agentic serving layer" in result.output.lower()

    def test_serve_command_has_port_and_host_options(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["serve", "--help"])

        assert "--port" in result.output
        assert "--host" in result.output

    def test_serve_command_does_not_expose_open_browser(self) -> None:
        """Agentic serving does not open a browser — the flag is UI-specific."""
        runner = CliRunner()
        result = runner.invoke(cli, ["serve", "--help"])

        assert "--open" not in result.output

    def test_serve_and_web_start_the_same_fastapi_app(self) -> None:
        """Both commands must call the same app factory via uvicorn.run."""
        runner = CliRunner()
        with patch("uvicorn.run") as mock_uvicorn:
            runner.invoke(cli, ["serve", "--port", "0"])
            serve_app = mock_uvicorn.call_args.args[0]

            mock_uvicorn.reset_mock()

            runner.invoke(cli, ["web", "--port", "0"])
            web_app = mock_uvicorn.call_args.args[0]

        # The two apps must share the same router surface — same
        # number of routes with the same paths. Exact-identity isn't
        # guaranteed because ``create_app`` constructs a fresh
        # FastAPI instance each call.
        assert {r.path for r in serve_app.routes} == {r.path for r in web_app.routes}

    def test_serve_command_labels_output_as_agentic_serving(self) -> None:
        runner = CliRunner()
        with patch("uvicorn.run"):
            result = runner.invoke(cli, ["serve", "--port", "0"])

        assert result.exit_code == 0
        assert "agentic serving layer" in result.stderr
