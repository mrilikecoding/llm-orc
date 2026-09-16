"""``llm-orc serve`` owns the llama-server router (#90)."""

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from llm_orc.cli import cli


class TestServeOwnsTheRouter:
    def test_serve_starts_router_exports_url_and_stops_after_uvicorn(self) -> None:
        supervisor = MagicMock()
        supervisor.base_url = "http://127.0.0.1:8080/v1"
        seen_env: dict[str, str | None] = {}

        def fake_uvicorn_run(*args: object, **kwargs: object) -> None:
            import os

            seen_env["url"] = os.environ.get("LLAMA_SERVER_URL")

        with (
            patch(
                "llm_orc.providers.llama_server.start_router_from_config",
                return_value=supervisor,
            ) as start,
            patch("uvicorn.run", side_effect=fake_uvicorn_run),
            patch.dict("os.environ", {}, clear=False),
        ):
            import os

            os.environ.pop("LLAMA_SERVER_URL", None)
            result = CliRunner().invoke(cli, ["serve", "--port", "8765"])

        assert result.exit_code == 0, result.output
        start.assert_called_once()
        assert seen_env["url"] == "http://127.0.0.1:8080/v1"
        supervisor.stop.assert_called_once()

    def test_no_backend_skips_the_router(self) -> None:
        with (
            patch("llm_orc.providers.llama_server.start_router_from_config") as start,
            patch("uvicorn.run"),
        ):
            result = CliRunner().invoke(cli, ["serve", "--no-backend"])

        assert result.exit_code == 0, result.output
        start.assert_not_called()

    def test_router_failure_is_reported_not_swallowed(self) -> None:
        with (
            patch(
                "llm_orc.providers.llama_server.start_router_from_config",
                side_effect=RuntimeError("llama-server exited with code 3"),
            ),
            patch("uvicorn.run") as run,
        ):
            result = CliRunner().invoke(cli, ["serve"])

        assert result.exit_code != 0
        assert "llama-server exited with code 3" in result.output
        run.assert_not_called()
