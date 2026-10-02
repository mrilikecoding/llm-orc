"""Model lifecycle on the serve's own API (#90): list what the router
serves and pull a model, so a remote operator never needs ssh or a
separate inference app."""

import threading
import time
from typing import Any
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from llm_orc.providers.llama_server import LlamaServerClient
from llm_orc.web.server import create_app


class TestModelsApi:
    def test_list_reports_router_models_with_status(self, client: TestClient) -> None:
        router = MagicMock()
        router.models.return_value = [
            {"id": "qwen3-8b", "status": {"value": "loaded"}},
            {"id": "qwen3-14b", "status": {"value": "unloaded"}},
        ]

        with patch("llm_orc.web.api.models.router_client", return_value=router):
            response = client.get("/api/models")

        assert response.status_code == 200
        assert response.json() == {
            "models": [
                {"name": "qwen3-8b", "status": "loaded"},
                {"name": "qwen3-14b", "status": "unloaded"},
            ]
        }

    def test_pull_waits_for_the_load_and_reports_the_real_status(
        self, client: TestClient
    ) -> None:
        """The router's load call returns as soon as loading starts (e2e
        2026-09-16: 'loaded' reported while the status was 'loading'); pull
        answers only once the router says otherwise."""
        listings = [
            [{"id": "qwen3-14b", "status": {"value": "loading"}}],
            [{"id": "qwen3-14b", "status": {"value": "loading"}}],
            [{"id": "qwen3-14b", "status": {"value": "loaded"}}],
        ]

        with (
            patch.object(LlamaServerClient, "load") as load,
            patch.object(LlamaServerClient, "_list", side_effect=listings) as listing,
            patch("llm_orc.web.api.models.PULL_POLL_S", 0),
        ):
            response = client.post("/api/models/qwen3-14b/pull")

        assert response.status_code == 200
        assert response.json() == {"name": "qwen3-14b", "status": "loaded"}
        load.assert_called_once_with("qwen3-14b")
        assert listing.call_count == 3

    def test_a_failed_load_is_reported_unloaded_not_loaded(
        self, client: TestClient
    ) -> None:
        """Probe 2026-10-01: a load of a source that does not exist is
        accepted, then the entry reads unloaded with failed true."""
        failed = [
            {
                "id": "bogus",
                "status": {"value": "unloaded", "exit_code": 1, "failed": True},
            }
        ]

        with (
            patch.object(LlamaServerClient, "load"),
            patch.object(LlamaServerClient, "_list", return_value=failed),
        ):
            response = client.post("/api/models/bogus/pull")

        assert response.status_code == 200
        assert response.json() == {"name": "bogus", "status": "unloaded"}

    def test_health_answers_while_a_pull_is_in_flight(self) -> None:
        """The pull used to poll with time.sleep inside an ``async def``,
        holding the event loop (health, /mcp, REST execute and chat all
        share it) for the whole download (#199). One shared loop needs
        the TestClient as a context manager."""
        pull_seconds = 1.0
        loading = threading.Event()

        def slow_load(self: LlamaServerClient, model: str) -> None:
            loading.set()
            time.sleep(pull_seconds)

        listing = [{"id": "big", "status": {"value": "loaded"}}]
        with (
            patch.object(LlamaServerClient, "load", slow_load),
            patch.object(LlamaServerClient, "_list", return_value=listing),
            TestClient(create_app()) as client,
        ):
            pulled: dict[str, Any] = {}
            puller = threading.Thread(
                target=lambda: pulled.update(
                    status=client.post("/api/models/big/pull").status_code
                )
            )
            puller.start()
            assert loading.wait(timeout=5)
            started = time.monotonic()
            health = client.get("/health")
            health_seconds = time.monotonic() - started
            puller.join()

        assert health.status_code == 200
        assert health_seconds < pull_seconds / 2
        assert pulled["status"] == 200

    def test_unreachable_router_is_a_503_not_a_500(self, client: TestClient) -> None:
        router = MagicMock()
        router.models.side_effect = OSError("connection refused")

        with patch("llm_orc.web.api.models.router_client", return_value=router):
            response = client.get("/api/models")

        assert response.status_code == 503
        assert "llama-server" in response.json()["error"]

    def test_router_client_targets_the_configured_router(
        self, monkeypatch: object
    ) -> None:
        from llm_orc.web.api.models import router_client

        with patch.dict(
            "os.environ", {"LLAMA_SERVER_URL": "http://remote-host:8080/v1"}
        ):
            assert router_client().root_url == "http://remote-host:8080"
