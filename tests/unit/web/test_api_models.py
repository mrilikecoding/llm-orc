"""Model lifecycle on the serve's own API (#90): list what the router
serves and pull a model, so a remote operator never needs ssh or a
separate inference app."""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


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
        router = MagicMock()
        router.models.side_effect = [
            [{"id": "qwen3-14b", "status": {"value": "loading"}}],
            [{"id": "qwen3-14b", "status": {"value": "loading"}}],
            [{"id": "qwen3-14b", "status": {"value": "loaded"}}],
        ]

        with (
            patch("llm_orc.web.api.models.router_client", return_value=router),
            patch("llm_orc.web.api.models.time.sleep"),
        ):
            response = client.post("/api/models/qwen3-14b/pull")

        assert response.status_code == 200
        assert response.json() == {"name": "qwen3-14b", "status": "loaded"}
        router.load.assert_called_once_with("qwen3-14b")
        assert router.models.call_count == 3

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

        with patch.dict("os.environ", {"LLAMA_SERVER_URL": "http://ng-mini:8080/v1"}):
            assert router_client().root_url == "http://ng-mini:8080"
