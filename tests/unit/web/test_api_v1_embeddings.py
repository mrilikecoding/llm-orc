"""Tests for the Serving Layer ``POST /v1/embeddings`` endpoint (#198).

Forwards an OpenAI-compatible embeddings request to the llama-server
router this serve owns
(``docs/plans/2026-09-16-embeddings-on-the-serve.md``). ``model`` may
be a llama-server profile id (resolved to its ``model`` field) or a
router model name directly. Stubs the router the same way
``test_api_models.py`` does: a ``MagicMock`` in place of the real
``LlamaServerClient``, never the real binary.
"""

from typing import Any
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

_PROFILES: dict[str, dict[str, Any]] = {
    "local-nomic-embed-text": {
        "model": "nomic-embed-text",
        "provider": "llama-server",
        "hf_repo": "nomic-ai/nomic-embed-text-v1.5-GGUF:Q8_0",
        "options": {"embeddings": True, "pooling": "mean"},
    },
}


def _config(profiles: dict[str, dict[str, Any]] | None = None) -> MagicMock:
    library = _PROFILES if profiles is None else profiles
    config = MagicMock()
    config.get_model_profiles.return_value = library
    config.get_model_profile.side_effect = lambda name: library.get(name)
    return config


class TestEmbeddingsApi:
    def test_forwards_body_and_returns_the_routers_response_verbatim(
        self, client: TestClient
    ) -> None:
        router = MagicMock()
        router.embeddings.return_value = (
            200,
            {"object": "list", "data": [{"embedding": [0.1, 0.2]}]},
        )

        with (
            patch(
                "llm_orc.web.api.v1_embeddings.get_config_manager",
                return_value=_config(),
            ),
            patch("llm_orc.web.api.v1_embeddings.router_client", return_value=router),
        ):
            response = client.post(
                "/v1/embeddings",
                json={"model": "nomic-embed-text", "input": "hello world"},
            )

        assert response.status_code == 200
        assert response.json() == {
            "object": "list",
            "data": [{"embedding": [0.1, 0.2]}],
        }
        router.embeddings.assert_called_once_with(
            {"model": "nomic-embed-text", "input": "hello world"}, timeout=600.0
        )

    def test_list_input_arrives_at_the_stub_verbatim(self, client: TestClient) -> None:
        router = MagicMock()
        router.embeddings.return_value = (200, {"data": []})

        with (
            patch(
                "llm_orc.web.api.v1_embeddings.get_config_manager",
                return_value=_config(),
            ),
            patch("llm_orc.web.api.v1_embeddings.router_client", return_value=router),
        ):
            response = client.post(
                "/v1/embeddings",
                json={"model": "nomic-embed-text", "input": ["a", "b"]},
            )

        assert response.status_code == 200
        router.embeddings.assert_called_once_with(
            {"model": "nomic-embed-text", "input": ["a", "b"]}, timeout=600.0
        )

    def test_optional_fields_forward_when_present(self, client: TestClient) -> None:
        router = MagicMock()
        router.embeddings.return_value = (200, {"data": []})

        with (
            patch(
                "llm_orc.web.api.v1_embeddings.get_config_manager",
                return_value=_config(),
            ),
            patch("llm_orc.web.api.v1_embeddings.router_client", return_value=router),
        ):
            client.post(
                "/v1/embeddings",
                json={
                    "model": "nomic-embed-text",
                    "input": "hi",
                    "encoding_format": "float",
                    "dimensions": 256,
                },
            )

        router.embeddings.assert_called_once_with(
            {
                "model": "nomic-embed-text",
                "input": "hi",
                "encoding_format": "float",
                "dimensions": 256,
            },
            timeout=600.0,
        )

    def test_profile_id_resolves_to_the_router_model_name(
        self, client: TestClient
    ) -> None:
        router = MagicMock()
        router.embeddings.return_value = (200, {"data": []})

        with (
            patch(
                "llm_orc.web.api.v1_embeddings.get_config_manager",
                return_value=_config(),
            ),
            patch("llm_orc.web.api.v1_embeddings.router_client", return_value=router),
        ):
            response = client.post(
                "/v1/embeddings",
                json={"model": "local-nomic-embed-text", "input": "hi"},
            )

        assert response.status_code == 200
        router.embeddings.assert_called_once_with(
            {"model": "nomic-embed-text", "input": "hi"}, timeout=600.0
        )

    def test_unknown_model_is_a_404_without_contacting_the_router(
        self, client: TestClient
    ) -> None:
        router = MagicMock()

        with (
            patch(
                "llm_orc.web.api.v1_embeddings.get_config_manager",
                return_value=_config(),
            ),
            patch("llm_orc.web.api.v1_embeddings.router_client", return_value=router),
        ):
            response = client.post(
                "/v1/embeddings",
                json={"model": "nonexistent-model", "input": "hi"},
            )

        assert response.status_code == 404
        assert response.json() == {
            "error": {
                "message": "model 'nonexistent-model' not found",
                "type": "invalid_request_error",
                "code": "model_not_found",
            }
        }
        router.embeddings.assert_not_called()

    def test_a_profile_id_for_a_non_llama_server_provider_is_a_404(
        self, client: TestClient
    ) -> None:
        """A profile id resolves through ``model``, but only a llama-server
        model name lands in the rendered preset -- a hosted-provider
        profile's ``model`` will never be a member, so it 404s too."""
        router = MagicMock()
        profiles = {
            **_PROFILES,
            "premium-claude": {
                "model": "claude-sonnet-4",
                "provider": "anthropic-api",
            },
        }

        with (
            patch(
                "llm_orc.web.api.v1_embeddings.get_config_manager",
                return_value=_config(profiles),
            ),
            patch("llm_orc.web.api.v1_embeddings.router_client", return_value=router),
        ):
            response = client.post(
                "/v1/embeddings",
                json={"model": "premium-claude", "input": "hi"},
            )

        assert response.status_code == 404
        router.embeddings.assert_not_called()

    def test_unreachable_router_is_a_503_not_a_500(self, client: TestClient) -> None:
        router = MagicMock()
        router.embeddings.side_effect = OSError("connection refused")

        with (
            patch(
                "llm_orc.web.api.v1_embeddings.get_config_manager",
                return_value=_config(),
            ),
            patch("llm_orc.web.api.v1_embeddings.router_client", return_value=router),
        ):
            response = client.post(
                "/v1/embeddings",
                json={"model": "nomic-embed-text", "input": "hi"},
            )

        assert response.status_code == 503
        assert "llama-server" in response.json()["error"]

    def test_the_routers_own_error_status_and_body_pass_through(
        self, client: TestClient
    ) -> None:
        router = MagicMock()
        router.embeddings.return_value = (400, {"error": "bad batch"})

        with (
            patch(
                "llm_orc.web.api.v1_embeddings.get_config_manager",
                return_value=_config(),
            ),
            patch("llm_orc.web.api.v1_embeddings.router_client", return_value=router),
        ):
            response = client.post(
                "/v1/embeddings",
                json={"model": "nomic-embed-text", "input": "hi"},
            )

        assert response.status_code == 400
        assert response.json() == {"error": "bad batch"}
