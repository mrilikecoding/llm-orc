"""Tests for the Serving Layer ``POST /v1/embeddings`` endpoint (#198).

Forwards an OpenAI-compatible embeddings request to the llama-server
router this serve owns
(``docs/plans/2026-09-16-embeddings-on-the-serve.md``). ``model`` may
be a llama-server profile id (resolved to its ``model`` field) or a
router model name directly. Stubs the router the same way
``test_api_models.py`` does: a ``MagicMock`` in place of the real
``LlamaServerClient``, never the real binary.
"""

import http.client
import inspect
import io
import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from llm_orc.providers.llama_server import _DIRECT
from llm_orc.web.api.v1_embeddings import (
    MODEL_LIST_TTL_S,
    _reset_model_list_cache,
    create_embeddings,
)

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


def _router(*names: str) -> MagicMock:
    """A stub router whose ``/models`` lists exactly ``names``."""
    router = MagicMock()
    router.models.return_value = [{"id": name} for name in names]
    return router


def _wire_listing(body: Any) -> Any:
    """Patch the router's HTTP opener so the real ``LlamaServerClient``
    reads ``body`` as its ``GET /models`` answer."""
    return patch.object(
        _DIRECT,
        "open",
        side_effect=lambda *a, **k: io.BytesIO(json.dumps(body).encode()),
    )


@pytest.fixture(autouse=True)
def _fresh_model_list_cache() -> None:
    _reset_model_list_cache()


class TestEmbeddingsApi:
    def test_forwards_body_and_returns_the_routers_response_verbatim(
        self, client: TestClient
    ) -> None:
        router = _router("nomic-embed-text")
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
        router = _router("nomic-embed-text")
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
        router = _router("nomic-embed-text")
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
        router = _router("nomic-embed-text")
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

    def test_unknown_model_is_a_404_without_forwarding(
        self, client: TestClient
    ) -> None:
        router = _router("nomic-embed-text")

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
        """A profile id resolves through ``model``, but the router only
        lists llama-server models -- a hosted-provider profile's ``model``
        is never in its list, so it 404s too."""
        router = _router("nomic-embed-text")
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
        router = _router("nomic-embed-text")
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
        router = _router("nomic-embed-text")
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

    def test_a_model_the_router_reports_is_accepted_without_a_profile(
        self, client: TestClient
    ) -> None:
        """Profiles edited after the router started: the router, not the
        files on disk, says what it can serve (#199)."""
        router = _router("nomic-embed-text")
        router.embeddings.return_value = (200, {"data": []})

        with (
            patch(
                "llm_orc.web.api.v1_embeddings.get_config_manager",
                return_value=_config({}),
            ),
            patch("llm_orc.web.api.v1_embeddings.router_client", return_value=router),
        ):
            response = client.post(
                "/v1/embeddings",
                json={"model": "nomic-embed-text", "input": "hi"},
            )

        assert response.status_code == 200
        router.embeddings.assert_called_once()

    def test_a_profile_only_model_the_router_never_loaded_is_a_404(
        self, client: TestClient
    ) -> None:
        router = _router("some-other-model")

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

        assert response.status_code == 404
        assert response.json()["error"]["code"] == "model_not_found"
        router.embeddings.assert_not_called()

    def test_conflicting_profiles_on_disk_do_not_break_a_router_model(
        self, client: TestClient
    ) -> None:
        router = _router("nomic-embed-text")
        router.embeddings.return_value = (200, {"data": []})
        conflicting = {
            "a": {"model": "dup", "provider": "llama-server", "hf_repo": "x/a:Q4"},
            "b": {"model": "dup", "provider": "llama-server", "hf_repo": "x/b:Q4"},
        }

        with (
            patch(
                "llm_orc.web.api.v1_embeddings.get_config_manager",
                return_value=_config(conflicting),
            ),
            patch("llm_orc.web.api.v1_embeddings.router_client", return_value=router),
        ):
            response = client.post(
                "/v1/embeddings",
                json={"model": "nomic-embed-text", "input": "hi"},
            )

        assert response.status_code == 200

    def test_router_model_list_is_cached_within_the_ttl(
        self, client: TestClient
    ) -> None:
        router = _router("nomic-embed-text")
        router.embeddings.return_value = (200, {"data": []})

        with (
            patch(
                "llm_orc.web.api.v1_embeddings.get_config_manager",
                return_value=_config({}),
            ),
            patch("llm_orc.web.api.v1_embeddings.router_client", return_value=router),
        ):
            for _ in range(2):
                response = client.post(
                    "/v1/embeddings",
                    json={"model": "nomic-embed-text", "input": "hi"},
                )
                assert response.status_code == 200

        router.models.assert_called_once()

    def test_router_model_list_is_asked_again_after_the_ttl(
        self, client: TestClient
    ) -> None:
        router = _router("nomic-embed-text")
        router.embeddings.return_value = (200, {"data": []})
        clock = [1000.0]

        with (
            patch(
                "llm_orc.web.api.v1_embeddings.get_config_manager",
                return_value=_config({}),
            ),
            patch("llm_orc.web.api.v1_embeddings.router_client", return_value=router),
            patch(
                "llm_orc.web.api.v1_embeddings.time.monotonic",
                side_effect=lambda: clock[0],
            ),
        ):
            client.post(
                "/v1/embeddings", json={"model": "nomic-embed-text", "input": "a"}
            )
            clock[0] += MODEL_LIST_TTL_S + 1
            client.post(
                "/v1/embeddings", json={"model": "nomic-embed-text", "input": "b"}
            )

        assert router.models.call_count == 2

    def test_router_unreachable_while_listing_is_a_503_and_never_forwards(
        self, client: TestClient
    ) -> None:
        router = MagicMock()
        router.models.side_effect = OSError("connection refused")

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
        router.embeddings.assert_not_called()

    def test_an_empty_model_is_a_404_even_when_the_router_lists_an_idless_entry(
        self, client: TestClient
    ) -> None:
        listing = {"data": [{"status": "loaded"}, {"id": "nomic-embed-text"}]}

        with (
            patch(
                "llm_orc.web.api.v1_embeddings.get_config_manager",
                return_value=_config({}),
            ),
            _wire_listing(listing) as opened,
        ):
            response = client.post("/v1/embeddings", json={"model": "", "input": "x"})

        assert response.status_code == 404
        assert response.json()["error"]["code"] == "model_not_found"
        opened.assert_called_once()  # the listing; nothing was forwarded

    def test_an_empty_model_is_never_forwarded_whatever_the_router_listed(
        self, client: TestClient
    ) -> None:
        router = MagicMock()
        router.models.return_value = [{"status": "loaded"}]
        router.embeddings.return_value = (200, {"data": []})

        with (
            patch(
                "llm_orc.web.api.v1_embeddings.get_config_manager",
                return_value=_config({}),
            ),
            patch("llm_orc.web.api.v1_embeddings.router_client", return_value=router),
        ):
            response = client.post("/v1/embeddings", json={"model": "", "input": "x"})

        assert response.status_code == 404
        router.embeddings.assert_not_called()

    @pytest.mark.parametrize(
        "opener",
        [
            patch.object(_DIRECT, "open", side_effect=http.client.IncompleteRead(b"{")),
            _wire_listing([]),
            _wire_listing({"data": None}),
        ],
        ids=["incomplete-read", "non-dict-body", "data-null"],
    )
    def test_a_broken_router_listing_is_a_503_not_a_500(
        self, client: TestClient, opener: Any
    ) -> None:
        with (
            patch(
                "llm_orc.web.api.v1_embeddings.get_config_manager",
                return_value=_config(),
            ),
            opener,
        ):
            response = client.post(
                "/v1/embeddings",
                json={"model": "nomic-embed-text", "input": "hi"},
            )

        assert response.status_code == 503
        assert "llama-server" in response.json()["error"]

    def test_a_hosted_profile_whose_model_matches_a_router_model_is_a_404(
        self, client: TestClient
    ) -> None:
        """Only llama-server profiles are aliases for router models; a
        hosted profile that happens to share a model name must not
        resolve to the router's."""
        router = _router("nomic-embed-text")
        profiles = {
            "premium-embed": {
                "model": "nomic-embed-text",
                "provider": "openai-compatible",
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
                json={"model": "premium-embed", "input": "hi"},
            )

        assert response.status_code == 404
        router.embeddings.assert_not_called()

    def test_a_value_error_from_the_router_listing_is_a_503(
        self, client: TestClient
    ) -> None:
        router = MagicMock()
        router.models.side_effect = ValueError("bad listing")

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
        router.embeddings.assert_not_called()

    def test_handler_is_a_plain_function_so_fastapi_threadpools_it(self) -> None:
        """The handler's own body is entirely synchronous (config load,
        the router listing and call), so it must be a plain
        ``def`` -- FastAPI runs a ``def`` path operation in its shared
        threadpool, which is what keeps a slow (up to 600s) router call
        from blocking the event loop the rest of the serve depends on
        (health, /mcp, REST execute, chat all share it). An ``async def``
        here would buy nothing while blocking everything (reviewer
        measurement: /health took 6.96s issued 1s into an embeddings call
        against a router that sleeps 8s, vs. a 1.3ms baseline)."""
        assert not inspect.iscoroutinefunction(create_embeddings)
