"""Serving Layer ``POST /v1/embeddings`` endpoint (#198).

Forwards an OpenAI-compatible embeddings request to the llama-server
router this serve owns
(``docs/plans/2026-09-16-embeddings-on-the-serve.md``). ``model`` may
name a llama-server profile (resolved to its ``model`` field) or a
router model name directly; either way the forwarded body carries the
router name. A model the running router does not list answers 404
without forwarding: the router only knows what its own preset loaded it
with, so its ``/models`` (cached briefly) is the source of truth, not
the profile files, which can change after it started (#199).
"""

import time
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.providers.llama_server import LLAMA_SERVER_PROVIDER
from llm_orc.web.api.models import router_client

router = APIRouter(prefix="/v1", tags=["openai-compat"])

#: Generous: embedding batches of 64 on CPU (docs/plans/2026-09-16-
#: embeddings-on-the-serve.md, "Alignment").
EMBEDDINGS_TIMEOUT_S = 600.0


#: How long the router's model list is reused (#199). Short enough that
#: a model loaded or dropped by a router restart is seen within seconds,
#: long enough that a batch of embedding calls asks the router once.
MODEL_LIST_TTL_S = 5.0

_model_list: tuple[float, frozenset[str]] | None = None


def _reset_model_list_cache() -> None:
    global _model_list
    _model_list = None


def get_config_manager() -> ConfigurationManager:
    """The project's Configuration Manager for the current request.

    Per-request construction picks up profile changes without a process
    restart, matching ``get_model_profile_allowlist`` in ``v1_models``.
    Tests monkeypatch this to inject an isolated manager.
    """
    return ConfigurationManager()


class _EmbeddingsRequest(BaseModel):
    """The OpenAI ``/v1/embeddings`` request subset this route accepts."""

    model: str
    input: str | list[str]
    encoding_format: str | None = None
    dimensions: int | None = None


def _unreachable(exc: Exception) -> HTTPException:
    return HTTPException(
        status_code=503, detail=f"llama-server router unreachable: {exc}"
    )


def _model_not_found(name: str) -> HTTPException:
    return HTTPException(
        status_code=404,
        detail={
            "message": f"model '{name}' not found",
            "type": "invalid_request_error",
            "code": "model_not_found",
        },
    )


def _resolve_router_model(config: ConfigurationManager, requested: str) -> str:
    """A llama-server profile id resolves to its ``model``; anything else
    (including another provider's profile id) is taken as a router model
    name, as ``render_preset`` only counts llama-server profiles."""
    profile = config.get_model_profile(requested) or {}
    if profile.get("provider") != LLAMA_SERVER_PROVIDER:
        return requested
    served = profile.get("model")
    return str(served) if served else requested


def _router_model_names() -> frozenset[str]:
    """The model names the running router lists, reused for
    ``MODEL_LIST_TTL_S``. An unreachable router is a 503, never a
    fallback to the profile files."""
    global _model_list
    now = time.monotonic()
    if _model_list is not None and now - _model_list[0] < MODEL_LIST_TTL_S:
        return _model_list[1]
    try:
        models = router_client().models()
    except (OSError, ValueError) as e:
        raise _unreachable(e) from e
    names = frozenset(
        m["id"] for m in models if isinstance(m.get("id"), str) and m["id"]
    )
    _model_list = (now, names)
    return names


@router.post("/embeddings")
def create_embeddings(payload: _EmbeddingsRequest) -> JSONResponse:
    """Forward to the router's ``/v1/embeddings``, rewriting ``model``.

    404s without forwarding when the resolved name is not in the router's
    own model list (``_router_model_names``); a router connection failure
    (listing or forwarding) is a
    503, matching ``/api/models/{name}/pull``. Otherwise the router's
    status and JSON body are returned as-is (no streaming).

    Plain ``def``, not ``async def``: every line here is synchronous
    (config load, the urllib router call up to
    ``EMBEDDINGS_TIMEOUT_S``), so an ``async def`` bought nothing while
    blocking the event loop -- and with it every other request the serve
    handles (health, /mcp, REST execute, chat) -- for the life of the
    router call. FastAPI runs a ``def`` path operation in its shared
    threadpool automatically, which is the standard fix for a
    sync-only handler.
    """
    config = get_config_manager()
    resolved = _resolve_router_model(config, payload.model)
    if resolved not in _router_model_names():
        raise _model_not_found(payload.model)

    body: dict[str, Any] = payload.model_dump(exclude_none=True)
    body["model"] = resolved

    try:
        status, response_body = router_client().embeddings(
            body, timeout=EMBEDDINGS_TIMEOUT_S
        )
    except (OSError, ValueError) as e:
        raise _unreachable(e) from e

    return JSONResponse(status_code=status, content=response_body)
