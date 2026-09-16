"""Model lifecycle API (#90).

The serve owns the local inference process, so listing what it serves
and pulling a model are serve operations: a remote operator on the
tailnet needs neither ssh nor a second application to manage models.
"""

import os
from typing import Any

from fastapi import APIRouter, HTTPException

from llm_orc.providers.llama_server import LlamaServerClient

router = APIRouter(prefix="/api/models", tags=["models"])

DEFAULT_LLAMA_SERVER_URL = "http://127.0.0.1:8080/v1"


def router_client() -> LlamaServerClient:
    """The router this serve talks to (owned or reached by URL)."""
    base_url = os.environ.get("LLAMA_SERVER_URL", DEFAULT_LLAMA_SERVER_URL)
    return LlamaServerClient.from_base_url(base_url)


def _unreachable(exc: Exception) -> HTTPException:
    return HTTPException(
        status_code=503, detail=f"llama-server router unreachable: {exc}"
    )


def _entry(model: dict[str, Any]) -> dict[str, str]:
    status = model.get("status")
    value = status.get("value") if isinstance(status, dict) else status
    return {"name": str(model.get("id", "")), "status": str(value or "unknown")}


@router.get("")
async def list_models() -> dict[str, Any]:
    """Every model the router can serve, with its load status."""
    try:
        models = router_client().models()
    except (OSError, ValueError) as e:
        raise _unreachable(e) from e
    return {"models": [_entry(m) for m in models]}


@router.post("/{name}/pull")
async def pull_model(name: str) -> dict[str, str]:
    """Load one model, downloading its GGUF first if the router has to."""
    try:
        router_client().load(name)
    except (OSError, ValueError) as e:
        raise _unreachable(e) from e
    return {"name": name, "status": "loaded"}
