"""Model lifecycle API (#90).

The serve owns the local inference process, so listing what it serves
and pulling a model are serve operations: a remote operator on the
tailnet needs neither ssh nor a second application to manage models.
"""

from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from llm_orc.providers.llama_server import (
    PULL_POLL_S,
    PULL_TIMEOUT_S,
)
from llm_orc.providers.llama_server import (
    router_client as router_client,
)

router = APIRouter(prefix="/api/models", tags=["models"])


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
    """Load one model, downloading its GGUF first if the router has to.

    The wait is ``LlamaServerClient.pull``: it ends when the router
    reports something other than ``loading``, and the answer is that real
    status. It blocks, so it runs in a worker thread; a download held on
    the event loop stalls every other request the serve has (#199).
    """
    client = router_client()
    try:
        result = await run_in_threadpool(
            client.pull, name, timeout_s=PULL_TIMEOUT_S, poll_s=PULL_POLL_S
        )
    except (OSError, ValueError) as e:
        raise _unreachable(e) from e
    return {"name": name, "status": str(result["status"])}
