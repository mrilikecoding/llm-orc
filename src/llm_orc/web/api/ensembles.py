"""Ensembles API endpoints.

Provides REST API for ensemble management, delegating to OrchestraService.
"""

import asyncio
import contextlib
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from llm_orc.services.handlers.scope import Scope
from llm_orc.web.api import get_orchestra_service

router = APIRouter(prefix="/api/ensembles", tags=["ensembles"])


class ExecuteRequest(BaseModel):
    """Request body for running the ensemble named in the path.

    Unknown keys are a 422: a misspelled ``bind`` must not run the call
    without its binding. A root (``ensemble_name`` or ``ensemble``) is an
    unknown key here, since the path names the root.
    """

    model_config = ConfigDict(extra="forbid")

    input: str
    ensembles: dict[str, dict[str, Any]] | None = None
    profiles: dict[str, dict[str, Any]] | None = None
    scripts: dict[str, str] | None = None
    bind: dict[str, str] | None = None
    pull: bool | None = None


class RunRequest(ExecuteRequest):
    """Request body for ``POST /api/ensembles/execute``: the full request.

    The root is ``ensemble_name`` (an installed ensemble) or ``ensemble``
    (an inline definition); the service refuses a request with both or
    neither.
    """

    ensemble_name: str | None = None
    ensemble: dict[str, Any] | None = None


class CreateEnsembleRequest(BaseModel):
    """Request body for ensemble creation."""

    name: str
    description: str = ""
    agents: list[dict[str, Any]] = Field(default_factory=list)
    from_template: str | None = None
    scope: Scope = "project"


class UpdateEnsembleRequest(BaseModel):
    """Request body for ensemble update."""

    changes: dict[str, Any] = Field(default_factory=dict)
    dry_run: bool = True
    backup: bool = True
    scope: Scope = "project"


# nginx's code for a client that closed the connection before the answer.
CLIENT_CLOSED_REQUEST = 499


async def _wait_for_disconnect(http_request: Request) -> None:
    """Return when the connection reports ``http.disconnect``."""
    while (await http_request.receive())["type"] != "http.disconnect":
        pass


async def _invoke_while_connected(
    http_request: Request, request: dict[str, Any]
) -> dict[str, Any] | None:
    """Run the request, cancelling it if the client goes away first.

    Returns the result, or None after a disconnect: nobody is listening,
    and cancelling the call is what kills the run's scripts and removes
    its layer.
    """
    run = asyncio.create_task(get_orchestra_service().invoke(request))
    gone = asyncio.create_task(_wait_for_disconnect(http_request))
    try:
        await asyncio.wait({run, gone}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        gone.cancel()
        if not run.done():
            run.cancel()
    if run.cancelled() or not run.done():
        with contextlib.suppress(asyncio.CancelledError):
            await run
        return None
    return run.result()


@router.get("")
async def list_ensembles() -> list[dict[str, Any]]:
    """List all available ensembles.

    Returns ensembles from local, library, and global sources.
    """
    service = get_orchestra_service()
    return await service.read_ensembles()


@router.get("/{name}")
async def get_ensemble(name: str) -> dict[str, Any]:
    """Get detailed configuration for a specific ensemble."""
    service = get_orchestra_service()
    result = await service.read_ensemble(name)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Ensemble '{name}' not found")
    return result


@router.post("/execute", response_model=dict[str, Any])
async def run_request(request: RunRequest, http_request: Request) -> Any:
    """Run one request: a named or inline ensemble with its injections.

    Returns the execution result, or the refusal envelope (HTTP 200, like
    every run outcome) when the host cannot run it. A client that
    disconnects first cancels the run.
    """
    result = await _invoke_while_connected(
        http_request, request.model_dump(exclude_none=True)
    )
    return result if result is not None else Response(status_code=CLIENT_CLOSED_REQUEST)


@router.post("/{name}/execute", response_model=dict[str, Any])
async def execute_ensemble(
    name: str, request: ExecuteRequest, http_request: Request
) -> Any:
    """Execute the ensemble ``name`` with the given input and injections.

    Returns the execution result including agent outputs. A client that
    disconnects first cancels the run.
    """
    result = await _invoke_while_connected(
        http_request, {**request.model_dump(exclude_none=True), "ensemble_name": name}
    )
    return result if result is not None else Response(status_code=CLIENT_CLOSED_REQUEST)


@router.post("/{name}/validate")
async def validate_ensemble(name: str) -> dict[str, Any]:
    """Validate an ensemble configuration.

    Returns validation result with any errors found.
    """
    service = get_orchestra_service()
    result = await service.validate_ensemble({"ensemble_name": name})
    return result


@router.get("/{name}/runnable")
async def check_ensemble_runnable(name: str) -> dict[str, Any]:
    """Check if ensemble can run with current providers.

    Returns runnable status including:
    - Whether the ensemble can run
    - Status of each agent's profile/provider
    - Suggested local alternatives for unavailable profiles
    - dependencies: every child ensemble, script, profile and model in the
      closure with status and resolve hint (docs/serving.md, Preflight)
    """
    service = get_orchestra_service()
    result = await service.check_ensemble_runnable({"ensemble_name": name})
    return result


@router.post("")
async def create_ensemble(request: CreateEnsembleRequest) -> dict[str, Any]:
    """Create a new ensemble.

    Args:
        request: Ensemble definition including name, description, agents,
            and scope ("project" default, or "global" to write under
            the XDG global config dir).

    Returns:
        Creation result with path and agents copied.
    """
    service = get_orchestra_service()
    result = await service.create_ensemble(
        {
            "name": request.name,
            "description": request.description,
            "agents": request.agents,
            "from_template": request.from_template,
            "scope": request.scope,
        }
    )
    return result


@router.put("/{name}")
async def update_ensemble(name: str, request: UpdateEnsembleRequest) -> dict[str, Any]:
    """Update an existing ensemble.

    Args:
        name: Name of the ensemble to update.
        request: Update parameters including changes, dry_run, backup,
            and scope (which tier holds the ensemble).

    Returns:
        Update result with preview or applied changes.
    """
    service = get_orchestra_service()
    result = await service.update_ensemble(
        {
            "ensemble_name": name,
            "changes": request.changes,
            "dry_run": request.dry_run,
            "backup": request.backup,
            "scope": request.scope,
        }
    )
    return result


@router.delete("/{name}")
async def delete_ensemble(name: str, scope: Scope = "project") -> dict[str, Any]:
    """Delete an ensemble.

    Args:
        name: Name of the ensemble to delete.
        scope: Tier to delete from, "project" or "global".

    Returns:
        Deletion result.
    """
    service = get_orchestra_service()
    result = await service.delete_ensemble(
        {"ensemble_name": name, "confirm": True, "scope": scope}
    )
    return result
