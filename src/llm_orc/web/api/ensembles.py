"""Ensembles API endpoints.

Provides REST API for ensemble management, delegating to OrchestraService.
"""

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from llm_orc.services.handlers.scope import Scope
from llm_orc.web.api import get_orchestra_service

router = APIRouter(prefix="/api/ensembles", tags=["ensembles"])


class ExecuteRequest(BaseModel):
    """Request body for ensemble execution."""

    input: str


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


@router.post("/{name}/execute")
async def execute_ensemble(name: str, request: ExecuteRequest) -> dict[str, Any]:
    """Execute an ensemble with the given input.

    Returns the execution result including agent outputs.
    """
    service = get_orchestra_service()
    result = await service.invoke({"ensemble_name": name, "input": request.input})
    return result


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
