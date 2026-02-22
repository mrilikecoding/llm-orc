"""Ensembles API endpoints.

Provides REST API for ensemble management, delegating to OrchestraService.
"""

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from llm_orc.web.api import get_orchestra_service

router = APIRouter(prefix="/api/ensembles", tags=["ensembles"])


class ExecuteRequest(BaseModel):
    """Request body for ensemble execution."""

    input: str


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
    """
    service = get_orchestra_service()
    result = await service.check_ensemble_runnable({"ensemble_name": name})
    return result
