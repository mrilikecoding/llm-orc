"""Artifacts API endpoints.

Provides REST API for artifact management, delegating to OrchestraService.
"""

from typing import Any

from fastapi import APIRouter, HTTPException

from llm_orc.web.api import get_orchestra_service

router = APIRouter(prefix="/api/artifacts", tags=["artifacts"])


@router.get("")
async def list_artifacts() -> list[dict[str, Any]]:
    """List all execution artifacts grouped by ensemble."""
    service = get_orchestra_service()
    ensembles = service.list_artifact_ensembles()
    return ensembles


@router.get("/{ensemble}")
async def get_ensemble_artifacts(ensemble: str) -> list[dict[str, Any]]:
    """List artifacts for a specific ensemble."""
    service = get_orchestra_service()
    result = await service.read_artifacts(ensemble)
    return result


@router.get("/{ensemble}/{artifact_id}")
async def get_artifact(ensemble: str, artifact_id: str) -> dict[str, Any]:
    """Get a specific artifact's details."""
    service = get_orchestra_service()
    result = await service.read_artifact(ensemble, artifact_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    return result


@router.delete("/{ensemble}/{artifact_id}")
async def delete_artifact(ensemble: str, artifact_id: str) -> dict[str, Any]:
    """Delete a specific artifact."""
    service = get_orchestra_service()
    result = await service.delete_artifact(
        {
            "artifact_id": f"{ensemble}/{artifact_id}",
            "confirm": True,
        }
    )
    return result


@router.post("/{artifact_id}/analyze")
async def analyze_artifact(artifact_id: str) -> dict[str, Any]:
    """Analyze an execution artifact."""
    service = get_orchestra_service()
    result = await service.analyze_execution({"artifact_id": artifact_id})
    return result
