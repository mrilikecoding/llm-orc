"""Artifacts API endpoints.

Provides REST API for artifact management, delegating to MCPServerV2.
"""

from typing import Any

from fastapi import APIRouter, HTTPException

from llm_orc.mcp import MCPServerV2

router = APIRouter(prefix="/api/artifacts", tags=["artifacts"])

_mcp_server: MCPServerV2 | None = None


def get_mcp_server() -> MCPServerV2:
    """Get or create the MCP server instance."""
    global _mcp_server
    if _mcp_server is None:
        _mcp_server = MCPServerV2()
    return _mcp_server


@router.get("")
async def list_artifacts() -> list[dict[str, Any]]:
    """List all execution artifacts grouped by ensemble."""
    mcp = get_mcp_server()
    # Get all ensembles with artifacts
    ensembles = mcp.artifact_manager.list_ensembles()
    return ensembles


@router.get("/{ensemble}")
async def get_ensemble_artifacts(ensemble: str) -> list[dict[str, Any]]:
    """List artifacts for a specific ensemble."""
    mcp = get_mcp_server()
    result = await mcp._read_artifacts_resource(ensemble)
    return result


@router.get("/{ensemble}/{artifact_id}")
async def get_artifact(ensemble: str, artifact_id: str) -> dict[str, Any]:
    """Get a specific artifact's details."""
    mcp = get_mcp_server()
    result = await mcp._read_artifact_resource(ensemble, artifact_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    return result


@router.delete("/{ensemble}/{artifact_id}")
async def delete_artifact(ensemble: str, artifact_id: str) -> dict[str, Any]:
    """Delete a specific artifact."""
    mcp = get_mcp_server()
    result = await mcp._delete_artifact_tool(
        {
            "artifact_id": f"{ensemble}/{artifact_id}",
            "confirm": True,
        }
    )
    return result


@router.post("/{artifact_id}/analyze")
async def analyze_artifact(artifact_id: str) -> dict[str, Any]:
    """Analyze an execution artifact."""
    mcp = get_mcp_server()
    result = await mcp._analyze_execution_tool({"artifact_id": artifact_id})
    return result
