"""Ensembles API endpoints.

Provides REST API for ensemble management, delegating to MCPServer.
"""

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from llm_orc.mcp import MCPServer

router = APIRouter(prefix="/api/ensembles", tags=["ensembles"])

# Singleton MCP server instance
_mcp_server: MCPServer | None = None


def get_mcp_server() -> MCPServer:
    """Get or create the MCP server instance."""
    global _mcp_server
    if _mcp_server is None:
        _mcp_server = MCPServer()
    return _mcp_server


class ExecuteRequest(BaseModel):
    """Request body for ensemble execution."""

    input: str


@router.get("")
async def list_ensembles() -> list[dict[str, Any]]:
    """List all available ensembles.

    Returns ensembles from local, library, and global sources.
    """
    mcp = get_mcp_server()
    return await mcp._read_ensembles_resource()


@router.get("/{name}")
async def get_ensemble(name: str) -> dict[str, Any]:
    """Get detailed configuration for a specific ensemble."""
    mcp = get_mcp_server()
    result = await mcp._read_ensemble_resource(name)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Ensemble '{name}' not found")
    return result


@router.post("/{name}/execute")
async def execute_ensemble(name: str, request: ExecuteRequest) -> dict[str, Any]:
    """Execute an ensemble with the given input.

    Returns the execution result including agent outputs.
    """
    mcp = get_mcp_server()
    result = await mcp._invoke_tool({"ensemble_name": name, "input": request.input})
    return result


@router.post("/{name}/validate")
async def validate_ensemble(name: str) -> dict[str, Any]:
    """Validate an ensemble configuration.

    Returns validation result with any errors found.
    """
    mcp = get_mcp_server()
    result = await mcp._validate_ensemble_tool({"ensemble_name": name})
    return result


@router.get("/{name}/runnable")
async def check_ensemble_runnable(name: str) -> dict[str, Any]:
    """Check if ensemble can run with current providers.

    Returns runnable status including:
    - Whether the ensemble can run
    - Status of each agent's profile/provider
    - Suggested local alternatives for unavailable profiles
    """
    mcp = get_mcp_server()
    result = await mcp._check_ensemble_runnable_tool({"ensemble_name": name})
    return result
