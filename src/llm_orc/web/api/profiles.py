"""Profiles API endpoints.

Provides REST API for model profile management, delegating to MCPServerV2.
"""

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from llm_orc.mcp import MCPServerV2

router = APIRouter(prefix="/api/profiles", tags=["profiles"])

# Singleton MCP server instance
_mcp_server: MCPServerV2 | None = None


def get_mcp_server() -> MCPServerV2:
    """Get or create the MCP server instance."""
    global _mcp_server
    if _mcp_server is None:
        _mcp_server = MCPServerV2()
    return _mcp_server


class CreateProfileRequest(BaseModel):
    """Request body for profile creation."""

    name: str
    provider: str
    model: str
    system_prompt: str | None = None
    timeout_seconds: int | None = None


@router.get("")
async def list_profiles() -> list[dict[str, Any]]:
    """List all configured model profiles."""
    mcp = get_mcp_server()
    return await mcp._read_profiles_resource()


@router.post("")
async def create_profile(request: CreateProfileRequest) -> dict[str, Any]:
    """Create a new model profile."""
    mcp = get_mcp_server()
    result = await mcp._create_profile_tool({
        "name": request.name,
        "provider": request.provider,
        "model": request.model,
        "system_prompt": request.system_prompt,
        "timeout_seconds": request.timeout_seconds,
    })
    return result


@router.delete("/{name}")
async def delete_profile(name: str) -> dict[str, Any]:
    """Delete a model profile."""
    mcp = get_mcp_server()
    result = await mcp._delete_profile_tool({"name": name, "confirm": True})
    return result
