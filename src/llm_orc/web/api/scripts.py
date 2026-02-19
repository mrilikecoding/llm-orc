"""Scripts API endpoints.

Provides REST API for script management, delegating to MCPServer.
"""

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from llm_orc.web.api import get_mcp_server

router = APIRouter(prefix="/api/scripts", tags=["scripts"])


class TestScriptRequest(BaseModel):
    """Request body for script testing."""

    input: str


@router.get("")
async def list_scripts() -> dict[str, Any]:
    """List all available scripts by category."""
    mcp = get_mcp_server()
    result = await mcp._list_scripts_tool({})
    return result


@router.get("/{category}/{name}")
async def get_script(category: str, name: str) -> dict[str, Any]:
    """Get script details."""
    mcp = get_mcp_server()
    result = await mcp._get_script_tool({"name": name, "category": category})
    return result


@router.post("/{category}/{name}/test")
async def test_script(
    category: str, name: str, request: TestScriptRequest
) -> dict[str, Any]:
    """Test a script with sample input."""
    mcp = get_mcp_server()
    result = await mcp._test_script_tool(
        {
            "name": name,
            "category": category,
            "input": request.input,
        }
    )
    return result
