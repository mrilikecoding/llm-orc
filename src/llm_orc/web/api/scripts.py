"""Scripts API endpoints.

Provides REST API for script management, delegating to OrchestraService.
"""

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from llm_orc.web.api import get_orchestra_service

router = APIRouter(prefix="/api/scripts", tags=["scripts"])


class TestScriptRequest(BaseModel):
    """Request body for script testing."""

    input: str


@router.get("")
async def list_scripts() -> dict[str, Any]:
    """List all available scripts by category."""
    service = get_orchestra_service()
    result = await service.list_scripts({})
    return result


@router.get("/{category}/{name}")
async def get_script(category: str, name: str) -> dict[str, Any]:
    """Get script details."""
    service = get_orchestra_service()
    result = await service.get_script({"name": name, "category": category})
    return result


@router.post("/{category}/{name}/test")
async def test_script(
    category: str, name: str, request: TestScriptRequest
) -> dict[str, Any]:
    """Test a script with sample input."""
    service = get_orchestra_service()
    result = await service.test_script(
        {
            "name": name,
            "category": category,
            "input": request.input,
        }
    )
    return result
