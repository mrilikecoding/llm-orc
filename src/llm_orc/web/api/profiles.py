"""Profiles API endpoints.

Provides REST API for model profile management, delegating to OrchestraService.
"""

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from llm_orc.web.api import get_orchestra_service

router = APIRouter(prefix="/api/profiles", tags=["profiles"])


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
    service = get_orchestra_service()
    return await service.read_profiles()


@router.post("")
async def create_profile(request: CreateProfileRequest) -> dict[str, Any]:
    """Create a new model profile."""
    service = get_orchestra_service()
    result = await service.create_profile(
        {
            "name": request.name,
            "provider": request.provider,
            "model": request.model,
            "system_prompt": request.system_prompt,
            "timeout_seconds": request.timeout_seconds,
        }
    )
    return result


class UpdateProfileRequest(BaseModel):
    """Request body for profile update."""

    provider: str | None = None
    model: str | None = None
    system_prompt: str | None = None
    timeout_seconds: int | None = None


@router.put("/{name}")
async def update_profile(name: str, request: UpdateProfileRequest) -> dict[str, Any]:
    """Update an existing model profile."""
    service = get_orchestra_service()
    changes = {k: v for k, v in request.model_dump().items() if v is not None}
    result = await service.update_profile({"name": name, "changes": changes})
    return result


@router.delete("/{name}")
async def delete_profile(name: str) -> dict[str, Any]:
    """Delete a model profile."""
    service = get_orchestra_service()
    result = await service.delete_profile({"name": name, "confirm": True})
    return result
