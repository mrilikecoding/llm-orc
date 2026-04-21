"""Serving Layer ``/v1/chat/completions`` endpoint (Group 4 skeleton).

Per ``docs/agentic-serving/system-design.md`` §Serving Layer (L3),
§Integration Contracts, and roadmap WP-B Group 4. The endpoint stands
up the serving edge that later work packages fill in:

* WP-C replaces ``_orchestrator_handoff`` with the real Runtime.
* WP-B Group 5 adds SSE streaming (``stream: true``).

This module owns three integration contracts for Group 4: Serving
Layer → Session Registry (identity resolution), Serving Layer →
``resolve_session_start_context`` (Phase 1 hook), and the placeholder
Serving Layer → Orchestrator Runtime edge (stubbed).
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from llm_orc.agentic.session_registry import ChatMessage, SessionRegistry
from llm_orc.agentic.session_start import SessionContext, SessionStartCache

router = APIRouter(prefix="/v1", tags=["openai-compat"])


_SHARED_REGISTRY = SessionRegistry()
_SHARED_SESSION_START_CACHE = SessionStartCache()


def get_session_registry() -> SessionRegistry:
    """Return the process-scoped Session Registry.

    Group 4 uses in-memory state (cycle-status FF #42). Tests override
    this factory to inject an isolated registry per case, matching the
    pattern used in ``v1_models.get_orchestrator_config_resolver``.
    """
    return _SHARED_REGISTRY


def get_session_start_cache() -> SessionStartCache:
    """Return the process-scoped session-start cache.

    Tests override this factory to inject an isolated cache (often
    wrapping a spy resolver) per case.
    """
    return _SHARED_SESSION_START_CACHE


class _ChatCompletionMessage(BaseModel):
    """One message in the OpenAI chat-completions request."""

    role: str
    content: str


class _ChatCompletionsRequest(BaseModel):
    """Minimal subset of the OpenAI ``/v1/chat/completions`` request body.

    Group 4 accepts ``tools`` and ``user`` for forward compatibility but
    does not yet route them. WP-C consumes ``messages``; WP-F consumes
    ``tools``; WP-B Group 5 consumes ``stream``.
    """

    model: str
    messages: list[_ChatCompletionMessage]
    stream: bool = False
    tools: list[dict[str, Any]] = Field(default_factory=list)
    user: str | None = None


@router.post("/chat/completions")
async def chat_completions(request: _ChatCompletionsRequest) -> dict[str, Any]:
    """Group 4 skeleton: resolve session, run session-start, return placeholder."""
    if request.stream:
        raise HTTPException(
            status_code=400,
            detail=(
                "stream=true is not implemented in WP-B Group 4. "
                "SSE streaming lands in Group 5."
            ),
        )
    messages = [
        ChatMessage(role=message.role, content=message.content)
        for message in request.messages
    ]

    registry = get_session_registry()
    identity = registry.resolve_identity(
        messages=messages,
        user_field=request.user,
    )
    state = registry.get_or_create_state(identity)
    context = SessionContext(
        messages=messages,
        tools=list(request.tools),
        state=state,
    )

    cache = get_session_start_cache()
    cache.resolve(context)

    _ = await _orchestrator_handoff(context)

    return _build_completion_body(model=request.model)


async def _orchestrator_handoff(context: SessionContext) -> str:
    """Placeholder for the Orchestrator Runtime (WP-C wires the real one).

    Returns empty content so the skeleton response is an honest
    OpenAI-shaped reply with nothing to say yet. WP-C replaces this
    with the ReAct loop.
    """
    del context
    return ""


def _build_completion_body(*, model: str) -> dict[str, Any]:
    """Shape the Group 4 skeleton response."""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": ""},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }
