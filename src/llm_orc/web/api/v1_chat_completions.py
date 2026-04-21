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
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from llm_orc.agentic.orchestrator_chunk import Completion, OrchestratorChunk
from llm_orc.agentic.session_registry import ChatMessage, SessionRegistry
from llm_orc.agentic.session_start import SessionContext, SessionStartCache
from llm_orc.web.api.sse_format import OpenAiSseFormatter

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


@router.post("/chat/completions", response_model=None)
async def chat_completions(
    request: _ChatCompletionsRequest,
) -> dict[str, Any] | StreamingResponse:
    """Resolve session, run session-start, then stream or return a body."""
    context = _resolve_context(request)

    if request.stream:
        return StreamingResponse(
            _stream_completion(context, model=request.model),
            media_type="text/event-stream",
        )

    _ = await _orchestrator_handoff(context)
    return _build_completion_body(model=request.model)


def _resolve_context(request: _ChatCompletionsRequest) -> SessionContext:
    """Run the pre-handoff work shared by streaming and non-streaming paths.

    Session identity, state retrieval, and session-start cache resolution
    all happen before the response is shaped so identity-resolution
    errors surface as proper HTTP errors rather than as SSE errors
    mid-stream.
    """
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
    return context


async def _orchestrator_handoff(context: SessionContext) -> str:
    """Placeholder for the Orchestrator Runtime (WP-C wires the real one).

    Returns empty content so the skeleton response is an honest
    OpenAI-shaped reply with nothing to say yet. WP-C replaces this
    with the ReAct loop.
    """
    del context
    return ""


async def _orchestrator_stream_handoff(
    context: SessionContext,
) -> AsyncIterator[OrchestratorChunk]:
    """Streaming placeholder for the Orchestrator Runtime.

    Yields the minimum chunk sequence that satisfies the Serving Layer
    → Orchestrator Runtime integration contract without actually
    running a ReAct loop: one ``Completion(stop)``. WP-C replaces this
    with the real Runtime, which will interleave content deltas,
    internal tool-call observations, and optional client tool-call
    final-turn chunks before completion.
    """
    del context
    yield Completion(finish_reason="stop")


async def _stream_completion(
    context: SessionContext, *, model: str
) -> AsyncIterator[bytes]:
    """Drive the Runtime's chunk stream through the OpenAI SSE formatter.

    The opener (``delta.role``) and the ``[DONE]`` terminator are
    OpenAI protocol conventions; the Runtime speaks only the neutral
    chunk vocabulary defined in ``orchestrator_chunk``.
    """
    formatter = OpenAiSseFormatter(
        stream_id=f"chatcmpl-{uuid.uuid4().hex}",
        model=model,
        created=int(time.time()),
    )
    yield formatter.start_assistant_turn()
    async for chunk in _orchestrator_stream_handoff(context):
        framed = formatter.format(chunk)
        if framed:
            yield framed
    yield formatter.done()


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
