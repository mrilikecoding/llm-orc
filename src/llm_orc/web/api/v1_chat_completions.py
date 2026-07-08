"""Serving Layer ``/v1/chat/completions`` endpoint.

Per ``docs/serving.md`` §Cycle 8 and ADR-046 §1.
Every chat-completions request is handled by the declarative Serving
Ensemble (classify -> seat -> marshal), executed by the L0 Ensemble
Engine. The endpoint:

1. Parses the OpenAI-compatible request.
2. Resolves a Session via :class:`SessionRegistry`; the session-start
   context resolves once per session (FC-9).
3. Hands the request to the :class:`ServingEnsembleCaller`, which runs
   the declarative serving turn and yields the shared
   :class:`OrchestratorChunk` vocabulary. Tool-driven clients receive a
   ``tool_calls`` turn; toolless clients a finish-with-text turn.
4. Produces either a non-streaming ``chat.completion`` body or an SSE
   stream.

The caller factory is process-scoped; module-level tests override it via
``monkeypatch.setattr``. The dissolved loop-driver serving surface
(ADR-033/034/043) was removed with the ``agentic/`` package at Cycle-8
WP-F8; the declarative Serving Ensemble is the only path.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from llm_orc.core.session.registry import SessionRegistry
from llm_orc.web.api.sse_format import OpenAiSseFormatter, encode_tool_call_for_message
from llm_orc.web.serving.chunks import (
    ClientToolCall,
    Completion,
    ContentDelta,
    OrchestratorChunk,
)
from llm_orc.web.serving.serving_ensemble_caller import ServingEnsembleCaller
from llm_orc.web.serving.session_start import (
    ChatMessage,
    SessionContext,
    SessionStartCache,
)

router = APIRouter(prefix="/v1", tags=["openai-compat"])

_SHARED_REGISTRY = SessionRegistry()
_SHARED_SESSION_START_CACHE = SessionStartCache()


def get_session_registry() -> SessionRegistry:
    """Return the process-scoped Session Registry.

    Tests override this factory to inject an isolated registry.
    """
    return _SHARED_REGISTRY


def get_session_start_cache() -> SessionStartCache:
    """Return the process-scoped session-start cache."""
    return _SHARED_SESSION_START_CACHE


def _resolve_serving_project_dir() -> Path:
    local = Path.cwd() / ".llm-orc"
    return local if local.exists() else Path.cwd()


def get_serving_ensemble_caller() -> ServingEnsembleCaller:
    """Return the Cycle-8 declarative Serving Ensemble caller (ADR-046 §1).

    Tests override this factory to point at a hermetic project dir whose
    ``code_generation`` seat is a deterministic echo (no model).
    """
    return ServingEnsembleCaller(project_dir=_resolve_serving_project_dir())


class _ChatCompletionMessage(BaseModel):
    """One message in the OpenAI chat-completions request.

    ``tool_call_id`` and ``tool_calls`` carry the OpenAI tool-round-trip
    shape: ``role: assistant`` messages whose prior turn closed with
    ``finish_reason: tool_calls`` echo their ``tool_calls[]`` back, and
    ``role: tool`` messages carry a ``tool_call_id`` plus the tool
    result as ``content``. Both are optional so the common case
    (``role: user``, ``role: system``) parses without change.
    ``content`` is nullable because OpenAI accepts ``content: null`` on
    an assistant message whose turn carried only tool calls.
    """

    role: str
    content: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class _ChatCompletionsRequest(BaseModel):
    """Minimal subset of the OpenAI ``/v1/chat/completions`` request body."""

    model: str
    messages: list[_ChatCompletionMessage]
    stream: bool = False
    tools: list[dict[str, Any]] = Field(default_factory=list)
    user: str | None = None


@router.post("/chat/completions", response_model=None)
async def chat_completions(
    request: _ChatCompletionsRequest,
) -> dict[str, Any] | StreamingResponse:
    """Resolve the session and run the declarative Serving Ensemble turn.

    Every request is handled by the Serving Ensemble (classify -> seat ->
    marshal, ADR-046 §1). Tool-driven clients receive a ``tool_calls``
    turn; toolless clients a finish-with-text turn. Serving-turn
    introspection is the vendor-neutral turn trace.
    """
    context = _resolve_context(request)
    serving_caller: _ChatCompletionsCaller = get_serving_ensemble_caller()
    if request.stream:
        return StreamingResponse(
            _stream_completion(context, serving_caller, model=request.model),
            media_type="text/event-stream",
        )
    return await _build_completion_body(context, serving_caller, model=request.model)


def _resolve_context(request: _ChatCompletionsRequest) -> SessionContext:
    """Run the pre-handoff work shared by streaming and non-streaming paths."""
    messages = [
        ChatMessage(
            role=message.role,
            content=message.content,
            tool_call_id=message.tool_call_id,
            tool_calls=tuple(message.tool_calls) if message.tool_calls else (),
        )
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


class _ChatCompletionsCaller(Protocol):
    """The chat-completions request driver.

    The :class:`ServingEnsembleCaller` (ADR-046 §1) satisfies this
    structurally — ``run`` yields the shared :class:`OrchestratorChunk`
    vocabulary the SSE formatter and non-streaming collector consume.
    Tests inject stub doubles through the same interface.
    """

    def run(self, context: SessionContext) -> AsyncIterator[OrchestratorChunk]: ...


@dataclass(frozen=True)
class _NonStreamingResult:
    """Collected chunk output shaped for the non-streaming response body.

    ``tool_calls`` is non-``None`` when the serving turn closed with a
    :class:`ClientToolCall` (the emit node's tool-driven finish) — the
    client-tool delegations are carried on the response's
    ``message.tool_calls`` field and ``finish_reason`` is ``"tool_calls"``
    per the OpenAI shape.
    """

    content: str
    finish_reason: str
    tool_calls: list[dict[str, Any]] | None = None


async def _collect_non_streaming(
    context: SessionContext, caller: _ChatCompletionsCaller
) -> _NonStreamingResult:
    """Drive the caller and flatten its chunks to the non-streaming shape.

    The serving turn yields ``ContentDelta`` text and a terminal
    ``Completion`` for finish-with-text turns (toolless clients); build
    turns close with a ``ClientToolCall``, which shapes into
    ``message.tool_calls`` + ``finish_reason: "tool_calls"`` per the
    OpenAI non-streaming tool-call shape (the streaming path renders the
    same chunk via the SSE formatter).
    """
    content_parts: list[str] = []
    finish_reason = "stop"
    tool_calls: list[dict[str, Any]] | None = None
    async for chunk in caller.run(context):
        if isinstance(chunk, ContentDelta):
            content_parts.append(chunk.content)
        elif isinstance(chunk, ClientToolCall):
            tool_calls = [encode_tool_call_for_message(tc) for tc in chunk.tool_calls]
            finish_reason = "tool_calls"
        elif isinstance(chunk, Completion):
            finish_reason = chunk.finish_reason
    return _NonStreamingResult(
        content="".join(content_parts),
        finish_reason=finish_reason,
        tool_calls=tool_calls,
    )


async def _stream_completion(
    context: SessionContext, caller: _ChatCompletionsCaller, *, model: str
) -> AsyncIterator[bytes]:
    """Drive the caller's chunk stream through the OpenAI SSE formatter."""
    formatter = OpenAiSseFormatter(
        stream_id=f"chatcmpl-{uuid.uuid4().hex}",
        model=model,
        created=int(time.time()),
    )
    yield formatter.start_assistant_turn()
    async for chunk in caller.run(context):
        framed = formatter.format(chunk)
        if framed:
            yield framed
    yield formatter.done()


async def _build_completion_body(
    context: SessionContext, caller: _ChatCompletionsCaller, *, model: str
) -> dict[str, Any]:
    """Shape the non-streaming response body from the caller's chunks.

    ``usage`` carries the Session's cumulative accounting ---
    ``SessionState.token_spend`` is the sum over every LLM call in the
    session. Per-request delta accounting can land in a follow-up when
    clients need it; the cumulative shape is what agentic coding tools
    display today.
    """
    pre_token_spend = context.state.token_spend
    result = await _collect_non_streaming(context, caller)
    turn_tokens = max(0, context.state.token_spend - pre_token_spend)
    message: dict[str, Any] = {"role": "assistant", "content": result.content}
    if result.tool_calls is not None:
        # OpenAI convention: content is null on a pure tool_calls turn.
        message["content"] = None
        message["tool_calls"] = result.tool_calls
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": result.finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": turn_tokens,
            "total_tokens": turn_tokens,
        },
    }
