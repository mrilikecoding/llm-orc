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

import hashlib
import json
import os
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator, model_validator

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


def _log_wire_shape(request: _ChatCompletionsRequest) -> None:
    """Append one JSONL row describing the request's message shape.

    Enabled by ``LLM_ORC_SERVE_WIRE_LOG=<path>`` — the issue #82 entry-gate
    instrumentation for observing client-side history rewrites (compaction,
    forks) on the wire. Records roles, content lengths, and a rolling
    prefix-hash chain; never the content itself. The prefix hashes make a
    client rewrite visible as a divergence point between requests.
    """
    log_path = os.environ.get("LLM_ORC_SERVE_WIRE_LOG")
    if not log_path:
        return
    rows = []
    digest = hashlib.sha256()
    for message in request.messages:
        content = _text_content(message.content) or ""
        digest.update(f"{message.role}\x00{content}\x00".encode())
        rows.append(
            {
                "role": message.role,
                "content_len": len(content),
                "tool_calls": len(message.tool_calls or []),
                "prefix_hash": digest.hexdigest()[:12],
            }
        )
    row = {
        "ts": time.time(),
        "message_count": len(request.messages),
        "messages": rows,
    }
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    except OSError:  # observation must never break serving
        pass


def _resolve_serving_project_dir() -> Path:
    local = Path.cwd() / ".llm-orc"
    return local if local.exists() else Path.cwd()


_SHARED_CALLERS: dict[Path, ServingEnsembleCaller] = {}


def get_serving_ensemble_caller() -> ServingEnsembleCaller:
    """Return the Cycle-8 declarative Serving Ensemble caller (ADR-046 §1).

    Shared per project dir (like _SHARED_REGISTRY) so the caller's
    ensemble-config cache survives across requests (issue #93). Tests
    override this factory to point at a hermetic project dir whose
    ``code_generation`` seat is a deterministic echo (no model).
    """
    project_dir = _resolve_serving_project_dir()
    caller = _SHARED_CALLERS.get(project_dir)
    if caller is None:
        caller = ServingEnsembleCaller(project_dir=project_dir)
        _SHARED_CALLERS[project_dir] = caller
    return caller


class _ChatCompletionMessage(BaseModel):
    """One message in the OpenAI chat-completions request.

    ``tool_call_id`` and ``tool_calls`` carry the OpenAI tool-round-trip
    shape: ``role: assistant`` messages whose prior turn closed with
    ``finish_reason: tool_calls`` echo their ``tool_calls[]`` back, and
    ``role: tool`` messages carry a ``tool_call_id`` plus the tool
    result as ``content``. Both are optional so the common case
    (``role: user``, ``role: system``) parses without change.
    ``content`` is nullable because OpenAI accepts ``content: null`` on
    an assistant message whose turn carried only tool calls, and may be
    list-shaped (content parts) — normalized to plain text at this
    boundary by :func:`_text_content` (issue #107) so the serving layer
    stays str-only downstream.
    """

    role: str
    content: str | list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None

    @field_validator("content")
    @classmethod
    def _parts_are_well_shaped(
        cls, value: str | list[dict[str, Any]] | None
    ) -> str | list[dict[str, Any]] | None:
        """Reject malformed content parts on any role.

        Every part needs a string ``type`` (wire-invalid otherwise — a
        typeless text part would silently drop from the join); a text part
        needs a string ``text`` (str() coercion would leak Python reprs
        into the transcript and session hash).
        """
        if not isinstance(value, list):
            return value
        for part in value:
            if not isinstance(part.get("type"), str):
                raise ValueError("content part requires a string 'type'")
            if part["type"] == "text" and not isinstance(part.get("text"), str):
                raise ValueError("text content part requires a string 'text'")
        return value

    @model_validator(mode="after")
    def _user_parts_carry_text(self) -> _ChatCompletionMessage:
        """Reject a USER parts message whose text joins to blank.

        Empty user content silently slides the turn boundary back to a
        stale task (PR #113 review blocker). Scoped to user messages:
        only they feed ``_task_from``/``_latest_user_index``, and an
        empty tool result as parts is wire-legal (silent-success
        commands) with an honest empty-string render.
        """
        if self.role == "user" and isinstance(self.content, list):
            if not _joined_text_parts(self.content).strip():
                raise ValueError("user content parts carry no text to route on")
        return self


def _joined_text_parts(parts: list[dict[str, Any]]) -> str:
    """VALIDATED text parts joined into plain text; non-text parts
    (``image_url``, …) carry nothing the serving layer can route on and
    are dropped. Callers must not hand this raw wire dicts — the bare
    indexing assumes ``_parts_are_well_shaped`` ran."""
    return "\n".join(part["text"] for part in parts if part["type"] == "text")


def _text_content(content: str | list[dict[str, Any]] | None) -> str | None:
    """Normalize validated message content to OpenAI's plain-string shape."""
    if not isinstance(content, list):
        return content
    return _joined_text_parts(content)


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
    _log_wire_shape(request)
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
            content=_text_content(message.content),
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
