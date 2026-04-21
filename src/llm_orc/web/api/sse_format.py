"""OpenAI-compatible SSE formatter for ``OrchestratorChunk`` variants.

Per ``docs/agentic-serving/system-design.md`` §Serving Layer (L3) and
§Integration Contracts (Serving Layer → Orchestrator Runtime). The
formatter owns the protocol translation from the neutral
``OrchestratorChunk`` type surface to the OpenAI ``chat.completion.chunk``
wire format, framed as SSE ``data:`` lines. Keeping the framer here —
next to the endpoint that owns the OpenAI protocol — preserves FC-4:
the Orchestrator Runtime never imports SSE-specific code.
"""

from __future__ import annotations

import json
from typing import Any

from llm_orc.agentic.orchestrator_chunk import (
    ClientToolCall,
    Completion,
    ContentDelta,
    ErrorChunk,
    InternalToolCallInFlight,
    InternalToolCallResult,
    OrchestratorChunk,
    ToolCallInvocation,
)

_CHUNK_OBJECT = "chat.completion.chunk"


class OpenAiSseFormatter:
    """Translate ``OrchestratorChunk`` values into framed SSE bytes."""

    def __init__(self, *, stream_id: str, model: str, created: int) -> None:
        self._stream_id = stream_id
        self._model = model
        self._created = created

    def start_assistant_turn(self) -> bytes:
        """Emit the opening chunk carrying OpenAI's ``delta.role`` convention.

        The Runtime does not speak OpenAI protocol; this framing is a
        Serving-Layer convention that precedes the Runtime's chunks.
        """
        return self._frame(self._chunk_envelope(delta={"role": "assistant"}))

    def done(self) -> bytes:
        """Emit OpenAI's literal ``[DONE]`` terminator."""
        return b"data: [DONE]\n\n"

    def format(self, chunk: OrchestratorChunk) -> bytes:
        """Translate a chunk into framed SSE bytes.

        Returns empty bytes for variants whose visibility is deferred
        (Phase-1 internal tool calls) so callers can forward the result
        to the wire uniformly.
        """
        match chunk:
            case ContentDelta(content):
                return self._frame(self._chunk_envelope(delta={"content": content}))
            case Completion(finish_reason):
                return self._frame(
                    self._chunk_envelope(delta={}, finish_reason=finish_reason)
                )
            case ClientToolCall(tool_calls):
                return self._frame(
                    self._chunk_envelope(
                        delta={"tool_calls": _encode_tool_calls(tool_calls)},
                        finish_reason="tool_calls",
                    )
                )
            case InternalToolCallInFlight() | InternalToolCallResult():
                # Phase 1 surfaces nothing — OQ #2 resolves in WP-E.
                # Directional lean from the Group 5 gate (2026-04-21):
                # emit SSE comment lines (``: {"event":...}\n\n``) for
                # operator in-stream visibility of the llm-conductor /
                # in-session ensemble-design pattern. OpenAI-compat
                # clients ignore SSE comments per spec; llm-orc operator
                # tooling parses them. Validation deferred to rdd-play.
                return b""
            case ErrorChunk(message, type_):
                return self._frame({"error": {"message": message, "type": type_}})

    def _chunk_envelope(
        self, *, delta: dict[str, Any], finish_reason: str | None = None
    ) -> dict[str, Any]:
        return {
            "id": self._stream_id,
            "object": _CHUNK_OBJECT,
            "created": self._created,
            "model": self._model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }

    @staticmethod
    def _frame(payload: dict[str, Any]) -> bytes:
        return b"data: " + json.dumps(payload).encode("utf-8") + b"\n\n"


def _encode_tool_calls(
    tool_calls: tuple[ToolCallInvocation, ...],
) -> list[dict[str, Any]]:
    """Translate ``ToolCallInvocation`` values to OpenAI delta shape.

    The ``index`` field enumerates the position in the ``tool_calls``
    array, which OpenAI streaming clients use to reconstruct partial
    tool-call deltas across chunks. Group 5 ships one chunk per final
    turn (all arguments at once); WP-F may later split into multiple
    chunks that share an ``index`` per tool call.
    """
    return [
        {
            "index": index,
            "id": call.id,
            "type": "function",
            "function": {"name": call.name, "arguments": call.arguments},
        }
        for index, call in enumerate(tool_calls)
    ]
