"""Tests for the OpenAI-compatible SSE formatter.

Per ``docs/agentic-serving/system-design.md`` §Serving Layer → Orchestrator
Runtime (Integration Contracts). The formatter translates
``OrchestratorChunk`` variants to OpenAI ``chat.completion.chunk``
wire format, frames each chunk as ``data: {json}\\n\\n``, and terminates
the stream with ``data: [DONE]\\n\\n``. Each test parses the framed
payload back to JSON so assertions live at the observable boundary
(bytes on the wire), not formatter internals.
"""

import json
from typing import Any

from llm_orc.agentic.orchestrator_chunk import (
    ClientToolCall,
    Completion,
    ContentDelta,
    ErrorChunk,
    InternalToolCallInFlight,
    InternalToolCallResult,
    ToolCallInvocation,
    VisibilityEvent,
)
from llm_orc.web.api.sse_format import OpenAiSseFormatter


def _parse_sse_data(raw: bytes) -> dict[str, Any]:
    """Extract and parse the JSON payload from a framed ``data:`` line."""
    assert raw.startswith(b"data: "), f"expected SSE data frame, got {raw!r}"
    assert raw.endswith(b"\n\n"), f"expected SSE frame terminator, got {raw!r}"
    payload = raw[len(b"data: ") : -len(b"\n\n")]
    parsed: dict[str, Any] = json.loads(payload)
    return parsed


class TestContentDeltaFormatting:
    """Content deltas become OpenAI ``delta.content`` chunks."""

    def test_content_delta_formats_as_openai_chunk(self) -> None:
        formatter = OpenAiSseFormatter(
            stream_id="chatcmpl-abc",
            model="primary",
            created=1234567890,
        )

        raw = formatter.format(ContentDelta(content="hello"))

        payload = _parse_sse_data(raw)
        assert payload["id"] == "chatcmpl-abc"
        assert payload["object"] == "chat.completion.chunk"
        assert payload["created"] == 1234567890
        assert payload["model"] == "primary"
        choices = payload["choices"]
        assert isinstance(choices, list)
        assert len(choices) == 1
        choice = choices[0]
        assert choice["index"] == 0
        assert choice["delta"] == {"content": "hello"}
        assert choice["finish_reason"] is None


class TestCompletionFormatting:
    """Completion chunks carry ``finish_reason`` and an empty delta."""

    def test_stop_completion_emits_stop_finish_reason(self) -> None:
        formatter = OpenAiSseFormatter(
            stream_id="chatcmpl-abc",
            model="primary",
            created=1234567890,
        )

        raw = formatter.format(Completion(finish_reason="stop"))

        payload = _parse_sse_data(raw)
        choice = payload["choices"][0]
        assert choice["delta"] == {}
        assert choice["finish_reason"] == "stop"


class TestStreamStart:
    """Stream opener emits OpenAI's first-chunk ``delta.role`` convention.

    OpenAI streaming clients expect the first chunk in a stream to carry
    ``delta: {"role": "assistant"}`` — a protocol convention rather than
    a Runtime concept. Keeping the role delta out of the Runtime's chunk
    types and inside the formatter preserves FC-4 (Runtime imports no
    OpenAI-specific code).
    """

    def test_start_assistant_turn_emits_role_delta(self) -> None:
        formatter = OpenAiSseFormatter(
            stream_id="chatcmpl-abc",
            model="primary",
            created=1234567890,
        )

        raw = formatter.start_assistant_turn()

        payload = _parse_sse_data(raw)
        choice = payload["choices"][0]
        assert choice["delta"] == {"role": "assistant"}
        assert choice["finish_reason"] is None


class TestStreamDone:
    """``done()`` emits OpenAI's literal ``[DONE]`` terminator."""

    def test_done_terminator_matches_openai_convention(self) -> None:
        formatter = OpenAiSseFormatter(
            stream_id="chatcmpl-abc",
            model="primary",
            created=1234567890,
        )

        assert formatter.done() == b"data: [DONE]\n\n"


class TestClientToolCallFormatting:
    """Final-turn delegation becomes OpenAI ``delta.tool_calls`` with
    ``finish_reason: tool_calls`` — the Client Tool Surface Commitment
    (system-design §Client Tool Surface Commitment, Option C).
    """

    def test_client_tool_call_emits_tool_calls_delta(self) -> None:
        formatter = OpenAiSseFormatter(
            stream_id="chatcmpl-abc",
            model="primary",
            created=1234567890,
        )

        raw = formatter.format(
            ClientToolCall(
                tool_calls=(
                    ToolCallInvocation(
                        id="call_abc",
                        name="bash",
                        arguments='{"cmd": "ls"}',
                    ),
                )
            )
        )

        payload = _parse_sse_data(raw)
        choice = payload["choices"][0]
        assert choice["delta"] == {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"cmd": "ls"}'},
                }
            ]
        }
        assert choice["finish_reason"] == "tool_calls"

    def test_multiple_tool_calls_enumerate_with_index(self) -> None:
        formatter = OpenAiSseFormatter(
            stream_id="chatcmpl-abc",
            model="primary",
            created=1234567890,
        )

        raw = formatter.format(
            ClientToolCall(
                tool_calls=(
                    ToolCallInvocation(id="c1", name="bash", arguments="{}"),
                    ToolCallInvocation(id="c2", name="file_read", arguments="{}"),
                )
            )
        )

        payload = _parse_sse_data(raw)
        tool_calls = payload["choices"][0]["delta"]["tool_calls"]
        assert [call["index"] for call in tool_calls] == [0, 1]
        assert [call["id"] for call in tool_calls] == ["c1", "c2"]
        assert [call["function"]["name"] for call in tool_calls] == [
            "bash",
            "file_read",
        ]


class TestInternalToolCallVisibility:
    """Internal tool calls (orchestrator's own five-tool surface) are
    invisible to clients in Phase 1 — visibility form is OQ #2 and
    resolves in WP-E. The formatter's silent-drop preserves the Tool
    User's "endpoint is a model" mental model until visibility is
    decided.
    """

    def test_internal_tool_in_flight_emits_no_bytes(self) -> None:
        formatter = OpenAiSseFormatter(
            stream_id="chatcmpl-abc",
            model="primary",
            created=1234567890,
        )

        raw = formatter.format(
            InternalToolCallInFlight(id="call_1", name="invoke_ensemble")
        )

        assert raw == b""

    def test_internal_tool_result_emits_no_bytes(self) -> None:
        formatter = OpenAiSseFormatter(
            stream_id="chatcmpl-abc",
            model="primary",
            created=1234567890,
        )

        raw = formatter.format(
            InternalToolCallResult(id="call_1", summary="ensemble completed")
        )

        assert raw == b""


class TestVisibilityEventFormatting:
    """Visibility events surface as ``delta.content`` narration (ADR-008, OQ #2).

    Chose ``delta.content`` over SSE comment lines so a vanilla OpenAI-
    compat client (OpenCode, Roo Code, Cline) shows the tool user what
    llm-orc is composing in the same stream they converse in — the
    llm-conductor tinkering loop only closes when composition is
    observable to the tool user, not only to operator tooling.
    """

    def test_visibility_event_formats_as_delta_content_narration(self) -> None:
        formatter = OpenAiSseFormatter(
            stream_id="chatcmpl-abc",
            model="primary",
            created=1234567890,
        )

        event = VisibilityEvent(
            kind="composition",
            payload={"tool": "compose_ensemble", "arguments": {"name": "new"}},
        )
        raw = formatter.format(event)

        payload = _parse_sse_data(raw)
        assert payload["object"] == "chat.completion.chunk"
        content = payload["choices"][0]["delta"]["content"]
        assert content.startswith("[composition:")
        assert content.endswith("]")
        # The payload's key/value pairs are visible so the tool user can
        # see what was composed — not just that *some* composition fired.
        assert "compose_ensemble" in content
        assert '"name": "new"' in content
        # finish_reason stays null — visibility events don't end turns.
        assert payload["choices"][0]["finish_reason"] is None

    def test_visibility_event_content_is_single_line(self) -> None:
        """Narration is one line so it sits adjacent to other deltas.

        Multi-line content would interleave awkwardly with assistant text
        in the visible conversation. The event's kind + JSON payload fits
        on one line.
        """
        formatter = OpenAiSseFormatter(
            stream_id="chatcmpl-abc",
            model="primary",
            created=1234567890,
        )

        raw = formatter.format(
            VisibilityEvent(
                kind="composition",
                payload={
                    "tool": "compose_ensemble",
                    "arguments": {"name": "multi\nline\nthreat"},
                },
            )
        )

        payload = _parse_sse_data(raw)
        content = payload["choices"][0]["delta"]["content"]
        # JSON-encoded strings escape newlines — the narration line
        # itself carries no raw ``\n`` that would split the content.
        assert "\n" not in content


class TestErrorChunkFormatting:
    """Runtime exceptions become an ``error`` payload on the SSE stream.

    Per ``system-design.md`` §Serving Layer → Orchestrator Runtime
    (Integration Contracts), error-handling clause.
    """

    def test_error_chunk_emits_openai_error_payload(self) -> None:
        formatter = OpenAiSseFormatter(
            stream_id="chatcmpl-abc",
            model="primary",
            created=1234567890,
        )

        raw = formatter.format(
            ErrorChunk(message="orchestrator crashed", type="server_error")
        )

        payload = _parse_sse_data(raw)
        assert payload == {
            "error": {"message": "orchestrator crashed", "type": "server_error"}
        }
