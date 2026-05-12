"""Cycle-acceptance integration test for Conversation Compaction (WP-E4).

Step 5.5 row 5 — "Conversation Compaction five-layer pipeline maintains
orchestrator coherence across long sessions". The
``scenarios.md`` cycle-acceptance table's layer-match column is **no**
because individual unit-scenario tests exercise each pipeline layer in
isolation; this integration fixture composes them through a multi-turn
ReAct loop and verifies the cheapest-first ordering's coherence
property — the FC-14 fitness criterion the system-design lists at
``test_compaction_holds_context_below_threshold_across_long_session``.

Fitness target (system-design.agents.md L207): across a multi-turn
fixture session that exceeds the configured context threshold by
≥ 200%, the orchestrator's per-turn LLM call receives a context whose
estimated token count stays at or below the trigger threshold for at
least 95% of iterations.

The fixture drives a real ``OrchestratorRuntime`` through a scripted
LLM that emits an ``invoke_ensemble`` tool call on every iteration and
a stub tool dispatch that returns a 50,000-character tool result each
turn. Without compaction the messages array would grow unboundedly;
with the ADR-012 pipeline wired in, Layer 0 (persist-large-tool-
results) keeps the per-turn token count bounded.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from llm_orc.agentic.budget_controller import BudgetController
from llm_orc.agentic.conversation_compaction import (
    CompactionDefaults,
    ConversationCompaction,
)
from llm_orc.agentic.orchestrator_chunk import OrchestratorChunk
from llm_orc.agentic.orchestrator_runtime import OrchestratorRuntime
from llm_orc.agentic.orchestrator_tool_dispatch import (
    InternalToolCall,
    PhantomToolCallError,
    ToolCallResult,
    ToolCallSuccess,
)
from llm_orc.agentic.session_registry import SessionIdentity, SessionState
from llm_orc.agentic.session_start import ChatMessage, SessionContext
from llm_orc.models.base import ToolCall, ToolCallingResponse, ToolCallUsage

# Trigger threshold sized small so the test runs in tens of
# milliseconds; the proportional test for the fitness criterion is
# what matters, not absolute token counts.
_TRIGGER_TOKEN_COUNT = 1_000
_PERSIST_THRESHOLD_CHARS = 200
_TOOL_RESULT_SIZE_CHARS = 5_000  # ~1,250 tokens — fires Layer 0 each turn
_NUM_TURNS = 10  # exceeds threshold by ≥ 200% in accumulation


class _MultiTurnLLM:
    """Emits an invoke_ensemble tool call for the first N turns, then
    stops. Records every messages array it sees so the fitness check
    can score per-iteration token counts post-compaction."""

    def __init__(self, num_tool_turns: int) -> None:
        self._num_tool_turns = num_tool_turns
        self._call_count = 0
        self.per_call_messages: list[list[dict[str, Any]]] = []

    async def generate_with_tools(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],  # noqa: ARG002
    ) -> ToolCallingResponse:
        self.per_call_messages.append([dict(m) for m in messages])
        self._call_count += 1
        if self._call_count <= self._num_tool_turns:
            call_id = f"call-{self._call_count}"
            return ToolCallingResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id=call_id,
                        name="invoke_ensemble",
                        arguments_json=('{"name": "summarize", "input": "do work"}'),
                    )
                ],
                usage=ToolCallUsage(
                    prompt_tokens=10, completion_tokens=5, total_tokens=15
                ),
                finish_reason="tool_calls",
            )
        return ToolCallingResponse(
            content="all done.",
            tool_calls=[],
            usage=ToolCallUsage(prompt_tokens=10, completion_tokens=3, total_tokens=13),
            finish_reason="stop",
        )


class _LargeResultToolDispatch:
    """Returns a 50K-char tool result on every dispatch."""

    async def dispatch(
        self,
        call: InternalToolCall,
        *,
        session_id: str = "",  # noqa: ARG002
    ) -> ToolCallResult:
        return ToolCallSuccess(
            id=call.id,
            name=call.name,
            content="X" * _TOOL_RESULT_SIZE_CHARS,
        )

    def validate_response(
        self,
        response_text: str,  # noqa: ARG002
        tool_call_names: tuple[str, ...],  # noqa: ARG002
        *,
        session_id: str = "",  # noqa: ARG002
    ) -> PhantomToolCallError | None:
        return None


async def _collect(chunks: AsyncIterator[OrchestratorChunk]) -> None:
    async for _ in chunks:
        pass


def _estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Mirror the production heuristic (char / 4)."""
    total_chars = 0
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
    return total_chars // 4


@pytest.mark.asyncio
async def test_compaction_holds_context_below_threshold_across_long_session(
    tmp_path: Path,
) -> None:
    """The fitness target from system-design.agents.md L207.

    Drives the Runtime through ``_NUM_TURNS`` iterations of tool calls
    whose results would, in aggregate, exceed the trigger threshold by
    far more than 200%. With ADR-012 wired in, the per-turn LLM call
    receives a context whose token count stays at or below the
    trigger threshold for ≥ 95% of iterations.
    """
    defaults = CompactionDefaults(
        persist_threshold_chars=_PERSIST_THRESHOLD_CHARS,
        idle_window_minutes=60,
        session_notes_token_cap=12_288,
        layer_4_circuit_breaker_threshold=3,
        trigger_token_count=_TRIGGER_TOKEN_COUNT,
        summarizer_ensemble=None,
    )
    compaction = ConversationCompaction(
        defaults=defaults,
        persistence_root=tmp_path,
    )
    llm = _MultiTurnLLM(num_tool_turns=_NUM_TURNS)
    runtime = OrchestratorRuntime(
        llm=llm,
        budget=BudgetController(turn_limit=_NUM_TURNS + 5, token_limit=10_000_000),
        tool_dispatch=_LargeResultToolDispatch(),
        compaction=compaction,
    )

    state = SessionState(
        identity=SessionIdentity(
            value=f"session-{uuid.uuid4().hex}", method="user_field"
        ),
    )
    context = SessionContext(
        messages=[ChatMessage(role="user", content="kick off a long agentic task")],
        tools=[],
        state=state,
    )

    await _collect(runtime.run(context))

    # Score per-iteration LLM token counts.
    per_call_tokens = [_estimate_tokens(messages) for messages in llm.per_call_messages]
    assert len(per_call_tokens) >= _NUM_TURNS, (
        f"Fixture under-drove the Runtime — only {len(per_call_tokens)} "
        f"LLM calls (expected at least {_NUM_TURNS})."
    )

    # Fitness target: ≥ 95% of iterations at-or-below threshold.
    at_or_below_count = sum(1 for n in per_call_tokens if n <= _TRIGGER_TOKEN_COUNT)
    fraction_below = at_or_below_count / len(per_call_tokens)
    assert fraction_below >= 0.95, (
        f"Compaction failed to bound context across the multi-turn "
        f"session: only {fraction_below:.0%} of iterations stayed at or "
        f"below the trigger threshold ({_TRIGGER_TOKEN_COUNT} tokens). "
        f"Per-call token counts: {per_call_tokens}"
    )

    # Context-rot exceedance sanity check: without compaction the
    # accumulated context would have far exceeded the threshold. The
    # check confirms the fixture actually stresses the pipeline (not
    # a vacuous pass).
    total_tool_chars = _NUM_TURNS * _TOOL_RESULT_SIZE_CHARS
    total_tool_tokens = total_tool_chars // 4
    assert total_tool_tokens >= 2 * _TRIGGER_TOKEN_COUNT, (
        "Fixture sizing is below the ≥ 200% exceedance target named "
        "by the fitness criterion — increase tool-result size or "
        "turn count."
    )
