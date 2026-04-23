"""Tests for the Orchestrator Runtime module.

Per `docs/agentic-serving/system-design.md` §Orchestrator Runtime (L2),
§Integration Contracts (Serving Layer → Orchestrator Runtime,
Orchestrator Runtime → Budget Controller, Orchestrator Runtime →
Orchestrator Tool Dispatch), and §Fitness Criteria (FC-4, FC-10).

Covers scenarios:

* §Session terminates gracefully on turn limit exhaustion
* §Session terminates gracefully on token limit exhaustion
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from llm_orc.agentic.budget_controller import BudgetController
from llm_orc.agentic.orchestrator_chunk import (
    Completion,
    ContentDelta,
    InternalToolCallInFlight,
    InternalToolCallResult,
    OrchestratorChunk,
)
from llm_orc.agentic.orchestrator_runtime import OrchestratorRuntime
from llm_orc.agentic.orchestrator_tool_dispatch import (
    InternalToolCall,
    ToolCallError,
    ToolCallResult,
    ToolCallSuccess,
)
from llm_orc.agentic.session_registry import SessionIdentity, SessionState
from llm_orc.agentic.session_start import ChatMessage, SessionContext
from llm_orc.models.base import ToolCall, ToolCallingResponse, ToolCallUsage


class _ScriptedLLM:
    """Plays back a prepared sequence of tool-calling responses.

    Satisfies ``OrchestratorLLM`` structurally — matches
    ``ModelInterface.generate_with_tools``'s signature. Records each
    call so tests can assert on the tool schemas and message state at
    each iteration.
    """

    def __init__(self, responses: list[ToolCallingResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []

    async def generate_with_tools(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ToolCallingResponse:
        self.calls.append((list(messages), list(tools)))
        if not self._responses:
            raise AssertionError("_ScriptedLLM ran out of canned responses")
        return self._responses.pop(0)


class _StubToolDispatch:
    """Returns canned tool results keyed by tool-call id.

    Satisfies ``ToolDispatcher`` structurally. Defaults to raising if
    asked to dispatch a call the test didn't expect — vacuous-mock
    hazard prevention.
    """

    def __init__(self, results: dict[str, ToolCallResult]) -> None:
        self._results = dict(results)
        self.calls: list[InternalToolCall] = []

    async def dispatch(self, call: InternalToolCall) -> ToolCallResult:
        self.calls.append(call)
        if call.id not in self._results:
            raise AssertionError(
                f"_StubToolDispatch received unexpected call id={call.id!r}"
            )
        return self._results[call.id]


def _make_session_context(messages: list[ChatMessage] | None = None) -> SessionContext:
    state = SessionState(
        identity=SessionIdentity(value="test-session", method="user_field"),
    )
    return SessionContext(
        messages=list(messages or [ChatMessage(role="user", content="hello")]),
        tools=[],
        state=state,
    )


async def _collect(
    chunks: AsyncIterator[OrchestratorChunk],
) -> list[OrchestratorChunk]:
    return [chunk async for chunk in chunks]


class TestRuntimeStopsCleanly:
    """The simplest path: LLM emits a stop response with no tool calls."""

    @pytest.mark.asyncio
    async def test_runtime_yields_content_and_completion(self) -> None:
        llm = _ScriptedLLM(
            responses=[
                ToolCallingResponse(
                    content="hello back",
                    tool_calls=[],
                    usage=ToolCallUsage(
                        prompt_tokens=10, completion_tokens=5, total_tokens=15
                    ),
                    finish_reason="stop",
                ),
            ]
        )
        budget = BudgetController(turn_limit=10, token_limit=1000)
        tool_dispatch = _StubToolDispatch(results={})
        runtime = OrchestratorRuntime(
            llm=llm, budget=budget, tool_dispatch=tool_dispatch
        )

        chunks = await _collect(runtime.run(_make_session_context()))

        content_chunks = [c for c in chunks if isinstance(c, ContentDelta)]
        completion_chunks = [c for c in chunks if isinstance(c, Completion)]
        assert [c.content for c in content_chunks] == ["hello back"]
        assert len(completion_chunks) == 1
        assert completion_chunks[0].finish_reason == "stop"


class TestBudgetExhaustion:
    """Scenarios: Session terminates gracefully on turn/token limit exhaustion.

    The Budget check is control-plane (AS-3): it runs before every
    iteration regardless of LLM intent. A session that opens already at
    its turn or token limit terminates before the first LLM call.
    """

    @pytest.mark.asyncio
    async def test_turn_limit_exhausted_before_first_iteration(self) -> None:
        llm = _ScriptedLLM(responses=[])
        # Session state already at turn_limit — check fires on entry.
        context = _make_session_context()
        context.state.turn_count = 5
        budget = BudgetController(turn_limit=5, token_limit=1000)
        runtime = OrchestratorRuntime(
            llm=llm, budget=budget, tool_dispatch=_StubToolDispatch(results={})
        )

        chunks = await _collect(runtime.run(context))

        # The LLM must not have been called — Budget blocks entry.
        assert llm.calls == []
        # Last chunk is a length-reason completion; prior chunks include
        # at least one ContentDelta naming the exhaustion.
        completions = [c for c in chunks if isinstance(c, Completion)]
        contents = [c for c in chunks if isinstance(c, ContentDelta)]
        assert len(completions) == 1
        assert completions[0].finish_reason == "length"
        assert len(contents) >= 1
        combined = " ".join(c.content for c in contents).lower()
        assert "turn" in combined
        assert "exhausted" in combined or "limit" in combined

    @pytest.mark.asyncio
    async def test_token_limit_exhausted_before_first_iteration(self) -> None:
        llm = _ScriptedLLM(responses=[])
        context = _make_session_context()
        context.state.token_spend = 1000
        budget = BudgetController(turn_limit=10, token_limit=1000)
        runtime = OrchestratorRuntime(
            llm=llm, budget=budget, tool_dispatch=_StubToolDispatch(results={})
        )

        chunks = await _collect(runtime.run(context))

        assert llm.calls == []
        completions = [c for c in chunks if isinstance(c, Completion)]
        contents = [c for c in chunks if isinstance(c, ContentDelta)]
        assert len(completions) == 1
        assert completions[0].finish_reason == "length"
        combined = " ".join(c.content for c in contents).lower()
        assert "token" in combined


class TestReActLoop:
    """Runtime dispatches tool calls and feeds results back as observations."""

    @pytest.mark.asyncio
    async def test_runtime_dispatches_tool_call_and_feeds_result_back(
        self,
    ) -> None:
        # Iteration 1: LLM emits a tool call.
        # Iteration 2: LLM sees the tool result and emits a stop response.
        llm = _ScriptedLLM(
            responses=[
                ToolCallingResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call_t1",
                            name="list_ensembles",
                            arguments_json="{}",
                        )
                    ],
                    usage=ToolCallUsage(
                        prompt_tokens=20, completion_tokens=5, total_tokens=25
                    ),
                    finish_reason="tool_calls",
                ),
                ToolCallingResponse(
                    content="I found 2 ensembles.",
                    tool_calls=[],
                    usage=ToolCallUsage(
                        prompt_tokens=40, completion_tokens=6, total_tokens=46
                    ),
                    finish_reason="stop",
                ),
            ]
        )
        canned_result = ToolCallSuccess(
            id="call_t1",
            name="list_ensembles",
            content=[{"name": "a"}, {"name": "b"}],
        )
        tool_dispatch = _StubToolDispatch(results={"call_t1": canned_result})
        budget = BudgetController(turn_limit=10, token_limit=1000)
        runtime = OrchestratorRuntime(
            llm=llm, budget=budget, tool_dispatch=tool_dispatch
        )

        chunks = await _collect(runtime.run(_make_session_context()))

        # Tool Dispatch saw one call with the parsed arguments.
        assert len(tool_dispatch.calls) == 1
        dispatched = tool_dispatch.calls[0]
        assert dispatched.id == "call_t1"
        assert dispatched.name == "list_ensembles"
        assert dispatched.arguments == {}

        # Chunk order: InFlight → Result → ContentDelta (from iter 2) → Completion
        in_flight = [c for c in chunks if isinstance(c, InternalToolCallInFlight)]
        results = [c for c in chunks if isinstance(c, InternalToolCallResult)]
        content = [c for c in chunks if isinstance(c, ContentDelta)]
        completion = [c for c in chunks if isinstance(c, Completion)]

        assert len(in_flight) == 1
        assert in_flight[0].id == "call_t1"
        assert in_flight[0].name == "list_ensembles"
        assert len(results) == 1
        assert results[0].id == "call_t1"
        assert len(content) == 1
        assert content[0].content == "I found 2 ensembles."
        assert len(completion) == 1
        assert completion[0].finish_reason == "stop"

        # Second LLM call must have seen the tool result as an observation.
        assert len(llm.calls) == 2
        second_messages = llm.calls[1][0]
        tool_messages = [m for m in second_messages if m.get("role") == "tool"]
        assert len(tool_messages) == 1
        assert tool_messages[0]["tool_call_id"] == "call_t1"

    @pytest.mark.asyncio
    async def test_runtime_accumulates_token_spend_into_session_state(
        self,
    ) -> None:
        """Each iteration's ``total_tokens`` feeds into SessionState — the
        next Budget check sees the accumulated spend."""
        llm = _ScriptedLLM(
            responses=[
                ToolCallingResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call_1", name="list_ensembles", arguments_json="{}"
                        )
                    ],
                    usage=ToolCallUsage(
                        prompt_tokens=30, completion_tokens=10, total_tokens=40
                    ),
                    finish_reason="tool_calls",
                ),
                ToolCallingResponse(
                    content="done",
                    tool_calls=[],
                    usage=ToolCallUsage(
                        prompt_tokens=50, completion_tokens=15, total_tokens=65
                    ),
                    finish_reason="stop",
                ),
            ]
        )
        tool_dispatch = _StubToolDispatch(
            results={
                "call_1": ToolCallSuccess(
                    id="call_1", name="list_ensembles", content=[]
                )
            }
        )
        context = _make_session_context()
        budget = BudgetController(turn_limit=10, token_limit=1000)
        runtime = OrchestratorRuntime(
            llm=llm, budget=budget, tool_dispatch=tool_dispatch
        )

        await _collect(runtime.run(context))

        # Two iterations ran; state accumulates both turn and token spend.
        assert context.state.turn_count == 2
        assert context.state.token_spend == 40 + 65

    @pytest.mark.asyncio
    async def test_runtime_terminates_mid_loop_when_budget_exhausted_between_iterations(
        self,
    ) -> None:
        """Scenario coverage: 'the last completed turn's output is returned'.

        Iteration 1 runs (yielding its content), token spend pushes past
        the limit, iteration 2 is blocked by the Budget check, the
        exhaustion message is surfaced, and the session terminates.
        """
        llm = _ScriptedLLM(
            responses=[
                ToolCallingResponse(
                    content="I'll call list_ensembles.",
                    tool_calls=[
                        ToolCall(id="c1", name="list_ensembles", arguments_json="{}")
                    ],
                    usage=ToolCallUsage(
                        prompt_tokens=50, completion_tokens=50, total_tokens=100
                    ),
                    finish_reason="tool_calls",
                ),
                # This response would be returned if the LLM were called again,
                # but budget check blocks iteration 2.
                ToolCallingResponse(
                    content="should not appear",
                    tool_calls=[],
                    usage=ToolCallUsage(
                        prompt_tokens=0, completion_tokens=0, total_tokens=0
                    ),
                    finish_reason="stop",
                ),
            ]
        )
        tool_dispatch = _StubToolDispatch(
            results={"c1": ToolCallSuccess(id="c1", name="list_ensembles", content=[])}
        )
        context = _make_session_context()
        budget = BudgetController(turn_limit=100, token_limit=100)
        runtime = OrchestratorRuntime(
            llm=llm, budget=budget, tool_dispatch=tool_dispatch
        )

        chunks = await _collect(runtime.run(context))

        # Iteration 1's content IS in the stream.
        contents = [c.content for c in chunks if isinstance(c, ContentDelta)]
        assert "I'll call list_ensembles." in contents
        # Iteration 2 did NOT run — only one LLM call.
        assert len(llm.calls) == 1
        # Final completion is the length-reason exhaustion.
        completions = [c for c in chunks if isinstance(c, Completion)]
        assert len(completions) == 1
        assert completions[0].finish_reason == "length"
        # The exhaustion message is explicit.
        combined = " ".join(contents).lower()
        assert "token" in combined

    @pytest.mark.asyncio
    async def test_runtime_propagates_tool_error_as_observation(self) -> None:
        """Scenario hookup: §Invocation outside the tool set is rejected.

        When Tool Dispatch returns a ``ToolCallError``, the Runtime
        must not raise — it surfaces the error as a ``role: tool``
        observation so the LLM can adjust its plan and the ReAct loop
        continues.
        """
        llm = _ScriptedLLM(
            responses=[
                ToolCallingResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="bad_call",
                            name="hallucinated_tool",
                            arguments_json="{}",
                        )
                    ],
                    usage=ToolCallUsage(
                        prompt_tokens=10, completion_tokens=5, total_tokens=15
                    ),
                    finish_reason="tool_calls",
                ),
                ToolCallingResponse(
                    content="I'll stop and report the error.",
                    tool_calls=[],
                    usage=ToolCallUsage(
                        prompt_tokens=30, completion_tokens=10, total_tokens=40
                    ),
                    finish_reason="stop",
                ),
            ]
        )
        tool_dispatch = _StubToolDispatch(
            results={
                "bad_call": ToolCallError(
                    id="bad_call",
                    name="hallucinated_tool",
                    kind="unknown_tool",
                    reason="not in the orchestrator's committed tool set",
                ),
            }
        )
        budget = BudgetController(turn_limit=10, token_limit=1000)
        runtime = OrchestratorRuntime(
            llm=llm, budget=budget, tool_dispatch=tool_dispatch
        )

        chunks = await _collect(runtime.run(_make_session_context()))

        # The Runtime did not raise; the ReAct loop completed normally.
        completions = [c for c in chunks if isinstance(c, Completion)]
        assert len(completions) == 1
        assert completions[0].finish_reason == "stop"
        # The second iteration saw the tool error as an observation.
        assert len(llm.calls) == 2
        second_messages = llm.calls[1][0]
        tool_messages = [m for m in second_messages if m.get("role") == "tool"]
        assert len(tool_messages) == 1
        assert tool_messages[0]["tool_call_id"] == "bad_call"
        # The content payload names the error kind.
        assert "unknown_tool" in tool_messages[0]["content"]
