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
from llm_orc.agentic.conversation_compaction import CompactedContext
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

    ``extra_validation_patterns`` extends ADR-017's default
    assertion-pattern set for the structural validation guard
    (``validate_response``); tests that exercise the operator-extension
    surface set this directly on the stub. The stub uses the real
    scanner from :mod:`llm_orc.agentic.tool_call_validation_guard` so
    the wiring matches production.
    """

    def __init__(
        self,
        results: dict[str, ToolCallResult],
        *,
        extra_validation_patterns: tuple[str, ...] = (),
    ) -> None:
        self._results = dict(results)
        self.calls: list[InternalToolCall] = []
        self.session_ids: list[str] = []
        self._extra_validation_patterns = extra_validation_patterns

    async def dispatch(
        self, call: InternalToolCall, *, session_id: str = ""
    ) -> ToolCallResult:
        self.calls.append(call)
        self.session_ids.append(session_id)
        if call.id not in self._results:
            raise AssertionError(
                f"_StubToolDispatch received unexpected call id={call.id!r}"
            )
        return self._results[call.id]

    def validate_response(
        self,
        response_text: str,
        tool_call_names: tuple[str, ...],
        *,
        session_id: str = "",
    ) -> Any:
        from llm_orc.agentic.tool_call_validation_guard import (
            scan_response_for_phantom_claims,
        )

        return scan_response_for_phantom_claims(
            response_text,
            tool_call_names,
            dispatch_context={"session_id": session_id},
            extra_patterns=self._extra_validation_patterns,
        )


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
        # Session identity plumbed from SessionContext.state through
        # dispatch (WP-H) so per-session Calibration Gate state keys on
        # the orchestrator LLM's actual Session.
        assert tool_dispatch.session_ids == ["test-session"]

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


class TestOrchestratorSystemPrompt:
    """The orchestrator's system prompt always prepends the LLM's message list.

    WP-F Group 3 introduces ``system_prompt`` as a Runtime construction
    parameter. It teaches the LLM about the five internal tools, Option
    C's one-kind-per-turn discipline, and the ``needs_client_tool``
    retry convention (roadmap ODP #8 mechanism i). The prompt is
    *always* prepended — agentic coding clients (OpenCode, Cursor,
    Roo Code) typically supply their own system message with tool-user
    guidance; the orchestrator's guidance sits ahead of the client's so
    the discipline survives competing instructions.
    """

    @pytest.mark.asyncio
    async def test_system_prompt_prepends_as_role_system(self) -> None:
        llm = _ScriptedLLM(
            responses=[
                ToolCallingResponse(
                    content="done",
                    tool_calls=[],
                    usage=ToolCallUsage(
                        prompt_tokens=10, completion_tokens=2, total_tokens=12
                    ),
                    finish_reason="stop",
                ),
            ]
        )
        budget = BudgetController(turn_limit=10, token_limit=1000)
        runtime = OrchestratorRuntime(
            llm=llm,
            budget=budget,
            tool_dispatch=_StubToolDispatch(results={}),
            system_prompt="Orchestrator discipline text.",
        )

        await _collect(runtime.run(_make_session_context()))

        messages = llm.calls[0][0]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Orchestrator discipline text."

    @pytest.mark.asyncio
    async def test_system_prompt_sits_ahead_of_client_system_message(
        self,
    ) -> None:
        """Client's system prompt (if any) follows the orchestrator's.

        When a client (e.g. OpenCode) sends its own ``role: system``
        message, the orchestrator's prompt prepends ahead of it so the
        orchestrator's discipline wins over competing client guidance.
        """
        llm = _ScriptedLLM(
            responses=[
                ToolCallingResponse(
                    content="ok",
                    tool_calls=[],
                    usage=ToolCallUsage(
                        prompt_tokens=8, completion_tokens=2, total_tokens=10
                    ),
                    finish_reason="stop",
                ),
            ]
        )
        budget = BudgetController(turn_limit=10, token_limit=1000)
        context = _make_session_context(
            messages=[
                ChatMessage(role="system", content="Client's guidance."),
                ChatMessage(role="user", content="task"),
            ]
        )
        runtime = OrchestratorRuntime(
            llm=llm,
            budget=budget,
            tool_dispatch=_StubToolDispatch(results={}),
            system_prompt="Orchestrator guidance.",
        )

        await _collect(runtime.run(context))

        messages = llm.calls[0][0]
        assert [m["role"] for m in messages[:3]] == ["system", "system", "user"]
        assert messages[0]["content"] == "Orchestrator guidance."
        assert messages[1]["content"] == "Client's guidance."

    @pytest.mark.asyncio
    async def test_empty_system_prompt_does_not_prepend(self) -> None:
        """An empty system_prompt is a no-op — the LLM sees client messages unchanged.

        Tests that use a minimal default constructor should not observe a
        blank ``role: system`` at position 0. Operators who explicitly
        configure an empty prompt get the same — no silent synthetic
        message inserted.
        """
        llm = _ScriptedLLM(
            responses=[
                ToolCallingResponse(
                    content="ok",
                    tool_calls=[],
                    usage=ToolCallUsage(
                        prompt_tokens=5, completion_tokens=1, total_tokens=6
                    ),
                    finish_reason="stop",
                ),
            ]
        )
        budget = BudgetController(turn_limit=10, token_limit=1000)
        runtime = OrchestratorRuntime(
            llm=llm,
            budget=budget,
            tool_dispatch=_StubToolDispatch(results={}),
            system_prompt="",
        )

        await _collect(runtime.run(_make_session_context()))

        messages = llm.calls[0][0]
        assert messages[0]["role"] != "system"
        assert [m["role"] for m in messages] == ["user"]


class TestPhantomToolCallGuard:
    """ADR-017 — structural validation guard interposed on the orchestrator
    response. The guard scans response text for assertion patterns; if a
    phantom claim is detected with no corresponding tool-call structure,
    the runtime injects a structural-feedback diagnostic and re-enters the
    ReAct loop so the orchestrator reformulates."""

    @pytest.mark.asyncio
    async def test_phantom_response_with_zero_tool_calls_loops_with_diagnostic(
        self,
    ) -> None:
        """The spike phrasing (essay 005 Wave 3.A Trial 3): the orchestrator
        emits prose claiming a tool call without emitting the structure.

        The runtime must NOT yield Completion(stop) on this response — the
        orchestrator's reasoning surface must receive the structural
        feedback and reformulate (per ADR-017 §Rejection)."""
        phantom_response = ToolCallingResponse(
            content=(
                "The tool call has been made and the result is displayed "
                "above as a `role:tool` observation."
            ),
            tool_calls=[],
            usage=ToolCallUsage(
                prompt_tokens=10, completion_tokens=20, total_tokens=30
            ),
            finish_reason="stop",
        )
        clean_followup = ToolCallingResponse(
            content="Apologies; let me retry without that prose.",
            tool_calls=[],
            usage=ToolCallUsage(
                prompt_tokens=20, completion_tokens=10, total_tokens=30
            ),
            finish_reason="stop",
        )
        llm = _ScriptedLLM(responses=[phantom_response, clean_followup])
        runtime = OrchestratorRuntime(
            llm=llm,
            budget=BudgetController(turn_limit=10, token_limit=1000),
            tool_dispatch=_StubToolDispatch(results={}),
        )

        chunks = await _collect(runtime.run(_make_session_context()))

        # The runtime called the LLM twice — phantom + reformulated turn.
        assert len(llm.calls) == 2
        # Second call's messages include the phantom assistant turn AND
        # the structural-feedback diagnostic appended after it.
        second_call_messages = llm.calls[1][0]
        assistant_turns = [m for m in second_call_messages if m["role"] == "assistant"]
        assert len(assistant_turns) == 1
        assert "the result is displayed above" in assistant_turns[0]["content"]
        # The diagnostic is the message immediately following the assistant
        # turn — surfaces the phantom_tool_call kind so the orchestrator's
        # reasoning surface incorporates the rejection.
        diagnostic = second_call_messages[-1]
        assert "phantom_tool_call" in diagnostic["content"]
        # Final completion is from the clean followup, not the phantom.
        completions = [c for c in chunks if isinstance(c, Completion)]
        assert len(completions) == 1
        assert completions[0].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_clean_response_with_no_tool_calls_proceeds_without_diagnostic(
        self,
    ) -> None:
        """Existing behavior preservation — a response with no tool_calls
        and no phantom assertion patterns terminates with finish_reason
        ``stop`` (one LLM call only)."""
        clean_response = ToolCallingResponse(
            content="Here is your answer with no fabricated tool claims.",
            tool_calls=[],
            usage=ToolCallUsage(
                prompt_tokens=10, completion_tokens=10, total_tokens=20
            ),
            finish_reason="stop",
        )
        llm = _ScriptedLLM(responses=[clean_response])
        runtime = OrchestratorRuntime(
            llm=llm,
            budget=BudgetController(turn_limit=10, token_limit=1000),
            tool_dispatch=_StubToolDispatch(results={}),
        )

        await _collect(runtime.run(_make_session_context()))

        assert len(llm.calls) == 1

    @pytest.mark.asyncio
    async def test_response_with_tool_calls_proceeds_regardless_of_prose(
        self,
    ) -> None:
        """Match scenario: a response with prose that would otherwise match
        an assertion pattern is NOT flagged when the structural anchor (an
        emitted tool_call) is present in the same response."""
        invoke_call = ToolCall(
            id="call-1",
            name="invoke_ensemble",
            arguments_json='{"name": "x", "input": "y"}',
        )
        narrating_response = ToolCallingResponse(
            content=("I called invoke_ensemble and the result was the answer."),
            tool_calls=[invoke_call],
            usage=ToolCallUsage(
                prompt_tokens=10, completion_tokens=10, total_tokens=20
            ),
            finish_reason="tool_calls",
        )
        clean_followup = ToolCallingResponse(
            content="Done.",
            tool_calls=[],
            usage=ToolCallUsage(prompt_tokens=20, completion_tokens=5, total_tokens=25),
            finish_reason="stop",
        )
        llm = _ScriptedLLM(responses=[narrating_response, clean_followup])
        # Stub dispatcher returns a successful result for the invoke_call.
        dispatch = _StubToolDispatch(
            results={
                "call-1": ToolCallSuccess(
                    id="call-1",
                    name="invoke_ensemble",
                    content={"summary": "ok"},
                ),
            }
        )
        runtime = OrchestratorRuntime(
            llm=llm,
            budget=BudgetController(turn_limit=10, token_limit=1000),
            tool_dispatch=dispatch,
        )

        await _collect(runtime.run(_make_session_context()))

        # Tool call dispatched; no phantom-rejection diagnostic injected.
        assert len(dispatch.calls) == 1
        # Followup messages should NOT contain a phantom-diagnostic
        # role:user shaped like the rejection payload.
        second_call_messages = llm.calls[1][0]
        for message in second_call_messages:
            assert "phantom_tool_call" not in str(message.get("content", ""))

    @pytest.mark.asyncio
    async def test_operator_extended_pattern_flows_through_to_guard(
        self,
    ) -> None:
        """Scenario 4 (operator-extensibility) at the runtime level: an
        operator-supplied pattern flows through the runtime constructor to
        the guard scanner; matching prose without a tool_call structure
        triggers the same diagnostic-then-loop behavior."""
        operator_phrasing_response = ToolCallingResponse(
            content="site-specific phantom phrasing happened in this turn",
            tool_calls=[],
            usage=ToolCallUsage(
                prompt_tokens=10, completion_tokens=10, total_tokens=20
            ),
            finish_reason="stop",
        )
        clean_followup = ToolCallingResponse(
            content="ok.",
            tool_calls=[],
            usage=ToolCallUsage(prompt_tokens=20, completion_tokens=2, total_tokens=22),
            finish_reason="stop",
        )
        llm = _ScriptedLLM(responses=[operator_phrasing_response, clean_followup])
        runtime = OrchestratorRuntime(
            llm=llm,
            budget=BudgetController(turn_limit=10, token_limit=1000),
            tool_dispatch=_StubToolDispatch(
                results={},
                extra_validation_patterns=(r"\bsite-specific phantom phrasing\b",),
            ),
        )

        await _collect(runtime.run(_make_session_context()))

        # Operator-extended pattern matched; loop continued with diagnostic.
        assert len(llm.calls) == 2
        diagnostic = llm.calls[1][0][-1]
        assert "phantom_tool_call" in diagnostic["content"]


class _StubCompaction:
    """Captures compaction invocations and replays scripted results.

    Satisfies ``Compaction`` structurally. Tests can pre-load the
    script with sentinel ``CompactedContext`` values to verify the
    Runtime threads the compacted messages into the next LLM call.
    """

    def __init__(self, script: list[CompactedContext] | None = None) -> None:
        self._script = list(script) if script else []
        self.invocations: list[tuple[list[dict[str, Any]], str]] = []

    def compact(
        self,
        messages: list[dict[str, Any]],
        *,
        session_id: str,
    ) -> CompactedContext:
        self.invocations.append(([dict(m) for m in messages], session_id))
        if self._script:
            return self._script.pop(0)
        # Default: do not trigger; messages flow through unchanged.
        return CompactedContext(
            messages=[dict(m) for m in messages],
            layers_applied=(),
            triggered=False,
        )


class TestRuntimeCompactionInvocation:
    """WP-E4 / ADR-012 — the Runtime invokes Conversation Compaction at
    every turn boundary; the resulting compacted messages array is what
    flows into the next LLM call (system-design.agents.md L612)."""

    @pytest.mark.asyncio
    async def test_runtime_invokes_compaction_before_each_llm_call(self) -> None:
        """compact() fires once per ReAct iteration, before
        generate_with_tools — the turn-boundary contract."""
        llm = _ScriptedLLM(
            responses=[
                ToolCallingResponse(
                    content="done.",
                    tool_calls=[],
                    usage=ToolCallUsage(
                        prompt_tokens=5, completion_tokens=2, total_tokens=7
                    ),
                    finish_reason="stop",
                )
            ]
        )
        compaction = _StubCompaction()
        runtime = OrchestratorRuntime(
            llm=llm,
            budget=BudgetController(turn_limit=10, token_limit=1000),
            tool_dispatch=_StubToolDispatch(results={}),
            compaction=compaction,
        )

        await _collect(runtime.run(_make_session_context()))

        # One LLM call, one compaction invocation — same count.
        assert len(compaction.invocations) == 1
        assert len(llm.calls) == 1

    @pytest.mark.asyncio
    async def test_runtime_threads_session_id_to_compaction(self) -> None:
        """The compaction module needs the session_id to key per-session
        state (circuit-breaker, tool first-seen timestamps); the Runtime
        threads it from the SessionContext's identity."""
        llm = _ScriptedLLM(
            responses=[
                ToolCallingResponse(
                    content="ok.",
                    tool_calls=[],
                    usage=ToolCallUsage(
                        prompt_tokens=5, completion_tokens=2, total_tokens=7
                    ),
                    finish_reason="stop",
                )
            ]
        )
        compaction = _StubCompaction()
        runtime = OrchestratorRuntime(
            llm=llm,
            budget=BudgetController(turn_limit=10, token_limit=1000),
            tool_dispatch=_StubToolDispatch(results={}),
            compaction=compaction,
        )

        await _collect(runtime.run(_make_session_context()))

        assert compaction.invocations[0][1] == "test-session"

    @pytest.mark.asyncio
    async def test_runtime_uses_compacted_messages_in_next_llm_call(self) -> None:
        """When compaction triggers, the LLM receives the compacted
        messages — not the original. The contract from
        system-design.agents.md L612: the resulting ``CompactedContext``
        is what flows into the next LLM call."""
        sentinel_message = {"role": "system", "content": "[compacted sentinel]"}
        compaction = _StubCompaction(
            script=[
                CompactedContext(
                    messages=[sentinel_message],
                    layers_applied=(0,),
                    triggered=True,
                )
            ]
        )
        llm = _ScriptedLLM(
            responses=[
                ToolCallingResponse(
                    content="ok.",
                    tool_calls=[],
                    usage=ToolCallUsage(
                        prompt_tokens=5, completion_tokens=2, total_tokens=7
                    ),
                    finish_reason="stop",
                )
            ]
        )
        runtime = OrchestratorRuntime(
            llm=llm,
            budget=BudgetController(turn_limit=10, token_limit=1000),
            tool_dispatch=_StubToolDispatch(results={}),
            compaction=compaction,
        )

        await _collect(runtime.run(_make_session_context()))

        # The LLM saw the sentinel — proving the compacted messages
        # flowed through. ``triggered=True`` activates the substitution.
        llm_messages = llm.calls[0][0]
        assert sentinel_message in llm_messages

    @pytest.mark.asyncio
    async def test_below_threshold_compaction_leaves_messages_unchanged(self) -> None:
        """When compaction does NOT trigger (``triggered=False``), the
        Runtime keeps the original messages — no spurious substitution."""
        compaction = _StubCompaction()  # default: triggered=False
        llm = _ScriptedLLM(
            responses=[
                ToolCallingResponse(
                    content="ok.",
                    tool_calls=[],
                    usage=ToolCallUsage(
                        prompt_tokens=5, completion_tokens=2, total_tokens=7
                    ),
                    finish_reason="stop",
                )
            ]
        )
        runtime = OrchestratorRuntime(
            llm=llm,
            budget=BudgetController(turn_limit=10, token_limit=1000),
            tool_dispatch=_StubToolDispatch(results={}),
            compaction=compaction,
        )

        await _collect(runtime.run(_make_session_context()))

        # Original user message survives.
        llm_messages = llm.calls[0][0]
        user_messages = [m for m in llm_messages if m.get("role") == "user"]
        assert user_messages == [{"role": "user", "content": "hello"}]

    @pytest.mark.asyncio
    async def test_runtime_without_compaction_works_unchanged(self) -> None:
        """``compaction=None`` is the prior-WP behavior — Runtime
        operates without the Compaction interposer."""
        llm = _ScriptedLLM(
            responses=[
                ToolCallingResponse(
                    content="ok.",
                    tool_calls=[],
                    usage=ToolCallUsage(
                        prompt_tokens=5, completion_tokens=2, total_tokens=7
                    ),
                    finish_reason="stop",
                )
            ]
        )
        runtime = OrchestratorRuntime(
            llm=llm,
            budget=BudgetController(turn_limit=10, token_limit=1000),
            tool_dispatch=_StubToolDispatch(results={}),
            # compaction defaults to None
        )

        chunks = await _collect(runtime.run(_make_session_context()))

        # Behavior preserved: single LLM call, stop completion.
        assert len(llm.calls) == 1
        completions = [c for c in chunks if isinstance(c, Completion)]
        assert len(completions) == 1
        assert completions[0].finish_reason == "stop"
