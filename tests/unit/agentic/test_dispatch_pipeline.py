"""Tests for the Dispatch Pipeline (Cycle 7 WP-A, ADR-027).

The pipeline is the framework-driven plan → dispatch → synthesize caller
that replaces ``OrchestratorRuntime`` on the chat-completions surface. It
yields the same ``OrchestratorChunk`` vocabulary so ``OpenAiSseFormatter``
consumes its output unchanged (ARCHITECT Finding 8 disposition).

Scenarios from ``docs/agentic-serving/scenarios.md`` §"Framework-Driven
Dispatch Pipeline (ADR-027)", including the two Spike-ν-driven plan-
validation scenarios.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from llm_orc.agentic.dispatch_envelope import DispatchEnvelope
from llm_orc.agentic.dispatch_event_substrate import DispatchEventSubstrate
from llm_orc.agentic.dispatch_pipeline import (
    DirectCompletionFallback,
    DispatchFired,
    DispatchPipeline,
    DispatchPlan,
    PlanEmitted,
    SynthesizerCompleted,
    plan_to_internal_tool_call,
)
from llm_orc.agentic.orchestrator_chunk import (
    Completion,
    ContentDelta,
    OrchestratorChunk,
)
from llm_orc.agentic.orchestrator_tool_dispatch import (
    InternalToolCall,
    ToolCallResult,
    ToolCallSuccess,
)
from llm_orc.agentic.session_registry import SessionIdentity, SessionState
from llm_orc.agentic.session_start import ChatMessage, SessionContext

CAPABILITY_NAMES = frozenset(
    {
        "web-searcher",
        "text-summarizer",
        "code-generator",
        "claim-extractor",
        "argument-mapper",
        "prose-improver",
    }
)


# --- Test doubles for the ports -------------------------------------------


class _FakePlanner:
    """RoutingPlanner double returning a pre-scripted plan (or None)."""

    def __init__(self, plan: DispatchPlan | None) -> None:
        self._plan = plan
        self.requests: list[str] = []

    async def plan(self, *, request: str) -> DispatchPlan | None:
        self.requests.append(request)
        return self._plan


class _FakeSynthesizer:
    """ResponseSynthesizer double yielding a pre-scripted response.

    Records the structured input it received so tests can assert the
    pipeline composed the (request, dispatched, planned-but-not-run,
    dispatch-results) tuple correctly.
    """

    def __init__(self, text: str) -> None:
        self._text = text
        self.calls: list[dict[str, object]] = []

    async def synthesize(
        self,
        *,
        original_request: str,
        dispatched: list[str],
        planned_but_not_run: list[str],
        dispatch_results: list[tuple[str, str]],
    ) -> AsyncIterator[str]:
        self.calls.append(
            {
                "original_request": original_request,
                "dispatched": dispatched,
                "planned_but_not_run": planned_but_not_run,
                "dispatch_results": dispatch_results,
            }
        )
        yield self._text


class _FakeToolDispatch:
    """ToolDispatcher double returning a pre-scripted envelope-bearing result."""

    def __init__(self, primary: str = "dispatch output") -> None:
        self._primary = primary
        self.calls: list[InternalToolCall] = []

    async def dispatch(
        self, call: InternalToolCall, *, session_id: str = ""
    ) -> ToolCallResult:
        self.calls.append(call)
        return ToolCallSuccess(
            id=call.id,
            name=call.name,
            content=self._primary,
            envelope=DispatchEnvelope(status="success", primary=self._primary),
        )


class _RecordingSink:
    """EventSink double capturing every event emitted through the substrate."""

    def __init__(self) -> None:
        self.events: list[object] = []

    def consume(self, event: object) -> None:
        self.events.append(event)


def _make_context(prompt: str = "write a sorting function") -> SessionContext:
    return SessionContext(
        messages=[ChatMessage(role="user", content=prompt)],
        tools=[],
        state=SessionState(
            identity=SessionIdentity(value="test-session", method="user_field")
        ),
    )


async def _collect(
    chunks: AsyncIterator[OrchestratorChunk],
) -> list[OrchestratorChunk]:
    return [chunk async for chunk in chunks]


class TestPlanDispatchSynthesizeHappyPath:
    """Scenario: chat-completions request flows through plan→dispatch→synthesize."""

    @pytest.mark.asyncio
    async def test_capability_matched_request_dispatches_then_synthesizes(
        self,
    ) -> None:
        planner = _FakePlanner(
            DispatchPlan(
                action="dispatch",
                ensemble="code-generator",
                input="write a sorting function",
                rationale="programming task",
            )
        )
        synthesizer = _FakeSynthesizer("Here is your sorting function.")
        tool_dispatch = _FakeToolDispatch(primary="def sort(xs): return sorted(xs)")
        substrate = DispatchEventSubstrate()
        pipeline = DispatchPipeline(
            planner=planner,
            synthesizer=synthesizer,
            tool_dispatch=tool_dispatch,
            capability_names=CAPABILITY_NAMES,
            event_substrate=substrate,
        )

        chunks = await _collect(pipeline.run(_make_context()))

        # Observable output: synthesizer content + clean stop completion.
        content = [c for c in chunks if isinstance(c, ContentDelta)]
        completions = [c for c in chunks if isinstance(c, Completion)]
        assert "".join(c.content for c in content) == "Here is your sorting function."
        assert len(completions) == 1
        assert completions[0].finish_reason == "stop"

        # The dispatch fired against the planned ensemble via the adapter.
        assert len(tool_dispatch.calls) == 1
        assert tool_dispatch.calls[0].name == "invoke_ensemble"
        assert tool_dispatch.calls[0].arguments["ensemble_name"] == "code-generator"

        # The synthesizer saw the dispatch result content.
        assert synthesizer.calls[0]["dispatched"] == ["code-generator"]
        assert synthesizer.calls[0]["dispatch_results"] == [
            ("code-generator", "def sort(xs): return sorted(xs)")
        ]

    @pytest.mark.asyncio
    async def test_happy_path_emits_plan_dispatch_synthesizer_events(self) -> None:
        substrate = DispatchEventSubstrate()
        sink = _RecordingSink()
        substrate.register_sink(sink)
        pipeline = DispatchPipeline(
            planner=_FakePlanner(
                DispatchPlan("dispatch", "code-generator", "x", "programming")
            ),
            synthesizer=_FakeSynthesizer("done"),
            tool_dispatch=_FakeToolDispatch(),
            capability_names=CAPABILITY_NAMES,
            event_substrate=substrate,
        )

        await _collect(pipeline.run(_make_context()))

        types = [type(e) for e in sink.events]
        assert types == [PlanEmitted, DispatchFired, SynthesizerCompleted]
        # No degradation event on the happy path.
        assert not any(isinstance(e, DirectCompletionFallback) for e in sink.events)


class TestDirectCompletionPath:
    """Scenario: no-capability-match request flows through the direct path."""

    @pytest.mark.asyncio
    async def test_direct_action_skips_dispatch_and_synthesizes(self) -> None:
        tool_dispatch = _FakeToolDispatch()
        synthesizer = _FakeSynthesizer("Reykjavik is the capital of Iceland.")
        substrate = DispatchEventSubstrate()
        sink = _RecordingSink()
        substrate.register_sink(sink)
        pipeline = DispatchPipeline(
            planner=_FakePlanner(
                DispatchPlan("direct", None, None, "no capability matches")
            ),
            synthesizer=synthesizer,
            tool_dispatch=tool_dispatch,
            capability_names=CAPABILITY_NAMES,
            event_substrate=substrate,
        )

        chunks = await _collect(pipeline.run(_make_context("what's the capital?")))

        # No dispatch fired; synthesizer ran with empty dispatch results.
        assert tool_dispatch.calls == []
        assert synthesizer.calls[0]["dispatch_results"] == []
        content = "".join(c.content for c in chunks if isinstance(c, ContentDelta))
        assert content == "Reykjavik is the capital of Iceland."
        assert any(isinstance(c, Completion) for c in chunks)
        # A degradation event records the direct-completion reason.
        fallbacks = [e for e in sink.events if isinstance(e, DirectCompletionFallback)]
        assert len(fallbacks) == 1
        assert fallbacks[0].request_shape_category == "no_capability_match"


class TestPlanValidationRejectsInvalidPlans:
    """Scenario (Spike ν): invalid action or unregistered ensemble → direct."""

    @pytest.mark.asyncio
    async def test_injected_invalid_action_is_rejected_to_direct(self) -> None:
        # Spike ν E1: planner steered to emit action "launch".
        tool_dispatch = _FakeToolDispatch()
        substrate = DispatchEventSubstrate()
        sink = _RecordingSink()
        substrate.register_sink(sink)
        pipeline = DispatchPipeline(
            planner=_FakePlanner(DispatchPlan("launch", "all", None, "injected")),
            synthesizer=_FakeSynthesizer("handled directly"),
            tool_dispatch=tool_dispatch,
            capability_names=CAPABILITY_NAMES,
            event_substrate=substrate,
        )

        chunks = await _collect(pipeline.run(_make_context()))

        assert tool_dispatch.calls == []
        fallbacks = [e for e in sink.events if isinstance(e, DirectCompletionFallback)]
        assert len(fallbacks) == 1
        assert fallbacks[0].request_shape_category == "invalid_plan"
        assert any(isinstance(c, Completion) for c in chunks)

    @pytest.mark.asyncio
    async def test_fabricated_ensemble_name_is_rejected_to_direct(self) -> None:
        # Spike ν E3: planner dispatched to a non-registered "oracle".
        tool_dispatch = _FakeToolDispatch()
        substrate = DispatchEventSubstrate()
        sink = _RecordingSink()
        substrate.register_sink(sink)
        pipeline = DispatchPipeline(
            planner=_FakePlanner(
                DispatchPlan("dispatch", "oracle", "secret", "explicit naming")
            ),
            synthesizer=_FakeSynthesizer("handled directly"),
            tool_dispatch=tool_dispatch,
            capability_names=CAPABILITY_NAMES,
            event_substrate=substrate,
        )

        await _collect(pipeline.run(_make_context()))

        assert tool_dispatch.calls == []
        fallbacks = [e for e in sink.events if isinstance(e, DirectCompletionFallback)]
        assert fallbacks[0].request_shape_category == "invalid_plan"


class TestUnparseablePlanRoutesToDirect:
    """Scenario (Spike ν A6): empty/unparseable planner output → direct."""

    @pytest.mark.asyncio
    async def test_none_plan_routes_to_direct_completion(self) -> None:
        tool_dispatch = _FakeToolDispatch()
        substrate = DispatchEventSubstrate()
        sink = _RecordingSink()
        substrate.register_sink(sink)
        pipeline = DispatchPipeline(
            planner=_FakePlanner(None),
            synthesizer=_FakeSynthesizer("direct answer"),
            tool_dispatch=tool_dispatch,
            capability_names=CAPABILITY_NAMES,
            event_substrate=substrate,
        )

        chunks = await _collect(pipeline.run(_make_context()))

        assert tool_dispatch.calls == []
        assert any(isinstance(c, Completion) for c in chunks)
        fallbacks = [e for e in sink.events if isinstance(e, DirectCompletionFallback)]
        assert fallbacks[0].request_shape_category == "unparseable_plan"


class TestPlanToInternalToolCallAdapter:
    """Scenario (integration): plan-stage output is InternalToolCall-compatible."""

    def test_adapter_builds_invoke_ensemble_call(self) -> None:
        plan = DispatchPlan("dispatch", "code-generator", "write a function", "code")
        call = plan_to_internal_tool_call(plan, "sess-dispatch-0001")
        assert call.name == "invoke_ensemble"
        assert call.id == "sess-dispatch-0001"
        assert call.arguments == {
            "ensemble_name": "code-generator",
            "input": "write a function",
        }
