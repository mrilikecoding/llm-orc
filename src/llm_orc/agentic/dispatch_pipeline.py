"""Dispatch Pipeline (Cycle 7 WP-A, ADR-027) — L2.

The framework-driven plan → dispatch → synthesize orchestration that
replaces ``OrchestratorRuntime`` as the chat-completions caller. Per
``docs/agentic-serving/system-design.agents.md`` §Module: Dispatch
Pipeline.

The pipeline owns:

* The three-stage sequence — Plan (via Routing Planner), Dispatch (via
  Orchestrator Tool Dispatch when ``action == "dispatch"``), Synthesize
  (via Response Synthesizer).
* The **Plan → InternalToolCall adapter** (:func:`plan_to_internal_tool_call`),
  a pure function translating the routing-planner's
  ``{action, ensemble, input, rationale}`` into the existing dispatch
  surface's :class:`InternalToolCall`.
* The streaming chunk surface — ``run`` yields the same
  ``OrchestratorChunk`` vocabulary the Runtime produces (``ContentDelta``,
  ``Completion``) so ``OpenAiSseFormatter`` consumes pipeline output
  without modification (ARCHITECT Finding 8 disposition).
* Emission of the four pipeline-stage events (``PlanEmitted``,
  ``DispatchFired``, ``SynthesizerCompleted``, ``DirectCompletionFallback``)
  through the Dispatch Event Substrate.

**Plan validation is a non-optional pipeline stage** (Spike ν Findings
ν.2/ν.3 + the deployment policy in ADR-031's Spike ν amendment). The
planner faithfully parses model output; the *pipeline* decides whether
the plan is actionable. A plan whose action is not ``dispatch``/``direct``
or whose ensemble is not in the registered capability set is rejected and
the request falls to direct completion — this is what neutralizes the
prompt-injection cases (E1 ``launch``, E3 ``oracle``) the planner alone
can be steered into. An absent/unparseable plan (the A6 reliability mode)
routes to direct completion as a defined path, not an exception.

Per the ports + spike-backed-adapters decomposition (BUILD entry,
2026-05-23): the pipeline depends on :class:`RoutingPlanner` and
:class:`ResponseSynthesizer` *protocols*. WP-A wires spike-ensemble-backed
adapters behind these ports; WP-B (production Routing Planner) and WP-C
(production Response Synthesizer) replace the adapters without changing
the pipeline.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

from llm_orc.agentic.dispatch_event_substrate import DispatchEventSubstrate
from llm_orc.agentic.orchestrator_chunk import (
    Completion,
    ContentDelta,
    OrchestratorChunk,
)
from llm_orc.agentic.orchestrator_tool_dispatch import (
    InternalToolCall,
    ToolCallError,
    ToolCallResult,
    ToolCallSuccess,
)
from llm_orc.agentic.session_start import SessionContext

__all__ = [
    "DirectCompletionFallback",
    "DispatchFired",
    "DispatchPipeline",
    "DispatchPlan",
    "PlanEmitted",
    "ResponseSynthesizer",
    "RoutingPlanner",
    "SynthesizerCompleted",
    "ToolDispatcher",
    "plan_to_internal_tool_call",
]


@dataclass(frozen=True)
class DispatchPlan:
    """The routing-planner's parsed decision.

    ``action`` is a plain ``str`` (not a constrained Literal) because the
    planner reports model output faithfully — an injected ``"launch"``
    must be representable so the pipeline's validation stage can reject
    it (Spike ν E1). The pipeline, not the planner, decides whether the
    action is actionable.
    """

    action: str
    ensemble: str | None
    input: str | None
    rationale: str


# --- Ports (replaced by production modules in WP-B / WP-C) ----------------


class RoutingPlanner(Protocol):
    """Stage 1 — produces a :class:`DispatchPlan` from request content.

    Returns ``None`` when the planner output is empty or unparseable (the
    Spike ν A6 reliability mode); the pipeline treats ``None`` as a
    defined direct-completion path.
    """

    async def plan(self, *, request: str) -> DispatchPlan | None: ...


class ResponseSynthesizer(Protocol):
    """Stage 3 — composes the user-facing response from dispatch results.

    Yields response text chunks; the pipeline wraps each as a
    :class:`ContentDelta`. WP-A adapters yield the full response as one
    chunk; WP-C makes this token-streaming via the EnsembleExecutor's
    per-ensemble streaming surface.
    """

    def synthesize(
        self,
        *,
        original_request: str,
        dispatched: list[str],
        planned_but_not_run: list[str],
        dispatch_results: list[tuple[str, str]],
    ) -> AsyncIterator[str]: ...


class ToolDispatcher(Protocol):
    """Stage 2 — the existing Orchestrator Tool Dispatch surface."""

    async def dispatch(
        self, call: InternalToolCall, *, session_id: str = ""
    ) -> ToolCallResult: ...


# --- Pipeline-stage events (additive event vocabulary, ADR-027) -----------


@dataclass(frozen=True)
class PlanEmitted:
    """Stage-1 event: the routing-planner produced (or failed to produce) a plan."""

    dispatch_id: str | None
    plan: DispatchPlan | None


@dataclass(frozen=True)
class DispatchFired:
    """Stage-2 event: a capability ensemble was dispatched."""

    dispatch_id: str | None
    ensemble_name: str


@dataclass(frozen=True)
class SynthesizerCompleted:
    """Stage-3 event: the response synthesizer finished composing."""

    dispatch_id: str | None
    finish_reason: str


@dataclass(frozen=True)
class DirectCompletionFallback:
    """Direct-completion path taken instead of a capability dispatch.

    ``request_shape_category`` records why: ``no_capability_match`` (the
    planner emitted ``action: direct``), ``invalid_plan`` (the plan named
    a non-`{dispatch,direct}` action or an unregistered ensemble — Spike ν
    E1/E3), or ``unparseable_plan`` (empty/unparseable planner output —
    Spike ν A6). ADR-032 §"Operator-observable degradation signaling".
    """

    dispatch_id: str | None
    request_shape_category: str
    planner_rationale: str


# --- Plan → InternalToolCall adapter (pure function) ----------------------


def plan_to_internal_tool_call(
    plan: DispatchPlan, dispatch_id: str
) -> InternalToolCall:
    """Translate a dispatch plan into the existing dispatch surface's call.

    Per ADR-027 §Dispatch stage + system-design Finding 11 disposition:
    the adapter lives inside the pipeline as a pure function and reuses
    ``invoke_ensemble`` rather than re-implementing the dispatch contract.
    """
    return InternalToolCall(
        id=dispatch_id,
        name="invoke_ensemble",
        arguments={"ensemble_name": plan.ensemble, "input": plan.input},
    )


def _direct_category(
    plan: DispatchPlan | None, capability_names: frozenset[str]
) -> str:
    """Classify why a request took the direct-completion path.

    ``unparseable_plan`` — no plan (empty/unparseable planner output, A6).
    ``no_capability_match`` — planner emitted ``action: direct``.
    ``invalid_plan`` — action not in ``{dispatch, direct}`` or ensemble not
    in the registered capability set (E1/E3).
    """
    if plan is None:
        return "unparseable_plan"
    if plan.action == "direct":
        return "no_capability_match"
    return "invalid_plan"


def _result_text(result: ToolCallResult) -> str:
    """Extract the human-readable deliverable from a dispatch result.

    Successful dispatches carry the deliverable on ``envelope.primary``
    (ADR-024). A typed error is rendered as text so the synthesizer can
    report honestly per ADR-029 Rule 2.
    """
    if isinstance(result, ToolCallSuccess):
        if result.envelope is not None:
            return result.envelope.primary
        return str(result.content)
    if isinstance(result, ToolCallError):
        return f"[dispatch error: {result.kind}] {result.reason}"
    return str(result)


class DispatchPipeline:
    """Framework-driven plan → dispatch → synthesize chat-completions caller."""

    def __init__(
        self,
        *,
        planner: RoutingPlanner,
        synthesizer: ResponseSynthesizer,
        tool_dispatch: ToolDispatcher,
        capability_names: frozenset[str],
        event_substrate: DispatchEventSubstrate | None = None,
    ) -> None:
        self._planner = planner
        self._synthesizer = synthesizer
        self._tool_dispatch = tool_dispatch
        self._capability_names = capability_names
        self._event_substrate = event_substrate

    async def run(self, context: SessionContext) -> AsyncIterator[OrchestratorChunk]:
        """Drive the three-stage pipeline for one chat-completions request."""
        request = _extract_request(context)
        session_id = context.state.identity.value
        dispatch_id = self._new_dispatch_id(session_id)

        plan = await self._planner.plan(request=request)
        self._emit(PlanEmitted(dispatch_id=dispatch_id, plan=plan))

        if (
            plan is not None
            and plan.action == "dispatch"
            and plan.ensemble in self._capability_names
        ):
            call = plan_to_internal_tool_call(plan, dispatch_id)
            result = await self._tool_dispatch.dispatch(call, session_id=session_id)
            self._emit(
                DispatchFired(dispatch_id=dispatch_id, ensemble_name=plan.ensemble)
            )
            async for chunk in self._synthesize(
                request=request,
                dispatch_id=dispatch_id,
                dispatched=[plan.ensemble],
                dispatch_results=[(plan.ensemble, _result_text(result))],
            ):
                yield chunk
            return

        # Direct-completion path. The category distinguishes a clean
        # no-capability-match (planner said "direct") from the two Spike ν
        # degradation modes the pipeline structurally backstops: an
        # invalid plan (bad action or unregistered ensemble — E1/E3) and
        # an unparseable/empty plan (A6).
        category = _direct_category(plan, self._capability_names)
        self._emit(
            DirectCompletionFallback(
                dispatch_id=dispatch_id,
                request_shape_category=category,
                planner_rationale=plan.rationale if plan is not None else "",
            )
        )
        async for chunk in self._synthesize(
            request=request,
            dispatch_id=dispatch_id,
            dispatched=[],
            dispatch_results=[],
        ):
            yield chunk

    async def _synthesize(
        self,
        *,
        request: str,
        dispatch_id: str,
        dispatched: list[str],
        dispatch_results: list[tuple[str, str]],
    ) -> AsyncIterator[OrchestratorChunk]:
        async for text in self._synthesizer.synthesize(
            original_request=request,
            dispatched=dispatched,
            planned_but_not_run=[],
            dispatch_results=dispatch_results,
        ):
            yield ContentDelta(content=text)
        self._emit(SynthesizerCompleted(dispatch_id=dispatch_id, finish_reason="stop"))
        yield Completion(finish_reason="stop")

    def _new_dispatch_id(self, session_id: str) -> str:
        if self._event_substrate is not None:
            return self._event_substrate.new_dispatch_id(session_id)
        return f"{session_id}-pipeline"

    def _emit(self, event: object) -> None:
        if self._event_substrate is not None:
            self._event_substrate.emit(event)


def _extract_request(context: SessionContext) -> str:
    """Compose the planner/synthesizer request string from session messages.

    The latest user message is the request; for WP-A the pipeline passes
    the last user message content. Multi-turn history handling (native
    ``messages[]`` per ADR-029) is WP-C synthesizer work.
    """
    for message in reversed(context.messages):
        if message.role == "user" and message.content:
            return message.content
    return ""
