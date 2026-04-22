"""Orchestrator Tool Dispatch — closed five-tool surface (ADR-003).

Per ``docs/agentic-serving/system-design.md`` §Orchestrator Tool
Dispatch (L2) and §Integration Contracts (Orchestrator Runtime →
Orchestrator Tool Dispatch, Orchestrator Tool Dispatch → Result
Summarizer Harness, Orchestrator Tool Dispatch → Autonomy Policy). The
Runtime calls ``dispatch(call)`` with an ``InternalToolCall``; this
module routes by name through the closed set or returns a typed tool
error for names outside the set.

The closed-set property is structurally enforced: the five tool names
live in ``TOOL_NAMES`` and correspond to five async methods on this
class. FC-5 checks the count of public async methods whose names are
in ``TOOL_NAMES`` — exactly five.

Per system design Amendment #3, Tool Dispatch also interposes the
Result Summarizer Harness on ``invoke_ensemble``'s return path.
Unsummarized ensemble results never reach the Orchestrator Runtime
unless the invoked ensemble's ``raw_output`` flag is set (ADR-004
escape hatch). The Runtime is unaware of the Harness — summarization
is a Tool-Dispatch-side concern (FC-4, FC-8).

WP-E interposes the Autonomy Policy gate between the unknown-tool
filter and tool routing (FC-11). Unknown tool names short-circuit
before Autonomy is consulted, so AS-6 closure stays structural. For
valid tool names the gate decides Allow (optionally with visibility
events) or Deny (returning a typed ``denied_by_autonomy`` error).
Decision events ride back on the result via the ``events`` tuple.

WP-C wires ``invoke_ensemble`` and ``list_ensembles`` by delegating to
the project's existing ``OrchestraService`` (invoke + read_ensembles).
This avoids a parallel find-and-execute code path — the orchestrator
tool surface is a thin adapter on the existing ensemble-operations
facade, not a re-implementation.

``compose_ensemble`` (WP-G), ``query_knowledge`` (WP-I), and
``record_outcome`` (WP-I) return typed not-yet-wired errors so the
closed-set property holds from WP-C onward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from llm_orc.agentic.autonomy_policy import Allow, AutonomyDecision, Deny
from llm_orc.agentic.orchestrator_chunk import VisibilityEvent
from llm_orc.agentic.result_summarizer_harness import (
    RawOutputPassthrough,
    ResultSummarizerHarness,
    SummarizationFailure,
    SummarizationSuccess,
)

TOOL_NAMES: frozenset[str] = frozenset(
    {
        "invoke_ensemble",
        "compose_ensemble",
        "list_ensembles",
        "query_knowledge",
        "record_outcome",
    }
)
"""The closed tool set committed by ADR-003."""


ToolErrorKind = Literal[
    "unknown_tool",
    "not_yet_wired",
    "invocation_failed",
    "invalid_arguments",
    "summarization_failed",
    "denied_by_autonomy",
]


@dataclass(frozen=True)
class InternalToolCall:
    """A tool call emitted by the orchestrator LLM.

    ``arguments`` is pre-parsed by the Runtime — the orchestrator LLM
    emits JSON-string arguments per OpenAI convention; the Runtime
    parses before handing off so JSON handling is centralized.
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolCallSuccess:
    """A successful tool result surfaced to the orchestrator's context.

    ``events`` carries visibility events produced by the dispatch-time
    Autonomy Policy decision. The Runtime forwards them as
    :class:`VisibilityEvent` chunks so the Serving Layer can surface
    composition-related narration to the tool user (ADR-008 tightened
    levels). An empty tuple is the baseline — silent operation.
    """

    id: str
    name: str
    content: Any
    events: tuple[VisibilityEvent, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ToolCallError:
    """A failed tool call surfaced as an observation, not an exception.

    The ReAct loop continues with this result — the LLM sees the
    error, may adjust its plan, and emits the next tool call.

    ``events`` mirrors the ``ToolCallSuccess`` field so visibility is
    surfaced consistently regardless of the dispatch outcome — an
    Autonomy Policy decision may attach events to either path.
    """

    id: str
    name: str
    kind: ToolErrorKind
    reason: str
    events: tuple[VisibilityEvent, ...] = field(default_factory=tuple)


ToolCallResult = ToolCallSuccess | ToolCallError


class EnsembleOperations(Protocol):
    """Narrow facade over ensemble invocation and listing.

    ``OrchestraService`` satisfies this structurally; tests pass a
    handwritten double. The Protocol names only the two operations
    the orchestrator tool surface delegates to, so Tool Dispatch is
    decoupled from the full service construction surface.
    """

    async def invoke(self, arguments: dict[str, Any]) -> dict[str, Any]: ...

    async def read_ensembles(self) -> list[dict[str, Any]]: ...


class AutonomyGate(Protocol):
    """Minimum Autonomy Policy surface Tool Dispatch consults.

    :class:`~llm_orc.agentic.autonomy_policy.AutonomyPolicy` satisfies
    this structurally. The Protocol lets tests substitute a recording
    double that scripts a Deny decision — Phase 1 policy code never
    produces Deny, but the dispatch-side Deny handling must be covered.
    """

    def decide(
        self, *, tool_name: str, arguments: dict[str, Any]
    ) -> AutonomyDecision: ...


class OrchestratorToolDispatch:
    """Closed five-tool dispatch surface (ADR-003, FC-5)."""

    def __init__(
        self,
        *,
        operations: EnsembleOperations,
        harness: ResultSummarizerHarness,
        autonomy_policy: AutonomyGate,
    ) -> None:
        self._operations = operations
        self._harness = harness
        self._autonomy_policy = autonomy_policy

    async def dispatch(self, call: InternalToolCall) -> ToolCallResult:
        """Route a tool call through the unknown-tool filter, gate, then routing.

        Three steps, in order:

        1. **Unknown-tool filter.** Names outside ``TOOL_NAMES`` return a typed
           ``unknown_tool`` error without consulting the gate — AS-6 closure
           lives here, not in Autonomy.
        2. **Autonomy gate.** The policy decides Allow (optionally with
           visibility events) or Deny. A Deny short-circuits with a typed
           ``denied_by_autonomy`` error; the tool method is not called.
        3. **Tool routing.** Match-case dispatches to the five committed
           methods; decision events attach to the returned result.
        """
        if call.name not in TOOL_NAMES:
            return ToolCallError(
                id=call.id,
                name=call.name,
                kind="unknown_tool",
                reason=(
                    f"tool '{call.name}' is not in the orchestrator's "
                    "committed tool set"
                ),
            )

        decision = self._autonomy_policy.decide(
            tool_name=call.name, arguments=call.arguments
        )
        if isinstance(decision, Deny):
            return ToolCallError(
                id=call.id,
                name=call.name,
                kind="denied_by_autonomy",
                reason=decision.reason,
            )

        events = decision.events if isinstance(decision, Allow) else ()
        return _with_events(await self._route(call), events)

    async def _route(self, call: InternalToolCall) -> ToolCallResult:
        """Dispatch a committed tool name to its method.

        Match-case makes the five committed tools visible at the dispatch
        site (FF #66 — chosen over ``getattr`` to keep return-type tracking
        through mypy). The ``case _`` arm is unreachable because
        ``dispatch`` filters unknown names above; the assertion guards
        against a future edit that bypasses the filter.
        """
        match call.name:
            case "invoke_ensemble":
                return await self.invoke_ensemble(call.id, call.arguments)
            case "compose_ensemble":
                return await self.compose_ensemble(call.id, call.arguments)
            case "list_ensembles":
                return await self.list_ensembles(call.id, call.arguments)
            case "query_knowledge":
                return await self.query_knowledge(call.id, call.arguments)
            case "record_outcome":
                return await self.record_outcome(call.id, call.arguments)
            case _:  # pragma: no cover — TOOL_NAMES filter makes this unreachable
                raise AssertionError(
                    f"_route received unfiltered tool name: {call.name!r}"
                )

    async def invoke_ensemble(
        self, id_: str, arguments: dict[str, Any]
    ) -> ToolCallResult:
        """Resolve an ensemble by name, execute, then interpose the Harness.

        Per Amendment #3: on every successful invocation, Tool Dispatch
        calls :class:`ResultSummarizerHarness` with the raw result and
        the invoked ensemble's ``raw_output`` flag. The Runtime never
        sees unsummarized output unless the ensemble explicitly opts
        into the ADR-004 escape hatch.
        """
        name = arguments.get("name")
        input_data = arguments.get("input", "")
        if not isinstance(name, str) or not name:
            return ToolCallError(
                id=id_,
                name="invoke_ensemble",
                kind="invalid_arguments",
                reason="invoke_ensemble requires 'name' (non-empty string)",
            )
        if not isinstance(input_data, str):
            return ToolCallError(
                id=id_,
                name="invoke_ensemble",
                kind="invalid_arguments",
                reason="invoke_ensemble 'input' must be a string",
            )

        try:
            result = await self._operations.invoke(
                {"ensemble_name": name, "input": input_data}
            )
        except ValueError as exc:
            return ToolCallError(
                id=id_,
                name="invoke_ensemble",
                kind="invocation_failed",
                reason=str(exc),
            )

        raw_output = bool(result.get("raw_output", False))
        summarization = await self._harness.summarize(result, raw_output=raw_output)
        match summarization:
            case SummarizationSuccess(summary=summary):
                return ToolCallSuccess(
                    id=id_,
                    name="invoke_ensemble",
                    content={"summary": summary},
                )
            case RawOutputPassthrough(content=passthrough):
                return ToolCallSuccess(
                    id=id_, name="invoke_ensemble", content=passthrough
                )
            case SummarizationFailure(reason=reason):
                return ToolCallError(
                    id=id_,
                    name="invoke_ensemble",
                    kind="summarization_failed",
                    reason=reason,
                )

    async def compose_ensemble(
        self, id_: str, arguments: dict[str, Any]
    ) -> ToolCallResult:
        """Compose a new ensemble (WP-G)."""
        del arguments
        return ToolCallError(
            id=id_,
            name="compose_ensemble",
            kind="not_yet_wired",
            reason="compose_ensemble lands in WP-G (Composition Validator)",
        )

    async def list_ensembles(
        self, id_: str, arguments: dict[str, Any]
    ) -> ToolCallResult:
        """Enumerate library ensembles via ``OrchestraService.read_ensembles``."""
        del arguments
        entries = await self._operations.read_ensembles()
        return ToolCallSuccess(id=id_, name="list_ensembles", content=entries)

    async def query_knowledge(
        self, id_: str, arguments: dict[str, Any]
    ) -> ToolCallResult:
        """Query the knowledge graph (WP-I Plexus Adapter)."""
        del arguments
        return ToolCallError(
            id=id_,
            name="query_knowledge",
            kind="not_yet_wired",
            reason="query_knowledge lands in WP-I (Plexus Adapter)",
        )

    async def record_outcome(
        self, id_: str, arguments: dict[str, Any]
    ) -> ToolCallResult:
        """Record a routing decision or outcome (WP-I Plexus Adapter)."""
        del arguments
        return ToolCallError(
            id=id_,
            name="record_outcome",
            kind="not_yet_wired",
            reason="record_outcome lands in WP-I (Plexus Adapter)",
        )


def _with_events(
    result: ToolCallResult, events: tuple[VisibilityEvent, ...]
) -> ToolCallResult:
    """Return a copy of ``result`` carrying ``events``.

    Branches explicitly on the two result variants rather than using
    :func:`dataclasses.replace` so the union stays type-narrowed through
    mypy strict. No-op when ``events`` is empty (the Allow-without-events
    path on every dispatch), keeping the common case allocation-free.
    """
    if not events:
        return result
    if isinstance(result, ToolCallSuccess):
        return ToolCallSuccess(
            id=result.id,
            name=result.name,
            content=result.content,
            events=events,
        )
    return ToolCallError(
        id=result.id,
        name=result.name,
        kind=result.kind,
        reason=result.reason,
        events=events,
    )
