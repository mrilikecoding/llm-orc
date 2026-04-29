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

import logging
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from llm_orc.agentic.autonomy_policy import Allow, AutonomyDecision, Deny
from llm_orc.agentic.calibration_gate import CalibrationGate, QualitySignal
from llm_orc.agentic.composition_validator import (
    CompositionAccepted,
    CompositionOutcome,
    CompositionRejected,
    CompositionRequest,
    EnsembleWriteError,
    LocalEnsembleWriter,
)
from llm_orc.agentic.orchestrator_chunk import VisibilityEvent
from llm_orc.agentic.result_summarizer_harness import (
    RawOutputPassthrough,
    ResultSummarizerHarness,
    SummarizationFailure,
    SummarizationSuccess,
)

_logger = logging.getLogger(__name__)
"""Operator-side log surface for tool-dispatch outcomes.

Emits an INFO-level result line on every dispatch so the operator can
diagnose orchestrator behavior. Error paths include the full ``reason``
field so misconfiguration (e.g., a summarizer profile pointing at an
Ollama model that isn't pulled — research log 2026-04-28, DIAG-1)
surfaces actionably without spike-only instrumentation. Success paths
log the kind only.
"""

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


class CompositionGate(Protocol):
    """Minimum Composition Validator surface Tool Dispatch consults.

    :class:`~llm_orc.agentic.composition_validator.CompositionValidator`
    satisfies this structurally. The Protocol lets tests substitute a
    scripted double that returns a canned :class:`CompositionOutcome`
    — the production validator's deeper dependency graph (primitive
    registry, depth limit) stays out of the dispatch-level tests.
    """

    def validate(self, request: CompositionRequest) -> CompositionOutcome: ...


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


class PlexusAccess(Protocol):
    """Minimum Plexus Adapter surface Tool Dispatch consults.

    :class:`~llm_orc.agentic.plexus_adapter.PlexusAdapter` satisfies
    this structurally. ``query`` services ``query_knowledge`` and
    ``record`` services ``record_outcome``. WP-I supplies the no-op
    Adapter; WP-K replaces the bodies with real plexus MCP calls.
    """

    async def query(self, arguments: dict[str, Any]) -> dict[str, Any]: ...

    async def record(self, arguments: dict[str, Any]) -> dict[str, Any]: ...


class OrchestratorToolDispatch:
    """Closed five-tool dispatch surface (ADR-003, FC-5)."""

    def __init__(
        self,
        *,
        operations: EnsembleOperations,
        harness: ResultSummarizerHarness,
        autonomy_policy: AutonomyGate,
        composition_validator: CompositionGate,
        local_ensemble_writer: LocalEnsembleWriter,
        calibration_gate: CalibrationGate | None = None,
        plexus_adapter: PlexusAccess | None = None,
    ) -> None:
        self._operations = operations
        self._harness = harness
        self._autonomy_policy = autonomy_policy
        self._composition_validator = composition_validator
        self._local_ensemble_writer = local_ensemble_writer
        self._calibration_gate = calibration_gate
        self._plexus_adapter = plexus_adapter

    async def dispatch(
        self, call: InternalToolCall, *, session_id: str = ""
    ) -> ToolCallResult:
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

        ``session_id`` identifies the Session for per-session state the
        dispatch consults (Calibration Gate in WP-H). Defaults to the
        empty string so existing call sites without session context (or
        tests that do not exercise calibration) continue to work — the
        Calibration Gate stores records under whatever string it is
        handed.
        """
        if call.name not in TOOL_NAMES:
            error = ToolCallError(
                id=call.id,
                name=call.name,
                kind="unknown_tool",
                reason=(
                    f"tool '{call.name}' is not in the orchestrator's "
                    "committed tool set"
                ),
            )
            _log_dispatch_result(call.name, error)
            return error

        decision = self._autonomy_policy.decide(
            tool_name=call.name, arguments=call.arguments
        )
        if isinstance(decision, Deny):
            error = ToolCallError(
                id=call.id,
                name=call.name,
                kind="denied_by_autonomy",
                reason=decision.reason,
            )
            _log_dispatch_result(call.name, error)
            return error

        events = decision.events if isinstance(decision, Allow) else ()
        result = _with_events(await self._route(call, session_id), events)
        _log_dispatch_result(call.name, result)
        return result

    async def _route(self, call: InternalToolCall, session_id: str) -> ToolCallResult:
        """Dispatch a committed tool name to its method.

        Match-case makes the five committed tools visible at the dispatch
        site (FF #66 — chosen over ``getattr`` to keep return-type tracking
        through mypy). The ``case _`` arm is unreachable because
        ``dispatch`` filters unknown names above; the assertion guards
        against a future edit that bypasses the filter.
        """
        match call.name:
            case "invoke_ensemble":
                return await self.invoke_ensemble(call.id, call.arguments, session_id)
            case "compose_ensemble":
                return await self.compose_ensemble(call.id, call.arguments, session_id)
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
        self, id_: str, arguments: dict[str, Any], session_id: str = ""
    ) -> ToolCallResult:
        """Resolve an ensemble by name, execute, then interpose the Harness.

        Per Amendment #3: on every successful invocation, Tool Dispatch
        calls :class:`ResultSummarizerHarness` with the raw result and
        the invoked ensemble's ``raw_output`` flag. The Runtime never
        sees unsummarized output unless the ensemble explicitly opts
        into the ADR-004 escape hatch.

        Per ADR-007 (WP-H): before summarization, the raw result is
        handed to the Calibration Gate when one is configured. The gate
        is consulted for every ``invoke_ensemble`` — it is a no-op for
        trusted or untracked ensembles, and runs the checker for
        composed ensembles still in calibration. Calibration failures
        **do not** fail invocation (ADR-007 clause 2) — exceptions are
        swallowed so the orchestrator's loop continues unimpeded.
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

        await self._calibration_check_safe(
            session_id=session_id, ensemble_name=name, raw_result=result
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
        self, id_: str, arguments: dict[str, Any], session_id: str = ""
    ) -> ToolCallResult:
        """Compose a new ensemble via the Composition Validator (WP-G).

        Parses the LLM-emitted arguments into a
        :class:`CompositionRequest`, delegates validation to
        :class:`CompositionValidator`, and on accept hands the
        validated :class:`EnsembleConfig` to the local-tier writer.
        Validation failure and write failure both surface as
        ``ToolCallError(kind="invocation_failed")`` so the ReAct loop
        continues with the error as an observation. AS-2 (no partial
        state on validation failure) is structurally enforced — the
        writer is only reached after :class:`CompositionAccepted`.

        Per ADR-007 (WP-H): on successful write, the newly composed
        ensemble is registered with the Calibration Gate (when
        configured) so subsequent ``invoke_ensemble`` calls on the same
        name run the Quality Signal check until the ensemble clears
        calibration. Registration happens only after the writer
        succeeds — a rejected composition never enters calibration.
        """
        name = arguments.get("name")
        description = arguments.get("description", "")
        agents = arguments.get("agents", [])
        raw_output = bool(arguments.get("raw_output", False))
        if not isinstance(name, str) or not name:
            return ToolCallError(
                id=id_,
                name="compose_ensemble",
                kind="invalid_arguments",
                reason="compose_ensemble requires 'name' (non-empty string)",
            )
        if not isinstance(description, str):
            return ToolCallError(
                id=id_,
                name="compose_ensemble",
                kind="invalid_arguments",
                reason="compose_ensemble 'description' must be a string",
            )
        if not isinstance(agents, list) or not all(isinstance(a, dict) for a in agents):
            return ToolCallError(
                id=id_,
                name="compose_ensemble",
                kind="invalid_arguments",
                reason="compose_ensemble 'agents' must be a list of objects",
            )

        request = CompositionRequest(
            name=name,
            description=description,
            agents=agents,
            raw_output=raw_output,
        )
        outcome = self._composition_validator.validate(request)
        if isinstance(outcome, CompositionRejected):
            return ToolCallError(
                id=id_,
                name="compose_ensemble",
                kind="invocation_failed",
                reason=outcome.reason,
            )
        accepted: CompositionAccepted = outcome
        try:
            path = self._local_ensemble_writer.write(accepted.config)
        except EnsembleWriteError as exc:
            return ToolCallError(
                id=id_,
                name="compose_ensemble",
                kind="invocation_failed",
                reason=str(exc),
            )
        if self._calibration_gate is not None:
            self._calibration_gate.mark_composed(
                session_id=session_id, ensemble_name=accepted.config.name
            )
        return ToolCallSuccess(
            id=id_,
            name="compose_ensemble",
            content={"name": accepted.config.name, "path": path},
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
        """Query the knowledge graph via the Plexus Adapter (WP-I).

        Delegates to the configured Adapter when present. Production
        always wires one; the absence path is a defensive fallback for
        tests that omit it. The Adapter's no-op fallback returns a
        well-formed empty result when Plexus is absent (AS-8 / FC-7);
        WP-K replaces the body with real plexus MCP query semantics
        without changing this call site.
        """
        if self._plexus_adapter is None:
            return ToolCallError(
                id=id_,
                name="query_knowledge",
                kind="not_yet_wired",
                reason="query_knowledge requires a Plexus Adapter",
            )
        result = await self._plexus_adapter.query(arguments)
        return ToolCallSuccess(id=id_, name="query_knowledge", content=result)

    async def record_outcome(
        self, id_: str, arguments: dict[str, Any]
    ) -> ToolCallResult:
        """Record a routing decision or outcome via the Plexus Adapter (WP-I).

        Delegates to the configured Adapter when present. The Adapter's
        no-op fallback returns acknowledgement immediately, satisfying
        ADR-009's "acknowledgement promptly" contract for the
        Plexus-absent case. WP-K's real implementation writes
        asynchronously and still returns an immediate ack (eventual
        consistency at the enrichment layer).
        """
        if self._plexus_adapter is None:
            return ToolCallError(
                id=id_,
                name="record_outcome",
                kind="not_yet_wired",
                reason="record_outcome requires a Plexus Adapter",
            )
        result = await self._plexus_adapter.record(arguments)
        return ToolCallSuccess(id=id_, name="record_outcome", content=result)

    async def _calibration_check_safe(
        self, *, session_id: str, ensemble_name: str, raw_result: dict[str, Any]
    ) -> QualitySignal | None:
        """Consult the Calibration Gate, swallowing any error.

        ADR-007 clause 2: the calibration check does not prevent
        invocation. A checker-ensemble crash, summarizer backoff, or
        any other raise from the gate surfaces here as ``None`` rather
        than as a tool error. The invocation's raw result still flows
        to the Summarizer Harness and back to the orchestrator.
        """
        if self._calibration_gate is None:
            return None
        try:
            return await self._calibration_gate.check_and_record(
                session_id=session_id,
                ensemble_name=ensemble_name,
                raw_result=raw_result,
            )
        except Exception:  # noqa: BLE001 — ADR-007 clause 2: never block invocation
            return None


def _log_dispatch_result(tool_name: str, result: ToolCallResult) -> None:
    """Emit an INFO-level result line for a tool dispatch.

    Success path logs ``name`` and ``kind=success``. Error path includes
    the full ``reason`` so misconfiguration surfaces actionably in the
    operator log (research log 2026-04-28, DIAG-1: surfacing the actual
    reason was the unblock for diagnosing summarization_failed root
    cause).
    """
    if isinstance(result, ToolCallSuccess):
        _logger.info("tool dispatch: result name=%s kind=success", tool_name)
        return
    _logger.info(
        "tool dispatch: result name=%s kind=error:%s reason=%s",
        tool_name,
        result.kind,
        result.reason,
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
