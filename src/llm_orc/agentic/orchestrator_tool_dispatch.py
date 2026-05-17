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

import dataclasses
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from llm_orc.agentic.autonomy_policy import Allow, AutonomyDecision, Deny
from llm_orc.agentic.calibration_gate import (
    CalibrationGate,
    CalibrationVerdict,
    CalibrationVerdictEvent,
    DispatchContext,
    QualitySignal,
)
from llm_orc.agentic.composition_validator import (
    CompositionAccepted,
    CompositionOutcome,
    CompositionRejected,
    CompositionRequest,
    EnsembleWriteError,
    LocalEnsembleWriter,
)
from llm_orc.agentic.dispatch_envelope import DispatchEnvelope
from llm_orc.agentic.dispatch_event_substrate import (
    DispatchEventSubstrate,
    DispatchTiming,
    ExitStatus,
)
from llm_orc.agentic.orchestrator_chunk import VisibilityEvent
from llm_orc.agentic.result_summarizer_harness import (
    RawOutputPassthrough,
    ResultSummarizerHarness,
    SummarizationFailure,
    SummarizationSuccess,
)
from llm_orc.agentic.tier_router import (
    EscalationBypassError,
    MissingSkillMetadataError,
    TierRouter,
    TierSelection,
)
from llm_orc.agentic.tier_router_audit import TierEscalationAuditor
from llm_orc.agentic.tool_call_validation_guard import (
    PhantomToolCallError,
    scan_response_for_phantom_claims,
)

__all__ = [
    "InternalToolCall",
    "OrchestratorToolDispatch",
    "OutputSchemaReader",
    "PhantomToolCallError",
    "TOOL_NAMES",
    "ToolCallError",
    "ToolCallResult",
    "ToolCallSuccess",
    "ToolErrorKind",
]

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
    "escalation_bypass",
    "missing_skill_metadata",
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

    ``dispatch_id`` is the ADR-023 correlation identifier when this
    result is the return of ``invoke_ensemble`` with an event
    substrate configured — ``None`` for the other four tools (which
    do not allocate a dispatch_id) and for legacy / no-substrate
    paths. The Runtime uses this to query the Orchestrator-Context
    Event Sink at the next turn boundary for a structured observation.

    ``envelope`` is the ADR-024 typed :class:`DispatchEnvelope` when
    this result is the return of ``invoke_ensemble`` on a successful
    dispatch — ``None`` for the other four tools and for legacy
    no-substrate ``invoke_ensemble`` call sites. Same additive
    pattern as ``dispatch_id``: the closed-five-tool
    :class:`ToolCallResult` union remains the uniform dispatch return
    type; the envelope is the typed structural contract on the
    ``invoke_ensemble`` slot per ADR-024 (Cycle 6 WP-D).
    """

    id: str
    name: str
    content: Any
    events: tuple[VisibilityEvent, ...] = field(default_factory=tuple)
    dispatch_id: str | None = None
    envelope: DispatchEnvelope | None = None


@dataclass(frozen=True)
class ToolCallError:
    """A failed tool call surfaced as an observation, not an exception.

    The ReAct loop continues with this result — the LLM sees the
    error, may adjust its plan, and emits the next tool call.

    ``events`` mirrors the ``ToolCallSuccess`` field so visibility is
    surfaced consistently regardless of the dispatch outcome — an
    Autonomy Policy decision may attach events to either path.

    ``dispatch_id`` is the ADR-023 correlation identifier when this
    error is the return of ``invoke_ensemble`` after dispatch_id
    allocation (e.g., post-tier-selection invocation failure or
    summarization failure). Pre-dispatch errors (argument validation,
    no-such-ensemble) leave it ``None`` — no dispatch occurred.
    """

    id: str
    name: str
    kind: ToolErrorKind
    reason: str
    events: tuple[VisibilityEvent, ...] = field(default_factory=tuple)
    dispatch_id: str | None = None


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


class ToolCallEmitLogger(Protocol):
    """Minimum tool-call-emit logging surface Tool Dispatch consults.

    Cycle 6 WP-B piece 4 (ADR-023 §"Liveness signals"). The
    :class:`~llm_orc.agentic.operator_terminal_event_sink.OperatorTerminalEventSink`
    satisfies this structurally via :meth:`emit_tool_call_log`. The
    Protocol stays L2-local so Tool Dispatch does not import the L3
    sink — FC-4 layering preserved (the wiring at the serve layer
    composes the two).
    """

    def emit_tool_call_log(self, *, tool_name: str, dispatch_id: str) -> None: ...


class OutputSchemaReader(Protocol):
    """Minimum ``output_schema:`` lookup surface Tool Dispatch consults.

    Cycle 6 WP-D (ADR-024). Production wiring at the Serving Layer (L3)
    constructs a reader backed by ``EnsembleLoader.find_ensemble`` (or
    ``OrchestraService.find_ensemble_by_name``) and passes it at
    construction time. The reader returns the operator-authored
    ``output_schema:`` dict for the named ensemble, or ``None`` when
    the field is absent. The Protocol stays L2-local — Tool Dispatch
    does not import the L0 config module directly, mirroring the
    pattern :class:`~llm_orc.agentic.tier_router.TopazSkillReader`
    establishes for tier-routing metadata.

    The schema lookup is advisory at dispatch time per ADR-024 §"Out
    of scope" — when a schema is declared, ``invoke_ensemble`` attempts
    to populate ``envelope.structured`` from the synthesizer's response
    via JSON parse; when the response is not JSON, ``structured``
    stays ``None`` without raising.
    """

    def output_schema_for(self, ensemble_name: str) -> dict[str, Any] | None: ...


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
        tool_call_validation_patterns: tuple[str, ...] = (),
        tier_router: TierRouter | None = None,
        tier_router_audit: TierEscalationAuditor | None = None,
        event_substrate: DispatchEventSubstrate | None = None,
        tool_call_emit_logger: ToolCallEmitLogger | None = None,
        output_schema_reader: OutputSchemaReader | None = None,
    ) -> None:
        if tier_router is not None and calibration_gate is None:
            raise ValueError(
                "tier_router requires calibration_gate per ADR-015 §Router "
                "logic — the router consumes the verdict the gate produces"
            )
        if tier_router_audit is not None and tier_router is None:
            raise ValueError(
                "tier_router_audit requires tier_router per ADR-018 — the "
                "(d)-analog audit observes the verdict→router edge that "
                "only exists when the router is configured"
            )
        self._operations = operations
        self._harness = harness
        self._autonomy_policy = autonomy_policy
        self._composition_validator = composition_validator
        self._local_ensemble_writer = local_ensemble_writer
        self._calibration_gate = calibration_gate
        self._plexus_adapter = plexus_adapter
        self._tool_call_validation_patterns = tool_call_validation_patterns
        """Operator-extension patterns appended to ADR-017's default
        assertion-pattern set in :meth:`validate_response`.
        Empty tuple = scan defaults only."""
        self._tier_router = tier_router
        """Per-role tier-escalation router (WP-G4-1, ADR-015).

        When configured, every ``invoke_ensemble`` consumes the
        Calibration Gate's verdict via the router and selects a per-
        skill Model Profile for the dispatch. ``None`` preserves the
        pre-Cycle-4 dispatch path for existing tests and call sites
        that have not been migrated. Construction requires that
        ``calibration_gate`` is also configured when ``tier_router``
        is — the router cannot fire without the verdict producer.
        """
        self._tier_router_audit = tier_router_audit
        """(d)-analog audit dispatch on the verdict→router edge
        (WP-G4-2, ADR-018).

        When configured, every router consultation records the
        consumed verdict (and the selected tier, when the router
        produced one) into the auditor's window; outcomes record
        after ``_operations.invoke``. The auditor's fail-safe state
        overrides verdict-driven tier selection under severe drift
        (route-all-to-escalated until operator review). Requires
        ``tier_router`` to be configured (the audit observes the
        router's interposition point).
        """
        self._event_substrate = event_substrate
        """Dispatch Event Substrate (Cycle 6 WP-A, ADR-023).

        When configured, ``invoke_ensemble`` allocates a ``dispatch_id``
        at entry and emits ``DispatchTiming(start)`` before tier
        selection and ``DispatchTiming(end)`` after the harness (or
        substrate-write per ADR-025 when WP-E ships). The substrate
        fans these events out to registered sinks (operator-terminal
        per WP-B; orchestrator-context per WP-C). ``None`` preserves
        the pre-Cycle-6 dispatch path for existing tests and call
        sites that have not been migrated.
        """
        self._tool_call_emit_logger = tool_call_emit_logger
        """Tool-call-emit liveness logger (Cycle 6 WP-B piece 4, ADR-023).

        When configured alongside ``event_substrate``, the logger fires
        ``emit_tool_call_log(tool_name="invoke_ensemble", dispatch_id=...)``
        between :meth:`new_dispatch_id` allocation and
        ``DispatchTiming(start)`` emission — the chronological-ordering
        property FC-23 verifies. Argument-validation failures and the
        no-substrate path both skip the call (no dispatch_id exists in
        either case). ``None`` preserves the pre-piece-4 dispatch path.
        """
        self._output_schema_reader = output_schema_reader
        """Per-ensemble ``output_schema:`` lookup (Cycle 6 WP-D, ADR-024).

        When configured, ``invoke_ensemble``'s envelope-construction
        step consults the reader for the dispatched ensemble's
        declared schema. A non-``None`` schema triggers advisory
        JSON-parse of the synthesizer's response; on parse success the
        parsed payload becomes ``envelope.structured``. ``None`` (no
        reader configured) and ``None`` (reader returns no schema for
        the ensemble) both leave ``envelope.structured`` as ``None``
        without raising.
        """

    def validate_response(
        self,
        response_text: str,
        tool_call_names: tuple[str, ...],
        *,
        session_id: str = "",
    ) -> PhantomToolCallError | None:
        """Run ADR-017 structural validation on an orchestrator response.

        Per system-design.agents.md §Orchestrator Tool Dispatch's
        post-Cycle 4 interposition order: this is step (1) on every
        ``invoke_ensemble`` (and other tool dispatches). Scans
        ``response_text`` against the union of
        :data:`DEFAULT_ASSERTION_PATTERNS` and the operator-supplied
        ``tool_call_validation_patterns`` provided at construction
        time. Returns the typed :class:`PhantomToolCallError` on
        mismatch (prose claim with zero tool-call structures); ``None``
        on pass (any tool-call structure emitted, or no assertion
        pattern matched).
        """
        return scan_response_for_phantom_claims(
            response_text,
            tool_call_names,
            dispatch_context={"session_id": session_id},
            extra_patterns=self._tool_call_validation_patterns,
        )

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

        Per ADR-023 (Cycle 6 WP-A): when an event substrate is configured,
        a ``dispatch_id`` is allocated at entry, ``DispatchTiming(start)``
        is emitted before tier selection, and ``DispatchTiming(end)`` is
        emitted in a ``finally`` block so the operator-terminal and
        orchestrator-context destinations observe every dispatch
        regardless of exit path.
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

        # ADR-023 WP-A: allocate dispatch_id and emit DispatchTiming(start)
        # before tier selection. Argument-validation failures above do not
        # constitute a dispatch attempt and do not produce events.
        dispatch_id, start_timestamp = self._open_dispatch_event(
            ensemble_name=name, session_id=session_id
        )
        exit_status: ExitStatus = "success"
        # WP-D: end event closes BEFORE envelope construction on success
        # paths so envelope.diagnostics can project from the emitted
        # DispatchTiming(end). ``dispatch_event_closed`` guards the
        # ``finally`` close so it only fires on exception / error paths
        # where the success path did not explicitly close.
        #
        # Invariant: every path through this function emits exactly one
        # DispatchTiming(end) event. The flag is set immediately after
        # an explicit close call; the ``finally`` block consults it and
        # only emits when the success path did not. A future edit that
        # adds a new return path (e.g., a new typed error post-tier-
        # selection) MUST either (a) set ``exit_status`` and rely on
        # the ``finally`` block (most error paths today), or (b) call
        # ``_close_dispatch_event`` explicitly and set the flag to True
        # (the WP-D success-path pattern). Adding a return without
        # either honors-the-invariant pattern would silently break
        # FC-23's "paired DispatchTiming events per dispatch" property.
        dispatch_event_closed = False
        try:
            # WP-G4-1, ADR-015: tier-escalation router interposes between
            # the autonomy gate (applied in ``dispatch``) and the actual
            # ensemble execution. When configured, the router consumes the
            # Calibration Gate's verdict for this dispatch and selects a
            # per-skill Model Profile (cheap or escalated). Abstain
            # verdicts and missing-metadata cases surface as typed
            # ToolCallErrors so the ReAct loop continues with the failure
            # as an observation rather than a crash.
            selection = self._select_tier_for(
                session_id=session_id,
                ensemble_name=name,
                call_id=id_,
                dispatch_id=dispatch_id,
            )
            if isinstance(selection, ToolCallError):
                exit_status = "error"
                return dataclasses.replace(selection, dispatch_id=dispatch_id)

            invocation_args: dict[str, Any] = {
                "ensemble_name": name,
                "input": input_data,
            }
            if selection is not None:
                invocation_args["model_profile_override"] = selection.model_profile

            try:
                result = await self._operations.invoke(invocation_args)
            except ValueError as exc:
                # ADR-018: record dispatch outcome as failure for the
                # escalation-vs-outcome correlation criterion.
                self._record_dispatch_outcome(
                    ensemble_name=name, selection=selection, success=False
                )
                exit_status = "error"
                return ToolCallError(
                    id=id_,
                    name="invoke_ensemble",
                    kind="invocation_failed",
                    reason=str(exc),
                    dispatch_id=dispatch_id,
                )

            # ADR-018: record dispatch outcome as success for the
            # escalation-vs-outcome correlation criterion. Outcome maps
            # to dispatch-completion-without-exception per WP-G4-2 scope;
            # a richer outcome signal (e.g., calibration quality signal)
            # is Cycle 5+ territory.
            self._record_dispatch_outcome(
                ensemble_name=name, selection=selection, success=True
            )

            await self._calibration_check_safe(
                session_id=session_id, ensemble_name=name, raw_result=result
            )

            raw_output = bool(result.get("raw_output", False))
            summarization = await self._harness.summarize(result, raw_output=raw_output)
            match summarization:
                case SummarizationSuccess(summary=summary):
                    # WP-D: close the dispatch event BEFORE envelope
                    # construction so envelope.diagnostics can read
                    # DispatchTiming(end)'s duration_seconds from the
                    # substrate's event log.
                    self._close_dispatch_event(
                        ensemble_name=name,
                        dispatch_id=dispatch_id,
                        start_timestamp=start_timestamp,
                        exit_status="success",
                    )
                    dispatch_event_closed = True
                    envelope = self._build_envelope(
                        ensemble_name=name,
                        dispatch_id=dispatch_id,
                        raw_result=result,
                        primary=summary,
                    )
                    return ToolCallSuccess(
                        id=id_,
                        name="invoke_ensemble",
                        content={"summary": summary},
                        dispatch_id=dispatch_id,
                        envelope=envelope,
                    )
                case RawOutputPassthrough(content=passthrough):
                    self._close_dispatch_event(
                        ensemble_name=name,
                        dispatch_id=dispatch_id,
                        start_timestamp=start_timestamp,
                        exit_status="success",
                    )
                    dispatch_event_closed = True
                    envelope = self._build_envelope(
                        ensemble_name=name,
                        dispatch_id=dispatch_id,
                        raw_result=result,
                        primary=_passthrough_primary(passthrough),
                    )
                    return ToolCallSuccess(
                        id=id_,
                        name="invoke_ensemble",
                        content=passthrough,
                        dispatch_id=dispatch_id,
                        envelope=envelope,
                    )
                case SummarizationFailure(reason=reason):
                    exit_status = "error"
                    return ToolCallError(
                        id=id_,
                        name="invoke_ensemble",
                        kind="summarization_failed",
                        reason=reason,
                        dispatch_id=dispatch_id,
                    )
        finally:
            if not dispatch_event_closed:
                self._close_dispatch_event(
                    ensemble_name=name,
                    dispatch_id=dispatch_id,
                    start_timestamp=start_timestamp,
                    exit_status=exit_status,
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

    def _open_dispatch_event(
        self, *, ensemble_name: str, session_id: str
    ) -> tuple[str | None, float]:
        """Allocate a dispatch_id and emit DispatchTiming(start) per ADR-023.

        Returns ``(dispatch_id, start_timestamp)``. When no event substrate
        is configured, ``dispatch_id`` is ``None`` (the close helper then
        skips end emission). ``start_timestamp`` is the wall-clock anchor
        for ``duration_seconds`` on the matching end event.

        ``model_profile`` is unknown at start (tier selection follows in
        the interposition order); the start event carries
        ``model_profile=None`` and the subsequent ``TierSelection`` event
        carries the selected profile.

        Cycle 6 WP-B piece 4: when a :class:`ToolCallEmitLogger` is also
        configured, the tool-call-emit liveness anchor fires after
        ``dispatch_id`` allocation and before ``DispatchTiming(start)``
        emission so the operator-terminal log stream observes the
        ``tool-call emit`` line first (FC-23).
        """
        start_timestamp = time.time()
        if self._event_substrate is None:
            return None, start_timestamp
        dispatch_id = self._event_substrate.new_dispatch_id(session_id)
        if self._tool_call_emit_logger is not None:
            self._tool_call_emit_logger.emit_tool_call_log(
                tool_name="invoke_ensemble", dispatch_id=dispatch_id
            )
        self._event_substrate.emit(
            DispatchTiming(
                phase="start",
                dispatch_id=dispatch_id,
                ensemble_name=ensemble_name,
                timestamp_seconds=start_timestamp,
            )
        )
        return dispatch_id, start_timestamp

    def _close_dispatch_event(
        self,
        *,
        ensemble_name: str,
        dispatch_id: str | None,
        start_timestamp: float,
        exit_status: ExitStatus,
    ) -> None:
        """Emit DispatchTiming(end) per ADR-023 — runs in invoke_ensemble's
        ``finally`` block so the operator-terminal and orchestrator-context
        destinations observe every dispatch regardless of exit path.
        """
        if self._event_substrate is None or dispatch_id is None:
            return
        end_timestamp = time.time()
        self._event_substrate.emit(
            DispatchTiming(
                phase="end",
                dispatch_id=dispatch_id,
                ensemble_name=ensemble_name,
                timestamp_seconds=end_timestamp,
                duration_seconds=end_timestamp - start_timestamp,
                exit_status=exit_status,
            )
        )

    def _select_tier_for(
        self,
        *,
        session_id: str,
        ensemble_name: str,
        call_id: str,
        dispatch_id: str | None = None,
    ) -> TierSelection | ToolCallError | None:
        """Consult the Tier-Escalation Router for this dispatch.

        Returns ``None`` when no router is configured (back-compat for
        pre-WP-G4-1 wiring). Otherwise consumes the Calibration Gate's
        verdict via ``verdict_for`` with a default
        :class:`DispatchContext` (AUQ confidence and trajectory features
        flow in once WP-H4 lands ADR-016's signal channel; until then
        the verdict defaults to ``proceed`` and the router selects
        cheap-tier). Typed router errors translate to
        ``ToolCallError`` so the orchestrator's ReAct loop continues
        with the failure as an observation per ADR-015 §Router logic.

        Per ADR-018 (WP-G4-2): when ``tier_router_audit`` is configured
        and ``fail_safe_active`` is True, the original verdict is
        recorded into the audit window but the router is invoked with
        a synthetic ``"reflect"`` verdict so every dispatch routes to
        the escalated tier. The audit records the *original* verdict
        (not the override) so drift detection is not corrupted by the
        fail-safe response itself.
        """
        if self._tier_router is None or self._calibration_gate is None:
            return None
        verdict = self._calibration_gate.verdict_for(
            session_id=session_id,
            ensemble_name=ensemble_name,
            dispatch_context=DispatchContext(),
        )
        # ADR-023 WP-A: emit the verdict event through the substrate so the
        # registered sinks (operator-terminal per WP-B; orchestrator-context
        # per WP-C) see the verdict alongside the DispatchTiming events.
        self._emit_calibration_verdict_event(
            verdict=verdict,
            ensemble_name=ensemble_name,
            dispatch_id=dispatch_id,
        )
        fail_safe = (
            self._tier_router_audit is not None
            and self._tier_router_audit.fail_safe_active
        )
        effective_verdict = "reflect" if fail_safe else verdict
        try:
            selection = self._tier_router.select_tier(
                ensemble_name=ensemble_name,
                verdict=effective_verdict,
                session_id=session_id,
            )
        except EscalationBypassError as exc:
            # Original verdict was Abstain (fail-safe inactive). Record
            # the bypass before returning the typed error so the audit's
            # bypass-rate trend criterion sees it.
            self._record_audit_consumption(
                verdict=verdict,
                selection=None,
                ensemble_name=ensemble_name,
                bypassed=True,
                dispatch_id=dispatch_id,
            )
            return ToolCallError(
                id=call_id,
                name="invoke_ensemble",
                kind="escalation_bypass",
                reason=exc.operator_diagnostic,
            )
        except MissingSkillMetadataError as exc:
            # Missing metadata is metadata-driven, not verdict-driven.
            # Don't pollute the audit's verdict-distribution baseline.
            return ToolCallError(
                id=call_id,
                name="invoke_ensemble",
                kind="missing_skill_metadata",
                reason=exc.operator_diagnostic,
            )
        # Stamp dispatch_id on the selection (ADR-023 WP-A) and emit the
        # event through the substrate. dataclasses.replace produces a new
        # frozen instance with the correlation identifier populated.
        if dispatch_id is not None:
            selection = dataclasses.replace(selection, dispatch_id=dispatch_id)
        if self._event_substrate is not None:
            self._event_substrate.emit(selection)
        # Successful selection (cheap or escalated tier). Record the
        # original verdict so the audit's verdict-distribution
        # criterion measures actual gate output, not fail-safe-coerced
        # values.
        self._record_audit_consumption(
            verdict=verdict,
            selection=selection,
            ensemble_name=ensemble_name,
            bypassed=False,
            dispatch_id=dispatch_id,
        )
        return selection

    def _emit_calibration_verdict_event(
        self,
        *,
        verdict: CalibrationVerdict,
        ensemble_name: str,
        dispatch_id: str | None,
    ) -> None:
        """Emit a CalibrationVerdictEvent through the substrate (ADR-023 WP-A).

        ``CalibrationVerdict`` is a literal; routing it requires the
        :class:`CalibrationVerdictEvent` wrapper carrying ``dispatch_id``
        and call-site context. No-op when no substrate is configured.
        """
        if self._event_substrate is None:
            return
        self._event_substrate.emit(
            CalibrationVerdictEvent(
                verdict=verdict,
                ensemble_name=ensemble_name,
                timestamp_seconds=time.time(),
                dispatch_id=dispatch_id,
            )
        )

    def _record_audit_consumption(
        self,
        *,
        verdict: CalibrationVerdict,
        selection: TierSelection | None,
        ensemble_name: str,
        bypassed: bool,
        dispatch_id: str | None = None,
    ) -> None:
        """Record a verdict consumption into the audit window (if configured).

        Per ADR-023 WP-A: when the consumption causes an audit window to
        close, the returned :class:`AuditDiagnostic` is stamped with the
        current dispatch_id (the dispatch that crossed the trigger) and
        emitted through the substrate.
        """
        if self._tier_router_audit is None:
            return
        diagnostic = self._tier_router_audit.record_consumption(
            verdict=verdict,
            selection=selection,
            ensemble_name=ensemble_name,
            bypassed=bypassed,
        )
        if diagnostic is None or self._event_substrate is None:
            return
        if dispatch_id is not None:
            diagnostic = dataclasses.replace(diagnostic, dispatch_id=dispatch_id)
        self._event_substrate.emit(diagnostic)

    def _record_dispatch_outcome(
        self,
        *,
        ensemble_name: str,
        selection: TierSelection | None,
        success: bool,
    ) -> None:
        """Record a dispatch outcome into the audit window (if configured).

        Tool Dispatch calls this after ``_operations.invoke`` either
        returns or raises. The auditor's escalation-vs-outcome
        correlation criterion compares cheap-tier and escalated-tier
        success rates within the current window per ADR-018 §"Drift
        criteria". Outcomes are recorded only when a tier was actually
        selected — bypassed and metadata-error dispatches do not
        contribute to the correlation criterion.
        """
        if self._tier_router_audit is None or selection is None:
            return
        self._tier_router_audit.record_outcome(
            ensemble_name=ensemble_name,
            tier=selection.tier,
            success=success,
        )

    def _build_envelope(
        self,
        *,
        ensemble_name: str,
        dispatch_id: str | None,
        raw_result: dict[str, Any],
        primary: str,
    ) -> DispatchEnvelope:
        """Construct an ADR-024 envelope for a successful dispatch.

        Cycle 6 WP-D. Reads dispatch events from the Dispatch Event
        Substrate (when configured) to populate ``diagnostics`` per
        the system-design interposition order step (10); reads the
        operator-authored ``output_schema:`` via :attr:`_output_schema_reader`
        and advisorily parses the synthesizer response as JSON to
        populate ``structured``.

        Legacy / no-substrate paths produce an envelope with sparse
        diagnostics (just ``ensemble`` and what the ``raw_result``
        ``metadata`` dict carries) — the envelope shape is preserved
        across both substrate-configured and legacy call sites.
        """
        diagnostics = self._collect_diagnostics(
            ensemble_name=ensemble_name,
            dispatch_id=dispatch_id,
            raw_result=raw_result,
        )
        structured = self._maybe_extract_structured(
            ensemble_name=ensemble_name, raw_result=raw_result
        )
        return DispatchEnvelope(
            status="success",
            primary=primary,
            structured=structured,
            diagnostics=diagnostics,
        )

    def _collect_diagnostics(
        self,
        *,
        ensemble_name: str,
        dispatch_id: str | None,
        raw_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Compose the envelope's ``diagnostics`` dict.

        Substrate-configured path: reads emitted events for
        ``dispatch_id`` and projects them into the envelope's
        diagnostic fields (``duration_seconds``, ``model_profile``,
        ``tier``, ``topaz_skill``, ``calibration_verdict``,
        ``audit_findings``). Legacy / no-substrate path: best-effort
        from ``raw_result['metadata']`` (the existing execution.json
        shape) with ``dispatch_id=None``.
        """
        diagnostics: dict[str, Any] = {"ensemble": ensemble_name}
        if dispatch_id is not None:
            diagnostics["dispatch_id"] = dispatch_id
        if self._event_substrate is not None and dispatch_id is not None:
            self._populate_from_events(diagnostics, dispatch_id)
        else:
            metadata = raw_result.get("metadata")
            if isinstance(metadata, dict):
                diagnostics["metadata"] = dict(metadata)
        return diagnostics

    def _populate_from_events(
        self, diagnostics: dict[str, Any], dispatch_id: str
    ) -> None:
        """Project the dispatch's emitted events into ``diagnostics``.

        Reads :meth:`DispatchEventSubstrate.events_for` and discriminates
        by ``isinstance`` — the substrate does not type-discriminate,
        so consumers do. Only the success-relevant fields are
        projected; audit findings accumulate as a list (zero or more
        per dispatch).
        """
        assert self._event_substrate is not None  # narrowed by caller
        events = self._event_substrate.events_for(dispatch_id)
        audit_findings: list[dict[str, Any]] = []
        for event in events:
            self._project_event(event, diagnostics, audit_findings)
        diagnostics["audit_findings"] = audit_findings

    @staticmethod
    def _project_event(
        event: object,
        diagnostics: dict[str, Any],
        audit_findings: list[dict[str, Any]],
    ) -> None:
        """Discriminate one event and project it into the diagnostics dict.

        Extracted from :meth:`_populate_from_events` to keep the
        iteration loop's cognitive complexity within the project's
        complexipy gate (15). Each branch projects a single event type
        onto the envelope's diagnostics surface.
        """
        if isinstance(event, DispatchTiming) and event.phase == "end":
            if event.duration_seconds is not None:
                diagnostics["duration_seconds"] = event.duration_seconds
            if event.exit_status is not None:
                diagnostics["exit_status"] = event.exit_status
        elif isinstance(event, TierSelection):
            diagnostics["model_profile"] = event.model_profile
            diagnostics["tier"] = event.tier
            diagnostics["topaz_skill"] = event.topaz_skill
        elif isinstance(event, CalibrationVerdictEvent):
            diagnostics["calibration_verdict"] = event.verdict
        elif type(event).__name__ == "AuditDiagnostic":
            # Collect by name rather than direct import so the L2
            # dependency surface stays narrow (audit module is imported
            # transitively through tier_router_audit).
            audit_findings.append(
                {
                    "kind": getattr(event, "kind", None),
                    "ensemble_name": getattr(event, "ensemble_name", None),
                    "detail": getattr(event, "detail", None),
                }
            )

    def _maybe_extract_structured(
        self, *, ensemble_name: str, raw_result: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Populate ``envelope.structured`` advisorily per ADR-024.

        Returns ``None`` when no schema is declared for the ensemble.
        When a schema is declared, attempts a JSON parse of the
        synthesizer's response text; on parse success returns the
        parsed dict (the synthesizer produced JSON conforming to the
        declared schema). On parse failure returns ``None`` without
        raising — schema validation is advisory at dispatch time per
        spike β's reframing of output-spec drift as ``input.data``
        override.
        """
        if self._output_schema_reader is None:
            return None
        schema = self._output_schema_reader.output_schema_for(ensemble_name)
        if schema is None:
            return None
        response_text = _extract_synthesizer_text(raw_result)
        if response_text is None:
            return None
        try:
            parsed = json.loads(response_text)
        except (json.JSONDecodeError, TypeError):
            return None
        if isinstance(parsed, dict):
            return parsed
        return None

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


def _extract_synthesizer_text(raw_result: dict[str, Any]) -> str | None:
    """Pull the synthesizer's response text from the execution result.

    Mirrors :func:`~llm_orc.agentic.result_summarizer_harness._extract_summary`'s
    forgiving contract: prefer ``synthesis`` when populated, otherwise
    use the single-agent ensemble's lone ``response`` field. Returns
    ``None`` when neither shape applies — multi-agent ensembles without
    a synthesizer carry per-agent responses that an envelope-time JSON
    parse cannot meaningfully unify.
    """
    synthesis = raw_result.get("synthesis")
    if isinstance(synthesis, str) and synthesis:
        return synthesis
    results = raw_result.get("results")
    if isinstance(results, dict) and len(results) == 1:
        only_result = next(iter(results.values()))
        if isinstance(only_result, dict):
            response = only_result.get("response")
            if isinstance(response, str) and response:
                return response
    return None


def _passthrough_primary(passthrough: Any) -> str:
    """Compose ``envelope.primary`` for the raw-output passthrough path.

    ADR-004's ``raw_output=True`` escape hatch lets the orchestrator
    observe the raw result dict; the envelope still needs a string
    ``primary``. ``json.dumps(default=str)`` produces a deterministic
    one-line representation suitable for the operator-readable
    summary-line slot. Non-serializable members fall back to their
    ``repr`` via ``default=str``.
    """
    if isinstance(passthrough, str):
        return passthrough
    return json.dumps(passthrough, default=str)


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

    Preserves the WP-C ``dispatch_id`` and WP-D ``envelope`` correlations
    so an Allow-with-events Autonomy decision composing with an
    ``invoke_ensemble`` dispatch does not drop the typed ADR-023 /
    ADR-024 correlation surfaces.
    """
    if not events:
        return result
    if isinstance(result, ToolCallSuccess):
        return ToolCallSuccess(
            id=result.id,
            name=result.name,
            content=result.content,
            events=events,
            dispatch_id=result.dispatch_id,
            envelope=result.envelope,
        )
    return ToolCallError(
        id=result.id,
        name=result.name,
        kind=result.kind,
        reason=result.reason,
        events=events,
        dispatch_id=result.dispatch_id,
    )
