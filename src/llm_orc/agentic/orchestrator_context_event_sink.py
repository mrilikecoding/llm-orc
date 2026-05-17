"""Orchestrator-Context Event Sink (Cycle 6 WP-C, ADR-023 Destination 2).

Per ``docs/agentic-serving/system-design.agents.md`` §Module:
Orchestrator-Context Event Sink (L2 — new in Cycle 6). The sink is the
orchestrator-context routing destination of the unified Dispatch Event
Substrate per Inversion N+2 — one substrate fans out to two
destinations (operator-terminal at L3, orchestrator-context at L2).

Architectural drivers:

* ADR-023 §"Destination 2 — Orchestrator-context" — structured-
  observation construction (the JSON-shaped block prepended to the
  orchestrator's next ReAct turn) and end-of-session ``dispatch_log``
  summary semantics.
* ADR-023 §"CalibrationSignal exclusion-by-default" — operator opt-in
  via ``agentic_serving.observability.orchestrator_context_routes_calibration_signal``.
* Cycle 6 DECIDE snapshot Finding 2 advisory carry-forward —
  ``dispatch_id`` is the single source-of-truth correlation across
  the event stream and the artifact path; this sink reads, never
  re-derives.
* Cycle 6 WP-B feed-forward advisory 3 — the sink uses the same
  ``dispatch_id`` session-prefix filter pattern as the inference-wait
  heartbeat scheduler so per-session isolation is uniformly enforced
  across all per-request substrate consumers. The dispatch_id format
  coupling (``f"{session_id}-dispatch-{counter:04d}"`` from
  :func:`new_dispatch_id`) is the substrate's single source of truth.

The sink owns the canonical seven-field observation schema (FC-24
sibling fitness) — exactly:
``{dispatched, duration_seconds, model_profile, tier, topaz_skill,
calibration_verdict, dispatch_id}``. Missing field values fall back
to ``None`` rather than omitting the key so the schema is uniform
across dispatches that emit subsets of the event types.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from llm_orc.agentic.calibration_gate import CalibrationVerdictEvent
from llm_orc.agentic.calibration_signal_channel import CalibrationSignal
from llm_orc.agentic.dispatch_event_substrate import (
    DispatchEventSubstrate,
    DispatchTiming,
)
from llm_orc.agentic.tier_router import TierSelection

__all__ = [
    "OrchestratorContextEventSink",
    "StructuredObservation",
]


_CANONICAL_FIELDS: tuple[str, ...] = (
    "dispatched",
    "duration_seconds",
    "model_profile",
    "tier",
    "topaz_skill",
    "calibration_verdict",
    "dispatch_id",
)
"""ADR-023 §Destination 2 — the structured observation's canonical
seven-field schema. Verified by the schema-validation fitness."""


@dataclass(frozen=True)
class StructuredObservation:
    """Seven-field canonical observation block per ADR-023 §Destination 2.

    The Runtime prepends this block (serialized as JSON) at each
    turn boundary so the orchestrator-LLM sees dispatch outcomes in
    its reasoning context — answering PLAY note 12's load-bearing
    practitioner question (*"What was the total run-time of the
    ensemble?"*) directly from ``duration_seconds``.

    Field set is exactly the seven canonical fields; ``None`` values
    are preserved in the serialized JSON rather than omitted so
    downstream consumers see a uniform schema regardless of which
    event types fired for the dispatch.
    """

    dispatched: str | None
    duration_seconds: float | None
    model_profile: str | None
    tier: str | None
    topaz_skill: str | None
    calibration_verdict: str | None
    dispatch_id: str

    def to_json(self) -> str:
        """Serialize to JSON with ``None`` preserved (not omitted).

        ``json.dumps`` renders ``None`` as ``null`` — the canonical
        schema's missing-field semantics. The result is the wire form
        the Runtime appends to the next-turn message stream.
        """
        return json.dumps(asdict(self))


class OrchestratorContextEventSink:
    """Registered :class:`EventSink` for the orchestrator-context destination.

    Implements the substrate ``EventSink`` Protocol via :meth:`consume`.
    Buffers events keyed by ``dispatch_id`` so the Runtime can query a
    completed dispatch's observation at the next turn boundary via
    :meth:`observations_for`. End-of-session summary writing is owned
    by :meth:`dispatch_log_entries` (read) and
    :meth:`write_dispatch_log` (write to filesystem).

    Per-session isolation. The sink takes a ``session_id`` at
    construction and filters events whose ``dispatch_id`` does not
    start with ``f"{session_id}-dispatch-"`` (the WP-B feed-forward
    advisory 3 pattern). Cross-session traffic is silently ignored;
    the sink is intended for per-request registration alongside the
    inference-wait heartbeat scheduler.

    CalibrationSignal handling. By default the sink ignores
    :class:`CalibrationSignal` events (excluded from orchestrator-
    context routing per ADR-023). The opt-in flag
    ``routes_calibration_signal=True`` (config-resolvable via
    ``agentic_serving.observability.orchestrator_context_routes_calibration_signal``)
    includes them in the per-dispatch buffer; signals surface in the
    end-of-session dispatch_log under a ``calibration_signals`` key.
    They never appear in the seven-field observation block — that
    schema is fixed.
    """

    def __init__(
        self,
        *,
        session_id: str,
        routes_calibration_signal: bool = False,
    ) -> None:
        self._session_id = session_id
        self._dispatch_prefix = f"{session_id}-dispatch-"
        self._routes_calibration_signal = routes_calibration_signal
        # Insertion-ordered: end-of-session summary writes dispatches
        # in the order they completed (Python dict preserves insertion).
        self._events_by_dispatch: dict[str, list[object]] = {}

    # ------------------------------------------------------------------
    # EventSink Protocol — substrate fan-out
    # ------------------------------------------------------------------

    def consume(self, event: object) -> None:
        """Buffer one event for the dispatch it correlates to.

        Events without a string ``dispatch_id`` are ignored — they
        cannot be correlated to a dispatch and the schema requires
        one. Events whose ``dispatch_id`` belongs to another session
        are ignored per the session-prefix filter. ``CalibrationSignal``
        events are ignored unless ``routes_calibration_signal=True``.
        """
        dispatch_id = getattr(event, "dispatch_id", None)
        if not isinstance(dispatch_id, str):
            return
        if not dispatch_id.startswith(self._dispatch_prefix):
            return
        if isinstance(event, CalibrationSignal) and not self._routes_calibration_signal:
            return
        self._events_by_dispatch.setdefault(dispatch_id, []).append(event)

    # ------------------------------------------------------------------
    # Query surface — Runtime turn-boundary integration
    # ------------------------------------------------------------------

    def observations_for(self, dispatch_id: str) -> StructuredObservation | None:
        """Compose the seven-field observation for ``dispatch_id``.

        Returns ``None`` for unknown ``dispatch_id`` values (the Runtime
        treats ``None`` as "no observation to prepend"; this guards
        against the dispatch's events being filtered out by the
        session-prefix check upstream).

        Missing-field handling: every canonical field falls back to
        ``None`` rather than omitting the key. The dispatch-id-only
        case (no other events fired) still yields a valid observation
        with six ``None`` fields plus the ``dispatch_id``.
        """
        events = self._events_by_dispatch.get(dispatch_id)
        if events is None:
            return None
        return _compose_observation(dispatch_id, events)

    def observation_message_for(self, dispatch_id: str) -> dict[str, Any] | None:
        """ContextObservationSink Protocol surface for the Runtime.

        Wraps :meth:`observations_for`'s seven-field block in a
        ``role: user`` message dict the Runtime appends directly to
        ``messages``. The ``role: user`` choice matches the phantom-
        rejection diagnostic pattern in the same Runtime module
        (ADR-017 §Rejection) — observations and rejection diagnostics
        both flow back to the orchestrator's reasoning surface via the
        user-role channel rather than as system instructions or tool
        results (which would require a ``tool_call_id``).

        **Alternative considered.** ``role: system`` would carry more
        weight on the orchestrator-LLM's incorporation but conflicts
        with the project's existing convention of placing a single
        ``system`` message at the top of the conversation that teaches
        the closed five-tool surface (ADR-003, Option C). Repeated
        ``role: system`` messages would weaken that convention and
        could compete with the operator-tuned system prompt. The
        post-BUILD PLAY phase is the natural place to characterize
        whether ``role: user`` proves sufficient or whether per-profile
        observation weighting needs a different message shape.

        Returns ``None`` when no observation exists for ``dispatch_id``
        (unknown id, or all events filtered out by the session-prefix
        check). The Runtime treats ``None`` as "no observation to
        prepend" and continues without modifying ``messages``.
        """
        observation = self.observations_for(dispatch_id)
        if observation is None:
            return None
        return {"role": "user", "content": observation.to_json()}

    def dispatch_log_entries(self) -> list[dict[str, Any]]:
        """All dispatches' observations in completion order.

        Used by :meth:`write_dispatch_log` and exposed for testing.
        Each entry is the seven-field observation as a dict; when
        ``routes_calibration_signal=True`` is set and the dispatch
        emitted any :class:`CalibrationSignal` events, an additional
        ``calibration_signals`` key carries the signal data as a list
        of per-signal dicts.
        """
        entries: list[dict[str, Any]] = []
        for dispatch_id, events in self._events_by_dispatch.items():
            observation = _compose_observation(dispatch_id, events)
            entry: dict[str, Any] = asdict(observation)
            if self._routes_calibration_signal:
                signals = [
                    _calibration_signal_to_dict(e)
                    for e in events
                    if isinstance(e, CalibrationSignal)
                ]
                if signals:
                    entry["calibration_signals"] = signals
            entries.append(entry)
        return entries

    # ------------------------------------------------------------------
    # End-of-session — dispatch_log write
    # ------------------------------------------------------------------

    def write_dispatch_log(self, path: Path) -> None:
        """Write the dispatch_log to ``path`` (one JSON file per session).

        The path is owned by the caller — the serve layer constructs
        it from the session_id and the configured
        ``agentic_sessions_root`` (default ``.llm-orc/agentic-sessions/``).
        WP-E lands the full session-dir layout; WP-C writes a standalone
        ``dispatch_log.json`` so the integration with execution.json
        can compose later without coupling WP-C to WP-E's tree shape.

        Creates parent directories if needed; overwrites existing file.
        Empty dispatch_log (no dispatches completed) still writes a
        valid file with an empty ``entries`` list — the file's
        existence is the structural signal that the session closed
        cleanly.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"dispatch_log": {"entries": self.dispatch_log_entries()}}
        path.write_text(json.dumps(payload, indent=2))

    # ------------------------------------------------------------------
    # Substrate lifecycle — per-request register/unregister
    # ------------------------------------------------------------------

    def register_with(self, substrate: DispatchEventSubstrate) -> None:
        """Register as a substrate sink (per-request open).

        Mirrors :meth:`OperatorTerminalEventSink.register_with` so
        serve-layer wiring uses one pattern across both destinations.
        """
        substrate.register_sink(self)

    def unregister_with(self, substrate: DispatchEventSubstrate) -> None:
        """Remove from the substrate (per-request close); idempotent."""
        substrate.unregister_sink(self)


def _compose_observation(
    dispatch_id: str, events: list[object]
) -> StructuredObservation:
    """Reduce a dispatch's events into the canonical seven-field block.

    Reads the latest value of each field from the event stream:

    * ``dispatched`` ← any event's ``ensemble_name`` (DispatchTiming or
      CalibrationVerdictEvent both carry it).
    * ``duration_seconds`` ← ``DispatchTiming(phase="end")`` only.
    * ``model_profile``, ``tier``, ``topaz_skill`` ← ``TierSelection``.
    * ``calibration_verdict`` ← ``CalibrationVerdictEvent.verdict``.

    Field absence (no contributing event) preserves ``None``.
    """
    dispatched: str | None = None
    duration_seconds: float | None = None
    model_profile: str | None = None
    tier: str | None = None
    topaz_skill: str | None = None
    calibration_verdict: str | None = None

    for event in events:
        if isinstance(event, DispatchTiming):
            dispatched = event.ensemble_name
            if event.phase == "end":
                duration_seconds = event.duration_seconds
            if event.model_profile is not None and model_profile is None:
                model_profile = event.model_profile
        elif isinstance(event, TierSelection):
            model_profile = event.model_profile
            tier = event.tier
            topaz_skill = event.topaz_skill
        elif isinstance(event, CalibrationVerdictEvent):
            calibration_verdict = event.verdict
            if dispatched is None:
                dispatched = event.ensemble_name

    return StructuredObservation(
        dispatched=dispatched,
        duration_seconds=duration_seconds,
        model_profile=model_profile,
        tier=tier,
        topaz_skill=topaz_skill,
        calibration_verdict=calibration_verdict,
        dispatch_id=dispatch_id,
    )


def _calibration_signal_to_dict(event: CalibrationSignal) -> dict[str, Any]:
    """Serialize a :class:`CalibrationSignal` for the dispatch_log entry.

    Opt-in only — called by :meth:`dispatch_log_entries` when
    ``routes_calibration_signal=True`` is set on the sink. The
    seven-field observation schema is unchanged; signals surface in a
    side-channel ``calibration_signals`` key.
    """
    return {
        "ensemble_name": event.ensemble_name,
        "dispatch_success": event.dispatch_success,
        "recent_token_entropy": event.recent_token_entropy,
        "deterministic_anchor": event.deterministic_anchor,
        "dispatch_id": event.dispatch_id,
    }
