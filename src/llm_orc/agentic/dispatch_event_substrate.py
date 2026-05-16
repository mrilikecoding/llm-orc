"""Dispatch Event Substrate (Cycle 6 WP-A, ADR-023).

Per ``docs/agentic-serving/system-design.agents.md`` §Module: Dispatch
Event Substrate (L1 — new in Cycle 6). The substrate is the unified
event-emission infrastructure per **Inversion N+2**: one substrate
fans out to two routing destinations (operator-terminal sink shipped
by WP-B; orchestrator-context sink shipped by WP-C).

Architectural drivers:

* ADR-023 — observability event-routing decision; the substrate's
  scope and the ``DispatchTiming`` extension shape.
* ADR-023 Open Decision Point C6-1 — ``dispatch_id`` generation
  strategy; the default is a session-scoped monotonic counter
  (simpler than UUID4, observably ordered, aligns with the artifact
  filesystem path's lexicographic-sortability for operator review).
* Cycle 6 DECIDE snapshot Finding 2 (advisory carry-forward) —
  ``dispatch_id`` is the single source-of-truth correlation
  identifier across the event stream, the envelope's
  ``diagnostics.dispatch_id`` (WP-D), and the artifact filesystem
  path's ``<dispatch_id>`` segment (WP-E).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Literal, Protocol

__all__ = [
    "DispatchEvent",
    "DispatchEventSubstrate",
    "DispatchPhase",
    "DispatchTiming",
    "EventSink",
    "ExitStatus",
    "new_dispatch_id",
]


DispatchPhase = Literal["start", "end"]
"""Whether a :class:`DispatchTiming` event marks the start or end of one
``invoke_ensemble`` dispatch per ADR-023 §"Event-emission substrate"."""


ExitStatus = Literal["success", "error", "timeout", "aborted"]
"""Dispatch-end exit status per ADR-023 §"Event-emission substrate".

``success`` covers normal returns; ``error`` covers
:class:`~llm_orc.models.structural_errors.LlmOrcStructuralError`
subclasses; ``timeout`` covers cheap-tier and escalated-tier wall-clock
limits; ``aborted`` covers Abstain verdicts and operator-driven cancellations.
"""


@dataclass(frozen=True)
class DispatchTiming:
    """``invoke_ensemble`` dispatch start or end event per ADR-023.

    Emitted twice per dispatch by Orchestrator Tool Dispatch (the
    producer):

    * ``phase="start"`` immediately after :meth:`new_dispatch_id`
      allocation and before tier selection (carries call-site context).
    * ``phase="end"`` after the harness (or substrate-write per ADR-025)
      returns control (carries ``duration_seconds`` and ``exit_status``).

    PLAY note 12's load-bearing practitioner question
    (*"What was the total run-time of the ensemble?"*) is answered
    directly by ``duration_seconds`` on the end event.
    """

    phase: DispatchPhase
    dispatch_id: str
    ensemble_name: str
    timestamp_seconds: float
    model_profile: str | None = None
    duration_seconds: float | None = None
    exit_status: ExitStatus | None = None


class DispatchEvent(Protocol):
    """Minimum contract for events routed through the substrate.

    All dispatch events expose ``dispatch_id`` (``str | None`` during
    the Cycle 6 transition; the field becomes required once the
    progressive conversion closes per ADR-023 §"Event-emission substrate").
    Sinks discriminate concrete event types via ``isinstance`` checks at
    consumption time — the substrate does not type-discriminate.
    """

    dispatch_id: str | None


class EventSink(Protocol):
    """Sink consumer contract per ADR-023.

    Sinks register with :meth:`DispatchEventSubstrate.register_sink` and
    receive every emitted event via :meth:`consume`. Exceptions inside
    :meth:`consume` are isolated by the substrate — one failing sink
    does not affect other sinks or the producing path.
    """

    def consume(self, event: object) -> None:
        """Consume one event from the substrate's fan-out."""
        ...


def new_dispatch_id(session_id: str, counter: int) -> str:
    """Compose a session-scoped dispatch identifier.

    Per ADR-023 Open Decision Point C6-1 — the monotonic counter
    strategy. Format ``<session_id>-dispatch-<counter:04d>`` keeps the
    identifier lexicographically sortable for operator review and aligns
    with the ADR-025 artifact path's ``<dispatch_id>`` segment.
    """
    return f"{session_id}-dispatch-{counter:04d}"


class DispatchEventSubstrate:
    """Unified event-emission substrate per ADR-023 (Inversion N+2).

    The substrate owns:

    * ``dispatch_id`` allocation via :meth:`new_dispatch_id` — a
      session-scoped monotonic counter that becomes the correlation
      identifier across the event stream, the envelope's
      ``diagnostics.dispatch_id`` (WP-D), and the artifact filesystem
      path's ``<dispatch_id>`` segment (WP-E).
    * Sink registration via :meth:`register_sink` — multiple sinks
      observe the same substrate (operator-terminal + orchestrator-
      context arrive at WP-B + WP-C as registered consumers).
    * Event fan-out via :meth:`emit` — synchronous delivery to every
      registered sink, with per-sink exception isolation.
    * Post-hoc query via :meth:`events_for` — used by Orchestrator Tool
      Dispatch (WP-D) to assemble the envelope's diagnostics, and by
      Orchestrator-Context Event Sink (WP-C) to construct the
      structured observation block at turn boundaries.

    Producer-side migration (WP-A): Tier Router, Calibration Gate,
    Tier-Router-Audit, Calibration Signal Channel, and Orchestrator
    Tool Dispatch emit through this substrate. Sink-side wiring lands
    with WP-B and WP-C.
    """

    def __init__(self) -> None:
        self._sinks: list[EventSink] = []
        self._counters: dict[str, itertools.count[int]] = {}
        self._log: dict[str, list[object]] = {}

    def new_dispatch_id(self, session_id: str) -> str:
        """Allocate the next dispatch identifier for ``session_id``.

        Counters are session-scoped and monotonic — the first dispatch
        in a session is ``-dispatch-0001``, the second ``-dispatch-0002``,
        and so on. Counters across sessions are independent.
        """
        counter = self._counters.setdefault(session_id, itertools.count(1))
        return new_dispatch_id(session_id, next(counter))

    def register_sink(self, sink: EventSink) -> None:
        """Register a sink to receive every subsequent emission."""
        self._sinks.append(sink)

    def emit(self, event: object) -> None:
        """Fan out one event to every registered sink.

        Events whose ``dispatch_id`` attribute is a string are recorded
        in the substrate's per-dispatch log so :meth:`events_for` can
        reconstruct the full picture later. Events with
        ``dispatch_id=None`` (legacy emission sites during the Cycle 6
        transition) still fan out to sinks but cannot be retrieved by
        :meth:`events_for`.

        Sink exceptions are isolated. The producer path always returns
        normally — operator-terminal sink WARNs surface sink failures
        once the sink module ships (WP-B).
        """
        dispatch_id = getattr(event, "dispatch_id", None)
        if isinstance(dispatch_id, str):
            self._log.setdefault(dispatch_id, []).append(event)
        for sink in self._sinks:
            try:
                sink.consume(event)
            except Exception:  # noqa: BLE001
                continue

    def events_for(self, dispatch_id: str) -> list[object]:
        """Return every event emitted for ``dispatch_id`` in emission order.

        Returns an empty list for unknown ``dispatch_id`` values, including
        identifiers whose events were emitted with ``dispatch_id=None``.
        """
        return list(self._log.get(dispatch_id, ()))
