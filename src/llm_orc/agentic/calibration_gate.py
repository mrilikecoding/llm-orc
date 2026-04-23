"""Calibration Gate — result-checking of composed ensembles (ADR-007, WP-H).

Per ``docs/agentic-serving/system-design.md`` §Calibration Gate (L1
Domain Policy) and §Integration Contracts (Orchestrator Tool Dispatch
→ Calibration Gate). ADR-007 commits every composed ensemble to a
calibration period: the first N invocations always run a check that
attaches a Quality Signal to the invocation. Accumulated signals
decide whether the ensemble transitions to ``trusted``. AS-5 forbids
frequency-only transitions.

The gate is L1 — it takes a plain ``session_id`` string on each call so
the module stays free of imports from Session Registry (L3). Per-session
records are held internally, indexed by ``(session_id, ensemble_name)``.
When Plexus lands (WP-I), persistence is added by composing a Plexus-
backed store behind the same surface; stateless mode continues to work
exactly as today (AS-8).

Key invariant for Phase 1: calibration check does **not** prevent
invocation (ADR-007 clause 2). Tool Dispatch calls ``check_and_record``
alongside ``invoke_ensemble``'s return; the signal is recorded but the
result still flows through the Summarizer Harness and back to the
orchestrator. A failing check produces a ``negative`` or ``absent``
signal; it does not raise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

DEFAULT_CALIBRATION_N = 3
"""Default number of invocations required to earn trust.

A WP-H build decision per ADR-007 clause 5. Three balances check cost
against single-invocation noise tolerance — one unlucky reading does
not prevent transition indefinitely; two unlucky readings extend the
calibration window, which is the intent. Operators tune via
``agentic_serving.orchestrator.calibration.default_n``.
"""


QualitySignal = Literal["positive", "negative", "absent"]
"""Per system design §Integration Contracts (Orchestrator Tool Dispatch
→ Calibration Gate). ``positive`` = plausible on-task output;
``negative`` = hallucinated or off-task; ``absent`` = too short, empty,
or not evaluable."""

CalibrationStatus = Literal["in_calibration", "trusted"]


@dataclass
class CalibrationRecord:
    """Per-ensemble calibration state within one Session.

    Mutable so the gate can update it in place — mirrors
    :class:`~llm_orc.agentic.session_registry.SessionState` which is
    mutable for cumulative turn/token accounting. ``signals`` stays an
    immutable tuple so readers cannot inadvertently corrupt the record.
    """

    ensemble_name: str
    status: CalibrationStatus = "in_calibration"
    signals: tuple[QualitySignal, ...] = ()
    invocations_seen: int = 0


class CalibrationChecker(Protocol):
    """Produces a :data:`QualitySignal` for a raw ensemble result.

    The default implementation (:class:`EnsembleBackedChecker`) invokes
    a configured checker ensemble; tests pass scripted doubles. Kept
    narrow — the checker is the one variable the operator tunes when
    calibration quality needs attention.
    """

    async def check(
        self, *, ensemble_name: str, raw_result: dict[str, Any]
    ) -> QualitySignal: ...


@dataclass
class _SessionRecords:
    """Per-session bundle of per-ensemble records.

    Exists so the gate's internal store is ``dict[str, _SessionRecords]``
    rather than ``dict[str, dict[str, CalibrationRecord]]`` — the former
    reads as "one bundle per session" at the call site, the latter as
    "two levels of key indirection."
    """

    records: dict[str, CalibrationRecord] = field(default_factory=dict)


class CalibrationGate:
    """Tracks Calibration state and runs Quality-Signal checks.

    Construction takes ``default_n`` (the calibration window) and a
    :class:`CalibrationChecker`. The gate owns per-session state
    internally; the Session Registry stays agnostic. WP-I layers a
    Plexus-backed store behind the same surface for cross-session trust.
    """

    def __init__(
        self, *, default_n: int = DEFAULT_CALIBRATION_N, checker: CalibrationChecker
    ) -> None:
        if default_n < 1:
            raise ValueError(
                f"Calibration Gate default_n must be >= 1, got {default_n}"
            )
        self._default_n = default_n
        self._checker = checker
        self._sessions: dict[str, _SessionRecords] = {}

    def mark_composed(self, *, session_id: str, ensemble_name: str) -> None:
        """Register a newly composed ensemble for calibration.

        Called by Tool Dispatch after a successful ``compose_ensemble``
        write. Idempotent: if the ensemble is already tracked, no state
        is reset — accidental double-calls from a retry path do not
        discard accumulated signals.
        """
        bundle = self._sessions.setdefault(session_id, _SessionRecords())
        bundle.records.setdefault(
            ensemble_name, CalibrationRecord(ensemble_name=ensemble_name)
        )

    def status(self, *, session_id: str, ensemble_name: str) -> CalibrationStatus:
        """Return the calibration status for ``ensemble_name`` in the session.

        Untracked ensembles (library entries, or any ensemble the
        orchestrator did not compose within this session) return
        ``trusted`` — calibration applies only to composed ensembles.
        """
        bundle = self._sessions.get(session_id)
        if bundle is None:
            return "trusted"
        record = bundle.records.get(ensemble_name)
        if record is None:
            return "trusted"
        return record.status

    def record_for(
        self, *, session_id: str, ensemble_name: str
    ) -> CalibrationRecord | None:
        """Return the underlying record; ``None`` if the ensemble is untracked.

        Tests read this; production paths should not need to — the gate's
        public API (``status``, ``check_and_record``) is the intended
        surface for Tool Dispatch.
        """
        bundle = self._sessions.get(session_id)
        if bundle is None:
            return None
        return bundle.records.get(ensemble_name)

    async def check_and_record(
        self,
        *,
        session_id: str,
        ensemble_name: str,
        raw_result: dict[str, Any],
    ) -> QualitySignal | None:
        """Run the check on in-calibration ensembles; record the signal.

        Returns the Quality Signal produced for this invocation, or
        ``None`` if the ensemble is ``trusted`` (untracked or previously
        cleared). Per ADR-007 clause 2 the gate never prevents
        invocation — a negative signal is recorded alongside the return,
        it does not raise.

        After recording, the record's status is recomputed from the
        last-N signals. A negative or absent signal in the last-N window
        keeps the ensemble in calibration even after more than N
        invocations — the "calibration period extends" semantic from
        scenario §Calibration fails to clear.
        """
        bundle = self._sessions.get(session_id)
        if bundle is None:
            return None
        record = bundle.records.get(ensemble_name)
        if record is None or record.status == "trusted":
            return None

        signal = await self._checker.check(
            ensemble_name=ensemble_name, raw_result=raw_result
        )
        record.signals = record.signals + (signal,)
        record.invocations_seen += 1
        record.status = self._compute_status(record.signals)
        return signal

    def _compute_status(self, signals: tuple[QualitySignal, ...]) -> CalibrationStatus:
        """Decide status from accumulated signals (AS-5).

        Transition to ``trusted`` requires two conditions:

        1. At least ``default_n`` signals have accumulated (not frequency
           alone — AS-5 — but enough evidence to judge).
        2. The most-recent ``default_n`` signals are all ``positive``.

        A negative or absent signal in the last-N window keeps the
        ensemble ``in_calibration`` — subsequent invocations keep
        checking until either a clean run of ``default_n`` positives
        accumulates, or the operator intervenes.
        """
        if len(signals) < self._default_n:
            return "in_calibration"
        last_n = signals[-self._default_n :]
        if all(s == "positive" for s in last_n):
            return "trusted"
        return "in_calibration"
