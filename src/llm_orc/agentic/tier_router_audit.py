"""Tier-Escalation Router (d)-analog audit dispatch (WP-G4-2, ADR-018).

Periodic out-of-band audit on the L1→L2 verdict→router edge. Analog
of ADR-016 mechanism (d), structurally transferred per Spike β
(research log ``005h-spike-bounding-mechanism-transfer-l1-l2.md``,
2026-05-11). Per system-design.agents.md §Module: Tier-Escalation
Router (L2 — extended at architect-gate close 2026-05-11).

The auditor accumulates verdict consumptions and dispatch outcomes
across audit windows. When the trigger condition fires (every
``trigger_count`` consumptions or every ``trigger_wall_clock_hours``
of wall-clock time, whichever first), the auditor closes the current
window, evaluates three drift criteria against the prior window, and
emits an audit verdict (``no_drift`` / ``advisory`` / ``severe``).
Severe drift activates fail-safe mode — the Tool Dispatch consults
:attr:`TierEscalationAuditor.fail_safe_active` before tier selection
and routes all dispatches to the escalated tier while fail-safe is
active.

**FC-19 preserved by design.** :class:`TierEscalationAuditor` holds
all audit state in its instance; :class:`llm_orc.agentic.tier_router.
TierRouter`'s ``select_tier`` reads no auditor state and remains a
pure stateless function. Fail-safe override lives at the Tool
Dispatch consultation point, not inside the router.

**Three drift criteria** (ADR-018 §Decision §"Drift criteria"):

* **verdict_distribution_shift** — relative-frequency shift of any
  verdict axis exceeding ``verdict_distribution_shift`` (default 0.15
  / ±15 pp) between consecutive audit windows.
* **escalation_outcome_correlation** — escalated-tier success rate
  minus cheap-tier success rate (within the current window) below
  ``escalation_outcome_correlation_pp`` (default +5 pp). This is
  the Sub-Q6 evidence surface — distinguishes routing-noise from
  tier-configuration signal.
* **bypass_rate_trend** — relative-rate increase in Abstain bypasses
  per dispatch exceeding ``bypass_rate_increase`` (default +25 pp)
  between consecutive windows. Undefined-baseline cases (prior window
  had zero Abstains, current has some) register as exceeding but not
  severe — the trend is real but the magnitude is statistically
  undefined.

**Severity rule.** Two or more criteria exceeding their thresholds
produce a severe verdict. A single criterion exceeding at ≥
``severe_drift_multiplier`` × threshold (default 2× — the "severe
magnitude" cutoff) also produces severe. Otherwise a single
exceeding criterion is advisory. Zero is no_drift.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol

from llm_orc.agentic.calibration_gate import CalibrationVerdict
from llm_orc.agentic.tier_router import Tier, TierSelection

__all__ = [
    "AuditClock",
    "AuditDiagnostic",
    "AuditVerdict",
    "CriterionFinding",
    "SystemAuditClock",
    "TierEscalationAuditor",
    "TierEscalationAuditThresholds",
]


AuditVerdict = Literal["no_drift", "advisory", "severe"]
"""Per ADR-018 §"Verdict shape" — three-value verdict trichotomy.

Parallel-by-construction to ADR-016 mechanism (d)'s verdict shape.
"""


_CRITERION_NAMES: tuple[str, ...] = (
    "verdict_distribution_shift",
    "escalation_outcome_correlation",
    "bypass_rate_trend",
)


class AuditClock(Protocol):
    """Wall-clock surface for trigger-by-time evaluation.

    Production wiring uses :class:`SystemAuditClock`; tests pass a
    controllable clock so the 24-hour bound can be exercised
    deterministically without sleeping.
    """

    def now_seconds(self) -> float: ...


class SystemAuditClock:
    """Default :class:`AuditClock` backed by :func:`time.time`."""

    def now_seconds(self) -> float:
        return time.time()


@dataclass(frozen=True)
class TierEscalationAuditThresholds:
    """ADR-018-derived audit thresholds.

    All thresholds are operator-tunable via OrchestratorConfig per
    ADR-018 §Decision. Defaults match ADR-018's stated defaults.
    """

    trigger_count: int
    """Audit fires after this many verdict consumptions in a window.
    Default 100 per ADR-018 §"Trigger"."""

    trigger_wall_clock_hours: float
    """Audit fires when the window age reaches this many wall-clock
    hours (whichever comes first against ``trigger_count``).
    Default 24.0 per ADR-018 §"Trigger"."""

    verdict_distribution_shift: float
    """Maximum absolute relative-frequency shift on any verdict axis
    permitted between consecutive windows. Default 0.15 (±15 pp)."""

    escalation_outcome_correlation_pp: float
    """Minimum required escalation-vs-cheap success-rate gap, in
    percentage points expressed as a fraction. Default 0.05 (+5 pp).
    The Sub-Q6 evidence surface per ADR-018."""

    bypass_rate_increase: float
    """Maximum tolerated relative-rate increase in Abstain bypasses
    between consecutive windows. Default 0.25 (+25 % relative rate)."""

    severe_drift_multiplier: float
    """Single-criterion severity threshold: a criterion that exceeds
    its baseline by at least this multiple of its threshold reaches
    severe magnitude on its own. Default 2.0."""

    def __post_init__(self) -> None:
        if self.trigger_count <= 0:
            raise ValueError(
                "TierEscalationAuditThresholds.trigger_count must be a "
                "positive integer per ADR-018 §Trigger"
            )
        if self.trigger_wall_clock_hours <= 0:
            raise ValueError(
                "TierEscalationAuditThresholds.trigger_wall_clock_hours "
                "must be positive per ADR-018 §Trigger"
            )
        if self.verdict_distribution_shift <= 0:
            raise ValueError(
                "TierEscalationAuditThresholds.verdict_distribution_shift "
                "must be positive per ADR-018 §Drift criteria"
            )
        if self.escalation_outcome_correlation_pp <= 0:
            raise ValueError(
                "TierEscalationAuditThresholds.escalation_outcome_"
                "correlation_pp must be positive per ADR-018 §Drift criteria"
            )
        if self.bypass_rate_increase <= 0:
            raise ValueError(
                "TierEscalationAuditThresholds.bypass_rate_increase "
                "must be positive per ADR-018 §Drift criteria"
            )
        if self.severe_drift_multiplier < 1.0:
            raise ValueError(
                "TierEscalationAuditThresholds.severe_drift_multiplier "
                "must be at least 1.0 (severe threshold cannot be below "
                "the advisory threshold)"
            )


@dataclass(frozen=True)
class CriterionFinding:
    """One drift-criterion evaluation result.

    Carried inside :class:`AuditDiagnostic` so operator review can
    inspect which criteria triggered the verdict and by how much.
    """

    name: str
    """One of :data:`_CRITERION_NAMES`."""

    value: float
    """The criterion's measured value in the current window."""

    threshold: float
    """The criterion's advisory threshold from
    :class:`TierEscalationAuditThresholds`."""

    exceeds: bool
    """``True`` if the criterion's measured value exceeds its
    advisory threshold."""

    severe: bool
    """``True`` if the criterion's measured value exceeds the
    severe-magnitude cutoff (advisory threshold ×
    ``severe_drift_multiplier``)."""


@dataclass(frozen=True)
class AuditDiagnostic:
    """One audit-window verdict, surfaced to operator review.

    Diagnostics accumulate in the auditor's
    :meth:`TierEscalationAuditor.diagnostics` tuple. Operators consult
    them at session boundaries per ADR-018 §"Asynchronous-operator-
    review dynamic".
    """

    window_id: int
    """Monotonic audit-window counter. ``1`` for the first audit
    fired, ``2`` for the second, etc."""

    verdict: AuditVerdict
    """Audit verdict per the severity rule (zero/one/multi
    criterion exceedance, plus severe-magnitude single-criterion
    promotion)."""

    timestamp_seconds: float
    """Wall-clock seconds at which the audit fired."""

    consumption_count: int
    """Number of verdict consumptions in the closed window."""

    criteria_findings: tuple[CriterionFinding, ...]
    """One :class:`CriterionFinding` per drift criterion, in the
    order declared by :data:`_CRITERION_NAMES`."""

    dispatch_id: str | None = None
    """ADR-023 correlation identifier (Cycle 6 WP-A — additive). Joins
    this audit verdict to the ``DispatchTiming`` start/end events and
    other per-dispatch events emitted through the Dispatch Event
    Substrate. ``None`` during the progressive conversion when the
    consumption call site does not yet pass the substrate's allocated
    identifier; the audit's drift-criteria semantics are unchanged
    (preservation scenario in scenarios.md §Observability Event Routing)."""


@dataclass
class _WindowAccumulator:
    """Per-window state — internal to the auditor."""

    window_started_seconds: float
    consumption_count: int = 0
    proceed_count: int = 0
    reflect_count: int = 0
    abstain_count: int = 0
    cheap_outcomes: list[bool] = field(default_factory=list)
    escalated_outcomes: list[bool] = field(default_factory=list)

    def fraction(self, count: int) -> float:
        if self.consumption_count == 0:
            return 0.0
        return count / self.consumption_count

    def proceed_fraction(self) -> float:
        return self.fraction(self.proceed_count)

    def reflect_fraction(self) -> float:
        return self.fraction(self.reflect_count)

    def abstain_fraction(self) -> float:
        return self.fraction(self.abstain_count)

    def cheap_success_rate(self) -> float | None:
        if not self.cheap_outcomes:
            return None
        return sum(1 for s in self.cheap_outcomes if s) / len(self.cheap_outcomes)

    def escalated_success_rate(self) -> float | None:
        if not self.escalated_outcomes:
            return None
        successes = sum(1 for s in self.escalated_outcomes if s)
        return successes / len(self.escalated_outcomes)


class TierEscalationAuditor:
    """Periodic out-of-band drift detector on verdict→router edge.

    Per ADR-018 §Decision. Holds all audit state internally; the
    :class:`llm_orc.agentic.tier_router.TierRouter` is unaware of the
    auditor (FC-19 preserved). The Tool Dispatch consultation point
    queries :attr:`fail_safe_active` before invoking the router and
    forwards selection to the escalated tier when fail-safe is on.
    """

    def __init__(
        self,
        *,
        thresholds: TierEscalationAuditThresholds,
        clock: AuditClock | None = None,
    ) -> None:
        self._thresholds = thresholds
        self._clock: AuditClock = clock if clock is not None else SystemAuditClock()
        self._window = _WindowAccumulator(
            window_started_seconds=self._clock.now_seconds()
        )
        self._prior_window: _WindowAccumulator | None = None
        self._diagnostics: list[AuditDiagnostic] = []
        self._fail_safe_active = False
        self._window_id = 0

    # ------------------------------------------------------------------
    # Public surface — Tool Dispatch consults these
    # ------------------------------------------------------------------

    @property
    def fail_safe_active(self) -> bool:
        """Whether severe-drift fail-safe is currently active.

        While ``True``, Tool Dispatch routes every ``invoke_ensemble``
        to the escalated tier per ADR-018 §"Severe-drift response",
        regardless of verdict. Operator clears with
        :meth:`clear_fail_safe`.
        """
        return self._fail_safe_active

    def diagnostics(self) -> tuple[AuditDiagnostic, ...]:
        """Accumulated audit diagnostics for operator review.

        Per ADR-018 §"Asynchronous-operator-review dynamic" — advisory
        diagnostics do not block dispatch flow; they accumulate for
        operator review at the next session boundary.
        """
        return tuple(self._diagnostics)

    def clear_fail_safe(self) -> None:
        """Operator action — clears fail-safe after review.

        Per ADR-018 §"Severe-drift response" the operator must
        explicitly clear fail-safe; the auditor does not self-clear
        even if subsequent audits show no_drift.
        """
        self._fail_safe_active = False

    # ------------------------------------------------------------------
    # Recording surfaces — called by Tool Dispatch
    # ------------------------------------------------------------------

    def record_consumption(
        self,
        *,
        verdict: CalibrationVerdict,
        selection: TierSelection | None,
        ensemble_name: str,
        bypassed: bool,
    ) -> AuditDiagnostic | None:
        """Record one verdict consumption; fire audit if trigger reached.

        Returns the produced :class:`AuditDiagnostic` if this call
        crossed the trigger boundary, else ``None``. Tool Dispatch
        does not need the return value — diagnostics accumulate in
        :meth:`diagnostics` regardless — but tests use it to assert
        on the specific firing.
        """
        del selection  # selection's content does not influence the audit
        del ensemble_name  # per-ensemble accounting is a Cycle 5+ surface

        self._window.consumption_count += 1
        if verdict == "proceed":
            self._window.proceed_count += 1
        elif verdict == "reflect":
            self._window.reflect_count += 1
        else:  # verdict == "abstain"
            self._window.abstain_count += 1
        # ``bypassed`` parameter retained on the API for clarity at
        # the call site; abstain verdicts already imply bypass.
        del bypassed

        if self._trigger_fired():
            return self._fire_audit()
        return None

    def record_outcome(self, *, ensemble_name: str, tier: Tier, success: bool) -> None:
        """Record one dispatch's success/failure outcome by tier.

        Tool Dispatch calls this after ``EnsembleExecutor.execute``
        completes. The outcome accumulates into the current audit
        window for the escalation-vs-outcome correlation criterion.
        """
        del ensemble_name  # per-ensemble accounting is Cycle 5+ territory
        if tier == "cheap":
            self._window.cheap_outcomes.append(success)
        else:
            self._window.escalated_outcomes.append(success)

    # ------------------------------------------------------------------
    # Test-only inspection surfaces
    # ------------------------------------------------------------------

    @property
    def outcome_snapshot_for_tests(self) -> Mapping[Tier, tuple[int, int]]:
        """Per-tier (success_count, failure_count) for the current window.

        Used only by unit tests verifying outcome accumulation. Audit
        verdicts surface the same information via
        :class:`CriterionFinding`, which is the operator-facing path.
        """
        cheap_success = sum(1 for s in self._window.cheap_outcomes if s)
        cheap_failure = len(self._window.cheap_outcomes) - cheap_success
        escalated_success = sum(1 for s in self._window.escalated_outcomes if s)
        escalated_failure = len(self._window.escalated_outcomes) - escalated_success
        return {
            "cheap": (cheap_success, cheap_failure),
            "escalated": (escalated_success, escalated_failure),
        }

    # ------------------------------------------------------------------
    # Internal — trigger and verdict computation
    # ------------------------------------------------------------------

    def _trigger_fired(self) -> bool:
        if self._window.consumption_count >= self._thresholds.trigger_count:
            return True
        wall_clock_window_seconds = self._thresholds.trigger_wall_clock_hours * 3600.0
        elapsed = self._clock.now_seconds() - self._window.window_started_seconds
        return elapsed >= wall_clock_window_seconds

    def _fire_audit(self) -> AuditDiagnostic:
        self._window_id += 1
        findings = self._evaluate_criteria()
        verdict = self._verdict_from_findings(findings)
        diagnostic = AuditDiagnostic(
            window_id=self._window_id,
            verdict=verdict,
            timestamp_seconds=self._clock.now_seconds(),
            consumption_count=self._window.consumption_count,
            criteria_findings=findings,
        )
        self._diagnostics.append(diagnostic)
        if verdict == "severe":
            self._fail_safe_active = True
        # Roll windows: current becomes prior; start fresh window.
        self._prior_window = self._window
        self._window = _WindowAccumulator(
            window_started_seconds=self._clock.now_seconds()
        )
        return diagnostic

    def _evaluate_criteria(self) -> tuple[CriterionFinding, ...]:
        return (
            self._verdict_distribution_shift_finding(),
            self._escalation_outcome_correlation_finding(),
            self._bypass_rate_trend_finding(),
        )

    def _verdict_distribution_shift_finding(self) -> CriterionFinding:
        threshold = self._thresholds.verdict_distribution_shift
        severe_threshold = threshold * self._thresholds.severe_drift_multiplier
        if self._prior_window is None:
            # No baseline — criterion cannot fire.
            return CriterionFinding(
                name="verdict_distribution_shift",
                value=0.0,
                threshold=threshold,
                exceeds=False,
                severe=False,
            )
        max_axis_shift = max(
            abs(
                self._window.proceed_fraction() - self._prior_window.proceed_fraction()
            ),
            abs(
                self._window.reflect_fraction() - self._prior_window.reflect_fraction()
            ),
            abs(
                self._window.abstain_fraction() - self._prior_window.abstain_fraction()
            ),
        )
        return CriterionFinding(
            name="verdict_distribution_shift",
            value=max_axis_shift,
            threshold=threshold,
            exceeds=max_axis_shift > threshold,
            severe=max_axis_shift >= severe_threshold,
        )

    def _escalation_outcome_correlation_finding(self) -> CriterionFinding:
        threshold = self._thresholds.escalation_outcome_correlation_pp
        severe_threshold = threshold * self._thresholds.severe_drift_multiplier
        cheap_rate = self._window.cheap_success_rate()
        escalated_rate = self._window.escalated_success_rate()
        if cheap_rate is None or escalated_rate is None:
            # Both tiers must have outcomes in the current window for
            # the criterion to be measurable.
            return CriterionFinding(
                name="escalation_outcome_correlation",
                value=0.0,
                threshold=threshold,
                exceeds=False,
                severe=False,
            )
        gap = escalated_rate - cheap_rate
        # Gap below threshold means escalation is not buying enough
        # outcome improvement to be interpretable as a tier signal.
        exceeds = gap < threshold
        # Severe when escalation is strictly worse than cheap by at
        # least the severe-magnitude cutoff (escalation actively hurts
        # — clear routing-noise signal).
        severe = gap < (threshold - severe_threshold)
        return CriterionFinding(
            name="escalation_outcome_correlation",
            value=gap,
            threshold=threshold,
            exceeds=exceeds,
            severe=severe,
        )

    def _bypass_rate_trend_finding(self) -> CriterionFinding:
        threshold = self._thresholds.bypass_rate_increase
        severe_threshold = threshold * self._thresholds.severe_drift_multiplier
        if self._prior_window is None:
            return CriterionFinding(
                name="bypass_rate_trend",
                value=0.0,
                threshold=threshold,
                exceeds=False,
                severe=False,
            )
        prior_rate = self._prior_window.abstain_fraction()
        current_rate = self._window.abstain_fraction()
        if prior_rate == 0.0:
            # Undefined baseline. Treat any nonzero current rate as
            # exceeding (the trend is real), but never severe — the
            # relative-magnitude denominator is undefined.
            exceeds = current_rate > 0.0
            return CriterionFinding(
                name="bypass_rate_trend",
                value=current_rate,
                threshold=threshold,
                exceeds=exceeds,
                severe=False,
            )
        relative_increase = (current_rate - prior_rate) / prior_rate
        return CriterionFinding(
            name="bypass_rate_trend",
            value=relative_increase,
            threshold=threshold,
            exceeds=relative_increase > threshold,
            severe=relative_increase >= severe_threshold,
        )

    @staticmethod
    def _verdict_from_findings(
        findings: tuple[CriterionFinding, ...],
    ) -> AuditVerdict:
        exceeding = [f for f in findings if f.exceeds]
        if not exceeding:
            return "no_drift"
        # Severe if any single criterion is at severe magnitude OR
        # if two or more criteria exceed simultaneously.
        if any(f.severe for f in exceeding) or len(exceeding) >= 2:
            return "severe"
        return "advisory"
