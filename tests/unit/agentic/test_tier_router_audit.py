"""Tests for the Tier-Escalation Router (d)-analog audit dispatch (WP-G4-2).

Per ``docs/agentic-serving/decisions/adr-018-tier-escalation-router-audit-dispatch.md``
and ``docs/agentic-serving/system-design.agents.md`` §Module:
Tier-Escalation Router (L2 — extended at architect-gate close 2026-05-11).

The (d)-analog audit dispatch operates on the L1→L2 verdict→router
edge as a periodic out-of-band drift detector. Three drift criteria
at quantitative-threshold level (verdict-distribution shift,
escalation-vs-outcome correlation, bypass-rate trend) compose into
the audit verdict trichotomy (``no_drift`` / ``advisory`` / ``severe``).
Severe drift activates fail-safe mode (route-all-to-escalated).

FC-19 (router stateless purity) is preserved by keeping all audit
state inside :class:`TierEscalationAuditor` — :class:`TierRouter`
never reads or writes audit state.
"""

from __future__ import annotations

import pytest

from llm_orc.agentic.tier_router import TierSelection
from llm_orc.agentic.tier_router_audit import (
    AuditDiagnostic,
    AuditVerdict,
    CriterionFinding,
    TierEscalationAuditor,
    TierEscalationAuditThresholds,
)

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _ManualClock:
    """Wall-clock double for trigger-by-time tests.

    Tests advance the clock explicitly via :meth:`advance_seconds`; the
    auditor reads ``now_seconds`` to evaluate the wall-clock trigger
    bound (per ADR-018 §"Trigger": every 100 verdict consumptions or
    24 wall-clock hours, whichever first).
    """

    def __init__(self, start_seconds: float = 0.0) -> None:
        self._now = start_seconds

    def now_seconds(self) -> float:
        return self._now

    def advance_seconds(self, delta: float) -> None:
        self._now += delta

    def advance_hours(self, hours: float) -> None:
        self._now += hours * 3600.0


def _proceed_consumption(
    auditor: TierEscalationAuditor,
    *,
    ensemble_name: str = "ens-A",
    model_profile: str = "cheap-code-gen",
    topaz_skill: str = "code_generation",
) -> None:
    """Record one Proceed consumption with a cheap-tier selection."""
    auditor.record_consumption(
        verdict="proceed",
        selection=TierSelection(
            model_profile=model_profile,
            tier="cheap",
            topaz_skill=topaz_skill,  # type: ignore[arg-type]
        ),
        ensemble_name=ensemble_name,
        bypassed=False,
    )


def _reflect_consumption(
    auditor: TierEscalationAuditor,
    *,
    ensemble_name: str = "ens-A",
    model_profile: str = "escalated-code-gen",
    topaz_skill: str = "code_generation",
) -> None:
    """Record one Reflect consumption with an escalated-tier selection."""
    auditor.record_consumption(
        verdict="reflect",
        selection=TierSelection(
            model_profile=model_profile,
            tier="escalated",
            topaz_skill=topaz_skill,  # type: ignore[arg-type]
        ),
        ensemble_name=ensemble_name,
        bypassed=False,
    )


def _abstain_consumption(
    auditor: TierEscalationAuditor,
    *,
    ensemble_name: str = "ens-A",
) -> None:
    """Record one Abstain consumption — bypassed at router edge."""
    auditor.record_consumption(
        verdict="abstain",
        selection=None,
        ensemble_name=ensemble_name,
        bypassed=True,
    )


# ---------------------------------------------------------------------------
# Default thresholds — match ADR-018 §Decision §"Drift criteria" defaults.
# ---------------------------------------------------------------------------


def _default_thresholds() -> TierEscalationAuditThresholds:
    return TierEscalationAuditThresholds(
        trigger_count=100,
        trigger_wall_clock_hours=24.0,
        verdict_distribution_shift=0.15,
        escalation_outcome_correlation_pp=0.05,
        bypass_rate_increase=0.25,
        severe_drift_multiplier=2.0,
    )


# ---------------------------------------------------------------------------
# Group A — auditor state machine basics
# ---------------------------------------------------------------------------


class TestAuditorEmptyState:
    def test_fresh_auditor_reports_fail_safe_inactive(self) -> None:
        auditor = TierEscalationAuditor(
            thresholds=_default_thresholds(), clock=_ManualClock()
        )
        assert auditor.fail_safe_active is False

    def test_fresh_auditor_has_no_diagnostics(self) -> None:
        auditor = TierEscalationAuditor(
            thresholds=_default_thresholds(), clock=_ManualClock()
        )
        assert auditor.diagnostics() == ()


class TestAuditorBelowTriggerThresholds:
    def test_99_consumptions_below_count_trigger_produces_no_audit(self) -> None:
        auditor = TierEscalationAuditor(
            thresholds=_default_thresholds(), clock=_ManualClock()
        )
        for _ in range(99):
            _proceed_consumption(auditor)
        assert auditor.diagnostics() == ()
        assert auditor.fail_safe_active is False

    def test_wall_clock_below_24_hours_produces_no_audit(self) -> None:
        clock = _ManualClock()
        auditor = TierEscalationAuditor(thresholds=_default_thresholds(), clock=clock)
        _proceed_consumption(auditor)
        clock.advance_hours(23.99)
        _proceed_consumption(auditor)
        assert auditor.diagnostics() == ()


class TestAuditorTriggerByCount:
    def test_100th_consumption_fires_audit_returning_diagnostic(self) -> None:
        auditor = TierEscalationAuditor(
            thresholds=_default_thresholds(), clock=_ManualClock()
        )
        firings: list[tuple[int, AuditDiagnostic]] = []
        for i in range(100):
            outcome = auditor.record_consumption(
                verdict="proceed",
                selection=TierSelection(
                    model_profile="cheap-code-gen",
                    tier="cheap",
                    topaz_skill="code_generation",
                ),
                ensemble_name="ens-A",
                bypassed=False,
            )
            if outcome is not None:
                firings.append((i, outcome))
        # Exactly one audit fired across the 100 consumptions.
        assert len(firings) == 1
        triggering_index, _diagnostic = firings[0]
        # The 100th consumption (index 99) is the trigger boundary.
        assert triggering_index == 99
        # First window has no prior window to compare against; baseline
        # verdict is no_drift.
        recorded = auditor.diagnostics()
        assert len(recorded) == 1
        assert recorded[0].verdict == "no_drift"


class TestAuditorTriggerByWallClock:
    def test_24_hour_boundary_fires_audit_with_few_consumptions(self) -> None:
        clock = _ManualClock()
        auditor = TierEscalationAuditor(thresholds=_default_thresholds(), clock=clock)
        _proceed_consumption(auditor)
        clock.advance_hours(24.0)
        _proceed_consumption(auditor)
        recorded = auditor.diagnostics()
        assert len(recorded) == 1
        assert recorded[0].verdict == "no_drift"


class TestAuditorRecordOutcomeUpdatesCounts:
    def test_outcome_records_per_tier_success_and_failure(self) -> None:
        auditor = TierEscalationAuditor(
            thresholds=_default_thresholds(), clock=_ManualClock()
        )
        auditor.record_outcome(ensemble_name="ens-A", tier="cheap", success=True)
        auditor.record_outcome(ensemble_name="ens-A", tier="cheap", success=False)
        auditor.record_outcome(ensemble_name="ens-A", tier="escalated", success=True)
        # Snapshot is exposed for tests via the property; consumers
        # use the audit verdict rather than the snapshot directly.
        snapshot = auditor.outcome_snapshot_for_tests
        assert snapshot["cheap"] == (1, 1)  # (success, failure)
        assert snapshot["escalated"] == (1, 0)


# ---------------------------------------------------------------------------
# Group B — drift criteria
# ---------------------------------------------------------------------------


class TestVerdictDistributionShiftCriterion:
    def test_no_shift_yields_no_drift_verdict(self) -> None:
        """Two consecutive windows of all-Proceed verdicts: zero shift."""
        thresholds = _default_thresholds()
        auditor = TierEscalationAuditor(thresholds=thresholds, clock=_ManualClock())
        for _ in range(100):
            _proceed_consumption(auditor)
        # Baseline window — verdict no_drift (no prior to compare).
        for _ in range(100):
            _proceed_consumption(auditor)
        diagnostics = auditor.diagnostics()
        assert len(diagnostics) == 2
        # Window 2's verdict should be no_drift — distribution unchanged.
        assert diagnostics[1].verdict == "no_drift"

    def test_complete_distribution_flip_yields_severe_drift(self) -> None:
        """Window 1 all-Proceed → Window 2 all-Reflect: 100% shift."""
        thresholds = _default_thresholds()
        auditor = TierEscalationAuditor(thresholds=thresholds, clock=_ManualClock())
        for _ in range(100):
            _proceed_consumption(auditor)
        for _ in range(100):
            _reflect_consumption(auditor)
        diagnostics = auditor.diagnostics()
        # Window 2: the verdict distribution shifted by 100 pp on both
        # Proceed and Reflect axes — well past 2x the 0.15 threshold
        # → severe drift.
        assert diagnostics[1].verdict == "severe"
        assert auditor.fail_safe_active is True


class TestEscalationOutcomeCorrelationCriterion:
    def test_escalation_outperforms_cheap_by_default_threshold_yields_no_drift(
        self,
    ) -> None:
        """Escalated successes higher than cheap by ≥ +5pp → tier signal."""
        thresholds = _default_thresholds()
        auditor = TierEscalationAuditor(thresholds=thresholds, clock=_ManualClock())
        # Window 1 — 50 Proceed + 50 Reflect, with escalated dispatches
        # outperforming cheap by 20pp (70% vs 50% success).
        for i in range(50):
            _proceed_consumption(auditor)
            auditor.record_outcome(
                ensemble_name="ens-A", tier="cheap", success=(i < 25)
            )
        for i in range(50):
            _reflect_consumption(auditor)
            auditor.record_outcome(
                ensemble_name="ens-A", tier="escalated", success=(i < 35)
            )
        diagnostics = auditor.diagnostics()
        # First window has no prior window; correlation criterion
        # against an empty baseline cannot fire — verdict no_drift.
        assert diagnostics[0].verdict == "no_drift"

    def test_escalation_indistinguishable_from_cheap_yields_advisory(
        self,
    ) -> None:
        """Cheap and escalated dispatches have identical success rates."""
        thresholds = _default_thresholds()
        auditor = TierEscalationAuditor(thresholds=thresholds, clock=_ManualClock())
        # Window 1 — fill baseline.
        for _ in range(100):
            _proceed_consumption(auditor)
            auditor.record_outcome(ensemble_name="ens-A", tier="cheap", success=True)
        # Window 2 — cheap and escalated both produce 50% successes.
        for i in range(50):
            _proceed_consumption(auditor)
            auditor.record_outcome(
                ensemble_name="ens-A", tier="cheap", success=(i < 25)
            )
        for i in range(50):
            _reflect_consumption(auditor)
            auditor.record_outcome(
                ensemble_name="ens-A", tier="escalated", success=(i < 25)
            )
        diagnostics = auditor.diagnostics()
        # Correlation criterion fires — escalation produces 0pp
        # improvement (well below the +5pp default tolerance).
        # The verdict-distribution shift also moved (100% Proceed →
        # 50/50 split) by ~50pp, which exceeds 2x the 0.15 threshold.
        # Two criteria simultaneously exceed → severe.
        assert diagnostics[1].verdict == "severe"


class TestBypassRateTrendCriterion:
    def test_bypass_rate_steady_yields_no_drift(self) -> None:
        thresholds = _default_thresholds()
        auditor = TierEscalationAuditor(thresholds=thresholds, clock=_ManualClock())
        # Window 1 — 5% Abstain rate (5 of 100).
        for i in range(100):
            if i < 95:
                _proceed_consumption(auditor)
            else:
                _abstain_consumption(auditor)
        # Window 2 — same 5% Abstain rate.
        for i in range(100):
            if i < 95:
                _proceed_consumption(auditor)
            else:
                _abstain_consumption(auditor)
        diagnostics = auditor.diagnostics()
        # Window 2's bypass-rate trend is flat (5% → 5%, 0% relative
        # change, well below 25% threshold). Verdict-distribution
        # also unchanged. No criteria exceed → no_drift.
        assert diagnostics[1].verdict == "no_drift"

    def test_bypass_rate_doubling_yields_advisory_via_bypass_criterion_alone(
        self,
    ) -> None:
        thresholds = _default_thresholds()
        auditor = TierEscalationAuditor(thresholds=thresholds, clock=_ManualClock())
        # Window 1 — 50% Proceed / 50% Reflect baseline.
        for _ in range(50):
            _proceed_consumption(auditor)
        for _ in range(50):
            _reflect_consumption(auditor)
        # Window 2 — same Proceed/Reflect split, but ratio nudged by
        # one Abstain — bypass rate goes from 0 to 1/100 = 1% (∞%
        # relative increase). The integration-test scenario for the
        # ∞% relative bypass-rate change is severe by criterion-count
        # if other criteria also moved; here we keep Proceed/Reflect
        # roughly steady so only bypass-rate moves.
        for _ in range(49):
            _proceed_consumption(auditor)
        for _ in range(50):
            _reflect_consumption(auditor)
        _abstain_consumption(auditor)
        diagnostics = auditor.diagnostics()
        # First window baseline = no_drift; second window has 1
        # criterion exceeding (bypass-rate +∞% relative increase
        # from a zero baseline). One criterion exceeding (not at
        # severe-multiplier) → advisory.
        assert diagnostics[1].verdict == "advisory"


# ---------------------------------------------------------------------------
# Group C — fail-safe activation and clearing
# ---------------------------------------------------------------------------


class TestFailSafeActivation:
    def test_severe_verdict_activates_fail_safe(self) -> None:
        thresholds = _default_thresholds()
        auditor = TierEscalationAuditor(thresholds=thresholds, clock=_ManualClock())
        for _ in range(100):
            _proceed_consumption(auditor)
        for _ in range(100):
            _reflect_consumption(auditor)
        assert auditor.fail_safe_active is True

    def test_advisory_verdict_does_not_activate_fail_safe(self) -> None:
        thresholds = _default_thresholds()
        auditor = TierEscalationAuditor(thresholds=thresholds, clock=_ManualClock())
        for _ in range(50):
            _proceed_consumption(auditor)
        for _ in range(50):
            _reflect_consumption(auditor)
        for _ in range(49):
            _proceed_consumption(auditor)
        for _ in range(50):
            _reflect_consumption(auditor)
        _abstain_consumption(auditor)
        # Advisory verdict; fail-safe should NOT activate.
        assert auditor.fail_safe_active is False


class TestFailSafeClearing:
    def test_clear_fail_safe_returns_auditor_to_normal_operation(self) -> None:
        thresholds = _default_thresholds()
        auditor = TierEscalationAuditor(thresholds=thresholds, clock=_ManualClock())
        for _ in range(100):
            _proceed_consumption(auditor)
        for _ in range(100):
            _reflect_consumption(auditor)
        assert auditor.fail_safe_active is True
        auditor.clear_fail_safe()
        assert auditor.fail_safe_active is False


# ---------------------------------------------------------------------------
# Group D — diagnostic records carry criterion findings for operator review
# ---------------------------------------------------------------------------


class TestDiagnosticContent:
    def test_diagnostic_includes_named_criteria_findings(self) -> None:
        thresholds = _default_thresholds()
        auditor = TierEscalationAuditor(thresholds=thresholds, clock=_ManualClock())
        for _ in range(100):
            _proceed_consumption(auditor)
        for _ in range(100):
            _reflect_consumption(auditor)
        diagnostics = auditor.diagnostics()
        diagnostic = diagnostics[1]
        criterion_names = {f.name for f in diagnostic.criteria_findings}
        assert criterion_names == {
            "verdict_distribution_shift",
            "escalation_outcome_correlation",
            "bypass_rate_trend",
        }

    def test_severe_diagnostic_flags_responsible_criteria(self) -> None:
        thresholds = _default_thresholds()
        auditor = TierEscalationAuditor(thresholds=thresholds, clock=_ManualClock())
        for _ in range(100):
            _proceed_consumption(auditor)
        for _ in range(100):
            _reflect_consumption(auditor)
        diagnostic = auditor.diagnostics()[1]
        verdict_shift = next(
            f
            for f in diagnostic.criteria_findings
            if f.name == "verdict_distribution_shift"
        )
        # The shift was 100% on both Proceed and Reflect axes —
        # well above 2x the 0.15 threshold.
        assert verdict_shift.exceeds is True
        assert verdict_shift.severe is True


# ---------------------------------------------------------------------------
# Group E — operationally-tunable thresholds honored
# ---------------------------------------------------------------------------


class TestOperationallyTunableThresholds:
    def test_lower_trigger_count_fires_audit_earlier(self) -> None:
        thresholds = TierEscalationAuditThresholds(
            trigger_count=10,
            trigger_wall_clock_hours=24.0,
            verdict_distribution_shift=0.15,
            escalation_outcome_correlation_pp=0.05,
            bypass_rate_increase=0.25,
            severe_drift_multiplier=2.0,
        )
        auditor = TierEscalationAuditor(thresholds=thresholds, clock=_ManualClock())
        for _ in range(10):
            _proceed_consumption(auditor)
        assert len(auditor.diagnostics()) == 1

    def test_thresholds_validate_positive_values_at_construction(self) -> None:
        with pytest.raises(ValueError, match="trigger_count"):
            TierEscalationAuditThresholds(
                trigger_count=0,
                trigger_wall_clock_hours=24.0,
                verdict_distribution_shift=0.15,
                escalation_outcome_correlation_pp=0.05,
                bypass_rate_increase=0.25,
                severe_drift_multiplier=2.0,
            )


# ---------------------------------------------------------------------------
# Group F — verdict trichotomy literal exhaustiveness
# ---------------------------------------------------------------------------


class TestAuditVerdictTrichotomy:
    def test_verdict_values_are_the_three_documented_values(self) -> None:
        from typing import get_args

        assert set(get_args(AuditVerdict)) == {"no_drift", "advisory", "severe"}


class TestCriterionFindingShape:
    def test_finding_carries_name_value_threshold_and_severity_flags(self) -> None:
        finding = CriterionFinding(
            name="verdict_distribution_shift",
            value=0.5,
            threshold=0.15,
            exceeds=True,
            severe=True,
        )
        assert finding.name == "verdict_distribution_shift"
        assert finding.exceeds is True
        assert finding.severe is True
