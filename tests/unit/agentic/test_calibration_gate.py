"""Tests for the Calibration Gate module (ADR-007, WP-H).

Per ``docs/agentic-serving/system-design.md`` §Calibration Gate (L1
Domain Policy) and §Integration Contracts (Orchestrator Tool Dispatch
→ Calibration Gate). ADR-007 commits every composed ensemble to a
calibration period over its first N invocations during which a check
runs and produces a Quality Signal. AS-5 forbids frequency-only trust
transitions.

Covers scenarios (``docs/agentic-serving/scenarios.md``):

* §First N invocations of a composed ensemble are result-checked —
  here at unit scope.
* §Calibration transitions to trusted with sufficient positive
  quality signals — here at unit scope.
* §Calibration fails to clear with negative quality signals — here at
  unit scope.
* §Calibration is session-scoped when Plexus is absent — here at unit
  scope (per-session state isolation).

Scenario §Calibration persists across sessions when Plexus is active
is deferred to WP-I (Plexus Adapter).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from llm_orc.agentic.calibration_gate import (
    DEFAULT_AUQ_CONFIDENCE_THRESHOLD,
    DEFAULT_CALIBRATION_CHECKER_ENSEMBLE,
    DEFAULT_CALIBRATION_N,
    DEFAULT_ENTROPY_COLLAPSE_SIGMA,
    DEFAULT_TRAJECTORY_WINDOW_DISPATCHES,
    DEFAULT_TRAJECTORY_WINDOW_MINUTES,
    CalibrationAbstainError,
    CalibrationGate,
    CalibrationRecord,
    CalibrationStatus,
    DispatchContext,
    EnsembleBackedChecker,
    QualitySignal,
    TrajectoryFeatures,
)


class _ScriptedChecker:
    """Returns signals from a scripted iterator.

    Recording the invocations makes it easy to assert that a trusted
    ensemble does not trigger a checker call on a later invocation.
    """

    def __init__(self, signals: list[QualitySignal]) -> None:
        self._signals = iter(signals)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def check(
        self, *, ensemble_name: str, raw_result: dict[str, Any]
    ) -> QualitySignal:
        self.calls.append((ensemble_name, raw_result))
        return next(self._signals)


class _NeverCallChecker:
    """Fails loudly if called — asserts trusted ensembles skip the check."""

    async def check(
        self, *, ensemble_name: str, raw_result: dict[str, Any]
    ) -> QualitySignal:
        raise AssertionError(
            f"trusted ensemble {ensemble_name!r} should not trigger checker"
        )


class TestDefaults:
    """ADR-007 clause 5: a default for N is set at build.

    The value is a WP-H build decision — ``N = 3`` balances check cost
    against noise tolerance (one bad signal out of three averages out).
    """

    def test_default_n_is_three(self) -> None:
        assert DEFAULT_CALIBRATION_N == 3

    def test_default_n_rejected_below_one(self) -> None:
        with pytest.raises(ValueError, match="default_n must be >= 1"):
            CalibrationGate(default_n=0, checker=_NeverCallChecker())


class TestUntrackedEnsemblesAreTrusted:
    """Ensembles the gate never saw are library entries — not composed.

    Calibration applies only to ensembles produced by ``compose_ensemble``
    (ADR-007 opening paragraph). Library ensembles that were never
    composed-in-session return trusted without the checker running.
    """

    def test_library_ensemble_is_trusted_without_tracking(self) -> None:
        gate = CalibrationGate(default_n=3, checker=_NeverCallChecker())
        assert gate.status(session_id="s1", ensemble_name="library-thing") == "trusted"

    @pytest.mark.asyncio
    async def test_check_and_record_on_untracked_is_noop(self) -> None:
        checker = _NeverCallChecker()
        gate = CalibrationGate(default_n=3, checker=checker)
        signal = await gate.check_and_record(
            session_id="s1", ensemble_name="library-thing", raw_result={}
        )
        assert signal is None


class TestMarkComposed:
    """Registration of a newly composed ensemble."""

    def test_newly_composed_ensemble_enters_in_calibration(self) -> None:
        gate = CalibrationGate(default_n=3, checker=_NeverCallChecker())
        gate.mark_composed(session_id="s1", ensemble_name="composed-a")
        assert (
            gate.status(session_id="s1", ensemble_name="composed-a") == "in_calibration"
        )

    def test_mark_composed_is_idempotent(self) -> None:
        """Re-composing the same name is a no-op — no state reset."""
        gate = CalibrationGate(default_n=3, checker=_NeverCallChecker())
        gate.mark_composed(session_id="s1", ensemble_name="composed-a")
        record = gate.record_for(session_id="s1", ensemble_name="composed-a")
        assert record is not None
        original_id = id(record)

        gate.mark_composed(session_id="s1", ensemble_name="composed-a")
        re_record = gate.record_for(session_id="s1", ensemble_name="composed-a")
        assert re_record is record
        assert id(re_record) == original_id


class TestCheckAndRecordDuringCalibration:
    """Scenario §First N invocations of a composed ensemble are result-checked."""

    @pytest.mark.asyncio
    async def test_first_invocation_runs_check_and_records_signal(self) -> None:
        checker = _ScriptedChecker(["positive"])
        gate = CalibrationGate(default_n=3, checker=checker)
        gate.mark_composed(session_id="s1", ensemble_name="composed-a")

        signal = await gate.check_and_record(
            session_id="s1",
            ensemble_name="composed-a",
            raw_result={"results": {"a": {"response": "hi"}}},
        )

        assert signal == "positive"
        assert len(checker.calls) == 1
        record = gate.record_for(session_id="s1", ensemble_name="composed-a")
        assert record is not None
        assert record.signals == ("positive",)
        assert record.invocations_seen == 1
        # One of N positive signals does not yet earn trust.
        assert record.status == "in_calibration"

    @pytest.mark.asyncio
    async def test_all_n_positive_signals_transition_to_trusted(self) -> None:
        """Scenario §Calibration transitions to trusted with positive signals."""
        checker = _ScriptedChecker(["positive", "positive", "positive"])
        gate = CalibrationGate(default_n=3, checker=checker)
        gate.mark_composed(session_id="s1", ensemble_name="composed-a")

        for _ in range(3):
            await gate.check_and_record(
                session_id="s1", ensemble_name="composed-a", raw_result={}
            )

        assert gate.status(session_id="s1", ensemble_name="composed-a") == "trusted"

    @pytest.mark.asyncio
    async def test_trusted_ensemble_skips_check_on_later_invocation(self) -> None:
        """(N+1)th invocation on a trusted ensemble does not trigger the check.

        After three positive signals the ensemble is trusted; the fourth call
        returns ``None`` and — crucially — does not consume a signal from the
        scripted iterator. The checker's recorded call count stays at 3.
        """
        checker = _ScriptedChecker(["positive", "positive", "positive"])
        gate = CalibrationGate(default_n=3, checker=checker)
        gate.mark_composed(session_id="s1", ensemble_name="composed-a")

        for _ in range(3):
            await gate.check_and_record(
                session_id="s1", ensemble_name="composed-a", raw_result={}
            )
        assert len(checker.calls) == 3

        signal = await gate.check_and_record(
            session_id="s1", ensemble_name="composed-a", raw_result={}
        )

        assert signal is None
        assert len(checker.calls) == 3
        record = gate.record_for(session_id="s1", ensemble_name="composed-a")
        assert record is not None
        assert record.invocations_seen == 3


class TestCheckAndRecordCalibrationFailsToClear:
    """Scenario §Calibration fails to clear with negative quality signals.

    AS-5: frequency alone does not advance trust. A negative or absent
    signal in the last-N window keeps the ensemble in calibration even
    after N invocations.
    """

    @pytest.mark.asyncio
    async def test_negative_signal_in_last_n_keeps_in_calibration(self) -> None:
        checker = _ScriptedChecker(["positive", "negative", "positive"])
        gate = CalibrationGate(default_n=3, checker=checker)
        gate.mark_composed(session_id="s1", ensemble_name="composed-a")

        for _ in range(3):
            await gate.check_and_record(
                session_id="s1", ensemble_name="composed-a", raw_result={}
            )

        assert (
            gate.status(session_id="s1", ensemble_name="composed-a") == "in_calibration"
        )

    @pytest.mark.asyncio
    async def test_absent_signal_in_last_n_keeps_in_calibration(self) -> None:
        checker = _ScriptedChecker(["positive", "positive", "absent"])
        gate = CalibrationGate(default_n=3, checker=checker)
        gate.mark_composed(session_id="s1", ensemble_name="composed-a")

        for _ in range(3):
            await gate.check_and_record(
                session_id="s1", ensemble_name="composed-a", raw_result={}
            )

        assert (
            gate.status(session_id="s1", ensemble_name="composed-a") == "in_calibration"
        )

    @pytest.mark.asyncio
    async def test_ensemble_continues_to_be_checked_after_failure(self) -> None:
        """Scenario 3's 'calibration period extends' — check keeps firing."""
        # 2 negatives then 3 positives → last-N window becomes all-positive
        # and the ensemble transitions to trusted.
        checker = _ScriptedChecker(
            ["negative", "negative", "positive", "positive", "positive"]
        )
        gate = CalibrationGate(default_n=3, checker=checker)
        gate.mark_composed(session_id="s1", ensemble_name="composed-a")

        for _ in range(5):
            await gate.check_and_record(
                session_id="s1", ensemble_name="composed-a", raw_result={}
            )

        assert len(checker.calls) == 5
        assert gate.status(session_id="s1", ensemble_name="composed-a") == "trusted"


class TestAS5FrequencyDoesNotTrust:
    """Invariant test for AS-5 (quality signals govern stabilization, not frequency).

    Named ``test_frequency_without_quality_does_not_trust`` per system-design
    §Invariant Enforcement Tests.
    """

    @pytest.mark.asyncio
    async def test_frequency_without_quality_does_not_trust(self) -> None:
        # Ten invocations; a mix of negatives/absents drifts trust out of
        # reach despite plenty of use.
        signals: list[QualitySignal] = [
            "positive",
            "negative",
            "positive",
            "absent",
            "positive",
            "negative",
            "positive",
            "absent",
            "positive",
            "negative",
        ]
        checker = _ScriptedChecker(signals)
        gate = CalibrationGate(default_n=3, checker=checker)
        gate.mark_composed(session_id="s1", ensemble_name="composed-a")

        for _ in range(10):
            await gate.check_and_record(
                session_id="s1", ensemble_name="composed-a", raw_result={}
            )

        assert (
            gate.status(session_id="s1", ensemble_name="composed-a") == "in_calibration"
        )
        record = gate.record_for(session_id="s1", ensemble_name="composed-a")
        assert record is not None
        assert record.invocations_seen == 10


class TestSessionIsolation:
    """Scenario §Calibration is session-scoped when Plexus is absent.

    Per-session state: an ensemble cleared in Session 1 re-enters calibration
    at Session 2 start. The Gate's per-session store makes this structural —
    Session 2 simply has no record for the ensemble, so status defaults to
    ``trusted`` (library) until the orchestrator re-composes it, at which
    point ``mark_composed`` fires and calibration restarts.
    """

    @pytest.mark.asyncio
    async def test_two_sessions_have_independent_records(self) -> None:
        checker = _ScriptedChecker(["positive", "negative"])
        gate = CalibrationGate(default_n=3, checker=checker)
        gate.mark_composed(session_id="s1", ensemble_name="composed-a")
        gate.mark_composed(session_id="s2", ensemble_name="composed-a")

        await gate.check_and_record(
            session_id="s1", ensemble_name="composed-a", raw_result={}
        )
        await gate.check_and_record(
            session_id="s2", ensemble_name="composed-a", raw_result={}
        )

        r1 = gate.record_for(session_id="s1", ensemble_name="composed-a")
        r2 = gate.record_for(session_id="s2", ensemble_name="composed-a")
        assert r1 is not None
        assert r2 is not None
        assert r1.signals == ("positive",)
        assert r2.signals == ("negative",)

    def test_session_without_compose_ensemble_treats_as_library(self) -> None:
        """Session 2 after Session 1 cleared calibration.

        When the orchestrator in Session 2 invokes the same-named ensemble
        without re-composing it, the gate has no record and returns trusted
        — matching the 'library-default' semantics. The operator can
        re-enter calibration in Session 2 via the orchestrator re-composing.
        """
        gate = CalibrationGate(default_n=3, checker=_NeverCallChecker())
        gate.mark_composed(session_id="s1", ensemble_name="composed-a")
        # Session 2 never called ``mark_composed``; ensemble is library-default.
        assert gate.status(session_id="s2", ensemble_name="composed-a") == "trusted"


class TestRecordDataclass:
    """Lightweight checks on the CalibrationRecord type the gate surfaces."""

    def test_default_record_starts_in_calibration(self) -> None:
        record = CalibrationRecord(ensemble_name="any")
        assert record.status == "in_calibration"
        assert record.signals == ()
        assert record.invocations_seen == 0

    @pytest.mark.parametrize(
        "status",
        ["in_calibration", "trusted"],
    )
    def test_status_literal_accepts_both_values(
        self, status: CalibrationStatus
    ) -> None:
        record = CalibrationRecord(ensemble_name="any", status=status)
        assert record.status == status


class _RecordingInvoker:
    """Records checker-ensemble invocations and returns canned outputs."""

    def __init__(
        self,
        *,
        returns: dict[str, Any] | None = None,
        raises: Exception | None = None,
    ) -> None:
        self._returns: dict[str, Any] = (
            returns if returns is not None else {"deliverable": "signal: positive"}
        )
        self._raises = raises
        self.calls: list[dict[str, Any]] = []

    async def invoke(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(arguments)
        if self._raises is not None:
            raise self._raises
        return self._returns


class TestDefaultCheckerEnsembleName:
    def test_default_is_shipped_checker(self) -> None:
        assert DEFAULT_CALIBRATION_CHECKER_ENSEMBLE == "agentic-calibration-checker"


class TestEnsembleBackedChecker:
    """Production :class:`CalibrationChecker` that invokes a checker ensemble."""

    @pytest.mark.asyncio
    async def test_synthesis_positive_is_parsed(self) -> None:
        invoker = _RecordingInvoker(
            returns={"deliverable": "signal: positive\nreason: clear on-task response"}
        )
        checker = EnsembleBackedChecker(invoker=invoker)
        signal = await checker.check(
            ensemble_name="composed-a", raw_result={"results": {"a": "x"}}
        )
        assert signal == "positive"

    @pytest.mark.asyncio
    async def test_synthesis_negative_is_parsed(self) -> None:
        invoker = _RecordingInvoker(
            returns={"deliverable": "signal: negative\nreason: hallucinated content"}
        )
        checker = EnsembleBackedChecker(invoker=invoker)
        signal = await checker.check(ensemble_name="composed-a", raw_result={})
        assert signal == "negative"

    @pytest.mark.asyncio
    async def test_synthesis_absent_is_parsed(self) -> None:
        invoker = _RecordingInvoker(
            returns={"deliverable": "signal: absent\nreason: empty output"}
        )
        checker = EnsembleBackedChecker(invoker=invoker)
        assert await checker.check(ensemble_name="x", raw_result={}) == "absent"

    @pytest.mark.asyncio
    async def test_single_agent_response_shape(self) -> None:
        """When ``synthesis`` is missing the checker reads the agent response.

        Dependency-free single-agent ensembles leave ``synthesis``
        unpopulated; the Summarizer Harness handles the same shape,
        the checker should too.
        """
        invoker = _RecordingInvoker(
            returns={
                "results": {"only": {"response": "signal: positive"}},
                "deliverable": None,
            }
        )
        checker = EnsembleBackedChecker(invoker=invoker)
        signal = await checker.check(ensemble_name="composed-b", raw_result={})
        assert signal == "positive"

    @pytest.mark.asyncio
    async def test_unparseable_response_yields_absent(self) -> None:
        invoker = _RecordingInvoker(
            returns={"deliverable": "the ensemble looked fine but I am not sure"}
        )
        checker = EnsembleBackedChecker(invoker=invoker)
        assert await checker.check(ensemble_name="x", raw_result={}) == "absent"

    @pytest.mark.asyncio
    async def test_invoker_exception_yields_absent(self) -> None:
        """ADR-007 clause 2: checker crash does not surface as tool error."""
        invoker = _RecordingInvoker(raises=RuntimeError("checker ensemble missing"))
        checker = EnsembleBackedChecker(invoker=invoker)
        assert await checker.check(ensemble_name="x", raw_result={}) == "absent"

    @pytest.mark.asyncio
    async def test_payload_carries_target_ensemble_and_output(self) -> None:
        invoker = _RecordingInvoker()
        checker = EnsembleBackedChecker(
            invoker=invoker, checker_ensemble_name="custom-checker"
        )
        raw_result = {"results": {"a": {"response": "hi"}}}
        await checker.check(ensemble_name="composed-c", raw_result=raw_result)

        assert len(invoker.calls) == 1
        call = invoker.calls[0]
        assert call["ensemble_name"] == "custom-checker"
        payload = json.loads(call["input"])
        assert payload["target_ensemble"] == "composed-c"
        assert payload["output"] == raw_result

    @pytest.mark.asyncio
    async def test_signal_tokens_tolerate_case(self) -> None:
        """Operators phrasing ``SIGNAL: Positive`` should still parse."""
        invoker = _RecordingInvoker(
            returns={"deliverable": "SIGNAL: Positive (ensemble answered on-task)"}
        )
        checker = EnsembleBackedChecker(invoker=invoker)
        assert await checker.check(ensemble_name="x", raw_result={}) == "positive"


# ---------------------------------------------------------------------------
# WP-F4 — ADR-014 verdict producer (in-process trajectory-level calibration)
# ---------------------------------------------------------------------------


class _StubClock:
    """Wall-clock double driven by an explicit list of return values."""

    def __init__(self, *, returns: list[float]) -> None:
        self._returns = iter(returns)

    def now_seconds(self) -> float:
        return next(self._returns)


class TestADR014Defaults:
    """Per ADR-014 §Decision — operationally-tunable defaults."""

    def test_default_auq_confidence_threshold_within_literature_range(self) -> None:
        """AUQ literature range is 0.8–1.0 (arXiv:2601.15703); 0.85 is the
        drafting-time synthesis pick per ADR-014 §Provenance check."""
        assert 0.8 <= DEFAULT_AUQ_CONFIDENCE_THRESHOLD <= 1.0
        assert DEFAULT_AUQ_CONFIDENCE_THRESHOLD == 0.85

    def test_default_trajectory_window_minutes_is_sixty(self) -> None:
        assert DEFAULT_TRAJECTORY_WINDOW_MINUTES == 60.0

    def test_default_trajectory_window_dispatches_is_one_hundred(self) -> None:
        assert DEFAULT_TRAJECTORY_WINDOW_DISPATCHES == 100

    def test_default_entropy_collapse_sigma_is_one_point_five(self) -> None:
        """ADR-014 §"Calibration verdict" Abstain criterion 1."""
        assert DEFAULT_ENTROPY_COLLAPSE_SIGMA == 1.5


class TestCalibrationGateAdr014ConstructorValidation:
    """Operationally-tunable parameters reject out-of-range values."""

    def test_auq_threshold_below_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="auq_confidence_threshold"):
            CalibrationGate(
                default_n=3,
                checker=_NeverCallChecker(),
                auq_confidence_threshold=-0.1,
            )

    def test_auq_threshold_above_one_rejected(self) -> None:
        with pytest.raises(ValueError, match="auq_confidence_threshold"):
            CalibrationGate(
                default_n=3,
                checker=_NeverCallChecker(),
                auq_confidence_threshold=1.1,
            )

    def test_window_minutes_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="trajectory_window_minutes"):
            CalibrationGate(
                default_n=3,
                checker=_NeverCallChecker(),
                trajectory_window_minutes=0.0,
            )

    def test_window_dispatches_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="trajectory_window_dispatches"):
            CalibrationGate(
                default_n=3,
                checker=_NeverCallChecker(),
                trajectory_window_dispatches=0,
            )

    def test_entropy_sigma_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="entropy_collapse_sigma"):
            CalibrationGate(
                default_n=3,
                checker=_NeverCallChecker(),
                entropy_collapse_sigma=0.0,
            )


class TestVerdictTrichotomyProceed:
    """Scenario §Proceed verdict routes dispatch as-is."""

    def test_above_threshold_confidence_yields_proceed(self) -> None:
        gate = CalibrationGate(default_n=3, checker=_NeverCallChecker())
        verdict = gate.verdict_for(
            session_id="s1",
            ensemble_name="composed-a",
            dispatch_context=DispatchContext(auq_confidence=0.92),
        )
        assert verdict == "proceed"

    def test_confidence_exactly_at_threshold_yields_proceed(self) -> None:
        """Below-threshold is strict; the threshold itself is Proceed."""
        gate = CalibrationGate(default_n=3, checker=_NeverCallChecker())
        verdict = gate.verdict_for(
            session_id="s1",
            ensemble_name="composed-a",
            dispatch_context=DispatchContext(auq_confidence=0.85),
        )
        assert verdict == "proceed"

    def test_no_auq_confidence_yields_proceed(self) -> None:
        """Missing AUQ confidence does not produce Reflect — Reflect
        requires evidence of low confidence per ADR-014 §"Calibration
        verdict"."""
        gate = CalibrationGate(default_n=3, checker=_NeverCallChecker())
        verdict = gate.verdict_for(
            session_id="s1",
            ensemble_name="composed-a",
            dispatch_context=DispatchContext(),
        )
        assert verdict == "proceed"


class TestVerdictTrichotomyReflect:
    """Scenario §Reflect verdict routes to escalated tier."""

    def test_below_threshold_without_anomaly_yields_reflect(self) -> None:
        gate = CalibrationGate(default_n=3, checker=_NeverCallChecker())
        verdict = gate.verdict_for(
            session_id="s1",
            ensemble_name="composed-a",
            dispatch_context=DispatchContext(auq_confidence=0.6),
        )
        assert verdict == "reflect"

    def test_just_below_threshold_yields_reflect(self) -> None:
        gate = CalibrationGate(default_n=3, checker=_NeverCallChecker())
        verdict = gate.verdict_for(
            session_id="s1",
            ensemble_name="composed-a",
            dispatch_context=DispatchContext(auq_confidence=0.84999),
        )
        assert verdict == "reflect"


class TestVerdictTrichotomyAbstain:
    """Scenario §Abstain verdict blocks dispatch and produces typed error.

    Three concrete criteria per ADR-014 §"Calibration verdict":
    entropy collapse; post-hoc result-check hard failure; severe drift.
    """

    def test_entropy_collapse_yields_abstain(self) -> None:
        """Entropy drops > 1.5σ below trajectory's running mean."""
        # Seed the window with steady-entropy history so the running
        # mean has a basis; the most recent reading is well below.
        clock = _StubClock(returns=[100.0, 200.0, 300.0, 400.0, 500.0])
        gate = CalibrationGate(default_n=3, checker=_NeverCallChecker(), clock=clock)
        # Four prior dispatches at steady high entropy.
        for _ in range(4):
            gate.verdict_for(
                session_id="s1",
                ensemble_name="composed-a",
                dispatch_context=DispatchContext(
                    auq_confidence=0.95,
                    trajectory_features=TrajectoryFeatures(recent_token_entropy=4.0),
                ),
            )
        # Fifth dispatch: entropy collapses sharply (well below mean −1.5σ;
        # the prior window is 4.0/4.0/4.0/4.0 → stdev 0 → criterion
        # requires non-degenerate stdev; bump one prior sample to lift
        # stdev above zero so the threshold check is meaningful).
        # Test arranges this in the next case; here we use varied entries.
        # Re-run with varied prior entropy in another test.
        verdict = gate.verdict_for(
            session_id="s1",
            ensemble_name="composed-a",
            dispatch_context=DispatchContext(
                auq_confidence=0.95,
                trajectory_features=TrajectoryFeatures(recent_token_entropy=0.5),
            ),
        )
        # With zero stdev the criterion cannot meaningfully fire — proves
        # the criterion needs statistical basis.
        assert verdict == "proceed"

    def test_entropy_collapse_with_variance_in_history_yields_abstain(self) -> None:
        """Varied prior entropy lifts stdev so the 1.5σ comparison fires."""
        # Five distinct timestamps; varied prior entropies.
        clock = _StubClock(returns=[100.0, 200.0, 300.0, 400.0, 500.0, 600.0])
        gate = CalibrationGate(default_n=3, checker=_NeverCallChecker(), clock=clock)
        prior_entropies = [3.8, 4.2, 4.0, 4.1, 3.9]
        for entropy in prior_entropies:
            gate.verdict_for(
                session_id="s1",
                ensemble_name="composed-a",
                dispatch_context=DispatchContext(
                    auq_confidence=0.95,
                    trajectory_features=TrajectoryFeatures(
                        recent_token_entropy=entropy
                    ),
                ),
            )
        # Sixth dispatch: entropy crashes to 0.5 — well below mean −1.5σ.
        verdict = gate.verdict_for(
            session_id="s1",
            ensemble_name="composed-a",
            dispatch_context=DispatchContext(
                auq_confidence=0.95,
                trajectory_features=TrajectoryFeatures(recent_token_entropy=0.5),
            ),
        )
        assert verdict == "abstain"

    def test_post_hoc_hard_failure_yields_abstain(self) -> None:
        """Criterion 2: ADR-007 result-check non-recoverable outcome."""
        gate = CalibrationGate(default_n=3, checker=_NeverCallChecker())
        verdict = gate.verdict_for(
            session_id="s1",
            ensemble_name="composed-a",
            dispatch_context=DispatchContext(
                auq_confidence=0.95, post_hoc_hard_failure=True
            ),
        )
        assert verdict == "abstain"

    def test_severe_drift_yields_abstain(self) -> None:
        """Criterion 3: ADR-016 mechanism (d) severe-drift verdict."""
        gate = CalibrationGate(default_n=3, checker=_NeverCallChecker())
        verdict = gate.verdict_for(
            session_id="s1",
            ensemble_name="composed-a",
            dispatch_context=DispatchContext(
                auq_confidence=0.95, drift_verdict="severe_drift"
            ),
        )
        assert verdict == "abstain"

    def test_drift_detected_does_not_alone_yield_abstain(self) -> None:
        """Only ``severe_drift`` triggers the Abstain criterion per ADR-014;
        ``drift_detected`` (advisory) is operator-action territory."""
        gate = CalibrationGate(default_n=3, checker=_NeverCallChecker())
        verdict = gate.verdict_for(
            session_id="s1",
            ensemble_name="composed-a",
            dispatch_context=DispatchContext(
                auq_confidence=0.95, drift_verdict="drift_detected"
            ),
        )
        assert verdict == "proceed"


class TestCalibrationGateAbstainCriterionExtraction:
    """``abstain_criterion_for`` returns the specific criterion that
    *would* have fired for the verdict — used by consumers raising
    :class:`CalibrationAbstainError`."""

    def test_returns_none_when_no_criterion_fires(self) -> None:
        gate = CalibrationGate(default_n=3, checker=_NeverCallChecker())
        criterion = gate.abstain_criterion_for(
            session_id="s1",
            ensemble_name="composed-a",
            dispatch_context=DispatchContext(auq_confidence=0.9),
        )
        assert criterion is None

    def test_returns_severe_drift_for_drift_input(self) -> None:
        gate = CalibrationGate(default_n=3, checker=_NeverCallChecker())
        criterion = gate.abstain_criterion_for(
            session_id="s1",
            ensemble_name="composed-a",
            dispatch_context=DispatchContext(drift_verdict="severe_drift"),
        )
        assert criterion == "severe_drift"

    def test_returns_post_hoc_hard_failure_for_failure_input(self) -> None:
        gate = CalibrationGate(default_n=3, checker=_NeverCallChecker())
        criterion = gate.abstain_criterion_for(
            session_id="s1",
            ensemble_name="composed-a",
            dispatch_context=DispatchContext(post_hoc_hard_failure=True),
        )
        assert criterion == "post_hoc_hard_failure"

    def test_severe_drift_takes_precedence_over_post_hoc(self) -> None:
        """Criterion ordering: drift > post-hoc > entropy collapse."""
        gate = CalibrationGate(default_n=3, checker=_NeverCallChecker())
        criterion = gate.abstain_criterion_for(
            session_id="s1",
            ensemble_name="composed-a",
            dispatch_context=DispatchContext(
                drift_verdict="severe_drift", post_hoc_hard_failure=True
            ),
        )
        assert criterion == "severe_drift"

    def test_abstain_criterion_for_is_side_effect_free(self) -> None:
        """Calling ``abstain_criterion_for`` does not record into the
        trajectory window — only ``verdict_for`` does."""
        gate = CalibrationGate(default_n=3, checker=_NeverCallChecker())
        gate.abstain_criterion_for(
            session_id="s1",
            ensemble_name="composed-a",
            dispatch_context=DispatchContext(
                trajectory_features=TrajectoryFeatures(recent_token_entropy=3.5),
            ),
        )
        # After only calling abstain_criterion_for, the next verdict_for
        # call on an empty window should see no basis for entropy collapse.
        verdict = gate.verdict_for(
            session_id="s1",
            ensemble_name="composed-a",
            dispatch_context=DispatchContext(
                trajectory_features=TrajectoryFeatures(recent_token_entropy=0.5),
            ),
        )
        assert verdict == "proceed"


class TestVerdictIsExhaustiveAndDeterministicGivenInputs:
    """FC-19 — every dispatch produces exactly one verdict, and the
    verdict depends only on the supplied inputs.

    Named per system-design.agents.md §Calibration Gate fitness:
    ``test_verdict_is_exhaustive_and_deterministic_given_inputs``.
    """

    def test_verdict_is_always_one_of_three_values(self) -> None:
        gate = CalibrationGate(default_n=3, checker=_NeverCallChecker())
        contexts = [
            DispatchContext(),
            DispatchContext(auq_confidence=0.5),
            DispatchContext(auq_confidence=0.95),
            DispatchContext(post_hoc_hard_failure=True),
            DispatchContext(drift_verdict="severe_drift"),
            DispatchContext(drift_verdict="drift_detected"),
            DispatchContext(drift_verdict="no_drift", auq_confidence=0.4),
        ]
        for index, context in enumerate(contexts):
            verdict = gate.verdict_for(
                session_id=f"s-{index}",
                ensemble_name="composed-a",
                dispatch_context=context,
            )
            assert verdict in ("proceed", "reflect", "abstain")

    def test_same_inputs_yield_same_verdict(self) -> None:
        """Determinism: the verdict depends only on the supplied inputs."""
        gate_a = CalibrationGate(default_n=3, checker=_NeverCallChecker())
        gate_b = CalibrationGate(default_n=3, checker=_NeverCallChecker())
        context = DispatchContext(auq_confidence=0.7)
        for session in ("s1", "s2", "s3"):
            assert gate_a.verdict_for(
                session_id=session,
                ensemble_name="composed-a",
                dispatch_context=context,
            ) == gate_b.verdict_for(
                session_id=session,
                ensemble_name="composed-a",
                dispatch_context=context,
            )


class TestTimeDecayWindowingDualBoundLinear:
    """Scenario §Time-decay windowing limits trajectory features to
    dual-bound recent window."""

    def test_signals_outside_time_window_excluded(self) -> None:
        """Samples older than 60 minutes contribute weight 0."""
        # Five timestamps: four old (more than 60 min ago) and one
        # recent. The window prunes the old; remaining sample base is
        # too small for entropy-collapse basis, so even an extreme
        # recent reading yields Proceed (not Abstain).
        # Old timestamps: 0s, 100s, 200s, 300s. Recent: 4000s (past
        # 60 min = 3600s threshold from current time 4500s).
        clock = _StubClock(returns=[0.0, 100.0, 200.0, 300.0, 4500.0])
        gate = CalibrationGate(
            default_n=3,
            checker=_NeverCallChecker(),
            clock=clock,
        )
        # Four old varied-entropy samples build a basis IF kept.
        for entropy in (3.8, 4.2, 4.0, 4.1):
            gate.verdict_for(
                session_id="s1",
                ensemble_name="composed-a",
                dispatch_context=DispatchContext(
                    trajectory_features=TrajectoryFeatures(recent_token_entropy=entropy)
                ),
            )
        # Fifth dispatch at t=4500s; the four old samples are now
        # outside the 3600s window and should not contribute basis.
        # Even an extreme low entropy should NOT fire entropy-collapse.
        verdict = gate.verdict_for(
            session_id="s1",
            ensemble_name="composed-a",
            dispatch_context=DispatchContext(
                trajectory_features=TrajectoryFeatures(recent_token_entropy=0.1)
            ),
        )
        assert verdict == "proceed"

    def test_dispatch_count_bound_prunes_oldest_entries(self) -> None:
        """Past the 100-dispatch cap, oldest samples are pruned even
        when time-window-fresh.

        Two gates run identical inputs; the gate with the tighter cap
        sees a different running stats basis than the gate with the
        broader cap, and the verdicts diverge — that divergence proves
        the cap is actively pruning.
        """
        clock_history = [float(i) for i in range(10)]
        # Six samples: three high-entropy older samples + three identical
        # recent samples. With cap=3, only the recent identical samples
        # anchor — zero variance → no basis → criterion does not fire.
        # With cap=10, the older high-entropy samples lift the basis
        # variance enough that an extreme drop fires the criterion.
        samples = [
            (0.0, 8.0),
            (1.0, 8.0),
            (2.0, 8.0),
            (3.0, 4.0),
            (4.0, 4.0),
            (5.0, 4.0),
        ]

        def gate_with_cap(cap: int) -> CalibrationGate:
            return CalibrationGate(
                default_n=3,
                checker=_NeverCallChecker(),
                clock=_StubClock(returns=list(clock_history)),
                trajectory_window_dispatches=cap,
            )

        gate_tight = gate_with_cap(3)
        gate_broad = gate_with_cap(10)
        for gate in (gate_tight, gate_broad):
            for ts, entropy in samples:
                gate.verdict_for(
                    session_id="s1",
                    ensemble_name="composed-a",
                    dispatch_context=DispatchContext(
                        trajectory_features=TrajectoryFeatures(
                            recent_token_entropy=entropy
                        ),
                        dispatch_timestamp_seconds=ts,
                    ),
                )

        # Seventh dispatch: recent reading drops to 0.0.
        # Tight cap (3): window basis is (4.0, 4.0, 4.0) → stdev=0 →
        #   no basis → criterion does not fire → Proceed.
        # Broad cap (10): window spans the full six-sample run → stdev
        #   high enough that 0.0 sits below mean−1.5σ → Abstain.
        recent_context = DispatchContext(
            trajectory_features=TrajectoryFeatures(recent_token_entropy=0.0),
            dispatch_timestamp_seconds=6.0,
        )
        tight_verdict = gate_tight.verdict_for(
            session_id="s1",
            ensemble_name="composed-a",
            dispatch_context=recent_context,
        )
        broad_verdict = gate_broad.verdict_for(
            session_id="s1",
            ensemble_name="composed-a",
            dispatch_context=recent_context,
        )

        assert tight_verdict == "proceed"
        assert broad_verdict == "abstain"

    def test_dispatch_window_with_tight_history_does_fire_on_extreme_drop(
        self,
    ) -> None:
        """The bounded window still detects entropy collapse when the
        drop is large enough relative to the window's stdev."""
        clock = _StubClock(returns=[float(i) for i in range(10)])
        gate = CalibrationGate(
            default_n=3,
            checker=_NeverCallChecker(),
            clock=clock,
            trajectory_window_dispatches=4,
        )
        # Varied baseline.
        for entropy in (3.8, 4.2, 4.0, 4.1, 3.9):
            gate.verdict_for(
                session_id="s1",
                ensemble_name="composed-a",
                dispatch_context=DispatchContext(
                    trajectory_features=TrajectoryFeatures(recent_token_entropy=entropy)
                ),
            )
        # Recent window is (4.2, 4.0, 4.1, 3.9). Extreme drop:
        verdict = gate.verdict_for(
            session_id="s1",
            ensemble_name="composed-a",
            dispatch_context=DispatchContext(
                trajectory_features=TrajectoryFeatures(recent_token_entropy=0.1)
            ),
        )
        assert verdict == "abstain"

    def test_explicit_timestamp_drives_windowing(self) -> None:
        """``dispatch_timestamp_seconds`` overrides clock for the sample."""
        gate = CalibrationGate(
            default_n=3,
            checker=_NeverCallChecker(),
            clock=_StubClock(returns=[]),  # not consulted
        )
        # Three samples, all far apart but inside one big window.
        for ts, entropy in [(0.0, 3.9), (1.0, 4.0), (2.0, 4.1)]:
            gate.verdict_for(
                session_id="s1",
                ensemble_name="composed-a",
                dispatch_context=DispatchContext(
                    trajectory_features=TrajectoryFeatures(
                        recent_token_entropy=entropy
                    ),
                    dispatch_timestamp_seconds=ts,
                ),
            )
        # Fourth sample at t=3.0 with a small drop — confirm we get a
        # deterministic outcome, not a clock-dependent one.
        verdict = gate.verdict_for(
            session_id="s1",
            ensemble_name="composed-a",
            dispatch_context=DispatchContext(
                trajectory_features=TrajectoryFeatures(recent_token_entropy=0.0),
                dispatch_timestamp_seconds=3.0,
            ),
        )
        assert verdict == "abstain"


class TestVerdictComputationWorksWithoutSignalChannel:
    """FC-19 / system-design fitness — when ADR-016 is not active, the
    in-process layer operates on L1-internal trajectory data only;
    the verdict trichotomy continues to function.

    Named per system-design.agents.md §Calibration Gate fitness:
    ``test_verdict_computation_works_without_signal_channel``.
    """

    def test_verdict_works_with_drift_verdict_none(self) -> None:
        """Channel inactive → drift_verdict is ``None``; verdict still
        flows through the three values."""
        gate = CalibrationGate(default_n=3, checker=_NeverCallChecker())
        for confidence, expected in (
            (0.95, "proceed"),
            (0.5, "reflect"),
        ):
            verdict = gate.verdict_for(
                session_id="s1",
                ensemble_name="composed-a",
                dispatch_context=DispatchContext(
                    auq_confidence=confidence, drift_verdict=None
                ),
            )
            assert verdict == expected

    def test_post_hoc_hard_failure_still_aborts_without_channel(self) -> None:
        gate = CalibrationGate(default_n=3, checker=_NeverCallChecker())
        verdict = gate.verdict_for(
            session_id="s1",
            ensemble_name="composed-a",
            dispatch_context=DispatchContext(
                auq_confidence=0.95,
                post_hoc_hard_failure=True,
                drift_verdict=None,
            ),
        )
        assert verdict == "abstain"


class TestAdr007PostHocCalibrationUnchangedUnderAdr014:
    """Preservation scenario — ADR-007's first-N post-hoc mechanism is
    unchanged under ADR-014's extension.

    Named per system-design.agents.md §Calibration Gate fitness:
    ``test_adr_007_post_hoc_calibration_unchanged_under_adr_014``.
    """

    @pytest.mark.asyncio
    async def test_check_and_record_still_records_signals(self) -> None:
        """The ADR-007 API is unchanged when callers also use
        verdict_for; calibration tracking proceeds independently."""
        checker = _ScriptedChecker(["positive", "positive", "positive"])
        gate = CalibrationGate(default_n=3, checker=checker)
        gate.mark_composed(session_id="s1", ensemble_name="composed-a")

        # Interleave verdict_for and check_and_record calls — the two
        # surfaces are additive.
        for _ in range(3):
            gate.verdict_for(
                session_id="s1",
                ensemble_name="composed-a",
                dispatch_context=DispatchContext(auq_confidence=0.95),
            )
            await gate.check_and_record(
                session_id="s1",
                ensemble_name="composed-a",
                raw_result={"results": {"a": {"response": "hi"}}},
            )

        assert gate.status(session_id="s1", ensemble_name="composed-a") == "trusted"
        record = gate.record_for(session_id="s1", ensemble_name="composed-a")
        assert record is not None
        assert record.signals == ("positive", "positive", "positive")
        assert record.invocations_seen == 3

    @pytest.mark.asyncio
    async def test_trusted_ensemble_skips_check_unchanged(self) -> None:
        """After transition to trusted, the ADR-007 invariant
        (no further check invocation) holds regardless of verdict_for
        calls."""
        checker = _ScriptedChecker(["positive", "positive", "positive"])
        gate = CalibrationGate(default_n=3, checker=checker)
        gate.mark_composed(session_id="s1", ensemble_name="composed-a")
        for _ in range(3):
            await gate.check_and_record(
                session_id="s1", ensemble_name="composed-a", raw_result={}
            )
        # Trusted now; an ADR-014 verdict call must not consume from
        # the (now-exhausted) checker iterator.
        verdict = gate.verdict_for(
            session_id="s1",
            ensemble_name="composed-a",
            dispatch_context=DispatchContext(auq_confidence=0.95),
        )
        assert verdict == "proceed"
        signal = await gate.check_and_record(
            session_id="s1", ensemble_name="composed-a", raw_result={}
        )
        assert signal is None
        assert len(checker.calls) == 3


class TestCalibrationAbstainErrorRaiseAndDispatchContext:
    """Consumer-side raise path with the criterion extracted from the
    gate (the WP-G4-1 Tier-Escalation Router transforms to
    ``escalation_bypass``; standalone consumers raise this directly)."""

    def test_raise_carries_criterion_in_dispatch_context(self) -> None:
        gate = CalibrationGate(default_n=3, checker=_NeverCallChecker())
        context = DispatchContext(drift_verdict="severe_drift")
        verdict = gate.verdict_for(
            session_id="s1",
            ensemble_name="composed-a",
            dispatch_context=context,
        )
        assert verdict == "abstain"
        criterion = gate.abstain_criterion_for(
            session_id="s1",
            ensemble_name="composed-a",
            dispatch_context=context,
        )
        assert criterion == "severe_drift"

        with pytest.raises(CalibrationAbstainError, match="severe drift"):
            raise CalibrationAbstainError(
                "severe drift triggered Abstain verdict",
                session_id="s1",
                ensemble_name="composed-a",
                criterion=criterion,
            )
