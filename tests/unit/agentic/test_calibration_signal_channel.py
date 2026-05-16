"""Unit tests for Calibration Signal Channel (WP-H4, ADR-016).

Covers the 11 scenarios in ``docs/agentic-serving/scenarios.md`` §Cross-
Layer Calibration Channel plus the four fitness criteria in
``docs/agentic-serving/system-design.agents.md`` §Module: Calibration
Signal Channel.

Per ADR-016's conditional acceptance the channel ships with explicit
falsification trigger; these unit tests validate that all five
bounding mechanisms (a)–(e) operate within ADR-002's L0–L3 structure
(the trigger has not fired).
"""

from __future__ import annotations

import inspect

import pytest

from llm_orc.agentic.calibration_gate import CalibrationVerdict
from llm_orc.agentic.calibration_signal_channel import (
    CalibrationChannelAuditThresholds,
    CalibrationSignal,
    CalibrationSignalChannel,
    MalformedSignalError,
    WindowedSignalFeatures,
)
from llm_orc.agentic.dispatch_event_substrate import DispatchEventSubstrate
from llm_orc.models.structural_errors import LlmOrcStructuralError


class _ScriptedClock:
    """Test-only :class:`ChannelClock` that advances on explicit ``tick``."""

    def __init__(self, start_seconds: float = 0.0) -> None:
        self._now = start_seconds

    def now_seconds(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


def _make_signal(
    *,
    timestamp_seconds: float = 0.0,
    ensemble_name: str = "ens-A",
    dispatch_success: bool = True,
    recent_token_entropy: float | None = None,
    deterministic_anchor: bool | None = None,
) -> CalibrationSignal:
    return CalibrationSignal(
        timestamp_seconds=timestamp_seconds,
        ensemble_name=ensemble_name,
        dispatch_success=dispatch_success,
        recent_token_entropy=recent_token_entropy,
        deterministic_anchor=deterministic_anchor,
    )


# ---------------------------------------------------------------------------
# Scenario 1: Calibration data flows L0 → L1 through read-only channel
# ---------------------------------------------------------------------------


class TestSignalFlowL0ToL1:
    """L0 emits → channel buffers → L1 reads via aggregated features."""

    def test_l1_consumer_receives_typed_aggregated_features(self) -> None:
        clock = _ScriptedClock(start_seconds=1000.0)
        channel = CalibrationSignalChannel(clock=clock)

        channel.record_signal(
            _make_signal(timestamp_seconds=1000.0, dispatch_success=True)
        )
        channel.record_signal(
            _make_signal(timestamp_seconds=1001.0, dispatch_success=False)
        )

        features = channel.windowed_features()

        assert isinstance(features, WindowedSignalFeatures)
        assert features.in_window_count == 2
        assert features.aggregated_success_rate == 0.5

    def test_aggregated_view_is_a_fresh_value_not_shared_state(self) -> None:
        """Mechanism (a) — consumer reads aggregated values, not signal refs."""
        clock = _ScriptedClock(start_seconds=1000.0)
        channel = CalibrationSignalChannel(clock=clock)
        channel.record_signal(
            _make_signal(timestamp_seconds=1000.0, recent_token_entropy=2.0)
        )
        channel.record_signal(
            _make_signal(timestamp_seconds=1001.0, recent_token_entropy=2.5)
        )

        features1 = channel.windowed_features()
        features2 = channel.windowed_features()

        # The two reads return value-equal features but are not the
        # same instance — the channel constructs a fresh view per call.
        assert features1 == features2
        assert features1 is not features2


# ---------------------------------------------------------------------------
# Scenario 2: Upward write attempt through channel is rejected (structurally)
# ---------------------------------------------------------------------------


class TestNoWritePath:
    """The channel intentionally exposes no L1 → L0 write API."""

    def test_channel_surface_has_no_l1_to_l0_write_method(self) -> None:
        """Structural test — no method on the channel mutates L0 state.

        The methods on the channel are signal-emission (L0 → channel),
        feature read (channel → L1), audit feedback (L1 → audit), and
        operator/test inspection. Nothing routes back through the
        channel to L0.
        """
        public_methods = {
            name
            for name, _ in inspect.getmembers(
                CalibrationSignalChannel, predicate=inspect.isfunction
            )
            if not name.startswith("_")
        }

        # The expected public surface — see system-design.agents.md
        # §Module: Calibration Signal Channel. Anything outside this
        # set requires explicit justification before BUILD-merge.
        expected = {
            "record_signal",
            "windowed_features",
            "record_verdict_outcome",
            "audit_diagnostics",
            "clear_fail_safe",
            "malformed_signal_count",
        }
        unexpected = public_methods - expected
        assert unexpected == set(), (
            "CalibrationSignalChannel grew an unexpected public method — "
            "verify it does not create an L1 → L0 write path: "
            f"{unexpected}"
        )

    def test_record_signal_does_not_accept_callback_to_mutate_l0(self) -> None:
        """``record_signal`` accepts a typed signal, not a callable."""
        channel = CalibrationSignalChannel()

        with pytest.raises(MalformedSignalError):
            channel.record_signal(lambda: None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Scenario 3: Non-calibration data attempt through channel is rejected
# ---------------------------------------------------------------------------


class TestNonCalibrationDataRejected:
    """Mechanism (e) — schema validation rejects arbitrary upward data."""

    def test_unknown_mapping_shape_raises_malformed_signal_error(self) -> None:
        channel = CalibrationSignalChannel()

        with pytest.raises(MalformedSignalError) as exc_info:
            channel.record_signal({"arbitrary": "data", "unrelated": 42})

        assert exc_info.value.error_kind == "malformed_signal"

    def test_l1_consumer_never_receives_rejected_data(self) -> None:
        clock = _ScriptedClock(start_seconds=1000.0)
        channel = CalibrationSignalChannel(clock=clock)

        with pytest.raises(MalformedSignalError):
            channel.record_signal({"arbitrary": "data"})

        features = channel.windowed_features()
        assert features.in_window_count == 0


# ---------------------------------------------------------------------------
# Scenario 4: Mechanism (a) — fresh-context isolation
# ---------------------------------------------------------------------------


class TestMechanismAFreshContextIsolation:
    """Consumer's evaluation context does not accumulate signal data."""

    def test_no_raw_signal_list_exposed_in_public_api(self) -> None:
        """Mechanism (a) is enforced by API shape — only aggregated reads."""
        channel = CalibrationSignalChannel()
        assert not hasattr(channel, "signals")
        assert not hasattr(channel, "get_signals")
        assert not hasattr(channel, "raw_signals")

    def test_two_consecutive_reads_return_equal_aggregated_views(self) -> None:
        """The aggregated view is deterministic from in-window signals."""
        clock = _ScriptedClock(start_seconds=1000.0)
        channel = CalibrationSignalChannel(clock=clock)

        for i in range(5):
            channel.record_signal(
                _make_signal(
                    timestamp_seconds=1000.0 + i,
                    recent_token_entropy=2.0 + 0.1 * i,
                )
            )

        a = channel.windowed_features()
        b = channel.windowed_features()
        assert a == b


# ---------------------------------------------------------------------------
# Scenario 5: Mechanism (b) — time-decay windowing
# ---------------------------------------------------------------------------


class TestMechanismBTimeDecayWindowing:
    """Stale signals do not influence current aggregation."""

    def test_signals_older_than_window_minutes_contribute_zero(self) -> None:
        clock = _ScriptedClock(start_seconds=0.0)
        channel = CalibrationSignalChannel(
            window_minutes=60.0, window_signals=1000, clock=clock
        )

        # Signal at t=0 — three hours before "now" — outside the
        # 60-minute window.
        channel.record_signal(
            _make_signal(timestamp_seconds=0.0, recent_token_entropy=10.0)
        )
        clock.advance(3 * 3600.0)
        # Three in-window signals — three is the minimum needed for the
        # linear-decay weighting to produce a non-degenerate weighted
        # variance (with n=2 the oldest sample receives weight 0 so the
        # window collapses to a single effective sample).
        for offset_seconds, entropy in [(0.0, 2.0), (30.0, 2.5), (60.0, 3.0)]:
            clock.advance(offset_seconds if offset_seconds > 0.0 else 0.0)
            channel.record_signal(
                _make_signal(
                    timestamp_seconds=clock.now_seconds(),
                    recent_token_entropy=entropy,
                )
            )

        features = channel.windowed_features()
        assert features.in_window_count == 3
        # The 10.0 outlier dropped — mean should be in [2.0, 3.0].
        assert features.running_entropy_mean is not None
        assert 2.0 <= features.running_entropy_mean <= 3.0

    def test_signals_past_dispatch_count_bound_dropped(self) -> None:
        clock = _ScriptedClock(start_seconds=0.0)
        channel = CalibrationSignalChannel(
            window_minutes=60.0, window_signals=3, clock=clock
        )

        for i in range(5):
            clock.advance(1.0)
            channel.record_signal(
                _make_signal(
                    timestamp_seconds=clock.now_seconds(),
                    recent_token_entropy=float(i),
                )
            )

        features = channel.windowed_features()
        # Only 3 most-recent signals retained.
        assert features.in_window_count == 3

    def test_linear_decay_weight_more_recent_carries_more_influence(self) -> None:
        """Most-recent signal should dominate the weighted mean.

        With three equally-spaced in-window samples and weights
        ``[0.0, 0.5, 1.0]`` the weighted mean equals
        ``(0.0*old + 0.5*mid + 1.0*new) / 1.5``. Setting the oldest to
        0.0 leaves the average dominated by the most-recent sample's
        contribution.
        """
        clock = _ScriptedClock(start_seconds=0.0)
        channel = CalibrationSignalChannel(window_minutes=60.0, clock=clock)

        channel.record_signal(
            _make_signal(timestamp_seconds=0.0, recent_token_entropy=0.0)
        )
        clock.advance(10.0)
        channel.record_signal(
            _make_signal(
                timestamp_seconds=clock.now_seconds(), recent_token_entropy=5.0
            )
        )
        clock.advance(10.0)
        channel.record_signal(
            _make_signal(
                timestamp_seconds=clock.now_seconds(), recent_token_entropy=10.0
            )
        )

        features = channel.windowed_features()
        # Expected weighted mean = (0.0*0 + 0.5*5 + 1.0*10) / 1.5 ≈ 8.333…
        # The most-recent sample's contribution (10.0) dominates.
        assert features.running_entropy_mean is not None
        assert features.running_entropy_mean == pytest.approx(12.5 / 1.5)
        assert features.running_entropy_mean > 5.0


# ---------------------------------------------------------------------------
# Scenarios 6 + 7: Mechanism (c) — deterministic anchor / LLM-only ensemble
# ---------------------------------------------------------------------------


class TestMechanismCDeterministicAnchor:
    """Categorical anchor is surfaced when available; LLM-only is graceful."""

    def test_anchor_signal_increments_anchor_count(self) -> None:
        clock = _ScriptedClock(start_seconds=0.0)
        channel = CalibrationSignalChannel(clock=clock)
        channel.record_signal(
            _make_signal(timestamp_seconds=0.0, deterministic_anchor=True)
        )
        channel.record_signal(
            _make_signal(timestamp_seconds=1.0, deterministic_anchor=False)
        )

        features = channel.windowed_features()
        assert features.deterministic_anchor_count == 2
        assert features.deterministic_anchor_positive_fraction == 0.5

    def test_llm_only_ensemble_has_zero_anchor_count(self) -> None:
        """Signals with ``deterministic_anchor=None`` produce empty (c)."""
        clock = _ScriptedClock(start_seconds=0.0)
        channel = CalibrationSignalChannel(clock=clock)
        channel.record_signal(
            _make_signal(timestamp_seconds=0.0, deterministic_anchor=None)
        )
        channel.record_signal(
            _make_signal(timestamp_seconds=1.0, deterministic_anchor=None)
        )

        features = channel.windowed_features()
        assert features.deterministic_anchor_count == 0
        assert features.deterministic_anchor_positive_fraction is None

    def test_llm_only_ensemble_still_aggregates_features_a_b_e(self) -> None:
        """Mechanisms (a), (b), (e) remain load-bearing without (c).

        Uses three samples — the minimum needed for the linear-decay
        weighting to produce a non-degenerate weighted variance.
        """
        clock = _ScriptedClock(start_seconds=0.0)
        channel = CalibrationSignalChannel(clock=clock)
        for timestamp, entropy in [(0.0, 2.0), (1.0, 2.5), (2.0, 3.0)]:
            channel.record_signal(
                _make_signal(
                    timestamp_seconds=timestamp,
                    recent_token_entropy=entropy,
                    deterministic_anchor=None,
                )
            )

        features = channel.windowed_features()
        assert features.in_window_count == 3
        assert features.has_entropy_basis is True
        assert features.deterministic_anchor_count == 0


# ---------------------------------------------------------------------------
# Scenario 8: Mechanism (d) — out-of-band audit fires at trigger frequency
# ---------------------------------------------------------------------------


class TestMechanismDAuditFiresAtTrigger:
    """Audit fires every ``trigger_count`` consumptions or wall-clock hours."""

    def test_audit_fires_after_trigger_count_consumptions(self) -> None:
        clock = _ScriptedClock(start_seconds=0.0)
        thresholds = CalibrationChannelAuditThresholds(trigger_count=5)
        channel = CalibrationSignalChannel(audit_thresholds=thresholds, clock=clock)

        for _ in range(4):
            diagnostic = channel.record_verdict_outcome(
                verdict="proceed",
                ensemble_name="ens-A",
                signal_features=WindowedSignalFeatures.empty(),
            )
            assert diagnostic is None

        diagnostic = channel.record_verdict_outcome(
            verdict="proceed",
            ensemble_name="ens-A",
            signal_features=WindowedSignalFeatures.empty(),
        )
        assert diagnostic is not None
        assert diagnostic.window_id == 1
        assert diagnostic.consumption_count == 5

    def test_audit_fires_after_wall_clock_hours_pass(self) -> None:
        clock = _ScriptedClock(start_seconds=0.0)
        thresholds = CalibrationChannelAuditThresholds(
            trigger_count=10_000, trigger_wall_clock_hours=1.0
        )
        channel = CalibrationSignalChannel(audit_thresholds=thresholds, clock=clock)

        channel.record_verdict_outcome(
            verdict="proceed",
            ensemble_name="ens-A",
            signal_features=WindowedSignalFeatures.empty(),
        )
        assert channel.audit_diagnostics() == ()

        clock.advance(3600.0 + 1.0)
        diagnostic = channel.record_verdict_outcome(
            verdict="proceed",
            ensemble_name="ens-A",
            signal_features=WindowedSignalFeatures.empty(),
        )
        assert diagnostic is not None

    def test_first_audit_has_no_prior_window_no_drift(self) -> None:
        clock = _ScriptedClock(start_seconds=0.0)
        thresholds = CalibrationChannelAuditThresholds(trigger_count=3)
        channel = CalibrationSignalChannel(audit_thresholds=thresholds, clock=clock)

        for _ in range(3):
            channel.record_verdict_outcome(
                verdict="proceed",
                ensemble_name="ens-A",
                signal_features=WindowedSignalFeatures.empty(),
            )

        diagnostics = channel.audit_diagnostics()
        assert len(diagnostics) == 1
        assert diagnostics[0].verdict == "no_drift"


# ---------------------------------------------------------------------------
# Scenario 9: Mechanism (e) — malformed signal produces typed error
# ---------------------------------------------------------------------------


class TestMechanismEMalformedSignalProducesTypedError:
    """Schema validation rejects malformed signals at the channel boundary."""

    def test_missing_ensemble_name_field_raises(self) -> None:
        channel = CalibrationSignalChannel()
        with pytest.raises(MalformedSignalError) as exc_info:
            channel.record_signal({"timestamp_seconds": 1.0, "dispatch_success": True})
        assert exc_info.value.error_kind == "malformed_signal"

    def test_missing_timestamp_field_raises(self) -> None:
        channel = CalibrationSignalChannel()
        with pytest.raises(MalformedSignalError):
            channel.record_signal({"ensemble_name": "ens-A", "dispatch_success": True})

    def test_missing_dispatch_success_field_raises(self) -> None:
        channel = CalibrationSignalChannel()
        with pytest.raises(MalformedSignalError):
            channel.record_signal({"ensemble_name": "ens-A", "timestamp_seconds": 1.0})

    def test_wrong_type_for_entropy_field_raises(self) -> None:
        channel = CalibrationSignalChannel()
        with pytest.raises(MalformedSignalError):
            channel.record_signal(
                {
                    "ensemble_name": "ens-A",
                    "timestamp_seconds": 1.0,
                    "dispatch_success": True,
                    "recent_token_entropy": "not a number",
                }
            )

    def test_malformed_signal_count_increments_on_rejection(self) -> None:
        channel = CalibrationSignalChannel()
        try:
            channel.record_signal({"invalid": True})
        except MalformedSignalError:
            pass
        assert channel.malformed_signal_count() == 1

    def test_malformed_signal_error_is_llm_orc_structural_error(self) -> None:
        """FC-17 — MalformedSignalError is the 8th concrete subclass."""
        channel = CalibrationSignalChannel()
        with pytest.raises(LlmOrcStructuralError):
            channel.record_signal({"invalid": True})

    def test_malformed_signal_dropped_from_verdict_computation(self) -> None:
        """Per ADR-016 §"Mechanism (e)" — malformed signal does not enter window."""
        clock = _ScriptedClock(start_seconds=0.0)
        channel = CalibrationSignalChannel(clock=clock)

        # Accept one well-formed signal first.
        channel.record_signal(
            _make_signal(timestamp_seconds=0.0, recent_token_entropy=2.0)
        )

        with pytest.raises(MalformedSignalError):
            channel.record_signal({"invalid": True})

        # Window count reflects only the well-formed signal.
        features = channel.windowed_features()
        assert features.in_window_count == 1


# ---------------------------------------------------------------------------
# Scenario 10: Severe drift triggers fail-safe mode
# ---------------------------------------------------------------------------


class TestSevereDriftActivatesFailSafe:
    """Severe-drift audit verdict activates fail-safe synchronously."""

    def test_severe_drift_activates_fail_safe(self) -> None:
        """Distribution shift from 100% Proceed to 100% Reflect is severe."""
        clock = _ScriptedClock(start_seconds=0.0)
        thresholds = CalibrationChannelAuditThresholds(trigger_count=5)
        channel = CalibrationSignalChannel(audit_thresholds=thresholds, clock=clock)

        # Window 1: all Proceed
        for _ in range(5):
            channel.record_verdict_outcome(
                verdict="proceed",
                ensemble_name="ens-A",
                signal_features=WindowedSignalFeatures.empty(),
            )
        assert channel.fail_safe_active is False

        # Window 2: all Reflect — 100% distribution shift, way over 2x×15%
        for _ in range(5):
            channel.record_verdict_outcome(
                verdict="reflect",
                ensemble_name="ens-A",
                signal_features=WindowedSignalFeatures.empty(),
            )

        diagnostics = channel.audit_diagnostics()
        assert len(diagnostics) == 2
        assert diagnostics[1].verdict == "severe"
        assert channel.fail_safe_active is True

    def test_clear_fail_safe_releases_fail_safe(self) -> None:
        clock = _ScriptedClock(start_seconds=0.0)
        thresholds = CalibrationChannelAuditThresholds(trigger_count=2)
        channel = CalibrationSignalChannel(audit_thresholds=thresholds, clock=clock)

        for _ in range(2):
            channel.record_verdict_outcome(
                verdict="proceed",
                ensemble_name="ens-A",
                signal_features=WindowedSignalFeatures.empty(),
            )
        for _ in range(2):
            channel.record_verdict_outcome(
                verdict="reflect",
                ensemble_name="ens-A",
                signal_features=WindowedSignalFeatures.empty(),
            )

        assert channel.fail_safe_active is True
        channel.clear_fail_safe()
        assert channel.fail_safe_active is False


# ---------------------------------------------------------------------------
# Scenario 11: ADR-016 not active — channel never instantiated
# ---------------------------------------------------------------------------


class TestADR016InactiveCase:
    """When channel is not instantiated, L1 calibration operates L1-internal."""

    def test_calibration_gate_operates_without_channel(self) -> None:
        """ADR-014 verdict producer works on L1-internal data only.

        Verified by the absence of any required-channel parameter on
        :class:`CalibrationGate`. The gate accepts an optional channel
        in the WP-H4 wiring path (next change in this WP) but operates
        identically without it.
        """
        # Importing here proves the gate is constructible without
        # importing the channel.
        from llm_orc.agentic.calibration_gate import (
            CalibrationGate,
            DispatchContext,
        )

        class _NullChecker:
            async def check(
                self, *, ensemble_name: str, raw_result: dict[str, object]
            ) -> str:
                return "absent"

        gate = CalibrationGate(checker=_NullChecker())  # type: ignore[arg-type]
        verdict = gate.verdict_for(
            session_id="s1",
            ensemble_name="ens-A",
            dispatch_context=DispatchContext(),
        )
        assert verdict in ("proceed", "reflect", "abstain")


# ---------------------------------------------------------------------------
# Preservation: ADR-002 layering rule unchanged for write/non-calibration
# ---------------------------------------------------------------------------


class TestPreservationLayeringRuleHolds:
    """The exception is the only one; other upward attempts still rejected."""

    def test_channel_api_only_records_signal_data(self) -> None:
        """The single L0 → L1 method accepts only typed signal data.

        Any non-calibration-shaped payload raises
        :class:`MalformedSignalError`. The structural validation guard
        is the runtime enforcement of "signal-channel-specific (calibration
        only)" per ADR-016 §"The exception".
        """
        channel = CalibrationSignalChannel()
        with pytest.raises(MalformedSignalError):
            channel.record_signal({"command": "do something to L0"})

    def test_record_signal_does_not_return_anything_callers_could_misuse(
        self,
    ) -> None:
        """``record_signal`` returns ``None`` — no L0-mutating handle.

        Checked via ``__annotations__``: the signature declares a
        ``None`` return type. A non-``None`` return would be the
        L0-mutating handle that scenarios.md §"Upward write attempt"
        rejects.
        """
        annotations = CalibrationSignalChannel.record_signal.__annotations__
        # PEP-563 deferred annotations — the return type is the string
        # ``"None"``. Either the runtime ``None`` type or the deferred
        # string form is acceptable.
        return_annotation = annotations.get("return")
        assert (
            return_annotation == "None"
            or return_annotation is type(None)
            or return_annotation is None
        )


# ---------------------------------------------------------------------------
# Channel constructor validation
# ---------------------------------------------------------------------------


class TestChannelConstructorValidation:
    def test_window_minutes_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="window_minutes"):
            CalibrationSignalChannel(window_minutes=0.0)

    def test_window_signals_must_be_at_least_one(self) -> None:
        with pytest.raises(ValueError, match="window_signals"):
            CalibrationSignalChannel(window_signals=0)


class TestAuditThresholdsValidation:
    def test_negative_trigger_count_rejected(self) -> None:
        with pytest.raises(ValueError, match="trigger_count"):
            CalibrationChannelAuditThresholds(trigger_count=0)

    def test_zero_wall_clock_hours_rejected(self) -> None:
        with pytest.raises(ValueError, match="trigger_wall_clock_hours"):
            CalibrationChannelAuditThresholds(trigger_wall_clock_hours=0.0)

    def test_negative_distribution_shift_rejected(self) -> None:
        with pytest.raises(ValueError, match="verdict_distribution_shift"):
            CalibrationChannelAuditThresholds(verdict_distribution_shift=-0.1)

    def test_severe_drift_multiplier_less_than_one_rejected(self) -> None:
        with pytest.raises(ValueError, match="severe_drift_multiplier"):
            CalibrationChannelAuditThresholds(severe_drift_multiplier=0.5)


# ---------------------------------------------------------------------------
# Fitness criteria verification (system-design.agents.md §Module: Calibration
# Signal Channel)
# ---------------------------------------------------------------------------


class TestFitnessCriteria:
    """The four declared fitness criteria for the module."""

    def test_channel_validates_signal_schema_at_boundary(self) -> None:
        """FC: Every signal traversing the channel is structurally typed.

        Verified by mechanism (e)'s schema validation test class above —
        replicating the FC name here so the linkage to system-design is
        explicit per ADR-076's named-fitness convention.
        """
        channel = CalibrationSignalChannel()
        with pytest.raises(MalformedSignalError):
            channel.record_signal({"missing required fields": True})
        assert channel.windowed_features().in_window_count == 0

    def test_consumer_runs_in_fresh_context_no_carryover(self) -> None:
        """FC: Each verdict computation runs in a fresh evaluation context.

        Structural enforcement: the channel exposes aggregated views
        per call (no signal-list reference). The same in-window state
        produces value-equal but distinct-instance views.
        """
        clock = _ScriptedClock(start_seconds=0.0)
        channel = CalibrationSignalChannel(clock=clock)
        channel.record_signal(
            _make_signal(timestamp_seconds=0.0, recent_token_entropy=2.0)
        )
        channel.record_signal(
            _make_signal(timestamp_seconds=1.0, recent_token_entropy=2.5)
        )

        a = channel.windowed_features()
        b = channel.windowed_features()
        assert a == b
        assert a is not b

    def test_channel_is_read_only_no_l1_to_l0_writes(self) -> None:
        """FC: The channel rejects any upward write attempt structurally.

        Replicated structural check — the channel has no L1 → L0 method.
        """
        public_methods = {
            name
            for name, _ in inspect.getmembers(
                CalibrationSignalChannel, predicate=inspect.isfunction
            )
            if not name.startswith("_")
        }
        # Restated for fitness-criteria traceability.
        assert "write_to_ensemble_engine" not in public_methods
        assert "mutate_l0_state" not in public_methods
        assert "send_command_to_l0" not in public_methods

    def test_audit_dispatch_fires_at_trigger_and_severe_drift_activates_fail_safe(
        self,
    ) -> None:
        """FC: Mechanism (d) audit fires at trigger; severe drift activates fail-safe.

        The integration FC named in system-design.agents.md L247. Drives
        through both surfaces in one test per the FC's combined shape.
        """
        clock = _ScriptedClock(start_seconds=0.0)
        thresholds = CalibrationChannelAuditThresholds(trigger_count=5)
        channel = CalibrationSignalChannel(audit_thresholds=thresholds, clock=clock)

        # First trigger window: all Proceed → no_drift, no fail-safe.
        for _ in range(5):
            channel.record_verdict_outcome(
                verdict="proceed",
                ensemble_name="ens-A",
                signal_features=WindowedSignalFeatures.empty(),
            )
        assert channel.fail_safe_active is False

        # Second trigger window: all Reflect → severe shift → fail-safe.
        for _ in range(5):
            channel.record_verdict_outcome(
                verdict="reflect",
                ensemble_name="ens-A",
                signal_features=WindowedSignalFeatures.empty(),
            )
        diagnostics = channel.audit_diagnostics()
        assert len(diagnostics) == 2
        assert diagnostics[1].verdict == "severe"
        assert channel.fail_safe_active is True


# ---------------------------------------------------------------------------
# Signal-to-verdict correlation criterion exercises
# ---------------------------------------------------------------------------


class TestGateConsumesChannel:
    """L1 Calibration Gate consumes the channel — the WP-H4 wiring."""

    def test_gate_records_verdict_to_channel_audit(self) -> None:
        """Verdict consumption flows from gate → channel.audit_window."""
        from llm_orc.agentic.calibration_gate import (
            CalibrationGate,
            DispatchContext,
        )

        clock = _ScriptedClock(start_seconds=0.0)
        channel = CalibrationSignalChannel(
            audit_thresholds=CalibrationChannelAuditThresholds(trigger_count=2),
            clock=clock,
        )

        class _NullChecker:
            async def check(
                self, *, ensemble_name: str, raw_result: dict[str, object]
            ) -> str:
                return "absent"

        gate = CalibrationGate(
            checker=_NullChecker(),  # type: ignore[arg-type]
            signal_channel=channel,
        )

        # Two dispatches → channel's trigger_count=2 fires.
        gate.verdict_for(
            session_id="s1",
            ensemble_name="ens-A",
            dispatch_context=DispatchContext(dispatch_timestamp_seconds=0.0),
        )
        gate.verdict_for(
            session_id="s1",
            ensemble_name="ens-A",
            dispatch_context=DispatchContext(dispatch_timestamp_seconds=1.0),
        )

        diagnostics = channel.audit_diagnostics()
        assert len(diagnostics) == 1
        # First audit fires with no prior window → no_drift.
        assert diagnostics[0].verdict == "no_drift"

    def test_fail_safe_from_channel_forces_reflect_verdict(self) -> None:
        """When channel is in fail-safe, gate defaults to Reflect."""
        from llm_orc.agentic.calibration_gate import (
            CalibrationGate,
            DispatchContext,
        )

        clock = _ScriptedClock(start_seconds=0.0)
        channel = CalibrationSignalChannel(
            audit_thresholds=CalibrationChannelAuditThresholds(trigger_count=2),
            clock=clock,
        )

        class _NullChecker:
            async def check(
                self, *, ensemble_name: str, raw_result: dict[str, object]
            ) -> str:
                return "absent"

        gate = CalibrationGate(
            checker=_NullChecker(),  # type: ignore[arg-type]
            signal_channel=channel,
        )

        # Drive the channel into severe-drift fail-safe via verdict
        # consumptions directly on the channel (bypass gate's verdict
        # production to drive specific verdicts).
        for _ in range(2):
            channel.record_verdict_outcome(
                verdict="proceed",
                ensemble_name="ens-A",
                signal_features=WindowedSignalFeatures.empty(),
            )
        for _ in range(2):
            channel.record_verdict_outcome(
                verdict="reflect",
                ensemble_name="ens-A",
                signal_features=WindowedSignalFeatures.empty(),
            )
        assert channel.fail_safe_active is True

        # Now the gate's verdict should be Reflect even with high AUQ
        # confidence (which would normally produce Proceed).
        verdict = gate.verdict_for(
            session_id="s1",
            ensemble_name="ens-A",
            dispatch_context=DispatchContext(
                auq_confidence=0.99,
                dispatch_timestamp_seconds=2.0,
            ),
        )
        assert verdict == "reflect"


class TestLifecycleCompositionL0ToL1:
    """Integration: EnsembleExecutor → channel → gate (lifecycle sequence)."""

    def test_executor_emits_signal_then_gate_reads_features(self) -> None:
        """L0 emit → channel buffer → L1 read returns the in-window count."""
        from llm_orc.agentic.calibration_signal_channel import CalibrationSignal

        clock = _ScriptedClock(start_seconds=1000.0)
        channel = CalibrationSignalChannel(clock=clock)

        # Simulate L0 emission (bypass instantiating EnsembleExecutor
        # because that requires heavy fixtures; this is the call
        # ``EnsembleExecutor._emit_calibration_signal`` makes).
        channel.record_signal(
            CalibrationSignal(
                timestamp_seconds=1000.0,
                ensemble_name="ens-A",
                dispatch_success=True,
            )
        )
        channel.record_signal(
            CalibrationSignal(
                timestamp_seconds=1001.0,
                ensemble_name="ens-A",
                dispatch_success=False,
            )
        )

        features = channel.windowed_features(now_seconds=1002.0)
        assert features.in_window_count == 2
        assert features.aggregated_success_rate == 0.5

    def test_channel_filter_by_ensemble_returns_per_ensemble_view(self) -> None:
        from llm_orc.agentic.calibration_signal_channel import CalibrationSignal

        clock = _ScriptedClock(start_seconds=1000.0)
        channel = CalibrationSignalChannel(clock=clock)

        channel.record_signal(
            CalibrationSignal(
                timestamp_seconds=1000.0,
                ensemble_name="ens-A",
                dispatch_success=True,
            )
        )
        channel.record_signal(
            CalibrationSignal(
                timestamp_seconds=1001.0,
                ensemble_name="ens-B",
                dispatch_success=False,
            )
        )

        features_a = channel.windowed_features(
            now_seconds=1002.0, ensemble_name="ens-A"
        )
        features_b = channel.windowed_features(
            now_seconds=1002.0, ensemble_name="ens-B"
        )
        assert features_a.in_window_count == 1
        assert features_a.aggregated_success_rate == 1.0
        assert features_b.in_window_count == 1
        assert features_b.aggregated_success_rate == 0.0


class TestSignalVerdictCorrelationCriterion:
    """The third drift criterion — Pearson on (entropy, verdict_numeric)."""

    def test_correlation_drift_below_threshold_no_exceedance(self) -> None:
        clock = _ScriptedClock(start_seconds=0.0)
        thresholds = CalibrationChannelAuditThresholds(
            trigger_count=4, signal_verdict_correlation_drift=0.20
        )
        channel = CalibrationSignalChannel(audit_thresholds=thresholds, clock=clock)

        def features_with_entropy(entropy: float) -> WindowedSignalFeatures:
            return WindowedSignalFeatures(
                in_window_count=1,
                running_entropy_mean=entropy,
                running_entropy_stdev=0.1,
                has_entropy_basis=True,
                deterministic_anchor_count=0,
                deterministic_anchor_positive_fraction=None,
                aggregated_success_rate=None,
            )

        # Window 1: positive entropy↔proceed correlation
        verdicts: list[CalibrationVerdict] = [
            "proceed",
            "proceed",
            "reflect",
            "reflect",
        ]
        entropies = [3.0, 3.0, 2.0, 2.0]
        for v, e in zip(verdicts, entropies, strict=True):
            channel.record_verdict_outcome(
                verdict=v,
                ensemble_name="ens-A",
                signal_features=features_with_entropy(e),
            )

        # Window 2: similar correlation (no drift)
        for v, e in zip(verdicts, entropies, strict=True):
            channel.record_verdict_outcome(
                verdict=v,
                ensemble_name="ens-A",
                signal_features=features_with_entropy(e),
            )

        diagnostics = channel.audit_diagnostics()
        assert len(diagnostics) == 2
        # Find the correlation criterion's finding in window 2
        corr = next(
            f
            for f in diagnostics[1].criteria_findings
            if f.name == "signal_verdict_correlation_drift"
        )
        assert corr.exceeds is False


# ---------------------------------------------------------------------------
# Cycle 6 WP-B: substrate emission (WP-A carry-forward)
# ---------------------------------------------------------------------------


class _RecordingSink:
    def __init__(self) -> None:
        self.events: list[object] = []

    def consume(self, event: object) -> None:
        self.events.append(event)


class TestDispatchEventSubstrateEmission:
    """Per ADR-023 §Destination 1 — CalibrationSignal at DEBUG.

    The channel emits each validated signal through the configured
    substrate at ``record_signal`` time. ``None`` substrate preserves
    the pre-Cycle-6 path (no emission, no test impact).
    """

    def test_signal_flows_through_substrate_when_configured(self) -> None:
        substrate = DispatchEventSubstrate()
        sink = _RecordingSink()
        substrate.register_sink(sink)
        channel = CalibrationSignalChannel(event_substrate=substrate)

        signal = CalibrationSignal(
            timestamp_seconds=1700000000.0,
            ensemble_name="code-generator",
            dispatch_success=True,
            recent_token_entropy=2.345,
        )
        channel.record_signal(signal)

        assert sink.events == [signal]

    def test_substrate_absent_preserves_pre_cycle_6_path(self) -> None:
        """The channel works as before when no substrate is configured."""
        clock = _ScriptedClock(start_seconds=1000.0)
        channel = CalibrationSignalChannel(clock=clock)
        signal = CalibrationSignal(
            timestamp_seconds=1000.0,
            ensemble_name="code-generator",
            dispatch_success=True,
        )
        # Must not raise.
        channel.record_signal(signal)
        # Channel still accumulates internally per ADR-016.
        features = channel.windowed_features(
            now_seconds=1000.0, ensemble_name="code-generator"
        )
        assert features.in_window_count == 1

    def test_malformed_signals_are_not_emitted(self) -> None:
        """Per mechanism (e) — malformed signals are rejected at the boundary;
        nothing flows through the substrate either.
        """
        substrate = DispatchEventSubstrate()
        sink = _RecordingSink()
        substrate.register_sink(sink)
        channel = CalibrationSignalChannel(event_substrate=substrate)

        with pytest.raises(MalformedSignalError):
            channel.record_signal({"ensemble_name": "x"})  # missing required fields

        assert sink.events == []
