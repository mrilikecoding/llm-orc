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
    DEFAULT_CALIBRATION_CHECKER_ENSEMBLE,
    DEFAULT_CALIBRATION_N,
    CalibrationGate,
    CalibrationRecord,
    CalibrationStatus,
    EnsembleBackedChecker,
    QualitySignal,
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
            returns if returns is not None else {"synthesis": "signal: positive"}
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
            returns={"synthesis": "signal: positive\nreason: clear on-task response"}
        )
        checker = EnsembleBackedChecker(invoker=invoker)
        signal = await checker.check(
            ensemble_name="composed-a", raw_result={"results": {"a": "x"}}
        )
        assert signal == "positive"

    @pytest.mark.asyncio
    async def test_synthesis_negative_is_parsed(self) -> None:
        invoker = _RecordingInvoker(
            returns={"synthesis": "signal: negative\nreason: hallucinated content"}
        )
        checker = EnsembleBackedChecker(invoker=invoker)
        signal = await checker.check(ensemble_name="composed-a", raw_result={})
        assert signal == "negative"

    @pytest.mark.asyncio
    async def test_synthesis_absent_is_parsed(self) -> None:
        invoker = _RecordingInvoker(
            returns={"synthesis": "signal: absent\nreason: empty output"}
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
                "synthesis": None,
            }
        )
        checker = EnsembleBackedChecker(invoker=invoker)
        signal = await checker.check(ensemble_name="composed-b", raw_result={})
        assert signal == "positive"

    @pytest.mark.asyncio
    async def test_unparseable_response_yields_absent(self) -> None:
        invoker = _RecordingInvoker(
            returns={"synthesis": "the ensemble looked fine but I am not sure"}
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
            returns={"synthesis": "SIGNAL: Positive (ensemble answered on-task)"}
        )
        checker = EnsembleBackedChecker(invoker=invoker)
        assert await checker.check(ensemble_name="x", raw_result={}) == "positive"
