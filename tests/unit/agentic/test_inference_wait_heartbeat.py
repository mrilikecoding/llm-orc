"""Tests for the Inference-Wait Heartbeat scheduler (Cycle 6 WP-B piece 5).

Per ``docs/agentic-serving/scenarios.md`` §Observability Event Routing
scenario "Inference-wait heartbeat fires after `heartbeat_interval_seconds`
of inactivity" and ``system-design.agents.md`` §Operator-Terminal Event
Sink (heartbeat fires from open-request tracking; sink formats the
``INFO: inference wait:`` line).

The scheduler:

* Resets the activity timer when a :class:`DispatchTiming` event whose
  ``dispatch_id`` carries the scheduler's ``session_id`` arrives via
  :meth:`consume` (registered as a substrate :class:`EventSink`).
* Resets the activity timer when :meth:`emit_tool_call_log` is invoked
  (the scheduler implements the ``ToolCallEmitLogger`` Protocol and is
  injected into Tool Dispatch in place of the bare operator-terminal
  sink so per-request emits are observable).
* Fires heartbeats via the sink's :meth:`emit_heartbeat` action when
  elapsed inactivity meets or exceeds ``interval_seconds``, recurring
  at the interval until activity resumes.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from llm_orc.agentic.calibration_signal_channel import CalibrationSignal
from llm_orc.agentic.dispatch_event_substrate import (
    DispatchEventSubstrate,
    DispatchTiming,
)
from llm_orc.agentic.inference_wait_heartbeat import (
    InferenceWaitHeartbeatScheduler,
)
from llm_orc.agentic.operator_terminal_event_sink import OperatorTerminalEventSink


class _MutableClock:
    """Deterministic clock for activity-elapsed assertions."""

    def __init__(self, *, start: float = 1_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _make_scheduler(
    *,
    clock: _MutableClock,
    interval: float = 30.0,
    session_id: str = "session-A",
) -> tuple[InferenceWaitHeartbeatScheduler, OperatorTerminalEventSink]:
    sink = OperatorTerminalEventSink()
    scheduler = InferenceWaitHeartbeatScheduler(
        sink=sink,
        session_id=session_id,
        interval_seconds=interval,
        clock=clock,
    )
    return scheduler, sink


class TestInactivityFiring:
    def test_emits_heartbeat_when_inactivity_meets_interval(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        clock = _MutableClock(start=1_000.0)
        scheduler, _sink = _make_scheduler(clock=clock, interval=30.0)
        caplog.set_level(logging.INFO, logger="llm_orc.agentic.operator_terminal")

        clock.advance(30.0)
        fired = scheduler.check_and_emit_if_inactive()

        assert fired is True
        (record,) = [
            r for r in caplog.records if r.name == "llm_orc.agentic.operator_terminal"
        ]
        assert "inference wait" in record.message
        assert "elapsed=30" in record.message
        assert "session_id=session-A" in record.message

    def test_does_not_emit_before_interval_elapses(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        clock = _MutableClock(start=1_000.0)
        scheduler, _sink = _make_scheduler(clock=clock, interval=30.0)
        caplog.set_level(logging.INFO, logger="llm_orc.agentic.operator_terminal")

        clock.advance(29.999)
        fired = scheduler.check_and_emit_if_inactive()

        assert fired is False
        assert caplog.records == []

    def test_recurs_at_interval_after_continued_inactivity(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        clock = _MutableClock(start=1_000.0)
        scheduler, _sink = _make_scheduler(clock=clock, interval=30.0)
        caplog.set_level(logging.INFO, logger="llm_orc.agentic.operator_terminal")

        clock.advance(30.0)
        assert scheduler.check_and_emit_if_inactive() is True
        clock.advance(30.0)
        assert scheduler.check_and_emit_if_inactive() is True

        emits = [r for r in caplog.records if "inference wait" in r.message]
        assert len(emits) == 2


class TestActivityReset:
    def test_consume_dispatch_timing_for_session_resets_activity(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        clock = _MutableClock(start=1_000.0)
        scheduler, _sink = _make_scheduler(
            clock=clock, interval=30.0, session_id="session-A"
        )
        caplog.set_level(logging.INFO, logger="llm_orc.agentic.operator_terminal")

        clock.advance(20.0)
        scheduler.consume(
            DispatchTiming(
                phase="start",
                dispatch_id="session-A-dispatch-0001",
                ensemble_name="code-generator",
                timestamp_seconds=clock.now,
            )
        )
        clock.advance(20.0)
        fired = scheduler.check_and_emit_if_inactive()

        assert fired is False
        assert [r for r in caplog.records if "inference wait" in r.message] == []

    def test_consume_dispatch_timing_for_other_session_does_not_reset(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        clock = _MutableClock(start=1_000.0)
        scheduler, _sink = _make_scheduler(
            clock=clock, interval=30.0, session_id="session-A"
        )
        caplog.set_level(logging.INFO, logger="llm_orc.agentic.operator_terminal")

        clock.advance(15.0)
        scheduler.consume(
            DispatchTiming(
                phase="start",
                dispatch_id="session-B-dispatch-0001",  # other session
                ensemble_name="code-generator",
                timestamp_seconds=clock.now,
            )
        )
        clock.advance(15.0)
        fired = scheduler.check_and_emit_if_inactive()

        assert fired is True

    def test_consume_non_dispatch_timing_event_does_not_reset(self) -> None:
        clock = _MutableClock(start=1_000.0)
        scheduler, _sink = _make_scheduler(clock=clock, interval=30.0)

        clock.advance(15.0)
        # Pass a CalibrationSignal with our session_id — only DispatchTiming
        # events count as activity per the scenario.
        scheduler.consume(
            CalibrationSignal(
                timestamp_seconds=clock.now,
                ensemble_name="code-generator",
                dispatch_success=True,
                recent_token_entropy=0.5,
                deterministic_anchor=None,
                dispatch_id="session-A-dispatch-0001",
            )
        )
        clock.advance(15.0)
        fired = scheduler.check_and_emit_if_inactive()
        assert fired is True

    def test_emit_tool_call_log_resets_activity_and_forwards(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        clock = _MutableClock(start=1_000.0)
        scheduler, _sink = _make_scheduler(clock=clock, interval=30.0)
        caplog.set_level(logging.INFO, logger="llm_orc.agentic.operator_terminal")

        clock.advance(20.0)
        scheduler.emit_tool_call_log(
            tool_name="invoke_ensemble", dispatch_id="session-A-dispatch-0001"
        )
        clock.advance(20.0)
        fired = scheduler.check_and_emit_if_inactive()

        assert fired is False
        # Forwarded line is present; no inference-wait yet.
        tool_call_records = [r for r in caplog.records if "tool-call emit" in r.message]
        assert len(tool_call_records) == 1
        assert "dispatch_id=session-A-dispatch-0001" in tool_call_records[0].message
        assert [r for r in caplog.records if "inference wait" in r.message] == []

    def test_emit_tool_call_log_for_other_session_does_not_reset_activity(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        clock = _MutableClock(start=1_000.0)
        scheduler, _sink = _make_scheduler(
            clock=clock, interval=30.0, session_id="session-A"
        )
        caplog.set_level(logging.INFO, logger="llm_orc.agentic.operator_terminal")

        clock.advance(15.0)
        # The emit always forwards (the line is still useful), but only
        # session-matching emits reset this scheduler's activity timer.
        scheduler.emit_tool_call_log(
            tool_name="invoke_ensemble", dispatch_id="session-B-dispatch-0001"
        )
        clock.advance(15.0)
        fired = scheduler.check_and_emit_if_inactive()

        assert fired is True


class TestSubstrateLifecycle:
    @pytest.mark.asyncio
    async def test_start_and_stop_register_and_unregister_with_substrate(
        self,
    ) -> None:
        """register_with()/unregister_with() compose with the substrate so
        the scheduler is added once, removed at request close."""
        substrate = DispatchEventSubstrate()
        clock = _MutableClock(start=1_000.0)
        scheduler, _sink = _make_scheduler(clock=clock)

        scheduler.register_with(substrate)
        # The substrate fans out an event — the scheduler observes.
        substrate.emit(
            DispatchTiming(
                phase="start",
                dispatch_id="session-A-dispatch-0001",
                ensemble_name="code-generator",
                timestamp_seconds=clock.now,
            )
        )
        # After unregister the scheduler must not observe further events.
        scheduler.unregister_with(substrate)
        substrate.emit(
            DispatchTiming(
                phase="start",
                dispatch_id="session-A-dispatch-0002",
                ensemble_name="code-generator",
                timestamp_seconds=clock.now,
            )
        )
        # No exceptions; un-registering twice is a no-op (idempotent).
        scheduler.unregister_with(substrate)


class TestAsyncLoop:
    @pytest.mark.asyncio
    async def test_run_emits_heartbeat_then_cancels_cleanly(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The async run() loop emits a heartbeat once after the configured
        interval and stops cleanly when cancelled."""
        sink = OperatorTerminalEventSink()
        scheduler = InferenceWaitHeartbeatScheduler(
            sink=sink, session_id="session-A", interval_seconds=0.05
        )
        caplog.set_level(logging.INFO, logger="llm_orc.agentic.operator_terminal")

        task = asyncio.create_task(scheduler.run())
        try:
            await asyncio.sleep(0.07)
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        heartbeats = [r for r in caplog.records if "inference wait" in r.message]
        assert len(heartbeats) >= 1
