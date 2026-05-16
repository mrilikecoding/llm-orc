"""Tests for the Dispatch Event Substrate (Cycle 6 WP-A, ADR-023).

Per ``docs/agentic-serving/system-design.agents.md`` §Module:
Dispatch Event Substrate (L1 — new in Cycle 6). The substrate is the
unified event-emission surface (Inversion N+2) — one infrastructure,
two routing destinations (operator-terminal and orchestrator-context,
shipped by WP-B and WP-C). It owns ``DispatchTiming`` event allocation,
``dispatch_id`` correlation identifier generation, registered-sink
fan-out, and post-hoc event query.

Per ADR-023 §"Event-emission substrate":

* ``DispatchTiming`` is emitted twice per dispatch (``phase="start"``
  before the ensemble executes; ``phase="end"`` after it returns
  control). ``phase="end"`` carries ``duration_seconds`` and
  ``exit_status``.
* ``dispatch_id`` is a session-scoped monotonic counter per
  open-decision-point C6-1; the same identifier flows through every
  event emitted during one dispatch, the envelope's
  ``diagnostics.dispatch_id`` (WP-D), and the artifact path's
  ``<dispatch_id>`` segment (WP-E).
* Sinks register via :meth:`DispatchEventSubstrate.register_sink` and
  consume events synchronously. Sink exceptions are isolated — one
  failing sink does not affect other sinks or the producing path.
"""

from __future__ import annotations

import dataclasses

import pytest

from llm_orc.agentic.dispatch_event_substrate import (
    DispatchEventSubstrate,
    DispatchTiming,
    EventSink,
)

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _RecordingSink:
    """Captures every event passed to :meth:`consume` for assertion."""

    def __init__(self) -> None:
        self.events: list[object] = []

    def consume(self, event: object) -> None:
        self.events.append(event)


class _ExplodingSink:
    """Always raises in :meth:`consume` — exercises sink-isolation."""

    def __init__(self) -> None:
        self.calls: int = 0

    def consume(self, event: object) -> None:
        self.calls += 1
        raise RuntimeError("sink intentionally fails for isolation test")


# ---------------------------------------------------------------------------
# DispatchTiming event shape
# ---------------------------------------------------------------------------


class TestDispatchTimingShape:
    """The DispatchTiming event satisfies ADR-023's bounded extension."""

    def test_start_phase_carries_call_site_context(self) -> None:
        event = DispatchTiming(
            phase="start",
            dispatch_id="session-1-dispatch-0001",
            ensemble_name="code-generator",
            timestamp_seconds=1700000000.0,
            model_profile="agentic-tier-cheap-general",
        )
        assert event.phase == "start"
        assert event.dispatch_id == "session-1-dispatch-0001"
        assert event.ensemble_name == "code-generator"
        assert event.model_profile == "agentic-tier-cheap-general"
        assert event.duration_seconds is None
        assert event.exit_status is None

    def test_end_phase_carries_duration_and_exit_status(self) -> None:
        event = DispatchTiming(
            phase="end",
            dispatch_id="session-1-dispatch-0001",
            ensemble_name="code-generator",
            timestamp_seconds=1700000061.44,
            duration_seconds=61.44,
            exit_status="success",
        )
        assert event.phase == "end"
        assert event.duration_seconds == pytest.approx(61.44)
        assert event.exit_status == "success"

    def test_event_is_frozen(self) -> None:
        """Per system-design.agents.md — events are frozen value types."""
        event = DispatchTiming(
            phase="start",
            dispatch_id="session-1-dispatch-0001",
            ensemble_name="code-generator",
            timestamp_seconds=1700000000.0,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            event.phase = "end"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# dispatch_id allocation
# ---------------------------------------------------------------------------


class TestDispatchIdAllocation:
    """``new_dispatch_id`` is session-scoped and monotonic per C6-1."""

    def test_first_dispatch_id_per_session(self) -> None:
        substrate = DispatchEventSubstrate()
        dispatch_id = substrate.new_dispatch_id("2026-05-15T14:32:08Z-a7f3")
        assert dispatch_id == "2026-05-15T14:32:08Z-a7f3-dispatch-0001"

    def test_counter_advances_within_one_session(self) -> None:
        substrate = DispatchEventSubstrate()
        first = substrate.new_dispatch_id("session-A")
        second = substrate.new_dispatch_id("session-A")
        third = substrate.new_dispatch_id("session-A")
        assert first == "session-A-dispatch-0001"
        assert second == "session-A-dispatch-0002"
        assert third == "session-A-dispatch-0003"

    def test_counters_are_independent_per_session(self) -> None:
        substrate = DispatchEventSubstrate()
        a1 = substrate.new_dispatch_id("session-A")
        b1 = substrate.new_dispatch_id("session-B")
        a2 = substrate.new_dispatch_id("session-A")
        b2 = substrate.new_dispatch_id("session-B")
        assert a1 == "session-A-dispatch-0001"
        assert b1 == "session-B-dispatch-0001"
        assert a2 == "session-A-dispatch-0002"
        assert b2 == "session-B-dispatch-0002"


# ---------------------------------------------------------------------------
# Sink registration and fan-out
# ---------------------------------------------------------------------------


class TestSinkFanOut:
    """Every registered sink receives every emitted event in order."""

    def test_single_sink_receives_emitted_event(self) -> None:
        substrate = DispatchEventSubstrate()
        sink = _RecordingSink()
        substrate.register_sink(sink)
        event = DispatchTiming(
            phase="start",
            dispatch_id="session-A-dispatch-0001",
            ensemble_name="code-generator",
            timestamp_seconds=1700000000.0,
        )
        substrate.emit(event)
        assert sink.events == [event]

    def test_multiple_sinks_each_receive_every_event(self) -> None:
        substrate = DispatchEventSubstrate()
        sink_a, sink_b = _RecordingSink(), _RecordingSink()
        substrate.register_sink(sink_a)
        substrate.register_sink(sink_b)
        start = DispatchTiming(
            phase="start",
            dispatch_id="session-A-dispatch-0001",
            ensemble_name="code-generator",
            timestamp_seconds=1700000000.0,
        )
        end = DispatchTiming(
            phase="end",
            dispatch_id="session-A-dispatch-0001",
            ensemble_name="code-generator",
            timestamp_seconds=1700000010.0,
            duration_seconds=10.0,
            exit_status="success",
        )
        substrate.emit(start)
        substrate.emit(end)
        assert sink_a.events == [start, end]
        assert sink_b.events == [start, end]

    def test_emit_isolates_sink_exceptions(self) -> None:
        """One failing sink does not block other sinks or producers."""
        substrate = DispatchEventSubstrate()
        bad, good = _ExplodingSink(), _RecordingSink()
        substrate.register_sink(bad)
        substrate.register_sink(good)
        event = DispatchTiming(
            phase="start",
            dispatch_id="session-A-dispatch-0001",
            ensemble_name="code-generator",
            timestamp_seconds=1700000000.0,
        )
        # Producer path must not see the bad sink's exception.
        substrate.emit(event)
        assert bad.calls == 1
        assert good.events == [event]

    def test_emit_without_registered_sinks_is_a_no_op(self) -> None:
        """Substrate-emit happens regardless of sink registration (WP-A)."""
        substrate = DispatchEventSubstrate()
        event = DispatchTiming(
            phase="start",
            dispatch_id="session-A-dispatch-0001",
            ensemble_name="code-generator",
            timestamp_seconds=1700000000.0,
        )
        substrate.emit(event)  # must not raise


# ---------------------------------------------------------------------------
# Post-hoc event query
# ---------------------------------------------------------------------------


class TestEventsForQuery:
    """``events_for`` reconstructs the event log per dispatch_id."""

    def test_returns_events_in_emission_order(self) -> None:
        substrate = DispatchEventSubstrate()
        dispatch_id = substrate.new_dispatch_id("session-A")
        start = DispatchTiming(
            phase="start",
            dispatch_id=dispatch_id,
            ensemble_name="code-generator",
            timestamp_seconds=1700000000.0,
        )
        end = DispatchTiming(
            phase="end",
            dispatch_id=dispatch_id,
            ensemble_name="code-generator",
            timestamp_seconds=1700000010.0,
            duration_seconds=10.0,
            exit_status="success",
        )
        substrate.emit(start)
        substrate.emit(end)
        assert substrate.events_for(dispatch_id) == [start, end]

    def test_separates_events_by_dispatch_id(self) -> None:
        substrate = DispatchEventSubstrate()
        first = substrate.new_dispatch_id("session-A")
        second = substrate.new_dispatch_id("session-A")
        event_first = DispatchTiming(
            phase="start",
            dispatch_id=first,
            ensemble_name="code-generator",
            timestamp_seconds=1700000000.0,
        )
        event_second = DispatchTiming(
            phase="start",
            dispatch_id=second,
            ensemble_name="text-summarizer",
            timestamp_seconds=1700000020.0,
        )
        substrate.emit(event_first)
        substrate.emit(event_second)
        assert substrate.events_for(first) == [event_first]
        assert substrate.events_for(second) == [event_second]

    def test_unknown_dispatch_id_returns_empty_list(self) -> None:
        substrate = DispatchEventSubstrate()
        assert substrate.events_for("never-emitted") == []

    def test_events_without_dispatch_id_are_not_indexed_for_query(self) -> None:
        """Legacy emission sites may produce dispatch_id=None during Cycle 6.

        Per ADR-023 §"`dispatch_id` correlation identifier on existing events"
        — ``dispatch_id: None`` is allowed during transition. Those events
        still fan out to sinks but cannot be retrieved by dispatch_id.
        """
        substrate = DispatchEventSubstrate()
        sink = _RecordingSink()
        substrate.register_sink(sink)

        class _OrphanEvent:
            dispatch_id: str | None = None

        orphan = _OrphanEvent()
        substrate.emit(orphan)
        assert sink.events == [orphan]
        assert substrate.events_for("anything") == []


# ---------------------------------------------------------------------------
# EventSink contract
# ---------------------------------------------------------------------------


class TestEventSinkProtocol:
    """The EventSink protocol is the minimum sink contract."""

    def test_recording_sink_satisfies_protocol(self) -> None:
        sink: EventSink = _RecordingSink()
        # If the assignment type-checks, the runtime is also fine.
        assert hasattr(sink, "consume")


# ---------------------------------------------------------------------------
# Bounded event-model extension (ADR-023 §"Event-emission substrate")
# ---------------------------------------------------------------------------


class TestBoundedEventModelExtension:
    """The Cycle 6 extension is bounded: one new event type + dispatch_id on
    four existing event types.

    Per ADR-023 §"Event-emission substrate" — the extension is additive only.
    """

    def test_dispatch_timing_is_the_one_new_event_type(self) -> None:
        from llm_orc.agentic import dispatch_event_substrate

        # The new event type lives on the substrate module per system-design
        # §"Module: Dispatch Event Substrate" (owns DispatchTiming).
        assert hasattr(dispatch_event_substrate, "DispatchTiming")

    def test_existing_event_types_gain_dispatch_id_field(self) -> None:
        """All four extended event types declare ``dispatch_id``."""
        from llm_orc.agentic.calibration_gate import CalibrationVerdictEvent
        from llm_orc.agentic.calibration_signal_channel import CalibrationSignal
        from llm_orc.agentic.tier_router import TierSelection
        from llm_orc.agentic.tier_router_audit import AuditDiagnostic

        for event_cls in (
            TierSelection,
            AuditDiagnostic,
            CalibrationSignal,
            CalibrationVerdictEvent,
        ):
            assert "dispatch_id" in {
                field.name for field in dataclasses.fields(event_cls)
            }, f"{event_cls.__name__} must declare a dispatch_id field per ADR-023"

    def test_dispatch_id_default_is_none_on_extended_event_types(self) -> None:
        """``None`` is allowed during the progressive conversion per ADR-023."""
        from llm_orc.agentic.calibration_signal_channel import CalibrationSignal
        from llm_orc.agentic.tier_router import TierSelection

        # Construct without specifying dispatch_id — default-None.
        selection = TierSelection(
            model_profile="agentic-tier-cheap-general",
            tier="cheap",
            topaz_skill="code_generation",
        )
        signal = CalibrationSignal(
            timestamp_seconds=1700000000.0,
            ensemble_name="code-generator",
            dispatch_success=True,
        )
        assert selection.dispatch_id is None
        assert signal.dispatch_id is None
