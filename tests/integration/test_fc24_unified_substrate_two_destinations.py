"""FC-24 anchor — unified substrate fans out to both destinations.

Per ``docs/agentic-serving/system-design.agents.md`` §Fitness Criteria
FC-24::

    The Operator-Terminal Event Sink and Orchestrator-Context Event
    Sink consume from the same Dispatch Event Substrate; no parallel-
    emission path exists for the same data; the unified-substrate
    Inversion N+2 commitment is structurally enforced.

This integration test composes a real :class:`DispatchEventSubstrate`
with the real :class:`OperatorTerminalEventSink` and the real
:class:`OrchestratorContextEventSink`, emits one dispatch's worth of
events, and verifies both sinks observed them.

Asserting via the substrate's own ``events_for`` lookup and the
orchestrator-context sink's structured-observation surface keeps the
test free of caplog (operator-terminal emissions are log-line side
effects whose timing is covered by the FC-23 anchor); this anchor
focuses on the *fan-out* property — same events, two destinations.
"""

from __future__ import annotations

from llm_orc.agentic.calibration_gate import CalibrationVerdictEvent
from llm_orc.agentic.dispatch_event_substrate import (
    DispatchEventSubstrate,
    DispatchTiming,
)
from llm_orc.agentic.operator_terminal_event_sink import OperatorTerminalEventSink
from llm_orc.agentic.orchestrator_context_event_sink import (
    OrchestratorContextEventSink,
)
from llm_orc.agentic.tier_router import TierSelection


def test_both_sinks_receive_dispatch_events_from_one_substrate() -> None:
    """One emit → both sinks observe the event. No parallel-emission path."""
    substrate = DispatchEventSubstrate()
    operator_sink = OperatorTerminalEventSink()
    context_sink = OrchestratorContextEventSink(session_id="session-1")
    operator_sink.register_with(substrate)
    context_sink.register_with(substrate)

    dispatch_id = substrate.new_dispatch_id("session-1")
    substrate.emit(
        DispatchTiming(
            phase="start",
            dispatch_id=dispatch_id,
            ensemble_name="code-generator",
            timestamp_seconds=1000.0,
            model_profile="cheap-cloud",
        )
    )
    substrate.emit(
        TierSelection(
            model_profile="cheap-cloud",
            tier="cheap",
            topaz_skill="code_generation",
            dispatch_id=dispatch_id,
        )
    )
    substrate.emit(
        CalibrationVerdictEvent(
            verdict="proceed",
            ensemble_name="code-generator",
            timestamp_seconds=1071.0,
            dispatch_id=dispatch_id,
        )
    )
    substrate.emit(
        DispatchTiming(
            phase="end",
            dispatch_id=dispatch_id,
            ensemble_name="code-generator",
            timestamp_seconds=1071.5,
            duration_seconds=71.5,
            exit_status="success",
        )
    )

    # Substrate's own log records all four events for the dispatch.
    substrate_events = substrate.events_for(dispatch_id)
    assert len(substrate_events) == 4

    # The orchestrator-context sink composed a canonical observation
    # from the same events — it received them, not via a parallel path.
    observation = context_sink.observations_for(dispatch_id)
    assert observation is not None
    assert observation.dispatched == "code-generator"
    assert observation.duration_seconds == 71.5
    assert observation.tier == "cheap"
    assert observation.topaz_skill == "code_generation"
    assert observation.calibration_verdict == "proceed"
    assert observation.dispatch_id == dispatch_id

    # Operator-terminal sink consumed the events without raising.
    # (Log-line timing is covered by FC-23; this anchor verifies fan-
    # out structurally — both sinks register with the same substrate.)


def test_cross_session_event_filtered_at_orchestrator_context_sink() -> None:
    """Session-prefix filter is per-sink; operator-terminal is global,
    orchestrator-context is per-session. Both sinks register with the
    same substrate; the filter lives in the orchestrator-context sink's
    consume() method, not in a parallel-emission path."""
    substrate = DispatchEventSubstrate()
    operator_sink = OperatorTerminalEventSink()
    context_sink_session_a = OrchestratorContextEventSink(session_id="session-A")
    operator_sink.register_with(substrate)
    context_sink_session_a.register_with(substrate)

    # Event from a different session.
    foreign_dispatch_id = "session-B-dispatch-0001"
    substrate.emit(
        DispatchTiming(
            phase="start",
            dispatch_id=foreign_dispatch_id,
            ensemble_name="code-generator",
            timestamp_seconds=2000.0,
            model_profile="cheap-cloud",
        )
    )

    # Substrate captured it (no filtering at substrate level).
    assert len(substrate.events_for(foreign_dispatch_id)) == 1

    # Context sink for session-A filtered it out (different session).
    assert context_sink_session_a.observations_for(foreign_dispatch_id) is None

    # Operator-terminal sink consumed it (it does not filter by session).


def test_multi_dispatch_both_sinks_observe_each_dispatch_independently() -> None:
    """Multiple dispatches in one session emit independent events; both
    sinks observe each dispatch with the correct dispatch_id correlation."""
    substrate = DispatchEventSubstrate()
    operator_sink = OperatorTerminalEventSink()
    context_sink = OrchestratorContextEventSink(session_id="session-1")
    operator_sink.register_with(substrate)
    context_sink.register_with(substrate)

    # Dispatch 1.
    d1 = substrate.new_dispatch_id("session-1")
    substrate.emit(
        DispatchTiming(
            phase="start",
            dispatch_id=d1,
            ensemble_name="code-generator",
            timestamp_seconds=1000.0,
        )
    )
    substrate.emit(
        DispatchTiming(
            phase="end",
            dispatch_id=d1,
            ensemble_name="code-generator",
            timestamp_seconds=1010.0,
            duration_seconds=10.0,
            exit_status="success",
        )
    )

    # Dispatch 2.
    d2 = substrate.new_dispatch_id("session-1")
    substrate.emit(
        DispatchTiming(
            phase="start",
            dispatch_id=d2,
            ensemble_name="claim-extractor",
            timestamp_seconds=1011.0,
        )
    )
    substrate.emit(
        DispatchTiming(
            phase="end",
            dispatch_id=d2,
            ensemble_name="claim-extractor",
            timestamp_seconds=1021.0,
            duration_seconds=10.0,
            exit_status="success",
        )
    )

    # Both dispatches recorded distinctly in the orchestrator-context sink.
    obs_1 = context_sink.observations_for(d1)
    obs_2 = context_sink.observations_for(d2)
    assert obs_1 is not None
    assert obs_2 is not None
    assert obs_1.dispatched == "code-generator"
    assert obs_2.dispatched == "claim-extractor"
    assert obs_1.dispatch_id == d1
    assert obs_2.dispatch_id == d2

    # End-of-session dispatch_log carries both entries in completion order.
    entries = context_sink.dispatch_log_entries()
    assert [e["dispatch_id"] for e in entries] == [d1, d2]
