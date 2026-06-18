"""Tests for the Orchestrator-Context Event Sink (Cycle 6 WP-C, ADR-023).

Per ``docs/agentic-serving/system-design.agents.md`` §Module:
Orchestrator-Context Event Sink (L2 — new in Cycle 6). The sink is the
orchestrator-context routing destination of the unified Dispatch Event
Substrate per Inversion N+2 — one substrate fans out to two destinations.

Per ADR-023 §Destination 2:

* The structured observation has exactly seven canonical fields:
  ``dispatched, duration_seconds, model_profile, tier, topaz_skill,
  calibration_verdict, dispatch_id``.
* Missing field values fall back to ``None``; no key is omitted.
* :class:`CalibrationSignal` events are excluded by default; the
  opt-in flag includes them in the end-of-session ``dispatch_log``.
* Session-prefix filtering isolates per-request consumers — same
  pattern as the inference-wait heartbeat scheduler (WP-B advisory).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_orc.agentic.calibration_gate import CalibrationVerdictEvent
from llm_orc.agentic.calibration_signal_channel import CalibrationSignal
from llm_orc.agentic.dispatch_event_substrate import (
    DispatchEventSubstrate,
    DispatchTiming,
)
from llm_orc.agentic.orchestrator_context_event_sink import (
    OrchestratorContextEventSink,
    StructuredObservation,
)
from llm_orc.agentic.tier_router import TierSelection

SESSION_ID = "session-abc"
DISPATCH_ID = "session-abc-dispatch-0001"
OTHER_DISPATCH_ID = "session-xyz-dispatch-0001"


def _start_event(dispatch_id: str = DISPATCH_ID) -> DispatchTiming:
    return DispatchTiming(
        phase="start",
        dispatch_id=dispatch_id,
        ensemble_name="code-generator",
        timestamp_seconds=1000.0,
        model_profile="cheap-cloud",
    )


def _end_event(dispatch_id: str = DISPATCH_ID) -> DispatchTiming:
    return DispatchTiming(
        phase="end",
        dispatch_id=dispatch_id,
        ensemble_name="code-generator",
        timestamp_seconds=1071.5,
        duration_seconds=71.5,
        exit_status="success",
    )


def _tier_selection(dispatch_id: str = DISPATCH_ID) -> TierSelection:
    return TierSelection(
        model_profile="cheap-cloud",
        tier="cheap",
        topaz_skill="code_generation",
        dispatch_id=dispatch_id,
    )


def _verdict_event(dispatch_id: str = DISPATCH_ID) -> CalibrationVerdictEvent:
    return CalibrationVerdictEvent(
        verdict="proceed",
        ensemble_name="code-generator",
        timestamp_seconds=1071.0,
        dispatch_id=dispatch_id,
    )


def _signal(dispatch_id: str = DISPATCH_ID) -> CalibrationSignal:
    return CalibrationSignal(
        timestamp_seconds=1070.0,
        ensemble_name="code-generator",
        dispatch_success=True,
        recent_token_entropy=0.42,
        deterministic_anchor=None,
        dispatch_id=dispatch_id,
    )


def test_observations_for_unknown_dispatch_returns_none() -> None:
    sink = OrchestratorContextEventSink(session_id=SESSION_ID)
    assert sink.observations_for(DISPATCH_ID) is None


def test_observation_carries_canonical_seven_fields() -> None:
    sink = OrchestratorContextEventSink(session_id=SESSION_ID)
    sink.consume(_start_event())
    sink.consume(_tier_selection())
    sink.consume(_verdict_event())
    sink.consume(_end_event())

    observation = sink.observations_for(DISPATCH_ID)

    assert observation == StructuredObservation(
        dispatched="code-generator",
        duration_seconds=71.5,
        model_profile="cheap-cloud",
        tier="cheap",
        topaz_skill="code_generation",
        calibration_verdict="proceed",
        dispatch_id=DISPATCH_ID,
    )


def test_observation_schema_serializes_with_exactly_seven_keys() -> None:
    sink = OrchestratorContextEventSink(session_id=SESSION_ID)
    sink.consume(_start_event())
    sink.consume(_tier_selection())
    sink.consume(_verdict_event())
    sink.consume(_end_event())

    observation = sink.observations_for(DISPATCH_ID)
    assert observation is not None
    payload = json.loads(observation.to_json())

    assert set(payload.keys()) == {
        "dispatched",
        "duration_seconds",
        "model_profile",
        "tier",
        "topaz_skill",
        "calibration_verdict",
        "dispatch_id",
    }


def test_missing_field_values_fall_back_to_none_without_omitting_keys() -> None:
    # Only DispatchTiming(start) — no tier selection, no verdict, no end.
    sink = OrchestratorContextEventSink(session_id=SESSION_ID)
    sink.consume(_start_event())

    observation = sink.observations_for(DISPATCH_ID)
    assert observation is not None
    payload = json.loads(observation.to_json())

    assert payload["dispatched"] == "code-generator"
    assert payload["dispatch_id"] == DISPATCH_ID
    assert payload["duration_seconds"] is None
    assert payload["tier"] is None
    assert payload["topaz_skill"] is None
    assert payload["calibration_verdict"] is None
    # model_profile populated by DispatchTiming.start when no TierSelection yet.
    assert payload["model_profile"] == "cheap-cloud"


def test_calibration_signal_excluded_by_default() -> None:
    sink = OrchestratorContextEventSink(session_id=SESSION_ID)
    sink.consume(_start_event())
    sink.consume(_signal())
    sink.consume(_end_event())

    observation = sink.observations_for(DISPATCH_ID)
    assert observation is not None
    # The seven-field schema does not include calibration_signal data;
    # ensure the signal did not leak into the dispatch_log entry either.
    entries = sink.dispatch_log_entries()
    assert len(entries) == 1
    assert "calibration_signals" not in entries[0]


def test_calibration_signal_included_under_opt_in() -> None:
    sink = OrchestratorContextEventSink(
        session_id=SESSION_ID, routes_calibration_signal=True
    )
    sink.consume(_start_event())
    sink.consume(_signal())
    sink.consume(_end_event())

    entries = sink.dispatch_log_entries()
    assert len(entries) == 1
    assert "calibration_signals" in entries[0]
    assert entries[0]["calibration_signals"][0]["dispatch_success"] is True
    # Seven-field observation schema is still exact — no signal fields.
    observation = sink.observations_for(DISPATCH_ID)
    assert observation is not None
    payload = json.loads(observation.to_json())
    assert set(payload.keys()) == {
        "dispatched",
        "duration_seconds",
        "model_profile",
        "tier",
        "topaz_skill",
        "calibration_verdict",
        "dispatch_id",
    }


def test_cross_session_events_are_ignored() -> None:
    sink = OrchestratorContextEventSink(session_id=SESSION_ID)
    # Different session-prefix → ignored.
    sink.consume(_start_event(dispatch_id=OTHER_DISPATCH_ID))
    sink.consume(_end_event(dispatch_id=OTHER_DISPATCH_ID))

    assert sink.observations_for(OTHER_DISPATCH_ID) is None
    assert sink.dispatch_log_entries() == []


def test_events_without_string_dispatch_id_are_ignored() -> None:
    sink = OrchestratorContextEventSink(session_id=SESSION_ID)
    # Legacy emission site with dispatch_id=None — substrate transition.
    legacy = TierSelection(
        model_profile="cheap-cloud",
        tier="cheap",
        topaz_skill="code_generation",
        dispatch_id=None,
    )
    sink.consume(legacy)

    assert sink.dispatch_log_entries() == []


def test_dispatch_log_preserves_completion_order() -> None:
    sink = OrchestratorContextEventSink(session_id=SESSION_ID)
    first = f"{SESSION_ID}-dispatch-0001"
    second = f"{SESSION_ID}-dispatch-0002"
    # Second dispatch starts before first ends — but completion order
    # is determined by first-event arrival, which equals dispatch order.
    sink.consume(_start_event(dispatch_id=first))
    sink.consume(_start_event(dispatch_id=second))
    sink.consume(_end_event(dispatch_id=first))
    sink.consume(_end_event(dispatch_id=second))

    entries = sink.dispatch_log_entries()
    assert [entry["dispatch_id"] for entry in entries] == [first, second]


def test_substrate_register_and_unregister_lifecycle() -> None:
    substrate = DispatchEventSubstrate()
    sink = OrchestratorContextEventSink(session_id=SESSION_ID)

    sink.register_with(substrate)
    substrate.emit(_start_event())
    substrate.emit(_end_event())
    assert sink.observations_for(DISPATCH_ID) is not None

    sink.unregister_with(substrate)
    # Emissions after unregister do not reach the sink.
    second = f"{SESSION_ID}-dispatch-0002"
    substrate.emit(_start_event(dispatch_id=second))
    assert sink.observations_for(second) is None


def test_unregister_is_idempotent() -> None:
    substrate = DispatchEventSubstrate()
    sink = OrchestratorContextEventSink(session_id=SESSION_ID)
    # Never registered — unregister should not raise.
    sink.unregister_with(substrate)
    # Register, unregister, unregister again — no exception.
    sink.register_with(substrate)
    sink.unregister_with(substrate)
    sink.unregister_with(substrate)


def test_verdict_alone_populates_dispatched_field() -> None:
    # Edge case: timing never fired but verdict did — dispatched still
    # populates from the verdict event's ensemble_name.
    sink = OrchestratorContextEventSink(session_id=SESSION_ID)
    sink.consume(_verdict_event())

    observation = sink.observations_for(DISPATCH_ID)
    assert observation is not None
    assert observation.dispatched == "code-generator"
    assert observation.calibration_verdict == "proceed"


def test_write_dispatch_log_creates_file_with_json_payload(tmp_path: Path) -> None:
    sink = OrchestratorContextEventSink(session_id=SESSION_ID)
    sink.consume(_start_event())
    sink.consume(_tier_selection())
    sink.consume(_verdict_event())
    sink.consume(_end_event())

    log_path = tmp_path / "agentic-sessions" / SESSION_ID / "dispatch_log.json"
    sink.write_dispatch_log(log_path)

    assert log_path.exists()
    payload = json.loads(log_path.read_text())
    assert "dispatch_log" in payload
    assert "entries" in payload["dispatch_log"]
    assert len(payload["dispatch_log"]["entries"]) == 1
    assert payload["dispatch_log"]["entries"][0]["dispatched"] == "code-generator"
    assert payload["dispatch_log"]["entries"][0]["duration_seconds"] == 71.5


def test_write_dispatch_log_with_no_dispatches_still_creates_valid_file(
    tmp_path: Path,
) -> None:
    sink = OrchestratorContextEventSink(session_id=SESSION_ID)
    log_path = tmp_path / "dispatch_log.json"
    sink.write_dispatch_log(log_path)

    assert log_path.exists()
    payload = json.loads(log_path.read_text())
    assert payload == {"dispatch_log": {"entries": []}}


def test_write_dispatch_log_accumulates_across_requests(tmp_path: Path) -> None:
    # A multi-turn session is multiple chat-completions requests, each with its
    # own per-request sink that holds only its own dispatches. Writing to the
    # shared per-session path must accumulate by dispatch_id, not overwrite.
    log_path = tmp_path / SESSION_ID / "dispatch_log.json"
    second_dispatch_id = "session-abc-dispatch-0002"  # same session, next turn

    first = OrchestratorContextEventSink(session_id=SESSION_ID)
    first.consume(_start_event())
    first.consume(_end_event())
    first.write_dispatch_log(log_path)

    second = OrchestratorContextEventSink(session_id=SESSION_ID)
    second.consume(_start_event(dispatch_id=second_dispatch_id))
    second.consume(_end_event(dispatch_id=second_dispatch_id))
    second.write_dispatch_log(log_path)

    entries = json.loads(log_path.read_text())["dispatch_log"]["entries"]
    assert {e["dispatch_id"] for e in entries} == {DISPATCH_ID, second_dispatch_id}


def test_write_dispatch_log_empty_final_request_does_not_wipe_prior(
    tmp_path: Path,
) -> None:
    # The reported bug: the final finish turn carries no dispatches; an
    # overwriting write would wipe the whole session's accumulated log.
    log_path = tmp_path / SESSION_ID / "dispatch_log.json"

    first = OrchestratorContextEventSink(session_id=SESSION_ID)
    first.consume(_start_event())
    first.consume(_end_event())
    first.write_dispatch_log(log_path)

    finish = OrchestratorContextEventSink(session_id=SESSION_ID)
    finish.write_dispatch_log(log_path)  # finish turn — no dispatches

    entries = json.loads(log_path.read_text())["dispatch_log"]["entries"]
    assert [e["dispatch_id"] for e in entries] == [DISPATCH_ID]


def test_signal_for_different_session_ignored_under_opt_in(
    tmp_path: Path,
) -> None:
    # Opt-in does not relax the session-prefix filter.
    sink = OrchestratorContextEventSink(
        session_id=SESSION_ID, routes_calibration_signal=True
    )
    sink.consume(_signal(dispatch_id=OTHER_DISPATCH_ID))
    assert sink.dispatch_log_entries() == []


def test_calibration_signal_alone_creates_no_dispatch_entry_by_default() -> None:
    # Even when the only event for a dispatch is a CalibrationSignal,
    # default exclusion means no dispatch entry is recorded.
    sink = OrchestratorContextEventSink(session_id=SESSION_ID)
    sink.consume(_signal())
    assert sink.dispatch_log_entries() == []
    assert sink.observations_for(DISPATCH_ID) is None


@pytest.mark.parametrize(
    "exit_status",
    ["success", "error", "timeout", "aborted"],
)
def test_duration_populates_for_any_exit_status(exit_status: str) -> None:
    sink = OrchestratorContextEventSink(session_id=SESSION_ID)
    sink.consume(_start_event())
    end_with_status = DispatchTiming(
        phase="end",
        dispatch_id=DISPATCH_ID,
        ensemble_name="code-generator",
        timestamp_seconds=1071.5,
        duration_seconds=71.5,
        exit_status=exit_status,  # type: ignore[arg-type]
    )
    sink.consume(end_with_status)

    observation = sink.observations_for(DISPATCH_ID)
    assert observation is not None
    assert observation.duration_seconds == 71.5
