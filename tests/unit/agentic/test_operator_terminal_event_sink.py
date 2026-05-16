"""Tests for the Operator-Terminal Event Sink (Cycle 6 WP-B, ADR-023).

Per ``docs/agentic-serving/system-design.agents.md`` §Module:
Operator-Terminal Event Sink (L3 — new in Cycle 6). The sink is the
operator-terminal routing destination of the unified Dispatch Event
Substrate per Inversion N+2 — one substrate, two destinations.

Per ADR-023 §Destination 1:

* Each event class formats to one or more human-readable lines.
* Lines emit at INFO level except ``CalibrationSignal`` at DEBUG.
* ``AuditDiagnostic`` emits one line per drift criterion finding.
* Tool-call-emit and inference-wait heartbeat lines are action-driven
  liveness signals — the Serving Layer calls into the sink rather than
  the sink scheduling timers itself.
"""

from __future__ import annotations

import logging

import pytest

from llm_orc.agentic.calibration_gate import CalibrationVerdictEvent
from llm_orc.agentic.calibration_signal_channel import CalibrationSignal
from llm_orc.agentic.dispatch_event_substrate import (
    DispatchEventSubstrate,
    DispatchTiming,
)
from llm_orc.agentic.operator_terminal_event_sink import OperatorTerminalEventSink
from llm_orc.agentic.tier_router_audit import AuditDiagnostic, CriterionFinding


@pytest.fixture
def sink_and_caplog(
    caplog: pytest.LogCaptureFixture,
) -> tuple[OperatorTerminalEventSink, pytest.LogCaptureFixture]:
    """Build a sink wired to caplog at DEBUG (so DEBUG lines are captured)."""
    caplog.set_level(logging.DEBUG, logger="llm_orc.agentic.operator_terminal")
    sink = OperatorTerminalEventSink()
    return sink, caplog


# ---------------------------------------------------------------------------
# DispatchTiming formatting
# ---------------------------------------------------------------------------


class TestDispatchTimingFormatting:
    """Per ADR-023 §Destination 1 — DispatchTiming start/end lines."""

    def test_start_line_carries_ensemble_profile_and_dispatch_id(
        self,
        sink_and_caplog: tuple[OperatorTerminalEventSink, pytest.LogCaptureFixture],
    ) -> None:
        sink, caplog = sink_and_caplog
        sink.consume(
            DispatchTiming(
                phase="start",
                dispatch_id="session-A-dispatch-0001",
                ensemble_name="code-generator",
                timestamp_seconds=1700000000.0,
                model_profile="agentic-tier-cheap-general",
            )
        )
        (record,) = caplog.records
        assert record.levelno == logging.INFO
        assert "dispatch start" in record.message
        assert "ensemble=code-generator" in record.message
        assert "profile=agentic-tier-cheap-general" in record.message
        assert "dispatch_id=session-A-dispatch-0001" in record.message

    def test_start_line_renders_unknown_profile_as_question_mark(
        self,
        sink_and_caplog: tuple[OperatorTerminalEventSink, pytest.LogCaptureFixture],
    ) -> None:
        sink, caplog = sink_and_caplog
        sink.consume(
            DispatchTiming(
                phase="start",
                dispatch_id="session-A-dispatch-0001",
                ensemble_name="code-generator",
                timestamp_seconds=1700000000.0,
            )
        )
        (record,) = caplog.records
        assert "profile=?" in record.message

    def test_end_line_carries_duration_and_exit_status(
        self,
        sink_and_caplog: tuple[OperatorTerminalEventSink, pytest.LogCaptureFixture],
    ) -> None:
        sink, caplog = sink_and_caplog
        sink.consume(
            DispatchTiming(
                phase="end",
                dispatch_id="session-A-dispatch-0001",
                ensemble_name="code-generator",
                timestamp_seconds=1700000061.44,
                duration_seconds=61.44,
                exit_status="success",
            )
        )
        (record,) = caplog.records
        assert record.levelno == logging.INFO
        assert "dispatch end" in record.message
        assert "duration=61.440" in record.message
        assert "exit=success" in record.message


# ---------------------------------------------------------------------------
# TierSelection / CalibrationVerdictEvent formatting
# ---------------------------------------------------------------------------


class TestTierSelectionFormatting:
    def test_emits_one_info_line(
        self,
        sink_and_caplog: tuple[OperatorTerminalEventSink, pytest.LogCaptureFixture],
    ) -> None:
        from llm_orc.agentic.tier_router import TierSelection

        sink, caplog = sink_and_caplog
        sink.consume(
            TierSelection(
                model_profile="agentic-tier-cheap-general",
                tier="cheap",
                topaz_skill="code_generation",
                dispatch_id="session-A-dispatch-0001",
            )
        )
        (record,) = caplog.records
        assert record.levelno == logging.INFO
        assert "tier selection" in record.message
        assert "profile=agentic-tier-cheap-general" in record.message
        assert "tier=cheap" in record.message
        assert "topaz_skill=code_generation" in record.message
        assert "dispatch_id=session-A-dispatch-0001" in record.message


class TestCalibrationVerdictFormatting:
    def test_emits_one_info_line(
        self,
        sink_and_caplog: tuple[OperatorTerminalEventSink, pytest.LogCaptureFixture],
    ) -> None:
        sink, caplog = sink_and_caplog
        sink.consume(
            CalibrationVerdictEvent(
                verdict="proceed",
                ensemble_name="code-generator",
                timestamp_seconds=1700000000.0,
                dispatch_id="session-A-dispatch-0001",
            )
        )
        (record,) = caplog.records
        assert record.levelno == logging.INFO
        assert "calibration verdict" in record.message
        assert "proceed" in record.message
        assert "ensemble=code-generator" in record.message
        assert "dispatch_id=session-A-dispatch-0001" in record.message


# ---------------------------------------------------------------------------
# AuditDiagnostic formatting (one line per criterion finding)
# ---------------------------------------------------------------------------


class TestAuditDiagnosticFormatting:
    """ADR-023 §Destination 1 — one line per drift criterion finding."""

    def test_one_line_per_criterion_finding(
        self,
        sink_and_caplog: tuple[OperatorTerminalEventSink, pytest.LogCaptureFixture],
    ) -> None:
        sink, caplog = sink_and_caplog
        diagnostic = AuditDiagnostic(
            window_id=7,
            verdict="warn",
            timestamp_seconds=1700000000.0,
            consumption_count=42,
            criteria_findings=(
                CriterionFinding(
                    name="verdict_distribution_shift",
                    value=0.18,
                    threshold=0.15,
                    exceeds=True,
                    severe=False,
                ),
                CriterionFinding(
                    name="escalation_vs_outcome_correlation",
                    value=0.08,
                    threshold=0.10,
                    exceeds=False,
                    severe=False,
                ),
            ),
            dispatch_id="session-A-dispatch-0042",
        )
        sink.consume(diagnostic)
        assert len(caplog.records) == 2
        messages = [r.message for r in caplog.records]
        assert all("audit diagnostic" in m for m in messages)
        assert "criterion=verdict_distribution_shift" in messages[0]
        assert "exceeds=True" in messages[0]
        assert "criterion=escalation_vs_outcome_correlation" in messages[1]
        assert "exceeds=False" in messages[1]
        assert all("dispatch_id=session-A-dispatch-0042" in m for m in messages)


# ---------------------------------------------------------------------------
# CalibrationSignal formatting (DEBUG level)
# ---------------------------------------------------------------------------


class TestCalibrationSignalFormatting:
    """ADR-023 §Destination 1 — CalibrationSignal at DEBUG level."""

    def test_emits_at_debug_level(
        self,
        sink_and_caplog: tuple[OperatorTerminalEventSink, pytest.LogCaptureFixture],
    ) -> None:
        sink, caplog = sink_and_caplog
        sink.consume(
            CalibrationSignal(
                timestamp_seconds=1700000000.0,
                ensemble_name="code-generator",
                dispatch_success=True,
                recent_token_entropy=2.345,
                deterministic_anchor=True,
                dispatch_id="session-A-dispatch-0001",
            )
        )
        (record,) = caplog.records
        assert record.levelno == logging.DEBUG
        assert "calibration signal" in record.message
        assert "ensemble=code-generator" in record.message
        assert "success=True" in record.message
        assert "entropy=2.3450" in record.message
        assert "anchor=true" in record.message

    def test_debug_lines_suppressed_at_info_level(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When the logger is at INFO, CalibrationSignal lines do not surface.

        Confirms the discrimination at the substrate boundary: the sink
        emits at DEBUG; the operator's configured verbosity controls
        whether it surfaces.
        """
        caplog.set_level(logging.INFO, logger="llm_orc.agentic.operator_terminal")
        sink = OperatorTerminalEventSink()
        sink.consume(
            CalibrationSignal(
                timestamp_seconds=1700000000.0,
                ensemble_name="code-generator",
                dispatch_success=True,
            )
        )
        assert caplog.records == []


# ---------------------------------------------------------------------------
# Action surfaces (liveness signals + startup warnings)
# ---------------------------------------------------------------------------


class TestActionSurfaces:
    def test_tool_call_emit_log(
        self,
        sink_and_caplog: tuple[OperatorTerminalEventSink, pytest.LogCaptureFixture],
    ) -> None:
        sink, caplog = sink_and_caplog
        sink.emit_tool_call_log(
            tool_name="invoke_ensemble", dispatch_id="session-A-dispatch-0001"
        )
        (record,) = caplog.records
        assert record.levelno == logging.INFO
        assert "tool-call emit" in record.message
        assert "tool=invoke_ensemble" in record.message
        assert "dispatch_id=session-A-dispatch-0001" in record.message

    def test_heartbeat(
        self,
        sink_and_caplog: tuple[OperatorTerminalEventSink, pytest.LogCaptureFixture],
    ) -> None:
        sink, caplog = sink_and_caplog
        sink.emit_heartbeat(session_id="session-A", elapsed_seconds=30.0)
        (record,) = caplog.records
        assert record.levelno == logging.INFO
        assert "inference wait" in record.message
        assert "elapsed=30" in record.message
        assert "session_id=session-A" in record.message

    def test_validation_warning(
        self,
        sink_and_caplog: tuple[OperatorTerminalEventSink, pytest.LogCaptureFixture],
    ) -> None:
        sink, caplog = sink_and_caplog
        sink.emit_validation_warning(
            yaml_path="/path/to/fan-out-test.yaml",
            error="Input should be a valid string [type=string_type]",
        )
        (record,) = caplog.records
        assert record.levelno == logging.WARNING
        assert "invalid ensemble yaml" in record.message
        assert "path=/path/to/fan-out-test.yaml" in record.message


# ---------------------------------------------------------------------------
# Substrate registration + unknown-event tolerance
# ---------------------------------------------------------------------------


class TestSubstrateRegistration:
    def test_register_with_routes_substrate_emissions(
        self,
        sink_and_caplog: tuple[OperatorTerminalEventSink, pytest.LogCaptureFixture],
    ) -> None:
        sink, caplog = sink_and_caplog
        substrate = DispatchEventSubstrate()
        sink.register_with(substrate)
        substrate.emit(
            DispatchTiming(
                phase="start",
                dispatch_id="session-A-dispatch-0001",
                ensemble_name="code-generator",
                timestamp_seconds=1700000000.0,
            )
        )
        assert len(caplog.records) == 1
        assert "dispatch start" in caplog.records[0].message


class TestUnknownEventTolerance:
    def test_unknown_event_is_dropped_silently(
        self,
        sink_and_caplog: tuple[OperatorTerminalEventSink, pytest.LogCaptureFixture],
    ) -> None:
        """Sink ignores events it does not format — substrate fans out to
        every registered sink, so subset-handling is the contract.
        """
        sink, caplog = sink_and_caplog

        class _UnknownEvent:
            dispatch_id: str | None = "session-A-dispatch-0001"

        sink.consume(_UnknownEvent())
        assert caplog.records == []
