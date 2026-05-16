"""Operator-Terminal Event Sink (Cycle 6 WP-B, ADR-023 Destination 1).

Per ``docs/agentic-serving/system-design.agents.md`` §Module:
Operator-Terminal Event Sink (L3 — new in Cycle 6). The sink is the
operator-terminal routing destination of the unified Dispatch Event
Substrate per Inversion N+2 — one substrate, two destinations
(WP-C ships the orchestrator-context destination).

Architectural drivers:

* ADR-023 §Destination 1 — operator-terminal destination's per-event
  format strings, log-level discrimination
  (``CalibrationSignal`` at DEBUG; all others at INFO), and the
  liveness-signal action surfaces (``emit_tool_call_log``,
  ``emit_heartbeat``).
* ADR-023 §"Noise-floor remediation (validate-once-at-load)" — the
  startup-validation-warning surface (one WARN per invalid YAML at
  startup; zero per-enumeration noise thereafter).

The sink owns the format strings; the Serving Layer owns the timing
(heartbeat scheduling, tool-call detection). The sink does not start
threads or schedule tasks — it formats and emits lines via Python's
``logging`` module so operators control verbosity through the standard
``logging`` configuration.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING

from llm_orc.agentic.calibration_gate import CalibrationVerdictEvent
from llm_orc.agentic.calibration_signal_channel import CalibrationSignal
from llm_orc.agentic.dispatch_event_substrate import DispatchTiming
from llm_orc.agentic.tier_router import TierSelection
from llm_orc.agentic.tier_router_audit import AuditDiagnostic

if TYPE_CHECKING:
    from llm_orc.agentic.dispatch_event_substrate import DispatchEventSubstrate
    from llm_orc.core.config.ensemble_config import EnsembleValidationResult

__all__ = [
    "OperatorTerminalEventSink",
]


_logger = logging.getLogger("llm_orc.agentic.operator_terminal")
"""Per-ADR-023 the operator-terminal destination is the serve console.

Python ``logging`` is the substrate so operators control verbosity via
standard logging configuration (``--verbose`` flag wiring, ``LOG_LEVEL``
env var, etc.). ``CalibrationSignal`` lines emit at DEBUG so they are
suppressed at default INFO verbosity per ADR-023 §Destination 1.
"""


class OperatorTerminalEventSink:
    """Registered :class:`EventSink` for the operator-terminal destination.

    Implements the :class:`~llm_orc.agentic.dispatch_event_substrate.EventSink`
    Protocol via :meth:`consume`. The Serving Layer (and other action-
    initiating callers) drive the liveness-signal surfaces directly
    through :meth:`emit_tool_call_log` and :meth:`emit_heartbeat` — the
    sink formats and logs, but does not schedule timers or detect
    tool-call structures itself.

    Startup-validation warnings flow through :meth:`emit_validation_warning`,
    called by the bootstrapping pipeline after the Ensemble Engine
    completes its validate-once-at-load library scan (WP-B Piece 3).
    Subsequent ``list_ensembles()`` calls return the cached validated
    subset without re-emitting warnings — the noise-floor remediation
    per ADR-023 §"Noise-floor remediation".
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
    ) -> None:
        self._logger = logger if logger is not None else _logger

    # ------------------------------------------------------------------
    # EventSink Protocol — substrate fan-out
    # ------------------------------------------------------------------

    def consume(self, event: object) -> None:
        """Format and log one event per ADR-023 §Destination 1.

        Unknown event types are ignored — the substrate fans out every
        emission to every registered sink; events the operator-terminal
        destination does not format are silently dropped here. The
        orchestrator-context sink (WP-C) handles a different subset.
        """
        if isinstance(event, DispatchTiming):
            self._log_dispatch_timing(event)
        elif isinstance(event, TierSelection):
            self._log_tier_selection(event)
        elif isinstance(event, CalibrationVerdictEvent):
            self._log_calibration_verdict(event)
        elif isinstance(event, AuditDiagnostic):
            self._log_audit_diagnostic(event)
        elif isinstance(event, CalibrationSignal):
            self._log_calibration_signal(event)

    # ------------------------------------------------------------------
    # Action surfaces — liveness signals + startup warnings
    # ------------------------------------------------------------------

    def emit_tool_call_log(self, *, tool_name: str, dispatch_id: str) -> None:
        """Log a tool-call-emit liveness anchor per ADR-023 §"Liveness signals".

        The Serving Layer calls this *before* dispatching when it detects
        an ``invoke_ensemble`` tool-call structure in the orchestrator's
        response stream. The line gives operators a "received tool call
        from cloud LLM at HH:MM:SS" anchor distinct from the post-dispatch
        ``DispatchTiming(start)`` event.

        FC-23 fitness — the ``test_tool_call_emit_log_precedes_dispatch_start``
        integration test asserts this line precedes the DispatchTiming
        start line for the same dispatch_id (WP-B Piece 4 wires the
        timing).
        """
        self._logger.info(
            "tool-call emit: tool=%s dispatch_id=%s", tool_name, dispatch_id
        )

    def emit_heartbeat(self, *, session_id: str, elapsed_seconds: float) -> None:
        """Log an inference-wait heartbeat per ADR-023 §"Liveness signals".

        The Serving Layer's inactivity timer calls this after
        ``heartbeat_interval_seconds`` (default 30s) of open-request
        inactivity (no tool-call emit, no DispatchTiming events). The
        line gives operators mid-stream signal during long cloud-LLM
        inference waits.
        """
        self._logger.info(
            "inference wait: elapsed=%.0f session_id=%s",
            elapsed_seconds,
            session_id,
        )

    def emit_validation_warning(self, *, yaml_path: str, error: str) -> None:
        """Log one WARN per invalid YAML at startup (ADR-023 noise-floor remediation).

        The Ensemble Engine's validate-once-at-load surface (WP-B Piece 3)
        produces these at library-load time. ``list_ensembles()`` returns
        the validated subset without re-emitting warnings, eliminating
        the per-enumeration noise pattern observed in the 2026-05-13
        Cycle 5 verification session (Cycle 6 DISCOVER finding 7).
        """
        self._logger.warning(
            "invalid ensemble yaml: path=%s error=%s", yaml_path, error
        )

    def report_validation_results(
        self, results: Iterable[EnsembleValidationResult]
    ) -> None:
        """Drain a loader's validation results into one WARN line per failure.

        Convenience wiring for the serve startup site. The serve hook
        calls :meth:`EnsembleLoader.prime` once at startup (or library
        reload) and then hands the resulting
        :meth:`~llm_orc.core.config.ensemble_config.EnsembleLoader.validation_results`
        tuple to this method. Each entry produces one WARN line via the
        existing :meth:`emit_validation_warning` action surface.
        """
        for result in results:
            self.emit_validation_warning(yaml_path=result.yaml_path, error=result.error)

    def register_with(self, substrate: DispatchEventSubstrate) -> None:
        """Register this sink with a Dispatch Event Substrate.

        Convenience method — equivalent to ``substrate.register_sink(sink)``.
        Production wiring calls this once at serve startup so all
        subsequent ``invoke_ensemble`` dispatches route through the sink.
        """
        substrate.register_sink(self)

    # ------------------------------------------------------------------
    # Format strings — one private method per event class (ADR-023 §Destination 1)
    # ------------------------------------------------------------------

    def _log_dispatch_timing(self, event: DispatchTiming) -> None:
        """``INFO: dispatch start: ...`` or ``INFO: dispatch end: ...``."""
        if event.phase == "start":
            self._logger.info(
                "dispatch start: ensemble=%s profile=%s dispatch_id=%s",
                event.ensemble_name,
                event.model_profile if event.model_profile is not None else "?",
                event.dispatch_id,
            )
            return
        # phase == "end"
        self._logger.info(
            "dispatch end: ensemble=%s duration=%s exit=%s dispatch_id=%s",
            event.ensemble_name,
            self._format_duration(event.duration_seconds),
            event.exit_status if event.exit_status is not None else "?",
            event.dispatch_id,
        )

    def _log_tier_selection(self, event: TierSelection) -> None:
        """One INFO line per ADR-023 §Destination 1 tier-selection format."""
        self._logger.info(
            "tier selection: profile=%s tier=%s topaz_skill=%s dispatch_id=%s",
            event.model_profile,
            event.tier,
            event.topaz_skill,
            event.dispatch_id if event.dispatch_id is not None else "?",
        )

    def _log_calibration_verdict(self, event: CalibrationVerdictEvent) -> None:
        """``INFO: calibration verdict: <verdict> dispatch_id=<id>``."""
        self._logger.info(
            "calibration verdict: %s ensemble=%s dispatch_id=%s",
            event.verdict,
            event.ensemble_name,
            event.dispatch_id if event.dispatch_id is not None else "?",
        )

    def _log_audit_diagnostic(self, event: AuditDiagnostic) -> None:
        """One line per criterion finding per ADR-023 §Destination 1.

        The audit diagnostic carries multiple ``CriterionFinding`` entries
        in its ``criteria_findings`` tuple. ADR-023 specifies one line
        per criterion finding so operators can see which criteria
        exceeded thresholds without parsing nested structure.
        """
        dispatch_id = event.dispatch_id if event.dispatch_id is not None else "?"
        for finding in event.criteria_findings:
            self._logger.info(
                (
                    "audit diagnostic: window_id=%d criterion=%s finding=%g "
                    "threshold=%g exceeds=%s dispatch_id=%s"
                ),
                event.window_id,
                finding.name,
                finding.value,
                finding.threshold,
                finding.exceeds,
                dispatch_id,
            )

    def _log_calibration_signal(self, event: CalibrationSignal) -> None:
        """``DEBUG: calibration signal: ...`` — suppressed at default verbosity.

        ADR-023 §Destination 1 specifies DEBUG level for calibration
        signals because the cross-layer channel (ADR-016) emits at high
        volume relative to other dispatch events. The line carries
        per-field detail so operators enabling DEBUG can inspect the
        signal's structural content.
        """
        self._logger.debug(
            (
                "calibration signal: ensemble=%s success=%s entropy=%s "
                "anchor=%s dispatch_id=%s"
            ),
            event.ensemble_name,
            event.dispatch_success,
            self._format_optional_float(event.recent_token_entropy),
            self._format_optional_bool(event.deterministic_anchor),
            event.dispatch_id if event.dispatch_id is not None else "?",
        )

    @staticmethod
    def _format_duration(seconds: float | None) -> str:
        """Format duration_seconds for the dispatch-end line.

        ``None`` becomes ``?`` — only happens during the transition window
        if a producer emits an end event without duration. Production
        emission via Orchestrator Tool Dispatch always supplies duration.
        """
        if seconds is None:
            return "?"
        return f"{seconds:.3f}"

    @staticmethod
    def _format_optional_float(value: float | None) -> str:
        if value is None:
            return "?"
        return f"{value:.4f}"

    @staticmethod
    def _format_optional_bool(value: bool | None) -> str:
        if value is None:
            return "?"
        return "true" if value else "false"
