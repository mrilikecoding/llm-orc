"""Inference-Wait Heartbeat scheduler (Cycle 6 WP-B piece 5).

Per ``docs/agentic-serving/system-design.agents.md`` §Operator-Terminal
Event Sink — heartbeats fire from the open-request side and the sink
formats the ``INFO: inference wait:`` line. The Serving Layer owns one
scheduler per open chat-completions request (C6-2 default — async
background task tied to the open-request lifetime, auto-cancelled on
request close).

The scheduler observes two signal paths to detect activity:

* Substrate :class:`DispatchTiming` events for the request's session
  (start or end of an ``invoke_ensemble`` dispatch). Registered as an
  :class:`~llm_orc.agentic.dispatch_event_substrate.EventSink` per
  request; removed at request close.
* Direct tool-call-emit calls — the scheduler implements the
  ``ToolCallEmitLogger`` Protocol that
  :class:`~llm_orc.agentic.orchestrator_tool_dispatch.OrchestratorToolDispatch`
  consults at ``invoke_ensemble`` entry. Forwarding the call to the
  underlying :class:`OperatorTerminalEventSink` preserves the FC-23
  ordering property; resetting the local activity timer keeps the
  heartbeat quiet while the dispatch is in flight.

Only the request's own session is treated as activity — events with
``dispatch_id`` values that do not start with ``f"{session_id}-dispatch-"``
do not reset the timer. This isolates per-request heartbeat timing
from cross-session traffic so operators see one heartbeat stream per
session.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable

from llm_orc.agentic.dispatch_event_substrate import (
    DispatchEventSubstrate,
    DispatchTiming,
)
from llm_orc.agentic.operator_terminal_event_sink import OperatorTerminalEventSink

__all__ = [
    "InferenceWaitHeartbeatScheduler",
]


_logger = logging.getLogger(__name__)


class InferenceWaitHeartbeatScheduler:
    """Per-request inference-wait heartbeat (Cycle 6 WP-B piece 5).

    Implements both the substrate ``EventSink`` protocol and the
    ``ToolCallEmitLogger`` protocol so the Serving Layer can register
    the scheduler with the dispatch event substrate at request open
    *and* substitute it for the bare operator-terminal sink as Tool
    Dispatch's tool-call-emit logger. Both paths reset the activity
    timer when the signal carries the scheduler's session_id.
    """

    def __init__(
        self,
        *,
        sink: OperatorTerminalEventSink,
        session_id: str,
        interval_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._sink = sink
        self._session_id = session_id
        self._dispatch_prefix = f"{session_id}-dispatch-"
        self._interval = interval_seconds
        self._clock = clock
        self._last_activity = clock()
        self._stopped = False

    # ------------------------------------------------------------------
    # EventSink Protocol — substrate fan-out
    # ------------------------------------------------------------------

    def consume(self, event: object) -> None:
        """Reset activity when a session-matching DispatchTiming arrives.

        The scenario language is *"30 seconds elapse without a
        tool-call-emit event or dispatch start/end event"*. Only
        :class:`DispatchTiming` events count as activity here; other
        substrate event types (CalibrationVerdictEvent, TierSelection,
        AuditDiagnostic, CalibrationSignal) ride on top of an active
        dispatch and are already bracketed by DispatchTiming's
        start/end pair.
        """
        if not isinstance(event, DispatchTiming):
            return
        if not event.dispatch_id.startswith(self._dispatch_prefix):
            return
        self._note_activity()

    # ------------------------------------------------------------------
    # ToolCallEmitLogger Protocol — forwards and resets activity
    # ------------------------------------------------------------------

    def emit_tool_call_log(self, *, tool_name: str, dispatch_id: str) -> None:
        """Forward to the underlying sink; reset activity on session match.

        Tool Dispatch calls this from inside
        :meth:`OrchestratorToolDispatch._open_dispatch_event` per the
        FC-23 ordering contract. Forwarding preserves the operator-
        terminal log line; the session-match check guards against
        cross-session emits incorrectly resetting this scheduler.
        """
        self._sink.emit_tool_call_log(tool_name=tool_name, dispatch_id=dispatch_id)
        if dispatch_id.startswith(self._dispatch_prefix):
            self._note_activity()

    # ------------------------------------------------------------------
    # Substrate lifecycle
    # ------------------------------------------------------------------

    def register_with(self, substrate: DispatchEventSubstrate) -> None:
        """Register as a substrate sink (per-request open)."""
        substrate.register_sink(self)

    def unregister_with(self, substrate: DispatchEventSubstrate) -> None:
        """Remove from the substrate (per-request close); idempotent."""
        substrate.unregister_sink(self)

    # ------------------------------------------------------------------
    # Tick logic — sync surface tested directly
    # ------------------------------------------------------------------

    def check_and_emit_if_inactive(self) -> bool:
        """Emit a heartbeat if inactivity ≥ interval. Returns ``True`` if fired.

        The async :meth:`run` loop wraps this; tests call it directly to
        assert the activity-elapsed branching without spinning real
        asyncio time.
        """
        now = self._clock()
        elapsed = now - self._last_activity
        if elapsed < self._interval:
            return False
        self._sink.emit_heartbeat(session_id=self._session_id, elapsed_seconds=elapsed)
        self._last_activity = now
        return True

    async def run(self) -> None:
        """Background loop — fires heartbeats while the request stays idle.

        The Serving Layer schedules :meth:`run` as an asyncio task at
        request open and cancels it at request close. ``CancelledError``
        terminates the loop cleanly; the scheduler emits no further
        heartbeats after cancellation.
        """
        try:
            while not self._stopped:
                now = self._clock()
                elapsed = now - self._last_activity
                if elapsed >= self._interval:
                    self._sink.emit_heartbeat(
                        session_id=self._session_id, elapsed_seconds=elapsed
                    )
                    self._last_activity = now
                    wait = self._interval
                else:
                    wait = self._interval - elapsed
                await asyncio.sleep(wait)
        except asyncio.CancelledError:
            self._stopped = True
            raise

    def stop(self) -> None:
        """Mark the scheduler stopped so ``run`` exits on its next iteration."""
        self._stopped = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _note_activity(self) -> None:
        self._last_activity = self._clock()
