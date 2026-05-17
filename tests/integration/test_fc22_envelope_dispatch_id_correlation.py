"""FC-22 envelope-leg anchor — diagnostics.dispatch_id correlates to ADR-023 events.

Per ``docs/agentic-serving/system-design.agents.md`` §Fitness Criteria
FC-22::

    `dispatch_id` consistency holds across three surfaces — the event
    stream's `dispatch_id`, the envelope's `diagnostics.dispatch_id`,
    and the artifact path's `<dispatch_id>` segment — for every
    substrate-routed dispatch.

The full three-surface check requires WP-E's Session Artifact Store
(the artifact-path leg). This Cycle 6 WP-D anchor verifies the
*envelope ↔ events* leg end-to-end against real component objects: the
real :class:`DispatchEventSubstrate`, the real
:class:`OrchestratorToolDispatch` with a scripted ensemble-operations
double, and the real :class:`ResultSummarizerHarness`. The envelope-leg
property is composable with the WP-E artifact-path leg — once WP-E
ships, the same dispatch_id flows through to the artifact filesystem
path's ``<dispatch_id>`` segment, closing the three-surface verification.

The anchor also verifies ADR-024's "envelope returns on every successful
dispatch" fitness criterion at integration scale: the real harness +
substrate composition produces an envelope on the dispatch's return
without further plumbing.
"""

from __future__ import annotations

from typing import Any

import pytest

from llm_orc.agentic.autonomy_policy import BASELINE_LEVEL, AutonomyPolicy
from llm_orc.agentic.composition_validator import (
    CompositionOutcome,
    CompositionRejected,
    CompositionRequest,
)
from llm_orc.agentic.dispatch_event_substrate import (
    DispatchEventSubstrate,
    DispatchTiming,
)
from llm_orc.agentic.orchestrator_tool_dispatch import (
    InternalToolCall,
    OrchestratorToolDispatch,
    ToolCallSuccess,
)
from llm_orc.agentic.result_summarizer_harness import ResultSummarizerHarness
from llm_orc.core.config.ensemble_config import EnsembleConfig


class _ScriptedOperations:
    """Programmable ``EnsembleOperations`` double for the boundary test.

    Returns a canned ``execution.json``-shaped result on ``invoke``; the
    real Tool Dispatch threads it through the real harness + substrate.
    """

    def __init__(self, *, invoke_result: dict[str, Any]) -> None:
        self._invoke_result = invoke_result

    async def invoke(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._invoke_result

    async def read_ensembles(self) -> list[dict[str, Any]]:
        return []


class _StubSummarizerInvoker:
    """Returns a canned summary so the harness produces SummarizationSuccess."""

    def __init__(self, *, summary: str) -> None:
        self._summary = summary

    async def invoke(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return {"synthesis": self._summary}


class _UnusedWriter:
    """LocalEnsembleWriter stub — compose path is not exercised here."""

    def write(self, config: EnsembleConfig) -> str:  # pragma: no cover
        raise AssertionError("local_ensemble_writer should not be called")


class _RejectingValidator:
    """CompositionValidator stub — compose path is not exercised here."""

    def validate(self, request: CompositionRequest) -> CompositionOutcome:
        return CompositionRejected(
            kind="missing_primitive",
            reason="unused in this anchor",
        )


@pytest.mark.asyncio
async def test_envelope_diagnostics_dispatch_id_matches_event_stream() -> None:
    """The envelope's diagnostics.dispatch_id == the event stream's dispatch_id.

    Real composition: real substrate, real Tool Dispatch, real harness.
    Asserts the cross-surface correlation property the WP-E artifact path
    will extend to three surfaces.
    """
    substrate = DispatchEventSubstrate()
    operations = _ScriptedOperations(
        invoke_result={
            "ensemble": "text-summarizer",
            "status": "completed",
            "synthesis": "the deliverable text",
            "metadata": {"tokens": 100},
        }
    )
    harness = ResultSummarizerHarness(
        invoker=_StubSummarizerInvoker(summary="a summary line"),
        summarizer_name="agentic-result-summarizer",
    )
    dispatch = OrchestratorToolDispatch(
        operations=operations,
        harness=harness,
        autonomy_policy=AutonomyPolicy(level_provider=lambda: BASELINE_LEVEL),
        composition_validator=_RejectingValidator(),
        local_ensemble_writer=_UnusedWriter(),
        event_substrate=substrate,
    )

    result = await dispatch.dispatch(
        InternalToolCall(
            id="call-1",
            name="invoke_ensemble",
            arguments={"name": "text-summarizer", "input": "source text"},
        ),
        session_id="session-FC22",
    )

    assert isinstance(result, ToolCallSuccess)
    assert result.envelope is not None
    envelope_dispatch_id = result.envelope.diagnostics["dispatch_id"]

    # FC-22 envelope-leg: every event emitted for this dispatch_id
    # carries the same identifier as envelope.diagnostics.dispatch_id.
    events = substrate.events_for(envelope_dispatch_id)
    assert len(events) >= 2  # at minimum start + end
    event_dispatch_ids = {getattr(event, "dispatch_id", None) for event in events}
    event_dispatch_ids.discard(None)
    assert event_dispatch_ids == {envelope_dispatch_id}, (
        "envelope.diagnostics.dispatch_id must equal the substrate event "
        "stream's dispatch_id for this dispatch — FC-22 envelope leg."
    )


@pytest.mark.asyncio
async def test_envelope_diagnostics_carry_duration_from_dispatch_timing_end() -> None:
    """envelope.diagnostics.duration_seconds projects from DispatchTiming(end).

    The envelope is constructed AFTER DispatchTiming(end) is emitted
    (per WP-D's ordering refactor of invoke_ensemble's finally-block
    pattern). The integration verifies the end event's
    duration_seconds shows up in the envelope's diagnostics.
    """
    substrate = DispatchEventSubstrate()
    operations = _ScriptedOperations(
        invoke_result={"synthesis": "deliverable", "metadata": {}}
    )
    harness = ResultSummarizerHarness(
        invoker=_StubSummarizerInvoker(summary="summary"),
        summarizer_name="agentic-result-summarizer",
    )
    dispatch = OrchestratorToolDispatch(
        operations=operations,
        harness=harness,
        autonomy_policy=AutonomyPolicy(level_provider=lambda: BASELINE_LEVEL),
        composition_validator=_RejectingValidator(),
        local_ensemble_writer=_UnusedWriter(),
        event_substrate=substrate,
    )

    result = await dispatch.dispatch(
        InternalToolCall(
            id="call-1",
            name="invoke_ensemble",
            arguments={"name": "text-summarizer", "input": "text"},
        ),
        session_id="session-FC22",
    )

    assert isinstance(result, ToolCallSuccess)
    assert result.envelope is not None
    diagnostics = result.envelope.diagnostics
    assert "duration_seconds" in diagnostics
    assert isinstance(diagnostics["duration_seconds"], float)
    assert diagnostics["duration_seconds"] >= 0.0

    # The same value lives on DispatchTiming(end) per WP-A; both
    # surfaces project from the same source.
    events = substrate.events_for(diagnostics["dispatch_id"])
    end_events = [
        event
        for event in events
        if isinstance(event, DispatchTiming) and event.phase == "end"
    ]
    assert len(end_events) == 1
    assert end_events[0].duration_seconds == diagnostics["duration_seconds"]
