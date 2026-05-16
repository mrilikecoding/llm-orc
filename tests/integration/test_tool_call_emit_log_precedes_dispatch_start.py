"""FC-23 anchor — tool-call-emit log precedes DispatchTiming(start).

Per ``docs/agentic-serving/system-design.agents.md`` §Operator-Terminal
Event Sink fitness::

    The tool-call-emit log line for an ``invoke_ensemble`` tool call
    fires **before** the dispatch — chronologically,
    ``INFO: tool-call emit: tool=invoke_ensemble dispatch_id=<id>``
    appears in the log stream before the corresponding
    ``INFO: dispatch start: ensemble=<name> ... dispatch_id=<id>`` —
    verified by ``test_tool_call_emit_log_precedes_dispatch_start``
    (integration, timestamp-ordered assertion).

This integration test composes the production Tool Dispatch with the
real :class:`OperatorTerminalEventSink` and a real
:class:`DispatchEventSubstrate`. A scripted operations double stands
in for the L0 ensemble executor (no real LLM) so the timing-order
assertion stays deterministic.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from llm_orc.agentic.autonomy_policy import BASELINE_LEVEL, AutonomyPolicy
from llm_orc.agentic.composition_validator import (
    CompositionOutcome,
    CompositionRejected,
    CompositionRequest,
)
from llm_orc.agentic.dispatch_event_substrate import DispatchEventSubstrate
from llm_orc.agentic.operator_terminal_event_sink import OperatorTerminalEventSink
from llm_orc.agentic.orchestrator_tool_dispatch import (
    InternalToolCall,
    OrchestratorToolDispatch,
    ToolCallSuccess,
)
from llm_orc.agentic.result_summarizer_harness import ResultSummarizerHarness
from llm_orc.core.config.ensemble_config import EnsembleConfig


class _ScriptedOperations:
    """Minimum :class:`EnsembleOperations` double — returns the scripted dict."""

    def __init__(self, result: dict[str, Any]) -> None:
        self._result = result

    async def invoke(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._result

    async def read_ensembles(self) -> list[dict[str, Any]]:  # pragma: no cover
        return []


class _UnusedValidator:
    """Validator slot — invoke_ensemble never consults it; rejects defensively."""

    def validate(
        self, request: CompositionRequest
    ) -> CompositionOutcome:  # pragma: no cover
        return CompositionRejected(
            kind="missing_primitive", reason="unused-in-this-test"
        )


class _UnusedWriter:
    """Writer slot — invoke_ensemble path never writes."""

    def write(self, config: EnsembleConfig) -> str:  # pragma: no cover
        raise AssertionError("write should not be called for invoke_ensemble")


def _build_dispatch(
    *,
    operations: _ScriptedOperations,
    substrate: DispatchEventSubstrate,
    sink: OperatorTerminalEventSink,
) -> OrchestratorToolDispatch:
    return OrchestratorToolDispatch(
        operations=operations,
        harness=ResultSummarizerHarness(
            invoker=operations, summarizer_name="not-used-raw-output"
        ),
        autonomy_policy=AutonomyPolicy(level_provider=lambda: BASELINE_LEVEL),
        composition_validator=_UnusedValidator(),
        local_ensemble_writer=_UnusedWriter(),
        event_substrate=substrate,
        tool_call_emit_logger=sink,
    )


@pytest.fixture
def _llm_orc_logger_propagation() -> Any:
    """Ensure the ``llm_orc`` logger propagates while this test runs.

    The ``llm-orc serve`` / ``web`` CLI commands disable propagation on
    the ``llm_orc`` parent logger so server output is insulated from
    uvicorn's root handler chain. When prior tests exercise those
    commands the flag stays mutated, which breaks pytest's caplog
    fixture (caplog hooks the root logger via propagation). Restoring
    propagate=True for this test is a self-contained workaround.
    """
    orc_logger = logging.getLogger("llm_orc")
    previous = orc_logger.propagate
    orc_logger.propagate = True
    try:
        yield
    finally:
        orc_logger.propagate = previous


@pytest.mark.usefixtures("_llm_orc_logger_propagation")
@pytest.mark.asyncio
async def test_tool_call_emit_log_precedes_dispatch_start(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """FC-23 — the operator-terminal log stream records ``tool-call emit``
    before ``dispatch start`` for the same dispatch_id."""
    substrate = DispatchEventSubstrate()
    sink = OperatorTerminalEventSink()
    sink.register_with(substrate)
    dispatch = _build_dispatch(
        operations=_ScriptedOperations({"synthesis": "done", "raw_output": True}),
        substrate=substrate,
        sink=sink,
    )

    with caplog.at_level(logging.INFO, logger="llm_orc.agentic.operator_terminal"):
        result = await dispatch.dispatch(
            InternalToolCall(
                id="call-1",
                name="invoke_ensemble",
                arguments={"name": "code-generator", "input": "task"},
            ),
            session_id="session-A",
        )

    assert isinstance(result, ToolCallSuccess)

    operator_records = [
        r for r in caplog.records if r.name == "llm_orc.agentic.operator_terminal"
    ]
    emit_indices = [
        i for i, r in enumerate(operator_records) if "tool-call emit" in r.message
    ]
    start_indices = [
        i for i, r in enumerate(operator_records) if "dispatch start" in r.message
    ]
    assert len(emit_indices) == 1, operator_records
    assert len(start_indices) == 1, operator_records
    assert emit_indices[0] < start_indices[0]

    # Same dispatch_id on both lines — operators can join them post-hoc.
    emit_record = operator_records[emit_indices[0]]
    start_record = operator_records[start_indices[0]]
    assert "dispatch_id=session-A-dispatch-0001" in emit_record.message
    assert "dispatch_id=session-A-dispatch-0001" in start_record.message
