"""Integration: delegation-rate instrumentation across the real wiring.

The unit tests exercise the meter (``classify_turn`` / ``delegation_rate``),
the loop driver's ``turn_shape`` stamping, and the sink's line in isolation.
This test wires the three real components as production does — the real
:class:`LoopDriver` emitting through the real
:class:`DispatchEventSubstrate` to the real
:class:`OperatorTerminalEventSink` — and verifies the boundary the unit tests
cannot: that the delegation rate is computable from the emitted
``TurnDecision`` events alone, with the sink holding no access to request
content, logs, or replay (FC-59 "events alone").

Per ``docs/agentic-serving/system-design.agents.md`` §Module: Delegation Rate
Meter (``test_delegation_rate_from_events_alone``) and §Module: Loop Driver
(``test_turn_decision_carries_turn_shape``), and the build skill's Step 5.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from llm_orc.agentic.budget_controller import BudgetController
from llm_orc.agentic.dispatch_envelope import DispatchEnvelope
from llm_orc.agentic.dispatch_event_substrate import DispatchEventSubstrate
from llm_orc.agentic.loop_driver import LoopDriver
from llm_orc.agentic.operator_terminal_event_sink import OperatorTerminalEventSink
from llm_orc.agentic.orchestrator_tool_dispatch import (
    InternalToolCall,
    ToolCallResult,
    ToolCallSuccess,
)
from llm_orc.agentic.session_action_record import SessionActionRecord
from llm_orc.agentic.session_registry import SessionIdentity, SessionState
from llm_orc.agentic.session_start import ChatMessage, SessionContext
from llm_orc.agentic.single_step_enforcer import SingleStepEnforcer
from llm_orc.models.base import ToolCall, ToolCallingResponse


class _ScriptedSeatFiller:
    """Seat-filler double returning one scripted response per turn, in order."""

    def __init__(self, responses: list[ToolCallingResponse]) -> None:
        self._responses = list(responses)

    async def generate_with_tools(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ToolCallingResponse:
        return self._responses.pop(0)


class _UnusedJudgmentSeat:
    """Judgment-seat double — first-turn contexts here never reach a judgment."""

    async def generate_response(self, message: str, role_prompt: str) -> str:
        return "VERDICT: REMAINING\n"


class _RemainingJudgmentSeat:
    """Judgment-seat double returning REMAINING with a descriptive anchor.

    The anchor carries no generation verb — the exact shape that mis-stamped
    trailing delegated writes ``carry`` before WP-LB-M.
    """

    async def generate_response(self, message: str, role_prompt: str) -> str:
        return "VERDICT: REMAINING\nThe test file still needs writing."


class _EchoToolDispatch:
    """Tool-dispatch double — the callee path is not the boundary under test."""

    async def dispatch(
        self,
        call: InternalToolCall,
        *,
        session_id: str = "",
        model_profile_override: str | None = None,
    ) -> ToolCallResult:
        return ToolCallSuccess(
            id=call.id,
            name=call.name,
            content="generated",
            envelope=DispatchEnvelope(status="success", primary="generated"),
        )


def _delegating_response() -> ToolCallingResponse:
    return ToolCallingResponse(
        content="",
        tool_calls=[
            ToolCall(
                id="g1",
                name="invoke_ensemble",
                arguments_json=json.dumps(
                    {"name": "code-generator", "input": "build it", "filePath": "m.py"}
                ),
            )
        ],
        finish_reason="tool_calls",
    )


def _read_response() -> ToolCallingResponse:
    return ToolCallingResponse(
        content="",
        tool_calls=[
            ToolCall(
                id="r1",
                name="read",
                arguments_json=json.dumps({"filePath": "m.py"}),
            )
        ],
        finish_reason="tool_calls",
    )


def _context(message: str, session: str) -> SessionContext:
    return SessionContext(
        messages=[ChatMessage(role="user", content=message)],
        tools=[{"type": "function", "function": {"name": "write"}}],
        state=SessionState(
            identity=SessionIdentity(value=session, method="user_field")
        ),
    )


def _trailing_context(message: str, session: str) -> SessionContext:
    """A trailing tool-result tail — the shape that opens the termination
    judgment, where a REMAINING verdict drives another delegated write."""
    return SessionContext(
        messages=[
            ChatMessage(role="user", content=message),
            ChatMessage(role="assistant", content=None),
            ChatMessage(role="tool", content="Wrote file successfully"),
        ],
        tools=[{"type": "function", "function": {"name": "write"}}],
        state=SessionState(
            identity=SessionIdentity(value=session, method="user_field")
        ),
    )


@pytest.mark.asyncio
async def test_delegation_rate_computes_from_emitted_events_alone() -> None:
    substrate = DispatchEventSubstrate()
    sink = OperatorTerminalEventSink()
    sink.register_with(substrate)

    driver = LoopDriver(
        seat_filler=_ScriptedSeatFiller([_delegating_response(), _read_response()]),
        enforcer=SingleStepEnforcer(),
        tool_dispatch=_EchoToolDispatch(),
        capabilities=frozenset({"code-generator"}),
        event_substrate=substrate,
        action_record=SessionActionRecord(),
        judgment_seat=_UnusedJudgmentSeat(),
        budget=BudgetController(turn_limit=1_000, token_limit=1_000_000),
    )

    # A generation-shaped turn that delegates, then a read turn that carries —
    # two distinct sessions so each is a first-turn shape.
    await driver.decide(
        _context("Write a python module m.py with a helper function.", "rate-gen")
    )
    await driver.decide(_context("Read m.py and tell me what it does.", "rate-read"))

    reading = sink.delegation_rate_reading()

    assert reading.generation_turns == 1
    assert reading.delegated == 1
    assert reading.rate == 1.0
    assert reading.considered == 2


@pytest.mark.asyncio
async def test_delegation_rate_counts_trailing_multi_file_generation() -> None:
    """WP-LB-M — a trailing REMAINING turn that delegates a write is counted
    in the denominator, so the rate instruments multi-file sessions, not only
    first turns. The shape follows the action; the descriptive remaining-work
    anchor (no generation verb) does not stamp the turn ``carry``.
    """
    substrate = DispatchEventSubstrate()
    sink = OperatorTerminalEventSink()
    sink.register_with(substrate)

    driver = LoopDriver(
        seat_filler=_ScriptedSeatFiller(
            [_delegating_response(), _delegating_response()]
        ),
        enforcer=SingleStepEnforcer(),
        tool_dispatch=_EchoToolDispatch(),
        capabilities=frozenset({"code-generator"}),
        event_substrate=substrate,
        action_record=SessionActionRecord(),
        judgment_seat=_RemainingJudgmentSeat(),
        budget=BudgetController(turn_limit=1_000, token_limit=1_000_000),
    )

    # Turn 1 (first turn) delegates; turn 2 (trailing tool-result tail, judged
    # REMAINING) delegates again — both are generation turns, so the rate
    # reflects the whole multi-file session.
    await driver.decide(_context("Write module a.py and its tests.", "multi-file"))
    await driver.decide(
        _trailing_context("Write module a.py and its tests.", "multi-file")
    )

    reading = sink.delegation_rate_reading()

    assert reading.generation_turns == 2
    assert reading.delegated == 2
    assert reading.rate == 1.0
