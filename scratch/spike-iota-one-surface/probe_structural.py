"""Spike ι Arm A — structural floor for no-tools graceful finish.

Drives the real ``LoopDriver.decide`` over a ``tools=[]`` (no client tools)
SessionContext across capability states and seat-filler outcomes. Confirms the
loop has no tools-required precondition and finishes gracefully on the no-tools
path that loop-back #9 will route to it. $0, no model, deterministic.

Run: python scratch/spike-iota-one-surface/probe_structural.py
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from llm_orc.agentic.budget_controller import BudgetController
from llm_orc.agentic.dispatch_envelope import DispatchEnvelope
from llm_orc.agentic.loop_driver import (
    ApplyWork,
    FinishTurn,
    LoopDriver,
)
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


class _FakeSeatFiller:
    def __init__(self, response: ToolCallingResponse) -> None:
        self._response = response
        self.calls: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []

    async def generate_with_tools(
        self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ToolCallingResponse:
        self.calls.append((messages, tools))
        return self._response


class _FakeToolDispatch:
    def __init__(self, deliverable: str = "def sort(xs): ...") -> None:
        self._deliverable = deliverable
        self.calls: list[InternalToolCall] = []

    async def dispatch(
        self,
        call: InternalToolCall,
        *,
        session_id: str = "",
        model_profile_override: str | None = None,
    ) -> ToolCallResult:
        self.calls.append(call)
        return ToolCallSuccess(
            id=call.id,
            name=call.name,
            content=self._deliverable,
            envelope=DispatchEnvelope(status="success", primary=self._deliverable),
        )


class _FakeJudgmentSeat:
    def __init__(self, response: str) -> None:
        self._response = response
        self.calls: list[tuple[str, str]] = []

    async def generate_response(self, message: str, role_prompt: str) -> str:
        self.calls.append((message, role_prompt))
        return self._response


def _no_tools_context() -> SessionContext:
    """A plain question with NO client tools — the loop-back-#9 no-tools path."""
    return SessionContext(
        messages=[ChatMessage(role="user", content="what is 2 + 2?")],
        tools=[],
        state=SessionState(
            identity=SessionIdentity(value="iota-session", method="user_field")
        ),
    )


def _build_driver(
    seat_filler: _FakeSeatFiller,
    *,
    capabilities: frozenset[str] = frozenset(),
    tool_dispatch: _FakeToolDispatch | None = None,
) -> LoopDriver:
    return LoopDriver(
        seat_filler=seat_filler,
        enforcer=SingleStepEnforcer(),
        tool_dispatch=tool_dispatch or _FakeToolDispatch(),
        capabilities=capabilities,
        event_substrate=None,
        action_record=SessionActionRecord(),
        judgment_seat=_FakeJudgmentSeat("VERDICT: REMAINING\n"),
        budget=BudgetController(turn_limit=1_000, token_limit=1_000_000),
        artifact_store=None,
        escalation_ladder=(),
        generation_target=None,
    )


def _offered_tool_names(seat_filler: _FakeSeatFiller) -> list[str]:
    _messages, tools = seat_filler.calls[0]
    names: list[str] = []
    for tool in tools:
        function = tool.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            names.append(function["name"])
    return names


def _check(label: str, condition: bool, detail: str) -> bool:
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {label}: {detail}")
    return condition


async def main() -> int:
    results: list[bool] = []

    # Cell 1 — no capabilities, seat proposes text. Expect FinishTurn(text);
    # the seat-filler is offered an EMPTY tool list (no delegation, no client).
    seat = _FakeSeatFiller(
        ToolCallingResponse(content="2 + 2 = 4.", tool_calls=[], finish_reason="stop")
    )
    driver = _build_driver(seat)
    outcome = await driver.decide(_no_tools_context())
    print("Cell 1 — no capabilities, plain question, seat finishes with text:")
    results.append(
        _check(
            "returns FinishTurn with text",
            outcome == FinishTurn(content="2 + 2 = 4."),
            repr(outcome),
        )
    )
    results.append(
        _check(
            "seat offered an empty tool list",
            _offered_tool_names(seat) == [],
            repr(_offered_tool_names(seat)),
        )
    )

    # Cell 2 — no capabilities, empty seat text. Expect FinishTurn(None).
    seat = _FakeSeatFiller(
        ToolCallingResponse(content="", tool_calls=[], finish_reason="stop")
    )
    driver = _build_driver(seat)
    outcome = await driver.decide(_no_tools_context())
    print("Cell 2 — no capabilities, empty seat text:")
    results.append(
        _check(
            "returns FinishTurn(content=None)",
            outcome == FinishTurn(content=None),
            repr(outcome),
        )
    )

    # Cell 3 — capabilities present, seat proposes text (no action). Expect
    # FinishTurn(text); seat offered ONLY invoke_ensemble (no client tools).
    seat = _FakeSeatFiller(
        ToolCallingResponse(
            content="The answer is 4.", tool_calls=[], finish_reason="stop"
        )
    )
    driver = _build_driver(seat, capabilities=frozenset({"code-generator"}))
    outcome = await driver.decide(_no_tools_context())
    print("Cell 3 — capabilities present, plain question, seat finishes with text:")
    results.append(
        _check(
            "returns FinishTurn with text",
            outcome == FinishTurn(content="The answer is 4."),
            repr(outcome),
        )
    )
    results.append(
        _check(
            "seat offered only invoke_ensemble (no client tools)",
            _offered_tool_names(seat) == ["invoke_ensemble"],
            repr(_offered_tool_names(seat)),
        )
    )

    # Cell 4 — capabilities present, seat delegates. Expect ApplyWork — a
    # capability-matched no-tools question still routes to an ensemble.
    seat = _FakeSeatFiller(
        ToolCallingResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id="t1",
                    name="invoke_ensemble",
                    arguments_json=json.dumps(
                        {
                            "name": "code-generator",
                            "input": "write a sorting function",
                            "filePath": "sort.py",
                        }
                    ),
                )
            ],
            finish_reason="tool_calls",
        )
    )
    dispatch = _FakeToolDispatch()
    driver = _build_driver(
        seat, capabilities=frozenset({"code-generator"}), tool_dispatch=dispatch
    )
    outcome = await driver.decide(_no_tools_context())
    print("Cell 4 — capabilities present, capability-matched question, seat delegates:")
    results.append(
        _check(
            "returns ApplyWork (delegation works without client tools)",
            isinstance(outcome, ApplyWork),
            repr(outcome),
        )
    )
    results.append(
        _check(
            "dispatched exactly one ensemble",
            len(dispatch.calls) == 1 and dispatch.calls[0].name == "invoke_ensemble",
            f"{len(dispatch.calls)} call(s)",
        )
    )

    passed = sum(results)
    total = len(results)
    print(f"\nArm A structural floor: {passed}/{total} assertions passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
