"""Tests for the Loop Driver (Cycle 7 loop-back WP-LB-B, ADR-033).

The Loop Driver is the layer-A control structure for the tool-driven
multi-turn surface. Each ``decide`` is one turn: it invokes the injected
seat-filler LLM to decide the next action, enforces single-action-per-turn,
and returns the per-turn :class:`TurnOutcome` — a literal client tool call is
carried through verbatim (grounded carry, :class:`CarryClientTool`), an
``invoke_ensemble`` call delegates per-turn generation to a single capability
ensemble (the callee) and returns the deliverable envelope for the Terminal
to marshal (:class:`ApplyWork`), and a no-action turn finishes with text
(:class:`FinishTurn`). Scenarios from ``docs/agentic-serving/scenarios.md``
§"Layer-A Loop-Driver and Surface-Mode Discrimination (ADR-033)". Tool-call
*emission* and deliverable-content marshalling are the Client-Tool-Action
Terminal's job (ADR-034); see ``test_client_tool_action_terminal.py``.
"""

from __future__ import annotations

import json
from typing import Any

from llm_orc.agentic.dispatch_envelope import DispatchEnvelope
from llm_orc.agentic.dispatch_event_substrate import DispatchEventSubstrate
from llm_orc.agentic.loop_driver import (
    ApplyWork,
    CarryClientTool,
    FinishTurn,
    LoopDriver,
    TurnDecision,
)
from llm_orc.agentic.orchestrator_tool_dispatch import (
    InternalToolCall,
    ToolCallResult,
    ToolCallSuccess,
)
from llm_orc.agentic.session_registry import SessionIdentity, SessionState
from llm_orc.agentic.session_start import ChatMessage, SessionContext
from llm_orc.agentic.single_step_enforcer import SingleStepEnforcer
from llm_orc.models.base import ToolCall, ToolCallingResponse


class _FakeSeatFiller:
    """Seat-filler double returning a pre-scripted tool-calling response.

    Records the messages and tools it was handed so tests can assert the
    Loop Driver surfaced the conversation (including prior observed tool
    results) to the model.
    """

    def __init__(self, response: ToolCallingResponse) -> None:
        self._response = response
        self.calls: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []

    async def generate_with_tools(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ToolCallingResponse:
        self.calls.append((messages, tools))
        return self._response


class _FakeToolDispatch:
    """Tool-dispatch double recording the calls it dispatched.

    Returns a successful ensemble result whose ``envelope.primary`` is the
    pre-scripted deliverable, mimicking the per-turn callee ensemble.
    """

    def __init__(self, deliverable: str = "generated content") -> None:
        self._deliverable = deliverable
        self.calls: list[InternalToolCall] = []

    async def dispatch(
        self, call: InternalToolCall, *, session_id: str = ""
    ) -> ToolCallResult:
        self.calls.append(call)
        return ToolCallSuccess(
            id=call.id,
            name=call.name,
            content=self._deliverable,
            envelope=DispatchEnvelope(status="success", primary=self._deliverable),
        )


def _make_context(
    messages: list[ChatMessage] | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> SessionContext:
    return SessionContext(
        messages=messages or [ChatMessage(role="user", content="what is 2 + 2?")],
        tools=tools if tools is not None else [{"type": "function"}],
        state=SessionState(
            identity=SessionIdentity(value="test-session", method="user_field")
        ),
    )


def _build_driver(
    seat_filler: _FakeSeatFiller,
    *,
    tool_dispatch: _FakeToolDispatch | None = None,
    event_substrate: DispatchEventSubstrate | None = None,
) -> LoopDriver:
    return LoopDriver(
        seat_filler=seat_filler,
        enforcer=SingleStepEnforcer(),
        tool_dispatch=tool_dispatch or _FakeToolDispatch(),
        event_substrate=event_substrate,
    )


class _CapturingSink:
    """Event sink recording every event the substrate fans out."""

    def __init__(self) -> None:
        self.events: list[object] = []

    def consume(self, event: object) -> None:
        self.events.append(event)


def _turn_decisions(sink: _CapturingSink) -> list[TurnDecision]:
    return [event for event in sink.events if isinstance(event, TurnDecision)]


class TestLoopDriverFinishesWithText:
    """Loop-driver finishes with a text completion when no further action.

    Per ADR-033 §Decision 1 the finish-with-text path is the safe terminal
    that makes engaging the driver on tools-presence safe: a tool-capable
    client asking a plain question is served correctly.
    """

    async def test_finishes_with_text_when_seat_filler_proposes_no_action(
        self,
    ) -> None:
        seat_filler = _FakeSeatFiller(
            ToolCallingResponse(
                content="2 + 2 = 4.", tool_calls=[], finish_reason="stop"
            )
        )
        driver = _build_driver(seat_filler)

        outcome = await driver.decide(_make_context())

        assert outcome == FinishTurn(content="2 + 2 = 4.")

    async def test_no_content_when_finish_text_is_empty(self) -> None:
        seat_filler = _FakeSeatFiller(
            ToolCallingResponse(content="", tool_calls=[], finish_reason="stop")
        )
        driver = _build_driver(seat_filler)

        outcome = await driver.decide(_make_context())

        assert outcome == FinishTurn(content=None)


class TestLoopDriverDelegatesToCallee:
    """FC-44 — per-turn generation routes to a single capability ensemble.

    The seat-filler emits an ``invoke_ensemble`` call to delegate generation;
    the driver dispatches exactly one ensemble (no routing-planner or
    response-synthesizer collaborator exists on the driver to invoke — the
    structural callee property) and returns the deliverable envelope plus the
    tool-mapping decision for the Terminal to emit.
    """

    async def test_delegates_generation_to_single_ensemble(self) -> None:
        seat_filler = _FakeSeatFiller(
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
        tool_dispatch = _FakeToolDispatch(deliverable="def sort(xs): ...")
        driver = _build_driver(seat_filler, tool_dispatch=tool_dispatch)

        outcome = await driver.decide(_make_context())

        assert len(tool_dispatch.calls) == 1
        call = tool_dispatch.calls[0]
        assert call.name == "invoke_ensemble"
        assert call.arguments["name"] == "code-generator"
        assert call.arguments["input"] == "write a sorting function"

        assert isinstance(outcome, ApplyWork)
        assert outcome.tool_name == "write"
        assert outcome.file_path == "sort.py"
        assert outcome.delegated_ensemble == "code-generator"
        # The envelope, not baked content, travels to the Terminal (which
        # marshals it); WP-LB-B's deliverable is the inline primary.
        assert outcome.envelope.primary == "def sort(xs): ..."


class TestLoopDriverGroundedCarry:
    """FC-45 — an action depending on a prior observed result uses that value.

    A literal client tool call carries the seat-filler's arguments verbatim:
    a value observed in a prior tool result reaches the client tool-call
    argument unchanged (no ``${...}`` template, no fabrication, no ensemble
    regeneration). The driver owns this verbatim guarantee; the Terminal emits
    the carried invocation as-is.
    """

    @staticmethod
    def _grounded_context() -> SessionContext:
        return _make_context(
            messages=[
                ChatMessage(
                    role="user",
                    content="run gen-token.sh and save the token to token.txt",
                ),
                ChatMessage(
                    role="assistant",
                    content=None,
                    tool_calls=(
                        {
                            "id": "b1",
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": '{"command": "./gen-token.sh"}',
                            },
                        },
                    ),
                ),
                ChatMessage(role="tool", content="TOKEN_7f3a9c", tool_call_id="b1"),
            ]
        )

    async def test_observed_value_carried_into_tool_call_verbatim(self) -> None:
        seat_filler = _FakeSeatFiller(
            ToolCallingResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="w1",
                        name="write",
                        arguments_json=json.dumps(
                            {"filePath": "token.txt", "content": "TOKEN_7f3a9c"}
                        ),
                    )
                ],
                finish_reason="tool_calls",
            )
        )
        tool_dispatch = _FakeToolDispatch()
        driver = _build_driver(seat_filler, tool_dispatch=tool_dispatch)

        outcome = await driver.decide(self._grounded_context())

        # No ensemble dispatch for a literal carry — the value is not
        # regenerated.
        assert tool_dispatch.calls == []
        assert isinstance(outcome, CarryClientTool)
        assert outcome.invocation.name == "write"
        assert json.loads(outcome.invocation.arguments)["content"] == "TOKEN_7f3a9c"
        assert "${" not in outcome.invocation.arguments

    async def test_prior_tool_result_is_surfaced_to_the_seat_filler(self) -> None:
        seat_filler = _FakeSeatFiller(
            ToolCallingResponse(content="done", tool_calls=[], finish_reason="stop")
        )
        driver = _build_driver(seat_filler)

        await driver.decide(self._grounded_context())

        surfaced_messages = seat_filler.calls[0][0]
        assert any(
            message.get("content") == "TOKEN_7f3a9c" for message in surfaced_messages
        )


class TestLoopDriverTurnDecision:
    """FC-51 — each turn emits a TurnDecision diagnostic.

    The event carries the action, the delegated ensemble (if any), whether a
    grounded carry was held, and whether the enforcer truncated a batch — so a
    failing long-horizon (axis-2) run reconstructs as split-incorrect (wrong
    action) vs callee-incorrect (wrong generated content).
    """

    @staticmethod
    def _capture(
        seat_filler: _FakeSeatFiller,
    ) -> tuple[DispatchEventSubstrate, _CapturingSink, LoopDriver]:
        substrate = DispatchEventSubstrate()
        sink = _CapturingSink()
        substrate.register_sink(sink)
        driver = _build_driver(seat_filler, event_substrate=substrate)
        return substrate, sink, driver

    async def test_finish_turn_emits_a_turn_decision(self) -> None:
        seat_filler = _FakeSeatFiller(
            ToolCallingResponse(content="done", tool_calls=[], finish_reason="stop")
        )
        _, sink, driver = self._capture(seat_filler)

        await driver.decide(_make_context())

        decisions = _turn_decisions(sink)
        assert len(decisions) == 1
        assert decisions[0].action == "finish"
        assert decisions[0].delegated_ensemble is None

    async def test_generation_turn_records_the_delegated_ensemble(self) -> None:
        seat_filler = _FakeSeatFiller(
            ToolCallingResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="t1",
                        name="invoke_ensemble",
                        arguments_json=json.dumps(
                            {
                                "name": "code-generator",
                                "input": "write a function",
                                "filePath": "f.py",
                            }
                        ),
                    )
                ],
                finish_reason="tool_calls",
            )
        )
        _, sink, driver = self._capture(seat_filler)

        await driver.decide(_make_context())

        decision = _turn_decisions(sink)[0]
        assert decision.action == "write"
        assert decision.delegated_ensemble == "code-generator"
        assert decision.grounded_carry_held is False

    async def test_literal_carry_batch_flags_grounded_and_truncation(self) -> None:
        seat_filler = _FakeSeatFiller(
            ToolCallingResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="w1",
                        name="write",
                        arguments_json=json.dumps(
                            {"filePath": "a.txt", "content": "A"}
                        ),
                    ),
                    ToolCall(
                        id="w2",
                        name="write",
                        arguments_json=json.dumps(
                            {"filePath": "b.txt", "content": "B"}
                        ),
                    ),
                ],
                finish_reason="tool_calls",
            )
        )
        _, sink, driver = self._capture(seat_filler)

        await driver.decide(_make_context())

        decision = _turn_decisions(sink)[0]
        assert decision.action == "write"
        assert decision.grounded_carry_held is True
        assert decision.delegated_ensemble is None
        assert decision.replanned_after_truncation is True
