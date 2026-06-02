"""Tests for the Loop Driver (Cycle 7 loop-back WP-LB-B, ADR-033).

The Loop Driver is the layer-A control structure for the tool-driven
multi-turn surface. Each ``run`` is one turn: it invokes the injected
seat-filler LLM to decide the next action, enforces single-action-per-turn,
and resolves that action one of two ways — a literal client tool call is
passed through verbatim (grounded carry), an ``invoke_ensemble`` call
delegates per-turn generation to a single capability ensemble (the callee)
and maps the deliverable to a client tool. Scenarios from
``docs/agentic-serving/scenarios.md`` §"Layer-A Loop-Driver and
Surface-Mode Discrimination (ADR-033)".
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from llm_orc.agentic.dispatch_envelope import DispatchEnvelope
from llm_orc.agentic.loop_driver import LoopDriver
from llm_orc.agentic.orchestrator_chunk import (
    ClientToolCall,
    Completion,
    ContentDelta,
    OrchestratorChunk,
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


async def _collect(
    chunks: AsyncIterator[OrchestratorChunk],
) -> list[OrchestratorChunk]:
    return [chunk async for chunk in chunks]


def _build_driver(
    seat_filler: _FakeSeatFiller,
    *,
    tool_dispatch: _FakeToolDispatch | None = None,
) -> LoopDriver:
    return LoopDriver(
        seat_filler=seat_filler,
        enforcer=SingleStepEnforcer(),
        tool_dispatch=tool_dispatch or _FakeToolDispatch(),
    )


def _one_tool_call(chunks: list[OrchestratorChunk]) -> Any:
    tool_calls = [c for c in chunks if isinstance(c, ClientToolCall)]
    assert len(tool_calls) == 1
    return tool_calls[0].tool_calls[0]


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

        chunks = await _collect(driver.run(_make_context()))

        assert chunks == [
            ContentDelta(content="2 + 2 = 4."),
            Completion(finish_reason="stop"),
        ]

    async def test_no_content_delta_when_finish_text_is_empty(self) -> None:
        seat_filler = _FakeSeatFiller(
            ToolCallingResponse(content="", tool_calls=[], finish_reason="stop")
        )
        driver = _build_driver(seat_filler)

        chunks = await _collect(driver.run(_make_context()))

        assert chunks == [Completion(finish_reason="stop")]


class TestLoopDriverDelegatesToCallee:
    """FC-44 — per-turn generation routes to a single capability ensemble.

    The seat-filler emits an ``invoke_ensemble`` call to delegate generation;
    the driver dispatches exactly one ensemble (no routing-planner or
    response-synthesizer collaborator exists on the driver to invoke — the
    structural callee property) and maps the deliverable to a client tool.
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

        chunks = await _collect(driver.run(_make_context()))

        assert len(tool_dispatch.calls) == 1
        call = tool_dispatch.calls[0]
        assert call.name == "invoke_ensemble"
        assert call.arguments["name"] == "code-generator"
        assert call.arguments["input"] == "write a sorting function"

        invocation = _one_tool_call(chunks)
        assert invocation.name == "write"
        args = json.loads(invocation.arguments)
        assert args["filePath"] == "sort.py"
        assert args["content"] == "def sort(xs): ..."


class TestLoopDriverGroundedCarry:
    """FC-45 — an action depending on a prior observed result uses that value.

    A literal client tool call carries the seat-filler's arguments verbatim:
    a value observed in a prior tool result reaches the client tool-call
    argument unchanged (no ``${...}`` template, no fabrication, no ensemble
    regeneration).
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

        chunks = await _collect(driver.run(self._grounded_context()))

        # No ensemble dispatch for a literal carry — the value is not
        # regenerated.
        assert tool_dispatch.calls == []
        invocation = _one_tool_call(chunks)
        assert invocation.name == "write"
        assert json.loads(invocation.arguments)["content"] == "TOKEN_7f3a9c"
        assert "${" not in invocation.arguments

    async def test_prior_tool_result_is_surfaced_to_the_seat_filler(self) -> None:
        seat_filler = _FakeSeatFiller(
            ToolCallingResponse(content="done", tool_calls=[], finish_reason="stop")
        )
        driver = _build_driver(seat_filler)

        await _collect(driver.run(self._grounded_context()))

        surfaced_messages = seat_filler.calls[0][0]
        assert any(
            message.get("content") == "TOKEN_7f3a9c" for message in surfaced_messages
        )
