"""Tests for the Loop Driver (Cycle 7 loop-back WP-LB-B, ADR-033).

The Loop Driver is the layer-A control structure for the tool-driven
multi-turn surface. Each ``run`` is one turn: it invokes the injected
seat-filler LLM to decide the next action, enforces single-action-per-turn,
delegates per-turn generation to a single capability ensemble (the
callee), and emits a ``TurnDecision`` diagnostic. Scenarios from
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


class _FakeCapabilitySelector:
    """Capability-selector double returning a fixed capability for any task.

    Records the tasks it was asked to match so tests can assert selection is
    a function of the turn's task content (AS-10), never a client signal.
    """

    def __init__(self, capability: str | None = "code-generator") -> None:
        self._capability = capability
        self.tasks: list[str] = []

    def select(self, *, task: str) -> str | None:
        self.tasks.append(task)
        return self._capability


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
    capability_selector: _FakeCapabilitySelector | None = None,
) -> LoopDriver:
    return LoopDriver(
        seat_filler=seat_filler,
        enforcer=SingleStepEnforcer(),
        tool_dispatch=tool_dispatch or _FakeToolDispatch(),
        capability_selector=capability_selector or _FakeCapabilitySelector(),
    )


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


def _write_call(call_id: str, arguments_json: str) -> ToolCall:
    return ToolCall(id=call_id, name="write", arguments_json=arguments_json)


class TestLoopDriverDelegatesToCallee:
    """FC-44 — per-turn generation routes to a single capability ensemble.

    Not the plan -> dispatch -> synthesize pipeline: the driver dispatches
    exactly one ``invoke_ensemble`` for the selected capability and emits the
    deliverable as a client tool call. There is no routing-planner or
    response-synthesizer collaborator on the driver to invoke (the structural
    callee property), so "zero planner / zero synthesizer" holds by absence.
    """

    async def test_delegates_generation_to_single_ensemble(self) -> None:
        seat_filler = _FakeSeatFiller(
            ToolCallingResponse(
                content="",
                tool_calls=[_write_call("t1", '{"filePath": "sort.py"}')],
                finish_reason="tool_calls",
            )
        )
        tool_dispatch = _FakeToolDispatch(deliverable="def sort(xs): ...")
        selector = _FakeCapabilitySelector(capability="code-generator")
        driver = _build_driver(
            seat_filler, tool_dispatch=tool_dispatch, capability_selector=selector
        )

        chunks = await _collect(driver.run(_make_context()))

        assert len(tool_dispatch.calls) == 1
        call = tool_dispatch.calls[0]
        assert call.name == "invoke_ensemble"
        assert call.arguments["name"] == "code-generator"

        tool_calls = [c for c in chunks if isinstance(c, ClientToolCall)]
        assert len(tool_calls) == 1
        invocation = tool_calls[0].tool_calls[0]
        assert invocation.name == "write"
        args = json.loads(invocation.arguments)
        assert args["filePath"] == "sort.py"
        assert args["content"] == "def sort(xs): ..."

    async def test_content_comes_from_the_ensemble_not_the_seat_filler(self) -> None:
        # Callee delegation: even if the seat-filler proposes its own content,
        # the client write carries the *ensemble's* deliverable. Generation is
        # delegated, not done by the driver's seat-filler.
        seat_filler = _FakeSeatFiller(
            ToolCallingResponse(
                content="",
                tool_calls=[
                    _write_call(
                        "t1",
                        '{"filePath": "sort.py", "content": "SEAT_FILLER_GUESS"}',
                    )
                ],
                finish_reason="tool_calls",
            )
        )
        tool_dispatch = _FakeToolDispatch(deliverable="def sort(xs): return xs")
        driver = _build_driver(seat_filler, tool_dispatch=tool_dispatch)

        chunks = await _collect(driver.run(_make_context()))

        invocation = next(
            c for c in chunks if isinstance(c, ClientToolCall)
        ).tool_calls[0]
        args = json.loads(invocation.arguments)
        assert args["content"] == "def sort(xs): return xs"
