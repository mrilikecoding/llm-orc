"""Tests for the Client-Tool-Action Terminal (Cycle 7 loop-back WP-LB-C, ADR-034).

The Terminal owns tool-call *emission*: it asks the Loop Driver to decide one
turn and maps the resulting :class:`TurnOutcome` to the shared
``OrchestratorChunk`` vocabulary. These tests drive the mapping in isolation
against a scripted decider (the N emission cases), then verify the composition
with the real Loop Driver (the +1 wiring test — the content the Terminal
marshals into a ``write`` comes from the driver's delegated ensemble).
Scenarios from ``docs/agentic-serving/scenarios.md`` §"Client-Tool-Action
Terminal and Artifact-Bridge (ADR-034)".
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from llm_orc.agentic.client_tool_action_terminal import ClientToolActionTerminal
from llm_orc.agentic.dispatch_envelope import DispatchEnvelope
from llm_orc.agentic.loop_driver import (
    ApplyWork,
    CarryClientTool,
    FinishTurn,
    LoopDriver,
    TurnOutcome,
)
from llm_orc.agentic.orchestrator_chunk import (
    ClientToolCall,
    Completion,
    ContentDelta,
    OrchestratorChunk,
    ToolCallInvocation,
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


class _ScriptedDecider:
    """A decider double returning one scripted ``TurnOutcome`` per turn.

    Records the context it was handed so loop-participation tests can assert
    the Terminal passed the full conversation (not a filtered request) to the
    driver.
    """

    def __init__(self, outcome: TurnOutcome) -> None:
        self._outcome = outcome
        self.contexts: list[SessionContext] = []

    async def decide(self, context: SessionContext) -> TurnOutcome:
        self.contexts.append(context)
        return self._outcome


def _make_context(messages: list[ChatMessage] | None = None) -> SessionContext:
    return SessionContext(
        messages=messages or [ChatMessage(role="user", content="write the config")],
        tools=[{"type": "function", "function": {"name": "write"}}],
        state=SessionState(
            identity=SessionIdentity(value="terminal-test", method="user_field")
        ),
    )


async def _collect(
    chunks: AsyncIterator[OrchestratorChunk],
) -> list[OrchestratorChunk]:
    return [chunk async for chunk in chunks]


class TestFinishOutcomeEmission:
    """A finish outcome emits assistant text (if any) then a stop completion."""

    async def test_finish_with_text_emits_content_delta_then_completion(self) -> None:
        terminal = ClientToolActionTerminal(
            loop_driver=_ScriptedDecider(FinishTurn(content="2 + 2 = 4."))
        )

        chunks = await _collect(terminal.run(_make_context()))

        assert chunks == [
            ContentDelta(content="2 + 2 = 4."),
            Completion(finish_reason="stop"),
        ]

    async def test_finish_without_text_emits_only_completion(self) -> None:
        terminal = ClientToolActionTerminal(
            loop_driver=_ScriptedDecider(FinishTurn(content=None))
        )

        chunks = await _collect(terminal.run(_make_context()))

        assert chunks == [Completion(finish_reason="stop")]


class TestCarryClientToolEmission:
    """A grounded-carry outcome emits the carried invocation verbatim (FC-45)."""

    async def test_carried_invocation_is_emitted_unchanged(self) -> None:
        invocation = ToolCallInvocation(
            id="w1",
            name="write",
            arguments=json.dumps({"filePath": "token.txt", "content": "TOKEN_7f3a9c"}),
        )
        terminal = ClientToolActionTerminal(
            loop_driver=_ScriptedDecider(CarryClientTool(invocation=invocation))
        )

        chunks = await _collect(terminal.run(_make_context()))

        assert chunks == [ClientToolCall(tool_calls=(invocation,))]


class TestApplyWorkEmission:
    """A generation outcome emits a tool call carrying the deliverable (FC-47)."""

    async def test_apply_work_emits_write_tool_call_with_deliverable(self) -> None:
        outcome = ApplyWork(
            invocation_id="t1",
            tool_name="write",
            file_path="sort.py",
            envelope=DispatchEnvelope(status="success", primary="def sort(xs): ..."),
            delegated_ensemble="code-generator",
        )
        terminal = ClientToolActionTerminal(loop_driver=_ScriptedDecider(outcome))

        chunks = await _collect(terminal.run(_make_context()))

        assert len(chunks) == 1
        tool_call = chunks[0]
        assert isinstance(tool_call, ClientToolCall)
        invocation = tool_call.tool_calls[0]
        assert invocation.name == "write"
        args = json.loads(invocation.arguments)
        assert args["filePath"] == "sort.py"
        assert args["content"] == "def sort(xs): ..."


class TestLoopParticipation:
    """FC-50 — the Terminal hands the full conversation to the driver.

    Loop participation is structural: the Terminal does not filter the request
    to the last user message (the single-turn pipeline's ``_extract_request``
    behavior, which drops ``role: "tool"`` results). The trailing tool result
    reaches the driver's per-turn decision.
    """

    async def test_trailing_tool_result_reaches_the_driver(self) -> None:
        decider = _ScriptedDecider(FinishTurn(content="done"))
        terminal = ClientToolActionTerminal(loop_driver=decider)
        context = _make_context(
            messages=[
                ChatMessage(role="user", content="write a token to token.txt"),
                ChatMessage(
                    role="assistant",
                    content=None,
                    tool_calls=(
                        {
                            "id": "w1",
                            "type": "function",
                            "function": {
                                "name": "write",
                                "arguments": '{"filePath": "token.txt"}',
                            },
                        },
                    ),
                ),
                ChatMessage(
                    role="tool", content="ok: wrote token.txt", tool_call_id="w1"
                ),
            ]
        )

        await _collect(terminal.run(context))

        surfaced = decider.contexts[0]
        assert any(
            message.role == "tool" and message.content == "ok: wrote token.txt"
            for message in surfaced.messages
        )


class _FakeSeatFiller:
    """Seat-filler double returning a fixed tool-calling response."""

    def __init__(self, response: ToolCallingResponse) -> None:
        self._response = response

    async def generate_with_tools(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ToolCallingResponse:
        return self._response


class _FakeToolDispatch:
    """Tool-dispatch double returning a fixed ensemble deliverable."""

    def __init__(self, deliverable: str) -> None:
        self._deliverable = deliverable

    async def dispatch(
        self, call: InternalToolCall, *, session_id: str = ""
    ) -> ToolCallResult:
        return ToolCallSuccess(
            id=call.id,
            name=call.name,
            content=self._deliverable,
            envelope=DispatchEnvelope(status="success", primary=self._deliverable),
        )


class TestTerminalComposesRealLoopDriver:
    """+1 wiring test — the Terminal over the real Loop Driver.

    Proves the composition: the driver delegates generation to an ensemble and
    the Terminal marshals that ensemble's deliverable into the ``write`` the
    client executes. (The N emission cases above stub the decider; this one
    runs the real per-turn decision path.)
    """

    async def test_generation_deliverable_reaches_the_client_write(self) -> None:
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
        driver = LoopDriver(
            seat_filler=seat_filler,
            enforcer=SingleStepEnforcer(),
            tool_dispatch=_FakeToolDispatch(deliverable="def sort(xs): return xs"),
        )
        terminal = ClientToolActionTerminal(loop_driver=driver)

        chunks = await _collect(terminal.run(_make_context()))

        assert len(chunks) == 1
        tool_call = chunks[0]
        assert isinstance(tool_call, ClientToolCall)
        invocation = tool_call.tool_calls[0]
        assert invocation.name == "write"
        args = json.loads(invocation.arguments)
        assert args["filePath"] == "sort.py"
        assert args["content"] == "def sort(xs): return xs"
