"""Integration: the termination judgment reads the production digest join.

Step 5 (Integration Verification) for WP-LB-K (ADR-037). The unit tests
exercise the judgment over a ``SessionActionRecord`` populated directly.
This test verifies the production **lifecycle sequence** across the real
Loop Driver and the real Session Action Record (no stubs on either side
of the join):

  turn N   — the driver records the emitted client-tool action
  (the framework's own emission), then the client executes it locally
  turn N+1 — the client's ``role: tool`` result arrives; the driver
  joins it to the pending record, and the termination judgment reads
  the joined digest.

This is the production digest join the Conditional Acceptance gate
requires (FC-64): the judgment's evidence is framework-recorded paths
joined with client results, not a constructed digest. The model seat is
a scripted double ($0); the real-client layer is the deferred acceptance
gate (Run 1).
"""

from __future__ import annotations

from typing import Any

from llm_orc.agentic.budget_controller import BudgetController
from llm_orc.agentic.dispatch_envelope import DispatchEnvelope
from llm_orc.agentic.loop_driver import FinishTurn, LoopDriver
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


class _DelegateThenJudge:
    """A seat-filler that delegates one write, then fills the judgment seat.

    Implements both driver model ports (FC-68 shared-profile default): the
    action-call seat (``generate_with_tools`` — one ``invoke_ensemble``
    delegation) and the judgment seat (``generate_response`` — the captured
    judgment digest is recorded so the test can assert the production join).
    """

    def __init__(self) -> None:
        self.judgment_messages: list[str] = []

    async def generate_with_tools(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ToolCallingResponse:
        return ToolCallingResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id="t1",
                    name="invoke_ensemble",
                    arguments_json=(
                        '{"name": "code-generator", "input": "write it", '
                        '"filePath": "string_utils.py"}'
                    ),
                )
            ],
            finish_reason="tool_calls",
        )

    async def generate_response(self, message: str, role_prompt: str) -> str:
        self.judgment_messages.append(message)
        return "VERDICT: COMPLETE\nWrote string_utils.py."


class _EchoDispatch:
    """Tool dispatch double returning an inline deliverable envelope."""

    async def dispatch(
        self,
        call: InternalToolCall,
        *,
        session_id: str = "",
        model_profile_override: str | None = None,
    ) -> ToolCallResult:
        return ToolCallSuccess(
            id="t1",
            name="invoke_ensemble",
            content="def reverse_words(s): ...",
            envelope=DispatchEnvelope(
                status="success", primary="def reverse_words(s): ..."
            ),
        )


def _build_real_driver(
    seat: _DelegateThenJudge, record: SessionActionRecord
) -> LoopDriver:
    return LoopDriver(
        seat_filler=seat,
        enforcer=SingleStepEnforcer(),
        tool_dispatch=_EchoDispatch(),
        action_record=record,
        judgment_seat=seat,
        budget=BudgetController(turn_limit=100, token_limit=1_000_000),
        capabilities=frozenset({"code-generator"}),
    )


def _context(messages: list[ChatMessage]) -> SessionContext:
    return SessionContext(
        messages=messages,
        tools=[{"type": "function", "function": {"name": "write"}}],
        state=SessionState(
            identity=SessionIdentity(value="join-int", method="user_field")
        ),
    )


# A task that names no files, so completeness routes to the stochastic judge
# (the J-3 deterministic gate fires only for named-file tasks — Spike σ). The
# recorded path the digest quotes comes from the seat's delegation, not the
# task, so the join assertions are unaffected by the task wording.
_NO_FILE_TASK = "Add a helper that reverses the words in a sentence."


async def test_judgment_reads_the_framework_recorded_path_joined_with_result() -> None:
    seat = _DelegateThenJudge()
    record = SessionActionRecord()
    driver = _build_real_driver(seat, record)

    # Turn N — a first-turn delegation. The driver records the emitted
    # write (framework's own emission); no judgment fires on a first turn.
    await driver.decide(_context([ChatMessage(role="user", content=_NO_FILE_TASK)]))
    assert seat.judgment_messages == []
    [recorded] = record.records("join-int")
    assert recorded.target_path == "string_utils.py"
    assert recorded.result is None  # client has not executed it yet

    # Turn N+1 — the client echoes its tool result; this is the trailing
    # tail. The driver joins the result and the judgment reads the digest.
    await driver.decide(
        _context(
            [
                ChatMessage(role="user", content=_NO_FILE_TASK),
                ChatMessage(role="assistant", content=None),
                ChatMessage(role="tool", content="Wrote file successfully"),
            ]
        )
    )

    # The production join happened: the pending record carries the client
    # result, and the judgment digest quotes the framework-recorded path
    # joined with that result — not a digest reconstructed from the bare
    # client message (which carries no path).
    [joined] = record.records("join-int")
    assert joined.result == "Wrote file successfully"
    assert len(seat.judgment_messages) == 1
    digest = seat.judgment_messages[0]
    assert "write string_utils.py — tool result:" in digest
    assert "Wrote file successfully" in digest


async def test_complete_verdict_converges_the_session_end_to_end() -> None:
    seat = _DelegateThenJudge()
    record = SessionActionRecord()
    # Seed an action whose result the judge reads in the digest; the no-file
    # task routes completeness to the judge, which returns COMPLETE.
    record.record_action("join-int", action_kind="write", target_path="a.py")
    record.join_result("join-int", "Wrote file successfully")
    driver = _build_real_driver(seat, record)

    outcome = await driver.decide(
        _context(
            [
                ChatMessage(role="user", content=_NO_FILE_TASK),
                ChatMessage(role="assistant", content=None),
                ChatMessage(role="tool", content="Wrote file successfully"),
            ]
        )
    )

    # The session converges: a text-only finish, no delegated phantom
    # revision (the Finding F refutation at the harness layer).
    assert outcome == FinishTurn(content="Wrote string_utils.py.")
