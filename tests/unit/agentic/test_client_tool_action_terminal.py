"""Tests for the Client-Tool-Action Terminal (Cycle 7 loop-back WP-LB-C, ADR-034).

The Terminal owns tool-call *emission* and the artifact-bridge marshalling: it
asks the Loop Driver to decide one turn and maps the resulting
:class:`TurnOutcome` to the shared ``OrchestratorChunk`` vocabulary, resolving
substrate-routed deliverable content through the Artifact Bridge. These tests
drive the mapping in isolation against a scripted decider (the N emission
cases), verify substrate-routed full-fidelity content + degradation paths
(FC-49 / FC-48), then verify composition with the real Loop Driver (the +1
wiring test). Scenarios from ``docs/agentic-serving/scenarios.md``
§"Client-Tool-Action Terminal and Artifact-Bridge (ADR-034)".
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from llm_orc.agentic.artifact_bridge import (
    ArtifactBridge,
    FormRefusedError,
    parse_check_form_gate,
)
from llm_orc.agentic.budget_controller import BudgetController
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
from llm_orc.agentic.session_action_record import SessionActionRecord
from llm_orc.agentic.session_artifact_store import SessionArtifactStore
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


class _FakeJudgmentSeat:
    """Judgment-seat double — the contexts here never reach a judgment."""

    async def generate_response(self, message: str, role_prompt: str) -> str:
        return "VERDICT: REMAINING\n"


def _unused_bridge() -> ArtifactBridge:
    """A bridge whose store is never read (inline / finish / carry outcomes).

    Only a substrate-routed ``ApplyWork`` reads the store; finish,
    grounded-carry, and inline-``primary`` outcomes resolve without touching
    disk, so an unrooted store is fine for those.
    """
    store = SessionArtifactStore(agentic_sessions_root=Path("unused-by-inline-tests"))
    return ArtifactBridge(store)


def _inline_terminal(decider: _ScriptedDecider) -> ClientToolActionTerminal:
    """A Terminal over a scripted decider whose bridge is never read."""
    return ClientToolActionTerminal(loop_driver=decider, bridge=_unused_bridge())


async def _collect(
    chunks: AsyncIterator[OrchestratorChunk],
) -> list[OrchestratorChunk]:
    return [chunk async for chunk in chunks]


def _one_invocation(chunks: list[OrchestratorChunk]) -> ToolCallInvocation:
    assert len(chunks) == 1
    tool_call = chunks[0]
    assert isinstance(tool_call, ClientToolCall)
    return tool_call.tool_calls[0]


class TestFinishOutcomeEmission:
    """A finish outcome emits assistant text (if any) then a stop completion."""

    async def test_finish_with_text_emits_content_delta_then_completion(self) -> None:
        terminal = _inline_terminal(_ScriptedDecider(FinishTurn(content="2 + 2 = 4.")))

        chunks = await _collect(terminal.run(_make_context()))

        assert chunks == [
            ContentDelta(content="2 + 2 = 4."),
            Completion(finish_reason="stop"),
        ]

    async def test_finish_without_text_emits_only_completion(self) -> None:
        terminal = _inline_terminal(_ScriptedDecider(FinishTurn(content=None)))

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
        terminal = _inline_terminal(
            _ScriptedDecider(CarryClientTool(invocation=invocation))
        )

        chunks = await _collect(terminal.run(_make_context()))

        assert chunks == [ClientToolCall(tool_calls=(invocation,))]


class TestApplyWorkEmission:
    """A generation outcome emits a tool call carrying the deliverable (FC-47)."""

    async def test_inline_deliverable_marshalled_into_write(self) -> None:
        outcome = ApplyWork(
            invocation_id="t1",
            tool_name="write",
            file_path="sort.py",
            envelope=DispatchEnvelope(status="success", primary="def sort(xs): ..."),
            delegated_ensemble="code-generator",
        )
        terminal = _inline_terminal(_ScriptedDecider(outcome))

        chunks = await _collect(terminal.run(_make_context()))

        invocation = _one_invocation(chunks)
        assert invocation.name == "write"
        args = json.loads(invocation.arguments)
        assert args["filePath"] == "sort.py"
        assert args["content"] == "def sort(xs): ..."


class _RecordingBridge(ArtifactBridge):
    """Bridge subclass recording the destination tool each marshal received."""

    def __init__(self, store: SessionArtifactStore) -> None:
        super().__init__(store)
        self.destinations: list[str | None] = []
        self.destination_paths: list[str | None] = []

    def marshal(
        self,
        envelope: DispatchEnvelope,
        *,
        destination_tool: str | None = None,
        destination_path: str | None = None,
    ) -> str | bytes:
        self.destinations.append(destination_tool)
        self.destination_paths.append(destination_path)
        return super().marshal(
            envelope,
            destination_tool=destination_tool,
            destination_path=destination_path,
        )


class TestTerminalThreadsDestinationTool:
    """FC-57 — the Terminal passes the turn's destination tool to the bridge.

    ADR-035 §4: the FormGate seam lives on the marshal surface; the
    Terminal threads ``outcome.tool_name`` through the existing
    Terminal→Bridge edge (a shared-type extension, not a new edge) so
    the gate knows which destination's form it is evaluating.
    """

    async def test_apply_work_marshals_with_the_decided_tool(
        self, tmp_path: Path
    ) -> None:
        bridge = _RecordingBridge(SessionArtifactStore(agentic_sessions_root=tmp_path))
        outcome = ApplyWork(
            invocation_id="t1",
            tool_name="write",
            file_path="sort.py",
            envelope=DispatchEnvelope(status="success", primary="def sort(xs): ..."),
            delegated_ensemble="code-generator",
        )
        terminal = ClientToolActionTerminal(
            loop_driver=_ScriptedDecider(outcome), bridge=bridge
        )

        await _collect(terminal.run(_make_context()))

        assert bridge.destinations == ["write"]
        # ADR-041 V-01: the destination *path* is threaded to the gate too,
        # not just the tool — the parse-check derives the extension from it.
        assert bridge.destination_paths == ["sort.py"]


class TestApplyWorkCapturesDeliverableContent:
    """ADR-039 V-04 — the Terminal captures the resolved deliverable content
    into the session action record, the content anchor's source.

    The capture rides the existing Terminal→Bridge marshalling: the same bytes
    that land in the client ``write`` are recorded onto the turn's action, so a
    later turn's callee can anchor on the produced sibling. The driver recorded
    the action at decide time; the Terminal joins the resolved content to it.
    """

    async def test_marshalled_content_is_recorded_on_the_latest_action(
        self,
    ) -> None:
        action_record = SessionActionRecord()
        action_record.record_action(
            "terminal-test", action_kind="write", target_path="converters.py"
        )
        outcome = ApplyWork(
            invocation_id="t1",
            tool_name="write",
            file_path="converters.py",
            envelope=DispatchEnvelope(
                status="success", primary="def c_to_f(c: float) -> float: ..."
            ),
            delegated_ensemble="code-generator",
        )
        terminal = ClientToolActionTerminal(
            loop_driver=_ScriptedDecider(outcome),
            bridge=_unused_bridge(),
            action_record=action_record,
        )

        await _collect(terminal.run(_make_context()))

        records = action_record.records("terminal-test")
        assert records[-1].content == "def c_to_f(c: float) -> float: ..."

    async def test_no_content_captured_when_the_bridge_fails(
        self, tmp_path: Path
    ) -> None:
        """A failed marshal degrades to a dispatch-failure completion and yields
        no usable deliverable, so the record carries no content to anchor on.
        """
        store = SessionArtifactStore(agentic_sessions_root=tmp_path)
        ref = store.write_deliverable(
            session_id="s1",
            dispatch_id="d1",
            deliverable_name="present",
            content="real content\n",
            content_type="application/python",
        )
        ghost = dataclasses.replace(ref, path=ref.path + ".missing")
        action_record = SessionActionRecord()
        action_record.record_action(
            "terminal-test", action_kind="write", target_path="ghost.py"
        )
        outcome = ApplyWork(
            invocation_id="t1",
            tool_name="write",
            file_path="ghost.py",
            envelope=DispatchEnvelope(
                status="success",
                primary="ghost summary",
                artifacts=[dataclasses.asdict(ghost)],
            ),
            delegated_ensemble="code-generator",
        )
        terminal = ClientToolActionTerminal(
            loop_driver=_ScriptedDecider(outcome),
            bridge=ArtifactBridge(store),
            action_record=action_record,
        )

        await _collect(terminal.run(_make_context()))

        assert action_record.records("terminal-test")[-1].content is None


class TestApplyWorkSubstrateFidelity:
    """FC-49 — a substrate-routed deliverable is marshalled at full fidelity.

    The bridge reads the full content from the Session Artifact Store, not the
    summary on ``envelope.primary`` (scenarios.md §ADR-034 "Artifact-bridge
    reads the substrate-routed deliverable and marshals it into tool-call
    content"). This is the live-loop wiring WP-LB-D's bridge enabled.
    """

    async def test_write_carries_full_artifact_content_not_the_summary(
        self, tmp_path: Path
    ) -> None:
        store = SessionArtifactStore(agentic_sessions_root=tmp_path)
        content = (
            "class Calculator:\n"
            "    def add(self, a: int, b: int) -> int:\n"
            "        return a + b\n"
        )
        ref = store.write_deliverable(
            session_id="2026-06-02T12:00:00Z-cc33",
            dispatch_id="dispatch-009",
            deliverable_name="calculator",
            content=content,
            content_type="application/python",
        )
        outcome = ApplyWork(
            invocation_id="t1",
            tool_name="write",
            file_path="calculator.py",
            envelope=DispatchEnvelope(
                status="success",
                primary="calculator.py: 3 lines — a Calculator class",  # summary
                artifacts=[dataclasses.asdict(ref)],
            ),
            delegated_ensemble="code-generator",
        )
        terminal = ClientToolActionTerminal(
            loop_driver=_ScriptedDecider(outcome), bridge=ArtifactBridge(store)
        )

        chunks = await _collect(terminal.run(_make_context()))

        invocation = _one_invocation(chunks)
        args = json.loads(invocation.arguments)
        assert args["content"] == content
        assert "summary" not in args["content"]


class TestApplyWorkDegradesOnBridgeFailure:
    """Error handling — a bridge failure degrades to a dispatch-failure completion.

    The Terminal never emits a ``write`` with empty or fabricated content
    (FC-48 / the fidelity FC forbid a paraphrase substitute); a missing
    artifact or a binary deliverable becomes a text completion instead
    (system-design.agents.md §Client-Tool-Action Terminal error handling).
    """

    async def test_unresolvable_reference_yields_a_text_completion(
        self, tmp_path: Path
    ) -> None:
        store = SessionArtifactStore(agentic_sessions_root=tmp_path)
        ref = store.write_deliverable(
            session_id="s1",
            dispatch_id="d1",
            deliverable_name="present",
            content="real content\n",
            content_type="application/python",
        )
        ghost = dataclasses.replace(ref, path=ref.path + ".missing")
        outcome = ApplyWork(
            invocation_id="t1",
            tool_name="write",
            file_path="ghost.py",
            envelope=DispatchEnvelope(
                status="success",
                primary="ghost summary",
                artifacts=[dataclasses.asdict(ghost)],
            ),
            delegated_ensemble="code-generator",
        )
        terminal = ClientToolActionTerminal(
            loop_driver=_ScriptedDecider(outcome), bridge=ArtifactBridge(store)
        )

        chunks = await _collect(terminal.run(_make_context()))

        assert not any(isinstance(chunk, ClientToolCall) for chunk in chunks)
        assert chunks[-1] == Completion(finish_reason="stop")

    async def test_form_refusal_yields_a_text_completion(self, tmp_path: Path) -> None:
        """A refusing FormGate degrades like a missing artifact (ADR-035 §4).

        The refusal channel pre-exists the detect-and-refuse gate so
        installing it touches only the seam (FC-57) — the Terminal
        already degrades a FormRefusedError to a dispatch-failure
        completion rather than sending a clearly-wrong deliverable to
        the client.
        """

        def refusing_gate(
            content: str | bytes, tool: str | None, path: str | None
        ) -> str | bytes:
            raise FormRefusedError("multi-fence deliverable; refusing to emit")

        store = SessionArtifactStore(agentic_sessions_root=tmp_path)
        outcome = ApplyWork(
            invocation_id="t1",
            tool_name="write",
            file_path="bad.py",
            envelope=DispatchEnvelope(
                status="success", primary="```python\n...\n```\nprose\n```...```"
            ),
            delegated_ensemble="code-generator",
        )
        terminal = ClientToolActionTerminal(
            loop_driver=_ScriptedDecider(outcome),
            bridge=ArtifactBridge(store, form_gate=refusing_gate),
        )

        chunks = await _collect(terminal.run(_make_context()))

        assert not any(isinstance(chunk, ClientToolCall) for chunk in chunks)
        assert chunks[-1] == Completion(finish_reason="stop")

    async def test_real_parse_gate_refuses_invalid_python_through_the_terminal(
        self, tmp_path: Path
    ) -> None:
        """ADR-041 §Decision 1 integration: the production parse-check gate,
        installed at the seam, degrades an unparseable ``.py`` deliverable to a
        dispatch-failure ``stop`` — no broken ``write`` reaches the client
        (the protection floor, verified with the real gate not a stub)."""
        store = SessionArtifactStore(agentic_sessions_root=tmp_path)
        bled = "def main():\n    return 1\nThis function returns one.\n"
        outcome = ApplyWork(
            invocation_id="t1",
            tool_name="write",
            file_path="cli.py",
            envelope=DispatchEnvelope(status="success", primary=bled),
            delegated_ensemble="code-generator",
        )
        terminal = ClientToolActionTerminal(
            loop_driver=_ScriptedDecider(outcome),
            bridge=ArtifactBridge(store, form_gate=parse_check_form_gate),
        )

        chunks = await _collect(terminal.run(_make_context()))

        assert not any(isinstance(chunk, ClientToolCall) for chunk in chunks)
        assert chunks[-1] == Completion(finish_reason="stop")

    async def test_real_parse_gate_passes_valid_python_through_the_terminal(
        self, tmp_path: Path
    ) -> None:
        """The gate is a no-op on valid content: a parseable ``.py`` deliverable
        emits as a normal ``write`` tool call (fidelity preserved)."""
        store = SessionArtifactStore(agentic_sessions_root=tmp_path)
        content = "def main() -> None:\n    print('ok')\n"
        outcome = ApplyWork(
            invocation_id="t1",
            tool_name="write",
            file_path="cli.py",
            envelope=DispatchEnvelope(status="success", primary=content),
            delegated_ensemble="code-generator",
        )
        terminal = ClientToolActionTerminal(
            loop_driver=_ScriptedDecider(outcome),
            bridge=ArtifactBridge(store, form_gate=parse_check_form_gate),
        )

        chunks = await _collect(terminal.run(_make_context()))

        invocation = _one_invocation(chunks)
        assert invocation.name == "write"
        args = json.loads(invocation.arguments)
        assert args["content"] == content

    async def test_binary_deliverable_yields_a_text_completion(
        self, tmp_path: Path
    ) -> None:
        store = SessionArtifactStore(agentic_sessions_root=tmp_path)
        ref = store.write_deliverable(
            session_id="s1",
            dispatch_id="d1",
            deliverable_name="blob",
            content=b"\x80\x81\x82\xff",  # not UTF-8 decodable
            content_type="application/octet-stream",
        )
        outcome = ApplyWork(
            invocation_id="t1",
            tool_name="write",
            file_path="blob.bin",
            envelope=DispatchEnvelope(
                status="success",
                primary="blob summary",
                artifacts=[dataclasses.asdict(ref)],
            ),
            delegated_ensemble="code-generator",
        )
        terminal = ClientToolActionTerminal(
            loop_driver=_ScriptedDecider(outcome), bridge=ArtifactBridge(store)
        )

        chunks = await _collect(terminal.run(_make_context()))

        assert not any(isinstance(chunk, ClientToolCall) for chunk in chunks)
        assert chunks[-1] == Completion(finish_reason="stop")


class TestLoopParticipation:
    """FC-50 — the Terminal hands the full conversation to the driver.

    Loop participation is structural: the Terminal does not filter the request
    to the last user message (the single-turn pipeline's ``_extract_request``
    behavior, which drops ``role: "tool"`` results). The trailing tool result
    reaches the driver's per-turn decision.
    """

    async def test_trailing_tool_result_reaches_the_driver(self) -> None:
        decider = _ScriptedDecider(FinishTurn(content="done"))
        terminal = _inline_terminal(decider)
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
        self,
        call: InternalToolCall,
        *,
        session_id: str = "",
        model_profile_override: str | None = None,
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
            action_record=SessionActionRecord(),
            judgment_seat=_FakeJudgmentSeat(),
            budget=BudgetController(turn_limit=1_000, token_limit=1_000_000),
        )
        terminal = ClientToolActionTerminal(loop_driver=driver, bridge=_unused_bridge())

        chunks = await _collect(terminal.run(_make_context()))

        invocation = _one_invocation(chunks)
        assert invocation.name == "write"
        args = json.loads(invocation.arguments)
        assert args["filePath"] == "sort.py"
        assert args["content"] == "def sort(xs): return xs"
