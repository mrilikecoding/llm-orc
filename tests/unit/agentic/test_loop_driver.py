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
from pathlib import Path
from typing import Any

import pytest

from llm_orc.agentic.budget_controller import BudgetController
from llm_orc.agentic.dispatch_envelope import DispatchEnvelope
from llm_orc.agentic.dispatch_event_substrate import DispatchEventSubstrate
from llm_orc.agentic.loop_driver import (
    ApplyWork,
    CarryClientTool,
    FinishTurn,
    LoopDriver,
    SeatFiller,
    ToolDispatcher,
    TurnDecision,
    compose_form_directive,
    compose_judgment_message,
    parse_verdict,
    strip_verdict,
)
from llm_orc.agentic.orchestrator_tool_dispatch import (
    InternalToolCall,
    ToolCallResult,
    ToolCallSuccess,
)
from llm_orc.agentic.session_action_record import (
    ActionRecord,
    SessionActionRecord,
)
from llm_orc.agentic.session_artifact_store import SessionArtifactStore
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
    seat_filler: SeatFiller,
    *,
    tool_dispatch: _FakeToolDispatch | ToolDispatcher | None = None,
    capabilities: frozenset[str] = frozenset(),
    event_substrate: DispatchEventSubstrate | None = None,
    action_record: SessionActionRecord | None = None,
    judgment_seat: _FakeJudgmentSeat | None = None,
    budget: BudgetController | None = None,
    artifact_store: SessionArtifactStore | None = None,
) -> LoopDriver:
    return LoopDriver(
        seat_filler=seat_filler,
        enforcer=SingleStepEnforcer(),
        tool_dispatch=tool_dispatch or _FakeToolDispatch(),
        capabilities=capabilities,
        event_substrate=event_substrate,
        action_record=action_record or SessionActionRecord(),
        judgment_seat=judgment_seat or _FakeJudgmentSeat("VERDICT: REMAINING\n"),
        budget=budget or BudgetController(turn_limit=1_000, token_limit=1_000_000),
        artifact_store=artifact_store,
    )


def _offered_tool_names(seat_filler: _FakeSeatFiller) -> list[str]:
    _messages, tools = seat_filler.calls[0]
    names: list[str] = []
    for tool in tools:
        function = tool.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            names.append(function["name"])
    return names


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


class TestLoopDriverOffersDelegation:
    """WP-LB-G/WP-LB-I — the seat-filler is offered ``invoke_ensemble`` + guidance.

    Finding B (WP-LB-C real-OpenCode validation): without being offered the
    delegation tool, a real seat-filler can only act directly and never
    delegates. When capability ensembles are configured the driver augments the
    seat-filler's tool list with an ``invoke_ensemble`` tool enumerating them,
    and composes delegation guidance into the user-turn region (ADR-036;
    FC-58 — a framework system message loses the attention contest against
    the client's system prompt, Finding E).
    """

    _NAMED_TOOLS = [{"type": "function", "function": {"name": "write"}}]

    def _finishing_filler(self) -> _FakeSeatFiller:
        return _FakeSeatFiller(
            ToolCallingResponse(content="ok", tool_calls=[], finish_reason="stop")
        )

    async def test_offers_invoke_ensemble_enumerating_capabilities(self) -> None:
        seat_filler = self._finishing_filler()
        driver = _build_driver(
            seat_filler,
            capabilities=frozenset({"code-generator", "web-searcher"}),
        )

        await driver.decide(_make_context(tools=self._NAMED_TOOLS))

        names = _offered_tool_names(seat_filler)
        assert "invoke_ensemble" in names
        # The client tools are still offered alongside the delegation tool.
        assert "write" in names
        _messages, tools = seat_filler.calls[0]
        delegation = next(
            tool for tool in tools if tool["function"]["name"] == "invoke_ensemble"
        )
        enum = delegation["function"]["parameters"]["properties"]["name"]["enum"]
        assert enum == ["code-generator", "web-searcher"]
        assert delegation["function"]["parameters"]["required"] == [
            "name",
            "input",
            "filePath",
        ]

    async def test_first_turn_guidance_composes_into_the_user_task(self) -> None:
        """FC-58 (ADR-036) — guidance lives in the user-turn region.

        The first-turn form Spikes ψ.2/ψ′-A measured (40/40): guidance
        merged into the user task message; no framework-authored system
        message anywhere in the composed request.
        """
        seat_filler = self._finishing_filler()
        driver = _build_driver(seat_filler, capabilities=frozenset({"code-generator"}))

        await driver.decide(_make_context(tools=self._NAMED_TOOLS))

        messages, _tools = seat_filler.calls[0]
        assert all(message["role"] != "system" for message in messages)
        task_message = messages[-1]
        assert task_message["role"] == "user"
        assert "invoke_ensemble" in task_message["content"]
        assert "delegat" in task_message["content"].lower()
        # The client's task is preserved after the merged guidance.
        assert task_message["content"].endswith("what is 2 + 2?")

    async def test_trailing_tool_result_tail_gets_standalone_guidance(self) -> None:
        """FC-58 (ADR-036) — the C3 trailing form Spike ψ′-C measured.

        A tool-result tail gets the guidance appended as its own
        ``role: "user"`` message after the conversation, without mutating
        any client-authored message content.
        """
        seat_filler = self._finishing_filler()
        driver = _build_driver(seat_filler, capabilities=frozenset({"code-generator"}))
        conversation = [
            ChatMessage(role="user", content="Create fib.py"),
            ChatMessage(role="assistant", content="writing fib.py"),
            ChatMessage(role="tool", content="wrote fib.py"),
        ]

        await driver.decide(
            _make_context(messages=conversation, tools=self._NAMED_TOOLS)
        )

        messages, _tools = seat_filler.calls[0]
        guidance = messages[-1]
        assert guidance["role"] == "user"
        assert "invoke_ensemble" in guidance["content"]
        # Every client-authored message is carried through unmutated.
        assert messages[:-1] == [
            {"role": "user", "content": "Create fib.py"},
            {"role": "assistant", "content": "writing fib.py"},
            {"role": "tool", "content": "wrote fib.py"},
        ]

    async def test_client_system_prompt_stands_alone_in_the_system_region(
        self,
    ) -> None:
        """FC-58 — the composed request's only system message is the client's.

        The client's system prompt is the measured suppressor (Finding E);
        ADR-036 leaves it alone rather than contesting the slot.
        """
        seat_filler = self._finishing_filler()
        driver = _build_driver(seat_filler, capabilities=frozenset({"code-generator"}))
        conversation = [
            ChatMessage(role="system", content="You are a coding agent."),
            ChatMessage(role="user", content="Create fib.py"),
        ]

        await driver.decide(
            _make_context(messages=conversation, tools=self._NAMED_TOOLS)
        )

        messages, _tools = seat_filler.calls[0]
        system_messages = [m for m in messages if m["role"] == "system"]
        assert system_messages == [
            {"role": "system", "content": "You are a coding agent."}
        ]

    async def test_seat_filler_tool_list_completeness(self) -> None:
        """FC-62 (ψ.4c) — the composed tool list always includes the client tools.

        qwen3:14b answers a judged-incompatible tool list with an empty
        response; ``invoke_ensemble``-only tool lists must be
        unconstructable through the driver's composition path.
        """
        seat_filler = self._finishing_filler()
        driver = _build_driver(seat_filler, capabilities=frozenset({"code-generator"}))

        await driver.decide(_make_context(tools=self._NAMED_TOOLS))

        _messages, tools = seat_filler.calls[0]
        for client_tool in self._NAMED_TOOLS:
            assert client_tool in tools

    async def test_no_delegation_tool_or_guidance_without_capabilities(self) -> None:
        seat_filler = self._finishing_filler()
        driver = _build_driver(seat_filler)  # no capabilities

        await driver.decide(_make_context(tools=self._NAMED_TOOLS))

        assert "invoke_ensemble" not in _offered_tool_names(seat_filler)
        messages, _tools = seat_filler.calls[0]
        assert all(message["role"] != "system" for message in messages)
        assert all("invoke_ensemble" not in message["content"] for message in messages)


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
        # The generation task leads the dispatch input; the ADR-035 form
        # directive rides after it (FC-53 covers the directive itself).
        assert call.arguments["input"].startswith("write a sorting function")

        assert isinstance(outcome, ApplyWork)
        assert outcome.tool_name == "write"
        assert outcome.file_path == "sort.py"
        assert outcome.delegated_ensemble == "code-generator"
        # The envelope, not baked content, travels to the Terminal (which
        # marshals it); WP-LB-B's deliverable is the inline primary.
        assert outcome.envelope.primary == "def sort(xs): ..."


class _SequencedToolDispatch:
    """Tool-dispatch double returning a scripted sequence of deliverables.

    Each ``dispatch`` returns the next inline-envelope deliverable; the last
    one repeats once the sequence is exhausted. This scripts the coder bleeding
    invalid output then (optionally) re-sampling a valid file — the input to
    the server-side recovery loop (ADR-041 §Decision 3).
    """

    def __init__(self, deliverables: list[str]) -> None:
        self._deliverables = deliverables
        self.calls: list[InternalToolCall] = []

    async def dispatch(
        self, call: InternalToolCall, *, session_id: str = ""
    ) -> ToolCallResult:
        index = min(len(self.calls), len(self._deliverables) - 1)
        self.calls.append(call)
        deliverable = self._deliverables[index]
        return ToolCallSuccess(
            id=call.id,
            name=call.name,
            content=deliverable,
            envelope=DispatchEnvelope(status="success", primary=deliverable),
        )


def _delegate_seat_filler(file_path: str) -> _FakeSeatFiller:
    """A seat-filler that delegates one generation bound for ``file_path``."""
    return _FakeSeatFiller(
        ToolCallingResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id="t1",
                    name="invoke_ensemble",
                    arguments_json=json.dumps(
                        {
                            "name": "code-generator",
                            "input": "write the module",
                            "filePath": file_path,
                        }
                    ),
                )
            ],
            finish_reason="tool_calls",
        )
    )


class TestLoopDriverFormRecovery:
    """ADR-041 §Decision 3 — server-side re-dispatch on a parse-invalid deliverable.

    When a delegated generation's deliverable does not parse as what its
    destination path claims, the Loop Driver re-dispatches within the serving
    turn (the coder re-samples) up to ``_FORM_REDISPATCH_CAP`` (=2) times,
    rather than letting the refusal end the client loop (the smoke finding).
    Recovery is gated on the artifact store — its content-resolution
    dependency, always wired in production. The action is recorded once
    regardless of how many re-dispatches run (the single-record property).
    """

    async def test_intermittent_bleed_self_heals_within_the_turn(
        self, tmp_path: Path
    ) -> None:
        """Scenario: an intermittent bleed self-heals — a valid re-sample is
        reachable within the cap, so the valid deliverable is the outcome."""
        valid = "def main() -> None:\n    print('hi')\n"
        dispatch = _SequencedToolDispatch(["def main(: !! not python", valid])
        action_record = SessionActionRecord()
        driver = _build_driver(
            _delegate_seat_filler("cli.py"),
            tool_dispatch=dispatch,
            action_record=action_record,
            artifact_store=SessionArtifactStore(agentic_sessions_root=tmp_path),
        )

        outcome = await driver.decide(_make_context())

        assert isinstance(outcome, ApplyWork)
        assert outcome.envelope.primary == valid  # recovered to the valid sample
        assert len(dispatch.calls) == 2  # initial + one re-dispatch
        # single-record property: re-dispatch reuses the delegation path, so the
        # action is recorded once regardless of the re-dispatch count.
        assert len(action_record.records("test-session")) == 1

    async def test_persistent_bleed_exhausts_the_cap(self, tmp_path: Path) -> None:
        """Scenario: a persistent bleed exhausts the cap (initial + 2
        re-dispatches), returning the last attempt — the terminal's FormGate
        is the final arbiter that degrades it to a dispatch-failure stop."""
        dispatch = _SequencedToolDispatch(["def main(: still broken"])
        driver = _build_driver(
            _delegate_seat_filler("cli.py"),
            tool_dispatch=dispatch,
            artifact_store=SessionArtifactStore(agentic_sessions_root=tmp_path),
        )

        outcome = await driver.decide(_make_context())

        assert isinstance(outcome, ApplyWork)
        assert len(dispatch.calls) == 3  # initial + _FORM_REDISPATCH_CAP (2)
        assert outcome.envelope.primary == "def main(: still broken"

    async def test_recovery_inert_without_an_artifact_store(self) -> None:
        """Without the store (recovery's content-resolution dependency) the
        loop is inert — no re-dispatch — so a store-less driver behaves as it
        did before the gate. Production always wires the store."""
        dispatch = _SequencedToolDispatch(["def main(: broken"])
        driver = _build_driver(
            _delegate_seat_filler("cli.py"),
            tool_dispatch=dispatch,
            artifact_store=None,
        )

        outcome = await driver.decide(_make_context())

        assert isinstance(outcome, ApplyWork)
        assert len(dispatch.calls) == 1  # no re-dispatch
        assert outcome.envelope.primary == "def main(: broken"

    async def test_valid_deliverable_is_not_re_dispatched(self, tmp_path: Path) -> None:
        """A valid first sample needs no recovery — one dispatch, passes
        straight through (the common case is a no-op)."""
        valid = "x = 1\n"
        dispatch = _SequencedToolDispatch([valid])
        driver = _build_driver(
            _delegate_seat_filler("config.py"),
            tool_dispatch=dispatch,
            artifact_store=SessionArtifactStore(agentic_sessions_root=tmp_path),
        )

        outcome = await driver.decide(_make_context())

        assert isinstance(outcome, ApplyWork)
        assert len(dispatch.calls) == 1
        assert outcome.envelope.primary == valid


class TestFormDirectiveDestinationKeying:
    """FC-54 — ``compose_form_directive`` keys the directive to the tool.

    Scenarios.md §Client-Tool Deliverable Form Contract: the injected
    directive matches the decided destination (``write`` → bare file
    bytes; ``bash`` → bare command; ``edit`` → bare replacement). The
    wording is framework-owned prose (LB-6, tunable); the keying is the
    fitness property.
    """

    def test_write_directive_names_bare_file_bytes(self) -> None:
        directive = compose_form_directive("write")

        assert "file" in directive
        assert "ONLY" in directive
        assert "fence" in directive.lower()

    def test_bash_directive_names_bare_command(self) -> None:
        directive = compose_form_directive("bash")

        assert "command" in directive
        assert "file" not in directive  # a write-form directive on bash fails

    def test_edit_directive_names_bare_replacement(self) -> None:
        directive = compose_form_directive("edit")

        assert "replacement" in directive
        assert "command" not in directive

    def test_unknown_destination_fails_loud(self) -> None:
        with pytest.raises(ValueError, match="read"):
            compose_form_directive("read")


class TestLoopDriverInjectsFormDirective:
    """FC-53 — every client-tool-bound callee dispatch carries the directive.

    ADR-035 decision 1: the marshalling boundary composes the
    destination-keyed bare-output directive and injects it into the
    callee ``invoke_ensemble`` dispatch input. The ensemble stays
    destination-agnostic — the directive arrives per-dispatch, never
    via ensemble YAML.
    """

    async def test_loop_driver_injects_form_directive(self) -> None:
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

        await driver.decide(_make_context())

        dispatch_input = tool_dispatch.calls[0].arguments["input"]
        # The seat-filler's generation task is preserved...
        assert "write a sorting function" in dispatch_input
        # ...and the write-keyed directive rides the same dispatch input
        # (a client-tool-bound dispatch lacking the directive fails FC-53).
        assert compose_form_directive("write") in dispatch_input


class TestContentAnchor:
    """ADR-039 (Finding H, V-01) — the content anchor on the callee dispatch.

    On a delegated write into a session with already-produced siblings, the
    driver builds the anchor from the records it already holds (the content
    captured at the Terminal, V-04) and injects it into the callee dispatch
    input, so the dependent deliverable references real sibling APIs instead
    of inventing them. The current target is excluded — a file never anchors
    on itself. Scenarios from scenarios.md §"Content Anchor (ADR-039, Finding
    H)".
    """

    _CAPS = frozenset({"code-generator"})

    @staticmethod
    def _delegating_filler(file_path: str, *, task: str) -> _FakeSeatFiller:
        return _FakeSeatFiller(
            ToolCallingResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="t1",
                        name="invoke_ensemble",
                        arguments_json=json.dumps(
                            {
                                "name": "code-generator",
                                "input": task,
                                "filePath": file_path,
                            }
                        ),
                    )
                ],
                finish_reason="tool_calls",
            )
        )

    @staticmethod
    def _record_with_sibling(path: str, content: str) -> SessionActionRecord:
        record = SessionActionRecord()
        record.record_action("test-session", action_kind="write", target_path=path)
        record.record_content("test-session", content)
        return record

    async def test_anchor_injects_prior_sibling_api_into_callee_dispatch(
        self,
    ) -> None:
        record = self._record_with_sibling(
            "converters.py",
            "def celsius_to_fahrenheit(c: float) -> float:\n"
            "    return c * 9 / 5 + 32\n",
        )
        seat_filler = self._delegating_filler("cli.py", task="write cli.py")
        tool_dispatch = _FakeToolDispatch()
        driver = _build_driver(
            seat_filler,
            tool_dispatch=tool_dispatch,
            capabilities=self._CAPS,
            action_record=record,
        )

        await driver.decide(_make_context())

        dispatch_input = tool_dispatch.calls[0].arguments["input"]
        # the produced sibling's API surface is anchored into the dispatch
        assert "celsius_to_fahrenheit" in dispatch_input
        assert "These files already exist" in dispatch_input
        # task still leads and the form directive is preserved (delegation +
        # form preserved under the anchor — the ADR-039 FC)
        assert dispatch_input.startswith("write cli.py")
        assert compose_form_directive("write") in dispatch_input

    async def test_anchor_fires_for_a_prose_callee_too(self) -> None:
        """ADR-039 callee-agnostic scope: the anchor fires regardless of which
        capability ensemble is invoked — a prose callee (the README) receives
        the produced code siblings' API exactly as a code callee does. The
        injection never consults the capability name, so it cannot be omitted
        for ``prose-improver`` (Spike ξ prose arm: blind 0/10 → anchored 10/10).
        """
        record = self._record_with_sibling(
            "converters.py", "def celsius_to_fahrenheit(c: float) -> float: ...\n"
        )
        seat_filler = _FakeSeatFiller(
            ToolCallingResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="t1",
                        name="invoke_ensemble",
                        arguments_json=json.dumps(
                            {
                                "name": "prose-improver",
                                "input": "write README.md",
                                "filePath": "README.md",
                            }
                        ),
                    )
                ],
                finish_reason="tool_calls",
            )
        )
        tool_dispatch = _FakeToolDispatch()
        driver = _build_driver(
            seat_filler,
            tool_dispatch=tool_dispatch,
            capabilities=frozenset({"prose-improver"}),
            action_record=record,
        )

        await driver.decide(_make_context())

        dispatch_input = tool_dispatch.calls[0].arguments["input"]
        assert "celsius_to_fahrenheit" in dispatch_input
        assert "These files already exist" in dispatch_input

    async def test_anchor_excludes_the_current_target(self) -> None:
        """A sibling list containing only the file being rewritten yields no
        anchor — the callee never anchors a file on itself.
        """
        record = self._record_with_sibling("cli.py", "def main() -> None: ...\n")
        seat_filler = self._delegating_filler("cli.py", task="rewrite cli.py")
        tool_dispatch = _FakeToolDispatch()
        driver = _build_driver(
            seat_filler,
            tool_dispatch=tool_dispatch,
            capabilities=self._CAPS,
            action_record=record,
        )

        await driver.decide(_make_context())

        dispatch_input = tool_dispatch.calls[0].arguments["input"]
        assert "These files already exist" not in dispatch_input
        assert dispatch_input == f"rewrite cli.py\n\n{compose_form_directive('write')}"

    async def test_no_anchor_without_prior_siblings(self) -> None:
        """A first delegation has no produced siblings, so the dispatch input
        is the unanchored task + directive — byte-equal to the pre-ADR-039 form
        (no regression on first-file or no-dependency writes).
        """
        seat_filler = self._delegating_filler(
            "converters.py", task="write converters.py"
        )
        tool_dispatch = _FakeToolDispatch()
        driver = _build_driver(
            seat_filler, tool_dispatch=tool_dispatch, capabilities=self._CAPS
        )

        await driver.decide(_make_context())

        dispatch_input = tool_dispatch.calls[0].arguments["input"]
        assert dispatch_input == (
            f"write converters.py\n\n{compose_form_directive('write')}"
        )

    async def test_turn_decision_marks_the_anchor_present(self) -> None:
        """V-05 — the anchor-presence FC is refutable from the event stream:
        an anchored generation turn stamps ``content_anchor_present=True`` so
        the discharge run reads presence from the TurnDecision, not the raw
        dispatch payload.
        """
        record = self._record_with_sibling(
            "converters.py", "def celsius_to_fahrenheit(c: float) -> float: ...\n"
        )
        substrate = DispatchEventSubstrate()
        sink = _CapturingSink()
        substrate.register_sink(sink)
        driver = _build_driver(
            self._delegating_filler("cli.py", task="write cli.py"),
            capabilities=self._CAPS,
            action_record=record,
            event_substrate=substrate,
        )

        await driver.decide(_make_context())

        assert _turn_decisions(sink)[0].content_anchor_present is True

    async def test_turn_decision_marks_the_anchor_absent_on_first_delegation(
        self,
    ) -> None:
        """A first delegation has no siblings, so no anchor — the event stamps
        ``content_anchor_present=False`` (the unanchored baseline is observable).
        """
        substrate = DispatchEventSubstrate()
        sink = _CapturingSink()
        substrate.register_sink(sink)
        driver = _build_driver(
            self._delegating_filler("converters.py", task="write converters.py"),
            capabilities=self._CAPS,
            event_substrate=substrate,
        )

        await driver.decide(_make_context())

        assert _turn_decisions(sink)[0].content_anchor_present is False

    async def test_records_without_content_do_not_anchor(self) -> None:
        """A carried action (a read, or a write whose dispatch failed) has no
        captured content, so it contributes nothing to the anchor.
        """
        record = SessionActionRecord()
        record.record_action("test-session", action_kind="read", target_path="a.py")
        seat_filler = self._delegating_filler("cli.py", task="write cli.py")
        tool_dispatch = _FakeToolDispatch()
        driver = _build_driver(
            seat_filler,
            tool_dispatch=tool_dispatch,
            capabilities=self._CAPS,
            action_record=record,
        )

        await driver.decide(_make_context())

        dispatch_input = tool_dispatch.calls[0].arguments["input"]
        assert dispatch_input == f"write cli.py\n\n{compose_form_directive('write')}"


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


class TestLoopDriverBudgetCap:
    """AS-3 (FC-69) — the turn cap is the absolute ceiling on this surface.

    ADR-037 names the BudgetController as the deterministic backstop beneath
    the termination mechanism. On the tool-driven surface the loop spans
    stateless HTTP requests, so the cap measures against the
    conversation-recovered turn index. When the cap is reached the session
    terminates regardless of judgment outcomes — refutable: a session
    exceeding the cap without termination violates AS-3 on this surface.
    """

    @staticmethod
    def _context_at_turn(turn: int) -> SessionContext:
        """A trailing context whose recovered turn index is ``turn``.

        ``_turn_index`` is one past the count of prior assistant turns, so
        ``turn - 1`` prior assistant turns puts this turn at ``turn``.
        """
        messages: list[ChatMessage] = [ChatMessage(role="user", content="write a.py")]
        for _ in range(turn - 1):
            messages.append(ChatMessage(role="assistant", content=None))
            messages.append(ChatMessage(role="tool", content="Wrote file"))
        return _make_context(messages=messages)

    async def test_session_terminates_at_the_turn_cap(self) -> None:
        judgment = _FakeJudgmentSeat("VERDICT: REMAINING\nmore to do")
        seat_filler = _FakeSeatFiller(
            ToolCallingResponse(content="", tool_calls=[], finish_reason="stop")
        )
        driver = _build_driver(
            seat_filler,
            capabilities=frozenset({"code-generator"}),
            judgment_seat=judgment,
            budget=BudgetController(turn_limit=3, token_limit=1_000_000),
        )

        outcome = await driver.decide(self._context_at_turn(3))

        assert isinstance(outcome, FinishTurn)
        # The cap is the absolute ceiling: it fires before the judgment and
        # before any action call.
        assert judgment.calls == []
        assert seat_filler.calls == []

    async def test_below_the_cap_the_turn_proceeds(self) -> None:
        seat_filler = _FakeSeatFiller(
            ToolCallingResponse(content="ok", tool_calls=[], finish_reason="stop")
        )
        driver = _build_driver(
            seat_filler,
            budget=BudgetController(turn_limit=10, token_limit=1_000_000),
        )

        await driver.decide(self._context_at_turn(2))

        assert len(seat_filler.calls) == 1

    async def test_cap_termination_emits_a_turn_decision(self) -> None:
        substrate = DispatchEventSubstrate()
        sink = _CapturingSink()
        substrate.register_sink(sink)
        driver = _build_driver(
            _FakeSeatFiller(
                ToolCallingResponse(content="", tool_calls=[], finish_reason="stop")
            ),
            budget=BudgetController(turn_limit=2, token_limit=1_000_000),
            event_substrate=substrate,
        )

        await driver.decide(self._context_at_turn(2))

        decision = _turn_decisions(sink)[0]
        assert decision.action == "finish"


class _FakeJudgmentSeat:
    """Judgment-seat double returning a pre-scripted verdict response.

    Records ``(message, role_prompt)`` pairs so tests can assert the
    bare-form composition (FC-63): a framework-authored judge system
    message plus one user message carrying the quoted task, the digest,
    and the deliverable-accounting question — no client prompt, no tools
    (the port shape itself carries no tools).
    """

    def __init__(self, response: str) -> None:
        self._response = response
        self.calls: list[tuple[str, str]] = []

    async def generate_response(self, message: str, role_prompt: str) -> str:
        self.calls.append((message, role_prompt))
        return self._response


def _trailing_context() -> SessionContext:
    """A trailing tool-result tail whose task names NO files — the shape the
    ADR-037 stochastic judge opens. (J-3 routes named-file tasks to the
    deterministic completeness gate instead; the judge is the no-named-files
    fallback these tests exercise.)
    """
    return _make_context(
        messages=[
            ChatMessage(role="system", content="You are OpenCode, a client."),
            ChatMessage(role="user", content="Summarize the project discussion notes."),
            ChatMessage(role="assistant", content=None),
            ChatMessage(role="tool", content="Wrote file successfully"),
        ]
    )


class TestLoopDriverTerminationJudgment:
    """ADR-037 — two-call trailing composition (V-01/02/04/05/08).

    Scenarios from scenarios.md §"Session-Termination Mechanism": the
    judgment opens every trailing tail on the delegation surface, the
    COMPLETE branch finishes protocol-clean, the REMAINING branch falls
    through to the unchanged ADR-036 C3 action call with the judgment
    exchange discarded.
    """

    _CAPS = frozenset({"code-generator"})

    @staticmethod
    def _finishing_filler() -> _FakeSeatFiller:
        return _FakeSeatFiller(
            ToolCallingResponse(content="ok", tool_calls=[], finish_reason="stop")
        )

    async def test_trailing_tail_judgment_first(self) -> None:
        """FC-63 (the first red test, gate-named): a trailing tool-result
        tail produces the judgment dispatch before any guidance-composed
        call — on COMPLETE the seat-filler is never called at all.
        """
        judgment = _FakeJudgmentSeat(
            "VERDICT: COMPLETE\nWrote string_utils.py and its tests."
        )
        seat_filler = self._finishing_filler()
        driver = _build_driver(
            seat_filler, capabilities=self._CAPS, judgment_seat=judgment
        )

        outcome = await driver.decide(_trailing_context())

        assert len(judgment.calls) == 1
        assert seat_filler.calls == []
        assert isinstance(outcome, FinishTurn)

    async def test_judgment_call_is_bare_form(self) -> None:
        """The judgment request carries the framework judge system message
        and one user message (quoted task + digest + accounting question);
        the client's system prompt does not ride along.
        """
        judgment = _FakeJudgmentSeat("VERDICT: COMPLETE\nDone.")
        record = SessionActionRecord()
        record.record_action(
            "test-session", action_kind="write", target_path="string_utils.py"
        )
        record.join_result("test-session", "Wrote file successfully")
        driver = _build_driver(
            self._finishing_filler(),
            capabilities=self._CAPS,
            judgment_seat=judgment,
            action_record=record,
        )

        await driver.decide(_trailing_context())

        message, role_prompt = judgment.calls[0]
        assert "judge whether the user's requested work has been completed" in (
            role_prompt
        )
        assert "You are OpenCode, a client." not in message
        assert "Summarize the project discussion notes." in message
        assert "write string_utils.py — tool result:" in message
        assert "Wrote file successfully" in message

    async def test_judgment_question_carries_the_accounting_standard(self) -> None:
        """V-04: the deliverable-accounting standard is in the question —
        a successful write counts as produced; code correctness is
        explicitly out of the judgment's scope (round 1's unanswerable
        standard, Form B 0/10).
        """
        judgment = _FakeJudgmentSeat("VERDICT: COMPLETE\nDone.")
        driver = _build_driver(
            self._finishing_filler(),
            capabilities=self._CAPS,
            judgment_seat=judgment,
        )

        await driver.decide(_trailing_context())

        message, _role_prompt = judgment.calls[0]
        assert (
            "A successful write of a requested file counts as that "
            "deliverable being produced" in message
        )
        assert "you are not being asked to verify code correctness" in message

    async def test_complete_verdict_yields_clean_finish(self) -> None:
        """FC-65 (V-05/V-07): COMPLETE returns the judgment summary as a
        text-only finish with no ``VERDICT:`` line leaked.
        """
        judgment = _FakeJudgmentSeat(
            "VERDICT: COMPLETE\nWrote string_utils.py and its tests."
        )
        driver = _build_driver(
            self._finishing_filler(),
            capabilities=self._CAPS,
            judgment_seat=judgment,
        )

        outcome = await driver.decide(_trailing_context())

        assert outcome == FinishTurn(content="Wrote string_utils.py and its tests.")

    async def test_remaining_verdict_call2_form_preserved(self) -> None:
        """FC-66 amended (ADR-038, V-38-3): REMAINING falls through to exactly
        one action call composed per ADR-036's trailing form (session messages
        + standalone trailing guidance) **plus the remaining-work anchor** —
        with the rest of the judgment exchange (judge system message, digest,
        Status-check question, VERDICT literal) absent from its context. The
        anchor is the routed-forward stripped statement; the byte-equality to
        the pre-ADR-037 E4b composition is intentionally broken by ADR-038.
        """
        judgment = _FakeJudgmentSeat("VERDICT: REMAINING\ntests not yet written")
        # An acting seat-filler (the normal REMAINING path): it delegates, so the
        # F-σ.1 no-tool-call retry does not fire and exactly one action call is
        # composed. (A finishing filler here would now legitimately retry — the
        # Finding I fix — which is covered by TestRemainingRetry.)
        seat_filler = _FakeSeatFiller(_delegation_response("test_string_utils.py"))
        driver = _build_driver(
            seat_filler, capabilities=self._CAPS, judgment_seat=judgment
        )
        context = _trailing_context()

        await driver.decide(context)

        assert len(seat_filler.calls) == 1
        messages, _tools = seat_filler.calls[0]
        # The session projection rides unchanged; only the trailing region
        # carries the guidance + anchor.
        assert messages[:-1] == [
            {"role": message.role, "content": message.content}
            for message in context.messages
        ]
        assert messages[-1]["role"] == "user"
        # The rest of the judgment exchange stays discarded (ADR-037
        # context-bounding preserved): no question, no digest provenance line.
        for composed in messages:
            content = composed["content"] or ""
            assert "Status check" not in content
            assert "action record" not in content.lower()
            # the literal verdict token never carries (strip_verdict removes it)
            assert "VERDICT:" not in content


class TestRemainingWorkAnchor:
    """ADR-038 — the remaining-work anchor (V-38-1/V-38-2).

    On a REMAINING verdict the judge's own stripped remaining-work statement
    is routed forward into call 2's trailing region, followed by the fixed
    framework imperative — instead of being discarded. Scenarios from
    scenarios.md §"Remaining-Work Anchor (ADR-038, Finding G)".
    """

    _CAPS = frozenset({"code-generator"})

    @staticmethod
    def _finishing_filler() -> _FakeSeatFiller:
        return _FakeSeatFiller(
            ToolCallingResponse(content="ok", tool_calls=[], finish_reason="stop")
        )

    @staticmethod
    def _trailing_content(seat_filler: _FakeSeatFiller) -> str:
        messages, _tools = seat_filler.calls[0]
        return str(messages[-1]["content"])

    async def test_remaining_anchor_carries_statement_and_imperative(self) -> None:
        judgment = _FakeJudgmentSeat(
            "VERDICT: REMAINING\nThe test file test_string_utils.py "
            "has not been written yet."
        )
        seat_filler = self._finishing_filler()
        driver = _build_driver(
            seat_filler, capabilities=self._CAPS, judgment_seat=judgment
        )

        await driver.decide(_trailing_context())

        trailing = self._trailing_content(seat_filler)
        # the judge's actual stripped statement (routed forward, not templated)
        assert "The test file test_string_utils.py has not been written yet." in (
            trailing
        )
        # the fixed framework imperative
        assert "Produce that next." in trailing
        # the ADR-036 guidance is still present (anchor appends, not replaces)
        assert "invoke_ensemble" in trailing

    async def test_no_anchor_on_first_turn(self) -> None:
        """A first turn fires no judgment, so no remaining-work anchor."""
        seat_filler = self._finishing_filler()
        driver = _build_driver(
            seat_filler,
            capabilities=self._CAPS,
            judgment_seat=_FakeJudgmentSeat("VERDICT: REMAINING\nx remains"),
        )

        await driver.decide(_make_context())

        assert "Produce that next." not in self._trailing_content(seat_filler)

    async def test_no_anchor_on_new_user_task_tail(self) -> None:
        """A trailing tail with a new user task is ADR-036's merge branch —
        no judgment, no anchor.
        """
        seat_filler = self._finishing_filler()
        driver = _build_driver(
            seat_filler,
            capabilities=self._CAPS,
            judgment_seat=_FakeJudgmentSeat("VERDICT: REMAINING\nx remains"),
        )
        context = _make_context(
            messages=[
                ChatMessage(role="user", content="write a.py"),
                ChatMessage(role="assistant", content=None),
                ChatMessage(role="tool", content="Wrote file successfully"),
                ChatMessage(role="user", content="now also write b.py"),
            ]
        )

        await driver.decide(context)

        assert "Produce that next." not in self._trailing_content(seat_filler)

    async def test_no_anchor_when_parse_miss(self) -> None:
        """A judgment with no parseable verdict falls through to the action
        call, but with no usable statement there is no anchor (the turn still
        delegates via the unanchored guidance; the next re-judgment re-anchors).
        """
        seat_filler = self._finishing_filler()
        driver = _build_driver(
            seat_filler,
            capabilities=self._CAPS,
            judgment_seat=_FakeJudgmentSeat("I am not sure what remains here"),
        )

        await driver.decide(_trailing_context())

        assert "Produce that next." not in self._trailing_content(seat_filler)

    async def test_unparseable_verdict_falls_through_to_the_action_call(
        self,
    ) -> None:
        """A judgment response with no parseable ``VERDICT:`` line must not
        end the session (false-stop drops work silently); the turn falls
        through to the action call and the parse miss is observable as
        ``judgment_verdict=None`` on a trailing turn.
        """
        judgment = _FakeJudgmentSeat("I think the work might be done?")
        seat_filler = self._finishing_filler()
        substrate = DispatchEventSubstrate()
        sink = _CapturingSink()
        substrate.register_sink(sink)
        driver = _build_driver(
            seat_filler,
            capabilities=self._CAPS,
            judgment_seat=judgment,
            event_substrate=substrate,
        )

        await driver.decide(_trailing_context())

        assert len(seat_filler.calls) == 1
        decision = _turn_decisions(sink)[0]
        assert decision.tail_kind == "trailing_tool_result"
        assert decision.judgment_verdict is None

    async def test_turn_decision_carries_the_verdict_on_both_branches(self) -> None:
        """FC-67: COMPLETE stamps action=finish + verdict; REMAINING stamps
        the action call's outcome + verdict.
        """
        substrate = DispatchEventSubstrate()
        sink = _CapturingSink()
        substrate.register_sink(sink)
        complete_driver = _build_driver(
            self._finishing_filler(),
            capabilities=self._CAPS,
            judgment_seat=_FakeJudgmentSeat("VERDICT: COMPLETE\nDone."),
            event_substrate=substrate,
        )
        await complete_driver.decide(_trailing_context())

        remaining_driver = _build_driver(
            self._finishing_filler(),
            capabilities=self._CAPS,
            judgment_seat=_FakeJudgmentSeat("VERDICT: REMAINING\nmore to do"),
            event_substrate=substrate,
        )
        await remaining_driver.decide(_trailing_context())

        complete, remaining = _turn_decisions(sink)
        assert complete.action == "finish"
        assert complete.judgment_verdict == "COMPLETE"
        assert remaining.judgment_verdict == "REMAINING"

    async def test_no_judgment_on_first_turn_or_new_user_task(self) -> None:
        """Preservation: ADR-036's merge branch is untouched — the judgment
        is specific to no-new-task tool-result tails (ψ/ψ′ first-turn
        evidence, 40/40, rides untouched).
        """
        judgment = _FakeJudgmentSeat("VERDICT: COMPLETE\nDone.")
        driver = _build_driver(
            self._finishing_filler(),
            capabilities=self._CAPS,
            judgment_seat=judgment,
        )

        await driver.decide(_make_context())
        await driver.decide(
            _make_context(
                messages=[
                    ChatMessage(role="user", content="write a.py"),
                    ChatMessage(role="assistant", content=None),
                    ChatMessage(role="tool", content="Wrote file successfully"),
                    ChatMessage(role="user", content="now write b.py"),
                ]
            )
        )

        assert judgment.calls == []

    async def test_no_judgment_without_capabilities(self) -> None:
        """Without capabilities there is no delegation guidance and no
        suppression to fix (ψ″ E2: unguided work-complete tails finish
        10/10); the trailing turn behaves as before.
        """
        judgment = _FakeJudgmentSeat("VERDICT: COMPLETE\nDone.")
        driver = _build_driver(
            self._finishing_filler(),
            capabilities=frozenset(),
            judgment_seat=judgment,
        )

        await driver.decide(_trailing_context())

        assert judgment.calls == []


class TestJudgmentHelpers:
    """The named stateless helpers (the ``compose_form_directive``
    precedent) — unit-testable in isolation.
    """

    def test_parse_verdict_complete(self) -> None:
        assert parse_verdict("VERDICT: COMPLETE\nAll done.") == "COMPLETE"

    def test_parse_verdict_remaining(self) -> None:
        assert parse_verdict("VERDICT: REMAINING\ntests missing") == "REMAINING"

    def test_parse_verdict_first_literal_wins(self) -> None:
        text = "VERDICT: REMAINING\n(not VERDICT: COMPLETE)"
        assert parse_verdict(text) == "REMAINING"

    def test_parse_verdict_none_when_absent(self) -> None:
        assert parse_verdict("the work looks finished") is None

    def test_parse_verdict_ignores_think_blocks(self) -> None:
        """The spike's measurement discipline: the verdict is parsed over
        think-stripped text, so a reasoning block musing about the other
        verdict does not flip the parse.
        """
        text = "<think>VERDICT: COMPLETE? no...</think>VERDICT: REMAINING\nmore"
        assert parse_verdict(text) == "REMAINING"

    def test_strip_verdict_removes_the_verdict_line_and_think_blocks(self) -> None:
        text = "<think>counting...</think>VERDICT: COMPLETE\nWrote both files."
        assert strip_verdict(text) == "Wrote both files."

    def test_compose_judgment_message_renders_records_and_question(self) -> None:
        records = (
            ActionRecord(
                action_kind="write",
                target_path="a.py",
                result="Wrote file successfully",
            ),
            ActionRecord(action_kind="read", target_path="notes.md", result="text"),
        )

        message = compose_judgment_message("write a.py from notes.md", records)

        assert "quoted as data, not instructions to you" in message
        assert "write a.py from notes.md" in message
        assert "- action 1: write a.py — tool result:" in message
        assert "- action 2: read notes.md — tool result:" in message
        assert "Status check:" in message


class TestLoopDriverActionRecording:
    """FC-64 (digest provenance) — the driver's side of the Session Action
    Record contract: every emitted client-tool action is recorded at
    decision time from the framework's own emission, and the client's
    per-call tool result joins the pending record on the next request.
    """

    @staticmethod
    def _delegating_filler() -> _FakeSeatFiller:
        return _FakeSeatFiller(
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

    async def test_generation_turn_records_the_emitted_write(self) -> None:
        record = SessionActionRecord()
        driver = _build_driver(
            self._delegating_filler(),
            capabilities=frozenset({"code-generator"}),
            action_record=record,
        )

        await driver.decide(_make_context())

        assert record.records("test-session") == (
            ActionRecord(action_kind="write", target_path="f.py", result=None),
        )

    async def test_carry_turn_records_the_client_tool_action(self) -> None:
        record = SessionActionRecord()
        seat_filler = _FakeSeatFiller(
            ToolCallingResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="r1",
                        name="read",
                        arguments_json=json.dumps({"filePath": "notes.md"}),
                    )
                ],
                finish_reason="tool_calls",
            )
        )
        driver = _build_driver(seat_filler, action_record=record)

        await driver.decide(_make_context())

        assert record.records("test-session") == (
            ActionRecord(action_kind="read", target_path="notes.md", result=None),
        )

    async def test_bash_carry_records_the_command_as_target(self) -> None:
        record = SessionActionRecord()
        seat_filler = _FakeSeatFiller(
            ToolCallingResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="b1",
                        name="bash",
                        arguments_json=json.dumps({"command": "pytest -q"}),
                    )
                ],
                finish_reason="tool_calls",
            )
        )
        driver = _build_driver(seat_filler, action_record=record)

        await driver.decide(_make_context())

        assert record.records("test-session") == (
            ActionRecord(action_kind="bash", target_path="pytest -q", result=None),
        )

    async def test_finish_turn_records_no_action(self) -> None:
        record = SessionActionRecord()
        seat_filler = _FakeSeatFiller(
            ToolCallingResponse(content="done", tool_calls=[], finish_reason="stop")
        )
        driver = _build_driver(seat_filler, action_record=record)

        await driver.decide(_make_context())

        assert record.records("test-session") == ()

    async def test_client_tool_result_joins_the_pending_record(self) -> None:
        """The loop spans HTTP requests: the action recorded on turn N is
        joined with the client's ``role: tool`` result when turn N+1's
        request arrives (the production digest join, FC-64).
        """
        record = SessionActionRecord()
        driver = _build_driver(
            self._delegating_filler(),
            capabilities=frozenset({"code-generator"}),
            action_record=record,
        )
        await driver.decide(
            _make_context(messages=[ChatMessage(role="user", content="write f.py")])
        )

        follow_up = _make_context(
            messages=[
                ChatMessage(role="user", content="write f.py"),
                ChatMessage(role="assistant", content=None),
                ChatMessage(role="tool", content="Wrote file successfully"),
                ChatMessage(role="user", content="now write g.py too"),
            ]
        )
        await driver.decide(follow_up)

        first_record = record.records("test-session")[0]
        assert first_record.result == "Wrote file successfully"


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

    async def test_first_turn_stamps_finish_policy_fields(self) -> None:
        """V-06 (ADR-037 §Decision 6, FC-67): the finish-policy fields ride
        every TurnDecision. A first turn is not a trailing tail, so no
        judgment fires — ``judgment_verdict`` is ``None``.
        """
        seat_filler = _FakeSeatFiller(
            ToolCallingResponse(content="done", tool_calls=[], finish_reason="stop")
        )
        _, sink, driver = self._capture(seat_filler)

        await driver.decide(_make_context())

        decision = _turn_decisions(sink)[0]
        assert decision.tail_kind == "first_turn"
        assert decision.judgment_verdict is None

    async def test_new_user_task_tail_stamps_tail_kind(self) -> None:
        """A trailing turn carrying a genuine new user task is ADR-036's
        merge branch — untouched by the judgment (the preservation
        scenario); the tail kind makes that discrimination observable.
        """
        seat_filler = _FakeSeatFiller(
            ToolCallingResponse(content="done", tool_calls=[], finish_reason="stop")
        )
        _, sink, driver = self._capture(seat_filler)
        context = _make_context(
            messages=[
                ChatMessage(role="user", content="write a.py"),
                ChatMessage(role="assistant", content=None),
                ChatMessage(role="tool", content="Wrote file successfully"),
                ChatMessage(role="user", content="now also write b.py"),
            ]
        )

        await driver.decide(context)

        decision = _turn_decisions(sink)[0]
        assert decision.tail_kind == "new_user_task"
        assert decision.judgment_verdict is None

    async def test_trailing_tool_result_tail_stamps_tail_kind(self) -> None:
        seat_filler = _FakeSeatFiller(
            ToolCallingResponse(content="done", tool_calls=[], finish_reason="stop")
        )
        _, sink, driver = self._capture(seat_filler)
        context = _make_context(
            messages=[
                ChatMessage(role="user", content="write a.py"),
                ChatMessage(role="assistant", content=None),
                ChatMessage(role="tool", content="Wrote file successfully"),
            ]
        )

        await driver.decide(context)

        decision = _turn_decisions(sink)[0]
        assert decision.tail_kind == "trailing_tool_result"

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


class TestTurnShapeStamping:
    """FC-59 — each TurnDecision carries the meter's turn-shape classification.

    The shape is derived from the turn's *outcome* (WP-LB-M): a write (a
    delegated ``ApplyWork`` or a literal ``write``/``edit`` carry) is
    ``generation`` (the denominator); a read, command, or finish is ``carry``;
    a repair-shaped or uncovered-domain instruction is ``boundary_excluded``
    regardless of the action. A generation turn that delegates is the rate
    numerator; a framework finish (cap reached, judge COMPLETE) is shaped
    ``carry`` so it never inflates the denominator. Scenarios.md
    §Delegation-Decision Mechanism (the classifier denominator; boundary turns
    excluded, not guessed).
    """

    _CAPS = frozenset({"code-generator"})

    def _capture(
        self,
        seat_filler: _FakeSeatFiller,
        *,
        judgment_seat: _FakeJudgmentSeat | None = None,
        budget: BudgetController | None = None,
    ) -> tuple[_CapturingSink, LoopDriver]:
        substrate = DispatchEventSubstrate()
        sink = _CapturingSink()
        substrate.register_sink(sink)
        driver = _build_driver(
            seat_filler,
            capabilities=self._CAPS,
            event_substrate=substrate,
            judgment_seat=judgment_seat,
            budget=budget,
        )
        return sink, driver

    async def test_generation_shaped_turn_stamps_generation(self) -> None:
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
                                "input": "sort a list",
                                "filePath": "sort.py",
                            }
                        ),
                    )
                ],
                finish_reason="tool_calls",
            )
        )
        sink, driver = self._capture(seat_filler)
        ctx = _make_context(
            messages=[
                ChatMessage(
                    role="user",
                    content="Write a python module sort.py with a sorting function.",
                )
            ]
        )

        await driver.decide(ctx)

        decision = _turn_decisions(sink)[0]
        assert decision.turn_shape == "generation"
        assert decision.delegated_ensemble == "code-generator"

    async def test_read_turn_stamps_carry(self) -> None:
        seat_filler = _FakeSeatFiller(
            ToolCallingResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="t1",
                        name="read",
                        arguments_json=json.dumps({"filePath": "sort.py"}),
                    )
                ],
                finish_reason="tool_calls",
            )
        )
        sink, driver = self._capture(seat_filler)
        ctx = _make_context(
            messages=[
                ChatMessage(
                    role="user", content="Read sort.py and tell me what it does."
                )
            ]
        )

        await driver.decide(ctx)

        decision = _turn_decisions(sink)[0]
        assert decision.turn_shape == "carry"
        assert decision.grounded_carry_held is True

    async def test_repair_shaped_turn_stamps_boundary_excluded(self) -> None:
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
                                "input": "fix it",
                                "filePath": "sort.py",
                            }
                        ),
                    )
                ],
                finish_reason="tool_calls",
            )
        )
        sink, driver = self._capture(seat_filler)
        ctx = _make_context(
            messages=[
                ChatMessage(
                    role="user",
                    content="Fix the bug in sort.py where it crashes on empty input.",
                )
            ]
        )

        await driver.decide(ctx)

        decision = _turn_decisions(sink)[0]
        assert decision.turn_shape == "boundary_excluded"

    async def test_complete_judgment_finish_stamps_carry(self) -> None:
        judgment = _FakeJudgmentSeat("VERDICT: COMPLETE\nWrote string_utils.py.")
        seat_filler = _FakeSeatFiller(
            ToolCallingResponse(content="ok", tool_calls=[], finish_reason="stop")
        )
        sink, driver = self._capture(seat_filler, judgment_seat=judgment)

        await driver.decide(_trailing_context())

        decision = _turn_decisions(sink)[0]
        assert decision.action == "finish"
        assert decision.judgment_verdict == "COMPLETE"
        assert decision.turn_shape == "carry"

    async def test_cap_finish_stamps_carry(self) -> None:
        seat_filler = _FakeSeatFiller(
            ToolCallingResponse(content="", tool_calls=[], finish_reason="stop")
        )
        sink, driver = self._capture(
            seat_filler,
            budget=BudgetController(turn_limit=2, token_limit=1_000_000),
        )
        ctx = _make_context(
            messages=[
                ChatMessage(role="user", content="write string_utils.py"),
                ChatMessage(role="assistant", content=None),
                ChatMessage(role="tool", content="ok"),
                ChatMessage(role="assistant", content=None),
                ChatMessage(role="tool", content="ok"),
            ]
        )

        await driver.decide(ctx)

        decision = _turn_decisions(sink)[0]
        assert decision.action == "finish"
        assert decision.turn_shape == "carry"

    async def test_remaining_delegated_write_stamps_generation(self) -> None:
        """WP-LB-M — a trailing REMAINING turn that delegates a write is a
        ``generation`` turn, even though the judge's remaining-work anchor is a
        descriptive statement with no generation verb. The shape follows the
        action taken, not the anchor wording — the gap the ladder surfaced
        (multi-file sessions under-instrumented because the instruction-side
        classification read the descriptive anchor and stamped ``carry``).
        """
        judgment = _FakeJudgmentSeat(
            "VERDICT: REMAINING\nThe test file test_string_utils.py is still missing."
        )
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
                                "input": "write the test file",
                                "filePath": "test_string_utils.py",
                            }
                        ),
                    )
                ],
                finish_reason="tool_calls",
            )
        )
        sink, driver = self._capture(seat_filler, judgment_seat=judgment)

        await driver.decide(_trailing_context())

        decision = _turn_decisions(sink)[0]
        assert decision.turn_shape == "generation"
        assert decision.delegated_ensemble == "code-generator"

    async def test_mixed_read_first_turn_stamps_carry(self) -> None:
        """WP-LB-M — a turn whose action is a read is ``carry``, even when the
        user task frames a write. The mixed read-then-write flow opens with a
        read; the instruction's write framing must not stamp the read turn
        ``generation`` (the axis-B mis-stamp).
        """
        seat_filler = _FakeSeatFiller(
            ToolCallingResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="t1",
                        name="read",
                        arguments_json=json.dumps({"filePath": "config.py"}),
                    )
                ],
                finish_reason="tool_calls",
            )
        )
        sink, driver = self._capture(seat_filler)
        ctx = _make_context(
            messages=[
                ChatMessage(
                    role="user",
                    content="Read config.py, then write a module loader.py from it.",
                )
            ]
        )

        await driver.decide(ctx)

        decision = _turn_decisions(sink)[0]
        assert decision.turn_shape == "carry"
        assert decision.grounded_carry_held is True

    async def test_literal_write_carry_stamps_generation(self) -> None:
        """WP-LB-M — a literal ``write`` carry (the C1 inline-write the model
        produces instead of delegating) is a ``generation`` turn: it counts in
        the denominator with no numerator, so the delegation rate drops. C1
        detection is preserved by reading the action, not the instruction (the
        literal instruction would classify ``carry``).
        """
        seat_filler = _FakeSeatFiller(
            ToolCallingResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="t1",
                        name="write",
                        arguments_json=json.dumps(
                            {"filePath": "greeting.py", "content": "print('hi')\n"}
                        ),
                    )
                ],
                finish_reason="tool_calls",
            )
        )
        sink, driver = self._capture(seat_filler)
        ctx = _make_context(
            messages=[
                ChatMessage(
                    role="user",
                    content=(
                        "Write the file greeting.py with exactly this content:\n"
                        "```\nprint('hi')\n```"
                    ),
                )
            ]
        )

        await driver.decide(ctx)

        decision = _turn_decisions(sink)[0]
        assert decision.turn_shape == "generation"
        assert decision.delegated_ensemble is None


class _SequencedSeatFiller:
    """Seat-filler returning a scripted response per call (for retry tests).

    Records each call so a test can assert how many times the driver dispatched
    the action call. After the list is exhausted it repeats the last response.
    """

    def __init__(self, responses: list[ToolCallingResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []

    async def generate_with_tools(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ToolCallingResponse:
        self.calls.append((messages, tools))
        return self._responses[min(len(self.calls) - 1, len(self._responses) - 1)]


def _no_action_response() -> ToolCallingResponse:
    return ToolCallingResponse(
        content="Making progress.", tool_calls=[], finish_reason="stop"
    )


def _delegation_response(file_path: str = "cli.py") -> ToolCallingResponse:
    return ToolCallingResponse(
        content="",
        tool_calls=[
            ToolCall(
                id="t1",
                name="invoke_ensemble",
                arguments_json=json.dumps(
                    {
                        "name": "code-generator",
                        "input": f"write {file_path}",
                        "filePath": file_path,
                    }
                ),
            )
        ],
        finish_reason="tool_calls",
    )


class TestRemainingRetry:
    """ADR-039 loop-back / Finding I — the REMAINING-retry (F-σ.1).

    On a REMAINING verdict the judge has affirmed work remains, so a seat-filler
    no-tool-call is an incoherent stall, not a legitimate finish (a finish here
    ends the client loop with deliverables missing — the premature-finish the
    real client exposed). The driver retries the action call once before honoring
    a finish; the AS-3 cap remains the ultimate backstop. The retry is
    REMAINING-only: first-turn / non-judgment no-action responses finish without
    a retry. Scenarios from scenarios.md §"Premature-finish (Finding I)".
    """

    _CAPS = frozenset({"code-generator"})

    @staticmethod
    def _remaining_judge() -> _FakeJudgmentSeat:
        return _FakeJudgmentSeat("VERDICT: REMAINING\ncli.py has not been written yet.")

    async def test_retry_recovers_a_stall_on_remaining(self) -> None:
        seat = _SequencedSeatFiller([_no_action_response(), _delegation_response()])
        driver = _build_driver(
            seat, capabilities=self._CAPS, judgment_seat=self._remaining_judge()
        )

        outcome = await driver.decide(_trailing_context())

        assert isinstance(outcome, ApplyWork)  # the retry recovered the stall
        assert len(seat.calls) == 2  # original action call + one retry

    async def test_finish_when_the_retry_also_stalls(self) -> None:
        seat = _SequencedSeatFiller([_no_action_response(), _no_action_response()])
        driver = _build_driver(
            seat, capabilities=self._CAPS, judgment_seat=self._remaining_judge()
        )

        outcome = await driver.decide(_trailing_context())

        assert isinstance(outcome, FinishTurn)  # both stalled → finish (AS-3 beyond)
        assert len(seat.calls) == 2  # retried exactly once, then gave up

    async def test_no_retry_off_the_remaining_branch(self) -> None:
        """A first-turn no-action is not a REMAINING stall — finish, no retry."""
        seat = _SequencedSeatFiller([_no_action_response()])
        driver = _build_driver(seat, capabilities=self._CAPS)

        outcome = await driver.decide(_make_context())  # first_turn tail, no judgment

        assert isinstance(outcome, FinishTurn)
        assert len(seat.calls) == 1  # no retry off the REMAINING branch


def _trailing_with_task(task: str) -> SessionContext:
    """A trailing tool-result tail carrying a custom user task."""
    return _make_context(
        messages=[
            ChatMessage(role="system", content="You are OpenCode, a client."),
            ChatMessage(role="user", content=task),
            ChatMessage(role="assistant", content=None),
            ChatMessage(role="tool", content="Wrote file successfully"),
        ]
    )


class TestDeterministicCompleteness:
    """J-3 (Spike σ) — deterministic completeness gate for named-file tasks.

    For a task that names its deliverables the framework checks requested vs
    produced deterministically (no stochastic judge → the false-COMPLETE failure
    mode cannot occur), and ``requested − produced`` composes the ADR-038 anchor.
    A task that names no files falls back to the ADR-037 judge. Scenarios from
    scenarios.md §"Deterministic completeness (Finding I / Spike σ)".
    """

    _CAPS = frozenset({"code-generator"})
    _TASK = "Create converters.py, test_converters.py, and README.md."

    @staticmethod
    def _record(*produced: str) -> SessionActionRecord:
        record = SessionActionRecord()
        for path in produced:
            record.record_action("test-session", action_kind="write", target_path=path)
        return record

    def test_extract_requested_deliverables_filters_non_files(self) -> None:
        from llm_orc.agentic.loop_driver import _extract_requested_deliverables

        got = _extract_requested_deliverables(
            "Create converters.py, test_converters.py, cli.py, test_cli.py, "
            "README.md. Convert 273.15 K and 9.5; e.g. the notes."
        )
        assert got == frozenset(
            {
                "converters.py",
                "test_converters.py",
                "cli.py",
                "test_cli.py",
                "README.md",
            }
        )

    async def test_complete_when_all_requested_produced_no_judge_call(self) -> None:
        # The judge would say COMPLETE here too, but the point is it is never
        # consulted — completeness is deterministic.
        judge = _FakeJudgmentSeat("VERDICT: COMPLETE\n")
        record = self._record("converters.py", "test_converters.py", "README.md")
        driver = _build_driver(
            _FakeSeatFiller(_no_action_response()),  # not reached on COMPLETE
            capabilities=self._CAPS,
            judgment_seat=judge,
            action_record=record,
        )

        outcome = await driver.decide(_trailing_with_task(self._TASK))

        assert isinstance(outcome, FinishTurn)
        assert len(judge.calls) == 0  # deterministic — the stochastic judge is skipped

    async def test_remaining_overrides_a_wrong_judge_and_anchors_missing(self) -> None:
        # The stochastic judge would WRONGLY say COMPLETE after one file (the
        # Spike σ false-COMPLETE). The deterministic gate ignores it and reports
        # REMAINING, anchoring the two unproduced files.
        judge = _FakeJudgmentSeat("VERDICT: COMPLETE\n")
        seat = _FakeSeatFiller(_delegation_response("test_converters.py"))
        record = self._record("converters.py")
        driver = _build_driver(
            seat, capabilities=self._CAPS, judgment_seat=judge, action_record=record
        )

        await driver.decide(_trailing_with_task(self._TASK))

        assert len(judge.calls) == 0  # deterministic — judge not consulted
        trailing = str(seat.calls[0][0][-1]["content"])
        assert "test_converters.py" in trailing
        assert "README.md" in trailing
        assert "converters.py" not in trailing.replace("test_converters.py", "")

    async def test_falls_back_to_judge_when_task_names_no_files(self) -> None:
        judge = _FakeJudgmentSeat("VERDICT: REMAINING\nstill summarizing")
        seat = _FakeSeatFiller(_delegation_response("notes.txt"))
        driver = _build_driver(
            seat,
            capabilities=self._CAPS,
            judgment_seat=judge,
            action_record=self._record(),
        )

        await driver.decide(_trailing_with_task("Summarize the meeting discussion."))

        assert len(judge.calls) == 1  # no requested set → stochastic-judge fallback

    async def test_persisted_requested_survives_a_compacted_later_turn(self) -> None:
        # An earlier turn named all the files (persist-once captured them). The
        # current turn's task is client-compacted — it names no files. Without
        # persist-once the gate re-derives an empty set and collapses to the
        # judge (which false-COMPLETEs here); with it, the persisted set still
        # anchors REMAINING. This is the live Spike σ run-2 failure mode.
        judge = _FakeJudgmentSeat("VERDICT: COMPLETE\n")
        seat = _FakeSeatFiller(_delegation_response("test_converters.py"))
        record = self._record("converters.py")  # 1 of 3 produced
        record.set_requested_if_absent(
            "test-session",
            frozenset({"converters.py", "test_converters.py", "README.md"}),
        )
        driver = _build_driver(
            seat, capabilities=self._CAPS, judgment_seat=judge, action_record=record
        )

        await driver.decide(_trailing_with_task("Continue with the remaining work."))

        assert len(judge.calls) == 0  # persisted set used → no stochastic fallback
        trailing = str(seat.calls[0][0][-1]["content"])
        assert "test_converters.py" in trailing
        assert "README.md" in trailing

    async def test_decide_persists_the_requested_set_once_on_first_naming(self) -> None:
        # The driver captures the requested set from the task it sees this turn,
        # so a later compacted turn reads the stable persisted set.
        judge = _FakeJudgmentSeat("VERDICT: COMPLETE\n")
        seat = _FakeSeatFiller(_delegation_response("test_converters.py"))
        record = self._record("converters.py")
        driver = _build_driver(
            seat, capabilities=self._CAPS, judgment_seat=judge, action_record=record
        )

        await driver.decide(_trailing_with_task(self._TASK))

        assert record.requested("test-session") == frozenset(
            {"converters.py", "test_converters.py", "README.md"}
        )
