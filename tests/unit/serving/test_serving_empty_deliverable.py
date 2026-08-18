"""#166: an empty deliverable is never a client write.

The seat contract asserts artifact PRESENCE, never non-emptiness;
``ast.parse("")`` succeeds so the form gate passes honestly; and the caller
maps any outcome carrying ``file`` and ``content`` to a client Write. So a
seat that SUCCEEDS with an empty artifact reaches the client as an empty
file, and on the re-fix route that is a clobber of a file the client already
has.

Pre-flight settled reachability by measurement (design:
docs/plans/2026-08-17-166-empty-deliverable-design.md). The route this issue
originally named — a DEAD seat — is not the fault: the real seat_contract
node rejects a dead seat on all three build routes and emit refuses. The two
captures below are the routes where ``accept`` is genuinely defeated with an
empty deliverable, and they drive the real nodes end to end rather than
hand-feeding a verdict, which is how the original reproduction went wrong.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO = Path(__file__).resolve().parents[3]
SCRIPTS = REPO / ".llm-orc" / "scripts" / "agentic_serving"

sys.path.insert(0, str(SCRIPTS))
from emit import TERMINALS  # type: ignore  # noqa: E402

from llm_orc.web.serving.chunks import ClientToolCall  # noqa: E402
from llm_orc.web.serving.serving_ensemble_caller import (  # noqa: E402
    _outcome_chunks,
    _reject_kind,
    _RejectPrefixes,
    _RejectTerminal,
)

# The same derivation the caller's own _load_emit_reject_prefixes uses: a
# terminal that mints nothing is filtered out there, so it is filtered here.
_PREFIXES: _RejectPrefixes = tuple(
    _RejectTerminal(terminal.prefix, terminal.mints)
    for terminal in TERMINALS.values()
    if terminal.mints
)

_HEALTHY = "def restock(item, n):\n    return n + 1\n"


def _node(script: str, deps: dict[str, Any], input_data: str = "") -> dict[str, Any]:
    """One script node, as the engine runs it."""
    payload: dict[str, Any] = {"dependencies": deps}
    if input_data:
        payload["input_data"] = input_data
    completed = subprocess.run(
        [sys.executable, str(SCRIPTS / script)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=True,
    )
    result: dict[str, Any] = json.loads(completed.stdout)
    return result


def _dep(value: Any) -> dict[str, str]:
    return {"response": value if isinstance(value, str) else json.dumps(value)}


def _seat_wire(envelope: dict[str, Any]) -> str:
    """What the skeleton's dispatched ``seat`` dep carries: the child
    ensemble's results with its TERMINAL node last, since _helpers.terminal
    peels ``results`` by last key."""
    return json.dumps({"results": {"gather": _dep("..."), "envelope": _dep(envelope)}})


def _serving_tail(target: str, seat_response: str, file: str) -> dict[str, Any]:
    """seat_contract -> shape -> form_gate -> emit, all real."""
    resolve = _dep(
        {
            "target": target,
            "kind": "python_module",
            "file": file,
            "build": True,
            "dispatch_input": "fix it",
            "is_build_ask": True,
        }
    )
    deps: dict[str, Any] = {"resolve": resolve, "seat": {"response": seat_response}}
    seat_contract = _node("seat_contract.py", deps)
    shape = _node("shape.py", {**deps, "seat_contract": _dep(seat_contract)})
    form_gate = _node("form_gate.py", {"shape": _dep(shape)})
    return _node("emit.py", {"form_gate": _dep(form_gate)})


def _chunks(outcome: dict[str, Any]) -> list[Any]:
    return _outcome_chunks(outcome, [], _PREFIXES)


def _text(chunks: list[Any]) -> str:
    return "".join(str(getattr(chunk, "content", "")) for chunk in chunks)


def _refix_envelope(model_edit: str) -> dict[str, Any]:
    """The real re-fix chain from the faulty node output down: gather ->
    select -> executor -> envelope. ``model_edit`` is the only input that
    varies, and it is the single fault."""
    dispatch_input = (
        "assistant: [ran pytest -q (failed)]\n"
        "  E   assert restock('x', 2) == 3\n"
        "\n"
        "[PRIOR CODE: this turn's write, before the re-fix]\n"
        "```python\n"
        "def restock(item, n):\n"
        "    return n\n"
        "```\n"
        "\n\nCurrent request: fix calc.py so restock adds one"
    )
    gather = _node("refix_gather.py", {}, input_data=dispatch_input)
    assert gather["needs_model_edit"], "the deterministic edit must not pin here"
    assert not gather["visible_test"].strip(), "the smoke-only path is the subject"
    select = _node(
        "refix_select.py",
        {"gather": _dep(gather), "model_edit": _dep(model_edit)},
    )
    executor = _node("accept_executor.py", {"select": _dep(select)})
    return _node(
        "refix_envelope.py", {"select": _dep(select), "executor": _dep(executor)}
    )


_TESTS = (
    "```python\n"
    "from inventory import restock\n"
    "def test_restock():\n"
    "    assert restock('x', 2) == 3\n"
    "```\n"
)


def _build_gated_envelope(code_writer: str, requirement: str) -> dict[str, Any]:
    """The real build-gated tail from the faulty node output down: gather ->
    executor -> judge -> accept_gate -> envelope.

    The rendered context carries ``inventory.py``, the module the tests
    exercise, which is what lets an EMPTY deliverable pass the executor:
    _materialize shadows the TARGET file only, and the target is whatever
    filename the requirement names. ``code_writer`` is the single fault and
    stays a raw seat response — gather owns the fence extraction, so the
    test never has to guess at it."""
    context = (
        "assistant: [read inventory.py]\n"
        + "".join(f"  {line}\n" for line in _HEALTHY.splitlines())
        + f"\n\nCurrent request: {requirement}"
    )
    gather = _node(
        "accept_gather.py",
        {"code_writer": _dep(code_writer), "test_writer": _dep(_TESTS)},
        input_data=context,
    )
    assert gather["workspace"], "the workspace block must materialize"
    executor = _node("accept_executor.py", {"gather": _dep(gather)})
    judge = _node("adequacy_check.py", {"executor": _dep(executor)})
    accept_gate = _node(
        "accept_gate.py", {"executor": _dep(executor), "judge": _dep(judge)}
    )
    return _node(
        "build_gated_envelope.py",
        {
            "code_writer": _dep(code_writer),
            "executor": _dep(executor),
            "accept_gate": _dep(accept_gate),
        },
    )


class TestAnEmptyDeliverableIsNeverAClientWrite:
    """The invariant pins. Each must go RED under deletion of the guard."""

    def test_the_refix_smoke_only_route_writes_nothing(self) -> None:
        """The destructive capture, and the reason this is not defence in
        depth. refix_select substitutes a smoke test whose body is ``pass``
        when rung 1.5 found no visible test, and ``pass`` passes against any
        code, including none. One faulty node output — an empty fence from
        model_edit — and the accept gate is satisfied with nothing in hand.
        The named file is one the client already has, so the write clobbers
        it."""
        envelope = _refix_envelope("Here is the corrected file:\n\n```python\n```\n")

        assert envelope["diagnostics"]["accept"] is True, (
            "the smoke test must still be satisfied, or this pins the wrong thing"
        )
        assert envelope["artifacts"][0]["content"] == ""

        outcome = _serving_tail("re-fix", _seat_wire(envelope), "calc.py")
        chunks = _chunks(outcome)

        assert not any(isinstance(chunk, ClientToolCall) for chunk in chunks), (
            "an empty re-fix candidate clobbered the client's file"
        )
        assert "calc.py" in _text(chunks), "the refusal must name the file"

    def test_the_build_gated_workspace_satisfied_route_writes_nothing(self) -> None:
        """The second route, and a distinct mechanism: the executor's ground
        truth is satisfied by the materialized WORKSPACE rather than by the
        deliverable. _materialize shadows the target file only, so tests
        importing another workspace module pass with an empty deliverable.
        adequacy_check reads the tests alone and never sees the code, so
        tests_adequate cannot notice either."""
        envelope = _build_gated_envelope("", "fix helpers.py so restock adds one")

        assert envelope["diagnostics"]["accept"] is True
        assert envelope["artifacts"][0]["content"] == ""

        outcome = _serving_tail("build-gated", _seat_wire(envelope), "solution.py")
        chunks = _chunks(outcome)

        assert not any(isinstance(chunk, ClientToolCall) for chunk in chunks)
        assert "solution.py" in _text(chunks)

    def test_a_whitespace_only_deliverable_refuses(self) -> None:
        """Kills a ``== ""`` implementation. The rule is emptiness after
        strip, because a file of blank lines is as empty as no file."""
        chunks = _chunks({"finish": False, "file": "a.py", "content": "  \n\t\n "})

        assert not any(isinstance(chunk, ClientToolCall) for chunk in chunks)

    def test_the_refusal_mints_a_refused_ledger_entry(self) -> None:
        """The finding-3 pin. Every existing caller-minted refusal uses
        "Refused: ", whose mints is "" — and _load_emit_reject_prefixes
        filters non-minting terminals out entirely, so following that idiom
        would record NO outcome for a refused build ask, indistinguishable
        from a question. That breaks #133/#134 recap grounding on exactly
        the ask a user is most likely to follow up on.

        test_every_build_reachable_emit_terminal_mints_a_ledger_entry
        iterates emit's own TERMINALS and is structurally blind to a
        caller-side terminal, so this invariant needs its own pin.
        """
        chunks = _chunks({"finish": False, "file": "a.py", "content": ""})
        message = SimpleNamespace(
            role="assistant", content=_text(chunks), tool_calls=None
        )

        kind, reason = _reject_kind(message, _PREFIXES)

        assert kind == "refused", "the refusal minted no ledger entry"
        assert reason, "a refused entry carries the reason verbatim"

    def test_the_prefix_is_not_hardcoded(self) -> None:
        """The caller placement rests on project scripts revving
        independently of the installed caller, so the refusal cannot own a
        literal that emit.py owns. With a project whose refused terminal
        carries different wording, the refusal must use THAT wording."""
        renamed: _RejectPrefixes = (
            _RejectTerminal("Declined the build: ", "refused"),
            _RejectTerminal("Another round needed: ", "rejected_gate"),
        )

        chunks = _outcome_chunks(
            {"finish": False, "file": "a.py", "content": ""}, [], renamed
        )

        assert _text(chunks).startswith("Declined the build: ")


class TestTheGuardDoesNotRefuseHealthyBuilds:
    """Over-refusal pins. These CANNOT fail under deletion of the guard —
    deleting a refusal keeps healthy builds writing — so they are named as
    what they are rather than passed off as invariant pins.
    """

    def test_a_healthy_build_still_writes(self) -> None:
        envelope = _build_gated_envelope(
            f"```python\n{_HEALTHY}```\n", "fix inventory.py so restock adds one"
        )
        assert envelope["artifacts"][0]["content"].strip()

        outcome = _serving_tail("build-gated", _seat_wire(envelope), "inventory.py")
        chunks = _chunks(outcome)

        assert any(isinstance(chunk, ClientToolCall) for chunk in chunks)

    def test_a_one_character_deliverable_still_writes(self) -> None:
        """The rule is emptiness, not a length heuristic."""
        chunks = _chunks({"finish": False, "file": "a.py", "content": "x"})

        call = chunks[0]
        assert isinstance(call, ClientToolCall)
        assert json.loads(call.tool_calls[0].arguments)["content"] == "x"
