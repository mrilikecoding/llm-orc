"""Unit tests for the caller-side ASK-OUTCOME ledger (#133/#134 recap
grounding, extending #82's write-history ledger; review round 1 blocker 2
adds the outcome-kind vocabulary and the "refused" minting class).

Four entry kinds, all derived from the serve's OWN wire emissions, never
free prose: "shipped" (a CONFIRMED write tool_call), "rejected_contract"
(the seat's own output contract), "rejected_gate" (the accept gate), and
"refused" (read-failed/glob-failed/build-invalid — never attributed to a
specific gate the record doesn't support). An ask with no build outcome
(a question, a read) is never an entry. Design: docs/plans/2026-07-17-
recap-grounding-design.md, amended 2026-08-12 (review round 1).
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
EMIT = SCRIPTS / "emit.py"

sys.path.insert(0, str(SCRIPTS))
from emit import (  # type: ignore  # noqa: E402
    ACCEPT_GATE_REJECT_PREFIX,
    BUILD_REFUSED_PREFIX,
    REFUSED_PREFIX,
    SEAT_CONTRACT_REJECT_PREFIX,
    TERMINALS,
)

from llm_orc.web.serving.serving_ensemble_caller import (  # noqa: E402
    ServingEnsembleCaller,
    _previous_ask,
    _recall_ledger,
    _reject_kind,
    _RejectPrefixes,
    _RejectTerminal,
)

# Review round 3 minor 1: derived by iterating emit's own TERMINALS registry
# — never a hand-built parallel mapping — the same discipline the caller's
# own `_load_emit_reject_prefixes` now uses. Review round 2 new blocker 2:
# a terminal whose `mints` is empty (the plain "Refused:" prefix — a
# read/glob refusal on a turn with no build signal) never mints a
# build-outcome entry, so it is filtered out here exactly as it is there.
_PREFIXES = tuple(
    _RejectTerminal(terminal.prefix, terminal.mints)
    for terminal in TERMINALS.values()
    if terminal.mints
)


def _user(text: str) -> SimpleNamespace:
    return SimpleNamespace(role="user", content=text, tool_calls=None)


def _assistant_prose(text: str) -> SimpleNamespace:
    return SimpleNamespace(role="assistant", content=text, tool_calls=None)


def _assistant_write(
    path: str, content: str, call_id: str = "call_1"
) -> SimpleNamespace:
    call: dict[str, Any] = {
        "id": call_id,
        "type": "function",
        "function": {
            "name": "write",
            "arguments": json.dumps({"filePath": path, "content": content}),
        },
    }
    return SimpleNamespace(role="assistant", content=None, tool_calls=[call])


def _tool_write_result(text: str, call_id: str = "call_1") -> SimpleNamespace:
    return SimpleNamespace(
        role="tool", content=text, tool_call_id=call_id, tool_calls=None
    )


def _assistant_read_request(path: str) -> SimpleNamespace:
    call: dict[str, Any] = {
        "id": "call_read",
        "type": "function",
        "function": {
            "name": "read",
            "arguments": json.dumps({"filePath": path}),
        },
    }
    return SimpleNamespace(role="assistant", content=None, tool_calls=[call])


def _tool_result(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        role="tool", content=text, tool_call_id="call_read", tool_calls=None
    )


def test_ledger_recognizes_a_rejected_build_from_the_seat_contract_prefix() -> None:
    messages = [
        _user("add a complete_todo function to todo.py"),
        _assistant_prose(SEAT_CONTRACT_REJECT_PREFIX + "Assertion 'x' failed"),
        _user("did you see my previous query?"),
    ]

    ledger = _recall_ledger(messages, _PREFIXES)

    assert len(ledger) == 1
    assert ledger[0]["ask"] == "add a complete_todo function to todo.py"
    assert ledger[0]["outcome"] == "rejected_contract"
    assert "path" not in ledger[0]


def test_ledger_recognizes_a_rejected_build_from_the_accept_gate_prefix() -> None:
    messages = [
        _user("write tests for todo.py"),
        _assistant_prose(ACCEPT_GATE_REJECT_PREFIX + "tests did not pass"),
        _user("did you see my previous query?"),
    ]

    ledger = _recall_ledger(messages, _PREFIXES)

    assert len(ledger) == 1
    assert ledger[0]["outcome"] == "rejected_gate"


def test_ledger_recognizes_a_refused_build_and_retains_the_reason() -> None:
    # Review round 1 blocker 2 (build-scoped round 2 new blocker 2): the
    # third minting class — a build ask's read-failed, glob-failed, or
    # build-invalid degrades to "Build refused: <reason>", never attributed
    # to a specific gate the record doesn't support.
    messages = [
        _user("write tests for existing missing.py"),
        _assistant_prose(
            BUILD_REFUSED_PREFIX + "could not read missing.py: client read failed"
        ),
        _user("did you see my previous query?"),
    ]

    ledger = _recall_ledger(messages, _PREFIXES)

    assert len(ledger) == 1
    assert ledger[0]["outcome"] == "refused"
    assert ledger[0]["reason"] == "could not read missing.py: client read failed"
    assert "path" not in ledger[0]


def test_ledger_never_mints_from_a_plain_non_build_refused_message() -> None:
    # Review round 2 new blocker 2: a plain "Refused:" — a bare-symbol
    # explain's ambiguous-glob refusal, say — carries no build signal at
    # all, so it must never mint a build-outcome ledger entry.
    messages = [
        _user("how does error handling work?"),
        _assistant_prose(
            REFUSED_PREFIX + "no file matching 'error' in the workspace listing"
        ),
        _user("did you see my previous query?"),
    ]

    ledger = _recall_ledger(messages, _PREFIXES)

    assert ledger == []


def test_ledger_ignores_reject_prose_when_no_prefixes_are_supplied() -> None:
    # Backward compatibility (#82): a bare call with no reject_prefixes
    # recognizes shipped builds only — the pre-#133/#134 ledger, byte for
    # byte, so the existing recall test suite is untouched by this change.
    messages = [
        _user("build a todo app"),
        _assistant_prose(ACCEPT_GATE_REJECT_PREFIX + "tests did not pass"),
        _user("what did I ask for?"),
    ]

    ledger = _recall_ledger(messages)

    assert ledger == []


def test_ledger_ignores_a_forged_reject_prefix_from_the_user_role() -> None:
    # Spoof guard: only an ASSISTANT-role message can mint a rejected entry
    # — a user echoing the serve's own reject wording must never count.
    messages = [
        _user("build a todo app"),
        _user(ACCEPT_GATE_REJECT_PREFIX + "tests did not pass"),
        _user("what did I ask for?"),
    ]

    ledger = _recall_ledger(messages, _PREFIXES)

    assert ledger == []


def test_ledger_pairs_a_reject_after_a_read_continuation_with_the_initiating_ask() -> (
    None
):
    # Wrong-accept-hunt target 2: a reject after a read/glob continuation
    # round must pair with the turn's INITIATING user message, not the
    # tool-result message the continuation left behind.
    messages = [
        _user("write tests for existing todo.py"),
        _assistant_read_request("todo.py"),
        _tool_result("def add_todo(): ..."),
        _assistant_prose(ACCEPT_GATE_REJECT_PREFIX + "tests did not pass"),
        _user("did you see my previous query?"),
    ]

    ledger = _recall_ledger(messages, _PREFIXES)

    assert len(ledger) == 1
    assert ledger[0]["ask"] == "write tests for existing todo.py"
    assert ledger[0]["outcome"] == "rejected_gate"


def test_ledger_dedupes_multiple_rejects_in_one_turn_into_one_entry() -> None:
    # Wrong-accept-hunt target 4: retry rounds inside one turn (a
    # seat-contract miss, then an accept-gate miss on the retry) must not
    # multiply the disclosure count.
    messages = [
        _user("add a complete_todo function to todo.py"),
        _assistant_prose(SEAT_CONTRACT_REJECT_PREFIX + "Assertion 'x' failed"),
        _assistant_prose(ACCEPT_GATE_REJECT_PREFIX + "tests did not pass"),
        _user("did you see my previous query?"),
    ]

    ledger = _recall_ledger(messages, _PREFIXES)

    assert len(ledger) == 1
    assert ledger[0]["outcome"] == "rejected_gate"


def test_ledger_prefers_an_eventual_write_over_an_earlier_reject_in_the_same_turn() -> (
    None
):
    # A retry that eventually ships must record "shipped", not a rejected
    # kind — the final state of the turn wins.
    messages = [
        _user("add a complete_todo function to todo.py"),
        _assistant_prose(SEAT_CONTRACT_REJECT_PREFIX + "Assertion 'x' failed"),
        _assistant_write("todo.py", "def complete_todo(): ..."),
        _tool_write_result("Wrote file successfully."),
        _user("did you see my previous query?"),
    ]

    ledger = _recall_ledger(messages, _PREFIXES)

    assert len(ledger) == 1
    assert ledger[0]["outcome"] == "shipped"
    assert ledger[0]["path"] == "todo.py"


def test_ledger_still_lists_shipped_builds_unaffected_by_reject_prefixes() -> None:
    messages = [
        _user("build a todo app"),
        _assistant_write("todo.py", "def add_item(): ..."),
        _user("build a calculator"),
        _assistant_write("calc.py", "def add(a, b): ..."),
        _user("what did I ask for?"),
    ]

    ledger = _recall_ledger(messages, _PREFIXES)

    assert [(entry["ask"], entry["outcome"], entry["path"]) for entry in ledger] == [
        ("build a todo app", "shipped", "todo.py"),
        ("build a calculator", "shipped", "calc.py"),
    ]


def test_ledger_does_not_ship_a_write_whose_result_is_failure_shaped() -> None:
    # Minor 1: a failed client write ("Error: permission denied") must not
    # mint "shipped" — reuses the same failure-shape check the fix-chain
    # path already trusts.
    messages = [
        _user("build a todo app"),
        _assistant_write("todo.py", "def add_item(): ..."),
        _tool_write_result("Error: permission denied"),
        _user("what did I ask for?"),
    ]

    ledger = _recall_ledger(messages, _PREFIXES)

    assert ledger == []


def test_ledger_ships_a_write_with_no_tool_result_present() -> None:
    # Backward compat: a hand-built fixture (or any wire never carrying an
    # explicit tool-result message) defaults to shipped — unchanged #82
    # behavior, only an EXPLICIT failure disqualifies.
    messages = [
        _user("build a todo app"),
        _assistant_write("todo.py", "def add_item(): ..."),
        _user("what did I ask for?"),
    ]

    ledger = _recall_ledger(messages, _PREFIXES)

    assert ledger == [
        {"ask": "build a todo app", "outcome": "shipped", "index": 0, "path": "todo.py"}
    ]


def test_ledger_truncates_a_long_ask_with_a_marker_within_the_cap() -> None:
    # Minor 2: a truncated ask must never present as verbatim.
    messages = [
        _user("build a todo app " + "x" * 1000),
        _assistant_write("todo.py", "def add_item(): ..."),
        _user("what was the first thing I asked?"),
    ]

    ledger = _recall_ledger(messages, _PREFIXES)

    ask = ledger[0]["ask"]
    assert len(ask) <= 200
    assert ask.startswith("build a todo app")
    assert ask.endswith("...")


def test_previous_ask_reports_a_shipped_outcome() -> None:
    messages = [
        _user("build a todo app"),
        _assistant_write("todo.py", "def add_item(): ..."),
        _user("did you see my previous query?"),
    ]

    result = _previous_ask(messages, _PREFIXES)

    assert result == {
        "ask": "build a todo app",
        "outcome": "shipped",
        "path": "todo.py",
        "reason": "",
    }


def test_previous_ask_reports_a_rejected_gate_outcome() -> None:
    messages = [
        _user("write tests for todo.py"),
        _assistant_prose(ACCEPT_GATE_REJECT_PREFIX + "tests did not pass"),
        _user("did you see my previous query?"),
    ]

    result = _previous_ask(messages, _PREFIXES)

    assert result == {
        "ask": "write tests for todo.py",
        "outcome": "rejected_gate",
        "path": "",
        "reason": "",
    }


def test_previous_ask_reports_a_refused_outcome_with_its_reason() -> None:
    messages = [
        _user("write tests for existing missing.py"),
        _assistant_prose(
            BUILD_REFUSED_PREFIX + "could not read missing.py: client read failed"
        ),
        _user("did you see my previous query?"),
    ]

    result = _previous_ask(messages, _PREFIXES)

    assert result == {
        "ask": "write tests for existing missing.py",
        "outcome": "refused",
        "path": "",
        "reason": "could not read missing.py: client read failed",
    }


def test_previous_ask_reports_no_outcome_for_a_read_or_question() -> None:
    # Wrong-accept-hunt target 5: "did you read/see FILE?" — the PREVIOUS
    # ask was about a read, not a build, so it must not claim one.
    messages = [
        _user("read storage.py"),
        _assistant_read_request("storage.py"),
        _tool_result("def save_todos(): ..."),
        _user("did you read that file?"),
    ]

    result = _previous_ask(messages, _PREFIXES)

    assert result == {"ask": "read storage.py", "outcome": "", "path": "", "reason": ""}


def test_previous_ask_is_empty_for_the_first_turn_of_a_session() -> None:
    messages = [_user("did you see my previous query?")]

    result = _previous_ask(messages, _PREFIXES)

    assert result == {"ask": "", "outcome": "", "path": "", "reason": ""}


def test_caller_reads_reject_prefixes_from_the_projects_real_emit_module(
    tmp_path: Path,
) -> None:
    # The design requires these be IMPORTED from a project's own emit.py,
    # never a literal duplicated in the caller — this proves the dynamic
    # per-project load actually reads the real TERMINALS registry (round 3
    # minor 1: iterated, not read as three individually-named constants).
    scripts = tmp_path / "scripts" / "agentic_serving"
    scripts.mkdir(parents=True)
    (scripts / "emit.py").write_text(
        "from typing import NamedTuple\n\n"
        "class Terminal(NamedTuple):\n"
        "    prefix: str\n"
        "    mints: str\n\n"
        "TERMINALS = {\n"
        '    "seat_contract": Terminal(\n'
        '        "Seat contract not met: ", "rejected_contract"\n'
        "    ),\n"
        '    "accept_gate": Terminal("Another round needed: ", "rejected_gate"),\n'
        '    "build_refused": Terminal("Build refused: ", "refused"),\n'
        '    "refused": Terminal("Refused: ", ""),\n'
        "}\n"
    )
    caller = ServingEnsembleCaller(project_dir=tmp_path)

    prefixes = caller._emit_reject_prefixes()

    assert prefixes == _PREFIXES


def test_caller_yields_no_prefixes_when_the_project_has_no_emit_module(
    tmp_path: Path,
) -> None:
    caller = ServingEnsembleCaller(project_dir=tmp_path)

    prefixes = caller._emit_reject_prefixes()

    assert prefixes == _RejectPrefixes()


# --- blocker 2b: every emit.py terminal reachable from a BUILD ask either
# ships (a real write tool_call, caught by the caller's OWN write
# detection — never emit prose) or mints a recognized reject/refuse kind.
# Runs the REAL emit.py subprocess for each shape, so a future emit
# terminal that mints nothing fails HERE, not silently in production. ---


def _emit(gated: dict[str, Any]) -> dict[str, Any]:
    payload = json.dumps(
        {"dependencies": {"form_gate": {"response": json.dumps(gated)}}}
    )
    out = subprocess.run(
        [sys.executable, str(EMIT)],
        input=payload,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    result: dict[str, Any] = json.loads(out)
    return result


def test_every_registry_terminal_agrees_with_the_ledgers_recognition() -> None:
    """Review round 2 major 3: the invariant test iterates the REGISTRY
    itself (``TERMINALS``, imported from emit.py) — never a hand-maintained
    parallel list, whose "every terminal declares a minting class" claim
    was false (nothing forced a new terminal onto that list too). For each
    registry entry, a synthetic message carrying that PREFIX must be
    recognized by ``_reject_kind`` as producing EXACTLY that minting class
    — including "" for the plain "refused" terminal, which must never mint.
    A newly added TERMINALS entry with a wrong (or missing) ``mints``
    declaration fails HERE, not silently in production.
    """
    for name, terminal in TERMINALS.items():
        message = SimpleNamespace(
            role="assistant", content=f"{terminal.prefix}some reason", tool_calls=None
        )
        kind, _ = _reject_kind(message, _PREFIXES)
        assert kind == terminal.mints, (
            f"TERMINALS[{name!r}] declares mints={terminal.mints!r} but "
            f"_reject_kind returned {kind!r} for a {terminal.prefix!r} message"
        )


# Wrong-accept-hunt-style integration check (blocker 2b, round 1): the
# REAL emit.py subprocess, for the concrete build-ask-reachable shapes, in
# case a future main() edit stops actually rendering from TERMINALS.
_BUILD_ASK_REJECT_SHAPES: list[dict[str, Any]] = [
    # seat contract not met
    {
        "build": True,
        "valid": True,
        "seat_admitted": False,
        "seat_contract_reason": "bad envelope",
    },
    # accept gate rejected
    {
        "build": True,
        "valid": True,
        "file": "a.py",
        "content": "x = 1",
        "accept": False,
        "accept_reason": "tests failed",
    },
    # read failed (a build ask needing to read an existing file first)
    {
        "build": False,
        "file": "x.py",
        "content": "n/a",
        "valid": True,
        "read_failed": "client read failed",
        "is_build_ask": True,
    },
    # glob failed (a build ask's discovery round found no/many candidates)
    {
        "build": False,
        "file": "x.py",
        "content": "n/a",
        "valid": True,
        "glob_failed": "no file matching 'x' in the workspace listing",
        "is_build_ask": True,
    },
    # build invalid (form-gate parse failure)
    {
        "build": True,
        "valid": False,
        "file": "a.py",
        "content": "bad",
        "reason": "not valid Python",
    },
    # #155: the seat-side gate died on a turn that would otherwise ship.
    # Build-reachable and minting, so it belongs in this list — review
    # found it missing, which is exactly what this list exists to catch.
    {
        "build": True,
        "valid": True,
        "file": "a.py",
        "content": "x = 1",
        "seat_gate_failed": "the seat contract node returned unreadable output",
    },
]


def test_every_build_reachable_emit_terminal_mints_a_ledger_entry() -> None:
    for gated in _BUILD_ASK_REJECT_SHAPES:
        outcome = _emit(gated)
        message = SimpleNamespace(
            role="assistant", content=outcome["content"], tool_calls=None
        )
        kind, _ = _reject_kind(message, _PREFIXES)
        assert kind, (
            f"emit terminal {gated} minted no ledger kind: {outcome['content']!r}"
        )


def test_non_build_ask_read_and_glob_refusals_never_mint() -> None:
    # Review round 2 new blocker 2's own invariant, via the real subprocess:
    # is_build_ask absent/False must never mint, even for the SAME
    # read_failed/glob_failed reasons a build ask would mint for.
    # `valid` on both: without it, #155's positive recognition refuses these
    # as a PIPELINE error, which also does not mint — so they would pass
    # while testing a different path than the one they name.
    shapes = [
        {
            "build": False,
            "valid": True,
            "file": "x.py",
            "content": "n/a",
            "read_failed": "client read failed",
        },
        {
            "build": False,
            "valid": True,
            "file": "x.py",
            "content": "n/a",
            "glob_failed": "no file matching 'x' in the workspace listing",
        },
    ]
    for gated in shapes:
        outcome = _emit(gated)
        # Guard against passing for the wrong reason: these must be the
        # read/glob refusals, not a pipeline error.
        assert "pipeline error" not in outcome["content"], outcome["content"]
        message = SimpleNamespace(
            role="assistant", content=outcome["content"], tool_calls=None
        )
        kind, _ = _reject_kind(message, _PREFIXES)
        assert kind == "", (
            f"non-build refusal {gated} unexpectedly minted {kind!r}: "
            f"{outcome['content']!r}"
        )
