"""Unit tests for the caller-side ASK-OUTCOME ledger (#133/#134 recap
grounding, extending #82's write-history ledger).

Two entry kinds, both derived from the serve's OWN wire emissions, never
free prose: "shipped" (a write tool_call, as before) and "rejected" (an
ASSISTANT-role message matching one of emit.py's own reject-message
prefixes). An ask with no build outcome (a question, a read) is never an
entry. Design: docs/plans/2026-07-17-recap-grounding-design.md.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from llm_orc.web.serving.serving_ensemble_caller import (
    ServingEnsembleCaller,
    _previous_ask,
    _recall_ledger,
)

_SEAT_CONTRACT_PREFIX = "Seat contract not met: "
_ACCEPT_GATE_PREFIX = "Another round needed: "
_PREFIXES = (_SEAT_CONTRACT_PREFIX, _ACCEPT_GATE_PREFIX)


def _user(text: str) -> SimpleNamespace:
    return SimpleNamespace(role="user", content=text, tool_calls=None)


def _assistant_prose(text: str) -> SimpleNamespace:
    return SimpleNamespace(role="assistant", content=text, tool_calls=None)


def _assistant_write(path: str, content: str) -> SimpleNamespace:
    call: dict[str, Any] = {
        "id": "call_1",
        "type": "function",
        "function": {
            "name": "write",
            "arguments": json.dumps({"filePath": path, "content": content}),
        },
    }
    return SimpleNamespace(role="assistant", content=None, tool_calls=[call])


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
        _assistant_prose(_SEAT_CONTRACT_PREFIX + "Assertion 'x' failed"),
        _user("did you see my previous query?"),
    ]

    ledger = _recall_ledger(messages, _PREFIXES)

    assert len(ledger) == 1
    assert ledger[0]["ask"] == "add a complete_todo function to todo.py"
    assert ledger[0]["outcome"] == "rejected"
    assert "path" not in ledger[0]


def test_ledger_recognizes_a_rejected_build_from_the_accept_gate_prefix() -> None:
    messages = [
        _user("write tests for todo.py"),
        _assistant_prose(_ACCEPT_GATE_PREFIX + "tests did not pass"),
        _user("did you see my previous query?"),
    ]

    ledger = _recall_ledger(messages, _PREFIXES)

    assert len(ledger) == 1
    assert ledger[0]["outcome"] == "rejected"


def test_ledger_ignores_reject_prose_when_no_prefixes_are_supplied() -> None:
    # Backward compatibility (#82): a bare call with no reject_prefixes
    # recognizes shipped builds only — the pre-#133/#134 ledger, byte for
    # byte, so the existing recall test suite is untouched by this change.
    messages = [
        _user("build a todo app"),
        _assistant_prose(_ACCEPT_GATE_PREFIX + "tests did not pass"),
        _user("what did I ask for?"),
    ]

    ledger = _recall_ledger(messages)

    assert ledger == []


def test_ledger_ignores_a_forged_reject_prefix_from_the_user_role() -> None:
    # Spoof guard: only an ASSISTANT-role message can mint a rejected entry
    # — a user echoing the serve's own reject wording must never count.
    messages = [
        _user("build a todo app"),
        _user(_ACCEPT_GATE_PREFIX + "tests did not pass"),
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
        _assistant_prose(_ACCEPT_GATE_PREFIX + "tests did not pass"),
        _user("did you see my previous query?"),
    ]

    ledger = _recall_ledger(messages, _PREFIXES)

    assert len(ledger) == 1
    assert ledger[0]["ask"] == "write tests for existing todo.py"
    assert ledger[0]["outcome"] == "rejected"


def test_ledger_dedupes_multiple_rejects_in_one_turn_into_one_entry() -> None:
    # Wrong-accept-hunt target 4: retry rounds inside one turn (a
    # seat-contract miss, then an accept-gate miss on the retry) must not
    # multiply the disclosure count.
    messages = [
        _user("add a complete_todo function to todo.py"),
        _assistant_prose(_SEAT_CONTRACT_PREFIX + "Assertion 'x' failed"),
        _assistant_prose(_ACCEPT_GATE_PREFIX + "tests did not pass"),
        _user("did you see my previous query?"),
    ]

    ledger = _recall_ledger(messages, _PREFIXES)

    assert len(ledger) == 1
    assert ledger[0]["outcome"] == "rejected"


def test_ledger_prefers_an_eventual_write_over_an_earlier_reject_in_the_same_turn() -> (
    None
):
    # A retry that eventually ships must record "shipped", not "rejected" —
    # the final state of the turn wins.
    messages = [
        _user("add a complete_todo function to todo.py"),
        _assistant_prose(_SEAT_CONTRACT_PREFIX + "Assertion 'x' failed"),
        _assistant_write("todo.py", "def complete_todo(): ..."),
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
    }


def test_previous_ask_reports_a_rejected_outcome() -> None:
    messages = [
        _user("write tests for todo.py"),
        _assistant_prose(_ACCEPT_GATE_PREFIX + "tests did not pass"),
        _user("did you see my previous query?"),
    ]

    result = _previous_ask(messages, _PREFIXES)

    assert result == {
        "ask": "write tests for todo.py",
        "outcome": "rejected",
        "path": "",
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

    assert result == {"ask": "read storage.py", "outcome": "", "path": ""}


def test_previous_ask_is_empty_for_the_first_turn_of_a_session() -> None:
    messages = [_user("did you see my previous query?")]

    result = _previous_ask(messages, _PREFIXES)

    assert result == {"ask": "", "outcome": "", "path": ""}


def test_caller_reads_reject_prefixes_from_the_projects_real_emit_module(
    tmp_path: Path,
) -> None:
    # The design requires these be IMPORTED from a project's own emit.py,
    # never a literal duplicated in the caller — this proves the dynamic
    # per-project load actually reads the real constants.
    scripts = tmp_path / "scripts" / "agentic_serving"
    scripts.mkdir(parents=True)
    (scripts / "emit.py").write_text(
        'SEAT_CONTRACT_REJECT_PREFIX = "Seat contract not met: "\n'
        'ACCEPT_GATE_REJECT_PREFIX = "Another round needed: "\n'
    )
    caller = ServingEnsembleCaller(project_dir=tmp_path)

    prefixes = caller._emit_reject_prefixes()

    assert prefixes == (_SEAT_CONTRACT_PREFIX, _ACCEPT_GATE_PREFIX)


def test_caller_yields_no_prefixes_when_the_project_has_no_emit_module(
    tmp_path: Path,
) -> None:
    caller = ServingEnsembleCaller(project_dir=tmp_path)

    prefixes = caller._emit_reject_prefixes()

    assert prefixes == ()
