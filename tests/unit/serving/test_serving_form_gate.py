"""Unit tests for the serving ``form_gate`` node (WP-A8).

The form-gate is the cheapest verification-ladder rung: a build deliverable must
parse as what its destination path claims, else it is refused before the client
sees it (scenarios.md "form-gate refuses a deliverable that does not parse";
ADR-046 §1, ADR-035 re-home).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
FORM_GATE = REPO / ".llm-orc" / "scripts" / "agentic_serving" / "form_gate.py"


def _gate(shaped: dict[str, Any]) -> dict[str, Any]:
    payload = json.dumps({"dependencies": {"shape": {"response": json.dumps(shaped)}}})
    out = subprocess.run(
        [sys.executable, str(FORM_GATE)],
        input=payload,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    result: dict[str, Any] = json.loads(out)
    return result


def test_valid_python_deliverable_passes() -> None:
    gated = _gate(
        {"build": True, "file": "add.py", "content": "def add(a, b):\n    return a + b"}
    )
    assert gated["valid"] is True
    assert gated["file"] == "add.py"


def test_unparseable_python_deliverable_is_refused() -> None:
    gated = _gate(
        {"build": True, "file": "add.py", "content": "Here's the code: def add("}
    )
    assert gated["valid"] is False
    assert "add.py" in gated["reason"]


def test_non_build_turn_is_inert() -> None:
    gated = _gate({"build": False, "file": "n/a", "content": "prose"})
    assert gated["valid"] is True


def test_form_gate_passes_the_accept_verdict_through() -> None:
    gated = _gate(
        {
            "build": True,
            "file": "a.py",
            "content": "x = 1",
            "accept": False,
            "accept_reason": "tests inadequate",
        }
    )
    assert gated["accept"] is False
    assert gated["accept_reason"] == "tests inadequate"


def test_form_gate_passes_read_fields_through() -> None:
    gated = _gate(
        {
            "build": False,
            "file": "test_storage.py",
            "content": "Requesting client files.",
            "needs_files": ["storage.py"],
            "read_failed": "",
            "accept": None,
            "accept_reason": "",
            "seat_admitted": None,
            "seat_contract_reason": "",
        }
    )
    assert gated["needs_files"] == ["storage.py"]
    assert gated["read_failed"] == ""
    assert gated["valid"] is True


def test_form_gate_passes_needs_run_through() -> None:
    gated = _gate(
        {
            "build": False,
            "file": "solution.py",
            "content": "Requesting a client test run.",
            "needs_files": [],
            "read_failed": "",
            "needs_run": "pytest -q",
            "accept": None,
            "accept_reason": "",
            "seat_admitted": None,
            "seat_contract_reason": "",
        }
    )
    assert gated["needs_run"] == "pytest -q"
    assert gated["valid"] is True


def test_form_gate_passes_glob_fields_through() -> None:
    gated = _gate(
        {
            "build": False,
            "file": "solution.py",
            "content": "Requesting a workspace listing.",
            "needs_files": [],
            "read_failed": "",
            "needs_run": "",
            "needs_glob": "storage",
            "glob_failed": "",
            "accept": None,
            "accept_reason": "",
            "seat_admitted": None,
            "seat_contract_reason": "",
        }
    )
    assert gated["needs_glob"] == "storage"
    assert gated["glob_failed"] == ""
    assert gated["valid"] is True


def test_form_gate_passes_not_grounded_through() -> None:
    gated = _gate(
        {
            "build": False,
            "file": "solution.py",
            "content": "Not grounded in this session.",
            "needs_files": [],
            "read_failed": "",
            "not_grounded": "todo.py",
            "accept": None,
            "accept_reason": "",
            "seat_admitted": None,
            "seat_contract_reason": "",
        }
    )
    assert gated["not_grounded"] == "todo.py"
    assert gated["valid"] is True


# --- phantom-symbol backstop (#133/#134 §4, defense in depth):
# docs/plans/2026-07-17-recap-grounding-design.md ---


def test_memory_shaped_content_with_a_grounded_claim_passes_unchanged() -> None:
    gated = _gate(
        {
            "build": False,
            "file": "solution.py",
            "content": "So far you've shipped `todo.py`.",
            "memory_shaped": True,
            "grounded_text": "todo.py\ndef add_todo(): ...",
            "ledger_recap": "Shipped so far: `todo.py`.",
        }
    )
    assert gated["content"] == "So far you've shipped `todo.py`."


def test_memory_shaped_content_with_a_phantom_claim_fails_closed_to_the_recap() -> None:
    gated = _gate(
        {
            "build": False,
            "file": "solution.py",
            "content": "You've built `todo.py` and a `complete_todo` function.",
            "memory_shaped": True,
            "grounded_text": "todo.py\ndef add_todo(): ...",
            "ledger_recap": "Shipped so far: `todo.py`.",
        }
    )
    assert gated["content"] == "Shipped so far: `todo.py`."


def test_non_memory_shaped_content_is_never_backstopped() -> None:
    # Never concept or named-file explains — the backstop is scoped narrowly
    # to defer_recall (memory_shaped) turns only.
    gated = _gate(
        {
            "build": False,
            "file": "solution.py",
            "content": "It uses a `PhantomHelper` class internally.",
            "memory_shaped": False,
            "grounded_text": "",
            "ledger_recap": "",
        }
    )
    assert gated["content"] == "It uses a `PhantomHelper` class internally."


def test_build_turn_is_never_backstopped() -> None:
    gated = _gate(
        {
            "build": True,
            "file": "add.py",
            "content": "def add(a, b):\n    return a + b",
            "memory_shaped": True,
            "grounded_text": "",
            "ledger_recap": "Nothing has been built in this session yet.",
        }
    )
    assert gated["content"] == "def add(a, b):\n    return a + b"


def test_backstop_never_treats_the_recaps_own_backticks_as_a_phantom_claim() -> None:
    # Wrong-accept-hunt target 6: if content already reads exactly like our
    # OWN recap template (all its backtick-quoted claims are legitimately
    # grounded — they came from the ledger), the backstop must not mangle
    # it further or treat it as a phantom claim needing replacement.
    gated = _gate(
        {
            "build": False,
            "file": "solution.py",
            "content": "Shipped so far: `todo.py`, `storage.py`.",
            "memory_shaped": True,
            "grounded_text": "todo.py\nstorage.py",
            "ledger_recap": "Shipped so far: `todo.py`, `storage.py`.",
        }
    )
    assert gated["content"] == "Shipped so far: `todo.py`, `storage.py`."
