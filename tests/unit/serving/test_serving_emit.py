"""Unit tests for the serving ``emit`` node (WP-A8).

emit is the terminal client-permission-seam node: a valid build deliverable
becomes a file write; a refused build degrades to a prose finish carrying the
reason (the serve never writes a form-gate-refused deliverable); a non-build
turn is a prose finish (scenarios.md "Per-Turn Serving Handler"; ADR-046 §1,
ADR-034 re-home).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
EMIT = REPO / ".llm-orc" / "scripts" / "agentic_serving" / "emit.py"


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


def test_valid_build_emits_a_file_write() -> None:
    outcome = _emit(
        {
            "build": True,
            "valid": True,
            "file": "add.py",
            "content": "def add():\n    pass",
        }
    )
    assert outcome["finish"] is False
    assert outcome["file"] == "add.py"
    assert outcome["content"] == "def add():\n    pass"


def test_refused_build_degrades_to_a_prose_finish_with_no_write() -> None:
    outcome = _emit(
        {
            "build": True,
            "valid": False,
            "file": "add.py",
            "content": "bad",
            "reason": "not valid Python",
        }
    )
    assert outcome["finish"] is True
    assert "Refused" in outcome["content"]
    assert "file" not in outcome


def test_non_build_is_a_prose_finish() -> None:
    outcome = _emit({"build": False, "valid": True, "content": "It adds two numbers."})
    assert outcome["finish"] is True
    assert "adds two numbers" in outcome["content"]


def test_rejected_accept_gate_emits_another_round_with_no_write() -> None:
    # accept=False routes another round (ODP-2: the client owns the loop); the
    # serve never writes a gate-rejected deliverable, even if it parses.
    outcome = _emit(
        {
            "build": True,
            "valid": True,
            "file": "a.py",
            "content": "def f():\n    pass",
            "accept": False,
            "accept_reason": "tests inadequate to verify the requirement",
        }
    )
    assert outcome["finish"] is True
    assert "another round" in outcome["content"].lower()
    assert "inadequate" in outcome["content"]
    assert "file" not in outcome


def test_accepted_build_emits_a_file_write() -> None:
    outcome = _emit(
        {
            "build": True,
            "valid": True,
            "file": "a.py",
            "content": "def f():\n    pass",
            "accept": True,
            "accept_reason": "tests pass and are adequate",
        }
    )
    assert outcome["finish"] is False
    assert outcome["file"] == "a.py"
