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
