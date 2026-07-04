"""Unit tests for the serving ``shape`` node (WP-A8, scenario 5).

shape reads the deliverable CONTENT from the seat's ADR-024 envelope and the
DESTINATION from classify, then produces the faithful deliverable. When the seat
did not envelope (a non-build explain seat returning raw prose), shape degrades
to the raw terminal text (scenarios.md "the marshal node consumes the seat's
real common I/O envelope"; ADR-046 §1).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
SHAPE = REPO / ".llm-orc" / "scripts" / "agentic_serving" / "shape.py"


def _shape(
    classify_decision: dict[str, Any], seat_child_result: dict[str, Any]
) -> dict[str, Any]:
    payload = json.dumps(
        {
            "dependencies": {
                "classify": {"response": json.dumps(classify_decision)},
                "seat": {"response": json.dumps(seat_child_result)},
            }
        }
    )
    out = subprocess.run(
        [sys.executable, str(SHAPE)],
        input=payload,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    result: dict[str, Any] = json.loads(out)
    return result


def test_shape_reads_deliverable_from_envelope_and_destination_from_classify() -> None:
    code = "def add(a, b):\n    return a + b"
    envelope = {
        "status": "success",
        "primary": code,
        "artifacts": [
            {"content": code, "content_type": "text/x-python", "summary": "add"}
        ],
    }
    shaped = _shape(
        {"build": True, "file": "add.py", "kind": "python_module"},
        {
            "results": {
                "generate": {"response": "..."},
                "envelope": {"response": json.dumps(envelope)},
            }
        },
    )
    assert shaped["build"] is True
    assert shaped["file"] == "add.py"
    assert shaped["content"] == code


def test_shape_degrades_to_raw_seat_terminal_when_no_envelope() -> None:
    shaped = _shape(
        {"build": False, "kind": "explanation"},
        {"results": {"out": {"response": "It adds two numbers."}}},
    )
    assert shaped["build"] is False
    assert "adds two numbers" in shaped["content"]


def test_shape_carries_the_accept_verdict_from_the_build_gated_envelope() -> None:
    code = "def add(a, b):\n    return a + b"
    envelope = {
        "status": "success",
        "primary": code,
        "artifacts": [{"content": code, "content_type": "text/x-python"}],
        "diagnostics": {
            "ensemble": "build-gated",
            "accept": False,
            "accept_reason": "tests inadequate",
        },
    }
    shaped = _shape(
        {"build": True, "file": "add.py", "kind": "python_module"},
        {"results": {"envelope": {"response": json.dumps(envelope)}}},
    )
    assert shaped["content"] == code
    assert shaped["accept"] is False
    assert "inadequate" in shaped["accept_reason"]


def test_shape_has_no_verdict_for_an_ungated_seat_envelope() -> None:
    # A code-seat / explainer envelope carries no accept diagnostics -> None.
    envelope = {
        "status": "success",
        "primary": "x = 1",
        "artifacts": [{"content": "x = 1"}],
    }
    shaped = _shape(
        {"build": True, "file": "a.py"},
        {"results": {"envelope": {"response": json.dumps(envelope)}}},
    )
    assert shaped["accept"] is None
