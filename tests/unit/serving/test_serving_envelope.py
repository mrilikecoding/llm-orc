"""Unit tests for the code-seat ``emit_envelope`` node (WP-A8, scenario 5).

The envelope node extracts the code deliverable from the code-generator's
chatty synthesizer prose and emits an ADR-024 ``DispatchEnvelope`` so the
serving marshal consumes a faithful structured artifact (ADR-024; ADR-046 §2).
This is the fix for the fidelity defect real-model grounding surfaced.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
ENVELOPE = REPO / ".llm-orc" / "scripts" / "agentic_serving" / "emit_envelope.py"


def _envelope(generate_child_result: dict[str, Any]) -> dict[str, Any]:
    payload = json.dumps(
        {"dependencies": {"generate": {"response": json.dumps(generate_child_result)}}}
    )
    out = subprocess.run(
        [sys.executable, str(ENVELOPE)],
        input=payload,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    result: dict[str, Any] = json.loads(out)
    return result


def test_envelope_extracts_clean_code_from_chatty_synthesizer_prose() -> None:
    env = _envelope(
        {
            "results": {
                "synthesizer": {
                    "response": (
                        "Here's the implementation for your file:\n"
                        "```python\ndef add(a, b):\n    return a + b\n```\n"
                        "**Analysis:** looks correct."
                    )
                }
            }
        }
    )
    assert env["status"] == "success"
    deliverable = env["artifacts"][0]["content"]
    assert deliverable == "def add(a, b):\n    return a + b"
    assert env["primary"] == deliverable
    assert "Analysis" not in deliverable
    assert "```" not in deliverable


def test_envelope_passes_through_already_clean_code() -> None:
    env = _envelope({"results": {"out": {"response": "x = 1\n"}}})
    assert env["artifacts"][0]["content"] == "x = 1"
