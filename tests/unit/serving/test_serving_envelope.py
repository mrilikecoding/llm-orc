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


def test_envelope_container_shape_survives_the_collapse_without_retired_subfields() -> (
    None
):
    """Preservation (scenarios.md "the common I/O envelope container shape is
    unchanged by the collapse"): a Cycle-8 seat emits the surviving ADR-024
    container shape, and ONLY the superseded ``diagnostics.calibration_verdict``
    and ``diagnostics.audit_findings`` sub-fields are absent (they went with
    their retired gates).
    """
    env = _envelope({"results": {"out": {"response": "x = 1\n"}}})
    # Surviving ADR-024 container fields.
    assert env["status"] == "success"
    for field in ("primary", "structured", "artifacts", "diagnostics"):
        assert field in env
    # The retired calibration sub-fields are gone.
    assert "calibration_verdict" not in env["diagnostics"]
    assert "audit_findings" not in env["diagnostics"]
