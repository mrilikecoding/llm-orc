"""Unit tests for the build-gated ``envelope`` node (WP-D8).

The build-gated shape's terminal: it carries the code deliverable (artifacts[0])
plus the accept-gate verdict (accept / reason / signals) in the ADR-024
diagnostics, so the serving marshal consumes a faithful structured artifact and
the client sees the accept/another-round decision (ODP-2). The surviving ADR-024
container shape is preserved; the retired calibration subfields stay absent.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
ENVELOPE = REPO / ".llm-orc" / "scripts" / "agentic_serving" / "build_gated_envelope.py"


def _sub_ensemble_response(terminal_text: str) -> str:
    node = {"response": terminal_text, "status": "success"}
    return json.dumps(
        {
            "ensemble": "code-generator",
            "status": "completed",
            "results": {"synthesizer": node},
        }
    )


def _envelope(code_terminal: str, gate_verdict: dict[str, Any]) -> dict[str, Any]:
    payload = json.dumps(
        {
            "dependencies": {
                "code_writer": {"response": _sub_ensemble_response(code_terminal)},
                "accept_gate": {"response": json.dumps(gate_verdict)},
            }
        }
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


def test_envelope_carries_code_and_accept_verdict() -> None:
    env = _envelope(
        "def f():\n    return 1",
        {"accept": True, "tests_pass": True, "tests_adequate": True, "reason": "ok"},
    )
    assert env["status"] == "success"
    assert env["artifacts"][0]["content"] == "def f():\n    return 1"
    assert env["primary"] == "def f():\n    return 1"
    diag = env["diagnostics"]
    assert diag["accept"] is True
    assert diag["tests_pass"] is True
    assert diag["tests_adequate"] is True


def test_envelope_carries_reject_verdict_and_reason() -> None:
    env = _envelope(
        "def f():\n    return 1",
        {
            "accept": False,
            "tests_pass": True,
            "tests_adequate": False,
            "reason": "tests inadequate to verify the requirement",
        },
    )
    diag = env["diagnostics"]
    assert diag["accept"] is False
    assert "inadequate" in diag["accept_reason"]


def test_envelope_extracts_code_from_fenced_seat_output() -> None:
    env = _envelope(
        "Here it is:\n```python\nx = 1\n```",
        {"accept": True, "reason": "ok"},
    )
    assert env["artifacts"][0]["content"] == "x = 1"


def test_envelope_preserves_adr024_container_without_retired_subfields() -> None:
    env = _envelope("x = 1", {"accept": True, "reason": "ok"})
    for field in ("primary", "structured", "artifacts", "diagnostics"):
        assert field in env
    assert "calibration_verdict" not in env["diagnostics"]
    assert "audit_findings" not in env["diagnostics"]
