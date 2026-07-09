"""Unit tests for the loop_unwrap node (issue #89, the retry-round seat).

The loop primitive wraps its body's terminal output in
``{"output": ..., "iterations": ..., "terminated": ...}``; unwrap restores
the bare ADR-024 envelope contract for the serving marshal and seat contract.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
UNWRAP = REPO / ".llm-orc" / "scripts" / "agentic_serving" / "loop_unwrap.py"


def _unwrap(round_response: str) -> dict[str, Any]:
    payload = json.dumps(
        {"input_data": "x", "dependencies": {"round": {"response": round_response}}}
    )
    out = subprocess.run(
        [sys.executable, str(UNWRAP)],
        input=payload,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    result: dict[str, Any] = json.loads(out)
    return result


def test_unwrap_restores_the_bare_envelope() -> None:
    envelope = {
        "status": "success",
        "primary": "def f(): pass",
        "artifacts": [{"content": "def f(): pass"}],
        "diagnostics": {"accept": True},
    }
    wrapper = json.dumps({"output": envelope, "iterations": 2, "terminated": "until"})
    assert _unwrap(wrapper) == envelope


def test_unwrap_degrades_to_empty_dict_on_malformed_input() -> None:
    assert _unwrap("not json") == {}
