"""Deterministic routing unit tests for the serving ``classify`` node (WP-A8).

``classify`` is a pure script node: it reads the turn and emits the routing
decision ``{target, kind, file, dispatch_input, build}`` the dispatch seat
resolves (scenarios.md "classify emits a structural routing decision
deterministically"; ADR-046 §1, where classify owns the build-vs-non-build
executable-deliverable determination). Driven via subprocess exactly as the L0
engine runs a script node.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
CLASSIFY = REPO / ".llm-orc" / "scripts" / "agentic_serving" / "classify.py"


def _classify(turn: dict[str, Any]) -> dict[str, Any]:
    envelope = json.dumps({"input": json.dumps(turn)})
    out = subprocess.run(
        [sys.executable, str(CLASSIFY)],
        input=envelope,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    result: dict[str, Any] = json.loads(out)
    return result


def test_build_turn_routes_to_the_code_generation_seat() -> None:
    decision = _classify({"task": "write a function that adds two numbers"})
    assert decision["target"] == "code-seat"
    assert decision["build"] is True


def test_filename_is_extracted_from_the_turn_text() -> None:
    decision = _classify({"task": "write a function that adds two numbers in add.py"})
    assert decision["file"] == "add.py"
    assert decision["build"] is True


def test_explain_turn_is_non_build_and_routes_away_from_the_code_seat() -> None:
    decision = _classify({"task": "explain what foo.py does"})
    assert decision["build"] is False
    assert decision["target"] != "code-generator"


def test_a_named_target_file_is_carried_structurally() -> None:
    decision = _classify({"task": "build a cli", "file": "cli.py"})
    assert decision["file"] == "cli.py"
    assert decision["build"] is True
