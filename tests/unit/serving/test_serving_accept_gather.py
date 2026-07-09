"""Unit tests for the build-gated ``gather`` node + executor dependency-mode (WP-D8).

``gather`` assembles the accept-gate's ``{requirement, code, tests}`` from the two
sub-ensemble seats (test_writer, code_writer), peeling the nested ensemble
envelopes and stripping code fences. The executor then reads that assembled
contract from its ``gather`` dependency (its flat direct-payload mode, exercised
in test_serving_accept_gate.py, is preserved).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
GATHER = REPO / ".llm-orc" / "scripts" / "agentic_serving" / "accept_gather.py"
EXECUTOR = REPO / ".llm-orc" / "scripts" / "agentic_serving" / "accept_executor.py"

TESTS = "def test_even():\n    assert is_even(4) is True"
CODE = "def is_even(n):\n    return n % 2 == 0"


def _sub_ensemble_response(terminal_text: str) -> str:
    """A sub-ensemble node's response: the nested ``{ensemble, results}`` envelope
    the engine returns for an ``ensemble:`` node (peeled by _terminal)."""
    return json.dumps(
        {
            "ensemble": "x",
            "status": "completed",
            "results": {"out": {"response": terminal_text, "status": "success"}},
        }
    )


def _gather(criteria: str, tests_terminal: str, code_terminal: str) -> dict[str, Any]:
    payload = json.dumps(
        {
            "input_data": criteria,
            "dependencies": {
                "test_writer": {"response": _sub_ensemble_response(tests_terminal)},
                "code_writer": {"response": _sub_ensemble_response(code_terminal)},
            },
        }
    )
    out = subprocess.run(
        [sys.executable, str(GATHER)],
        input=payload,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    result: dict[str, Any] = json.loads(out)
    return result


def _executor_from_gather(gather_out: dict[str, Any]) -> dict[str, Any]:
    payload = json.dumps(
        {
            "input_data": gather_out.get("requirement", ""),
            "dependencies": {"gather": {"response": json.dumps(gather_out)}},
        }
    )
    out = subprocess.run(
        [sys.executable, str(EXECUTOR)],
        input=payload,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    result: dict[str, Any] = json.loads(out)
    return result


def test_gather_assembles_requirement_code_tests_from_seats() -> None:
    out = _gather("Write is_even(n).", TESTS, CODE)
    assert out["requirement"] == "Write is_even(n)."
    assert out["code"] == CODE
    assert out["tests"] == TESTS


def test_gather_strips_markdown_fences_from_seat_output() -> None:
    fenced_code = "Here you go:\n```python\n" + CODE + "\n```\n"
    fenced_tests = "```python\n" + TESTS + "\n```"
    out = _gather("Write is_even(n).", fenced_tests, fenced_code)
    assert out["code"] == CODE
    assert out["tests"] == TESTS


def test_executor_reads_contract_from_gather_dependency() -> None:
    gathered = _gather("Write is_even(n).", TESTS, CODE)
    result = _executor_from_gather(gathered)
    assert result["tests_pass"] is True
    assert result["n_tests"] == 1
    # passthrough carries the contract + artifact to the judge
    assert result["code"] == CODE
    assert result["tests"] == TESTS


def test_gather_strips_conversation_context_from_the_requirement() -> None:
    """With rung-1 context threading, the shape's base input carries the
    conversation ahead of the 'Current request:' marker. The requirement the
    verifier chain echoes (and the judge reads) is the clean turn only —
    conversation context must not reach verifier seats (ADR-048 isolation).
    """
    criteria = (
        "Conversation so far:\n"
        "user: write is_even in even.py\nassistant: [wrote even.py]"
        "\n\nCurrent request: Write is_even(n)."
    )
    out = _gather(criteria, TESTS, CODE)
    assert out["requirement"] == "Write is_even(n)."
