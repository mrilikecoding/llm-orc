"""Unit tests for the serving accept gate (WP-D8): sandboxed executor + AND gate.

The accept gate is the verification-ladder rung above form-gate: does the produced
code RUN and pass its tests (deterministic executor, sandboxed in a subprocess per
ODP-1 so a runaway test cannot hang the serve), and does an isolated judge find the
tests adequate. The gate ANDs the two (ADR-048 §1; scenarios.md "accept = tests_pass
AND tests_adequate"). Executor and gate are deterministic and hermetic here; the
judge is the ensemble's only model call and is fed as a fixed verdict.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
EXECUTOR = REPO / ".llm-orc" / "scripts" / "agentic_serving" / "accept_executor.py"
GATE = REPO / ".llm-orc" / "scripts" / "agentic_serving" / "accept_gate.py"

CORRECT = "def celsius_to_fahrenheit(c):\n    return c * 9 / 5 + 32\n"
WRONG = "def celsius_to_fahrenheit(c):\n    return c * 9 / 5\n"  # missing + 32
REAL_TESTS = (
    "def test_freezing():\n    assert celsius_to_fahrenheit(0) == 32\n"
    "def test_boiling():\n    assert celsius_to_fahrenheit(100) == 212\n"
)
TRIVIAL_TESTS = "def test_callable():\n    assert callable(celsius_to_fahrenheit)\n"


def _executor(requirement: str, code: str, tests: str) -> dict[str, Any]:
    payload = json.dumps({"requirement": requirement, "code": code, "tests": tests})
    env = {**os.environ, "LLM_ORC_ACCEPT_EXECUTOR_TIMEOUT": "3"}
    out = subprocess.run(
        [sys.executable, str(EXECUTOR)],
        input=payload,
        capture_output=True,
        text=True,
        check=True,
        env=env,
    ).stdout
    result: dict[str, Any] = json.loads(out)
    return result


def _gate(executor_resp: dict[str, Any], judge_resp: dict[str, Any]) -> dict[str, Any]:
    payload = json.dumps(
        {
            "dependencies": {
                "executor": {"response": json.dumps(executor_resp)},
                "judge": {"response": json.dumps(judge_resp)},
            }
        }
    )
    out = subprocess.run(
        [sys.executable, str(GATE)],
        input=payload,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    result: dict[str, Any] = json.loads(out)
    return result


# --- executor: deterministic ground truth, sandboxed in a subprocess ---


def test_executor_passes_correct_code_against_real_tests() -> None:
    r = _executor("c * 9/5 + 32", CORRECT, REAL_TESTS)
    assert r["tests_pass"] is True
    assert r["n_tests"] == 2


def test_executor_fails_wrong_code() -> None:
    r = _executor("c * 9/5 + 32", WRONG, REAL_TESTS)
    assert r["tests_pass"] is False


def test_executor_is_fooled_by_trivial_tests() -> None:
    # The executor passes trivial tests — that is the point; the judge catches it.
    r = _executor("c * 9/5 + 32", CORRECT, TRIVIAL_TESTS)
    assert r["tests_pass"] is True


def test_executor_sandbox_times_out_on_a_runaway_test() -> None:
    # ODP-1: the executor runs the produced tests in a subprocess with a timeout,
    # so a runaway test is reported as a failure, not a frozen serve.
    runaway = "def test_hang():\n    while True:\n        pass\n"
    r = _executor("loops forever", CORRECT, runaway)
    assert r["tests_pass"] is False
    assert "timeout" in r["report"].lower()


def test_executor_passes_the_contract_through_for_the_judge() -> None:
    # The judge seat receives {requirement, code, tests} + execution result via
    # the executor's passthrough (isolation: contract + artifact + result only).
    r = _executor("the requirement", CORRECT, REAL_TESTS)
    assert r["requirement"] == "the requirement"
    assert r["code"] == CORRECT
    assert r["tests"] == REAL_TESTS


# --- gate: accept = tests_pass AND tests_adequate (orthogonal catches) ---


def test_gate_accepts_when_both_signals_green() -> None:
    g = _gate({"tests_pass": True}, {"tests_adequate": True})
    assert g["accept"] is True


def test_gate_rejects_when_judge_inadequate() -> None:
    g = _gate({"tests_pass": True}, {"tests_adequate": False})
    assert g["accept"] is False
    assert g["tests_pass"] is True
    assert g["tests_adequate"] is False


def test_gate_rejects_when_executor_fails() -> None:
    g = _gate({"tests_pass": False}, {"tests_adequate": True})
    assert g["accept"] is False


def test_quoted_string_false_from_the_judge_does_not_pass_the_gate() -> None:
    """Small models sometimes quote booleans; bool("false") is True in
    Python, which would wave inadequate tests through the gate."""
    verdict = _gate(
        {"tests_pass": True, "report": "all passed"},
        {"tests_adequate": "false", "reason": "trivial asserts"},
    )
    assert verdict["tests_adequate"] is False
    assert verdict["accept"] is False
