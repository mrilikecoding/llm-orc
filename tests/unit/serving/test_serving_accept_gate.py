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
import time
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


def _executor(
    requirement: str,
    code: str,
    tests: str,
    env_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    payload = json.dumps({"requirement": requirement, "code": code, "tests": tests})
    env = {
        **os.environ,
        "LLM_ORC_ACCEPT_EXECUTOR_TIMEOUT": "3",
        **(env_overrides or {}),
    }
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


def test_aggregate_budget_bounds_a_many_hang_suite() -> None:
    # Per-test isolation can spawn up to 21 children, each hanging for the
    # full per-child timeout — an interactive serve cannot absorb that. The
    # aggregate budget caps total wall time across the suite and reports it.
    hang_five = "".join(
        f"def test_hang_{i}():\n    while True:\n        pass\n" for i in range(5)
    )
    start = time.monotonic()
    r = _executor(
        "loops forever x5",
        CORRECT,
        hang_five,
        env_overrides={
            "LLM_ORC_ACCEPT_EXECUTOR_TIMEOUT": "1",
            "LLM_ORC_ACCEPT_EXECUTOR_BUDGET": "2",
        },
    )
    elapsed = time.monotonic() - start
    assert elapsed < 10
    assert r["tests_pass"] is False
    assert "aggregate budget exhausted after 2s" in r["report"]
    assert "of 5 tests not run" in r["report"]


def test_aggregate_budget_does_not_fire_on_a_normal_suite() -> None:
    r = _executor("c * 9/5 + 32", CORRECT, REAL_TESTS)
    assert r["tests_pass"] is True
    assert "aggregate budget" not in r["report"]


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


def _gate_no_judge(
    executor_resp: dict[str, Any], gather_resp: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Drive the gate as the held round wires it: executor + gather, no judge."""
    deps: dict[str, Any] = {"executor": {"response": json.dumps(executor_resp)}}
    if gather_resp is not None:
        deps["gather"] = {"response": json.dumps(gather_resp)}
    out = subprocess.run(
        [sys.executable, str(GATE)],
        input=json.dumps({"dependencies": deps}),
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    result: dict[str, Any] = json.loads(out)
    return result


def test_gate_carries_round1_adequacy_on_a_held_round() -> None:
    """The TDD retry round (issue #100): the held path only fires when round
    1's judge passed the tests, so with no judge seat the gate carries that
    verdict deterministically — the executor stays the live ground truth."""
    g = _gate_no_judge({"tests_pass": True}, {"held": True})
    assert g["tests_adequate"] is True
    assert g["accept"] is True
    assert "carried" in g["reason"]


def test_gate_held_round_still_rejects_when_tests_fail() -> None:
    g = _gate_no_judge({"tests_pass": False}, {"held": True})
    assert g["accept"] is False
    assert g["tests_adequate"] is True
    # the carried verdict must not mask the live failure cause
    assert "tests did not pass" in g["reason"]


def test_gate_without_judge_and_not_held_rejects() -> None:
    """No judge and no held flag is a miswired shape, not a free pass."""
    g = _gate_no_judge({"tests_pass": True}, {"held": False})
    assert g["accept"] is False
    g2 = _gate_no_judge({"tests_pass": True})
    assert g2["accept"] is False


def test_gate_reads_the_verdict_through_a_sub_ensemble_judge_envelope() -> None:
    """With the judge extracted to its own ensemble (#84), the judge dep's
    response is the nested sub-ensemble result envelope, not bare model
    JSON — the gate must peel it (the escaped nesting also defeats the
    regex fallback, which silently rejected every round)."""
    judge_child = json.dumps(
        {
            "ensemble": "adequacy-judge",
            "status": "completed",
            "results": {
                "judge": {
                    "response": json.dumps(
                        {"tests_adequate": True, "reason": "real behavior checks"}
                    ),
                    "status": "success",
                }
            },
        }
    )
    payload = json.dumps(
        {
            "dependencies": {
                "executor": {"response": json.dumps({"tests_pass": True})},
                "judge": {"response": judge_child},
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
    verdict = json.loads(out)
    assert verdict["tests_adequate"] is True
    assert verdict["accept"] is True


def test_quoted_string_false_from_the_judge_does_not_pass_the_gate() -> None:
    """Small models sometimes quote booleans; bool("false") is True in
    Python, which would wave inadequate tests through the gate."""
    verdict = _gate(
        {"tests_pass": True, "report": "all passed"},
        {"tests_adequate": "false", "reason": "trivial asserts"},
    )
    assert verdict["tests_adequate"] is False
    assert verdict["accept"] is False


LEAKY_STATE_CODE = "todos = []\ndef add(item):\n    todos.append(item)\n"
LEAKY_STATE_TESTS = (
    "def test_one():\n    add('a')\n    assert len(todos) == 1\n"
    "def test_two():\n    add('a')\n    add('b')\n    assert len(todos) == 2\n"
)
LEAKY_FILE_TESTS = (
    "import os, json\n"
    "def save(t):\n    json.dump(t, open('t.json', 'w'))\n"
    "def test_writes():\n    save([1])\n    assert os.path.exists('t.json')\n"
    "def test_fresh():\n    assert not os.path.exists('t.json')\n"
)


def test_executor_isolates_module_state_across_tests() -> None:
    result = _executor("add todos", LEAKY_STATE_CODE, LEAKY_STATE_TESTS)
    assert result["tests_pass"] is True
    assert result["n_tests"] == 2


def test_executor_isolates_filesystem_across_tests() -> None:
    result = _executor("save todos", "", LEAKY_FILE_TESTS)
    assert result["tests_pass"] is True
    assert result["n_tests"] == 2


def test_executor_still_fails_genuinely_wrong_code() -> None:
    result = _executor(
        "adds",
        "def add(a, b):\n    return a - b\n",
        "def test_add():\n    assert add(1, 2) == 3\n",
    )
    assert result["tests_pass"] is False
    assert "test_add" in result["report"]
