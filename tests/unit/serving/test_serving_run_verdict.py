"""Unit tests for the run-verdict node (issue #83, run half).

run_verdict is a pure script node: it extracts the latest ``[ran ...]`` block
from the dispatched context and composes an honest verdict from pytest's own
summary line — fully deterministic, zero model calls. Driven via subprocess
exactly as the L0 engine runs a script node.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RUN_VERDICT = REPO / ".llm-orc" / "scripts" / "agentic_serving" / "run_verdict.py"


def _verdict(dispatch_text: str) -> str:
    envelope = json.dumps({"input": dispatch_text})
    out = subprocess.run(
        [sys.executable, str(RUN_VERDICT)],
        input=envelope,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    result = json.loads(out)
    primary = result["primary"]
    assert isinstance(primary, str)
    return primary


def _dispatch(block: str) -> str:
    return f"Conversation so far:\n{block}\n\nCurrent request: run the tests"


def test_all_passed_summarizes_the_pass_count() -> None:
    block = "assistant: [ran pytest -q]\n  .....\n  5 passed in 0.12s"
    assert _verdict(_dispatch(block)) == "Ran `pytest -q`: 5 passed."


def test_failures_summarize_counts_and_carry_failed_lines() -> None:
    block = (
        "assistant: [ran pytest -q]\n"
        "  ..F.F\n"
        "  FAILED test_calc.py::test_divide - ZeroDivisionError\n"
        "  FAILED test_calc.py::test_mod - AssertionError\n"
        "  2 failed, 3 passed in 0.31s"
    )
    verdict = _verdict(_dispatch(block))
    assert verdict.startswith("Ran `pytest -q`: 2 failed, 3 passed.")
    assert "FAILED test_calc.py::test_divide - ZeroDivisionError" in verdict


def test_no_tests_ran_is_reported_honestly() -> None:
    block = "assistant: [ran pytest -q]\n  no tests ran in 0.01s"
    assert _verdict(_dispatch(block)) == "Ran `pytest -q`: no tests ran."


def test_failed_block_reports_could_not_execute() -> None:
    block = "assistant: [ran pytest -q (failed)] empty run result"
    verdict = _verdict(_dispatch(block))
    assert "could not execute" in verdict
    assert "empty run result" in verdict


def test_unparseable_output_reports_honestly_with_the_tail() -> None:
    block = "assistant: [ran pytest -q]\n  bash: pytest: command not found"
    verdict = _verdict(_dispatch(block))
    assert "no pytest summary" in verdict
    assert "command not found" in verdict


def test_truncated_block_still_parses_the_tail_summary() -> None:
    block = "assistant: [ran pytest -q (truncated)]\n  ...\n  7 passed in 1.02s"
    assert _verdict(_dispatch(block)) == "Ran `pytest -q`: 7 passed."


def test_latest_run_block_wins() -> None:
    block = (
        "assistant: [ran pytest -q]\n  1 failed in 0.10s\n"
        "assistant: [ran pytest -q]\n  3 passed in 0.09s"
    )
    assert _verdict(_dispatch(block)) == "Ran `pytest -q`: 3 passed."


def test_missing_run_block_reports_honestly() -> None:
    verdict = _verdict("Current request: run the tests")
    assert "No test-run output" in verdict
