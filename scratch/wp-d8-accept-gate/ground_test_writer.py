#!/usr/bin/env python3
"""WP-D8 slice 2 grounding — the test-writer seat with a real model.

Grounds the tests-first half of the gated build shape at the earliest runnable
point (standing BUILD directive, no vacuum): run the real test-writer seat on a
criteria, extract its tests, then run them against a CORRECT and a WRONG
implementation via the shipped accept_executor. Discriminating tests pass the
correct impl and fail the wrong one — that is what makes the executor a real
ground-truth signal downstream.

Run: uv run python scratch/wp-d8-accept-gate/ground_test_writer.py
"""

import asyncio
import json
import re
import subprocess
import sys
from pathlib import Path

from llm_orc.core.config.ensemble_config import EnsembleLoader
from llm_orc.core.execution.executor_factory import ExecutorFactory

PROJECT = Path("/Users/nathangreen/Development/eddi-lab/llm-orc/.llm-orc")
TEST_WRITER = PROJECT / "ensembles" / "agentic-serving" / "test-writer.yaml"
EXECUTOR = PROJECT / "scripts" / "agentic_serving" / "accept_executor.py"

CRITERIA = "Write is_even(n): return True if n is even, False otherwise."
CORRECT = "def is_even(n):\n    return n % 2 == 0\n"
WRONG = "def is_even(n):\n    return n % 2 == 1\n"  # inverted


def _extract_tests(raw: str) -> str:
    """Pull the test code out of the seat's output (strip fences / prose)."""
    fences = re.findall(r"```(?:python)?\n(.*?)```", raw, re.DOTALL)
    if fences:
        return "\n".join(block.strip() for block in fences)
    lines = raw.splitlines()
    for i, line in enumerate(lines):
        if line.startswith(("def test_", "import ", "from ")):
            return "\n".join(lines[i:])
    return raw


def _run_executor(requirement: str, code: str, tests: str) -> dict:
    payload = json.dumps({"requirement": requirement, "code": code, "tests": tests})
    out = subprocess.run(
        [sys.executable, str(EXECUTOR)],
        input=payload,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return json.loads(out)


async def main() -> None:
    cfg = EnsembleLoader().load_from_file(str(TEST_WRITER))
    executor = ExecutorFactory.create_root_executor(project_dir=PROJECT)

    print(f"criteria: {CRITERIA}\n")
    result = await executor.execute(cfg, CRITERIA)
    raw = result.get("results", {}).get("test_writer", {}).get("response", "")
    tests = _extract_tests(raw)
    print("--- test-writer produced ---")
    print(tests)
    print()

    on_correct = _run_executor(CRITERIA, CORRECT, tests)
    on_wrong = _run_executor(CRITERIA, WRONG, tests)
    print(
        f"vs CORRECT impl: tests_pass={on_correct['tests_pass']} "
        f"n_tests={on_correct['n_tests']} report={on_correct['report']!r}"
    )
    print(
        f"vs WRONG impl:   tests_pass={on_wrong['tests_pass']} "
        f"n_tests={on_wrong['n_tests']} report={on_wrong['report']!r}"
    )
    print()
    discriminates = on_correct["tests_pass"] is True and on_wrong["tests_pass"] is False
    print(
        "RESULT:",
        "DISCRIMINATES — the generated tests pass correct code and fail wrong code"
        if discriminates
        else "*** does not discriminate — the tests are weak or misnamed ***",
    )


if __name__ == "__main__":
    asyncio.run(main())
