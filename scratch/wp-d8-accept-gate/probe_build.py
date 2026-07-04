#!/usr/bin/env python3
"""WP-D8 slice-3 probe: test_writer -> code_writer (build-against-tests).

Runs the 2-node probe ensemble, extracts the test-writer's tests and the
code-writer's code, and runs the tests against the code via the shipped executor.
The question: does a dependent code-generator, given {criteria + tests}, produce
code that PASSES the tests (naming coordination + tractable build-against-tests)?

Run: uv run python scratch/wp-d8-accept-gate/probe_build.py
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
PROBE = Path(__file__).with_name("probe-build.yaml")
EXECUTOR = PROJECT / "scripts" / "agentic_serving" / "accept_executor.py"

CRITERIA = "Write a function that returns the nth Fibonacci number (fib(0)=0, fib(1)=1)."


def _terminal(text: str) -> str:
    """Peel the sub-ensemble envelope layers (deliverable / output / results) to
    the terminal node's raw output, matching emit_envelope._terminal."""
    current = text
    for _ in range(6):
        try:
            obj = json.loads(current)
        except (json.JSONDecodeError, TypeError):
            return current
        if not isinstance(obj, dict):
            return current
        if isinstance(obj.get("deliverable"), str):
            current = obj["deliverable"]
            continue
        if isinstance(obj.get("output"), str):
            current = obj["output"]
            continue
        results = obj.get("results")
        if isinstance(results, dict) and results:
            node = results[list(results.keys())[-1]]
            current = node.get("response", "") if isinstance(node, dict) else str(node)
            continue
        return current
    return current


def _extract_code(raw: str) -> str:
    text = _terminal(raw)
    fences = re.findall(r"```(?:[a-zA-Z0-9_+-]+)?\n(.*?)```", text, re.DOTALL)
    if fences:
        return "\n".join(block.strip() for block in fences)
    return text.strip()


def _extract_tests(raw: str) -> str:
    text = _terminal(raw)
    fences = re.findall(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if fences:
        return "\n".join(block.strip() for block in fences)
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith(("def test_", "import ", "from ")):
            return "\n".join(lines[i:])
    return text


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
    cfg = EnsembleLoader().load_from_file(str(PROBE))
    executor = ExecutorFactory.create_root_executor(project_dir=PROJECT)

    print(f"criteria: {CRITERIA}\n")
    result = await executor.execute(cfg, CRITERIA)
    results = result.get("results", {})
    tests = _extract_tests(results.get("test_writer", {}).get("response", ""))
    code = _extract_code(results.get("code_writer", {}).get("response", ""))

    print("--- test_writer tests ---")
    print(tests)
    print("\n--- code_writer code ---")
    print(code)
    print()

    verdict = _run_executor(CRITERIA, code, tests)
    print(
        f"executor: tests_pass={verdict['tests_pass']} "
        f"n_tests={verdict['n_tests']} report={verdict['report']!r}"
    )
    print()
    if verdict["tests_pass"] and verdict["n_tests"] > 0:
        print("PROBE RESULT: PASS — code_writer built code that passes test_writer's")
        print("tests (naming coordinated, build-against-tests worked). Reuse code-generator.")
    else:
        print("PROBE RESULT: the dependent code-generator did not pass the tests")
        print("(naming mismatch or the composed input was too implicit). Fallback:")
        print("a prep node making the 'pass these tests' instruction explicit.")


if __name__ == "__main__":
    asyncio.run(main())
