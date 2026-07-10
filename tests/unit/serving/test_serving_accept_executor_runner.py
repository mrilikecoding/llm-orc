"""Single-test mode for the accept-gate runner (per-test isolation,
seat-quality design 2026-07-09). Driven via subprocess exactly as
accept_executor.py invokes it."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
RUNNER = REPO / ".llm-orc" / "scripts" / "agentic_serving" / "accept_executor_runner.py"

CODE = "todos = []\ndef add(item):\n    todos.append(item)\n"
LEAKY_TESTS = (
    "def test_one():\n    add('a')\n    assert len(todos) == 1\n"
    "def test_two():\n    add('a')\n    add('b')\n    assert len(todos) == 2\n"
)


def _run(code: str, tests: str, tmp: Path, only: str | None = None) -> dict[str, Any]:
    (tmp / "solution.py").write_text(code)
    (tmp / "tests.py").write_text(tests)
    argv = [
        sys.executable,
        str(RUNNER),
        str(tmp / "solution.py"),
        str(tmp / "tests.py"),
    ]
    if only is not None:
        argv += ["--only", only]
    out = subprocess.run(argv, capture_output=True, text=True, cwd=tmp, check=True)
    result: dict[str, Any] = json.loads(out.stdout)
    return result


def test_only_runs_exactly_the_named_test(tmp_path: Path) -> None:
    verdict = _run(CODE, LEAKY_TESTS, tmp_path, only="test_two")
    assert verdict["n_tests"] == 1
    assert verdict["tests_pass"] is True  # fresh namespace: no leaked 'a'


def test_only_unknown_name_reports_zero_tests(tmp_path: Path) -> None:
    verdict = _run(CODE, LEAKY_TESTS, tmp_path, only="test_missing")
    assert verdict["n_tests"] == 0
    assert verdict["tests_pass"] is False


def test_no_flag_keeps_legacy_whole_run(tmp_path: Path) -> None:
    verdict = _run(CODE, LEAKY_TESTS, tmp_path)
    assert verdict["n_tests"] == 2
    assert verdict["tests_pass"] is False  # the shared-state leak, as today
