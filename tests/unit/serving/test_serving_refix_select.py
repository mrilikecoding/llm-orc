"""Unit tests for the re-fix select node (rung 2, convergent-fix design).

refix_select picks the candidate (deterministic edit, else the model edit's
extracted code) and hands {code, tests} to the accept executor. When rung 1.5
found no visible test to re-gate against, select injects a minimal smoke test
so the executor still verifies the candidate at least LOADS cleanly before it
can ship — a re-fix must never clobber the original with a candidate that
parses but fails to import (F3, merge-gate review). Driven via subprocess
exactly as the L0 engine runs a script node.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
REFIX_SELECT = REPO / ".llm-orc" / "scripts" / "agentic_serving" / "refix_select.py"


def _select(gather: dict[str, Any]) -> dict[str, Any]:
    envelope = json.dumps(
        {"input_data": "", "dependencies": {"gather": {"response": json.dumps(gather)}}}
    )
    out = subprocess.run(
        [sys.executable, str(REFIX_SELECT)],
        input=envelope,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    result: dict[str, Any] = json.loads(out)
    return result


def test_deterministic_candidate_wins_and_carries_the_visible_test() -> None:
    selected = _select(
        {
            "deterministic_code": "def f(): return 1\n",
            "visible_test": "def test_f(): assert f() == 1\n",
            "task": "fix f in f.py",
        }
    )
    assert selected["code"] == "def f(): return 1\n"
    assert selected["tests"] == "def test_f(): assert f() == 1\n"
    assert selected["edit_kind"] == "deterministic"
    assert selected["smoke_only"] is False


def test_no_visible_test_injects_a_smoke_test_and_flags_smoke_only() -> None:
    selected = _select(
        {
            "deterministic_code": "def f(): return 1\n",
            "visible_test": "",
            "task": "fix f in f.py",
        }
    )
    # a real test function so the executor does not short-circuit to
    # "no tests found" — the runner execs the CODE before it, so a
    # non-loading candidate fails this smoke gate
    assert "def test_" in selected["tests"]
    assert selected["smoke_only"] is True


def test_visible_test_is_not_smoke_only() -> None:
    selected = _select(
        {
            "deterministic_code": "def f(): return 1\n",
            "visible_test": "def test_f(): assert f() == 1\n",
            "task": "fix f in f.py",
        }
    )
    assert selected["smoke_only"] is False
