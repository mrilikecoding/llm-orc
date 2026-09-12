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

import pytest

REPO = Path(__file__).resolve().parents[3]
REFIX_SELECT = REPO / ".llm-orc" / "scripts" / "agentic_serving" / "refix_select.py"
EXECUTOR = REPO / ".llm-orc" / "scripts" / "agentic_serving" / "accept_executor.py"


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


def _select_with_model_edit(prior_code: str, model_edit: str) -> dict[str, Any]:
    """Same wiring as ``_select``, plus a ``model_edit`` dep — the shape a
    smoke-only round with no deterministic edit actually takes."""
    gather = {
        "deterministic_code": "",
        "visible_test": "",
        "prior_code": prior_code,
        "task": "fix restock in calc.py",
    }
    payload = json.dumps(
        {
            "dependencies": {
                "gather": {"response": json.dumps(gather)},
                "model_edit": {"response": model_edit},
            }
        }
    )
    out = subprocess.run(
        [sys.executable, str(REFIX_SELECT)],
        input=payload,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    result: dict[str, Any] = json.loads(out)
    return result


def _executor_result(selected: dict[str, Any]) -> dict[str, Any]:
    payload = json.dumps(
        {"dependencies": {"select": {"response": json.dumps(selected)}}}
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


# --- #171: the smoke test is surface-derived, not a bare `pass` ------------
#
# The old `_SMOKE_TEST` was `def ...(): pass` — it references no name, so on
# the smoke-only path (no visible test) participation was unsatisfiable and
# any content that merely loaded shipped, clobbering the original (post-#173:
# `x = 1` / `import os` / an unrelated def all still accepted). The smoke
# test is now derived from the PRIOR module's top-level names (`prior_code`,
# already carried by refix_gather): a candidate that drops the surface fails.

_PRIOR_WITH_SURFACE = "def restock(item, n):\n    return n\n"


def test_smoke_test_is_derived_from_the_prior_modules_surface() -> None:
    selected = _select(
        {
            "deterministic_code": "",
            "visible_test": "",
            "prior_code": _PRIOR_WITH_SURFACE,
            "task": "fix restock in calc.py",
        }
    )
    assert "restock" in selected["tests"]


def test_missing_prior_code_still_injects_a_loads_cleanly_smoke_test() -> None:
    """No prior_code known (or nothing to preserve) makes ``_smoke_test``
    emit the "loads cleanly" bar alone — an ``import solution`` assertion
    with no per-name checks, rather than crashing or emitting an
    unsatisfiable test — the existing smoke-only pins above never supply
    prior_code. This pins only the TEXT this function emits; the
    downstream ACCEPT decision for a surface-less prior is refix_envelope's
    F-1 fallback (pre-#171 bar: load cleanly plus #173's inertness), pinned
    in test_serving_refix_envelope.py, not here."""
    selected = _select(
        {
            "deterministic_code": "def f(): return 1\n",
            "visible_test": "",
            "task": "fix f in f.py",
        }
    )
    assert "def test_" in selected["tests"]
    assert "import solution" in selected["tests"]


def test_smoke_surface_empty_flag_is_true_with_no_prior_surface() -> None:
    """M3 (review): refix_envelope needs to tell "no surface to check"
    apart from "these tests happen not to exercise the deliverable" so it
    can give the surface-less sub-path its own actionable reason instead of
    the generic participation constant."""
    selected = _select(
        {
            "deterministic_code": "def f(): return 1\n",
            "visible_test": "",
            "task": "fix f in f.py",
        }
    )
    assert selected["smoke_surface_empty"] is True


def test_smoke_surface_empty_flag_is_false_with_a_real_prior_surface() -> None:
    selected = _select(
        {
            "deterministic_code": "",
            "visible_test": "",
            "prior_code": _PRIOR_WITH_SURFACE,
            "task": "fix restock in calc.py",
        }
    )
    assert selected["smoke_surface_empty"] is False


def test_smoke_surface_empty_flag_is_false_with_a_visible_test() -> None:
    """The flag is scoped to the smoke-only path — a real visible test is
    never "surface-less" in this sense, it just isn't the smoke test."""
    selected = _select(
        {
            "deterministic_code": "def f(): return 1\n",
            "visible_test": "def test_f(): assert f() == 1\n",
            "task": "fix f in f.py",
        }
    )
    assert selected["smoke_surface_empty"] is False


@pytest.mark.parametrize(
    ("label", "candidate"),
    [
        ("empty", ""),
        ("assignment-only", "x = 1\n"),
        ("comment-only", "# nope\n"),
        ("import-only", "import os\n"),
        ("unrelated-def", "def other():\n    return 1\n"),
    ],
)
def test_junk_candidates_fail_the_surface_derived_smoke_test(
    label: str, candidate: str
) -> None:
    selected = _select_with_model_edit(_PRIOR_WITH_SURFACE, candidate)
    result = _executor_result(selected)

    assert result["tests_pass"] is False, f"{label}: {result['report']}"


def test_a_real_fix_passes_the_surface_derived_smoke_test() -> None:
    selected = _select_with_model_edit(
        _PRIOR_WITH_SURFACE, "def restock(item, n):\n    return n + 1\n"
    )
    result = _executor_result(selected)

    assert result["tests_pass"] is True, result["report"]


def test_a_fix_that_drops_a_public_name_refuses() -> None:
    """Recorded bound (#171): the smoke test is a surface check, not a
    semantic diff — a fix that intentionally drops a name the prior module
    exported refuses, whichever was the right call to make."""
    prior = "def restock(item, n):\n    return n\ndef audit(item):\n    return item\n"
    selected = _select_with_model_edit(
        prior, "def restock(item, n):\n    return n + 1\n"
    )
    result = _executor_result(selected)

    assert result["tests_pass"] is False, result["report"]


def test_a_fix_that_removes_a_private_helper_still_accepts() -> None:
    """F-2 (regression, #171 round 2 review): the smoke surface used to be
    EVERY top-level def/class name, underscore-prefixed ones included, so a
    legitimate re-fix that inlines or deletes a private helper was refused
    for dropping a name nothing public ever promised. The surface must be
    derived from PUBLIC names only (no leading underscore, dunders
    excluded)."""
    prior = (
        "def _round2(v):\n"
        "    return round(v, 2)\n\n\n"
        "def discount(price, pct):\n"
        "    return _round2(price - price * pct / 100)\n"
    )
    candidate = (
        "def discount(price, pct):\n    return round(price - price * pct / 100, 2)\n"
    )
    selected = _select_with_model_edit(prior, candidate)
    result = _executor_result(selected)

    assert result["tests_pass"] is True, result["report"]
