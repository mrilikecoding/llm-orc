"""Unit tests for the deterministic adequacy check (#84).

The 2026-07-09 measurement (benchmarks/judge_adequacy, 3 x 128 samples)
showed the model judge held FAR = 0 on garbage tests but FRR 25-67% on
adequate ones, near-deterministically per fixture — and every inadequate
class is detectable statically. The gate's adequacy signal is therefore a
deterministic AST check: tests are adequate when at least one assert is
value-bearing (checks the code's output against an independent expected
value). The measurement fixtures are this checker's ground truth: it must
label every one correctly, which the model judge never did.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO = Path(__file__).resolve().parents[3]
CHECK = REPO / ".llm-orc" / "scripts" / "agentic_serving" / "adequacy_check.py"
FIXTURES = REPO / "benchmarks" / "judge_adequacy" / "fixtures.yaml"


def _check(requirement: str, code: str, tests: str) -> dict[str, Any]:
    contract = {"requirement": requirement, "code": code, "tests": tests}
    payload = json.dumps(
        {
            "input_data": requirement,
            "dependencies": {"executor": {"response": json.dumps(contract)}},
        }
    )
    out = subprocess.run(
        [sys.executable, str(CHECK)],
        input=payload,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    result: dict[str, Any] = json.loads(out)
    return result


def _fixtures() -> list[dict[str, Any]]:
    data = yaml.safe_load(FIXTURES.read_text(encoding="utf-8"))
    return list(data["fixtures"])


@pytest.mark.parametrize("fixture", _fixtures(), ids=lambda f: str(f["name"]))
def test_checker_labels_every_measurement_fixture_correctly(
    fixture: dict[str, Any],
) -> None:
    verdict = _check(fixture["requirement"], fixture["code"], fixture["tests"])
    assert verdict["tests_adequate"] is fixture["expected_adequate"], verdict["reason"]


def test_truthy_assert_on_the_function_under_test_is_value_bearing() -> None:
    """`assert is_even(4)` checks real output truthiness on a real input —
    unlike `assert callable(is_even)`, which any implementation passes."""
    verdict = _check(
        "is_even",
        "def is_even(n):\n    return n % 2 == 0",
        "def test_even():\n    assert is_even(4)",
    )
    assert verdict["tests_adequate"] is True


def test_expected_value_derived_from_the_same_call_is_tautological() -> None:
    verdict = _check(
        "clamp",
        "def clamp(n, lo, hi):\n    return max(lo, min(n, hi))",
        "def test_echo():\n"
        "    expected = clamp(15, 0, 10)\n"
        "    assert clamp(15, 0, 10) == expected",
    )
    assert verdict["tests_adequate"] is False


def test_unittest_assert_methods_are_value_bearing() -> None:
    verdict = _check(
        "add",
        "def add(a, b):\n    return a + b",
        "import unittest\n\n"
        "class TestAdd(unittest.TestCase):\n"
        "    def test_add(self):\n"
        "        self.assertEqual(add(2, 3), 5)\n",
    )
    assert verdict["tests_adequate"] is True


def test_exception_expectation_tests_are_value_bearing() -> None:
    verdict = _check(
        "remove",
        "class Store:\n"
        "    def __init__(self):\n"
        "        self.d = {}\n"
        "    def remove(self, k):\n"
        "        del self.d[k]",
        "def test_remove_missing_raises():\n"
        "    s = Store()\n"
        "    try:\n"
        "        s.remove(99)\n"
        "        assert False, 'expected KeyError'\n"
        "    except KeyError:\n"
        "        pass",
    )
    assert verdict["tests_adequate"] is True


def test_unparseable_tests_are_inadequate_not_a_crash() -> None:
    verdict = _check("f", "def f():\n    return 1", "def test_(:\n    broken")
    assert verdict["tests_adequate"] is False
    assert "parse" in verdict["reason"]


def test_verdict_shape_matches_the_judge_contract() -> None:
    """The checker fills the judge seat: same {tests_adequate, reason} JSON
    the accept gate already reads."""
    verdict = _check(
        "add",
        "def add(a, b):\n    return a + b",
        "def test_add():\n    assert add(2, 3) == 5",
    )
    assert set(verdict) == {"tests_adequate", "reason"}
    assert isinstance(verdict["reason"], str)
