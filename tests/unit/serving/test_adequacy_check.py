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
EXECUTOR = REPO / ".llm-orc" / "scripts" / "agentic_serving" / "accept_executor.py"
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


def test_mutation_pattern_assert_is_value_bearing() -> None:
    """The spike's turn1_s5 r1 false-reject (2026-07-10): a None-returning
    mutator is verified by passing a local to the call and comparing that
    local to an independent literal afterwards. For a task shaped 'add an
    item to a list', mutation style is the natural correct test."""
    verdict = _check(
        "add_todo",
        "def add_todo(todos, item):\n    todos.append(item)",
        'def test_add():\n    todos = []\n    add_todo(todos, "x")\n'
        '    assert todos == ["x"]',
    )
    assert verdict["tests_adequate"] is True


def test_mutation_pattern_subscript_assert_is_value_bearing() -> None:
    verdict = _check(
        "add_todo",
        "def add_todo(todos, item):\n    todos.append(item)",
        'def test_add():\n    todos = []\n    add_todo(todos, "x")\n'
        '    assert todos[0] == "x"',
    )
    assert verdict["tests_adequate"] is True


def test_compare_of_two_untouched_names_is_not_value_bearing() -> None:
    """A compare with no call anywhere in the body stays inadequate — the
    mutation rule needs the name to have been passed to a real call first."""
    verdict = _check(
        "add_todo",
        "def add_todo(todos, item):\n    todos.append(item)",
        'def test_add():\n    todos = ["x"]\n    assert todos == ["x"]',
    )
    assert verdict["tests_adequate"] is False


def test_mutation_compare_against_a_non_literal_is_not_value_bearing() -> None:
    """Comparing the mutated name to another plain name is not an
    independent expected value — only literals/containers count."""
    verdict = _check(
        "add_todo",
        "def add_todo(todos, item):\n    todos.append(item)",
        "def test_add():\n    todos = []\n    expected = todos\n"
        '    add_todo(todos, "x")\n    assert todos == expected',
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


def test_loose_try_except_pass_is_not_value_bearing() -> None:
    """``try: f(2) / except Exception: pass`` asserts nothing and passes any
    implementation — counting it adequate is a wrong-accept channel (review
    finding, PR #102). An exception expectation needs a failure signal after
    the call (assert False / raise / self.fail())."""
    verdict = _check(
        "f",
        "def f(n):\n    return n",
        "def test_swallows_everything():\n"
        "    try:\n"
        "        f(2)\n"
        "    except Exception:\n"
        "        pass",
    )
    assert verdict["tests_adequate"] is False


def test_with_assert_raises_is_value_bearing() -> None:
    """The canonical unittest raises idiom must count."""
    verdict = _check(
        "remove",
        "class Store:\n    def remove(self, k):\n        raise KeyError(k)",
        "import unittest\n\n"
        "class TestStore(unittest.TestCase):\n"
        "    def test_remove_missing(self):\n"
        "        with self.assertRaises(KeyError):\n"
        "            Store().remove(99)\n",
    )
    assert verdict["tests_adequate"] is True


def test_with_pytest_raises_is_value_bearing() -> None:
    verdict = _check(
        "remove",
        "class Store:\n    def remove(self, k):\n        raise KeyError(k)",
        "import pytest\n\n"
        "def test_remove_missing():\n"
        "    with pytest.raises(KeyError):\n"
        "        Store().remove(99)\n",
    )
    assert verdict["tests_adequate"] is True


def test_testcase_subclass_without_test_prefix_is_a_test_unit() -> None:
    """The runner collects ANY unittest.TestCase subclass; the checker must
    see the same dialect or it rejects suites the executor happily runs."""
    verdict = _check(
        "add",
        "def add(a, b):\n    return a + b",
        "import unittest\n\n"
        "class MyStoreTests(unittest.TestCase):\n"
        "    def test_add(self):\n"
        "        self.assertEqual(add(2, 3), 5)\n",
    )
    assert verdict["tests_adequate"] is True


def test_unparseable_tests_are_inadequate_not_a_crash() -> None:
    verdict = _check("f", "def f():\n    return 1", "def test_(:\n    broken")
    assert verdict["tests_adequate"] is False
    assert "parse" in verdict["reason"]


def test_adequacy_judges_the_repaired_suite_not_the_raw_one() -> None:
    """Ordering (round-2 repairs, 2026-07-10): the judge reads the
    executor's ECHOED contract, so repairs run before the adequacy check.
    The inverted exception-expectation suite is inadequate raw (no failure
    signal in the try body) but adequate once the executor rewrites it to
    pytest.raises — the repaired suite is what gets judged."""
    code = (
        "def save_todos(todos):\n"
        "    if todos is None:\n"
        "        raise TypeError('todos must be a list')\n"
    )
    tests = (
        "def test_save_todos_handles_exceptions():\n"
        "    try:\n"
        "        save_todos(None)\n"
        "    except TypeError:\n"
        '        assert False, "Expected TypeError"\n'
        "    else:\n"
        '        assert False, "Did not raise TypeError"\n'
    )
    raw_verdict = _check("save todos", code, tests)
    assert raw_verdict["tests_adequate"] is False

    executor_out = subprocess.run(
        [sys.executable, str(EXECUTOR)],
        input=json.dumps({"requirement": "save todos", "code": code, "tests": tests}),
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    payload = json.dumps(
        {
            "input_data": "save todos",
            "dependencies": {"executor": {"response": executor_out}},
        }
    )
    out = subprocess.run(
        [sys.executable, str(CHECK)],
        input=payload,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    chained: dict[str, Any] = json.loads(out)
    assert chained["tests_adequate"] is True


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
