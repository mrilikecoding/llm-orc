#!/usr/bin/env python3
"""Deterministic test-adequacy check — the gate's adequacy signal (#84).

Replaces the model adequacy judge in the gated build round. The 2026-07-09
measurement (benchmarks/judge_adequacy, 3 prompt variants x 128 samples)
showed the model seat never false-accepted garbage tests but false-rejected
25-67% of adequate ones, near-deterministically per fixture — and every
inadequate class has a static signature. The deterministic rule:

    Tests are ADEQUATE when at least one assert is VALUE-BEARING — it
    checks the code's output against an independent expected value.

Value-bearing forms: a comparison containing a real call whose two sides
are not the same expression (directly or via a variable assigned from the
same call — the tautology channel); a truthy assert that IS a real call
(``assert is_even(4)``); unittest assert methods over a real call; an
exception-expectation try/except around a real call. Reflection calls
(callable/hasattr/isinstance/type/...) are never value-bearing.

Emits the judge seat's contract: {"tests_adequate": bool, "reason": str} —
the accept gate reads it unchanged. Deterministic control per the standing
constraint: model judgment only where a closed check cannot decide.
"""

from __future__ import annotations

import ast
import json
import sys
from typing import Any

from _helpers import deps as _deps
from _helpers import payload as _payload
from _helpers import response as _response

# Reflection/introspection calls prove existence, not behavior.
_EXCLUDED_CALLS = {
    "callable",
    "hasattr",
    "isinstance",
    "type",
    "getattr",
    "dir",
    "id",
    "vars",
}


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _real_calls(node: ast.AST) -> list[ast.Call]:
    """Calls that exercise behavior (reflection calls excluded)."""
    return [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.Call) and _call_name(n) not in _EXCLUDED_CALLS
    ]


def _signature(node: ast.AST, assigned: dict[str, str]) -> str:
    """Structural identity of a compared side; a bare name resolves to the
    call it was assigned from (the ``expected = f(x); assert f(x) == expected``
    tautology channel)."""
    if isinstance(node, ast.Name) and node.id in assigned:
        return assigned[node.id]
    return ast.dump(node)


def _compare_is_value_bearing(node: ast.Compare, assigned: dict[str, str]) -> bool:
    if not _real_calls(node):
        return False
    if len(node.comparators) == 1:
        return _signature(node.left, assigned) != _signature(
            node.comparators[0], assigned
        )
    return True


def _unittest_assert_is_value_bearing(node: ast.Call, assigned: dict[str, str]) -> bool:
    name = _call_name(node)
    if not name.startswith("assert"):
        return False
    if name.startswith("assertRaises"):
        return True
    args = node.args
    if len(args) >= 2:
        if not any(_real_calls(arg) for arg in args[:2]):
            return False
        return _signature(args[0], assigned) != _signature(args[1], assigned)
    if len(args) == 1:
        return bool(_real_calls(args[0]))
    return False


def _assigned_call_signatures(unit: ast.AST) -> dict[str, str]:
    assigned: dict[str, str] = {}
    for node in ast.walk(unit):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigned[target.id] = ast.dump(node.value)
    return assigned


def _has_failure_signal(statements: list[ast.stmt]) -> bool:
    """A failure marker inside an expect-raise try body: ``assert False``,
    a bare ``raise``, or ``self.fail()`` — without one, a try/except-pass
    swallows everything and passes any implementation (a wrong-accept
    channel, PR #102 review)."""
    for stmt in statements:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Assert):
                test = node.test
                if isinstance(test, ast.Constant) and test.value is False:
                    return True
            elif isinstance(node, ast.Raise):
                return True
            elif isinstance(node, ast.Call) and _call_name(node) == "fail":
                return True
    return False


def _expect_raise_with(node: ast.With | ast.AsyncWith) -> bool:
    """``with self.assertRaises(...):`` / ``with pytest.raises(...):`` — the
    canonical exception-expectation idioms."""
    for item in node.items:
        expr = item.context_expr
        if isinstance(expr, ast.Call):
            name = _call_name(expr)
            if name.startswith("assertRaises") or name == "raises":
                return True
    return False


def _value_bearing_asserts(unit: ast.AST) -> int:
    """Value-bearing asserts within one test unit (function or class)."""
    assigned = _assigned_call_signatures(unit)
    count = 0
    for node in ast.walk(unit):
        if isinstance(node, ast.Assert):
            test = node.test
            if isinstance(test, ast.Compare):
                if _compare_is_value_bearing(test, assigned):
                    count += 1
            elif isinstance(test, ast.Call) and _call_name(test) not in _EXCLUDED_CALLS:
                count += 1
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            if _unittest_assert_is_value_bearing(node.value, assigned):
                count += 1
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            if _expect_raise_with(node) and any(
                _real_calls(stmt) for stmt in node.body
            ):
                count += 1
        elif isinstance(node, ast.Try):
            specific = any(handler.type is not None for handler in node.handlers)
            if (
                specific
                and any(_real_calls(stmt) for stmt in node.body)
                and _has_failure_signal(node.body)
            ):
                count += 1
    return count


def _is_test_class(node: ast.ClassDef) -> bool:
    """Match the runner's dialect: Test*-named classes OR any
    unittest.TestCase subclass regardless of name."""
    if node.name.startswith("Test"):
        return True
    for base in node.bases:
        base_name = base.attr if isinstance(base, ast.Attribute) else ""
        if isinstance(base, ast.Name):
            base_name = base.id
        if "TestCase" in base_name:
            return True
    return False


def _test_units(tree: ast.Module) -> list[ast.AST]:
    units: list[ast.AST] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test_"):
                units.append(node)
        elif isinstance(node, ast.ClassDef) and _is_test_class(node):
            units.append(node)
    return units


def check_tests(tests: str) -> dict[str, Any]:
    try:
        tree = ast.parse(tests)
    except SyntaxError as error:
        return {
            "tests_adequate": False,
            "reason": f"tests do not parse: {error.msg}",
        }
    units = _test_units(tree)
    if not units:
        return {
            "tests_adequate": False,
            "reason": "no test functions or TestCase classes found",
        }
    total = sum(_value_bearing_asserts(unit) for unit in units)
    if total == 0:
        return {
            "tests_adequate": False,
            "reason": (
                "no value-bearing asserts: the tests never check the code's "
                "output against an expected value"
            ),
        }
    return {
        "tests_adequate": True,
        "reason": f"{total} value-bearing assert(s) check real outputs",
    }


def _contract(payload: dict[str, Any]) -> dict[str, Any]:
    """The executor's echoed {requirement, code, tests} from the dependency
    envelope (flat direct payload as fallback, mirroring the executor)."""
    for dep in _deps(payload).values():
        try:
            parsed = json.loads(_response(dep))
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict) and "tests" in parsed:
            return parsed
    return payload


def main() -> None:
    payload = _payload(sys.stdin.read().strip())
    contract = _contract(payload)
    print(json.dumps(check_tests(str(contract.get("tests", "")))))


if __name__ == "__main__":
    main()
