#!/usr/bin/env python3
"""Q2 grounding spike — deterministic executor seat.

Reads {requirement, code, tests} from the ensemble input, runs the tests
against the code in-process (no pytest dependency), and emits the artifact
plus the deterministic pass/fail result. This is the builder-INDEPENDENT
ground-truth half of the accept gate: execution is execution, the builder
cannot argue a failing test into passing.

Emits JSON: {requirement, code, tests, tests_pass, n_tests, report}

Note (spike scope): tests run in-process on trusted fixtures. A shipped
version would sandbox execution; that is a BUILD concern, not this probe's.
"""

import json
import sys


def run_tests(code: str, tests: str) -> tuple[bool, str, int]:
    """Exec code + tests in a shared namespace, call every test_* function."""
    ns: dict[str, object] = {}
    try:
        exec(compile(code, "solution.py", "exec"), ns)
    except Exception as exc:  # noqa: BLE001 - spike surface
        return False, f"code failed to load: {exc!r}", 0
    try:
        exec(compile(tests, "test_solution.py", "exec"), ns)
    except Exception as exc:  # noqa: BLE001
        return False, f"tests failed to load: {exc!r}", 0

    test_fns = [
        (name, fn)
        for name, fn in ns.items()
        if name.startswith("test_") and callable(fn)
    ]
    if not test_fns:
        return False, "no test_* functions found", 0

    failures: list[str] = []
    for name, fn in test_fns:
        try:
            fn()  # type: ignore[operator]
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: {exc!r}")

    if failures:
        return False, "; ".join(failures), len(test_fns)
    return True, "all passed", len(test_fns)


def _payload(envelope: dict[str, object]) -> dict[str, object]:
    """Extract the task payload across llm-orc's envelope formats.

    Root nodes arrive as ``{"input": "<json>"}`` (legacy wrapper); dependent
    nodes as ``{"input_data": "<json>", "dependencies": {...}}``. Handle both,
    plus a direct dict.
    """
    for key in ("input_data", "input"):
        value = envelope.get(key)
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    return envelope


def main() -> None:
    raw = sys.stdin.read().strip()
    data: dict[str, object] = {}
    try:
        envelope = json.loads(raw)
        if isinstance(envelope, dict):
            data = _payload(envelope)
    except (json.JSONDecodeError, TypeError):
        data = {}

    requirement = str(data.get("requirement", ""))
    code = str(data.get("code", ""))
    tests = str(data.get("tests", ""))

    tests_pass, report, n_tests = run_tests(code, tests)

    print(
        json.dumps(
            {
                "requirement": requirement,
                "code": code,
                "tests": tests,
                "tests_pass": tests_pass,
                "n_tests": n_tests,
                "report": report,
            }
        )
    )


if __name__ == "__main__":
    main()
