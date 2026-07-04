#!/usr/bin/env python3
"""Sandboxed test runner for the serving accept gate (WP-D8, ODP-1).

Runs the produced tests against the produced code in a fresh subprocess, so a
runaway or crashing test cannot hang the serve. Invoked by ``accept_executor.py``
as ``python accept_executor_runner.py <code_path> <tests_path>``: reads the two
files, execs them in one shared namespace (so the tests reference the code's
names), runs every ``test_*`` callable, and prints the deterministic verdict.

Emits JSON: {tests_pass, n_tests, report}

Sandbox scope (MVP): process isolation plus a wall-clock timeout enforced by the
caller. Heavy sandboxing (container / seccomp / resource limits) is the named
hardening follow-up (ADR-048 §Consequences, "sandboxed execution is BUILD work").
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def run_tests(code: str, tests: str) -> tuple[bool, str, int]:
    """Exec code + tests in a shared namespace, call every ``test_*`` function."""
    namespace: dict[str, object] = {}
    try:
        exec(compile(code, "solution.py", "exec"), namespace)
    except Exception as error:  # noqa: BLE001 - executing produced code
        return False, f"code failed to load: {error!r}", 0
    try:
        exec(compile(tests, "test_solution.py", "exec"), namespace)
    except Exception as error:  # noqa: BLE001
        return False, f"tests failed to load: {error!r}", 0

    test_fns = [
        (name, fn)
        for name, fn in namespace.items()
        if name.startswith("test_") and callable(fn)
    ]
    if not test_fns:
        return False, "no test_* functions found", 0

    failures: list[str] = []
    for name, fn in test_fns:
        try:
            fn()
        except Exception as error:  # noqa: BLE001
            failures.append(f"{name}: {error!r}")

    if failures:
        return False, "; ".join(failures), len(test_fns)
    return True, "all passed", len(test_fns)


def main() -> None:
    code = Path(sys.argv[1]).read_text(encoding="utf-8")
    tests = Path(sys.argv[2]).read_text(encoding="utf-8")
    tests_pass, report, n_tests = run_tests(code, tests)
    print(json.dumps({"tests_pass": tests_pass, "n_tests": n_tests, "report": report}))


if __name__ == "__main__":
    main()
