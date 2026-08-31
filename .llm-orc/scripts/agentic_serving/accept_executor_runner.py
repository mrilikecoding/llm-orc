#!/usr/bin/env python3
"""Sandboxed test runner for the serving accept gate (WP-D8, ODP-1).

Runs the produced tests against the produced code in a fresh subprocess, so a
runaway or crashing test cannot hang the serve. Invoked by ``accept_executor.py``
as ``python accept_executor_runner.py <code_path> <tests_path> [--only <name>]``:
reads the two files, execs them in one shared namespace (so the tests reference
the code's names), runs every ``test_*`` callable, and prints the deterministic
verdict. Optional ``--only <name>`` runs a single test function by name;
``--only __cases__`` runs only unittest.TestCase classes.

Emits JSON: {tests_pass, n_tests, report, leaked_classes}

Sandbox scope (MVP): process isolation plus a wall-clock timeout enforced by the
caller. Heavy sandboxing (container / seccomp / resource limits) is the named
hardening follow-up (ADR-048 §Consequences, "sandboxed execution is BUILD work").
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path


def _filter_by_only(
    test_fns: list, case_classes: list, only: str | None
) -> tuple[list, list]:
    """Restrict the run per --only: a named test_* function (TestCase
    classes skipped), or "__cases__" for just the TestCase classes."""
    if only == "__cases__":
        return [], case_classes
    if only is not None:
        return [(n, f) for n, f in test_fns if n == only], []
    return test_fns, case_classes


def _failing_line(error: Exception, tests: str) -> str:
    """The failing source line from the tests, when the last traceback frame
    lands there — a bare 'AssertionError()' gives the retry round nothing to
    move on; the offending expectation is the actionable evidence."""
    tb = error.__traceback__
    if tb is None:
        return ""
    frames = traceback.extract_tb(tb)
    for frame in reversed(frames):
        if frame.filename == "test_solution.py" and frame.lineno:
            lines = tests.splitlines()
            if 0 < frame.lineno <= len(lines):
                return lines[frame.lineno - 1].strip()
    return ""


def _run_test_fns(test_fns: list, tests: str) -> tuple[int, list[str]]:
    """Execute each test_* function; return count and failures list."""
    import asyncio

    n_tests = 0
    failures: list[str] = []
    for name, fn in test_fns:
        n_tests += 1
        try:
            result = fn()
            if asyncio.iscoroutine(result):
                # an async test returns a coroutine that raised nothing yet —
                # uncollected it would count as a silent pass (wrong accept)
                asyncio.run(result)
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except BaseException as error:  # noqa: BLE001
            # BaseException, not Exception: pytest's Failed (the
            # pytest.raises DID-NOT-RAISE outcome) derives from
            # BaseException and must report as a clean per-test failure,
            # not crash the runner child (validation replay 2026-07-10)
            line = _failing_line(error, tests)
            detail = f"{name}: {error!r}"
            if line:
                detail += f" at: {line}"
            failures.append(detail)
    return n_tests, failures


def _leaked_test_classes(namespace: dict[str, object], testcase: type) -> list[str]:
    """Names of module-level classes, in the REAL post-exec namespace, that
    carry a callable ``test_*`` attribute and are NOT ``testcase``
    subclasses — the pytest ``class TestFoo: def test_x(self): ...``
    dialect the runner has no mechanism to execute (#176 F8).

    ``getattr``/``dir`` walk the MRO, so this catches every syntactic
    shape the class could reach a ``test_*`` callable through: a method
    defined directly on the class, one inherited from a base defined in
    the CODE file rather than the tests file, or a name simply ASSIGNED
    to an imported callable (``test_x = imported_fn``, no ``def`` at
    all). #176 review round 2 (N-1, rule 18's third instance of this file
    guessing what runs from AST shape): a prior version of this check
    parsed the tests SOURCE for a class defining a `test_*` method and
    compared the executed-test count against the top-level function
    count, and every one of those three shapes evaded it — the truth
    only exists in the namespace after both files have actually loaded,
    so the judgment moves here.
    """
    leaked = []
    for name, obj in namespace.items():
        if not isinstance(obj, type) or issubclass(obj, testcase):
            continue
        if any(
            attr.startswith("test_") and callable(getattr(obj, attr, None))
            for attr in dir(obj)
        ):
            leaked.append(name)
    return leaked


def run_tests(
    code: str, tests: str, only: str | None = None
) -> tuple[bool, str, int, list[str]]:
    """Exec code + tests in a shared namespace, call every ``test_*`` function.

    ``only`` (per-test isolation, seat-quality design 2026-07-09) restricts
    the run to one named top-level test function — the executor spawns one
    runner per test so module and filesystem state cannot leak across
    tests. ``only="__cases__"`` runs just the unittest.TestCase classes.

    Returns (tests_pass, report, n_tests, leaked_classes) — the last is
    the names any ``_leaked_test_classes`` found, always computed from the
    full namespace regardless of ``only``, so every child (and the legacy
    whole-run) reports the same file-level fact consistently.
    """
    namespace: dict[str, object] = {}
    try:
        exec(compile(code, "solution.py", "exec"), namespace)
    except Exception as error:  # noqa: BLE001 - executing produced code
        return False, f"code failed to load: {error!r}", 0, []
    try:
        exec(compile(tests, "test_solution.py", "exec"), namespace)
    except Exception as error:  # noqa: BLE001
        return False, f"tests failed to load: {error!r}", 0, []

    test_fns = [
        (name, fn)
        for name, fn in namespace.items()
        if name.startswith("test_") and callable(fn) and not isinstance(fn, type)
    ]

    # unittest.TestCase classes are the other common seat-model dialect
    # ('no test_* functions found' otherwise rejects perfectly good tests)
    import unittest

    case_classes = [
        obj
        for obj in namespace.values()
        if isinstance(obj, type)
        and issubclass(obj, unittest.TestCase)
        and obj is not unittest.TestCase
    ]
    leaked_classes = _leaked_test_classes(namespace, unittest.TestCase)

    test_fns, case_classes = _filter_by_only(test_fns, case_classes, only)

    n_tests, failures = _run_test_fns(test_fns, tests)

    if case_classes:
        loader = unittest.defaultTestLoader
        suite = unittest.TestSuite(
            loader.loadTestsFromTestCase(case) for case in case_classes
        )
        result = unittest.TestResult()
        suite.run(result)
        n_tests += result.testsRun
        for test, trace in result.failures + result.errors:
            failures.append(f"{test}: {trace.strip().splitlines()[-1]}")

    if leaked_classes:
        failures.append(
            "tests define a class-based suite the runner cannot execute: "
            + ", ".join(sorted(leaked_classes))
        )

    if n_tests == 0 and not leaked_classes:
        detail = (
            f"no test named {only!r} found"
            if only and only != "__cases__"
            else ("no test_* functions or TestCase classes found")
        )
        return False, detail, 0, leaked_classes
    if failures:
        return False, "; ".join(failures), n_tests, leaked_classes
    return True, "all passed", n_tests, leaked_classes


def main() -> None:
    # sandbox dir on sys.path so tests can import materialized workspace
    # modules (conversation-written files) as siblings
    sys.path.insert(0, str(Path(sys.argv[1]).resolve().parent))
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1]
    code = Path(sys.argv[1]).read_text(encoding="utf-8")
    tests = Path(sys.argv[2]).read_text(encoding="utf-8")
    tests_pass, report, n_tests, leaked_classes = run_tests(code, tests, only)
    print(
        json.dumps(
            {
                "tests_pass": tests_pass,
                "n_tests": n_tests,
                "report": report,
                "leaked_classes": leaked_classes,
            }
        )
    )


if __name__ == "__main__":
    main()
