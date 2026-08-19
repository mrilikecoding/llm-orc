#!/usr/bin/env python3
"""Sandboxed test runner for the serving accept gate (WP-D8, ODP-1).

Runs the produced tests against the produced code in a fresh subprocess, so a
runaway or crashing test cannot hang the serve. Invoked by ``accept_executor.py``
as ``python accept_executor_runner.py <code_path> <tests_path> [--only <name>]``:
reads the two files, execs them in one shared namespace (so the tests reference
the code's names), runs every ``test_*`` callable, and prints the deterministic
verdict. Optional ``--only <name>`` runs a single test function by name;
``--only __cases__`` runs only unittest.TestCase classes.

Emits JSON: {tests_pass, n_tests, report}

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


# #168 review round 2. An exception MESSAGE is the wire channel: a
# CalledProcessError carries the full argv (the serve's own interpreter, so
# the operator's home and username), and OSError's __str__ adds the filename
# its repr hides. But dropping messages wholesale destroyed "DID NOT RAISE",
# which a pre-existing pin holds precisely because losing it "starved the
# retry round of evidence".
#
# So the rule is a PROPERTY of the output, not a list of bad producers: a
# message may pass only if it contains no path separator. Every absolute
# path on either platform contains one, so this cannot pass a path — and it
# keeps every message that is genuinely about the test rather than the
# filesystem. Same discipline as shape.py's summary, checked on what is
# emitted rather than on what produced it.
_SEPARATORS = ("/", "\\")


def _path_free(text: str) -> bool:
    return not any(sep in text for sep in _SEPARATORS)


def _safe_reason(error: BaseException) -> str:
    """What a test failure may say on the wire (#168).

    The class name always; the message only when it names no path. A
    ``SyntaxError``'s line and column ride along regardless — they are
    integers, and they are the part a developer actually uses, while the
    executor's report is clipped server-side by the trace snippet, so the
    class name alone would be the only surviving record anywhere.
    """
    name = type(error).__name__
    lineno = getattr(error, "lineno", None)
    if isinstance(lineno, int):
        offset = getattr(error, "offset", None)
        where = f" at line {lineno}"
        if isinstance(offset, int):
            where += f", column {offset}"
        return name + where
    message = str(error).strip()
    if message and _path_free(message):
        return f"{name}: {message}"
    return name


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
            # #168 review round 2: `{error!r}` here was the FOURTH wire
            # channel and the most reachable — a failing test is the ordinary
            # outcome of a re-fix round, and refix_envelope binds this report
            # straight to accept_reason, which emit ships. A produced module
            # that shells out to sys.executable and fails puts the serve's own
            # interpreter path — hence the operator's home and username — in
            # the repr, with no cooperation from the client. The class name
            # says what happened and cannot carry a path.
            detail = f"{name}: {_safe_reason(error)}"
            if line:
                detail += f" at: {line}"
            failures.append(detail)
    return n_tests, failures


def run_tests(code: str, tests: str, only: str | None = None) -> tuple[bool, str, int]:
    """Exec code + tests in a shared namespace, call every ``test_*`` function.

    ``only`` (per-test isolation, seat-quality design 2026-07-09) restricts
    the run to one named top-level test function — the executor spawns one
    runner per test so module and filesystem state cannot leak across
    tests. ``only="__cases__"`` runs just the unittest.TestCase classes.
    """
    namespace: dict[str, object] = {}
    try:
        exec(compile(code, "solution.py", "exec"), namespace)
    except Exception as error:  # noqa: BLE001 - executing produced code
        return False, f"code failed to load: {_safe_reason(error)}", 0
    try:
        exec(compile(tests, "test_solution.py", "exec"), namespace)
    except Exception as error:  # noqa: BLE001
        return False, f"tests failed to load: {_safe_reason(error)}", 0

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
            # The last traceback line is "Type: {str(exc)}", and OSError's
            # __str__ includes the filename its repr hides (#168 round 2).
            # Keep the type, drop the message.
            # The last traceback line is "Type: {str(exc)}"; the same
            # property check applies, and the type survives either way.
            last = trace.strip().splitlines()[-1]
            if not _path_free(last):
                last = last.split(":", 1)[0].strip()
            failures.append(f"{test}: {last}")

    if n_tests == 0:
        detail = (
            f"no test named {only!r} found"
            if only and only != "__cases__"
            else ("no test_* functions or TestCase classes found")
        )
        return False, detail, 0
    if failures:
        return False, "; ".join(failures), n_tests
    return True, "all passed", n_tests


def main() -> None:
    # sandbox dir on sys.path so tests can import materialized workspace
    # modules (conversation-written files) as siblings
    sys.path.insert(0, str(Path(sys.argv[1]).resolve().parent))
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1]
    code = Path(sys.argv[1]).read_text(encoding="utf-8")
    tests = Path(sys.argv[2]).read_text(encoding="utf-8")
    tests_pass, report, n_tests = run_tests(code, tests, only)
    print(json.dumps({"tests_pass": tests_pass, "n_tests": n_tests, "report": report}))


if __name__ == "__main__":
    main()
