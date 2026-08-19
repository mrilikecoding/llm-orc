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


def _wire_safe(text: str, fallback: str) -> str:
    """``text`` if it names no path, else ``fallback`` (#168 round 3).

    Checked on what is EMITTED, never merely on what was inspected. Round
    2 introduced the property rule but enforced it at the producer, and
    that gap leaked twice: an exception whose message is MULTI-LINE has a
    last traceback line with no colon, so the ``split(":", 1)[0]`` fallback
    returned it unchanged; and ``type(error).__name__`` was emitted with no
    check at all, which produced code controls. Both are derived strings,
    and a property asserted about an input says nothing about a derivation
    of it.
    """
    if _path_free(text):
        return text
    # Round 4: the FALLBACK was unchecked, and a caller built one by
    # interpolating a produced-code-controlled test name — the guard's own
    # output escaped the property the guard exists to enforce, which is
    # round 3's defect one level down.
    #
    # Every caller now passes a fallback built from already-checked pieces
    # or a constant, so no shipped path reaches the second check. It stays
    # because this is the function's POSTCONDITION: what it returns is
    # path-free, full stop. A postcondition that holds only while callers
    # are careful is the thing this arc keeps being bitten by, and it is
    # not pinnable precisely because the callers are correct.
    return fallback if _path_free(fallback) else "failed"


def _exception_line(trace: str) -> str:
    """The ``Type: message`` line of a traceback, or its last line.

    Found by the traceback's own GRAMMAR: frames are indented, so an
    exception line is the first column-0 line after an indented block, and
    the LAST such line is the exception that actually terminated. Three
    shapes defeat the obvious alternatives, all ordinary rather than
    adversarial —

    - taking the last LINE loses the type whenever the message is
      multi-line, which is what ``assertEqual`` produces for any sequence;
    - scanning backward for the first ``identifier:`` line picks a MESSAGE
      line reading ``Key: value`` over the real type line above it, so
      ``AssertionError('Response mismatch:\nStatus: 404\nBody: not found')``
      reported ``Body: not found`` (#168 review round 6);
    - taking the FIRST such line reports the CAUSE of a chained exception
      rather than the one that failed the test, so ordinary
      ``except ValueError: raise RuntimeError(...)`` cleanup reported
      ``ValueError`` (#168 review round 7). A chained traceback has one
      frame block per link, so resetting after each capture and keeping the
      last lands on the terminating exception — and ``raise ... from None``
      has one link, so it is unaffected.

    Measured: an ``assertEqual`` diff's continuation lines are all at
    column 0, so they never look like a frame and cannot start a new
    candidate.

    Falls back to the last line when there are no frames at all, which is
    the shape a bare message has.
    """
    lines = trace.strip().splitlines()
    found = ""
    after_frame = False
    for line in lines:
        if line[:1].isspace():
            after_frame = True
        elif after_frame:
            found = line.strip()
            after_frame = False
    return found or (lines[-1].strip() if lines else "")


def _safe_reason(error: BaseException) -> str:
    """What a test failure may say on the wire (#168).

    The class name always; the message only when it names no path. A
    ``SyntaxError``'s line and column ride along regardless — they are
    integers, and they are the part a developer actually uses, while the
    executor's report is clipped server-side by the trace snippet, so the
    class name alone would be the only surviving record anywhere.
    """
    name = _wire_safe(type(error).__name__, "error")
    lineno = getattr(error, "lineno", None)
    if isinstance(lineno, int):
        offset = getattr(error, "offset", None)
        where = f" at line {lineno}"
        if isinstance(offset, int):
            where += f", column {offset}"
        return name + where
    message = str(error).strip()
    # `name` is already checked and the separator is not in ": ", so a
    # further check on the join would be dead (round 4). One guard, once.
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
            # Each PIECE is checked as it enters the string (round 4).
            # Checking only the whole discarded the class name and the
            # message whenever the source echo happened to name a RELATIVE
            # path — `assert os.path.exists('data/out.txt')` reduced an
            # ordinary failing test to "failed", which is the evidence loss
            # round 2 paid to avoid and which also feeds the next round's
            # retry prompt.
            safe_name = _wire_safe(name, "test")
            detail = f"{safe_name}: {_safe_reason(error)}"
            if line and _path_free(line):
                detail += f" at: {line}"
            failures.append(_wire_safe(detail, f"{safe_name}: failed"))
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
        return (
            False,
            _wire_safe(
                f"code failed to load: {_safe_reason(error)}",
                "code failed to load",
            ),
            0,
        )
    try:
        exec(compile(tests, "test_solution.py", "exec"), namespace)
    except Exception as error:  # noqa: BLE001
        return (
            False,
            _wire_safe(
                f"tests failed to load: {_safe_reason(error)}",
                "tests failed to load",
            ),
            0,
        )

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
            # Same piece-by-piece discipline as _run_test_fns (#168 round 5).
            # Checking only the composed string lost the test name, the
            # exception class and the message together on an ordinary
            # `assertEqual` over a sequence containing a relative path: the
            # diff is multi-line, its LAST line is a diff fragment with no
            # colon, the type salvage returned it unchanged, and everything
            # went. `test failed` is what the client and the retry prompt
            # both got for a real, ordinary failure.
            detail = _exception_line(trace)
            if not _path_free(detail):
                detail = detail.split(":", 1)[0].strip()
            failures.append(
                f"{_wire_safe(str(test), 'test')}: {_wire_safe(detail, 'failed')}"
            )

    if n_tests == 0:
        detail = _wire_safe(
            f"no test named {only!r} found"
            if only and only != "__cases__"
            else ("no test_* functions or TestCase classes found"),
            "no tests found",
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
