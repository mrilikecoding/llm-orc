#!/usr/bin/env python3
"""Serving accept gate — deterministic executor node (WP-D8, ADR-048 §1).

The builder-independent ground-truth half of the accept gate: runs the produced
tests against the produced code and reports a deterministic pass/fail. Execution
is sandboxed in a subprocess with a wall-clock timeout (ODP-1), so a runaway or
crashing test is reported as a failure, never a frozen serve. The verifier seats
downstream (judge, gate) receive only {requirement / acceptance criteria, produced
code, produced tests, execution result} and never builder reasoning, so the signal
is independent of the builder (ADR-048 §3, isolation via seat isolation).

Reads {requirement, code, tests} from the node payload; emits
{requirement, code, tests, tests_pass, n_tests, report} — the passthrough carries
the contract and artifact to the isolated judge seat.
"""

from __future__ import annotations

import ast
import builtins
import json
import keyword
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

RUNNER = Path(__file__).with_name("accept_executor_runner.py")
DEFAULT_TIMEOUT = 15.0

# per-test isolation (seat-quality design 2026-07-09): each test function
# runs in its own subprocess with its own fresh materialized directory, so
# module globals and written files cannot leak across tests. Bounded and
# surfaced — no silent caps.
_MAX_ISOLATED_TESTS = 20

# A bare-name assert on a name the tests never assign (``assert load`` /
# ``assert load, "msg"``) checks only module-object truthiness — a defined
# function is always truthy, so the line carries no test value, and the
# 4-arm tier spike (2026-07-09) showed no seat at any tier/thinking mode
# satisfies a garbage one. Stripped before execution; the echoed (shipped)
# tests are the sanitized suite. ``assert result`` on an assigned local is
# a real truthiness check and is kept.
_BARE_ASSERT_RE = re.compile(r"^\s*assert\s+([A-Za-z_]\w*)\s*(?:,.*)?$")


def _sanitize_tests(tests: str) -> tuple[str, int]:
    """Tests with value-free bare-name assert lines removed, and the count.

    "Assigned" is computed via the AST (``_bound_names``): every
    Store-context name, for/with/walrus target, parameter, def/class name,
    and import alias — not just a plain ``name = ...`` line — so a bare
    assert on any of those is a real check, not a value-free reference.
    Unparseable tests skip sanitization entirely (mirrors the injector).
    """
    try:
        tree = ast.parse(tests)
    except SyntaxError:
        return tests, 0
    assigned = _bound_names(tree)
    kept: list[str] = []
    dropped = 0
    # Skip Python keywords and built-in constants that are intentionally asserted
    skip_names = {"True", "False", "None", "__name__", "__doc__"}
    for line in tests.splitlines():
        match = _BARE_ASSERT_RE.match(line)
        if (
            match
            and match.group(1) not in assigned
            and not keyword.iskeyword(match.group(1))
            and match.group(1) not in skip_names
        ):
            dropped += 1
            continue
        kept.append(line)
    return "\n".join(kept) + ("\n" if tests.endswith("\n") else ""), dropped


# The 8b test-writer omits its own imports (live probe 2026-07-09:
# os.path.exists / pytest.raises with no import lines) — a defect no code
# regeneration can fix, since the code cannot define pytest. Deterministic
# repair, same shape as gather's workspace-import injection: unbound
# whitelisted module names get their import prepended, and the echoed
# (shipped) tests carry it, making the artifact self-contained. Names
# outside the whitelist stay uninjected and reject honestly.
_INJECTABLE_MODULES = frozenset(
    {
        "collections",
        "functools",
        "itertools",
        "json",
        "math",
        "os",
        "pathlib",
        "pytest",
        "re",
        "string",
        "sys",
        "tempfile",
        "time",
        "unittest",
    }
)

_BUILTIN_NAMES = frozenset(dir(builtins))


def _param_names(args: ast.arguments) -> set[str]:
    """Every name a function's parameter list binds, including *args/**kwargs."""
    names = {a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)}
    if args.vararg:
        names.add(args.vararg.arg)
    if args.kwarg:
        names.add(args.kwarg.arg)
    return names


def _bound_names(tree: ast.Module) -> set[str]:
    """Every name the tests source binds: assignments, for/with targets (all
    ``ast.Name`` in Store context), def/class names, function parameters, and
    import aliases — the rest are candidates for import injection."""
    bound = {
        n.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
    }
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            bound |= _param_names(node.args)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            bound |= {a.asname or a.name.split(".")[0] for a in node.names}
    return bound


def _inject_test_imports(tests: str, code: str) -> tuple[str, int]:
    """Tests with missing whitelisted imports prepended, and the count.

    Injection candidates exclude any name the code binds — an injected
    ``import os`` must never shadow a code-defined ``os`` in the runner's
    shared namespace, or a code bug (``os = None``) would be masked
    instead of failing its test.
    """
    try:
        tree = ast.parse(tests)
    except SyntaxError:
        return tests, 0
    try:
        code_bound = _bound_names(ast.parse(code))
    except SyntaxError:
        code_bound = set()
    loaded = {
        n.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }
    unbound = loaded - _bound_names(tree) - _BUILTIN_NAMES - code_bound
    missing = sorted(unbound & _INJECTABLE_MODULES)
    if not missing:
        return tests, 0
    prelude = "\n".join(f"import {name}" for name in missing)
    return f"{prelude}\n\n{tests}", len(missing)


def _calls_unbound_name(func: ast.AST, bound: set[str]) -> bool:
    """Whether a test function calls a bare name outside ``bound`` — a
    guaranteed NameError no code regeneration can convert."""
    return any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id not in bound
        for n in ast.walk(func)
    )


def _excise_unbound_callable_tests(tests: str, code: str) -> tuple[str, int]:
    """Tests minus the ones calling a name bound nowhere, and the count.

    Spike 2026-07-10 (turn6_s2/turn6_s5): ``assert file_exists(...)`` — a
    call to a bare name the tests never bind, the code deliverable never
    defines, and neither builtins nor the injectable-module whitelist can
    supply. The line NameErrors in every round; excising that one test and
    running the remainder is honest, where name-mapping (``file_exists ->
    os.path.exists``) would risk changing intent. Scope: top-level
    ``test_*`` functions (per-test isolation's unit). Bounded: if excision
    would drop ALL test units, the suite is returned unchanged so the
    round rejects on the real NameError. The echoed (shipped) tests are
    the excised suite, per the v0.18.7 repair convention.
    """
    try:
        tree = ast.parse(tests)
    except SyntaxError:
        return tests, 0
    try:
        code_bound = _bound_names(ast.parse(code))
    except SyntaxError:
        code_bound = set()
    bound = _bound_names(tree) | code_bound | _BUILTIN_NAMES | _INJECTABLE_MODULES
    test_funcs = [
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name.startswith("test_")
    ]
    doomed = [f for f in test_funcs if _calls_unbound_name(f, bound)]
    has_cases = any(isinstance(n, ast.ClassDef) for n in tree.body)
    if not doomed or (len(doomed) == len(test_funcs) and not has_cases):
        return tests, 0
    lines = tests.splitlines()
    for func in sorted(doomed, key=lambda f: f.lineno, reverse=True):
        start = min([func.lineno, *(d.lineno for d in func.decorator_list)]) - 1
        del lines[start : (func.end_lineno or func.lineno)]
    return "\n".join(lines) + ("\n" if tests.endswith("\n") else ""), len(doomed)


def _from_dependencies(envelope: dict[str, object]) -> dict[str, object] | None:
    """The ``{requirement, code, tests}`` contract from a dependency (the
    build-gated ``gather`` node), when the executor runs inside the ensemble.
    Returns ``None`` in flat direct-payload mode (no such dependency)."""
    deps = envelope.get("dependencies")
    if not isinstance(deps, dict):
        return None
    for node in deps.values():
        response = node.get("response") if isinstance(node, dict) else None
        if not isinstance(response, str):
            continue
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "code" in parsed and "tests" in parsed:
            return parsed
    return None


def _payload(envelope: dict[str, object]) -> dict[str, object]:
    """Extract the task payload across llm-orc's envelope formats: a root node
    arrives as ``{"input": "<json>"}``, a dependent node as
    ``{"input_data": ..., "dependencies": {...}}``; else the envelope itself."""
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


def _timeout() -> float:
    raw = os.environ.get("LLM_ORC_ACCEPT_EXECUTOR_TIMEOUT", "")
    try:
        return float(raw) if raw else DEFAULT_TIMEOUT
    except ValueError:
        return DEFAULT_TIMEOUT


_BUDGET_MULTIPLIER = 3.0


def _aggregate_budget() -> float:
    """Total wall budget across isolated children: per-child timeout × 3
    by default, LLM_ORC_ACCEPT_EXECUTOR_BUDGET overrides (seconds). The
    per-child timeout bounds one runaway test; this bounds the SUITE —
    per-test isolation multiplied the old single-run bound by the child
    count, and an interactive serve cannot absorb 21× (review finding,
    seat-quality arc 2026-07-09)."""
    raw = os.environ.get("LLM_ORC_ACCEPT_EXECUTOR_BUDGET", "")
    try:
        return float(raw) if raw else _timeout() * _BUDGET_MULTIPLIER
    except ValueError:
        return _timeout() * _BUDGET_MULTIPLIER


def _enumerate_tests(tests: str) -> tuple[list[str], bool] | None:
    """(top-level test_* function names, has_testcase_classes), or ``None``
    when the tests don't parse — the caller falls back to one legacy run.

    Completeness invariant: per-test isolation only ever runs the names
    returned here, but only ``tree.body`` is visible to the caller's child
    dispatch. A ``test_*`` def nested anywhere else (module-level if/try/
    for, or inside another def) would therefore be silently dropped from
    the run — legacy's single whole-module exec picks it up regardless of
    nesting. So if any such nested test def exists, return ``None`` and
    let the caller fall back to that legacy run instead of losing it.
    """
    try:
        tree = ast.parse(tests)
    except SyntaxError:
        return None
    top_level_defs = [
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name.startswith("test_")
    ]
    top_level_ids = {id(n) for n in top_level_defs}
    nested_exists = any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name.startswith("test_")
        and id(n) not in top_level_ids
        for n in ast.walk(tree)
    )
    if nested_exists:
        return None
    names = [n.name for n in top_level_defs]
    has_cases = any(isinstance(n, ast.ClassDef) for n in tree.body)
    return names, has_cases


def _materialize(
    tmp: str,
    code: str,
    tests: str,
    workspace: dict[str, str] | None,
    target_file: str,
) -> tuple[Path, Path]:
    """Write one child's fresh sandbox dir: conversation-written workspace
    files, then the edit turn's target-file shadow, then solution.py and
    tests.py — exactly the materialization the legacy single run did."""
    for name, body in (workspace or {}).items():
        safe = Path(name).name
        if safe and safe not in ("solution.py", "tests.py"):
            (Path(tmp) / safe).write_text(str(body), encoding="utf-8")
    safe_target = Path(target_file).name if target_file else ""
    if safe_target and safe_target not in ("solution.py", "tests.py"):
        (Path(tmp) / safe_target).write_text(code, encoding="utf-8")
    code_path = Path(tmp) / "solution.py"
    tests_path = Path(tmp) / "tests.py"
    code_path.write_text(code, encoding="utf-8")
    tests_path.write_text(tests, encoding="utf-8")
    return code_path, tests_path


def _run_one(
    code: str,
    tests: str,
    workspace: dict[str, str] | None,
    target_file: str,
    only: str | None,
    timeout: float,
) -> tuple[bool, str, int]:
    with tempfile.TemporaryDirectory() as tmp:
        code_path, tests_path = _materialize(tmp, code, tests, workspace, target_file)
        argv = [sys.executable, str(RUNNER), str(code_path), str(tests_path)]
        if only is not None:
            argv += ["--only", only]
        try:
            completed = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmp,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False, f"timeout after {timeout:g}s", 0
    if completed.returncode != 0:
        detail = (completed.stderr.strip() or completed.stdout.strip())[:200]
        return False, f"runner crashed: {detail}", 0
    try:
        verdict = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return False, f"unreadable runner output: {completed.stdout[:200]!r}", 0
    return (
        bool(verdict.get("tests_pass", False)),
        str(verdict.get("report", "")),
        int(verdict.get("n_tests", 0)),
    )


def _run_children(
    code: str,
    tests: str,
    workspace: dict[str, str] | None,
    target_file: str,
    timeout: float,
    children: list[str | None],
) -> tuple[list[str], int]:
    """Run each isolated child in turn, stopping early once the aggregate
    wall budget across the suite is spent — the per-child timeout bounds
    one runaway test, this bounds the whole suite (review finding: 21
    children × per-child timeout is too much blocking time for an
    interactive serve). The budget check happens before spawning each
    child, not after, so the real worst case is budget + at most one
    per-child timeout (check-before-spawn slack), not the budget alone."""
    budget = _aggregate_budget()
    start = time.monotonic()
    n = len(children)
    failures: list[str] = []
    total = 0
    for i, only in enumerate(children):
        if time.monotonic() - start >= budget:
            remaining = n - i
            failures.append(
                f"aggregate budget exhausted after {budget:g}s; "
                f"{remaining} of {n} tests not run"
            )
            break
        ok, report, n_tests = _run_one(
            code, tests, workspace, target_file, only, timeout
        )
        total += n_tests
        if not ok:
            failures.append(report)
    return failures, total


def _run_sandboxed(
    code: str,
    tests: str,
    workspace: dict[str, str] | None = None,
    target_file: str = "",
) -> tuple[bool, str, int]:
    timeout = _timeout()
    enumerated = _enumerate_tests(tests)
    if enumerated is None or (not enumerated[0] and not enumerated[1]):
        # unparseable or nothing enumerable: one legacy run reports it
        return _run_one(code, tests, workspace, target_file, None, timeout)

    names, has_cases = enumerated
    capped = len(names) > _MAX_ISOLATED_TESTS
    children: list[str | None] = list(names[:_MAX_ISOLATED_TESTS])
    if has_cases:
        children.append("__cases__")

    failures, total = _run_children(
        code, tests, workspace, target_file, timeout, children
    )
    if capped:
        failures.append(f"…capped at {_MAX_ISOLATED_TESTS} isolated tests")

    # a load failure repeats identically in every child — report it once
    deduped = list(dict.fromkeys(failures))
    if deduped:
        return False, "; ".join(deduped), total
    return True, "all passed", total


def main() -> None:
    raw = sys.stdin.read().strip()
    data: dict[str, object] = {}
    try:
        envelope = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        envelope = None
    if isinstance(envelope, dict):
        data = _from_dependencies(envelope) or _payload(envelope)

    requirement = str(data.get("requirement", ""))
    code = str(data.get("code", ""))
    tests = str(data.get("tests", ""))
    tests, tests_sanitized = _sanitize_tests(tests)
    tests, tests_excised = _excise_unbound_callable_tests(tests, code)
    tests, tests_imports_injected = _inject_test_imports(tests, code)
    raw_workspace = data.get("workspace")
    workspace = (
        {str(k): str(v) for k, v in raw_workspace.items()}
        if isinstance(raw_workspace, dict)
        else {}
    )

    target_file = str(data.get("target_file", ""))

    tests_pass, report, n_tests = _run_sandboxed(code, tests, workspace, target_file)

    print(
        json.dumps(
            {
                "requirement": requirement,
                "code": code,
                "tests": tests,
                "tests_pass": tests_pass,
                "n_tests": n_tests,
                "tests_sanitized": tests_sanitized,
                "tests_excised": tests_excised,
                "tests_imports_injected": tests_imports_injected,
                "report": report,
            }
        )
    )


if __name__ == "__main__":
    main()
