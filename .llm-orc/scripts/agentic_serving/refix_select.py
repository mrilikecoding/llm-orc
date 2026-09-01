#!/usr/bin/env python3
"""re-fix select node — the two-rung ladder's outcome (rung 2, convergent-fix
design). gather's pinned deterministic edit wins when present; otherwise the
model edit's extracted code (drop_test_blocks — a code-writer response
sometimes echoes a copy of the test alongside the fix, and shipping both
would embed the test suite in the deliverable). Emits the flat
{requirement, code, tests} the accept executor verifies (accept_executor's
own dependency scan picks this response up unmodified).

When rung 1.5 found no visible test to re-gate against, a smoke test is
injected so the executor still verifies the candidate at least LOADS
cleanly (the runner execs the code before any test, so a candidate that
parses but fails to import fails this gate) — a re-fix must never clobber
the original with an unvalidated whole-file regen (F3, merge-gate review).
The smoke test is internal-only; the deliverable is the code alone.

Surface-derived (#171): a bare ``def ...(): pass`` references no name, so
participation was unsatisfiable on this path and any loadable content —
``x = 1``, an unrelated def, a comment — shipped. The smoke test now
asserts the candidate still binds every top-level name the PRIOR module
(``gather``'s ``prior_code``) defined; a candidate that drops one refuses.
"""

from __future__ import annotations

import ast
import json
import sys

from _helpers import deps as _deps
from _helpers import extract_code as _extract_code
from _helpers import payload as _payload
from _helpers import response as _response
from _helpers import terminal as _terminal


def _smoke_test(prior_code: str) -> str:
    """A smoke test asserting the candidate imports cleanly and still binds
    every top-level function/class name ``prior_code`` defined. No prior
    surface (unparseable, or the prior module bound nothing at module
    level) degrades to the "loads cleanly" bar alone — deterministic,
    import/getattr shape only, no behavior claims."""
    try:
        tree = ast.parse(prior_code)
    except SyntaxError:
        names: list[str] = []
    else:
        names = sorted(
            n.name
            for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        )
    asserts = "".join(
        f'    assert hasattr(solution, "{name}"), "{name}"\n' for name in names
    )
    return "def test_refix_candidate_loads_cleanly():\n    import solution\n" + asserts


def main() -> None:
    payload = _payload(sys.stdin.read().strip())
    deps = _deps(payload)
    try:
        gathered = json.loads(_response(deps.get("gather", {})))
    except (json.JSONDecodeError, TypeError):
        gathered = {}
    if not isinstance(gathered, dict):
        gathered = {}

    deterministic_code = str(gathered.get("deterministic_code", ""))
    if deterministic_code:
        code, edit_kind = deterministic_code, "deterministic"
    else:
        generated = _terminal(_response(deps.get("model_edit", {})))
        code = _extract_code(generated, drop_test_blocks=True)
        edit_kind = "model"

    visible_test = str(gathered.get("visible_test", ""))
    smoke_only = not visible_test.strip()
    prior_code = str(gathered.get("prior_code", ""))
    tests = _smoke_test(prior_code) if smoke_only else visible_test

    print(
        json.dumps(
            {
                "requirement": str(gathered.get("task", "")),
                "code": code,
                "tests": tests,
                "target_file": str(gathered.get("target_file", "")),
                "edit_kind": edit_kind,
                "smoke_only": smoke_only,
            }
        )
    )


if __name__ == "__main__":
    main()
