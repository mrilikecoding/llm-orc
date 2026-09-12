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
asserts the candidate still binds every PUBLIC top-level name the PRIOR
module (``gather``'s ``prior_code``) defined; a candidate that drops one
refuses.

F-2 (#171 round 2 review): the surface is PUBLIC names only — a leading
underscore (private helpers and dunders alike) is excluded. A legitimate
re-fix that inlines or deletes a private helper binds nothing the module's
public surface ever promised, and refusing it for that is refusing a
correct fix.

F-1, round 3 (widened, not skipped): a constants-only settings module or a
dict-only rates table has no def/class, so the F-1 fallback originally
skipped the participation check for it entirely — which reopened the exact
clobber #173 closed for def-bearing modules (a junk ``x = 1`` shipped
against a constants-only prior). The surface now also includes top-level
assignment targets (``Assign``/``AnnAssign`` with a plain ``Name`` target,
simple tuple/list unpacking too), public ones only, same as def/class. The
"loads cleanly" fallback is now reserved for a prior with literally ZERO
public bindings of any kind (an empty module, or one that only imports
names — imports were never part of the surface, before or after this).

Round 2b (independent confirmation review): the fallback above was only
ever meant for a prior that IS readable and PROVABLY has zero public
bindings — not for "we could not determine the surface at all". Two such
undetermined shapes: ``prior_code`` is empty (the ``[PRIOR CODE]`` marker
was never populated) and ``prior_code`` is present but fails to parse
(reachable live via the renderer's ``(truncated)``/``(oversize)`` write
variants). Both now refuse honestly instead of silently degrading to
"loads cleanly" — see ``_smoke_test``'s ``prior_status`` and
``refix_envelope``'s handling of it. A prior with only private/dunder
bindings stays on the fallback (named bound, recorded in the design
brief, not closed).
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


def _assign_target_names(target: ast.expr) -> set[str]:
    """Public plain-``Name`` targets an assignment target binds: a bare
    ``Name`` (excluding a leading underscore, same rule as def/class), or a
    ``Tuple``/``List`` of them for simple unpacking (``A, B = 1, 2``).
    Anything else — ``Attribute``, ``Subscript``, ``Starred`` — contributes
    no name, same as it always has for def/class (this function only ever
    ADDS names, never removes one the old rule already caught)."""
    if isinstance(target, ast.Name):
        return set() if target.id.startswith("_") else {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for elt in target.elts:
            names |= _assign_target_names(elt)
        return names
    return set()


def _public_top_level_names(tree: ast.Module) -> list[str]:
    """Every PUBLIC top-level binding ``prior_code`` makes: def/class names
    (F-2), plus (F-1 round 3, widened rather than skipped) plain-assignment
    targets — ``Assign`` and ``AnnAssign`` with a ``Name`` target, simple
    tuple/list unpacking included. A settings module's ``PORT = 8080`` is a
    public binding exactly the way a function name is; a fix that drops it
    must refuse for the same reason a fix that drops a function refuses.
    Import statements bind names too but are deliberately NOT surface here
    (never were) — an import-only module (an ``__init__`` re-export, say)
    has zero public bindings by this function's own definition, which is
    what routes it to the "loads cleanly" fallback below."""
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                names.add(node.name)
        elif isinstance(node, ast.Assign):
            for assign_target in node.targets:
                names |= _assign_target_names(assign_target)
        elif isinstance(node, ast.AnnAssign):
            names |= _assign_target_names(node.target)
    return sorted(names)


_BARE_SMOKE_TEXT = "def test_refix_candidate_loads_cleanly():\n    import solution\n"


def _smoke_test(prior_code: str) -> tuple[str, bool, str]:
    """(test text, has_surface, prior_status).

    ``prior_status`` is ``"missing"`` when ``prior_code`` is empty (the
    ``[PRIOR CODE]`` marker was never populated — ``refix_gather`` has NO
    information about the client's file, not "the file is empty"),
    ``"unparseable"`` when ``prior_code`` is present but fails to parse
    (reachable live via the renderer's ``(truncated)``/``(oversize)`` write
    variants, which garble the file's actual content), or ``"ok"``
    otherwise.

    Round 2b correction (independent confirmation review): ``has_surface``
    is only meaningful when ``prior_status`` is ``"ok"`` — "missing" and
    "unparseable" mean the caller could not determine whether the module
    had a surface AT ALL, and the caller (refix_envelope) must refuse
    outright rather than silently degrade to the "loads cleanly" bar; that
    fallback is reserved for a prior that IS readable and PROVABLY has zero
    public bindings (an empty module, one that only imports names, or one
    whose only top-level names are private/dunder — named bound: junk
    against that last shape still ships under the loads-cleanly bar,
    recorded not closed, see the design brief)."""
    if not prior_code:
        return _BARE_SMOKE_TEXT, False, "missing"
    try:
        tree = ast.parse(prior_code)
    except SyntaxError:
        return _BARE_SMOKE_TEXT, False, "unparseable"
    names = _public_top_level_names(tree)
    asserts = "".join(
        f'    assert hasattr(solution, "{name}"), "{name}"\n' for name in names
    )
    return _BARE_SMOKE_TEXT + asserts, bool(names), "ok"


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
    smoke_surface_empty = False
    smoke_prior_status = "ok"
    if smoke_only:
        tests, has_surface, smoke_prior_status = _smoke_test(prior_code)
        smoke_surface_empty = not has_surface
    else:
        tests = visible_test

    print(
        json.dumps(
            {
                "requirement": str(gathered.get("task", "")),
                "code": code,
                "tests": tests,
                "target_file": str(gathered.get("target_file", "")),
                "edit_kind": edit_kind,
                "smoke_only": smoke_only,
                "smoke_surface_empty": smoke_surface_empty,
                "smoke_prior_status": smoke_prior_status,
            }
        )
    )


if __name__ == "__main__":
    main()
