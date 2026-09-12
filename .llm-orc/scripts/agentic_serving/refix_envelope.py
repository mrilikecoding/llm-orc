#!/usr/bin/env python3
"""re-fix terminal — emit an ADR-024 envelope with the re-gated verdict
(rung 2, convergent-fix design, docs/plans/2026-07-12-convergent-fix-
design.md). Carries the candidate deliverable and the executor's verdict in
``diagnostics.accept``, exactly mirroring build_gated_envelope's shape, so
the serving marshal (shape/form_gate/emit — all unmodified) ships a full
write on accept.

When no visible test exists to re-gate against (rung 1.5 found none, or the
client's failing suite lives outside test_<stem>.py), select injects a
smoke test so the executor still confirms the candidate LOADS cleanly
before it can ship — a re-fix must never clobber the original with an
unvalidated whole-file regen (F3, merge-gate review). A candidate that
fails to load is rejected here, the original preserved; the client's own
pytest re-run remains the semantic verifier once a loadable fix ships.
"""

from __future__ import annotations

import ast
import json
import re
import sys

from _helpers import deps as _deps
from _helpers import payload as _payload
from _helpers import response as _response
from _helpers import workspace_unplaced_reason as _workspace_unplaced_reason

# The surface-derived smoke test's own assert message IS the dropped name
# (refix_select's `assert hasattr(solution, "<name>"), "<name>"`) — #171
# review M3: on the smoke-only path this is the ONLY test that can fail, so
# a report matching this shape names a dropped name, never a user assertion.
_DROPPED_NAME_RE = re.compile(r"AssertionError: ([A-Za-z_]\w*)\b")

# #173 review round 1's adjudicated predicate: a closed whitelist of node
# types that can bind no name and call nothing. Structure (Module, Pass, If,
# While, the operator/context base classes) plus constant-only expression
# forms (Constant, JoinedStr/FormattedValue for f-strings, BinOp/UnaryOp/
# BoolOp/Compare/IfExp over constants, and the constant-literal containers).
# Anything outside this set — Name, Call, Attribute, Subscript, any def/
# class/assign/import, Raise, Assert, With, Try, ... — means NOT inert, and
# the candidate is judged by the load gate and tests exactly as before.
_INERT_NODE_TYPES: tuple[type[ast.AST], ...] = (
    ast.Module,
    ast.Pass,
    ast.Expr,
    ast.Constant,
    ast.JoinedStr,
    ast.FormattedValue,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.IfExp,
    ast.Tuple,
    ast.List,
    ast.Dict,
    ast.Set,
    ast.If,
    ast.While,
    ast.operator,
    ast.unaryop,
    ast.boolop,
    ast.cmpop,
    ast.expr_context,
)


def _is_inert(code: str) -> bool:
    """True when every node in the parsed candidate is drawn from the closed
    whitelist above (#173 review round 1's adjudicated predicate, replacing
    round 1's "every statement is a bare string" — which implemented a
    narrower rule than its own docstring claimed). A subtree built only from
    structure and constants can bind no name and call nothing, so it cannot
    make the injected smoke test — or any test — pass for a real reason; a
    real fix always binds (assignment, def, import, ...) or calls.

    Measured through the real chain to clobber under round 1's predicate: a
    bare ``pass``, a bare ``...``, a bare numeric/None/bool constant, an
    f-string (with or without a placeholder), a ``BinOp`` of two constants
    (``"a" + "b"``), a bytes literal, ``if False: pass``, ``while False:
    pass``, a docstring followed by ``pass``, a comment followed by ``pass``,
    and ``pass; pass`` — 15 members, this whitelist catches all of them.

    A bare Name (``x``) is NOT in the whitelist and stays not-inert — it
    fails honestly at load with a NameError (measured), a real reason, not a
    hole in this guard.

    A candidate that fails to parse is not inert by this check either — the
    executor's load gate (compile/exec, #169's mechanism) already rejects
    it via ``tests_pass=False``. If ``ast.parse`` here raises anyway, fail
    closed (treat as inert) rather than let the exception escape this node.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return True
    return all(isinstance(node, _INERT_NODE_TYPES) for node in ast.walk(tree))


def _smoke_failure_reason(target: str, report: str) -> str:
    """A plain statement of what the surface-derived smoke test found
    (#171 review M3). The smoke-only candidate can fail this gate two
    ways: it does not load at all, or it loads fine but no longer defines
    a name the prior module provided. "Failed to load" is only true of
    the first — the second used to say it anyway, and quoted the smoke
    test's own internal assert source line (an implementation detail of
    the ablation's own check, not anything the user wrote). Naming the
    DROPPED name is fine; that is the fact the check exists to report.
    """
    dropped = _DROPPED_NAME_RE.search(report)
    if dropped:
        return (
            f"re-fix candidate for {target} no longer defines "
            f"{dropped.group(1)}, which the prior module provided"
        )
    return f"re-fix candidate for {target} failed to load: {report}"


def _executor_verdict(
    deps: dict[str, object],
) -> tuple[bool, str, bool, str, list[str]]:
    """(tests_pass, report, participates, participation_reason,
    workspace_unplaced). Re-fix has no adequacy seat at all — this route's
    own accept formula is the only place the #171 ablation's verdict (and
    #184 A2's workspace-placement fact) can land. ``participates`` defaults
    True (an executor response predating the field, or the ablation never
    ran): necessity not disproven, never a free pass nor a refusal for a
    missing key."""
    try:
        parsed = json.loads(_response(deps.get("executor", {})))
    except (json.JSONDecodeError, TypeError):
        parsed = {}
    if not isinstance(parsed, dict):
        parsed = {}
    raw_unplaced = parsed.get("workspace_unplaced", [])
    workspace_unplaced = (
        [str(name) for name in raw_unplaced] if isinstance(raw_unplaced, list) else []
    )
    return (
        bool(parsed.get("tests_pass", False)),
        str(parsed.get("report", "")),
        bool(parsed.get("participates", True)),
        str(parsed.get("participation_reason", "")),
        workspace_unplaced,
    )


def main() -> None:
    payload = _payload(sys.stdin.read().strip())
    deps = _deps(payload)
    try:
        selected = json.loads(_response(deps.get("select", {})))
    except (json.JSONDecodeError, TypeError):
        selected = {}
    if not isinstance(selected, dict):
        selected = {}

    code = str(selected.get("code", ""))
    edit_kind = str(selected.get("edit_kind", ""))
    smoke_only = bool(selected.get("smoke_only", False))
    summary = code.splitlines()[0][:80] if code.strip() else "re-fix deliverable"

    # Always the executor verdict now (F3): with a visible test it re-gates
    # the fix against the client's real test; without one, select injected a
    # smoke test so the executor still confirms the candidate LOADS cleanly
    # before it can clobber the original. A candidate that fails either gate
    # is rejected here -> honest-red terminal, original preserved.
    tests_pass, report, participates, participation_reason, workspace_unplaced = (
        _executor_verdict(deps)
    )
    workspace_placed = not workspace_unplaced
    # #169: an EMPTY candidate is never a fix, and the executor cannot say
    # so. The injected smoke test's body is `pass`, which passes against any
    # code including none, so an empty model_edit used to report accept:true
    # over nothing — and since the target is a file the client already has,
    # the resulting write was a clobber rather than an empty new file.
    #
    # Not scoped to smoke_only: measured, a VISIBLE test that does not
    # reference the target module passes against an empty candidate too, and
    # rung 1.5's visible test is whatever test_<stem>.py was found.
    candidate_present = bool(code.strip())
    # #173: a candidate that parses but binds nothing and calls nothing
    # (structure and constants only — see _is_inert) is never a fix either.
    # Only checked when something survived .strip(), so this cannot change
    # the emptiness branch below.
    inert = candidate_present and _is_inert(code)
    # F-1 (#171 round 2/3 review): refix_select's smoke surface now covers
    # every public top-level BINDING, not just def/class (round 3 widened
    # constants/dict-only settings modules into the surface instead of
    # exempting them — round 2's original fallback for those reopened the
    # exact clobber #173 closed). What is left surface-less is a prior with
    # ZERO public bindings of any kind (an empty module, or one that only
    # imports names) — there the smoke-only bar is unconditionally "loads
    # cleanly", which the ablation control's empty-code run satisfies
    # identically (an "import solution" check observes nothing about the
    # candidate), so `participates` would be False for EVERY candidate
    # against that narrow class and the route could never converge. Fall
    # back to the pre-#171 bar (load cleanly, plus #173's inertness
    # whitelist) instead of applying the participation gate in exactly
    # that case.
    smoke_surface_empty = smoke_only and bool(
        selected.get("smoke_surface_empty", False)
    )
    # Round 2b (independent confirmation review): the fallback above is
    # only warranted when the prior IS readable and PROVABLY has zero
    # public bindings — not when its surface could not be determined at
    # all. `smoke_prior_status` (from refix_select's `_smoke_test`) tells
    # the two apart: "missing" (no `[PRIOR CODE]` marker — no information,
    # not "empty file") and "unparseable" (present but fails to parse,
    # reachable live via the renderer's (truncated)/(oversize) write
    # variants) both mean "we don't know", and this route must fail
    # CLOSED rather than silently accept whatever loads. "ok" is the only
    # status the surface-empty fallback above was ever meant to cover.
    smoke_prior_status = str(selected.get("smoke_prior_status", "ok"))
    prior_unreadable = smoke_only and smoke_prior_status in ("missing", "unparseable")
    effective_participates = participates or smoke_surface_empty
    accept = (
        tests_pass
        and candidate_present
        and not inert
        and workspace_placed
        and not prior_unreadable
        and effective_participates
    )
    # Names the target (review round 1): #166's caller guard names the file
    # it declined to write, and a refusal the client cannot map to a file is
    # worth less. The target comes from select, which took it from gather's
    # own extraction — never from a path on this server.
    target = str(selected.get("target_file", "")) or "the file"
    if not candidate_present:
        reason = f"re-fix candidate for {target} is empty; the original is unchanged"
    elif inert:
        reason = (
            f"re-fix candidate for {target} has no executable statement; "
            "the original is unchanged"
        )
    elif not workspace_placed:
        # #184 A2: a workspace entry root-resolution could not place in the
        # sandbox (still absolute, or escaping) makes the whole run
        # untrustworthy — checked before prior_unreadable/participation,
        # same priority accept_gate gives it on the build routes.
        reason = _workspace_unplaced_reason(workspace_unplaced)
    elif prior_unreadable:
        # Path-free (#168 discipline) and never "loads cleanly" — the
        # candidate may well load fine, but there is nothing to check that
        # against, and this route must not say otherwise.
        reason = (
            f"no prior content for {target} was available to check the fix "
            "against; the original is unchanged"
            if smoke_prior_status == "missing"
            else (
                f"the current version of {target} could not be read whole, "
                "so the fix cannot be checked against it; the original is "
                "unchanged"
            )
        )
    elif not effective_participates:
        # #171: re-fix has no adequacy seat — this is the only place its
        # own accept formula can catch a suite the deliverable never
        # touched. Scoped to a REAL surface now (F-1): a surface-less
        # prior is exempted from this gate above, so what reaches here is
        # a visible test the target module never appears in, or a
        # smoke-only surface the candidate genuinely failed to satisfy
        # some other way the ablation caught.
        reason = participation_reason or "the tests never exercise the deliverable"
    elif smoke_only:
        reason = (
            "candidate loads cleanly; no visible test, the client run verifies"
            if accept
            else _smoke_failure_reason(target, report)
        )
    else:
        reason = report or ("tests pass" if accept else "tests did not pass")

    envelope = {
        "status": "success",
        "primary": code,
        "structured": {"content": code},
        "artifacts": [
            {"content_type": "text/x-python", "content": code, "summary": summary}
        ],
        "diagnostics": {
            "ensemble": "re-fix",
            "accept": accept,
            "accept_reason": reason,
            "edit_kind": edit_kind,
        },
    }
    print(json.dumps(envelope))


if __name__ == "__main__":
    main()
