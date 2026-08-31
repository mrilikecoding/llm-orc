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
import sys

from _helpers import deps as _deps
from _helpers import payload as _payload
from _helpers import response as _response


def _is_inert(code: str) -> bool:
    """True when no statement in ``code`` does anything: comment-only,
    whitespace-only, a bare docstring, or several bare docstrings with
    nothing else (#173). #169's ``code.strip()`` emptiness check is one
    ``#`` character wide — a comment or a bare docstring both parse and
    satisfy the injected smoke test (a ``pass`` body is satisfied by no
    code), so either would clobber a file the client already has.

    A candidate that fails to parse is not inert by this check — the
    executor's load gate (compile/exec, #169's mechanism) already rejects
    it via ``tests_pass=False``. If ``ast.parse`` here raises anyway, fail
    closed (treat as inert) rather than let the exception escape this node.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return True
    return all(
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
        for stmt in tree.body
    )


def _executor_verdict(deps: dict[str, object]) -> tuple[bool, str]:
    try:
        parsed = json.loads(_response(deps.get("executor", {})))
    except (json.JSONDecodeError, TypeError):
        parsed = {}
    if not isinstance(parsed, dict):
        parsed = {}
    return bool(parsed.get("tests_pass", False)), str(parsed.get("report", ""))


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
    tests_pass, report = _executor_verdict(deps)
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
    # #173: a candidate that parses but carries no executable statement
    # (comment-only, docstring-only, or both) is never a fix either — see
    # _is_inert. Only checked when something survived .strip(), so this
    # cannot change the emptiness branch below.
    inert = candidate_present and _is_inert(code)
    accept = tests_pass and candidate_present and not inert
    if not candidate_present:
        # Names the target (review round 1): #166's caller guard names the
        # file it declined to write, and a refusal the client cannot map to
        # a file is worth less. The target comes from select, which took it
        # from gather's own extraction — never from a path on this server.
        target = str(selected.get("target_file", "")) or "the file"
        reason = f"re-fix candidate for {target} is empty; the original is unchanged"
    elif inert:
        target = str(selected.get("target_file", "")) or "the file"
        reason = (
            f"re-fix candidate for {target} has no executable statement "
            "(comment or docstring only); the original is unchanged"
        )
    elif smoke_only:
        reason = (
            "candidate loads cleanly; no visible test, the client run verifies"
            if accept
            else f"re-fix candidate failed to load: {report}"
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
