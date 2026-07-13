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

import json
import sys

from _helpers import deps as _deps
from _helpers import payload as _payload
from _helpers import response as _response


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
    accept = tests_pass
    if smoke_only:
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
