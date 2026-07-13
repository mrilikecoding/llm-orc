#!/usr/bin/env python3
"""re-fix terminal — emit an ADR-024 envelope with the re-gated verdict
(rung 2, convergent-fix design, docs/plans/2026-07-12-convergent-fix-
design.md). Carries the candidate deliverable and the executor's verdict in
``diagnostics.accept``, exactly mirroring build_gated_envelope's shape, so
the serving marshal (shape/form_gate/emit — all unmodified) ships a full
write on accept.

When no visible test exists to re-gate against (rung 1.5 found none, or the
client's failing suite lives outside test_<stem>.py), the candidate ships
unconditionally: the client's own pytest re-run is the true verifier in
that case, and the internal executor gate has nothing to check it against.
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
    tests = str(selected.get("tests", ""))
    edit_kind = str(selected.get("edit_kind", ""))
    summary = code.splitlines()[0][:80] if code.strip() else "re-fix deliverable"

    if tests.strip():
        tests_pass, report = _executor_verdict(deps)
        accept = tests_pass
        reason = report or ("tests pass" if accept else "tests did not pass")
    else:
        accept = True
        reason = "no visible test to re-gate against; shipping for the client run"

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
