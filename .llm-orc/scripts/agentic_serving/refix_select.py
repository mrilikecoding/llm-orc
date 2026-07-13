#!/usr/bin/env python3
"""re-fix select node — the two-rung ladder's outcome (rung 2, convergent-fix
design). gather's pinned deterministic edit wins when present; otherwise the
model edit's extracted code (drop_test_blocks — a code-writer response
sometimes echoes a copy of the test alongside the fix, and shipping both
would embed the test suite in the deliverable). Emits the flat
{requirement, code, tests} the accept executor verifies (accept_executor's
own dependency scan picks this response up unmodified).

When rung 1.5 found no visible test to re-gate against, a minimal smoke
test is injected so the executor still verifies the candidate at least
LOADS cleanly (the runner execs the code before any test, so a candidate
that parses but fails to import fails this gate) — a re-fix must never
clobber the original with an unvalidated whole-file regen (F3, merge-gate
review). The smoke test is internal-only; the deliverable is the code
alone.
"""

from __future__ import annotations

import json
import sys

from _helpers import deps as _deps
from _helpers import extract_code as _extract_code
from _helpers import payload as _payload
from _helpers import response as _response
from _helpers import terminal as _terminal

_SMOKE_TEST = "def test_refix_candidate_loads_cleanly():\n    pass\n"


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
    tests = _SMOKE_TEST if smoke_only else visible_test

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
