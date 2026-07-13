#!/usr/bin/env python3
"""re-fix select node — the two-rung ladder's outcome (rung 2, convergent-fix
design). gather's pinned deterministic edit wins when present; otherwise the
model edit's extracted code (drop_test_blocks — a code-writer response
sometimes echoes a copy of the test alongside the fix, and shipping both
would embed the test suite in the deliverable). Emits the flat
{requirement, code, tests} the accept executor verifies (accept_executor's
own dependency scan picks this response up unmodified).
"""

from __future__ import annotations

import json
import sys

from _helpers import deps as _deps
from _helpers import extract_code as _extract_code
from _helpers import payload as _payload
from _helpers import response as _response
from _helpers import terminal as _terminal


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

    print(
        json.dumps(
            {
                "requirement": str(gathered.get("task", "")),
                "code": code,
                "tests": str(gathered.get("visible_test", "")),
                "target_file": str(gathered.get("target_file", "")),
                "edit_kind": edit_kind,
            }
        )
    )


if __name__ == "__main__":
    main()
