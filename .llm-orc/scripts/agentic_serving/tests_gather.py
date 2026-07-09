#!/usr/bin/env python3
"""write-tests gather node — assemble the test-deliverable contract (#98).

The write-tests shape's deliverable IS the test file: one test source
(test_writer), empty code, the conversation workspace materialized for the
executor to run the tests against. No target_file — nothing shadows, so the
artifact that ships is exactly the artifact that executed (the issue #98
wrong-accept was the gate validating a shadowed composite).

Reuses accept_gather's extraction, context-splitting, and deterministic
workspace-import injection via sibling import (one implementation).

Emits JSON: {requirement, code, tests, workspace, target_file, held}
"""

from __future__ import annotations

import json
import sys

from _helpers import payload as _payload
from _helpers import response as _response
from _helpers import terminal as _terminal
from accept_gather import _REQUEST_MARKER, _extract_tests, _inject_workspace_imports
from accept_gather import _workspace as _extract_workspace


def main() -> None:
    payload = _payload(sys.stdin.read().strip())
    requirement = str(payload.get("input_data", ""))
    workspace: dict[str, str] = {}
    if _REQUEST_MARKER in requirement:
        context, requirement = requirement.rsplit(_REQUEST_MARKER, 1)
        workspace = _extract_workspace(context)
    deps = payload.get("dependencies", {})
    if not isinstance(deps, dict):
        deps = {}

    tests = _extract_tests(_terminal(_response(deps.get("test_writer", {}))))
    tests = _inject_workspace_imports(tests, workspace)

    print(
        json.dumps(
            {
                "requirement": requirement,
                "code": "",
                "tests": tests,
                "workspace": workspace,
                "target_file": "",
                "held": False,
            }
        )
    )


if __name__ == "__main__":
    main()
