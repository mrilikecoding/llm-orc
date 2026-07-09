#!/usr/bin/env python3
"""write-tests terminal — the ADR-024 envelope for a test deliverable (#98).

Ships the EXECUTOR-ECHOED tests (the exact artifact that ran against the
workspace) as the primary artifact, with the accept verdict in diagnostics.
On reject, composes a fresh-regeneration retry carry — the write-tests
shape has no held mode: tests are the moving side by definition, and the
executor's real-workspace failure report is the retry evidence.
"""

from __future__ import annotations

import json
import sys

from _helpers import deps as _deps
from _helpers import payload as _payload
from _helpers import response as _response


def _parsed(deps: dict[str, object], name: str) -> dict[str, object]:
    try:
        parsed = json.loads(_response(deps.get(name, {})))
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def main() -> None:
    payload = _payload(sys.stdin.read().strip())
    deps = _deps(payload)
    executor = _parsed(deps, "executor")
    verdict = _parsed(deps, "accept_gate")
    tests = str(executor.get("tests", ""))
    summary = tests.splitlines()[0][:80] if tests.strip() else "test deliverable"

    diagnostics = {
        "ensemble": "write-tests",
        "accept": bool(verdict.get("accept", False)),
        "accept_reason": str(verdict.get("reason", "")),
        "tests_pass": bool(verdict.get("tests_pass", False)),
        "tests_adequate": bool(verdict.get("tests_adequate", False)),
    }
    if not diagnostics["accept"]:
        report = str(executor.get("report", ""))
        diagnostics["retry_input"] = (
            f"{payload.get('input_data', '')}\n\n"
            f"[Previous round rejected: {diagnostics['accept_reason']}."
            f" Executor report against the workspace: {report}."
            f" Regenerate the tests to exercise the workspace code's real"
            f" behavior.]"
        )

    envelope = {
        "status": "success",
        "primary": tests,
        "structured": {"content": tests},
        "artifacts": [
            {
                "content_type": "text/x-python",
                "content": tests,
                "summary": summary,
            }
        ],
        "diagnostics": diagnostics,
    }
    print(json.dumps(envelope))


if __name__ == "__main__":
    main()
