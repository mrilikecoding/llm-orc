#!/usr/bin/env python3
"""build-gated terminal — emit an ADR-024 envelope with the accept verdict (WP-D8).

Terminal node of the gated build shape. Carries the code deliverable
(``artifacts[0]`` / ``primary``, extracted from the code_writer seat) and the
accept-gate verdict (``accept`` / ``accept_reason`` / signal booleans) in the
ADR-024 ``diagnostics``, so the serving marshal consumes a faithful structured
artifact and the client sees the accept/another-round decision (ODP-2: the client
owns the loop). The surviving ADR-024 container shape is preserved; the retired
``calibration_verdict`` / ``audit_findings`` subfields stay absent (Cycle-8
collapse preservation).
"""

from __future__ import annotations

import json
import sys

from _helpers import deps as _deps
from _helpers import extract_code as _extract_code
from _helpers import payload as _payload
from _helpers import response as _response
from _helpers import terminal as _terminal






def _executor_report(deps: dict[str, object]) -> str:
    try:
        parsed = json.loads(_response(deps.get("executor", {})))
    except (json.JSONDecodeError, TypeError):
        return ""
    return str(parsed.get("report", "")) if isinstance(parsed, dict) else ""







def _verdict(deps: dict[str, object]) -> dict[str, object]:
    try:
        parsed = json.loads(_response(deps.get("accept_gate", {})))
    except (json.JSONDecodeError, TypeError):
        parsed = {}
    return parsed if isinstance(parsed, dict) else {}


def main() -> None:
    payload = _payload(sys.stdin.read().strip())
    deps = _deps(payload)
    code = _extract_code(_terminal(_response(deps.get("code_writer", {}))))
    verdict = _verdict(deps)
    summary = code.splitlines()[0][:80] if code.strip() else "code deliverable"

    diagnostics = {
        "ensemble": "build-gated",
        "accept": bool(verdict.get("accept", False)),
        "accept_reason": str(verdict.get("reason", "")),
        "tests_pass": bool(verdict.get("tests_pass", False)),
        "tests_adequate": bool(verdict.get("tests_adequate", False)),
    }
    if not diagnostics["accept"]:
        # the bounded retry round's carry REPLACES the next iteration's
        # input (loop primitive semantics), so compose turn + evidence here
        report = _executor_report(deps)
        diagnostics["retry_input"] = (
            f"{payload.get('input_data', '')}\n\n"
            f"[Previous round rejected: {diagnostics['accept_reason']}."
            f" Executor report: {report}."
            f" Regenerate, fixing the failing tests or wrong expectations.]"
        )

    envelope = {
        "status": "success",
        "primary": code,
        "structured": {"content": code},
        "artifacts": [
            {
                "content_type": "text/x-python",
                "content": code,
                "summary": summary,
            }
        ],
        "diagnostics": diagnostics,
    }
    print(json.dumps(envelope))


if __name__ == "__main__":
    main()
