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

# /// llm-orc
# files = ["_helpers.py"]
# ///

from __future__ import annotations

import json
import sys

from _helpers import HELD_TESTS_MARKER as _HELD_MARKER
from _helpers import deps as _deps
from _helpers import extract_code as _extract_code
from _helpers import payload as _payload
from _helpers import response as _response
from _helpers import terminal as _terminal


_UNVERIFIED_REASON = "verified code unavailable; nothing was written"


def _executor_result(deps: dict[str, object]) -> dict[str, object]:
    try:
        parsed = json.loads(_response(deps.get("executor", {})))
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}







def _executed_code(deps: dict[str, object]) -> str | None:
    """The code the executor ran, echoed back (#181), or None when the
    executor's output is unreadable or carries no string ``code``.

    accept_gather repairs the candidate (injected workspace imports,
    trimmed trailing prose) before the suite runs, and the executor echoes
    the result. Never stringified: a non-string is not the verified code."""
    code = _executor_result(deps).get("code")
    return code if isinstance(code, str) else None


def _deliverable_code(deps: dict[str, object], executed: str | None) -> str:
    """What the envelope carries: the executed code when there is one. The
    raw extraction is only the fallback for a round with no executed code,
    and such a round can never be accepted (see main)."""
    if executed is not None:
        return executed
    return _extract_code(
        _terminal(_response(deps.get("code_writer", {}))), drop_test_blocks=True
    )


def _verdict(deps: dict[str, object]) -> dict[str, object]:
    try:
        parsed = json.loads(_response(deps.get("accept_gate", {})))
    except (json.JSONDecodeError, TypeError):
        parsed = {}
    return parsed if isinstance(parsed, dict) else {}


def _held_round(deps: dict[str, object]) -> bool:
    """Whether this round ran held (build-code-round wires gather in as an
    envelope dep; the fresh round has no gather dep here) — the round's mode
    is only knowable live from the envelope diagnostics."""
    try:
        parsed = json.loads(_response(deps.get("gather", {})))
    except (json.JSONDecodeError, TypeError):
        return False
    return bool(parsed.get("held", False)) if isinstance(parsed, dict) else False


def main() -> None:
    payload = _payload(sys.stdin.read().strip())
    deps = _deps(payload)
    executed = _executed_code(deps)
    code = _deliverable_code(deps, executed)
    verdict = _verdict(deps)
    summary = code.splitlines()[0][:80] if code.strip() else "code deliverable"

    diagnostics = {
        "ensemble": "build-gated",
        "accept": bool(verdict.get("accept", False)),
        "accept_reason": str(verdict.get("reason", "")),
        "tests_pass": bool(verdict.get("tests_pass", False)),
        "tests_adequate": bool(verdict.get("tests_adequate", False)),
        "held_round": _held_round(deps),
    }
    if diagnostics["accept"] and not (executed and executed.strip()):
        # accept is a claim about the shipped bytes; with no verified code in
        # hand there is nothing to claim it about, whatever the gate said
        diagnostics["accept"] = False
        diagnostics["accept_reason"] = _UNVERIFIED_REASON
    if not diagnostics["accept"]:
        # the bounded retry round's carry REPLACES the next iteration's
        # input (loop primitive semantics), so compose turn + evidence here
        executor = _executor_result(deps)
        report = str(executor.get("report", ""))
        tests = str(executor.get("tests", ""))
        n_tests = int(executor.get("n_tests", 0) or 0)
        held = bool(diagnostics["tests_adequate"] and n_tests > 0 and tests.strip())
        if held:
            # TDD retry (issue #100): the round's tests collected, ran, and
            # were judged adequate — they ARE the spec; round 2 holds them
            # fixed and regenerates only the code (route dispatches on the
            # marker to the held round)
            diagnostics["retry_input"] = (
                f"{payload.get('input_data', '')}\n\n"
                f"[Previous round rejected: {diagnostics['accept_reason']}."
                f" Executor report: {report}."
                f" The tests below are the spec; write code that passes them.]\n\n"
                f"{_HELD_MARKER}\n```python\n{tests}\n```"
            )
        else:
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
