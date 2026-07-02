#!/usr/bin/env python3
"""Q2 grounding spike — deterministic accept gate.

Reads the executor's deterministic result and the isolated judge's
test-adequacy verdict from the dependency envelope, ANDs them, and emits a
single boolean the guard can route on. The AND-ing lives here (deterministic)
rather than in the guard, because the guard predicate grammar is truthiness /
== literal only.

accept = tests_pass AND tests_adequate

Independence: neither input comes from a builder the produced artifact could
steer. tests_pass is real execution; tests_adequate is a fresh-context judge.

Emits JSON: {accept, tests_pass, tests_adequate, reason}
"""

import json
import re
import sys


def _dep_response(deps: dict, name: str) -> str:
    node = deps.get(name, {})
    if isinstance(node, dict):
        resp = node.get("response", "")
        return resp if isinstance(resp, str) else json.dumps(resp)
    return ""


def _extract_bool(resp: str, key: str) -> bool | None:
    """Lenient bool extraction: JSON first, then a `"key": true/false` regex."""
    try:
        obj = json.loads(resp)
        if isinstance(obj, dict) and key in obj:
            return bool(obj[key])
    except (json.JSONDecodeError, TypeError):
        pass
    match = re.search(rf'"{key}"\s*:\s*(true|false)', resp, re.IGNORECASE)
    if match:
        return match.group(1).lower() == "true"
    return None


def main() -> None:
    raw = sys.stdin.read().strip()
    deps: dict = {}
    try:
        envelope = json.loads(raw)
        if isinstance(envelope, dict):
            deps = envelope.get("dependencies", {}) or {}
    except (json.JSONDecodeError, TypeError):
        deps = {}

    executor_resp = _dep_response(deps, "executor")
    judge_resp = _dep_response(deps, "judge")

    tests_pass = _extract_bool(executor_resp, "tests_pass")
    tests_adequate = _extract_bool(judge_resp, "tests_adequate")

    reasons: list[str] = []
    if tests_pass is None:
        tests_pass = False
        reasons.append("executor verdict unreadable")
    if tests_adequate is None:
        tests_adequate = False
        reasons.append("judge verdict unreadable")

    accept = bool(tests_pass and tests_adequate)
    if not accept and not reasons:
        if not tests_pass:
            reasons.append("tests did not pass")
        if not tests_adequate:
            reasons.append("tests inadequate to verify the requirement")

    print(
        json.dumps(
            {
                "accept": accept,
                "tests_pass": tests_pass,
                "tests_adequate": tests_adequate,
                "reason": "; ".join(reasons) if reasons else "tests pass and are adequate",
            }
        )
    )


if __name__ == "__main__":
    main()
