#!/usr/bin/env python3
"""Serving accept gate — deterministic AND node (WP-D8, ADR-048 §1).

Reads the executor's deterministic result and the isolated judge's adequacy
verdict from the dependency envelope, ANDs them, and emits a single boolean plus
reason the client can route the loop on (ODP-2: the client owns the
accept/another-round loop; this node produces the verdict, it does not iterate).
The AND lives here (deterministic) rather than in a guard, because the guard
predicate grammar is truthiness / == literal only.

    accept = tests_pass AND tests_adequate

The two catch orthogonal failures: the executor catches wrong code real tests
exercise; the isolated judge catches trivially-tested or under-covering outputs
the executor passes (ADR-048 §1). Independence: neither input comes from a builder
the produced artifact could steer — tests_pass is real sandboxed execution,
tests_adequate is a fresh-context judge (ADR-048 §3).

Emits JSON: {accept, tests_pass, tests_adequate, reason}
"""

from __future__ import annotations

import json
import re
import sys

from _helpers import terminal as _terminal


def _dep_response(deps: dict[str, object], name: str) -> str:
    node = deps.get(name, {})
    if isinstance(node, dict):
        resp = node.get("response", "")
        return resp if isinstance(resp, str) else json.dumps(resp)
    return ""


def _extract_bool(resp: str, key: str) -> bool | None:
    """Lenient bool extraction: JSON first, then a ``"key": true/false`` regex."""
    try:
        obj = json.loads(resp)
        if isinstance(obj, dict) and key in obj:
            value = obj[key]
            if isinstance(value, str):
                # small models sometimes quote booleans; "false" must not
                # truthy its way through the gate
                return value.strip().lower() == "true"
            return bool(value)
    except (json.JSONDecodeError, TypeError):
        pass
    match = re.search(rf'"{key}"\s*:\s*(true|false)', resp, re.IGNORECASE)
    if match:
        return match.group(1).lower() == "true"
    return None


def _read_deps(raw: str) -> dict[str, object]:
    try:
        envelope = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    if isinstance(envelope, dict):
        found = envelope.get("dependencies", {})
        return found if isinstance(found, dict) else {}
    return {}


def _resolve_adequacy(
    deps: dict[str, object], reasons: list[str]
) -> tuple[bool, bool]:
    """(tests_adequate, carried) from the judge verdict or the held flag.

    The judge is a sub-ensemble seat (#84): its dep response is the nested
    child-result envelope — peel to the model's raw verdict (terminal() is
    a no-op on a bare verdict, so both shapes read). Held round (issue
    #100): no judge seat — the held path only fires when round 1's judge
    passed these exact tests, so the verdict carries deterministically;
    the executor stays the live gate. No judge AND no held flag is a
    miswired shape, not a free pass.
    """
    tests_adequate = _extract_bool(
        _terminal(_dep_response(deps, "judge")), "tests_adequate"
    )
    if tests_adequate is not None:
        return tests_adequate, False
    if _extract_bool(_dep_response(deps, "gather"), "held"):
        return True, True
    reasons.append("judge verdict unreadable")
    return False, False


def main() -> None:
    deps = _read_deps(sys.stdin.read().strip())

    tests_pass = _extract_bool(_dep_response(deps, "executor"), "tests_pass")
    reasons: list[str] = []
    if tests_pass is None:
        tests_pass = False
        reasons.append("executor verdict unreadable")
    tests_adequate, carried = _resolve_adequacy(deps, reasons)

    accept = bool(tests_pass and tests_adequate)
    if not accept and not reasons:
        if not tests_pass:
            reasons.append("tests did not pass")
        if not tests_adequate:
            reasons.append("tests inadequate to verify the requirement")

    if reasons:
        reason = "; ".join(reasons)
    elif carried:
        reason = "tests pass; adequacy carried from round 1"
    else:
        reason = "tests pass and are adequate"
    print(
        json.dumps(
            {
                "accept": accept,
                "tests_pass": tests_pass,
                "tests_adequate": tests_adequate,
                "reason": reason,
            }
        )
    )


if __name__ == "__main__":
    main()
