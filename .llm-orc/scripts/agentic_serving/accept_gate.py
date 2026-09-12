#!/usr/bin/env python3
"""Serving accept gate — deterministic AND node (WP-D8, ADR-048 §1).

Reads the executor's deterministic result and the isolated judge's adequacy
verdict from the dependency envelope, ANDs them, and emits a single boolean plus
reason the client can route the loop on (ODP-2: the client owns the
accept/another-round loop; this node produces the verdict, it does not iterate).
The AND lives here (deterministic) rather than in a guard, because the guard
predicate grammar is truthiness / == literal only.

    accept = tests_pass AND tests_adequate AND participates AND surface_kept

The first two catch orthogonal failures: the executor catches wrong code real
tests exercise; the isolated judge catches trivially-tested or under-covering
outputs the executor passes (ADR-048 §1). ``participates`` (#171) is the
executor's own runtime ablation control: a suite that passes identically with
the deliverable's bytes absent never actually exercised it, which neither of
the other two inputs can see. ``surface_kept`` (#182 slice B-1) is a fourth,
independent fact: whether the deliverable's own top-level names are a superset
of the prior module's public surface (gather's ``prior_surface``) — an edit
that silently drops a public name can still pass the OTHER three checks (the
surviving tests exercise only what's left), so this is not subsumed by them.
When it fails, the refusal sentence names what was dropped and stands ALONE —
never joined with "tests did not pass"/"tests inadequate...", which may be
true only incidentally. Independence: none of the four comes from a builder
the produced artifact could steer — tests_pass, participates, and
surface_kept are real sandboxed execution / deterministic AST facts,
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


def _extract_str(resp: str, key: str) -> str:
    """The string at ``key`` in ``resp``'s JSON, or "" — the executor is a
    deterministic script (never a model seat), so a lenient regex fallback
    buys nothing here."""
    try:
        obj = json.loads(resp)
        if isinstance(obj, dict):
            value = obj.get(key, "")
            return value if isinstance(value, str) else ""
    except (json.JSONDecodeError, TypeError):
        pass
    return ""


def _extract_list(resp: str, key: str) -> list[str]:
    """The list at ``key`` in ``resp``'s JSON, or [] — same lenient-JSON,
    no-regex-fallback rule as ``_extract_str`` (the executor is a
    deterministic script, never a model seat)."""
    try:
        obj = json.loads(resp)
        if isinstance(obj, dict):
            value = obj.get(key, [])
            if isinstance(value, list):
                return [str(v) for v in value]
    except (json.JSONDecodeError, TypeError):
        pass
    return []


def _surface_reason(target_file: str, missing: list[str]) -> str:
    """#182 slice B-1: the refusal sentence for a deliverable that dropped
    a name the prior module's public surface defined. Composed HERE, not
    in the executor (review F1 rework) — so it can stand ALONE as the
    reason. It is never joined with "tests did not pass" or "tests
    inadequate to verify the requirement", which may be true only
    incidentally (the surviving tests can still pass, or fail for the
    unrelated reason the missing surface causes) — the surface loss is
    what the next round needs to fix, and is the only thing said."""
    listed = missing[:3]
    names = ", ".join(listed)
    remaining = len(missing) - len(listed)
    if remaining > 0:
        names += f", and {remaining} more"
    target = target_file or "the file"
    return (
        f"the deliverable for {target} no longer defines {names}; "
        "an edit must ship the whole updated file"
    )


def _read_deps(raw: str) -> dict[str, object]:
    try:
        envelope = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    if isinstance(envelope, dict):
        found = envelope.get("dependencies", {})
        return found if isinstance(found, dict) else {}
    return {}


def _resolve_adequacy(deps: dict[str, object], reasons: list[str]) -> tuple[bool, bool]:
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


def _refusal_reasons(
    executor_resp: str,
    reasons: list[str],
    *,
    tests_pass: bool,
    tests_adequate: bool,
    participates: bool,
    surface_missing: list[str],
) -> list[str]:
    """The refusal reason(s) for a rejected turn.

    A surface loss (#182 slice B-1) stands ALONE — never joined with a
    tests_pass/tests_adequate clause that may be true only incidentally
    once the surface itself has gone (the surviving tests can still pass,
    or fail for the unrelated reason the missing surface causes). The
    surface loss is what the next round needs to fix, so it is the only
    thing said.

    Otherwise, the pre-existing orthogonal-catches composition — but only
    when ``reasons`` is still empty (an unreadable executor/judge verdict
    already explains itself and is not compounded with the others)."""
    if surface_missing:
        target_file = _extract_str(executor_resp, "target_file")
        return [_surface_reason(target_file, surface_missing)]
    if reasons:
        return reasons
    composed = list(reasons)
    if not participates:
        composed.append(
            _extract_str(executor_resp, "participation_reason")
            or "the tests never exercise the deliverable"
        )
    if not tests_pass:
        composed.append("tests did not pass")
    if not tests_adequate:
        composed.append("tests inadequate to verify the requirement")
    return composed


def main() -> None:
    deps = _read_deps(sys.stdin.read().strip())

    executor_resp = _dep_response(deps, "executor")
    tests_pass = _extract_bool(executor_resp, "tests_pass")
    reasons: list[str] = []
    if tests_pass is None:
        tests_pass = False
        reasons.append("executor verdict unreadable")
    # #171: absent (an executor response predating this field) means the
    # runtime ablation control was never run, which is the SAME "necessity
    # not disproven" default the executor itself uses — never a free pass
    # for a missing key, never a refusal for one.
    participates = _extract_bool(executor_resp, "participates")
    if participates is None:
        participates = True
    tests_adequate, carried = _resolve_adequacy(deps, reasons)

    # #182 slice B-1 (review F1 rework): a fourth, independent AND input —
    # a deterministic fact about the deliverable's own source, computed by
    # the executor alongside (never instead of) the real run.
    surface_missing = _extract_list(executor_resp, "surface_missing")
    surface_kept = not surface_missing

    accept = bool(tests_pass and tests_adequate and participates and surface_kept)
    if not accept:
        reasons = _refusal_reasons(
            executor_resp,
            reasons,
            tests_pass=tests_pass,
            tests_adequate=tests_adequate,
            participates=participates,
            surface_missing=surface_missing,
        )

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
