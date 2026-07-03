#!/usr/bin/env python3
"""Serving resolve node — merge classify + the guarded decider into routing.

The final routing decision the ``seat`` dispatches on. When ``classify`` resolved
the turn structurally (``needs_decider: false``) its decision passes through
unchanged and the model-backed ``decide`` node never ran. When classify deferred
(``needs_decider: true``), resolve reads the decider's bounded target and derives
build/kind deterministically — the model only classifies into the closed seat
set, so the control decision stays deterministic (ADR-046 §1; determinism-over-
carve-outs). An out-of-set decider output leaves ``target`` empty so the dispatch
node fails deterministically rather than guessing a default seat.
"""

from __future__ import annotations

import json
import re
import sys

# The closed seat set the decider chooses from, and the build/kind each implies.
_DERIVED = {
    "code-seat": ("python_module", True),
    "explainer": ("explanation", False),
}
_JSON_RE = re.compile(r"\{[^{}]*\}")


def _deps(raw: str) -> dict:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data.get("dependencies", {}) if isinstance(data, dict) else {}


def _response(dep: object) -> str:
    return dep.get("response", "") if isinstance(dep, dict) else ""


def _decider_target(response: str) -> str:
    """A known seat target from the decider's output, or "" when none resolves.

    Strict first: the first JSON object's ``target`` if it is a known seat.
    Fallback: exactly one known token present in the raw text. No token, or an
    ambiguous both-tokens output, resolves to "" (deterministic dispatch fail).
    """
    match = _JSON_RE.search(response or "")
    if match:
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict):
            target = obj.get("target")
            if isinstance(target, str) and target in _DERIVED:
                return target
    present = [t for t in _DERIVED if t in (response or "")]
    return present[0] if len(present) == 1 else ""


def main() -> None:
    deps = _deps(sys.stdin.read().strip())
    try:
        classify = json.loads(_response(deps.get("classify", {})))
    except json.JSONDecodeError:
        classify = {}
    if not isinstance(classify, dict):
        classify = {}

    file = classify.get("file", "solution.py")
    dispatch_input = classify.get("dispatch_input", "")

    if classify.get("needs_decider"):
        target = _decider_target(_response(deps.get("decide", {})))
        kind, build = _DERIVED.get(target, ("", False))
    else:
        target = classify.get("target", "")
        kind = classify.get("kind", "")
        build = bool(classify.get("build", False))

    print(
        json.dumps(
            {
                "target": target,
                "kind": kind,
                "file": file,
                "dispatch_input": dispatch_input,
                "build": build,
            }
        )
    )


if __name__ == "__main__":
    main()
