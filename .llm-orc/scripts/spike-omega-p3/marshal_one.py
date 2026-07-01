#!/usr/bin/env python3
"""Ω-P3 marshal-one: join the guarded branches into the per-file deliverable.

Terminal node of the build-one sub-ensemble. Exactly one of build-code /
build-prose fired (the other was skipped by its guard, so it is absent from the
dependencies). Emits whichever branch produced — this becomes the sub-ensemble's
deliverable for the fan-out gather.
"""

from __future__ import annotations

import json
import sys


def main() -> None:
    try:
        deps = json.loads(sys.stdin.read()).get("dependencies", {})
    except (json.JSONDecodeError, ValueError, AttributeError):
        deps = {}
    chosen: dict | None = None
    branch = ""
    for name in ("build-code", "build-prose"):
        node = deps.get(name)
        if isinstance(node, dict) and node.get("status") == "success" and node.get("response"):
            chosen = _unwrap(json.loads(node["response"]))
            branch = name
            break
    out = chosen or {"error": "no branch produced"}
    out["fired_branch"] = branch
    print(json.dumps(out))


def _unwrap(response: dict) -> dict:
    """A loop branch wraps its result as {output, iterations, terminated}; a plain
    branch returns the deliverable directly. Normalize to the deliverable."""
    if "output" in response and isinstance(response["output"], dict):
        return response["output"]
    return response


if __name__ == "__main__":
    main()
