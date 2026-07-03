#!/usr/bin/env python3
"""Serving marshal — emit node (client permission seam).

Terminal node of the serving ensemble: emits the serve outcome the caller maps
onto the client permission seam — a file-write for a valid build deliverable, a
prose finish otherwise (scenarios.md "Per-Turn Serving Handler"; ADR-046 §1,
ADR-034 re-homes the Client-Tool-Action Terminal). A build deliverable the
form-gate refused degrades to a prose finish carrying the refusal reason: the
serve never writes a deliverable that failed destination-validity.

    build + valid:   {"finish": false, "file": "<path>", "content": "<source>"}
    build + refused: {"finish": true, "content": "Refused: <reason>"}
    non-build:       {"finish": true, "content": "<prose>"}
"""

from __future__ import annotations

import json
import sys


def _deps(raw: str) -> dict:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data.get("dependencies", {}) if isinstance(data, dict) else {}


def _response(dep: object) -> str:
    return dep.get("response", "") if isinstance(dep, dict) else ""


def main() -> None:
    deps = _deps(sys.stdin.read().strip())
    try:
        gated = json.loads(_response(deps.get("form_gate", {})))
    except json.JSONDecodeError:
        gated = {}
    if not isinstance(gated, dict):
        gated = {}

    build = bool(gated.get("build", False))
    content = str(gated.get("content", ""))

    if build and gated.get("valid", False):
        outcome = {
            "finish": False,
            "file": gated.get("file", "solution.py"),
            "content": content,
        }
    elif build:
        outcome = {
            "finish": True,
            "content": f"Refused: {gated.get('reason', 'invalid deliverable')}",
        }
    else:
        outcome = {"finish": True, "content": content}

    print(json.dumps(outcome))


if __name__ == "__main__":
    main()
