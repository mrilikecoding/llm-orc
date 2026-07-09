#!/usr/bin/env python3
"""Peel the loop primitive's wrapper back to the body's terminal output.

A ``loop:`` node responds with ``{"output": <body terminal dict>,
"iterations": N, "terminated": ...}``. Downstream consumers of a looped seat
(the serving marshal, the seat contract) expect the bare ADR-024 envelope the
body's terminal emits — this node restores that contract deterministically.
"""

from __future__ import annotations

import json
import sys


def main() -> None:
    try:
        payload = json.loads(sys.stdin.read().strip())
    except json.JSONDecodeError:
        payload = {}
    deps = payload.get("dependencies", {}) if isinstance(payload, dict) else {}
    response = ""
    for dep in deps.values() if isinstance(deps, dict) else []:
        if isinstance(dep, dict) and isinstance(dep.get("response"), str):
            response = dep["response"]
            break
    try:
        wrapper = json.loads(response)
    except json.JSONDecodeError:
        wrapper = {}
    output = wrapper.get("output") if isinstance(wrapper, dict) else None
    print(json.dumps(output if isinstance(output, dict) else {}))


if __name__ == "__main__":
    main()
