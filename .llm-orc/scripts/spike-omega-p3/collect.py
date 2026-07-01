#!/usr/bin/env python3
"""Ω-P3 collect stage: gather the fanned build results.

Reads the ScriptAgentInput `dependencies` (the gathered fan-out results live
under the build node) and emits a compact summary so the harness can see how
many instances ran and what each returned.
"""

from __future__ import annotations

import json
import sys


def main() -> None:
    try:
        payload = json.loads(sys.stdin.read())
    except (json.JSONDecodeError, ValueError):
        payload = {}
    deps = payload.get("dependencies", {}) if isinstance(payload, dict) else {}
    print(json.dumps({"dependency_keys": sorted(deps.keys()), "dependencies": deps}))


if __name__ == "__main__":
    main()
