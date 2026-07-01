#!/usr/bin/env python3
"""Ω-P3 score: gather the fanned build results into the flow's deliverable.

Terminal node of the full serving flow. Reads the gathered fan-out results
(a list of per-file sub-ensemble child results), unwraps each instance's
deliverable, and reports per file which tier/branch built it and whether it has
content. Stand-in for the real execution gate (write files + run the tests).
"""

from __future__ import annotations

import json
import sys
from typing import Any


def _instances() -> list[Any]:
    try:
        deps = json.loads(sys.stdin.read()).get("dependencies", {})
    except (json.JSONDecodeError, ValueError, AttributeError):
        return []
    raw = deps.get("build", {}).get("response", "[]")
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def _deliverable(instance: Any) -> dict[str, Any]:
    child = json.loads(instance) if isinstance(instance, str) else instance
    if not isinstance(child, dict):
        return {}
    deliv = child.get("deliverable", "{}")
    try:
        return json.loads(deliv) if isinstance(deliv, str) else deliv
    except (json.JSONDecodeError, ValueError):
        return {}


def main() -> None:
    built = []
    for inst in _instances():
        d = _deliverable(inst)
        built.append(
            {
                "file": d.get("file"),
                "tier": d.get("tier"),
                "fired": d.get("fired_branch"),
                "has_content": bool(d.get("content")),
            }
        )
    print(json.dumps({"count": len(built), "built": built}))


if __name__ == "__main__":
    main()
