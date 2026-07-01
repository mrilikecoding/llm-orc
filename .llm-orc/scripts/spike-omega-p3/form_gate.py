#!/usr/bin/env python3
"""Ω-P3 form-gate: the build loop's terminal gate (3c).

Terminal node of the code-attempt body. Minimal stand-in: accepts the attempt's
content and emits the loop contract {ok, content, next_input}. Always ok here so
the loop converges on iteration 1 — 3c only confirms a loop node runs inside the
fanned sub-ensemble; carry-driven convergence is already proven by spike Ω-loop.
"""

from __future__ import annotations

import json
import sys


def main() -> None:
    try:
        deps = json.loads(sys.stdin.read()).get("dependencies", {})
        attempt = json.loads(deps.get("attempt", {}).get("response", "{}"))
    except (json.JSONDecodeError, ValueError, AttributeError):
        attempt = {}
    print(
        json.dumps(
            {
                "ok": True,
                "file": attempt.get("file", ""),
                "tier": attempt.get("tier", "code"),
                "content": attempt.get("content", ""),
                "next_input": "",
            }
        )
    )


if __name__ == "__main__":
    main()
