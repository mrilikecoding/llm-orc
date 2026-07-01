#!/usr/bin/env python3
"""Ω-P3 plan: project the resolved contract into per-file build tasks.

Bridges the resolve-contract loop to the build fan-out. The loop wraps its
terminal output as {output: {ok, contract, ...}, iterations, terminated}, so this
unwraps `output.contract` and emits a BARE JSON array — one composite
{deliverable, contract} per file — for the downstream `fan_out: true` build node.
Each element carries the full contract so every fanned build instance has the
sibling APIs it must integrate against (fan-out hands an instance only its chunk).
"""

from __future__ import annotations

import json
import sys


def main() -> None:
    try:
        payload = json.loads(sys.stdin.read())
        loop_out = json.loads(payload["dependencies"]["resolve-contract"]["response"])
    except (json.JSONDecodeError, ValueError, KeyError, TypeError):
        loop_out = {}
    contract = loop_out.get("output", {}).get("contract", [])
    tasks = [{"deliverable": d, "contract": contract} for d in contract]
    # Emit a dict (not a bare array) so the build node selects the array via
    # input_key — the routing-demo convention. A top-level array trips the script
    # request-processor (it expects dict-shaped output).
    print(json.dumps({"tasks": tasks}))


if __name__ == "__main__":
    main()
