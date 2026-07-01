#!/usr/bin/env python3
"""Ω-P3 probe: reveal exactly what a fanned sub-ensemble instance receives.

Root node of the build-one sub-ensemble. Echoes its raw input (stdin
ScriptAgentInput + the INPUT_TEXT env) so the spike can see how a fan-out
instance's chunk is delivered into the sub-ensemble.
"""

from __future__ import annotations

import json
import os
import sys


def main() -> None:
    raw_stdin = sys.stdin.read()
    try:
        payload = json.loads(raw_stdin)
    except (json.JSONDecodeError, ValueError):
        payload = None
    out = {
        "env_INPUT_TEXT": os.environ.get("INPUT_TEXT"),
        "stdin_type": type(payload).__name__,
        "stdin_keys": sorted(payload.keys()) if isinstance(payload, dict) else None,
        "stdin_input_data": payload.get("input_data") if isinstance(payload, dict) else None,
        "raw_stdin_head": raw_stdin[:300],
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
