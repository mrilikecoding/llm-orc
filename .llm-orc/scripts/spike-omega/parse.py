#!/usr/bin/env python3
"""Spike Ω-1 parse stage.

Reads the incoming request (input_data string or JSON), extracts the
task and any prior context. For Ω-1 turn 1, prior_context is always
empty. The stage preserves the §4 topology: a real script where a
parse would read substrate state across turns (Ω-2).

Emits JSON: {"task": "<str>", "prior_context": "<str or empty>"}
"""

import json
import sys


def main() -> None:
    raw = sys.stdin.read().strip()
    task = ""
    prior_context = ""

    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            # ScriptAgentInput wraps our request inside input_data.
            input_data = data.get("input_data") or data.get("input") or ""
            # input_data may itself be a JSON string of our request shape.
            try:
                inner = json.loads(input_data) if isinstance(input_data, str) else None
            except json.JSONDecodeError:
                inner = None
            if isinstance(inner, dict):
                task = inner.get("task", "") or inner.get("input", "")
                prior_context = inner.get("last_tool_result", "") or inner.get(
                    "prior_context", ""
                )
            elif isinstance(input_data, str):
                task = input_data
            else:
                task = ""
        elif isinstance(data, str):
            task = data
    except (json.JSONDecodeError, TypeError):
        task = raw

    if not task:
        print(json.dumps({"success": False, "error": "No task in input"}))
        return

    print(json.dumps({"task": task, "prior_context": prior_context}))


if __name__ == "__main__":
    main()