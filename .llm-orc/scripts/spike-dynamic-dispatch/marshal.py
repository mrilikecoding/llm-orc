#!/usr/bin/env python3
"""Dynamic-dispatch skeleton — marshal (deterministic finalize) stage.

Reads the classify decision and the dispatched seat's output and shapes the
serve outcome: a code kind becomes a file write, an explanation becomes a
finishing prose message. Deterministic, no model. Terminal node of the
skeleton, so its response IS the turn's serve outcome:
    code:        {"finish": false, "file": "<path>", "content": "<source>"}
    explanation: {"finish": true, "content": "<prose>"}
"""

import json
import re
import sys


def _deps(raw: str) -> dict:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data.get("dependencies", {}) if isinstance(data, dict) else {}


def _response(dep: object) -> str:
    return dep.get("response", "") if isinstance(dep, dict) else ""


def _seat_terminal(seat_response: str) -> str:
    """The dispatched seat's terminal node output (dispatch returns the child
    ensemble result as JSON)."""
    try:
        child = json.loads(seat_response)
    except json.JSONDecodeError:
        return seat_response
    results = child.get("results", {}) if isinstance(child, dict) else {}
    if not results:
        return seat_response
    node = results[list(results.keys())[-1]]
    return node.get("response", "") if isinstance(node, dict) else ""


def _clean_code(content: str) -> str:
    s = content.strip()
    if s.startswith("```"):
        m = re.search(r"```(?:[a-zA-Z]+)?\n(.*?)```", s, re.DOTALL)
        if m:
            s = m.group(1).strip()
    return s


def main() -> None:
    deps = _deps(sys.stdin.read().strip())
    try:
        decision = json.loads(_response(deps.get("classify", {})))
    except json.JSONDecodeError:
        decision = {}

    kind = decision.get("kind", "python_module")
    content = _seat_terminal(_response(deps.get("seat", {})))

    if kind == "explanation":
        print(json.dumps({"finish": True, "content": content.strip()}))
    else:
        print(
            json.dumps(
                {
                    "finish": False,
                    "file": decision.get("file", "solution.py"),
                    "content": _clean_code(content),
                }
            )
        )


if __name__ == "__main__":
    main()
