#!/usr/bin/env python3
"""code-seat terminal — emit an ADR-024 ``DispatchEnvelope`` for the deliverable.

The code-generation flow (``code-generator``) ends in a chatty synthesizer
response wrapping the code in prose. This deterministic node extracts the code
deliverable and emits it as an ADR-024 ``DispatchEnvelope`` so the serving
marshal (the shape node) consumes a faithful structured artifact rather than
guessing at prose (scenarios.md "the marshal node consumes the seat's real
common I/O envelope"; ADR-024; ADR-046 §2). The seat owns the deliverable
CONTENT; the serving classify owns the DESTINATION path — shape combines them.
"""

from __future__ import annotations

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


def _terminal(text: str) -> str:
    """The child ensemble's deliverable, unwrapping the layers the engine adds.

    A child ensemble result carries ``deliverable`` (the terminal node's output);
    an inline-shell script node wraps its stdout as ``{"success", "output"}``; a
    nested result carries ``results``. Peel these until a plain deliverable
    string remains (a model seat's raw text, or a script seat's output).
    """
    current = text
    for _ in range(6):
        try:
            obj = json.loads(current)
        except (json.JSONDecodeError, TypeError):
            return current
        if not isinstance(obj, dict):
            return current
        if isinstance(obj.get("deliverable"), str):
            current = obj["deliverable"]
            continue
        if isinstance(obj.get("output"), str):
            current = obj["output"]
            continue
        results = obj.get("results")
        if isinstance(results, dict) and results:
            node = results[list(results.keys())[-1]]
            current = node.get("response", "") if isinstance(node, dict) else str(node)
            continue
        return current
    return current


def _extract_code(text: str) -> str:
    stripped = text.strip()
    match = re.search(r"```(?:[a-zA-Z0-9_+-]+)?\n(.*?)```", stripped, re.DOTALL)
    if match:
        return match.group(1).strip()
    return stripped


def main() -> None:
    deps = _deps(sys.stdin.read().strip())
    generated = _terminal(_response(deps.get("generate", {})))
    code = _extract_code(generated)
    summary = code.splitlines()[0][:80] if code.strip() else "code deliverable"

    envelope = {
        "status": "success",
        "primary": code,
        "structured": {"content": code},
        "artifacts": [
            {
                "content_type": "text/x-python",
                "content": code,
                "summary": summary,
            }
        ],
        "diagnostics": {"ensemble": "code-seat"},
    }
    print(json.dumps(envelope))


if __name__ == "__main__":
    main()
