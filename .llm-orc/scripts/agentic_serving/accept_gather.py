#!/usr/bin/env python3
"""build-gated gather node — assemble the accept-gate contract (WP-D8).

Reads the test_writer and code_writer sub-ensemble outputs plus the turn's
criteria (the node's base input), peels the nested ensemble envelopes, strips
code fences, and emits the flat ``{requirement, code, tests}`` the accept
executor verifies. Tests-first: the tests come from test_writer (authored from
the criteria) and the code from code_writer (built against those tests).

Emits JSON: {requirement, code, tests}
"""

from __future__ import annotations

import json
import re
import sys


def _payload(raw: str) -> dict[str, object]:
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def _response(dep: object) -> str:
    return dep.get("response", "") if isinstance(dep, dict) else ""


def _terminal(text: str) -> str:
    """Peel the sub-ensemble envelope layers (deliverable / output / results) to
    the terminal node's raw output (matches emit_envelope._terminal)."""
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


def _fenced(text: str) -> str | None:
    blocks = re.findall(r"```(?:[a-zA-Z0-9_+-]+)?\n(.*?)```", text, re.DOTALL)
    if blocks:
        return "\n".join(block.strip() for block in blocks)
    return None


def _extract_code(text: str) -> str:
    fenced = _fenced(text)
    return fenced if fenced is not None else text.strip()


def _extract_tests(text: str) -> str:
    fenced = _fenced(text)
    if fenced is not None:
        return fenced
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith(("def test_", "import ", "from ")):
            return "\n".join(lines[i:]).strip()
    return text.strip()


def main() -> None:
    payload = _payload(sys.stdin.read().strip())
    requirement = str(payload.get("input_data", ""))
    deps = payload.get("dependencies", {})
    if not isinstance(deps, dict):
        deps = {}

    tests = _extract_tests(_terminal(_response(deps.get("test_writer", {}))))
    code = _extract_code(_terminal(_response(deps.get("code_writer", {}))))

    print(json.dumps({"requirement": requirement, "code": code, "tests": tests}))


if __name__ == "__main__":
    main()
