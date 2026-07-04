#!/usr/bin/env python3
"""Serving marshal — form-gate node (deterministic destination-validity).

Applies the cheapest rung of the verification ladder: a deliverable must parse
as what its destination path claims. A ``.py`` deliverable must ``ast.parse``;
a ``.json`` deliverable must load. A deliverable that does not parse is refused
before it reaches the client (scenarios.md "form-gate refuses a deliverable
that does not parse as its path claims"; ADR-046 §1, ADR-035 re-home). Passes
the shaped deliverable through with a ``valid`` verdict; a non-build turn is
inert here (nothing to parse).
"""

from __future__ import annotations

import ast
import json
import sys


def _deps(raw: str) -> dict:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data.get("dependencies", {}) if isinstance(data, dict) else {}


def _response(dep: object) -> str:
    return dep.get("response", "") if isinstance(dep, dict) else ""


def _validity(file: str, content: str) -> tuple[bool, str]:
    if file.endswith(".py"):
        try:
            ast.parse(content)
        except SyntaxError as error:
            return False, f"deliverable for {file} is not valid Python: {error}"
        return True, "ok"
    if file.endswith(".json"):
        try:
            json.loads(content)
        except json.JSONDecodeError as error:
            return False, f"deliverable for {file} is not valid JSON: {error}"
        return True, "ok"
    return True, "ok"


def main() -> None:
    deps = _deps(sys.stdin.read().strip())
    try:
        shaped = json.loads(_response(deps.get("shape", {})))
    except json.JSONDecodeError:
        shaped = {}
    if not isinstance(shaped, dict):
        shaped = {}

    build = bool(shaped.get("build", False))
    file = str(shaped.get("file", "solution.py"))
    content = str(shaped.get("content", ""))

    if not build:
        valid, reason = True, "ok"
    else:
        valid, reason = _validity(file, content)

    print(
        json.dumps(
            {
                "build": build,
                "file": file,
                "content": content,
                "valid": valid,
                "reason": reason,
                # Pass the accept-gate verdict through to emit unchanged (the
                # form-gate is the cheaper syntax rung; the accept gate ran in the
                # build shape). ``None`` when the seat carries no verdict.
                "accept": shaped.get("accept"),
                "accept_reason": str(shaped.get("accept_reason", "")),
            }
        )
    )


if __name__ == "__main__":
    main()
