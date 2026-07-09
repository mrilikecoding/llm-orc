#!/usr/bin/env python3
"""Shared helpers for the agentic_serving script nodes (issue #92).

Python sets ``sys.path[0]`` to the script's directory, so sibling scripts
import this module directly. One implementation each for the three things
every node re-implemented: the script-node payload contract, the
sub-ensemble envelope peel, and fenced-code extraction — the extractors had
drifted into three divergent copies, so the accept gate could judge
different code than emit shipped.
"""

from __future__ import annotations

import json
import re
from typing import Any

# Fences tagged with a shell-ish language are usage examples, not code.
SHELL_LANGS = {"bash", "sh", "shell", "console", "zsh", "text"}

_FENCE_RE = re.compile(r"```([a-zA-Z0-9_+-]*)\n(.*?)```", re.DOTALL)


def payload(raw: str) -> dict[str, Any]:
    """The script-node stdin payload as a dict ({} on anything malformed)."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def deps(payload_dict: dict[str, Any]) -> dict[str, Any]:
    """The ``dependencies`` mapping from a script-node payload."""
    value = payload_dict.get("dependencies", {})
    return value if isinstance(value, dict) else {}


def response(dep: Any) -> str:
    """A dependency node's response string ('' when absent or non-string)."""
    if isinstance(dep, dict):
        resp = dep.get("response", "")
        return resp if isinstance(resp, str) else json.dumps(resp)
    return ""


def terminal(text: str) -> str:
    """Peel sub-ensemble envelope layers (deliverable / output / results) to
    the terminal node's raw output."""
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


def extract_code(text: str) -> str:
    """The code deliverable from a (possibly chatty) seat response.

    Non-shell fenced blocks joined; falls back to all fences, then to the
    raw text. The ONE set of semantics for the gate, the envelope, and
    emit — divergent copies meant the gate could approve code that was not
    the code shipped.
    """
    tagged = _FENCE_RE.findall(text)
    blocks = [
        body for lang, body in tagged if lang.lower() not in SHELL_LANGS
    ] or [body for _, body in tagged]
    if blocks:
        return "\n".join(block.strip() for block in blocks)
    return text.strip()
