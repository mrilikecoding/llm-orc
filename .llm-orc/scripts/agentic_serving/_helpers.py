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

import ast
import json
import re
from typing import Any

# Fences tagged with a shell-ish language are usage examples, not code.
SHELL_LANGS = {"bash", "sh", "shell", "console", "zsh", "text"}

# The TDD retry sentinel (issue #100): a rejected round whose tests collected
# and were judged adequate carries them under this marker; route dispatches
# the held round on it, and gather reads the tests back out. One constant —
# the envelope writes it, route and gather read it.
HELD_TESTS_MARKER = "[HELD TESTS: round 1 spec; regenerate ONLY the code]"

# The convergent-fix sentinel (rung 2, docs/plans/2026-07-12-convergent-fix-
# design.md): classify composes the fix-led write's content under this
# marker in the re-fix dispatch_input; refix_gather reads it back out. One
# constant — classify writes it, refix_gather reads it.
PRIOR_CODE_MARKER = "[PRIOR CODE: this turn's write, before the re-fix]"

_FENCE_RE = re.compile(r"```([a-zA-Z0-9_+-]*)\n(.*?)```", re.DOTALL)

# A ``[ran <command>]`` block header, with its optional failed/truncated
# variant and any inline trailing detail — shared by run_verdict (verdict
# prose) and classify (rung 2's failure-shape routing signal), issue #83 /
# convergent-fix rung 2.
_RAN_HEADER_RE = re.compile(r"^assistant: \[ran (.+?)( \((failed|truncated)\))?\](.*)$")


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


def extract_code(text: str, *, drop_test_blocks: bool = False) -> str:
    """The code deliverable from a (possibly chatty) seat response.

    Non-shell fenced blocks joined; falls back to all fences, then to the
    raw text. The ONE set of semantics for the gate, the envelope, and
    emit — divergent copies meant the gate could approve code that was not
    the code shipped.

    ``drop_test_blocks`` is for the CODE consumers only: seat models
    sometimes emit the code and a copy of the tests as two fences, and
    joining them ships a file with the test suite embedded. Pure-test
    blocks are dropped when a non-test block exists; the tests extraction
    never sets this (test blocks are its point).
    """
    tagged = _FENCE_RE.findall(text)
    blocks = [
        body for lang, body in tagged if lang.lower() not in SHELL_LANGS
    ] or [body for _, body in tagged]
    if drop_test_blocks and len(blocks) > 1:
        non_test = [block for block in blocks if not _is_pure_test_block(block)]
        blocks = non_test or blocks
    if blocks:
        return "\n".join(block.strip() for block in blocks)
    return text.strip()


def latest_ran_block(text: str) -> tuple[str, str, str, str] | None:
    """(command, variant, inline detail, body) of the LAST ``[ran ...]``
    block in ``text``, or ``None`` when no block is present.

    The body is the block's two-space-indented lines, de-indented (a blank
    line inside the body stays blank); untrusted output text can never be
    confused with block headers since headers live only at column 0 (fenced
    block grammar). Shared by run_verdict's verdict-prose parse and
    classify's rung-2 failure-shape classification — moved out of
    run_verdict so both read the identical block shape (no duplicate parser
    to drift).
    """
    lines = text.splitlines()
    found: tuple[int, re.Match[str]] | None = None
    for index, line in enumerate(lines):
        match = _RAN_HEADER_RE.match(line)
        if match:
            found = (index, match)
    if found is None:
        return None
    index, match = found
    body_lines: list[str] = []
    for line in lines[index + 1 :]:
        if line.startswith("  "):
            body_lines.append(line[2:])
        elif not line.strip():
            body_lines.append("")
        else:
            break
    command = match.group(1)
    variant = match.group(3) or ""
    detail = (match.group(4) or "").strip()
    return command, variant, detail, "\n".join(body_lines).strip()


def _is_pure_test_block(block: str) -> bool:
    """A parsed block whose top-level defs/classes are all test-named
    (imports and docstrings allowed). Non-parsing blocks are kept."""
    try:
        tree = ast.parse(block)
    except SyntaxError:
        return False
    named = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    if not named:
        return False
    return all(
        node.name.startswith("test_") or node.name.startswith("Test")
        for node in named
    )
