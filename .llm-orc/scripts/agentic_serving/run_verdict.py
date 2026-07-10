#!/usr/bin/env python3
"""run-verdict shape — deterministic test-run verdict node (issue #83).

Extracts the latest ``assistant: [ran <command>]`` block from the dispatched
context and composes an honest verdict from pytest's own summary text. Fully
deterministic — a "run the tests" turn costs zero model calls end to end.
The parser reads the block BODY (two-space indented by the caller's render;
the indent is stripped here), so untrusted output text can never be confused
with block headers.
"""

from __future__ import annotations

import json
import re
import sys

_RAN_HEADER_RE = re.compile(r"^assistant: \[ran (.+?)( \((failed|truncated)\))?\](.*)$")
_NO_TESTS_RE = re.compile(r"\bno tests ran\b", re.IGNORECASE)
_FAILING_LINE_RE = re.compile(r"^(?:FAILED|ERROR)\b")
_MAX_FAILING_LINES = 5
_TAIL_LINES = 10


def _dispatch_text(raw: str) -> str:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if not isinstance(data, dict):
        return str(data)
    inner = data.get("input_data")
    if inner is None:
        inner = data.get("input")
    if inner is None:
        return ""
    return inner if isinstance(inner, str) else json.dumps(inner)


def _latest_run(text: str) -> tuple[str, str, str, str] | None:
    """(command, variant, inline detail, body) of the LAST run block."""
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


def _count(pattern: str, body: str) -> int | None:
    match = re.search(pattern, body)
    return int(match.group(1)) if match else None


def _verdict(command: str, variant: str, detail: str, body: str) -> str:
    if variant == "failed":
        reason = detail or "empty run result"
        return f"The test run could not execute: {reason}"
    if _NO_TESTS_RE.search(body):
        return f"Ran `{command}`: no tests ran."
    failed = _count(r"\b(\d+) failed\b", body)
    passed = _count(r"\b(\d+) passed\b", body)
    errors = _count(r"\b(\d+) errors?\b", body)
    if failed is None and passed is None and errors is None:
        tail = "\n".join(body.splitlines()[-_TAIL_LINES:])
        return (
            f"Ran `{command}`, but the output carried no pytest summary. "
            f"Output tail:\n{tail}"
        )
    counts = ((failed, "failed"), (errors, "errored"), (passed, "passed"))
    parts = [f"{count} {label}" for count, label in counts if count]
    verdict = f"Ran `{command}`: {', '.join(parts) or '0 tests'}."
    if failed or errors:
        failing = [line for line in body.splitlines() if _FAILING_LINE_RE.match(line)]
        if failing:
            verdict += "\n" + "\n".join(failing[:_MAX_FAILING_LINES])
    return verdict


def main() -> None:
    text = _dispatch_text(sys.stdin.read().strip())
    run = _latest_run(text)
    if run is None:
        primary = "No test-run output found for this turn."
    else:
        primary = _verdict(*run)
    print(json.dumps({"status": "ok", "primary": primary}))


if __name__ == "__main__":
    main()
