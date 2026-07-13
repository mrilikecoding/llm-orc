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

from _helpers import latest_ran_block as _latest_ran_block

_NO_TESTS_RE = re.compile(r"\bno tests ran\b", re.IGNORECASE)
# pytest's authoritative summary is its LAST line and always carries the
# "in N.NNs" duration anchor — counts parse from that line ONLY, so
# count-shaped text in captured stdout ("0 failed so far") can never shadow
# the real result (review finding 2026-07-09). The last few lines are
# scanned, latest match wins, to tolerate a client-appended trailer.
_SUMMARY_SHAPE_RE = re.compile(
    r"\b(?:\d+ (?:passed|failed|errors?|skipped|deselected|xfailed|xpassed"
    r"|warnings?)|no tests ran)\b.*\bin [\d.]+s\b",
    re.IGNORECASE,
)
_FAILING_LINE_RE = re.compile(r"^(?:FAILED|ERROR)\b")
_MAX_FAILING_LINES = 5
_TAIL_LINES = 10
_SUMMARY_SCAN_LINES = 3


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


def _summary_line(body: str) -> str:
    """pytest's summary line, or "" when the output carries none."""
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    for line in reversed(lines[-_SUMMARY_SCAN_LINES:]):
        if _SUMMARY_SHAPE_RE.search(line):
            return line
    return ""


def _count(pattern: str, summary: str) -> int | None:
    match = re.search(pattern, summary)
    return int(match.group(1)) if match else None


def _verdict(command: str, variant: str, detail: str, body: str) -> str:
    if variant == "failed":
        reason = detail or "empty run result"
        return f"The test run could not execute: {reason}"
    summary = _summary_line(body)
    if not summary:
        # summary-less bodies only — the duration-anchored summary line
        # stays authoritative, so phrase-shaped stdout from a REAL run can
        # never shadow the result (PR #115 review, both rounds)
        if "rejected permission" in body.lower():
            return f"The test run was not permitted by the client (`{command}`)."
        tail = "\n".join(body.splitlines()[-_TAIL_LINES:])
        return (
            f"Ran `{command}`, but the output carried no pytest summary. "
            f"Output tail:\n{tail}"
        )
    if _NO_TESTS_RE.search(summary):
        return f"Ran `{command}`: no tests ran."
    failed = _count(r"\b(\d+) failed\b", summary)
    passed = _count(r"\b(\d+) passed\b", summary)
    errors = _count(r"\b(\d+) errors?\b", summary)
    if failed is None and passed is None and errors is None:
        # a summary with none of the verdict-bearing counts (all skipped /
        # deselected) — echo pytest's own line rather than claiming no summary
        return f"Ran `{command}`: {summary}"
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
    run = _latest_ran_block(text)
    if run is None:
        primary = "No test-run output found for this turn."
    else:
        primary = _verdict(*run)
    print(json.dumps({"status": "ok", "primary": primary}))


if __name__ == "__main__":
    main()
