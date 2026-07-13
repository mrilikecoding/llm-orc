#!/usr/bin/env python3
"""re-fix gather node — assemble the re-fix candidate (rung 2, convergent-fix
design, docs/plans/2026-07-12-convergent-fix-design.md).

Splits classify's dispatch_input into {prior_code, failure_body,
visible_test, target_file, task} and attempts the ONE pinned deterministic
string-literal edit — the narrow, safe case where a
``pytest.raises(..., match=...)`` mismatch names both the expected pattern
and the actual raised message in the captured output. When the failure
isn't that exact shape, or the actual value can't be unambiguously located
in the prior code, ``needs_model_edit`` signals the fallback (fails CLOSED
to the model edit rather than guess).
"""

from __future__ import annotations

import json
import re
import sys

from _helpers import PRIOR_CODE_MARKER as _PRIOR_CODE_MARKER
from _helpers import latest_ran_block as _latest_ran_block
from _helpers import payload as _payload

_REQUEST_MARKER = "\n\nCurrent request: "

# The narrow, safe pinnable case (turn 13's exact shape, confirmed against
# a real pytest run): pytest.raises(..., match=...) states both the
# expected pattern and the actual raised message in its own failure text —
# no need to re-read the test file to pin the replacement.
_MATCH_MISMATCH_RE = re.compile(
    r"Expected regex:\s*'([^']*)'\s*\n\s*E?\s*Actual message:\s*'([^']*)'"
)

_FILE_RE = re.compile(r"\b([\w./-]+\.py)\b")

# A file block in the rendered context ([wrote ...] or [read ...]) — mirrors
# accept_gather.py's _FILE_HEADER_RE/_workspace: body lines carry a two-
# space indent the renderer added, so a header lookalike in untrusted
# content can never be mistaken for a real header (fenced block grammar).
_FILE_HEADER_RE = re.compile(
    r"^assistant: \[(?:wrote|read) ([^\]]+?)( \((?:truncated|failed|oversize)\))?\]$"
)


def _raw_input(payload: dict[str, object]) -> str:
    for key in ("input_data", "input"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _split_dispatch_input(text: str) -> tuple[str, str, str]:
    """(conversation, prior_code, task) from classify's composed
    dispatch_input — the PRIOR_CODE_MARKER sentinel splits prior_code out
    from the conversation (mirrors accept_gather's HELD_TESTS_MARKER
    split); the trailing "Current request:" marker splits the clean task
    out from the rest, exactly as accept_gather does."""
    conversation, task = text, ""
    prior_code = ""
    if _PRIOR_CODE_MARKER in text:
        conversation, rest = text.split(_PRIOR_CODE_MARKER, 1)
        conversation = conversation.strip()
        # the marker's own trailing newline is formatting, not content —
        # strip only that, never the write's own trailing whitespace, so
        # the candidate stays byte-identical to the fix pass's write apart
        # from the pinned literal
        rest = rest.removeprefix("\n")
        if _REQUEST_MARKER in rest:
            prior_code, task = rest.split(_REQUEST_MARKER, 1)
        else:
            prior_code = rest
    elif _REQUEST_MARKER in text:
        conversation, task = text.split(_REQUEST_MARKER, 1)
        conversation = conversation.strip()
    return conversation, prior_code, task.strip()


def _visible_test(context: str) -> str:
    """The latest ``[read test_*.py]`` block's body — the visible test
    rung 1.5 read, when it fired. "" when none exists (rung 1.5 skipped, or
    the failure came from a wider client-run suite)."""
    lines = context.splitlines()
    index = 0
    latest = ""
    while index < len(lines):
        header = _FILE_HEADER_RE.match(lines[index])
        index += 1
        if not header:
            continue
        body_lines: list[str] = []
        while index < len(lines):
            line = lines[index]
            if line.startswith("  "):
                body_lines.append(line[2:])
            elif not line.strip():
                body_lines.append("")
            else:
                break
            index += 1
        name = header.group(1).rsplit("/", 1)[-1]
        if not header.group(2) and name.startswith("test_") and name.endswith(".py"):
            latest = "\n".join(body_lines).strip()
    return latest


def _deterministic_edit(prior_code: str, failure_body: str) -> str:
    """The pinned string-literal replacement, or "" when the failure isn't
    the pinnable match-mismatch shape, or the actual value can't be
    unambiguously located in the source — fails CLOSED to the model edit
    (never guesses which occurrence to replace)."""
    match = _MATCH_MISMATCH_RE.search(failure_body)
    if not match:
        return ""
    expected, actual = match.group(1), match.group(2)
    if not actual or prior_code.count(actual) != 1:
        return ""
    return prior_code.replace(actual, expected, 1)


def main() -> None:
    payload = _payload(sys.stdin.read().strip())
    text = _raw_input(payload)
    conversation, prior_code, task = _split_dispatch_input(text)

    run = _latest_ran_block(conversation)
    failure_body = run[3] if run else ""
    visible_test = _visible_test(conversation)
    file_match = _FILE_RE.search(task)
    target_file = file_match.group(1).rsplit("/", 1)[-1] if file_match else ""

    deterministic_code = _deterministic_edit(prior_code, failure_body)

    print(
        json.dumps(
            {
                "prior_code": prior_code,
                "failure_body": failure_body,
                "visible_test": visible_test,
                "target_file": target_file,
                "task": task,
                "deterministic_code": deterministic_code,
                "needs_model_edit": not deterministic_code,
            }
        )
    )


if __name__ == "__main__":
    main()
