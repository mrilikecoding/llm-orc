"""Offset-continuation read stitching (#153).

Pure text functions over OpenCode read-tool wire results: trailer
parsing, per-part normalization, and the stitcher that reassembles a
50KB-capped multi-part read into the whole file — or refuses.

Design: docs/plans/2026-08-14-offset-reads-design.md (v1.1). The load-
bearing honesty rule is POSITIVE completeness (pre-flight blocker 2): a
stitch is complete only when its final part carries the wire's own
``(End of file - total N lines)`` trailer AND N equals the stitched line
count. An unrecognized cap-trailer variant (the unverified 2000-line-cap
wording, or a silently line-capped continuation part) therefore fails
closed instead of rendering a corrupt whole.

The wire-shape regexes here mirror ``serving_ensemble_caller``'s
single-read normalizer (this module must stay a leaf — the caller
imports it); a drift pin test asserts the patterns stay equal.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

_CONTENT_TAG_RE = re.compile(r"<content>(.*)</content>", re.DOTALL)
_END_OF_FILE_TRAILER_RE = re.compile(r"^\(End of file - total (\d+) lines?\)$")
_OPENCODE_GUTTER_RE = re.compile(r"^\d+: ?")
# The captured cap trailer (gate-recall-ledger-run2.jsonl, 2026-08-13),
# with the line range and continuation offset captured.
_CAP_TRAILER_RE = re.compile(
    r"^\(Output capped at .+\. Showing lines (\d+)-(\d+)\. "
    r"Use offset=(\d+) to continue\.\)$",
    re.MULTILINE,
)


def parse_cap_trailer(raw: str) -> tuple[int, int, int] | None:
    """(showing_start, showing_end, continue_offset) from a capped read
    result's trailer, or ``None`` when the result carries no recognized
    cap trailer."""
    match = _CAP_TRAILER_RE.search(raw or "")
    if not match:
        return None
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def _part_lines(raw: str) -> tuple[list[str], int | None]:
    """(source lines, eof_total) for one wire part: content-tag extract,
    cap-trailer strip BEFORE the gutter-uniformity check (pre-flight
    minor 5 — the ungutted trailer line otherwise defeats it), EOF
    trailer strip (its total captured), then the gutter strip."""
    match = _CONTENT_TAG_RE.search(raw or "")
    body = match.group(1) if match else (raw or "")
    lines = body.strip().splitlines()
    lines = [line for line in lines if not _CAP_TRAILER_RE.match(line.strip())]
    eof_total: int | None = None
    kept: list[str] = []
    for line in lines:
        eof = _END_OF_FILE_TRAILER_RE.match(line.strip())
        if eof:
            eof_total = int(eof.group(1))
            continue
        kept.append(line)
    while kept and not kept[-1].strip():
        kept.pop()
    non_empty = [line for line in kept if line.strip()]
    if non_empty and all(_OPENCODE_GUTTER_RE.match(line) for line in non_empty):
        kept = [_OPENCODE_GUTTER_RE.sub("", line, count=1) for line in kept]
    return kept, eof_total


def stitch_parts(parts: list[tuple[int, str]]) -> str | None:
    """The complete stitched source for ``[(offset_param, raw_result)]``,
    or ``None`` (refuse). Rules (design v1.1): latest result per offset,
    ascending order, contiguity (each capped part's continue-offset
    equals the next part's offset param), and POSITIVE completeness —
    the final part carries the EOF trailer and its total equals the
    stitched line count."""
    if not parts:
        return None
    latest: dict[int, str] = {}
    for offset, raw in parts:
        latest[offset] = raw
    ordered = sorted(latest.items())
    stitched: list[str] = []
    for index, (offset, raw) in enumerate(ordered):
        cap = parse_cap_trailer(raw)
        lines, eof_total = _part_lines(raw)
        if index < len(ordered) - 1:
            next_offset = ordered[index + 1][0]
            if not _middle_part_valid(cap, offset, next_offset, eof_total):
                return None
            stitched.extend(lines)
            continue
        # final part: POSITIVE completeness — the EOF trailer must be
        # present and its total must equal the stitched line count.
        stitched.extend(lines)
        if cap is not None or eof_total != len(stitched):
            return None
    return "\n".join(stitched)


def _middle_part_valid(
    cap: tuple[int, int, int] | None,
    offset: int,
    next_offset: int,
    eof_total: int | None,
) -> bool:
    """A non-final part must be capped, its showing-start must equal its
    own requested offset, its continue-offset must be exactly the next
    part's offset (contiguity), and it must not carry an EOF trailer."""
    if cap is None or eof_total is not None:
        return False
    showing_start, _showing_end, continue_offset = cap
    return showing_start == offset and continue_offset == next_offset


# --- wire orchestration (the caller imports these back) ---------------------
# These live here rather than in serving_ensemble_caller because the
# caller's own rendered read block borders the session token budget (the
# whale-pin economics; the grep_render extraction is the precedent). The
# tiny message-shape helpers below are duplicated deliberately as leaf
# copies; the caller's originals remain the single home for every other
# consumer and the shapes are pinned by the caller's own suites.

# At most this many read CALLS per path per turn (the 96KB serve window
# needs 2 parts at the client's 50KB cap; one spare).
_READ_PART_BOUND = 3
_READ_FILE_CAP = 98304  # mirror of the caller's cap; drift-pinned below


def _parsed_read_arguments(call: Any) -> dict[str, Any] | None:
    function = call.get("function", {}) if isinstance(call, dict) else {}
    raw = function.get("arguments")
    if not isinstance(raw, str):
        return None
    try:
        arguments = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(arguments, dict):
        return None
    if not arguments.get("filePath") or "command" in arguments:
        return None
    if "content" in arguments:
        return None
    return arguments


def _read_call_info(messages: Sequence[Any]) -> dict[str, tuple[str, int]]:
    """tool_call_id -> (filePath, offset) for every read-shaped call; a
    call without an ``offset`` param reads from line 1."""
    info: dict[str, tuple[str, int]] = {}
    for message in messages:
        for call in getattr(message, "tool_calls", ()) or ():
            arguments = _parsed_read_arguments(call)
            if arguments is not None and isinstance(call, dict) and call.get("id"):
                try:
                    offset = int(arguments.get("offset", 1) or 1)
                except (TypeError, ValueError):
                    offset = 1
                info[str(call["id"])] = (str(arguments["filePath"]), offset)
    return info


def _read_part_groups(
    messages: Sequence[Any],
) -> tuple[dict[tuple[int, str], list[tuple[int, str]]], list[tuple[int, str]]]:
    """Read results grouped per (turn segment, path) with their offsets,
    plus first-occurrence order — a segment is the span between
    consecutive user messages (#153 design v1.1: parts from different
    segments never stitch)."""
    call_info = _read_call_info(messages)
    groups: dict[tuple[int, str], list[tuple[int, str]]] = {}
    order: list[tuple[int, str]] = []
    segment = 0
    for message in messages:
        content = getattr(message, "content", None)
        if getattr(message, "role", None) == "user" and (content or "").strip():
            segment += 1
            continue
        if getattr(message, "role", None) != "tool":
            continue
        info = call_info.get(getattr(message, "tool_call_id", None) or "")
        if not info:
            continue
        path, offset = info
        key = (segment, path)
        groups.setdefault(key, []).append((offset, content or ""))
        if key not in order:
            order.append(key)
    return groups, order


def _last_user_index(messages: Sequence[Any]) -> int:
    items = list(messages)
    for index in range(len(items) - 1, -1, -1):
        content = getattr(items[index], "content", None)
        if getattr(items[index], "role", None) == "user" and (content or "").strip():
            return index
    return -1


def _read_continuation(messages: Sequence[Any]) -> tuple[str, int] | None:
    """(path, offset) for the next offset-continuation read (#153), or
    ``None``. THIS turn's read results only; for each path whose LAST
    part carries the cap trailer: the READ CALL COUNT for the path this
    turn must be under ``_READ_PART_BOUND`` (a call count, never a dict
    keyed by offset — a non-conforming client repeating a window would
    freeze that and spin, pre-flight blocker 1), the trailer's
    continue-offset must exceed the part's own offset param
    (monotonicity), and the trailer's showing-start must equal the
    requested offset — any violation stops continuing so the render
    refuses instead."""
    items = list(messages)
    post_user = items[_last_user_index(items) + 1 :]
    call_info = _read_call_info(post_user)
    call_counts: dict[str, int] = {}
    for _call_id, (path, _offset) in call_info.items():
        call_counts[path] = call_counts.get(path, 0) + 1
    last_part: dict[str, tuple[int, str]] = {}
    for message in post_user:
        if getattr(message, "role", None) != "tool":
            continue
        info = call_info.get(getattr(message, "tool_call_id", None) or "")
        if info:
            path, offset = info
            last_part[path] = (offset, getattr(message, "content", None) or "")
    for path, (offset, raw) in last_part.items():
        continuation = _continuation_offset(raw, offset, call_counts.get(path, 0))
        if continuation is not None:
            return path, continuation
    return None


def _continuation_offset(raw: str, offset: int, call_count: int) -> int | None:
    """The validated continuation offset for one capped part, or ``None``:
    under the call-count bound, monotonic (continue-offset > the part's
    own offset), and showing-start equals the requested offset."""
    cap = parse_cap_trailer(raw)
    if cap is None or call_count >= _READ_PART_BOUND:
        return None
    showing_start, _showing_end, continue_offset = cap
    if continue_offset <= offset or showing_start != offset:
        return None
    return continue_offset


def _render_stitched_read_block(path: str, stitched: str) -> tuple[str, bool]:
    """A complete stitched source rendered through the read-block tail —
    the per-file cap applies to the WHOLE, and the block enters the same
    token-budget accounting as any single read (budget parity)."""
    if not stitched.strip():
        return f"assistant: [read {path} (failed)] empty read result", False
    if len(stitched) > _READ_FILE_CAP:
        return f"assistant: [read {path} (oversize)]", False
    body = "\n".join(f"  {line}" for line in stitched.splitlines())
    return f"assistant: [read {path}]\n{body}", True
