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

import re

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
