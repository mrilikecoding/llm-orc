"""Grep wire mechanics — the #121 content-grep round's pattern template,
echo validation, result normalization, and block render.

Extracted from ``serving_ensemble_caller`` (structural, behavior-neutral):
the caller's own rendered read block sits just under the session token
budget, and the whale pin (`test_real_repo_files_admit_or_refuse_at_
current_size`) exists precisely so growth surfaces as a conscious
decision — this module is that decision. Everything here is pure
string/regex work over the wire result; the caller re-exports the names.

Design: docs/plans/2026-08-13-content-grep-design.md. Wire grammar per
the opencode 1.17.15 binary extraction (design resolution 2): 100-match
client cap, header count computed FROM the capped array (never used for
truncation arithmetic), a " (more matches available)" header suffix
and/or a results-truncated footer when cut, "No files found" when empty,
and blank lines between file groups. Render caps (resolution 8): 50
post-filter lines AND a 4,096-char ceiling; any cut marks (truncated).
"""

from __future__ import annotations

import re
from pathlib import Path

# The shared stem charset (the run-command discipline): classify emits
# charset-checked stems; both the glob and grep templates re-assert each
# part before it may enter a pattern. The caller imports these back so
# there is exactly one definition.
_GLOB_STEM_RE = re.compile(r"^[A-Za-z_]\w*$")
_UNTRUSTED_STEM = "untrusted-stem"

_GREP_INCLUDE = "*.py"
_GREP_LINE_CAP = 50
_GREP_CHAR_CAP = 4096
_GREP_EMPTY_RESULT = "No files found"
_GREP_TRUNCATED_FOOTER = (
    "(Results truncated. Consider using a more specific path or pattern.)"
)
_GREP_FOUND_RE = re.compile(r"^Found \d+ matches( \(more matches available\))?$")
_GREP_ROW_RE = re.compile(r"^  Line (\d+): (.*)$")
# Echo validation (the _RUN_COMMAND_RE discipline): the issued pattern is
# reconstructed from the echoed alternation and compared for equality, so
# only a template-shaped echo may name the rendered header's stems.
_GREP_ECHO_STEMS_RE = re.compile(
    r"^\(\?i\)\^\\s\*\(def\|class\)\\s\+\[A-Za-z0-9_\]\*\(([A-Za-z0-9_|]+)\)"
)


def _grep_pattern(grep_stems: str) -> str | None:
    """The closed def-anchored grep pattern template for classify's
    ``grep`` outcome (#121): definition-shaped lines only — def/class
    names or module-level assignment targets CONTAINING a stem, both
    sides optional, case-insensitive. Each comma-joined stem is
    charset-re-asserted before it may enter the template (the
    run-command discipline); an unsafe stem returns ``None`` so the
    caller refuses."""
    parts = grep_stems.split(",")
    if not parts or not all(_GLOB_STEM_RE.match(part) for part in parts):
        return None
    alternation = "|".join(parts)
    return (
        rf"(?i)^\s*(def|class)\s+[A-Za-z0-9_]*({alternation})[A-Za-z0-9_]*"
        rf"|^[A-Za-z0-9_]*({alternation})[A-Za-z0-9_]* *="
    )


def _grep_echo_stems(pattern: str) -> str | None:
    """The comma-joined stems an echoed grep pattern was issued for, or
    ``None`` when the echo does not exactly reconstruct the closed
    template — only a template-shaped echo may enter the rendered
    header."""
    match = _GREP_ECHO_STEMS_RE.match(pattern or "")
    if not match:
        return None
    stems = ",".join(match.group(1).split("|"))
    return stems if _grep_pattern(stems) == pattern else None


def _normalize_grep(
    raw: str, root: Path
) -> tuple[list[tuple[str, int, str]], bool] | None:
    """Client grep output as (rows, wire_truncated), or ``None`` for an
    empty result. Rows are (relativized path, line number, line text) in
    wire order; paths outside ``root`` keep their absolute form."""
    text = (raw or "").strip()
    if not text or text.startswith(_GREP_EMPTY_RESULT):
        return None
    lines = text.splitlines()
    wire_truncated = False
    body = lines
    header = lines[0].strip()
    if _GREP_FOUND_RE.match(header):
        wire_truncated = header.endswith("(more matches available)")
        body = lines[1:]
    rows, footer_truncated = _grep_result_rows(body, str(root).rstrip("/") + "/")
    return rows, wire_truncated or footer_truncated


def _grep_result_rows(
    body: list[str], root_prefix: str
) -> tuple[list[tuple[str, int, str]], bool]:
    """(rows, footer_truncated) parsed from a grep result's group lines —
    per-file groups separated by blank lines, ``  Line N: text`` rows,
    the results-truncated footer recognized anywhere."""
    rows: list[tuple[str, int, str]] = []
    footer_truncated = False
    current_path: str | None = None
    for line in body:
        if line.strip() == _GREP_TRUNCATED_FOOTER:
            footer_truncated = True
            continue
        if not line.strip():
            current_path = None
            continue
        row = _GREP_ROW_RE.match(line)
        if row and current_path:
            rows.append((current_path, int(row.group(1)), row.group(2)))
            continue
        candidate = line.strip()
        if candidate.endswith(":"):
            path = candidate[:-1]
            if path.startswith(root_prefix):
                path = path[len(root_prefix) :]
            current_path = path
    return rows, footer_truncated


def _render_grep_block(pattern: str, raw: str, root: Path) -> str:
    """A grep result as a context block (#121 grammar):
    ``assistant: [grepped <stems>]`` plus two-space-indented
    ``<path>: Line <N>: <text>`` rows, relativized against the workspace
    root (halves the per-line charge — design final-review change 2).
    ANY cut — the wire's own truncation signals or the render caps —
    marks the header ``(truncated)`` so classify's menu semantics can
    hedge honestly; an empty result is a single-line failed variant."""
    stems = _grep_echo_stems(pattern)
    if stems is None:
        return (
            f"assistant: [grepped {_UNTRUSTED_STEM} (failed)] "
            "pattern echo did not match the issued template"
        )
    normalized = _normalize_grep(raw, root)
    if normalized is None:
        return f"assistant: [grepped {stems} (failed)] no definition matches"
    rows, truncated = normalized
    if not rows:
        return f"assistant: [grepped {stems} (failed)] no definition matches"
    rendered: list[str] = []
    chars = 0
    for path, lineno, line_text in rows:
        line = f"  {path}: Line {lineno}: {line_text}"
        if len(rendered) >= _GREP_LINE_CAP or chars + len(line) + 1 > _GREP_CHAR_CAP:
            truncated = True
            break
        rendered.append(line)
        chars += len(line) + 1
    header = (
        f"assistant: [grepped {stems} (truncated)]"
        if truncated
        else f"assistant: [grepped {stems}]"
    )
    return header + "\n" + "\n".join(rendered)
