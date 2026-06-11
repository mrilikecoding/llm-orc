"""Spike π — candidate FormGate implementations (ADR-035 §Decision 4 escalation).

Mirrors the artifact_bridge.py FormGate seam. The real seam contract is
``Callable[[content, destination_tool], content]``; the parse-check arm needs
the *destination path* (extension), so these gates take ``destination_path`` —
the seam extension ADR-035's parse-check escalation requires. A gate either
returns content unchanged (pass) or raises FormRefusedError (refuse → the
Terminal degrades to a dispatch-failure completion, FC-57).

None of these gates extracts, paraphrases, or summarizes (Spike χ F-χ.1 rejected
heuristic extraction). They only *recognize* a clearly-wrong deliverable.
"""

from __future__ import annotations

import ast
import json
import os


class FormRefusedError(Exception):
    """A gate refused a clearly-wrong deliverable (mirror of the real channel)."""


def _ext(destination_path: str) -> str:
    return os.path.splitext(destination_path)[1].lower()


# --- Arm B: parse-check (destination-validity) -----------------------------

def parse_check_gate(content: str, destination_path: str) -> str:
    """Validate the deliverable parses as what its destination path claims.

    ``.py`` -> ast.parse; ``.json`` -> json.loads; everything else (``.md``,
    unknown) -> pass-through (no parser; the deliverable's form is not
    structurally checkable, so the gate does not inspect it). Refuses on a
    parse/validity failure.
    """
    ext = _ext(destination_path)
    if ext == ".py":
        try:
            ast.parse(content)
        except SyntaxError as exc:
            raise FormRefusedError(
                f"{destination_path}: not valid Python ({exc.msg})"
            ) from exc
    elif ext == ".json":
        try:
            json.loads(content)
        except json.JSONDecodeError as exc:
            raise FormRefusedError(
                f"{destination_path}: not valid JSON ({exc})"
            ) from exc
    return content


# --- Arm C: fence-only (content-only) --------------------------------------

def fence_only_gate(content: str, destination_path: str) -> str:
    """Refuse if any markdown code fence is present (the directive forbade fences).

    Content-only; ignores the destination path. This is what surfaces the
    markdown-deliverable problem: a legitimate README carries fenced examples.
    """
    if "```" in content:
        raise FormRefusedError(f"{destination_path}: markdown fence present")
    return content


# --- Arm D: marker-detection (ADR-035 §4 literal text) ----------------------

# Common LLM scaffolding-prose openers + the §4 "prose-scaffolding markers".
# Independently justifiable (these are stock explanation phrases), NOT
# reverse-engineered from the corpus.
_PROSE_MARKERS = (
    "here is the",
    "here's the",
    "this script",
    "this code",
    "this function",
    "this implementation",
    "the following",
    "below is",
    "note that",
    "you can",
    "i hope this",
    "let me know",
    "feel free",
)


def marker_detection_gate(content: str, destination_path: str) -> str:
    """Fence + prose-scaffolding markers (ADR-035 §4 as literally written).

    The heuristic pole the determinism principle distrusts; measured here for
    false-positives on legitimate comments/docstrings and markdown prose.
    """
    if "```" in content:
        raise FormRefusedError(f"{destination_path}: markdown fence present")
    low = content.lower()
    for marker in _PROSE_MARKERS:
        if marker in low:
            raise FormRefusedError(
                f"{destination_path}: prose-scaffolding marker {marker!r}"
            )
    return content


# --- Arm A: pass-through (today's baseline) ---------------------------------

def passthrough_gate(content: str, destination_path: str) -> str:
    return content


GATES = {
    "A_passthrough": passthrough_gate,
    "B_parse_check": parse_check_gate,
    "C_fence_only": fence_only_gate,
    "D_marker": marker_detection_gate,
}
