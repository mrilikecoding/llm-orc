#!/usr/bin/env python3
"""Serving Ensemble — classify (decider) node.

Emits the routing decision the dispatch seat resolves:

    {"target", "kind", "file", "dispatch_input", "build"}

Routing is deterministic where the signal is structural: an explain-shaped turn
is non-build prose; a turn with a build verb or a named target file is a build
routed to the default code-generation seat. The build-vs-non-build (executable-
deliverable) determination is classify's own responsibility (ADR-046 §1
responsibility matrix) — ``build`` gates marshal's file-vs-prose shaping and, at
WP-D8, the gated accept shape.

When neither structural signal resolves the turn, classify does NOT guess a
default seat: it emits ``needs_decider: true`` and leaves ``target`` empty, so a
guarded model-backed ``decide`` node reads the turn intent and a ``resolve`` node
merges the two (scenarios.md "classify reads intent with a model-backed decider
when the signal is not structural"; ADR-046 §1, classify is the decider seat).
Determinism is preserved: the model runs only on the guarded ambiguous path, its
output is a closed target set, and an unresolved target fails at dispatch.

The seat is filled by dynamic dispatch on ``${resolve.target}``, so swapping a
seat strategy is a change to this decision or the operator default, never to the
skeleton (AS-11).
"""

from __future__ import annotations

import json
import re
import sys

_EXPLAIN_MARKERS = (
    "explain",
    "what does",
    "how does",
    "describe",
    "summarize",
    "why does",
    "what is",
    "tell me",
)
_DEFAULT_CODE_SEAT = "code-seat"
_EXPLAIN_SEAT = "explainer"
_FILE_RE = re.compile(
    r"\b([\w./-]+\.(?:py|js|ts|jsx|tsx|json|md|txt|ya?ml|sh|go|rs|java|c|cpp|h))\b"
)
# A structural build signal: an imperative verb that asks for code to be
# produced or changed. Word-boundaried so "add" does not fire on "address".
_BUILD_RE = re.compile(
    r"\b(write|implement|create|build|generate|refactor|fix|add|code)\b",
    re.IGNORECASE,
)


def _extract_file(task: str) -> str:
    """A structural filename signal from the turn (e.g. 'in add.py')."""
    match = _FILE_RE.search(task)
    return match.group(1) if match else ""


def _turn(raw: str) -> dict:
    """Recover the turn dict from the ScriptAgent wrapper or a bare task.

    A no-dependency phase-0 script receives ``{"input": "<turn>", ...}``; a
    dependent script receives ``{"input_data": "<turn>", "dependencies": {...}}``.
    Handle both keys plus a bare turn dict for direct use.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"task": raw}
    if not isinstance(data, dict):
        return {"task": str(data)}
    inner = data.get("input_data")
    if inner is None:
        inner = data.get("input")
    if inner is None:
        inner = data
    if isinstance(inner, dict):
        return inner
    if isinstance(inner, str):
        try:
            parsed = json.loads(inner)
            return parsed if isinstance(parsed, dict) else {"task": inner}
        except json.JSONDecodeError:
            return {"task": inner}
    return {"task": ""}


def main() -> None:
    turn = _turn(sys.stdin.read().strip())
    task = str(turn.get("task", "")).strip()
    is_explain = any(marker in task.lower() for marker in _EXPLAIN_MARKERS)
    named_file = turn.get("file") or _extract_file(task)
    has_build_signal = bool(named_file) or bool(_BUILD_RE.search(task))

    if is_explain:
        target, kind, build, needs_decider = _EXPLAIN_SEAT, "explanation", False, False
    elif has_build_signal:
        target = _DEFAULT_CODE_SEAT
        kind = turn.get("kind", "python_module")
        build, needs_decider = True, False
    else:
        # No structural signal — hand the routing to the guarded model decider.
        target, kind, build, needs_decider = "", "", False, True

    file = named_file or "solution.py"

    print(
        json.dumps(
            {
                "target": target,
                "kind": kind,
                "file": file,
                "dispatch_input": task or turn.get("dispatch_input", ""),
                "build": build,
                "needs_decider": needs_decider,
            }
        )
    )


if __name__ == "__main__":
    main()
