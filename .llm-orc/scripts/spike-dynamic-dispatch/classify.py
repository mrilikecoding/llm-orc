#!/usr/bin/env python3
"""Dynamic-dispatch skeleton — classify (decider) stage.

Reads the turn and emits the routing decision the dispatch seat resolves:
    {"target": "<seat ensemble name>", "kind": "...", "file": "...",
     "dispatch_input": "<payload the seat receives via input_key>"}

Routing is deterministic (F4: within-category routing is often a lookup, not an
inference). An explicit ``seat`` override wins (used to force a seat strategy
for swap probes); else an explain-shaped turn routes to the explain seat; else
it defaults to the solo code seat. The seat is filled by dynamic dispatch on
``${classify.target}``: swapping the seat strategy is a change to this decision
or the operator default, never to the skeleton.
"""

import json
import sys

_EXPLAIN_MARKERS = (
    "explain",
    "what does",
    "how does",
    "describe",
    "summarize",
    "why does",
)
_DEFAULT_CODE_SEAT = "dd-seat-code-solo"
_EXPLAIN_SEAT = "dd-seat-explain"


def _turn(raw: str) -> dict:
    """Recover the turn dict from the script wrapper or a bare task.

    A no-dependency phase-0 script receives the ScriptAgent wrapper
    ``{"input": "<turn>", ...}``; a dependent script receives the dependency
    envelope ``{"input_data": "<turn>", "dependencies": {...}}``. Handle both
    keys (as the other spike scripts do), plus a bare turn dict for direct use.
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
    override = turn.get("seat")
    is_explain = any(m in task.lower() for m in _EXPLAIN_MARKERS)

    if override:
        target = override
        kind = turn.get("kind", "explanation" if is_explain else "python_module")
    elif is_explain:
        target = _EXPLAIN_SEAT
        kind = "explanation"
    else:
        target = _DEFAULT_CODE_SEAT
        kind = turn.get("kind", "python_module")

    print(
        json.dumps(
            {
                "target": target,
                "kind": kind,
                "file": turn.get("file", "solution.py"),
                "dispatch_input": task or turn.get("dispatch_input", ""),
            }
        )
    )


if __name__ == "__main__":
    main()
