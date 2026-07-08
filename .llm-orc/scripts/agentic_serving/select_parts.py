#!/usr/bin/env python3
"""gen -> review exemplar shape — deterministic slot selector (WP-C8, scenario 4).

Fills the shape's two slots with runtime-chosen REGISTERED parts. It reads the
Topaz-keyed capability registry and picks the registered building-block part for
each slot's capability (gen=code_generation, review=logical_reasoning); the shape's
gen/review nodes then dynamic-dispatch ``${select_parts.gen}`` / ``${select_parts.
review}`` onto those parts. Selection is deterministic (first registered part per
key, the registry sorts) and external to any capability ensemble's self-routing
(Strategy A, ADR-047 §2). A slot with no registered part resolves empty so the
dispatch fails deterministically rather than guessing (determinism-over-carve-outs).

This is the shape's declarative composition made concrete: the slot -> capability
map lives here (the shape's definition), the parts come from the registry at
runtime. An operator authors a shape by adding a skeleton + a selector, never the
engine (AS-11).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from llm_orc.core.serving.capability_registry import capability_parts

_REGISTRY_DIR = Path(__file__).resolve().parents[2] / "ensembles" / "agentic-serving"

# The shape's declarative slot -> capability map (its composition definition).
_SLOTS = {"gen": "code_generation", "review": "logical_reasoning"}


def _task(raw: str) -> str:
    """The turn text from the ScriptAgent wrapper (``input``/``input_data``) or a
    bare task string."""
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
        inner = data
    if isinstance(inner, dict):
        return str(inner.get("task", ""))
    return str(inner)


def _pick(parts: dict[str, list[str]], skill: str) -> str:
    candidates = parts.get(skill, [])
    return candidates[0] if candidates else ""


def main() -> None:
    task = _task(sys.stdin.read().strip())
    parts = capability_parts(_REGISTRY_DIR)
    selected = {slot: _pick(parts, skill) for slot, skill in _SLOTS.items()}
    selected["task"] = task
    print(json.dumps(selected))


if __name__ == "__main__":
    main()
