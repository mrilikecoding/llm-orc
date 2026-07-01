#!/usr/bin/env python3
"""Deterministic fixture-replay "architect" for the Ω-loop validation.

Stands in for the frontier architect so the engine's bounded-loop primitive can
be validated WITHOUT spending frontier tokens. It is a state machine, not a
reasoner: on a fresh request it emits the `fresh_fixture` contract; once it sees
the coherence gate's rejection feedback carried into its input, it emits the
`repair_fixture`. Pointing both at the same incoherent fixture models a
non-converging architect (the loop must then trip its bound).

I/O matches the real architect's contract (scratch/spike-omega-e/run_e.py): the
input text arrives via the engine's INPUT_TEXT env var (the loop's per-iteration
input — the task on iteration 1, the carried feedback after), fixtures arrive
via AGENT_PARAMETERS, and the output is {"deliverables": [...]} on stdout. So the
gate node and the loop work unchanged when the live architect is swapped in.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REJECTION_MARKER = "REJECTED by the coherence gate"


def _input_text() -> str:
    """The per-iteration input text (task, or carried feedback on repair)."""
    text = os.environ.get("INPUT_TEXT")
    if text is not None:
        return text
    # Fallback: read ScriptAgentInput from stdin.
    try:
        payload = json.loads(sys.stdin.read())
    except (json.JSONDecodeError, ValueError):
        return ""
    return str(payload.get("input_data", "")) if isinstance(payload, dict) else ""


def _resolve_fixture(ref: str) -> Path:
    """Resolve a fixture ref: absolute as-is, else relative to the repo root."""
    path = Path(ref)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[3] / path


def main() -> None:
    params = json.loads(os.environ.get("AGENT_PARAMETERS", "{}"))
    is_repair = REJECTION_MARKER in _input_text()
    fixture = params["repair_fixture"] if is_repair else params["fresh_fixture"]
    deliverables = json.loads(_resolve_fixture(fixture).read_text())
    print(json.dumps({"deliverables": deliverables}))


if __name__ == "__main__":
    main()
