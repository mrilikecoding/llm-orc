#!/usr/bin/env python3
"""Architect-coherence gate as an engine script node (Ω-loop; lifts Ω-E).

The deterministic, domain-free contract check from
scratch/spike-omega-e/coherence_gate.py, lifted to a substrate `script:` node so
the engine's bounded-loop primitive can drive the architect-repair cycle
declaratively (instead of the Python `resolve_contract` driver). The gate logic
is reused unchanged — no fork — via sys.path; this node is only the I/O shim.

Reads the upstream architect's output from the ScriptAgentInput `dependencies`
on stdin, runs the gate, and emits the loop's terminal contract on stdout:

    {ok, reasons, contract, next_input}

  - ok         -> drives the loop's `until: ${ok}` (stop when coherent)
  - next_input -> drives the loop's `carry: ${next_input}` (the WHOLE next
                  iteration input). Because `carry` replaces the entire input,
                  next_input must carry the task forward too: it is
                  `task + DELIM + feedback`, so a live architect sees both the
                  original task and the rejection on a repair iteration. The
                  pristine task is recovered from this node's own input_data by
                  splitting on DELIM (the whole thing on iteration 1).

Robust to real LLM output: extracts the JSON object even when fenced/wrapped,
and treats an unparseable/empty contract as a coherence failure (so the loop
repairs or trips its bound instead of false-passing on no contract).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

# Reuse the Ω-E gate logic unchanged (this node is the substrate lift of it).
_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "scratch" / "spike-omega-e"))

from coherence_gate import _format_feedback, coherence_gate  # noqa: E402

DELIM = "\n\n----- COHERENCE GATE FEEDBACK -----\n"


def _payload() -> dict[str, Any]:
    """The ScriptAgentInput dict on stdin (input_data + dependencies)."""
    try:
        data = json.loads(sys.stdin.read())
    except (json.JSONDecodeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _task(payload: dict[str, Any]) -> str:
    """The pristine task: the part of this node's input before the first DELIM."""
    input_data = payload.get("input_data", "")
    return input_data.split(DELIM, 1)[0] if isinstance(input_data, str) else ""


def _architect_response(payload: dict[str, Any]) -> str:
    """The upstream architect node's raw response from the dependencies."""
    arch = payload.get("dependencies", {}).get("architect", {})
    return arch.get("response", "") if isinstance(arch, dict) else ""


def _parse_json(text: str) -> Any:
    """Parse JSON, falling back to the first {...} block in wrapped output."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except (json.JSONDecodeError, ValueError):
                return {}
    return {}


def _contract(architect_response: str) -> list[dict[str, Any]]:
    """Extract the deliverables list from the architect's (maybe wrapped) JSON."""
    parsed = _parse_json(architect_response)
    if isinstance(parsed, list):
        return parsed
    deliverables = parsed.get("deliverables", []) if isinstance(parsed, dict) else []
    return deliverables if isinstance(deliverables, list) else []


def main() -> None:
    payload = _payload()
    contract = _contract(_architect_response(payload))
    if not contract:
        ok, reasons = False, ["architect produced no parseable contract (deliverables)"]
    else:
        ok, reasons = coherence_gate(contract)
    next_input = "" if ok else _task(payload) + DELIM + _format_feedback(reasons)
    print(
        json.dumps(
            {
                "ok": ok,
                "reasons": reasons,
                "contract": contract,
                "next_input": next_input,
            }
        )
    )


if __name__ == "__main__":
    main()
