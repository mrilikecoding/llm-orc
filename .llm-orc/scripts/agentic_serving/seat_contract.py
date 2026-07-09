#!/usr/bin/env python3
"""Serving seat contract — per-seat pass/fail admission (WP-E8; ADR-046 §2).

Between ``seat`` and the marshal, this node loads the resolved seat's seat-owned
contract (the seat's ``validation:`` block) and admits or rejects the seat's output
envelope through a real ``ValidationEvaluator.evaluate`` — closing the ADR-046 §2 F3
gap where seat correctness was asserted by inspection. The contract is applied by the
skeleton (the seat's owner), not self-asserted by the candidate, and only through its
black-box, deterministic-first projection (the core/validation Seat Contract policy).

Emits ``{seat_admitted, seat_contract_reason}``. The verdict is a different
granularity from the loop-level accept gate (WP-D8) and rides alongside it: shape
carries it to emit, which refuses a rejected seat before the deliverable ships. A
seat with no contract (an ungated seat, a raw-prose explainer) is vacuously admitted.
"""

from __future__ import annotations

import asyncio
import json

from _helpers import terminal as _terminal
import sys
from pathlib import Path
from typing import Any

import yaml

from llm_orc.core.validation.models import ValidationConfig
from llm_orc.core.validation.seat_contract import SEAT_OUTPUT_KEY, admit

ENSEMBLES = Path(__file__).resolve().parents[2] / "ensembles"


def _deps(raw: str) -> dict[str, Any]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data.get("dependencies", {}) if isinstance(data, dict) else {}


def _response(dep: object) -> str:
    return dep.get("response", "") if isinstance(dep, dict) else ""




def _seat_output(seat_terminal: str) -> dict[str, Any]:
    """The seat's output as a dict the contract validates. An ADR-024 envelope is
    used as-is; a raw terminal (an explainer's prose) is adapted to a minimal
    success dict so a contract-less seat stays vacuously admitted."""
    try:
        obj = json.loads(seat_terminal)
    except (json.JSONDecodeError, TypeError):
        obj = None
    if isinstance(obj, dict) and "status" in obj:
        return obj
    return {"status": "success", "primary": seat_terminal}


def _contract_for(target: str) -> ValidationConfig:
    """Load the resolved seat's seat-owned contract (its ``seat_contract:`` block).

    This is a distinct field from the engine's ``validation:`` — the engine
    auto-runs ``validation:`` against the ensemble's own agents at execution,
    whereas the admission contract references the skeleton's ``seat`` adapter key.
    An unknown or contract-less seat yields an empty (vacuously-admitting) contract.
    """
    if not target:
        return ValidationConfig()
    for ext in ("yaml", "yml"):
        candidate = ENSEMBLES / f"{target}.{ext}"
        if candidate.exists():
            raw = yaml.safe_load(candidate.read_text()) or {}
            return ValidationConfig.model_validate(raw.get("seat_contract") or {})
    return ValidationConfig()


def main() -> None:
    deps = _deps(sys.stdin.read().strip())
    try:
        decision = json.loads(_response(deps.get("resolve", {})))
    except json.JSONDecodeError:
        decision = {}
    target = decision.get("target", "") if isinstance(decision, dict) else ""

    seat_terminal = _terminal(_response(deps.get("seat", {})))
    output = {SEAT_OUTPUT_KEY: _seat_output(seat_terminal)}
    contract = _contract_for(str(target))

    admission = asyncio.run(admit(str(target), output[SEAT_OUTPUT_KEY], contract))

    print(
        json.dumps(
            {
                "seat_admitted": admission.admitted,
                "seat_contract_reason": "" if admission.admitted else admission.reason,
            }
        )
    )


if __name__ == "__main__":
    main()
