#!/usr/bin/env python3
"""Ω-P3 plan (fixed-contract, for the local-only de-risk run).

Emits the per-file build tasks from the pinned coherent calc contract instead of
the resolve-contract loop, so the REAL local build + execution-gate path can be
validated WITHOUT spending architect tokens. Same {tasks: [{deliverable,
contract}]} shape the live plan emits.
"""

from __future__ import annotations

import json
from pathlib import Path

FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "scratch" / "spike-omega-e" / "fixtures" / "calc_coherent_contract.json"
)


def main() -> None:
    contract = json.loads(FIXTURE.read_text())
    tasks = [{"deliverable": d, "contract": contract} for d in contract]
    print(json.dumps({"tasks": tasks}))


if __name__ == "__main__":
    main()
