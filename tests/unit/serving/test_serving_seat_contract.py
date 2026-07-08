"""Serving seat-contract wiring (WP-E8; ADR-046 §2 F3 gap).

The ``seat_contract`` node loads the resolved seat's seat-owned contract and admits
or rejects the seat's output through a wired ``ValidationEvaluator.evaluate`` — not
by inspection. The per-seat admission is a different granularity from the loop-level
accept gate (WP-D8) and the two compose: a seat can clear its own contract while the
accept gate still rejects the turn (scenarios.md preservation).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from llm_orc.core.validation.models import ValidationConfig

REPO = Path(__file__).resolve().parents[3]
SEAT_CONTRACT = REPO / ".llm-orc" / "scripts" / "agentic_serving" / "seat_contract.py"
EMIT = REPO / ".llm-orc" / "scripts" / "agentic_serving" / "emit.py"
CODE_SEAT_YAML = REPO / ".llm-orc" / "ensembles" / "agentic-serving" / "code-seat.yaml"

CODE = "def add(a, b):\n    return a + b\n"
SUCCESS_ENVELOPE: dict[str, Any] = {
    "status": "success",
    "primary": CODE,
    "structured": {"content": CODE},
    "artifacts": [{"content": CODE, "content_type": "text/x-python", "summary": "add"}],
    "diagnostics": {"ensemble": "code-seat"},
}
MALFORMED_ENVELOPE: dict[str, Any] = {
    "status": "error",
    "primary": "",
    "artifacts": [],
    "diagnostics": {"ensemble": "code-seat"},
}


def _seat_child(envelope: dict[str, Any]) -> dict[str, Any]:
    """The wrapped shape a dispatched code-seat child returns to the skeleton."""
    return {
        "results": {
            "generate": {"response": "..."},
            "envelope": {"response": json.dumps(envelope)},
        }
    }


def _seat_contract(target: str, seat_response: str) -> dict[str, Any]:
    payload = json.dumps(
        {
            "dependencies": {
                "resolve": {"response": json.dumps({"target": target, "build": True})},
                "seat": {"response": seat_response},
            }
        }
    )
    out = subprocess.run(
        [sys.executable, str(SEAT_CONTRACT)],
        input=payload,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    result: dict[str, Any] = json.loads(out)
    return result


def _emit(form_gate: dict[str, Any]) -> dict[str, Any]:
    payload = json.dumps(
        {"dependencies": {"form_gate": {"response": json.dumps(form_gate)}}}
    )
    out = subprocess.run(
        [sys.executable, str(EMIT)],
        input=payload,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    result: dict[str, Any] = json.loads(out)
    return result


# --- the ADR-046 §2 wiring gap: admitted/rejected by a real evaluate ---


def test_admits_a_conforming_code_seat_envelope() -> None:
    verdict = _seat_contract("code-seat", json.dumps(_seat_child(SUCCESS_ENVELOPE)))
    assert verdict["seat_admitted"] is True


def test_rejects_a_malformed_code_seat_envelope() -> None:
    verdict = _seat_contract("code-seat", json.dumps(_seat_child(MALFORMED_ENVELOPE)))
    assert verdict["seat_admitted"] is False
    assert verdict["seat_contract_reason"]  # a reason from the failing layer


def test_seat_swap_applies_the_resolved_seats_own_contract() -> None:
    # Swap the resolved seat at zero skeleton change: the explainer carries no
    # contract, so raw prose is vacuously admitted through the same node.
    verdict = _seat_contract("explainer", json.dumps({"response": "some prose"}))
    assert verdict["seat_admitted"] is True


def test_code_seat_contract_uses_only_io_facing_layers() -> None:
    # Fitness (system-design §Cycle 8): the contract names no internal agents
    # (no structural required_agents coupling) — black-box interchangeability.
    raw = yaml.safe_load(CODE_SEAT_YAML.read_text())
    contract = ValidationConfig.model_validate(raw.get("validation") or {})
    assert contract.structural is None
    assert contract.behavioral  # the contract does gate on I/O-facing output


# --- preservation: per-seat admission composes with the loop-level accept gate ---


def test_seat_admission_and_accept_gate_are_independent_gates() -> None:
    # A seat that clears its own contract can still be rejected by the loop-level
    # accept gate (different granularities compose, scenarios.md preservation).
    passed_seat_but_rejected_accept = {
        "build": True,
        "file": "add.py",
        "content": CODE,
        "valid": True,
        "reason": "ok",
        "seat_admitted": True,
        "accept": False,
        "accept_reason": "tests inadequate",
    }
    outcome = _emit(passed_seat_but_rejected_accept)
    assert outcome["finish"] is True
    assert "Another round needed" in outcome["content"]


def test_a_rejected_seat_contract_refuses_before_the_deliverable_ships() -> None:
    rejected_seat = {
        "build": True,
        "file": "add.py",
        "content": CODE,
        "valid": True,
        "reason": "ok",
        "seat_admitted": False,
        "seat_contract_reason": "envelope status is not success",
        "accept": None,
    }
    outcome = _emit(rejected_seat)
    assert outcome["finish"] is True
    assert "Seat contract" in outcome["content"]
