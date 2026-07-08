"""Seat Contract — per-seat pass/fail admission over the validation framework.

ADR-046 §2: the Cycle-8 serving seats are swappable behind contracts, and the
contract IS the surviving ``core/validation`` framework (``ValidationConfig`` /
``ValidationEvaluator``). This module wires that framework as the seat's admission
gate, applying the three conventions ADR-046 §2 names:

- **seat-owned, not candidate-owned** — the seat declares the contract and the
  candidate's output is validated against it. A candidate grading its own homework
  is the §6.2b independence trap relocated to the registry; the caller passes the
  seat-owned contract in, the candidate never self-asserts.
- **black-box** — only the I/O-facing layers gate admission. The ``structural``
  layer names internal agents (``required_agents``) and would couple a seat to a
  candidate's internals, breaking interchangeability, so it is dropped.
- **deterministic-first** — the ``semantic`` (LLM-judge) layer is advisory; seat
  eligibility rests on the deterministic layers, so it is dropped from the gate.

The admission runs a real :meth:`ValidationEvaluator.evaluate` against the projected
contract (not an inspection re-check), closing the ADR-046 §2 F3 wiring gap.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from llm_orc.core.validation.evaluator import ValidationEvaluator
from llm_orc.core.validation.models import (
    EnsembleExecutionResult,
    ValidationConfig,
    ValidationResult,
)

SEAT_OUTPUT_KEY = "seat"
"""The canonical agent key the seat's output is adapted under. A seat-owned
contract references the seat's output as ``results['seat'][...]`` in its
behavioral assertions and ``agent: seat`` in its schema validations."""


class SeatAdmission(BaseModel):
    """The result of gating a seat's output against its contract."""

    admitted: bool
    reason: str
    result: ValidationResult


def admissible_layers(contract: ValidationConfig) -> ValidationConfig:
    """Project a declared contract to the layers that gate seat admission.

    Black-box drops the ``structural`` layer (internal-agent coupling);
    deterministic-first drops the ``semantic`` layer (advisory LLM judge). The
    I/O-facing deterministic layers (schema / behavioral / quantitative) are kept.
    Returns a copy — the seat's declared contract is left untouched.
    """
    return contract.model_copy(update={"structural": None, "semantic": None})


async def admit(
    seat_name: str,
    seat_output: dict[str, Any],
    contract: ValidationConfig,
) -> SeatAdmission:
    """Gate a seat's output against its seat-owned contract.

    Adapts ``seat_output`` (the seat's ADR-024 envelope, or any output dict) into
    the ``EnsembleExecutionResult`` the evaluator consumes — under the canonical
    :data:`SEAT_OUTPUT_KEY` so a contract's assertions can reference it — and runs
    a real :meth:`ValidationEvaluator.evaluate` against the black-box,
    deterministic-first projection of the contract.
    """
    projected = admissible_layers(contract)
    results = EnsembleExecutionResult(
        ensemble_name=seat_name,
        execution_order=[SEAT_OUTPUT_KEY],
        agent_outputs={SEAT_OUTPUT_KEY: seat_output},
        execution_time=0.0,
    )
    result = await ValidationEvaluator().evaluate(seat_name, results, projected)

    reasons = [
        error
        for layer in result.results.values()
        if layer is not None and not layer.passed
        for error in layer.errors
    ]
    reason = "; ".join(reasons) if reasons else "seat contract satisfied"
    return SeatAdmission(admitted=result.passed, reason=reason, result=result)
