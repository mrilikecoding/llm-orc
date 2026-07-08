"""Unit tests for the Seat Contract policy (WP-E8; ADR-046 §2).

The seat contract is the per-seat pass/fail admission check wired over the
surviving ``core/validation`` framework. It is used seat-owned (the seat declares
the contract; the candidate is validated against it), black-box (only I/O-facing
layers — the ``structural`` layer names internal agents and would couple a seat to
a candidate's internals), and deterministic-first (the ``semantic`` LLM-judge layer
is advisory; eligibility rests on the deterministic layers). These tests hit the
policy directly; the skeleton wiring is exercised in the serving suite.
"""

from __future__ import annotations

from typing import Any

import pytest

from llm_orc.core.validation.models import (
    BehavioralAssertion,
    SemanticValidationConfig,
    StructuralValidationConfig,
    ValidationConfig,
)
from llm_orc.core.validation.seat_contract import (
    SEAT_OUTPUT_KEY,
    admissible_layers,
    admit,
)

SUCCESS_ENVELOPE: dict[str, Any] = {
    "status": "success",
    "primary": "def f():\n    return 1\n",
    "artifacts": [{"content": "def f():\n    return 1\n", "summary": "f"}],
    "diagnostics": {"ensemble": "code-seat"},
}
FAILURE_ENVELOPE: dict[str, Any] = {
    "status": "error",
    "primary": "",
    "artifacts": [],
    "diagnostics": {},
}


def _status_success_contract() -> ValidationConfig:
    return ValidationConfig(
        behavioral=[
            BehavioralAssertion(
                name="envelope-status-success",
                description="the seat produced a success envelope",
                assertion=f"results['{SEAT_OUTPUT_KEY}']['status'] == 'success'",
            ),
            BehavioralAssertion(
                name="envelope-carries-artifact",
                description="the seat envelope carries a deliverable artifact",
                assertion=f"len(results['{SEAT_OUTPUT_KEY}']['artifacts']) > 0",
            ),
        ]
    )


# --- black-box + deterministic-first projection ---


def test_admissible_layers_drops_the_structural_layer() -> None:
    # black-box: the structural layer names internal agents (required_agents) and
    # would couple a seat to a candidate's internals — it must never gate admission.
    config = _status_success_contract()
    config.structural = StructuralValidationConfig(required_agents=["generate"])

    projected = admissible_layers(config)

    assert projected.structural is None


def test_admissible_layers_drops_the_semantic_layer() -> None:
    # deterministic-first: the LLM-judge layer is advisory, not eligibility-gating.
    config = _status_success_contract()
    config.semantic = SemanticValidationConfig(enabled=True, validator_model="x")

    projected = admissible_layers(config)

    assert projected.semantic is None


def test_admissible_layers_keeps_the_io_facing_layers() -> None:
    projected = admissible_layers(_status_success_contract())

    assert len(projected.behavioral) == 2


# --- admit: a real ValidationEvaluator.evaluate, not inspection ---


@pytest.mark.asyncio
async def test_admit_passes_a_conforming_envelope() -> None:
    admission = await admit("code-seat", SUCCESS_ENVELOPE, _status_success_contract())

    assert admission.admitted is True


@pytest.mark.asyncio
async def test_admit_rejects_a_nonconforming_envelope() -> None:
    admission = await admit("code-seat", FAILURE_ENVELOPE, _status_success_contract())

    assert admission.admitted is False
    assert admission.reason  # a non-empty reason drawn from the failing layer


@pytest.mark.asyncio
async def test_admit_ignores_a_structural_layer_the_candidate_would_fail() -> None:
    # The black-box guarantee at the admit level: a contract that names a
    # required internal agent the candidate does not expose still admits when the
    # I/O-facing layers pass — the structural coupling is dropped before evaluate.
    contract = _status_success_contract()
    contract.structural = StructuralValidationConfig(required_agents=["nonexistent"])

    admission = await admit("code-seat", SUCCESS_ENVELOPE, contract)

    assert admission.admitted is True


@pytest.mark.asyncio
async def test_admit_is_vacuous_for_an_empty_contract() -> None:
    admission = await admit("code-seat", SUCCESS_ENVELOPE, ValidationConfig())

    assert admission.admitted is True
