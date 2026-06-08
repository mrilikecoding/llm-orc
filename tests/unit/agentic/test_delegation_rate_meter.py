"""Tests for the Delegation Rate Meter (Cycle 7 loop-back #3, ADR-036 §Decision 3).

The meter graduates the Spike ψ.4a classification rule
(``scratch/spike-psi-delegation-rate/psi4a_prefilter.py``) into the package.
Two surfaces:

* ``classify_turn`` — the deterministic turn-shape rule (FC-59 denominator).
  The labeled-set regression reproduces the spike's clear-case result
  (0 misclassifications on the 12 clear cases); the 4 ambiguous boundary
  cases are reported against their recorded disposition, not thresholded.
* ``delegation_rate`` — the rate computed from emitted events alone
  (FC-59 numerator × denominator), no request content, logs, or replay.
"""

from __future__ import annotations

from dataclasses import dataclass

from llm_orc.agentic.delegation_rate_meter import (
    TurnShape,
    classify_turn,
    delegation_rate,
    domains_for,
)

_CODE_DOMAINS = domains_for(["code-generator"])


# The graduated labeled set — the spike's 16 cases, re-expressed against the
# three-way TurnShape vocabulary. 12 clear cases (thresholded at 0 errors)
# + 4 ambiguous boundary cases (recorded disposition, reported not thresholded).
@dataclass(frozen=True)
class _LabeledCase:
    id: str
    message: str
    shape: TurnShape
    clear: bool
    observed: tuple[str, ...] = ()


_LABELED_SET: tuple[_LabeledCase, ...] = (
    # --- clear: generation (real session tasks; capability-matched code) ---
    _LabeledCase(
        "c01",
        "Write a python module string_utils.py with a function that reverses "
        "the word order of a sentence and a function that counts vowels.",
        "generation",
        True,
    ),
    _LabeledCase(
        "c02",
        "Write a python module matrix_utils.py with functions to transpose a "
        "matrix and multiply two matrices.",
        "generation",
        True,
    ),
    _LabeledCase(
        "c03",
        "Create a python script inventory.py that tracks items with add, "
        "remove, and total-count functions.",
        "generation",
        True,
    ),
    _LabeledCase(
        "c04",
        "Implement a Stack class in python with push, pop, and peek methods, "
        "in stack.py.",
        "generation",
        True,
    ),
    # --- clear: carry (literal / observed-value) ---
    _LabeledCase(
        "c05",
        "Write exactly this to config.ini: [server]\\nport=8080\\nhost=localhost",
        "carry",
        True,
    ),
    _LabeledCase(
        "c06",
        "Append the output above to results.log verbatim.",
        "carry",
        True,
        ("test passed: 42 of 42",),
    ),
    _LabeledCase(
        "c07",
        "Replace the string 'DEBUG = True' with 'DEBUG = False' in settings.py.",
        "carry",
        True,
    ),
    # --- clear: read ---
    _LabeledCase("c08", "Read stack.py and tell me what it does.", "carry", True),
    _LabeledCase("c09", "Show the contents of pyproject.toml.", "carry", True),
    # --- clear: bash ---
    _LabeledCase("c10", "Run the tests and report whether they pass.", "carry", True),
    _LabeledCase("c11", "Execute ls -la in the project root.", "carry", True),
    # --- clear: finish/conversational ---
    _LabeledCase("c12", "What does this repository do?", "carry", True),
    # --- ambiguous boundary (recorded disposition; reported, not thresholded) ---
    _LabeledCase(
        "a01",
        "Fix the bug in stack.py where pop crashes on an empty stack.",
        "boundary_excluded",  # repair-shaped: read-then-generate, multi-step
        False,
    ),
    _LabeledCase(
        "a02",
        "Rename the variable counter to item_count in inventory.py.",
        "carry",  # mechanical edit; 'rename' is not a generation verb
        False,
    ),
    _LabeledCase(
        "a03",
        "Summarize this repository into NOTES.md.",
        "carry",  # 'summarize' reads as non-generation; spike labeled it delegate
        False,
    ),
    _LabeledCase(
        "a04",
        "Update the README with the test results from the run above.",
        "carry",  # observed values + connective prose; 'update' not a gen verb
        False,
        ("42 passed in 1.32s",),
    ),
)


class TestClassifierLabeledSet:
    """FC-59 — the graduated classifier reproduces the spike's labeled set."""

    def test_turn_shape_classifier_labeled_set(self) -> None:
        clear_errors = []
        for case in _LABELED_SET:
            got = classify_turn(case.message, list(case.observed), _CODE_DOMAINS)
            if case.clear and got != case.shape:
                clear_errors.append((case.id, case.shape, got))
        assert clear_errors == [], f"clear-case misclassifications: {clear_errors}"

    def test_ambiguous_cases_hold_recorded_disposition(self) -> None:
        # Reported, not thresholded — the recorded disposition is the contract,
        # documenting where the boundary is fuzzy.
        for case in _LABELED_SET:
            if case.clear:
                continue
            got = classify_turn(case.message, list(case.observed), _CODE_DOMAINS)
            assert got == case.shape, f"{case.id}: disposition drift {got}"

    def test_classification_is_deterministic(self) -> None:
        # No RNG, no LLM, no clock — identical inputs yield identical shapes.
        for case in _LABELED_SET:
            runs = {
                classify_turn(case.message, list(case.observed), _CODE_DOMAINS)
                for _ in range(5)
            }
            assert len(runs) == 1


class TestClassifierShapes:
    """The three turn shapes, isolated."""

    def test_generation_shaped_turn(self) -> None:
        assert (
            classify_turn(
                "Create a python module token_bucket.py with a rate limiter.",
                [],
                _CODE_DOMAINS,
            )
            == "generation"
        )

    def test_read_turn_is_carry(self) -> None:
        assert (
            classify_turn("Read config.py and summarize.", [], _CODE_DOMAINS) == "carry"
        )

    def test_repair_shaped_turn_is_boundary_excluded(self) -> None:
        assert (
            classify_turn(
                "Fix the bug in string_utils.py where count_vowels misses "
                "uppercase vowels.",
                [],
                _CODE_DOMAINS,
            )
            == "boundary_excluded"
        )

    def test_uncovered_domain_is_boundary_excluded(self) -> None:
        # Generation-shaped but no registered capability covers the domain —
        # excluded from the denominator, never guessed into it.
        assert (
            classify_turn(
                "Write an essay about climate policy in essay.md", [], _CODE_DOMAINS
            )
            == "boundary_excluded"
        )

    def test_no_capabilities_excludes_generation_turn(self) -> None:
        # With no registered capabilities, a generation-shaped turn cannot be
        # delegated — it is boundary-excluded, not a phantom denominator entry.
        assert (
            classify_turn("Write a python module foo.py with a helper.", [], [])
            == "boundary_excluded"
        )


# A minimal structural stand-in for a TurnDecision — proves delegation_rate
# reads only the narrow protocol (turn_shape, delegated_ensemble), with no
# access to request content, logs, or replay (FC-59 "events alone").
@dataclass(frozen=True)
class _FakeEvent:
    turn_shape: TurnShape | None
    delegated_ensemble: str | None


class TestDelegationRate:
    """FC-59 — the rate is computable from emitted events alone."""

    def test_delegation_rate_from_events_alone(self) -> None:
        events = [
            _FakeEvent("generation", "code-generator"),
            _FakeEvent("generation", "code-generator"),
            _FakeEvent("generation", None),  # generation that failed to delegate
            _FakeEvent("carry", None),
            _FakeEvent("boundary_excluded", None),
        ]
        reading = delegation_rate(events)
        assert reading.delegated == 2
        assert reading.generation_turns == 3
        assert reading.rate == 2 / 3
        assert reading.boundary_excluded == 1
        assert reading.considered == 5

    def test_rate_is_none_with_no_generation_turns(self) -> None:
        reading = delegation_rate(
            [_FakeEvent("carry", None), _FakeEvent("boundary_excluded", None)]
        )
        assert reading.rate is None
        assert reading.generation_turns == 0
        assert reading.boundary_excluded == 1

    def test_boundary_excluded_share_is_observable(self) -> None:
        # The denominator-degradation signal: boundary turns are counted and
        # surfaced, not silently dropped.
        reading = delegation_rate(
            [
                _FakeEvent("generation", "code-generator"),
                _FakeEvent("boundary_excluded", None),
                _FakeEvent("boundary_excluded", None),
            ]
        )
        assert reading.boundary_excluded == 2
        assert reading.considered == 3

    def test_empty_stream_is_a_null_reading(self) -> None:
        reading = delegation_rate([])
        assert reading.rate is None
        assert reading.delegated == 0
        assert reading.generation_turns == 0
        assert reading.boundary_excluded == 0
        assert reading.considered == 0
