"""Deterministic routing unit tests for the serving ``classify`` node (WP-A8).

``classify`` is a pure script node: it reads the turn and emits the routing
decision ``{target, kind, file, dispatch_input, build}`` the dispatch seat
resolves (scenarios.md "classify emits a structural routing decision
deterministically"; ADR-046 §1, where classify owns the build-vs-non-build
executable-deliverable determination). Driven via subprocess exactly as the L0
engine runs a script node.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
CLASSIFY = REPO / ".llm-orc" / "scripts" / "agentic_serving" / "classify.py"


def _classify(turn: dict[str, Any]) -> dict[str, Any]:
    envelope = json.dumps({"input": json.dumps(turn)})
    out = subprocess.run(
        [sys.executable, str(CLASSIFY)],
        input=envelope,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    result: dict[str, Any] = json.loads(out)
    return result


def test_build_turn_routes_to_the_code_generation_seat() -> None:
    decision = _classify({"task": "write a function that adds two numbers"})
    assert decision["target"] == "code-seat"
    assert decision["build"] is True


def test_filename_is_extracted_from_the_turn_text() -> None:
    decision = _classify({"task": "write a function that adds two numbers in add.py"})
    assert decision["file"] == "add.py"
    assert decision["build"] is True


def test_explain_turn_is_non_build_and_routes_away_from_the_code_seat() -> None:
    decision = _classify({"task": "explain what foo.py does"})
    assert decision["build"] is False
    assert decision["target"] != "code-generator"


def test_a_named_target_file_is_carried_structurally() -> None:
    decision = _classify({"task": "build a cli", "file": "cli.py"})
    assert decision["file"] == "cli.py"
    assert decision["build"] is True


def test_ambiguous_turn_defers_to_the_model_backed_decider() -> None:
    """A turn with no structural signal (no explain marker, no build verb, no
    named file) cannot be routed deterministically, so classify flags it for the
    model-backed decider rather than defaulting to a seat (scenarios.md "classify
    reads intent with a model-backed decider when the signal is not structural").
    """
    decision = _classify({"task": "I'm not sure how to approach this"})
    assert decision["needs_decider"] is True


def test_structural_turns_do_not_need_the_decider() -> None:
    """The deterministic fast-path resolves structural turns without a model, so
    the guarded decider never runs for them (determinism preserved)."""
    for task in (
        "write a function that adds two numbers in add.py",
        "explain what foo.py does",
        "implement a stack",
    ):
        assert _classify({"task": task})["needs_decider"] is False


def test_a_question_naming_a_file_routes_to_explain_not_build() -> None:
    """An interrogative turn is a request for understanding even when it names
    a file — the named-file build signal must not outrank the question shape
    (battery finding 2026-07-08: "What approach does palindrome.py use?" ran
    the full gated build and returned a reject verdict).
    """
    decision = _classify({"task": "What approach does palindrome.py use, briefly?"})
    assert decision["build"] is False
    assert decision["target"] == "explainer"


def test_an_imperative_build_request_phrased_politely_still_builds() -> None:
    decision = _classify({"task": "Can you write a function to add numbers in add.py"})
    assert decision["build"] is True
    assert decision["target"] == "code-seat"


def test_context_composes_into_dispatch_input_after_the_marker() -> None:
    """With conversation context present, dispatch_input carries it ahead of
    the deterministic 'Current request:' marker (rung 1, memory design)."""
    decision = _classify(
        {
            "task": "add tests for it in test_even.py",
            "context": "user: write is_even in even.py\nassistant: [wrote even.py]",
        }
    )
    assert decision["dispatch_input"] == (
        "Conversation so far:\n"
        "user: write is_even in even.py\nassistant: [wrote even.py]"
        "\n\nCurrent request: add tests for it in test_even.py"
    )
    # the clean turn stays available for consumers that must not see history
    assert decision["task"] == "add tests for it in test_even.py"


def test_routing_reads_the_task_never_the_context() -> None:
    """A past build request in the context must not re-trigger a build; the
    latest turn alone decides the route."""
    decision = _classify(
        {
            "task": "What does the helper do?",
            "context": "user: write a helper in util.py\nassistant: [wrote util.py]",
        }
    )
    assert decision["build"] is False
    assert decision["target"] == "explainer"


def test_no_context_leaves_dispatch_input_as_the_bare_task() -> None:
    decision = _classify({"task": "write a function that adds two numbers"})
    assert decision["dispatch_input"] == "write a function that adds two numbers"


# --- test-primary turns route to the tests-seat (#98) ---


def test_write_tests_for_a_named_file_routes_to_the_tests_seat() -> None:
    """A test-primary turn (tests as the OBJECT): the deliverable is a test
    file run against the workspace, never build-gated's code/tests duality
    (the shadowed-composite wrong-accept, issue #98)."""
    decision = _classify({"task": "Write tests for storage.py"})
    assert decision["target"] == "tests-seat"
    assert decision["build"] is True
    assert decision["kind"] == "python_tests"
    assert decision["file"] == "test_storage.py"


def test_add_tests_for_it_routes_to_the_tests_seat() -> None:
    decision = _classify({"task": "add tests for it"})
    assert decision["target"] == "tests-seat"
    assert decision["file"] == "test_solution.py"


def test_a_named_test_file_routes_to_the_tests_seat() -> None:
    decision = _classify({"task": "update test_models.py with a case for done"})
    assert decision["target"] == "tests-seat"
    assert decision["file"] == "test_models.py"


def test_with_tests_is_not_test_primary() -> None:
    """'write is_even with tests' wants code (tests as a trailing mention);
    routing it to the tests-seat would ship only tests."""
    decision = _classify({"task": "write is_even with tests in even.py"})
    assert decision["target"] == "code-seat"
    assert decision["file"] == "even.py"


def test_explain_still_outranks_the_tests_signal() -> None:
    decision = _classify({"task": "what do the tests in test_foo.py cover?"})
    assert decision["build"] is False
    assert decision["target"] == "explainer"
