"""Unit tests for the serving ``resolve`` node (WP-A8, scenario 4).

``resolve`` merges the deterministic ``classify`` decision with the guarded
model-backed ``decide`` node into the final routing the ``seat`` dispatches on
(scenarios.md "classify reads intent with a model-backed decider when the signal
is not structural"; ADR-046 §1). When classify resolved the turn structurally
(``needs_decider: false``), resolve passes it through unchanged and ``decide``
never ran. When classify deferred (``needs_decider: true``), resolve reads the
decider's bounded target and derives build/kind deterministically — the model
only classifies into the closed seat set; an unrecognized target is left empty
so dispatch fails deterministically rather than guessing.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
RESOLVE = REPO / ".llm-orc" / "scripts" / "agentic_serving" / "resolve.py"


def _resolve(
    classify_decision: dict[str, Any], decide_response: str | None = None
) -> dict[str, Any]:
    deps: dict[str, Any] = {"classify": {"response": json.dumps(classify_decision)}}
    if decide_response is not None:
        deps["decide"] = {"response": decide_response}
    out = subprocess.run(
        [sys.executable, str(RESOLVE)],
        input=json.dumps({"dependencies": deps}),
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    result: dict[str, Any] = json.loads(out)
    return result


def _structural(**overrides: Any) -> dict[str, Any]:
    decision = {
        "target": "code-seat",
        "kind": "python_module",
        "file": "add.py",
        "dispatch_input": "write add in add.py",
        "build": True,
        "needs_decider": False,
    }
    decision.update(overrides)
    return decision


def _ambiguous(dispatch_input: str) -> dict[str, Any]:
    return {
        "target": "",
        "kind": "",
        "file": "solution.py",
        "dispatch_input": dispatch_input,
        "build": False,
        "needs_decider": True,
    }


def test_structural_decision_passes_through_and_maps_to_the_build_shape() -> None:
    resolved = _resolve(_structural())
    # A code intent resolves to the default gated build shape (WP-D8, default-on).
    assert resolved["target"] == "build-gated"
    assert resolved["build"] is True
    # build/kind/file/dispatch_input pass through from classify unchanged.
    assert resolved["file"] == "add.py"
    assert resolved["dispatch_input"] == "write add in add.py"


def test_code_intent_maps_to_the_gated_build_shape() -> None:
    # The semantic intent 'code-seat' maps to the gated build shape at the
    # routing layer; explain stays a prose seat, unknown stays unresolved.
    assert _resolve(_structural(target="code-seat"))["target"] == "build-gated"
    prose = _resolve(_structural(target="explainer", build=False))
    assert prose["target"] == "explainer"


def test_ambiguous_turn_takes_the_decider_target_and_derives_build() -> None:
    resolved = _resolve(
        _ambiguous("sort this data for me"), json.dumps({"target": "code-seat"})
    )
    assert resolved["target"] == "build-gated"
    assert resolved["build"] is True
    assert resolved["kind"] == "python_module"
    # file and dispatch_input still come from classify
    assert resolved["file"] == "solution.py"
    assert resolved["dispatch_input"] == "sort this data for me"


def test_decider_can_route_an_ambiguous_turn_to_prose() -> None:
    resolved = _resolve(
        _ambiguous("tell me about recursion"), json.dumps({"target": "explainer"})
    )
    assert resolved["target"] == "explainer"
    assert resolved["build"] is False


def test_decider_target_is_read_from_prose_wrapped_output() -> None:
    resolved = _resolve(
        _ambiguous("do the thing"),
        'Sure, {"target": "explainer"} is the right seat here.',
    )
    assert resolved["target"] == "explainer"
    assert resolved["build"] is False


def test_unrecognized_decider_target_leaves_target_empty() -> None:
    """An out-of-set decider output does not guess a default: target stays empty
    so the dispatch node fails deterministically (the closed-set discipline)."""
    resolved = _resolve(_ambiguous("???"), "I really cannot tell what this is")
    assert resolved["target"] == ""


def test_intent_to_shape_routing_is_catalog_driven() -> None:
    """WP-C8: the intent->shape mapping is the operator-curated Shape Catalog, not
    a hardcoded map. resolve's code-seat routing matches the derived catalog, so
    re-hardcoding (or an operator retagging the shape) would diverge and fail."""
    from llm_orc.core.serving.shape_catalog import shape_catalog

    catalog_dir = REPO / ".llm-orc" / "ensembles" / "agentic-serving"
    catalog = shape_catalog(catalog_dir)
    assert catalog.get("code-seat") == "build-gated"  # the shipped default lane
    assert _resolve(_structural(target="code-seat"))["target"] == catalog["code-seat"]


def test_decider_tests_seat_derives_python_tests_build() -> None:
    """tests-seat joins the decider's closed set (#98)."""
    out = _resolve(
        {"needs_decider": True, "file": "test_x.py", "dispatch_input": "t"},
        '{"target": "tests-seat"}',
    )
    assert out["kind"] == "python_tests"
    assert out["build"] is True


def test_needs_files_and_read_failed_pass_through_resolve() -> None:
    routing = _resolve(
        {
            "target": "need-files",
            "kind": "need_files",
            "file": "test_storage.py",
            "dispatch_input": "write tests for existing storage.py",
            "build": False,
            "needs_decider": False,
            "needs_files": ["storage.py"],
            "read_failed": "",
        }
    )
    assert routing["target"] == "need-files"
    assert routing["needs_files"] == ["storage.py"]
    assert routing["read_failed"] == ""


def test_decider_path_defaults_read_fields_empty() -> None:
    routing = _resolve(
        {
            "target": "",
            "kind": "",
            "file": "solution.py",
            "dispatch_input": "coverage for the storage module",
            "build": False,
            "needs_decider": True,
        },
        '{"target": "tests-seat"}',
    )
    assert routing["needs_files"] == []
    assert routing["read_failed"] == ""


def test_needs_run_passes_through_resolve() -> None:
    routing = _resolve(
        _structural(
            target="need-run",
            kind="need_run",
            build=False,
            needs_run="pytest -q",
        )
    )
    assert routing["target"] == "need-run"
    assert routing["needs_run"] == "pytest -q"


def test_decider_path_defaults_needs_run_empty() -> None:
    routing = _resolve(
        _structural(target="", kind="", build=False, needs_decider=True),
        decide_response='{"target": "explainer"}',
    )
    assert routing["needs_run"] == ""


def test_needs_glob_and_glob_failed_pass_through_resolve() -> None:
    routing = _resolve(
        _structural(
            target="need-glob",
            kind="need_glob",
            build=False,
            needs_glob="storage",
            glob_failed="",
        )
    )
    assert routing["target"] == "need-glob"
    assert routing["needs_glob"] == "storage"
    assert routing["glob_failed"] == ""


def test_glob_failed_refusal_passes_through_resolve() -> None:
    reason = "no file matching 'storage' in the workspace listing"
    routing = _resolve(
        _structural(
            target="need-glob",
            kind="need_glob",
            build=False,
            needs_glob="",
            glob_failed=reason,
        )
    )
    assert routing["glob_failed"] == reason


def test_decider_path_defaults_glob_fields_empty() -> None:
    routing = _resolve(
        _structural(target="", kind="", build=False, needs_decider=True),
        decide_response='{"target": "explainer"}',
    )
    assert routing["needs_glob"] == ""
    assert routing["glob_failed"] == ""


def test_not_grounded_passes_through_resolve() -> None:
    # grounded-explain design: not_grounded rides classify's own structural
    # decision — never the guarded decider, so no _DERIVED entry is needed.
    routing = _resolve(
        _structural(
            target="not-grounded",
            kind="not_grounded",
            build=False,
            not_grounded="todo.py",
        )
    )
    assert routing["target"] == "not-grounded"
    assert routing["not_grounded"] == "todo.py"


def test_decider_path_defaults_not_grounded_empty() -> None:
    routing = _resolve(
        _structural(target="", kind="", build=False, needs_decider=True),
        decide_response='{"target": "explainer"}',
    )
    assert routing["not_grounded"] == ""


def test_decider_recall_vote_routes_to_the_recall_answer_shape() -> None:
    # #82 detection layer 2: a deferred recall turn whose decider votes "recall"
    # routes to the recall-answer shape, keeping classify's pre-computed honest
    # message (selection was already structural — the model decided recall vs
    # explain, never which file).
    classify_decision = {
        "target": "",
        "kind": "",
        "file": "solution.py",
        "dispatch_input": "what was the earliest thing you built?",
        "build": False,
        "needs_decider": True,
        "recall_answer": "The first thing built was `todo.py`. Ask me to read it.",
    }
    routing = _resolve(classify_decision, decide_response='{"target": "recall"}')
    assert routing["target"] == "recall-answer"
    assert routing["build"] is False
    assert "todo.py" in routing["recall_answer"]


def test_recall_vote_without_a_precomputed_message_falls_back_to_explainer() -> None:
    # Adversarial review finding 5: a recall vote on a turn where classify did
    # NOT pre-compute a message (the cheap decider mis-fired on a non-deferred
    # ambiguous turn) must not dispatch a degenerate empty recall shape — fall
    # back to the explainer.
    classify_decision = {
        "target": "",
        "kind": "",
        "file": "solution.py",
        "dispatch_input": "sort this data for me",
        "build": False,
        "needs_decider": True,
        "recall_answer": "",
    }
    routing = _resolve(classify_decision, decide_response='{"target": "recall"}')
    assert routing["target"] == "explainer"
    assert routing["recall_answer"] == ""


def test_decider_non_recall_vote_drops_the_precomputed_recall_answer() -> None:
    # If the decider votes a normal seat on a deferred turn, the pre-computed
    # recall message MUST be dropped — emit fires on recall_answer PRESENCE, so
    # leaving it would emit the recall message over the explainer's real output.
    classify_decision = {
        "target": "",
        "kind": "",
        "file": "solution.py",
        "dispatch_input": "what is the first-class function pattern?",
        "build": False,
        "needs_decider": True,
        "recall_answer": "The first thing built was `todo.py`. Ask me to read it.",
    }
    routing = _resolve(classify_decision, decide_response='{"target": "explainer"}')
    assert routing["target"] == "explainer"
    assert routing["recall_answer"] == ""


def test_memory_shaped_backstop_survives_when_the_decider_agrees_it_is_recall() -> None:
    # Review round 1 blocker 3: the backstop applies ONLY when the decider's
    # OWN vote confirms recall intent — here it votes "recall" but classify
    # never precomputed a message (finding-5 edge case), so it still falls
    # through to the explainer, and the backstop substrate must survive.
    classify_decision = {
        "target": "",
        "kind": "",
        "file": "solution.py",
        "dispatch_input": "what did you build originally?",
        "build": False,
        "needs_decider": True,
        "recall_answer": "",
        "memory_shaped": True,
        "grounded_text": "todo.py\ndef add_todo(): ...",
        "ledger_recap": "Shipped so far: `todo.py`.",
    }
    routing = _resolve(classify_decision, decide_response='{"target": "recall"}')
    assert routing["target"] == "explainer"
    assert routing["memory_shaped"] is True
    assert routing["grounded_text"] == "todo.py\ndef add_todo(): ..."
    assert routing["ledger_recap"] == "Shipped so far: `todo.py`."


def test_memory_shaped_backstop_clears_when_the_decider_disagrees() -> None:
    # Review round 1 blocker 3 (the measured false positive): classify's
    # loose maybe_recall pre-filter fires on "explain how first-class
    # functions work" (the incidental "first"), setting memory_shaped=True —
    # but the decider correctly votes "explainer", not "recall". The
    # backstop must NOT survive, so the seat's real answer is never
    # replaced.
    classify_decision = {
        "target": "",
        "kind": "",
        "file": "solution.py",
        "dispatch_input": "explain how first-class functions work in python",
        "build": False,
        "needs_decider": True,
        "recall_answer": "",
        "memory_shaped": True,
        "grounded_text": "",
        "ledger_recap": "Nothing has been built in this session yet.",
    }
    routing = _resolve(classify_decision, decide_response='{"target": "explainer"}')
    assert routing["target"] == "explainer"
    assert routing["memory_shaped"] is False


def test_memory_shaped_backstop_clears_on_the_deterministic_recall_answer() -> None:
    # No seat runs on this branch (the message was already precomputed and
    # deterministic) — the backstop must never check our OWN template's
    # backticks (wrong-accept-hunt target 6).
    classify_decision = {
        "target": "",
        "kind": "",
        "file": "solution.py",
        "dispatch_input": "what did you build originally?",
        "build": False,
        "needs_decider": True,
        "recall_answer": "The first thing built was `todo.py`. Ask me to read it.",
        "memory_shaped": True,
        "grounded_text": "",
        "ledger_recap": "Shipped so far: `todo.py`.",
    }
    routing = _resolve(classify_decision, decide_response='{"target": "recall"}')
    assert routing["target"] == "recall-answer"
    assert routing["memory_shaped"] is False


def test_missing_memory_shaped_fields_default_safely() -> None:
    # A classify decision predating #133/#134 (or a hand-built test fixture)
    # carries none of these keys — resolve must not crash, and must default
    # to "not memory-shaped".
    resolved = _resolve(_structural())
    assert resolved["memory_shaped"] is False
    assert resolved["grounded_text"] == ""
    assert resolved["ledger_recap"] == ""
