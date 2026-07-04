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
