"""Unit tests for the serving ``shape`` node (WP-A8, scenario 5).

shape reads the deliverable CONTENT from the seat's ADR-024 envelope and the
DESTINATION from classify, then produces the faithful deliverable. When the seat
did not envelope (a non-build explain seat returning raw prose), shape degrades
to the raw terminal text (scenarios.md "the marshal node consumes the seat's
real common I/O envelope"; ADR-046 §1).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
SHAPE = REPO / ".llm-orc" / "scripts" / "agentic_serving" / "shape.py"


def _shape_raw(deps: dict[str, Any]) -> dict[str, Any]:
    out = subprocess.run(
        [sys.executable, str(SHAPE)],
        input=json.dumps({"dependencies": deps}),
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    result: dict[str, Any] = json.loads(out)
    return result


def _shape(
    classify_decision: dict[str, Any], seat_child_result: dict[str, Any]
) -> dict[str, Any]:
    return _shape_raw(
        {
            "classify": {"response": json.dumps(classify_decision)},
            "seat": {"response": json.dumps(seat_child_result)},
        }
    )


def test_shape_reads_deliverable_from_envelope_and_destination_from_classify() -> None:
    code = "def add(a, b):\n    return a + b"
    envelope = {
        "status": "success",
        "primary": code,
        "artifacts": [
            {"content": code, "content_type": "text/x-python", "summary": "add"}
        ],
    }
    shaped = _shape(
        {
            "target": "code-seat",
            "build": True,
            "file": "add.py",
            "kind": "python_module",
        },
        {
            "results": {
                "generate": {"response": "..."},
                "envelope": {"response": json.dumps(envelope)},
            }
        },
    )
    assert shaped["build"] is True
    assert shaped["file"] == "add.py"
    assert shaped["content"] == code


def test_shape_degrades_to_raw_seat_terminal_when_no_envelope() -> None:
    shaped = _shape(
        {"target": "explainer", "build": False, "kind": "explanation"},
        {"results": {"out": {"response": "It adds two numbers."}}},
    )
    assert shaped["build"] is False
    assert "adds two numbers" in shaped["content"]


def test_shape_carries_the_accept_verdict_from_the_build_gated_envelope() -> None:
    code = "def add(a, b):\n    return a + b"
    envelope = {
        "status": "success",
        "primary": code,
        "artifacts": [{"content": code, "content_type": "text/x-python"}],
        "diagnostics": {
            "ensemble": "build-gated",
            "accept": False,
            "accept_reason": "tests inadequate",
        },
    }
    shaped = _shape(
        {
            "target": "code-seat",
            "build": True,
            "file": "add.py",
            "kind": "python_module",
        },
        {"results": {"envelope": {"response": json.dumps(envelope)}}},
    )
    assert shaped["content"] == code
    assert shaped["accept"] is False
    assert "inadequate" in shaped["accept_reason"]


def test_shape_has_no_verdict_for_an_ungated_seat_envelope() -> None:
    # A code-seat / explainer envelope carries no accept diagnostics -> None.
    envelope = {
        "status": "success",
        "primary": "x = 1",
        "artifacts": [{"content": "x = 1"}],
    }
    shaped = _shape(
        {"target": "code-seat", "build": True, "file": "a.py"},
        {"results": {"envelope": {"response": json.dumps(envelope)}}},
    )
    assert shaped["accept"] is None


def test_unreadable_routing_decision_fails_closed_to_refusal() -> None:
    """An empty/unparseable routing decision must not default the turn onto
    the build path NOR pass the dead seat's prose through — it refuses
    (#152 strengthens the old prose fail-closed to an honest refusal)."""
    out = _shape({}, {"results": {"out": {"response": "x = 1", "status": "success"}}})
    assert out["build"] is False
    assert out["routing_failed"]


# The engine's crash wrap (script_agent.py:161-168): a serving node that
# exits nonzero RETURNS this envelope as a normal response, so the ensemble
# records the node status=success and the envelope rides into dependents'
# deps looking like a readable result. Captured live 2026-08-13 (#144 gate
# misfire, docs/plans/2026-08-13-144-live-gate/).
_FAILURE_ENVELOPE = {
    "success": False,
    "error": "Script failed with exit code 1",
    "stderr": "ModuleNotFoundError: No module named 'llm_orc'",
}


def test_failed_resolve_dep_fails_closed_to_a_routing_refusal() -> None:
    """The #152 capture: the old truthiness fallback picked the failed
    resolve dep (it exists), parsed the failure envelope to a non-empty
    dict with no "kind" key, and the build default landed True — a
    crashed ROUTING node degraded into a junk empty solution.py write.
    No readable routing decision must refuse instead."""
    shaped = _shape_raw(
        {
            "resolve": {"response": json.dumps(_FAILURE_ENVELOPE)},
            "seat": {"response": ""},
        }
    )
    assert shaped["build"] is False
    assert "Script failed with exit code 1" in shaped["routing_failed"]


def test_empty_target_decision_is_unreadable_and_refuses() -> None:
    """target must be NON-EMPTY — key presence is not enough (#152
    pre-flight finding 1): a crashed classify laundered through a healthy
    resolve emits target "" (resolve's else branch over the failure
    envelope), as does an out-of-set decider vote. Both are routing
    failures, and the seat that dispatched on "" is dead either way."""
    shaped = _shape_raw(
        {
            "resolve": {
                "response": json.dumps(
                    {"target": "", "kind": "", "file": "solution.py", "build": False}
                )
            },
            "seat": {"response": ""},
        }
    )
    assert shaped["build"] is False
    assert shaped["routing_failed"]


def test_decision_with_target_but_no_build_or_kind_refuses() -> None:
    """The retained build default (kind != "explanation" -> True) must
    never be what stands between a target-only decision and a build
    (#152 pre-flight finding 3): a decision carrying target but neither
    build nor kind is outside the producers' contract and refuses."""
    shaped = _shape_raw(
        {
            "resolve": {"response": json.dumps({"target": "code-seat"})},
            "seat": {"response": "x = 1"},
        }
    )
    assert shaped["build"] is False
    assert shaped["routing_failed"]


def test_failed_resolve_with_readable_classify_routes_via_classify() -> None:
    """The source-preference chain stays alive for the unit harness and
    any pre-resolve wiring: a readable classify behind an unreadable
    resolve routes the turn normally (live serving.yaml never wires
    classify into shape's deps, so this is not an absorbing live
    fallback — see the #152 design's deviation section)."""
    shaped = _shape_raw(
        {
            "resolve": {"response": json.dumps(_FAILURE_ENVELOPE)},
            "classify": {
                "response": json.dumps(
                    {"target": "explainer", "kind": "explanation", "build": False}
                )
            },
            "seat": {
                "response": json.dumps(
                    {"results": {"out": {"response": "It adds two numbers."}}}
                )
            },
        }
    )
    assert shaped["build"] is False
    assert shaped["routing_failed"] == ""
    assert "adds two numbers" in shaped["content"]


def test_shape_passes_read_fields_from_the_routing_decision() -> None:
    shaped = _shape(
        {
            "target": "need-files",
            "kind": "need_files",
            "file": "test_storage.py",
            "build": False,
            "needs_files": ["storage.py"],
            "read_failed": "",
        },
        {"results": {"out": {"response": "Requesting client files."}}},
    )
    assert shaped["needs_files"] == ["storage.py"]
    assert shaped["read_failed"] == ""


def test_shape_passes_needs_run_from_the_routing_decision() -> None:
    shaped = _shape(
        {
            "target": "need-run",
            "kind": "need_run",
            "file": "solution.py",
            "build": False,
            "needs_files": [],
            "read_failed": "",
            "needs_run": "pytest -q",
        },
        {"status": "ok", "primary": "Requesting a client test run."},
    )
    assert shaped["needs_run"] == "pytest -q"


def test_shape_passes_glob_fields_from_the_routing_decision() -> None:
    shaped = _shape(
        {
            "target": "need-glob",
            "kind": "need_glob",
            "file": "solution.py",
            "build": False,
            "needs_files": [],
            "read_failed": "",
            "needs_run": "",
            "needs_glob": "storage",
            "glob_failed": "",
        },
        {"status": "ok", "primary": "Requesting a workspace listing."},
    )
    assert shaped["needs_glob"] == "storage"
    assert shaped["glob_failed"] == ""


def test_shape_passes_not_grounded_from_the_routing_decision() -> None:
    shaped = _shape(
        {
            "target": "not-grounded",
            "kind": "not_grounded",
            "file": "solution.py",
            "build": False,
            "not_grounded": "todo.py",
        },
        {"status": "ok", "primary": "Not grounded in this session."},
    )
    assert shaped["not_grounded"] == "todo.py"
    assert shaped["build"] is False


def test_shape_passes_is_build_ask_from_the_routing_decision() -> None:
    # Review round 2 new blocker 2.
    shaped = _shape(
        {
            "target": "need-files",
            "kind": "need_files",
            "file": "test_storage.py",
            "build": False,
            "is_build_ask": True,
        },
        {"results": {"out": {"response": "Requesting client files."}}},
    )
    assert shaped["is_build_ask"] is True


def test_shape_defaults_is_build_ask_false_when_absent() -> None:
    shaped = _shape(
        {"target": "explainer", "kind": "explanation", "file": "solution.py"},
        {"status": "ok", "primary": "It adds two numbers."},
    )
    assert shaped["is_build_ask"] is False


def test_needs_self_files_passes_through_shape() -> None:
    # #144 serve-native self-reference: rides the routing decision.
    shaped = _shape(
        {
            "target": "need-self-files",
            "kind": "need_self_files",
            "file": "solution.py",
            "build": False,
            "needs_self_files": [".llm-orc/scripts/agentic_serving/resolve.py"],
        },
        {"status": "ok", "primary": "Requesting self files."},
    )
    assert shaped["needs_self_files"] == [".llm-orc/scripts/agentic_serving/resolve.py"]


def test_non_string_or_whitespace_target_refuses() -> None:
    """Review finding 8: the gate demands a non-empty STRING target — a
    drifted producer emitting a number, list, or whitespace target must
    refuse, never ride the build default past the gate."""
    for target in (5, ["code-seat"], "   "):
        shaped = _shape_raw(
            {
                "resolve": {
                    "response": json.dumps(
                        {"target": target, "kind": "python_module", "build": True}
                    )
                },
                "seat": {"response": "x = 1"},
            }
        )
        assert shaped["build"] is False, f"target={target!r}"
        assert shaped["routing_failed"], f"target={target!r}"
