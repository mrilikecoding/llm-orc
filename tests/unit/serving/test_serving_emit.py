"""Unit tests for the serving ``emit`` node (WP-A8).

emit is the terminal client-permission-seam node: a valid build deliverable
becomes a file write; a refused build degrades to a prose finish carrying the
reason (the serve never writes a form-gate-refused deliverable); a non-build
turn is a prose finish (scenarios.md "Per-Turn Serving Handler"; ADR-046 §1,
ADR-034 re-home).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPTS = REPO / ".llm-orc" / "scripts" / "agentic_serving"
EMIT = SCRIPTS / "emit.py"

sys.path.insert(0, str(SCRIPTS))
from emit import (  # type: ignore  # noqa: E402
    ACCEPT_GATE_REJECT_PREFIX,
    BUILD_REFUSED_PREFIX,
    REFUSED_PREFIX,
    SEAT_CONTRACT_REJECT_PREFIX,
    TERMINALS,
)


def _emit(gated: dict[str, Any]) -> dict[str, Any]:
    payload = json.dumps(
        {"dependencies": {"form_gate": {"response": json.dumps(gated)}}}
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


def test_valid_build_emits_a_file_write() -> None:
    outcome = _emit(
        {
            "build": True,
            "valid": True,
            "file": "add.py",
            "content": "def add():\n    pass",
        }
    )
    assert outcome["finish"] is False
    assert outcome["file"] == "add.py"
    assert outcome["content"] == "def add():\n    pass"


def test_refused_build_degrades_to_a_prose_finish_with_no_write() -> None:
    outcome = _emit(
        {
            "build": True,
            "valid": False,
            "file": "add.py",
            "content": "bad",
            "reason": "not valid Python",
        }
    )
    assert outcome["finish"] is True
    assert "refused" in outcome["content"].lower()
    assert "file" not in outcome


def test_non_build_is_a_prose_finish() -> None:
    outcome = _emit({"build": False, "valid": True, "content": "It adds two numbers."})
    assert outcome["finish"] is True
    assert "adds two numbers" in outcome["content"]


def test_rejected_accept_gate_emits_another_round_with_no_write() -> None:
    # accept=False routes another round (ODP-2: the client owns the loop); the
    # serve never writes a gate-rejected deliverable, even if it parses.
    outcome = _emit(
        {
            "build": True,
            "valid": True,
            "file": "a.py",
            "content": "def f():\n    pass",
            "accept": False,
            "accept_reason": "tests inadequate to verify the requirement",
        }
    )
    assert outcome["finish"] is True
    assert "another round" in outcome["content"].lower()
    assert "inadequate" in outcome["content"]
    assert "file" not in outcome


def test_accepted_build_emits_a_file_write() -> None:
    outcome = _emit(
        {
            "build": True,
            "valid": True,
            "file": "a.py",
            "content": "def f():\n    pass",
            "accept": True,
            "accept_reason": "tests pass and are adequate",
        }
    )
    assert outcome["finish"] is False
    assert outcome["file"] == "a.py"


def test_needs_files_emits_a_reads_outcome() -> None:
    outcome = _emit(
        gated={
            "build": False,
            "file": "test_storage.py",
            "content": "Requesting client files.",
            "valid": True,
            "reason": "ok",
            "needs_files": ["storage.py"],
            "read_failed": "",
            "accept": None,
            "accept_reason": "",
            "seat_admitted": None,
            "seat_contract_reason": "",
        }
    )
    assert outcome == {"finish": False, "reads": ["storage.py"]}


def test_read_failed_emits_an_honest_refusal() -> None:
    # is_build_ask absent/False: a non-build turn's read refusal (e.g. a
    # bare-symbol explain-discovery's ambiguous read) never claims a build
    # outcome (review round 2 new blocker 2).
    outcome = _emit(
        gated={
            "build": False,
            "file": "test_storage.py",
            "content": "Requesting client files.",
            "valid": True,
            "reason": "ok",
            "needs_files": [],
            "read_failed": "could not read storage.py: client read failed",
            "accept": None,
            "accept_reason": "",
            "seat_admitted": None,
            "seat_contract_reason": "",
        }
    )
    assert outcome["finish"] is True
    assert outcome["content"] == (
        "Refused: could not read storage.py: client read failed"
    )


def test_build_ask_read_failed_uses_the_build_refused_prefix() -> None:
    # Review round 2 new blocker 2: a build ask (write tests for existing
    # X) needing to read X first, where the read fails, must mint a BUILD
    # outcome — the plain "Refused:" prefix never claims build-ness, so the
    # caller-side ledger would silently drop this disclosure otherwise.
    outcome = _emit(
        gated={
            "build": False,
            "file": "test_storage.py",
            "content": "Requesting client files.",
            "valid": True,
            "reason": "ok",
            "needs_files": [],
            "read_failed": "could not read storage.py: client read failed",
            "is_build_ask": True,
            "accept": None,
            "accept_reason": "",
            "seat_admitted": None,
            "seat_contract_reason": "",
        }
    )
    assert outcome["finish"] is True
    assert outcome["content"] == (
        "Build refused: could not read storage.py: client read failed"
    )


def test_needs_run_emits_a_run_outcome() -> None:
    outcome = _emit(
        {
            "build": False,
            "file": "solution.py",
            "content": "Requesting a client test run.",
            "valid": True,
            "reason": "ok",
            "needs_files": [],
            "read_failed": "",
            "needs_run": "pytest -q test_calc.py",
            "accept": None,
            "accept_reason": "",
            "seat_admitted": None,
            "seat_contract_reason": "",
        }
    )
    assert outcome == {"finish": False, "run": "pytest -q test_calc.py"}


def test_needs_glob_emits_a_glob_outcome() -> None:
    outcome = _emit(
        {
            "build": False,
            "file": "solution.py",
            "content": "Requesting a workspace listing.",
            "valid": True,
            "reason": "ok",
            "needs_files": [],
            "read_failed": "",
            "needs_run": "",
            "needs_glob": "storage",
            "glob_failed": "",
            "accept": None,
            "accept_reason": "",
            "seat_admitted": None,
            "seat_contract_reason": "",
        }
    )
    assert outcome == {"finish": False, "glob": "storage"}


def test_glob_failed_emits_an_honest_refusal() -> None:
    # is_build_ask absent/False: an explain-discovery glob refusal.
    outcome = _emit(
        {
            "build": False,
            "file": "solution.py",
            "content": "Requesting a workspace listing.",
            "valid": True,
            "reason": "ok",
            "needs_files": [],
            "read_failed": "",
            "needs_run": "",
            "needs_glob": "",
            "glob_failed": "no file matching 'storage' in the workspace listing",
            "accept": None,
            "accept_reason": "",
            "seat_admitted": None,
            "seat_contract_reason": "",
        }
    )
    assert outcome["finish"] is True
    assert outcome["content"] == (
        "Refused: no file matching 'storage' in the workspace listing"
    )


def test_build_ask_glob_failed_uses_the_build_refused_prefix() -> None:
    # Review round 2 new blocker 2: a build ask's discovery glob refusal
    # must mint a BUILD outcome too.
    outcome = _emit(
        {
            "build": False,
            "file": "solution.py",
            "content": "Requesting a workspace listing.",
            "valid": True,
            "reason": "ok",
            "needs_files": [],
            "read_failed": "",
            "needs_run": "",
            "needs_glob": "",
            "glob_failed": "no file matching 'storage' in the workspace listing",
            "is_build_ask": True,
            "accept": None,
            "accept_reason": "",
            "seat_admitted": None,
            "seat_contract_reason": "",
        }
    )
    assert outcome["finish"] is True
    assert outcome["content"] == (
        "Build refused: no file matching 'storage' in the workspace listing"
    )


def test_not_grounded_emits_the_honest_message_without_a_seat_call() -> None:
    # grounded-explain design (docs/plans/2026-07-12-grounded-explain-
    # design.md): the message is deterministic and non-speculative — no
    # "Refused:" prefix, since this is not a request refusal, it is an
    # honest report that the target was never seen on the wire.
    outcome = _emit(
        {
            "build": False,
            "file": "solution.py",
            "content": "Not grounded in this session.",
            "valid": True,
            "reason": "ok",
            "needs_files": [],
            "read_failed": "",
            "needs_run": "",
            "needs_glob": "",
            "glob_failed": "",
            "not_grounded": "todo.py",
            "accept": None,
            "accept_reason": "",
            "seat_admitted": None,
            "seat_contract_reason": "",
        }
    )
    assert outcome == {
        "finish": True,
        "content": (
            "No `todo.py` in this session (no successful build or read of "
            "it), so I can't explain its internals without guessing. If "
            "it's in your workspace, ask me to read it."
        ),
    }


def test_not_grounded_reason_states_it_not_the_failed_action() -> None:
    # minor 3 (review round 1): a target with a RECORDED attempt reason
    # (classify's not_grounded_reason, threaded from _visibility's
    # attempted dict) must not be told to do the exact thing that just
    # failed — the message states the reason instead.
    outcome = _emit(
        {
            "build": False,
            "file": "solution.py",
            "content": "Not grounded in this session.",
            "valid": True,
            "reason": "ok",
            "needs_files": [],
            "read_failed": "",
            "needs_run": "",
            "needs_glob": "",
            "glob_failed": "",
            "not_grounded": "big.py",
            "not_grounded_reason": "file exceeds the 96 KB read cap",
            "accept": None,
            "accept_reason": "",
            "seat_admitted": None,
            "seat_contract_reason": "",
        }
    )
    assert outcome == {
        "finish": True,
        "content": (
            "No `big.py` in this session: file exceeds the 96 KB read cap, "
            "so I can't explain its internals without guessing."
        ),
    }
    assert "ask me to read it" not in outcome["content"]


def test_not_grounded_without_a_reason_keeps_the_original_message() -> None:
    # backward compatibility: a gated payload that never sets
    # not_grounded_reason (the field defaults to "") keeps today's message
    # byte for byte.
    outcome = _emit(
        {
            "build": False,
            "file": "solution.py",
            "content": "Not grounded in this session.",
            "valid": True,
            "reason": "ok",
            "needs_files": [],
            "read_failed": "",
            "needs_run": "",
            "needs_glob": "",
            "glob_failed": "",
            "not_grounded": "todo.py",
            "not_grounded_reason": "",
            "accept": None,
            "accept_reason": "",
            "seat_admitted": None,
            "seat_contract_reason": "",
        }
    )
    assert outcome["content"].endswith("ask me to read it.")


def test_normal_decisions_carry_empty_not_grounded() -> None:
    outcome = _emit(
        {
            "build": False,
            "file": "solution.py",
            "content": "It adds two numbers.",
            "valid": True,
            "reason": "ok",
            "needs_files": [],
            "read_failed": "",
            "needs_run": "",
            "needs_glob": "",
            "glob_failed": "",
            "not_grounded": "",
            "accept": None,
            "accept_reason": "",
            "seat_admitted": None,
            "seat_contract_reason": "",
        }
    )
    assert outcome == {"finish": True, "content": "It adds two numbers."}


def test_seat_contract_rejection_uses_the_exported_prefix() -> None:
    # #133/#134 recap grounding: the ask-outcome ledger recognizes a rejected
    # build from this EXACT wire prefix, so emit's own literal must stay in
    # lockstep with the exported constant (never drift independently).
    outcome = _emit(
        {
            # `valid` because #155 made emit positively recognise its
            # form_gate dep, and a real form_gate output always carries it.
            # Do NOT relax the check to accommodate a partial fixture — that
            # reinstates the denylist the change exists to remove.
            "valid": True,
            "build": True,
            "seat_admitted": False,
            "seat_contract_reason": "Assertion 'x' raised exception",
        }
    )
    assert outcome["finish"] is True
    assert outcome["content"] == (
        f"{SEAT_CONTRACT_REJECT_PREFIX}Assertion 'x' raised exception"
    )


def test_accept_gate_rejection_uses_the_exported_prefix() -> None:
    outcome = _emit(
        {
            "build": True,
            "valid": True,
            "file": "a.py",
            "content": "def f():\n    pass",
            "accept": False,
            "accept_reason": "tests did not pass",
        }
    )
    assert outcome["finish"] is True
    assert outcome["content"] == f"{ACCEPT_GATE_REJECT_PREFIX}tests did not pass"


def test_build_invalid_refusal_uses_the_build_refused_prefix() -> None:
    # Review round 2 new blocker 2: a form-gate parse failure only ever
    # happens on a build turn (this branch requires build=True already —
    # proof enough on its own, no is_build_ask threading needed here), so
    # it always mints a BUILD outcome, never the ambiguous plain "Refused:".
    outcome = _emit(
        {
            "build": True,
            "valid": False,
            "file": "add.py",
            "content": "bad",
            "reason": "not valid Python",
        }
    )
    assert outcome["content"] == f"{BUILD_REFUSED_PREFIX}not valid Python"


def test_terminals_registry_agrees_with_the_exported_prefix_constants() -> None:
    # Round 2 major 3: emit renders from TERMINALS, not inline literals — the
    # registry entries must stay in lockstep with the individually exported
    # prefix constants (kept exported for the caller's dynamic import).
    assert TERMINALS["seat_contract"] == (
        SEAT_CONTRACT_REJECT_PREFIX,
        "rejected_contract",
    )
    assert TERMINALS["accept_gate"] == (ACCEPT_GATE_REJECT_PREFIX, "rejected_gate")
    assert TERMINALS["build_refused"] == (BUILD_REFUSED_PREFIX, "refused")
    assert TERMINALS["refused"] == (REFUSED_PREFIX, "")


def test_recall_answer_field_emits_the_honest_message() -> None:
    # #82 deep recall: the deterministic recall answer rides the routing
    # decision (a composed string field), emitted as a prose finish with no
    # seat involvement — the same shape as the not_grounded honest message.
    message = "You haven't asked me to build anything yet."
    # `valid` for the #155 reason above: a real form_gate output carries it.
    outcome = _emit({"valid": True, "recall_answer": message})
    assert outcome == {"finish": True, "content": message}


def test_needs_self_files_emits_a_self_reads_outcome() -> None:
    # #144 serve-native self-reference: the caller executes this read
    # natively — no client tool call, so the outcome is its own vocabulary.
    outcome = _emit(
        gated={
            "build": False,
            "file": "solution.py",
            "content": "Requesting self files.",
            "valid": True,
            "reason": "ok",
            "needs_files": [],
            "read_failed": "",
            "needs_self_files": [".llm-orc/scripts/agentic_serving/resolve.py"],
            "accept": None,
            "accept_reason": "",
            "seat_admitted": None,
            "seat_contract_reason": "",
        }
    )
    assert outcome == {
        "finish": False,
        "self_reads": [".llm-orc/scripts/agentic_serving/resolve.py"],
    }


def test_needs_grep_emits_a_grep_outcome() -> None:
    # #121 content-grep: the caller maps this to ONE grep tool call.
    outcome = _emit(
        gated={
            "build": False,
            "file": "solution.py",
            "content": "Searching definitions.",
            "valid": True,
            "reason": "ok",
            "needs_files": [],
            "read_failed": "",
            "needs_grep": "recall,ledger,built",
            "accept": None,
            "accept_reason": "",
            "seat_admitted": None,
            "seat_contract_reason": "",
        }
    )
    assert outcome == {"finish": False, "grep": "recall,ledger,built"}


def test_routing_failed_refuses_with_the_plain_prefix_and_never_writes() -> None:
    """#152 fail-closed routing: a turn with no readable routing decision
    refuses FIRST — before seam fields, gates, and the build path — with
    the non-minting plain prefix (an unreadable decision makes
    is_build_ask unknowable; under-report, never misreport). The junk
    content a dead seat produced must never ride out as a write or as a
    prose finish."""
    reason = (
        "serving pipeline error: no readable routing decision this turn "
        "(resolve: Script failed with exit code 1); nothing was built or "
        "written"
    )
    outcome = _emit(
        gated={
            "build": False,
            "file": "solution.py",
            "content": "",
            "valid": True,
            "reason": "ok",
            "accept": None,
            "accept_reason": "",
            "seat_admitted": None,
            "seat_contract_reason": "",
            # Competing seam field (review finding 9): shape zeroes every
            # seam field on the refusal path, so this state is only
            # reachable through a drifted shape — the ordering pin makes
            # "refuses FIRST" a tested property, not a comment.
            "recall_answer": "a drifted shape left this behind",
            "routing_failed": reason,
        }
    )
    assert outcome == {"finish": True, "content": f"Refused: {reason}"}
    assert "file" not in outcome


# --- #155 Arc A: emit positively recognises its form_gate dep ---------------


def _emit_raw(form_gate_response: str) -> dict[str, Any]:
    """Feed emit a RAW form_gate response, including one that is not JSON."""
    payload = json.dumps(
        {"dependencies": {"form_gate": {"response": form_gate_response}}}
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


_ENGINE_WRAP = (
    '{"success": false, "data": null, "error": "Schema JSON execution failed: '
    'Command \'[...]\' returned non-zero exit status 1.", "agent_requests": []}'
)


def test_a_crashed_form_gate_refuses_instead_of_finishing_empty() -> None:
    """#155: the permissive `except json.JSONDecodeError: gated = {}` turned
    "I could not read my input" into `{"finish": true, "content": ""}` — a
    successful-looking empty answer the client cannot tell from "the model
    had nothing to say". The engine's wrap for a dead serving node is the
    four-key schema-json shape, since these nodes always take the
    ScriptAgentInput path."""
    outcome = _emit_raw(_ENGINE_WRAP)

    assert outcome["finish"] is True
    assert outcome["content"].startswith(REFUSED_PREFIX)
    assert outcome["content"] != REFUSED_PREFIX


def test_form_gate_output_that_is_not_json_at_all_refuses() -> None:
    """Stdout pollution in a stdlib-only node used to degrade silently."""
    outcome = _emit_raw("Traceback (most recent call last):\n  boom\n")

    assert outcome["finish"] is True
    assert outcome["content"].startswith(REFUSED_PREFIX)


def test_a_form_gate_output_missing_valid_refuses() -> None:
    """Positive recognition, not a denylist: form_gate emits `valid` on every
    path (build and non-build alike), so its absence means this is not a
    form_gate output — whatever else it may be."""
    outcome = _emit_raw(json.dumps({"build": False, "content": "hello"}))

    assert outcome["finish"] is True
    assert outcome["content"].startswith(REFUSED_PREFIX)


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("needs_files", ["src/foo.py"], {"finish": False, "reads": ["src/foo.py"]}),
        (
            "needs_self_files",
            ["shape.py"],
            {"finish": False, "self_reads": ["shape.py"]},
        ),
        ("needs_glob", "foo", {"finish": False, "glob": "foo"}),
        ("needs_grep", "def foo", {"finish": False, "grep": "def foo"}),
        ("needs_run", "pytest -q", {"finish": False, "run": "pytest -q"}),
    ],
    ids=["reads", "self-reads", "glob", "grep", "run"],
)
def test_delegation_survives_the_new_readability_gate(
    field: str, value: Any, expected: dict[str, Any]
) -> None:
    """The delegation branches still fire once emit's `_readable_gate`
    check is in front of them.

    Renamed after review: an earlier name claimed this covered "a dead
    seat", which its fixture does not carry — no seat, no seat_contract,
    no failure of any kind. The route-level regression it was meant to
    guard is pinned end to end instead, with a really-crashed node, in
    test_a_crashed_seat_contract_does_not_break_a_delegation_route.
    """
    outcome = _emit({"valid": True, "build": False, "content": "", field: value})

    assert outcome == expected


def test_a_threaded_node_failure_refuses() -> None:
    """The THREADING, which the direct-feed pins above do not reach.

    A crashed shape is caught by form_gate, which still emits a
    well-formed output (carrying `valid`) with `node_failed` set. So
    emit's own readability check passes and the refusal depends entirely
    on emit honouring the threaded reason. Mutation showed that ignoring
    it left every other pin green, because none of them exercised the
    full shape -> form_gate -> emit path.
    """
    outcome = _emit(
        {
            "valid": True,
            "build": False,
            "content": "",
            "node_failed": "the shape node returned unreadable output",
        }
    )

    assert outcome["finish"] is True
    assert outcome["content"].startswith(REFUSED_PREFIX)
    assert "shape node" in outcome["content"]


def test_a_threaded_node_failure_beats_a_delegation_request() -> None:
    """Ordering: an unreadable upstream node means nothing downstream is
    trustworthy, including a delegation request that rode along with it."""
    outcome = _emit(
        {
            "valid": True,
            "build": False,
            "content": "",
            "needs_files": ["src/foo.py"],
            "node_failed": "the shape node returned unreadable output",
        }
    )

    assert outcome["finish"] is True
    assert outcome["content"].startswith(REFUSED_PREFIX)


# --- #155: the seat-gate branch, which round 2 shipped with no unit pin ----


def test_a_dead_seat_gate_refuses_a_build_that_would_otherwise_ship() -> None:
    """The wrong-accept this branch exists to prevent, and the only place
    it fires: the admission verdict is unknown and the turn would ship."""
    outcome = _emit(
        {
            "valid": True,
            "build": True,
            "file": "a.py",
            "content": "x = 1",
            "seat_gate_failed": "the seat contract node returned unreadable output",
        }
    )

    assert outcome["finish"] is True
    assert outcome["content"].startswith(BUILD_REFUSED_PREFIX)
    assert "nothing was built or written" in outcome["content"]


def test_a_dead_seat_gate_never_refuses_a_non_build_turn() -> None:
    """The `build and` guard, which review found unpinned: deleting it left
    all 3977 tests green while converting a healthy prose turn into a
    MINTING `Build refused:` — a ledger entry on a turn that carried no
    build ask, which is the #133/#134 invariant."""
    outcome = _emit(
        {
            "valid": True,
            "build": False,
            "content": "A tuple is immutable.",
            "seat_gate_failed": "the seat contract node returned unreadable output",
        }
    )

    assert outcome == {"finish": True, "content": "A tuple is immutable."}


def test_the_accept_gate_outranks_a_dead_seat_gate() -> None:
    """Gate precedence, which the design doc demanded a decision on and
    round 2 answered silently in the wrong direction.

    The accept gate holds a REAL verdict the system computed, carrying a
    retry invitation; the seat gate only reports that the admission
    verdict is unknown. Refusing ahead of it discarded the better answer
    and converted a `rejected_gate` ledger entry into a `refused` one.
    """
    outcome = _emit(
        {
            "valid": True,
            "build": True,
            "file": "a.py",
            "content": "x = 1",
            "accept": False,
            "accept_reason": "tests do not pass",
            "seat_gate_failed": "the seat contract node returned unreadable output",
        }
    )

    assert outcome["content"].startswith(ACCEPT_GATE_REJECT_PREFIX)
    assert "tests do not pass" in outcome["content"]


def test_a_seat_contract_rejection_outranks_a_dead_seat_gate() -> None:
    """Cannot co-occur in live wiring — both derive from one parse of one
    dep — but the ordering is asserted so a future split cannot silently
    demote an explicit rejection to a pipeline error."""
    outcome = _emit(
        {
            "valid": True,
            "build": True,
            "file": "a.py",
            "content": "x = 1",
            "seat_admitted": False,
            "seat_contract_reason": "no artifact",
            "seat_gate_failed": "the seat contract node returned unreadable output",
        }
    )

    assert outcome["content"].startswith(SEAT_CONTRACT_REJECT_PREFIX)
