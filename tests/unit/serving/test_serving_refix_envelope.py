"""#169: a re-fix candidate that is empty is never accepted.

``refix_select`` injects a smoke test when rung 1.5 found no visible test,
and its body is ``pass`` — which passes against any code, including none. So
an empty ``model_edit`` produced ``accept: true`` over nothing at all, and
the whole downstream chain admitted it: the seat contract asserts artifact
PRESENCE, ``ast.parse("")`` succeeds, and the caller writes any outcome
carrying ``file`` and ``content``. The target file is one the client already
has, so the write is a clobber.

Driven via subprocess exactly as the L0 engine runs a script node.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
SCRIPTS = REPO / ".llm-orc" / "scripts" / "agentic_serving"

# _SMOKE below is the PRODUCTION constant, not a copy. Review round 1: a
# frozen copy left the whole suite green when refix_select._SMOKE_TEST was
# replaced with an always-failing test or emptied outright — so every pin
# here could go green for a different reason and `candidate_present` become
# silently deletable. #169's own issue lists "make the smoke test assert
# something about the candidate" as the alternative fix, which is exactly
# that edit.
sys.path.insert(0, str(SCRIPTS))

from refix_select import _SMOKE_TEST as _SMOKE  # type: ignore  # noqa: E402

# A visible test that does NOT reference the target module. Realistic: rung
# 1.5's "visible test" is whatever test_<stem>.py was found, and a suite
# routinely contains cases that do not touch the module under repair. With an
# empty candidate this passes, which is what makes the pin below able to fail.
_VISIBLE_TEST = "def test_unrelated():\n    assert 1 + 1 == 2\n"
_REAL = "def restock(n):\n    return n + 1\n"


def _node(script: str, deps: dict[str, Any], input_data: str = "") -> dict[str, Any]:
    payload: dict[str, Any] = {"input_data": input_data, "dependencies": deps}
    out = subprocess.run(
        [sys.executable, str(SCRIPTS / script)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    result: dict[str, Any] = json.loads(out)
    return result


def _dep(value: Any) -> dict[str, str]:
    return {"response": value if isinstance(value, str) else json.dumps(value)}


def _envelope(code: str, *, visible_test: str = "") -> dict[str, Any]:
    """executor -> envelope, both real, over a SYNTHESIZED select output.

    ``refix_select.py`` is not run here — the dict below is built to the
    shape it emits, verified key-for-key against the real node — so the
    smoke test comes from the production constant rather than from select
    having injected it. ``code`` is the candidate and the only fault.
    """
    selected = {
        "requirement": "fix calc.py so restock adds one",
        "code": code,
        "tests": visible_test or _SMOKE,
        "target_file": "calc.py",
        "edit_kind": "model",
        "smoke_only": not visible_test,
    }
    executor = _node("accept_executor.py", {"select": _dep(selected)})
    envelope = _node(
        "refix_envelope.py",
        {"select": _dep(selected), "executor": _dep(executor)},
    )
    # The premise: the executor's verdict is NOT what stops an empty
    # candidate. If the smoke test ever starts failing on nothing, these
    # pins would go green for a reason that has nothing to do with the
    # guard, and the guard would be deletable (review round 1).
    envelope["_tests_pass"] = executor["tests_pass"]
    return envelope


class TestAnEmptyCandidateIsNeverAccepted:
    """Invariant pins — each must go RED under deletion of the guard."""

    def test_an_empty_candidate_on_the_smoke_only_path_is_rejected(self) -> None:
        """The issue's reproduction. The smoke test really does pass with no
        code, so the executor's verdict is not what stops this."""
        envelope = _envelope("")

        assert envelope["_tests_pass"] is True, (
            "the smoke test must still pass against no code, or this pins "
            "the wrong thing"
        )
        assert envelope["diagnostics"]["accept"] is False

    def test_the_reject_reason_names_emptiness(self) -> None:
        """Two gates can refuse a smoke-only re-fix — the candidate failed to
        load, or the candidate was empty — and they call for different
        operator responses. The reason has to tell them apart."""
        envelope = _envelope("")
        reason = envelope["diagnostics"]["accept_reason"]

        assert "empty" in reason.lower()
        assert "calc.py" in reason, "the refusal must name what was not written"

    def test_an_empty_candidate_with_a_visible_test_is_rejected_too(self) -> None:
        """The rule is not scoped to the smoke-only path, and this pin can
        fail: measured, a visible test that does not reference the target
        module reports tests_pass=True against an EMPTY candidate, exactly as
        the injected smoke test does. Scoping the guard to smoke_only would
        leave that open."""
        envelope = _envelope("", visible_test=_VISIBLE_TEST)

        assert envelope["_tests_pass"] is True, (
            "the visible test must still pass against no code, or this pins "
            "the wrong thing"
        )
        assert envelope["diagnostics"]["accept"] is False

    def test_a_whitespace_only_candidate_is_rejected(self) -> None:
        """Kills a ``== ""`` implementation."""
        envelope = _envelope("   \n\t\n")

        assert envelope["diagnostics"]["accept"] is False


class TestTheGuardDoesNotRejectRealCandidates:
    """Over-refusal pins. These CANNOT fail under deletion of the guard —
    they are here so the fix does not become "reject every re-fix"."""

    def test_a_real_candidate_on_the_smoke_only_path_still_accepts(self) -> None:
        envelope = _envelope(_REAL)

        assert envelope["diagnostics"]["accept"] is True
        assert envelope["artifacts"][0]["content"] == _REAL

    def test_a_one_character_candidate_still_accepts(self) -> None:
        """The rule is emptiness, not a length heuristic. ``0`` rather than
        ``x``: a bare name raises NameError at module exec, so it fails the
        load gate for a real reason and would pin nothing here."""
        envelope = _envelope("0\n")

        assert envelope["diagnostics"]["accept"] is True


def _serving_tail(envelope: dict[str, Any]) -> dict[str, Any]:
    """seat_contract -> shape -> form_gate -> emit, all real, with the
    envelope wired as the dispatched seat's child result (terminal node
    LAST, since _helpers.terminal peels ``results`` by last key)."""
    seat = json.dumps({"results": {"select": _dep("..."), "envelope": _dep(envelope)}})
    resolve = _dep(
        {
            "target": "re-fix",
            "kind": "re_fix",
            "file": "calc.py",
            "build": True,
            "dispatch_input": "fix it",
            "is_build_ask": True,
        }
    )
    deps: dict[str, Any] = {"resolve": resolve, "seat": {"response": seat}}
    seat_contract = _node("seat_contract.py", deps)
    shape = _node("shape.py", {**deps, "seat_contract": _dep(seat_contract)})
    form_gate = _node("form_gate.py", {"shape": _dep(shape)})
    return _node("emit.py", {"form_gate": _dep(form_gate)})


def test_an_empty_candidate_reaches_the_client_as_a_rejection() -> None:
    """End to end, and distinct from #166's pin for the same input: that one
    asserts the empty deliverable is not WRITTEN, this one asserts the turn
    is REJECTED. Without this the smoke-only path keeps reporting a verdict
    of accept for nothing at all, so the ledger would record a turn that
    shipped-then-refused rather than one the gate turned down.

    A node-level pin does not prove the chain (#155's lesson), so the
    envelope goes through the real marshal nodes rather than being read
    directly.
    """
    outcome = _serving_tail(_envelope(""))

    assert outcome.get("finish") is True, outcome
    assert "file" not in outcome, "an empty candidate reached the client as a write"
    assert "empty" in str(outcome.get("content", "")).lower()
