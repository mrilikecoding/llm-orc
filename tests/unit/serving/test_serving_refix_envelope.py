"""#169: a re-fix candidate that is empty is never accepted.

``refix_select`` injects a smoke test when rung 1.5 found no visible test,
and its body is ``pass`` — which passes against any code, including none. So
an empty ``model_edit`` produced ``accept: true`` over nothing at all, and
the whole downstream chain admitted it: the seat contract asserts artifact
PRESENCE, ``ast.parse("")`` succeeds, and the caller writes any outcome
carrying ``file`` and ``content``. The target file is one the client already
has, so the write is a clobber.

Driven via subprocess the way the L0 engine runs a script node — the payload
carries ``input_data`` and ``dependencies``, which is what these nodes read,
not the engine's full four-key envelope. Same shape as every sibling harness
in this directory.
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

# _smoke below is the PRODUCTION function, not a copy, so the pins in THIS
# file cannot go green for a reason unrelated to the guard if the smoke test
# changes. #169's own issue lists "make the smoke test assert something about
# the candidate" as the alternative fix, so that edit is likely — and #171
# made it: the smoke test is now surface-derived from prior_code, so it is
# no longer a bare string constant.
#
# Round 1 justified this with a measurement that review round 2 falsified:
# it claimed the whole suite stayed green when refix_select._SMOKE_TEST was
# emptied or made always-failing. It did not — pre-existing pins in
# test_serving_refix_select.py and test_serving_ensemble_endpoint.py catch
# both. What the import actually buys is local: these pins stop depending on
# a copy that could silently diverge from the constant they are about.
#
# Every fixture in this file calls ``_smoke("")`` (no prior surface known) —
# this file is about #169/#173's emptiness/inertness guards, which are
# orthogonal to #171's participation guard; #171's own pins live in
# test_serving_refix_select.py and test_serving_gate_participation.py.
sys.path.insert(0, str(SCRIPTS))

from refix_select import _smoke_test as _smoke  # type: ignore  # noqa: E402

# A visible test that does NOT reference the target module. Realistic: rung
# 1.5's "visible test" is whatever test_<stem>.py was found, and a suite
# routinely contains cases that do not touch the module under repair. With an
# empty candidate this passes, which is what makes the pin below able to fail.
_VISIBLE_TEST = "def test_unrelated():\n    assert 1 + 1 == 2\n"
_REAL = "def restock(n):\n    return n + 1\n"
# The buggy stale version _REAL fixes — #171's smoke test is surface-
# derived from prior_code, so a genuine fix pin needs one to preserve.
_PRIOR_RESTOCK = "def restock(n):\n    return n\n"

# #173 review round 1: measured through the real chain, all 15 clobber under
# the pre-round-2 predicate ("every statement is a bare string"). Each binds
# no name and calls nothing, so none can make the smoke test pass for a real
# reason. Distinct from ``test_a_bare_numeric_constant_supersedes_the_one_
# character_pin`` below, which documents the #169 pin this class of fix
# supersedes rather than sweeping it into this list.
_INERT_MEMBERS: list[tuple[str, str]] = [
    ("pass-only", "pass\n"),
    ("ellipsis-only", "...\n"),
    ("bare-int", "42\n"),
    ("bare-none", "None\n"),
    ("bare-bool", "True\n"),
    ("f-string-no-placeholder", 'f"a"\n'),
    ("f-string-with-placeholder", 'f"{1}"\n'),
    ("string-concat", '"a" + "b"\n'),
    ("bytes-literal", 'b"x"\n'),
    ("if-false-pass", "if False:\n    pass\n"),
    ("while-false-pass", "while False:\n    pass\n"),
    ("docstring-then-pass", '"""doc"""\npass\n'),
    ("comment-then-pass", "# nope\npass\n"),
    ("pass-semicolon-pass", "pass; pass\n"),
]


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


def _envelope(
    code: str, *, visible_test: str = "", prior_code: str = ""
) -> dict[str, Any]:
    """executor -> envelope, both real, over a SYNTHESIZED select output.

    ``refix_select.py`` is not run here — the dict below is built to the
    shape it emits, verified key-for-key against the real node — so the
    smoke test comes from the production function rather than from select
    having injected it. ``code`` is the candidate and the only fault.

    ``prior_code`` defaults to "" (no surface known) — every fixture in
    this file is about #169/#173's emptiness/inertness guards, which do not
    need one; the #171 participation pins that DO need a real surface pass
    it explicitly (see TestTheGuardDoesNotRejectRealCandidates below).
    """
    smoke_only = not visible_test.strip()
    smoke_text, smoke_has_surface = _smoke(prior_code)
    selected = {
        "requirement": "fix calc.py so restock adds one",
        "code": code,
        "tests": visible_test or smoke_text,
        "target_file": "calc.py",
        "edit_kind": "model",
        "smoke_only": smoke_only,
        "smoke_surface_empty": smoke_only and not smoke_has_surface,
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

        assert envelope["_tests_pass"] is True, (
            "the smoke test must still pass against whitespace, or this pins "
            "the wrong thing"
        )
        assert envelope["diagnostics"]["accept"] is False


class TestTheGuardDoesNotRejectRealCandidates:
    """Over-refusal pins. These CANNOT fail under deletion of the guard —
    they are here so the fix does not become "reject every re-fix"."""

    def test_a_real_candidate_on_the_smoke_only_path_still_accepts(self) -> None:
        """#171: the smoke test is surface-derived from prior_code, so a
        genuine fix needs a prior surface to preserve — without one an
        "import solution"-only check cannot prove participation for
        anything, real or junk (the shape the tests just above this class
        exist to refuse)."""
        envelope = _envelope(_REAL, prior_code=_PRIOR_RESTOCK)

        assert envelope["diagnostics"]["accept"] is True
        assert envelope["artifacts"][0]["content"] == _REAL

    def test_a_one_line_edit_with_no_prior_surface_accepts_under_the_fallback_bar(
        self,
    ) -> None:
        """F-1 (#171 round 2 review): supersedes this test's own prior claim.
        #171 landed a rule where a surface-less prior refused EVERY
        candidate: the smoke-only bar degrades to "import solution" alone,
        which the ablation control satisfies identically with the
        deliverable's bytes absent (an import-only check observes nothing),
        so ``participates`` was False for the whole class of surface-less
        prior modules and the route could never converge.

        Round 3 correction: the fallback is scoped to priors with ZERO
        PUBLIC BINDINGS OF ANY KIND (an empty module, same as no prior_code
        known at all — see below for the import-only ``__init__`` shape).
        A constants-only or dict-only prior is NOT this case any more (the
        surface now includes top-level assignment targets); this test's own
        prior is "" (nothing known), which has none. ``x = 1`` is not on
        #173's inert whitelist (``Assign`` isn't a member), so it accepts
        here exactly as it did pre-#171 — #173's own whitelist, unchanged,
        is what still catches genuine junk against a zero-binding prior;
        this measured case is not that."""
        envelope = _envelope("x = 1\n")

        assert envelope["diagnostics"]["accept"] is True, envelope["diagnostics"][
            "accept_reason"
        ]

    def test_an_import_only_prior_also_falls_back_to_the_loads_cleanly_bar(
        self,
    ) -> None:
        """The other zero-public-binding shape (an ``__init__`` re-export
        module): imports were never part of the surface, before or after
        the round-3 widening, so a prior that only imports names still has
        nothing to check and falls back to the pre-#171 bar."""
        prior = "from foo import bar\nfrom baz import qux\n"
        envelope = _envelope("x = 1\n", prior_code=prior)

        assert envelope["diagnostics"]["accept"] is True, envelope["diagnostics"][
            "accept_reason"
        ]

    def test_a_constants_only_prior_with_a_legitimate_fix_accepts(self) -> None:
        """F-1's own repro, round 3 (widened, not skipped): a constants-only
        settings module's top-level assignment targets (PORT/DEBUG/RETRIES)
        are now PART of the smoke surface — a legitimate one-value fix that
        keeps every name still accepts, via the genuine per-name check, not
        a bypass."""
        prior = "PORT = 8080\nDEBUG = False\nRETRIES = 3\n"
        fixed = "PORT = 9090\nDEBUG = False\nRETRIES = 3\n"
        envelope = _envelope(fixed, prior_code=prior)

        assert envelope["diagnostics"]["accept"] is True, envelope["diagnostics"][
            "accept_reason"
        ]

    def test_a_dict_only_rates_table_fix_accepts(self) -> None:
        """F-1's second repro shape, round 3: a dict-only rates table's
        ``RATES`` binding is part of the widened surface; a fix that keeps
        the name (only the dict's value changes) accepts."""
        prior = "RATES = {'a': 1, 'b': 2}\n"
        fixed = "RATES = {'a': 1, 'b': 3}\n"
        envelope = _envelope(fixed, prior_code=prior)

        assert envelope["diagnostics"]["accept"] is True, envelope["diagnostics"][
            "accept_reason"
        ]

    def test_a_junk_edit_against_a_constants_only_prior_now_refuses(self) -> None:
        """Round 3 correction (the coordinator's own finding): F-1's
        fallback re-opened the exact clobber #173 closed for def-bearing
        modules — ``x = 1`` shipped against a constants-only prior. The
        surface is widened instead of skipped: PORT/DEBUG/RETRIES are
        public top-level bindings same as a function name, so a junk
        deliverable that drops all three refuses, with the same
        dropped-name reason wording a dropped function gets (F-2)."""
        prior = "PORT = 8080\nDEBUG = False\nRETRIES = 3\n"
        envelope = _envelope("x = 1\n", prior_code=prior)
        reason = envelope["diagnostics"]["accept_reason"]

        assert envelope["diagnostics"]["accept"] is False
        assert "no longer defines" in reason
        assert any(name in reason for name in ("PORT", "DEBUG", "RETRIES"))

    def test_a_fix_that_drops_a_public_constant_refuses(self) -> None:
        """Recorded bound, same as functions (F-2's own dropped-name bound,
        now extended to constants): a fix that intentionally drops a public
        constant the prior module exported refuses, whichever was the
        right call to make."""
        prior = "PORT = 8080\nDEBUG = False\nRETRIES = 3\n"
        candidate = "PORT = 9090\nDEBUG = False\n"
        envelope = _envelope(candidate, prior_code=prior)
        reason = envelope["diagnostics"]["accept_reason"]

        assert envelope["diagnostics"]["accept"] is False
        assert "RETRIES" in reason

    def test_a_dropped_name_reason_states_the_fact_without_quoting_test_source(
        self,
    ) -> None:
        """Review M3: the smoke-only reason used to say "failed to load"
        for a candidate that loaded FINE but dropped a name the prior
        module provided, and it quoted the internal smoke test's own
        assert source line — an implementation detail of the ablation's
        OWN test, not something the user wrote. Naming the dropped name is
        fine; quoting internal test source is not."""
        prior = (
            "def restock(item, n):\n    return n + 1\n"
            "def audit(item):\n    return item\n"
        )
        envelope = _envelope(_REAL, prior_code=prior)
        reason = envelope["diagnostics"]["accept_reason"]

        assert envelope["diagnostics"]["accept"] is False
        assert "failed to load" not in reason, reason
        assert "assert hasattr" not in reason, reason
        assert "audit" in reason, reason


class TestAnInertCandidateIsNeverAccepted:
    """#173: #169's emptiness check (``code.strip()``) is one ``#`` character
    wide. A comment-only or docstring-only candidate parses, satisfies the
    injected smoke test (a ``pass`` body is satisfied by no code), and would
    otherwise ship — clobbering a file the client already has."""

    def test_a_comment_only_candidate_is_rejected(self) -> None:
        envelope = _envelope("# TODO: implement the fix\n")

        assert envelope["_tests_pass"] is True, (
            "the smoke test must still pass against a comment, or this pins "
            "the wrong thing"
        )
        assert envelope["diagnostics"]["accept"] is False

    def test_a_docstring_only_candidate_is_rejected(self) -> None:
        envelope = _envelope('"""placeholder - could not fix."""\n')

        assert envelope["_tests_pass"] is True, (
            "the smoke test must still pass against a bare docstring, or "
            "this pins the wrong thing"
        )
        assert envelope["diagnostics"]["accept"] is False

    def test_a_comment_and_docstring_with_no_code_is_rejected(self) -> None:
        envelope = _envelope('# nope\n"""placeholder"""\n')

        assert envelope["_tests_pass"] is True, (
            "the smoke test must still pass against a comment plus a "
            "docstring, or this pins the wrong thing"
        )
        assert envelope["diagnostics"]["accept"] is False

    def test_the_reject_reason_names_the_target_and_the_defect(self) -> None:
        envelope = _envelope("# TODO: implement the fix\n")
        reason = envelope["diagnostics"]["accept_reason"]

        assert "executable" in reason.lower()
        assert "calc.py" in reason, "the refusal must name what was not written"

    def test_an_unparseable_candidate_is_rejected_and_does_not_crash(self) -> None:
        """#173's guard must not itself raise on code the load gate already
        rejects (SyntaxError at load, #169's mechanism) — fail closed, never
        crash the node."""
        envelope = _envelope("def broken(:\n")

        assert envelope["_tests_pass"] is False, (
            "the load gate must still catch this, or this pins the wrong thing"
        )
        assert envelope["diagnostics"]["accept"] is False

    def test_a_bare_numeric_constant_supersedes_the_one_character_pin(self) -> None:
        """Supersedes #169's ``test_a_one_character_candidate_still_accepts``,
        which pinned ``_envelope("0\\n")`` as accept=True with the rationale
        "the rule is emptiness, not a length heuristic." That rationale was
        correct against a length heuristic and is superseded by #173's
        effect-based rule: a bare ``0`` binds no name and calls nothing, so
        it can satisfy the smoke test for no reason at all. It now rejects."""
        envelope = _envelope("0\n")

        assert envelope["_tests_pass"] is True, (
            "the smoke test must still pass against a bare constant, or "
            "this pins the wrong thing"
        )
        assert envelope["diagnostics"]["accept"] is False

    @pytest.mark.parametrize(
        "code",
        [code for _, code in _INERT_MEMBERS],
        ids=[label for label, _ in _INERT_MEMBERS],
    )
    def test_a_structurally_inert_candidate_is_rejected(self, code: str) -> None:
        """#173 review round 1: the emptiness-plus-bare-string-literal
        predicate implemented "every statement is a bare string" while its
        own docstring claimed "no statement does anything" — a 15-member
        gap, measured through the real chain to clobber. This sweep pins
        the members not already covered by a dedicated test above."""
        envelope = _envelope(code)

        assert envelope["_tests_pass"] is True, (
            "the smoke test must still pass against this candidate, or "
            "this pins the wrong thing"
        )
        assert envelope["diagnostics"]["accept"] is False


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
    envelope = _envelope("")
    assert envelope["_tests_pass"] is True, (
        "the smoke test must still pass against no code, or this pins the wrong thing"
    )
    # The premise key is stripped before the marshal runs: driving the real
    # chain with a shape production never emits is the habit #155's lesson is
    # about, even when the extra key is inert (measured: emit's output is
    # byte-identical either way).
    outcome = _serving_tail({k: v for k, v in envelope.items() if k != "_tests_pass"})

    assert outcome.get("finish") is True, outcome
    assert "file" not in outcome, "an empty candidate reached the client as a write"
    assert "empty" in str(outcome.get("content", "")).lower()


def test_an_inert_candidate_reaches_the_client_as_a_rejection() -> None:
    """#173 review round 1, F2 / the issue's instrument 5: a node-level pin
    does not prove the chain (#155's lesson), so a ``pass``-only candidate —
    the headline case, literally the injected smoke test's own body — goes
    through the real marshal nodes rather than being read directly. Same
    shape as ``test_an_empty_candidate_reaches_the_client_as_a_rejection``
    above, for the inert rather than the empty case.
    """
    envelope = _envelope("pass\n")
    assert envelope["_tests_pass"] is True, (
        "the smoke test must still pass against a pass-only candidate, or "
        "this pins the wrong thing"
    )
    outcome = _serving_tail({k: v for k, v in envelope.items() if k != "_tests_pass"})

    assert outcome.get("finish") is True, outcome
    assert "file" not in outcome, "an inert candidate reached the client as a write"
    assert "executable" in str(outcome.get("content", "")).lower()
