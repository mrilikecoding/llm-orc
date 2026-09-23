"""Unit tests for the dependency-response accessors' honest-absence
contract (SF7).

A failed or skipped dependency's ``response`` field is PRESENT but
``None`` (AgentResult's default, or the skip record's), not absent.
``_helpers.response`` and ``accept_gate._dep_response`` used to read it
with ``dep.get("response", "")`` and, finding it non-string, fall to
``json.dumps(resp)`` — which turns ``None`` into the literal three-byte
string ``"null"``: a value every downstream ``json.loads`` and every
code/text extractor accepts as real content (``extract_code`` shipping
the literal string "null" as a code deliverable, a routing decision
reading a parsed ``None`` where its dict was expected). Both accessors
now treat a ``None`` response identically to an absent dependency: the
empty string, matching the shape the callers' own ``except
(JSONDecodeError, TypeError): ...`` / ``isinstance(..., dict)`` guards
already expect.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / ".llm-orc" / "scripts" / "agentic_serving"))

from _helpers import (  # type: ignore[import-not-found]  # noqa: E402
    extract_code,
    response,
    terminal,
)
from accept_gate import _dep_response  # type: ignore[import-not-found]  # noqa: E402

_extract_code = extract_code
_response = response
_terminal = terminal


class TestHelpersResponseNeverStringifiesNoneToNull:
    def test_none_response_is_the_empty_string(self) -> None:
        dep = {"status": "failed", "error": "boom", "response": None}

        assert _response(dep) == ""

    def test_absent_response_key_is_also_the_empty_string(self) -> None:
        """Present-but-None and entirely-absent behave identically —
        both mean "nothing to read", not "the JSON value null"."""
        assert _response({"status": "failed"}) == ""

    def test_a_real_string_response_is_unaffected(self) -> None:
        dep = {"status": "success", "response": '{"ok": true}'}

        assert _response(dep) == '{"ok": true}'

    def test_extract_code_never_receives_the_literal_word_null(self) -> None:
        """The bug's concrete failure mode: a failed code_writer used to
        make its deliverable the 4-character string "null" rather than
        an honest empty string."""
        dep = {"status": "failed", "error": "exited non-zero", "response": None}

        code = _extract_code(_terminal(_response(dep)), drop_test_blocks=True)

        assert code == ""

    def test_terminal_of_a_failed_dependency_is_empty_not_null(self) -> None:
        dep = {"status": "failed", "response": None}

        assert _terminal(_response(dep)) == ""

    def test_terminal_of_a_null_last_results_node_is_empty_not_none(self) -> None:
        """SF3: terminal peels a results envelope whose LAST node's own
        response is present-but-None (a failed dependency INSIDE that
        envelope, e.g. a crashed sub-ensemble terminal) — must return
        "", not None. extract_code(None) used to raise TypeError in
        every caller (build_gated_envelope.py, accept_gather.py,
        emit_envelope.py, refix_select.py, seat_contract.py)."""
        import json

        envelope = json.dumps(
            {"results": {"last": {"response": None, "status": "failed"}}}
        )

        result = _terminal(envelope)

        assert result == ""
        assert isinstance(result, str)
        # the actual failure mode: extract_code must not raise TypeError
        assert _extract_code(result) == ""


class TestAcceptGateDepResponseNeverStringifiesNoneToNull:
    def test_none_response_is_the_empty_string(self) -> None:
        deps = {"executor": {"status": "failed", "error": "boom", "response": None}}

        assert _dep_response(deps, "executor") == ""

    def test_missing_dependency_is_also_the_empty_string(self) -> None:
        assert _dep_response({}, "executor") == ""

    def test_a_real_string_response_is_unaffected(self) -> None:
        deps = {"executor": {"status": "success", "response": '{"tests_pass": true}'}}

        assert _dep_response(deps, "executor") == '{"tests_pass": true}'
