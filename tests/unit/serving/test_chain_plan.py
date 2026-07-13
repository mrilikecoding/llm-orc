"""Unit tests for the chain-plan table (WS-3 / issue #120).

``chain_plan.advance`` replaces ``classify._route``/``_fix_chain_route`` — a
flat first-match scan over a declarative ``Step`` table instead of an
if/elif cascade. The routing corpus below pins every table row's happy path
plus the ordering-sensitive edges (docs/plans/2026-07-12-chain-executor-
design.md; the byte-identical migration contract): each expected
``(target, kind, build, needs_decider)`` tuple was captured from a green
parallel run against the (now-deleted) ``classify._route`` before it was
removed, so this suite stays the durable proof the transpose never changed
a routing decision.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
SCRIPTS = REPO / ".llm-orc" / "scripts" / "agentic_serving"
sys.path.insert(0, str(SCRIPTS))

from chain_plan import SignalBundle as _SignalBundle  # type: ignore  # noqa: E402
from chain_plan import advance as _advance  # noqa: E402

_DEFAULTS: dict[str, Any] = {
    "is_explain": False,
    "explain_ungrounded": False,
    "run_signal": False,
    "fix_chain": False,
    "has_run_block": False,
    "needs_another_run": False,
    "has_refixed": False,
    "failure_shape": "",
    "needs_glob": "",
    "glob_failed": "",
    "needs_files": [],
    "read_failed": "",
    "tests_primary": False,
    "has_build_signal": False,
    "kind_hint": "python_module",
}


def _decide(**overrides: Any) -> tuple[str, str, bool, bool]:
    bundle = _SignalBundle(**{**_DEFAULTS, **overrides})
    decision = _advance(bundle)
    return decision.target, decision.kind, decision.build, decision.needs_decider


# --- one case per table row (happy path) ---


def test_row1_fix_cont_needs_another_run() -> None:
    assert _decide(fix_chain=True, needs_another_run=True) == (
        "need-run",
        "need_run",
        False,
        False,
    )


def test_row2_fix_cont_localized_re_fixes() -> None:
    assert _decide(fix_chain=True, has_refixed=False, failure_shape="localized") == (
        "re-fix",
        "re_fix",
        True,
        False,
    )


def test_row3_fix_cont_falls_through_to_run_verdict() -> None:
    assert _decide(fix_chain=True, failure_shape="structural") == (
        "run-verdict",
        "run_verdict",
        False,
        False,
    )


def test_row4_run_with_run_block_is_run_verdict() -> None:
    assert _decide(run_signal=True, has_run_block=True) == (
        "run-verdict",
        "run_verdict",
        False,
        False,
    )


def test_row5_run_without_run_block_needs_run() -> None:
    assert _decide(run_signal=True, has_run_block=False) == (
        "need-run",
        "need_run",
        False,
        False,
    )


def test_row6_explain_ungrounded_is_not_grounded() -> None:
    assert _decide(is_explain=True, explain_ungrounded=True) == (
        "not-grounded",
        "not_grounded",
        False,
        False,
    )


def test_row7_explain_grounded_routes_to_the_explainer_seat() -> None:
    assert _decide(is_explain=True, explain_ungrounded=False) == (
        "explainer",
        "explanation",
        False,
        False,
    )


def test_defer_recall_falls_through_to_the_decider() -> None:
    # #82 deep recall: a loose maybe_recall turn (is_explain, but the tight
    # _RECALL_RE did NOT resolve it structurally) defers to the guarded
    # model-decider rather than assuming the guessing explainer seat. It
    # outranks _explain_explainer so an ambiguous ordinal question is never
    # answered by speculation.
    assert _decide(is_explain=True, defer_recall=True) == ("", "", False, True)


def test_row8_build_needs_glob() -> None:
    assert _decide(needs_glob="storage") == (
        "need-glob",
        "need_glob",
        False,
        False,
    )


def test_row8_build_glob_failed_alone_also_needs_glob_target() -> None:
    assert _decide(glob_failed="no file matching 'storage'") == (
        "need-glob",
        "need_glob",
        False,
        False,
    )


def test_row9_build_needs_files() -> None:
    assert _decide(needs_files=["a.py"]) == (
        "need-files",
        "need_files",
        False,
        False,
    )


def test_row9_build_read_failed_alone_also_needs_files_target() -> None:
    assert _decide(read_failed="could not read a.py: client read failed") == (
        "need-files",
        "need_files",
        False,
        False,
    )


def test_row10_build_tests_primary_routes_to_the_tests_seat() -> None:
    assert _decide(tests_primary=True) == (
        "tests-seat",
        "python_tests",
        True,
        False,
    )


def test_row11_build_code_seat_kind_is_the_dynamic_kind_hint() -> None:
    assert _decide(has_build_signal=True, kind_hint="python_module") == (
        "code-seat",
        "python_module",
        True,
        False,
    )


def test_row11_build_code_seat_kind_hint_passes_through_unchanged() -> None:
    # proves row 11's kind is genuinely dynamic, not a memorized literal
    assert _decide(has_build_signal=True, kind_hint="python_script") == (
        "code-seat",
        "python_script",
        True,
        False,
    )


def test_row12_empty_bundle_falls_through_to_the_decider() -> None:
    assert _decide() == ("", "", False, True)


# --- ordering-sensitive edges (brief: fix_chain x {needs_another_run,
# localized+not-refixed, has_refixed, structural}) ---


def test_fix_chain_has_refixed_blocks_re_fix_even_when_localized() -> None:
    # the one-round bound: a second write's run has already come back —
    # even a localized red verdict must not re-fix a second time
    assert _decide(fix_chain=True, has_refixed=True, failure_shape="localized") == (
        "run-verdict",
        "run_verdict",
        False,
        False,
    )


def test_fix_chain_structural_stays_on_the_honest_red_terminal() -> None:
    assert _decide(fix_chain=True, has_refixed=False, failure_shape="structural") == (
        "run-verdict",
        "run_verdict",
        False,
        False,
    )


# --- ordering-sensitive edges: needs_glob + needs_files both set (glob
# must win) ---


def test_glob_and_files_both_set_glob_wins() -> None:
    assert _decide(needs_glob="storage", needs_files=["a.py"]) == (
        "need-glob",
        "need_glob",
        False,
        False,
    )


# --- ordering-sensitive edges: tests_primary + has_build_signal both set
# (tests wins) ---


def test_tests_primary_and_build_signal_both_set_tests_wins() -> None:
    assert _decide(tests_primary=True, has_build_signal=True) == (
        "tests-seat",
        "python_tests",
        True,
        False,
    )


# --- cross-chain priority proofs: the table order is the whole contract ---


def test_fix_chain_outranks_every_other_signal() -> None:
    decision = _decide(
        fix_chain=True,
        needs_another_run=True,
        run_signal=True,
        has_run_block=True,
        is_explain=True,
        explain_ungrounded=True,
        needs_glob="storage",
        needs_files=["a.py"],
        tests_primary=True,
        has_build_signal=True,
    )
    assert decision == ("need-run", "need_run", False, False)


def test_run_signal_outranks_explain_and_build_when_not_fix_chain() -> None:
    decision = _decide(
        run_signal=True,
        has_run_block=True,
        is_explain=True,
        explain_ungrounded=True,
        needs_glob="storage",
        tests_primary=True,
        has_build_signal=True,
    )
    assert decision == ("run-verdict", "run_verdict", False, False)


def test_explain_outranks_build_when_not_fix_chain_or_run() -> None:
    decision = _decide(
        is_explain=True,
        explain_ungrounded=False,
        needs_glob="storage",
        needs_files=["a.py"],
        tests_primary=True,
        has_build_signal=True,
    )
    assert decision == ("explainer", "explanation", False, False)


# --- the table walk itself: first-match order and step bookkeeping ---


def test_advance_tags_the_firing_step_with_its_chain_and_index() -> None:
    decision = _advance(_SignalBundle(**{**_DEFAULTS, "tests_primary": True}))
    assert decision.chain == "build"
    assert decision.step_index == 2  # need-glob(0), need-files(1), tests-seat(2)


def test_advance_reports_the_decider_step_index() -> None:
    decision = _advance(_SignalBundle(**_DEFAULTS))
    assert decision.chain == "decider"
    assert decision.step_index == 0
