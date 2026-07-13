#!/usr/bin/env python3
"""Serving Ensemble — chain-plan table (WS-3 / issue #120).

``classify._route``/``_fix_chain_route`` (docs/plans/2026-07-12-chain-
executor-design.md), transposed to data: a ``Step`` is one routing branch —
``{chain_label, target, kind, build, guard}`` — and ``advance()`` is a flat
first-match scan over the steps in the SAME priority order the original
if/elif cascade checked its conditions. The four ad-hoc serving chains
(read->build, glob->read->build, write->run->verdict, convergent re-fix)
read off this table statelessly; the ``chain`` label groups steps by
lifecycle but never affects which step fires — that IS the guard's own
condition (the full ``_route`` condition, chain-selecting signal included),
so the transpose cannot silently reorder anything.

Run outranks marker-based explain (``run_signal`` is already false on an
interrogative or marker-led turn), so "run the tests and tell me what
failed" delegates the run instead of narrating one — the run rows sit ahead
of the explain rows below for that reason.

``classify.main()`` builds a ``SignalBundle`` from its already-computed
signals and calls ``advance(bundle)``; this module owns no signal
extraction of its own.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

# The canonical seat-target strings (moved from classify.py). Two of the
# three — _EXPLAIN_SEAT and _TESTS_SEAT — are imported back into classify.py
# for its post-routing composition, so a rename can't desync the two;
# _DEFAULT_CODE_SEAT is referenced only here.
_DEFAULT_CODE_SEAT = "code-seat"
_EXPLAIN_SEAT = "explainer"
_TESTS_SEAT = "tests-seat"


@dataclass(frozen=True)
class SignalBundle:
    """The routing signals ``_route`` used to read as keyword arguments —
    its former signature IS this schema, copied field for field, so no
    signal can be silently dropped in the transpose."""

    is_explain: bool
    explain_ungrounded: bool
    run_signal: bool
    fix_chain: bool
    has_run_block: bool
    needs_another_run: bool
    has_refixed: bool
    failure_shape: str
    needs_glob: str
    glob_failed: str
    needs_files: list[str]
    read_failed: str
    tests_primary: bool
    has_build_signal: bool
    kind_hint: str
    # #82 deep recall: an ordinal-recall turn whose deterministic selection
    # resolves to an honest message (rejected/none/built-deep), NOT the
    # guessing explainer seat. Defaulted so existing bundles are unchanged;
    # the shipped-and-visible case rides grounded-explain via a named_file
    # injection instead, never this flag.
    is_recall_answer: bool = False
    # #82 deep recall (detection layer 2): a loose maybe_recall turn — is_explain
    # plus a first-ordinal word, no named file — that the tight _RECALL_RE did
    # NOT resolve structurally. Defers to the guarded model-decider (recall vs
    # explain) instead of the guessing explainer seat. Defaulted so existing
    # bundles are unchanged.
    defer_recall: bool = False


_Guard = Callable[[SignalBundle], bool]


@dataclass(frozen=True)
class Step:
    """One ``_route``/``_fix_chain_route`` branch, transposed to data: the
    ``(target, kind, build)`` it returns, plus the guard predicate that
    decides whether it fires. ``kind=None`` means the kind is resolved
    dynamically from the bundle's ``kind_hint`` (row 11, code-seat) rather
    than a literal string — ``_route`` returns ``kind_hint`` itself there,
    not a constant."""

    chain_label: str
    target: str
    kind: str | None
    build: bool
    guard: _Guard


@dataclass(frozen=True)
class Chain:
    """A labeled, ordered group of steps sharing a lifecycle. No
    ``max_rounds`` yet — the round-budget backstop is a later, additive
    task (docs/plans/2026-07-12-chain-executor-design.md, commit 2)."""

    label: str
    steps: tuple[Step, ...]


@dataclass(frozen=True)
class Decision:
    """(target, kind, build, needs_decider) — ``_route``'s old return
    tuple — plus which step produced it, for diagnosability."""

    target: str
    kind: str
    build: bool
    needs_decider: bool
    chain: str
    step_index: int


def _fix_cont_need_run(bundle: SignalBundle) -> bool:
    """fix-execution run leg (rung 1) or rung 2's re-fix run leg: a write
    this turn — the fix's own or the re-fix's — has no run of its own yet.
    ``wrote_path``/``write_count`` are structural, from the caller's
    post-boundary tool_calls; delegate ONE closed-template run."""
    return bundle.fix_chain and bundle.needs_another_run


def _fix_cont_re_fix(bundle: SignalBundle) -> bool:
    """rung 2, convergent-fix design: a red, localized verdict on a
    fix-led turn routes to the bounded one-round re-fix instead of today's
    honest-red terminal."""
    return (
        bundle.fix_chain
        and not bundle.has_refixed
        and bundle.failure_shape == "localized"
    )


def _fix_cont_run_verdict(bundle: SignalBundle) -> bool:
    """every write this turn has a matching run — the LATEST verdict is
    terminal (green, structural-red, or already-refixed)."""
    return bundle.fix_chain


def _run_run_verdict(bundle: SignalBundle) -> bool:
    """issue #83 run half: the client ran the command — the deliverable is
    the deterministic verdict, one run round per turn."""
    return bundle.run_signal and bundle.has_run_block


def _run_need_run(bundle: SignalBundle) -> bool:
    """issue #83 run half: delegate one closed-template test run."""
    return bundle.run_signal


def _explain_not_grounded(bundle: SignalBundle) -> bool:
    """grounded-explain design: a real named-file target (never the
    "solution.py" default) with no visible build or read on the wire gets
    the deterministic honest refusal — the explainer seat is never called,
    so there is no speculation path."""
    return bundle.is_explain and bundle.explain_ungrounded


def _explain_recall_answer(bundle: SignalBundle) -> bool:
    """#82 deep recall: an ordinal-recall turn resolved to a deterministic
    honest message (nothing shipped / never asked / built-deep). Ahead of
    the explainer seat so recall never guesses; the shipped-and-visible
    case does NOT set this flag and rides the grounded explainer instead."""
    return bundle.is_recall_answer


def _explain_defer_recall(bundle: SignalBundle) -> bool:
    """#82 deep recall (detection layer 2): a loose maybe_recall turn the tight
    _RECALL_RE did not resolve structurally. Outranks the explainer seat so an
    ambiguous ordinal question defers to the guarded model-decider (recall vs
    explain) rather than being answered by speculation. Its empty target makes
    advance() emit needs_decider, exactly like the terminal fallthrough."""
    return bundle.defer_recall


def _explain_explainer(bundle: SignalBundle) -> bool:
    """a grounded (or conceptual, file-less) explain turn dispatches to the
    explainer seat."""
    return bundle.is_explain


def _build_need_glob(bundle: SignalBundle) -> bool:
    """issue #83 discovery: one glob round (or its honest refusal) before
    the read seam. Exclusive with needs_files/read_failed by construction —
    a discovering turn names no source file, a reading turn does — so the
    order here only mirrors the seam chain (discover -> read -> build)."""
    return bool(bundle.needs_glob or bundle.glob_failed)


def _build_need_files(bundle: SignalBundle) -> bool:
    """issue #83: request the client files (or refuse a failed request)
    before any seat runs — the need-files shape is a cheap script echo."""
    return bool(bundle.needs_files or bundle.read_failed)


def _build_tests_seat(bundle: SignalBundle) -> bool:
    """the deliverable IS a test file, run against the workspace alone
    (issue #98) — never build-gated's code/tests duality."""
    return bundle.tests_primary


def _build_code_seat(bundle: SignalBundle) -> bool:
    """the default code-generation seat; only reached once every discovery,
    read, and tests-primary guard above has passed."""
    return bundle.has_build_signal


def _decider_fallthrough(_bundle: SignalBundle) -> bool:
    """No structural signal — hand the routing to the guarded model
    decider. Always true; the terminal row of the table."""
    return True


CHAIN_FIX_CONT = Chain(
    label="fix-cont",
    steps=(
        Step(
            chain_label="fix-cont",
            target="need-run",
            kind="need_run",
            build=False,
            guard=_fix_cont_need_run,
        ),
        Step(
            chain_label="fix-cont",
            target="re-fix",
            kind="re_fix",
            build=True,
            guard=_fix_cont_re_fix,
        ),
        Step(
            chain_label="fix-cont",
            target="run-verdict",
            kind="run_verdict",
            build=False,
            guard=_fix_cont_run_verdict,
        ),
    ),
)

CHAIN_RUN = Chain(
    label="run",
    steps=(
        Step(
            chain_label="run",
            target="run-verdict",
            kind="run_verdict",
            build=False,
            guard=_run_run_verdict,
        ),
        Step(
            chain_label="run",
            target="need-run",
            kind="need_run",
            build=False,
            guard=_run_need_run,
        ),
    ),
)

CHAIN_EXPLAIN = Chain(
    label="explain",
    steps=(
        Step(
            chain_label="explain",
            target="recall-answer",
            kind="recall",
            build=False,
            guard=_explain_recall_answer,
        ),
        Step(
            chain_label="explain",
            target="not-grounded",
            kind="not_grounded",
            build=False,
            guard=_explain_not_grounded,
        ),
        Step(
            chain_label="explain",
            target="",  # defer to the guarded decider (needs_decider)
            kind="",
            build=False,
            guard=_explain_defer_recall,
        ),
        Step(
            chain_label="explain",
            target=_EXPLAIN_SEAT,
            kind="explanation",
            build=False,
            guard=_explain_explainer,
        ),
    ),
)

CHAIN_BUILD = Chain(
    label="build",
    steps=(
        Step(
            chain_label="build",
            target="need-glob",
            kind="need_glob",
            build=False,
            guard=_build_need_glob,
        ),
        Step(
            chain_label="build",
            target="need-files",
            kind="need_files",
            build=False,
            guard=_build_need_files,
        ),
        Step(
            chain_label="build",
            target=_TESTS_SEAT,
            kind="python_tests",
            build=True,
            guard=_build_tests_seat,
        ),
        Step(
            chain_label="build",
            target=_DEFAULT_CODE_SEAT,
            kind=None,  # dynamic: resolved from bundle.kind_hint
            build=True,
            guard=_build_code_seat,
        ),
    ),
)

CHAIN_DECIDER = Chain(
    label="decider",
    steps=(
        # An empty-target step — advance() derives needs_decider from
        # `not step.target`. The explain chain's defer-recall step (#82) is
        # the only other empty-target step; every non-decider step must carry
        # a non-empty target or it silently emits needs_decider too.
        Step(
            chain_label="decider",
            target="",
            kind="",
            build=False,
            guard=_decider_fallthrough,
        ),
    ),
)

CHAINS: tuple[Chain, ...] = (
    CHAIN_FIX_CONT,
    CHAIN_RUN,
    CHAIN_EXPLAIN,
    CHAIN_BUILD,
    CHAIN_DECIDER,
)


def advance(bundle: SignalBundle) -> Decision:
    """The first-match scan over ``CHAINS``, in priority order — the same
    order ``_route``/``_fix_chain_route`` checked its conditions.
    ``needs_decider`` is derived from an empty target, mirroring ``_route``'s
    own invariant: every branch but the terminal fallthrough returns a
    non-empty target and ``needs_decider=False``; only the fallthrough
    returns both empty and ``True``."""
    for chain in CHAINS:
        for step_index, step in enumerate(chain.steps):
            if not step.guard(bundle):
                continue
            kind = bundle.kind_hint if step.kind is None else step.kind
            return Decision(
                target=step.target,
                kind=kind,
                build=step.build,
                needs_decider=not step.target,
                chain=chain.label,
                step_index=step_index,
            )
    # Unreachable: the decider chain's fallthrough guard is always true.
    raise AssertionError("advance: no step matched — the decider row must always fire")
