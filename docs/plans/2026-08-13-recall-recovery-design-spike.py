#!/usr/bin/env python3
"""Spike for issue #143 / docs/plans/2026-08-13-recall-recovery-design.md.

Resolves the design doc's fork ("Spike to resolve the fork" section) BEFORE
any production change: evaluates three candidate glob-candidate matching
rules (A: unique-rare-component match, B: question-stems-subset-of-
components, C: B plus a minimal question-machinery residue set) against a
labeled question set built from the real repo tree.

Read-only. Makes NO production code changes — this script and the doc
addendum it feeds are the only writes for this spike.

Run from the repo root: `uv run python
docs/plans/2026-08-13-recall-recovery-design-spike.py`
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# --- _explain_stems: IMPORTED from the real module (not replicated). The
# design doc's own tests already exercise it; importing keeps this spike
# honest about what the production tokenizer actually does, rather than
# risking drift from a hand-copied version. classify.py's sibling imports
# (_helpers, chain_plan) resolve fine once its own directory is on the path.
sys.path.insert(0, str(REPO_ROOT / ".llm-orc/scripts/agentic_serving"))
import classify  # noqa: E402  (path insert must precede this import)

_explain_stems = classify._explain_stems

# --- 1. Repo stem-component frequency table -------------------------------
# Same method as the design doc's own measurement (lines 8-16): non-test
# .py files under src/ and .llm-orc/scripts, components = splitting the
# lowercased filename stem on non-alphanumerics, keeping pieces len >= 3
# that are not pure digits. This mirrors _explain_glob_candidates's own
# component-splitting exactly (classify.py ~line 768-772).
_CORPUS_ROOTS = ("src", ".llm-orc/scripts")


def _is_test_file(path: Path) -> bool:
    name = path.name
    return name.startswith("test_") or name.endswith("_test.py") or "tests" in path.parts


def iter_corpus_files() -> list[Path]:
    files: list[Path] = []
    for root in _CORPUS_ROOTS:
        for path in sorted((REPO_ROOT / root).rglob("*.py")):
            if not _is_test_file(path):
                files.append(path.relative_to(REPO_ROOT))
    return files


def components_of(stem: str) -> set[str]:
    return {
        component
        for component in re.split(r"[^a-z0-9]+", stem.lower())
        if len(component) >= 3 and not component.isdigit()
    }


CORPUS_FILES = iter_corpus_files()
FILE_COMPONENTS: dict[Path, set[str]] = {p: components_of(p.stem) for p in CORPUS_FILES}
FREQUENCY: Counter[str] = Counter()
for _comps in FILE_COMPONENTS.values():
    for _c in _comps:
        FREQUENCY[_c] += 1

# --- Candidate listing (approximating the [globbed ...] step) -------------
# Per the spike brief: all repo non-test .py files whose stem contains any
# question stem as a COMPONENT (not a raw substring) — the same corpus used
# to build the frequency table above, kept as one consistent universe.


def candidates_for(stems: list[str]) -> list[tuple[Path, set[str]]]:
    stem_set = set(stems)
    return [
        (path, comps)
        for path, comps in FILE_COMPONENTS.items()
        if comps & stem_set
    ]


# --- 2. The three candidate rules ------------------------------------------
#
# Rule A — unique-rare-component match: a candidate qualifies if it shares a
# question stem whose repo-wide component frequency is exactly 1.
#
# Rule B — "question-stems <= candidate components", taken LITERALLY as
# written in the design doc's "Two naive rules" section (its own worked
# example: "{dispatcher, work} ⊄ {agent, dispatcher} refuses" puts the
# STEMS on the left and the file's COMPONENTS on the right). This is the
# OPPOSITE subset direction from the already-shipped
# `_explain_glob_candidates` (which checks components <= stems, per its own
# docstring and code: `if components and components <= stem_set`). That
# divergence is deliberate on this spike's part, not an oversight: only the
# stems<=components direction makes Rule C's proposed fix (stripping
# residue words FROM THE STEMS) logically capable of helping — stripping
# extra stems can only make a stems<=components check MORE permissive,
# never less, whereas under the shipped components<=stems direction,
# stripping stems could only ever make the check LESS permissive. See the
# addendum for the flagged uncertainty this produces (possible doc typo)
# and its consequences.
#
# Rule C — Rule B, with a minimal residue set removed from the question's
# stems before the subset check (never from the candidate-listing step
# above, so no candidate is ever hidden by residue-stripping — it only
# loosens the subset test).

RESIDUE: frozenset[str] = frozenset({"work"})
# Deliberately a singleton. The design doc's own demonstrating gap is
# "how does the dispatcher work?" — "work" is the one word that appears in
# EVERY generated "how does the <component> work?" question below and in
# no repo filename stem. Spike 1 (2026-07-14) warns against tuning a
# stopword/residue list to the gate; two of the eight real spike-1
# questions below ALSO fail the stems<=components check for reasons that
# have nothing to do with "work" (extraneous CONTENT verbs like "verify"/
# "build", not question machinery) — those are reported as honest
# wrong-refuses for Rule C too, rather than padding the residue set until
# they pass.


def _decide(qualifying: list[Path]) -> tuple[str, Path | None, list[Path]]:
    """("ground", path, []) | ("refuse", None, qualifying) — >=2 qualifying
    candidates is the existing ambiguous-refuse discipline (design doc
    Invariants), collapsed into "refuse" for the confusion table, with the
    candidate list kept for the per-question detail line."""
    if len(qualifying) == 1:
        return "ground", qualifying[0], []
    return "refuse", None, qualifying


def rule_a(stems: list[str]) -> tuple[str, Path | None, list[Path]]:
    stem_set = set(stems)
    qualifying = [
        path
        for path, comps in candidates_for(stems)
        if any(FREQUENCY[c] == 1 for c in (comps & stem_set))
    ]
    return _decide(qualifying)


def rule_b(stems: list[str]) -> tuple[str, Path | None, list[Path]]:
    stem_set = set(stems)
    qualifying = [path for path, comps in candidates_for(stems) if stem_set <= comps]
    return _decide(qualifying)


def rule_c(stems: list[str]) -> tuple[str, Path | None, list[Path]]:
    filtered = set(stems) - RESIDUE
    qualifying = [path for path, comps in candidates_for(stems) if filtered <= comps]
    return _decide(qualifying)


RULES = {"A (unique-rare-component)": rule_a, "B (stems<=components)": rule_b, "C (B + residue)": rule_c}


# --- 3. Labeled question set -------------------------------------------
# Each entry: (question, expected ("ground", relative Path) | ("refuse",
# None), rationale). Expected labels are this spike's own best-judgment
# ground truth, cross-checked against the classify test suite
# (tests/unit/serving/test_serving_classify.py) wherever a case coincides
# with an existing pinned routing-corpus test.

Question = tuple[str, tuple[str, Path | None], str]

QUESTIONS: list[Question] = [
    (
        "how does context management work?",
        ("refuse", None),
        "THE BLOCKER (2026-07 adversarial review): project_context.py's "
        "components are {project, context} — 'project' is never named, so "
        "no rule may ground here. Regression-pinned by the design doc's "
        "Invariants section.",
    ),
    (
        "explain the dispatcher",
        ("ground", Path("src/llm_orc/core/execution/phases/agent_dispatcher.py")),
        "The rung's demonstrating case: 'dispatcher' is repo-unique "
        "(frequency 1) to agent_dispatcher.py. Pinned by the design doc's "
        "Invariants section.",
    ),
    (
        "how does classify decide routing?",
        ("ground", Path(".llm-orc/scripts/agentic_serving/classify.py")),
        "Spike-1 Q1. classify.py's stem is the single component 'classify' "
        "(repo-unique) and IS the file this question is about — matches "
        "the pinned production test "
        "test_bare_symbol_explain_single_candidate_requests_its_read.",
    ),
    (
        "where is the recall ledger built?",
        ("refuse", None),
        "Spike-1 Q2. The recall ledger is implemented inside classify.py "
        "itself (_recall_select/_ledger_recap), which shares no stem with "
        "this question at all. recall_echo.py only coincidentally shares "
        "'recall' — it is a routing-target echo/stub script, not the "
        "ledger's real implementation. No exact named-after-the-symbol "
        "target exists.",
    ),
    (
        "what does the chain executor do?",
        ("refuse", None),
        "Spike-1 Q3. No file combines both 'chain' and 'executor' — "
        "chain_plan.py only has 'chain', three different *executor*.py "
        "files only have 'executor'. Ambiguous/no exact target.",
    ),
    (
        "how are tool calls emitted to the client?",
        ("refuse", None),
        "Spike-1 Q4. The real emission path (emit_envelope.py) doesn't "
        "even share a stem ('emitted' != 'emit', no lemmatization in "
        "either the real tokenizer or this spike). tool_call_guard.py "
        "only coincidentally shares 'tool' — it's a validation guard, not "
        "the emission path.",
    ),
    (
        "how does the accept gate verify a build?",
        ("ground", Path(".llm-orc/scripts/agentic_serving/accept_gate.py")),
        "Spike-1 Q5. accept_gate.py's components {accept, gate} are both "
        "named; 'verify'/'build' are extraneous behavioral verbs, same "
        "shape as the pinned production test "
        "test_bare_symbol_explain_grounds_a_fully_named_multi_word_file.",
    ),
    (
        "where does grounded explain refuse?",
        ("refuse", None),
        "Spike-1 Q6. 'refuse' is never a filename component anywhere in "
        "the repo (checked against the frequency table below); "
        "not_grounded_echo.py only coincidentally shares 'grounded'. "
        "Treated as a synonym gap outside any component-matching rule's "
        "scope, not a target this spike expects a structural rule to "
        "reach.",
    ),
    (
        "what is the write history selector?",
        ("refuse", None),
        "Spike-1 Q7. The real selection logic (_recall_select) lives "
        "inside classify.py, sharing no stem with this question. "
        "write_file.py only coincidentally shares 'write'; 'selector' != "
        "'select' (no lemmatization) even if it had been the target.",
    ),
    (
        "how does the serve normalize read results?",
        ("refuse", None),
        "Spike-1 Q8. Three different files (read_file.py, "
        "results_processor.py, results_display.py) each share only ONE "
        "stem; none is fully named. No unambiguous single target.",
    ),
]

# --- Generated partial-name questions -------------------------------------
# One per repo-unique component, from the first 10 unique-component
# multi-component files, taken ALPHABETICALLY by repo-relative path
# (deterministic, not cherry-picked). Both phrasings count.

_MULTI_UNIQUE_FILES = sorted(
    (
        path
        for path, comps in FILE_COMPONENTS.items()
        if len(comps) >= 2 and any(FREQUENCY[c] == 1 for c in comps)
    ),
    key=lambda p: str(p).lower(),
)[:10]

for _path in _MULTI_UNIQUE_FILES:
    _comps = FILE_COMPONENTS[_path]
    _unique_comps = sorted(c for c in _comps if FREQUENCY[c] == 1)
    # Deterministic pick when a file has more than one repo-unique
    # component: the alphabetically-first one.
    _component = _unique_comps[0]
    QUESTIONS.append(
        (
            f"how does the {_component} work?",
            ("ground", _path),
            f"Generated partial-name question: '{_component}' is "
            f"repo-unique to {_path}.",
        )
    )
    QUESTIONS.append(
        (
            f"explain the {_component}",
            ("ground", _path),
            f"Generated partial-name question (explain phrasing): "
            f"'{_component}' is repo-unique to {_path}.",
        )
    )


# --- 4/5. Simulate each rule, print confusion tables -----------------------


def classify_outcome(expected: tuple[str, Path | None], actual_kind: str, actual_path: Path | None) -> str:
    exp_kind, exp_path = expected
    if exp_kind == "ground" and actual_kind == "ground" and actual_path == exp_path:
        return "correct-ground"
    if exp_kind == "refuse" and actual_kind == "refuse":
        return "correct-refuse"
    if actual_kind == "ground":
        # Covers both a should-refuse question that wrongly grounds, and a
        # should-ground question that grounds on the WRONG file — both are
        # the dangerous "confidently answers from the wrong material"
        # failure mode this spike exists to catch.
        return "wrong-accept"
    return "wrong-refuse"


def main() -> None:
    print(f"Repo corpus: {len(CORPUS_FILES)} non-test .py files under {_CORPUS_ROOTS}")
    print(f"Labeled question set: {len(QUESTIONS)} questions "
          f"({len(QUESTIONS) - len(_MULTI_UNIQUE_FILES) * 2} fixed + "
          f"{len(_MULTI_UNIQUE_FILES) * 2} generated)")
    print(f"Rule C residue set (full): {sorted(RESIDUE)}")
    print()
    print("First 10 unique-component multi-component files alphabetically, "
          "with the component used to generate their questions:")
    for _path in _MULTI_UNIQUE_FILES:
        _comps = FILE_COMPONENTS[_path]
        _unique_comps = sorted(c for c in _comps if FREQUENCY[c] == 1)
        print(f"  {_path}  components={sorted(_comps)}  used='{_unique_comps[0]}'"
              f"  (all repo-unique components: {_unique_comps})")

    for rule_name, rule_fn in RULES.items():
        counts: Counter[str] = Counter()
        wrong_accepts: list[str] = []
        wrong_refuses: list[str] = []
        for question, expected, rationale in QUESTIONS:
            stems = _explain_stems(question)
            actual_kind, actual_path, ambiguous = rule_fn(stems)
            outcome = classify_outcome(expected, actual_kind, actual_path)
            counts[outcome] += 1
            if outcome == "wrong-accept":
                exp_kind, exp_path = expected
                exp_desc = "REFUSE" if exp_kind == "refuse" else str(exp_path)
                wrong_accepts.append(
                    f"  Q: {question!r}\n"
                    f"     stems={stems}\n"
                    f"     expected={exp_desc}\n"
                    f"     actual=GROUND({actual_path})"
                )
            elif outcome == "wrong-refuse":
                exp_kind, exp_path = expected
                amb = f" (ambiguous candidates: {ambiguous})" if ambiguous else ""
                wrong_refuses.append(
                    f"  Q: {question!r}\n"
                    f"     stems={stems}\n"
                    f"     expected=GROUND({exp_path})\n"
                    f"     actual=REFUSE{amb}"
                )

        print(f"\n{'=' * 70}\nRule {rule_name}\n{'=' * 70}")
        print(f"  correct-ground: {counts['correct-ground']}")
        print(f"  correct-refuse: {counts['correct-refuse']}")
        print(f"  wrong-accept:   {counts['wrong-accept']}")
        print(f"  wrong-refuse:   {counts['wrong-refuse']}")
        if wrong_accepts:
            print(f"\n  --- wrong-accept detail ({len(wrong_accepts)}) ---")
            for entry in wrong_accepts:
                print(entry)
        if wrong_refuses:
            print(f"\n  --- wrong-refuse detail ({len(wrong_refuses)}) ---")
            for entry in wrong_refuses:
                print(entry)


if __name__ == "__main__":
    main()
