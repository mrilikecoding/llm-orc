#!/usr/bin/env python3
"""C1 stop-condition spike for issue #143 / docs/plans/2026-08-13-recall-
recovery-design.md ("Pre-flight 2 verdict" + "Changes bound to the arc").

Decides whether the model-backed file-directed GATE is buildable AT ALL,
before any implementation. The gate's job (per the design doc's Architecture
section): given a user turn, and — only when the deterministic C-prong
selection already narrowed the repo to exactly ONE candidate file — that
candidate's path, decide whether the turn is asking about that specific code
file (YES -> the serve grounds a read of it) or about a general concept / the
assistant's own conversation-session-self (NO -> today's conceptual answer).
The model never chooses a file; it only gates whether the deterministic
unique match is used.

Pre-flight 2 already ran the naive two-answer contract and found it collapses
to a constant at both information levels tested (turn-alone: 12/12
"concept"; with the deterministic candidate shown: 11/12 "file") — no
measured calibration point between always-NO and always-YES. This script
tests three alternative contracts (plus a 14b upsize) against a bar that was
fixed BEFORE any of them ran.

PRE-REGISTERED BAR (fixed before running anything — do not edit after seeing
results): over the labeled question set below, a variant PASSES only if:
  (a) "explain the dispatcher" and "how does the dispatcher work?" -> YES
  (b) "explain the context" and "how does context management work?" -> NO
  (c) ALL meta/self questions -> NO (or the dedicated meta answer, in the
      three-answer variants)
  (d) zero YES on the labeled-conceptual set
  (e) >=60% YES on the labeled file-directed set
The full confusion table is reported for every variant regardless of pass/
fail (design doc doctrine: report the whole table, not just the cells that
flatter the hypothesis).

Read-only against production code and against the real repo tree (git
ls-files, file reads) — no production file is modified. This script plus the
design doc addendum it feeds are the only writes for this spike.

Run from the repo root: `uv run python
docs/plans/2026-08-13-gate-precision-spike.py`
Requires a local Ollama serving qwen3:8b and qwen3:14b on localhost:11434
(both free, both already pulled on the rig).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[2]

# _explain_stems: IMPORTED from the real classify.py (not replicated), same
# discipline as the earlier 2026-08-13-recall-recovery-design-spike.py.
sys.path.insert(0, str(REPO_ROOT / ".llm-orc/scripts/agentic_serving"))
import classify  # noqa: E402  (path insert must precede this import)

_explain_stems = classify._explain_stems

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_8B = "qwen3:8b"
MODEL_14B = "qwen3:14b"

# --- Deterministic candidate computation ------------------------------------
# Mirrors the #143 "Fork resolved: option (b)" Architecture section: stage 1
# is the shipped full-naming rule (components <= stems); only when stage 1
# yields ZERO candidates does stage 2 run, the C-prong
# ((stems - residue) <= components) over the same listing. A unique winner at
# either stage is "the deterministic unique candidate file" the gate is shown
# (per the design doc, the ONLY shape the gate is ever invoked in); zero or
# >=2 qualifying candidates at a stage mean no single file to show — those
# questions are tested turn-alone (no candidate line) below, since that is
# what the deterministic layer would actually hand the gate: nothing, because
# the gate never fires for them in production either.
#
# The candidate LISTING step (approximating the caller's [globbed ...] block)
# is built from git-tracked ONLY (not the earlier recall-recovery spike's
# src+.llm-orc/scripts-only corpus) — the pre-flight's own blocker 4 flagged
# that narrower corpus as "wrong universe (202 src-only stems vs 440+
# whole-workspace)"; this spike uses the whole tracked tree so a question
# like "explain the model" surfaces its real single-component collision
# (benchmarks/agentic_serving/model.py) instead of hiding it.
RESIDUE: frozenset[str] = frozenset({"work", "works", "working"})
# The closed question-machinery family decided in the design doc's "Fork
# resolved: option (b)" section — three words, not just "work" (the earlier
# recall-recovery spike's singleton), pinned both ways: family members
# admissible, "management" and other content words are not.


def _is_workspace_test_file(rel_path: str) -> bool:
    path = Path(rel_path)
    parts = path.parts
    return (
        path.name.startswith("test_")
        or path.name.endswith("_test.py")
        or "tests" in parts
        or "worktrees" in parts
    )


def _iter_workspace_py_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout
    return [
        p for p in out.splitlines() if p.endswith(".py") and not _is_workspace_test_file(p)
    ]


def _components_of(stem: str) -> set[str]:
    return {
        c
        for c in re.split(r"[^a-z0-9]+", stem.lower())
        if len(c) >= 3 and not c.isdigit()
    }


WORKSPACE_FILES: list[str] = _iter_workspace_py_files()
FILE_COMPONENTS: dict[str, set[str]] = {p: _components_of(Path(p).stem) for p in WORKSPACE_FILES}


class Candidate(NamedTuple):
    """``stage``: "stage1" | "stage2" | "none" | "ambiguous". ``path`` is the
    single winning path, or None. ``all_matches`` carries every qualifying
    path when ambiguous (diagnostic only — never shown to the model, since an
    ambiguous stage already refuses-with-candidates deterministically and
    never reaches the gate)."""

    stage: str
    path: str | None
    all_matches: tuple[str, ...]


def deterministic_candidate(question: str) -> Candidate:
    stems = set(_explain_stems(question))
    listing = [(p, c) for p, c in FILE_COMPONENTS.items() if c & stems]
    stage1 = sorted(p for p, c in listing if c and c <= stems)
    if len(stage1) == 1:
        return Candidate("stage1", stage1[0], tuple(stage1))
    if len(stage1) > 1:
        return Candidate("ambiguous", None, tuple(stage1))
    filtered = stems - RESIDUE
    if not filtered:
        return Candidate("none", None, ())
    stage2 = sorted(p for p, c in listing if filtered <= c)
    if len(stage2) == 1:
        return Candidate("stage2", stage2[0], tuple(stage2))
    if len(stage2) > 1:
        return Candidate("ambiguous", None, tuple(stage2))
    return Candidate("none", None, ())


# --- Labeled question set (~30 questions) -----------------------------------
# Three categories: file-positive (expect "file"/"yes"), concept-negative
# (expect "concept"/"no" — general software or repo-concept questions with no
# intended single file), meta-negative (expect "conversation" in three-answer
# variants, "no" in two-answer — questions about the assistant/session/its
# own config, never the repo).


class LabeledQuestion(NamedTuple):
    category: str  # "file-positive" | "concept-negative" | "meta-negative"
    question: str
    expected3: str  # "file" | "concept" | "conversation"
    note: str


# File-directed positives: 9 real, git-ls-files-verified multi-component
# files, each carrying a repo-unique component (git ls-files verification and
# frequency computation done interactively before writing this list; see the
# addendum for the raw table). Phrased both ways per the design doc's own
# generated-question convention ("explain the X" / "how does the X work?").
# Same 9 of the design doc's first-10-alphabetically unique-component list
# used by the earlier recall-recovery spike, plus agent_dispatcher.py (the
# rung's pinned demonstrating case) — kept deliberately the SAME files for
# continuity between the two spikes.
FILE_TARGETS: list[tuple[str, str, str]] = [
    (
        "dispatcher",
        "src/llm_orc/core/execution/phases/agent_dispatcher.py",
        "The rung's demonstrating case (design doc Invariants) — required "
        "bar cell (a).",
    ),
    (
        "adequacy",
        ".llm-orc/scripts/agentic_serving/adequacy_check.py",
        "Repo-unique component of a real multi-component file.",
    ),
    (
        "chain",
        ".llm-orc/scripts/agentic_serving/chain_plan.py",
        "Repo-unique component; 'explain the chain' also appears as a "
        "variant-2 few-shot example (contamination flagged in the addendum).",
    ),
    (
        "form",
        ".llm-orc/scripts/agentic_serving/form_gate.py",
        "Repo-unique component of a real multi-component file.",
    ),
    (
        "round",
        ".llm-orc/scripts/agentic_serving/route_round.py",
        "Repo-unique component of a real multi-component file.",
    ),
    (
        "verdict",
        ".llm-orc/scripts/agentic_serving/run_verdict.py",
        "Repo-unique component of a real multi-component file.",
    ),
    (
        "files",
        ".llm-orc/scripts/agentic_serving/need_files_echo.py",
        "Repo-unique component of a real multi-component file.",
    ),
    (
        "glob",
        ".llm-orc/scripts/agentic_serving/need_glob_echo.py",
        "Repo-unique component of a real multi-component file.",
    ),
    (
        "grounded",
        ".llm-orc/scripts/agentic_serving/not_grounded_echo.py",
        "Repo-unique component of a real multi-component file. ('recall' "
        "was the originally planned 9th example, matching the earlier "
        "recall-recovery spike's picks, but the whole-workspace corpus "
        "used here — see the corpus-scope note above — found it collides "
        "with this very design's own sibling spike script "
        "(docs/plans/2026-08-13-recall-recovery-design-spike.py, whose "
        "filename also stems to 'recall'), making it genuinely ambiguous "
        "rather than repo-unique; swapped to keep the file-positive set "
        "clean of self-referential collisions.)",
    ),
]

QUESTIONS: list[LabeledQuestion] = []
for _component, _path, _note in FILE_TARGETS:
    QUESTIONS.append(LabeledQuestion("file-positive", f"explain the {_component}", "file", _note))
    QUESTIONS.append(
        LabeledQuestion("file-positive", f"how does the {_component} work?", "file", _note)
    )

QUESTIONS += [
    LabeledQuestion(
        "concept-negative",
        "how does context management work?",
        "concept",
        "THE BLOCKER (2026-07 adversarial review). Zero deterministic "
        "candidate at either stage ('management' is never a filename "
        "component) — never reaches the gate in production, tested here "
        "turn-alone as the paired required-bar-cell(b) phrasing.",
    ),
    LabeledQuestion(
        "concept-negative",
        "explain the context",
        "concept",
        "The blocker's bare-symbol phrasing — DOES surface a real unique "
        "stage-2 candidate (project_context.py, the blocker file itself), "
        "so this is the sharp test of cell (b): the gate must say NO even "
        "with the wrong-accept file sitting right in front of it.",
    ),
    LabeledQuestion(
        "concept-negative",
        "explain the errors",
        "concept",
        "Same shape as the blocker: 'errors' is a repo-unique component "
        "(structural_errors.py) coincidentally, but the question is a "
        "general 'what error handling exists' ask, not about that one "
        "module.",
    ),
    LabeledQuestion(
        "concept-negative",
        "how does routing work?",
        "concept",
        "Zero deterministic candidate ('routing' names no file stem "
        "exactly) — general concept question about the serving pipeline.",
    ),
    LabeledQuestion(
        "concept-negative",
        "where is the recall ledger built?",
        "concept",
        "Zero deterministic candidate — the real implementation "
        "(_recall_select in classify.py) shares no stem with this "
        "question; recall_echo.py is a coincidental non-match here because "
        "'ledger'/'built' are not its components either.",
    ),
    LabeledQuestion(
        "concept-negative",
        "why is Python dynamically typed?",
        "concept",
        "General software question, zero deterministic candidate, no "
        "intended file.",
    ),
    LabeledQuestion(
        "concept-negative",
        "what is Big O notation?",
        "concept",
        "General software question, zero deterministic candidate, no "
        "intended file.",
    ),
    LabeledQuestion(
        "concept-negative",
        "what's the difference between a list and a tuple?",
        "concept",
        "General software question, zero deterministic candidate, no "
        "intended file.",
    ),
    LabeledQuestion(
        "concept-negative",
        "what is a decorator?",
        "concept",
        "General software question, zero deterministic candidate, no "
        "intended file.",
    ),
]

QUESTIONS += [
    LabeledQuestion(
        "meta-negative",
        "explain the session",
        "conversation",
        "Pre-flight 2's worst-case class: a real unique stage-2 candidate "
        "(session_start.py) exists, but the honest answer lives in the "
        "#133/#134 ledger, not that file — sourced-but-irrelevant is "
        "functionally dishonest here.",
    ),
    LabeledQuestion(
        "meta-negative",
        "explain the model",
        "conversation",
        "Has a real STAGE-1 (not stage-2) candidate "
        "(benchmarks/agentic_serving/model.py) — meaning this one already "
        "grounds TODAY via the shipped full-naming rule, pre-empting the "
        "gate entirely (single-component files are already open in stage "
        "1, per Pre-flight 2). Tested for completeness; not in the gate's "
        "actual blast radius.",
    ),
    LabeledQuestion(
        "meta-negative",
        "explain the config",
        "conversation",
        "Deterministic layer is AMBIGUOUS here (5 qualifying files) -> "
        "refuse-with-candidates before the gate would ever run. Tested "
        "turn-alone for completeness; not in the gate's actual blast "
        "radius either.",
    ),
    LabeledQuestion(
        "meta-negative",
        "explain the state",
        "conversation",
        "Zero deterministic candidate — tested turn-alone.",
    ),
    LabeledQuestion(
        "meta-negative",
        "what is the current profile?",
        "conversation",
        "Zero deterministic candidate — tested turn-alone.",
    ),
]

CANDIDATES: dict[str, Candidate] = {q.question: deterministic_candidate(q.question) for q in QUESTIONS}

# Verbatim overlap with variant 2's fixed few-shot examples (see
# _GATE_SYSTEM_3ANS_FEWSHOT below) — these questions are trivially "seen" by
# the model and must be reported separately, not folded into a claim of
# generalization (design doc doctrine 5: self-confirming-metric pathology).
FEWSHOT_LEAKED_QUESTIONS: frozenset[str] = frozenset(
    {
        "explain the dispatcher",
        "how does context management work?",
        "explain the session",
        "explain the chain",
    }
)


# --- Ollama call plumbing ----------------------------------------------------


def call_ollama(model: str, system: str, user: str) -> tuple[str, float, str]:
    """(raw content, wall-clock seconds, error string). ``error`` is "" on a
    clean response; any transport/HTTP failure is caught and recorded rather
    than raised, so one bad call never aborts the run."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "think": False,
        "format": "json",
        "options": {"temperature": 0},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=data, headers={"Content-Type": "application/json"}
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        elapsed = time.perf_counter() - start
        return body.get("message", {}).get("content", ""), elapsed, ""
    except Exception as exc:  # noqa: BLE001 -- spike script: record, never raise
        elapsed = time.perf_counter() - start
        return "", elapsed, str(exc)


_JSON_OBJ_RE = re.compile(r"\{.*?\}", re.DOTALL)


def _no_default(valid: frozenset[str]) -> str:
    """The fail-closed answer for a parse failure — mirrors Pre-flight 2's
    stated production contract: "any parse failure / timeout / third answer
    = NOT-file-directed = today's behavior exactly"."""
    if "concept" in valid:
        return "concept"
    return "no"


def parse_answer(raw: str, valid: frozenset[str]) -> tuple[str, bool]:
    """(answer, parse_failed) — NO-default on any parse failure: no JSON
    object found, invalid JSON, missing/non-string "answer", or an "answer"
    outside the closed set all count as a parse failure and resolve to the
    fail-closed non-accept default, counted separately in the tables below."""
    match = _JSON_OBJ_RE.search(raw)
    if not match:
        return _no_default(valid), True
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return _no_default(valid), True
    answer = obj.get("answer") if isinstance(obj, dict) else None
    if not isinstance(answer, str) or answer not in valid:
        return _no_default(valid), True
    return answer, False


# --- Prompt contracts ---------------------------------------------------


_GATE_SYSTEM_3ANS = (
    "You are the file-directed gate in a coding assistant's serving "
    "pipeline. A deterministic search already ran; if it found exactly one "
    "repo file whose name plausibly matches a word in the user's turn, that "
    "file's path is shown below as the candidate. Decide what the turn is "
    "really asking for.\n\n"
    'Answer "file" when the turn is asking to understand that ONE SPECIFIC '
    "code file in the repo, and reading it would answer the question.\n"
    'Answer "concept" when the turn is asking about a general software or '
    "programming concept, or about the repo's design in the abstract — not "
    "about that candidate file specifically, even if a word in the turn "
    "happens to overlap the candidate's name.\n"
    'Answer "conversation" when the turn is asking about the assistant '
    "itself, this chat session, or the assistant's own configuration, "
    "model, or current state — not about the code repository at all.\n\n"
    "Respond with ONLY a JSON object, no other text and no code fences: "
    '{"answer": "file"}, {"answer": "concept"}, or {"answer": "conversation"}'
)

# Variant 2: two-polarity few-shot. Exactly the 4 examples named in the
# spike brief (dispatcher->file, context management->concept,
# session->conversation, chain->file) — deliberately fixed, not tuned after
# seeing results.
_GATE_SYSTEM_3ANS_FEWSHOT = (
    _GATE_SYSTEM_3ANS
    + "\n\nExamples:\n"
    'Turn: "explain the dispatcher"\n'
    "Candidate file: src/llm_orc/core/execution/phases/agent_dispatcher.py\n"
    '{"answer": "file"}\n\n'
    'Turn: "how does context management work?"\n'
    '{"answer": "concept"}\n\n'
    'Turn: "explain the session"\n'
    "Candidate file: src/llm_orc/web/serving/session_start.py\n"
    '{"answer": "conversation"}\n\n'
    'Turn: "explain the chain"\n'
    "Candidate file: .llm-orc/scripts/agentic_serving/chain_plan.py\n"
    '{"answer": "file"}'
)

_RELEVANCE_SYSTEM = (
    "A deterministic search found exactly one repo file whose name matches "
    "a word in the user's question — shown below with its first lines. "
    "Decide: is this candidate file actually what the question is asking "
    "about?\n\n"
    'Answer "yes" only if the question is specifically about this file. '
    'Answer "no" if the question is about a general concept, the repo\'s '
    "design in the abstract, or the assistant/session/its own "
    "configuration, even if the file's name happens to share a word with "
    "the question.\n\n"
    "Respond with ONLY a JSON object, no other text and no code fences: "
    '{"answer": "yes"} or {"answer": "no"}'
)


def user_message_candidate_shown(question: str, candidate: Candidate) -> str:
    if candidate.path:
        return f'Turn: "{question}"\nCandidate file: {candidate.path}'
    return f'Turn: "{question}"'


def read_head(path: str, n: int = 40) -> str:
    lines = (REPO_ROOT / path).read_text().splitlines()
    return "\n".join(lines[:n])


def user_message_relevance(question: str, candidate: Candidate) -> str:
    assert candidate.path is not None
    head = read_head(candidate.path)
    return (
        f'Question: "{question}"\n'
        f"Candidate file: {candidate.path}\n"
        f"--- first 40 lines ---\n{head}"
    )


# --- Runner -------------------------------------------------------------


class Result(NamedTuple):
    variant: str
    model: str
    question: LabeledQuestion
    candidate: Candidate
    raw: str
    answer: str
    parse_failed: bool
    error: str
    elapsed: float


def run_3answer_variant(
    variant_name: str, model: str, system: str, questions: list[LabeledQuestion]
) -> list[Result]:
    valid = frozenset({"file", "concept", "conversation"})
    results = []
    for q in questions:
        cand = CANDIDATES[q.question]
        user = user_message_candidate_shown(q.question, cand)
        raw, elapsed, error = call_ollama(model, system, user)
        answer, parse_failed = parse_answer(raw, valid)
        results.append(Result(variant_name, model, q, cand, raw, answer, parse_failed, error, elapsed))
    return results


def run_relevance_variant(
    variant_name: str, model: str, questions: list[LabeledQuestion]
) -> list[Result]:
    """Only runs on questions with a real deterministic candidate — this
    framing is meaningless without one (there is no file to judge relevance
    against). Questions with no candidate are skipped and reported as N/A,
    not silently passed."""
    valid = frozenset({"yes", "no"})
    results = []
    for q in questions:
        cand = CANDIDATES[q.question]
        if not cand.path:
            continue
        user = user_message_relevance(q.question, cand)
        raw, elapsed, error = call_ollama(model, _RELEVANCE_SYSTEM, user)
        answer, parse_failed = parse_answer(raw, valid)
        results.append(Result(variant_name, model, q, cand, raw, answer, parse_failed, error, elapsed))
    return results


# --- Confusion table + bar check -----------------------------------------


def outcome(result: Result, accept_answer: str) -> str:
    is_accept = result.answer == accept_answer
    if result.question.category == "file-positive":
        return "TP" if is_accept else "FN"
    return "FP" if is_accept else "TN"


def print_confusion_table(variant_name: str, results: list[Result], accept_answer: str) -> None:
    counts = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    parse_failures = 0
    fp_details: list[Result] = []
    for r in results:
        o = outcome(r, accept_answer)
        counts[o] += 1
        if r.parse_failed:
            parse_failures += 1
        if o == "FP":
            fp_details.append(r)

    total = len(results)
    elapsed_vals = [r.elapsed for r in results]
    avg_elapsed = sum(elapsed_vals) / total if total else 0.0

    print(f"\n{'=' * 78}\nVariant: {variant_name}\n{'=' * 78}")
    print(f"  questions evaluated: {total}  (parse failures: {parse_failures})")
    print(f"  wall-clock: avg={avg_elapsed:.2f}s  min={min(elapsed_vals, default=0):.2f}s"
          f"  max={max(elapsed_vals, default=0):.2f}s  total={sum(elapsed_vals):.2f}s")
    print(f"  | TP={counts['TP']:2d} | FP={counts['FP']:2d} | TN={counts['TN']:2d} | FN={counts['FN']:2d} |")

    if fp_details:
        print(f"\n  --- wrong-accept (FP) detail: {len(fp_details)} ---")
        for r in fp_details:
            leaked = " [FEWSHOT-LEAKED]" if r.question.question in FEWSHOT_LEAKED_QUESTIONS else ""
            print(
                f"    Q: {r.question.question!r} [{r.question.category}]{leaked}\n"
                f"       candidate={r.candidate.path} (stage={r.candidate.stage})\n"
                f"       raw={r.raw!r}  parsed_answer={r.answer!r}"
            )

    print("\n  --- per-question verdicts ---")
    for r in results:
        leaked = " [FEWSHOT-LEAKED]" if r.question.question in FEWSHOT_LEAKED_QUESTIONS else ""
        print(
            f"    [{outcome(r, accept_answer)}] {r.question.category:16s} "
            f"{r.question.question!r:50s} -> {r.answer!r}"
            f"{' (parse-fail)' if r.parse_failed else ''}"
            f"  cand={'yes(' + r.candidate.stage + ')' if r.candidate.path else 'none(' + r.candidate.stage + ')'}"
            f"{leaked}"
        )


def check_bar(variant_name: str, results: list[Result], accept_answer: str, meta_answer: str | None) -> None:
    by_q = {r.question.question: r for r in results}

    def ans(question: str) -> str | None:
        r = by_q.get(question)
        return r.answer if r else None

    def tested(question: str) -> bool:
        return question in by_q

    cell_a = all(
        tested(q) and ans(q) == accept_answer
        for q in ("explain the dispatcher", "how does the dispatcher work?")
    )
    cell_a_na = not all(tested(q) for q in ("explain the dispatcher", "how does the dispatcher work?"))

    cell_b_qs = ("explain the context", "how does context management work?")
    cell_b = all(tested(q) and ans(q) != accept_answer for q in cell_b_qs)
    cell_b_na = not all(tested(q) for q in cell_b_qs)

    meta_qs = [q.question for q in QUESTIONS if q.category == "meta-negative"]
    meta_tested = [q for q in meta_qs if tested(q)]
    meta_not_accept = all(ans(q) != accept_answer for q in meta_tested)
    meta_exact = (
        all(ans(q) == meta_answer for q in meta_tested) if meta_answer else meta_not_accept
    )
    cell_c = meta_not_accept  # loose reading: NO is sufficient even in 3-answer variants
    cell_c_exact = meta_exact  # strict reading: lands on the dedicated meta answer
    cell_c_na = len(meta_tested) < len(meta_qs)

    concept_qs = [q.question for q in QUESTIONS if q.category == "concept-negative"]
    concept_tested = [q for q in concept_qs if tested(q)]
    cell_d = all(ans(q) != accept_answer for q in concept_tested)
    cell_d_na = len(concept_tested) < len(concept_qs)

    file_qs = [q.question for q in QUESTIONS if q.category == "file-positive"]
    file_tested = [q for q in file_qs if tested(q)]
    file_hits = sum(1 for q in file_tested if ans(q) == accept_answer)
    cell_e_rate = file_hits / len(file_tested) if file_tested else 0.0
    cell_e = cell_e_rate >= 0.60
    cell_e_na = len(file_tested) < len(file_qs)

    overall_testable = cell_a and cell_b and cell_c and cell_d and cell_e
    coverage_complete = not (cell_a_na or cell_b_na or cell_c_na or cell_d_na or cell_e_na)

    print(f"\n  --- pre-registered bar check: {variant_name} ---")
    print(f"    (a) dispatcher pair -> {accept_answer}: "
          f"{'PASS' if cell_a else 'FAIL'}{' [incomplete coverage]' if cell_a_na else ''}")
    print(f"    (b) blocker pair -> not-{accept_answer}: "
          f"{'PASS' if cell_b else 'FAIL'}{' [incomplete coverage]' if cell_b_na else ''}")
    print(f"    (c) meta/self -> not-{accept_answer}: "
          f"{'PASS' if cell_c else 'FAIL'} ({len(meta_tested)}/{len(meta_qs)} tested)"
          f"{' [incomplete coverage]' if cell_c_na else ''}"
          f"  | exact meta-bucket match: {'PASS' if cell_c_exact else 'FAIL'}")
    print(f"    (d) zero wrong-accept on concept-negatives: "
          f"{'PASS' if cell_d else 'FAIL'} ({len(concept_tested)}/{len(concept_qs)} tested)"
          f"{' [incomplete coverage]' if cell_d_na else ''}")
    print(f"    (e) >=60% correct on file-positives: "
          f"{'PASS' if cell_e else 'FAIL'} ({file_hits}/{len(file_tested)} = {cell_e_rate:.0%})"
          f"{' [incomplete coverage]' if cell_e_na else ''}")
    verdict = "CLEARS THE BAR" if overall_testable else "DOES NOT CLEAR THE BAR"
    if overall_testable and not coverage_complete:
        verdict += " on its testable subset (incomplete question-set coverage — see notes)"
    print(f"    => {verdict}")


def main() -> None:
    print(f"Workspace corpus (git ls-files, non-test .py): {len(WORKSPACE_FILES)} files")
    print(f"Labeled question set: {len(QUESTIONS)} questions "
          f"({sum(1 for q in QUESTIONS if q.category == 'file-positive')} file-positive, "
          f"{sum(1 for q in QUESTIONS if q.category == 'concept-negative')} concept-negative, "
          f"{sum(1 for q in QUESTIONS if q.category == 'meta-negative')} meta-negative)")
    print(f"Residue set: {sorted(RESIDUE)}")
    with_candidate = sum(1 for c in CANDIDATES.values() if c.path)
    print(f"Questions with a real deterministic unique candidate: {with_candidate}/{len(QUESTIONS)}")
    print("\nPer-question candidate computation:")
    for q in QUESTIONS:
        c = CANDIDATES[q.question]
        detail = c.path if c.path else f"({c.stage}" + (f": {list(c.all_matches)})" if c.all_matches else ")")
        print(f"  [{q.category:16s}] {q.question!r:50s} -> {detail}")

    # Variant 1: three-answer contract, candidate shown when available.
    v1 = run_3answer_variant("1 (8b, three-answer)", MODEL_8B, _GATE_SYSTEM_3ANS, QUESTIONS)
    print_confusion_table("1 (8b, three-answer)", v1, "file")
    check_bar("1 (8b, three-answer)", v1, "file", "conversation")

    # Variant 2: three-answer contract + fixed 4-example few-shot.
    v2 = run_3answer_variant(
        "2 (8b, three-answer + few-shot)", MODEL_8B, _GATE_SYSTEM_3ANS_FEWSHOT, QUESTIONS
    )
    print_confusion_table("2 (8b, three-answer + few-shot)", v2, "file")
    check_bar("2 (8b, three-answer + few-shot)", v2, "file", "conversation")

    # Variant 3: candidate-relevance framing (yes/no), candidate-bearing
    # questions only.
    v3 = run_relevance_variant("3 (8b, candidate-relevance)", MODEL_8B, QUESTIONS)
    print_confusion_table("3 (8b, candidate-relevance)", v3, "yes")
    check_bar("3 (8b, candidate-relevance)", v3, "yes", None)
    skipped = [q.question for q in QUESTIONS if not CANDIDATES[q.question].path]
    print(f"\n  variant 3 skipped {len(skipped)} questions with no deterministic "
          f"candidate (framing does not apply): {skipped}")

    # Variant 4: qwen3:14b, the best-performing contract from 1-3 — but
    # variants 1-3 ALL failed the bar at 8b (same two wrong-accepts,
    # "explain the context" / "explain the errors", survive every framing
    # tested), so per the spike brief's rule 5 this runs variant 3's
    # candidate-relevance framing at 14b rather than picking a "winner"
    # among three failures.
    v4 = run_relevance_variant("4 (14b, candidate-relevance)", MODEL_14B, QUESTIONS)
    print_confusion_table("4 (14b, candidate-relevance)", v4, "yes")
    check_bar("4 (14b, candidate-relevance)", v4, "yes", None)


if __name__ == "__main__":
    main()
