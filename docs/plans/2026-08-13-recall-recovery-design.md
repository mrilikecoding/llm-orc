# Recall recovery via distinctive-component matching (#143)

Meta-task rung: reach multi-component files from partial names ("the
dispatcher" → `agent_dispatcher.py`) without regressing the wrong-accept
the current rule exists to prevent. WS-3; extends glob→read
grounded-explain slice 1.

## The gap (measured)

`_explain_glob_candidates` (`.llm-orc/scripts/agentic_serving/classify.py`)
matches only when candidate components ⊆ question stems. 202 non-test
`.py` stems in src + scripts; 120 are multi-component; the subset rule
makes a multi-component file reachable only when EVERY component is
named. 58 of 120 (48%) carry a repo-unique component and are the rung's
recoverable population (issue #143's ~53% was measured on an earlier
tree; same order).

## Two naive rules, both refuted by measurement (2026-08-13)

1. **Unique-rare-component match** (candidate carries a question stem
   with repo stem-frequency 1): REFUTED as sole rule — "context" has
   freq 1 (`project_context.py` is the only context-stem file), so "how
   does context management work?" re-grounds on the exact file the
   2026-07 adversarial review flagged as the wrong-accept blocker.
2. **Question-stems ⊆ candidate components**: kills the blocker
   ({context, management, work} ⊄ {project, context}) but ALSO kills the
   rung's own target phrasings — "how does the dispatcher work?" carries
   "work", which survives `_EXPLAIN_STOPWORDS`, so {dispatcher, work} ⊄
   {agent, dispatcher} refuses.

The fork: what residue is admissible between the question's stem set and
the candidate's component set. Spike 1 (2026-07-14) warns against tuning
a stopword list to the gate; that warning applies to any closed
"question-machinery" residue list (work/works/happen/mean/…), which is
otherwise the cheapest fix to rule 2.

## Spike to resolve the fork (build before deciding)

Evaluate BOTH rules (and rule 2 + a minimal question-machinery residue
set, listed in full in the spike output) against:

- the routing corpus (`tests/` fixtures for classify) — must stay green,
- the 8 spike-1 questions (2026-07-14 record),
- the blocker case and ≥10 partial-name questions generated one per
  repo-unique component from the measurement above (deterministic, not
  cherry-picked: take the first N unique-component files alphabetically).

Deliverable: one confusion table per rule (wrong-accept / wrong-refuse /
correct-ground / correct-refuse). The rule with zero wrong-accepts and
the most correct-grounds wins; a tie prefers the rule with no new word
list (doctrine: structural lever over list tuning).

## Invariants (whatever rule wins)

- The blocker case ("how does context management work?") REFUSES —
  regression test pinned.
- "explain the dispatcher" GROUNDS on `agent_dispatcher.py` when it is
  the unique carrier — the rung's demonstrating case, pinned.
- Ambiguous matches (≥2 qualifying candidates) refuse-with-candidates —
  existing discipline, corpus-pinned.
- Charset/stem-safety discipline unchanged (tokens still from
  `_EXPLAIN_TOKEN_RE`; the glob template contract untouched).
- Routing corpus green; no other rung's routing changes.

## Sequencing

Spike (in-repo script + table in this doc's addendum) → reviewer
pre-flight on the winning rule (one exchange, before implementation) →
TDD on a branch → hermetic green → live battery row (RIG, available) →
author-independent adversarial review with wrong-accept hunt → merge.

## Spike results (2026-08-13)

Script: `docs/plans/2026-08-13-recall-recovery-design-spike.py`. Run with
`uv run python docs/plans/2026-08-13-recall-recovery-design-spike.py` from
the repo root. Read-only — no production code touched; `_explain_stems` was
imported from the real `classify.py`, not replicated. Corpus: 202 non-test
`.py` files under `src/` + `.llm-orc/scripts` (same count as the design
doc's own measurement). Routing corpus (`tests/unit/serving/
test_serving_classify.py`) confirmed green (219 passed) — unaffected, since
this spike changed no production code.

30 labeled questions: the blocker case, the dispatcher demonstrating case,
the 8 spike-1 (2026-07-14) questions with best-judgment labels, and 20
generated partial-name questions (both phrasings) — one pair per file, from
the first 10 unique-component multi-component files alphabetically:
`adequacy_check.py`, `build_gated_envelope.py`, `chain_plan.py`,
`form_gate.py`, `need_files_echo.py`, `need_glob_echo.py`,
`not_grounded_echo.py`, `recall_echo.py`, `route_round.py`,
`run_verdict.py`.

**Interpretive flag on Rule B/C's subset direction.** The design doc's own
"Two naive rules" section states rule 2 as "Question-stems ⊆ candidate
components," and its worked example ("{dispatcher, work} ⊄ {agent,
dispatcher} refuses") puts stems on the left, components on the right. That
is the OPPOSITE direction from the already-shipped
`_explain_glob_candidates` (`components <= stem_set` — components is the
subset). This spike implemented the doc's rule 2 literally (stems ⊆
components), because only that direction makes the doc's own proposed
residue-set fix (rule 3) logically capable of helping: stripping words out
of the stems can only make a stems⊆components check MORE permissive, never
a components⊆stems check. If this was actually meant to describe the
already-shipped direction, rule B below is a different, novel rule, not a
re-test of production behavior — worth confirming before the reviewer
pre-flight.

**Residue set for Rule C (full): `{"work"}`.** A singleton, deliberately.
It is the one word every generated "how does the `<component>` work?"
question shares and no repo filename stem contains. Two of the eight real
spike-1 questions still wrong-refuse under Rule C for reasons unrelated to
"work" (extraneous content verbs "verify"/"build", not question
machinery) — left unpadded rather than added to the residue set, per spike
1's overfit warning.

### Confusion tables (30 questions each)

| Rule | correct-ground | correct-refuse | wrong-accept | wrong-refuse |
|---|---|---|---|---|
| A — unique-rare-component | 22 | 0 | **8** | 0 |
| B — stems ⊆ components | 11 | 7 | 0 | 12 |
| C — B + `{"work"}` residue | 21 | 7 | 0 | 2 |

**Rule A's 8 wrong-accepts** (every spike-1 question except the two with an
exact named-after-the-symbol target grounds on a coincidental freq-1 hit):

- "how does context management work?" → `src/llm_orc/mcp/project_context.py` (the blocker itself)
- "where is the recall ledger built?" → `.llm-orc/scripts/agentic_serving/recall_echo.py`
- "what does the chain executor do?" → `.llm-orc/scripts/agentic_serving/chain_plan.py`
- "how are tool calls emitted to the client?" → `src/llm_orc/core/validation/tool_call_guard.py`
- "how does the accept gate verify a build?" → `.llm-orc/scripts/agentic_serving/build_gated_envelope.py` (wrong file — the correct target, `accept_gate.py`, never even qualifies under rule A, since neither "accept" nor "gate" is repo-unique)
- "where does grounded explain refuse?" → `.llm-orc/scripts/agentic_serving/not_grounded_echo.py`
- "what is the write history selector?" → `src/llm_orc/primitives/file_ops/write_file.py`
- "how does the serve normalize read results?" → `src/llm_orc/primitives/file_ops/read_file.py`

Rule B and Rule C have zero wrong-accepts on this question set.

### Recommendation

Rule A is disqualified outright — 8 wrong-accepts on 30 questions, not just
the one known blocker, confirms the design doc's refutation and shows the
failure mode is systemic (any question sharing one coincidentally-rare word
with an unrelated file grounds on it), not a single edge case a patch could
close. Between B and C, both clear the zero-wrong-accept bar, so the
decision comes down to correct-grounds: C reaches 21 against B's 11 — B
wrong-refuses every generated "how does X work?" phrasing (10 of 10) plus
2 of the 8 real spike-1 questions, because a single trailing verb it
doesn't recognize as machinery defeats an otherwise-exact match. C's
one-word residue set recovers all 10 generated wrong-refuses at zero
wrong-accept cost, and is not required to close the remaining 2 (which are
extraneous content words, not machinery, and stay honestly refused rather
than chasing the gate with a bigger list). C wins on the design doc's own
criterion (zero wrong-accepts first, most correct-grounds second) without
needing the no-new-word-list tiebreak — the tiebreak never triggers because
this isn't a tie. Recommend Rule C, carrying the interpretive flag above
into the reviewer pre-flight before implementation.

### Lead resolution of the direction flag (2026-08-13)

The doc's rule 2 direction was intentional: the shipped
`components ⊆ stems` check is what CREATES the partial-naming gap, so
rule C is a new prong, not a re-test. Production shape for
implementation: the UNION — a candidate qualifies when EITHER the
shipped fully-named check passes (preserves every currently-matching
case by construction, including questions that name extra concepts
beyond one file's components) OR rule C passes
(`(stems − {"work"}) ⊆ components`, the partial-naming recovery). The
ambiguity discipline (≥2 qualifying candidates → refuse-with-candidates)
applies AFTER the union. Blocker check under the union: the shipped
prong fails ("project" unnamed) and the C prong fails ("management"
unexplained) — dead by both prongs. Pre-flight question to the
reviewer: union-vs-replacement, the residue singleton, and any
wrong-accept the union enables that neither prong enables alone.

## Reviewer pre-flight verdict (2026-08-13): REDESIGN (scoped)

The pre-flight (adversarial, end-to-end against a patched classify copy
in scratch; no repo files touched) upheld the union SHAPE (proof: with
one listing, union of qualifiers, ambiguity after, no prong priority,
union wrong-accepts ⊆ per-prong wrong-accepts) and REFUTED the rest:

1. **Class-level unsatisfiability.** Invariant 1 (blocker refuses) and
   invariant 2 (dispatcher grounds) cannot both hold at class level
   under ANY component-subset rule: `agent_dispatcher.py` and
   `project_context.py` are the same shape (two components, one
   repo-unique named, one generic unnamed). One-word probes re-open the
   blocker class end-to-end: "explain the context" →
   `project_context.py`, "explain the errors" → `structural_errors.py`.
   The pinned July regression tests survive as strings, not as a class.
2. **Residue collapse.** `_explain_stems("how does this work?") =
   ['work']`; minus the residue the C prong tests `∅ ⊆ components`,
   which everything satisfies — verified grounding `network_client.py`
   via the `*work*` substring glob. Any residue rule needs a non-empty
   guard, plus a deliberate decision on works/working inflections.
3. **Union loses 16 currently-working grounds** (second candidate
   qualifies via C, ambiguity-refuse after union): `explain emit`,
   `explain the cli`, `explain shape`, … The doc's "preserves every
   currently-matching case by construction" claim is RETRACTED —
   candidates are preserved, outcomes are not.
4. **The spike's evidence does not transfer**: wrong universe (202
   src-only stems vs 440+ whole-workspace), wrong listing step
   (component-intersection vs basename-substring glob), no
   `_GLOB_MAX_PATHS=50` truncation modeling (mtime-ordered, so
   truncated listings are nondeterministic), two-bucket refuse
   conflating ambiguity-refuse with conceptual fall-through, and a
   question set generated from the rule's own success condition (its
   entire 10-point win). Doctrine 5's self-confirming-metric pathology,
   caught before code.

**Fork for the practitioner (re-scopes the rung, so recorded rather
than adjudicated autonomously):**

- **(a) Phrasing-specific invariant 1**: ship the union with guards
  (non-empty residue-filtered stems, truncated-listing refuse, the 16
  flips pinned as accepted losses or mitigated), accepting that
  one-word conceptual questions ("explain the context") ground on
  coincidental unique-component files with the source disclosed. This
  overturns a review-established wrong-accept class phrasing-by-
  phrasing.
- **(b) Class-level invariant 1 holds** (lead recommendation —
  consistent with fail-closed honesty doctrine and the four-times-
  proven wrong-accept review culture): the C prong cannot ship;
  #143 re-scopes to the #82 two-layer pattern — a bounded, closed-set
  model gate ("file-directed ask?") in front of the DETERMINISTIC
  candidate discipline, where a wrong gate decision degrades to
  today's conceptual answer or a refuse-with-candidates, never a
  silent wrong-file ground.

Standalone hardening extracted regardless of the fork: refuse to
ground on a `(truncated)` glob listing (deterministic, cheap; the
pre-flight measured real stem families busting the 50-cap with
mtime-ordered nondeterminism). [Shipped: #148, merged 5a59515,
live-validated run 6.]

## Fork resolved: option (b) (practitioner, 2026-08-13)

The blocker invariant holds at CLASS level. No deterministic rule change
may open the class; the rung re-scopes to the #82 two-layer split, using
the decider plumbing classify already has (`needs_decider` deferral,
fail-closed fallback — the #133/#134 machinery).

### Architecture

**Deterministic SELECTION (unchanged + staged):** stage 1 is the shipped
full-naming rule (`components ⊆ stems`), byte-identical outcomes — no
16-flip regression by construction. Only when stage 1 yields ZERO
candidates AND the model gate says file-directed does stage 2 run: the
C-prong (`(stems − residue) ⊆ components`) over the same complete
listing, with the pre-flight guards — non-empty residue-filtered stem
set required, truncated listings already refuse (#148), unique candidate
grounds, ≥2 refuse-with-candidates, zero falls through to the
conceptual explainer.

**Model DETECTION (bounded, closed-set, fail-closed):** a decider
question "is this asking about a specific code file, or about a general
concept?" — closed two-answer set, structured output, any parse
failure / timeout / third answer = NOT-file-directed = today's behavior
exactly. The gate runs ONLY on the stage-1-zero + stage-2-unique shape,
so its blast radius is: false-NO = today's conceptual answer (honest
miss); false-YES = a grounded explain of the ONE real file uniquely
carrying the named component, transparently sourced on the wire — the
#82 irrelevant-but-true class, backstopped by deterministic selection.
The model never chooses a file; it only gates whether the deterministic
unique match is used.

**Residue set, decided deliberately (pre-flight ask):** the closed
question-machinery family `{"work", "works", "working"}` — three words,
pinned both ways in the corpus (family members admissible; "management"
and content verbs are not). Spike-1's overfit warning is honored by the
corpus pinning adversarial phrasings, not by pretending the set can be
empty.

### Sequencing

Reviewer pre-flight on this architecture (one exchange: the
stage-1-first fallback vs the round-2 union proof, gate contract, gate
placement in serving.yaml) → TDD on a branch → gate-precision table
(the 30 spike questions through the real decider seat, RIG) → live
partial-name validation (real OpenCode: "explain the dispatcher"-shaped
ask in this repo) → adversarial review with wrong-accept hunt → merge.

## Pre-flight 2 verdict (2026-08-13): PROCEED-WITH-CHANGES, C1 a stop-condition

The reviewer probed the REAL gate (qwen3:8b, think off, the cheap-tier
profile) before any code existed — 24 free local calls:

- **The two-answer gate collapses to a constant at both information
  levels.** Turn-alone: 12/12 "concept" (including the rung's own
  demonstrating case — the rung would deliver zero grounds). With the
  deterministic candidate shown: 11/12 "file" (including the blocker's
  own file). No measured calibration point exists between always-NO and
  always-YES.
- **The staged fallback is sound** (byte-identical to today whenever
  stage 1 yields a candidate; the round-2 union proof is not an
  objection since today, not the union, is the baseline). Deterministic
  stops absorb a third of the exposed population (49 stage-2 ambiguity,
  7 stage-1 ambiguity, 3 truncation).
- **The invariant as stated does not describe shipped code**: the
  one-word-question class is ALREADY OPEN for single-component files —
  49 of 203 repo-vocabulary probes ground TODAY via stage 1 ("explain
  the cli" → cli.py, "explain the model" → benchmarks model.py). The
  honest invariant: multi-component files stay closed behind the gate;
  single-component files are already open in stage 1.
- **The meta/self collision class is the worst case**: "explain the
  session" → session_start.py while the #133/#134 ledger holds the real
  answer — sourced-but-irrelevant is functionally dishonest where an
  existing honest path was pre-empted (session, model, config, profile,
  client, state, registry, catalog). A two-answer set has no cell for
  it; the existing decider carries five answers for exactly this reason.
- **Plumbing**: `needs_decider` cannot gate a read (emit reads
  needs_files before seat output; `_decider_target`'s parse default is
  a hard dispatch failure, the inverse of fail-closed-to-today). The
  right precedent is `recall_answer` promote-or-drop: a new classify
  field, a gate node with its own `when:`, NO-default parse. The gate
  must fire once, on the read-issuing pass (idempotency across passes).

**Changes bound to the arc**: (C1, stop-condition) gate-precision spike
FIRST with a pre-registered bar — variants: 14b seat, a third
"about-this-session/meta" answer, two-polarity few-shot, and
candidate-relevance framing ("is this file what the question is
about?") — required cells: "explain the dispatcher"=YES, "explain the
context"=NO, meta/self class=NO. If no variant clears the bar, the rung
CLOSES as an honest miss (fail-closed applied to the process). (C2)
gate wire shape per the recall_answer precedent. (C3) idempotent
across passes. (C4) meta/self class handled (third answer or
deterministic exclusion). (C5) the 49 recorded — done above.

## Gate-precision spike results (2026-08-13)

Script: `docs/plans/2026-08-13-gate-precision-spike.py`. Run with `uv run
python docs/plans/2026-08-13-gate-precision-spike.py` from the repo root.
Read-only — no production code touched. 108 real Ollama calls total
(qwen3:8b x86 questions, qwen3:14b x22), think off, temperature 0, JSON
format forced, zero transport/parse failures across every call.

**Pre-registered bar (fixed before running anything)**: a variant PASSES
only if (a) the dispatcher pair → YES; (b) the context/context-management
pair → NO; (c) all meta/self questions → NO (or the dedicated meta answer
in three-answer variants); (d) zero YES on the labeled-conceptual set; (e)
≥60% YES on the labeled file-directed set.

**Labeled question set (32 questions)**: 18 file-positive (9 real,
git-ls-files-verified repo-unique-component files — dispatcher, adequacy,
chain, form, round, verdict, files, glob, grounded — each phrased both
"explain the X" and "how does the X work?"), 9 concept-negative (the
blocker pair, "explain the errors", "how does routing work?", "where is
the recall ledger built?", and 4 general software questions with no
intended file), 5 meta-negative (session, model, config, state, current
profile).

**Corpus-scope correction found while building the set**: the deterministic
candidate for each question was computed with the same staged rule the
design's Architecture section specifies — stage 1 `components ⊆ stems`
(shipped), stage 2 `(stems − {work, works, working}) ⊆ components` (new
prong) — but over the WHOLE git-tracked tree (224 non-test `.py` files),
not the earlier recall-recovery spike's src+`.llm-orc/scripts`-only 202,
per Pre-flight 2's own blocker 4 ("wrong universe"). This caught a real
self-referential collision before it shipped: "recall" was the planned 9th
file-positive example (`recall_echo.py`), but under the whole-workspace
corpus it collides with this design's own sibling spike script
(`docs/plans/2026-08-13-recall-recovery-design-spike.py`, whose filename
also stems to "recall"), making it genuinely ambiguous rather than
repo-unique. Swapped for `not_grounded_echo.py` ("grounded") — a clean,
verified-unique replacement. Reported here rather than silently fixed,
since it is itself a small demonstration of the whole-workspace-corpus
argument: naming collisions are more common than the narrower measurement
suggested.

Only 22/32 questions carry a real deterministic unique candidate (the ONLY
shape the gate is ever invoked in per the Architecture section); the other
10 (`how does context management work?`, `how does routing work?`, `where
is the recall ledger built?`, the 4 general questions, `explain the
config` [ambiguous, 5 files], `explain the state`, `what is the current
profile?`) never reach the gate in production at all — the deterministic
layer already resolves them (empty or ambiguous) before any model call
would happen. They are tested turn-alone (no candidate shown) for
completeness since the spike's contracts all take an optional candidate,
but a wrong-accept on them is structurally impossible in production and is
reported separately from the 22 that matter.

### Confusion tables (accept-answer: "file" for variants 1-2, "yes" for 3-4)

| Variant | questions | TP | FP | TN | FN | parse-fail | (a) | (b) | (c) loose | (d) | (e) | BAR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 — 8b, three-answer (file/concept/conversation) | 32 | 14 | **2** | 12 | 4 | 0 | PASS | FAIL | PASS | FAIL | PASS (78%) | **FAIL** |
| 2 — 8b, three-answer + 4-example few-shot | 32 | 14 | **3** | 11 | 4 | 0 | PASS | FAIL | FAIL | FAIL | PASS (78%) | **FAIL** |
| 3 — 8b, candidate-relevance (yes/no) | 22† | 10 | **2** | 2 | 8 | 0 | PASS | FAIL‡ | PASS‡ | FAIL‡ | FAIL (56%) | **FAIL** |
| 4 — 14b, candidate-relevance (yes/no) | 22† | 11 | **2** | 2 | 7 | 0 | PASS | FAIL‡ | PASS‡ | FAIL‡ | PASS (61%) | **FAIL** |

† Variants 3-4 only cover the 22 candidate-bearing questions — the
relevance framing has nothing to evaluate on the other 10, so they are
skipped, not silently passed. ‡ Cells (b)/(d) are evaluated only on the
half of their question set that variants 3-4 can test (`explain the
context` / `explain the errors`, both candidate-bearing) — `how does
context management work?` has no candidate and is untestable under this
framing; both testable halves still FAIL.

**The single finding that decides this spike**: `"explain the context"`
and `"explain the errors"` wrong-accept (answer "file"/"yes") in **every
one of the four variants**, at both model sizes, under both the
directedness framing (1/2/4... variants 1-2) and the relevance framing
(3-4). Swapping contracts, adding few-shot examples, or moving from 8b to
14b changes nothing about this pair. This is the strongest possible
negative result for C1: the wrong-accept is not a capacity or
prompt-engineering problem, it reproduces identically across every axis
tested.

### Wrong-accept (FP) detail — every occurrence across all 4 variants

- **Variant 1** (2 FP): `explain the context` → `file` (candidate
  `src/llm_orc/mcp/project_context.py`, the blocker file itself);
  `explain the errors` → `file` (candidate
  `src/llm_orc/models/structural_errors.py`).
- **Variant 2** (3 FP): the same 2 as variant 1, **plus** `explain the
  model` (meta-negative) → `file` (candidate
  `benchmarks/agentic_serving/model.py`, a stage-1 pre-existing-grounding
  case, not even in the gate's normal blast radius — the few-shot examples
  made this ONE case worse, not better, versus variant 1's `concept`
  answer on the same question).
- **Variant 3** (2 FP): `explain the context` → `yes`; `explain the
  errors` → `yes` — same two files, now under a framing that explicitly
  asks "is this file what the question is about" rather than "is this
  file-directed." The model still says yes: the file genuinely IS
  topically relevant to the question, which is exactly the wrong-accept
  mechanism the blocker exists to name (sourced-but-not-what-was-asked).
- **Variant 4** (2 FP): identical to variant 3, same two questions, same
  two files, at 14b.

No other question in the 32-question set ever produced a wrong-accept, at
any variant. The failure is confined entirely to the two questions shaped
exactly like the original blocker: a general-English word that happens to
be a repo-unique filename component of an unrelated file.

### Secondary finding: false-negative rate on genuine file-positives

Variants 3-4 (candidate-relevance framing) also badly under-ground real
file-directed questions: `explain the form`, `explain the files`, `explain
the glob`, `explain the grounded` and their "how does X work?" pairings
answer "no" even though the candidate file is exactly what was asked
about (56%/61% hit rate, barely at/under the 60% floor) — the relevance
framing is worse on both axes at once, not a safer trade against variant
1/2's wrong-accepts. Variant 2's few-shot examples also produced one
FN-adjacent regression not shown above: no new FN, but the meta wrong-
accept on "explain the model" shows few-shot can move mass in either
direction unpredictably on a 4-example prompt.

**Methodological flag on variant 2**: its four few-shot examples
(`explain the dispatcher`, `how does context management work?`, `explain
the session`, `explain the chain`) are verbatim members of the labeled
question set — those 4 results are trivially "seen," not evidence of
generalization, and are marked `[FEWSHOT-LEAKED]` in the script's raw
output. None of the 4 leaked questions happens to be a wrong-accept case,
so the leak doesn't inflate variant 2's FP count, but it does inflate its
TP/TN counts slightly; the bar-relevant failures (the context/errors pair)
are entirely outside the leaked set regardless.

### Wall-clock

| Variant | model | calls | avg | min | max | total |
|---|---|---|---|---|---|---|
| 1 | qwen3:8b | 32 | 0.49s | 0.43s | 0.57s | 15.8s |
| 2 | qwen3:8b | 32 | 0.50s | 0.43s | 0.83s | 16.1s |
| 3 | qwen3:8b | 22 | 0.92s | 0.34s | 1.50s | 20.2s |
| 4 | qwen3:14b | 22 | 2.72s | 1.58s | 7.73s | 59.8s |

Full run (all 4 variants, 108 calls): ~112s wall-clock.

### Verdict

**None of the four variants clears the pre-registered bar — the rung
closes as an honest miss per C1.** Every variant passes cell (a) (the
dispatcher case grounds cleanly everywhere) and most pass cell (e), so the
gate is not simply non-functional — but every variant fails cell (b)/(d)
on the exact same pair of questions, `explain the context` and `explain
the errors`, which are structurally identical to the pinned blocker (a
generic English word that coincidentally is a repo-unique filename
component of an unrelated file). Neither a three-answer contract, few-shot
examples, a relevance-scoring reframe, nor a 2x larger model (8b → 14b)
moves this pair off "wrong-accept" even once across 4 attempts and 8 total
calls on those 2 questions. This confirms the reviewer pre-flight's class-
level unsatisfiability finding from first principles: the model cannot
reliably distinguish "this file is topically relevant to a generic
question" from "this question is specifically about this file" when both
share a name, which is precisely the ambiguity the C-prong's own candidate
selection creates by design. Per the design doc's Fork resolution (option
b) and the class-level invariant it re-affirmed, #143 does not get a
model-gated C-prong: the multi-component partial-naming gap stays closed,
and the rung closes here rather than shipping a gate that reopens the
class the 2026-07 adversarial review closed.
