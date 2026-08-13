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
mtime-ordered nondeterminism).
