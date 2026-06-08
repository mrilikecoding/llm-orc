# Spike ρ — Remaining-Work Anchor (Cycle 7 loop-back #6, Finding G)

**Status:** COMPLETE (run 2026-06-08). Pre-registered + methods-reviewed
(2026-06-07, all 2 P1 / 3 P2 / 2 P3 applied — see
`housekeeping/audits/research-methods-spike-rho.md`); revisions review-directed
(the control arm is P1-B's recommendation; the statement-vs-imperative split
addresses the named incongruity), recorded as the visible-flag disposition.
**Verdict: PASS — route-the-signal-forward grounded; remaining-work content
causally isolated (control 0/10 vs anchored 8–10/10). Amendment form: judge's
statement + framework imperative. Grounds ADR-038.**
**Trigger:** Finding G (cycle-status §"Finding G" + `scratch/spike-multifile-progress/RESULTS.md`):
the WP-LB-K real-OpenCode acceptance run showed multi-file sessions do not
converge — the seat-filler's call-2 next-action selection never advances to the
second deliverable. Root cause: ADR-037's FC-66 discards the judgment exchange
to keep call 2 byte-equal to the measured E4b composition, so the judge's own
"what remains" statement is thrown away and the action call has no
remaining-deliverable anchor. The informal rung-1 probe (advance 0/10 → 8/10
with a *hardcoded* correct-filename anchor) validated the fix direction. This
spike grounds the ADR-037 amendment (DECIDE loop-back #6).

**Class:** DECIDE-driving evaluation (the θ/ω class, not the χ/φ bounded
BUILD-gate class) — research-methods review of this design is dispatched before
any run.

---

## What the rung-1 probe left open (the reason this spike exists)

The rung-1 probe (`scratch/spike-multifile-progress/`) anchored call 2 with a
**hardcoded** correct filename (`test_string_utils.py`). It measured only the
second of two composing factors:

- **Factor 1 (unmeasured):** P(the judge's REMAINING statement correctly names
  the unproduced deliverable). The production anchor is the judge's *own*
  one-sentence "what remains" output, not a hardcoded string.
- **Factor 2 (measured at 8/10):** P(anchored call 2 advances to the named
  deliverable | a correct anchor).

The production end-to-end rate composes Factor 1 × Factor 2. A judge that
mis-names what remains would poison the anchor — the amendment cannot be
credited on Factor 2 alone. The rung-1 probe also observed a **2/10
no-tool-call rate** on the anchored call 2 (a premature-finish risk); this
spike characterizes it at full n.

---

## Hypotheses

- **H-ρ.1:** The judge reliably names the unproduced deliverable in its
  REMAINING statement (the production anchor is trustworthy).
- **H-ρ.2:** Call 2 anchored with the judge's *actual* REMAINING statement
  advances to the named deliverable at a rate comparable to the rung-1
  hardcoded-anchor result (8/10), while preserving delegation.
- **H-ρ.3 (null-guard):** The anchor does not collapse delegation into inline
  writes (the Finding B shape) and does not raise the no-tool-call rate beyond a
  tolerable bound.

## Arms (qwen3:14b via Ollama /v1, $0 local; composition via the real landed
code path — `compose_judgment_message`, `_seat_filler_messages`,
`_delegation_tools` — not hand-built)

Two bases (the θ multi-base discipline — one base could hide a base-specific
effect):

- **Base B2:** 2-deliverable task (`string_utils.py` + `test_string_utils.py`),
  file 1 written (one trailing tool pair). The rung-1 base.
- **Base B3:** θ's E4′ task text verbatim (P3-A) — "Write a python module
  string_utils.py with a function that reverses the word order of a string, a
  number_utils.py with a function that formats integers with thousands
  separators, and a test_string_utils.py with unit tests for the string
  module." Two files written (`string_utils.py`, `number_utils.py` — two
  trailing tool pairs); the unproduced deliverable is `test_string_utils.py`.
  The deeper-tail robustness base.

Per base, n=10:

- **ρ.1 — judge remaining-naming.** Run the real judgment call
  (`compose_judgment_message(task, records)` + the judge seat). Record: the
  parsed verdict; and, when REMAINING, the one-sentence statement. Classify the
  statement against the **pre-registered three-level sufficiency standard**
  (P1-A), adjudicated from the retained response text against the known base
  state:
  - `specific-correct` — names the unproduced deliverable by filename
    (`test_string_utils.py`) or an identity-equivalent token.
  - `description-correct` — refers to the unproduced deliverable by unambiguous
    description ("the test module for string_utils", "unit tests for the
    reverse function") without the filename, and does NOT assert any
    already-produced file is missing.
  - `ambiguous` — references "remaining work" / "the tests" with no anchor to
    *which* deliverable, such that a seat-filler could not identify the target.
  - `names-wrong` — asserts an already-produced file is missing, or names a
    deliverable not in the task.
  - `verdict-COMPLETE` — a false COMPLETE on a work-remaining base (the θ
    false-stop; a separate failure).

  `names-correct` for the pass rule = `specific-correct ∪ description-correct`.
  The standard is fixed before running so adjudication is refutable from the
  retained text, not disposition-dependent (P1-A). Denominator n.

- **ρ.2 — end-to-end composed advance (statement-only, the minimal production
  form).** For each ρ.1 trial whose verdict is REMAINING, anchor call 2 with the
  judge's *actual* statement and nothing more — the production composition if the
  amendment routes the judge's sentence forward verbatim: trailing C3 guidance +
  the judge's remaining-work sentence, no framework-added imperative. Run call 2
  (`_seat_filler_messages` + `_delegation_tools` → the seat). Record target
  (`advance` / `stuck` / `other` / `none`) and `delegated`. Denominator n; a
  verdict-COMPLETE / no-statement / ambiguous trial contributes `none` to ρ.2
  (no usable anchor was produced). **Track and report the runnable-trial count**
  (trials that produced a usable REMAINING anchor) alongside the n=10 rate — the
  conditional-denominator interaction (P2-A): a verdict-COMPLETE at 1/10 leaves
  9 runnable trials, so the n=10 advance rate and the per-runnable rate are both
  reported.

- **ρ.2-imp — statement + framework imperative (the rung-1-shaped form).** Same
  as ρ.2 but the anchor appends a fixed framework imperative after the judge's
  statement ("Produce that next."). This isolates whether the rung-1 8/10 came
  from the imperative (which the judge's bare statement lacks — the named
  incongruity) versus the remaining-work content alone. n per runnable ρ.1
  trial.

- **ρ.control — content-neutral trailing perturbation (mechanism isolation,
  P1-B).** Call 2 anchored with a trailing addition of *similar length and
  format* to the judge's statement but carrying **no remaining-work content** —
  the delegation standard re-stated ("Remember: delegate generation to a
  capability ensemble rather than writing inline."), no filename, no "what
  remains." If ρ.control advances at the same rate as ρ.2, the effect is mere
  trailing-token perturbation, not the remaining-work content, and the amendment
  ships for the wrong reason with different brittleness. n=10 on B2 (the rung-1
  base); B3 optional. This arm is the load-bearing causal control.

## Measurement definitions

- **names-correct:** the judge's statement references the unproduced
  deliverable's identity (filename or unambiguous description) and does not
  assert an already-produced file is missing. Adjudicated against the known base
  state (file 1 [and 2] produced; the test [and/or second module] not).
- **advance:** the call-2 tool call targets the unproduced deliverable.
- **stuck:** targets an already-produced file (the Finding G signature).
- **delegated:** the call-2 tool call is `invoke_ensemble` (not an inline
  client `write` of generated content — the Finding B shape).
- **no-tool-call:** the call-2 response carries no tool call (the
  premature-finish risk). Sub-classified (P3-B) from the retained text:
  `none-finish` (a completion/summary — the premature-finish shape),
  `none-text` (prose continuing the task without a tool call), `none-other`
  (refusal/confusion). The sub-classes have different remediation paths.
- Verdict parse: `parse_verdict` over think-stripped text (the landed helper).
  Remaining-statement extraction: `strip_verdict` (the landed helper) yields the
  post-VERDICT text; the statement is that text.

## Pre-registered decision rule (pass = the amendment is grounded)

All thresholds at n=10 are pass/fail boundaries with **wide confidence
intervals** (P2-C): a 7/10 vs 6/10 difference is not a distinguishable
population-rate difference; the rule is a go/no-go for a single-spike adoption
decision, not a precise rate estimate. The real-OpenCode multi-file convergence
run is the layer-matching confirmation regardless of the in-harness rate.

- **ρ.1 passes** if `names-correct ≥ 8/10` on *each* base (the anchor is
  trustworthy) AND `verdict-COMPLETE ≤ 1/10` on each base (no regression of the
  θ false-stop rate the termination mechanism already validated).
- **ρ.2 passes** if, on *each* base, `advance ≥ 7/10` AND `delegated ≥ 7/10`
  AND `no-tool-call ≤ 2/10` (the rung-1 no-tool-call rate as the tolerance
  ceiling; a higher rate means the anchor wording needs a BUILD-gate revision
  before the amendment ships). Both the n=10 rate and the per-runnable-trial
  rate are reported (P2-A).
- **Causal-isolation read (P1-B):** the remaining-work content is credited as
  the mechanism only if `ρ.2 advance − ρ.control advance ≥ 0.3` on B2. If
  ρ.control advances comparably to ρ.2, the effect is trailing-token
  perturbation, not remaining-work content — the amendment's framing is wrong
  even if the rate is high, and the ADR records the anchor as "trailing
  perturbation that happens to break churn" rather than "routes the computed
  signal forward."
- **Production-form read:** ρ.2 (statement-only) is the default amendment form;
  ρ.2-imp informs whether a framework imperative should wrap the judge's
  statement. If ρ.2-imp materially exceeds ρ.2 (≥ 0.2 advance gap), the
  amendment specifies statement + imperative; else statement-only (the simpler
  form).

**Mixed-result backstop (ADR-097 pattern), with the two failure modes named
(P2-B):**
- If ρ.1 passes but ρ.2 advance ∈ **[0.5, 0.7)** — this band is "materially
  above the A_current 0/10 baseline but below single-spike adoption confidence"
  — the amendment proceeds as **Conditional Acceptance** with the real-OpenCode
  multi-file convergence run as the discharge gate (the discipline ADR-037
  itself carries). The 0.5 lower bound is the floor for "the fix direction is
  real"; below it the fix is not working.
- **ρ.1 passes AND ρ.2 advance < 0.5** — distinct diagnostic: the anchor is
  *trustworthy* (judge names it right) but call 2 *ignores* it. This refutes
  route-the-signal-forward as composed and points at the action-call composition
  (the anchor placement/wording in `_seat_filler_messages`), not the judge —
  re-opens the alternatives at the composition layer.
- **ρ.1 fails (anchor poisoned)** — the judge cannot reliably say what remains;
  route-the-signal-forward is refuted at the source. Re-opens toward the
  framework-checklist alternative (deterministic requested-vs-written diff)
  since the model-judged remaining signal is unreliable.
- **delegation collapses (ρ.2 delegated < 0.7)** — the anchor reintroduces the
  Finding B inline-write shape; refuted regardless of advance rate.

## Fidelity discipline

- Composition through the landed code path (the rung-1 harness already does
  this; ρ extends it with the real judgment call + statement extraction +
  real-statement anchoring). The anchor in ρ.2 is the judge's actual output, not
  a hardcoded string — this is the production form.
- Assistant tool-call turns carry `content=""` not `None` (Ollama rejects null
  content — rung-1 fix, recorded).
- Full response text retained per trial for adjudication (names-correct is a
  judgment call read against the base state).
- Bases derived from the real ψ-capture request shape where possible; the
  task-text and tail-truncation edits are recorded.

## Scope (stated, not guessed across)

qwen3:14b, n=10×2 bases, file-write deliverables, tails to depth two. Same scope
class as θ. Non-write deliverables, deeper tails, and other seats are out of
scope (the ADR-037 recorded boundary carries forward). The real-OpenCode
multi-file convergence run is the layer-matching confirmation, deferred to BUILD
as the amendment's discharge gate.

## Harness

Extends `scratch/spike-multifile-progress/probe.py` →
`scratch/spike-rho-remaining-anchor/probe.py` (the real judgment call + statement
extraction + real-statement anchoring). Reuse the rung-1 classifier.

---

## Results (run 2026-06-08, $0 local qwen3:14b; n=10 per arm/base)

*(Run note: the first battery attempt stalled overnight when the machine slept —
process suspended, not hung; the single-call latency check on wake measured
24.6s/call and the battery was re-run clean.)*

### ρ.1 — judge remaining-naming (Factor 1)

| Base | REMAINING | verdict-COMPLETE | names-correct (specific) |
|------|-----------|------------------|--------------------------|
| B2 | 10/10 | 0/10 | **10/10 specific-correct** |
| B3 | 10/10 | 0/10 | **10/10 specific-correct** |

Every statement named `test_string_utils.py` by filename (e.g. "The test file
test_string_utils.py has not been created yet."). **ρ.1 PASSES** on both bases
(names-correct ≥ 8/10; verdict-COMPLETE ≤ 1/10). Factor 1 ≈ 1.0 — the judge's
remaining-work signal is trustworthy; the production anchor is not poisoned.

### ρ.2 / ρ.2-imp / control — anchored call 2 (Factor 2 + isolation)

| Arm | advance | stuck | other | none | delegated |
|-----|---------|-------|-------|------|-----------|
| ρ.2 B2 (statement-only) | 8/10 | 1 | 0 | 1 | 9/10 |
| ρ.2-imp B2 (statement + "Produce that next.") | **10/10** | 0 | 0 | 0 | 10/10 |
| **control B2 (content-neutral, same length)** | **0/10** | 7 | 3 | 0 | 10/10 |
| ρ.2 B3 (statement-only) | 9/10 | 1 | 0 | 0 | 10/10 |
| ρ.2-imp B3 (statement + imperative) | 9/10 | 0 | 0 | 1 | 9/10 |

### Decision-rule verdict — PASS (route-the-signal-forward grounded)

- **ρ.2 passes** both bases (advance ≥ 7/10; delegated ≥ 7/10; none ≤ 2/10).
- **Causal isolation (P1-B) — the load-bearing result.** ρ.2 B2 advance −
  control B2 advance = 8/10 − 0/10 = **0.8 ≥ 0.3**. The content-neutral
  perturbation (same length/format, no remaining-work content) advances 0/10
  and is stuck 7/10 — identical to the A_current baseline (rung-1: 0/10 advance,
  7/10 stuck). **The remaining-work content is the mechanism, not trailing-token
  perturbation.** The amendment's framing ("routes the computed signal forward")
  is correct, not an artifact. This is the strongest single result in the spike.
- **Production form (statement vs imperative).** ρ.2-imp ≥ ρ.2 on both bases
  (B2: 10 vs 8 — the 0.2 gap meets the imperative-specifying threshold and
  matches the methods-review incongruity that rung-1's 8/10 carried an
  imperative; B3: 9 vs 9 — equal). Combined ρ.2-imp 19/20 vs ρ.2 17/20.
  The imperative is never worse and removes B2's lone stuck + none cases. **The
  amendment specifies statement + framework imperative** ("Produce that next.").
- **Delegation preserved.** 9–10/10 across all production arms — no Finding B
  inline-write collapse. **no-tool-call** ≤ 1/10 (within the 2/10 ceiling; the
  two `none` were not inline writes).

**Composed production estimate:** Factor 1 (judge names correct) ≈ 20/20 ×
Factor 2 (imperative-anchored advance) ≈ 19/20 → the end-to-end multi-file
progress rate is ~0.9 at qwen3:14b, n=20 across two bases. Labeled composed (the
two factors measured on the same trials for ρ.2-imp; ρ.1 measured separately
confirms Factor 1 is not the bottleneck). n=10-per-cell precision caveat (P2-C):
boundary differences (8 vs 10) are not distinguishable population rates; the
go/no-go is clear, the exact rate is not.

**Scope:** qwen3:14b, file-write deliverables, tails to depth two — the θ scope
class. The real-OpenCode multi-file convergence run is the layer-matching
discharge gate (BUILD), carried as the amendment's Conditional Acceptance
condition. Artifacts: `scratch/spike-rho-remaining-anchor/` (probe, per-arm
results JSON, run logs, statements).
