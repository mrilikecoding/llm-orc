# ADR-038: Remaining-Work Anchor — Routing the Judge's Signal Forward

> **Conditional Acceptance discharged 2026-06-08** (jointly with ADR-037). The
> real-OpenCode multi-file run (WP-LB-L) passed in a single session: turn 1
> wrote `string_utils.py`, turn 2 (REMAINING) advanced to `test_string_utils.py`
> via the anchor (no churn), turn 3 (COMPLETE) finished text-only and the client
> loop ended. Evidence: `scratch/wp-lb-l-acceptance/RESULTS.md`.

> **Updated by ADR-040 on 2026-06-10.** For tasks that name their deliverables,
> the REMAINING anchor is now the framework-computed `requested − produced` set,
> not the judge's one-sentence statement. Once ADR-040's deterministic
> completeness gate exists for the verdict (forced by Spike σ's judge
> false-COMPLETE), the remaining set is already computed, so the anchor rides it
> at no extra cost. The judge-statement anchor this ADR validated (Spike ρ, 19/20
> advance) still applies to tasks that name no files (the general-task fallback).
> See ADR-040.

**Status:** Accepted, Updated by ADR-040 (Cycle 7 loop-back #6 DECIDE, gate closed 2026-06-08;
Conditional Acceptance per ADR-097 discharged 2026-06-08 by the WP-LB-L
real-OpenCode multi-file run — see §Empirical grounding)

## Context

Finding G (cycle-status §"Finding G," 2026-06-07): the WP-LB-K real-OpenCode
acceptance run validated ADR-037's two-call termination mechanism — single-file
sessions converge — but exposed that **multi-file sessions do not converge**. On
a two-deliverable task ("write `string_utils.py` and `test_string_utils.py`"),
every trailing turn the seat-filler re-wrote `string_utils.py` and never
advanced to the test file; the session was kept alive (correctly) by the
termination judgment returning REMAINING because a requested deliverable was
genuinely missing, but it made no forward progress on the deliverable list.

ADR-037 solved *termination* (when to stop) and, in doing so, starved
*progress* (what to do next). Its two-call composition routes the judge's
output to the stop-decision and discards the judgment exchange before call 2
(ADR-037 §Decision 4 / FC call-2 form preservation — "the judgment exchange is
discarded; it does not ride into call 2's context, byte-equal to the measured
E4b composition"). But the judge's question ends *"If REMAINING, state in one
sentence what remains."* So the framework **computes the remaining-work signal
and then throws it away** — call 2's action selection has no
remaining-deliverable anchor and re-derives "write file 1" each turn.

This is a multi-file-parity blocker, not a PLAY-deferrable nicety (practitioner
disposition at the Finding G gate): the north star is parity with regular
model flows, and "we won't be done until all that remains is ensemble iteration
and improvement." ADR-037 stays Conditional Acceptance until multi-file
convergence lands.

## Decision

**On a REMAINING verdict, call 2's composition carries the judge's own
remaining-work statement forward as a next-step anchor, followed by a fixed
framework imperative.** The signal ADR-037 computes for the stop-decision is no
longer discarded — it is routed forward to steer the action call.

Concretely, the call-2 trailing region (the ADR-036 C3 standalone trailing
guidance) gains, on the REMAINING branch only, an appended anchor of the form:

> `{judge's one-sentence remaining-work statement}` + `" Produce that next."`

The judge's statement is its verbatim `VERDICT:`-stripped REMAINING output (the
existing `strip_verdict` helper); the imperative is a fixed framework string.
First turns, new-user-task tails, and COMPLETE verdicts are untouched — the
anchor exists only where a REMAINING verdict has produced a remaining-work
statement.

The imperative is included on the measured margin, not because it strictly
dominates: ρ.2-imp advanced 10/10 vs ρ.2 statement-only 8/10 on B2, but **tied
9/10 vs 9/10 on B3** (Spike ρ). At n=10/cell a 9-vs-10 difference is not a
distinguishable population rate, so the honest claim is *the imperative is never
worse across the two bases and modestly better on one* — it is adopted because
it costs one fixed string and removed B2's lone stuck and no-tool-call cases,
not because the evidence shows a population-level advantage.

This **updates ADR-037's FC (call-2 form preservation)**: call 2 is no longer
byte-equal to the pre-ADR-037 E4b composition; it is the E4b composition plus
the remaining-work anchor on the REMAINING branch. The rest of ADR-037 — the
judgment-first composition, the bare-form judgment call, the digest provenance,
the COMPLETE protocol-clean finish, the AS-3 backstop — stands unchanged. (See
ADR-037's dated update header.)

### Fitness criteria introduced / amended

- **FC (remaining-work anchor presence):** on a REMAINING trailing turn, the
  composed call-2 request contains the judge's remaining-work statement and the
  framework imperative in its trailing region. Refutable from composed-request
  inspection.
- **FC (call-2 form preservation — amended from ADR-037):** call 2 on REMAINING
  equals the ADR-036 C3 composition **plus** the remaining-work anchor, and
  nothing else of the judgment exchange (not the judge system message, not the
  digest, not the verdict literal) rides into call 2's context. The judgment
  question/digest remain discarded; only the stripped remaining-work sentence
  plus the imperative carry forward. Refutable from composed-request inspection.
- **FC (delegation preserved under the anchor):** the anchored call 2 still
  delegates generation (does not reintroduce the Finding B inline-write shape).
  Refutable: an inline `write` of generated content on an anchored call-2 turn,
  **or** a rise in no-tool-call (premature-finish) turns above the ρ-measured
  ≤1/10 — the anchor must not push the seat into finishing or writing inline
  instead of delegating the named deliverable.

## Rejected alternatives

### Keep FC-66 as-is; accept multi-file non-convergence (defer to PLAY)

Rejected by the practitioner at the Finding G gate: multi-file convergence is
essential to north-star parity, not an experiential-discovery nicety. A serving
layer that converges single-file but churns on the second deliverable does not
reach "all that remains is ensemble iteration." Deferring it would ship a known
parity gap as if it were an unknown to be discovered.

### Framework-tracked deliverable checklist (deterministic requested-vs-written diff)

The framework decomposes the task into a deliverable list up front and tracks
completion by diffing requested deliverables against the Session Action Record's
written paths; the next-action anchor is the deterministic diff. **Rejected
primarily as redundant:** Spike ρ.1 measured the judge naming the unproduced
deliverable at 20/20 specific-correct — a signal that is already trustworthy, so
building a deterministic decomposer to replace it adds a subsystem for no
reliability gain. (A secondary point, not the load-bearing one: the checklist's
"which deliverables were requested" is lighter than the *completeness-quality*
judgment ADR-037 established as semantically hard — "is this work adequate?" is
harder than "which filenames were asked for" — so the checklist is not as
infeasible as ADR-037's semantic argument might suggest; the rejection rests on
redundancy, not infeasibility.) Held as the fallback *if* ρ.1 had failed — a
judge that mis-named what remains would have forced a deterministic source; ρ.1's
20/20 closed that branch.

### Per-task routing-planner (decompose → track → dispatch)

A planning stage decomposes the task into a deliverable plan and drives the
sequence — the ADR-027 plan→dispatch→synthesize shape applied to the multi-turn
loop. **Rejected primarily because it re-opens the planner-confabulation surface
Cycle 6/7 spent effort bounding** (the evidence-backed reason): a plan-ahead
role narrates a deliverable sequence it has not yet observed, the ungrounded
composition AS-9's note-22 case removed. The route-the-signal-forward fix stays
inside the grounded per-turn loop — each anchor is the judge's read of *observed*
action records, not a forward plan. (Secondary, an engineering judgment without
its own evidence: the planner is also a heavier subsystem — a third role — for a
problem the judge's already-computed output solves with a one-sentence anchor;
this proportionality point reinforces but does not carry the rejection.)

### Statement-only anchor (no framework imperative)

Measured, not assumed: ρ.2 statement-only advanced 8/10 (B2) and 9/10 (B3) —
already a working fix on its own. ρ.2-imp (statement + "Produce that next.")
advanced 10/10 (B2) and 9/10 (B3) — better on B2, tied on B3. The imperative is
never worse across the two bases and modestly better on one (matching the
methods-review incongruity that rung-1's 8/10 already carried an imperative).
Statement-only is a legitimate simpler form and the fallback if the imperative
later proves brittle; the imperative is adopted on the B2 margin at a one-string
cost, not on a demonstrated population-level advantage (the n=10/cell caveat
means 9-vs-10 is not a distinguishable rate).

### Mere trailing-token perturbation (the null mechanism the control tests)

The causal-isolation control (P1-B): a content-neutral trailing addition of the
same length and format as the judge's statement, carrying no remaining-work
content. If it advanced comparably, the effect would be perturbation, not the
remaining-work signal, and this ADR's framing would be wrong. It advanced
**0/10** (stuck 7/10 — identical to the unanchored baseline) versus the
anchor's 8–10/10. The remaining-work content is causally responsible; the
framing holds. This is not a rejected design — it is the refuted null
hypothesis that licenses the decision's mechanism claim. **Qualification:** the
control text was a delegation-style reminder ("Remember: delegate generation…"),
not a semantically inert filler, so it may have actively held attention on
delegation *style* rather than being neutral; a stricter control would be
length-matched nonsense. The 0.8 gap is practically decisive regardless, but the
precise isolation claim is "remaining-work content advances where a plausible
non-target trailing sentence does not," not "where arbitrary tokens do not."

## Consequences

**Positive:**
- Multi-file sessions advance: the seat-filler moves to the next deliverable
  instead of churning on the first. The end-to-end measured rate is ρ.2-imp's
  **19/20 advance** across B2/B3 (10/10 + 9/10) — this is a direct observation
  of the production composition (the judge's real statement anchoring call 2),
  not an independence-valid Factor 1 × Factor 2 product: ρ.2-imp ran on the
  judge's actual statements, so Factor 1 is already folded into the 19/20. ρ.1's
  separate 20/20 confirms the anchor was not poisoned (it explains *why* the
  19/20 holds), rather than being an independent multiplicand. Baseline:
  A_current 0/10 advance. The Finding G blocker is removed at the composition
  layer.
- The fix reuses an already-computed signal — no new model call, no new role.
  Call 1 (the judgment) was already paying for the remaining-work statement;
  the amendment stops discarding it. Zero added latency on the REMAINING branch
  (the statement is already in the judgment response).
- The remaining-work content is causally isolated (control 0/10 vs anchored
  8–10/10), so the mechanism is understood, not a lucky perturbation. Whether it
  generalizes beyond the measured scope (qwen3:14b, file-write deliverables) is
  not claimed here — see the Negative scope boundary.
- Delegation is preserved (9–10/10 across arms): the anchor does not reintroduce
  Finding B inline writes.

**Negative:**
- Call 2 is no longer byte-equal to the measured E4b composition (the ADR-037
  property this amends). The anchored call-2 advance/delegation rates are
  Spike-ρ-measured (8–10/10), not inherited from ADR-036's E4b 9/10 — a fresh
  per-profile empirical property that composes with ADR-036/ADR-037's existing
  profile-swap re-validation FCs (a judgment-seat or action-seat profile change
  re-validates the anchored composition too).
- The amendment is validated at the composition layer (Spike ρ, harness), not
  yet end-to-end against the real client on a multi-file session. The
  real-OpenCode multi-file convergence run is the Conditional Acceptance
  discharge gate — a join/anchoring defect could degrade it.
- The imperative is a second fixed framework string (after the judgment
  question); like ADR-037's, it is tunable at the FC-58 evidence bar (wording
  revisions re-validate the affected ρ arms).
- **The judge prompt and the action anchor are now coupled.** Because the
  routed-forward anchor *is* the judge's remaining-work statement, a change to
  the judge question's wording (the θ-harness tuning surface) changes the
  statement that anchors call 2 — so a judge-prompt revision re-validates both
  the θ judgment arms AND the ρ anchor arms, not the judgment alone. The two
  re-validation surfaces are no longer independent.
- Scope is the θ class (qwen3:14b, file-write deliverables, tails to depth two).
  Deeper tails, non-write deliverables, and other seats remain the recorded
  boundary carried from ADR-037.

**Neutral:**
- The judgment exchange's *question and digest* remain discarded (ADR-037's
  privacy/context-bounding property holds); only the stripped remaining-work
  sentence plus the imperative carry forward. The client never sees the anchor
  (it exists on the framework → seat-filler hop only, like the ADR-036
  guidance).
- The progressive task-shape ladder (the WP-LB-K post-gate plan) is partly
  consumed: multi-file convergence was its rung 1, now grounded; the ladder
  continues from deeper/mixed shapes.

## Empirical grounding (ADR-097 filter)

**Grounding path: spike validation.** Spike ρ — pre-registered, methods-reviewed
before any run (2 P1 / 3 P2 / 2 P3 applied; the causal-isolation control arm and
the statement-vs-imperative split are reviewer-directed), full-n denominators,
pre-registered decision rule passed, two bases (B2/B3), causal-isolation control
refuting the perturbation null (0/10 vs 8–10/10). Research log
`essays/research-logs/cycle-7-spike-rho-remaining-work-anchor.md`.

**Conditional Acceptance: the discharge gate is the BUILD real-OpenCode
multi-file convergence run** — a **single** real-OpenCode session on a
multi-deliverable task in which the session both advances through all
deliverables (no churn on file 1) **and then** converges (the COMPLETE finish
ADR-037 validated), both observed in that one session — separate single-file and
advance-only runs do not satisfy it. Verified from serve-log evidence within the
run: `turn decision:` lines showing REMAINING with a `dispatch start` for each
*distinct* deliverable in sequence, then a COMPLETE `action=finish`. This is the
same layer-match discipline ADR-037 itself carries; it folds into ADR-037's
existing Conditional Acceptance discharge (both clear together on the multi-file
run).

## Provenance check

Driver-derived: Finding G and the root-cause diagnosis (cycle-status §Finding G;
the WP-LB-K acceptance run evidence — all writes targeted file 1); the
rung-1 probe (advance 0/10 → 8/10 with a hardcoded anchor;
`scratch/spike-multifile-progress/`); all Spike ρ rates (ρ.1 20/20, ρ.2 8–9/10,
ρ.2-imp 9–10/10, control 0/10 — `scratch/spike-rho-remaining-anchor/` + the
research log); the causal-isolation control and the statement-vs-imperative
discrimination (research-methods review P1-B + the named incongruity); the
practitioner's "multi-file parity is essential, not PLAY-deferrable" disposition
(Finding G gate); the framework-checklist and routing-planner rejections (the
orchestrator loop-back brief's belief-map request; the semantic-decomposition
argument is ADR-037 §Context).

Drafting-time synthesis, labeled as such: the "statement + imperative strictly
dominates" framing (true on the measured margin — 19/20 vs 17/20 — but n=10/cell
cannot distinguish 9 from 10, so "strictly dominates" is a drafting-time
characterization of a modest measured edge, not a population claim); the
composed ~0.9 production estimate (Factor 1 × Factor 2 arithmetic from
same-and-separate-trial measurements, not an end-to-end multi-file measurement —
that is the deferred discharge gate); the framing that the imperative "costs one
fixed string" (an implementation-cleanliness read).
