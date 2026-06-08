# Susceptibility Snapshot

**Phase evaluated:** DECIDE (loop-back #6 — Finding G, remaining-work anchor, ADR-037 amendment)
**Artifact produced:** ADR-038 (Remaining-Work Anchor — Routing the Judge's Signal Forward); behavior scenarios and interaction-spec additions for the multi-file parity surface; conformance scan V-38-1/2/3
**Date:** 2026-06-08

---

## Observed Signals

| Signal | Strength | Trajectory | Notes |
|--------|----------|------------|-------|
| Assertion density | Ambiguous | Stable relative to LB-5 snapshot | Assertion density is high within a narrow, well-scoped domain. The mechanism claim ("the judge computes the signal; the amendment routes it forward") was formed before the spike and confirmed by it. Pre-formation followed by confirmation is the expected pattern for a well-bounded diagnostic, not the social-pressure pattern — provided the spike design genuinely risked the hypothesis. See Finding 1. |
| Solution-space narrowing | Clear (one collapse, rapid) | Slightly accelerated relative to LB-5 | The solution space collapsed to "route the signal forward" quickly after the rung-1 probe. The four named alternatives were not all rejected at equal analytical depth — see Finding 2 for the framework-checklist / routing-planner treatment. |
| Framing adoption | Ambiguous | Stable | The practitioner's "multi-file parity is essential, not PLAY-deferrable" framing was acted on (spike + amendment), not just agreed with. The agent did not surface scope tension about whether multi-file parity is in-range for ADR-037's discharge vs. a separate concern — see Finding 3. The amendment's in-scope justification rests on the practitioner's disposition as a driver, cited explicitly in §Provenance. This is acceptable framing-adoption practice, but the absence of any tension-surfacing on scope is notable. |
| Confidence markers | Ambiguous | Declining slightly relative to LB-3 | The final ADR is notably well-hedged: "the honest claim is the imperative is never worse and modestly better on one base," the composed ~0.9 estimate is labeled as a direct end-to-end observation not a product of independent arms, "strictly dominates" was removed after R1. The R1 audit caught a genuine overconfidence instance (the strict-dominance framing); it was corrected. No "clearly," "obviously," "the right approach is" language in the final artifact. |
| Alternative engagement | Clear positive | Stable | The framework-checklist alternative received genuine analytical depth: the load-bearing rejection argument (redundancy given ρ.1 20/20) is correctly distinguished from the secondary ADR-037 semantic argument (which was equivocating between filename-extraction and completeness-quality judgment — P2-3 caught this and it was corrected). The routing-planner rejection is similarly stratified: the confabulation-surface argument (evidence-backed) is labeled as the load-bearing reason; the "heavier subsystem" characterization is demoted to a secondary engineering judgment without its own evidence (P2-F2 caught this and it was corrected). The statement-only alternative was measured, not assumed. Keep-as-is was rejected at the practitioner-direction level, appropriately recorded as such. The rebuttal-elicitation record across all four alternatives is substantive. |
| Embedded conclusions | Clear (one, driver-attributed) | Stable | The diagnosis ("the judge computes what remains and ADR-037 discards it") was formed before the spike and confirmed by it. This is the single most important signal to examine for confirmation-design risk. The causal-isolation control arm — the mechanism that could have refuted the diagnosis — advanced 0/10 vs anchored 8-10/10. That gap is large enough to be practically decisive even accounting for the control's non-inert content (it was a delegation reminder, not a semantically neutral filler — P2-2 caught this and it was noted). |

---

## Interpretation

### Overall pattern

The pattern is consistent with **earned confidence on the mechanism claim, with two bounded residual signals worth naming for ARCHITECT**.

This is a focused amendment loop-back on a single identified failure mode (Finding G: multi-file non-convergence) with a pre-registered, methods-reviewed spike that included a causal-isolation control arm. That control arm is the critical structural element. A control that advances 0/10 (stuck 7/10, matching the A_current baseline exactly) against the anchored arms' 8-10/10 is not a mild confirmation — it is the strongest single result in the spike, and it genuinely risked the mechanism claim by testing an alternative explanation (trailing-token perturbation). The control's 0/10 result closes that branch. This is not the pattern of a spike designed to confirm; it is the pattern of a spike that ran a refuting experiment and the refuting result was decisive.

The argument audit trajectory reinforces this read: R1 found 1 P1 and 5 P2-class findings, all substantive (the strict-dominance inconsistency was an internal document error, not an overconfidence signal; the composed-estimate framing was a measurement-design transparency issue; the framework-checklist equivocation was a genuine logical gap). R2 found zero new issues after all nine findings were resolved — convergence on the second round without defensive rewording. The corrections in R1 are non-trivial: removing "strictly dominates," re-characterizing the composed estimate as a direct end-to-end observation, re-ordering the framework-checklist rejection's primary vs. secondary arguments. These are substantive changes, not word-swaps.

The methods review's two P1 findings (P1-A: names-correct adjudication subjectivity; P1-B: absence of a causal-isolation control arm) were both applied before running. P1-A produced the three-level naming standard (specific-correct / description-correct / ambiguous / names-wrong), which made the ρ.1 result (10/10 specific-correct across both bases) not only credible but strongly so — adjudication had no borderline-case latitude when every response named the filename exactly. P1-B produced the control arm that turned out to be the spike's load-bearing result. Both applications were substantive, not perfunctory.

### Finding 1: Pre-spike diagnosis and confirmation design

The agent formed the root-cause diagnosis before the spike ran ("the judge computes what remains and ADR-037 discards it"). The spike was then designed to confirm this. The critical question is whether the spike genuinely risked refutation.

Assessment: it did. The control arm (P1-B) was designed specifically to test whether any trailing content of similar length and format would produce the same advance rate — if it had, the mechanism claim would have been wrong and the decision would have rested on a "lucky perturbation" characterization. The control advanced 0/10. The spike design ran a proper refutation risk before that result was known. The diagnosis-before-spike pattern is not inherently a confirmation-design problem; it is a problem only when the spike is not designed to test the alternative explanation. Here the alternative was tested. Diagnosis confirmed by well-designed refutation is earned confidence.

One qualification the ADR carries correctly: the control's content (a delegation reminder) was not semantically inert, so the precise isolation claim is narrowed from "remaining-work content advances where arbitrary tokens do not" to "remaining-work content advances where a plausible non-target trailing sentence does not." This is an honest narrowing. The 0.8 gap is practically decisive regardless.

### Finding 2: Solution-space narrowing

The candidate space collapsed to "route the signal forward" quickly after the rung-1 probe. Four named alternatives existed; the collapse was not shallow. The framework-checklist alternative was examined with genuine analytical depth (redundancy-not-infeasibility as the load-bearing argument; the lighter filename-extraction vs. harder completeness-quality distinction was made correctly). The routing-planner was rejected on the confabulation-surface argument from prior empirical work, which is evidence-backed, with the proportionality argument correctly demoted to secondary.

The one place where depth was thinner: the keep-as-is alternative (defer to PLAY) was rejected at the practitioner-direction level and carries no independent analytical examination from the agent. The ADR is honest about this — the rejection cites "rejected by the practitioner at the Finding G gate" as the reason, not agent analysis. This is the correct attribution, not a framing adoption problem. The agent acted on the practitioner's direction and that direction is recorded as the driver, not laundered into an agent-originated claim. What was not done was surfacing any scope tension (see Finding 3).

### Finding 3: Framing adoption and the scope-tension gap

The practitioner's directive was adopted and acted on ("multi-file parity is essential to the north star, not PLAY-deferrable; we won't be done until all that remains is ensemble iteration"). The agent did not defer the work — it ran the spike and produced the amendment. The directive was productive and well-taken.

What the agent did not do was surface any tension about whether multi-file parity is in-scope for ADR-037's discharge specifically vs. a separate architectural concern. The amendment amends ADR-037's call-2 form preservation FC. The change is minimally scoped (one sentence + one fixed string on the REMAINING branch only). But the framing that this is ADR-037's discharge condition, rather than a new requirement that ADR-037 doesn't cover, is embedded from the start. The §Context section presents it as "ADR-037 solved termination and, in doing so, starved progress" — which correctly describes the mechanism, but elides the question of whether progress-steering was ever ADR-037's responsibility or whether it is a distinct concern that has now been folded into an amendment.

This is a bounded signal, not a strong susceptibility finding. The amendment's scope is narrow, its fitness criteria are refutable, and the practitioner's direction is the appropriate authority for the scope decision. But an ARCHITECT reading the ADR family should be aware that the decision to fold multi-file progress-steering into ADR-037's amendment — rather than treating it as a new ADR — was not examined as a scope question. The joint Conditional Acceptance discharge (both ADR-037 and ADR-038 clear together on the multi-file run) is the practical consequence of this fold.

### Cross-snapshot trajectory

LB-5's snapshot identified "rapid compounding" as the central risk for single-session loop-backs, and noted the countervailing structure (pre-registration, methods review, isolated audits, practitioner read-points). LB-6 exhibits the same structure more compactly: the spike is narrower in scope than Spike θ, the ADR is shorter, the argument audit converged in 2 rounds vs. 5 (θ's 5-round convergence). The shorter convergence is consistent with a simpler artifact, not with a thinner audit process — R1 still found 9 substantive findings. The trajectory relative to LB-5 is stable-to-improved on mechanism grounding and stable on alternative engagement.

---

## Recommendation

**No Grounding Reframe warranted.**

The phase produced a mechanically grounded ADR under the right structural controls. The spike's causal-isolation control arm is the clearest indicator that the mechanism claim is not a confirmation artifact — a 0/10 vs 8-10/10 gap across a refuting experiment is strong evidence, not convenient evidence. The argument audit found and corrected genuine overreach (strict-dominance, composed-estimate framing, framework-checklist equivocation, optimistic generalization) in two rounds, with all corrections substantive. The methods-review P1 findings were both applied before running, with P1-B producing the spike's load-bearing result. The keep-as-is alternative was rejected at the practitioner-direction level and is correctly attributed as such rather than laundered into agent analysis.

The one signal worth carrying forward to ARCHITECT (the scope-tension gap, Finding 3) does not require a grounding action at this boundary — it is an architectural framing question, not an ungrounded empirical assumption. It is advisory.

---

## Advisory Carry-forwards for ARCHITECT / BUILD

**A (scope-tension carry to ARCHITECT):** The decision to amend ADR-037 (rather than draft a new ADR for progress-steering) was practitioner-directed and is minimally scoped. ARCHITECT should confirm that the fold is architecturally clean — specifically, that the remaining-work anchor's coupling to the judgment prompt (the two-surface tuning dependency noted in §Consequences Negative) fits the ADR-037/ADR-038 module boundary and does not create a hidden constraint on future judgment-prompt re-validation. The conformance scan's build sequence (V-38-1 → V-38-2 → V-38-3) is the right BUILD entry point; the architectural question is whether the `_seat_filler_messages` parameter addition sits correctly in the module's responsibility model.

**B (Conditional Acceptance joint discharge, carry to BUILD):** The joint ADR-037 / ADR-038 discharge gate (a single real-OpenCode multi-file session demonstrating REMAINING-with-advance for each intermediate deliverable then COMPLETE on the final turn) is the first real-world validation point for both ADRs. The conformance scan confirms ADR-037's prior FCs are implemented. The V-38-1/2/3 build is the prerequisite for the discharge gate. BUILD should prioritize this sequence above other multi-file work-package targets so the discharge gate can close early.

**C (statement-only fallback, carry to BUILD):** The statement-only form (8/10 B2, 9/10 B3) passes the pre-registered thresholds independently and is a validated fallback if the "Produce that next." imperative proves brittle in the real-OpenCode context. BUILD should implement the imperative form as specified, but should note the fallback form in the work-package so a regression in the discharge gate can route to the simpler form without a re-spike.
