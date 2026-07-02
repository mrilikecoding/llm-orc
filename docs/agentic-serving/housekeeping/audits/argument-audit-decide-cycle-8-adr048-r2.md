# Argument Audit Report — R2 (re-audit after revision)

**Audited document:** `docs/agentic-serving/decisions/adr-048-grounded-acceptance-composed-verification-gate.md` (revised/restructured since R1)
**R1 report checked against:** `docs/agentic-serving/housekeeping/audits/argument-audit-decide-cycle-8-adr048.md`
**Source material:** `docs/agentic-serving/essays/research-logs/cycle-8-spike-q2-grounded-acceptance.md`, `docs/agentic-serving/decisions/adr-046-target-architecture-per-turn-handler-one-ensemble.md`, `docs/agentic-serving/decisions/adr-014-calibration-gate-trajectory-level-extension.md`, `docs/agentic-serving/domain-model.md` (Invariants AS-2/AS-5/AS-11; Amendment Log #23)
**Genre:** ADR
**Date:** 2026-07-02

## Section 1: Argument Audit

### Summary

- **Genre:** ADR
- **Argument chains mapped:** 5 (renumbered §Decision-1 composed-gate mechanism; §Decision-2 acceptance-criteria-as-contract, now primary; §Decision-3 independence-via-isolation, now design-intent/validation-pending; §Decision-4 deterministic-primary signal ordering with AUQ/HTC as anchored candidate; §Decision-5 minimal-then-ladder)
- **R1 findings resolved:** 3/3 P1, 5/5 P2, 1/1 targeted P3 (P3-3); 1 P3 (P3-1, Status field) left unaddressed but was low-priority and non-blocking
- **New issues found this round:** 2 P2, 2 P3
- **Pyramid coverage map:** N/A (ADR genre, not Essay-Outline)

### R1 Finding-by-Finding Resolution

**P1-1 (independence-vs-discrimination, artifact channel) — RESOLVED.** The revision cleanly separates what the spike established from what remains committed. Context now states outright: "The spike tested discrimination on static fixtures; it did not exercise independence against a live builder (see §3)" (lines 41-42). §Decision-3 is retitled "Independence comes from architectural isolation plus determinism... (design intent; validation-pending)" (line 86) and its body states "This is a design commitment, not a spike-proven property" (line 94), naming both validation targets the R1 recommendation asked for almost verbatim: (a) the produced-artifact channel (lines 96-99) and (b) builder/judge model-weight correlation (line 99-101). §Conditional Acceptance moves independence out of the "accepted... spike-grounded" list and into validation-pending, flagging it as "the item most load-bearing on the ADR's title" (lines 205-206). §Provenance check states the same distinction plainly (lines 220-224). This does not over-correct into "the gate proves nothing" — the discrimination claims (executor catches wrong code, judge catches trivial-test gaming) remain stated with their original, well-supported confidence, and the hedge is scoped specifically to independence, not to the gate's demonstrated behavior.

**P1-2 (AUQ/HTC demotion by assertion) — RESOLVED per practitioner's option (b).** §Decision-4 now lists trajectory confidence as "an anchored, non-primary candidate, parallel to the judge: imperfect (builder-internal, weak on confident-wrong), so never the standalone gate, but ADR-046 named it as catching contract-conforming-but-anomalous generations a deterministic output check cannot see. Whether it earns a place in the gate is a BUILD-phase calibration question... This ADR does not foreclose ADR-046's open validation question by asserting the signal out; it carries it forward as a candidate pending evidence" (lines 112-121). §Rejected Alternatives now reads "Trajectory confidence as a standalone or primary gate. Rejected... It is retained as an anchored, non-primary candidate (§4), not excluded" (lines 158-160) — explicitly distinguishing rejection-as-standalone from exclusion. §Conditional Acceptance lists "the anchored trajectory-confidence candidate (§4): whether AUQ/HTC earns a place in the gate" among the validation-pending items (line 209), giving it the same open status as judge reliability. This is internally consistent with how the judge is treated (both anchored, non-primary, never standalone, pending calibration) and stays anchored — the text never proposes AUQ/HTC as a standalone gate. Per the practitioner's brief this was deliberately left NOT argued-out (rather than resolving whether "blind to confident-wrong" and "correlated-error-prone" are comparable failure magnitudes); the revision honors that scope — see also Framing C below.

**P1-3 (seat-contract relationship) — RESOLVED, with a new precision issue introduced (see NEW-P2-1, NEW-P2-2 below).** Context now adds "Relationship to ADR-046's seat contract" (lines 19-28), stating the granularity distinction the R1 recommendation asked for: the seat contract is per-seat admission, this ADR's gate is the loop-level accept/another-round decision, and "the two compose, they do not compete: a seat may clear its per-seat contract and the turn still fail the loop-level gate" (lines 27-28). This is a coherent compositional claim — a lower-level check passing does not guarantee a higher-level check passes, a standard and defensible layering pattern, and it resolves the "which of (a)/(b)/(c) is it" ambiguity R1 raised in favor of (a) (composes on top of seat-contract admission). However, the paragraph also asserts "The gate's deterministic half is the same `core/validation/` machinery applied at the turn level" (lines 26-27) — an unhedged technical-identity claim the spike does not establish (the spike's built executor is a bespoke script node running code+tests in-process, not a call into `ValidationConfig`/`ValidationEvaluator`) and that sits uneasily against ADR-046's own caveat that "seat-contract *wiring* (not just its sufficiency) is unvalidated" (ADR-046 §2). See NEW-P2-1.

**P2-1 (probe-vs-pre-registered) — RESOLVED.** §Decision-1 now states the coverage-awareness result "is a single exploratory probe on a second task, not one of the pre-registered F1–F3 falsification results, and rests on one stochastic sample. It is promising, not established" (lines 61-64).

**P2-2 (hedge placement) — RESOLVED.** §Decision-2 now places the single-sample qualifier inline, in the same sentence as the claim: "the judge caught the coverage gap only because the requirement stated the century rule it missed (one fixture, one fully-explicit requirement, no comparison case with a vaguer requirement and the same bug, so this is a single-sample result, not a measured scaling relationship)" (lines 69-72). The Negative Consequences hedge is retained as a second, reinforcing mention (line 187), not the only one.

**P2-3 (log-probability entropy reduction) — RESOLVED.** §Decision-4's AUQ/HTC parenthetical now reads "verbalized confidence plus generation-trajectory features, token-entropy patterns, attention-weight distributions over tool-call sequences, decision-confidence trajectories, with an entropy-collapse Abstain criterion" (lines 112-114) — restoring ADR-014's own multi-component vocabulary instead of the single-feature "log-probability entropy" shorthand.

**P2-4 (false-adequate double-use) — RESOLVED.** The revision now uses "coverage-gap" for the caught case (Context line 39: "discriminated correct / trivially-tested / wrong-code / coverage-gap build outputs") and "unstated-input ceiling" for the un-spikeable case (§Decision-5 line 132: "the unstated-input ceiling"; §Deferred line 141: "the unstated-input oracle rung"; §Consequences Negative line 186: "the unstated-input ceiling stands"). The two layers no longer share a label; grepping the revised document, "false-adequate" as a standalone label does not recur outside the fixture name `d_false_adequate` used only as a spike-artifact identifier.

**P2-5 (N-signal conjunction) — RESOLVED.** §Decision-5 now states "The composition rule for added rungs is itself an open question: a strict AND across more signals compounds false-rejection risk, so whether rungs join the same conjunction or a weighted/hierarchical combination is a design decision the two-signal spike did not settle" (lines 134-137), cross-referenced from §Consequences Neutral (lines 193-195).

**P3-1 (Status field vs. Conditional Acceptance) — NOT ADDRESSED, still P3.** Status remains "Proposed (2026-07-02)" (line 3) while §Conditional Acceptance describes what happens "on acceptance" (line 212). Given the ADR has not yet been accepted by the practitioner, "Proposed" pending a future conditional-acceptance transition is a defensible reading, not a contradiction — carried forward as a low-priority open item, not a regression.

**P3-2 (ADR-097 path numbering) — N/A, unaffected.** Pre-existing corpus-level inconsistency between ADR-046 and ADR-048's path citations; not an ADR-048 defect, untouched by this revision.

**P3-3 (single non-frontier judge doing both jobs) — RESOLVED.** §Rejected Alternatives now adds the clarifying parenthetical distinguishing the two independent axes (model capability; signal-source count) and names "a single non-frontier judge doing both code-correctness and adequacy without a separate executor" as "a different, untested variant, not adopted because the executor's deterministic ground truth is the gate's anchor" (lines 163-167).

### New Issues Found This Round

**NEW-P2-1. The seat-contract reconciliation paragraph asserts a mechanism-identity claim the spike and ADR-046 do not establish, and does not carry forward ADR-046's own "unwired" caveat.**

- **Location:** ADR-048 §Context, "Relationship to ADR-046's seat contract" (lines 19-28): "The gate's deterministic half is **the same `core/validation/` machinery** applied at the turn level; the isolated judge is precisely the 'more' ADR-046 flagged as possibly needed." Compare ADR-046 §2 (required source material): "the spike *analyzed* this framework (F3, design analysis); it did not wire `ValidationConfig`/`ValidationEvaluator` as the seat's pass/fail gate — the swap tests asserted correctness by inspection. So seat-contract *wiring* (not just its sufficiency) is unvalidated (§Open)." Compare the spike log's own Setup (`cycle-8-spike-q2-grounded-acceptance.md`, lines 28-31): the built executor is "deterministic: runs produced code + tests in-process," a bespoke script node, not a call into `core/validation/`'s `ValidationEvaluator`.
- **Claim:** The composed gate's deterministic half is architecturally identical to (not merely analogous to, or intended to reuse) the `core/validation/` seat-contract machinery.
- **Evidence gap:** Nothing in the spike or in ADR-046 demonstrates this identity; ADR-046 explicitly flags the opposite — that `core/validation/` wiring as a seat's pass/fail gate is unvalidated, and the spike's own executor is a separate, purpose-built script. The claim reads as settled architecture stated with the same confidence the ADR applies to its now-corrected independence claim, but without the parallel hedge. It is the one place in this Context section where the ADR's own newly-adopted discipline (distinguish spike-grounded fact from design commitment) is not applied.
- **Recommendation:** Hedge this sentence the same way §3 hedges independence — e.g., "the gate's deterministic half is *intended to reuse* `core/validation/` machinery... whether that wiring is shared code or parallel logic is BUILD-phase design," rather than asserting identity as fact.
- **Routing:** N/A (ADR genre; not an Essay-Outline pyramid-boundary finding).

**NEW-P2-2. "Contract" now names two distinct objects in adjacent Context/Decision text without disambiguation: ADR-046's seat contract (a deterministic `core/validation/` schema check) and ADR-048's newly-primary "gate's contract" (user/spec-supplied acceptance criteria fed to the judge).**

- **Location:** ADR-048 §Context (lines 19-20): "ADR-046 §2 named **the seat contract** (the `core/validation/` framework..." — a structural/schema/behavioral/quantitative/semantic validation config. Compare §Decision-2 heading (line 66): "Acceptance criteria are **the gate's contract**, and the gate's power scales with them. (primary)"; and body (lines 72-73): "acceptance criteria... are the primary lever, threaded to the verifier seats **as their contract**."
- **Claim (implicit):** "Contract," "seat contract," and "the gate's contract" are used precisely enough that a reader will not conflate ADR-046's deterministic validation-config object with ADR-048's natural-language acceptance-criteria input to the judge.
- **Evidence gap:** These are semantically different objects (a machine-checkable schema/behavioral spec vs. natural-language requirement text handed to an LLM judge) that now share the word "contract" across two ADRs the reader is required to hold in mind together, in the same section of the same document. The qualifiers ("seat," "gate's," "their") are the only disambiguators, and "their contract" (Decision-2, referring to verifier seats) is a close enough echo of "the seat contract" (Context, referring to ADR-046's `core/validation/`) that a reader skimming could reasonably conflate the two, especially since this is new text this round and the two paragraphs sit close together.
- **Recommendation:** Either rename Decision-2's object (e.g., "acceptance-criteria input" or "the judge's brief") to avoid the word "contract" entirely, or add one clause explicitly noting these are different contract objects at different levels (deterministic schema vs. judge-facing criteria).
- **Routing:** N/A.

**NEW-P3-1 (minor).** §Decision-2 states that "enriching the contract before the gate... raises the gate's ceiling more than judge-side improvements do" (lines 80-82). This comparative claim is a defensible structural inference (if the judge is bounded by what the criteria name, no judge capability improvement closes a criteria-shaped gap) rather than an empirical finding, but it is stated with the same unhedged confidence as an established design principle immediately after a sentence that carefully hedges the underlying single-sample result. Consider a light qualifier ("plausibly raises... more than," or noting this is a reasoned inference, not a tested comparison) for consistency with the surrounding paragraph's hedging discipline.

**NEW-P3-2 (minor).** §Decision-3's heading states "Independence comes from architectural isolation plus determinism, composed inside the ensemble" (line 86) as an unqualified assertion; the parenthetical tag "(design intent; validation-pending)" and the body's first sentence ("The *intended* source of independence is...") immediately hedge it, so no reader is actually misled, but the heading's phrasing itself has not been brought in line with the rest of the section's hedged language. Consider retitling to "Independence is intended to come from..." for full consistency.

### P1 — Must Fix

None. All three R1 P1 findings are resolved (see Finding-by-Finding Resolution above); no new P1s were introduced this round.

### P2 — Should Fix

- **NEW-P2-1** — seat-contract mechanism-identity overclaim (see above). Location: §Context, lines 26-28.
- **NEW-P2-2** — "contract" terminology overlap between ADR-046's seat contract and ADR-048's newly-primary acceptance-criteria-as-contract. Location: §Context lines 19-20 vs. §Decision-2 lines 66-73.

All five R1 P2 findings (P2-1 through P2-5) are resolved and are not repeated here; see Finding-by-Finding Resolution.

### P3 — Consider

- **NEW-P3-1** — unhedged comparative claim on criteria-elicitation vs. judge-side improvements. §Decision-2, lines 80-82.
- **NEW-P3-2** — §Decision-3 heading not yet brought into the section's own hedging register. Line 86.
- **P3-1 (carried over from R1, unaddressed)** — Status field ("Proposed") vs. §Conditional Acceptance's forward-looking "on acceptance" language. Low priority; plausibly intentional (not yet accepted).
- **P3-2 (carried over from R1, N/A to ADR-048)** — corpus-level ADR-097 path-numbering inconsistency between ADR-046 and ADR-048; not this ADR's defect.

## Section 2: Framing Audit

### Question 1: What alternative framings did the evidence support?

The three alternative framings R1 raised (A: specification-quality-primary; B: discrimination-established/independence-deferred; C: AUQ/HTC-as-candidate-third-signal) have each been substantially adopted into the ADR's own dominant framing this round, rather than remaining unacknowledged alternatives — this is the expected and correct outcome of a revision responding to a framing audit, not a new finding.

One residual, minor alternative surfaces from the *new* Context material: **Alternative D — granularity-reconciliation-primary.** Under this framing, the ADR's real center of gravity is not the gate mechanism (§1) or the acceptance-criteria lever (§2), but the newly-added act of placing this ADR at a specific point in ADR-046's granularity hierarchy (per-seat admission vs. loop-level accept). Belief-mapping: for the current framing (contract-quality-primary) to be right rather than granularity-reconciliation-primary, a reader needs to believe the *content* of the gate's decision (what makes it discriminate well) is more consequential than *where* it sits in the seat/loop hierarchy (whether it composes correctly with per-seat admission at all). This is a low-materiality alternative — the ADR's Context paragraph already states the granularity relationship plainly enough that a reader is not left to infer it — and is noted for completeness rather than as an omission requiring a fix.

### Question 2: What truths were available but not featured?

- **ADR-046's "seat-contract wiring... is unvalidated" caveat** (ADR-046 §2, required source material) is not carried into ADR-048's new "Relationship to ADR-046's seat contract" paragraph, which instead states the mechanism-identity claim as settled. Feeds NEW-P2-1.
- Everything else R1 identified under this question (the spike's static-fixture methodology, ADR-046's AUQ/HTC complementarity argument, ADR-014's full feature composition, the `(probe)` label, the spike's hedged "might not" language) is now featured in the revised document — this is the expected effect of a well-executed revision and is not re-flagged here.

### Question 3: What would change if the dominant framing were inverted?

**Dominant framing (post-revision):** acceptance-criteria quality is the primary, load-bearing lever (§2, marked primary); the composed executor+judge gate is necessary infrastructure that the criteria make bite; independence and judge-reliability are named, validation-pending BUILD/PLAY targets rather than accepted architecture; AUQ/HTC is an anchored, non-primary candidate carried forward rather than excluded.

**Inverted framing:** the verification-gate *mechanism* (isolation, determinism, the AND of two orthogonal signals) is what actually does the accepting; acceptance criteria are an important *input* to that mechanism, not a claim more fundamental than the mechanism itself — a perfectly-specified acceptance criterion with no isolated verifier consuming it produces nothing (documentation, not a gate).

Under this inversion: the claim that most needs restating is not "the criteria are primary" but a symmetric-necessity framing — mechanism and criteria are each necessary and neither is sufficient alone. Checking the actual text against this: §Context already states "The composed gate is necessary infrastructure; the acceptance criteria are what make it bite" (lines 33-34), which is a reasonably symmetric formulation, not an overclaim that criteria alone suffice. §Decision-2's "the deterministic executor remains the anchor" under weak criteria (line 79) reinforces this. **This inversion mostly confirms the elevation avoided the over-correction the practitioner flagged as the risk to watch for** — the revision does not claim the gate architecture is inert, and does not claim criteria alone would produce grounded acceptance without the gate. No P1/P2 framing finding results from this inversion check.

### Framing Issues

**P1:** None.

**P2 (cross-referenced from Section 1, not double-counted):**
- NEW-P2-1 — the seat-contract mechanism-identity overclaim, also visible as an excluded truth (ADR-046's unwired-wiring caveat) under Question 2.

**P3:**
- Alternative D (granularity-reconciliation-primary) — low-materiality, noted for completeness under Question 1, not a fix-requiring finding.

## Convergence-Saturation Signal (ADR-094)

**Convergence-Saturation Signal:** NOT TRIGGERED
- Round number: R2
- P1 count this round: 0 (down from R1's 3 — all three P1s resolved; no new P1s)
- P2 count this round (new, non-carry-over): 2 (NEW-P2-1 seat-contract mechanism-identity overclaim; NEW-P2-2 "contract" terminology overlap). All five R1 P2s are resolved and do not recur; not counted as carry-over.
- New framings or claim-scope expansions: none material. Alternative D (granularity-reconciliation-primary) is named under Question 1 as a low-materiality observation, not a claim-scope expansion requiring practitioner review — the revision's own Context text already forecloses the strong form of the concern (symmetric necessity is stated explicitly), so this is not being escalated.
- Recommendation: **CONTINUE to next round.** P1 count (0) meets the trigger threshold, but new-P2 count (2) exceeds the threshold (≤ 1), so the signal does not trigger on the numeric condition alone — no ESCALATE branch is needed since the shortfall is a clean count, not a judgment-call ambiguity about framing. The two new P2s are narrow and mechanical to fix (a hedge on one sentence; a rename or disambiguating clause on one term), unlike R1's P1s which required structural rework — R3 should be a short confirmation pass once those two sentences are edited, and is likely to trigger if no further new issues surface.

*This is a re-audit-after-revision that also sweeps for new issues per the dispatch brief; the verdict line above is included as directed (R2 of the standard sequence, not a single-purpose repair-verification re-audit).*
