# Argument Audit Report — R3 (re-audit after revision)

**Audited document:** `docs/agentic-serving/decisions/adr-048-grounded-acceptance-composed-verification-gate.md`
**R2 report checked against:** `docs/agentic-serving/housekeeping/audits/argument-audit-decide-cycle-8-adr048-r2.md`
**Source material:** `docs/agentic-serving/decisions/adr-046-target-architecture-per-turn-handler-one-ensemble.md`, `docs/agentic-serving/essays/research-logs/cycle-8-spike-q2-grounded-acceptance.md`, `docs/agentic-serving/domain-model.md`
**Genre:** ADR
**Date:** 2026-07-02

## Section 1: Argument Audit

### Summary

- **Genre:** ADR
- **Argument chains mapped:** 5 (unchanged from R2 — §Decision-1 composed-gate mechanism; §Decision-2 acceptance-criteria-as-contract, primary; §Decision-3 independence-via-isolation, design-intent/validation-pending; §Decision-4 deterministic-primary signal ordering with AUQ/HTC as anchored candidate; §Decision-5 minimal-then-ladder)
- **R2 findings resolved:** 2/2 P2 (NEW-P2-1 seat-contract mechanism-identity overclaim; NEW-P2-2 "contract" terminology overlap)
- **R2 findings left unaddressed (as expected — the dispatch scoped this revision to the two P2s):** NEW-P3-1 (unhedged criteria-elicitation comparative claim), NEW-P3-2 (§Decision-3 heading not yet in the section's hedged register), P3-1 carried from R1 (Status field), P3-2 carried from R1 (N/A, corpus-level)
- **New issues found this round:** 0
- **Pyramid coverage map:** N/A (ADR genre, not Essay-Outline)

### R2 Finding-by-Finding Resolution

**NEW-P2-1 (seat-contract mechanism-identity overclaim) — RESOLVED.** The flagged sentence previously read "The gate's deterministic half is **the same** `core/validation/` machinery applied at the turn level." It now reads: "The gate's deterministic half performs the same *kind* of check the seat contract does (execution and behavioral validation) and could reuse the `core/validation/` framework at the turn level, though the spike's executor was a bespoke script, so whether they share an implementation is a BUILD wiring choice, not something established here." This matches the R2 recommendation almost exactly ("hedge this sentence the same way §3 hedges independence... rather than asserting identity as fact"). Three checks:

1. **Does it still overclaim?** No. "Performs the same kind of check" is a category claim (both are execution/behavioral checks), not an implementation-identity claim. "Could reuse" is explicitly modal, not asserted as fact. The sentence names its own epistemic status ("not something established here") and routes the open question to BUILD, the same move §3 makes for independence.
2. **Does it now carry forward ADR-046's own caveat** (seat-contract *wiring* is unvalidated; the spike's executor was a bespoke script, not a `ValidationEvaluator` call)? Yes — "the spike's executor was a bespoke script" states the same underlying fact ADR-046 §2 flags ("the swap tests asserted correctness by inspection... seat-contract wiring... is unvalidated"), in ADR-048's own words. This also resolves the parallel Framing Audit finding from R2 (Question 2: ADR-046's unwired-wiring caveat was an excluded truth) — it is no longer excluded.
3. **Does the hedge contradict "the two compose, they do not compete... a seat may clear its per-seat contract and the turn still fail the loop-level gate"** (the sentence immediately following)? No. "The two" in that sentence resolves to the two *decision points* named at the top of the paragraph — the seat contract (per-seat admission) and the accept gate (loop-level accept/another-round) — not to the deterministic-half/judge-half split inside ADR-048's own gate. The explanatory clause after the colon ("a seat may clear its per-seat contract and the turn still fail the loop-level gate") confirms this reading unambiguously: it is restating the granularity distinction from the paragraph's second sentence, not the implementation-sharing question the hedge addresses. Composability-of-decisions and shared-implementation are independent axes; hedging one says nothing about the other. No contradiction.

**NEW-P2-2 ("contract" terminology overlap) — RESOLVED.** §Decision-2 now reads: "threaded to the verifier seats as their **acceptance-criteria contract** (the natural-language spec the judge checks against, distinct from ADR-046's deterministic seat contract)." This is the disambiguating clause R2 recommended as one of its two acceptable fixes. Checked against every other use of "contract" in the document:

| Location | Phrase | Refers to | Consistent with disambiguation? |
|---|---|---|---|
| §Context (seat-contract paragraph) | "the seat contract", "its own contract", "per-seat contract" | ADR-046's deterministic `core/validation/` object | Yes — matches "ADR-046's deterministic seat contract" in the new clause verbatim in substance |
| §Context ("load-bearing variable") | "contract quality" | The acceptance-criteria contract (paragraph closes with "why acceptance-criteria-as-contract (§2) is treated as primary") | Yes, resolved within the same paragraph; pre-existing forward reference, untouched by this round's edit, not a regression |
| §Decision-2 heading | "the gate's contract" | Acceptance criteria (equated in the heading itself) | Yes — heading's equation is then given its formal, disambiguated name in the body |
| §Decision-2 body | "acceptance-criteria contract", "the contract" (elicitation sentence), "the contract is thin" | Acceptance criteria | Yes, all local uses after the defining sentence refer back to the same disambiguated object |
| §Decision-1 | "isolated contract-confidence judge" | The judge that assesses confidence against acceptance criteria | Yes — "contract-confidence" ties to the acceptance-criteria contract, never to the seat contract, since the judge never checks schema/structural conformance |
| §Decision-4 | "contract-confidence questions execution cannot answer (test adequacy, requirement coverage)" | Same object as above | Yes |
| §Deferred | "L1 contract-confidence judge", "the contracts threaded down are rich" | Same object | Yes |

No location uses "contract" in a way that now conflicts with the new disambiguating clause. The only pre-existing internal variation ("seat contract" vs. "per-seat contract," both referring to ADR-046's object) predates this round's edit (present at R2, quoted in R2's P1-3 resolution) and is not something this round's fix touched or worsened.

**R1 findings (3 P1, 5 P2) — spot-checked, all still hold, none disturbed.** The two edits this round are narrowly scoped to the seat-contract paragraph (§Context) and one clause in §Decision-2. Confirmed unchanged and intact:
- P1-1 (independence-vs-discrimination): §Context's "The spike tested discrimination on static fixtures; it did not exercise independence against a live builder (see §3)" and §Decision-3's "design commitment, not a spike-proven property" language — untouched.
- P1-2 (AUQ/HTC demotion): §Decision-4's "anchored, non-primary candidate... not excluded" and §Rejected Alternatives' matching language — untouched.
- P1-3 (seat-contract relationship, granularity distinction) — the granularity claim itself ("per-seat admission" vs. "loop-level accept/another-round decision") and the composability claim ("the two compose, they do not compete") are both preserved verbatim; only the mechanism-identity sentence between them was edited (see NEW-P2-1 above).
- P2-1 through P2-5 (probe-vs-pre-registered hedge, hedge placement, AUQ/HTC vocabulary, coverage-gap/unstated-input-ceiling terminology split, N-signal conjunction) — all quoted language confirmed present and unchanged at its R2 location.
- Carried P3s (Status field, ADR-097 path numbering, NEW-P3-1 criteria-elicitation comparative claim, NEW-P3-2 §Decision-3 heading register) — confirmed unchanged, as expected since the dispatch scoped this revision to the two P2s only.

### New Issues Found This Round

None. The two edits are narrow, additive hedges/clauses that resolve their target findings without introducing new overclaims, contradictions, or terminology drift elsewhere in the document.

### P1 — Must Fix

None. Zero P1s at R2, zero at R3.

### P2 — Should Fix

None. Both R2 P2s (NEW-P2-1, NEW-P2-2) are resolved; no new P2s introduced.

### P3 — Consider

Carried forward, unaddressed by design (out of this round's scope) and not regressed:
- **NEW-P3-1** — §Decision-2's "raises the gate's ceiling more than judge-side improvements do" remains an unhedged comparative/structural inference stated with more confidence than the surrounding single-sample-hedged text. Location: §Decision-2.
- **NEW-P3-2** — §Decision-3's heading ("Independence comes from architectural isolation plus determinism, composed inside the ensemble") remains unqualified at the heading level even though the parenthetical tag and body hedge it immediately. Location: §Decision-3 heading.
- **P3-1 (carried from R1)** — Status field ("Proposed") vs. §Conditional Acceptance's forward-looking "on acceptance" language. Low priority, plausibly intentional.
- **P3-2 (carried from R1, N/A to ADR-048)** — corpus-level ADR-097 path-numbering inconsistency between ADR-046 and ADR-048; not this ADR's defect.

## Section 2: Framing Audit

### Question 1: What alternative framings did the evidence support?

Unchanged from R2. The three R1 alternative framings remain substantially adopted into the ADR's own dominant framing. Alternative D (granularity-reconciliation-primary), surfaced at R2 as a low-materiality observation about the new "Relationship to ADR-046's seat contract" paragraph, is unaffected by this round's two sentence-level edits — the paragraph's granularity claim (the actual content Alternative D responds to) was not touched; only the mechanism-identity clause nested inside it was hedged. No new alternative framing is introduced by these edits.

### Question 2: What truths were available but not featured?

**Resolved this round:** ADR-046's "seat-contract wiring... is unvalidated" caveat, flagged at R2 as excluded from ADR-048's "Relationship to ADR-046's seat contract" paragraph, is now present in substance — "the spike's executor was a bespoke script" states the same underlying fact in ADR-048's own words, and defers the open question to BUILD exactly as ADR-046 does. This closes the Question-2 finding from R2 (previously feeding NEW-P2-1); it is not re-flagged.

No other previously-unfeatured truth surfaces from this round's two edits.

### Question 3: What would change if the dominant framing were inverted?

Unchanged from R2. The dominant framing (acceptance-criteria quality as the primary, load-bearing lever; the composed gate as necessary infrastructure the criteria make bite) still states a symmetric-necessity position in §Context ("The composed gate is necessary infrastructure; the acceptance criteria are what make it bite") and in §Decision-2 ("the deterministic executor remains the anchor" under weak criteria). Neither of this round's edits touches this framing balance — the mechanism-identity hedge concerns implementation-sharing with ADR-046, not the criteria-primacy vs. mechanism-primacy question the inversion probes. No P1/P2 framing finding results.

### Framing Issues

**P1:** None.

**P2:** None. NEW-P2-1, previously cross-referenced here from Section 1, is resolved (see Question 2 above).

**P3:**
- Alternative D (granularity-reconciliation-primary) — carried from R2, low-materiality, unaffected by this round's edits.

## Convergence-Saturation Signal (ADR-094)

**Convergence-Saturation Signal:** TRIGGERED
- Round number: R3
- P1 count this round: 0 (trend: R1=3 → R2=0 → R3=0)
- P2 count this round (new, non-carry-over): 0 (both R2 P2s — NEW-P2-1, NEW-P2-2 — resolved; no new P2s introduced by this round's edits)
- New framings or claim-scope expansions: none. Alternative D was named at R2 and is unaffected by this round's edits; no new warrant or claim-scope characterization surfaces this round.
- Recommendation: **STOP at this round.** All three trigger conditions hold: P1 = 0, new-P2 count (0) ≤ 1, and no new framings surfaced. The P1 trend (3 → 0 → 0) and the P2 trend (0 R1-carry / 5 resolved at R2 / 2 new-then-resolved at R3) show the document converging rather than drifting — each round's findings are narrower in scope than the last (structural rework at R1, two isolated sentences at R2/R3), and this round's fixes did not introduce any new issue at any severity. Remaining open items (four P3s) are all either out of this round's scope by design or corpus-level and not ADR-048's defect; none blocks Conditional Acceptance.

*This is a re-audit-after-revision (verifying the R2 P2 repairs) that also sweeps for new issues, per the dispatch brief. The verdict line above is included as directed — this is R3 of the standard sequence (R1 → R2 → R3), not a single-purpose repair-verification re-audit that would omit it.*
