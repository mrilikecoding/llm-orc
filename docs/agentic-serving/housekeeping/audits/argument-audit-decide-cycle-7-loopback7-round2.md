# Argument Audit Report — Round 2

**Audited document:** `docs/agentic-serving/decisions/adr-039-content-anchor.md`
**Source material:**
- `docs/agentic-serving/essays/research-logs/cycle-7-spike-xi-content-anchor.md`
- `docs/agentic-serving/housekeeping/audits/argument-audit-decide-cycle-7-loopback7.md` (R1 findings)
- `scratch/spike-xi-content-anchor/` (rates confirmed in R1; not re-read this round)
**Genre:** ADR
**Date:** 2026-06-09

---

## Section 1: Argument Audit

### Summary

- **Genre:** ADR
- **Argument chains mapped:** 5 (Decision rationale; form selection; causal isolation; rejected alternatives; scope of consequences)
- **Issues found:** 1 (0 P1, 1 P2, 0 P3)
- **Pyramid coverage map:** N/A
- **Expansion-fidelity findings:** N/A

### R1 closure verification

**P2-1 (Positive overclaim) — CLOSED.**

The revised Consequences/Positive first bullet now reads:

> "Dependent files reference real sibling APIs: cross-file resolution moved from 3/10 to 10/10 (Base T) and 0/10 to 10/10 (Base V) on the cheap qwen3:8b coder, in the coder-generation harness. The Finding H blocker is removed at the dispatch layer in that harness; the real-client confirmation is the discharge gate, not yet cleared."

The summary-impression point now carries the harness-only scope explicitly, and names the discharge gate as the thing that is not yet cleared. This matches the scope of the evidence. The Negative bullet (harness scope, not yet end-to-end) still does the structural work; the Positive bullet no longer overclaims relative to it. Adequate.

---

**P2-2 (Decision/FC "all produced siblings" scope) — CLOSED.**

The Decision's fourth paragraph now reads (lines 37-38):

> "The spike validated injecting the **single relevant** sibling's signatures. Whether production injects all produced files' signatures or a dependency-inferred subset is a selection-policy detail deferred to BUILD, bounded by the same context-budget reasoning that selected signatures over full content... The Decision commits the mechanism (framework-sourced signatures into the callee dispatch); it does not commit a specific multi-sibling selection policy, which is unmeasured here."

The FC (content-anchor presence) gained the parenthetical (line 43):

> "(The selection scope, all produced files versus a dependency-inferred subset, is a BUILD detail; Spike ξ validated injecting the single relevant sibling, not a multi-sibling selection policy.)"

Both the Decision and the FC now flag the untested-design-choice at the point where a reader would form a commitment. A reader cannot mistake the all-siblings scope for a measured property. Adequate.

---

**P2-3 (ADR-038 epistemic asymmetry) — CLOSED.**

The Empirical Grounding section now ends with (lines 93-94):

> "Unlike ADR-038, whose Conditional Acceptance is already discharged (the WP-LB-L real-OpenCode multi-file run), ADR-039's discharge is pending. The route-the-signal-forward rhyme with ADR-038 is structural; it does not transfer ADR-038's cleared empirical confidence to this ADR, whose end-to-end behavior under the real client is still unmeasured."

The asymmetry is named explicitly where the empirical grounding discipline is stated, which is the authoritative location. The rhyme language in Context and Consequences still reads as it did (the rhyme is accurate), and the new paragraph gives the reader the correct confidence calibration before they leave the section. One sentence in Context saying "ADR-038's discharge is already cleared; ADR-039's is not" would be slightly more proximate to the rhyme invocation, but the Empirical Grounding placement is the standard location for this type of epistemic flag, and a reader reading Context would reach the section in normal forward reading. The closure is adequate.

---

**FI-P2-1 (Discharge gate README criterion) — CLOSED, with one residual observation (see P2-1 below).**

The Empirical Grounding discharge gate now reads (lines 92):

> "The README's prose-to-code coherence is *observed* at the gate but is not a discharge requirement: the anchor targets code-generating callees, and whether it reaches a prose-generating callee (`prose-improver`) is the recorded prose-coherence scope question (see Negative consequences), not a behavior this ADR's mechanism is designed to deliver."

This is the recommended resolution (option a from the R1 framing): README coherence is now scoped as a characterization observation, not a gate criterion. The gate verifies `cli.py` and `test_*` cross-file references; the README is observed but a README failure would open a prose-anchor follow-up, not block ADR-039's full acceptance. The gate is now internally consistent with the Negative consequences boundary on prose-to-code coherence.

One sentence that was in the R1 discharge gate description — "the README documents real functions" — no longer appears as an unqualified verification step. Adequate.

---

**P3-1 (Causal isolation softening) — CLOSED.**

The revised Context bullet (lines 20-21) now reads:

> "Neither API-shape alone nor extra tokens alone reliably breaks the guessing; the real sibling API does (B minus decoy = 1.0, B minus filler = 0.9, both far past the 0.3 isolation gate). The model's own priors are weak but insufficient (the filler arm resolved one trial fully and two only partially), so the anchor supplies what the priors cannot."

The phrase "only the real sibling API does" from R1 has been replaced by "reliably breaks the guessing," and the parenthetical adds "the filler arm resolved one trial fully and two only partially." This captures both the binary-resolves verdict (1/10 is weak) and the graded-resolution observation that the R1 P3-1 flagged without burying the causal conclusion. The language is now precise rather than absolute. Adequate.

---

**P3-2 (Research log stale qwen3:14b sentence) — STATUS UNCHANGED.**

The R1 P3-2 was a housekeeping edit to the research log (lines 41-45, the superseded pre-correction body text naming qwen3:14b). The dispatch brief does not mention this finding was addressed, and nothing in the research-log read confirms a change. This is a research-log artifact, not an ADR argument error, so its continued presence does not block the ADR gate. It remains open as a minor housekeeping item. Not re-counted as a new finding this round.

---

### P1 — Must Fix

No P1 findings this round.

---

### P2 — Should Fix

**P2-1 (carry-over, partially resolved): The Empirical Grounding discharge-gate / prose-coherence closure is adequate, but a new internal tension is introduced by the revised framing.**

Location: §Empirical Grounding, discharge gate; §Consequences, Negative, final bullet.

The FI-P2-1 fix correctly scopes the README observation as "not a discharge requirement." However, the Negative consequences section still states (line 81):

> "prose-to-code coherence (the README's invented Rankine scale, a prose deliverable referencing code) remain the recorded boundary; the prose failure shape in particular may need a prose-targeted variant this ADR does not cover."

And the discharge gate now states the README is "observed at the gate but is not a discharge requirement." These are consistent. The tension is minor: the discharge gate says the README's coherence is *observed* (implying it will be checked as a characterization), but the Negative consequences section does not tell an evaluator what "observed" produces — is a README failure at the gate a signal that opens a new ADR, or an expected result? The gate text says it "would open a prose-anchor follow-up," which is an adequate answer, but the Negative section does not cross-reference this. A reader reading only Consequences and then the gate criteria would have to deduce this relationship.

This is a P2 rather than a P1 because the information is present (in the gate text) and the document is internally consistent; the issue is that the two places where a reader forms expectations about README behavior (Consequences/Negative and the discharge gate) do not directly reference each other. An evaluator at the gate needs to know a README failure is expected and opens a follow-up, not a re-run. The Consequences section is the natural place to say this explicitly.

Recommendation: add one sentence to the Negative consequences prose-boundary bullet: "At the discharge gate, a README coherence failure is expected-and-characterization, not a gate block; it opens a prose-anchor follow-up ADR." This makes the evaluation criterion legible without reading the full discharge gate text.

Note: this is a genuine residual from the FI-P2-1 fix, not a re-opening of R1's finding. The R1 fix was correct; the residual is a cross-section linkage gap introduced by scoping the gate criterion precisely.

---

### P3 — Consider

No new P3 findings this round.

---

### Regression check: cross-edit consistency

The "single-sibling validated" qualifier introduced in the Decision (P2-2 fix) was checked against all other sections:

- **Consequences/Positive (line 70):** reports the 10/10 harness rate, which was measured with a single sibling. No overclaim introduced; the rate is still accurate for the measured unit.
- **Consequences/Negative (line 77-78):** "Validated at the coder-generation layer (Spike ξ harness, qwen3:8b, single-hop dependencies), not yet end to end against the real client on a multi-file trajectory." The "single-hop" language already covered the single-sibling scope; the P2-2 fix adds precision without contradicting this bullet.
- **FC (content-anchor presence, line 43):** now reads "the callee dispatch context contains the relevant produced siblings' API signatures" with the BUILD-detail parenthetical. The word "relevant" in the FC hedges the selection scope appropriately and aligns with the Decision's "single relevant sibling" language. Consistent.
- **Provenance check (line 100):** already stated the single-sibling caveat accurately. The P2-2 fix brings the Decision and FC into alignment with Provenance without changing Provenance. No contradiction.

No cross-edit contradictions detected.

---

## Section 2: Framing Audit

The framing audit examines what the content selection excludes.

### Question 1: What alternative framings did the evidence support?

The three alternative framings identified in R1 (path-only injection, API-format-as-constraint, generate-then-repair) are still available from the source material. None have been taken up in the revised ADR, and none need to be: the R1 framing assessment was that the dominant framing is defensible and that the exclusions were soundly reasoned. This round confirms no new alternative framing has been introduced by the edits. The revisions tighten scope claims; they do not change the argument's framing.

### Question 2: What truths were available but not featured?

The three observations from R1 (filler graded-resolution data, now handled by the P3-1 closure; the common-name prior predictability dimension; and the discharge gate README criterion) were the material underrepresentations. The P3-1 closure addressed the filler graded-resolution data directly in Context. The discharge gate README criterion is addressed by the FI-P2-1 closure. The common-name prior predictability dimension (R1 FI-P3-1 territory) remains unaddressed, but as noted in R1 this is a characterization point about the discharge gate's scope evaluation, not a gap that changes the ADR's argument. No new omissions are introduced by the revisions.

### Question 3: What would change if the dominant framing were inverted?

The inverted framing ("make the model read, not compensate for it not reading") was analyzed in R1 and the Rejected Alternatives / "Induce the model to read" section engages with it directly. The revisions do not change this section. The framing is unchanged and the assessment stands.

### Framing Issues

**FI-P3-1 (carry-over): The path-only injection alternative remains unnamed in Rejected Alternatives.**

This was a P3 in R1 (not a dispatch blocking finding). The R1 recommendation was to add a brief Rejected Alternatives entry for "framework-injected path reference without content extraction," citing the live-trajectory zero-read evidence as the rejection basis. This has not been addressed. Carried as P3 for completeness; it does not block the gate.

---

## Convergence-Saturation Signal (ADR-094)

**Convergence-Saturation Signal:** TRIGGERED

- Round number: R2
- P1 count this round: 0 (Section 1 + Section 2 combined)
- P2 count this round (new, non-carry-over): 1 (P2-1 — the Consequences/discharge-gate cross-reference gap, a new finding introduced by the FI-P2-1 fix; not present in R1)
- New framings or claim-scope expansions: none. The P2-1 residual is a cross-section linkage gap, not a new framing or claim-scope characterization.
- Recommendation: STOP at R2

Signal triggers: P1 = 0, new P2 = 1 (at the ≤1 threshold), no new framings surfaced. The single new P2 is a minor linkage gap in the Consequences section that does not change the ADR's argument or gate criteria — it is an editorial precision improvement. The ADR is substantively clean: all R1 P1s were never present; all four R1 P2 closures are adequate; no regression or cross-edit contradiction detected; data layer confirmed clean in R1 and unchanged. The gate may proceed. If the practitioner addresses the P2-1 residual in the same revision pass (one sentence added to the Consequences/Negative prose-boundary bullet), R3 would be a confirmatory pass with 0 P2 new. That pass is not required by the signal, but is available if the practitioner wants a clean record.

*Standard-sequence audit: the verdict line applies.*
