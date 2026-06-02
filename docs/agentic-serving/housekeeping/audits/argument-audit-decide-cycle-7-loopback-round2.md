# Argument Audit Report — Round 2

**Audited documents:**
- `docs/agentic-serving/decisions/adr-033-layer-a-loop-driver-multi-turn-agentic-surface.md` (revised after R1)
- `docs/agentic-serving/decisions/adr-034-client-tool-action-terminal-artifact-bridge.md` (revised after R1)
- `docs/agentic-serving/decisions/adr-027-framework-driven-dispatch-pipeline.md` (partial-update header only)

**Source material:**
- `docs/agentic-serving/essays/research-logs/cycle-7-spike-tau-upsilon-decide-entry-probes.md` (Spikes τ/τ′/υ)
- `docs/agentic-serving/essays/research-logs/006b-client-tool-action-terminal.md` (Spikes π/ρ/σ; F-ρ.1)
- `docs/agentic-serving/essays/essay-outline-006-cross-compatibility-routing-surface.md` (§C8 / Amendment B)
- `docs/agentic-serving/domain-model.md` (AS-9 scope boundary; AS-10; OQ #26; OQ #27; Amendment Log #15)

**Prior audit:** `docs/agentic-serving/housekeeping/audits/argument-audit-decide-cycle-7-loopback.md` (R1)

**Genre:** ADR set
**Date:** 2026-06-02

---

## Section 1: Argument Audit

### Summary

- **Genre:** ADR set
- **Argument chains re-verified:** 8 (same chains as R1; focus on whether R1 P2/P3 corrections resolved the gaps and whether corrections introduced new issues)
- **Issues found this round:** 0 P1, 1 P2, 1 P3
- **Pyramid coverage map:** N/A (ADR genre)
- **Expansion-fidelity findings:** N/A (ADR genre)

**R1 corrections applied — verification results:**

| R1 Finding | Correction applied | R2 verdict |
|---|---|---|
| P2-1 (discriminator marked as decided, not drafting-time) | Qualifier added to ADR-033 §Decision ¶1 marking discriminator as drafting-time design choice for ARCHITECT/BUILD, including the tool-capable-client edge case | Resolved |
| P2-2 (turn-count cost × axis-2 risk interaction unnamed) | "Compound-cost scenario" added to ADR-033 Consequences/Negative | Resolved |
| P2-3 (tool-mapping assigned to loop-driver as settled, no spike evidence) | Scope note added to ADR-034 §Decision ¶4 flagging multi-tool-type decisions as BUILD/PLAY work | Resolved |
| P3-1 (wrapper niche stated without "not tested") | "A niche no spike tested" added to ADR-033 Rejected/Wrapper | Resolved |
| P3-2 (deterministic-delivery conflates marshalling with fidelity) | ADR-034 Consequences/Positive rewritten to separate marshalling determinism from fidelity-at-scale | Resolved |
| F2-1 (τ′ stand-in status absent from Decision text) | Scope note added to ADR-033 §Decision ¶3 naming enforcement technique as unresolved ARCHITECT/BUILD design | Resolved |
| F2-2 (latency comparison scope) | Latency-comparison scope qualifier added to ADR-033 Rejected/Wrapper | Resolved |

---

### P1 — Must Fix

None.

The core inference chains remain logically sound after revision. Callee resolution is grounded in Spike υ with scope appropriately qualified. The Conditional Acceptance is correctly shaped (axis-1 conditionally validated under enforcement, axis-2 deferred to BUILD/PLAY). The artifact-bridge necessity follows from F-ρ.1 and ADR-025. The ADR-027 partial-update header is consistent with ADR-027's body. AS-9 and AS-10 are honored. No conclusion requires a premise the revised ADRs fail to establish.

---

### P2 — Should Fix

**P2-1 (carry-forward, partially resolved): The compound-cost scenario addition introduces a framing that slightly overreaches the evidence it draws from.**

- **Location:** ADR-033 §Consequences §Negative — newly added "Compound-cost scenario" paragraph.
- **Claim in the addition:** "if axis-2 validation forces a capable (frontier-tier) driver, the two cost dimensions multiply rather than add — a higher per-turn cost (capable driver on every turn) times the higher turn count single-action-per-turn enforcement produces... The worst case for the cost-distribution value proposition is 'capable driver required AND long horizon,' where both factors compound."
- **Assessment:** the substance of the compound-cost finding is correct and the addition resolves the R1 P2-2 gap — the interaction is now named. However, the new paragraph introduces the phrase "times the higher turn count" in a way that could be read as asserting that single-action-per-turn enforcement specifically multiplies turn count compared to the unconstrained driver on the same task. That is true for tasks the unconstrained driver would have batched (as τ showed), but it is not universally true: for tasks with genuine sequential dependencies the turn count under enforcement matches the logical minimum anyway. The framing should read as "may multiply" rather than stating the multiplication as unconditional.
- **Recommendation:** change "a higher per-turn cost ... times the higher turn count single-action-per-turn enforcement produces" to "a higher per-turn cost ... combined with the higher turn count single-action-per-turn enforcement produces on batchable tasks." This preserves the finding without overstating it.
- **Severity note:** this is a precision issue in a Consequences/Negative paragraph — it does not affect the Decision or Fitness Criteria. P2 is appropriate; it does not rise to P1.

---

### P3 — Consider

**P3-1 (new, minor): The ADR-033 §Decision ¶3 scope note on enforcement technique mentions three candidate mechanisms but does not indicate relative maturity or practicability.**

- **Location:** ADR-033 §Decision ¶3 parenthetical — "candidates are batch-truncation, a re-planning prompt that requests one action at a time, or a one-tool `tool_choice` constraint on the driver."
- **Observation:** all three candidates are listed as equivalent alternatives. Batch-truncation is what τ′ tested (the scratch proxy truncated the batch). The re-planning prompt and `tool_choice` constraint are untested in any probe. Naming all three without flagging which has empirical backing (even at scratch-proxy level) leaves ARCHITECT without a starting prior.
- **Recommendation:** consider noting "(batch-truncation is what τ′ used as its scratch proxy; the other two are untested)" to give ARCHITECT a differential starting point. Minor clarity improvement, not a logical gap.

---

### Re-verification of core chains after revision

**Callee resolution from υ (n=1).** ADR-033 §Decision ¶2 and the Rejected alternatives §Wrapper now both carry the latency-comparison scope qualifier identifying the measurement as made on a batchable two-write task under unconstrained conditions and noting that the per-turn directional advantage holds while absolute session numbers shift. The qualifier is substantive, not hand-waving. The callee resolution remains grounded.

**Grounded-loop Conditional Acceptance vs. OQ #27 two-axis framing.** ADR-033 §Decision ¶4 correctly characterizes the Conditional Acceptance as conditional on the framework-enforced single-action-per-turn constraint; axis-2 is designated a BUILD/PLAY validation target. The AS-9 scope-boundary annotation (OQ #27 authoritative version, domain-model Amendment Log #15) is honored: the ADR does not claim AS-9 extends to the loop-driver role.

**ADR-033 / ADR-034 split coherence.** ADR-033 decides when and which tool (loop-driver decision logic); ADR-034 specifies how the deliverable becomes a client-executed tool call (marshalling mechanism). The naming disambiguation note in ADR-034 (artifact-bridge vs. ADR-030 Bridge mechanism) remains in place and is accurate. The split is coherent.

**ADR-027 scoping consistency.** The partial-update header on ADR-027 correctly scopes the pipeline to single-turn non-tool-driven requests and layer-B generation. The header does not supersede ADR-027's body, which remains operative for its scoped surface. The scoping is consistent across ADR-033 §Relationship to ADR-027, ADR-034 §Relationship to prior ADRs, and the header.

**AS-9 / AS-10 honored.** AS-9 is honored by the Conditional Acceptance structure (single-decision-shaped scope limitation acknowledged, loop-driver extended conditionally only). AS-10 is honored by ADR-034 §Relationship to prior ADRs ¶4 (the terminal introduces no client-side opt-in signal; Spike ρ reaffirmed AS-10 on the tool-driven surface).

---

## Section 2: Framing Audit

### Re-flagging F3-1 and F3-2 (held for gate per R1 discipline)

Per the R1 dispatch: F3-1 (wrapper-residual-as-concession-vs-contingency) and F3-2 (deterministic-delivery framing) were NOT auto-corrected after R1; they were held for practitioner gate adjudication. Both are re-assessed here.

**F3-1 re-assessment.** ADR-033 §Rejected alternatives §Wrapper now explicitly states "a niche no spike tested" for the per-turn multi-capability composition case. The wrapper residual is accurately positioned as an untested niche, not a tested fallback. The R1 P3-1 correction (marking the niche as untested) partly addresses the framing concern. The remaining framing question — whether the wrapper should be positioned as a "named contingency if axis-2 fails" rather than a "residualized rejected alternative" — is a practitioner judgment call about future-cycle risk management posture. The current text records the wrapper as a residual and names frontier-tier as the fallback for axis-2 failure; there is no explicit "revert to wrapper" specification. This gap is still present and is a real practitioner-facing risk if axis-2 fails during BUILD/PLAY, but it is not a logical error in the current ADR. **F3-1 stands as a P3 framing note for the gate.**

**F3-2 re-assessment.** ADR-034 §Consequences/Positive was rewritten after R1 to separate marshalling determinism from fidelity-at-scale. The rewritten paragraph distinguishes: (a) the marshalling step is framework code, not an LLM generation — that is the determinism claim; (b) fidelity across large or structurally complex deliverables is BUILD scope, not established by the small-content spike evidence. The rewrite is substantive and resolves the R1 P3-2 finding. **F3-2 is resolved.**

---

### Question 1: New framing issues introduced by the R1 corrections?

No new framing issues were introduced by the corrections. The compound-cost scenario addition (R1 P2-2 → new ADR-033 text) is factually accurate in its core claim; the P2-1 finding above flags a precision issue in one phrase, not a framing shift. The other additions (scope notes, qualifiers) are narrowly targeted and do not introduce new framings.

---

### Question 2: Do the prior framing observations still stand?

**F2-1 (τ′ enforcement stand-in status).** Resolved by the R1 correction. ADR-033 §Decision ¶3 now carries the parenthetical explicitly naming the scratch-proxy stand-in and naming the enforcement technique as ARCHITECT/BUILD design. The source material's scope-of-claim (τ′ enforcement is a stand-in) is now represented in the Decision text.

**F2-2 (latency comparison scope).** Resolved by the R1 correction. ADR-033 §Rejected alternatives §Wrapper now carries the qualifier identifying the ~3× figure as measured on a batchable two-write task, and noting that the per-generation ratio is the stable comparison while absolute session numbers shift with horizon.

**F3-1 (wrapper residual posture).** Stands as P3, as assessed above.

---

### Question 3: What would change if the dominant framing were inverted?

The R1 framing-inversion analysis (callee as higher-upside/higher-risk vs. wrapper as lower-upside/lower-risk) is unchanged by the R1 corrections. The compound-cost scenario addition actually strengthens the inverted framing's salience: it makes visible the scenario under which the callee's cost advantage collapses entirely. Practitioners reading the Consequences/Negative section now have a clearer picture of the inverted-framing scenario's worst case. The framing itself (callee as the evidence-backed primary choice) remains appropriate given the evidence — the inversion is a risk-management lens, not a finding that the decision was wrong.

---

### Framing Issues

**P1 — Consequential omissions:**

None. The R1 corrections addressed both primary disclosure gaps (τ′ stand-in status; latency comparison scope). No finding in the source material is omitted in a way that would change the ADRs' conclusions.

---

**P2 — Underrepresented alternatives:**

None new this round. F2-1 and F2-2 are resolved.

---

**P3 — Minor framing choices:**

**F3-1 (carried from R1).** The wrapper-residual's positioning as a residualized rejected alternative rather than a live contingency specification is a practitioner framing choice the gate should adjudicate. The risk: if BUILD/PLAY axis-2 validation fails, the practitioner needs to know what "revert to wrapper" means architecturally — that specification is absent. The ADR is not logically wrong in its current form, but a future-cycle BUILD practitioner facing axis-2 failure would benefit from a named contingency posture rather than a rejected-alternative footnote.

- **Location:** ADR-033 §Rejected alternatives §Wrapper, closing sentence; ADR-033 §Consequences §Neutral ¶2.
- **Recommendation for gate consideration:** either add a "contingency specification" note to the wrapper section (naming what "revert to wrapper" would require architecturally: the loop-driver remains, per-turn generation invokes the full pipeline, the synthesizer-redundancy risk is accepted), or explicitly document that frontier-tier driver is the preferred axis-2 fallback and wrapper reversion is the second-order fallback. The current text implies the first but does not state it.

---

## Convergence-Saturation Signal (ADR-094)

**Convergence-Saturation Signal:** TRIGGERED

- Round number: R2
- P1 count this round: 0 (Section 1: 0; Section 2: 0)
- P2 count this round (new, non-carry-over): 1 (Section 1 P2-1 — precision issue in the newly-added compound-cost sentence; no Section 2 P2 findings)
- New framings or claim-scope expansions: none. The compound-cost addition does not introduce a new framing — it extends the scope of an R1-flagged gap. No new warrants or claim-scope characterizations were surfaced that prior rounds did not name.
- Recommendation: STOP at R2

*All three signal conditions are met: P1 count = 0; new P2 count ≤ 1 (exactly 1, precision-class); no new framings. The one remaining P2 is a precision correction in a Consequences/Negative paragraph, not a structural gap. F3-1 carries as a P3 gate note; that is within normal gate discipline and does not prevent stopping. The ADR set is logically sound after R1 corrections.*
