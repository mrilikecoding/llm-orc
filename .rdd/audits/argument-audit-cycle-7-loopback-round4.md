# Argument Audit Report

**Audited document:** `docs/agentic-serving/essays/essay-outline-006-cross-compatibility-routing-surface.md`
**Source material:** `docs/agentic-serving/essays/research-logs/research-log.md`
**Genre:** Essay-Outline
**Date:** 2026-05-30
**Round:** R4 — single-purpose re-audit (verify round-3 corrections; omit convergence-saturation verdict line per ADR-094)

---

## Section 1: Argument Audit

### Summary

- **Genre:** Essay-Outline
- **Argument chains mapped:** 8 (C1–C8)
- **Issues found:** 1 (P3)
- **Pyramid coverage map:** included
- **Expansion-fidelity findings:** P1: 0, P2: 0, P3: 1

---

### Pyramid Coverage Map

| Abstract Conclusion | Argument-Graph Nodes | Body Sections | References Cited |
|---------------------|----------------------|---------------|------------------|
| C1. NL-routing fraction ≈ zero | C1, W1.1, W1.2, E1.1.1–E1.1.3, E1.2.1–E1.2.3 | §2 (C1) | cycle-6-spike-gamma, cycle-6-field-notes-play |
| C2. tool_choice not free baseline (D0) | C2, W2.1, W2.2, W2.3, E2.1.1–E2.1.2, E2.2.1–E2.2.2, E2.3.1–E2.3.2 | §3 (C2) | cycle-7-spike-kappa, cycle-7-spike-lambda, cycle-7-oq-19-build-complexity-comparison |
| C3. Routing-planner ensemble is PRIMARY mechanism | C3, W3.1, W3.2, W3.3, E3.1.1–E3.1.2, E3.2.1–E3.2.3, E3.3.1–E3.3.3 | §4 (C3) | cycle-7-spike-zeta, research-design-review-cycle-7, cycle-7-oq-20-population-a-timeouts |
| C4. Framework-driven composition continuation required | C4, W4.1, W4.2, W4.3, E4.1.1–E4.1.3, E4.2.1–E4.2.2, E4.3.1–E4.3.3 | §5 (C4) | cycle-7-spike-lambda, cycle-6-field-notes-play, adr-025, cycle-6-spike-delta |
| C5. I/O enforcement at ensemble-execution paths | C5, W5.1, W5.2, W5.3, E5.1.1–E5.1.3, E5.2.1–E5.2.3, E5.3.1–E5.3.3 | §6 (C5) | cycle-6-field-notes-play, cycle-6-spike-delta, research-design-review-cycle-7, adr-014, adr-018 |
| C6. Fallback in tension with value; capability discovery first-order | C6, W6.1, W6.2, W6.3, E6.1.1, E6.2.1–E6.2.3, E6.3.1–E6.3.5 | §7 (C6) | agentic-serving-cycle-status, cycle-6-field-notes-play, cycle-7-oq-18-cost-distribution-validation |
| C7. ADR-027 as PRIMARY; hybrid as conditional alternative | C7, W7.1, W7.2, W7.3, W7.4, E7.1.1–E7.1.2, E7.2.1–E7.2.3, E7.3.1–E7.3.3, E7.4.1–E7.4.2 | §8 (C7; META scope) | cycle-7-spike-kappa, cycle-7-oq-19-build-complexity-comparison, domain-model-as9, cycle-7-spike-mu-confabulation-generalization, cycle-6-spike-delta, cycle-6-field-notes-play, cycle-7-spike-lambda |
| C8. Client-tool-action terminal necessary | C8, W8.1, W8.2, W8.3, W8.4, W8.5, W8.6, E8.1.1–E8.1.2, E8.2.1, E8.3.1–E8.3.2, E8.4.1–E8.4.2, E8.5.1, E8.6.1–E8.6.3 | §9 (C8) | research-log-loopback, opencode, adr-025 |

**META-anchored sections:** §1 (Methodology preamble), §8 (Cycle 7 recommended architecture — `C7; META scope`)

**Orphan body sections:** none.
**Argument-Graph nodes with no body anchor:** none.

---

### Expansion-Fidelity Findings

**P1 findings (pyramid violations):** none.

**P2 findings (weak expansion or META misclassification):** none.

**P3 findings (minor coverage gaps):**

- **Section 9 heading anchor format.** The section heading reads `### Section 9: The client-tool-action terminal — BUILD → RESEARCH loop-back (C8)`. The `(C8)` appears mid-title rather than as a clean terminal parenthetical anchoring the section to the graph node. All other developmental sections follow the `(Cx)` parenthetical-at-end convention (Section 2 ends `(C1)`, Section 3 ends `(C2)`, etc.). Section 9's anchor identifier is unambiguous — it clearly maps to C8 — but the formatting is inconsistent with the rest of the document. No functional impact; cosmetic.

---

### Round-3 Corrections Verification

**P2-1 (E8.6.3 thin body development):** CONFIRMED RESOLVED. E8.6.3 is now a standalone bullet: `"Discriminating evidence appropriate to BUILD-phase validation against the real built terminal, not another stand-in spike..."` It is distinct from E8.6.2 (grounded-loop hypothesis) and the separation is clean. The Section 9 body also carries a parallel dedicated EVIDENCE bullet: `"Why no further stand-in spiking is warranted before DECIDE/ARCHITECT."` The two occurrences are complementary rather than redundant — the Argument-Graph node (E8.6.3) states the conclusion; the Citation-Embedded Outline EVIDENCE bullet states the reasoning. No remaining thinness issue.

**P2-2 (AS-9 reopening condition understated):** CONFIRMED RESOLVED. E8.6.2 now reads: `"AS-9's scope reopens specifically around whether the grounded-vs-ungrounded distinction is the correct framing of the structural property — not whether the bounded-role pattern holds generally."` The parenthetical precisely scopes the reopening to the framing of the distinction, not to AS-9 broadly. ARCHITECT will know the reopening question is "is grounded-vs-ungrounded the right axis?" not "is the bounded-role pattern valid at all?" The correction is well-targeted and does not introduce new ambiguity.

**P2-F1 (candidate (1) asymmetric framing depth):** CONFIRMED RESOLVED. E8.6.1 is now structured as a sub-bulleted list with a distinct `*AS-9 relevance:*` note for each of (1), (2), and (3). Candidate (2)'s note explicitly names the scope-expansion-outside-Spike-ζ-task-shape observation: `"the scope expansion from single dispatch-plan JSON to per-turn agentic-action decisions takes the planner outside the task shape Spike ζ validated (request → {action, ensemble, rationale}), so AS-9's empirical basis would need re-establishment at the new task shape."` Candidate (3)'s note names the design-from-scratch advantage: `"designed-from-scratch as a bounded role under AS-9; requires the most BUILD investment but avoids re-litigating AS-9's empirical basis on an existing component."` The treatment is structurally parallel. The ordering (1) → (2) → (3) still gives σ-tested candidate (1) priority of appearance, but each candidate's distinctive AS-9 profile is now legible independently, which was the goal of the correction. No first-candidate salience tilt remains.

**P3-1 (Amendment B propagation line W8.1–W8.5 vs. W8.1–W8.6):** CONFIRMED RESOLVED. The "Propagation applied 2026-05-24" summary in Amendment B now reads: `"new C8 sub-tree W8.1–W8.6 in the Argument-Graph (W8.6 + the layer-A seat-filler candidates + grounded-loop hypothesis added in a post-gate refinement after round 2..."` The range is correct and includes a brief explanatory note about the post-gate addition. Audit trail is clean.

**P3-F1 (σ's success → σ.1's success):** CONFIRMED RESOLVED. All occurrences of the grounded-loop hypothesis in the Argument-Graph now reference `σ.1` specifically: `"if σ.1's success and Cycle 6 PLAY note 22's failure reflect..."` The one remaining use of bare `σ` in E8.4.1 (`"Spike σ confirms layer A is required and is fillable by a local model on a short grounded task (σ.1 used qwen3:14b directly as the seat-filler)"`) is accurate — `Spike σ` refers to the spike family and `σ.1` identifies the specific sub-spike. The distinction is now precise throughout.

---

### P1 — Must Fix

None.

---

### P2 — Should Fix

None.

---

### P3 — Consider

**P3-1 (Section 9 heading anchor format — cosmetic)**

- **Location:** Line 266, Citation-Embedded Outline heading `### Section 9: The client-tool-action terminal — BUILD → RESEARCH loop-back (C8)`
- **Claim:** The section maps to C8. This is unambiguous.
- **Evidence gap:** The other seven developmental sections place the Argument-Graph anchor as a parenthetical terminating the heading (`### Section 2: ... (C1)`, etc.). Section 9 embeds `(C8)` within the subtitle after a dash, which technically satisfies the anchoring requirement but departs from the established convention.
- **Recommendation:** Reformat to `### Section 9: The client-tool-action terminal — BUILD → RESEARCH loop-back (C8)` is already close; the only question is whether the dash-parenthetical read as `(title — subtitle (anchor))` is clear enough. Optionally move the anchor to the end: `### Section 9: The client-tool-action terminal (C8)` with the loop-back descriptor absorbed into the section body or dropped from the heading. Purely cosmetic; does not affect the argument.

---

## Section 2: Framing Audit

### Question 1: What alternative framings did the evidence support?

The source material is the loop-back research log, which documents Spikes π/ρ/σ on the client-tool-action terminal question. The primary document's C8 sub-tree accurately incorporates all three spikes. Three possible alternative framings merit examination:

**Alt-1: "Layer A is the primary finding; the terminal is a corollary."** The research log presents the layer-A/B distinction as emerging from Spike σ and characterizes it as the "distinct architectural role ADR-027 lacks." The primary document orders it the other way: the terminal is the conclusion (C8) and layer A is a structural gap the terminal surfaces (W8.4). Both orderings are defensible; the research log's framing gives slightly more weight to the architectural implication (a missing role is deeper than a missing terminal). The primary document's ordering keeps C8's narrative coherent — necessity → mechanism → composition → structural gaps — so the choice is appropriate.

**Alt-2: "The bounded routing planner (C1 finding) is the structural reason the terminal works; C8 depends on this."** The research log records that Spike ρ's planner-driven delegation worked specifically because the planner routes on request content (AS-10), not on the client's declared tools, so the C1 suppression did not recur. The primary document names this (E8.2.1) but the dependency chain — C1 established the suppression problem → AS-10 dissolves it for the planner → this is why the planner-driven terminal is viable — is present but not foregrounded as a framing choice. The current framing treats AS-10 as an explanatory aside rather than a load-bearing dependency. This is a reasonable editorial choice; ARCHITECT needs to know C8 depends on AS-10 holding, which the document does note.

**Alt-3: "The σ.2 integrated-pattern finding is stronger than the document implies; it tests the full north-star loop, not merely mechanism composition."** The research log records that σ.2 drove layer-A decisions while delegating layer-B generation to the `code-generator` ensemble, and the ensemble-generated test passed. The primary document frames this as "the integrated north-star pattern composes at the mechanism level" (scoped in E8.3.2 with caveats about stand-in server and 2-turn batching). The framing is accurate but conservative; the research log emphasizes that this is the "north-star pattern validated end-to-end." Given the scoping caveats (stand-in server, headless mode, 2 turns), the document's conservatism is defensible.

### Question 2: What truths were available but not featured?

The research log contains several observations that are present in the primary document's scope qualifications but are not foregrounded as standalone findings:

**Available but backgrounded: σ.2's single-turn batching.** The research log explicitly notes that σ.2's driver "batched all three actions in one planning turn (so this was 2 turns: plan + finish, not a long decide-act-observe chain)." The primary document references this in C8's scope qualification (Section 9 SCOPE QUALIFICATION) and in E8.3.2's parenthetical, but not in the Abstract or C8 conclusion. For a DECIDE/ARCHITECT reader skimming the Abstract, the "integrated north-star pattern composes end-to-end" language could overstate the validation breadth. The scope qualification in Section 9 addresses this; the Abstract does not. This is a marginal framing concern given the explicit qualification exists in the body.

**Available but not surfaced: OpenCode's native `skill` and `task` tools.** Spike π Phase 0 observed that OpenCode declares native `skill` and `task` (subagent) tools in addition to the filesystem tools the document focuses on. The research log notes this as "noted, not tested here" and flags it as "north-star-relevant." The primary document repeats this note in Section 9's SCOPE QUALIFICATION. This is correctly scoped as an ARCHITECT observation; it does not affect C8's argument but ARCHITECT needs it. It is available and noted; its backgrounding is appropriate.

**Available but not surfaced: the σ.2 delegation mechanism used a stand-in, not a production ensemble.** The research log notes that σ.2 used `spike-pi-code-generator` as a stand-in for the production `code-generator` because of F-ρ.1 (artifact substrate). The primary document records F-ρ.1 explicitly and the σ.2 scope qualification names the stand-in. The gap between the stand-in test and the production path is the most consequential unresolved item; it is correctly flagged throughout but it is worth confirming that ARCHITECT receives it clearly. Section 9's CLAIM block, SCOPE QUALIFICATION, and E8.3.2's parenthetical all reference it; the Abstract mentions F-ρ.1 by name. Coverage is adequate.

### Question 3: What would change if the dominant framing were inverted?

The dominant framing in C8 is: "the terminal is necessary; the architecture requires it; the north-star is achievable with the right BUILD work." The inverted framing: "the north-star was always more complex than ADR-027 acknowledged; the terminal is a symptom of an architectural mismatch between the pipeline design and the agentic-loop interaction model."

Under the inverted framing:
- The layer-A gap becomes not a discovered omission but an expected consequence of designing a pipeline (batch mode) for a use case that is inherently conversational (turn mode). The "ADR-027 is incomplete, not wrong" framing would soften to "ADR-027 solved a different problem than the north-star requires."
- The grounded-loop hypothesis becomes less important; even if grounding works, a pipeline designed for text-terminal responses will need significant restructuring to participate in a multi-turn agentic loop, and the needed restructuring may challenge ADR-027's philosophy.
- The two structural gaps (layer A + F-ρ.1) would be presented not as gaps to fill but as indicators that the architecture needs a different model at its center.

The primary document pre-empts part of this inversion: W8.4 explicitly characterizes the pipeline as "structurally under-specified for the multi-turn agentic behavior the north-star requires, not merely missing a stage." This is honest. The inverted framing is acknowledged at the level of the gap characterization but not at the level of ADR-027's appropriateness as the base design. A DECIDE reader who holds the inverted framing might question whether to extend ADR-027 or redesign around a conversational loop model from the start; the document treats this as DECIDE/ARCHITECT's choice (W8.6) without advocating for either direction.

This is a defensible editorial choice. The document's position is that the pipeline philosophy is sound (deterministic framework wrapping beats orchestrator-LLM chaining) and the terminal/layer-A additions are extensions of that philosophy, not a refutation of it. The inverted framing would want the document to acknowledge more strongly that the pipeline model and the agentic-loop model are architecturally distinct and that the north-star may require choosing one over the other rather than extending the pipeline.

### Framing Issues

**No P1 findings.**

**No P2 findings.**

**P3-F1 (Abstract C8 sentence underweights the "incomplete, not wrong" framing vs. the structural-gap framing)**

- **Location:** Abstract, C8 sentence: `"ADR-027 is incomplete, not wrong; the fix aligns with its philosophy..."` vs. W8.4 in the Argument-Graph which characterizes the pipeline as "structurally under-specified for the multi-turn agentic behavior the north-star requires, not merely missing a stage."
- **Observation:** The Argument-Graph uses the more specific and honest formulation ("structurally under-specified... not merely missing a stage") while the Abstract uses the softer "incomplete, not wrong." The Abstract formulation is accurate but the Argument-Graph version carries more information for an ARCHITECT reader who needs to understand the depth of the gap. This is a minor calibration point, not a logical inconsistency; the body sections elaborate correctly.
- **Recommendation:** Consider aligning the Abstract language to mirror the Argument-Graph's "structurally under-specified" characterization, or add a brief qualifier: "ADR-027 is structurally under-specified for multi-turn agentic behavior (not merely missing a terminal), but the fix aligns with its philosophy." This keeps the claim honest without overstating the architectural challenge.

---

*Single-purpose re-audit dispatched per ADR-094 re-audit-after-revision rule. Convergence-Saturation Signal verdict line omitted.*
