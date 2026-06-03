# Argument Audit Report — R3 Scoped Verification

**Audited document:** docs/agentic-serving/decisions/adr-035-client-tool-deliverable-form-contract.md
**Source material (reference, for consistency checks only):**
- docs/agentic-serving/decisions/adr-034-client-tool-action-terminal-artifact-bridge.md
- docs/agentic-serving/decisions/adr-033-layer-a-loop-driver-multi-turn-agentic-surface.md
**Prior audits:** argument-audit-decide-cycle-7-loopback2.md (R1), argument-audit-decide-cycle-7-loopback2-round2.md (R2)
**Genre:** ADR
**Date:** 2026-06-03

---

## Scope Declaration

This is a scoped R3 verification of four gate-derived additions only. Per the dispatch brief, R2 converged (signal TRIGGERED; 0 P1, 1 new P2 which has since been corrected). The full argument structure is not re-derived; the audit verifies that the four additions introduce no new logical gaps, overreach, or contradictions with the existing audited body and with ADR-033/034.

The R2 P2-1 finding (Provenance check parenthetical "corrects Spike β premise" contradicting the body) is verified as corrected first, then each addition is assessed.

---

## Section 1: Argument Audit

### Summary

- **Genre:** ADR
- **Round:** R3 — scoped verification of four gate-derived additions
- **Scope:** Addition 1 (Seam framing paragraph), Addition 2 (Decision 4 revised), Addition 3 (Why a hard form-guarantee is neither available nor required), Addition 4 (Conditional Acceptance revisions)
- **R2 carry-forward items:** NEW-P2-1 (Provenance check parenthetical); carry-over P3-3 (terminology interchangeability) — see verification below
- **Issues found:** 0 P1, 0 P2, 1 P3 (carry-over P3-3 — still present in new additions; not a new finding)
- **Pyramid coverage map:** N/A
- **Expansion-fidelity findings:** N/A

---

### R2 P2-1 Correction Verification

**R2 finding:** The Provenance check bullet for `default_task` inert still used the parenthetical "(corrects Spike β premise)" after the ADR body had been correctly revised to say "refines ... without disturbing" at P2-2.

**Verification:** Corrected. The current Provenance check reads:

> "`default_task` inert; dispatch input is the only contract surface (refines Spike β's mechanism; its headline survives): Spike φ Part 3 grep + path trace (driver)."

The parenthetical now matches the ADR body's framing. The inconsistency R2 identified is resolved. **VERIFIED.**

---

### Addition 1: Seam Framing Paragraph

**Location:** Context §"Seam framing (practitioner, DECIDE gate)"

**Claims to verify:**
1. Form seam is a specific cost of delegation architecture; absent in single-model flows.
2. ADR-034 restored the execution-model half of parity; ADR-035 restores the generation-form half.
3. Semantic-coherence seam is distinct and NOT addressed by this ADR.
4. "Parity of mechanism is claimed; parity of content quality is not."

**Verification:**

Claim 1 (form seam as delegation cost) is sound. The single-model case — where the deciding model also generates the content — gets destination-awareness implicitly, in-context. The delegation architecture separates decider from producer across a dispatch seam, making implicit destination-awareness unavailable. This is the structural logic for why the form contract is required in this architecture and not in the single-model alternative. No premise is missing.

Claim 2 (two-halves framing) is accurate and internally consistent. ADR-034 Decision 1 establishes that "the client executes the tool itself," grounding the execution-model parity claim. ADR-035 Decision 1 establishes the form directive, grounding the generation-form parity claim. The ADR-034 Consequences §Positive states "Parity is achieved" — this refers to the execution-model parity ADR-034 was scoped to address, not content-form parity, so the seam framing's "half of parity" characterization is not contradicted by ADR-034's own conclusion language; the two documents are using "parity" at different levels of abstraction, and the seam paragraph makes this hierarchy explicit.

Claim 3 (semantic seam scope exclusion) is consistently maintained. The paragraph names where the semantic seam lives: "the seat-filler's dispatch-input composition (axis-2 territory, OQ #27) and ensemble quality (declared orthogonal at DISCOVER)." The Conditional Acceptance section also separately maintains the scope exclusion: FC-51 `TurnDecision` diagnostics "distinguishes a wrong-*form* deliverable (this contract) from a wrong-*action* turn (driver/split) — and, per the seam framing above, from a wrong-*content* turn (the semantic-coherence seam this ADR does not address)." The Decision itself says nothing about semantic coherence. The scope boundary is consistently maintained across all three locations.

Claim 4 (mechanism vs. content quality parity) is scoped correctly and does not overreach. The ADR's n=4 compliance evidence (Spike χ) covers bare-form production, not content quality relative to a frontier model. The sentence correctly limits the parity claim to mechanism.

**Verdict: No issues. Internally consistent, consistent with Decision and Conditional Acceptance, scope boundary maintained.**

---

### Addition 2: Decision 4 Revised

**Location:** Decision §4

**Claims to verify:**
1. Backstop's escalated form is detect-and-refuse, not extract — consistent with the rejected bridge-side-shaper alternative.
2. Escalation sequence is consistent with Conditional Acceptance escalation order.

**Verification:**

Decision 4's current text distinguishes two backstop tiers: (a) the conservative normalization safety net ("strip a single enclosing code fence if one slips through") and (b) the escalated form ("a fail-safe detection gate: when the marshalled deliverable is clearly non-bare... the bridge refuses to emit the tool call and degrades to a dispatch-failure completion"). The text explicitly states: "Detection-and-refusal only has to *recognize* a clearly-wrong deliverable; it never attempts heuristic extraction from multi-fence output (Spike χ F-χ.1 — that path is fragile)."

The rejected bridge-side-shaper alternative is rejected as the *primary contract* (heuristic extraction from unconstrained output). Decision 4 retains a conservative normalization (strip a single fence) as a safety net — a much narrower operation than the rejected extraction approach, operating only on an edge-case slip-through rather than as the primary shaping mechanism. The escalated form (detect-and-refuse) is explicitly distinguished from extraction. The refusal-vs.-extraction distinction holds cleanly.

The Conditional Acceptance escalation order lists: "(1) escalate the backstop to its detect-and-refuse gate (decision 4); (2) `output_schema`-as-enforcement reject-and-retry for the client-tool path; (3) a frontier seat-filler (ADR-033 §6b)." Decision 4 is correctly named as escalation step (1). Consistent.

**Verdict: No issues. Refusal-vs.-extraction distinction holds; escalation sequence is internally consistent.**

---

### Addition 3: "Why a hard form-guarantee is neither available nor required"

**Location:** Decision §"Why a hard form-guarantee is neither available nor required (DECIDE-gate exchange)"

**Claims to verify:**
1. Form not structurally enforceable unlike ADR-033's truncation mechanism.
2. Schema-validation of bare code inherits shaper heuristics.
3. `submit_file` guarantees slot not form.
4. "Not required" argument is honestly scoped to surfaces with client-side execution affordances.

**Verification:**

Claim 1 (form not structurally enforceable): ADR-033 Decision 3 describes single-action-per-turn enforcement as a mechanical operation (the framework "executes one client tool call per turn, returns its result to the loop-driver, and forces re-planning before the next action"). The enforcement is purely structural: the framework truncates or constrains what the driver can emit, independent of content. Enforcing bare-code form would require the framework to mechanically convert markdown-with-fences to bare code — the same operation Spike χ F-χ.1 shows has no robust general rule. The comparison is sound.

Claim 2 (schema-validation inherits heuristics): "Is this bare code?" is not representable as a clean JSON Schema rule — the validation itself would require the same fence-detection logic as the rejected shaper. The claim is sound. The rejected `output_schema`-as-enforcement alternative is rejected on similar grounds ("heavier than the evidence warrants" and "Candidate B-strong shape ADR-024 rejected for enforcing 'at a layer that isn't the source'"). The new paragraph adds a different angle (the validation logic problem) that is consistent with, and not in tension with, the existing rejection reasoning.

Claim 3 (`submit_file` guarantees slot not form): The rejected static-ensemble-coupling alternative is described in the Rejected alternatives section as coupling the ensemble to file-production, which erodes ADR-025 reusability. The new paragraph adds the observation that `submit_file` guarantees a tool *slot* but not the *content form* within that slot (fences can appear inside the argument). This is accurate — a structured tool call constrains the argument slot name, not the argument value's internal formatting. Consistent with the rejection reasoning and adds a precise mechanical observation.

Claim 4 ("not required" scoping — client-side affordances): The paragraph states: "A wrong-form deliverable surfaces as a rejectable diff; it does not silently corrupt the workspace (that was the rejected co-located-write shape)." And: "on a surface without client-side execution affordances, the detection-gate backstop (decision 4) would be warranted from the start." The bounded-failure-cost argument rests on ADR-034's execution model (Decision 1: "the surface never writes to the client's filesystem behind the client's back"; Consequences §Positive: "executes and observes its own tool calls, with permission gates, diffs, and tool-result feedback intact"). The scoping is honest: "not required" is conditioned on the presence of client-side affordances, and the carve-out for surfaces without them is explicit. The ADR's overall scope is the tool-driven multi-turn surface (ADR-033 Decision 1 discriminator: "When a chat-completions request carries client `tools[]`..."). The argument does not overgeneralize.

One observation at P3 severity: the "not required" argument implicitly relies on the client *actually using* the permission gate and diff affordances — a compliance assumption on the client side, not just a structural affordance. The argument is still honest because the rejection of the co-located-write shape in ADR-034 explicitly turns on the absence of the permission gate, and any tool-driven client that bypasses its own permission gate would be behaving outside the surface's operating assumptions. This is a minor P3 observation (noted below), not a P1 or P2 gap.

**Verdict: Claims are sound and honestly scoped. One P3 observation (see below).**

---

### Addition 4: Conditional Acceptance Revisions

**Location:** Decision §"Conditional Acceptance (ADR-097)" — new delegation-precondition bullet + escalation (1) rewording + wrong-content-turn mention

**Claims to verify:**
1. New delegation-precondition bullet is consistent with FC-51's described discrimination capability and ADR-033 §6b.
2. Escalation (1) rewording to detect-and-refuse gate is consistent with Decision 4.
3. Wrong-content-turn mention is consistent with seam framing and within scope.

**Verification:**

Delegation-precondition bullet: "Precondition, not an independent target: the form contract exercises only when the Loop Driver actually delegates (`invoke_ensemble`). Delegation reliability (Finding B's resolution, WP-LB-G) has held for one real-client run; PLAY validation of the form contract is gated on delegation continuing to fire across prompts and clients." This is logically correct: if the loop-driver never fires `invoke_ensemble`, there is no dispatch input to inject the form directive into, so the form contract is never exercised. The precondition relationship is structurally sound. The Finding B reference (WP-LB-G delegation validated) grounds the claim that delegation reliability has been established for one real-client run without asserting it is fully settled — appropriately qualified.

FC-51 discrimination: The bullet that closes Conditional Acceptance states FC-51 instrumentation distinguishes wrong-*form* from wrong-*action* from wrong-*content* turns. FC-51 is referenced in ADR-033 §6b's context through the PLAY validation mechanism (the axis-2 validation is north-star-shaped, driven by the real-client run). ADR-033 §6b does not name FC-51 explicitly — FC-51 is an instrumentation artifact at the FC-51 `TurnDecision` level, which is within the ADR-033 architecture (loop-driver decisions are `TurnDecision` typed). The reference is consistent with ADR-033 §6b's validation apparatus. The three-way discrimination (form / action / content) maps cleanly to the three seams the Context paragraph names, so the FC-51 description is internally consistent with the seam framing addition.

Escalation (1) rewording: Now reads "(1) escalate the backstop to its detect-and-refuse gate (decision 4)." Direct reference to Decision 4 by section number. Fully consistent with Decision 4's revised text.

Wrong-content-turn mention: "FC-51 `TurnDecision` instrumentation distinguishes a wrong-*form* deliverable (this contract) from a wrong-*action* turn (driver/split) — and, per the seam framing above, from a wrong-*content* turn (the semantic-coherence seam this ADR does not address)." The cross-reference to the seam framing is explicit and correct. The wrong-content-turn is the semantic-coherence seam the Context paragraph explicitly excludes from this ADR's scope. This sentence correctly characterizes the FC-51 diagnostic as capable of distinguishing all three failure modes without claiming this ADR addresses the third. Consistent and within scope.

**Verdict: No issues. All three revised elements are internally consistent, consistent with Decision 4, and consistent with the seam framing.**

---

### Provenance Check Lines

Three new provenance lines are added for the gate-derived content:

1. **Seam framing:** "practitioner framing at the DECIDE gate (driver — user framing at gate, 2026-06-03). The semantic-coherence-seam scope boundary and its FC-51/PLAY disposition compose the practitioner's framing with ADR-033 §6b and the DISCOVER ensemble-quality-orthogonal declaration." Attribution is accurate — the two-halves-of-parity framing is practitioner-originated; the scope-boundary elaboration (OQ #27 / ensemble quality orthogonal) is drawn from prior ADRs and the domain model, correctly characterized as composition rather than a new driver finding.

2. **"Hard form-guarantee neither available nor required":** "DECIDE-gate exchange (driver — the practitioner's guarantee-justification challenge + the agent's mechanism analysis). The bounded-failure-cost argument rests on ADR-034's execution-model decision (driver, prior ADR). The detect-and-refuse escalated backstop is gate-exchange-derived synthesis." Attribution is accurate — the guarantee challenge originated at the gate, the mechanism analysis is the agent's response at the gate, and ADR-034's execution model is the load-bearing prior ADR for the bounded-failure-cost claim. The provenance correctly distinguishes the driver (gate exchange) from the evidentiary anchor (ADR-034).

3. **Conditional Acceptance escalation + delegation precondition:** These are covered under the existing Conditional Acceptance provenance bullet ("drafting-time synthesis applying ADR-097's grounding filter... composing ADR-033 §6b axis-2 escalation levers") and the seam framing bullet. The detect-and-refuse escalated backstop appears in the guarantee-section bullet as "gate-exchange-derived synthesis." The delegation-precondition bullet in Conditional Acceptance is consistent with the Seam framing provenance (FC-51/PLAY disposition composes ADR-033 §6b). The coverage is adequate; a separate provenance bullet for the delegation-precondition would be tighter but the current attribution is not misleading.

**Verdict: Provenance attribution is fair and accurate. No misattributions.**

---

### Carry-Over P3-3 Status

P3-3 (terminology interchangeability: "marshalling boundary" and "Loop Driver / Client-Tool-Action Terminal" used for the same entity across sections) was not corrected per the R2 dispatch brief scope. The new additions use "marshalling boundary" in the seam framing paragraph ("the loop-driver's decision side and the 'producer' side... the marshalling boundary composes an output-form directive"). This is the same minor terminology inconsistency — "marshalling boundary" is a logical-role label (where the form directive is composed), while "Loop Driver / Client-Tool-Action Terminal (ADR-033/034)" is the architectural-entity label. The carry-over is still present in the new additions. Not a new finding; still P3.

---

### P1 — Must Fix

None.

---

### P2 — Should Fix

None. The R2 P2-1 is verified corrected. The four gate-derived additions introduce no new P2 issues.

---

### P3 — Consider

**P3-1 (new): "Not required" argument — client-side affordance compliance assumption**

- **Location:** Decision §"Why a hard form-guarantee is neither available nor required" — the "Not required" paragraph.
- **Claim:** "A wrong-form deliverable surfaces as a rejectable diff; it does not silently corrupt the workspace."
- **Gap:** The argument correctly identifies that the client's permission gate and diff affordances are structurally present (ADR-034 grounds this). It implicitly assumes the client will use them rather than auto-accepting. The auto-accept case (a client configured to accept all diffs without review) is outside the surface's stated operating assumptions, so this is not a gap in the argument — but a reader unfamiliar with the tool-driven client model might not see this. A single parenthetical clarifying that the argument holds "for clients operating with their permission gate active" would close any ambiguity.
- **Recommendation:** Consider adding a short qualifier: "for a client operating with its permission gate active (the operating assumption of the tool-driven surface), a wrong-form deliverable surfaces as a rejectable diff." Optional — the ADR's scope (ADR-033 discriminator: tool-driven clients) implies this, but making it explicit would pre-empt the question.
- **Severity note:** This is a minor clarity gap in an otherwise sound argument, not a logical error. The bounded-failure-cost claim is valid within the surface's operating assumptions; the assumptions are defined by the ADR scope.

**Carry-over P3-3 (from R1/R2, unchanged):** "Marshalling boundary" vs. "Loop Driver / Client-Tool-Action Terminal" terminology interchangeability. Still present in the new seam framing paragraph. Not corrected per scope; not a new finding.

---

## Section 2: Framing Audit

This R3 is a scoped verification of gate-derived additions. The framing audit is correspondingly scoped: do the additions introduce new framing choices that exclude evidence available in the source material, or do they misrepresent the alternatives the gate exchange considered?

### Question 1: Do the additions foreclose alternative framings the gate exchange made available?

The seam framing paragraph introduces a three-seam decomposition (execution-model seam, generation-form seam, semantic-coherence seam) and explicitly excludes the semantic-coherence seam from this ADR. This framing is the practitioner's framing at the gate; it is attributed as such in the Provenance check. The Conditional Acceptance section correctly names FC-51 as the diagnostic that distinguishes all three seam failure modes at PLAY. No alternative framing from the gate exchange is foreclosed — the seam decomposition is an additive categorization that clarifies scope without suppressing alternatives.

The "not available / not required" section presents the guarantee question as a choice between probabilistic mechanisms at different costs, not between guaranteed and probabilistic. This framing closes the possibility of a true structural guarantee. Given Spike χ F-χ.1's evidence (no robust general extraction rule; form not mechanically derivable from schema), the framing is grounded. The one alternative that might feel foreclosed — a frontier-tier model that reliably produces bare output without directives — is not a structural guarantee either, and the ADR's Conditional Acceptance escalation path (step 3: frontier seat-filler) explicitly keeps that door open for PLAY.

### Question 2: Any truths from the gate exchange underrepresented in the additions?

The gate exchange grounded the "not required" argument in ADR-034's execution model. The additions accurately represent this. The seam framing's scope-exclusion of the semantic seam is accompanied by a disposition (OQ #27 axis-2 territory; FC-51 PLAY observation target), which is honest about what happens to the excluded seam — it is not dismissed, it is deferred with a named accountability mechanism.

One truth from the gate exchange that is present but lightly surfaced: the "not required" argument's bounded-failure-cost reasoning is explicitly conditional ("on a surface without client-side execution affordances, the detection-gate backstop would be warranted from the start"). This conditionality is in the text. It could be made more prominent by cross-referencing the surface-mode discriminator (ADR-033 Decision 1), but that would be a tightening, not a correction.

### Question 3: Does inverting the dominant framing of the additions reveal anything?

The seam framing's dominant move is to frame the form problem as a *seam cost of delegation* — implicit in single-model flows, explicit in ensemble flows. Inverted: the form problem is a fundamental limitation of the ensemble approach, not a seam cost that can be fully closed. Under this inversion, the claim "ADR-035 restores the generation-form half of parity" becomes harder to sustain — if bare-form production is model-compliance-dependent (not structurally guaranteed), parity is probabilistic, not restored. The ADR is fully aware of this tension: it records the mechanism as model-compliance-dependent in Consequences §Negative and in the "not available" paragraph. The Conditional Acceptance shape exists precisely to hold this tension open for PLAY validation. The additions do not suppress the inverted framing; they incorporate it into the ADR's honest qualification structure.

### Framing Issues

No new P1 or P2 framing issues introduced by the four additions. The held P2-F1 (AS-9 analogy tension) and P2-F2 (granularity invariant / structured-multi-file alternative not probed) from R1 remain at the practitioner gate; the additions do not affect their status.

---

## Convergence-Saturation Signal (ADR-094)

**Convergence-Saturation Signal:** TRIGGERED

- Round number: R3 (same document form; no form-change event; baseline continues from R2)
- P1 count this round: 0 (Section 1: 0; Section 2: 0)
- P2 count this round (new, non-carry-over): 0 — the R2 P2-1 is verified corrected; the four additions introduce no new P2 findings
- New framings or claim-scope expansions: none — the seam framing paragraph names three seams already implicit in the prior body (execution-model parity was ADR-034's claim; form-contract is ADR-035's claim; semantic seam was already excluded from scope). The three-seam categorization is a clarifying articulation of the existing scope structure, not a new claim-scope expansion. No new warrants surfaced.
- Recommendation: **STOP at this round.** All three signal conditions hold: P1 = 0; new P2 = 0; no new framings. The remaining open items are practitioner-gate decisions: the two held R1 framing P2s (P2-F1, P2-F2), the carry-over P3-3, and the new P3-1 (permission-gate compliance assumption — minor clarity gap, not a logical error). Further audit rounds are not warranted.

*The held framing items and P3s are practitioner judgment calls, not argument-audit blockers. TRIGGERED means the audit round-count discipline is satisfied; the practitioner decides whether the held items require action before the ADR advances.*
