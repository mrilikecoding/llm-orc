# Argument Audit Report

**Audited documents:**
- `docs/agentic-serving/decisions/adr-033-layer-a-loop-driver-multi-turn-agentic-surface.md`
- `docs/agentic-serving/decisions/adr-034-client-tool-action-terminal-artifact-bridge.md`
- `docs/agentic-serving/decisions/adr-027-framework-driven-dispatch-pipeline.md` (partial-update header only)

**Source material:**
- `docs/agentic-serving/essays/research-logs/cycle-7-spike-tau-upsilon-decide-entry-probes.md`
- `docs/agentic-serving/essays/research-logs/006b-client-tool-action-terminal.md`
- `docs/agentic-serving/essays/essay-outline-006-cross-compatibility-routing-surface.md` (§C8, W8.x, Section 9)
- `docs/agentic-serving/domain-model.md` (AS-9, AS-10, OQ #26, OQ #27, Amendment Log #15)

**Dependent prior ADRs read:**
- ADR-025 (artifact-as-substrate)
- ADR-026 (capability matching from request content alone)
- ADR-027 (framework-driven dispatch pipeline)
- ADR-028 (routing-planner ensemble) — via references; full text not re-read
- ADR-030 (tool_choice disposition)

**Genre:** ADR set
**Date:** 2026-06-02

---

## Section 1: Argument Audit

### Summary

- **Genre:** ADR set
- **Argument chains mapped:** 8 (callee resolution; grounded-loop Conditional Acceptance; single-action-per-turn enforcement; surface-mode discrimination; ADR-027 scoping; artifact-bridge necessity; tool-call terminal necessity; naming disambiguation)
- **Issues found:** 5 (0 P1, 3 P2, 2 P3)
- **Pyramid coverage map:** N/A (ADR genre)
- **Expansion-fidelity findings:** N/A (ADR genre)

---

### P1 — Must Fix

None.

The core inference chains are logically sound: callee resolution is grounded in Spike υ's measured evidence and explicitly flags n=1; the Conditional Acceptance is correctly shaped per ADR-097 (axis-1 conditionally resolved under enforcement, axis-2 deferred); the artifact-bridge necessity follows from F-ρ.1 and ADR-025's `output_substrate: artifact` design; the ADR-027 scoping is consistent with ADR-027's body; AS-9 and AS-10 are honored rather than amended. No conclusion requires a premise the ADRs fail to establish.

---

### P2 — Should Fix

**P2-1. ADR-033 §Decision point 1: the surface-mode discriminator is marked as "drafting-time synthesis" in the Provenance check but is not flagged as such in the Decision text.**

- **Location:** ADR-033 §Decision ¶1 ("When a chat-completions request carries client `tools[]`...") vs. ADR-033 §Provenance check ¶4 ("drafting-time synthesis... not directly driver-derived. ARCHITECT/BUILD validates it.")
- **Claim in Decision:** the surface engages the layer-A loop-driver when a request carries client `tools[]` and uses the ADR-027 pipeline when it does not. This is stated as a decided mechanism.
- **Evidence gap:** the Provenance check correctly identifies this discriminator as drafting-time synthesis, not driver-derived. The decision text does not carry a comparable epistemic qualifier. A BUILD implementer reading §Decision ¶1 without reading the Provenance check sees a settled design choice; one reading the Provenance check sees a VALIDATE-in-BUILD synthesis. The gap is real: no spike tested what happens to a tool-driven request that hits the single-turn pipeline, or whether the `tools[]` presence signal is the right branch condition (versus, say, request-history length, or an explicit marker from the client). The discriminator could be wrong — a client might send `tools[]` for bookkeeping reasons without expecting agentic loop behavior.
- **Recommendation:** add a parenthetical qualifier in §Decision ¶1: "The specific discriminator (`tools[]` presence) is a drafting-time synthesis — ARCHITECT/BUILD validates it against production traffic and may refine the signal." This surfaces the epistemic gap to implementers without changing the design intent.

**P2-2. ADR-033 §Consequences (Negative): the turn-count multiplication cost is understated as an interaction with axis-2 risk.**

- **Location:** ADR-033 §Consequences §Negative ¶2 ("Single-step enforcement multiplies turn count.")
- **Claim:** turn-count multiplication is named as a latency cost, accepted under the parity-is-behavioral-not-latency commitment.
- **Evidence gap:** the latency consequence compounds with the axis-2 risk in a way the text does not make visible. If a cheap-tier driver requires a capable driver on long horizons (the axis-2 contingency), every-turn driver cost rises AND the turn count is high (single-step enforcement). The cost-distribution value proposition could erode on two dimensions simultaneously. The Consequences section treats each risk separately; the interaction — "high turn count × more capable driver = much higher per-session cost" — is not named. This is a practitioner-facing risk that deserves surface-level acknowledgment.
- **Recommendation:** add a sentence in the turn-count negative: "If axis-2 validation requires a more capable driver, the per-session cost rises on both dimensions (higher per-turn driver cost × higher turn count); this is the compound scenario the Conditional Acceptance backstops."

**P2-3. ADR-034 §Decision point 4: tool-mapping is assigned to ADR-033 (loop-driver decision logic) without evidence that the loop-driver was designed or tested with the full `write`/`edit`/`bash` mapping logic.**

- **Location:** ADR-034 §Decision ¶4 ("Tool-mapping is loop-driver decision logic.")
- **Claim:** the loop-driver decides which client tool a deliverable maps to; `edit`-in-place requires a `read` round-trip the loop-driver initiates.
- **Evidence gap:** Spike τ/τ′ tested a two-step value-carry task (bash → write) with a passthrough loop driver. The loop driver in those spikes made no tool-mapping decisions beyond "emit the first tool call in the batch." The claim that the loop-driver correctly decides `write` vs `edit` vs `bash` conditioned on observed client state is a design assertion, not a spike finding. ADR-033 correctly names this as ARCHITECT/BUILD work in its seat-filler discussion. ADR-034 assigns the decision logic to ADR-033 cleanly, but neither ADR flags that no spike tested the loop driver making multi-tool-type decisions. Spike υ used the full plan→dispatch→synthesize pipeline (not a bare loop-driver) and generated only `write` calls.
- **Recommendation:** add a scope note in ADR-034 §Decision ¶4: "Tool-mapping logic is loop-driver decision logic per ADR-033; no spike tested the loop-driver making `write`/`edit`/`bash` distinctions — this is ARCHITECT/BUILD design work, consistent with the `edit`/`bash`/multi-file items already listed in the Negative consequences."

---

### P3 — Consider

**P3-1. ADR-033 §Rejected alternatives: the wrapper's "per-turn multi-capability composition" niche residual is treated as strictly architectural but has no evidence base.**

- **Location:** ADR-033 §Rejected alternatives §Wrapper, final sentence ("if production traffic surfaces a real need for per-turn multi-capability composition, the pipeline-as-subroutine remains available")
- **Claim:** the wrapper earns its keep for per-turn multi-capability composition (search-then-summarize-then-write within one step).
- **Minor gap:** Spike υ tested single-capability generation only (code-gen). The wrapper's per-turn multi-capability niche was not probed. The claim that the wrapper "would earn its keep" in that niche is plausible-but-untested, stated as if the niche is validated. The residual is architecturally honest (not foreclosing the option), but it could be tightened: the niche is speculative, not confirmed.
- **Recommendation:** replace "would earn their keep" with "would in principle earn their keep" or add "(not tested)" to the parenthetical. Minor tightening of epistemic register.

**P3-2. ADR-034 §Consequences (Positive): "deterministic delivery" claim is slightly overreached.**

- **Location:** ADR-034 §Consequences §Positive ¶3 ("The delivery mechanism is deterministic.")
- **Claim:** the framework marshals the deliverable into a tool call deterministically.
- **Minor gap:** "deterministic" here means the marshalling step is framework-code (read artifact → insert into tool call), not LLM-generated. That framing is accurate. But the claim also implicitly covers artifact-bridge fidelity (the content equals the ensemble's deliverable). The one scenario where this is non-trivially non-deterministic is a large deliverable with content-encoding complexities (double JSON-escaping, binary content, etc.) — Spike π noted content integrity held through double JSON-escaping for a trivial case. For large code files or structured data, the marshalling chain could introduce encoding artifacts. This is not a P2 risk given the FC (artifact-bridge fidelity) fitness criterion covers it, but the word "deterministic" slightly overstates what has been verified at small n.
- **Recommendation:** consider qualifying: "The delivery mechanism is deterministic in the marshalling step (framework-code, not LLM-generated)" to clarify the scope of the determinism claim.

---

## Section 2: Framing Audit

The framing audit examines what the evidence base made available that the ADRs did not choose, and whether the chosen framings overreach or underrepresent the evidence.

---

### Question 1: What alternative framings did the evidence support?

**Alternative framing A: "Wrapper is the safer near-term choice; callee is the higher-upside bet."**

The evidence base — particularly Spike υ (n=1, correct result, test passed) and the latency measurements — supports reading the wrapper as the more conservative, lower-risk option in the near term. Spike υ demonstrated the wrapper works for the tested task shape. The callee shape's risks are explicit (axis-2 driver coherence unknown, seat-filler selection deferred), while the wrapper's risks are measured and bounded (latency cost, synthesizer redundancy for simple cases).

A framing that foregrounds the callee as the riskier, higher-upside bet and the wrapper as the risk-managed fallback would read as: "Adopt callee for cost and architectural cleanliness; accept that the driver-capability bet is the primary open risk; keep the wrapper specification available as a concrete fallback if the driver bet loses." The ADRs effectively endorse this framing in the Conditional Acceptance mechanism, but they frame the wrapper as the alternative that "lost" rather than as the live risk backstop. Whether the wrapper should be a named "deferred to BUILD/PLAY contingency" vs. a "rejected and residualized" shape is a practitioner framing choice the evidence supports revisiting.

*What would the reader need to believe for this alternative framing to be right?* That the axis-2 driver-capability risk is high enough that a concrete fallback mechanism should be formally specified at DECIDE, not left as a residual. If BUILD/PLAY axis-2 failure probability is judged non-trivial, the wrapper deserves a more explicit contingency posture than it currently occupies.

**Alternative framing B: "The surface-mode discriminator is itself an open design question, not a decided mechanism."**

The evidence base supports framing the surface-mode discriminator (`tools[]` presence) as one of several open design questions that ARCHITECT/BUILD must resolve, rather than as a decided mechanism with a note in the Provenance check. The ADRs' Provenance checks are unusually transparent about attributing this to drafting-time synthesis — but the Decision section presents it as decided.

*What would the reader need to believe for this to matter?* That a BUILD implementer who reads the Decision section without the Provenance check will implement the `tools[]` branch without validating whether that signal correctly discriminates agentic clients from non-agentic clients. This is a real implementation risk: some clients might send `tools[]` for metadata/introspection purposes without expecting an agentic loop.

**Alternative framing C: "Small-n evidence supports the control-structure split, not just the callee resolution."**

The ADRs treat the layer-A/layer-B split as a research finding (from Spike σ) and the callee resolution as a DECIDE-entry evidence-based choice (from Spikes τ/τ′/υ). But the layer-A/layer-B split itself rests on σ.2 (n=1, short task, driver batched all three actions). A framing that maintains more epistemic humility about the split — treating it as well-motivated but lightly-evidenced — would surface that the architectural split and the callee resolution are both resting on small-n evidence, and that axis-2 validation will test both assumptions simultaneously.

---

### Question 2: What truths were available but not featured?

**Finding 1: The τ′ enforcement mechanism is a stand-in, not a production specification.**

The spike research log is explicit (§Scope of claim): "The framework-forced single-step was implemented by truncating the driver's batch in a scratch proxy. The production mechanism (and whether truncation vs. a re-planning prompt vs. a one-tool `tool_choice` constraint is the right enforcement) is DECIDE/ARCHITECT/BUILD design."

ADR-033 §Decision ¶3 codifies single-action-per-turn enforcement as a fitness criterion but does not name the production enforcement mechanism as an open design question. The FC definition is refutable ("a turn that dispatches two client tool calls before returning the first's result violates this") but does not specify how the framework prevents this structurally. The Provenance check does not flag the enforcement-mechanism as unresolved. A BUILD implementer arrives at this FC without knowing that the spike's enforcement was a scratch proxy, not a validated production mechanism.

This finding is available in the source material and absent in the ADR's primary text. It does not change the decision but it is a genuine unresolved design question the ADR could make visible.

*Why might it have been excluded?* Likely treated as ARCHITECT/BUILD scope, consistent with other mechanism decisions deferred there. But the FC wording implies a production mechanism exists, which is slightly stronger than warranted.

*Would its inclusion change the argument?* No — but it would prevent a BUILD implementer from treating the FC as fully specified.

**Finding 2: The wrapper was tested on a batchable task, not a non-collapsible one.**

The Spike υ evidence base used the "same multi-step calc task σ.2 ran" (write calc.py, write test_calc.py, run test). This is a batchable task — Spike τ demonstrated that batchable tasks are exactly the ones where the unconstrained driver succeeds. The wrapper's redundancy finding (synthesizer returns output verbatim, planner stage re-decides) holds for any generation task. But the latency comparison was made on a batchable task where the callee's per-turn structure is not tested against its actual production shape (single-step enforcement, which adds turns).

Under single-step enforcement, the callee costs (driver round-trip + ensemble call) × N turns. Under the wrapper per Spike υ measurements, it costs ~50s per write. The comparison that supports the callee choice implicitly assumed single-step enforcement on the callee side but measured the wrapper on the unconstrained (two-write) task. The actual per-session latency comparison under equal single-step enforcement was not computed. The ADR's latency argument ("~3× per-turn latency") is directionally correct but the specific comparison was not apples-to-apples.

*Would this change the conclusion?* Probably not — the direction (callee faster per turn) likely holds. But the latency difference under production single-step enforcement may be smaller or differently distributed than the υ measurement implies. The Framing section should acknowledge this.

**Finding 3: Spike ρ validated one capability-matched task through one round-trip + one continuation.**

ADR-034 §Context states "Spike ρ confirmed the framework's routing-planner decides to delegate on a real tool-rich OpenCode request and the delegated work returns via the `tool_calls` terminal with parity, together, end-to-end." This is accurate but the scope is tight: one capability-matched code-generation task, one round-trip plus a canned closer. The research log itself names this scope limitation ("planner-driven delegation + `tool_call` terminal validated for one capability-matched code-generation task through one round-trip + one continuation"). ADR-034 presents this as the end-to-end validation without restating the tight scope. The ADR's Negative Consequences mentions unbuilt items (`edit`, `bash`, multi-file, streaming) but does not restate that the "combined validation" is single-task, single-capability.

*Would inclusion change the argument?* The ADR's overall scope-of-claim framing is honest, but restating the ρ scope limitation at the point of the "combined validation" claim would tighten it.

---

### Question 3: What would change if the dominant framing were inverted?

**The dominant framing:** the callee resolution is the evidence-backed choice; the wrapper is an eliminated alternative with a narrow niche residual.

**Inverted framing:** the callee resolution is the higher-upside, higher-risk bet; the wrapper is the more conservative choice with measured, bounded costs.

Under the inverted framing:

- **Claims that become weaker:** the cost advantage of callee assumes a cheap-tier driver succeeds at axis-2, which is the unresolved bet. Under the inverted framing, the wrapper's ~3× per-turn latency becomes the cost of axis-2 certainty, not waste.
- **Claims that become stronger:** the wrapper's pipeline stages "earn their keep" for multi-capability composition becomes a broader argument — if real-world agentic sessions routinely need within-turn composition (edit + verify + rewrite in one turn), the wrapper's overhead is amortized differently.
- **Evidence that becomes more salient:** Spike υ's n=1 result showing the wrapper works correctly and passes the test. The fact that the wrapper's synthesizer returned output verbatim is a finding that the synthesizer is redundant for *code-gen content*, but it also shows the synthesizer did not corrupt the output — one of ADR-033's stated concerns.
- **What the ADRs would need to address under the inverted framing:** a more explicit contingency specification — not just "wrapper is the named fallback if axis-2 fails" but "here is the concrete specification for adopting wrapper-per-turn as the generation subroutine, which remains available at a defined cost." ADR-033 records the rejected alternative substantively but does not provide a re-activation specification. If axis-2 fails in BUILD/PLAY, the practitioner needs to know what "revert to wrapper" means architecturally.

---

### Framing Issues

**P1 — Consequential omissions where source material contains findings that would change the document's conclusions:**

None. The ADRs' scope-of-claim language is faithful to the evidence, and no finding in the source material is omitted in a way that would invalidate a conclusion.

---

**P2 — Underrepresented alternatives that should be acknowledged:**

**F2-1. The τ′ enforcement mechanism's "stand-in" status is not surfaced in the Decision text.**

- **Location:** ADR-033 §Decision ¶3 + §Fitness criteria
- **Source:** Spike τ/τ′ research log §Scope of claim ("τ′ enforcement is a stand-in. The framework-forced single-step was implemented by truncating the driver's batch in a scratch proxy.")
- **Why not featured:** likely scoped to ARCHITECT/BUILD as a mechanism decision. But the FC implies a production-ready enforcement mechanism in a way the Decision text does not qualify.
- **Recommendation:** add a brief note in §Decision ¶3 or the single-step FC: "The production enforcement mechanism (truncation vs. re-planning prompt vs. `tool_choice` constraint) is an ARCHITECT/BUILD decision — the τ′ probe used a scratch proxy; the mechanism is validated in principle, not implemented."

**F2-2. The latency comparison supporting callee over wrapper was made on a batchable task under the unconstrained (multi-write) case; under production single-step enforcement the comparison is apples-to-apples-ish but not identical.**

- **Location:** ADR-033 §Rejected alternatives §Wrapper (citing Spike υ's ~3× latency finding)
- **Source:** Spike τ/τ′ research log §Spike τ (non-collapsible task design); Spike υ research log (batchable task)
- **Why not featured:** the latency finding is directionally robust regardless of task-collapsibility (three serialized model calls vs. one is three calls vs. one on any task). But the specific "~3× per-turn" figure was measured on two writes in one task run, not across multiple single-step turns under enforcement.
- **Recommendation:** add a scope qualifier in the latency discussion: "Spike υ measured wrapper latency on a batchable two-write task; under production single-step enforcement on the callee side the per-session comparison shifts (N driver round-trips + N ensemble calls vs. wrapper's 3 serialized calls × M turns), but the per-turn directional advantage holds."

---

**P3 — Minor framing choices that could be more balanced:**

**F3-1. The "wrapper remains available for the narrow niche" residual is positioned as a design concession rather than a live contingency.**

- **Location:** ADR-033 §Rejected alternatives §Wrapper, last sentence; ADR-033 §Consequences §Neutral ¶2
- **Source material:** the research log names the niche residual directly and honestly; the ADR preserves it. No omission — this is a framing tone observation.
- **Note:** the wrapper residual is named in both the rejected-alternatives and consequences sections. Whether it reads as a live contingency vs. a gracious acknowledgment depends on BUILD/PLAY axis-2 outcome. The current framing is defensible; a future-cycle practitioner reviewing axis-2 failure would benefit from a slightly more explicit "how to re-activate the wrapper" note.

**F3-2. ADR-034's "deterministic delivery" framing does not distinguish "deterministic marshalling" from "fidelity-validated delivery".**

- **Location:** ADR-034 §Consequences §Positive ¶3
- **Cross-reference:** Argument Audit P3-2 above.
- **Note:** this framing issue mirrors the argument-audit P3-2 finding; the two findings are related and can be addressed together.

---

## Convergence-Saturation Signal (ADR-094)

**Convergence-Saturation Signal:** NOT TRIGGERED

- Round number: R1 (first audit on the ADR-033 + ADR-034 artifact pair; form-change baseline — new ADRs, not a re-audit of a prior form)
- P1 count this round: 0 (Section 1: 0; Section 2: 0)
- P2 count this round (new, non-carry-over): 5 (P2-1, P2-2, P2-3 from Section 1; F2-1, F2-2 from Section 2)
- New framings or claim-scope expansions: surface-mode discriminator as open design question (P2-1 / F2-1); turn-count × axis-2 cost interaction (P2-2); latency comparison apples-to-apples qualification (F2-2); τ′ enforcement stand-in status (F2-1); tool-mapping as untested loop-driver decision logic (P2-3)
- Recommendation: CONTINUE to R2

*P2 count exceeds 1 and new framings are present; signal does not trigger. R2 is warranted after the orchestrator reviews and applies corrections.*
