# ADR-014: Calibration Gate Trajectory-Level Extension

> **Superseded by ADR-046 on 2026-07-01.** The imperative per-dispatch Proceed/Reflect/Abstain machinery is superseded — the seat contract's per-dispatch pass/fail (ADR-046 §2) covers output-correctness gating. **The process-level AUQ/HTC confidence signal is NOT discarded:** it is a different signal type (verbalized confidence + generation-trajectory features — token-entropy, attention-weight distribution — that catch a contract-conforming-but-anomalous generation an output check cannot see) and is carried forward as a named candidate for Q2 grounded acceptance (ADR-046 §Open). Body immutable.

**Status:** Superseded by ADR-046 on 2026-07-01 (was: Proposed)

**Date:** 2026-05-05

---

## Context

ADR-007 establishes the Calibration Gate (L1) as a post-hoc output-level mechanism: composed ensembles enter calibration on creation, their first N invocations are result-checked, and quality signals accumulate to determine trust transition. ADR-007's Open Question #5 — *what counts as a quality signal* — was left open because the cycle that produced ADR-007 had not yet surveyed the calibration literature.

Cycle 4's Wave 2 literature review surveyed the calibration-gated routing surface and surfaced converged-on mechanisms. The cycle's empirical territory adds two specific findings: AUQ's Dual-Process framework (arXiv:2601.15703) achieves +10.7 percent on ALFWorld and +13.6 percent on WebShop training-free with a System 1 verbalized-confidence soft propagation through attention plus a System 2 binary gate at thresholds 0.8–1.0; HTC (arXiv:2601.15778) extracts process-level features across entire trajectories — more informative than output-level calibration for long-horizon decisions, with cross-domain transfer validated without retraining. Chuang et al. (arXiv:2502.04428) establish across 1,500-plus settings that the choice of uncertainty-quantification method dominates threshold choice as a design lever — a direct recommendation about where Calibration Gate implementation effort matters most. ReDAct (arXiv:2604.07036) shows that deferring only 15 percent of decisions to a large model matches full large-model quality; the calibration verdict is what enables that selective deferral economically.

Essay 005 §"ADR candidate #3" frames the elaboration: in-process trajectory-level calibration alongside the existing post-hoc promotion tracking, with HTC trajectory-level features and AUQ-style soft-then-binary gating filling in ADR-007's Open Question #5. The cycle's RDD-phase decomposition (essay 005, §"RDD-Cycle Decomposition") locates the value: Cluster 1 phases (RESEARCH, DECIDE, SYNTHESIZE) and the verification work in Cluster 2 phases need calibration verdicts at known dispatch points; Cluster 2's continuous-routing work needs in-process trajectory-level calibration as inputs accumulate.

OI-MAS (arXiv:2601.04861) is the closest published analogue to an in-process Calibration Gate — confidence below threshold triggers tier escalation, with 17–78 percent cost reduction at +12.88 percent accuracy. ADR-014's calibration verdict is the data input that ADR-015's per-role tier-escalation router consumes; the two ADRs compose to form the cycle's in-process calibration system. ADR-014 specifies the calibration verdict; ADR-015 specifies the routing behavior the verdict drives.

The cross-layer extension — L0 ensemble output signals influencing L1 dispatch decisions through an upward signal channel — is held for ADR-016. ADR-014's scope is in-layer (L1 → L1) and in-process; ADR-014 does not depend on ADR-016's acceptance. ADR-016 is currently in conditional-acceptance status (synthetic-data and structural-transfer validation completed 2026-05-06; first-deployment evidence pending). If ADR-016's conditional acceptance proceeds to full acceptance after first-deployment evidence, ADR-014's input set expands to include cross-layer signal data through the read-only channel ADR-016 establishes; if ADR-016 is ultimately rejected at any validation gate, ADR-014 continues operating on L1-internal trajectory data only.

The framing commitment from research-gate Grounding Action 2 (recorded 2026-05-05, *elaboration-by-evidence*) holds: ADR-014's expanded responsibilities concentrate within the existing Calibration Gate module rather than warranting a dedicated trajectory-monitor module orthogonal to L1.

The practitioner's "no token-limit pre-optimization on free local models" guidance (gate-conversation, 2026-05-05) relaxes a constraint that might otherwise minimize trajectory-feature extraction. The relaxed constraint allows fuller HTC-style feature extraction without per-token economy concern; this design space is recorded in the Provenance check.

---

## Decision

The Calibration Gate (L1) extends ADR-007's post-hoc output-level mechanism with an **in-process trajectory-level calibration layer**. ADR-014 fills in ADR-007's Open Question #5 (what counts as a quality signal) with an explicit composition:

### Quality signal composition

A quality signal is a composition of three sources, all available to the Calibration Gate:

1. **Post-hoc result check** (per ADR-007, unchanged) — the existing first-N invocation result-check produces a quality signal attached to the invocation.

2. **In-process AUQ confidence** — verbalized confidence emitted by the dispatched ensemble's component agents, propagated through attention as a soft constraint (System 1) and gated at a binary threshold for action decisions (System 2). Threshold defaults to 0.85 within the literature-supported 0.8–1.0 range; operationally tunable per deployment.

3. **HTC trajectory features** — process-level features extracted across the entire dispatch trajectory: token-level entropy patterns, attention-weight distributions over tool-call sequences, decision-confidence trajectories across reasoning steps. The full HTC feature surface is extracted; per the practitioner's "no token-limit pre-optimization" guidance, the extraction is generous rather than minimized.

   **Feature-extraction location (composition with ADR-016, argument-audit P2.4 finding 2026-05-06).** When ADR-016 is in conditional-acceptance or fully-accepted status (cross-layer signal channel active), HTC trajectory features are extracted **once at L0** during ensemble dispatch and propagated upward through the read-only signal channel; L1's Calibration Gate consumes the cross-layer features rather than re-extracting them in-layer. When ADR-016 is rejected (no cross-layer channel), HTC trajectory features are extracted in-layer at L1 from the L1-internal trajectory data the Calibration Gate has access to. The single-extraction-point property is load-bearing for performance (avoid double-cost compute) and consistency (the same trajectory features inform both the cross-layer signal channel's bounding mechanisms and the in-layer calibration verdict).

### Calibration verdict

The Calibration Gate produces a **calibration verdict** for each dispatch decision. The verdict is one of three values:

- **Proceed** — confidence is high (System 2 threshold met, post-hoc check positive, trajectory features in normal range). Dispatch proceeds without modification.
- **Reflect** — confidence is below threshold (System 2 not met) but not failure. Dispatch triggers a reflection step rather than blocking; the orchestrator is informed of the low-confidence verdict and may reformulate, escalate per ADR-015's router, or proceed with explicit acknowledgement of the uncertainty.
- **Abstain** — dispatch is blocked; the orchestrator receives a typed abstention error and must take a different action. Triggered by any of three concrete criteria (per argument-audit P2.3 finding 2026-05-06):
   - **Entropy collapse:** token-level entropy in the dispatched ensemble's most recent N tokens drops more than 1.5 standard deviations below the trajectory's running mean. Threshold operationally tunable.
   - **Post-hoc result-check hard failure:** the result-check from ADR-007's existing first-N calibration mechanism produces a verdict-incompatible outcome (e.g., the dispatched ensemble's output fails the result-check criteria with non-recoverable error rather than recoverable low-quality).
   - **Multiple drift-detection criteria simultaneously exceed thresholds** (when ADR-016's mechanism (d) is active and reporting drift). Severe-drift verdict from mechanism (d) propagates to Abstain at the Calibration Gate level.

  If none of these criteria fire, dispatch routes to Proceed (above-threshold confidence) or Reflect (below-threshold confidence without anomaly). The trichotomy is structurally exhaustive given the criterion specifications above.

Low confidence triggers reflection rather than blocking by default — this preserves AUQ's empirical pattern (low confidence → reflection produced the +10.7% / +13.6% gains). Abstain is reserved for severe-confidence-collapse cases, not routine low-confidence cases.

### In-process versus post-hoc layering

The two layers compose as follows:

- **Post-hoc layer (ADR-007)** — applies to composed ensembles in their first-N calibration window. The result-check signal feeds the post-hoc promotion tracking that determines trusted-status transition. Cross-session under Plexus, session-scoped under stateless mode (per ADR-007).

- **In-process layer (ADR-014, this ADR)** — applies to *every* dispatch decision the orchestrator makes, for all ensembles regardless of calibration status. The calibration verdict drives the dispatch-time decision (proceed / reflect / abstain). The verdict is recorded alongside the dispatch outcome; recorded verdicts feed the post-hoc layer's signal accumulation when the dispatched ensemble is in its first-N window.

The two layers are complementary, not substitute. Post-hoc tracks *whether an ensemble can be trusted*; in-process tracks *whether a specific dispatch should proceed right now*. Both signals govern stabilization (per AS-5, when Plexus is active).

### Time-decay windowing on trajectory features

The trajectory-feature extraction window is **time-bounded**: only trajectory data within the most recent 60 minutes (or N most recent dispatches, whichever is shorter) influences the current verdict. Older trajectory data does not propagate to current calibration verdicts. The windowing is applied at feature-extraction time, not at signal-storage time — historical signals remain in the artifact record (and, when Plexus is active, in the knowledge graph) for analysis and post-hoc review, but they do not influence current verdicts.

The windowing is the in-layer instance of essay 005's bounding-mechanism (b) — *time-decay windowing for the bias-compounding horizon*. ADR-014 applies the windowing to its in-layer trajectory features; ADR-016 applies a (potentially different) windowing to the cross-layer signal channel. Same mechanism class, different application contexts.

---

## Rejected alternatives

**(a) Hard binary threshold only (no System 1 soft attention propagation).** Rejected: AUQ's empirical evidence (+10.7% ALFWorld, +13.6% WebShop) is for the *dual-process* system, not for binary-only. Single-stage binary gating loses the soft-attention-propagation benefit and reduces to a coarser uncertainty signal that does not capture intra-trajectory uncertainty evolution. The cycle's task class (long-horizon) makes intra-trajectory uncertainty consequential.

**(b) Trajectory-level only, dropping post-hoc layer.** Rejected: post-hoc promotion tracking remains valuable for cross-session stabilization — it is the mechanism by which composed ensembles earn trusted status (per ADR-007). Trajectory-level adds in-process responsiveness *without replacing* the cross-session signal. Dropping post-hoc would weaken the calibration → trust → stabilization chain that AS-5 requires.

**(c) Output-level confidence only, skipping trajectory features.** Rejected: HTC's evidence shows trajectory-level features are more informative than output-level for long-horizon decisions. The cycle's task class (long-horizon multi-step orchestrator sessions) is exactly the territory HTC's evidence covers. Output-level only would be a less-informative calibration signal for the same compute cost.

**(d) Block-on-low-confidence (rather than reflect-on-low-confidence) by default.** Rejected: AUQ's empirical pattern is that low-confidence-triggers-reflection is what produces the accuracy gains. Block-on-low-confidence shifts the failure mode from "agent reformulates and proceeds" to "agent halts and operator intervenes" — the latter has higher friction without proportionate quality gain. The Abstain verdict is reserved for severe cases (trajectory anomaly, post-hoc failure); routine low-confidence triggers Reflect.

**(e) Dedicated trajectory-monitor module orthogonal to Calibration Gate (cross-cutting infrastructure).** Rejected: per the cycle's framing commitment from research-gate Grounding Action 2 (elaboration-by-evidence), expanded calibration responsibilities concentrate within the Calibration Gate module rather than warranting a separate cross-cutting module. The module-shape inheritance from essay 005's verdict is held; a dedicated trajectory-monitor would introduce module-shape strain without clear architectural benefit. (Falsification trigger applies — if BUILD finds the trajectory-feature-extraction logic structurally awkward inside the Calibration Gate, the elaboration-by-evidence framing commitment is what's being tested.)

**(f) No time-decay windowing on trajectory features.** Rejected: feedback paths compound bias when stale signals influence current decisions. Khanal et al.'s universal non-improvement finding from episodic memory augmentation is the cycle's load-bearing literature evidence on this risk. Time-decay windowing is the in-layer bounding mechanism that addresses the risk for L1's own internal feedback loop. Skipping it would expose ADR-014's in-process layer to the same compounding-bias failure mode the bounding mechanisms in ADR-016 are designed to bound for the cross-layer signal.

---

## Consequences

**Positive:**
- Fills ADR-007's Open Question #5 with an explicit, literature-supported quality-signal composition (post-hoc + AUQ + HTC)
- In-process calibration enables ADR-015's per-role tier-escalation router to make confidence-gated dispatch decisions; the two ADRs compose to form the in-process calibration system
- Reflect-on-low-confidence preserves AUQ's empirical pattern (the mechanism that produced the +10.7% / +13.6% gains in published results)
- Time-decay windowing addresses the in-layer feedback-bias risk before it can compound; the mechanism is consistent with ADR-016's cross-layer bounding (same mechanism class, different scope)
- HTC's cross-domain transfer evidence (validated without retraining) means trajectory features generalize across the cycle's task classes (RESEARCH, DECIDE, BUILD, ARCHITECT) without per-task tuning
- Per the practitioner's no-token-limit-pre-optimization guidance, fuller HTC feature extraction is design-allowed — the larger feature set increases calibration-signal quality at the cost of compute, and on free local models the cost trade is favorable

**Negative:**
- Trajectory-feature extraction adds compute cost at every dispatch; HTC's full feature set is more expensive than output-level confidence alone
- HTC is novel-without-codebase-precedent in llm-orc; BUILD will require new infrastructure for trajectory-feature extraction (token-level entropy, attention-weight access, decision-confidence trajectory recording)
- Three threshold values (System 2 confidence threshold default 0.85; trajectory-feature anomaly threshold; abstention severity threshold) are operationally tunable; default tuning may need adjustment per deployment
- Time-decay windowing's window-shape choice (60 minutes vs. last-N-dispatches) is itself a design parameter; the dual-bound (whichever is shorter) is drafting-time synthesis and may need adjustment after first-deployment evidence
- Reflect verdicts increase orchestrator-side complexity — the orchestrator must respond to Reflect by reformulating, escalating per ADR-015, or proceeding with acknowledgement; this is additional per-dispatch logic on the orchestrator's reasoning surface
- Abstain verdicts produce typed errors that the orchestrator must handle; failure to handle them produces session-level failures rather than per-dispatch failures (the cost of structural enforcement)

**Neutral:**
- The composition with ADR-007 is additive — ADR-007's first-N output-level mechanism is unchanged; ADR-014 layers in-process trajectory-level on top
- The composition with ADR-015 is data-flow: ADR-014 produces calibration verdicts; ADR-015 consumes them as router input
- The composition with ADR-016 is conditional: if ADR-016 is accepted, ADR-014's input set expands to include cross-layer signals; if ADR-016 is rejected, ADR-014 operates on L1-internal data only
- Post-hoc promotion tracking remains the cross-session stabilization mechanism (under Plexus per AS-5, session-scoped under stateless mode per ADR-007)
- Time-decay windowing applies at feature-extraction time, not at signal-storage time — historical signals remain in artifact and Plexus records for analysis purposes

---

## Provenance check

- **Driver-derived content (mechanism specifications).** AUQ's System 1 + System 2 dual-process structure, threshold range 0.8–1.0, and reflect-on-low-confidence pattern are direct adoption from arXiv:2601.15703, surfaced via essay 005 §"ADR candidate #3" and §"Long-Horizon Reliability Infrastructure." HTC's trajectory-level feature framing is from arXiv:2601.15778. OI-MAS's confidence-gated tier-escalation pattern is from arXiv:2601.04861 (and is the input that ADR-015 consumes).

- **Driver-derived content (filling ADR-007 Open Question #5).** The three-source quality-signal composition (post-hoc + AUQ + HTC) is essay-derived. Essay 005's §"ADR candidate #3" specifies the composition; ADR-014 operationalizes it as a calibration-verdict mechanism.

- **Drafting-time synthesis (verdict trichotomy: Proceed / Reflect / Abstain).** Essay 005 specifies AUQ's reflect-on-low-confidence pattern but does not specify a trichotomous verdict structure with an Abstain category for severe cases. The Abstain verdict is drafting-time synthesis applying defensive design — without an Abstain path, severe trajectory anomalies must either be silently passed-through (which violates the calibration purpose) or treated as routine low-confidence (which produces unworkable retry loops). The trichotomy is drafting-time judgment; the alternative (binary Proceed/Reflect with no Abstain) is workable but loses defensive coverage of severe cases.

- **Drafting-time synthesis (System 2 threshold default 0.85).** The literature range is 0.8–1.0; the default of 0.85 is drafting-time synthesis picking a value within the range. Chuang et al. (arXiv:2502.04428) establish that uncertainty-quantification method choice dominates threshold choice, suggesting the threshold is a tuning parameter rather than a load-bearing design choice. The default value is an operational starting point.

- **Drafting-time synthesis (time-decay windowing as in-layer bounding mechanism).** Essay 005's bounding-mechanism (b) — time-decay windowing — is specified for ADR-016's cross-layer signal channel. ADR-014 applies the same mechanism class to its in-layer trajectory-feature extraction. The application to L1-internal trajectory data is drafting-time extension of essay 005's bounding-mechanism reasoning. The 60-minute / last-N-dispatches dual-bound is drafting-time synthesis; alternative windowing shapes (sliding-window, exponential-decay, hard-cutoff) are not examined in the rejected alternatives because the load-bearing question is *whether* to window, not the specific window shape. BUILD's first-deployment evidence will inform window-shape tuning.

- **Drafting-time synthesis (the practitioner's friction-vs-discovery guidance applied).** Per the gate-conversation guidance "willing to trade more friction for the opportunity to discover," ADR-014 specifies the fuller dual-process + trajectory-feature composition rather than a thinner version (e.g., binary-threshold-only, or output-level-confidence-only). The thinner versions are valid friction-minimizing alternatives; the fuller version is the friction-trades-for-discovery alternative. The choice is drafting-time application of practitioner guidance.

- **No-token-limit-pre-optimization guidance applied.** Per the gate-conversation guidance "not concerned with token limits on free local models," HTC trajectory-feature extraction is specified as the full feature set rather than a token-economized subset. The token cost on free local models is not a binding constraint for the cycle's deployment shape; on cost-bearing tiers, the feature set may need scoping. This is operational tuning territory, not architectural-decision territory.

- **Vocabulary impact.** ADR-014 introduces three terms candidate for domain-model addition at Tranche-C close:
  - **Calibration verdict** — proposed new term in §Concepts (operator voice; the value the Calibration Gate produces)
  - **In-process trajectory-level calibration** — proposed new term in §Methodology Vocabulary (research voice; the elaboration's category)
  - **Trajectory feature** — proposed new term in §Methodology Vocabulary (research voice; the HTC-derived data type)

  The verdict trichotomy values (Proceed / Reflect / Abstain) are operational vocabulary that may surface in operator-facing tooling; whether they belong in the domain model's Product Vocabulary is a Tranche-C-close editorial decision.

- **Asymmetric DECIDE budget per research-gate carry-forward #4.** ADR-014 is in the "novel architectural territory warranting full DECIDE pressure-testing" group. Argument-audit on this ADR should concentrate on (i) the verdict trichotomy's drafting-time design (whether Abstain is justified or overengineering), (ii) the time-decay windowing parameter shape (whether the dual-bound is well-specified for the cycle's deployment), and (iii) the composition with ADR-015 and ADR-016 (whether the data-flow assumptions are coherent). The literature-derived mechanism specifications (AUQ, HTC, OI-MAS adoption) are adoption-decision discipline; the drafting-time synthesis flagged above is novel-architectural pressure-test territory.
