# Argument Audit Report — Cycle 4 DECIDE-Phase ADRs

**Audited documents:**
- `docs/agentic-serving/decisions/adr-012-conversation-compaction-five-layer-pipeline.md`
- `docs/agentic-serving/decisions/adr-013-session-registry-initializer-then-resume-schema.md`
- `docs/agentic-serving/decisions/adr-014-calibration-gate-trajectory-level-extension.md`
- `docs/agentic-serving/decisions/adr-015-per-role-tier-escalation-router.md`
- `docs/agentic-serving/decisions/adr-016-upward-l0-l1-read-only-signal-channel.md`
- `docs/agentic-serving/decisions/adr-017-tool-call-structural-validation-guard.md`
- `docs/agentic-serving/decisions/adr-deferred-005-summarizer-harness-reconsideration.md`

**Source material:**
- `docs/agentic-serving/essays/005-layer-conditional-composition.md`
- `docs/agentic-serving/domain-model.md`
- `docs/agentic-serving/product-discovery.md`
- `docs/agentic-serving/decisions/adr-002-four-layer-architecture-plexus-optional.md`
- `docs/agentic-serving/decisions/adr-004-result-summarization-mandatory.md`
- `docs/agentic-serving/decisions/adr-007-calibration-gate-for-composed-ensembles.md`
- `docs/agentic-serving/decisions/adr-011-orchestrator-llm-is-a-model-profile.md`
- `docs/agentic-serving/essays/research-logs/005e-spike-adr016-b-time-decay-windowing.md`
- `docs/agentic-serving/essays/research-logs/005f-spike-adr016-d-structural-transfer-audit.md`

**Date:** 2026-05-06

---

## Section 1: Argument Audit

### Summary

- **Argument chains mapped:** 22
- **Issues found:** 14 (2 P1, 7 P2, 5 P3)

---

### P1 — Must Fix

---

**Issue 1.1 — ADR-015: ADR-011 compatibility argument is partially circular**

- **Location:** ADR-015 §Decision "ADR-011 compatibility — explicit verification"
- **Claim:** "The orchestrator's own LLM remains scoped at the session-boundary event per ADR-011 unchanged. The router escalates *the dispatched task's* Model Profile, not the orchestrator's. [...] The pattern is consistent with ADR-011's framing: tiered behavior is implemented in the dispatch path, not as a special case in the orchestrator."
- **Evidence gap:** ADR-011's Decision text reads: "If tiered behavior is desired, it is expressed as a **composed ensemble** invokable by the orchestrator — not as a mechanism special to the orchestrator." The key phrase is "composed ensemble." ADR-015 places the tier-escalation router inside Tool Dispatch (L2 interposition), which is architectural infrastructure — not a composed ensemble invokable by the orchestrator. The compatibility argument says "tiered behavior is implemented in the dispatch path," but ADR-011 specifically contemplated tiered behavior in an ensemble that the orchestrator calls, not in the dispatch layer the orchestrator's tool calls pass through invisibly. The argument asserts these are equivalent; they are structurally different. The essay's framing in §"ADR candidate #4" states this is "consistent with ADR-011's existing rule: ADR-011 already says tiered behavior is expressible as ensemble-composition; OI-MAS at Tool Dispatch is an *implementation* of ADR-011, not an amendment" — but this claim in the essay is itself the assertion under audit. A composed ensemble invokable by the orchestrator and an invisible dispatch-layer router are different architectural placements. The orchestrator that calls `invoke_ensemble` is unaware of the tier-escalation happening in the dispatch layer; an orchestrator calling a `triage-route` composed ensemble is choosing to delegate that routing. The compatibility argument elides this distinction.
- **Recommendation:** Either (a) acknowledge that ADR-015 extends ADR-011 in scope (rather than being a pure "implementation" of it) and add a brief justification for why L2-interposition is preferable to composed-ensemble placement, or (b) explicitly note that the "composed ensemble" framing in ADR-011 was the anticipated mechanism for tiered behavior but that L2 interposition achieves the same architectural separation with less orchestrator-side complexity and lower per-dispatch overhead. The current compatibility claim is true in spirit (the orchestrator's LLM is unchanged per session; tiered behavior is not baked into the orchestrator's reasoning logic) but overstates identity with ADR-011's explicit mechanism. This should be surfaced as "consistent with ADR-011's intent, extending its mechanism class" rather than "an implementation of ADR-011's existing pattern."

---

**Issue 1.2 — ADR-016: Conditional-acceptance status has a structural gap that partially undermines its function**

- **Location:** ADR-016 §Status, §Provenance check (conditional-acceptance synthesis), §Empirical validation pathway
- **Claim:** "Proposed (conditional acceptance — synthetic-data and structural-transfer validation completed 2026-05-06; first-deployment evidence pending for full operational validation)." The ADR ships with mechanisms (b) and (d) specified and claims that DECIDE-phase spike validation has narrowed the conditionality to "first-deployment evidence pending."
- **Evidence gap:** The conditional-acceptance status has a meaningful structural gap: the ADR is functionally "accepted for architecture" once it enters the decisions directory as Proposed with conditional acceptance, because downstream ADRs (ADR-014's composition with ADR-016, ADR-015's via ADR-014, FC-2 and FC-3 updates) will be written against it. There is no mechanism specified in the ADR or the corpus that automatically triggers re-deliberation if first-deployment evidence falsifies mechanisms (b) or (d). The falsification trigger in §"Empirical validation pathway" reads: "If any validation pathway produces evidence that mechanism (b) or (d) cannot be operationalized within L1 [...] the elaboration-by-evidence framing commitment is invalidated. The reorganization branch re-opens; ADR-016 is re-deliberated." But this re-deliberation is not governed by any structural gate — it depends on the practitioner or the next cycle reading the evidence and choosing to re-open. The conditional-acceptance status's practical effect is therefore that the ADR functions as accepted for BUILD purposes, with the "conditional" label tracking a monitoring responsibility rather than a genuine architectural deferral. This is not necessarily wrong, but the ADR's framing (and the practitioner's guidance on "not codifying unsupported assumptions without evidence") suggests the conditionality is structurally meaningful. As written, it is softer than that framing implies.
- **Recommendation:** Add a concrete monitoring specification to the empirical validation pathway: name which artifact, which phase, and which human action would constitute first-deployment evidence receipt and trigger re-deliberation. For example: "First-deployment evidence receipt = the cycle's North-Star benchmark produces at least N sessions of calibration log data; the responsible party (operator or next-cycle research entry) reads the calibration log against mechanism (b)'s bias-bound and mechanism (d)'s drift-detection criteria; if evidence is negative, files a re-deliberation note in `decisions/` per the deferred-ADR pattern." Without this, the conditional-acceptance status tracks a hope rather than a commitment.

---

### P2 — Should Fix

---

**Issue 2.1 — ADR-012: Threshold defaults justified by appeal to a Claude Code-specific failure history**

- **Location:** ADR-012 §Context, §Decision, §Rejected alternatives (a), §Provenance check
- **Claim:** "The four threshold values (50,000-character Layer 0 trigger; 60-minute Layer 2 idle window; 12,288-token Layer 3 cap; 3-failure Layer 4 circuit-breaker) are operationally tunable; defaults match Claude Code's specification." The 250,000 API calls per day waste figure is cited repeatedly.
- **Evidence gap:** The ADR appropriately flags the 250,000-API-calls figure as "a Claude Code-specific failure history, not a general finding, but the cost shape generalizes." However, the inference that the defaults match Claude Code's specification then goes unexamined — Claude Code is a different system with different dispatch frequencies, different tool-output sizes, and different operator workload shapes than llm-orc. The threshold defaults are presented as operationally-tunable starting points, which is correct, but the ADR does not establish why Claude Code's defaults are the right starting point for llm-orc versus, say, something more conservative. The 50,000-character threshold for Layer 0 and the 60-minute idle window for Layer 2 are never rationalized for llm-orc's workload shape. This is a scope-accuracy issue: the ADR implicitly claims the defaults are calibrated to llm-orc's deployment, but they are calibrated to Claude Code's deployment.
- **Recommendation:** Add a one-sentence acknowledgment in the Decision or Consequences section that the threshold defaults are Claude Code's values and have not been calibrated to llm-orc's workload shape. The existing Consequences §Negative entry about "four operationally-tunable thresholds carry deployment-tuning cost" partially covers this, but it should explicitly say the defaults are an external starting point, not an llm-orc-derived calibration.

---

**Issue 2.2 — ADR-013: Cluster-conditional applicability rule lacks a specified error case for cluster-misidentification**

- **Location:** ADR-013 §Decision "Cluster-conditional applicability," §Provenance check
- **Claim:** "Cluster determination is a session-start decision based on the operator's session-shape declaration; default behavior is required (operators opt-out for Cluster 1 / Cluster 3 contexts rather than opt-in for Cluster 2)."
- **Evidence gap:** The ADR specifies the cluster-conditional applicability rule and the default (required for Cluster 2, optional for Cluster 1 / 3), but does not specify what happens when the operator's session-shape declaration is wrong — i.e., an operator declares a Cluster 1 session that turns into a Cluster 2 session mid-flight, or vice versa. This is not an edge case in the cycle's North-Star benchmark: a DECIDE session (Cluster 1) that requires substantial BUILD work (Cluster 2) during the same session straddles clusters. The artifact set is enabled or disabled at session-start; there is no mid-session reclassification mechanism. The ADR is silent on this failure mode. The cluster-conditional rule is presented as complete, but the cross-cluster session shape — which the North-Star benchmark explicitly requires — is not addressed.
- **Recommendation:** Add a brief specification of the failure mode and a disposition. Options: (a) session-start determination is binding (operators must restart the session with the correct declaration), which simplifies the implementation but adds friction for mis-declared sessions; (b) the artifact set can be activated mid-session by the operator if cluster reclassification occurs, at the cost of the session not having a complete progress log from the start; (c) the artifact set defaults to required-unconditionally in the first cycle of deployment, with cluster-conditional applicability deferred to a future ADR once the deployment evidence clarifies which cluster transitions are common. At minimum, the ADR should acknowledge the cross-cluster session failure mode rather than leaving it as an uncovered case.

---

**Issue 2.3 — ADR-014: The Abstain verdict's trigger condition is underspecified relative to its function**

- **Location:** ADR-014 §Decision "Calibration verdict," §Provenance check (verdict trichotomy as drafting-time synthesis)
- **Claim:** "Abstain — confidence is severely below threshold (e.g., trajectory features show entropy collapse, post-hoc check failed) or HTC features indicate the trajectory is anomalous. Dispatch is blocked."
- **Evidence gap:** The ADR specifies Abstain as the response to "severe" confidence collapse but leaves "severe" operationally underspecified. The examples given ("entropy collapse," "post-hoc check failed") are illustrative but are not quantitative thresholds. In the Consequences section, the ADR notes "three threshold values [...] are operationally tunable," and the Provenance check acknowledges the Abstain verdict is drafting-time synthesis without literature grounding specifically for the three-class trichotomy. However, the Abstain case is the highest-stakes verdict (dispatch is blocked; the orchestrator must take a different action entirely) and is the one that most requires a concrete trigger specification. If the trigger is left to BUILD-phase tuning, BUILD engineers have no specification to work from. The AUQ paper (arXiv:2601.15703) specifies binary gating at thresholds 0.8–1.0; the mapping from that binary gate to a trichotomous verdict is not established in any cited source.
- **Recommendation:** Specify the Abstain trigger at the criterion level, even if thresholds are tunable. For example: "Abstain triggers when System 2 threshold is not met AND either (i) the post-hoc check from ADR-007 is negative for this ensemble, or (ii) HTC trajectory features show entropy collapse (defined as: token-level entropy below [configurable floor] for [configurable consecutive steps])." The exact thresholds can be tunable, but the criterion shape should be concrete enough for BUILD to implement without architectural re-deliberation.

---

**Issue 2.4 — ADR-014 / ADR-016 composition: the input-expansion claim requires a precise handshake specification**

- **Location:** ADR-014 §Context "The cross-layer extension," §Consequences "Neutral," §Provenance check; ADR-016 §The signal channel
- **Claim:** ADR-014: "If ADR-016's conditional acceptance proceeds to full acceptance after first-deployment evidence, ADR-014's input set expands to include cross-layer signal data through the read-only channel ADR-016 establishes; if ADR-016 is ultimately rejected at any validation gate, ADR-014 continues operating on L1-internal trajectory data only."
- **Evidence gap:** The composition claim is sound at the high level, but neither ADR specifies the interface between them with precision. ADR-016 §"The signal channel" specifies the data shape: ensemble output trajectory features (extracted at L0 per ADR-014's HTC specification), ensemble dispatch outcomes, and deterministic-tool-output signals. ADR-014 §"Quality signal composition" does not reference the cross-layer signal data shape at all — it specifies in-layer sources (post-hoc result check, in-process AUQ confidence, HTC trajectory features), but the HTC features in ADR-014 are described as extracted by the Calibration Gate from its own dispatch decisions, not received from L0 via the channel. There is a potential double-counting ambiguity: ADR-014 extracts HTC trajectory features in-layer AND ADR-016 sends HTC trajectory features from L0 through the upward channel. Are these the same features (extracted once, sent upward) or different features (extracted at two distinct points)? The composition's data-flow is ambiguous at the handshake boundary.
- **Recommendation:** In ADR-014's Consequences §Neutral (or in a new composition note), specify: when ADR-016 is active, the HTC trajectory features in the quality-signal composition are the cross-layer signals received via the channel rather than re-extracted in-layer, eliminating redundant extraction. When ADR-016 is inactive, the in-layer extraction serves as the fallback. This makes the data-flow direction concrete and eliminates the double-counting ambiguity.

---

**Issue 2.5 — ADR-016: Falsification trigger specificity is adequate for mechanisms but not for the "within L1" operationalization test**

- **Location:** ADR-016 §Falsification trigger
- **Claim:** "If any validation pathway produces evidence that mechanism (b) or (d) cannot be operationalized within L1 (e.g., the windowing's bias-compounding bound fails empirically; the audit dispatch's drift criteria don't transfer; either mechanism requires module-shape orthogonal to L0–L3), the elaboration-by-evidence framing commitment is invalidated."
- **Evidence gap:** The falsification trigger lists three triggering conditions. The first two (bias-compounding bound fails empirically; drift criteria don't transfer) are mechanism-specific and relatively clear. The third ("either mechanism requires module-shape orthogonal to L0–L3") is structurally ambiguous — BUILD is where module-shape decisions are made, and any complex-enough mechanism can be implemented in ways that look cross-cutting. What constitutes "orthogonal to L0–L3" as a module shape is not defined. A BUILD implementation of mechanism (d) that extracts its audit-dispatch logic into a shared class (reused by other L2 modules) would technically be "cross-cutting code" but might not violate the spirit of the "within L1" constraint. The falsification trigger needs a more concrete criterion for what it means to require orthogonal module shape.
- **Recommendation:** Replace the third condition with a more concrete criterion: e.g., "either mechanism (b) or (d) requires a process, thread, or service boundary that does not exist within the L1 Calibration Gate module's implementation scope (i.e., BUILD determines the logic cannot be hosted in the Calibration Gate's existing or extended class structure without introducing a new top-level module)." This makes the trigger concrete enough to fire rather than remaining formal language.

---

**Issue 2.6 — ADR-017: Conservative false-positive discipline's asymmetric cost-weighting is asserted but not grounded**

- **Location:** ADR-017 §Decision "Conservative false-positive discipline"
- **Claim:** "Under-detection is preferred over over-detection. The cost of a false-positive (rejecting a legitimate response) is operator-visible session disruption; the cost of a false-negative (allowing a phantom tool-call through) is an orchestrator-side fabrication that the orchestrator's downstream reasoning will incorporate as if the tool call had actually run. Both costs are non-zero; the pattern set's calibration favors false-negatives because the cycle's spike evidence does not establish that confabulation is high-frequency at cheap-cloud under operational conditions."
- **Evidence gap:** The reasoning is partially circular. The rationale for accepting false-negatives is that confabulation may be low-frequency under operational conditions. But the pattern set's scope determines how often the guard triggers and therefore whether this calibration is actually conservative. A very narrow pattern set produces few false-positives regardless of the underlying confabulation rate — the conservatism is structural (small set), not calibrated (matched to confabulation frequency). The ADR conflates "conservative pattern set" (meaning few patterns, so few triggers) with "conservative calibration" (meaning calibrated against the frequency of the failure mode). If confabulation is low-frequency but the pattern set is also small, the guard will rarely trigger and rarely catch anything — which is not a conservative outcome, it is a low-signal outcome. The Provenance check correctly notes that the conservative discipline is "drafting-time judgment applying the spike's prompt-design caveat," but the argument in the Decision section presents it as a calibration against the evidence, which it is not.
- **Recommendation:** Reframe the conservative false-positive discipline as: "The default pattern set is minimal rather than calibrated, because no deployment evidence exists for the confabulation rate under operational conditions. Operators are expected to extend the pattern set as deployment evidence accumulates (per the operator-extensibility model). The minimal default is a starting point, not a calibrated optimum." This is more accurate and avoids the false precision of presenting under-detection as a justified calibration against the spike evidence.

---

**Issue 2.7 — Deferred candidate #5: The evidentiary threshold for deferral is sound, but the spike specification has an implicit scope assumption**

- **Location:** Deferred-005 §"What spike would close the gap"
- **Claim:** The output-size sweep, ensemble-configuration sweep, and N>1 trials would collectively constitute "empirical evidence sufficient to justify (or reject) the amendment to ADR-004's mandatory framing."
- **Evidence gap:** The spike specification is appropriately scoped, but it has an implicit assumption: that the specificity-loss mechanism at the harness-interposition stage is attributable primarily to the summarizer's compression behavior, not to the orchestrator's downstream misquotation behavior. Wave 3.A Trial 2's observation was that "the orchestrator quoted 'verbatim from ensemble return' what was actually the summarizer's compressed blob." This is technically two failure modes: (a) the summarizer compressed a 600-character output unnecessarily, and (b) the orchestrator then presented the summarizer's output as if it were the ensemble's output. The proposed spike measures specificity-loss as a function of output size and ensemble shape, which addresses (a). But if (b) is the dominant failure mode — if the orchestrator's downstream misrepresentation is what produces the practical harm — then the spike's design does not directly measure the harm. An output-size sweep might show that the summarizer skips short outputs (eliminating (a)), but the orchestrator could still misrepresent outputs in ways the spike does not capture.
- **Recommendation:** Add a brief scope note to the spike specification acknowledging the two-failure-mode structure: "The spike measures specificity-loss at the harness-interposition stage (failure mode (a): unnecessary compression). The orchestrator's downstream representation of summarized output (failure mode (b)) is a separate surface; if the spike's results suggest (a) is not the dominant failure mode, the downstream representation pattern should be examined before amendment."

---

### P3 — Consider

---

**Issue 3.1 — ADR-012: Layer 4 circuit-breaker's three-failure window is per-session, but session boundary definition is ambiguous for long-running deployments**

- **Location:** ADR-012 §Decision item 5 (Layer 4), §Consequences §Negative
- **Claim:** "Layer 4's circuit-breaker introduces session-scoped error state that must be reset between sessions."
- **Note:** The ADR's definition of "session" is from domain-model.md: a stateful conversation between a client and the orchestrator agent, bounded by budget constraints. This is clear for a single user-session. For a long-running deployed serve instance with many sessions, the "three consecutive failures within a session" criterion is clear. However, the ADR does not specify whether the circuit-breaker state resets automatically at session end or requires explicit operator action. The Consequences §Negative entry says "must be reset between sessions and observable for operator debugging" — "observable" implies the state can be read, but the reset mechanism is not specified.
- **Recommendation:** One sentence in the Decision or Consequences sections specifying that the circuit-breaker state is session-scoped and resets automatically at session teardown would clarify the implementation requirement without architectural significance.

---

**Issue 3.2 — ADR-013: Write-gate validation item (iii) — signed-script integrity — conflates hash verification with integrity**

- **Location:** ADR-013 §Decision item 4 (iii)
- **Claim:** "The init.sh content is hashed at operator-authoring time and the hash is recorded in the Session Registry's configuration; execution is gated on hash match."
- **Note:** Hash verification establishes that the executed file matches the file the operator hashed. It does not establish that the file is safe to execute — a deliberately malicious init.sh that the operator hashed would pass the gate. The ADR labels this "signed-script integrity verification," but "integrity" in the security sense typically means the file has not been tampered with by a third party, not that the file is safe. The concern is not that the gate is wrong (it is correct for what it does), but that "integrity verification" may create a false impression of security scope. This is a terminology precision issue.
- **Recommendation:** Rename item (iii) to "operator-authoring-time hash verification" or "tamper-detection for init.sh" in the Decision section, and add a scope note: "The gate detects unauthorized modification of init.sh by a third party after operator authoring; it does not validate the content of the init.sh the operator authored."

---

**Issue 3.3 — ADR-015: The 8-skill taxonomy's claim that it enables "discovery of which capability dimensions matter" depends on an implicit assumption about deployment diversity**

- **Location:** ADR-015 §Decision "Per-skill tier defaults," §Rejected alternatives (b) and (d)
- **Claim:** "The full 8-skill taxonomy enables discovery of which skill dimensions actually matter for the cycle's task class and deployment shape. Subsetting before deployment evidence is premature optimization."
- **Note:** The discovery claim assumes that the deployment will exercise all eight Topaz skills with enough frequency to produce distinguishable calibration evidence. The cycle's primary task class (RDD-cycle work) concentrates in code_generation, tool_use, logical_reasoning, and instruction_following. Mathematical_reasoning, factual_knowledge, writing_quality, and summarization are secondary. If the deployment concentrates in a subset of skills, the configuration surface remains 16 slots but most slots produce no calibration evidence. The discovery framing is sound as a general principle, but the claim that "all 8 first-class" is necessary for discovery may overstate the evidence value of the less-used skills in the cycle's deployment.
- **Recommendation:** Acknowledge in a Consequences §Neutral note that the discovery value of the full 8-skill taxonomy is proportional to deployment coverage across skills; the cycle's primary task class may not exercise all eight, in which case the configuration surface is 16 slots but the empirical evidence accumulates in a subset.

---

**Issue 3.4 — ADR-016: The mechanism (c) "cannot be argued away" claim requires a scope condition**

- **Location:** ADR-016 §The five bounding mechanisms (c)
- **Claim:** "Deterministic outputs cannot be argued away by LLM consensus, so the feedback loop cannot drift on probabilistic noise."
- **Note:** The claim is drawn from Wisdom and Delusion of LLM Ensembles (arXiv:2510.21513) on CrossHair counterexample feedback in code-generation ensembles. The original paper's context is CrossHair generating formal counterexamples that falsify LLM-generated code — a case where the tool output is binary (the code passes or fails a formally-specified test). The generalization to "deterministic tool outputs" in llm-orc's context covers a broader class of script-model outputs, not all of which are binary-pass/fail. A script that outputs a structured analysis report (deterministic in the sense of not LLM-generated) is not an uncounterable anchor in the same way a formal counterexample is. The "cannot be argued away" property is strong for binary-verifiable deterministic outputs; it is weaker for deterministic-but-interpretable outputs.
- **Recommendation:** Scope the "cannot be argued away" claim to binary-verifiable deterministic outputs (e.g., test-pass/fail, lint-pass/fail, schema-validation-pass/fail). Note that deterministic outputs that are interpretable (structured reports, numerical scores) provide a weaker anchor — they are not LLM-judgment, but they can still be misinterpreted by downstream LLM consumers.

---

**Issue 3.5 — ADR-017: "Operator-readable diagnostics" referenced in ADR-012's Layer 4 typed error and in ADR-017's phantom_tool_call error are both described as typed errors, but the error taxonomy is not unified**

- **Location:** ADR-012 §Decision item 5; ADR-013 §Decision item 4; ADR-014 §Decision "Calibration verdict"; ADR-017 §Decision "Rejection"
- **Note:** Four ADRs independently specify typed errors (Layer 4 failure in ADR-012; write-gate violations in ADR-013; Abstain verdicts in ADR-014; phantom_tool_call in ADR-017) and reference the codebase precedent at commit `9f86d0b`. Each ADR names its error with different conventions: ADR-013 says "typed error" without naming the error class; ADR-014 says "typed abstention error" without naming the class; ADR-017 names `phantom_tool_call` explicitly. The ADRs are correct to reference a unified typed-error pattern, but the inconsistent specificity across them means BUILD engineers have to determine whether these are the same error class, subclasses, or distinct top-level types. This is not an argument error but a specification gap that affects downstream implementation.
- **Recommendation:** In ADR-017 (as the ADR most focused on the typed-error pattern), add a brief note specifying the error taxonomy relationship: "The `phantom_tool_call` error is a member of the same typed-error family as the Layer 4 circuit-breaker error (ADR-012), write-gate violations (ADR-013), and Abstain verdicts (ADR-014). BUILD should establish a shared error base class or error registry from which all four error types derive, consistent with the pattern established at commit `9f86d0b`."

---

## Section 2: Framing Audit

The framing audit examines what the source material made available that the primary documents did not choose, and what alternative framings the evidence supports.

---

### Question 1: What alternative framings did the evidence support?

**Alternative framing A: Risk-first ordering (address failure modes before adding capability mechanisms)**

The source material — particularly essay 005's Wave 1 + Wave 3.A findings and Khanal et al.'s universal non-improvement from episodic memory — supports an alternative framing where the ADR set is ordered by failure-mode severity rather than by architectural elaboration. Under this framing:

- ADR-017 (tool-call structural validation, addressing a documented and observed failure mode) would be the first ADR, operationalizing the most empirically-concrete failure mode the cycle surfaced.
- ADR-013's write-gate validation (addressing a failure mode documented in the literature, recommended but never implemented) would be second.
- ADR-012's circuit-breaker (addressing the pre-circuit-breaker failure history Claude Code documents) would be third.
- ADR-014's in-process calibration, ADR-015's tier escalation, and ADR-016's cross-layer channel would be later phases, as they add capability mechanisms rather than closing failure modes.

What would the document's central argument look like under this framing? The cycle would be read as a defensive design cycle — closing known failure modes in sequence — rather than a capability-expansion cycle. The argument would be: "the cycle's primary contribution is failure-mode closure at the structural level; the capability expansions (cross-layer calibration, tier escalation) are secondary, contingent on the failure-mode closures being in place."

Belief-mapping: "What would the reader need to believe for this framing to be right?" The reader would need to believe that: (a) the cycle's deployment is currently running with open failure modes (confabulation, specificity-loss, memory-poisoning) that are causing harm; and (b) adding calibration and tier-escalation mechanisms before closing the failure modes compounds the risk rather than independently adding value. Neither belief is well-supported by the current evidence — the cycle's deployment is pre-BUILD, so the failure modes are not yet causing production harm. The dominant framing (elaboration-by-evidence, capability expansion) is appropriate for the cycle's pre-BUILD state.

**Alternative framing B: Two-cycle strategy (close gaps in cycle 4; invest in novel mechanisms in cycle 5)**

The source material supports an alternative framing where ADR-012, ADR-013, and ADR-017 (all adoption-decision or documented-failure-mode territory) are the cycle's primary deliverables and ADR-014, ADR-015, ADR-016 (novel architectural territory) are deferred to Cycle 5 pending BUILD evidence from the adoption-decision ADRs.

What would the argument look like? The cycle's framing commitment is "elaboration-by-evidence." Under alternative framing B, "evidence" means BUILD-phase operational evidence, not DECIDE-phase synthetic-data validation. The deferred-candidate model (deferred-005) demonstrates that the corpus has a mechanism for principled deferral; the same model could be applied to ADR-014, 015, and 016 as a coherent strategy.

Belief-mapping: "What would the reader need to believe?" That the synthetic-data spike on mechanism (b) and the structural transfer audit on mechanism (d) are insufficient evidence for architectural decision-making, and that only BUILD-phase operational evidence qualifies. The corpus does not take this belief — it treats the DECIDE-phase spikes as adequate for conditional acceptance. The dominant framing is internally consistent; the alternative framing is a stronger application of the "validate before codifying" principle the practitioner articulated.

**Alternative framing C: Plexus-activation as the organizing axis**

Essay 005 explicitly states that the experience-accumulation arc (Framing D in the composite framing) partially supersedes pre-specifiable routing (Framing B) as Plexus activation enables retrieval-grounded selection. The source material supports a framing where the cycle's ADRs are organized by Plexus-conditional value:

- ADR-012, ADR-013, ADR-014, ADR-015, ADR-017 provide value in Plexus-absent mode.
- ADR-016's cross-layer calibration channel provides significantly more value when Plexus is active (because the post-hoc promotion tracking mechanism that relies on Plexus's cross-session persistence is what ADR-016's signals feed into).
- The ADR set's value proposition is stronger when read as a Plexus-absent baseline (ADRs 012–015, 017) with ADR-016 as the Plexus-adjacent elaboration.

What would the argument look like? The ADRs' sequencing and implementation priority would be governed by Plexus activation status rather than by architectural elaboration order. ADR-016's conditional-acceptance status has a natural Plexus dimension — its full value is Plexus-conditional — but neither ADR-016 nor the set as a whole makes this visible.

---

### Question 2: What truths were available but not featured?

**Underrepresented truth A: The cycle's autonomous-routing evidence gap remains fully open**

Essay 005 §"Open Questions and Scope-of-Claim" states explicitly: "Multi-iteration scale, fixture diversity [...] and N>1 trials are required before autonomous routing can be claimed reliably." The Sub-Q6 transfer-test — does context growth degrade ensemble-routing judgment at the session lengths the North-Star benchmark requires — "remains entirely open at cycle close."

This finding is largely absent from the ADR set. ADR-015's per-role tier-escalation router is described as a reliability mechanism for ensemble routing; its value depends on the orchestrator routing correctly to the cheap tier in the first place, which is precisely the autonomous-routing question the cycle did not settle. ADR-015's Consequences §Positive entry does not carry a caveat about the autonomous-routing evidence gap. The ADR assumes the routing path works reliably enough for the escalation to operate on top of it, but the cycle's own evidence base says routing reliability at multi-iteration scale is unknown.

This is not in the source material's research logs (the gap is explicitly documented in essay 005) but it is absent from the ADRs. An operator reading ADR-015 in isolation would not know that the routing surface it operates on is empirically unvalidated at the deployment scale the ADR targets.

**Underrepresented truth B: The Attention-MoA finding's implication for the orchestrator as aggregator**

Essay 005 §"Composition Shapes Per Layer" states: "On AlpacaEval 2.0 and MT-Bench (instruction-following benchmarks), the aggregation-agent quality drives a 12.82-percentage-point gap in outcomes; the cloud orchestrator quality is the bottleneck, not the member models." This means "cheap" must be calibrated against orchestrator quality at the aggregation moment.

This finding is absent from ADR-015 (per-role tier escalation) and ADR-014 (calibration gate). ADR-015 escalates the dispatched task's model tier; the orchestrator's aggregation role after receiving the escalated result is not addressed. If the orchestrator is the aggregation bottleneck for ensemble-of-ensembles work, escalating the member-model tier without addressing the orchestrator-as-aggregator quality gap may not produce the accuracy gains ADR-015's literature evidence (OI-MAS) attributes to tier escalation. OI-MAS's +12.88% accuracy finding is for a system where the orchestrator is a routing agent, not primarily an aggregation agent; the Attention-MoA finding suggests these may have different bottleneck profiles.

**Underrepresented truth C: The MOP paradox's alternative reading**

Essay 005 §"Long-Horizon Reliability" surfaces an alternative reading of Khanal et al.'s meltdown-on-paradox finding: "An alternative reading available in the same data is worth recording: the cheap orchestrator's reliability advantage on long-horizon may be partly due to capability-bounded failure (failing before the meltdown threshold) rather than architectural reliability compensation." This alternative reading — cheap models fail earlier and more uniformly, not less — is present in the essay but absent from the ADRs. ADR-015's rejected alternative (a) cites the heterogeneous-role-staffing evidence (OI-MAS, SC-MAS, MasRouter) but not the Khanal et al. meltdown data. The MOP finding is relevant to ADR-015's framing of tier-escalation as a capability expansion — if the capability-bounded-failure reading is correct, escalating to a more capable tier on Reflect verdicts might sometimes *increase* meltdown risk rather than resolve the low-confidence case. This alternative reading was available in the source material and was not surfaced in ADR-015's rejected alternatives or Consequences.

---

### Question 3: What would change if the dominant framing were inverted?

The dominant framing across the ADR set is **elaboration-by-evidence within the existing layer structure**, operationalizing essay 005's architectural verdict that long-horizon reliability infrastructure fits within existing layers without requiring architectural reorganization.

The inverted framing is **evidence of insufficient elaboration** — the mechanisms are architecturally coherent but too thin for the North-Star benchmark's demands, and the elaboration-within-layers framing is a Procrustean fit that constrains the design surface prematurely.

Under the inverted framing:

- ADR-016's conditional-acceptance status becomes the dominant signal rather than the exception. The conditionality means the cycle's most novel architectural decision ships without empirical validation; the inverted reading would say the cycle has not resolved its central architectural question — it has only committed to monitoring it.

- ADR-014's trajectory-feature extraction adds "compute cost at every dispatch" (acknowledged in Consequences §Negative). Under the inverted framing, the accumulating compute cost of full HTC feature extraction on every dispatch is the dominant concern, not the calibration signal quality. The "no token-limit pre-optimization on free local models" guidance that licensed the full feature set may not apply to cost-bearing deployments; the inverted framing would ask whether the calibration benefit justifies the compute cost at the deployment scale the North-Star benchmark requires.

- ADR-015's 16-slot configuration surface becomes the dominant friction concern rather than a discovery-trade. Under the inverted framing, configuration-burden-without-deployment-evidence is the structural failure mode, not a justified friction-for-discovery trade. The practitioner accepted this trade, but the inverted framing surfaces it as the dominant cost of the elaboration-by-evidence approach.

- ADR-013's cluster-conditional applicability rule becomes the primary complexity surface. Under the inverted framing, the session-start cluster-determination requirement is the mechanism that makes the elaboration-by-evidence approach least workable — it requires operators to pre-classify sessions at a time when the cycle's own evidence (the Sub-Q6 gap) says the session's decision-class profile cannot be reliably predicted.

What would the document need to address if it took the inverted framing seriously? The ADR set would need to engage with the question: "What is the failure mode if the elaboration-within-layers approach produces mechanisms that are too thin, and when would we know?" The falsification trigger in ADR-016 partially answers this, but only for mechanism (b) and (d). A set-level falsification criterion — "what evidence would cause us to revisit the elaboration-by-evidence framing for the set as a whole" — is absent.

---

### Framing Issues

**P1 — Consequential omission: autonomous-routing evidence gap not carried into ADR-015**

- **Location:** ADR-015 (entire document, specifically §Consequences §Positive and §Rejected alternatives)
- **Source material:** Essay 005 §"Open Questions and Scope-of-Claim," specifically: "Autonomous routing remains partially validated [...] Multi-iteration scale, fixture diversity [...] and N>1 trials are required before autonomous routing can be claimed reliably."
- **Omission:** ADR-015 presents the tier-escalation router as a mechanism that improves routing reliability. The mechanism's value depends on the underlying routing being reliable enough for escalation to operate on top of it. The cycle's evidence base says multi-iteration routing reliability is unknown. This omission is consequential: an operator deploying ADR-015 without reading essay 005 would not know that the routing surface the escalation operates on is empirically unvalidated at deployment scale. The source material explicitly documents this gap; the ADR does not carry it.
- **Recommendation:** Add a Consequences §Neutral entry: "ADR-015's escalation mechanism operates on the orchestrator's routing decisions. Cycle 4's behavioral spike (Wave 3.A Trial 1) validated the dispatch path at N=1 on a single fixture. Multi-iteration routing reliability at the North-Star benchmark's session length is not yet established (essay 005 §Open Questions). Deployment evidence on routing reliability is a prerequisite for interpreting escalation-rate calibration evidence."

**P2 — Underrepresented alternative: Plexus-conditional framing of ADR-016's value**

- **Location:** ADR-016 §Consequences, §Rejected alternatives
- **Source material:** Essay 005's composite framing element D (experience-accumulation arc enabled by Plexus activation); ADR-014's composition note that "Post-hoc promotion tracking remains the cross-session stabilization mechanism (under Plexus per AS-5, session-scoped under stateless mode per ADR-007)."
- **Issue:** ADR-016's value proposition is partially Plexus-conditional: the post-hoc promotion tracking that receives signals from the cross-layer channel (the signal data that persists across sessions for stabilization per AS-5) requires Plexus. In Plexus-absent mode, the cross-layer channel provides in-session calibration signals but not the cross-session stabilization value that makes the architecture's experience-accumulation arc operate. ADR-016 presents the channel's value without this scope condition. Essay 005 is explicit that "performance gains [from retrieval-grounded selection] are contingent on Plexus enablement"; ADR-016's value proposition should carry a parallel scope condition.
- **Recommendation:** Add a Consequences §Neutral note specifying: "Cross-layer calibration signals contribute to in-session dispatch decisions in both Plexus-active and Plexus-absent modes. The post-hoc promotion tracking (ADR-007) that uses cross-session signal accumulation for stabilization (AS-5) requires Plexus; in Plexus-absent mode, calibration state is session-scoped per ADR-007. The channel's cross-session value proposition is Plexus-conditional."

**P2 — Underrepresented alternative: the Attention-MoA orchestrator-as-aggregator finding and its relevance to ADR-015**

- **Location:** ADR-015 §Context, §Rejected alternatives
- **Source material:** Essay 005 §"Composition Shapes Per Layer": "the aggregation-agent quality drives a 12.82-percentage-point gap in outcomes; the cloud orchestrator quality is the bottleneck, not the member models."
- **Issue:** ADR-015 escalates the dispatched task's model tier on Reflect verdicts. The Attention-MoA finding suggests the orchestrator's aggregation quality — not the member model quality — is the bottleneck for ensembles-of-ensembles on instruction-following tasks. If the orchestrator is the aggregation bottleneck, escalating the member model while keeping the orchestrator at its configured tier (per ADR-011's session-boundary constraint) may not produce the accuracy gains the OI-MAS evidence attributes to tier escalation in a different architecture. The ADR does not engage with this tension. The practitioner's guidance is "discover which capability dimensions matter" — the Attention-MoA finding is a specific capability dimension (orchestrator-at-aggregation) the ADR's discovery apparatus does not surface.
- **Recommendation:** Add to ADR-015's Consequences §Neutral: "The Attention-MoA finding (essay 005 §'Composition Shapes Per Layer') suggests orchestrator quality at the aggregation moment drives accuracy on instruction-following tasks — a different bottleneck from the member-model quality ADR-015 addresses. Discovery evidence should track whether escalation gains are concentrated in tasks where member-model quality is the bottleneck (code review, verification) or whether orchestrator-aggregation is the binding constraint for the cycle's task class."

**P3 — Minor framing imbalance: ADR-012's adoption framing underrepresents the adaptation choices**

- **Location:** ADR-012 §Provenance check
- **Issue:** The Provenance check correctly flags the operator-readable error surface for Layer 4 and the typed-error coupling as drafting-time synthesis. However, the ADR's §Context and §Decision present all five layers as direct adoption from Claude Code's specification. The threshold defaults (50,000-character, 60-minute, 12,288-token, 3-failure) and the nine-section template structure are adopted unchanged, but the application context (llm-orc's orchestrator sessions, which have different workload shapes than Claude Code's coding sessions) is not Claude Code's context. The "adoption" framing implies fit-by-default; the honest framing is "adopted mechanism, unvalidated defaults."
- **Recommendation:** Add a one-sentence note in §Provenance check: "No llm-orc-specific workload data informs the threshold defaults; they are Claude Code's operational values applied as starting points. BUILD-phase tuning against llm-orc's actual dispatch patterns is the validation mechanism."
