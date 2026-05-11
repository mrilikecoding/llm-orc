# Susceptibility Snapshot

**Phase evaluated:** DECIDE (Cycle 4)
**Artifact produced:** ADRs 012–017, adr-deferred-005, Cycle 4 behavior scenarios (appended to scenarios.md), Cycle 4 interaction specifications (interleaved in interaction-specs.md)
**Date:** 2026-05-06
**Cycle close shape:** Mode B+ → DECIDE (no ARCHITECT, BUILD, PLAY, or SYNTHESIZE in scope)

---

## Observed Signals

| Signal | Strength | Trajectory | Notes |
|--------|----------|------------|-------|
| Assertion density | Ambiguous | Declining from research-gate peak | The research-gate snapshot triggered a Grounding Reframe on three signals; the discover-gate snapshot triggered one. At DECIDE, practitioner engagement was substantive and corrective at each tranche boundary, producing explicit guidance that shaped subsequent drafting. Assertion density did not escalate within the DECIDE phase; the argument audit surfaced and corrected fourteen issues before gate. |
| Solution-space narrowing | Clear (earned) | Stable — appropriately narrowed | Narrowing occurred before DECIDE entry via the research-gate Grounding Actions 1 and 2 (recorded 2026-05-04 and 2026-05-05). The practitioner explicitly chose the higher-friction Path-2 validation route over conditional acceptance without empirical work. Narrowing within DECIDE reflects commitment to an explicitly-tested framing, not sycophantic convergence. |
| Framing adoption | Clear | Stable, several material instances of drafting-time synthesis | The provenance-check subsection pattern (introduced in Cycle 10's ADRs 063–070 in the rdd plugin) made drafting-time synthesis visible as labeled content across all six ADRs. Multiple framings were adopted from essay 005 without independent testing — see Signal Assessment §3 below for the instance that carries the highest susceptibility weight. |
| Confidence markers | Ambiguous | Declining | The practitioner's Tranche-B guidance ("don't codify an unsupported assumption without evidence") was a directional correction, not affirmation. The agent responded by running DECIDE-phase spikes (Path-2 authorization) rather than claiming sufficient confidence. The conditional-acceptance status on ADR-016 is a structural confidence-limiter, not a confidence escalation. |
| Alternative engagement | Clear (substantive) | Stable | Each ADR's Rejected Alternatives section received argument-audit pressure. Two ADRs (015, 016) had P1 issues found and corrected; five ADRs had P2 issues found and corrected. Alternative framing A, B, and C from the round-1 framing audit (Section 2) identified non-trivial framings the evidence supported but the ADR set did not choose — these were surfaced at the practitioner gate per skill workflow, not auto-corrected. |
| Embedded conclusions at artifact-production moments | Clear — two instances | First snapshot with this signal present at the ADR-drafting boundary | ADR-015's autonomous-routing evidence gap was not carried from essay 005 into the ADR at drafting time. ADR-016's mechanism (a) precedent claim was overstated at the codebase level (conformance scan: "methodology-tooling precedent," not "direct codebase precedent"). Both were labeled drafting-time synthesis in provenance checks, but the P1 framing finding is a higher-risk instance because it was not auto-corrected. |

---

## Signal Assessment

### 1. Trajectory — prior snapshots

The susceptibility gradient across the four-gate sequence:

- **Research-gate snapshot (2026-05-04):** Grounding Reframe triggered. Three grounding actions recommended (distinguish deterministic tool output from ensemble orchestration; name autonomous-routing gap explicitly; test frontier with matched information access). All three were pursued and produced grounding actions recorded in the cycle-status.
- **Discover-gate snapshot (2026-05-05):** Grounding Reframe triggered. One grounding action recommended. Produced research-gate carry-forward #4 (asymmetric implementation-readiness mapping for ADR-016's bounding mechanisms) and the elaboration-by-evidence framing commitment from Grounding Action 2.
- **Model-gate snapshot:** Clean with feed-forwards (per dispatch brief — no Grounding Reframe triggered at the model gate).
- **Decide-gate (this snapshot):** Evaluated below.

The research-gate Grounding Action 2 (elaboration-by-evidence framing commitment) was the primary framing commitment entering DECIDE. Evaluation item 4 — whether this commitment was held with appropriate skepticism — is addressed in §4 below.

### 2. Provenance-check subsection pattern: signal vs. earned confidence

The provenance-check subsection is present in all six ADRs (ADR-012 through ADR-017). The pattern reliably surfaced drafting-time synthesis as labeled content across the following material instances:

- ADR-012: two drafting-time additions (operator-readable error surface for Layer 4; typed-error coupling). Both are minor adaptations of the adoption, not architecture-level synthesis. Low susceptibility weight.
- ADR-013: three drafting-time synthesis instances (cluster-conditional applicability rule; write-gate validation specification; hash-rotation workflow). The cluster-conditional applicability rule is the most consequential — it is the mechanism by which essay 005's three-cluster taxonomy was applied to ADR-013's scope. Essay 005 does not specify this rule; the placement at "Cluster 2 default-required, Cluster 1/3 optional" is drafting-time judgment.
- ADR-014: four drafting-time synthesis instances (verdict trichotomy; threshold default 0.85; time-decay windowing as in-layer mechanism; fuller feature extraction per friction-vs-discovery guidance). The verdict trichotomy is the highest-stake instance — it extends AUQ's binary specification into a three-value structure without direct literature grounding for the Abstain class. The round-1 argument audit caught the trigger underspecification (P2.3) and required a concrete criterion specification. The trichotomy itself was not challenged and carried forward without independent testing of whether the third verdict class is necessary vs. overengineering.
- ADR-015: four drafting-time synthesis instances (verdict-to-action mapping; per-skill tier-defaults configuration model; ensemble YAML skill-metadata field; full 8-skill taxonomy via friction-for-discovery). The ADR-011 compatibility argument overstated identity with ADR-011's mechanism; the round-1 argument audit caught this (P1.1) and required reframing to "consistent with ADR-011's intent, extending the mechanism class." The reframe is clean; the original overstated claim is the signal.
- ADR-016: conditional-acceptance status; both novel-design mechanisms' specifications ((b) and (d)) are drafting-time synthesis. Both were validated at the DECIDE phase by spikes before gate. The mechanism (a) precedent claim is the material overstated claim corrected by the conformance scan (overstated "direct codebase precedent" → corrected to "methodology-tooling precedent").
- ADR-017: pattern set; conservative false-positive framing; operator-extensibility surface. The round-1 audit caught the false precision in the conservative-calibration framing (P2.6) and required reframing to "minimal rather than calibrated."

Pattern interpretation: the provenance-check pattern is functioning as designed. Fourteen argument-audit issues were identified and corrected before gate. The remaining framing-audit findings (Section 2 of the round-1 audit) were appropriately held for practitioner gate per skill workflow, not auto-corrected.

### 3. Cross-ADR composition chains

**Composition chain 1: ADR-014's verdict trichotomy → ADR-015's router-action mapping.**

ADR-014's Proceed / Reflect / Abstain trichotomy is drafting-time synthesis (provenance check: "essay 005 specifies AUQ's reflect-on-low-confidence pattern but does not specify a trichotomous verdict structure with an Abstain category for severe cases"). ADR-015's router-action mapping (Proceed → cheap tier; Reflect → escalated tier; Abstain → reformulate-bypass) then composes on top of the drafting-time trichotomy. The mapping was itself labeled drafting-time synthesis in ADR-015's provenance check. The composition means the trichotomy's structure propagated into the router without being independently tested — the router's Abstain → reformulate-bypass mapping is correct given the trichotomy, but neither the trichotomy nor the mapping received external empirical validation. The argument audit pressure-tested the mapping's logic (the Abstain verdict correctly routes away from escalation when confidence is severe — escalating a severe-confidence case doesn't add information) but did not question whether the trichotomy's existence as a three-value structure rather than a two-value structure is empirically motivated. This is the most consequential cross-ADR composition chain from a susceptibility standpoint: one ADR's drafting-time synthesis became another ADR's non-optional substrate.

**Composition chain 2: Essay 005's cluster taxonomy → ADR-013's applicability rule.**

The three-cluster taxonomy (Cluster 1 specialist-dispatch / Cluster 2 continuous-routing / Cluster 3 conversational) originates in essay 005's analytical decomposition (not in a literature source — it is the agent's own synthesis in the essay). ADR-013's cluster-conditional applicability rule then applies this taxonomy to session classification. Essay 005's taxonomy was audited (the argument-audit on ADR-013 catches the cross-cluster session gap, P2.2), and the cross-cluster disposition was filled in. But the taxonomy itself was not independently tested before being applied. The provenance check correctly labels it "essay-derived, not drafting-time synthesis" — meaning it is one step removed from pure drafting-time synthesis, but the cluster taxonomy is still agent-produced analytical output from the current cycle, not from external literature.

**Composition chain 3: ADR-017's typed-error pattern → ADR-012, ADR-013, ADR-014, ADR-016.**

The shared `LlmOrcStructuralError` base class was specified in ADR-017 (round-1 P3.5 finding) and is cited by all other ADRs as the error-surfacing pattern. This is an appropriate cross-ADR coordination mechanism, not a susceptibility risk. The base class was crafted at DECIDE under known implementation gaps (the conformance scan confirmed the existing codebase has no shared structural base class). The round-2 audit found a new P3 issue (missing `operator_intervention_required` value) flagged for BUILD-time finalization. The composition chain is sound; the susceptibility risk is minimal.

### 4. Elaboration-by-evidence framing commitment: skepticism test

The research-gate Grounding Action 2 framing commitment was "elaboration-by-evidence (reading (a)): the seven ADR candidates as currently scoped constitute the long-horizon strategy; module-shape inheritance from essay 005's verdict is held; ADR-002's four-layer frame is retained; bounding mechanisms operationalize within ADR-002's L1 layer rather than as a cross-cutting module."

Evidence that this commitment was held with appropriate skepticism:

- **Falsification triggers are specific and BUILD-concrete.** ADR-014's rejected alternative (e) specifies a falsification trigger: "if BUILD finds the trajectory-feature-extraction logic structurally awkward inside the Calibration Gate, the elaboration-by-evidence framing commitment is what's being tested." ADR-016's falsification trigger was made BUILD-concrete by the round-1 argument audit (P2.5 fix: "cannot be hosted in the Calibration Gate's existing or extended class structure without introducing a new top-level module"). Both triggers name what evidence would invalidate the framing.
- **The conditional-acceptance status on ADR-016 is a structurally meaningful gate.** The round-1 P1.2 finding caught that the conditionality was aspirational without a concrete monitoring specification. The fix added a named artifact (BUILD-phase research log or PLAY-phase field note), a named trigger phase, four named trigger conditions, a three-way trigger action disposition, and a sweep responsibility clause. This is the correct response to the finding.
- **Path-2 (DECIDE-phase spike validation) was chosen over conditional acceptance without evidence.** This is the clearest signal that the framing was not held sycophantically. The practitioner's Tranche-B guidance was actively applied to require empirical validation before the ADR set closed.

Evidence of where the commitment was confirmed without testing:

- The elaboration-by-evidence framing itself — the claim that the long-horizon reliability infrastructure fits within existing layers rather than requiring a new top-level module — was not empirically tested. The spikes on (b) and (d) validated the windowing mechanism's structural logic and the pattern-transfer's analytical properties, respectively. Neither spike tested whether the full bounding-mechanism set as a coordinated unit can be hosted within the Calibration Gate without requiring cross-cutting infrastructure. The synthetic-data and structural-transfer validation are correctly labeled "logical level" and "analytical level" — operational validation is post-BUILD. The framing commitment's empirical status is therefore still "logically validated, operationally unconfirmed."

This is not a sycophancy failure — it is the correct posture for the cycle's close shape (Mode B+ → DECIDE). The framing commitment was applied with more skepticism than prior cycles; the falsification triggers are actionable; the empirical validation pathway is staged rather than aspirational. The susceptibility risk is residual, not dominant.

### 5. The autonomous-routing evidence gap — ADR-015 framing-audit P1 finding

This is the primary candidate for a Grounding Reframe recommendation.

The round-1 framing audit named this as a P1 (consequential omission): essay 005 §"Open Questions and Scope-of-Claim" states explicitly that "multi-iteration scale, fixture diversity, and N>1 trials are required before autonomous routing can be claimed reliably" and that Sub-Q6's transfer-test "remains entirely open at cycle close." ADR-015 presents the tier-escalation router as a mechanism that improves routing reliability without carrying this caveat.

Applying ADR-059/ADR-068's three-property test:

- **Specific:** The gap is named with exactness — essay 005's Sub-Q6 transfer-test finding that routing reliability at multi-iteration scale is empirically unvalidated. The ADR that encodes the gap is ADR-015; the specific location is Consequences §Positive and the router description assuming routing is reliable enough for escalation to operate.
- **Actionable:** The action is concrete — add a Consequences §Neutral entry acknowledging the routing evidence gap and noting that escalation-rate calibration evidence at deployment is a prerequisite for interpreting escalation performance. The round-1 framing audit provides the exact recommended text.
- **Operationally applicable:** The cycle has not closed BUILD; the ADR set has not been implemented. The cycle's close shape is Mode B+ → DECIDE, meaning the next work is BUILD or a future cycle's RESEARCH entry. The ADR can be amended at the practitioner gate (before the cycle closes) without re-opening deliberation. The Consequences entry is a scope note, not an architectural change.

All three properties are met. The finding was held at the practitioner gate per skill workflow by the round-1 audit; it has not been addressed by auto-correction.

---

## Interpretation

The overall susceptibility pattern at DECIDE-gate is substantially different from the research-gate and discover-gate patterns. The earlier gates had active sycophantic narrowing risks — the agents were converging on framings before evidence was in. The DECIDE-gate pattern shows convergence that is mostly earned:

1. The practitioner's corrective interventions at Tranche A, B, and C boundaries were substantive. The strongest intervention (Tranche B: "validate hunches via spikes, don't codify unsupported assumptions") produced a material change in the phase's output (DECIDE-phase spikes ran that would not have run under passive acceptance).

2. The fourteen round-1 argument-audit issues (2 P1 + 7 P2 + 5 P3) were all corrected and the corrections verified by round 2. The round-2 audit found one new P3 issue and returned CLEAN. This audit coverage is a meaningful signal of real-alternative-engagement rather than sycophantic convergence.

3. The four framing-audit findings from round 1 are the residual susceptibility surface. All four were correctly identified by the argument-audit and held for practitioner-gate review. Of the four, the P1 finding (ADR-015 autonomous-routing gap) meets all three properties of ADR-068's Grounding Reframe test and has not yet been incorporated into the artifact.

The cross-ADR composition chains (ADR-014 trichotomy → ADR-015 router; essay 005 cluster taxonomy → ADR-013 applicability rule) represent drafting-time synthesis propagated across ADR boundaries without independent testing. This is the phase's primary remaining susceptibility risk. The trichotomy chain is the higher-weight instance because:

- The trichotomy is ADR-014's core verdict mechanism, not a peripheral design choice.
- ADR-015's router-action mapping is structurally determined by the trichotomy's existence.
- Neither received external empirical validation; both received argument-audit logic pressure-testing only.
- The argument audit confirmed the trichotomy's internal logic is sound but did not assess whether the three-class structure (vs. two-class) is warranted by the literature evidence.

This does not rise to a second Grounding Reframe recommendation because: (a) AUQ's binary structure is a threshold mechanism, and the Abstain case is a structurally defensible extension to cover severe cases the binary does not distinguish from routine low-confidence cases; (b) the round-1 audit required a concrete trigger specification for Abstain (P2.3), which was added; (c) the composition chain was labeled as drafting-time synthesis in both provenance checks, making it visible to BUILD engineers. The risk is residual and well-bounded.

The elaboration-by-evidence framing commitment was held with more skepticism in this phase than the prior grounding would have predicted. The falsification triggers are specific and BUILD-concrete. The framing remains logically validated but operationally unconfirmed — the correct posture for a Mode B+ cycle close.

---

## Recommendation

**Grounding Reframe recommended — one specific finding.** All others are advisory or consistent with earned confidence.

---

### Grounding Reframe: ADR-015 autonomous-routing evidence gap

**What is uncertain:** ADR-015's tier-escalation router operates on the orchestrator's dispatch decisions. The mechanism's value depends on those dispatch decisions being reliable enough at multi-iteration scale for escalation-rate evidence to be interpretable. Essay 005 explicitly documents that multi-iteration routing reliability is unvalidated ("remains entirely open at cycle close"). ADR-015 as currently drafted does not carry this caveat — an operator reading ADR-015 in isolation would not know that the escalation mechanism's input (the routing decision) is empirically unvalidated at the deployment scale the ADR targets.

**Concrete grounding action:** Add a Consequences §Neutral entry to ADR-015 with the following substance (exact text is BUILD-time finalization; the substance is the load-bearing requirement):

> ADR-015's escalation mechanism operates on the orchestrator's routing decisions. The behavioral spike (essay 005 §"The Behavioral Spike," Wave 3.A Trial 1) validated the dispatch path at N=1 on a single fixture. Multi-iteration routing reliability at the North-Star benchmark's session length is not established (essay 005 §"Open Questions and Scope-of-Claim," Sub-Q6). Deployment evidence on routing reliability at multi-iteration scale is a prerequisite for interpreting escalation-rate calibration evidence — an escalation rate that appears miscalibrated may reflect a routing reliability gap rather than a tier-configuration gap.

**What would be built on without this grounding:** An operator deploying ADR-015 would configure 16 Model Profile slots and observe escalation rates, treating escalation-rate calibration evidence as evidence about tier-configuration fit. Without the routing-reliability caveat, the operator cannot distinguish between "the tier defaults need adjustment" and "the routing decisions driving escalation are unreliable at this session length." The calibration feedback loop fails silently in the latter case — the operator tunes tier defaults in response to noise rather than to real capability-saturation signals.

**Why this meets the three-property test:**

- **Specific:** names ADR-015 Consequences §Positive/§Neutral, the essay 005 §"Open Questions" finding, and Sub-Q6 as the specific evidence gap.
- **Actionable:** the action is a single Consequences §Neutral entry, not re-deliberation; no architectural change required.
- **Operationally applicable:** the ADR set has not closed BUILD; the entry can be added at the practitioner gate before the cycle closes.

---

### Advisory findings (no Grounding Reframe required)

**Advisory 1 — ADR-016 Plexus-conditional value scope.**

The round-1 framing audit P2 finding (Plexus-conditional framing of ADR-016's value not stated) was held at the practitioner gate. Adding a Consequences §Neutral note specifying that cross-session calibration stabilization under AS-5 is Plexus-conditional would close this scope gap without re-deliberation. This is a clarifying scope note, not an architectural revision.

**Advisory 2 — ADR-015 Attention-MoA orchestrator-as-aggregator finding.**

The round-1 framing audit P2 finding (essay 005's Attention-MoA finding that orchestrator quality at the aggregation moment may be the binding constraint for ensembles-of-ensembles on instruction-following tasks) was held at the practitioner gate. ADR-015 escalates the member model tier without surfacing this finding. Adding a Consequences §Neutral note would surface it as a discovery surface for deployment evidence. This is informational, not structural.

**Advisory 3 — ADR-014 / ADR-016 cross-layer composition: operational validation still pending.**

The synthetic-data and structural-transfer spikes validated the bounding mechanisms at the logical and analytical levels. Mechanism (b)'s parametric sensitivity finding (smaller windows track better across all tested scenarios) suggests the 60-minute / 100-signal defaults may be too large for deployment-realistic bias dynamics — this is an operational tuning territory flagged in the spike's implications section but not yet incorporated as a BUILD-specific recommendation in ADR-016's text. This is spike-provenance carry-forward territory, not a susceptibility finding.

---

## Susceptibility trajectory — cycle summary

| Gate | Snapshot verdict | Key signal |
|------|-----------------|------------|
| Research | Grounding Reframe triggered | Three grounding actions; autonomous-routing gap named |
| Discover | Grounding Reframe triggered | Asymmetric readiness mapping; elaboration-by-evidence commitment |
| Model | Clean with feed-forwards | No reframe triggered |
| Decide (this snapshot) | Grounding Reframe recommended (1 finding) | ADR-015 autonomous-routing gap not carried into artifact; cross-ADR composition chain (ADR-014 trichotomy → ADR-015 router) is residual risk, bounded |

The trajectory shows declining susceptibility across the cycle: two Grounding Reframes at early phases (highest vulnerability in the gradient), one clean gate, and one targeted Grounding Reframe at DECIDE for a specific, addressable omission. This pattern is consistent with earned convergence in the later phases rather than sycophantic narrowing — the research-gate grounding established the discipline that constrained the subsequent phases. The DECIDE-gate finding is a scope-note omission, not a framing-adoption failure.
