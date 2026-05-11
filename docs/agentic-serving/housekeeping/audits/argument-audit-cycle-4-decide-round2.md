# Argument Audit Report — Cycle 4 DECIDE-Phase ADRs (Round 2)

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

**Round 1 audit report:** `docs/agentic-serving/housekeeping/audits/argument-audit-cycle-4-decide.md`

**Date:** 2026-05-06

---

## Round 2 Scope

This round verifies that each Round 1 fix resolves the cited issue without introducing new logical gaps, overreach, or framing issues. Round 1 found 2 P1 + 7 P2 + 5 P3 argument-audit issues. All fourteen have been addressed in the revised ADRs.

The Round 1 framing-audit findings (Section 2) are not assessed in this round — those are surfaced at the practitioner gate per skill workflow, not auto-corrected.

---

## Section 1: Argument Audit

### Round 1 Fix Verification

---

#### P1 Fix 1.1 — ADR-015: ADR-011 compatibility argument

**Fix applied:** Context section now reads "consistent with ADR-011's intent and extending the mechanism class" rather than "implementation of ADR-011's existing pattern." A justification for L2-interposition over composed-ensemble placement is provided: the calibration verdict is per-dispatch and changes within a session; placing the tier decision inside an ensemble would either duplicate calibration logic across every tier-aware ensemble (configuration burden) or require operator-authored ensembles to consume calibration verdicts directly (coupling that crosses the L0-L1 boundary). The Provenance check records the drafting-time precision applied to essay 005's overstated framing.

**Verification:** The fix resolves the circularity. The original ADR-011 text — "tiered behavior is expressed as a composed ensemble invokable by the orchestrator" — is now acknowledged as the anticipated mechanism that ADR-015 extends rather than implements. The compatibility argument now rests on the preserved architectural property (orchestrator's LLM session-boundary-event scoped; tiered behavior not baked into orchestrator reasoning) rather than on identity with ADR-011's specific mechanism. The L2-interposition justification is concrete and avoids the elision the round 1 audit cited.

**New gap check:** The justification for L2-interposition introduces the claim that placing the tier decision inside an ensemble "would require operator-authored ensembles to consume calibration verdicts directly (coupling that crosses the L0-L1 boundary the layering rule constrains)." This claim deserves a brief examination: would a composed ensemble implementing tier decisions internally actually require cross-boundary coupling? ADR-011's contemplated `triage-route` ensemble would implement tier logic internally — it would not need to read ADR-014's calibration verdict; it would make its own confidence assessment. The L2-interposition argument is not "coupling crosses L0-L1" (which is imprecise) but rather "centralization vs. per-ensemble duplication": every tier-aware ensemble would need to replicate calibration logic rather than consuming one shared verdict. The L2 argument holds on configuration-burden grounds; the boundary-crossing claim as stated is slightly imprecise. This does not reopen the P1 issue — the compatibility argument is sound and the justification valid — but the boundary-crossing framing should be read as configuration-coupling rather than architectural-boundary-crossing. **No new issue requiring correction; observation noted.**

**Status: RESOLVED.**

---

#### P1 Fix 1.2 — ADR-016: Concrete monitoring specification

**Fix applied:** A "Concrete monitoring specification (post-spike-validation)" subsection has been added to the Empirical validation pathway section, specifying: trigger artifact (BUILD-phase research log or PLAY-phase field note on non-trivial fixture), trigger phase (earlier of first BUILD dispatch exercising the channel end-to-end or first PLAY scenario involving multi-iteration session continuation), trigger conditions for re-deliberation (four named failure modes: residual bias accumulation uncorrectable by parameter tuning, mechanism (d) failing to produce actionable diagnostics, either mechanism requiring a top-level module outside ADR-002's L0-L3 structure, (b)/(d) coupling failure), trigger action (practitioner review with three-way disposition: full acceptance, preserved conditionality with notes, falsification trigger fires), and sweep responsibility (practitioner at end of each cycle exercising the cross-layer channel, with absence of the row in the cycle-status Phase Status table constituting structural evidence the channel was not exercised).

**Verification:** The fix directly addresses the structural gap the round 1 audit identified: the conditional-acceptance status now has a named artifact, a named phase, a named responsible party, and a named action that constitutes re-deliberation. The sweep responsibility clause is particularly sound — it converts the conditionality from a monitoring aspiration to a recurring structural obligation with a falsification signal (the absent row). The four trigger conditions are concrete enough to be actionable.

**New gap check:** The monitoring specification creates a dependency on ADR-068's pattern for "monitoring-conditional decisions," cited as the reference for the sweep responsibility. ADR-068 is from the rdd plugin corpus rather than the agentic-serving corpus. The reference is legitimate (the rdd methodology's conformance ADRs govern the cycle's process discipline), but a reader of the agentic-serving corpus without access to the rdd plugin ADRs cannot verify the pattern being referenced. This is a minor discoverability concern rather than a logical gap — the mechanism is fully specified in ADR-016's text; ADR-068 is cited as a precedent, not as a dependency. **No new issue requiring correction.**

**Status: RESOLVED.**

---

#### P2 Fix 2.1 — ADR-012: Defaults provenance addition

**Fix applied:** Provenance check now includes an explicit "Defaults provenance" note stating that the threshold defaults are Claude Code's published operational values and that "No llm-orc-specific workload data informs these defaults." The note explicitly flags that llm-orc's dispatch-frequency, tool-output-shape, and session-cadence profiles may differ, and that operational deployment should validate the defaults before treating them as calibrated. The Decision section's existing "operationally tunable" caveat is noted as the load-bearing acknowledgement.

**Verification:** The fix resolves the scope-accuracy issue. The ADR now honestly distinguishes between "adopted from Claude Code" and "calibrated for llm-orc." The Layer 4 circuit-breaker reset mechanism (P3 fix 3.1) is also confirmed present in the Decision text: "Circuit-breaker state is automatically reset at session start."

**Status: RESOLVED.**

---

#### P2 Fix 2.2 — ADR-013: Cross-cluster sessions disposition

**Fix applied:** A "Cross-cluster sessions" subsection has been added under cluster-conditional applicability, presenting three explicit dispositions: (i) default to required when ambiguous (adopted as BUILD-time starting point), (ii) mid-session reclassification via explicit declaration mechanism, (iii) always required unconditionally. The rationale for disposition (i) is documented: cost is friction in mixed-cluster sessions; benefit is structural-non-regression and narrative-continuity cover the load-bearing Cluster 2 sub-portions.

**Verification:** The fix directly addresses the unspecified failure mode the round 1 audit identified. The cross-cluster session shape — which the North-Star benchmark explicitly requires — is now handled by disposition (i) with a clear fallback path to (ii) if BUILD evidence shows reclassification is useful, and (iii) if cluster-determination proves unworkable. The three-option structure is well-formed: each disposition is named, its cost and benefit identified, and the BUILD-time starting point is explicit.

**New gap check:** Disposition (ii) — mid-session reclassification — is described as "an explicit declaration mechanism" but the mechanism itself is not specified. The text says "Transitioning into Cluster 2 mid-session activates the artifact set; transitioning out of Cluster 2 deactivates it (existing artifacts persist; new writes pause)." The declarant is unstated: is the operator, the orchestrator, or the Session Registry the entity that makes the declaration? For a disposition flagged as an available BUILD-time refinement, this level of underspecification is acceptable — the ADR correctly defers the mechanism to BUILD rather than specifying it prematurely. The pattern is consistent with the ADR's general posture of specifying constraint shapes and leaving BUILD-concrete implementation to BUILD. **No new issue requiring correction at this phase.**

**Status: RESOLVED.**

---

#### P2 Fix 2.3 — ADR-014: Abstain verdict trigger criteria

**Fix applied:** The Abstain verdict now specifies three concrete criteria: (1) entropy collapse — token-level entropy drops more than 1.5 standard deviations below the trajectory's running mean, with operationally tunable threshold; (2) post-hoc result-check hard failure — ADR-007's first-N calibration produces a verdict-incompatible outcome with non-recoverable error; (3) multiple drift-detection criteria simultaneously exceed thresholds when ADR-016's mechanism (d) is active and reporting severe drift. The text confirms "if none of these criteria fire, dispatch routes to Proceed or Reflect" — establishing the exhaustiveness of the trichotomy under the criteria.

**Verification:** The fix resolves the underspecification. The 1.5-standard-deviation entropy-collapse criterion is concrete enough for BUILD implementation while remaining operationally tunable. The post-hoc result-check hard failure criterion clearly links back to ADR-007's existing mechanism. The third criterion appropriately conditions on ADR-016's mechanism (d) being active.

**New gap check:** One interaction to verify: the third Abstain criterion fires when "multiple drift-detection criteria simultaneously exceed thresholds" from ADR-016's mechanism (d). Mechanism (d) produces a three-level verdict: no drift / drift detected (advisory) / severe drift. The Abstain criterion says "severe drift verdict from mechanism (d) propagates to Abstain at the Calibration Gate level." This is the correct mapping — the severe-drift signal from mechanism (d) is exactly the kind of multi-criteria simultaneous breach that warrants dispatch blocking. The data-flow is coherent: mechanism (d) produces a typed verdict; ADR-014 consumes it as one of the Abstain trigger inputs. **No new issue.**

**Status: RESOLVED.**

---

#### P2 Fix 2.4 — ADR-014: HTC feature-extraction location

**Fix applied:** A "Feature-extraction location" note has been added under the HTC trajectory features item in the quality-signal composition: when ADR-016 is active, features are extracted once at L0 and propagated upward through the read-only signal channel; L1 consumes rather than re-extracts. When ADR-016 is rejected, features are extracted in-layer at L1. The single-extraction-point property is stated as load-bearing for both performance (avoid double-cost compute) and consistency (same features inform both the cross-layer channel and the in-layer calibration verdict).

**Verification:** The fix resolves the double-counting ambiguity the round 1 audit identified. The conditional structure (ADR-016 active vs. rejected) is clean and matches the composition dependency noted in ADR-014's Consequences §Neutral. The performance and consistency rationale for single-extraction is sound.

**New gap check:** The ADR now claims that when ADR-016 is active, the features are "extracted once at L0." This introduces a dependency: L0 (Ensemble Engine) must perform HTC feature extraction on behalf of L1's Calibration Gate. ADR-016 §The signal channel confirms this is in scope: "Ensemble output trajectory features (extracted at L0 per ADR-014's HTC specification, available for L1's calibration verdict)" — the channel specification explicitly routes extraction to L0. The dependency is coherent across both ADRs and the assignment of extraction responsibility to L0 is mutually consistent. **No new issue.**

**Status: RESOLVED.**

---

#### P2 Fix 2.5 — ADR-016: Falsification trigger specificity

**Fix applied:** The falsification trigger now reads: "cannot be hosted in the Calibration Gate's existing or extended class structure without introducing a new top-level module." This replaces the imprecise "module-shape orthogonal to L0-L3" phrasing.

**Verification:** The revised criterion is BUILD-concrete. A BUILD engineer evaluating whether mechanism (b) or (d) fires the falsification trigger can make a deterministic judgment: does the implementation require a new top-level module, or can it fit in the Calibration Gate's existing or extended class structure? The criterion no longer depends on the ambiguous concept of "orthogonality to L0-L3." The concrete monitoring specification (P1 fix 1.2 above) maps this to the trigger condition "either mechanism requiring a top-level module outside ADR-002's L0-L3 structure to implement."

**Status: RESOLVED.**

---

#### P2 Fix 2.6 — ADR-017: Conservative false-positive discipline reframed

**Fix applied:** The "Minimal default pattern set with operator-extension surface" section now states that the default pattern set is "minimal rather than calibrated." The primary justification is reframed as "the spike evidence does not support a richer default pattern set" rather than as a calibration against confabulation frequency. The text explicitly says: "A richer default would presuppose calibration data the cycle does not have."

**Verification:** The fix resolves the false-precision issue the round 1 audit identified. The ADR no longer presents under-detection as a calibrated trade — it presents it as a consequence of the evidentiary state (no calibration data). The operator-extensibility is now correctly framed as "the operational discovery path, not a fallback." The Rejected alternatives (c) retains the original cost-asymmetry language about operator-visible session disruption, which is appropriate as a secondary consideration; the primary justification in the Decision section is now the evidentiary-state framing. These are consistent.

**Status: RESOLVED.**

---

#### P2 Fix 2.7 — Deferred-005: Two-failure-mode scope note

**Fix applied:** A "Two-failure-mode scope note" has been added to the spike specification section, distinguishing failure mode (a) — summarizer compressed unnecessarily — from failure mode (b) — orchestrator misrepresented summarizer's output as verbatim ensemble output. The note explicitly states that the output-size sweep primarily measures (a) and that if (b) is the dominant harm, a clean spike result on (a) would not resolve whether ADR-004 amendment is warranted. Future-cycle spike designers are instructed to measure (a) and (b) separately and report by failure mode.

**Verification:** The fix resolves the implicit scope assumption the round 1 audit identified. The note correctly observes that failure mode (b) — if observed independent of (a) — is "possibly closer to ADR-017's structural-validation guard territory than to ADR-004's mandatory-summarization framing." This is an accurate mapping: orchestrator confabulation about source is a structural-representation failure, which ADR-017 addresses at the tool-call level. Whether the same guard mechanism extends to orchestrator misrepresentation of summarizer output is a meaningful open question that the scope note appropriately surfaces without overcommitting.

**Status: RESOLVED.**

---

#### P3 Fix 3.1 — ADR-012: Layer 4 circuit-breaker reset mechanism

**Fix applied:** Decision section item 5 now reads: "Circuit-breaker state is automatically reset at session start (per argument-audit P3.1 finding 2026-05-06); no operator action is required between sessions."

**Verification:** The addition is present, clear, and consistent with the Consequences §Negative entry that "must be reset between sessions." The automatic reset removes the ambiguity about whether operator action was required.

**Status: RESOLVED.**

---

#### P3 Fix 3.2 — ADR-013: Signed-script integrity verification scope

**Fix applied:** Write-gate validation item (iii) now includes an explicit scope note: "the integrity check is tamper-detection — it detects modification of init.sh content between operator-authoring time and session-execution time. It does not validate that the operator-authored init.sh is itself safe; the operator's authoring step is the trust boundary."

**Verification:** The scope note correctly characterizes the verification's reach. The terminology "signed-script integrity verification" is retained in the subsection heading, but the scope note disambiguates what "integrity" means in this context. The round 1 audit recommended renaming to "tamper-detection for init.sh" — the fix opted for a scope note instead. The scope note achieves the same disambiguation without renaming. Either approach is valid; the note is sufficient.

**Status: RESOLVED.**

---

#### P3 Fix 3.3 — ADR-015: Discovery value proportional to deployment coverage

**Fix applied:** Consequences §Negative now includes: "Discovery value is proportional to deployment coverage (per argument-audit P3.3 finding 2026-05-06). The cycle's primary task class may exercise only 4-5 of the 8 Topaz skills routinely; the remaining slots produce no calibration evidence in deployment. The friction-trades-for-discovery argument holds for the exercised skills; for unexercised skills, the configuration burden is friction without proportionate discovery."

**Verification:** The addition accurately reflects the constraint. The note goes further than the minimum: it correctly identifies the operational response (operators concentrated in a sub-taxonomy may collapse unused skills to shared Model Profiles) and notes that the structural taxonomy remains available for the discovery surface to expand into. This is sound — it neither retracts the full-taxonomy decision nor pretends the discovery value is uniform across skills.

**Status: RESOLVED.**

---

#### P3 Fix 3.4 — ADR-016: Mechanism (c) anchor strength scope condition

**Fix applied:** Mechanism (c) now includes an explicit "Anchor-strength scope condition": the "cannot be argued away" property is strongest for binary-verifiable deterministic outputs (the CrossHair formal-counterexample shape); weaker for interpretable-numerical outputs; weakest for interpretable-prose outputs. The mechanism's anchor strength scales with output verifiability.

**Verification:** The scope condition is accurate and well-grounded. The CrossHair literature precedent (arXiv:2510.21513) is explicitly about formal counterexamples — binary pass/fail — and the ADR correctly does not extend "cannot be argued away" to interpretable deterministic outputs. The three-level hierarchy (binary-verifiable / interpretable-numerical / interpretable-prose) is a reasonable decomposition that a BUILD engineer can apply when evaluating specific ensemble configurations.

**Status: RESOLVED.**

---

#### P3 Fix 3.5 — ADR-017: Shared typed-error base class

**Fix applied:** A "Shared typed-error base class (cross-ADR coordination)" section has been added to the Decision, specifying `LlmOrcStructuralError` as the base class with four common fields: `error_kind`, `dispatch_context`, `recovery_action_required`, and `operator_diagnostic`. The `error_kind` enum lists all error types (existing and new). The note explicitly cross-references ADR-012, ADR-013, ADR-014, ADR-016, and ADR-017.

**Verification:** The addition resolves the BUILD-coherence gap. The common fields cover the relevant information for each error context: `recovery_action_required` maps to the three recovery paths (reformulate / escalate / abstain); `operator_diagnostic` provides the human-readable surface. The field set is consistent with what each individual ADR's rejection specification requires. The base class name is appropriately qualified as "or equivalent name to be finalized in BUILD" — this is the correct posture for a name that is drafting-time synthesis.

**Status: RESOLVED.**

---

### Summary of Fix Verification

All fourteen Round 1 issues (2 P1 + 7 P2 + 5 P3) are resolved in the revised ADRs. No fix introduces a new P1 or P2 logical gap. One observation (the boundary-crossing framing in ADR-015's L2-interposition justification) is noted as imprecise but does not reopen the P1 issue — the compatibility argument is sound. One deferral (ADR-013's disposition (ii) mechanism underspecification) is appropriate given the ADR's posture of deferring BUILD-concrete implementation to BUILD.

---

### New Issues Introduced by Revisions

Reviewing the revised ADRs for logical gaps, overreach, or contradictions introduced by the round 1 corrections:

**Scan of ADR-012:** The Provenance check defaults-provenance addition is consistent with the Decision's "operationally tunable" language. No new issues.

**Scan of ADR-013:** The cross-cluster sessions disposition (i) introduces a default of "required" when session-shape declaration is ambiguous. The Consequences §Negative entry acknowledges "Cluster 2 default-required behavior may friction-cost early adoption." The three-disposition structure is internally consistent. No new issues.

**Scan of ADR-014:** The feature-extraction location specification introduces an ADR-016-conditional data-flow path. Both branches (ADR-016 active / ADR-016 rejected) are specified and consistent with ADR-016's signal channel description. The Abstain trigger criteria are self-consistent: the entropy-collapse criterion uses "running mean" as the baseline, which presupposes a trajectory of sufficient length to compute a running mean. For very short trajectories (first 1-2 dispatches), the running mean may be unstable. This is an operational tuning concern rather than a logical gap — the threshold is operationally tunable and would require warm-up data. **No new issue requiring correction; implementation-level concern noted.**

**Scan of ADR-015:** The L2-interposition justification introduces the claim about boundary-crossing as noted above. No other new gaps.

**Scan of ADR-016:** The concrete monitoring specification (new addition) introduces the sweep responsibility clause citing ADR-068. The clause creates a commitment: "Absence of the row constitutes structural evidence that the cycle has not exercised the channel." This is a strong structural claim — it converts absence-of-evidence into evidence-of-absence. The logic holds for the monitoring purpose (if no row is present, the channel has not been tracked) but could produce false negatives if a cycle exercises the channel but the cycle-status update is missed. This is an implementation concern — the structural claim is sound as a monitoring discipline even if it can produce gaps in practice. **No new issue requiring correction.**

**Scan of ADR-017:** The shared typed-error base class section introduces `LlmOrcStructuralError`. The `recovery_action_required` field is specified as one of {reformulate / escalate / abstain}. Cross-checking against the individual ADRs: ADR-012's Layer 4 circuit-breaker failure surfaces to the operator — the appropriate recovery action for that case would be operator intervention, not orchestrator reformulate. The base class field values may not map cleanly to every error type in the enum.

- **Location:** ADR-017 §Decision "Shared typed-error base class," field `recovery_action_required`
- **Claim:** The field is specified as `recovery_action_required: whether the orchestrator must reformulate, retry-with-different-tier, or abstain entirely`
- **Issue:** The Layer 4 circuit-breaker failure (ADR-012) is an operator-facing error — the appropriate action is for the operator to investigate the LLM summary service, not for the orchestrator to reformulate or escalate. The three-value enum {reformulate / escalate / abstain} covers orchestrator-side recovery actions but not operator-required interventions. If the field can only take one of these three values, Layer 4 failures would need to be mapped to the closest available value (probably `abstain` in the sense of "orchestrator cannot continue compaction at this layer"), but that is semantically imprecise. The field's documentation of Layer 4 circuit-breaker errors via the shared base class may produce misleading `recovery_action_required` values.
- **Severity:** P3 — this is a field-mapping precision concern in a field whose finalization is deferred to BUILD. The ADR correctly says "Naming and field finalization is BUILD work." The concern is that the field's three-value set should anticipate a fourth value: `operator_intervention_required` for cases where the orchestrator cannot self-recover.
- **Recommendation:** Add `operator_intervention_required` as a fourth value in the `recovery_action_required` field specification, covering cases where the error requires human operator response (Layer 4 circuit-breaker failure after three consecutive failures; severe drift in ADR-016 mechanism (d); potentially write-gate violations that indicate a policy compliance issue rather than a transient error). This is a BUILD-time finalization concern; the ADR can note the fourth value as a candidate without requiring immediate field definition.

---

### Summary

- **Argument chains mapped:** 14 (same as Round 1) + 2 new chains from the monitoring specification and shared-error-base additions
- **Round 1 issues resolved:** 14 of 14
- **New issues found:** 1 (P3)

---

### New P3 Issue

**Issue R2-3.1 — ADR-017: Shared typed-error base class `recovery_action_required` field omits operator-intervention cases**

- **Location:** ADR-017 §Decision "Shared typed-error base class"
- **Claim:** The `recovery_action_required` field covers "whether the orchestrator must reformulate (reformulate), retry-with-different-tier (escalate), or abstain entirely (abstain)."
- **Evidence gap:** The three values cover orchestrator-side recovery actions. At least two of the five error types in the base class require operator intervention rather than orchestrator action: Layer 4 circuit-breaker failure after three consecutive LLM-summary failures (ADR-012) is an operator-facing event requiring investigation of the LLM service; severe drift in mechanism (d) (ADR-016) requires operator review and calibration-system parameter action. Mapping these to `abstain` is semantically imprecise: "abstain" implies the orchestrator has a dispatch decision to block, but a Layer 4 circuit-breaker failure affects the compaction subsystem, not a specific dispatch. The field as specified cannot accurately express the recovery action for these error types.
- **Recommendation:** Add `operator_intervention_required` as a fourth candidate value in the `recovery_action_required` field specification, with a note that this value covers errors where the orchestrator cannot self-recover and the next action is operator review. Flag this as BUILD-time finalization territory, consistent with the ADR's existing posture on field naming. The addition requires no architectural change — it is a field-value extension in drafting-time synthesis that the BUILD-time finalization should address.

---

## VERDICT: ROUND 2 CLEAN (subject to P3 note)

All fourteen Round 1 issues have been correctly resolved. The revisions are logically sound and do not introduce new P1 or P2 gaps. One new P3 issue (ADR-017 shared-error-base `recovery_action_required` field missing operator-intervention cases) was identified in the scan of new additions. This is a BUILD-time precision concern that can be addressed without re-deliberation. The ADR set is clear to proceed to the epistemic gate.

---

## Section 2: Framing Audit

The Round 1 framing-audit findings (Section 2 of the Round 1 report) are not being addressed in this round per the skill workflow — they are surfaced at the practitioner gate. Round 2 does not re-audit framing.

For reference, the Round 1 framing findings were:
- **P1 (consequential omission):** Autonomous-routing evidence gap not carried into ADR-015
- **P2:** Plexus-conditional framing of ADR-016's value not stated
- **P2:** Attention-MoA orchestrator-as-aggregator finding not surfaced in ADR-015
- **P3:** ADR-012 adoption framing underrepresents the adaptation choices

These remain in the Round 1 report for practitioner-gate review. Round 2 did not find any new framing issues introduced by the Round 1 corrections.
