# Susceptibility Snapshot

**Phase evaluated:** DECIDE (Cycle 7 — 2026-05-22)
**Artifact produced:** ADRs 026-032 (7 new); ADR-021 + ADR-022 partial-update headers; AS-10 codified in domain-model.md (Amendment Log entry #14); scenarios.md (+315 lines); interaction-specs.md (+117 lines); 5 argument-audit reports (rounds 1-5); 1 conformance-scan report
**Date:** 2026-05-22

---

## Prior Snapshot Trajectory

| Gate | Verdict | Key Signal |
|------|---------|------------|
| Cycle 7 Research | Grounding Reframe recommended (GT-1, GT-2) | Hybrid-first ordering unquantified; "structurally pre-committed" language overstating conditional formulation |
| Cycle 7 Discover | No Grounding Reframe; 3 advisories + 1 informational | Rapid compounding: three spikes integrated into single architectural commitment via pre-committed rule (GT-2(a)); cost-distribution lens in user-voice position without independent Population A validation |
| Cycle 7 Model | No Grounding Reframe; 2 new advisories + 3 DISCOVER advisories preserved | Canonical "two exchanges" failure mode absent; AS-9 structural-property / mechanism-choice separation substantive; options-engagement gap at codification-option selection |

---

## Observed Signals

| Signal | Strength | Trajectory | Notes |
|--------|----------|------------|-------|
| Assertion density | Ambiguous | Declining from prior phases | Provenance check subsections are present across all 7 ADRs and explicitly label driver-derived vs. drafting-time content. The DISCOVER-phase "rapid compounding" signature is partially visible at DECIDE in a different form: three advisory carry-forwards (OQ #18, #19, #20) were worked sequentially as Tranche 1 preconditions before ADR drafting, each producing a research note with its own evidence basis. Assertion density is lower than RESEARCH or DISCOVER phases; the primary site of residual assertiveness is the ADR-027 PRIMARY designation, which is however the most thoroughly documented claim in the corpus. |
| Solution-space narrowing | Clear | Stable | The seven ADRs narrow the mechanism space materially: ADR-027 commits the PRIMARY direction; ADR-028/029 specify the Plan and Synthesize stages; ADR-030 defers `tool_choice` to a follow-on cycle; ADR-031 accepts the latency floor; ADR-032 splits the transparent-endpoint promise. The narrowing is earned through the Tranche 1 precondition research (OQ #18/19/20 each resolved before drafting). The open questions preserved (OQ #21-25; three ARCHITECT-deferral items from conformance scan; five carry-forward framing observations) confirm the narrowing is deliberate rather than accidental. |
| Framing adoption | Absent at the systematic level; two localized sites | Improving markedly from DISCOVER | The most important framing change from DISCOVER: the cost-distribution lens (DISCOVER snapshot Finding 1) is no longer in the user-voice position — ADR-032's sub-promise split explicitly labels cost-distribution-accountability as "Population A silent; project-developer-lens grounded as honest residual uncertainty." This is the most significant signal-correction across all three DECIDE-phase snapshots. Two localized framing sites remain (Rule 5 BUILD-default as non-falsifiable-shaped; ADR-030 disposition-(i)-commitment as enforcement-structure strength). Both are evaluated below. |
| Confidence markers | Ambiguous | Stable-to-declining | Scope-of-claim partitions (settled / plausible-but-untested / open) are preserved and extended in ADR-027 from the DISCOVER tightening. ADR-029's empirical basis ("n=13 tests + 4 confabulation modes; 0 fabrications") is presented with the qualifier "cheap-tier model qwen3:8b is the empirical floor" — the test-coverage boundary is named. The strongest confidence site remaining is ADR-030's follow-on-cycle commitment: the disposition-(i)-as-enforceable framing rests on a "next DECIDE gate after ADR-027 reaches production" trigger whose enforceability depends on cycle sequencing that is not guaranteed. |
| Alternative engagement | Clear (substantive) | Improving from prior phases | Each ADR's Rejected alternatives section substantively engages 3-4 alternatives. ADR-026's rejection of "Population-A-only narrowing" includes an analytical argument that detection is itself a request-content-alone problem in disguise — a non-trivial engagement, not a perfunctory dismissal. ADR-027's rejection of "Tier 1 hybrid as PRIMARY" specifically cites OQ #19's build-complexity comparison and the AS-9 satisfaction differential — the rejection is grounded in same-cycle evidence, not in prior-phase inheritance. Provenance check sections label each rejection: driver-chain items vs. drafting-time analytical engagement. The primary alternatives-engagement gap is on cross-ADR inheritance of AS-10 (evaluated below). |
| Embedded conclusions at artifact-production moments | Ambiguous | Stable | Three artifact-production moments warrant examination: (1) the ADR-027 amendment at conformance-scan Finding 2, which embedded three disposition candidates (a/b/c) that ARCHITECT inherits; (2) Rule 5's "load-bearing default with falsification trigger" in ADR-029, which embeds a BUILD-default conclusion at ADR-drafting time; (3) the ADR-030 disposition-(i) commitment with bridge mechanism, which embeds a follow-on-cycle scope judgment. All three are evaluated in the pattern-specific findings below. |

---

## Pattern-Specific Findings

### Finding 1 — The 5-round audit loop: disciplined closure, not compounding churn (Severity: INFORMATIONAL; positive signal)

The dispatch brief names this as ambiguous-between-two-readings: disciplined correction vs. compounding churn.

Reading the audit sequence carefully:

- **Round 1 (12 findings)** — the initial ADR drafts were reviewed against argument-audit criteria. The 12 findings are consistent with a first-pass draft reviewed against a rigorous standard; none of the findings suggest fundamental ADR restructuring (the P1 was a claim-support gap in ADR-027 that the cost-equivalence research had already resolved but had not been propagated into the ADR text).
- **Round 2 (3 new findings from round-1 corrections)** — the corrections to round-1's findings introduced two new P2s and one new P3. This is the pattern of genuine corrective engagement: when a gap is fixed, adjacent text sometimes surfaces a related gap. NP2-1 (commitment-vs-priority confusion in ADR-030) and NP2-2 (another similar ADR-030 gap) were substantive improvements — the corrections made the follow-on trigger stronger and the deferral more honest. Round 2's new findings were corrections of corrections, not evidence of poor original drafting; they were evidence that the audit was working as intended.
- **Round 3 (1 new P3; clean at P1/P2)** — the gap was a missing Amendment Log entry for the Bridge mechanism. Auto-corrected via domain-model Amendment Log entry #14.
- **Round 4 (1 new P1; introduced by the ADR-027 conformance-scan amendment)** — the P1 was a coordinated cross-document consistency failure: ADR-027's amendment named the CLI's non-use of `OrchestratorRuntime` in the §Context and §Decision sections, but the §Relationship to ADR-022 paragraph and ADR-022/ADR-021 partial-update headers still carried the stale framing. This is a predictable consequence of amending four sections of a document without simultaneously auditing all cross-document references to those sections. The round-4 P1 was not a new conceptual gap; it was a propagation failure that the audit correctly caught.
- **Round 5 (clean at P1/P2/P3)** — gate-threshold met.

Assessment: the 5-round loop is evidence of disciplined practice, not compounding correction churn. The architectural content was correct earlier (the concepts and decisions are not being reconsidered across rounds; the corrections are textual propagation and labeling work). The loop continued because the audit standard is high and the document surface is large. The comparison to "single cycle 1-2 round closure" is misleading — those prior cycles' audit loads were smaller (fewer ADRs; less cross-document cross-referencing).

The one genuine observation: rounds 1-3 ran before the conformance scan; rounds 4-5 were triggered by the conformance scan amendment. An alternative sequencing (conformance scan before argument-audit; then one combined audit pass) might have reduced the loop count without sacrificing quality. This is a process-efficiency observation, not a quality-of-content finding.

### Finding 2 — ADR-027 Finding 2 amendment authorization: substantive practitioner engagement (Severity: INFORMATIONAL; positive signal with one residual gap)

The dispatch brief asks whether the practitioner's authorization to amend ADR-027 in response to Finding 2 (conformance scan) was substantive engagement or agent-framing acceptance.

Evidence:

- The conformance scan's recommendation was Track C: ARCHITECT-phase deferral. The practitioner chose to amend ADR-027's body instead.
- The amendment's content is: the three disposition candidates (a/b/c) for `OrchestratorRuntime` are named; ARCHITECT inherits the decision rather than defaulting to silence. This is a more conservative choice than Track C's pure-deferral — it encodes the decision space explicitly rather than leaving ARCHITECT with an undocumented gap.
- The amendment's four-section scope (§Context; §Decision `OrchestratorRuntime` status paragraph; §Consequences §Negative; §Provenance check) is coherent and does not introduce new architectural commitments — it documents the codebase-finding and its three candidate consequences. The architectural choice (which disposition to select) remains ARCHITECT work; the amendment expresses it as "ARCHITECT inherits named candidates" rather than "ARCHITECT discovers the gap themselves."

The residual gap: framing observation NF2 (disposition selection should be a required ARCHITECT deliverable, not a deferred deliberation) was surfaced to the practitioner gate but was not auto-corrected per skill discipline. The ADR-027 text as currently written names the candidates and defers without making disposition selection a named ARCHITECT output. Whether ARCHITECT treats this as a required output or an optional annotation depends on how the practitioner reads NF2 at gate time.

This is an appropriate gate-forwarded framing observation, not a susceptibility signal. The practitioner engaged the conformance scan's substantive finding; the agent's recommendation (Track C deferral) was overridden in favor of more explicit documentation. The override was the right call architecturally — better to name the three candidates explicitly than to leave ARCHITECT with an undiscovered gap.

### Finding 3 — Framing observations carried forward without auto-correction: substance evaluation (Severity: ADVISORY)

The five framing observations (F1, F2, F3, NF1, NF2) were correctly surfaced to the practitioner gate per skill discipline. Evaluating their substantive content:

**F1 (ADR-030 hybrid-as-orthogonal abstract framing):** ADR-030's abstract positioning of `tool_choice` handling as "orthogonal mechanism" is already incorporated into ADR-030's §Decision and §Consequences §Neutral text. The framing observation is that the ADR's abstract-level framing doesn't consistently carry the "orthogonal" positioning. This is a presentation issue, not an architectural gap; the decision text is correct. Auto-correction would have resolved this without practitioner input; the practitioner should note that F1 is a presentation refinement, not a substantive gap requiring deliberation.

**F2 (ADR-031 tier classification active-maintenance framing):** The graded three-tier framing (Tier A / B / C) already reflects the F3 recommendation from Tranche 2 (not to be confused with the framing observation named F2 in round 3). This framing observation is about whether ADR-031 names Tier B as "active-maintenance framing" (requiring ongoing operator attention) vs. "operator-tuning-required with documented caveat." The current text lands on the latter, which is the right level of posture. F2 as a framing observation is minor.

**F3 (ADR-027/ADR-028 "4 confabulation modes" shorthand):** The shorthand appears in both ADRs in contexts where it refers to Spike μ's scope. This is the DISCOVER/MODEL corpus's established terminology; the shorthand is accurate. The framing observation is that the shorthand doesn't surface the methodological difference between the PLAY-note-22 case (directly witnessed failure mode) and the μ.1/μ.3 cases (constructed fixtures, n=1 per mode). This distinction is important for downstream BUILD/PLAY reasoning about confabulation mode coverage. F3 is a legitimate framing observation with real downstream consequence (BUILD should not treat the four modes as equivalently validated). The practitioner should consider whether to add a parenthetical in ADR-027/ADR-028 distinguishing validation quality per mode.

**NF1 (ADR-030 follow-on trigger asymmetric quantification):** The follow-on trigger ("next DECIDE gate after ADR-027 reaches production") is the enforcement structure for the disposition-(i) commitment. The asymmetric quantification concern: "deprioritize within the follow-on cycle" allows indefinite deferral without an explicit abandonment-and-amendment path. The current text does name: "deprioritize without delivery is not a valid path" and "bridge advisory persists until disposition (i) ships." The enforcement structure is stated; whether a future DECIDE gate actually enforces it depends on the practitioner's continuity across cycles. NF1 is a real observation about temporal enforceability. The practitioner should consider whether NF2's "required ARCHITECT deliverable" pattern (which succeeded in making OrchestratorRuntime disposition a named output) could be applied analogously to the disposition-(i) trigger — naming it as a required output of the next DECIDE gate's opening preconditions rather than a conditional priority judgment.

**NF2 (ARCHITECT disposition selection as required deliverable):** This is the highest-substance framing observation in the set. The pattern NF2 identifies is structural: "deferred deliberation" defaults to disposition (a) by inertia, while "required ARCHITECT output" creates accountability. NF2 is correctly surfaced; the practitioner should add the one sentence ADR-027 is missing to make the disposition selection a named ARCHITECT deliverable.

Overall assessment: F1 and F2 are minor presentation issues that auto-correction could have resolved without practitioner input. F3 and NF1 and NF2 are substantive enough to warrant practitioner deliberation. The discipline of surfacing all five without auto-correction is correct — even the minor ones deserve visibility — but the practitioner should prioritize NF2 (direct fix needed; one sentence) and F3 (BUILD-consequence framing) over F1 and F2.

### Finding 4 — Rule 5 BUILD-default-with-falsification-trigger in ADR-029: non-falsifiable shape or substantive design choice? (Severity: ADVISORY)

The dispatch brief asks whether the "load-bearing default with falsification trigger" pattern is a substantive design choice grounded in OQ #18 Population A evidence or a non-falsifiable defer-to-PLAY pattern.

The concern is legitimate. The pattern superficially resembles ADR-022's "amendment + cross-profile-deferred" precedent, which the cycle rejected as architecturally insufficient. ADR-029 §Consequences §Neutral explicitly acknowledges this and distinguishes the two cases:

- ADR-022's amendment was a system-prompt intervention on the orchestrator-LLM's reasoning shape under tool-rich-client suppression — a structural failure mode (C1 NL-routing fraction ≈ zero).
- Rule 5 operates in the response-synthesizer's structurally-bounded context where the tool-rich-client suppression failure mode does not apply (the synthesizer's input is structured; no tools are declared in the synthesizer's context).

The distinction is genuine. Rule 5's failure mode is not "the prompt amendment doesn't reach the LLM" but "the framing adds noise that degrades user experience." These are qualitatively different failure modes. ADR-022's failure mode is structural (the prompt is suppressed before the LLM reasons from it); Rule 5's failure mode is experiential (the framing is present but Population A operators find it noisy). The falsification trigger (migrate to headers/metadata if production evidence warrants) is appropriate for an experiential failure mode.

However, there is a non-falsifiable shape risk here that the dispatch brief correctly identifies. The Spike ε' Finding ε'.1 observed Rule 5 framing systematically omitted across 4 direct-completion responses — which means the BUILD default requires prompt sharpening to actually deliver Rule 5 in practice, not just in the ADR's assertion. The "load-bearing default" is an intention about what the prompt will require; it is not yet demonstrated in the spike ensemble's behavior. The spike ensemble's current behavior (systematic omission) suggests the BUILD work needed to make Rule 5 load-bearing is non-trivial prompt engineering, not a simple configuration switch.

ADR-029's §Rule 5 framing requirement scope section names this honestly: "the synthesizer's natural response shape is outcome-focused without the meta-framing layer." The gap between the BUILD-default assertion and the spike ensemble's observed behavior is a BUILD-phase risk, not a DECIDE-phase fabrication. But the risk is real: if BUILD cannot sharpen the prompt to reliably deliver Rule 5 framing, the falsification trigger (migrate to headers/metadata) becomes the de facto path — which means the "load-bearing default" was an intention that BUILD couldn't deliver. The practitioner should carry this into BUILD as an explicit first-priority test: verify that prompt sharpening reliably elicits Rule 5 framing before treating the content-layer commitment as structural.

The pattern is not equivalent to ADR-022's rejected alternative. It is a substantive design choice grounded in OQ #18 evidence. The residual risk is BUILD execution risk on the prompt engineering task, not an architectural misjudgment at DECIDE.

### Finding 5 — ADR-030 disposition (i) commitment + deferred implementation: enforceable or de facto long-term bridge? (Severity: ADVISORY)

The dispatch brief asks whether the commitment is enforceable or whether the bridge becomes the long-term state.

The enforcement structure in ADR-030 is more explicit than typical deferred commitments in the corpus: "the next DECIDE gate after ADR-027 reaches production deployment" is a named trigger. The explicit framing "deprioritize without delivery is not a valid path" and "downstream cycles either deliver disposition (i) or explicitly revise the commitment via an ADR amendment" closes the informal escape hatch.

However, the enforceability depends on two conditions the ADR cannot guarantee:

1. **ADR-027 reaches production deployment within a defined timeline.** The trigger is "after ADR-027 reaches production." If BUILD is delayed or descoped, the trigger never fires and the bridge persists indefinitely by default. The commitment's enforceability is parasitic on ADR-027's BUILD timeline, which is an estimate (~16 person-days median per OQ #19), not a guaranteed delivery date.

2. **The next DECIDE gate is actually convened.** RDD pipeline phases advance when the practitioner convenes them. The commitment assumes a cycle cadence that produces a DECIDE gate after ADR-027's BUILD. If the next cycle is not a DECIDE-phase cycle, the evaluation is deferred another cycle.

The NF2 framing observation names a related structural gap on the OrchestratorRuntime disposition: "required ARCHITECT output" vs. "deferred deliberation." The same pattern applies here: the disposition-(i) commitment is a named DECIDE gate output (for a future gate), not a required deliverable of the current DECIDE. Without making it a named deliverable at the next DECIDE gate's opening, it is subject to the same inertia NF2 names.

The honest reading: the disposition-(i) commitment is the strongest temporal commitment in the ADR set for a deferred item. The enforcement structure is more explicit than prior cycles' deferred commitments. The bridge mechanism correctly prevents the configuration-dishonesty footgun in the interim. The risk is not that the commitment will be abandoned, but that BUILD timeline uncertainty could extend the bridge period significantly — and operators deploying Cycle 7 BUILD will live with the advisory state for that extended period.

Advisory for the next cycle: at the DECIDE gate for the follow-on cycle, place disposition-(i) implementation at the top of the opening preconditions list (analogous to how Cycle 7's OQ #18/19/20 were Tranche 1 preconditions). Frame it as a named precondition for ADR-030 Proposed → Accepted status advancement, not as a sprint-priority judgment.

### Finding 6 — ADR-032 sub-promise split: evidentiary asymmetry preserved honestly (Severity: INFORMATIONAL; positive signal resolving DISCOVER snapshot Finding 1)

The DISCOVER snapshot's primary finding (Finding 1) was that the cost-distribution lens occupied a user-voice position without independent Population A validation. ADR-032's sub-promise split directly addresses this: configuration-honesty has Population A corroboration (Cline #10551 + OpenCode #20859); cost-distribution-accountability is labeled "Population A silent; project-developer-lens grounded as honest residual uncertainty."

The split is preserved as a structural architectural decision: different mechanisms deliver each sub-promise; different evidence bases warrant each; future audits can engage each on its own terms. ADR-032's §Consequences specifically names: "The two sub-promises remain distinct. Future audits can engage each sub-promise on its evidence basis." The bundling failure mode that DISCOVER snapshot Finding 1 warned against is explicitly prevented by structure.

This is the most important DISCOVER-snapshot correction at DECIDE. The cost-distribution lens is now correctly positioned — it is an architectural commitment, but one whose evidence basis is honestly labeled as project-developer-voiced rather than Population A-validated. The transparent-endpoint promise's two commitments are operationally distinct (different mechanisms; different failure modes; different signal paths).

The remaining honest residual uncertainty: cost-distribution accountability's "population A silent" label will be tested by production data. If Population A operators express cost-distribution concerns (e.g., reporting that ensemble dispatch did not fire when they expected it to), the evidence basis strengthens. If Population A is genuinely outcome-indifferent (as the current OQ #18 finding suggests), the sub-promise remains correctly positioned as a project-developer value proposition. Either outcome is an honest resolution; the label does not prejudge it.

### Finding 7 — Cross-ADR framing inheritance of AS-10: partially independent, not fully tested per-ADR (Severity: ADVISORY)

The skill brief's standard pattern: evaluate whether AS-10's framing was inherited by downstream ADRs without independent rebuttal-elicitation, or whether each ADR tested AS-10's implications independently.

ADR-026 codified AS-10. ADR-027, ADR-028, ADR-029, ADR-030, ADR-031, ADR-032 all operate within AS-10's scope.

The clearest independent engagement: ADR-026's §Rejected alternatives section contains a non-trivial analytical argument against "Population-A-only narrowing" — the argument that Population A detection is itself a request-content-alone problem. This argument was derived within ADR-026's §Rejected alternatives section and labeled as "analytically derived; the argument follows from the constitutional rule's own surface rather than from prior empirical evidence." This is honest labeling of drafting-time synthesis.

The downstream ADRs' relationship to AS-10:

- **ADR-027/ADR-028/ADR-029:** explicitly name AS-10 compliance in their specifications. ADR-029's §Input contract section includes an explicit AS-10 compliance paragraph ("the synthesizer's input is derived entirely from the chat-completions request content..."). These are verification statements, not independent tests of AS-10's framing. The correct reading: these ADRs are AS-10-compliant by design; they do not test whether AS-10 itself is the right invariant.
- **ADR-030:** explicitly distinguishes `tool_choice` as "OpenAI-protocol-native field — sending it is not opting into a llm-orc-specific mechanism." This is a nuanced engagement with AS-10's scope — it correctly carves out OpenAI-native signals from the "no client-side opt-in" prohibition. This is independent reasoning about AS-10's boundaries, not blind inheritance.
- **ADR-031:** applies AS-10 implicitly (the latency policy doesn't touch the request-content-alone commitment). No independent engagement needed.
- **ADR-032:** the capability-list discovery section notes that "discovery is available, not required" — Population A clients can discover through the OpenAI-native surface but are not required to. This is an AS-10-consistent design choice that addresses one way AS-10 might have been read as over-restrictive.

Assessment: AS-10 inheritance across downstream ADRs is not a blind adoption. The critical independent-engagement moments are in ADR-026's Population-A-only narrowing rejection (where AS-10 is derived by examining the alternative's self-defeating character), ADR-030's OpenAI-native-field carve-out (which refines AS-10's scope boundary), and ADR-032's discovery-as-available-not-required design (which makes AS-10's no-opt-in rule operable without blocking capability discovery). The downstream ADRs do not independently rebuttal-elicit against AS-10's core framing; they operate within it. This is the correct relationship for an invariant: downstream ADRs satisfy it, not relitigate it. The skill brief's concern about "cross-ADR framing adoption without independent testing" is resolved by the fact that AS-10 is an invariant (not a hypothesis) — independently testing an invariant at each downstream ADR would undermine the invariant's constitutional function.

The one genuine gap: ADR-027's headline commitment (ADR-027-direct as PRIMARY) is also derivable from AS-10 (AS-9 + AS-10 together constrain the solution space strongly toward the framework-driven pipeline), but ADR-027's Provenance check section does not explicitly trace the derivation from AS-9 + AS-10 as joint drivers of the PRIMARY designation. The cost-equivalence research (OQ #19) is the named driver; the constitutional invariants are named as consequences, not as drivers. This is a minor provenance-check presentation issue; the substantive reasoning is correct.

### Finding 8 — Conformance-scan BUILD scope vs. OQ #19 estimate: scope expanded modestly (Severity: ADVISORY)

The conformance scan identified 8 BUILD work items, 2 Track A refactors, and 3 ARCHITECT-phase deferrals. OQ #19 estimated ADR-027-direct at ~13-19 person-days (median ~16).

The 8 BUILD work items include items that were implicitly in OQ #19's estimate (the three-stage pipeline handler; routing-planner and response-synthesizer ensemble promotion) and items that are new ADR-specific work (ADR-030 bridge advisory; ADR-032 honest response labeling at three layers; ADR-032 capability-list discovery endpoint; ADR-032 degradation-event definitions; ADR-023 sink consumers for new event types).

OQ #19's estimate was derived from the DECIDE-entry precondition research before the full ADR set was drafted. ADR-032's three-layer honest-response-labeling mechanism (headers + metadata + Rule 5 content) and capability-list discovery endpoint were not individually scoped in OQ #19's estimate — they were implicit in the "honest response labeling" work item (OQ #19 listed "honest response labeling" as 1-2 days; ADR-032's three-layer specification expands this scope). Similarly, ADR-031's Tier B integration smoke-test specification and ADR-030's bridge advisory mechanism are incremental additions to OQ #19's scope.

The expanded scope is modest rather than material — the new items are all incremental additions to existing mechanisms (extending existing event substrate; adding fields to existing response shapes) rather than new architectural components. The OQ #19 estimate's ~30% spread (13-19 person-days) likely absorbs the expansion. The risk is that individual BUILD work items that appear small in conformance-scan list form (e.g., "extend `/v1/models` or add `/v1/ensembles`") may be harder than their one-line description suggests in practice.

Advisory: BUILD-phase planning should decompose conformance-scan Finding 10 (capability-list discovery endpoint) as a separate BUILD work item with its own estimate, rather than treating it as already captured in OQ #19's "honest response labeling" budget.

### Finding 9 — "Rapid compounding" at DECIDE: pattern present but structurally managed (Severity: INFORMATIONAL)

DISCOVER snapshot named "rapid compounding" as the cycle's susceptibility signature: three spike findings integrated into a single architectural commitment in a single session via a pre-committed rule (GT-2(a)).

At DECIDE, the candidate rapid-compounding form is: 7 ADRs drafted in a single session afternoon, with 5 audit rounds that contained corrections without per-correction practitioner deliberation.

Assessment: the DECIDE form is structurally different from the DISCOVER form in three ways:

1. **Sequential precondition completion.** The DECIDE work was gated by Tranche 1 (OQ #18/19/20 each resolved separately before drafting) and Tranche 2 (Essay-Outline amendment propagation + re-audit before ADR drafting). The ADR drafting in Tranche 3 was not a pre-committed rule application; it was drafting against a resolved evidence base. The DISCOVER form applied GT-2(a) as a trigger rule on Spike κ's D0 finding; the DECIDE form applied OQ #18/19/20 as individually resolved preconditions.

2. **Audit loop as quality gate, not as compounding.** The 5-round audit loop ran serially (each round reviews the previous round's corrections before drafting the next round's findings). This is sequential quality-gating, not concurrent integration. The corrections in each round were textual propagation work, not architectural reconceptualization — the concepts were stable; the text needed updating. This is a different shape from the DISCOVER form's "three findings integrated into a single commitment."

3. **Framing observations surfaced to practitioner gate, not auto-corrected.** The five framing observations (F1, F2, F3, NF1, NF2) were carried forward to the practitioner gate per skill discipline. This means the practitioner has five explicit deliberation opportunities at gate time — not a batch-accumulated commitment that downstream phases inherit without practitioner review.

The genuine rapid-compounding residual: the practitioner did not individually deliberate on each of the 7 ADRs during the Tranche 3 session. The ADR set was drafted by the agent and "presented for practitioner review before Tranche 4 audit dispatch" (per cycle-status §Pause Log, Pause #3). Whether the practitioner's review of the full 7-ADR set before Tranche 4 constitutes substantive individual ADR engagement is not visible in the artifact trail. The audit loop (Tranche 4) and conformance scan served as the quality gate; the framing observations as the practitioner engagement surface. But there is no record of the practitioner challenging any individual ADR's rejected-alternatives section or provenance check in the way the DISCOVER gate's EPISTEMIC GATE challenged the orchestrator-LLM claim.

This is the residual rapid-compounding signal at DECIDE: 7 ADRs, individually audited but not individually practitioner-challenged on their specific alternatives engagement. The audit loop is a substitute quality gate for alternatives engagement; it is not equivalent to practitioner challenge on specific rejected alternatives.

### Finding 10 — New role-stakeholder interaction specs: substantive content, appropriate abstraction level (Severity: INFORMATIONAL; positive signal)

The two new role-stakeholders (Routing-Planner Ensemble, Response-Synthesizer Ensemble) were introduced in interaction-specs.md. Evaluating per the dispatch brief's criteria:

**Super-objective traces to a job in product-discovery:**
- Routing-Planner Ensemble: "Produce a deterministic JSON dispatch plan from chat-completions request content + framework's capability list, such that the framework can execute the plan without further LLM reasoning." This traces directly to the Plan stage's job in the ADR-027 pipeline; the "without further LLM reasoning" qualifier names the structural property AS-9 requires.
- Response-Synthesizer Ensemble: "Produce the user-facing chat-completion response from structured `(ORIGINAL REQUEST + PLAN + DISPATCH RESULTS)` input under strict-fidelity rules, such that the response correctly represents the dispatched work (or honest direct-completion fallback) without confabulation." This traces to the Synthesize stage's job and names the anti-fabrication commitment AS-9 + Spike ε+ μ established.

**Tasks describe workflow-level mechanics at appropriate abstraction:**
Both stakeholders have 3-4 tasks each. The tasks describe input consumption, output emission, and interaction with the broader pipeline. The abstraction level is appropriate — specific enough to be playable in a PLAY session (an observer could verify whether the stakeholder is fulfilling its tasks by watching the pipeline execute), abstract enough to survive ARCHITECT-phase wire changes.

**Playable surface quality:**
The Calibration Gate Reflect task for the Response-Synthesizer Ensemble (Task 3) correctly ties to ADR-029's three Reflect-trigger criteria, naming the specific observable signals (Rule 5 framing absence; Rule 4 rounding-drift; Rule 1 fabrication signal). This is a genuinely playable surface — during PLAY, an observer inhabiting the Response-Synthesizer Ensemble role would know when to signal a Reflect verdict.

**One observation:** the Routing-Planner Ensemble's Task 3 ("Defer multi-step composition to the framework chain-heuristic") describes a default behavior under OQ #21's open question, correctly naming it as an open question. The task's interaction mechanics are appropriately hedged. This is a placeholder task that will be superseded when OQ #21 resolves — it's honest about its scope.

Overall: the new role-stakeholder entries have substantive content and are more playable than several of the prior-cycle interaction-spec entries. No concerns.

---

## Interpretation

### Pattern assessment

The DECIDE phase produced the most architecturally substantial artifact set in the agentic-serving corpus to date. Seven ADRs, each with a Provenance check section distinguishing driver-chain items from drafting-time synthesis, constitutes a significant quality-discipline advance relative to any prior DECIDE phase. The argument-audit loop closure (5 rounds to P1/P2/P3 clean) is evidence of genuine corrective engagement, not ceremonial quality theater — each round found real issues and the corrections were substantive.

The "rapid compounding" susceptibility signature from DISCOVER has evolved at DECIDE into a different form: high-volume drafting in a concentrated session, with audit loop as the quality gate. The audit loop is a strong corrective mechanism; its weakness is that it operates on textual precision (logical gaps, claim-support failures, cross-document consistency) rather than on architectural framing (whether the rejected alternatives were engaged at sufficient depth; whether the practitioner challenged individual ADRs on their specific design choices). The five framing observations are the practitioners' deliberation surface for the latter.

Three prior advisory carry-forwards are formally closed or substantially addressed:

- **Advisory 1 (cost-distribution lens, DISCOVER):** Resolved. ADR-032's sub-promise split correctly positions cost-distribution accountability with honest-residual-uncertainty labeling. This is the most significant advisory resolution in the DECIDE phase.
- **Advisory 2 (build-complexity comparison, DISCOVER):** Resolved. OQ #19 produced the explicit build-complexity comparison; ADR-027's rejection of Tier 1 hybrid specifically cites the comparison.
- **Advisory 3 (latency floor research, DISCOVER):** Resolved. ADR-031's graded three-tier coverage (Tier A/B/C) addresses the Population A tool-family timeout research with honest Cline and Cursor positioning.

MODEL advisories A and B carry forward partially:
- **Advisory A (spike methodology pre-specification):** Carried forward to BUILD — the conformance scan's Track A refactors (Spike ζ schema `input` field; Spike ε Rule 6 addition) are the BUILD-phase artifact of the pre-specification discipline.
- **Advisory B (options-engagement documentation gap):** Partially resolved. AS-9 enters DECIDE as an invariant; downstream ADRs reference it correctly. The deferral case was not re-opened in DECIDE because the invariant's status is settled — the amendment pathway remains open per Advisory B's framing.

### Earned confidence vs. sycophantic reinforcement

The primary test: do the 7 ADRs' framings originate from their driver chains or from drafting-time synthesis?

The Provenance check sections answer this question honestly. The driver-chain items (prior spikes, prior research notes, prior ADRs, practitioner-voice verbatim) are distinguishable from drafting-time synthesis items in each ADR. The drafting-time synthesis items are labeled as such. No ADR claims a driver-chain basis for a claim that is actually drafting-time synthesis.

The clearest driver-derived framings: ADR-026's constitutional scope (practitioner-voice + OQ #18 validation); ADR-027's PRIMARY designation (OQ #19 build-complexity comparison + AS-9 satisfaction differential + OQ #18/20 research); ADR-032's sub-promise split (OQ #18 Population A validation, directly). The clearest drafting-time synthesis items: ADR-026's Population-A-only-narrowing rejection reasoning; ADR-030's bridge mechanism specification; ADR-032's three candidate capability-list discovery surfaces. All are correctly labeled.

The overall pattern is earned confidence, not sycophantic reinforcement. The three Tranche 1 preconditions (OQ #18/19/20) resolved their respective advisory carry-forwards with evidence-grounded findings. The ADR drafting operated on those findings. The audit loop corrected textual and logical gaps. The framing observations were surfaced without auto-correction.

---

## Recommendation

**No Grounding Reframe warranted.**

The DECIDE phase closed its three advisory carry-forwards from DISCOVER with substantive research (OQ #18/19/20 each produced research notes with independent evidence), addressed the DISCOVER snapshot's primary concern (cost-distribution lens in user-voice position) with an explicit architectural split in ADR-032, and produced a 7-ADR set with consistent Provenance check discipline distinguishing driver-derived from drafting-time content. The 5-round audit loop is evidence of disciplined correction, not compounding churn.

**Five advisory carry-forwards for ARCHITECT and BUILD:**

**Advisory 1 (NF2 — OrchestratorRuntime disposition as required ARCHITECT deliverable):** ARCHITECT should treat the disposition selection (a/b/c per ADR-027) as a named required deliverable, not a deferred deliberation. The current ADR-027 text names the candidates and defers without making selection required. Add one sentence in ARCHITECT's system-design update naming which disposition is selected and why; this prevents the class from stranding by inertia.

**Advisory 2 (F3 — "4 confabulation modes" validation quality differentiation):** BUILD should carry the distinction between the PLAY-note-22 case (directly witnessed failure mode, large-n) and the Spike μ.1/μ.3 cases (constructed fixtures, n=1 per mode) into its regression-test design. The four modes are not equivalently validated; BUILD's confabulation regression suite should weight coverage accordingly — the PLAY-note-22 case is the highest-priority regression test target.

**Advisory 3 (Rule 5 BUILD execution risk):** BUILD should treat Rule 5 prompt sharpening as an explicit first-priority test before treating the content-layer commitment as delivered. Spike ε' Finding ε'.1 established that the current spike ensemble systematically omits Rule 5 framing across direct-completion responses. The BUILD-default assertion ("load-bearing in the synthesizer's system prompt") requires prompt engineering work to actually deliver; the falsification trigger (migrate to headers/metadata) should be held in reserve until prompt sharpening is confirmed to work reliably.

**Advisory 4 (disposition-(i) commitment enforceability):** When the cycle following Cycle 7's BUILD convenes its DECIDE phase, place ADR-030's disposition-(i) implementation as a named opening precondition (analogous to OQ #18/19/20 in Cycle 7). Frame it as a required deliverable for ADR-030 Proposed → Accepted status advancement, not as a sprint-priority judgment. The bridge advisory is explicitly provisional; the next DECIDE gate should not allow the provisional state to persist by default.

**Advisory 5 (7-ADR set without individual practitioner challenge on rejected alternatives):** The practitioner reviewed the 7-ADR draft set before Tranche 4 audit dispatch but no individual ADR received a practitioner-challenge on its specific rejected alternatives (comparable to the EPISTEMIC GATE challenge on the orchestrator-LLM scope claim in DISCOVER). This is the residual rapid-compounding signal. ARCHITECT should treat the ADR set as well-audited (argument-audit loop closed at P1/P2/P3) but not as fully practitioner-deliberated on every rejected alternative. Specifically: if ARCHITECT's system design surfaces a reason to revisit ADR-027's rejection of "Tier 1 hybrid as PRIMARY" or ADR-028's rejection of "rule-based classifier as primary," the record supports revisiting — the rejections are evidence-grounded but not practitioner-challenged.

---

*Snapshot produced in isolated evaluation context. Advisory only; does not block ARCHITECT phase progression.*
