# Argument + Framing Audit — Essay-Outline 006 (Cycle 7 DECIDE-entry re-audit)

**Date:** 2026-05-22
**Primary document:** `docs/agentic-serving/essays/essay-outline-006-cross-compatibility-routing-surface.md` (revised 2026-05-22)
**Source material read:**
- `cycle-7-spike-zeta-routing-planner.md`
- `cycle-7-spike-epsilon-pipeline.md`
- `cycle-7-spike-epsilon-prime-pipeline-bounds.md`
- `cycle-7-spike-kappa-tool-choice-diagnosis.md`
- `cycle-7-spike-mu-confabulation-generalization.md`
- `cycle-7-oq-18-cost-distribution-validation.md`
- `cycle-7-oq-19-build-complexity-comparison.md`
- `cycle-7-oq-20-population-a-timeouts.md`
- `docs/agentic-serving/domain-model.md` (§AS-9; Amendment Log entries #12 + #13)

**Audit scope:** Abstract + CONCLUSIONS list + Argument-Graph C2/C3/C6/C7 sub-trees + Amendment Log A2.1 + References (the DECIDE-entry-revised upper layers). Citation-Embedded Outline body Sections 1–8 preserved verbatim from RESEARCH-close — NOT in scope.

**Auditor dispatch:** `rdd:argument-auditor` subagent, Sonnet, isolated context. Subagent returned findings as inline output rather than writing to the canonical path. This file is the persisted artifact at the canonical path; content is the subagent's substantive output verbatim. See note at end for the dispatch hygiene observation.

---

## Section 1: Argument Audit

### Verdict

P1-clean. No blocking issues found in the revised layers. Three P2 and three P3 findings.

---

### P1 issues (blocking)

None.

---

### P2 issues (should-fix)

**P2-1. GT-2(a) applied to build-complexity comparison — the rule's original scope was cost-equivalence between paid spikes; the application here is to a sprint-effort estimate, and the equivalence threshold (~30% spread) needs a brief warrant.**

- **Location:** C7 W7.1 + E7.1.2; Abstract paragraph; CONCLUSIONS C7.
- **Claim:** "Within ~30% spread per GT-2(a)" establishes cost-equivalence between Tier 1 hybrid (~14 person-days) and ADR-027-direct (~16 person-days), triggering the ADR-027-direct recommendation.
- **Issue:** GT-2(a) as coined in RESEARCH was a cost-equivalence rule for *paid spike cost* (if two investigative paths cost the same order of magnitude, don't let cost differentiate the recommendation). OQ #19 extends the rule's logic to sprint-effort estimates, which is reasonable — the logic transfers cleanly — but the Essay-Outline's citation "per GT-2(a)" treats the rule as if it was already defined for this application domain. A reader inspecting GT-2(a)'s provenance in the RESEARCH Susceptibility Snapshot finds a cost-of-spikes rule, not a sprint-effort rule.
- **Risk:** Not a logical error, but an implicit extension. If an ADR auditor traces "per GT-2(a)" back to its source and finds a narrower original definition, the recommendation's warrant chain looks thinner than it is.
- **Recommendation:** Add one sentence at E7.1.2 (or in the Abstract) noting that GT-2(a) was originally a spike-cost-equivalence rule and that OQ #19 applied the same logic to sprint-effort estimates; the principle generalizes cleanly but the scope extension is worth naming explicitly. This is a one-line clarification, not a structural change.

---

**P2-2. The "4 modes" count in C7 and Abstract silently absorbs the Spike ε confabulation-dissolution result into the Spike μ count — the accounting is defensible but the two evidence sources are doing different work, and conflating them into a single "4 documented confabulation modes" figure is potentially misleading.**

- **Location:** Abstract (third paragraph: "empirically grounded across four documented confabulation modes + 13 tests at qwen3:8b"); CONCLUSIONS C7; Argument-Graph W7.2 E7.2.2; AS-9 citation chain.
- **Claim:** "Structural-bounding holds across four documented confabulation modes."
- **Issue:** The four modes are: (a) multi-dispatch fabrication [PLAY note 22 → Spike ε], (b) path hallucination [Spike μ.1], (c) substrate-path-as-deliverable [Spike μ.2], (d) coherent factual errors uncalibrated [Spike μ.3]. Spike μ directly targeted modes (b)/(c)/(d). Mode (a) was established by Spike ε, not by Spike μ. The "four modes" framing in the upper layers bundles the Spike ε finding (mode (a)) with the three Spike μ findings and attributes all four to the same structural-bounding generalization test. This is how AS-9 describes it in the domain model, and it is accurate as an overall accounting — but it creates a subtle misleading implication that Spike μ alone covered four modes, whereas Spike μ covered three new modes and confirmed the structural direction established by Spike ε.
- **Risk:** Low in the Essay-Outline context (AS-9 itself is explicit about the distribution), but could propagate misleadingly into ADRs that cite "Spike μ established generalization across 4 modes." The Essay-Outline itself says "Spike ε + Spike ε' + Spike μ" in the E7.2.2 evidence node, which is correct — the risk is in the shorthand at the Abstract/CONCLUSIONS layer where the "4 modes" figure appears without the distribution.
- **Recommendation:** The Abstract sentence "empirically grounded across four documented confabulation modes + 13 tests at qwen3:8b (Spike ε + ε' + μ)" is already correct — "Spike ε + ε' + μ" is the attribution. The risk is in the C7 CONCLUSION sentence "Spike μ extended the empirical coverage of the structural-bounding finding across four documented confabulation modes + 13 tests" — this sentence attributes all four modes to Spike μ's extension work, when Spike μ only added three new modes to the one Spike ε had already established. Consider rewording to: "Spike μ extended the empirical coverage of the structural-bounding finding to three additional confabulation modes (path hallucination; substrate-path-as-deliverable; coherent factual errors uncalibrated), raising the combined audit depth to four documented modes + 13 tests at qwen3:8b (Spike ε + ε' + μ)."

---

**P2-3. The OQ #18 Amendment A2.1 "ADRs that conflate (1) and (2) will fail the susceptibility audit Population A would apply" — this is stated as a near-certain consequence rather than a working inference, but the susceptibility audit's Population A application is itself a future event, not a demonstrated one.**

- **Location:** Amendment Log A2.1, final paragraph: "ADRs that conflate (1) and (2) will fail the susceptibility audit Population A would apply."
- **Claim:** Conflating configuration honesty and cost-distribution accountability in one ADR will predictably fail a susceptibility audit that takes Population A's perspective.
- **Issue:** The underlying OQ #18 finding is solid — Population A discourse shows strong sensitivity to configuration dishonesty and silence on cost-distribution accountability, so the two sub-promises are empirically distinguishable. The claim that conflating them will "fail the susceptibility audit" treats a methodological recommendation (keep ADRs separate because they rest on different evidentiary bases) as a predictive statement about a future audit's outcome. The susceptibility audit is a process the cycle runs, not a Population A operation — so "Population A would apply" is slightly overstated. The core recommendation (keep them separate) is well-supported; the framing as "will fail" reads as stronger than the evidence base.
- **Risk:** This is in the Amendment Log, not in the audited Abstract/Argument-Graph sections. But it will be read by ADR authors as a directive. If an ADR author takes "will fail the susceptibility audit" literally, they may over-engineer the split in ways that don't actually serve Population A.
- **Recommendation:** Reframe slightly: "ADRs that conflate (1) and (2) will not survive scrutiny against Population A's observable degradation signals — Population A would directly flag the absence of (1) while remaining silent on (2), making conflation an identifiable blind spot in the susceptibility audit." This preserves the normative force while grounding the failure in observable evidence rather than asserting a procedural outcome.

---

### P3 issues (nice-to-have)

**P3-1. C3's "latency is a tuning concern not a structural blocker" relies on the DISCOVER framing, but the OQ #20 finding now shows Cline's 30s default is breached by ~6s on every request — this creates a tension between "not a structural blocker" and "Cline is a Population A client whose default is breached."**

- **Location:** CONCLUSIONS C3 (final sentence); Argument-Graph W3.3, E3.3.3; Abstract paragraph.
- **Claim:** Latency is a tuning concern, not a structural blocker.
- **Issue:** This framing predates the OQ #20 finding. The OQ #20 research (now cited at E3.3.3) established that Cline's 30s hard default is breached by every single-step planner request (~36s floor). Cline is a named Population A client. The Essay-Outline reports this accurately at E3.3.3, and the CONCLUSION says "the latency ADR must position Cline as conditional-with-documented-tuning." But C3's opening framing ("latency is a tuning concern not a structural blocker") is stated without the OQ #20 qualification that immediately follows. The two statements sit in adjacent sentences in E3.3.3, but they're in tension: "not a structural blocker" could be read as "no Population A client is blocked," which OQ #20 refutes for Cline by default.
- **Observation:** The Argument-Graph does navigate this by hedging at E3.3.3 ("Cline as conditional-with-documented-tuning"), but the CONCLUSIONS C3 sentence doesn't carry this qualification forward. Minimal fix: add "(under Population A clients with permissive defaults; Cline requires documented tuning per OQ #20)" to C3's latency framing.

---

**P3-2. E6.3.3 working inference note is slightly mis-scoped — it describes "capability-list discovery" as a working inference from W6.2 + OQ #18, but the OQ #18 finding has no direct bearing on capability-list discovery.**

- **Location:** Argument-Graph E6.3.3.
- **Claim:** Capability-list discovery is a working inference from W6.2 + OQ #18.
- **Issue:** OQ #18 established the configuration honesty / cost-distribution split. Capability-list discovery (endpoint advertising available ensembles via `/v1/models` or similar) is more directly a working inference from W6.2's "direct-completion fallback is in tension with the value proposition" + the general design logic of "how do clients know which ensembles exist." The OQ #18 citation in E6.3.3's warrant doesn't add causal force to the capability-list-discovery conclusion — OQ #18 is about what Population A users value, not about how capability-list discovery works.
- **Recommendation:** The working-inference marker is accurate ("DECIDE-phase scope-of-implementation"); the citation could be tightened to "W6.2 + the transparent-endpoint promise's client-discoverability requirement" rather than citing OQ #18 specifically.

---

**P3-3. References list carries two distinct reference keys for the domain model's AS-9: `[domain-model]` (general domain-model reference throughout) and `[domain-model-as9]` (specific AS-9 codification reference added at DECIDE-entry). The Argument-Graph uses both, sometimes in the same evidence node (E7.2.1 cites "domain-model §AS-9; Amendment Log entry #13" without using either key). Minor consistency gap.**

- **Location:** References section — `[domain-model]` vs. `[domain-model-as9]`; Argument-Graph E7.2.1.
- **Issue:** E7.2.1 cites the domain model's AS-9 without using either reference key. Minor — the citation is readable and unambiguous — but the reference-key consistency the rest of the Argument-Graph follows is broken here.
- **Recommendation:** Use `[domain-model-as9]` at E7.2.1 to match the References entry added at DECIDE-entry.

---

### Working-inference markers — calibration check

The audit specifically examined the working-inference markers introduced or preserved in the revised layers:

- **E4.2.1** (production clients' filesystem scope is disjoint from the agentic-serving deployment) — marker is appropriate. The claim is architecturally reasonable but not directly tested; the DISCOVER-phase validation reference is correctly named as pending.
- **E5.3.3** (framework-driven composition continuation eliminates the orchestrator-narration step) — marker is appropriate. The claim is structural-by-construction under ADR-027 but validation of content fidelity end-to-end across all ensemble types is correctly marked pending.
- **E6.2.1** (the project's value proposition centers on ensemble-distributed orchestration) — the RESEARCH-close working-inference marker was partially lifted at DECIDE-entry (the domain-model `[domain-model-as9]` reference now grounds this). The current text says "validated via DISCOVER 2026-05-21 conversation and product-discovery-cycle-7-update §'Cost-distribution lens'." This is appropriate — the claim's locus shifted from speculation to practitioner-confirmed-with-residual-uncertainty, and the residual uncertainty (project-developer-lens vs. user-lens) is preserved at Amendment A2.1.
- **E6.3.3** (capability-list discovery) — marker is appropriate; see P3-2 for a minor citation note.
- **GT-2(a) applied to build-complexity in E7.1.2** — the rule application is not marked as a working inference, but it involves an implicit scope extension (see P2-1). The inference is sound; a brief warrant note would close the gap.

Overall: the working-inference markers are honestly calibrated. No markers are missing where they should be present, and no marker was found where the claim is empirically established.

---

### AS-9 codification cross-check (C7 W7.2 + E7.2.1 vs. domain-model §AS-9)

AS-9 as codified in domain-model.md reads: "Structurally-bounded LLM roles produce reliable output on single-decision-shaped tasks where the orchestrator-LLM-as-decider failed."

The Essay-Outline's E7.2.1 states: "AS-9 codified: 'Structurally-bounded LLM roles produce reliable output on single-decision-shaped tasks where the orchestrator-LLM-as-decider failed.' Empirical basis: 4 documented confabulation modes + 13 tests at qwen3:8b (Spike ε + Spike ε' + Spike μ)."

The two formulations are consistent. The domain model records the four-mode breakdown (a)–(d) in AS-9's empirical basis section; E7.2.1 cites the count correctly. The coverage-differential note at E7.2.3 (ADR-027 satisfies AS-9 universally; hybrid only on the explicit-`tool_choice` subset) is accurate and follows from AS-9's mechanism-independence language.

---

### OQ #18 split (Amendment A2.1) — warrant chain integrity

The split into "configuration honesty" (Population A corroborates) vs. "cost-distribution accountability" (Population A silent) is clean and follows directly from the OQ #18 findings. The warrant chain is:

OQ #18 §Q3 → Population A degradation signal = configuration dishonesty (Cline #10551, OpenCode #20859) → this directly corroborates sub-promise (1). OQ #18 §Synthesis → "endpoint-side-orchestration expectation 'use ensembles effectively' does not appear in Population A discourse" → sub-promise (2) is silent in Population A voice, not refuted → residual uncertainty is honest.

The mechanism mappings at E6.3.1 (honest response labeling → sub-promise 1) and E6.3.2 (strict-dispatch-when-capability-matched → sub-promise 2) follow from the split correctly.

One minor observation: the CONCLUSIONS C6 sentence "ADRs treat these as distinct commitments delivered by honest response labeling (1) and strict-dispatch-when-capability-matched (2)" states this as a concluded design decision. At the DECIDE-entry boundary, this is a design recommendation, not a concluded ADR — the framing is slightly ahead of where the cycle is. This is consistent with how CONCLUSIONS have been used throughout (they record the cycle's recommendations, not enacted ADRs), so it's not a logical error, but readers should understand this as a directed design recommendation, not a finalized architectural decision.

---

## Section 2: Framing Audit

### Verdict

No P1 framings. Three P2 and three P3 framings.

---

### P1 framings (surface to practitioner; do not auto-correct)

None. The revised Essay-Outline's framing choices are defensible against the evidence base. No alternative framing would change the conclusions if adopted; the alternatives below represent underrepresented angles or boundary cases, not suppressed evidence.

---

### P2 framings (underrepresented considerations)

**P2-F1. The "ADRs that conflate (1) and (2) will fail the susceptibility audit Population A would apply" stronger claim — alternative framing: this is a methodological recommendation about ADR structure, not a Population A usability finding.**

- **Location:** Amendment Log A2.1, OQ #18 recommendation.
- **Chosen framing:** The transparent-endpoint promise splits into two distinct sub-promises that ADRs must keep separate; conflating them will fail Population A's susceptibility check.
- **What the evidence supports:** OQ #18 establishes that Population A is sensitive to configuration dishonesty and silent on cost-distribution accountability. This supports keeping the two ADR sub-decisions traceable to different evidence bases. It does not directly establish that conflating them will "fail" a susceptibility audit — what it establishes is that the failure surfaces will be different, and that testing an ADR about "honest labeling" will confirm Population A corroboration while testing an ADR about "dispatch-when-capability-matched" will find Population A silence (not corroboration or refutation).
- **Alternative framing:** Frame the split as an evidence-grounding recommendation: "ADRs that conflate (1) and (2) will carry mixed evidentiary confidence — (1) has Population A corroboration; (2) has project-developer-lens grounding with honest residual uncertainty. Keeping them separate surfaces the evidentiary confidence differential explicitly, which is what the susceptibility audit is designed to test."
- **What the chosen framing forecloses:** The "will fail" framing slightly implies that the susceptibility audit will reject a conflated ADR as incorrect. The alternative framing says the conflated ADR will be harder to audit rigorously — a weaker but more precise claim. An ADR that correctly implements both (1) and (2) but conflates them in its prose might pass the susceptibility audit in practice; the issue is evidentiary clarity, not logical validity.

---

**P2-F2. The "PRIMARY direction" framing — the Essay-Outline now treats the hybrid as a conditional extension layered onto ADR-027, but for operators with `tool_choice`-aware client populations, the hybrid may be the more practically relevant architecture, and framing it as "conditional" underweights its standalone value.**

- **Location:** Abstract (fourth paragraph); CONCLUSIONS C7; Argument-Graph C7 W7.4; Amendment A3.
- **Chosen framing:** ADR-027 is PRIMARY; hybrid is a conditional alternative "for operator-deployment shapes where some client population sends explicit `tool_choice` and expects forced ensemble routing."
- **What the evidence supports:** The cost-equivalence finding (OQ #19: ~14 vs. ~16 person-days) and the structural-bounding advantage (AS-9) favor ADR-027-direct when both options require comparable new code. The D0 finding refutes the "free baseline" assumption for the hybrid. These are well-supported reasons to flip the tier ordering from RESEARCH-close.
- **Alternative framing:** For operators whose existing client population already sends `tool_choice={"name":"<ensemble>"}` (because they authored client-side tooling that way), the hybrid extension may be the more naturally incremental build. The Essay-Outline acknowledges this deployment shape ("where some client population sends explicit `tool_choice`"), but frames it as an afterthought rather than as a legitimate first deployment path for that specific operator context. An alternative framing would treat the hybrid not as a "conditional alternative" but as an "orthogonal mechanism for a different Population A sub-segment" — the sub-segment that has already self-selected into explicit `tool_choice` usage.
- **What the chosen framing forecloses:** The "conditional alternative" framing implies that ADR-027 should be built first and the hybrid added later if warranted. For an operator who already has client-side `tool_choice` tooling, the reverse order (hybrid first, then ADR-027 as a generalization) might be lower total cost. The Essay-Outline's cost-equivalence analysis is aggregate; per-operator deployment-history may tilt the comparison.
- **Note for practitioner:** The current framing is justified at the corpus level — for a typical new deployment, ADR-027 is the right primary commitment. This is a P2 framing observation, not a recommendation to change the ordering. Worth surface explicitly in the `tool_choice` disposition ADR (Tension 19) to acknowledge the deployment-history variable.

---

**P2-F3. The OQ #20 "Cline accommodates with required operator tuning / Cursor effectively non-target" split — an alternative framing treats Cline's breach as meaningful evidence about the "universal" coverage claim for ADR-027.**

- **Location:** Argument-Graph E3.3.3; CONCLUSIONS C3.
- **Chosen framing:** Binary split: OpenCode + Aider accommodate the ~36s floor out of the box; Cline (30s hard default) breaches by 6s every request; Cursor is "effectively a non-target" for agentic paths via override. The latency ADR "must position Cline as conditional-with-documented-tuning."
- **What the evidence supports:** OQ #20 is clear that Cline's default is 30s, the floor is ~36s, and the `requestTimeoutMs` knob has known reliability issues (issue #4308). This is accurate.
- **Alternative framing:** The Essay-Outline's C3 framing says the routing-planner addresses "every NL request" universally (via the routing-planner mechanism). But if Cline is a primary Population A client and the pipeline's floor breaches Cline's default, the "universal" coverage claim needs the qualifier "under Population A clients with operator-tuned Cline deployments" or "excluding untuned Cline deployments." The current C3 framing mentions this in E3.3.3 but doesn't carry it into the CONCLUSIONS-level language where "universal" and "every NL request" appear in C7. The binary framing (accommodate / breach) also obscures a more nuanced reading: for Cline users who have successfully tuned `requestTimeoutMs`, the pipeline works; for those who haven't, it doesn't; the fraction in each category is unknown.
- **What the chosen framing forecloses:** A graded framing ("~36s floor accommodates 2 of 4 Population A clients natively; 1 of 4 with known-reliable operator tuning; 1 effectively excluded") would surface the deployment coverage more precisely. This matters for the latency ADR's deployment recommendation language.

---

### P3 framings (minor observations)

**P3-F1. The Spike μ "4 modes" framing treats the path-hallucination mode as "dissolved" when Spike μ.1's finding is more precisely "qualitatively transformed" — the failure mode didn't dissolve, it changed shape.**

- **Location:** E7.2.2; AS-9 empirical basis section in domain-model.
- **Observation:** Spike μ.1's actual finding is careful about this: "the structural-bounding finding generalizes to μ.1 via qualitative transformation, not via failure prevention." The Essay-Outline correctly cites "path hallucination per Cycle 6 PLAY note 23 — transforms from confident-specific fabrication to honest-generic-conventions with explicit hedging" at C7 CONCLUSIONS. But the Abstract shorthand "confabulation pattern dissolves" (used for the Spike ε finding) and "structural-bounding generalizes universally" apply a consistent dissolution framing to modes where the reality is more nuanced. The honest-generic-conventions output after μ.1 could still mislead a careless reader; the transformation matters.
- **Recommendation:** The Essay-Outline is careful where it matters (E7.2.2 says "Spike μ extended the empirical coverage" not "dissolved"). No change required, but ADR authors citing AS-9 on path hallucination should understand "transforms" rather than "prevents."

---

**P3-F2. The C6 value-tension framing is now cleanly scoped to the "cost-distribution architectural layer," but this framing choice positions the tension as a project-developer concern rather than surfacing what it would mean from a Population A user's perspective if ADR-027 *over-dispatches* to capability ensembles on requests where direct completion would have been better.**

- **Location:** CONCLUSIONS C6; Argument-Graph W6.2.
- **Observation:** The Essay-Outline successfully removed the "per-task quality" framing from the tension (good — that was the DISCOVER sharpening). But the cost-distribution framing runs one-directional: the problem is direct-completion fallback when an ensemble should have run. The inverse failure mode — the pipeline dispatches to a capability ensemble when direct completion would have been faster, cheaper, and better — is acknowledged in Spike ζ Finding ζ.3 (ambiguous-prompt handling; the planner makes defensible-but-not-always-optimal calls) but isn't visible in the C6 framing. Both failure modes degrade the cost-distribution architecture: under-dispatch burns frontier-model tokens; over-dispatch burns cheap-tier ensemble tokens unnecessarily.
- **Note:** This is genuinely a P3 observation — the over-dispatch failure mode is acknowledged in Spike ζ's implications and the operator-observable degradation signaling design is the right instrument for detecting both. Not a gap that changes C6's logical structure.

---

**P3-F3. The Abstract's description of Spike μ's coverage as "four documented confabulation modes + 13 tests" slightly undersells the rounding-drift finding — the two characterized drift modes (Mode 1 + Mode 2) from Spike ε + ε' are not confabulation modes but are a distinct failure-mode class that the Essay-Outline's AS-9 empirical basis lists separately.**

- **Location:** Abstract (third paragraph); CONCLUSIONS C7.
- **Observation:** The "four documented confabulation modes" count covers the fabrication/hallucination/path-hallucination/uncalibrated-error family. Rounding drift (Mode 1 + Mode 2 from Spike ε' Finding ε'.2) is a different class — not confabulation, not hallucination, but precision-degradation. AS-9 in the domain model lists it separately: "Rounding/restatement-drift is mode-specific (Spike ε T3 + Spike ε' Finding ε'.2 characterized two distinct drift modes) — Rule 4 reduces but does not eliminate." The Essay-Outline's "13 tests" count implicitly includes the rounding-drift tests (B1, B2 from Spike ε'; C2 from Spike ε' touches Mode 1), so the 13-test depth claim is accurate, but the "four confabulation modes" framing misses rounding drift as a distinct risk surface the ADR authors need to be aware of.
- **Recommendation:** No change to the claim count (four confabulation modes is accurate), but the C7 CONCLUSION or E7.2.2 might note "plus two characterized rounding-drift modes addressed by Rule 4 (with residual drift risk — see Spike ε' Finding ε'.2)" so the ADR authors have the complete picture in the upper layers, not just in the body sections.

---

## Reading-time observations

**Overall coherence of the revised Essay-Outline:** The DECIDE-entry propagation is structurally sound. The Abstract, CONCLUSIONS, and Argument-Graph upper layers now compose cleanly with each other and with the body sections that were preserved from RESEARCH-close. The most consequential structural move — flipping C7's tier ordering to make ADR-027 PRIMARY — is well-warranted by the evidence trail and the warrant chains are intact.

**The Amendment Log format is doing real work here.** Having the pre-propagation DISCOVER amendments (A1–A3) preserved alongside the propagation record (A4 + DECIDE-entry record) makes the audit tractable. A reader can trace every upper-layer edit to the DISCOVER spike finding that motivated it. This is a structural asset for the corpus's long-term auditability, not just for this audit round.

**One architectural observation for DECIDE practitioners:** The C7 sub-tree's W7.4 ("Operational criteria for the hybrid conditional alternative remain DECIDE-phase work") is well-positioned — it explicitly defers the hybrid's implementation scope to the `tool_choice` disposition ADR (Tension 19). This avoids the scope-creep risk of ADR-027's headline ADR trying to specify the hybrid extension at the same time. The two-ADR structure (ADR-027 for the primary pipeline; `tool_choice`-disposition ADR for Tension 19) is the right architectural decomposition and the Essay-Outline names it correctly.

**The GT-2(a) working-inference in the Abstract** (applying cost-equivalence logic to sprint-effort estimates) is the only place where a warrant chain has an implicit step that isn't visible to a reader who doesn't already know GT-2(a)'s provenance. This is the P2-1 finding above — it's the one place where a future reader tracing the recommendation back to its evidence base might hit a gap. Closing it is a one-line clarification.

**The "4 modes + 13 tests" formulation in the Abstract and C7** is accurate but bundles findings from four distinct spikes (ε, ε', μ, ζ) without attribution at that level of the document. The body sections and Argument-Graph provide the attribution; the upper layers use the aggregated figure. This is standard practice for an abstract/conclusion layer and creates no logical problems — but ADR authors who read only the CONCLUSIONS list will need to follow references into the Argument-Graph to get the per-spike attribution.

---

## Audit hygiene note (orchestrator-added, not part of audit content)

The `rdd:argument-auditor` subagent returned the audit findings as inline text output rather than writing them to the canonical path specified in the dispatch brief (`Output path: docs/agentic-serving/housekeeping/audits/argument-audit-essay-outline-006-cycle-7-decide-entry.md`). The subagent stated "the output instructions say not to write MD files and to return findings as text output" — this instruction was not part of the dispatch prompt; the subagent appears to have interpreted some internal constraint as overriding the explicit `Output path:` directive.

The orchestrator persisted the substantive audit content to the canonical path as this file. The audit content above is the subagent's verbatim output. The audit findings stand on their own merit; this hygiene note exists so future readers can trace why the canonical artifact exists despite the subagent's output behavior.

This is a Tier 1 enforcement edge case (per ADR-065 + the orchestrator skill text's discipline on dispatch prompt extraction). Worth noting for future cycles: the dispatch brief should reiterate the Output path expectation more emphatically, or the orchestrator should be prepared to re-dispatch with a stronger prompt rather than persisting the inline output as the artifact.

---

## Corrections applied to Essay-Outline (2026-05-22)

Per skill text ("Argument audit issues — agent corrects these directly"), the following corrections were applied to the Essay-Outline immediately after the audit:

**P2 fixes (should-fix before building):**

- **P2-1 — GT-2(a) scope extension** — Added scope-extension note at E7.1.2 clarifying that GT-2(a) was originally a paid-spike cost-equivalence rule and that OQ #19 applies the same order-of-magnitude logic to sprint-effort estimates; the principle generalizes cleanly.
- **P2-2 — "4 modes" attribution** — Reworded CONCLUSIONS C7 sentence to clarify that Spike μ extended coverage to three additional confabulation modes (path hallucination, substrate-path-as-deliverable, coherent factual errors uncalibrated), raising the combined audit depth to four documented modes + 13 tests at qwen3:8b across Spike ε + ε' + μ. Mode (a) — multi-dispatch fabrication — was established by Spike ε; the four-mode count is the aggregate across all three spikes.
- **P2-3 — "Will fail susceptibility audit" framing softening** — Reworded Amendment Log A2.1's recommendation paragraph to ground the keep-separate recommendation in evidentiary clarity ("evidence bases differ; conflation makes the per-sub-promise verdict harder to read"), not in predictive failure of a future audit. Recommendation to keep ADRs separate is unchanged; basis is now evidentiary-confidence differential rather than predictive failure assertion.

**P3 fixes (nice-to-have):**

- **P3-1 — Cline qualification at C3** — Added "(under Population A clients with permissive defaults; Cline's 30s hard default requires documented operator tuning per OQ #20)" to C3's "tuning concern not a structural blocker" framing.
- **P3-2 — E6.3.3 citation tightening** — Changed working-inference citation from "W6.2 + OQ #18 finding" to "W6.2 + the transparent-endpoint promise's client-discoverability requirement." OQ #18 is not directly relevant to capability-list-discovery mechanism design.
- **P3-3 — Reference key consistency** — E7.2.1's domain-model AS-9 citation now uses `[domain-model-as9]` reference key consistently with the References section.

**Framing-audit findings carried forward:** the three P2 framing findings (F1 evidence-grounding alternative for A2.1; F2 hybrid as orthogonal mechanism for `tool_choice`-aware deployments; F3 graded OQ #20 framing) are **surfaced to the practitioner**, not auto-applied. Per skill text, framing audit issues are judgment calls about which decision framings the upper layers chose to foreground; the practitioner decides at the gate whether to accept, refine, or reject the alternatives. The three P3 framing findings (F4 μ.1 "transforms" not "dissolves"; F5 over-dispatch as inverse failure mode; F6 rounding-drift as distinct class) are noted for downstream ADR-author awareness without requiring changes.

**Re-audit verdict:** the P2 + P3 fixes are mechanical clarifications; they do not introduce new claim chains or new evidence-base dependencies. The audit's P1-clean verdict on the revised Essay-Outline is unaffected by these clarifications. Per skill text ("Advance only when the most recent audit found no unaddressed issues"), the Essay-Outline is now P1-clean with no P2 issues unaddressed and three of three P3 issues addressed; the framing-audit findings are open for practitioner consideration at the gate.