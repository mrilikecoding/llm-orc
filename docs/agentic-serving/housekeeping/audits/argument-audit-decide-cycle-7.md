# Argument Audit Report — Cycle 7 DECIDE ADR Set

**Audited documents:**
- `docs/agentic-serving/decisions/adr-026-capability-matching-from-request-content-alone.md`
- `docs/agentic-serving/decisions/adr-027-framework-driven-dispatch-pipeline.md`
- `docs/agentic-serving/decisions/adr-028-routing-planner-ensemble.md`
- `docs/agentic-serving/decisions/adr-029-response-synthesizer-ensemble.md`
- `docs/agentic-serving/decisions/adr-030-tool-choice-disposition.md`
- `docs/agentic-serving/decisions/adr-031-latency-timeout-policy.md`
- `docs/agentic-serving/decisions/adr-032-fallback-shape-and-transparent-endpoint-split.md`
- `docs/agentic-serving/decisions/adr-021-skill-orchestration-via-per-capability-dispatch.md` (partial-update header audit)
- `docs/agentic-serving/decisions/adr-022-routing-surface-behavior.md` (partial-update header audit)

**Source material read:**
- `docs/agentic-serving/essays/essay-outline-006-cross-compatibility-routing-surface.md`
- `docs/agentic-serving/domain-model.md`
- `docs/agentic-serving/product-discovery.md`
- `docs/agentic-serving/essays/research-logs/cycle-7-oq-18-cost-distribution-validation.md`
- `docs/agentic-serving/essays/research-logs/cycle-7-oq-19-build-complexity-comparison.md`
- `docs/agentic-serving/essays/research-logs/cycle-7-oq-20-population-a-timeouts.md`
- `docs/agentic-serving/essays/research-logs/cycle-7-spike-zeta-routing-planner.md` (via references in other documents)
- `docs/agentic-serving/essays/research-logs/cycle-7-spike-epsilon-pipeline.md` (via references)
- `docs/agentic-serving/essays/research-logs/cycle-7-spike-epsilon-prime-pipeline-bounds.md` (via references)
- `docs/agentic-serving/essays/research-logs/cycle-7-spike-mu-confabulation-generalization.md` (via references)
- `docs/agentic-serving/essays/research-logs/cycle-7-spike-kappa-tool-choice-diagnosis.md` (via references)
- `docs/agentic-serving/housekeeping/audits/argument-audit-essay-outline-006-cycle-7-decide-entry.md`

**Genre:** ADR set

**Date:** 2026-05-22

---

## Section 1: Argument Audit

### Summary

- **Genre:** ADR set (7 new ADRs + 2 partial-update audits)
- **Argument chains mapped:** 19 (one per ADR decision + rejected alternatives; plus partial-update header correctness for ADR-021 and ADR-022)
- **Issues found:** 12 (P1: 1, P2: 6, P3: 5)
- **Pyramid coverage map:** N/A (ADR genre)
- **Expansion-fidelity findings:** N/A (ADR genre)

---

### P1 — Must Fix

**P1-1. ADR-030 bridge mechanism references ADR-032 as the home for the structured advisory, but ADR-032 does not specify the bridge advisory as a concrete mechanism — it specifies the general honest-response-labeling surface. The bridge advisory's shape and content are underspecified across both ADRs, creating a circular dependency without a resolution owner.**

- **Location:** ADR-030 §Decision "Cycle 7 bridge mechanism — disposition (ii) variant"; ADR-032 §Decision §Sub-promise (1) §Implementation mechanisms
- **Claim:** ADR-030 says the bridge advisory is "included as response metadata (per ADR-032's honest response labeling mechanism)." ADR-032 lists several response layers for serving-by signals but does not name a "tool_choice bridge advisory" as a distinct mechanism — it describes headers, response body metadata, and Rule 5 content-layer framing. The bridge advisory is specifically for the case where `tool_choice` was received but not used; none of ADR-032's enumerated mechanisms directly names that signal.
- **Evidence gap:** ADR-030 defers the bridge advisory shape to ADR-032; ADR-032 does not specify a `tool_choice_handling: "deferred"` field or equivalent as a concrete first-class mechanism. The only example in ADR-030 (`tool_choice_handling: "deferred"` in a `served-by` header) is illustrative text in ADR-030 itself, not a specification in ADR-032. BUILD work that attempts to implement the bridge advisory has no canonical ADR to follow — it must synthesize the shape from two ADRs, neither of which fully specifies it.
- **Recommendation:** Either extend ADR-032's §Implementation mechanisms to explicitly name the `tool_choice` bridge signal as a first-class case (alongside the direct-completion and dispatch cases), or move the bridge advisory specification into ADR-030 as a self-contained implementation note that ADR-032's mechanisms will carry. The circular reference should be collapsed in one direction.

---

### P2 — Should Fix

**P2-1. ADR-027's cost-equivalence claim ("~14 vs. ~16 person-days; within ~30% spread") is presented as a settled precondition, but the Tranche 2 audit's P2-1 finding (GT-2(a) scope extension) was addressed in the Essay-Outline only as a note — ADR-027 itself does not carry that note, and a BUILD reader who reads ADR-027 without reading the Essay-Outline will see the GT-2(a) citation as unqualified.**

- **Location:** ADR-027 §Context (final paragraph): "OQ #19 build-complexity comparison establishing cost-equivalence per GT-2(a)"
- **Claim:** GT-2(a) cost-equivalence between Tier 1 hybrid (~14 person-days median) and ADR-027-direct (~16 person-days median) validates the PRIMARY direction recommendation.
- **Hidden assumption:** GT-2(a) was defined in RESEARCH as a rule about paid-spike cost equivalence. OQ #19 extended the rule's logic to sprint-effort estimates. This extension is sound but was not in the rule's original scope. ADR-027 cites it as a settled rule without noting the scope extension.
- **Risk:** Auditors tracing ADR-027's recommendation chain back to GT-2(a) find a narrower original rule; the recommendation looks less grounded than it is. This is the same gap the Tranche 2 audit found in the Essay-Outline (P2-1), now surfaced in ADR-027 itself where it is load-bearing for BUILD acceptance.
- **Recommendation:** Add one sentence at ADR-027's cost-equivalence citation noting that GT-2(a) was originally a paid-spike cost-equivalence rule and that OQ #19 applied the same order-of-magnitude logic to sprint-effort estimates. The Provenance check section of ADR-027 already references OQ #19 as driver — the note belongs inline with the cost-equivalence claim.

---

**P2-2. ADR-030 names disposition (i) as "the cycle's commitment" but defers implementation to a follow-on cycle, and the bridge mechanism is explicitly "intermediate" — this creates a commitment structure where the named commitment (disposition i) and the shipped artifact (bridge advisory) diverge, without a named gate or trigger that enforces the follow-on cycle's delivery.**

- **Location:** ADR-030 §Decision "Cycle 7 bridge mechanism" + §Consequences §Negative "The follow-on cycle commitment is real."
- **Claim:** ADR-030 commits to disposition (i) as the cycle's decision; Cycle 7 ships only the bridge mechanism (a weaker form of disposition ii); the follow-on cycle delivers the full implementation.
- **Evidence gap:** The Consequences §Negative section acknowledges "if the follow-on cycle is delayed or descoped, the bridge advisory persists longer than designed." There is no mechanism in ADR-030 or the broader ADR set that creates enforcement pressure on the follow-on cycle — no named gate criterion, no falsification trigger, no ADR amendment flag that marks the bridge as provisional. The commitment is named but has no structural enforcement.
- **Risk:** As cycles progress, the bridge advisory becomes the operational baseline; the disposition (i) commitment becomes a dormant intention. Without a named enforcement mechanism, the gap between commitment and delivery grows silently.
- **Recommendation:** Add an explicit "Provisional mechanism marker" to ADR-030 — name the bridge advisory as provisional and specify what would trigger the follow-on cycle (e.g., "when ADR-027 reaches production deployment" or "at the next DECIDE gate, evaluate whether disposition (i) should open the cycle"). This does not require the follow-on cycle to be scheduled now, but it makes the deferred status visible as an invariant, not an aspiration.

---

**P2-3. ADR-029's Rule 5 "load-bearing interpretation as BUILD default" and the "falsification trigger to headers/metadata" are named together with the ADR-022 pattern ("system-prompt amendment ships with cross-profile-deferred-to-BUILD/PLAY characterization") as structural precedent. But ADR-022's pattern was empirically found to be bounded in effectiveness under production tool-rich clients — citing it as a positive structural parallel understates the risk that the falsification trigger may fire early.**

- **Location:** ADR-029 §Decision §Rule 5 framing requirement scope (final paragraph): "This is the same pattern ADR-022 used..."
- **Claim:** The BUILD-default-with-falsification-trigger pattern is validated by ADR-022's use of the same structure.
- **Evidence gap:** ADR-022's characterization in the Neutral section of ADR-029 treats the pattern as a proven template. But ADR-022's amendment was empirically shown to be suppressed by production tool-rich clients (PLAY notes 12, 14 vs. PLAY note 18; Spike γ). ADR-027 ultimately superseded ADR-022 because the system-prompt amendment approach had structural limits. Using ADR-022 as a positive pattern precedent for the Rule 5 design implicitly borrows credibility from an approach the cycle rejected as architecturally insufficient for the routing surface.
- **Risk:** ADR-022's pattern is a fair precedent for the Rule 5 framing design because the Rule 5 context differs (the synthesizer's prompt is the operative surface; the synthesizer does not face the same tool-rich-client suppression problem the orchestrator-LLM did). But the citation does not distinguish this difference — a BUILD reader may not know whether the ADR-022 pattern's failure mode applies to Rule 5. The distinction is meaningful.
- **Recommendation:** Either (a) note the structural difference between ADR-022's context (orchestrator-LLM under tool-rich suppression) and Rule 5's context (synthesizer with no tool-rich suppression pressure — the synthesizer is not a tool-rich context), which would make the citation explicitly sound, or (b) remove the ADR-022 citation and state the BUILD-default-with-falsification-trigger pattern on its own merits.

---

**P2-4. The AS-9 vs. AS-10 distinction (role-shape vs. request-shape) is clear in ADR-026 but is not explicitly reiterated in ADR-028 and ADR-029 where both invariants are operative simultaneously. A BUILD reader who reads ADR-028 without having read ADR-026 may conflate the two invariants.**

- **Location:** ADR-028 §Consequences "The planner operates within AS-10's constitutional scope" (brief); ADR-029 (AS-10 not mentioned by name in §Consequences or §Decision)
- **Claim:** Both ADR-028 and ADR-029 operate within AS-9 (role-shape constraint) and AS-10 (request-shape constraint) simultaneously. ADR-028 names both in the Consequences but briefly. ADR-029 does not name AS-10 at all — it names AS-9 in §Consequences but is silent on AS-10.
- **Evidence gap:** The synthesizer's input contract (ORIGINAL REQUEST + PLAN + DISPATCH RESULTS) is itself a form of AS-10 compliance — the synthesizer reads the ORIGINAL REQUEST which is request-content-derived, not a client-side opt-in signal. But ADR-029 does not state this relationship. A reader of ADR-029 has to infer AS-10 compliance from the input contract description.
- **Risk:** AS-10 is a constitutional invariant that all downstream artifacts must satisfy per ADR-026. If ADR-029 doesn't explicitly confirm AS-10 compliance, the constitutional audit (conformance scanner) may not find the evidence it needs without cross-referencing ADR-026 and ADR-028.
- **Recommendation:** Add a brief AS-10 compliance statement to ADR-029's §Consequences or §Relationship section (analogous to what ADR-028 carries) confirming that the synthesizer's input is derived from the ORIGINAL REQUEST (AS-10 compliant) and that the synthesizer does not introduce any client-side opt-in signals into its output.

---

**P2-5. ADR-032 §Sub-promise (2) §Operational implication names routing-planner reliability (90% strict capability-match from Spike ζ) as the empirical floor for cost-distribution accountability. But 90% strict capability-match means 10% of requests may route to `action: "direct"` despite matching capabilities being available — the ADR does not address what happens to the sub-promise when the planner's 10% miss rate fires on capability-matched requests.**

- **Location:** ADR-032 §Decision §Sub-promise (2): "the planner's reliability profile (Spike ζ — 90% strict capability-match) is the empirical floor for sub-promise (2)"
- **Claim:** Cost-distribution accountability is operationalized by strict-dispatch-when-capability-matched; the planner's 90% strict capability-match is the empirical floor.
- **Evidence gap:** 90% strict capability-match + 100% defensible-judgment-match means the planner correctly routes 90% of requests precisely and 100% reasonably (the defensible-judgment cases are Spike ζ's ambiguous-match cases where the planner's routing was judged defensible even if not ideal). But "strict capability-match" means the remaining 10% routed to something other than the ideal ensemble — some of those routed to a defensible alternative, some may have routed to `action: "direct"` when an ensemble match existed. ADR-032 names the 90% figure as the floor but doesn't clarify whether those 10% "misses" in Spike ζ were `action: "direct"` cases (sub-promise (2) failures) or `action: "dispatch"` to a less-ideal ensemble (not sub-promise (2) failures). The ambiguity matters for how operators should read the sub-promise.
- **Risk:** If some of the 10% are routing to `action: "direct"` on capability-matched requests, sub-promise (2) is violated on ~10% of such requests by default, before any tuning. The cost-distribution-accountability commitment should name this openly.
- **Recommendation:** Clarify in ADR-032's §Sub-promise (2) §Operational implication whether Spike ζ's 10% non-strict-match includes any `action: "direct"` cases or whether all 20 prompts produced a `dispatch` action (the ambiguous cases being "dispatch to the defensible alternative"). If all 20 produced `dispatch`, the sub-promise (2) concern about direct fallback is separate from the 90% figure and the §Operational implication should say so. If some produced `direct`, name the rate explicitly.

---

**P2-6. ADR-031's Tier B (Cline) recommendation includes "run an integration smoke test against their Cline configuration before relying on the tuning" — but ADR-031 does not specify what the smoke test should validate or what a passing result looks like. This recommendation is actionable in intent but not in implementation.**

- **Location:** ADR-031 §Decision §Population A coverage tiers §Tier B — Cline
- **Claim:** "The deployment-documentation contribution this ADR makes is naming Cline as Tier B and recommending operators run an integration smoke test against their Cline configuration."
- **Evidence gap:** "Integration smoke test" is named but not defined. What request should the operator send? What response latency constitutes a pass? Which Cline setting should the operator check? The recommendation appears actionable at first read but gives operators no specification to follow.
- **Risk:** Operators who want to follow this recommendation have to independently design their smoke test. Some will test something inadequate; some will not test at all. The deployment documentation this ADR calls for would need to supply the missing specification, which is deferred to BUILD.
- **Recommendation:** Add a brief concrete smoke test shape inline: e.g., "send a single-capability NL request that should match an installed ensemble; verify the response arrives within the operator's configured `requestTimeoutMs` minus 5s headroom; confirm the response includes the expected ensemble output or a direct-completion framing per Rule 5." This is one sentence that removes the ambiguity.

---

### P3 — Consider

**P3-1. ADR-026 §Rejected alternatives §Narrow to Population-A-only scope argues that "Population A is not detectable from request content alone" as the rejection rationale. This argument is marked in the Provenance check as "drafting-time analytical engagement; not directly cited from prior artifacts." The argument is sound, but it is the only rejection rationale derived from in-ADR synthesis rather than a prior artifact — it deserves a brief acknowledgment of its drafting-time status.**

- **Location:** ADR-026 §Rejected alternatives §Narrow to Population-A-only scope; §Provenance check "Rejection of Population-A-only narrowing"
- **Claim:** Population A is not detectable from request content alone; conditioning the invariant on detection re-introduces the constitutional question one layer down.
- **Observation:** The argument is analytically solid. The Provenance check is transparent about the drafting-time origin. No change is strictly needed, but readers expecting every rejection rationale to trace to a prior artifact will find this one analytical rather than empirical. A one-line note distinguishing "analytically derived" from "empirically grounded" at the rejection rationale level (parallel to how AS-9's "plausible-but-untested" qualifiers work) would make the evidentiary status visible without undermining the argument.

---

**P3-2. ADR-028 §Ensemble structure says the routing-planner ensemble's `topaz_skill` is `tool_use` — "the routing decision is structurally a tool-use task (choose which tool to invoke from a capability list)." This framing is reasonable but the Topaz 8-skill taxonomy classifies `tool_use` as the skill of selecting and invoking tools, not of choosing which tool from a list. The routing-planner produces a JSON output; it does not issue a tool call. The `summarization` skill (used for the synthesizer) would be equally or more defensible for a "produce JSON output from given context" role.**

- **Location:** ADR-028 §Decision §Ensemble structure: "`topaz_skill` is `tool_use`"
- **Observation:** The `tool_use` classification makes the tier-router treat the routing-planner as a tool-use ensemble for calibration and escalation purposes. If the routing-planner's failure mode looks more like a summarization failure (poor JSON schema conformance under long inputs) than a tool-selection failure, the tier-router's calibration criteria for `tool_use` may not be the ideal detection surface. This is a classification boundary question, not an error. But it is worth noting as a BUILD consideration — if the routing-planner's calibration Reflect verdicts accumulate under a `tool_use` taxonomy when the failure mode is schema non-conformance under complex inputs, a reconsideration at ARCHITECT may be warranted.
- **Recommendation:** Note in ADR-028 §Ensemble structure that the `tool_use` classification is a pragmatic choice and that if calibration evidence suggests a different Topaz skill better captures the planner's failure mode, the classification can be revised at ARCHITECT or BUILD without structural consequence.

---

**P3-3. ADR-029 §Decision §Strict-fidelity rule set names "Rule 6 candidate (Spike μ.1 surfaced)" as a candidate rule but §Decision's heading and the opening paragraph present it as decided: "Five rules + one candidate sixth." The hedging word "candidate" appears in the heading but the rule's description in the body reads as a specification ("Rule 6 codifies the pattern"). The §Rejected alternatives has an entry "Don't codify Rule 6" that confirms the ADR decided to codify it. The "candidate" language in the heading is a minor inconsistency.**

- **Location:** ADR-029 §Decision §Strict-fidelity rule set (opening: "Five rules + one candidate sixth"); §Rejected alternatives ("Don't codify Rule 6 — Rejected because Rule 6 codifies the pattern")
- **Observation:** If the ADR decided to codify Rule 6, the heading should read "Five rules + one codified candidate (Rule 6)" or simply "Six rules." The "candidate" qualification was correct before the ADR made the decision; it is a residual draft artifact now that the decision is made.
- **Recommendation:** Update the heading to reflect that Rule 6 is codified: "Five strict-fidelity rules + Rule 6 (framework-convention enumeration; codified from Spike μ.1)."

---

**P3-4. Vocabulary consistency across the 7 new ADRs: ADR-027 uses "framework-driven dispatch pipeline" and "plan → dispatch → synthesize"; ADR-028 and ADR-029 use "Plan-stage" and "Synthesize-stage" with capitalization. ADR-030 uses "request boundary" to describe where `tool_choice` interception occurs. ADR-031 uses "latency floor" without distinguishing between the pipeline's wall-clock latency and the client's timeout. These are minor and do not create logical inconsistencies, but a consistent vocabulary would support BUILD readers who need to cross-reference ADRs.**

- **Location:** Terminology across ADR-027 through ADR-032
- **Observation:** No new domain-model vocabulary terms are introduced by the 7 ADRs that are not already present in domain-model.md. The terms used are consistent with domain-model.md vocabulary. The minor inconsistencies noted above (capitalization of Plan/Synthesize stage names; "request boundary" vs. "request-parsing boundary") are editorial, not semantic.
- **Recommendation:** At ARCHITECT, a vocabulary-alignment pass over the 7 ADRs would normalize stage naming and technical terms for BUILD implementation. Not a blocker for gate acceptance.

---

**P3-5. ADR-021's fourth partial-update header ("Updated by ADR-027 on 2026-05-22") correctly records the actor shift for the routing decision on the chat-completions surface. The header's body text includes the statement: "The two supported dispatch shapes ADR-021 names (explicit ensemble naming; natural-language prompt) are now both routed through the routing-planner ensemble." This is accurate for the chat-completions surface but may be read as applying to all surfaces — including `llm-orc invoke`, where the actor has NOT shifted. The header's body text lacks a scope qualifier on this sentence.**

- **Location:** ADR-021 §Updated by ADR-027 header, second sentence of the header body: "The two supported dispatch shapes ADR-021 names...are now both routed through the routing-planner ensemble."
- **Observation:** The header's opening sentence correctly scopes the actor shift: "the actor producing the routing decision on the agentic-serving chat-completions surface shifts." But the second sentence drops the surface qualifier — a reader scanning quickly could interpret "both routed through the routing-planner" as applying universally. On `llm-orc invoke`, the orchestrator-LLM remains the routing decider and the two dispatch shapes work as ADR-021 originally specified.
- **Recommendation:** Add the surface qualifier to the second sentence: "The two supported dispatch shapes ADR-021 names (explicit ensemble naming; natural-language prompt) are now both routed through the routing-planner ensemble **on the chat-completions surface**."

---

## Section 2: Framing Audit

The framing audit examines what the ADR set's choices excluded from view. The source material (Essay-Outline 006; OQ #18/19/20 research; domain-model AS-9; product-discovery) is compared against the ADRs' framing decisions.

---

### Question 1: What alternative framings did the evidence support?

**Alternative framing 1: Cost-distribution accountability as a population-detection problem, not an architectural commitment.**

OQ #18 established that Population A clients locate cost-distribution inside the client; they don't expect the endpoint to exercise the distribution. The ADR set's framing (ADR-032) treats cost-distribution accountability as an architectural commitment the project-developer makes — and commits ADR-032's "strict-dispatch-when-capability-matched" mechanism to deliver it. An alternative framing would treat cost-distribution accountability as a *monitoring problem* rather than an architectural one: instead of committing the pipeline to dispatch on every capability-matched request as an architectural invariant, commit only to making the dispatch rate observable (which ADR-032 also does, via operator-observable degradation signaling), and leave the dispatch rate as a deployment tuning concern.

Under this framing, the sub-promise split ADR-032 makes would collapse the cost-distribution sub-promise into the monitoring layer, treating it as an operational concern rather than an architectural commitment. The evidence that would favor this framing: Population A's silence on cost-distribution accountability (OQ #18); the fact that "strict-dispatch-when-capability-matched" is already enforced by the routing-planner's behavior, not by ADR-032 committing a new mechanism.

What would a reader need to believe for this alternative to be right: that the project-developer's cost-distribution concern is better served by transparent reporting than by architectural enforcement; that the routing-planner's reliability (90% capability-match) is sufficient to deliver cost-distribution accountability without a separate architectural commitment; and that Population A's silence on this sub-promise means the sub-promise's enforcement is net zero value for the primary stakeholder.

**Alternative framing 2: ADR-027 as a reversible BUILD experiment, not a PRIMARY direction commitment.**

The evidence base for ADR-027 is solid but bounded: n=13 tests at qwen3:8b; 4 confabulation modes; a 20-prompt routing battery. The cycle commits to ADR-027 as the PRIMARY direction for every chat-completions request. An alternative framing would have shipped the pipeline as a BUILD experiment behind a feature flag (perhaps enabled by operator configuration or for a subset of request shapes) with explicit criteria for when the experiment would graduate to the PRIMARY path.

Under this framing, the PLAY phase would be the gate at which "PRIMARY direction" is confirmed rather than assumed; BUILD would ship the pipeline with rollback capability preserved. The evidence that would favor this framing: OQ #19's comparison shows both approaches are comparable in build cost; the scope-of-claim partition (settled/plausible-but-untested/open) in ADR-027 is substantial; shipping a behavioral change to every chat-completions request without a gradual rollout is a concentration-of-risk choice the ADR set acknowledges but does not mitigate with a feature-flag or gradual-rollout mechanism.

What would a reader need to believe: that the plausible-but-untested scope (generalization beyond qwen3:8b; production traffic diversity) is significant enough to warrant a cautious BUILD path; that the pipeline's failure mode (routing-planner failure affects every request) justifies a staged rollout.

**Alternative framing 3: The transparent-endpoint promise as a constraint on the framework's internals, not a promise to users.**

The ADR set frames AS-10 and the transparent-endpoint promise as commitments to Population A users — the endpoint behaves as they expect. An alternative framing would position the transparent-endpoint promise as a *self-imposed architectural constraint* — the framework agrees not to add out-of-band opt-in mechanisms to itself, because that would make the framework more complex and less portable. Under this framing, the promise is an engineering discipline commitment, not a user-facing contract; its violation would harm the framework's internal consistency rather than (directly) Population A's experience.

This distinction matters because the ADR set's framing implies Population A would detect violations of AS-10 — and OQ #18 finds they detect configuration dishonesty but not capability routing decisions. The alternative framing would make AS-10's enforcement about framework discipline rather than about Population A's experience.

---

### Question 2: What truths were available but not featured?

**Available but underrepresented: The over-dispatch failure mode.**

The source material (Spike ζ Finding ζ.3; Tranche 2 framing audit P3-F5) noted that the routing-planner can route a request to a capability ensemble when direct completion would have been faster, cheaper, and better. This over-dispatch failure mode is symmetric to the under-dispatch failure mode (which ADR-032 addresses extensively via strict-dispatch-when-capability-matched and degradation signaling). The ADR set does not discuss over-dispatch as a risk surface in any of the 7 new ADRs.

Where it appears in source material: Tranche 2 audit P3-F5; Spike ζ's ambiguous-match cases (2 of 20 prompts required defensible-judgment routing — some may have been over-dispatched to a capability ensemble when direct completion was superior).

Why it may have been excluded: the cycle's primary failure mode in RESEARCH/DISCOVER was under-dispatch (NL routing fraction ≈ 0), so the ADR set's framing focus on ensuring dispatch fires is appropriate given the cycle's asymmetric evidence base. Over-dispatch is acknowledged implicitly by the Tier-Router Audit's drift criteria but not named as a first-class risk.

Would its inclusion change the argument: Not structurally, but it would add a symmetry to ADR-032's degradation signaling specification — the `direct_completion_rate` metric catches under-dispatch; a `dispatch_rate_on_ambiguous_requests` metric or similar would catch over-dispatch. This is BUILD design work.

**Available but underrepresented: Spike ζ's testing scope limitations for production traffic diversity (OQ #25).**

ADR-028 names OQ #25 (routing-planner reliability under production traffic diversity) as an open question in the scope-of-claim partition but does not carry the scope limitation into ADR-028's Consequences §Negative with the same specificity that ADR-027's Consequences §Negative carries it. The routing-planner is the single point of failure for every chat-completions request; its 20-prompt battery represents a deliberate but narrow test set. The source material (OQ #19 build-complexity comparison §"What this comparison does not resolve") explicitly flags production traffic diversity as unresolved.

Why it may have been underrepresented: ADR-027 carries the scope-of-claim partition in detail (settled/plausible-but-untested/open); ADR-028 references ADR-027's partition rather than repeating it. The risk is real but is addressed at the ADR-027 level where it belongs architecturally.

**Available but underrepresented: The asymmetry between the routing-planner's failure surface and existing ensemble failure surfaces.**

ADR-028 and ADR-027 both note that the routing-planner is invoked on every chat-completions request, unlike capability ensembles which are invoked only when matched. The ADR set's Consequences sections acknowledge this ("single point of failure" language appears in ADR-027 and ADR-028). But the existing infrastructure (Calibration Gate, Tier-Escalation Router, Audit Dispatch) operates within dispatched ensembles — for the routing-planner specifically, a schema-non-conformance failure after Calibration Gate retries has no graceful degradation path to direct completion (unlike a capability ensemble failure, which can fall through to direct completion via the ADR-032 fallback shape). The ADR set does not specify a failure-mode recovery path for routing-planner infrastructure failure itself.

Where it appears in source material: ADR-027 §Consequences §Negative (routing-planner becomes a new single point of failure); ADR-028 §Consequences §Negative (planner failure is a chat-completions-surface failure). The source material names the risk but does not specify a recovery path.

---

### Question 3: What would change if the dominant framing were inverted?

The ADR set's dominant framing: **the orchestrator-LLM is the problem; structural bounding is the solution; the routing-planner + response-synthesizer are reliable because they are bounded.** This framing drives ADR-027's architecture, ADR-028 and ADR-029's design, and AS-9's codification.

Inverting the framing: **the task surfaces tested are narrow; the orchestrator-LLM's failures are task-surface-specific, not architectural; a more capable orchestrator-LLM would handle those surfaces reliably.**

Under this inverted framing:

- C1 (NL-to-ensemble routing fraction ≈ zero) would be attributed to orchestrator-LLM capability at cheap/mid tier under tool-rich suppression, not to the orchestrator-LLM architecture being structurally incapable. A frontier-tier orchestrator-LLM (Claude Opus 4.5, GPT-5) might route correctly under NL framing without suppression.
- AS-9 (structurally-bounded LLM roles produce reliable output) would be read as "cheap-tier LLM roles bounded to single decisions produce reliable output" — which is narrower than AS-9's current formulation.
- The build-complexity comparison (ADR-027: ~16 days) would favor the Tier 1 hybrid or the frontier-orchestrator approach if the assumption holds that frontier models handle routing more reliably than the routing-planner ensemble.

What becomes weaker under the inverted framing:
- The rejected alternative "frontier-tier orchestrator-LLM" (ADR-027) is rejecting something not well-tested on the routing-decision surface — Spike ζ tested qwen3:8b as the routing-planner; no spike tested a frontier-tier model as a routing-planner. The rejection is argued on cost grounds (frontier cost for every request) rather than on reliability grounds.
- The scope-of-claim partition (settled/plausible-but-untested/open) would need to acknowledge that the settled findings are all cheap-tier-specific.

What becomes stronger under the inverted framing:
- The cost argument for the inverted framing weakens it (using frontier-tier for every routing decision is expensive); ADR-027's rejected alternative correctly identifies cost as the distinguishing factor, not AS-9.
- The inverted framing would need to explain the PLAY note 22 confabulation at a paid model tier (MiniMax M2.5 is not cheap-tier; it confabulated). Spike ε ε.1's finding — that the same cheap-tier model produces faithful output when structurally bounded — directly rebuts the "cheap tier is the problem" reading.

What the ADR set would need to address: the possibility that the frontier-tier orchestrator provides a less costly path to reliable routing and synthesis than the routing-planner + synthesizer ensemble stack, particularly as frontier-tier model costs continue to fall. The ADR set would benefit from naming this as a downstream-cycle consideration rather than a closed question.

---

### Framing Issues

**F1 (P2): F2 carry-forward landing — ADR-030 correctly positions the hybrid as orthogonal, but the "orthogonal mechanism for `tool_choice`-aware Population A clients" framing in ADR-030 does not name the specific deployment scenario where operators have already built client-side `tool_choice` tooling. The framing is accurate but abstract.**

- **Location:** ADR-030 §Decision §Architectural commitment
- **Chosen framing:** `tool_choice` handling is "orthogonal to the ADR-027 framework-driven dispatch pipeline." This correctly lands the F2 recommendation from Tranche 2 (not "conditional alternative" but "orthogonal mechanism").
- **What the evidence supports:** The F2 recommendation said the hybrid should be positioned as "an orthogonal mechanism for a self-selected Population A sub-segment, not a 'conditional alternative' subordinate to ADR-027." ADR-030 does this.
- **What remains underweighted:** The specific scenario that makes the orthogonal mechanism practically relevant — operators who have already authored client-side skill frameworks that send `tool_choice` shapes — is not named. A reader of ADR-030 understands the mechanism is orthogonal but may not understand who would actually use it and why it matters to them now (before the follow-on cycle delivers it).
- **Alternative framing:** Name the concrete operator scenario: "Operators who have authored client-side skill frameworks that emit `tool_choice={"name":"<ensemble>"}` shapes (for example, Claude Code skill plugins that construct explicit ensemble names in their tool payloads) benefit from the deterministic honoring that disposition (i) delivers. For these operators, the orthogonal mechanism is a first-class capability need, not a conditional extension."

---

**F2 (P2): F3 carry-forward landing — ADR-031's graded framing (Tier A / Tier B / Tier C) correctly lands the F3 recommendation. The landing is solid. One underweighted aspect: the graded framing implies these tiers are stable, but OQ #20 explicitly notes that all four clients have shown timeout-related regressions in the last 12 months, and Cline's `requestTimeoutMs` knob behavior across providers is documented as unreliable. The tier classification may need more frequent revision than ADR-031 suggests.**

- **Location:** ADR-031 §Consequences §Neutral: "The Population A coverage tiers may evolve. Cline issue #4308 may be resolved upstream; Cursor's base-URL override may extend to agentic paths; new Population A clients may emerge. ADR-031's tier classification is a Cycle 7-empirical reading."
- **Observation:** The Neutral section does acknowledge tier evolution. The issue is that the framing positions tier changes as upstream client changes (Cline resolves #4308; Cursor extends its override), not as a cycle-by-cycle responsibility of the framework's deployment documentation. The OQ #20 source material notes "all four have shown timeout-related regressions in the last 12 months" — timeout behavior is an actively shifting target.
- **Alternative framing:** Name the tier classification as requiring active maintenance — "the tier classification should be revisited at each cycle's PLAY phase as client behavior evolves; this is not a static reference document." This is a framing choice about documentation maintenance burden, not a logical error in the ADR.

---

**F3 (P3): The ADR set consistently frames AS-9's empirical basis as "n=13 tests across 4 confabulation modes." The Tranche 2 framing audit (P3-F3) noted that rounding drift (Mode 1 + Mode 2, Spike ε' Finding ε'.2) is a distinct failure-mode class, not a confabulation mode. ADR-029 correctly separates Rule 4 (rounding drift) from Rules 1-3 (confabulation/fabrication). But ADR-027's §Scope-of-claim partition and ADR-028's §Consequences still use the "4 confabulation modes" shorthand without noting the separate rounding-drift risk surface. BUILD readers working from ADR-027 and ADR-028 may underestimate the rounding-drift risk if they don't also read ADR-029.**

- **Location:** ADR-027 §Decision §Scope-of-claim partition §Settled; ADR-028 (no explicit mention of rounding drift in §Consequences)
- **Observation:** The rounding-drift finding is well-specified in ADR-029 (Rule 4 + mitigation playbook). The ADR-027 and ADR-028 references to "4 confabulation modes" don't suppress the finding — they just don't feature it. A reader who reads ADR-027 and ADR-028 in isolation may not know rounding drift is a named risk surface requiring its own mitigation playbook.
- **Note for practitioner:** This is a documentation completeness observation, not a logical gap. The risk is adequately addressed in ADR-029; cross-referencing or brief mention in ADR-027's scope-of-claim partition would make the complete risk surface visible from the headline ADR.

---

### Specific Audit Focus Findings

The dispatch brief named 10 specific audit focuses. Findings per focus:

**Focus 1: AS-9 propagation correctness.**

The partial-update headers on ADR-021 and ADR-022 correctly reflect AS-9's scope. ADR-021's header records the actor shift (orchestrator-LLM → routing-planner ensemble) and correctly scopes it to the chat-completions surface. ADR-022's header records that the system-prompt amendment is structurally moot for chat-completions under ADR-027 and remains operative for `OrchestratorRuntime` surfaces. The headers preserve the body text of both ADRs without modification.

ADR-026 (AS-10) does not contradict AS-9 — the two are explicitly distinguished as role-shape vs. request-shape. ADR-027, ADR-028, and ADR-029 build correctly on AS-9 without overreach. No propagation errors found.

Minor: ADR-021's header body sentence drops the surface qualifier on the two dispatch shapes (see P3-5 above).

**Focus 2: AS-10 codification soundness.**

The AS-9 vs. AS-10 distinction is internally consistent across ADR-026/027/028/029/032 in substance. ADR-026 defines the distinction precisely; ADR-027 and ADR-028 apply it correctly. AS-10's scope (chat-completions surface; not `llm-orc invoke` or direct ensemble HTTP API) is consistently applied across all 7 ADRs.

ADR-029 does not name AS-10 (see P2-4 above) — minor gap, not a contradiction.

The rejected alternatives in ADR-026 (Population-A-only narrowing; project-level scope; defer further) all hold up against scrutiny. The Population-A-only narrowing rejection is the strongest drafting-time analytical argument (see P3-1 above) and is correct — the detection problem re-introduces the constitutional question. No issue with the scope-of-claim partition.

**Focus 3: Cost-equivalence claim in ADR-027.**

The ~14 vs. ~16 person-days cost-equivalence claim is fairly characterized and consistent with OQ #19. The P2-2 note in the Tranche 2 audit (GT-2(a) scope extension) is the active issue (see P2-1 above). The structural-coverage differential (universal AS-9 satisfaction vs. partial) is fairly characterized — the three structural factors named (AS-9 satisfaction surface; NL-routing-fraction reduction; confabulation-mode mitigation) accurately reflect OQ #19's analysis.

The Essay-Outline §C7 W7.1 E7.1.2 P2-1 scope-extension note was applied as a correction at Tranche 2 close, but the correction was applied to the Essay-Outline, not to ADR-027. ADR-027 drafts after the corrected Essay-Outline and should carry the qualified citation — it doesn't (P2-1 above).

**Focus 4: Scope-of-claim partition in ADR-027.**

The settled/plausible-but-untested/open partition in ADR-027 §Decision §Scope-of-claim partition matches Essay-Outline 006 Amendment A3 tightened by Spike ε' + Spike μ. The partition is reproduced accurately and without overreach. The seven "settled" items, four "plausible-but-untested" items, and six "open" items all correctly reflect the source material's characterization.

No overreach found. The framing note in P3-3 (Tranche 2 audit) about over-dispatch is not present in ADR-027's settled partition but it was not part of the settled findings — the settled partition is accurate for what it claims.

**Focus 5: Sub-promise split in ADR-032.**

The configuration-honesty vs. cost-distribution-accountability split is well-executed. The two sub-promises have distinct mechanisms (honest response labeling; strict-dispatch-when-capability-matched), distinct evidence bases (Population A corroboration; project-developer-lens with honest residual uncertainty), and are named throughout ADR-032 without conflation.

One issue (P2-5 above): the 90% capability-match empirical floor needs clarification on whether Spike ζ's 10% non-strict-match cases include any `action: "direct"` outcomes that would constitute direct sub-promise (2) failures.

The rejected alternative "single transparent-endpoint promise without sub-promise split" is correctly rejected on evidentiary-clarity grounds. The rejection rationale cites OQ #18's original "will fail the susceptibility audit Population A would apply" language — the Tranche 2 audit's P2-3 softened this in the Essay-Outline, but ADR-032's Provenance check footnote correctly notes the softening: "the recommendation to keep ADRs separate is unchanged; basis is now evidentiary-confidence differential rather than predictive failure assertion." The ADR's rejected-alternative text uses the stronger "will fail" language, which is now weaker than the corrected Essay-Outline. Minor inconsistency; the rejection rationale is still sound but the "will fail" framing predates the Tranche 2 correction.

**Focus 6: Framing-audit F1-F3 carry-forward landings.**

- **F1 (A2.1 framing softening):** Applied at Tranche 2 close to the Essay-Outline. ADR-032's Provenance check references the softened framing (citing the argument-audit's P2-3 softening). The "will fail" language in ADR-032's rejected alternative body text predates the softening (see Focus 5 above). Minor.
- **F2 (hybrid as orthogonal mechanism):** Landing in ADR-030 is correct — "orthogonal mechanism for `tool_choice`-aware Population A clients" rather than "conditional alternative subordinate to ADR-027." See F1 framing finding above for the one underweighted aspect.
- **F3 (graded framing for OQ #20):** Landing in ADR-031 is correct — three-tier framing (Tier A / Tier B / Tier C) rather than binary accommodate/breach. The graded framing names Cline as Tier B with documented-tuning caveat and Cursor as Tier C structurally outside scope. The F3 intent is fully honored.

**Focus 7: Rule 5 framing scope (OQ #23) two-state design in ADR-029.**

The load-bearing-default-with-falsification-trigger design is internally consistent. The falsification trigger (production evidence that Rule 5 framing degrades user experience → migrate to headers/metadata per ADR-032) correctly names ADR-032's headers mechanism as the escape hatch.

The ADR-022 precedent citation issue (P2-3 above) is the one concern — the citation borrows credibility from an approach the cycle rejected. The structural difference (synthesizer vs. orchestrator-LLM context) is real and makes the citation defensible; naming it explicitly would remove the ambiguity.

The two-state design is consistent with ADR-022's "amendment + cross-profile-deferred-to-BUILD/PLAY" precedent in structure (commit to a default; named falsification trigger; production evidence required to trigger). The content of the trigger (user experience vs. routing effectiveness) differs appropriately from ADR-022's trigger (cross-profile routing effectiveness).

**Focus 8: ADR-030 disposition (i) with deferred implementation.**

The deferral rationale (Cycle 7 BUILD scope; ~30% scope growth avoided) is sound and correctly grounded in OQ #19's work-item breakdown. The bridge mechanism preserves AS-10 and Population A configuration-honesty commitments (the silent-strip footgun is addressed). The model-layer non-honoring out-of-scope clause is appropriately bounded (Spike λ-paid establishes the factual basis; the clause names the structural cost of the framework-layer / model-layer decoupling).

The structural enforcement gap (P2-2 above) is the main concern — the commitment exists without a mechanism to enforce the follow-on cycle's delivery.

**Focus 9: Cross-document consistency.**

No contradictions found between the 7 new ADRs and each other or with:
- ADR-001 (ReAct loop preservation): ADR-027 explicitly preserves the ReAct loop for non-chat-completions surfaces.
- ADR-019 (skill-framework-agnostic): The routing-planner and response-synthesizer are system ensembles under the `agentic-` prefix convention, consistent with ADR-019. The framework remains skill-framework-agnostic.
- ADR-021 (post-partial-update): The actor shift is correctly scoped; the dispatch contract's structural commitments are unchanged.
- ADR-023 (observability event routing): ADR-032 names new event types (`direct_completion_fallback`, `direct_completion_rate`) and correctly states they are delivered by ADR-023's routing mechanism. No contradiction.
- ADR-024 (envelope): ADR-029's input contract reads envelope content (`primary` + `artifacts[0]` summary fields) consistent with ADR-024's envelope contract.
- ADR-025 (substrate routing): ADR-029's input contract note — "the synthesizer does not have access to the dispatched ensemble's substrate paths directly — substrate routing per ADR-025 produces summary-shaped content in the envelope" — is accurate and consistent with ADR-025.

The one cross-document tension is ADR-030's bridge advisory specification referencing ADR-032 (P1-1 above) — this is the only structural gap found in cross-document consistency.

**Focus 10: Domain vocabulary usage.**

All 7 ADRs use domain-model.md vocabulary consistently. The key terms — Population A, Population B, routing-planner ensemble, response-synthesizer ensemble, transparent OpenAI-compatible endpoint, AS-9, AS-10, structurally-bounded role, framework-driven dispatch pipeline — are used consistently with their domain-model definitions. No novel terms introduced by the ADRs that should be added to domain-model.md. No vocabulary inconsistencies found.

One observation: ADR-030 introduces "bridge mechanism" as a term for the Cycle 7 intermediate `tool_choice` handling. This term does not appear in domain-model.md. It is a BUILD-phase-specific term for a provisional mechanism; adding it to domain-model.md with a "(provisional; pending disposition (i) implementation)" qualifier would support the backward propagation sweep (Step 3.7).
