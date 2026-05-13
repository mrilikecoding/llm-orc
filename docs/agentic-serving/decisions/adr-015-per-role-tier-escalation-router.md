# ADR-015: Per-Role Tier-Escalation Router via OI-MAS

**Status:** Updated by ADR-018; Updated by ADR-019

**Date:** 2026-05-05

> Updated by ADR-018 on 2026-05-11.
> ADR-018 amends this ADR by adding a periodic out-of-band audit dispatch responsibility to the Tier-Escalation Router module (analog of ADR-016 mechanism (d), operating on the L1→L2 verdict→router edge). The amendment does not change the primary-skill framing, the verdict-to-tier mapping, or the rejected alternatives — those remain unchanged. ADR-018 is the empirical-anchor record for the (d)-analog audit dispatch and the Sub-Q6 coupling noted in §Consequences §Neutral below. The amendment is empirically anchored by Spike β (research log `005h-spike-bounding-mechanism-transfer-l1-l2.md`, 2026-05-11). Spike α (research log `005g-spike-topaz-skill-classification.md`, 2026-05-11) confirmed that the primary-skill framing in this ADR stands — 21 of 21 classified library ensembles satisfy the clean-primary criterion; rejected alternative §(b) (per-ensemble overrides) remains unwarranted.

> Updated by ADR-019 on 2026-05-12.
> The reframing this header records applies to ADR-015 §Negative *as characterized in the proposal `proposals/agentic-serving-library-structure.md` and the Cycle 4 / Cycle 5 product-discovery rendering of ADR-015 §Negative* — specifically the "operator-driven library migration" reading. ADR-015's body §Negative addresses one concrete instance of operator-driven migration (Topaz-metadata addition on existing ensembles); ADR-019 leaves that body bullet undisturbed (existing ensembles still need one-time metadata migration to participate in tier-router dispatch). What ADR-019 *does* amend is the *broader* operator-driven-migration framing that the proposal and product-discovery extended from ADR-015's §Negative: the assumption that working defaults for the agentic-serving deployment (a profile file, an `agentic-serving/` subdirectory, a minimum-viable capability ensemble set, a rewritten `agentic_serving:` config section) live outside the cycle's BUILD scope. ADR-019 moves those into Cycle 5 BUILD scope. The verdict-to-tier mapping, primary-skill framing, per-skill tier defaults configuration shape, and rejected alternatives in ADR-015's body all remain unchanged. ADR-019 is the architectural-commitment record for the skill-framework-agnostic orchestrator + operation-named capability ensemble library framings the reframing rests on. The amendment is grounded in Cycle 4 PLAY note 1 (practitioner verbatim: *"the agentic-serving config is to me part of the build"*) and the Cycle 5 DISCOVER gate's settled commitment.

---

## Context

ADR-011 establishes that the orchestrator agent's LLM is configured via a standard Model Profile, with no hard-coded tiered fallback in the orchestrator. ADR-011's Decision text reads: *"If tiered behavior is desired (local triage with cloud escalation), it is expressed as a composed ensemble invokable by the orchestrator — not as a mechanism special to the orchestrator."* Essay 003 produced the "default-not-ceiling" reading of ADR-011: ADR-011 is defensible as a default-orchestrator decision but not as a ceiling on tiered behavior elsewhere in the system. The cycle's task is to operationalize the default-not-ceiling reading without violating ADR-011's no-special-case-in-orchestrator principle.

Cycle 4's Wave 2 literature review found that heterogeneous role-staffing beats homogeneous-staffing repeatedly: SC-MAS achieves +3.35 percent accuracy and −15.38 percent cost on MMLU; MasRouter delivers +1.8 to +8.2 percent accuracy and −52 percent cost on MBPP/HumanEval; OI-MAS (arXiv:2601.04861) provides +12.88 percent accuracy at −17 to −78 percent cost via confidence-gated tier escalation. The mechanism is named explicitly: capability saturation. Homogeneous systems over-allocate expensive capability to tasks that don't require it. The Topaz eight-skill taxonomy (code generation, tool use, mathematical reasoning, logical reasoning, factual knowledge, writing quality, instruction following, summarization) is directly adoptable as a role-profiling vocabulary. From Topaz's authors: *"Efficiency gains stemmed from capability saturation rather than hidden quality loss."*

Essay 005 §"ADR candidate #4" frames the elaboration: the Tool Dispatch (L2) interposition logic adds a confidence-gated tier-escalation pattern via OI-MAS — cheap model by default, escalate to a more capable tier when confidence is below threshold. The pattern is **consistent with ADR-011's intent** (the orchestrator's own LLM remains session-boundary-event scoped; tiered behavior is not a special case in the orchestrator's reasoning logic) **and extends the mechanism class** from "explicitly-invoked composed ensemble that implements tier decisions internally" (ADR-011's contemplated mechanism) to "dispatch-layer interposition that selects tier per dispatch based on calibration verdict" (ADR-015's mechanism). The two mechanisms produce the same architectural property — orchestrator-LLM-unchanged + tier-of-dispatched-work-variable — through different placements. Why L2-interposition is preferable to composed-ensemble placement for the cycle's deployment shape: the calibration verdict is per-dispatch and changes within a session; placing the tier decision inside an ensemble would either duplicate calibration logic across every tier-aware ensemble in the library (configuration burden) or require operator-authored ensembles to consume calibration verdicts directly (coupling that crosses the L0-L1 boundary the layering rule constrains). L2 interposition centralizes the tier decision once at the dispatch boundary where the calibration verdict is naturally available.

ADR-014 produces a calibration verdict (Proceed / Reflect / Abstain) for each dispatch decision. The verdict is the input that drives ADR-015's tier-escalation router. The two ADRs compose to form the cycle's in-process calibration-and-escalation system: ADR-014 specifies *whether to proceed and at what confidence*; ADR-015 specifies *which tier to dispatch to* given the verdict.

Pre-specifiable routing exceeds LLM-judgment routing across CAAF, LLMCompiler, MetaAgent, and Gu (essay 005's Wave 1 lit-review). The router's mechanism is structurally pre-specified (skill profile + calibration verdict → tier), not LLM-decided. This is a class (c) decomposition pattern in the intervention-class taxonomy.

The framing commitment from research-gate Grounding Action 2 (recorded 2026-05-05, *elaboration-by-evidence*) holds: tier-escalation responsibilities concentrate within Tool Dispatch (L2 interposition) rather than warranting a dedicated routing module orthogonal to L2.

The capability-floor open question (domain-model.md OQ #9) interacts with ADR-015: the cheap-tier defaults must meet the capability floor for the skills they cover. Early-deployment evidence may surface mismatches (the cheap-tier doesn't meet the floor for one or more skills); the router's tier-defaults configuration is the operationally-tunable surface where mismatches are corrected.

---

## Decision

The Tool Dispatch (L2) interposition logic adds a **per-role tier-escalation router** between the orchestrator's tool calls and the dispatched ensembles.

### Per-skill role profiling using Topaz's eight-skill taxonomy

Each ensemble in the library declares its **primary Topaz skill** in its YAML configuration as a metadata field:

- `code_generation`
- `tool_use`
- `mathematical_reasoning`
- `logical_reasoning`
- `factual_knowledge`
- `writing_quality`
- `instruction_following`
- `summarization`

Multiple ensembles may claim the same skill; ensembles representing distinct skills are catalogued separately. The skill metadata is operator-authored at ensemble creation; existing ensembles in the library require a one-time migration to add the skill metadata field.

### Per-skill tier defaults

For each of the eight Topaz skills, the operator configures a **cheap-tier Model Profile** and an **escalated-tier Model Profile**. The configuration surface is per-skill, not per-ensemble — ensembles claiming the same skill share the tier defaults. The configuration thus has 8 skills × 2 tiers = 16 Model Profile slots, with deployment-time defaults that may share Model Profiles across skills (e.g., a single local-7B profile may serve as cheap-tier for code_generation and tool_use).

Tier-default Model Profiles can be:
- **Cheap-tier:** operator-local (when local capability meets the floor for the skill) or cheap-cloud (when local falls below the floor)
- **Escalated-tier:** cheap-cloud, or operator-cloud (frontier)

### Router logic

On each dispatch via `invoke_ensemble`:

1. **Skill lookup.** The router reads the dispatched ensemble's Topaz skill metadata.
2. **Calibration verdict consumption.** The router reads the calibration verdict from ADR-014 for the dispatch context.
3. **Tier selection per OI-MAS pattern:**
   - **Proceed** → route to the cheap-tier Model Profile configured for the ensemble's skill
   - **Reflect** → route to the escalated-tier Model Profile configured for the ensemble's skill
   - **Abstain** → bypass routing entirely; produce a typed `escalation_bypass` error to the orchestrator. The orchestrator must take a different action (reformulate, dispatch to a different ensemble, or abandon the task) — escalation does not happen on Abstain.

### ADR-011 compatibility — explicit verification

The orchestrator's own LLM remains scoped at the session-boundary event per ADR-011 unchanged. The router escalates *the dispatched task's* Model Profile, not the orchestrator's. The orchestrator's tool-call surface (`invoke_ensemble`, per ADR-003's closed five-tool surface) is unchanged; the router operates inside Tool Dispatch (L2 interposition), invisible to the orchestrator's reasoning surface. The pattern is consistent with ADR-011's framing: tiered behavior is implemented in the dispatch path, not as a special case in the orchestrator.

### Capability-floor interaction

The cheap-tier defaults must meet the capability floor (domain-model.md OQ #9) for the skills they cover. The router does not enforce the floor — that is the configuration responsibility of the operator and the territory of the capability-floor scenario candidates DECIDE will deliberate (static specification + runtime probe per discover-gate carry-forward #7). When a cheap-tier default does not meet the floor, the router's tier-escalation behavior degrades: Reflect verdicts will frequently arrive on dispatches the cheap-tier cannot handle; the operator's calibration evidence will surface the mismatch; tier-defaults reconfiguration is the corrective action.

---

## Rejected alternatives

**(a) Single-tier-only (no escalation pattern).** Rejected: the heterogeneous role-staffing literature evidence base is overwhelming (OI-MAS +12.88% / −17-78%; SC-MAS +3.35% / −15.38%; MasRouter +1.8-8.2% / −52%). Single-tier-only either saturates capability (escalated-tier-everywhere) or wastes capability (cheap-tier-everywhere); the OI-MAS pattern's value is the calibration-driven discrimination between cases.

**(b) Per-ensemble tier alternatives (no per-skill profiling).** Rejected per practitioner friction-trades-for-discovery guidance and per the literature evidence base. Per-ensemble tier specification multiplies configuration burden across the library; per-skill profiling enables operator-managed tier defaults at the skill level. The full Topaz 8-skill taxonomy is the discovery-friendly choice — subsetting the taxonomy before deployment evidence indicates which skills matter is premature optimization. The discovery question is which capability dimensions actually drive value for the cycle's task class; that answer requires all 8 first-class.

**(c) Orchestrator-side tier decision (router in L2 Orchestrator Runtime, not Tool Dispatch).** Rejected: this would be a special case in the orchestrator, contradicting ADR-011's no-special-case principle. The router belongs in Tool Dispatch (L2 interposition) where it can intercept dispatches without altering the orchestrator's reasoning surface. ADR-011's principle was that tiered behavior is implemented in the dispatch path, not in the orchestrator; ADR-015 honors that placement.

**(d) Subset of Topaz's eight skills (e.g., only `code_generation` + `tool_use`).** Rejected per practitioner friction-trades-for-discovery guidance. The full 8-skill taxonomy enables discovery of which skill dimensions actually matter for the cycle's task class and deployment shape. Subsetting before deployment evidence is premature optimization. The friction cost (16 Model Profile slots to configure rather than 4) is the trade for discovery; the practitioner has explicitly accepted that trade.

**(e) LLM-judgment tier selection (orchestrator chooses tier per dispatch via prompt or reasoning).** Rejected: pre-specifiable routing exceeds LLM-judgment routing across the literature reviewed. The router's mechanism is structurally pre-specified (skill profile + calibration verdict → tier), which is class (c) decomposition. LLM-judgment tier selection at every dispatch is class (b) prompt-suggestion, which has documented reliability problems (CAAF arXiv:2604.17025: "apparent LLM reliability in safety-critical domains is often a prompt engineering artifact").

**(f) Skill metadata as runtime classification (router infers skill from task content) rather than ensemble-declared metadata.** Rejected: runtime classification reintroduces LLM-judgment into the routing path. The orchestrator selecting an ensemble *means* the orchestrator believes the ensemble matches the task; the ensemble's skill metadata is what makes the orchestrator's selection meaningful at the router level. Runtime inference would be redundant LLM work on data the ensemble selection already encodes.

**(g) Reflect verdict triggers retry-with-cheap-tier rather than escalation.** Rejected: AUQ's empirical pattern (which ADR-014's Reflect verdict implements) is that low-confidence triggers a reformulation step, and the escalation is the reformulation's higher-capability instantiation. Retry-with-cheap-tier on Reflect produces the same low-confidence verdict on the second attempt with the same tier — no information has been added. Escalation is the action that changes the calibration trajectory.

---

## Consequences

**Positive:**
- Operationalizes ADR-011's "default-not-ceiling" reading at the Tool Dispatch level without violating ADR-011's no-special-case-in-orchestrator principle
- Per-skill tier defaults enable operator-managed economy without per-ensemble configuration burden
- ADR-014's calibration verdict drives escalation; the two ADRs compose to form the in-process calibration-and-escalation system
- Heterogeneous role-staffing's empirical advantages apply to llm-orc deployments (literature evidence base spans SC-MAS, MasRouter, OI-MAS, Topaz)
- Discovery-friendly: full 8-skill taxonomy lets operator deployment evidence surface which skills benefit most from escalation, which skill cheap-tiers fall below the capability floor, and which skill Model Profile defaults need adjustment
- The Abstain → reformulate path (rather than escalate) keeps the orchestrator engaged in cases where escalation would not help, avoiding the failure mode where every Abstain produces an escalated dispatch that also fails

**Negative:**
- Operator must configure 8 skills × 2 tiers = 16 Model Profile slots at deployment; defaults can be shared across skills (collapsing some slots), but the configuration surface itself is 16
- Topaz skill metadata required on every ensemble in the library; existing ensembles need a one-time migration to add the skill metadata field (BUILD-time work)
- Cheap-tier and escalated-tier defaults interact with the capability-floor open question (OQ #9); early operators may produce calibration evidence that cheap-tier defaults fall below the floor for one or more skills, forcing reconfiguration
- Router adds latency overhead per dispatch (skill-metadata lookup + verdict consumption + Model Profile selection); the overhead is bounded but non-zero
- The 8-skill taxonomy may not partition cleanly for the cycle's task class — some dispatches may legitimately span two skills (e.g., code_generation + tool_use), and the primary-skill declaration forces a choice; the alternative (multi-skill metadata) was not adopted to keep the router's logic pre-specifiable
- **Discovery value is proportional to deployment coverage** (per argument-audit P3.3 finding 2026-05-06). The cycle's primary task class may exercise only 4–5 of the 8 Topaz skills routinely; the remaining slots produce no calibration evidence in deployment. The friction-trades-for-discovery argument holds for the *exercised* skills; for *unexercised* skills, the configuration burden is friction without proportionate discovery. Operators whose deployment is concentrated in a sub-taxonomy (e.g., predominantly code_generation + tool_use + summarization) may legitimately collapse unused skills to shared Model Profiles to reduce configuration surface; the structural taxonomy remains for the discovery surface to expand into

**Neutral:**
- The router operates in Tool Dispatch (L2 interposition), invisible to the orchestrator's reasoning surface — the orchestrator's tool-call API (`invoke_ensemble`) is unchanged
- Composition with ADR-014 is strict data-flow: calibration verdict input → tier selection output
- Composition with ADR-011 is strict compatibility: orchestrator's session-boundary-event scope preserved
- The router does not consume the cross-layer signal channel from ADR-016 directly; it consumes the calibration verdict ADR-014 produces, which (under ADR-016 acceptance) may itself incorporate cross-layer signals upstream
- Topaz skill metadata is a deployment-portable schema that other LLM orchestration systems can adopt; the metadata field's existence in llm-orc's ensemble YAML is an interoperability surface
- **Autonomous-routing evidence gap (Grounding Reframe action 2026-05-06, per round-1 framing audit P1 + decide-phase susceptibility snapshot).** The cycle's research (essay 005 §"Open Questions and Scope-of-Claim", Sub-Q6) explicitly documents that *multi-iteration routing reliability at the North-Star benchmark's session length is empirically unvalidated*. ADR-015's value proposition depends on the orchestrator routing correctly to the cheap-tier — which is precisely the unvalidated surface. Operators interpreting the router's escalation-rate calibration evidence should be aware that the rate may reflect *routing reliability noise* rather than tier-configuration mismatches. Calibration-driven tier-default tuning (per the operator action surfaces above) is interpretable only once multi-iteration routing reliability is established at the deployment scale the ADR targets; until then, the router's tier-selection behavior reflects a composition of routing accuracy AND tier capability, and isolating the two requires either explicit routing-quality measurement or first-deployment evidence on the cycle's North-Star benchmark. **(Sub-Q6 coupling per ADR-018, 2026-05-11.)** Spike β surfaced that ADR-018's (d)-analog audit dispatch — specifically its *escalation-vs-outcome correlation* drift criterion — is the operational measurement that distinguishes routing-noise from tier-configuration signal. The audit's first-deployment evidence on this drift criterion is what Sub-Q6 needs; OQ #14 partial closure for the L1→L2 stage and Sub-Q6 structural closure are addressed by the same mechanism.
- **Attention-MoA orchestrator-as-aggregator dependency (advisory, per round-1 framing audit P2 + decide-phase snapshot).** Essay 005 documents that on instruction-following tasks, orchestrator quality at the aggregation moment drives a 12.82-percentage-point gap in ensembles-of-ensembles outcomes — the orchestrator can be the bottleneck rather than the member models. ADR-015's tier escalation acts on the dispatched member model's tier; the orchestrator's aggregation role after receiving the escalated result is unaddressed. Deployment evidence should track whether escalation gains are concentrated where member-model quality is the bottleneck (where ADR-015 helps) versus where orchestrator-aggregation is the binding constraint (where ADR-015 may not help; the orchestrator's own Model Profile becomes the design surface, which is ADR-011 territory)

---

## Provenance check

- **Driver-derived content (mechanism specifications).** The OI-MAS confidence-gated tier-escalation pattern is direct adoption from arXiv:2601.04861. The Topaz eight-skill taxonomy is direct adoption from the Topaz paper, surfaced via essay 005 §"Long-Horizon Reliability Infrastructure" and §"ADR candidate #4." The capability-saturation mechanism explanation is from Topaz's authors. Heterogeneous role-staffing's empirical advantage (vs. homogeneous) is established by SC-MAS, MasRouter, OI-MAS evidence base.

- **Driver-derived content (ADR-011 compatibility argument).** Essay 005 §"ADR candidate #4" frames the tier-escalation router as compatible with ADR-011. **Argument-audit refinement (2026-05-06):** essay 005's framing — "OI-MAS at Tool Dispatch is an implementation of ADR-011, not an amendment" — overstates identity with ADR-011's specified mechanism. ADR-011 contemplated tier decisions inside explicitly-invoked composed ensembles, not at the dispatch-layer interposition. The Context section's reframing — "consistent with ADR-011's intent and extending the mechanism class" — is drafting-time precision applied to essay 005's framing. The ADR-011-compatible property (orchestrator's own LLM session-boundary-event scoped) is preserved by both mechanism placements; ADR-015 chose L2 interposition over composed-ensemble placement for the configuration-burden and coupling-boundary reasons documented.

- **Drafting-time synthesis (Reflect→escalate, Abstain→reformulate-bypass mapping).** Essay 005 specifies the OI-MAS confidence-gated tier-escalation pattern but does not specify the mapping from ADR-014's verdict trichotomy (Proceed / Reflect / Abstain) to router actions. The mapping is drafting-time synthesis composing ADR-014 and ADR-015. Specifically, the choice that Abstain produces an `escalation_bypass` typed error rather than an escalation is drafting-time judgment — alternatives (Abstain triggers escalation; Abstain produces silent fallback) are documented in the rejected alternatives section. The chosen mapping preserves the calibration mechanism's purpose (severe low-confidence signals require *different action*, not *more capability*).

- **Drafting-time synthesis (per-skill tier defaults configuration model).** Essay 005 specifies the OI-MAS pattern at Tool Dispatch but does not specify the per-skill tier-defaults configuration model (16 Model Profile slots per deployment). The alternative (per-ensemble tier alternatives) was rejected for friction-trades-for-discovery reasons. The 16-slot configuration is drafting-time synthesis structuring the OI-MAS pattern's deployment surface.

- **Drafting-time synthesis (ensemble YAML skill-metadata field).** The requirement that ensembles declare their primary Topaz skill in YAML metadata is drafting-time synthesis. Essay 005 names the Topaz taxonomy as the role-profiling vocabulary but does not specify how skill metadata is encoded in the ensemble library. The YAML-metadata-field choice is drafting-time judgment about the interoperability surface; alternatives (separate skill registry, runtime classification) were rejected for the reasons documented.

- **Capability-floor interaction is essay-derived flag, not drafting-time synthesis.** Essay 005 and discover-gate carry-forward #7 both flag the capability-floor question. ADR-015 documents the interaction (router degrades when cheap-tier defaults fall below floor) but does not resolve the floor specification — that is DECIDE-phase scenario candidates territory.

- **Practitioner friction-trades-for-discovery guidance applied.** The full 8-skill taxonomy (rather than a subset) and per-skill tier defaults (rather than per-ensemble) are drafting-time choices applying the practitioner's gate-conversation guidance. Both choices increase deployment-configuration friction; both choices are friction-trades-for-discovery applications. The discovery question (which skills/dimensions matter most for the cycle's task class) requires the full taxonomy as a starting point.

- **Vocabulary impact.** ADR-015 introduces three terms candidate for domain-model addition at Tranche-C close:
  - **Tier-escalation router** — proposed new term in §Concepts (operator voice; the router's name)
  - **Topaz skill profile** — proposed new term in §Methodology Vocabulary (research voice; the role-profiling vocabulary's source attribution)
  - **Per-skill tier default** — proposed new term in §Concepts (operator voice; the configuration unit operators work with)

  The existing product-vocabulary entry "tier escalation" (operator voice, MODEL phase Amendment Log entry #4) is the parent term; ADR-015's terms are sub-specifications.

- **Asymmetric DECIDE budget per research-gate carry-forward #4.** ADR-015 is the bridge case between adoption-decision discipline (#1, #2) and novel-architectural-territory pressure-testing (#3, #5, #6, #7). Argument-audit on this ADR should concentrate on (i) the ADR-011 compatibility argument (does the Tool Dispatch placement actually preserve the no-special-case-in-orchestrator principle, or is it special-casing-by-another-name?), (ii) the verdict-to-action mapping (whether Abstain→reformulate is correctly designed for the cycle's failure modes), and (iii) the friction trade (whether the 16-slot configuration burden is justified by the discovery value). The OI-MAS / Topaz adoption is adoption-decision discipline; the synthesis choices flagged above are bridge-case pressure-test territory.
