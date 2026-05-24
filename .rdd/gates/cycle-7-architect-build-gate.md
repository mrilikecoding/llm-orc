# Gate Reflection: Cycle 7 — Framework-driven orchestration: routing as code — architect → build

**Date:** 2026-05-23
**Phase boundary:** architect → build
**Cycle:** Cycle 7 — Framework-driven orchestration: routing as code

## Belief-mapping question composed for this gate

"What would you need to believe for the alternative — Tier 1 hybrid as the chat-completions mechanism, with `tool_choice` interception as the primary routing path and `OrchestratorRuntime` continuing to handle non-`tool_choice` NL routing — to be the better Cycle 7 architecture?

The DECIDE rationale rejected Tier 1 hybrid as primary on three structural grounds (AS-9 satisfaction differential, NL-routing-fraction reduction differential, confabulation-mode mitigation differential — all universal under ADR-027, partial under Tier 1). Cost-equivalence was established at OQ #19 (~14 vs. ~16 person-days median; same order of magnitude). What is the belief-shape under which a reasonable architect, reviewing this corpus now, would conclude the hybrid was the right Cycle 7 primary?"

## User's response

"If the ensembles are simply not capable enough to handle generalized long-horizon tasks then perhaps hybrid would be preferred. But we haven't invested enough to answer that question."

## Pedagogical move selected

Challenge (belief-mapping). The practitioner had engaged deeply across the gate (Track C triage with three substantive selections; module decomposition acceptance with reasoning-engagement on the boundary choices). The belief-mapping question targeted the cycle's most consequential architectural commitment (ADR-027 as primary direction) against its rejected alternative (Tier 1 hybrid), inviting the practitioner to name the belief-shape under which the rejected alternative would have been correct.

The practitioner's response was substantively gap-identifying — they named the long-horizon capability ceiling question (the empirical gap the cycle's Spike battery did not exercise) and flagged the corpus's own concession ("we haven't invested enough to answer that question"). The response was not a defense of the rejected alternative; it was an honest surfacing of the empirical territory the cycle's commitment rests on but did not test.

The agent's response composed three moves: (1) acknowledged the gap substantively; (2) reframed via the asymmetry argument — under Tier 1 hybrid, non-`tool_choice` NL routing continues through the orchestrator-LLM-as-decider (the failure locus per Cycle 6 PLAY note 22 + Spike λ-paid F-paid-4), so Tier 1 does not rescue the long-horizon case; both candidate Cycle 7 mechanisms face the same gap; (3) reframed the question to "does ANY single-step-plus-bounded-role architecture handle generalized long-horizon at cheap tier?" and offered three options (proceed-with-question-logged; architecturally revise toward optionality; add a Spike ν probe).

The practitioner chose **Spike ν** as architect→build boundary work. The agent added Spike ν to roadmap Track A.3 with pre-specified qualitative criteria (per MODEL snapshot Advisory A — pre-specify before running) and three trigger conditions (Pass / Fail / Intermediate). Architect-snapshot Advisory B added the multiple-intermediate rule before snapshot dispatch closed.

## Commitment gating outputs

**Settled premises (the practitioner is building on these going into BUILD):**

1. **AS-9 + AS-10 are constitutional commitments for the chat-completions surface.** Both invariants codified across MODEL (AS-9, 2026-05-22) and DECIDE (AS-10, 2026-05-22) boundaries. ARCHITECT honors them structurally — Routing Planner + Response Synthesizer satisfy AS-9 (FC-29); Routing Planner + Serving Layer + Capability List Builder satisfy AS-10 (FC-30). The invariants bind downstream BUILD work; future technically-permitted-but-spirit-violating extensions surface as constitutional amendments.

2. **ADR-027 framework-driven dispatch pipeline is Cycle 7's primary direction for the chat-completions surface.** Six new modules implement it; the orchestrator-LLM is removed from the routing-decision and post-dispatch-synthesis surfaces; `OrchestratorRuntime` preserved as architectural option per disposition (a). Commitment stands pending Spike ν empirical outcome.

3. **The three Track C ARCHITECT-phase deferrals are resolved.** Finding 2 → preserve `OrchestratorRuntime` as architectural option (with the dormant-class cost-accounting honest-residual-uncertainty Advisory D carries to BUILD). Finding 8 → Dispatch Pipeline yields same chunk vocabulary as runtime; `OpenAiSseFormatter` unchanged. Finding 11 → Plan→InternalToolCall adapter inside Dispatch Pipeline module.

4. **ADR-076 qualitative-claim decomposition is complete for Cycle 7.** Fourteen new fitness criteria FC-28..FC-41 cover the Cycle 7 qualitative claims; two annotations (direction-not-constraint for latency tuning; honest-residual-uncertainty for cost-distribution-accountability) preserve the structurally-non-decomposable surfaces visibly. Architect-snapshot positive signal: most complete decomposition in the corpus.

5. **Cycle 7 BUILD shape is 5 WPs + 2 Track A refactors + Spike ν probe.** Dependency graph + transition states (TS-10 + TS-11) + open decision points are roadmap-recorded. Spike ν gates WP-A entry.

**Open questions (carried forward to BUILD):**

1. **Long-horizon capability ceiling generalization (Spike ν direct test).** The practitioner-named gap that prompted Spike ν addition. The spike's three surfaces (multi-step composition; production-scale numerical content; adversarial routing) test the structural-bounding generalization AS-9 + ADR-027 commit to. Three trigger paths (all-Pass / any-Fail / single-Intermediate) plus the multiple-intermediate rule (per architect-snapshot Advisory B) govern post-spike action.

2. **OrchestratorRuntime dormant-class cost-accounting honesty (architect-snapshot Advisory D).** The "architectural-option preservation, not actively-maintained dual surfaces" framing is optimistic. BUILD should document the maintenance reality explicitly — integration-repair cost as surrounding infrastructure evolves; version-drift risk between dormant code and active code; deprecation cost should the class need to be re-activated by a future cycle. Implications for ADR-001 + ADR-011's continuing architectural-commitment claim.

3. **Inversion Principle checks across the 6 new modules (architect-snapshot Advisory C).** The 3-stage decomposition (Dispatch Pipeline + Routing Planner + Response Synthesizer) serves operator-configurability + developer-testability more than Population A's mental model. Capability Discovery Endpoint's inversion note may be absent. BUILD should clarify which model each boundary serves; the decomposition is correct but the inversion checks lean architectural rather than user-centered.

4. **The architect-snapshot asymmetry-argument partial-reductiveness (architect-snapshot Advisory A).** The Tier-1-doesn't-rescue-long-horizon argument is correct on bounded-role coverage differential but elides that the capability ceiling applies equally to both mechanisms' bounded-role components. Spike ν correctly frames this as a shared empirical question. BUILD should not over-rely on the asymmetry argument when interpreting Spike ν results; the spike tests the shared bounded-role question directly.

5. **WP-A streaming wiring + WP-C EnsembleExecutor per-ensemble streaming API (architect-snapshot Advisory D BUILD-discovery risks).** WP-A's pipeline chunk surface replication needs to honor `OrchestratorRuntime` streaming internals without using them; WP-C's synthesizer streaming relies on EnsembleExecutor exposing per-ensemble streaming. Neither is an architectural gap; both are BUILD-discovery risks that surface during work.

6. **Routing Planner production traffic diversity (OQ #25).** Spike ζ established mechanism viability on a 20-prompt battery; production traffic diversity surfaces during PLAY (or first-deployment evidence). Spike ν's adversarial routing surface partially exercises this; production-scale validation remains PLAY territory.

7. **Multi-step composition mechanism beyond single-step planner + framework-chain-heuristic (OQ #21).** Cycle 7 BUILD defaults to single-step planner; Spike ν tests this default at 2-step + 3-step composition; OQ #21 names three candidate mechanisms (single-step planner + framework chain-heuristic; multi-step planner; planner-loops-with-context). BUILD or PLAY may surface evidence warranting one of the alternatives.

8. **AS-10 incremental opt-in erosion risk (architect-snapshot Pattern 4 informational).** Latent: future BUILD or follow-on-cycle work could erode AS-10's "no client-side opt-in" commitment through technically-permitted-but-spirit-violating extensions. BUILD should flag any pattern that introduces client-supplied routing signals not on the OpenAI-protocol-native list.

**Specific commitments carried forward to BUILD:**

1. **Run Spike ν before WP-A starts.** Pre-specified qualitative criteria locked in (multi-step composition + production-scale numerical + adversarial routing). Three trigger conditions + multiple-intermediate rule. Pass → WP-A proceeds; Fail → Design Amendment re-opens architecture; Intermediate → caveat-with-deployment-policy + practitioner authorizes; multiple-Intermediate → practitioner authorizes one of three paths. Writeup target: `essays/research-logs/cycle-7-spike-nu-long-horizon-ceiling.md`.

2. **Apply Track A refactors before WP-B / WP-C BUILD work.** A.1 — routing-planner spike YAML adds `input` field per ADR-028 §Output contract (precedes WP-B). A.2 — synthesizer spike YAML adds Rule 6 per ADR-029 §"Strict-fidelity rule set" (precedes WP-C).

3. **BUILD documents `OrchestratorRuntime` dormant-class maintenance reality** (architect-snapshot Advisory D). The "architectural-option preservation" framing in system-design.md + ORIENTATION.md gets a cost-accounting annotation visible to future cycles. The annotation explicitly names: integration-repair cost as surrounding code evolves; version-drift risk; deprecation cost should re-activation be needed; ADR-001 + ADR-011's continuing-commitment claim qualified accordingly.

4. **BUILD attends to architect-snapshot Advisory C on Inversion Principle checks.** The 3-stage decomposition's user-mental-model-vs-developer-convenience distinction surfaces in field-guide entries when BUILD generates them. The Capability Discovery Endpoint's inversion check is added explicitly.

5. **BUILD honors the asymmetry-argument partial-reductiveness caveat** (architect-snapshot Advisory A). Spike ν's results are interpreted as shared-mechanism empirical evidence on the bounded-role-capability question, not as evidence for ADR-027 vs. Tier 1 per se. If Spike ν triggers Design Amendment, the re-opened architectural deliberation considers both ADR-027 and Tier 1 (and multi-mechanism architectures) on their merits at the time, not on the asymmetry argument.

6. **BUILD respects the multiple-intermediate rule on Spike ν** (architect-snapshot Advisory B). The rule prevents the "many non-fatal results collectively mask a ceiling" failure mode that single-surface evaluation does not catch.

7. **WP-A and WP-C BUILD entries surface the BUILD-discovery risks explicitly** (architect-snapshot Advisory D). The work decomposition + integration test plan handle streaming chunk surface replication and EnsembleExecutor per-ensemble streaming API as named risks rather than implicit assumptions.

8. **Cycle 7 BUILD-mode declaration: gated.** Recommended at ARCHITECT close given the central architectural pivot character + design-alternative surfaces (`tool_choice` disposition + multi-step composition mechanism + capability-list discovery surface choice). Auto mode appropriate only after WP-A structural shape is in place + the remaining WPs reduce to mechanical wiring.
