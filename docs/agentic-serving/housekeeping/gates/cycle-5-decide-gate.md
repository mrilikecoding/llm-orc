# Gate Reflection: Agentic Serving DECIDE → BUILD (Cycle 5)

**Date:** 2026-05-12
**Phase boundary:** DECIDE → BUILD
**Cycle:** Cycle 5 — Agentic-serving library structure (capability ensembles + multi-skill-framework-consumer surface)

## Belief-mapping question composed for this gate

> ADR-021 settles per-capability dispatch as the composition contract and rejects methodology-composer ensembles. The falsification trigger names *output-quality divergence across skill frameworks consuming the same Topaz slot* as the empirical condition that would invalidate the contract.
>
> What would have to be true for the parameterized-capability-ensembles resolution path (path 1 — `argument-mapper(skill_framework=...)`) to be the right starting point at Cycle 5 already, rather than as a falsification-only response?
>
> Conversely: what would have to be true for the plain operation-named shape (current commitment) to remain the right starting point even after operating with two or more skill frameworks?

## User's response

> My concern with path 1 is that there's an infinite space of skills and skill frameworks that could be leveraged by llm-orc agentic serving — they are pluggable, or more accurately, they can drive an agent (or orchestrator of agents in our case). The resting assumption here is that our cheap cloud orchestrator can employ local free models to do work. This is a cost savings. Ultimately we care most about the results of long-horizon tasks. I'd have to see that orchestration does not produce good results with our general topaz system and also need to see evidence that a framework we know about (RDD or similar) encoded into agentic serving is the only way to get good orchestrated results. As in a skill framework would become an agentic serving modality in order to achieve a good result, versus a generalized orchestration scheme that could be leveraged by any skill/skill framework to achieve the tasks requested by said framework.

## Pedagogical move selected

**Challenge** via belief-mapping. The question presented two sharp alternatives on the parameterization timing (Cycle 5 already vs. falsification-only response) and asked what would have to be true for each. The practitioner answered substantively with three reinforcing components: (1) the value-proposition framing (cheap-cloud orchestrator employing local free models for cost savings on long-horizon tasks); (2) the conjunctive evidence standard (general scheme fails AND framework-encoding is the only recovery path); and (3) the architectural commitment that skill frameworks are pluggable consumers, not modalities of the orchestrator. The conjunctive standard was sharper than the agent's original falsification framing in ADR-021; the practitioner specifically targeted the premature-inversion failure mode where one capability ensemble's serving one skill framework better than another might be mistaken for inverting the commitment. ADR-021's falsification trigger was revised to reflect the conjunctive standard and the long-horizon-task-outcome measurement surface.

## Commitment gating outputs

**Settled premises (the practitioner is building on these going into BUILD):**

1. **Skill-framework-agnostic dispatch is the architectural commitment**, and the falsification standard for inverting it is conjunctive: (a) the general scheme must fail to produce good long-horizon task results AND (b) framework-encoding into agentic serving must be empirically the *only* way to recover good results. Sub-task verdict divergence alone does not invalidate the commitment.

2. **The value proposition the commitment serves is cost savings via local-free-model leverage under cheap-cloud orchestration**, and the measurement surface is *long-horizon task outcomes*, not per-sub-task calibration verdicts. The Tier-Router Audit's drift criteria (ADR-018) measure tier-routing signal; long-horizon task-outcome signal is the BUILD-and-beyond test surface.

3. **The minimum-viable capability ensemble set (5 ensembles)** is the BUILD-scope authoring target. The set is RDD-research-workflow-representative rather than agnostically balanced across all 8 Topaz slots; the agnostic commitment is at the *contract level*, not at the initial-shape selection level.

4. **The on-ramp authoring (profile file, subdirectory, README, rewritten config section)** is in BUILD scope. The cycle's commitment is operator-runnable deployment shape, not just mechanism architecture.

5. **The web-searcher script-agent with Tavily default** is the `tool_use` slot's authored capability. Brave/Exa/Serper adapters are deferred to operator-driven extension under the adapter pattern.

6. **Per-capability dispatch contract is the skill-orchestration composition shape.** Explicit-naming dispatch is preferred; natural-language dispatch is supported with retrieval-not-evaluative-classification framing.

**Open questions (the practitioner is holding these open going into BUILD):**

1. **Non-RDD skill-framework empirical evidence**: snapshot Advisory 1 — the skill-framework-agnostic commitment's scope claim (covering Anthropic Skills, OpenAI Assistants, MCP-based frameworks) is grounded in n=1-framework structural verification (RDD only). BUILD-phase scope is RDD-decomposition only; the scope claim's evidential breadth gap persists at BUILD close unless a non-RDD framework is exercised.

2. **No-dispatch-fallback resolution durability**: snapshot Advisory 2 — the "intended scope" framing closes the discover gate's examination commitment at minimum threshold (PLAY-note citation rather than discursive argument). The orchestrator's reliability profile claim ("high on derivable, low on integration") is now scoped to Cycle 4 PLAY inhabitation range. Whether this resolution holds under broader deployment evidence remains observational.

3. **Vocabulary candidacy**: "capability ensemble", "operation-named ensemble", "three-layer architecture" enter BUILD as candidates pending settled-by-use confirmation. The product-discovery vocabulary table updates at BUILD close: settled (survived concrete authoring) or relocated to research-voice in domain-model.md §Methodology Vocabulary.

4. **Cycle Acceptance Criteria Table layer-match gaps**: the Cycle 5 table identifies 4 emergent/aggregate criteria — three with Layer-match "no" requiring integration tests or live-deployment evidence at BUILD Step 5.5. One criterion ("compose into a runnable deployment on first encounter without operator manual authoring beyond environment-variable setup") is a live-install verification surface; BUILD Step 5.5 closes this with an integration test or harness exercise.

**Specific commitments carried forward to BUILD (from snapshot advisories):**

1. **Advisory 1 (scope-claim breadth)**: BUILD's RDD-decomposition exercise is the scope of empirical verification. If BUILD-phase work produces decomposition evidence for any non-RDD framework, capture it; otherwise, the scope claim persists as candidate at BUILD close.

2. **Advisory 2 (no-dispatch-fallback minimum-threshold resolution)**: this is the cycle's primary residual exposure. BUILD work that surfaces orchestrator-natural-language-response errors should be recorded as candidate evidence for either continued "intended scope" resolution or future-cycle reconsideration as coverage-gap territory.

3. **BUILD mode declaration**: the cycle-status's `**BUILD mode:**` field defaults to `gated` per ADR-091. The practitioner declares the mode at BUILD entry — gated for per-scenario-group review, or auto for mechanical-character work where stewardship concentrates at start-and-end. Given the cycle's BUILD work is YAML/config authoring (profile file + subdirectory + 5 ensemble YAMLs + README + config rewrite + web-searcher script-agent), `auto` mode may fit the mechanical character; the practitioner's call.

4. **Downstream-artifact sweep for ADR-019's update of ADR-015**: the four-artifact sweep (system-design.md, ORIENTATION.md, domain-model.md, field-guide.md) is **deferred to BUILD** per Step 2.5's allowance. ADR-019 records the deferral; at BUILD close, the sweep is performed (and may be brief — the reframing is at the proposal/product-discovery characterization layer, not at the system-design module layer).

## Susceptibility snapshot outcome

Snapshot at `housekeeping/audits/susceptibility-snapshot-cycle-5-decide.md`: **no Grounding Reframe warranted**. The phase shows the corpus's strongest audit discipline to date — three-round audit pattern functioned as substantive correction (round 2 caught new issues from round 1 revisions; round 3 was clean). The conjunctive falsification standard was practitioner-generated with architectural precision, targeting premature-inversion failure mode; low susceptibility weight on that element.

Two advisory carry-forwards integrated above:

- **Advisory 1 (scope-claim breadth, BUILD-phase settlement)**: the n=1-framework scope assertion will persist at BUILD close unless BUILD produces non-RDD framework decomposition evidence.
- **Advisory 2 (no-dispatch-fallback resolution at minimum threshold)**: the resolution closes the discover gate's examination commitment by assertion + PLAY-note citation rather than discursive argument; carried as the cycle's primary residual exposure.

Three-round audit pattern outcome: substantive correction, not convergence-toward-friendly-text. Round-2 issues were *generated by* round-1 revisions — characteristic of substantive correction, not cosmetic convergence.
