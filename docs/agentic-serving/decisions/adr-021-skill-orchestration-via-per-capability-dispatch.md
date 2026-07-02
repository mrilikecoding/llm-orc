# ADR-021: Skill-Orchestration Composition via Per-Capability Dispatch

> **Superseded by ADR-045 on 2026-07-01 (Cycle-8 clean-slate collapse, AS-11).** The `OrchestratorRuntime` per-capability ReAct-dispatch mechanism is retired (imperative orchestration in the removed `src/llm_orc/agentic/`). The per-capability dispatch requirement — route each sub-task to its capability ensemble — carries forward as topaz-keyed dynamic dispatch (see ADR-045).

> **Updated by ADR-022 on 2026-05-15.** The "natural-language prompt" dispatch shape (§Decision §"Topaz-skill signal path: how the orchestrator identifies the right ensemble") is narrowed: "supported" now means "preferred when capability match exists, via the amended orchestrator system prompt." The permissive reading (supported as one option among several) is replaced by the amended commitment. The rest of this ADR (per-capability dispatch contract, `compose_ensemble` scope, cross-sub-task state, rejected alternatives, falsification trigger, consequences) remains current.

> **Updated by ADR-024 and ADR-025 on 2026-05-15.** §Decision §"Per-capability dispatch contract" step 3 ("The orchestrator returns the capability ensemble's output (already summarized per AS-7) to the skill framework as the chat completion response") is updated: the response shape is now the typed envelope per ADR-024; summarization is conditional on substrate-routing per ADR-025 (substrate-routed ensembles' deliverables are not summarized at content level; inline-response ensembles retain summarization per amended AS-7). Substantive commitments of the per-capability dispatch contract are preserved.

> **Updated by ADR-027 on 2026-05-22.** Per AS-9 §Propagation (domain-model.md, codified 2026-05-22) and ADR-027 (Framework-Driven Dispatch Pipeline as Primary Direction for the Chat-Completions Surface), the actor producing the routing decision on the agentic-serving chat-completions surface shifts from the orchestrator-LLM (current via `OrchestratorRuntime` ReAct loop) to the **routing-planner ensemble** per ADR-028. The two supported dispatch shapes ADR-021 names (explicit ensemble naming; natural-language prompt) are now both routed through the routing-planner ensemble **on the chat-completions surface** — explicit names become an explicit-naming intent signal the planner honors, and NL prompts are routed by the planner's capability-match decision. The per-capability dispatch contract's structural commitments (one capability sub-task per request; client-side workflow state; fresh-context property; calibration-gate-per-sub-task) are unchanged; only the **actor of the routing decision** shifts on the chat-completions surface. The `OrchestratorRuntime` ReAct loop remains available as an architectural option (per ADR-001 + ADR-011) for any future surface that adopts the ReAct execution model. **Per the Cycle 7 Tranche 4 conformance scan (Finding 2), `OrchestratorRuntime` currently has no production caller other than the chat-completions handler being replaced by ADR-027** — the `llm-orc invoke` CLI and other REST endpoints route through `OrchestraService` directly, not through `OrchestratorRuntime`. The actor-shift described above is chat-completions-surface scoped; non-chat-completions surfaces (CLI; REST endpoints for ensembles, artifacts, scripts, profiles; future surfaces) continue to operate per ADR-021's original dispatch shapes via `OrchestraService` directly.

**Status:** Superseded by ADR-045 (2026-07-01); formerly Updated by ADR-022, ADR-024, ADR-025, ADR-027

**Date:** 2026-05-12

---

## Context

ADR-019 establishes the skill-framework-agnostic orchestrator commitment: the orchestrator routes by capability (Topaz skill + calibration verdict per ADR-014/015) without knowing which skill framework is composing against it. The proposal `proposals/agentic-serving-library-structure.md` §OD-3 names three concrete shape options for *how* skill frameworks actually compose against the capability library:

- **(a) Skill prompts decompose to multiple `invoke_ensemble` calls** in a single client-side turn. The skill framework (client-side) emits one orchestrator request per capability-typed sub-task; the orchestrator routes each independently.
- **(b) Skill prompts dispatch to a top-level methodology-composer ensemble** that internally invokes capability ensembles via `compose_ensemble`. The composer ensemble holds the methodology's decomposition shape; the orchestrator dispatches against the composer; the composer dispatches against capability ensembles.
- **(c) Hybrid** — phase entry dispatches a methodology-shape ensemble; sub-tasks dispatch capability ensembles.

The Cycle 5 DISCOVER gate settled the architectural commitment: the orchestrator is skill-framework-agnostic. The OD-3 question is which composition shape *operationalizes* that commitment — and snapshot Advisory 1 flagged that the agnostic commitment was settled at the gate before its seam-case inversion (does Topaz-skill routing produce routing-quality parity across skill-framework contexts?) was examined. Snapshot Advisory 2 named four inversion questions for OD-3 to dispatch:

1. What would have to be true for the three-layer separation to be the wrong abstraction?
2. What would have to be true for "operation-named ensembles" to be wrong (vs. methodology-named)?
3. What would have to be true for the `agentic-` prefix / `agentic-serving/` subdirectory convention to be wrong?
4. What would the right ensemble decomposition look like if the orchestrator were *not* skill-framework-agnostic?

The four inversion questions are dispatched within this ADR's Rejected alternatives section (each option's rejection rationale examines one or more of the inversion questions against the option's structural commitments).

Cycle 4 PLAY note 14 (architectural-isolation mapping) and note 15 (three-layer separation, practitioner-generated) plus the Cycle 5 DISCOVER gate refinement establish the substrate: the orchestrator's `invoke_ensemble` fresh-context dispatch property is the architectural property skill frameworks need.

The skill orchestration layer is client-side (per ADR-019); the client side is one of:
- A Claude Code skill plugin (e.g., RDD's `rdd:*` skills) that decomposes phase prompts into `invoke_ensemble` calls
- An agentic-coding client (OpenCode, Roo Code, Cline) consuming the orchestrator's response surface and producing skill-decomposed prompts
- A future skill-framework client (Anthropic Skills runtime, OpenAI Assistants, MCP-skill-framework client, ...)

In each case, the skill framework's decomposition logic lives on the client side, not on the orchestrator side.

---

## Decision

Skill frameworks compose against the orchestrator's capability library via **per-capability dispatch** (option (a) of OD-3): the skill framework (client-side) decomposes its workflow into capability-typed sub-tasks, and emits one orchestrator request per sub-task. The orchestrator's job per request is *single-capability dispatch*: route the sub-task to the matching capability ensemble in the library by Topaz skill, calibration verdict, and tier-router decision.

### Per-capability dispatch contract

For each capability-typed sub-task the skill framework needs:

1. The skill framework (client-side) emits an OpenAI-compatible chat completion request to the orchestrator, with the sub-task framed as the user's prompt (or as `invoke_ensemble` tool-call arguments — both shapes work).
2. The orchestrator's ReAct loop receives the request, identifies the capability ensemble for the sub-task, and dispatches via `invoke_ensemble`.
3. The orchestrator returns the capability ensemble's output (already summarized per AS-7) to the skill framework as the chat completion response.
4. The skill framework consumes the response and proceeds to the next capability sub-task in its decomposition.

The orchestrator processes one capability sub-task at a time per request. Skill-framework decomposition logic (which sub-skills run, in what order, with what dependencies) is **client-side only**. The orchestrator does not know that a sequence of three `invoke_ensemble` calls came from an RDD lit-review phase versus from a security-review source-extraction phase versus from an ad-hoc workflow.

#### Topaz-skill signal path: how the orchestrator identifies the right ensemble

Two dispatch shapes are supported, with structurally different signal paths:

- **Explicit ensemble naming**: the skill framework names the capability ensemble directly in `invoke_ensemble` tool-call arguments (`invoke_ensemble("claim-extractor", {...})`). The orchestrator's routing is *pre-specified* — no LLM-judgment classification — and the Tier-Escalation Router (ADR-015) reads the ensemble's pre-tagged `topaz_skill` metadata from its YAML to select the tier. This is the preferred shape; it preserves ADR-015's pre-specified-routing commitment end-to-end. Skill frameworks that can name ensembles directly (because they know the deployment's library topology via the `skill-framework-capability-registry.md` artifact or via runtime `list_ensembles()`) use this path.
- **Natural-language prompt**: the skill framework sends the sub-task as user prompt content; the orchestrator selects the ensemble using its routing logic (`list_ensembles()` consultation plus LLM-judgment matching of prompt to ensemble description). This shape reintroduces LLM-judgment at the *capability-selection* boundary — the orchestrator chooses which capability ensemble matches a natural-language task description. This is accepted as a *narrower* LLM-judgment scope than the alternative ADR-015 rejected in §(f) ("Skill metadata as runtime classification"): ADR-015 §(f) rejected having the orchestrator classify *output quality* via LLM judgment; this shape uses LLM judgment for *input-to-ensemble matching*, which is a different judgment task — matching a task description to an ensemble description is closer to retrieval (with the deployment-time `list_ensembles()` result as the retrieval corpus) than to evaluative classification. Skill frameworks unable or unwilling to maintain library-topology knowledge use this path.

The pre-specified shape is preferred; the natural-language shape is supported. The `skill-framework-capability-registry.md` artifact (Cycle 5 OD-6 resolution) documents the deployment's library topology for skill framework authors maintaining the registry. Skill frameworks emitting natural-language sub-tasks accept LLM-judgment routing at the capability-selection boundary as a deployment trade-off; the Tier-Router Audit's drift criteria (ADR-018) measure the operational impact.

### `compose_ensemble` retains its existing scope

`compose_ensemble` (ADR-006) remains scoped to runtime composition of *capability ensembles* from existing library primitives. It is **not** the pathway for skill-framework composition. A skill framework wanting to compose new capabilities can `compose_ensemble` from existing capability ensembles, but the composition is *capability-level*, not *methodology-level* — the composed ensemble is itself a capability ensemble (tagged with a Topaz skill, library-eligible, dispatchable via `invoke_ensemble`), not a methodology-shape ensemble.

### Cross-sub-task state lives client-side

State that crosses capability sub-tasks (e.g., RDD's "the lit-review found these claims; the citation-audit should verify them") is **client-side state**, held by the skill framework or passed forward via the next sub-task's prompt content. The orchestrator does not maintain skill-framework-level state across `invoke_ensemble` calls. `invoke_ensemble`'s fresh-context property (no orchestrator history bleeds into the dispatched ensemble's context) is preserved as the load-bearing architectural property.

---

## Rejected alternatives

### (b) Skill prompts dispatch to a top-level methodology-composer ensemble

The skill framework emits a single orchestrator request per phase; the orchestrator dispatches against a "methodology-composer" ensemble (e.g., `rdd-research-composer`) that holds the methodology's decomposition shape internally. The composer ensemble uses `compose_ensemble` (or a chain of `invoke_ensemble` calls inside its own definition) to dispatch capability ensembles.

**Rejected because:** the methodology-composer ensemble is structurally methodology-specific. Either the orchestrator's library contains one composer ensemble per methodology (`rdd-research-composer`, `security-review-composer`, `code-review-composer`, ...) — which is exactly the methodology-coarse library shape ADR-019 rejected — or there is a single generic composer ensemble that takes the methodology's decomposition shape as input, which (1) reproduces the per-capability-dispatch pattern at one layer down (the composer's input is the decomposition; the generic-composer pattern just adds a passthrough layer) and (2) couples the orchestrator-side composer to whatever decomposition-language the skill framework speaks (a methodology-specific input schema in disguise).

This option fails inversion question 4 ("what would the right ensemble decomposition look like if the orchestrator were not skill-framework-agnostic?"): a methodology-composer ensemble *is* the answer to "orchestrator-aware decomposition" — it lives in the orchestrator's library, names the methodology, and runs the decomposition. The Cycle 5 DISCOVER gate's settled commitment rules out the orchestrator-aware-decomposition direction.

This option also concentrates decomposition in a single ensemble call, which mismatches the existing `invoke_ensemble` fresh-context property — a composer ensemble would have to maintain decomposition state across its internal sub-dispatches, either via `compose_ensemble`'s ensemble-composition mechanism (which is library-state, not session-state) or via internal context that bleeds across sub-tasks (which defeats the fresh-context property note 14 identified as load-bearing).

### (c) Hybrid — phase entry dispatches a methodology-shape ensemble; sub-tasks dispatch capability ensembles

A compromise between (a) and (b): the skill framework's phase entry dispatches a methodology-shape ensemble that prepares context (loads prior phase outputs, sets up decomposition state, etc.); sub-tasks within the phase dispatch capability ensembles.

**Rejected because:** the methodology-shape ensemble at phase entry is itself methodology-specific — it lives in the library, names the methodology, and inherits all the problems of option (b) for the entry-point dispatch. The "hybrid" framing doesn't escape the per-capability-dispatch alternative for the within-phase sub-tasks; it just adds an entry-point coupling on top.

A second concern: the "phase preparation" work the entry-point ensemble would do is *workflow state management* — loading prior phase outputs, setting up decomposition state. That state is client-side state in option (a); the hybrid would move it server-side without changing what the state *is*. Server-side workflow state doesn't fit the orchestrator's per-request model (each `invoke_ensemble` call is independently routed; no session-level workflow state). The hybrid would require a parallel state-management mechanism, which is non-trivial.

### Per-capability dispatch via a "skill framework registry" the orchestrator consults

A registry maps skill-framework identifiers (`rdd`, `code-review`, `security-review`) to decomposition rules. Skill frameworks declare themselves at request time; the orchestrator looks up the decomposition and routes capability sub-tasks accordingly.

**Rejected because:** declaring skill frameworks to the orchestrator violates the agnostic commitment — the orchestrator would know skill frameworks by name. The registry is the methodology-aware alternative dressed in registry-pattern clothing.

This option fails inversion question 1 ("what would have to be true for the three-layer separation to be the wrong abstraction?"): a registry-based orchestrator collapses the skill-orchestration layer into the orchestrator (since the orchestrator now needs to know about each skill framework), making the three-layer separation a fiction. ADR-019's settled commitment rules this out.

---

## Inversion-question dispatch (per snapshot Advisory 2)

The four inversion questions named at DISCOVER-gate close are dispatched against the option-rejection rationales above:

| # | Inversion question | Examined in rejected-alternative slot |
|---|---|---|
| 1 | What would have to be true for the three-layer separation to be the wrong abstraction? | "Registry alternative" rejection — three-layer separation collapses if the orchestrator knows skill frameworks by name |
| 2 | What would have to be true for "operation-named ensembles" to be wrong (vs. methodology-named)? | Resolved at ADR-019 §Rejected alternative (a); revisited here through option (b)'s methodology-composer-ensemble framing — methodology-named composer ensembles are methodology-coarse library entries in disguise |
| 3 | What would have to be true for the `agentic-` prefix / `agentic-serving/` subdirectory convention to be wrong? | Not load-bearing for OD-3 (the question is a layout choice, not a composition choice). ADR-019 §Rejected alternative (c) addresses *per-framework subdirectories* (a different inversion). The convention's wrongness conditions — that `agentic-serving/` conflates system ensembles (`agentic-result-summarizer`, `agentic-calibration-checker`) with capability ensembles (`code-generator`, `claim-extractor`, etc.) in a way that confuses operator navigation, or that the `agentic-` prefix ages poorly as the serving feature becomes the deployment default — are BUILD-phase authoring concerns. ADR-019's BUILD-scope commitment includes an `agentic-serving/README.md` that distinguishes the two ensemble categories explicitly; that README is the operator-navigation mitigation. Not further examined here |
| 4 | What would the right ensemble decomposition look like if the orchestrator were *not* skill-framework-agnostic? | Option (b) rejection — methodology-composer ensembles are the right shape for skill-framework-aware orchestration; the cycle's settled commitment rules out that direction |

Inversion question 3 is not load-bearing for OD-3's composition-shape decision; the question's natural home was ADR-019 (resolved there). The remaining three questions are examined in the rejection rationales above and would each have produced a different ADR if the cycle's settled commitment had been the inverse direction.

### Seam-case inversion (per snapshot Advisory 1)

Snapshot Advisory 1 named the seam-case inversion: *does Topaz-skill routing produce routing-quality parity across skill-framework contexts, or do framework-specific dispatch needs surface?* The question is **empirical, not design-time**. The Topaz 8-skill taxonomy was validated empirically at Cycle 4 architect-gate Spike α (research log `005g-spike-topaz-skill-classification.md`) against the existing library — 21 of 21 production ensembles satisfied the clean-primary criterion. The validation was against ensembles authored without methodology-specific intent in mind; framework-specific dispatch needs surfacing at deployment would be visible in the Tier-Router-Audit's verdict-distribution shift criterion (ADR-018), the escalation-vs-outcome correlation criterion, or the bypass-rate-trend criterion.

The seam-case inversion is **conditional on deployment evidence**, not on this ADR's design-time choice. The per-capability dispatch contract this ADR establishes does not foreclose either resolution: if framework-specific dispatch needs surface (e.g., RDD's argument-mapper sub-tasks consistently calibrate Reflect-verdict at the cheap tier while code-review's argument-mapper sub-tasks consistently calibrate Proceed), the resolution path is *per-skill-framework tier defaults* — an extension to ADR-015's `per_skill_tier_defaults` configuration to permit per-skill-framework override on a Topaz-skill slot — without disturbing the per-capability dispatch contract.

#### Falsification trigger

The value proposition the agnostic commitment serves: **the cheap-cloud orchestrator can employ local free models to do work, producing good long-horizon task results via generalized orchestration — a cost-savings architecture that scales to any skill framework that decomposes its workflow into Topaz-typed sub-tasks.** The measurement surface is *long-horizon task outcomes*, not per-sub-task calibration verdicts; the load-bearing claim is *outcome parity (or better)* across skill frameworks, achieved via the generalized scheme.

The per-capability dispatch contract is invalidated only when deployment evidence satisfies a **conjunctive standard**:

> **(a)** The generalized agnostic scheme — operation-named capability ensembles dispatched via Topaz routing under cheap-cloud orchestration with local-free-model tiers — does not produce good results on long-horizon tasks, with "good results" assessed at the task-outcome level (lit-review citation soundness, code-review architectural-issue-surfacing rate, security-review source-claim coverage, etc.) rather than at the sub-task calibration level.
>
> **AND**
>
> **(b)** A skill framework encoded into agentic serving — implemented as either parameterized capability ensembles (`argument-mapper(skill_framework=...)`) or per-skill-framework capability ensembles (`rdd-argument-mapper`) — is empirically the *only* way to recover good long-horizon task results, with the encoded shape producing results the agnostic scheme cannot.

Both conditions must hold. Sub-task verdict divergence alone (which would surface in Tier-Router-Audit's drift criteria per ADR-018) does **not** invalidate the contract — that's tier-routing-level signal, not task-outcome-level signal, and the resolution path for it is per-skill-framework tier defaults extending ADR-015 (already named as the lighter available extension). Output-quality divergence at the sub-task level is similarly insufficient if long-horizon task outcomes remain comparable across skill frameworks via the agnostic scheme.

The conjunctive standard rules out the most common form of premature inversion: discovering one capability ensemble's outputs serve one skill framework better than another and concluding the agnostic commitment was wrong. Per-capability-dispatch is structurally agnostic; the test surface for *whether agnostic-as-architectural-commitment is correct* is at the *long-horizon task outcome* layer, where the value proposition's claim is staked. The cost-savings value (local-free-model leverage) is independent of which skill framework consumes the dispatch and is preserved regardless.

#### Resolution paths under falsification

If and only if conditions (a) AND (b) both hold across multiple skill frameworks operating against the same capability library, the contract is invalidated. Three resolution paths are available, in increasing distance from the agnostic commitment:

1. **Parameterized capability ensembles** — `argument-mapper` becomes `argument-mapper(skill_framework=...)`, taking the consuming skill-framework's identifier as a parameter that conditions the ensemble's prompt or grading rubric. The library entries remain operation-named and methodology-agnostic at the dispatch surface; the parameter handles output-quality divergence. This is the *lightest* extension — preserves ADR-019's operation-named library principle but admits some capability ensembles need skill-framework-conditional behavior.
2. **Per-skill-framework capability ensembles** — `rdd-argument-mapper`, `security-review-argument-mapper`, etc. This is the methodology-coarse library shape ADR-019 §Rejected alternative (a) rejected. ADR-019's library-shape commitment re-opens.
3. **Explicit acceptance that the agnostic commitment was over-broad** — operates only at the dispatch level, not the output-quality level. The skill-framework-agnostic library serves dispatch routing; output-quality coverage requires per-skill-framework adaptation in some other layer (the methodology composer's prompt shape, the calibration checker's grading criteria, etc.).

ADR-019's commitment re-opens under resolution paths (2) and (3); path (1) preserves the commitment with refinement. The trigger fires the question of which resolution path the cycle revisitation chooses; this ADR does not foreclose any of them. **The trigger does not fire on sub-task signal alone; it requires task-outcome signal that the agnostic scheme cannot recover.**

The trigger is **filed as Cycle 5+ research territory**. The Tier-Router-Audit's drift criteria (ADR-018) are the operational measurement surface; the per-skill-framework override is the lighter available extension when evidence shows tier-routing divergence; the falsification trigger fires when output-quality divergence is structurally persistent.

---

## Consequences

### Positive

- **Skill-framework-agnostic commitment is operationalized as a contract** — per-capability dispatch is the structural shape that keeps the orchestrator's library and the skill framework's decomposition logic decoupled. No new methodology-coarse library entries; no orchestrator-side state that crosses sub-tasks; no skill-framework registry.
- **Fresh-context property is preserved.** The architectural fact: `invoke_ensemble`'s dispatch mechanics give each dispatched ensemble's agents `input + system_prompt` only — no orchestrator conversation history bleeds into the dispatched context. This property is real (it is how `invoke_ensemble` works); the *characterization* of it as load-bearing for sycophancy-resistance-style architectural patterns is an agent-introduced analogy from RDD's ADR-058 (Architectural Isolation) captured in Cycle 4 PLAY note 14 with an attribution flag, pending BUILD-phase or future-cycle testing in non-RDD methodology contexts. Skill frameworks needing the architectural-isolation property get the property as a structural consequence of `invoke_ensemble`'s mechanics; whether the property carries the *same load-bearing significance* for non-RDD methodologies is the part that remains candidate.
- **`compose_ensemble` scope is preserved** — runtime capability composition stays in scope for `compose_ensemble`; methodology composition stays out. The two scopes are independent and stay independent.
- **Skill-framework portability is structural** — a skill framework written for one capability library serves any other library exposing the same Topaz skills. RDD's `rdd:*` skill plugin running against Deployment A's library serves the same decomposition against Deployment B's library, even if the underlying ensembles differ.
- **Per-capability quality infrastructure fires per sub-task** — Calibration Gate, Tier-Router-Audit, cross-layer signal channel all fire on each `invoke_ensemble` call. A skill framework's workflow is quality-instrumented at the capability granularity, which is finer than the methodology granularity would be.

### Negative

- **Skill-framework decomposition logic is the client's responsibility, in entirety** — there is no orchestrator-side composer to lean on; the skill framework owns the full decomposition. Client-side complexity grows linearly with the skill framework's complexity (more sub-skills, more decomposition logic). Skill frameworks unable to host decomposition logic client-side cannot use the orchestrator productively.
- **Topaz-taxonomy-aligned decomposition is a precondition on the skill framework side.** The orchestrator routes by Topaz skill (per ADR-015); skill frameworks composing against the orchestrator must decompose their workflows into sub-tasks that map cleanly to the eight Topaz skills. A skill framework with its own internal decomposition vocabulary (RDD's phase shapes are *not* inherently Topaz-skill-named) requires either (a) the skill framework internally maps its decomposition vocabulary to Topaz skills before emitting dispatch requests, or (b) an adapter layer between the skill framework's vocabulary and the Topaz routing vocabulary. RDD's `rdd:*` skill plugin is the immediate methodology consumer; its phase-to-capability mapping is documented in `skill-framework-capability-registry.md` (Cycle 5 OD-6 resolution). Other skill frameworks integrating against the orchestrator absorb the same precondition; the integration burden grows linearly with skill-framework diversity. The Topaz 8-skill taxonomy's coarse granularity (8 categories spanning all LLM-capability space) is the constraint; the registry artifact is the deployment-time mitigation.
- **Cross-sub-task state is the skill framework's responsibility** — workflow state (e.g., "lit-review's claims feed the citation-audit's verification list") lives client-side. The orchestrator does not provide a workflow-state primitive. Skill frameworks needing rich workflow state author their own state-passing mechanism (typically: pass forward via the next sub-task's prompt content, or via the client-side framework's own state model).
- **Quality infrastructure does not fire on skill-framework-level metrics** — the orchestrator does not know which sub-tasks are part of an RDD lit-review (versus a code-review), so it cannot calibrate methodology-level patterns (e.g., "RDD lit-reviews escalate twice as often as code-reviews"). Skill-framework-level quality measurement, if needed, is the skill framework's responsibility (client-side analytics). The Tier-Router-Audit measures *capability-level* drift, not *methodology-level* drift.
- **Per-skill-framework tier defaults are not in Cycle 5 scope** — if the seam-case inversion's empirical resolution requires per-skill-framework overrides on Topaz-skill slots, the ADR-015 extension is a future cycle's work. Cycle 5 ships the per-Topaz-skill defaults (ADR-015 as-is); deployment evidence may surface need for the extension.

### Neutral

- **The orchestrator's per-request model fits OpenAI-compatible chat completions cleanly** — each `invoke_ensemble` is naturally one chat completion. Skill frameworks composing against the OpenAI-compatible surface (most existing agentic-coding clients) compose against per-capability dispatch without protocol adaptation.
- **`invoke_ensemble`'s recovery paths (ADR-015's typed errors, ADR-017's phantom-tool-call guard) operate per sub-task** — a skill framework's sub-task that hits MissingSkillMetadataError reformulates its sub-task; the framework's other sub-tasks proceed independently. Failure isolation is at the capability level, not the methodology level.
- **Methodology consumers seeking orchestrator-side support for their specific shape don't get it** — RDD's `rdd-research-composer` doesn't exist as a library entry; nor does any other methodology-shape ensemble. Methodology consumers absorb this cost (the skill framework runs client-side; decomposition logic is theirs). The architectural commitment is *for* this trade.

## Provenance check

- **Per-capability dispatch contract**: Cycle 4 PLAY note 14 (architectural-isolation mapping) + note 15 (three-layer separation) + Cycle 5 DISCOVER gate (skill-framework-agnostic commitment) + ADR-019. Driver chain: practitioner-generated + same-cycle-prior-ADR.
- **`compose_ensemble` scope clarification**: ADR-006 (driver). Driver chain: prior-ADR-derived.
- **Fresh-context property as load-bearing**: Cycle 4 PLAY note 14, attribution-flagged as agent-introduced framing of an empirical property of `invoke_ensemble`. Driver chain: empirical property (load-bearing) + agent-introduced framing (per Cycle 4 PLAY snapshot advisory).
- **Rejection of methodology-composer ensembles (option b)**: drafting-time synthesis examining option (b) against ADR-019's settled commitment. Driver chain: prior-cycle-ADR-derived (ADR-019).
- **Seam-case inversion deferral to deployment evidence**: snapshot Advisory 1 (driver) + ADR-018's drift criteria (existing operational measurement surface). Driver chain: snapshot-derived + prior-ADR-derived.
- **Per-skill-framework tier defaults as future extension territory**: drafting-time synthesis; the extension path is the natural next-cycle work if seam-case evidence warrants. Not an ADR commitment; flagged as available extension path.
- **Inversion-question dispatch table**: snapshot Advisory 2 (driver). The dispatch of each question to a rejection-rationale slot is drafting-time synthesis bridging snapshot-derived input to ADR structure.
