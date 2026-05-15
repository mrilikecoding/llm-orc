# ADR-019: Skill-Framework-Agnostic Orchestrator + Operation-Named Capability Ensemble Library

> **Updated by ADR-022 on 2026-05-15.** §Consequences §Positive's profile-portability claim is qualified — the library's portability across deployments is structural; the orchestrator's routing behavior across orchestrator profiles is empirically not uniform (per spike γ). See the inline qualification in §Consequences §Positive. The rest of this ADR (skill-framework-agnostic dispatch contract, operation-named library principle, three-layer separation, rejected alternatives, falsification trigger) remains current.

**Status:** Updated by ADR-022

**Date:** 2026-05-12

---

## Context

Cycle 4 BUILD landed the mechanism architecture for tier-escalation, calibration verdicts, audit dispatch, and the cross-layer calibration channel (WP-A4 through WP-H4). The Cycle 4 PLAY inhabitation surfaced that the cycle shipped *mechanism architecture* but not the *operator-facing on-ramp* — `.llm-orc/config.yaml` had inline tier profiles, only two ensembles carried `topaz_skill` metadata, and the operator-driven library migration framing (ADR-015 §Negative) reads cleanly at decide time but produced a first-encounter gap. The Cycle 4 PLAY note 15 (practitioner-generated, verbatim during inhabitation) surfaced the reframing: *"Llm-orc doesn't need to have an rdd lit-reviewer ensemble any more than it needs a 'rdd-research' ensemble, but it would need to be able to invoke an ensemble capable of doing the tasks needed by that skill or subskill."* Cycle 5 DISCOVER gate (2026-05-12) refined this further: the architectural commitment is **skill-framework-agnostic**, not just methodology-agnostic, covering RDD and emerging skill standards (Anthropic Skills, OpenAI Assistants, MCP-based skill frameworks). Practitioner verbatim at the gate: *"Better is to be able to leverage any agentic skill (as 'skills' are more-or-less standardized these days). [Encoding] a specific flow into llm-orc... that's not my first choice."*

The Topaz 8-skill taxonomy (introduced in ADR-015 for tier-escalation routing) is already the lingua franca: any skill framework that decomposes its workflow into capability-typed sub-tasks can tag those sub-tasks with Topaz skills, and the orchestrator routes by Topaz skill + calibration verdict without needing to know which skill framework is composing against it. ADR-015 §Negative's "operator-driven library migration" framing reframes under this commitment: not "tag every existing ensemble with whatever topaz_skill fits"; rather, "author the operation-named capability ensemble set that serves the deployment's skill frameworks."

Cycle 4 PLAY note 14 confirms the structural feasibility: `invoke_ensemble`'s fresh-context dispatch property — the dispatched ensemble's agents receive `input + system_prompt` only, no orchestrator conversation history — maps cleanly to the architectural property skill frameworks need (each capability sub-task runs in fresh context, no contamination from prior sub-tasks). No ADR amendment required to make the existing primitive surface support skill-framework-agnostic composition.

The product-discovery update (Cycle 5) introduces the **Skill Orchestration User** role: a distinct role whose defining characteristic is *using a skill orchestration process* — a client-side framework that decomposes higher-level workflows into capability-typed sub-tasks. The role is distinct from Tool User and Ensemble Author / Operator; humans can wear multiple roles concurrently.

---

## Decision

*Vocabulary note before the decision*: "capability ensemble", "operation-named ensemble", and "three-layer architecture" enter this ADR with the candidate status recorded in `product-discovery.md`'s vocabulary table (Cycle 5 update). The terms are used throughout this ADR's body as the working organizing vocabulary; the ADR's concrete library-authoring decisions and §Consequences are the DECIDE-phase test those terms' candidacy was pending. If BUILD-phase concrete authoring (Cycle 5 BUILD scope per the §Working defaults section) confirms the terms serve operator and architecture-reader use, the product-discovery vocabulary table moves them from "candidate under DECIDE examination" to "settled (survived BUILD-phase authoring)" at cycle close.

The orchestrator is **skill-framework-agnostic**: it routes by capability (Topaz 8-skill taxonomy + calibration verdict per ADR-015) without knowing which skill framework is composing against it. The capability ensemble library that serves agentic-serving deployments is **operation-named** and **capability-fine-grained**.

### The architectural commitment

The orchestrator's responsibility ends at **capability routing**. Three layers of responsibility separate cleanly:

1. **Skill orchestration layer** — client-side. Any standardized skill framework that decomposes a higher-level workflow into capability-typed sub-tasks. RDD's `rdd:*` skill plugin is one instance; Anthropic Skills, OpenAI Assistants, MCP-based skill frameworks, and emerging skill standards are others. The skill framework owns *what is decomposed and in what order*; the orchestrator never knows by name which specific framework is composing against it.

2. **Capability dispatch layer** — the orchestrator. Routes each capability-typed sub-task to an ensemble in the library by consulting (a) the ensemble's primary Topaz skill (ADR-015), and (b) the per-skill tier defaults and calibration verdict (ADR-014, ADR-015). The orchestrator owns *which ensemble runs the sub-task at which tier*.

3. **Capability instance layer** — the ensemble library. Each ensemble is named for the **operation** it performs (e.g., `claim-extractor`, `argument-mapper`, `web-searcher`, `code-generator`, `prose-improver`, `text-summarizer`), not for the skill framework that invokes it. Each capability ensemble is tagged with a primary `topaz_skill` (ADR-015) and is invokable by any skill framework that decomposes its workflow into Topaz-typed sub-tasks.

The Topaz 8-skill taxonomy is the **lingua franca** between layers 1 and 2: skill frameworks tag their decomposed sub-tasks with Topaz skills; the orchestrator routes by them.

### Library shape principle: capability-fine-grained, operation-named

The capability ensemble library is **fine-grained**: one ensemble per operation. `claim-extractor` extracts claims; `argument-mapper` maps argument structure; `web-searcher` performs web search. The same library serves many skill frameworks — RDD's lit-review consumes `web-searcher` + `claim-extractor`; a security-review-as-methodology skill consumes `claim-extractor` for source claims; a code-review-as-methodology skill consumes `code-generator` for refactoring suggestions.

The library is **not** methodology-coarse: no `rdd-lit-reviewer`, no `security-source-extractor`, no `code-reviewer` ensembles. Methodology-coarse naming would couple library entries to specific skill frameworks and require duplicating library work for each new skill framework.

Operator-driven migration (ADR-015 §Negative) reads under this principle as: *"author the operation-named capability ensemble set that serves the deployment's skill frameworks."* The set is deployment-specific (RDD-only deployments need a different set than RDD+security-review deployments); the set's *shape principle* (operation-named, capability-fine-grained) is invariant across deployments.

### Working defaults are in Cycle 5 BUILD scope

Cycle 4 PLAY note 1 surfaced that auto-mode BUILD landed mechanism architecture without operator-facing working defaults (practitioner verbatim: *"the agentic-serving config is to me part of the build"*). Cycle 5 BUILD scope includes:

- An agentic-serving **profile file** (`.llm-orc/profiles/agentic-serving-profiles.yaml`) isolating all agentic-serving Model Profiles so model swaps are single-file edits. Profile names referenced from `agentic_serving.orchestrator.model_profile` and `per_skill_tier_defaults` slots in `.llm-orc/config.yaml`.
- An agentic-serving **ensemble subdirectory** (`.llm-orc/ensembles/agentic-serving/`) namespacing capability ensembles separately from the rest of the library. System ensembles (`agentic-result-summarizer`, `agentic-calibration-checker`) move into the subdirectory; the names are kept.
- A **minimum-viable capability ensemble set** authored in BUILD: `code-generator` (promoted from Cycle 4 PLAY's `agentic-coding-helper`, code_generation), `claim-extractor` (factual_knowledge), `argument-mapper` (logical_reasoning), `prose-improver` (writing_quality), `text-summarizer` (summarization). The set is the minimum that demonstrates the principle and serves RDD's research workflow; deployments needing more author them at their leisure under the same shape principle. *Selection criterion acknowledgment:* the set's choice of five ensembles is driven by RDD's concrete BUILD-time demand (the only methodology consumer with concrete BUILD-time presence at Cycle 5). This makes the initial library shape *RDD-representative* rather than agnostically balanced across all eight Topaz slots — covering 5 of 8 slots from RDD's research-workflow needs. The skill-framework-agnostic commitment is at the *contract level* (any skill framework can compose against this library); the *initial shape's selection* is single-methodology-consumer-driven. A future deployment whose dominant methodology consumer needs different slots would author different ensembles under the same shape principle.
- A **rewritten `agentic_serving:` section** in `.llm-orc/config.yaml` referencing the renamed profiles and the new subdirectory layout.
- An **operator-facing README** at `.llm-orc/ensembles/agentic-serving/README.md` documenting the structure and the extension pattern.

The `mathematical_reasoning` Topaz slot remains **unauthored** in Cycle 5's minimum-viable set (resolving OD-1 to option (b) from the proposal). The slot is configured in `per_skill_tier_defaults` (the resolver requires all 8 slots covered); dispatches routing to math hit `MissingSkillMetadataError` because no library ensemble carries `topaz_skill: mathematical_reasoning`. The orchestrator's recovery path reformulates and tries another approach. Cycle 4 PLAY note 4 empirically validated the recovery path works under the cheap-cloud-orchestrator pattern. A `math-solver` capability ensemble is authored when a skill framework actually needs the slot — operator-driven, deployment-specific.

The `tool_use` slot's ensemble shape is resolved in ADR-020.

The skill-orchestration composition shape is resolved in ADR-021.

---

## Rejected alternatives

### (a) Methodology-coarse library: `rdd-lit-reviewer`, `code-reviewer`, etc.

Each skill framework's workflow shape is encoded directly into ensemble names. The orchestrator becomes methodology-aware: it knows that `rdd-lit-reviewer` is for RDD's lit-review phase, that `code-reviewer` is for code-review-as-methodology, and so on.

**Rejected because:** this couples library entries to specific skill frameworks and requires duplicating library work for each new methodology consumer. A deployment running RDD + security-review needs to maintain `rdd-lit-reviewer` + `security-source-extractor` even though both consume `claim-extractor`-shaped capability. The same skill framework's evolution (new sub-skills, renamed phases) churns library entries that have no genuine capability change. The Cycle 5 DISCOVER gate explicitly rejected this framing — practitioner verbatim: *"[Encoding] a specific flow into llm-orc... that's not my first choice."*

### (b) Encode RDD specifically (or any single skill framework) into the orchestrator's routing logic

The orchestrator is given direct awareness of RDD's phase shapes — `invoke_ensemble("rdd-research-step", ...)` accepts an additional argument naming the RDD phase, and the orchestrator's routing logic dispatches RDD-phase-aware. Other skill frameworks integrate the same way: the orchestrator gains awareness of each.

**Rejected because:** rules out the architectural commitment from the Cycle 5 DISCOVER gate. The orchestrator's value as a substrate for *any* skill framework is exactly that it doesn't know about any of them specifically. Methodology-aware routing is a coupling that grows linearly with the number of skill frameworks supported; the skill-framework-agnostic commitment scales to whatever skill standard emerges.

### (c) Skill-framework-specific subdirectories: `.llm-orc/ensembles/rdd/`, `.llm-orc/ensembles/code-review/`, etc.

Each skill framework gets its own namespace under `.llm-orc/ensembles/`. The orchestrator's ensemble walk path covers all subdirectories; methodology-named ensembles live under their methodology's subdirectory.

**Rejected because:** at the directory layout level, this is the same methodology-coarse coupling as alternative (a). The directory namespace would be load-bearing for which capability ensembles are available; library reuse across methodologies would either require symlinks, duplicate authoring, or a separate "shared" subdirectory. Operation-named ensembles in a single `agentic-serving/` namespace serve all methodology consumers without per-methodology directory machinery.

### (d) Inline library spec inside `agentic_serving:` config.yaml section (no separate profile file or subdirectory)

The Cycle 4 deployment shape — model profiles inline in `.llm-orc/config.yaml`, ensembles distributed across `.llm-orc/ensembles/*.yaml` at the top level — is kept. The Cycle 5 BUILD scope authors capability ensembles at the top level and references their profiles inline.

**Rejected because:** the practitioner-stated job *"I want working defaults on first encounter"* (Cycle 4 PLAY note 1) trades against config-file ergonomics. Inline profiles couple model-swap edits to the config file's structure; cross-skill-framework deployments add config-file complexity. A separate profile file (`agentic-serving-profiles.yaml`) makes model swaps single-file edits; a separate subdirectory (`agentic-serving/`) makes the namespace boundary structural; a single rewritten `agentic_serving:` section in config.yaml references both. The on-ramp clarity is load-bearing for the Cycle 5 BUILD scope decision (per tension #11).

### (e) Defer the library-structure decision to operator authoring at deployment time

ADR-015 §Negative's "operator-driven library migration" stands literally: the cycle ships mechanism architecture; the library is the operator's responsibility entirely. No working defaults; no minimum-viable set; no profile file.

**Rejected because:** Cycle 4 PLAY note 1 is the rejection. Auto-mode BUILD that ships mechanism without on-ramp produces a first-encounter gap the operator cannot close from the shipped state without a separate authoring pass. The cycle's BUILD-mode declaration (auto vs. gated) is independent from the question of *what* BUILD ships; auto-mode can still ship operator-facing artifacts when those are scoped into the cycle. Cycle 5 scopes them in.

---

## Consequences

### Positive

- **Skill-framework-agnostic dispatch scales to any skill standard.** RDD today; Anthropic Skills, OpenAI Assistants, MCP-based skill frameworks tomorrow; whatever emerges next year. The orchestrator's commitment is structural and survives skill-standard evolution. *Scope-of-claim qualifier (per Cycle 5 BUILD snapshot, finding 3):* this "scales to any skill standard" property is **structural — the dispatch contract is invariant across skill frameworks** — but **empirically verified for n=1 skill framework only** (RDD) at Cycle 5 close. The skill-framework-capability-registry's entries for Anthropic Skills, OpenAI Assistants, and MCP-based skill frameworks are *placeholders pending operator evaluation*, not verified deployments. The Cycle 5 cycle acceptance criteria table's Layer-match `no` entry for "at least two distinct skill orchestration users" names this gap directly; resolution path is per-deployment operator evidence as additional skill frameworks integrate.
- **Operation-named library is reusable across skill frameworks.** Each capability ensemble serves multiple methodology consumers. Library investment compounds across deployments.
- **Three-layer responsibility separation is debuggable.** A failed sub-task localizes to one of three layers: skill-framework decomposition (was the right capability requested?), capability dispatch (was the right ensemble routed?), or capability execution (did the ensemble execute correctly?). Each layer is independently diagnosable.
- **Working defaults close the first-encounter gap.** Cycle 5 BUILD ships an operator-runnable deployment shape, not just mechanism architecture.
- **Profile file isolates model swaps.** `.llm-orc/profiles/agentic-serving-profiles.yaml` makes model swaps a single-file edit. Cross-deployment portability improves. *Cycle 6 qualification (per spike γ, 2026-05-15):* the **library's** portability across deployments is structural (any deployment-time profile swap operates on the same library). The **orchestrator's routing behavior** across orchestrator profiles is empirically **not** uniform — spike γ Cell A vs Cell B documented MiniMax M2.5-free under-delegating to direct completion while qwen3:14b over-delegating to client tools under identical NL framing. ADR-022's system-prompt amendment is the cycle's remediation; effectiveness is configuration-conditional per ADR-022's disposition (iii) framing. The portability-improvement claim above scopes to library shape and deployment-time model swaps, not to cross-orchestrator-profile routing-behavior uniformity.
- **Subdirectory namespace clarifies extension.** Operators adding new capability ensembles know where they go; readers know what's in scope for agentic-serving versus the rest of the library.

### Negative

- **Operator-driven library authoring remains the operator's responsibility for non-minimum-viable capabilities.** A deployment needing a capability not in the minimum-viable set (e.g., `math-solver`, `web-searcher` if not authored) authors it. Cycle 5 reduces this from "author the entire library" to "author what's missing from the minimum-viable set"; it does not eliminate operator authoring.
- **The Topaz 8-skill taxonomy becomes a constraint operators learn.** Capability ensembles must tag with a Topaz skill; skill frameworks decompose against Topaz skills. Operators learn the taxonomy at deployment time. The Topaz authors' framing (capability saturation; 8 skills span the meaningful capability space) is the basis for accepting this constraint, but the constraint is real.
- **Methodology-specific affordances live client-side, not in the orchestrator.** A skill framework wanting orchestrator-side support for its specific shape (e.g., RDD wanting `invoke_ensemble` to take an `rdd_phase:` parameter) does not get it. The skill framework decomposes against Topaz skills and lives with that. The decoupling is the architectural commitment; the cost is borne by skill-framework authors.

### Neutral

- **The no-dispatch fallback path is intended scope, not a coverage gap.** Cycle 4 PLAY note 19 observed that prompts the orchestrator answers directly — without invoking any library ensemble — bypass the entire dispatch-conditional quality infrastructure (Calibration Gate per ADR-014, Tier-Router Audit per ADR-018, cross-layer signal channel per ADR-016). The Cycle 5 DISCOVER gate routed this as a DECIDE-phase framing-examination question (coverage gap requiring infrastructure extension vs. intended scope). The Cycle 5 DECIDE framing resolution: the orchestrator's direct natural-language response is the appropriate behavior when no library ensemble matches the task's shape. The quality infrastructure's scope is *dispatch-conditional by design* — Calibration Gate, Tier-Router Audit, and the signal channel measure *dispatched-ensemble* quality; they do not (and were not designed to) measure orchestrator-natural-language-response quality. Library coverage expansion (more capability ensembles, more skill-framework decomposition coverage) extends the dispatch infrastructure's reach but never to 100% — evaluative and meta-tasks naturally fall on the no-dispatch path. The orchestrator's reliability profile (high on derivable claims, low on integration claims, observed within Cycle 4 PLAY's inhabitation-session task range across notes 8, 10, 11, 12, 13, 18; n=1 inhabitation, single orchestrator profile) is the load-bearing property for no-dispatch path quality at the evidence currently available. The "consistent across task types" characterization is scoped to the Cycle 4 PLAY session — broader task ranges and other orchestrator profiles may reveal exceptions (Cycle 4 PLAY note 3 already recorded one routing-failure case where the orchestrator misrouted a meta-introspection prompt to a code-generation ensemble). Calibration on orchestrator-own-narration (Cycle 4 PLAY note 16's coverage observation) is *separate* infrastructure territory — not in Cycle 5 scope, not foreclosed for future cycles; whether it becomes a coverage-gap concern is conditional on deployment-evidence accumulation of orchestrator-narration error rates beyond what the Cycle 4 inhabitation range showed.
- **The `mathematical_reasoning` slot's MissingSkillMetadataError-recovery path is empirically validated** (Cycle 4 PLAY note 4) under one orchestrator profile (MiniMax M2.5-free via Zen). Other orchestrator profiles may handle the typed error differently. Operator-driven authoring of `math-solver` remains the resolution path when the recovery path's reliability is insufficient for the deployment's task class.
- **The Skill Orchestration User role is distinct but typically wearer-collapsed with Ensemble Author / Operator.** Solo developers running their own llm-orc backend, authoring their own skill plugin, and using it through OpenCode wear all three roles (Tool User, Ensemble Author / Operator, Skill Orchestration User). The role separation is meaningful even when collapsed; concerns are distinct. Multi-person deployments may split the roles across humans.
- **The Cycle 4 PLAY snapshot's attribution discipline carries forward.** The "three-layer architecture", "operation-named ensembles", and "capability ensemble" framings derive from one inhabitation session (n=1) plus the Cycle 5 DISCOVER gate refinement. This ADR settles them as cycle-decisions on the strength of (a) the practitioner-generated gate refinement, (b) the structural fit with existing `invoke_ensemble` properties (note 14), and (c) BUILD-phase concrete-authoring as the practical test. Survival through BUILD's concrete authoring decisions is the test these framings face this cycle; future cycles may revisit.

## Provenance check

- **Skill-framework-agnostic commitment**: Cycle 5 DISCOVER gate (2026-05-12), practitioner verbatim refinement of the proposal's "methodology-agnostic" framing. Driver chain: practitioner-generated.
- **Three-layer architecture (skill orchestration / capability dispatch / capability instance)**: Cycle 4 PLAY note 15 (practitioner-generated verbatim refinement) + proposals/agentic-serving-library-structure.md §"Three-layer architecture (load-bearing framing)". Driver chain: practitioner-generated, captured in substrate. *Vocabulary status*: "three-layer architecture" is marked research-voice in product-discovery (candidate for relocation to `domain-model.md` §Methodology Vocabulary). Its use as the organizing structure of this ADR (and ADR-021's inversion-question dispatch table) effectively promotes the term to settled-as-structural-vocabulary in this cycle's corpus; after BUILD-phase concrete authoring exercises the term in practice, the product-discovery vocabulary table is updated with this ADR as the settlement basis, or — if the term has not entered operator voice by BUILD close — the term is relocated to `domain-model.md` §Methodology Vocabulary as research voice.
- **Operation-named library principle**: Cycle 4 PLAY note 15 (practitioner-generated) + proposal substrate.
- **Topaz 8-skill taxonomy as lingua franca**: ADR-015 (driver). The Cycle 5 use of "lingua franca between layers" framing is drafting-time synthesis of the existing ADR-015 mechanism with the Cycle 5 three-layer separation.
- **Minimum-viable capability ensemble set (5 ensembles)**: proposal substrate §"Capability ensemble specs (minimum viable set)". Driver chain: substrate-derived; the proposal's selection of 5 reflects RDD's immediate research-workflow needs (the only methodology consumer with concrete BUILD-time demand).
- **Profile file isolation pattern**: proposal substrate §"Proposed model profile file". Driver: substrate-derived (n=1 practitioner-confirmed at proposal authoring time).
- **Subdirectory namespace pattern**: proposal substrate §"Proposed directory structure". Driver: substrate-derived.
- **`mathematical_reasoning` slot unauthored**: Cycle 4 PLAY note 4 (empirical observation that MissingSkillMetadataError recovery works) + proposal §"Open decisions" OD-1 recommended option. Driver chain: empirical + substrate-derived.
- **Working-defaults-in-BUILD reframing**: Cycle 4 PLAY note 1 (practitioner verbatim) + Cycle 5 product-discovery tension #11. Driver chain: practitioner-generated, captured in substrate and updated discovery artifact.
- **ADR-015 §Negative reframing**: drafting-time synthesis bridging the substrate framing to the Cycle 5 architectural commitment. Will be reflected in ADR-015 partial-update header.
