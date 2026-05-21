# RDD Cycle Status — Agentic Serving (Scoped)

**Artifact base:** `docs/agentic-serving/`
**Plugin version at cycle open:** v0.8.6
**Migration version:** 0.8.5 (`housekeeping/.migration-version`)

## Cycle Stack

### Active: Cycle 7 — Framework-driven orchestration: routing as code (provisional title)

**Cycle number:** 7
**Started:** 2026-05-20 (cycle prepared; RESEARCH not yet entered)
**Current phase:** research (not yet entered)
**Cycle type:** standard (full pipeline)
**Plugin version:** v0.8.6
**Artifact base:** `docs/agentic-serving/`
**BUILD mode:** gated (declared at cycle preparation; this cycle is design-alternative-heavy — auto mode would miss surfacing trade-offs between routing-planner ensemble, `tool_choice` constrained decoding, and hybrid approaches; gated stewardship reviews are appropriate)

**Origin:** Cycle 6 PLAY (2026-05-20) surfaced that the orchestrator-LLM's chain-handling behavior is the load-bearing failure for composition pipelines; ADR-022 amendment is empirically bounded to bare-endpoint mode and does not shift NL routing under tool-rich production clients. Spike δ confirmed that framework-driven Python chaining preserves data correctly across the same `web-searcher → claim-extractor` composition that failed under orchestrator-LLM routing. PLAY-tail grounding work (per susceptibility-snapshot recommendation) softened the initial "framework-driven pipeline" framing into bounded claims:

- Framework-driven **chain-handling** is well-grounded (Spike δ + alternatives belief-map confirm)
- Framework-driven **routing-decision** is contested — `tool_choice` constrained decoding is a viable lighter-weight alternative to a full routing-planner ensemble
- "Must-delegate to ensembles" is bounded to capability-matched requests, not universal
- The form-vs-content drift bifurcation may collapse if schema-as-enforcement unifies them

Cycle 6 status archived at `cycle-archive/cycle-6-ensemble-contract-observability-routing.md`.

**Cycle 7 question framing (provisional, for RESEARCH entry):**

Three linked architectural questions:

1. **Where does the routing decision live?** Options: (a) framework-driven routing-planner ensemble; (b) `tool_choice` constrained decoding forcing `invoke_ensemble` when a capability match is detected; (c) hybrid where a small classifier decides which mechanism applies per request. The chain-handling commitment is already framework-driven (Spike δ + belief-map); the routing-decision is the open territory.

2. **How are I/O contracts enforced?** Output_schema currently documentary (per Cycle 6 WP-D moderate advisory 1 resolution); claim-extractor's form drift demonstrates documentary schemas don't deliver compliance. Options: (a) schema-as-enforcement with reject-and-retry; (b) tool-call-as-output-format (agents call a `submit_results` tool whose params are the schema); (c) deterministic shaper after the agent (extra dispatch per ensemble with declared schema). Connects to form-vs-content drift collapse question — schema-as-enforcement might address both.

3. **What's the fallback shape for non-capability-matched requests?** Options: (a) general-completion fallback ensemble (preserves infrastructure, adds dispatch overhead); (b) direct LLM completion as residual through the orchestrator-LLM (preserves latency, loses infrastructure); (c) lightweight shim that wraps direct completion with minimal infrastructure (calibration verdict + audit log but no full ensemble overhead). The must-delegate-as-bounded framing means the fallback shape is design territory, not a forced choice.

These three questions are linked: the routing-decision mechanism + the contract-enforcement mechanism + the fallback shape together determine the cycle 7 architecture.

**Specific carry-forwards from Cycle 6 to address in Cycle 7 (or as standalone Thread A fixes between cycles):**

1. **`dispatch_log.json` overwrite vs. append.** Currently each chat-completions request overwrites the file with its own dispatches. Operator-facing post-hoc review use case is broken — the latest empty-dispatch request wipes prior session entries. Fix shape: append-mode, OR per-request files, OR reconstruct-from-artifacts at session close. Small fix; could land as Thread A (outside methodology cycle) or fold into Cycle 7 architecture work if the persistence semantics intersect the planner+synthesizer roles.

2. **`profile=?` placeholder in `dispatch start` log line.** Cosmetic — the profile is unknown at dispatch_start (resolved in `tier selection` one line later). Could be left blank, populated from the resolution step, or removed. One-line fix.

3. **Capability ensemble form drift at synthesizer/agent layer.** Claim-extractor's output is non-conformant to its `default_task` spec across all invocation paths. Cycle 7 DECIDE will land an ADR on I/O contract enforcement policy that addresses this directly.

**MODEL handling for Cycle 7:** Likely-needed. The architectural pivot introduces new vocabulary candidates: `routing-planner`, `response-synthesizer`, `general-completion fallback`, `plan-dispatch-synthesize pipeline`, `framework-as-orchestrator`. If RESEARCH + DISCOVER surface enough new vocabulary, formalize in `domain-model.md` before DECIDE. Decision deferred to DISCOVER's tail.

**DISCOVER handling for Cycle 7:** Required. The architectural pivot has stakeholder-mental-model implications:

- Tool User's "endpoint is a model" abstraction: holds or breaks under the new pipeline?
- Skill Orchestration User's "the orchestrator routes by capability under NL" mental model: now needs re-grounding given the routing-decision mechanism is design-open
- Ensemble Author / Operator: routing-planner authoring is a new authoring surface; what does the operator need to understand about the planner ensemble?
- Orchestrator LLM (as actor): its role shifts from decider to executor or disappears entirely — this is a stakeholder-model change worth surfacing

**ARCHITECT handling for Cycle 7:** Required. The chat-completions handler refactor is non-trivial: the orchestrator-LLM-as-decider goes away (or shrinks); new modules for routing-planner integration, synthesizer integration, plan-execution loop. Fitness criteria need to capture the new pipeline's invariants (e.g., "every chat-completions request resolves to ≥1 ensemble dispatch when a capability match exists"). Substantial ARCHITECT phase.

## Phase Status

| Phase | Status | Artifact | Key Epistemic Response |
|-------|--------|----------|----------------------|
| RESEARCH | ☐ Pending (next) | — | — |
| DISCOVER | ☐ Pending | — | — |
| MODEL | ☐ Pending | — | — |
| DECIDE | ☐ Pending | — | — |
| ARCHITECT | ☐ Pending | — | — |
| BUILD | ☐ Pending | — | — |
| PLAY | ☐ Optional | — | — |
| SYNTHESIZE | ☐ Optional | — | — |

## Spike candidates for Cycle 7

To be ordered after RESEARCH establishes the question shape, but pre-named for orientation:

- **Spike ε:** minimal routing-planner ensemble + response-synthesizer ensemble + Python harness for end-to-end plan→dispatch→synthesize pipeline. Run against the prompts that broke under orchestrator-LLM routing.
- **Spike ζ:** routing-planner reliability — JSON-output adherence across prompt shapes, including adversarial / ambiguous prompts. Tests the cheap-tier-model-as-structured-output assumption.
- **Spike η:** synthesizer behavior on multi-dispatch results — does the narrower role + structured input prevent the confabulation pattern the orchestrator-LLM exhibited?
- **Spike θ:** general-completion fallback ensemble shape — does wrapping direct completion in an ensemble degrade response quality (compared to note 1 / note 18 quality on the string-reverse task)?
- **Spike ι:** schema-as-enforcement reliability — does qwen3:8b produce schema-conformant JSON when retried with feedback after non-conformant output?
- **Spike κ (alternative-architecture):** `tool_choice` constrained decoding probe under Zen + MiniMax M2.5 — does Zen support the parameter? Does the model honor it? Latency + reliability shape?

## Feed-Forward Signals

### From Cycle 6 PLAY (closed at gate 2026-05-20)

1. **Composition drift is in the orchestrator-LLM.** Empirically confirmed by Spike δ. The chain-handling failure does not survive removing the LLM from the chain step.
2. **ADR-022 amendment is bounded to bare-endpoint mode.** Tool-rich clients suppress the amendment regardless of model tier. Production deployments are all tool-rich.
3. **Must-delegate is a bounded claim.** For capability-matched requests it is a current-state constraint; for non-matched it is a future-state aspiration. The fallback path needs explicit design.
4. **Form-vs-content drift may unify.** Schema-as-enforcement could address both in one mechanism.
5. **WP-C orchestrator-context observation works for single-dispatch lookup, fabricates on multi-dispatch summary.** Reading is reliable; narrating across dispatches degrades. Cycle 7 should account for this if any synthesis-of-results role uses the same observation surface.
6. **dispatch_log.json per-request overwrite breaks the operator-facing review use case.** Fix needed (Thread A or in-cycle).
7. **MiniMax-native XML tool-call leakage under bare-endpoint mode** is a model-output-format artifact when no client tools are declared. Not a defect; a tool-surface gap. Worth knowing for client-integration design.
8. **OpenCode interaction shape is preserved under the framework-driven pipeline candidate.** External API stays OpenAI-compatible; internal architecture is what shifts. Tool User mental model holds.

### From Cycle 6 ARCHITECT (closed at gate 2026-05-15)

(retained for reference; see archive for full detail) — `system-design.md` v4.0 + `roadmap.md` v6.0 capture the architecture as of Cycle 6 close. Cycle 7 ARCHITECT will need to update these substantially.

### From earlier cycles

See `cycle-archive/cycle-{1..6}-*.md` for prior-cycle feed-forward signals.

## Context for Resumption

Anyone resuming this cycle in a new session should read in this order:

1. This file (Cycle 7 cycle-status.md) — current state and framing.
2. `cycle-archive/cycle-6-ensemble-contract-observability-routing.md` — what closed Cycle 6 and what carries forward.
3. `essays/research-logs/cycle-6-play-grounding.md` — PLAY-tail grounding work that bounded the Cycle 7 premises.
4. `essays/research-logs/cycle-6-spike-delta-framework-chaining.md` — Spike δ writeup (the empirical foundation for the chain-handling claim).
5. `housekeeping/audits/susceptibility-snapshot-cycle-6-play.md` — the PLAY-boundary snapshot that triggered the grounding work.
6. `essays/reflections/field-notes.md` Cycle 6 PLAY section (notes 11-25 + Spike δ + BUILD verdicts table) — the empirical substrate.
7. `system-design.md` + `roadmap.md` — Cycle 6 close state of the architecture (Cycle 7 will update substantially).
8. `ORIENTATION.md` — high-level system framing.

When ready to enter RESEARCH, `/rdd-research` with the Cycle 7 question framing above as the entry topic.
