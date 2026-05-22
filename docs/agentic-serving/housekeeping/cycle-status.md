# RDD Cycle Status — Agentic Serving (Scoped)

**Artifact base:** `docs/agentic-serving/`
**Plugin version at cycle open:** v0.8.6
**Migration version:** 0.8.5 (`housekeeping/.migration-version`)

## Cycle Stack

### Active: Cycle 7 — Framework-driven orchestration: routing as code (provisional title)

**Cycle number:** 7
**Started:** 2026-05-20 (cycle prepared; RESEARCH not yet entered)
**Current phase:** discover
**In-progress phase:** discover
**Cycle type:** standard (full pipeline)
**Plugin version:** v0.8.6
**Artifact base:** `docs/agentic-serving/`
**BUILD mode:** gated (declared at cycle preparation; this cycle is design-alternative-heavy — auto mode would miss surfacing trade-offs between routing-planner ensemble, `tool_choice` constrained decoding, and hybrid approaches; gated stewardship reviews are appropriate)
**DISCOVER entered:** 2026-05-21

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
| RESEARCH | ✅ Complete | `essays/essay-outline-006-cross-compatibility-routing-surface.md` + reflection `essays/reflections/006-cross-compatibility-routing-surface.md` + research log `essays/research-logs/006-cross-compatibility-routing-surface.md` | Practitioner pushed back on hybrid framing in favor of ADR-027 stronger stance at the gate; cycle responded with tiered-architecture revision (hybrid as starting commitment + ADR-027 as structurally pre-committed escalation). 5 audit rounds verified P1-clean. Susceptibility snapshot surfaced 2 Grounding Reframes (GT-1 C6 elevation; GT-2 hybrid-first ordering language drift). Practitioner authorized pursuing both grounding actions in DISCOVER/DECIDE. |
| DISCOVER | ▶ In Progress (paused 2026-05-21 after Spike ζ) | partial — see In-Progress DISCOVER state below | — (gate not yet run) |
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

### From Cycle 7 RESEARCH (closed at gate 2026-05-21)

1. **NL-routing fraction under production tool-rich clients is empirically ~0.** Empirically established across Cycle 6 Spike γ + Cycle 6 PLAY notes 1-25 + Spike λ (qwen3:14b) + Spike λ-paid (paid MiniMax M2.5). ADR-022 amendment effectiveness is bounded to bare-endpoint mode; production tool-rich clients (OpenCode confirmed; Aider/Cursor/Cline inferred) all suppress the amendment. Empirically settled.

2. **OpenAI `tool_choice` mechanism is model-portability-gapped.** The framework implements `tool_choice` correctly (validated under qwen3:14b — Spike λ Cell λ.3 dispatched ensemble + clean NL synthesis). The cross-compatibility-relevant production model (paid MiniMax M2.5 via OpenCode Zen) does NOT honor `tool_choice={"name":"invoke_ensemble"}` under tool-rich conditions (Spike λ-paid Cell λ.3-paid). Three candidate diagnoses (Zen proxy stripping; MiniMax non-conformance; framework tool-list interaction) are unresolved — disambiguation is C2 follow-up.

3. **Tiered architecture is the cycle's DECIDE input.** Hybrid (C3 server-side `tool_choice` interception + C4 framework-driven composition continuation + C5 form-drift enforcement at ensemble-authoring layer + C6 fallback design) as starting commitment, ADR-027 framework-driven dispatch pipeline (orchestrator-LLM removed from dispatch path entirely) as structurally pre-committed escalation triggered by hybrid-effectiveness measurement during BUILD/PLAY. Operational measurement criteria are DECIDE-phase work; the cycle commits to criteria existing.

4. **Orchestrator-LLM is the consistent failure surface across three distinct failure modes.** Composition confabulation (Cycle 6 PLAY note 22 — 8 cache-hit web-searcher dispatches + 0 claim-extractor + fabricated final response); positive control (Spike δ — framework-driven chaining produces faithful citations when orchestrator-LLM is removed); post-dispatch protocol-format failure (Spike λ-paid F-paid-4 — substrate-path file-read attempts in XML or as unreachable client-tool calls). The three modes share the orchestrator-LLM as locus; this is the structural finding motivating the ADR-027 escalation path.

5. **Direct-completion fallback is in tension with the project's value proposition.** Practitioner's RESEARCH-gate stance (2026-05-21): *"llm-orc's stance is that ensembles are used to distribute costs and token to distribute, declarative orchestrations of smaller models. So the orchestrator NEEDs to do that. Otherwise we'd simply use the frontier model."* C6 elevates capability-list discovery to first-order requirement (not documentation work), names structured advisory for Population B, and surfaces operator-observable degradation signaling as a deployment concern.

6. **GROUNDING REFRAME GT-1 (DISCOVER entry):** C6's "first-order requirement" elevation was driven by practitioner stance + one PLAY-constructed stakeholder persona (Skill Orchestration User). Independent product-discovery validation is needed before DECIDE treats the value-misalignment framing as load-bearing. DISCOVER should: (a) treat "capability-list discovery is first-order" as a hypothesis for product-discovery examination, not as inherited commitment; (b) apply Population A / Population B belief-mapping (tool-call-aware OpenAI-family clients vs. developer/script clients); (c) cross-reference Cycle 6 PLAY notes 1/18 (direct-completion quality on string-reverse task) vs. note 7 (code-generator ensemble multi-agent run) for empirical quality contrast.

7. **GROUNDING REFRAME GT-2 (DISCOVER→DECIDE boundary):** C7's "hybrid as starting commitment" ordering is asserted to rest on architectural-continuity cost but the cost differential is not quantified. The Abstract's "structurally pre-committed" language slightly overstates the practitioner's verbatim "if hybrid doesn't work empirically → stronger measures" conditional formulation. At the DISCOVER→DECIDE boundary, produce: (a) explicit build-complexity comparison between Tier 1 hybrid implementation and ADR-027-direct implementation; if costs are within same order of magnitude, ADR-027 as primary recommendation; (b) C2 diagnosis disambiguation BEFORE BUILD (Zen-proxy-specific vs. model-specific vs. framework-interaction) — mechanism design for C3 should be conditional on which diagnosis holds.

8. **Empirical-grounding cluster (four working-inference nodes).** E3.1.1 (server-side `tool_choice` interception preserves client-facing API contract); E4.2.1 (production-client filesystem scope is disjoint from agentic-serving deployment's filesystem); E5.3.3 (framework-driven composition continuation eliminates the orchestrator-narration substitution surface); E6.2.1 (project value-proposition centers on ensemble-distributed orchestration such that direct-completion fallback is a value misalignment). Each needs DECIDE-phase corroboration or explicit residual-uncertainty acceptance before BUILD work proceeds.

9. **Validation-spike-as-research-method precedent extended.** Cycle 6 PLAY ran Spike δ in-loop during PLAY; Cycle 7 RESEARCH extended this to Phase A. Two spikes ran during RESEARCH itself (Spike λ cost-free; Spike λ-paid user-authorized ~$0.05-0.30). The anti-elaboration positioning honored — spikes pruned speculative claims at source rather than allowing them to elaborate downstream into ADRs. Future cycles should treat ADR-087 validation spikes as standard RESEARCH method when reframes or design hypotheses touch the production deployment surface.

10. **Stakeholder model shifts for DISCOVER to surface:**
    - **Tool User / Population A (tool-call-aware client without alternative-surface access)** — receives current direct-completion behavior; under C6 framing this is a degradation surface, not a satisfactory default
    - **Tool User / Population B (developer/script client with alternative-surface access)** — can be redirected to `llm-orc invoke` via structured advisory
    - **Skill Orchestration User (Cycle 6 PLAY stakeholder)** — super-objective "Compose a skill framework against the orchestrator's capability library; expect dispatch when a capability slot fits" was empirically unmet; expectation now mediated by `tool_choice` mechanism + capability-list discovery
    - **Orchestrator LLM (as actor)** — role constrained in Tier 1 (routing potentially intercepted server-side; composition continuation removed); removed from dispatch path entirely in Tier 2 (ADR-027)
    - **Ensemble Author / Operator** — new authoring surfaces: routing-planner ensemble (Tier 1 (i') option (i)); synthesizer ensemble (Tier 2 ADR-027); tool-call-as-output-format authoring (Q2 (b))

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

## In-Progress DISCOVER state (2026-05-21 session pause)

DISCOVER entered 2026-05-21. Substantial product-thinking work completed in conversation; session paused mid-spike-sequence. Resumption picks up at Spike ε.

**Completed in 2026-05-21 session:**

1. **Phase 1: C6 hypothesis examination** — Population A/B mapping confirmed; "degradation surface" framing sharpened to cost-distribution from project-developer perspective (not per-task quality from user perspective); ensemble design declared orthogonal (Cycle 7 ships delegation reliability as prerequisite); north star framed (orchestrator-designs-ensembles long-horizon vision) as context that stays in product-discovery without expanding Essay-Outline scope.

2. **Phase 2 section walk** — Stakeholder Map, Jobs and Mental Models, Value Tensions, Assumption Inversions, Product Vocabulary all walked. Practitioner reactions integrated: Population A primary; Population B important-but-not-Cycle-7-focus; transparent OpenAI-compatible endpoint as project identity ("skills authored without llm-orc knowledge"); latency as tuning concern not structural blocker.

3. **Spike ζ — routing-planner reliability** — completed. 20-prompt battery against qwen3:8b routing-planner ensemble. **100% JSON conformance + 100% schema validity + 90% strict capability-match (100% with defensible judgment).** Latency p50=10s, p90=13s, mean=10s. Mechanism viability confirmed. Writeup at `essays/research-logs/cycle-7-spike-zeta-routing-planner.md`. Spike ensemble + scratch artifacts retained per `feedback_spike_artifact_retention` directive.

**Pending in next DISCOVER session:**

4. **Spike ε** — end-to-end plan→dispatch→synthesize pipeline. Python harness wrapping (Spike ζ planner) + dispatch execution + response-synthesizer ensemble. Run against PLAY note 22's Iceland composition prompt (the historical confabulation case) + Spike δ's web-searcher → claim-extractor chain (the positive-control comparison). Free-tier; ~60-90min agent work.

5. **Spike κ** — C2 diagnosis disambiguation. Three diagnostic probes (Zen direct; MiniMax direct; framework-side payload inspection). Paid κ probes authorized by practitioner 2026-05-21 (~$0.05 estimated). Addresses GT-2(b) from RESEARCH gate.

6. **Section walk integration with spike findings** — Spike ζ findings already lightly anticipated in Section additions; ε and κ findings may revise specific items.

7. **Write updated product-discovery.md** — assemble all Cycle 7 additions with attribution markers per the existing Cycle 4/5/6 update pattern.

8. **Backward-propagation drafts for Essay-Outline §C6, §C7, §C3** — captured during DISCOVER as feed-forward; needs explicit edit draft before DECIDE inherits.

9. **EPISTEMIC GATE (reflection time)** — belief-mapping conversation at DISCOVER → MODEL boundary. Compose against specific DISCOVER artifact content.

10. **Susceptibility snapshot dispatch** — phase-boundary subagent at `docs/agentic-serving/housekeeping/audits/susceptibility-snapshot-cycle-7-discover.md` (per agentic-serving convention).

11. **Gate reflection note** — `.rdd/gates/cycle-7-discover-model-gate.md`. Then phase exit (remove `**In-progress phase:**`; advance Current phase only after manifest check clean).

**Key conversational reframings to preserve (not yet in product-discovery.md):**

- **Cost-distribution framing for "degradation":** the value-misalignment claim is about cost-distribution from the project-developer perspective, not per-task quality from the user perspective. The user is correctly outcome-focused.
- **Ensemble design orthogonality:** Cycle 7 ships delegation reliability; ensemble quality work iterates independently. "If we can effectively delegate to ensembles every time, then we can figure out how to make the ensembles better."
- **Transparent OpenAI-compatible endpoint as project identity:** "The point is to make llm-orc agentic-serving compatible with any tool that would want to use an OpenAI-compatible chat completions endpoint. The user trusts that llm-orc would use / create ensembles effectively. Full stop."
- **North star** (lives in product-discovery, not Essay-Outline): orchestrator-designs-ensembles future; long-horizon agentic coding via skill frameworks; meta-ensembles + ensemble primitives + specialist substrate ensembles. Cycle 7 is the prerequisite, not the destination.
- **Skill Orchestration User sharpening:** client-side skill framework; orchestrator sees only the resulting chat-completion requests; capability matching must work from request content alone with no client-side opt-in.
- **C3 mechanism priority FLIPPED from RESEARCH framing:** routing-planner ensemble (or equivalent NL→capability inference) is PRIMARY (not residual); explicit-naming extractor and tool_choice interception are bonus paths layered onto NL inference.

## Pause Log

| # | Paused | Resumed | Reason |
|---|--------|---------|--------|
| 1 | 2026-05-21 (mid-DISCOVER, after Spike ζ) | — | Practitioner pause after foundational spike; resume with Spike ε in next session |

## Context for Resumption

Anyone resuming this cycle in a new session should read in this order:

1. This file (Cycle 7 cycle-status.md) — current state and framing. **For resumption of paused DISCOVER work**, the "In-Progress DISCOVER state (2026-05-21 session pause)" section above is the load-bearing entry point.
2. `essays/research-logs/cycle-7-spike-zeta-routing-planner.md` — Spike ζ findings (foundational; mechanism viability confirmed). Most consequential DISCOVER-phase finding so far.
3. `essays/essay-outline-006-cross-compatibility-routing-surface.md` — the Cycle 7 RESEARCH Essay-Outline (P1-clean across 5 audit rounds). The structural input to DISCOVER.
3. `essays/reflections/006-cross-compatibility-routing-surface.md` — Cycle 7 RESEARCH reflection (narrative form per ADR-092 §8). Captures the two reframings (Phase A agent-initiated; RESEARCH-gate user-initiated), the validation-spike methodology observation, and the conditional-vs-pre-commitment language-drift finding.
4. `housekeeping/audits/susceptibility-snapshot-cycle-7-research.md` — phase-boundary snapshot with GT-1 (C6 elevation) + GT-2 (hybrid-first ordering language drift) Grounding Reframes the practitioner authorized pursuing in DISCOVER/DECIDE.
5. `essays/research-logs/006-cross-compatibility-routing-surface.md` — Cycle 7 RESEARCH research log (archived). Records Phase A grounding + question-set revision history + Spike λ + Spike λ-paid syntheses + Validation-Spike Decision recording.
6. `essays/research-logs/cycle-7-spike-lambda-tool-choice.md` — Spike λ + Spike λ-paid empirical writeup. The two probe sequences that produced the model-portability finding.
7. `cycle-archive/cycle-6-ensemble-contract-observability-routing.md` — what closed Cycle 6 and what carries forward.
8. `essays/reflections/field-notes.md` Cycle 6 PLAY section (notes 1-25) — the empirical substrate Cycle 7 inherited.
9. `system-design.md` + `roadmap.md` — Cycle 6 close state of the architecture (Cycle 7 ARCHITECT will update substantially).
10. `ORIENTATION.md` — high-level system framing.

When ready to enter DISCOVER, `/rdd-discover` with the Cycle 7 RESEARCH artifacts above as the entry input. The DISCOVER phase should attend specifically to the GT-1 Grounding Reframe (treat C6 "first-order requirement" elevation as a hypothesis for product-discovery examination; apply Population A / Population B belief-mapping) and prepare the GT-2 follow-up (build-complexity comparison + C2 diagnosis disambiguation) as DECIDE-phase work items.
