# ADR-027: Framework-Driven Dispatch Pipeline as Primary Direction for the Chat-Completions Surface

**Status:** Proposed

**Date:** 2026-05-22

---

## Context

The agentic-serving chat-completions surface (`/v1/chat/completions`) currently routes every request through the `OrchestratorRuntime` ReAct loop over the orchestrator-LLM (per ADR-011 — Orchestrator LLM is a Model Profile; per ADR-001 — Internal ReAct loop execution model). The orchestrator-LLM is the routing decider (chooses whether to invoke an ensemble, complete directly, or delegate to a client tool) and the post-dispatch synthesizer (composes the chat-completion response from dispatch results). The system-prompt amendment from ADR-022 is the architectural intervention designed to shift the orchestrator-LLM's NL routing toward `invoke_ensemble` for capability-matched requests. **Codebase finding (per Cycle 7 Tranche 4 conformance scan):** `OrchestratorRuntime` is instantiated only at `src/llm_orc/web/api/v1_chat_completions.py:587`. No other production code path constructs it. The `llm-orc invoke` CLI surface routes through `OrchestraService` directly (`src/llm_orc/cli_commands.py:28`); other REST endpoints (ensembles, artifacts, scripts, profiles) similarly use `OrchestraService` without involving `OrchestratorRuntime`.

Cycle 7 RESEARCH + DISCOVER + MODEL + DECIDE-entry research established that this architecture has a structurally-recurring failure surface on the chat-completions path:

- **C1 — NL-to-ensemble routing fraction is approximately zero under production tool-rich clients.** Spike γ (Cycle 6) + Spike λ + Spike λ-paid (Cycle 7) + Cycle 6 PLAY notes 1-25 all converge: production tool-rich clients (OpenCode confirmed empirically; Aider/Cursor/Cline inferred) suppress the ADR-022 amendment's effectiveness. The orchestrator-LLM under NL framing reliably routes to direct completion or client-tool delegation, not to `invoke_ensemble`. (Essay-Outline 006 §C1 W1.1-W1.2.)
- **C2 — `tool_choice` is not a free baseline.** Spike κ source-code inspection (Cycle 7 DISCOVER 2026-05-21) established D0: `_ChatCompletionsRequest` Pydantic model silently strips `tool_choice` at the request-parsing boundary; no code path in `src/llm_orc/` reads or forwards the parameter. The original Phase A reframe ("the OpenAI `tool_choice` contract already addresses forced routing") is empirically invalidated — implementing `tool_choice` requires new framework code at any layer. (Essay-Outline 006 §C2 W2.1-W2.3.)
- **Composition confabulation is reproducible at the orchestrator-LLM layer.** Cycle 6 PLAY note 22 (8 cache-hit web-searcher dispatches + 0 claim-extractor dispatches + fabricated final response under paid MiniMax M2.5); Spike λ-paid F-paid-4 (substrate-path file-read attempts in XML or as unreachable client-tool calls). The orchestrator-LLM-as-decider is the consistent failure surface across three distinct failure modes.
- **AS-9 codified at MODEL boundary (2026-05-22).** Structurally-bounded LLM roles produce reliable output on single-decision-shaped tasks where the orchestrator-LLM-as-decider failed. Empirically established across four documented confabulation modes + 13 tests at qwen3:8b (Spike ε + Spike ε' + Spike μ). The invariant names the structural property (role-shape, not model-shape) independent of mechanism choice. (Domain-model AS-9.)
- **Tranche 1 DECIDE-entry preconditions cleared (2026-05-22).** OQ #18 Population A voice validation (`cycle-7-oq-18-cost-distribution-validation.md`); OQ #19 build-complexity comparison (`cycle-7-oq-19-build-complexity-comparison.md`) establishing cost-equivalence per GT-2(a) — Tier 1 hybrid ~14 person-days median; ADR-027-direct ~16 person-days median; within ~30% spread; OQ #20 Population A tool-family timeout research (`cycle-7-oq-20-population-a-timeouts.md`). Essay-Outline 006 Amendment A1-A4 propagated; P1-clean argument-audit verified at DECIDE-entry. *Scope-extension note for GT-2(a) (per Essay-Outline §C7 W7.1 E7.1.2; Tranche 2 argument-audit P2-1):* GT-2(a) was originally a paid-spike cost-equivalence rule (don't let same-order-of-magnitude spike cost differentiate the recommendation); OQ #19 applies the same order-of-magnitude logic to sprint-effort estimates. The rule's underlying logic generalizes cleanly (cost equivalence at order-of-magnitude, independent of whether cost is spike-dollars or person-days); the ~30% spread observed for OQ #19 is well within the order-of-magnitude tolerance the original GT-2(a) framing used.

Two candidate mechanisms remained at DISCOVER close — the Tier 1 hybrid (server-side `tool_choice` interception layered onto NL inference + framework-driven composition continuation, preserving the orchestrator-LLM ReAct loop for non-`tool_choice` requests) and ADR-027-direct (the framework-driven `plan → dispatch → synthesize` pipeline, with the orchestrator-LLM removed from the routing-decision and post-dispatch-synthesis surfaces). DISCOVER 2026-05-21 settled ADR-027 as the PRIMARY direction; DECIDE-entry preconditions (Tranche 1) corroborated the direction with the cost-equivalence comparison and the structural-coverage differential.

The candidate has been carried in the corpus as "ADR-027" since Cycle 6 PLAY closeout. This ADR makes the commitment.

---

## Decision

**Adopt the framework-driven dispatch pipeline as the PRIMARY direction for the agentic-serving chat-completions surface.** Every chat-completions request flows through a deterministic three-stage pipeline:

1. **Plan.** The routing-planner ensemble (per ADR-028) reads the chat-completions request content (`messages[]`, `model`, optional `tools[]`) and produces a JSON dispatch plan: `{"action": "dispatch" | "direct", "ensemble": "<name>" | null, "rationale": "..."}`. For multi-step composition, the plan extends to a sequence of dispatch steps (the multi-step composition mechanism is open per OQ #21; default to single-step-planner + framework-chain-heuristic for the initial BUILD).
2. **Dispatch.** The framework executes the plan deterministically. When `action: dispatch`, the framework invokes the named ensemble via the existing `OrchestratorToolDispatch` machinery (per ADR-021 — per-capability dispatch contract; per ADR-024 — common I/O envelope; per ADR-025 — artifact-as-substrate routing). When `action: direct`, the framework falls through to the synthesizer's direct-completion path (per ADR-032). No LLM is in the dispatch-decision loop at this stage.
3. **Synthesize.** The response-synthesizer ensemble (per ADR-029) reads `(ORIGINAL REQUEST + PLAN + DISPATCH RESULTS)` as structured input and produces the user-facing chat-completion response under strict-fidelity rules. The synthesizer is the only LLM that touches the response after dispatch; the orchestrator-LLM does not chain through file-reads of dispatched ensemble substrate paths (per C4 failure mode established in Essay-Outline §C4).

**The `OrchestratorRuntime` ReAct loop's status under ADR-027.** Per the Cycle 7 Tranche 4 conformance scan (Finding 2), `OrchestratorRuntime` currently has no production caller outside the chat-completions handler — the `llm-orc invoke` CLI and other REST endpoints route through `OrchestraService` directly without involving `OrchestratorRuntime`. ADR-001 (Internal ReAct loop execution model) and ADR-011 (Orchestrator LLM is a Model Profile) remain operative as architectural commitments — the ReAct execution model is a documented option that any future llm-orc surface may adopt — but under ADR-027, removing the chat-completions handler's `OrchestratorRuntime` usage leaves the class without an active production caller. **The disposition of `OrchestratorRuntime` after ADR-027 lands is deferred to the Cycle 7 ARCHITECT phase**, where the three candidate dispositions are: (a) **preserve as architecture-for-future-surfaces** — keep the class in the codebase as a documented capability without an active caller, available for any future surface (e.g., a Population B direct-ensemble-HTTP-API endpoint, an autonomous-orchestrator REST surface) that warrants ReAct execution; (b) **wire `llm-orc invoke` to use `OrchestratorRuntime`** — extend the CLI's execution model to include a ReAct-loop path under `OrchestratorRuntime` for use cases where the CLI's current `OrchestraService`-direct path is insufficient (e.g., interactive CLI workflows that benefit from autonomous orchestrator reasoning); (c) **mark for removal as unused code** — file a follow-on `refactor:` commit removing `OrchestratorRuntime` after the BUILD phase ships ADR-027 and the chat-completions handler no longer references the class. ADR-027 does not foreclose any disposition; the ARCHITECT phase's system-design + roadmap work selects.

ADR-001 and ADR-011 are not superseded by ADR-027 — they govern the ReAct execution model as an architectural option that the agentic-serving project may invoke. ADR-027 supersedes ADR-001 and ADR-011 *for the chat-completions surface only*; the ReAct model itself remains a first-class llm-orc execution shape for surfaces that choose it.

### Scope-of-claim partition (load-bearing for downstream artifacts)

The "orchestrator-LLM removed from the dispatch path" framing is precise as direction; empirical coverage and residual bounds are partitioned (per Essay-Outline 006 Amendment A3 tightened by Spike ε' + Spike μ):

**Settled (empirically grounded by Spike ζ + ε + ε' + μ at qwen3:8b):**

- Orchestrator-LLM removal from the routing-decision surface (Spike ζ — 20-prompt battery; 100% JSON conformance + 90% strict capability-match).
- Orchestrator-LLM removal from the post-dispatch synthesis surface on the historical confabulation case (Cycle 6 PLAY note 22) + 1 positive-control chain + numerical-density fidelity + precise-roundable fidelity + multi-turn continuity (Spike ε + Spike ε'; n=10 tests; 0 fabrications).
- Synthesizer's direct-completion path (Rule 5) produces useful responses across 4 distinct request shapes (Spike ε' A1/A2/A3 + C1 Turn 2 fallback).
- Multi-turn continuity preserved when prior turns are included in the synthesizer's input (Spike ε' C1 + C2).
- Generalization across four documented confabulation modes (Spike μ — multi-dispatch fabrication; path hallucination transforms to honest-generic-conventions; substrate-path-as-deliverable structurally avoided by text-only synthesizer surface; coherent factual errors uncalibrated bounded by strict-fidelity Rules 1 + 5).

**Plausible-but-untested (evidence-strength qualifiers, not scope exclusions):**

- Generalization beyond qwen3:8b to other cheap-tier models.
- Production-scale numerical content broader than Spike ε' B1's 25 figures (longer dispatched outputs with hundreds of figures, structured tabular content).
- Cheap-tier reliability for direct-completion-of-factual-questions in training-data-error-prone domains (Spike ε' A2's "Urga / Khovd" factual error is one data point).
- Coherent factual errors uncalibrated on the direct-completion path under adversarial pressure (Spike μ.3 tested the dispatch-driven path).
- Routing-planner reliability under production traffic diversity (OQ #25).

**Open as downstream-phase design questions:**

- Multi-step composition mechanism (OQ #21) — single-step planner + framework chain-heuristic, multi-step planner, planner-loops-with-context. Default to single-step-planner + framework-chain-heuristic for initial BUILD; revisit if production traffic diversity warrants.
- Rule 5 framing requirement scope (OQ #23) — addressed in ADR-029.
- Rule 6 candidate for framework-convention enumeration in direct-completion mode (Spike μ.1) — addressed in ADR-029.
- Rounding/restatement drift mitigation playbook (OQ #24) — addressed in ADR-029.
- Tier escalation policy for direct-completion — addressed in ADR-031 + ADR-032.
- Native `messages[]` handling architecture — mechanical ARCHITECT-phase work per Spike ε' Finding ε'.3.

### Relationship to existing infrastructure

The pipeline reuses the existing in-ensemble infrastructure:

- **Calibration Gate (ADR-007, ADR-014)** operates within each dispatched ensemble; the pipeline does not change calibration semantics. The routing-planner ensemble and response-synthesizer ensemble each carry their own calibration gates within their dispatches.
- **Tier-Escalation Router (ADR-015) + audit dispatch (ADR-018)** operate within each dispatched ensemble; tier escalation continues to fire per-dispatch.
- **Common I/O envelope (ADR-024) + artifact-as-substrate (ADR-025)** govern dispatch outputs feeding into the response-synthesizer's structured input. The synthesizer reads envelope content (`primary` + `artifacts[0]` summary fields) rather than chained file-reads of substrate paths.
- **Dispatch event substrate (Cycle 6 WP-A)** extends with pipeline-stage events (plan-emitted, dispatch-fired, synthesizer-completed) for observability.
- **Session Registry (ADR-013) + budget enforcement (ADR-005, AS-3) + autonomy levels (ADR-008)** apply to the pipeline as session-level control plane concerns; the pipeline does not bypass these.

### Relationship to AS-9 and AS-10

- **AS-9 (structurally-bounded LLM roles)** is the constitutional invariant the pipeline satisfies on the chat-completions surface. The routing-planner ensemble and response-synthesizer ensemble are the structurally-bounded roles; the orchestrator-LLM-as-decider is removed from this surface (the role remains available as an architectural option for future surfaces per ADR-001 + ADR-011).
- **AS-10 (capability matching from request content alone — per ADR-026)** is the constitutional invariant the pipeline operates within on the chat-completions surface. The routing-planner ensemble's input is the chat-completions request content + the framework's capability list; no client-side opt-in mechanism is required.

### Relationship to ADR-022 (Routing surface behavior)

ADR-022's system-prompt amendment was the prior intervention designed to steer the orchestrator-LLM toward `invoke_ensemble` for capability-matched NL requests. Under ADR-027, the orchestrator-LLM is no longer the routing decider on the chat-completions surface; the amendment is structurally moot for this surface.

ADR-022 carries a dated `> Updated by ADR-027 on 2026-05-22.` partial-update header recording that the chat-completions surface is no longer governed by the amendment. The amendment remains operative for any future surface that adopts `OrchestratorRuntime` (per ADR-001 + ADR-011's continuing architectural commitment to the ReAct execution model). Per the Cycle 7 Tranche 4 conformance scan (Finding 2), `OrchestratorRuntime` currently has no production caller other than the chat-completions handler being replaced by ADR-027 — so the amendment has no live codebase surface until ARCHITECT selects disposition (a) preserve-for-future or (b) wire-the-CLI-to-use-it. Under disposition (c) (remove as unused), the amendment becomes dormant code preserved in version history per ADR-022's body-immutable record. The body of ADR-022 is preserved; the scope-narrowing and codebase-state clarification are the only changes.

### Relationship to ADR-021 (Per-capability dispatch contract)

ADR-021's per-capability dispatch contract (one capability sub-task per request; client-side workflow state; fresh-context property) is preserved structurally. The actor producing the routing decision shifts from the orchestrator-LLM (current) to the routing-planner ensemble (under ADR-027); the event shape — a single `invoke_ensemble` dispatch per capability sub-task — is unchanged. The two supported dispatch shapes ADR-021 names (explicit ensemble naming; natural-language prompt) are now both routed through the routing-planner ensemble — explicit names are recognized as an explicit-naming intent signal the planner honors, and NL prompts are routed by the planner's capability-match decision.

ADR-021 carries a dated `> Updated by ADR-027 on 2026-05-22.` partial-update header recording the actor shift. The body is preserved; the contract's structural commitments remain.

---

## Rejected alternatives

### Tier 1 hybrid as PRIMARY (server-side `tool_choice` interception layered onto NL inference + framework-driven composition continuation)

The chat-completions handler intercepts requests with explicit `tool_choice={"name":"<ensemble>"}` and routes them through a deterministic dispatch path bypassing the orchestrator-LLM; non-intercepted requests continue through the `OrchestratorRuntime` ReAct loop with the ADR-022 system-prompt amendment active. Framework-driven composition continuation surfaces dispatch envelope content as the chat-completion response directly (rather than `finish_reason: tool_calls` requiring the orchestrator-LLM to chain).

**Rejected because:** the build-complexity comparison (OQ #19) established cost-equivalence per GT-2(a) — ~14 person-days median for Tier 1 vs. ~16 person-days median for ADR-027-direct, within ~30% spread. The "free baseline" assumption that motivated Tier 1's ordering at RESEARCH close was refuted by Spike κ D0; `tool_choice` requires new framework code regardless of approach. With cost-equivalence established, three structural factors favor ADR-027:

1. **AS-9 satisfaction is universal under ADR-027, partial under Tier 1.** Tier 1's structural-bounding applies only to requests carrying explicit `tool_choice={"name":"<ensemble>"}`; non-`tool_choice` NL requests continue to hit the orchestrator-LLM-as-decider failure surface that C1 establishes (NL-to-ensemble routing fraction ≈ zero under production tool-rich clients). ADR-027 routes every chat-completions request through structurally-bounded roles.
2. **NL-routing-fraction reduction is universal under ADR-027.** Tier 1 leaves NL routing in the C1 failure space for non-`tool_choice` requests; ADR-027 routes every NL request through the validated routing-planner ensemble.
3. **The confabulation-mode mitigation generalizes universally under ADR-027.** Spike μ established that the structural-bounding finding holds across four documented confabulation modes. Tier 1 mitigates only the modes that arise from the explicit-`tool_choice` interception path; ADR-027 mitigates them universally on the chat-completions surface.

The hybrid remains viable as a **conditional extension** of ADR-027 (server-side `tool_choice` interception layered on top of the routing-planner ensemble, for operator-deployment shapes where some client population sends explicit `tool_choice` shapes). The conditional extension is meaningful only under the "implement `tool_choice` handling" disposition (per ADR-030 — `tool_choice` disposition). ADR-027 does not foreclose the hybrid extension; it positions the extension as orthogonal layering, not as the cycle's primary mechanism.

### Preserve the `OrchestratorRuntime` ReAct loop on the chat-completions surface with stronger system-prompt amendments

The chat-completions surface continues to route through the orchestrator-LLM with iteratively stronger system-prompt amendments (per ADR-022's "amend the prompt's commitment" pattern); ADR-027 is not built.

**Rejected because:** Cycle 6 PLAY note 13 documented orchestrator self-modeling reliability — orchestrators name routing defects accurately ("bad routing on my part") but predict fix effectiveness optimistically. ADR-022's amendment was tested empirically and confirmed bounded to bare-endpoint mode under production tool-rich clients (per Spike γ Cycle 6 + PLAY note 18). Stronger system-prompt amendments without architectural change carry the same risk: the prompt's commitment is the design surface; the orchestrator-LLM's reasoning shape under tool-rich client conditions is the empirical surface; the two have not converged through prompt-amendment work.

The recurring orchestrator-LLM failure surface across three distinct failure modes (composition confabulation per PLAY note 22; positive control via Spike δ when the orchestrator-LLM is removed; post-dispatch protocol-format failure per Spike λ-paid F-paid-4) is structural, not prompt-tunable. AS-9's codification at MODEL boundary names this structural property; system-prompt amendments operate within the structure AS-9 deems failure-prone for these task surfaces, while ADR-027 shifts the structure.

### Frontier-tier orchestrator-LLM as the routing-decision substrate

The chat-completions surface routes through `OrchestratorRuntime` with the orchestrator-LLM at frontier tier (Claude Opus 4.5, GPT-5, or comparable), exploiting the assumption that "the more capable the model, the better the routing decisions."

**Rejected because:** the project's value proposition (per Essay-Outline §C6 + cost-distribution lens; per OQ #18 validation) is that ensembles distribute load across cheaper-tier models — the chat-completions surface using frontier-tier orchestration for every request defeats the cost-distribution architecture. Population A clients at deployment scale would bear frontier-tier cost for every chat-completion request; the framework would be using its most expensive resource for a single-decision task structurally bounded enough that a cheap-tier ensemble handles it reliably (per AS-9 + Spike ζ).

The Khanal et al. MOP finding (open-source-models cohort frontier models exhibit highest meltdown rates on long-horizon tasks — Essay 005 reference) also weakens the assumption that frontier-tier orchestrators are reliably better routers; on the routing-decision task as Spike ζ tested it, the cheap-tier qwen3:8b routing-planner ensemble achieved 100% JSON conformance + 90% strict capability-match. Frontier-tier substitution adds cost without empirically validated reliability gain on this task surface.

### Defer ADR-027 to a later cycle; ship Tier 1 hybrid as Cycle 7's primary

The Tier 1 hybrid ships in Cycle 7 BUILD; ADR-027 is named as a future-cycle escalation path conditional on Tier 1 falling short of operational criteria.

**Rejected because:** this was the Cycle 7 RESEARCH-close framing the cycle later revised. The deferral creates a known-deferred architectural debt — the cycle has empirical evidence (Spike ζ + ε + μ; AS-9 codification) that ADR-027 is the structurally-sound direction on the cost-equivalence calculus, and deferring it locks the project into building a layered hybrid whose architectural cost the cycle would have to pay twice (once now to ship the hybrid; once later to ship ADR-027 and migrate). The cost-equivalence comparison per OQ #19 shows the migration cost is comparable to the original Tier 1 build cost; building Tier 1 first then ADR-027 later is approximately 2× the cost of building ADR-027 directly.

The deferral also creates an asymmetric-grounding risk — the cycle would be shipping the architecturally weaker option (Tier 1 mitigates AS-9 partially) while having empirical evidence for the stronger option (ADR-027 mitigates AS-9 universally). Per Invariant 8's structural-correctness principle, the cycle's commitment should align with the strongest evidence-direction available at DECIDE; if BUILD evidence surfaces ADR-027 limitations, the cycle is positioned to revise from a stronger architectural baseline.

---

## Consequences

### Positive

- **AS-9 is satisfied universally on the chat-completions surface.** Every chat-completions request flows through structurally-bounded roles (routing-planner + response-synthesizer). The orchestrator-LLM-as-decider failure surface that C1 + PLAY note 22 + Spike λ-paid F-paid-4 documented is removed from this surface entirely.
- **The NL-to-ensemble routing fraction can rise above the C1 baseline of approximately zero.** Spike ζ established that the routing-planner ensemble at qwen3:8b achieves 90% strict capability-match across the 20-prompt battery; production traffic will produce empirical evidence on the population-of-requests-the-pipeline-routes-to-ensemble metric. The C1 baseline is a structural floor the architecture has been engineered to leave.
- **The cost-distribution accountability sub-promise (per OQ #18 split; per ADR-032) is delivered by strict-dispatch-when-capability-matched.** The routing-planner ensemble dispatches when a capability match exists; direct-completion fallback fires only on no-capability-match requests. The project-developer-lens expectation that ensembles handle capability-matched work is encoded as the pipeline's default behavior, not as an aspiration the orchestrator-LLM intermittently honors.
- **The configuration honesty sub-promise (per OQ #18 split) is delivered structurally** — the response synthesizer's strict-fidelity Rule 5 (per ADR-029) requires honest direct-completion framing when DISPATCH RESULTS is empty. Population A's degradation signal (configuration dishonesty per Cline #10551 + OpenCode #20859) is structurally prevented; the synthesizer cannot silently disguise direct-completion as ensemble-dispatched.
- **The post-dispatch composition surface is structurally protected.** The orchestrator-LLM's emergent "chain through file-read of substrate path" failure mode (Spike λ-paid F-paid-4; PLAY note 22) is removed because the synthesizer's input is structured (`REQUEST + PLAN + DISPATCH RESULTS`), not a `finish_reason: tool_calls` requiring the orchestrator-LLM to issue further tool calls.
- **The pipeline's architectural surface is simpler than Tier 1's would have been.** A single mechanism (plan → dispatch → synthesize) governs the chat-completions surface; Tier 1 would have layered a new `tool_choice` interception path on top of the preserved orchestrator-LLM ReAct loop, growing the surface.

### Negative

- **The pipeline's behavior change is substantial on the chat-completions surface.** Every chat-completions request now flows through the new path; bugs in any of the three stages affect every request. Migration risk is concentrated on this surface.
- **Cheap-tier reliability beyond qwen3:8b is plausible-but-untested.** The empirical floor (n=13 tests + 20-prompt routing battery + 4 confabulation modes) is qwen3:8b-grounded. Operators deploying with other cheap-tier models (Llama-3.1-8B, Mistral-7B, Phi-3.5-mini, etc.) operate within AS-9's structural property but without the model-specific empirical floor. BUILD-phase work + PLAY-phase experiential discovery is required before declaring the pipeline production-ready across the cheap-tier model space.
- **Multi-step composition is an open design question (OQ #21).** The initial BUILD defaults to single-step-planner + framework-chain-heuristic per Spike δ's pattern; production traffic diversity may surface composition shapes the heuristic does not handle. The risk depends on the population of multi-step composition requests Population A clients produce; characterization is BUILD/PLAY work.
- **Latency floor is ~36s single-step / ~64s chained at qwen3:8b (Spike ε).** The latency tuning playbook is DECIDE-phase work per ADR-031; integration with OQ #20's Population A timeout findings is load-bearing. Cline (30s hard default) breaches the single-step floor by 6s every request; Cursor (4-20s reported ceilings on agentic paths) is structurally outside scope; OpenCode and Aider accommodate the floor with permissive defaults.
- **The routing-planner ensemble becomes a new single point of failure on the chat-completions surface.** If the routing-planner ensemble fails (model unavailable, schema-non-conformance, infrastructure error), the chat-completions surface degrades. The framework's existing infrastructure (calibration gate; tier-escalation router; budget controller; autonomy policy) operates within dispatched ensembles, so a routing-planner failure has the same recovery surface as any other ensemble failure — but the routing-planner is invoked on every request, raising its operational profile relative to capability ensembles.
- **The `OrchestratorRuntime` ReAct loop's disposition after ADR-027 is ARCHITECT-phase work.** Per the Tranche 4 conformance scan Finding 2, `OrchestratorRuntime` has no production caller outside the chat-completions handler; after ADR-027 lands, the class becomes unused production code unless ARCHITECT selects disposition (a) preserve-for-future or (b) wire the CLI to use it. If ARCHITECT selects (c) remove-as-unused, a follow-on `refactor:` commit cleans up the class after BUILD ships the pipeline. ADR-001 + ADR-011's ReAct execution model remains an architectural commitment as an available shape regardless of `OrchestratorRuntime`'s codebase presence; the model is not deprecated. The framework will potentially maintain two execution models long-term — the framework-driven pipeline (chat-completions) and the ReAct loop (future surfaces that adopt it) — but the latter has no current caller; the cost is "architectural-option preservation" rather than "actively-maintained dual surfaces."

### Neutral

- **The pipeline preserves the OpenAI chat-completions API contract.** External clients (Population A) see no change in protocol surface — the request and response shapes remain OpenAI-compatible. The architectural change is entirely internal to the framework.
- **The Tier 1 hybrid remains available as a conditional extension** (per ADR-030 `tool_choice` disposition). The cycle does not foreclose layering server-side `tool_choice` interception on top of the routing-planner ensemble for operator-deployment shapes that warrant it; the conditional extension is positioned as orthogonal to ADR-027, not as a competing primary.
- **Framing-audit F2 (per Tranche 2)** — the "PRIMARY direction" framing carries operator-foreclosure risk for client populations with `tool_choice`-aware behavior. ADR-027 names the hybrid extension as orthogonal (not "conditional alternative" or "secondary"); ADR-030's disposition is the proper home for the operator-foreclosure deliberation. The framing softening lands in ADR-030.

## Provenance check

- **C1 NL-to-ensemble routing fraction ≈ zero**: Essay-Outline 006 §C1 (driver). Driver chain: prior research log derived (Spike γ + Spike λ + Spike λ-paid + PLAY notes 1-25; documented in essay).
- **C2 Spike κ D0 finding (framework strips `tool_choice` at input)**: Spike κ research log (driver) + Essay-Outline 006 §C2 (driver). Driver chain: same-cycle spike + essay.
- **AS-9 structural-property invariant**: domain-model §AS-9 (driver). Driver chain: MODEL-phase codification 2026-05-22; established by Spike ε + Spike ε' + Spike μ; recorded in domain-model Amendment Log entry #13.
- **OQ #19 build-complexity comparison establishing cost-equivalence per GT-2(a)**: Tranche 1 research note `cycle-7-oq-19-build-complexity-comparison.md` (driver). Driver chain: same-cycle DECIDE-entry research; estimates source-code-inspection-derived.
- **OQ #18 cost-distribution lens / configuration-honesty split**: Tranche 1 research note `cycle-7-oq-18-cost-distribution-validation.md` (driver). Driver chain: same-cycle DECIDE-entry research; Population A voice corroborates configuration honesty (Cline #10551 + OpenCode #20859) + silent on cost-distribution accountability (project-developer-lens grounded).
- **OQ #20 Population A tool-family timeout research**: Tranche 1 research note `cycle-7-oq-20-population-a-timeouts.md` (driver). Driver chain: same-cycle DECIDE-entry research.
- **Scope-of-claim partition (settled / plausible-but-untested / open)**: Essay-Outline 006 Amendment A3 (driver, tightened by Spike ε' + Spike μ). Driver chain: DISCOVER 2026-05-21 settled with practitioner verbatim scope-of-claim challenge; MODEL 2026-05-22 codified AS-9 within the partition.
- **Relationship to ADR-022 (chat-completions surface no longer governed by amendment)**: drafting-time synthesis bridging ADR-022's mechanism to the ADR-027 architectural change. The ADR-022 amendment remains operative for `OrchestratorRuntime` surfaces *as an architectural commitment* — the amendment text is preserved in the class for any future surface that adopts the ReAct execution model. The partial-update header on ADR-022 (separate task) records the scope narrowing. **Note (Tranche 4 conformance scan Finding 2):** if ARCHITECT selects the (c) remove-as-unused disposition for `OrchestratorRuntime`, ADR-022's amendment becomes dormant code — preserved in version history via ADR-022's body-immutable record but not present in the live codebase. Re-introduction would require either ARCHITECT selecting disposition (a) or (b), or a future surface explicitly adopting the ReAct model and re-incorporating the amendment.
- **`OrchestratorRuntime` codebase-disposition deferred to ARCHITECT (Tranche 4 Finding 2)**: Cycle 7 Tranche 4 conformance scan `housekeeping/audits/conformance-scan-cycle-7-decide.md` (driver, Finding 2 with explicit ARCHITECT-phase deferral recommendation) + drafting-time synthesis (the three candidate dispositions a/b/c). Driver chain: same-cycle conformance scan + drafting-time enumeration. The disposition decision is ARCHITECT-phase work; ADR-027 names the candidates and the trigger (BUILD's removal of the chat-completions handler's `OrchestratorRuntime` usage).
- **Relationship to ADR-021 (actor shift from orchestrator-LLM to routing-planner ensemble)**: ADR-021 (driver) + domain-model AS-9 §Propagation note (driver). Driver chain: prior-ADR-derived + invariant-derived. The partial-update header on ADR-021 (separate task) records the actor shift.
- **Rejected alternative — Tier 1 hybrid as primary**: OQ #19 build-complexity comparison (driver) + structural-coverage differential (drafting-time synthesis enumerating AS-9 satisfaction differential, NL-routing-fraction reduction differential, confabulation-mode mitigation differential across the two mechanisms). Driver chain: same-cycle research + drafting-time analytical engagement.
- **Rejected alternative — preserve `OrchestratorRuntime` with stronger amendments**: ADR-022 (driver, with empirical bounds named in §Decision §"Effectiveness is configuration-conditional") + Cycle 6 PLAY note 13 (driver, orchestrator self-modeling reliability). Driver chain: prior-ADR-derived + prior-cycle PLAY-derived.
- **Rejected alternative — frontier-tier orchestrator-LLM**: Essay-Outline 006 §C6 cost-distribution lens (driver) + OQ #18 Population A voice validation (driver) + Khanal et al. MOP finding via essay 005 (driver). Driver chain: same-cycle essay-derived + same-cycle research-derived + prior-essay-cited literature.
- **Rejected alternative — defer ADR-027 to later cycle**: OQ #19 cost-equivalence + cost-of-double-build analysis (drafting-time synthesis applying the build-complexity comparison to the migration scenario). Driver chain: same-cycle research + drafting-time analytical engagement.
- **Framing-audit F2 carry-forward to ADR-030**: Tranche 2 framing audit (driver). Driver chain: same-cycle Tranche 2 audit; the carry-forward is drafting-time positioning of where the framing softening lands.
