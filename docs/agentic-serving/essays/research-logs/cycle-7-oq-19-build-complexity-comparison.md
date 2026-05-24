# Cycle 7 OQ #19 — Build-Complexity Comparison: Tier 1 Hybrid vs. ADR-027-direct

**Date:** 2026-05-22
**Context:** Cycle 7 DECIDE precondition (DISCOVER snapshot Advisory 2; MODEL snapshot preserved active). Per GT-2(a) cost-equivalence rule: if costs are within same order of magnitude, ADR-027 as primary recommendation.
**Method:** Source-code inspection of current chat-completions code path + architectural reasoning against Cycle 7 spike findings. Estimates are sprint-level magnitude, not point-precision.

---

## Summary

Both candidate mechanisms require comparable new framework code; the "free baseline" assumption for `tool_choice` is refuted by Spike κ (D0 — Pydantic strips at input; no code path in `src/llm_orc/` reads or forwards the parameter). The build-complexity comparison resolves to ~9-14 person-days for Tier 1 hybrid and ~13-19 person-days for ADR-027-direct — within the same order of magnitude (~30-50% spread) per GT-2(a). The structural advantage carried by AS-9 (orchestrator-LLM removed from routing-decision and post-dispatch-synthesis surfaces; ~95% confabulation dissolution at qwen3:8b per Spike ε + μ; 4 documented confabulation modes generalized) tips the recommendation to ADR-027-direct as PRIMARY direction.

**One observation worth surfacing:** the Tier 1 hybrid's "framework-driven composition continuation" component (Section 5 below) is structurally near-identical to ADR-027's `synthesize` step applied conditionally. The cost premium of Tier 1 over ADR-027 mostly comes from layering the new mechanisms ON TOP OF the existing orchestrator-LLM ReAct loop rather than replacing it for the chat-completions surface. ADR-027 simplifies the architectural surface; Tier 1 grows it.

---

## Current chat-completions code path (anchor for the comparison)

Source-code inspection 2026-05-22:

| Module | LoC | Role |
|--------|-----|------|
| `web/api/v1_chat_completions.py` | 801 | HTTP handler; `_ChatCompletionsRequest` Pydantic model; entry to runtime |
| `agentic/orchestrator_runtime.py` | 742 | `OrchestratorRuntime` — ReAct loop over the orchestrator-LLM |
| `agentic/orchestrator_tool_dispatch.py` | 1887 | Tool-dispatch implementation (`invoke_ensemble`, `list_ensembles`, `query_knowledge_graph`, `submit_feedback`, etc.) |
| `agentic/tier_router.py` | 409 | In-ensemble tier-escalation logic |
| `agentic/dispatch_envelope.py` | 160 | Envelope contract (`primary`, `artifacts[0]`, `diagnostics`) |
| `agentic/dispatch_event_substrate.py` | 209 | Typed event surface (Cycle 6 WP-A) |
| `agentic/calibration_gate.py` | (~) | In-ensemble Calibration Gate (ADR-007, -014) |

Key observations from the source:

- **`_ChatCompletionsRequest` does not declare a `tool_choice` field** (lines 401-408 of `v1_chat_completions.py`). Pydantic's default `extra="ignore"` discards the parameter at parsing — confirms Spike κ D0.
- **The handler delegates to `OrchestratorRuntime` immediately** (line 430: `runtime = await _build_runtime(...)`). Every chat-completions request goes through the same ReAct-loop path regardless of intent.
- **`OrchestratorRuntime` is a single ReAct-loop class** (line 204; `run()` method at line 255) that drives the orchestrator-LLM through `generate_with_tools` calls with the framework's tool surface (`TOOL_NAMES` per ADR-003).
- **Framework-driven composition continuation does NOT exist today.** The orchestrator-LLM is the chaining agent; the framework is the dispatch executor and event substrate. ADR-025 substrate routing produces the substrate artifact, but the orchestrator-LLM is what composes the final chat-completion response from dispatch results — and per Spike λ-paid F-paid-4 + PLAY note 22, that composition fails in production.

Both candidate mechanisms thus start from the same code baseline: no `tool_choice` handling at the framework boundary; no framework-driven composition continuation; one ReAct-loop entry point.

---

## Tier 1 Hybrid — work breakdown

**Definition (from Essay-Outline Amendment A3 §Tier 2 reframe):** server-side `tool_choice` interception layered onto NL inference, paired with framework-driven composition continuation, with capability-list discovery + Q2 enforcement + Q3 fallback advisory as additional components. The hybrid is meaningful only under "implement `tool_choice` handling" disposition (Tension 19).

### Work items

1. **`tool_choice` at the request boundary** (1-2 days)
   - Add `tool_choice` field to `_ChatCompletionsRequest` (Pydantic schema change; ~10 lines).
   - Add typed parsing for the three OpenAI shapes (`"auto"`, `"required"`, `{"type":"function","function":{"name":"X"}}`).
   - Tests: unit tests for parsing + integration test for request acceptance.

2. **Ensemble-name resolver for explicit-naming intent** (1 day)
   - When `tool_choice={"name":"<ensemble>"}`, route to deterministic dispatch path bypassing the orchestrator-LLM.
   - Resolver reads the framework's loaded-ensemble registry; rejects unknown names; emits typed observability event.
   - Tests: resolver-level unit tests + integration test for explicit-naming path.

3. **Server-side dispatch path** (2-3 days)
   - New code path in `v1_chat_completions.py` that, on `tool_choice` interception, calls the dispatch executor directly (bypasses `OrchestratorRuntime`).
   - Reuses existing `OrchestratorToolDispatch.dispatch()` machinery (already exists; ADR-021).
   - Tests: integration tests for explicit-naming dispatch end-to-end.

4. **Framework-driven composition continuation** (3-4 days) — **load-bearing**
   - When dispatch fires (by any mechanism), surface the dispatch envelope's `primary` content as the chat-completion response directly. Do NOT return `finish_reason: tool_calls` requiring the orchestrator-LLM to chain.
   - Two implementation options:
     - (a) Lightweight: return envelope `primary` verbatim with minimal framing.
     - (b) Structured: pass envelope content + original request to a single-turn synthesizer step (response-synthesizer ensemble shape — but applied conditionally).
   - Option (b) is structurally equivalent to ADR-027's `synthesize` step, applied only when dispatch fires.
   - Option (a) breaks if envelope `primary` is artifact-reference-shaped rather than answer-shaped (ADR-025 substrate routing). Option (b) handles this case.
   - Tests: integration tests for envelope→response surface across substrate-routed and inline-response ensembles.

5. **Capability-list discovery endpoint** (1-2 days)
   - New endpoint `/v1/ensembles` or extension to `/v1/models` advertising available ensembles.
   - Returns capability identifiers usable in `tool_choice={"name":"X"}` shapes.
   - Tests: endpoint contract tests + capability-list shape tests.

6. **Q2 enforcement** (2-3 days)
   - Tool-call-as-output-format authoring conventions documented in field guide + new ensemble templates (no framework change).
   - Calibration Gate Reflect retrofit for existing form drift (`claim-extractor`): integrate Reflect verdict-driven retry on schema-non-conformance.
   - Tests: ensemble-conformance integration tests for the new authoring pattern + retrofit test on `claim-extractor`.

7. **Q3 fallback advisory** (1-2 days)
   - When request reaches the orchestrator-LLM ReAct path (no `tool_choice` interception; no capability match), modify the response composition to embed structured advisory for Population B (suggest `llm-orc invoke` for matched ensembles).
   - Operator-observable degradation signal: emit a typed event when fallback-to-direct-completion fires (extend `dispatch_event_substrate.py`).
   - Tests: response-shape tests for advisory; event-emission tests.

8. **Migration / regression tests** (1-2 days)
   - Ensure non-intercepted requests still flow through `OrchestratorRuntime` correctly.
   - Add regression tests against historical confabulation patterns to verify mitigation.

**Tier 1 estimate: 11-19 person-days (median ~14).**

### Tier 1 risk surface

- **The orchestrator-LLM ReAct loop is preserved for non-intercepted requests.** Production tool-rich clients (Population A) sending NL requests without `tool_choice` continue to hit the orchestrator-LLM-as-decider failure surface. The hybrid mitigates only the explicit-`tool_choice` path; NL routing remains in the failure space established by C1 (NL-to-ensemble routing fraction ≈ zero under production tool-rich clients).
- **`tool_choice` is not honored by the cross-compatibility-relevant production model** (Spike λ-paid C2). Even if implemented server-side, the population that benefits is "clients that send `tool_choice` AND the framework intercepts." This intersection may be small in practice if Population A clients do not construct `tool_choice` themselves.
- **Architectural surface grows.** New mechanism + preserved ReAct loop + structured advisory + capability-list endpoint live alongside each other. Each adds independent maintenance surface.

---

## ADR-027-direct — work breakdown

**Definition (from Essay-Outline Amendment A3 §Tier 1 reframe):** a deterministic `plan → dispatch → synthesize` pipeline where the orchestrator-LLM is removed from the routing-decision and post-dispatch-synthesis surfaces. Routing-planner ensemble produces JSON plan; framework dispatches; response-synthesizer ensemble produces user-facing response from `(REQUEST + PLAN + DISPATCH RESULTS)` under strict-fidelity rules.

### Work items

1. **Routing-planner ensemble (hardening from Spike ζ)** (1 day)
   - Validated at qwen3:8b — 100% JSON conformance + 90% strict capability-match across 20-prompt battery (cycle-7-spike-zeta).
   - Promote from spike YAML to production ensemble; add to default ensemble library.
   - Tests: existing 20-prompt battery as regression suite; add edge-case prompts.

2. **Response-synthesizer ensemble (hardening from Spike ε)** (1-2 days)
   - Scratch artifact already exists: `.llm-orc/ensembles/spike-cycle7-epsilon-response-synthesizer.yaml`.
   - Validated at qwen3:8b — 0 fabrications on PLAY-note-22 confabulation case + Spike δ positive control + numerical-density fidelity (cycle-7-spike-epsilon, cycle-7-spike-epsilon-prime, cycle-7-spike-mu).
   - Promote to production ensemble; refine Rule 5 framing scope (open DECIDE question OQ #23) and Rule 6 framework-convention enumeration (open DECIDE question; Spike μ.1).
   - Tests: existing pipeline tests as regression suite.

3. **Plan-dispatch-synthesize pipeline module** (4-5 days) — **load-bearing**
   - New module (likely `agentic/dispatch_pipeline.py` or similar) orchestrating: routing-planner invoke → framework dispatch loop → response-synthesizer invoke.
   - Framework chaining pattern from Spike δ (`resolve_input(step.input, results)`) for multi-step dispatch.
   - Integration with existing infrastructure: tier_router (in-ensemble; unchanged), calibration_gate (in-ensemble; unchanged), dispatch_envelope (unchanged), dispatch_event_substrate (extends with pipeline-stage events), autonomy_policy, budget_controller, session_registry.
   - Multi-step composition: single-step planner + framework chain-heuristic (Spike ε.6 design question OQ #21 — DECIDE resolves).
   - Tests: integration tests for the full pipeline across single-step + multi-step + direct-completion paths.

4. **Chat-completions handler routing** (1-2 days)
   - Replace ReAct-loop entry in `v1_chat_completions.py` chat_completions handler with pipeline entry.
   - Preserve `OrchestratorRuntime` for non-chat-completions surfaces (orchestrator CLI per `llm-orc invoke`, future surfaces).
   - Native `messages[]` handling in synthesizer input (Spike ε' Finding ε'.3 — mechanical ARCHITECT-phase work; serialize prior turns into synthesizer input).
   - Tests: integration tests for handler routing; preserve existing tests for the ReAct-loop on `llm-orc invoke`.

5. **Direct-completion path** (1 day)
   - Routing-planner emits `action: direct` when no capability match; synthesizer's direct-completion mode handles the response (Rule 5 + Rule 6 — DECIDE rule-set design).
   - Tests: integration tests for direct-completion path across the 4 request shapes Spike ε' tested.

6. **Capability-list discovery endpoint** (1-2 days)
   - Same as Tier 1 (item 5) — independent of mechanism choice; both approaches need it.

7. **Q2 enforcement** (2-3 days)
   - Same as Tier 1 (item 6) — independent of mechanism choice.

8. **Migration / regression tests for preserved ReAct surfaces** (2-3 days)
   - Ensure `OrchestratorRuntime` continues to work for non-chat-completions surfaces.
   - Regression tests against historical confabulation patterns (PLAY note 22; substrate-path failure modes) under the new pipeline.

**ADR-027-direct estimate: 13-19 person-days (median ~16).**

### ADR-027-direct risk surface

- **The pipeline replaces the chat-completions surface's ReAct loop.** Behavioral change is substantial — every chat-completions request now flows through plan → dispatch → synthesize. Bugs in any of the three stages affect every request.
- **Cheap-tier reliability is empirically established at qwen3:8b across 13 tests / 4 confabulation modes** (Spike ε + ε' + μ). Generalization beyond qwen3:8b is plausible-but-untested (AS-9 §Plausible-but-untested).
- **Multi-step composition is an open design question (OQ #21).** Single-step planner + framework chain-heuristic, multi-step planner, planner-loops-with-context — DECIDE resolves. Risk depends on choice; the simplest option (single-step planner + framework chain-heuristic) reuses Spike δ's validated pattern.
- **Latency floor is ~36s single-step / ~64s chained at qwen3:8b** (Spike ε). Tuning axes named in Essay-Outline Amendment A3; integration with OQ #20 timeout research is load-bearing for the latency ADR.

---

## Comparison summary

| Dimension | Tier 1 Hybrid | ADR-027-direct |
|-----------|---------------|----------------|
| **Estimate (person-days, median)** | ~14 | ~16 |
| **Range** | 11-19 | 13-19 |
| **Within order of magnitude per GT-2(a)?** | yes | yes |
| **New ensembles** | 0 (extends existing dispatch) | 2 (routing-planner + response-synthesizer) |
| **New core modules** | 1 (composition-continuation; conditional pipeline) | 1 (plan-dispatch-synthesize pipeline) |
| **Touches existing modules** | `v1_chat_completions.py`, `dispatch_envelope.py`, `dispatch_event_substrate.py` | `v1_chat_completions.py`, `dispatch_event_substrate.py` |
| **Preserves `OrchestratorRuntime` for chat-completions surface?** | yes | no — replaced for chat-completions; preserved for other surfaces |
| **AS-9 satisfaction surface** | partial (explicit-`tool_choice` requests only) | full (all chat-completions requests) |
| **NL-routing-fraction reduction (C1 baseline ≈ 0)** | limited to clients that send `tool_choice` AND the framework intercepts | universal — every NL request flows through the routing-planner |
| **Confabulation-mode mitigation (4 modes per Spike μ)** | partial (only when dispatch fires AND composition continuation engages) | universal (orchestrator-LLM removed from the failure surfaces) |
| **Architectural surface area** | grows (new mechanism layered on existing ReAct loop) | simpler (pipeline replaces ReAct loop for chat-completions) |
| **Existing infrastructure reused** | dispatch envelope, calibration gate, tier router, autonomy policy, budget controller, session registry, dispatch event substrate | same as Tier 1 — both reuse the in-ensemble infrastructure |

### Per GT-2(a) cost-equivalence rule

Costs are within the same order of magnitude (~14 vs. ~16 median person-days; spread ~30%). The "free baseline" assumption for Tier 1 is refuted by Spike κ D0 — `tool_choice` requires new code regardless of approach. The cost-equivalence rule triggers: **ADR-027-direct as PRIMARY recommendation.**

### Structural advantage tipping factor

Beyond cost-equivalence, three structural factors favor ADR-027-direct:

1. **AS-9 is satisfied universally, not partially.** Tier 1's structural-bounding applies only to explicit-`tool_choice` requests; ADR-027 applies it to every chat-completions request.
2. **NL-routing-fraction reduction is universal.** Tier 1 leaves NL routing in the C1 failure space (≈ zero ensemble dispatch) for non-`tool_choice` requests. ADR-027 routes every NL request through the validated routing-planner.
3. **The confabulation-mode mitigation generalizes.** Spike μ established that the structural-bounding finding holds across 4 documented confabulation modes (multi-dispatch fabrication; path hallucination; substrate-path-as-deliverable; coherent factual errors uncalibrated). The Tier 1 hybrid mitigates only the modes that arise from the explicit-`tool_choice` interception path; ADR-027 mitigates them universally on the chat-completions surface.

---

## What this comparison does not resolve

- **Generalization beyond qwen3:8b.** Spike ε + ε' + μ are all qwen3:8b-grounded. AS-9's `*Plausible-but-untested*` qualifier (production-scale numerical content broader than Spike ε' B1's 25 figures; cheap-tier reliability for direct-completion-of-factual-questions in training-data-error-prone domains; coherent factual errors uncalibrated on direct-completion path under adversarial pressure; generalization beyond qwen3:8b) lists residual evidence-strength bounds. Mitigation: BUILD-phase work + PLAY-phase experiential discovery before declaring the pipeline production-ready.
- **Multi-step composition mechanism (OQ #21).** Single-step planner + framework chain-heuristic vs. multi-step planner vs. planner-loops-with-context — DECIDE resolves. The estimate above assumes the simplest option (Spike δ's `resolve_input(step.input, results)` pattern) since it has the strongest empirical grounding.
- **Latency tuning (paired with OQ #20).** The current ~36s floor at qwen3:8b is the starting state; tuning playbook is DECIDE-phase work. OQ #20 timeout research (running in parallel) determines whether the floor is compatible with Population A defaults.

---

## Recommendation for DECIDE

Adopt ADR-027-direct as the cycle's PRIMARY mechanism per GT-2(a) + structural advantage. The hybrid extension (server-side `tool_choice` interception layered on top of the routing-planner) remains a conditional alternative for operator-deployment shapes where some client population sends explicit `tool_choice={"name":"<ensemble>"}`. The conditional extension is meaningful only under "implement `tool_choice` handling" disposition (Tension 19) — that disposition is a separate DECIDE question (OQ in scope of the `tool_choice` ADR).

The Essay-Outline Amendment A3 framing — "ADR-027 framework-driven dispatch pipeline as PRIMARY direction; hybrid as conditional alternative" — is corroborated by this comparison.

## Sources

- `src/llm_orc/web/api/v1_chat_completions.py` (current chat-completions handler; 801 LoC)
- `src/llm_orc/agentic/orchestrator_runtime.py` (current ReAct loop; 742 LoC)
- `src/llm_orc/agentic/orchestrator_tool_dispatch.py` (current tool-dispatch implementation; 1887 LoC)
- `src/llm_orc/agentic/dispatch_envelope.py` (envelope contract; 160 LoC)
- `src/llm_orc/agentic/dispatch_event_substrate.py` (event surface; 209 LoC)
- `.llm-orc/ensembles/spike-cycle7-epsilon-response-synthesizer.yaml` (scratch synthesizer ensemble — Spike ε)
- `essays/research-logs/cycle-7-spike-kappa-tool-choice-diagnosis.md` (D0 finding — `tool_choice` strip-at-input)
- `essays/research-logs/cycle-7-spike-zeta-routing-planner.md` (routing-planner viability — 100% JSON conformance at qwen3:8b)
- `essays/research-logs/cycle-7-spike-epsilon-pipeline.md` (end-to-end pipeline validation)
- `essays/research-logs/cycle-7-spike-epsilon-prime-pipeline-bounds.md` (gate-tail bounds)
- `essays/research-logs/cycle-7-spike-mu-confabulation-generalization.md` (structural-bounding generalizes across 4 modes)
- `essays/essay-outline-006-cross-compatibility-routing-surface.md` Amendment Log A1-A4 (Cycle 7 DISCOVER backward-propagation)
- `domain-model.md` AS-9 (structurally-bounded LLM roles invariant)
