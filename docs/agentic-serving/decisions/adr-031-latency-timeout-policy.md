# ADR-031: Latency and Timeout Policy for the Agentic-Serving Chat-Completions Surface

**Status:** Proposed

**Date:** 2026-05-22

---

## Context

The framework-driven dispatch pipeline (per ADR-027 — plan → dispatch → synthesize) shifts the chat-completions latency profile relative to the orchestrator-LLM-as-decider baseline. Spike ε (Cycle 7) measured the empirical floor at qwen3:8b via local Ollama:

- **Single-step planner-driven** (one dispatched ensemble): ~36 seconds end-to-end.
- **Multi-step deterministic chain** (planner + dispatched chain via framework chain-heuristic per Spike δ): ~64 seconds end-to-end for two-step chains.

The latency floor stacks the routing-planner ensemble (p50=10s per Spike ζ) + the dispatched capability ensemble (varies by capability and tier; ~20s at cheap-tier for simple capabilities) + the response-synthesizer ensemble (varies by input complexity; ~6s on simple structured input). Each stage's latency adds; the floor is the sum of cheap-tier latencies the pipeline composes.

Tranche 1 research note `cycle-7-oq-20-population-a-timeouts.md` characterized Population A tool-family timeout behavior:

- **OpenCode** — 300,000 ms (5-minute) default; configurable to 600,000 ms; streams by default. Accommodates both floors.
- **Aider** — `None` default (delegates to LiteLLM 600,000 ms / openai-python 10-minute default); configurable via `--timeout`; streams by default. Accommodates both floors.
- **Cline** — 30,000 ms (30 s) hard default; `requestTimeoutMs` knob has documented unreliability across providers (issue #4308); streams by default. Breaches single-step floor by ~6 s every request without operator tuning.
- **Cursor** — base-URL override is plan-mode-only (most agentic traffic routes through Cursor's backend); ceilings reported across forum threads at 4 s / 20 s / 200 s per code path; no configurable knob exposed. Structurally outside scope for non-plan-mode traffic.

Tranche 2 framing audit F3 surfaced that the binary "accommodate / breach" framing underweights the Cline operator-tuning fraction — operators who tune Cline's `requestTimeoutMs` can accommodate the floor (within the bounds of issue #4308's documented unreliability). The F3 recommendation: graded framing in the latency ADR's deployment-recommendation language.

This ADR specifies the latency posture, deployment-target coverage, tuning playbook, and the tier escalation policy for direct-completion under latency pressure.

---

## Decision

**Adopt the framework-driven dispatch pipeline's empirical latency profile** (~36 s single-step floor; ~64 s chained floor at qwen3:8b via local Ollama) as the cycle's BUILD-default latency posture. Tuning axes are available to operators; the cycle does not specify a single mandatory latency floor.

### Population A coverage tiers (graded framing per F3)

The latency policy classifies Population A clients into three coverage tiers based on their timeout-handling behavior relative to the empirical floors:

**Tier A — Default-accommodation:**

- **OpenCode** — 5-minute default accommodates both floors out of the box. Streaming-default masks single-step floor as long as tokens flow within underlying HTTP read timeout. Operators upgrading to multi-step chains may tune to 10-minute via `provider.<name>.options.timeout`. No mitigation required for Cycle 7 BUILD.
- **Aider** — `None`-default delegates to LiteLLM/openai-python's 10-minute default. Streaming-default masks both floors. Operators forced to non-streaming for reasoning models can set `--timeout 180` (3 minutes) to cover chained scenarios with headroom. No mitigation required for Cycle 7 BUILD.

**Tier B — Operator-tuning-required:**

- **Cline** — 30 s hard default breaches single-step floor by ~6 s every request and breaches chained floor by ~34 s every request. Operator tuning via "Request Timeout" UI setting (`requestTimeoutMs` internally) to ≥ 120,000 ms covers both floors with ~60-second headroom. **Caveat:** Cline issue #4308 documents `requestTimeoutMs` as unreliable across providers — set values can still abort early. Population A coverage for Cline is conditional-with-documented-tuning + integration-smoke-test recommendation. The deployment-documentation contribution this ADR makes is naming Cline as Tier B and recommending operators run an integration smoke test against their Cline configuration before relying on the tuning.

  **Integration smoke test specification (operator-facing):** send a single-capability NL request that should match an installed capability ensemble (e.g., a code-generation request against a deployment with `code-generator` installed); verify the response arrives within the operator's configured `requestTimeoutMs` minus 5 s headroom; confirm the response includes the expected ensemble output OR a direct-completion framing per Rule 5 (per ADR-029). Repeat for a chained-composition request (e.g., a search-then-extract request that triggers two-step framework chaining); verify the response arrives within `requestTimeoutMs` minus 10 s headroom. The smoke test passes when both single-step and chained scenarios complete within the configured timeout; the smoke test fails when either scenario aborts early or returns a Cline-side timeout error. Operators failing the smoke test should investigate Cline issue #4308 in their specific Cline build before deploying.

**Tier C — Structurally outside scope:**

- **Cursor** — base-URL override is plan-mode-only (most agentic traffic stays on Cursor's backend); reported ceilings of 4-20 s on agentic paths with no configurable knob; reports of Cursor sending `Responses` API payloads to `/v1/chat/completions` endpoints break format compatibility independent of timeout. Cursor coverage is limited to plan-mode chat panel use; agentic-path traffic is structurally outside scope. The deployment-documentation contribution this ADR makes is naming Cursor as Tier C and scoping Population A coverage to plan-mode-only use.

The graded framing replaces a binary "accommodate / breach" reading with a three-tier coverage statement that names the operator-tuning fraction explicitly. Per F3, the cycle's commitment is honest about where coverage is conditional rather than absolute.

### Tuning playbook (operator-facing)

The tuning playbook combines four mechanisms; operators apply them based on their deployment profile and Population A coverage requirements:

- **Classifier pre-filter at request boundary** — a lightweight deterministic classifier (regex / keyword / structural rules) that identifies clearly-determinable cases (explicit ensemble naming; obviously-no-capability-match patterns) and short-circuits the routing-planner ensemble. Reduces planner invocations on the populations the classifier covers. Lossy by design — the planner is the fallback for non-classifier-covered requests. BUILD-phase work to specify the classifier rules; operator-tuneable for deployment-specific request shapes.
- **Cached planner decisions** — memoize routing-planner output for identical or near-identical requests; subsequent identical requests skip the planner. Reduces planner invocations on the population of repeated request shapes. Cache key design (exact-match vs. semantic-similarity-match) is BUILD-phase design; the cache TTL is operator-tuneable.
- **Smaller faster planner model** — alternative cheap-tier models with lower latency (e.g., qwen3:4b, phi-3.5-mini) substituted via the routing-planner ensemble's model-profile override. Reliability profile relative to qwen3:8b is empirically open; operators substituting models bear the characterization burden per ADR-028.
- **Streaming synthesizer response** — chat-completions response streams synthesizer output as it's produced, masking some end-to-end latency for streaming-default clients (OpenCode, Aider, Cline). Synthesizer's typical first-token latency at qwen3:8b under structured input is ~2-3 s; streaming lets clients begin displaying response well before the full ~6-8 s synthesizer-stage completion. Operator-configurable via the existing `stream` parameter in chat-completions request; the framework honors `stream: true` on the response-synthesizer's output.

Each tuning mechanism is independent and additive. Operators with strict latency budgets stack mechanisms; operators on permissive deployments may apply none. The default Cycle 7 BUILD configuration ships without classifier pre-filter or caching; the streaming-synthesizer-response mechanism ships with the synthesizer ensemble per ADR-029.

### Spike ν amendment (2026-05-23): classifier pre-filter and caching elevated optional → recommended for adversarial-exposed deployments

Spike ν (Track A.3 architect→build boundary probe; research log `cycle-7-spike-nu-long-horizon-ceiling.md`) ran a 40-prompt adversarial routing battery against the planner and returned a single-surface Intermediate verdict (37/40 = 92.5% JSON conformance, in the 80-95% band). The three conformance misses split into two robustness modes:

- **Prompt-injection susceptibility (E1, E3).** The planner obeyed an instruction-override injection (emitting `{"action": "launch", "ensemble": "all"}`) and dispatched to a fabricated ensemble (`oracle`), the latter exploiting the planner's own Decision Rule 1 ("trust explicit naming").
- **Empty-response reliability (A6).** The planner returned an empty plan (only a `<think>` block) on a legitimate ambiguous request.

**Primary defense is structural, at the pipeline, not the planner.** ADR-027's framework-driven dispatch pipeline validates the plan before acting: `action` must be in `{dispatch, direct}` and `ensemble` must be in the registered capability set, or the request falls to direct completion. Under that validation E1's `launch` action and E3's `oracle` ensemble are rejected; the injection cannot become a dispatch fault. Plan validation is therefore non-optional at the pipeline layer (see the ADR-027 candidate scenarios added at BUILD entry). A6's empty/unparseable plan also routes to direct completion under the same stage.

**Elevation.** Given the planner-layer attack surface Spike ν surfaced, the **classifier pre-filter** is elevated from optional to **recommended for any deployment exposed to untrusted or adversarial input**. Its security value is defense-in-depth: screening injection-shaped, degenerate, and obviously-determinable inputs at the request boundary reduces the population of requests reaching the planner, narrowing the planner-layer susceptibility window the pipeline backstop already covers for dispatch safety. **Cached planner decisions** are likewise elevated to **recommended** where request shapes repeat, primarily for latency but secondarily to bound planner re-evaluation of recurring adversarial probes. Both remain absent from the zero-configuration default (the pipeline backstop is the load-bearing safety mechanism); the recommendation is graded by deployment exposure, consistent with the F3 graded-framing posture above. Planner-layer prompt-injection hardening (system-prompt changes to the planner) remains an open robustness question (carried in the Spike ν scope-of-claim partition); the pipeline backstop and the classifier pre-filter are the Cycle 7 BUILD answers, not planner-prompt hardening.

### Tier escalation policy for direct-completion under latency pressure

When `action: direct` (no capability match) — the response-synthesizer's direct-completion path — operators may want a higher-tier model for substantive responses on factual or reasoning-heavy direct-completion requests. The tier escalation policy:

- **Default tier:** cheap-tier (qwen3:8b empirically baseline per ADR-029).
- **Escalation triggers:**
  - Calibration Gate Reflect verdict on the synthesizer's direct-completion output (per ADR-029's three Reflect-trigger criteria including Rule 1 fabrication signal).
  - Operator-configured tier override for the direct-completion path (per a deployment-level config field — BUILD-phase design).
  - Per-request `model` field in the chat-completions request maps to a higher tier (the framework's existing model-profile-resolution mechanism per ADR-011).
- **Escalation target:** operator-configured higher-tier model (e.g., gpt-4o-mini, Claude Haiku 4.5) or operator-configured escalation chain.

Spike ε' A2's "Urga / Khovd" factual error (a training-data error in qwen3:8b's direct-completion path) is the data point motivating the escalation policy — cheap-tier models have domain-specific training-data error patterns; tier escalation for direct-completion mitigates without requiring per-domain spikes for every direct-completion request class.

The tier escalation policy is BUILD-phase design work; ADR-031 names the structure (default + named triggers + operator-configured target). Production traffic determines which triggers operators apply; the default for Cycle 7 BUILD is cheap-tier with no automatic escalation, leaving the policy operator-configurable for deployment-specific shapes.

### Streaming as a load-bearing surface

Streaming response delivery (chunked transfer per OpenAI chat-completions semantics) is a load-bearing surface for accommodating Tier A clients' streaming-default behavior. The framework's existing chat-completions handler supports streaming; ADR-031 confirms streaming as a load-bearing commitment under the framework-driven pipeline:

- The response-synthesizer ensemble's output is streamed token-by-token (or chunk-by-chunk per the underlying model's streaming behavior) when the client requests `stream: true`.
- The routing-planner ensemble and dispatched capability ensembles run synchronously upstream of the synthesizer; streaming is at the *response* layer, not the *pipeline* layer.
- For streaming-default Population A clients (OpenCode, Aider, Cline), streaming masks the synthesizer-stage latency partly — first-token delivery happens ~2-3 s after synthesizer invocation rather than ~6-8 s after; the perceived response latency drops accordingly.

Streaming is independent of the tuning playbook above (a fifth mechanism orthogonal to the four). Cycle 7 BUILD ships streaming support as a default; operators or clients that disable streaming pay the full synthesizer-stage latency.

### Out of scope for ADR-031

- **Multi-step composition latency under planner-loops-with-context architecture** — OQ #21 names three multi-step composition mechanisms; their latency profiles vary. The chained floor of ~64 s applies to deterministic two-step chains per Spike δ; mechanisms involving planner-loops-with-context (re-invoking the planner with results context) compound latency by an additional planner-stage per step. Multi-step composition mechanism choice is downstream-phase work; latency ADR confirms the floor for the initial BUILD's single-step + framework-chain-heuristic default.
- **Cold-start latency for first-time deployments** — Ollama model cold-start, framework initialization, and ensemble-loading add startup latency. ADR-031 governs steady-state per-request latency; cold-start is a separate operational concern outside this ADR's scope.
- **End-to-end SLA commitments** — the cycle does not commit to specific SLA bounds (e.g., "p99 < 60 s"). The empirical floor is documented; the tuning playbook is named; operators bear the SLA-design burden for their deployments.

---

## Rejected alternatives

### Binary "accommodate / breach" framing for Population A coverage

The latency ADR classifies Population A clients into two categories: those whose defaults accommodate the empirical floor, and those whose defaults breach it. No operator-tuning carve-out is provided.

**Rejected because:** the binary framing underweights the Cline operator-tuning fraction (per F3 framing-audit carry-forward). Cline operators who tune `requestTimeoutMs` accommodate the floor (within issue #4308's documented unreliability bounds); the binary framing classifies Cline as "breach" without acknowledging the operator-mitigation path. The graded three-tier framing (Default-accommodation / Operator-tuning-required / Structurally-outside-scope) is honest about where coverage is conditional.

The binary framing also conflates Cursor (structurally outside scope; no operator-tuning mitigation) with Cline (Tier B; tuning available with documented caveats). The two cases have different operational implications — Cursor coverage requires architectural restriction (plan-mode-only use); Cline coverage requires operator configuration + smoke-test verification. The graded framing distinguishes.

### Mandate a single deployment latency floor (e.g., "agentic-serving deployments must complete chat-completions requests within 30 s")

The ADR specifies a hard latency floor; deployments exceeding the floor are unsupported.

**Rejected because:** the empirical floor at qwen3:8b is ~36 s single-step; mandating a 30-s floor would require operators to use higher-tier (faster) models, which contradicts the project's cost-distribution value proposition (per OQ #18). Mandating a 60-s floor would exclude Cline operators who can't tune `requestTimeoutMs` past its unreliable limits. Mandating any specific floor concentrates the operational burden on a class of deployments without addressing the underlying tuning fraction.

The cycle's commitment is empirical-floor documentation + tuning-playbook availability; SLA commitments are deployment-specific.

### Defer all latency tuning to a follow-on cycle

The ADR specifies the empirical floor and names the tuning axes; no implementation of any tuning mechanism in Cycle 7 BUILD.

**Rejected because:** streaming-synthesizer-response is the load-bearing tuning mechanism for Tier A clients (OpenCode, Aider, Cline-when-streaming-default-engaged) and is structurally part of the response-synthesizer ensemble's behavior — supporting streaming is a low-cost addition to ADR-029's implementation. Cycle 7 BUILD ships streaming. The other three tuning mechanisms (classifier pre-filter, cached planner decisions, smaller faster planner model) are operator-configurable and can land in BUILD or in follow-on cycles as deployment evidence warrants.

The deferral alternative would leave Tier A clients paying the full ~36-s floor for first-token delivery on streaming-default workflows, when the streaming mechanism reduces first-token latency to ~2-3 s after synthesizer invocation. The cost of including streaming in BUILD is small; the user-experience improvement is substantial.

### Cursor-Plan-Mode-only scoping notice in ADR rather than as deployment commitment

Acknowledge Cursor's structural limitation but treat it as informational rather than committing to plan-mode-only-supported scope.

**Rejected because:** the structural limitation is operational, not informational. Operators deploying llm-orc agentic-serving for Cursor users must understand that agentic-path traffic does not reach llm-orc; treating the limitation as informational creates a Population A coverage gap operators discover empirically. Naming Cursor as Tier C (Structurally-outside-scope) makes the gap explicit at the ADR layer — operators can plan deployments accordingly.

---

## Consequences

### Positive

- **The empirical latency floor is documented honestly.** Operators see the ~36 s single-step / ~64 s chained baseline at qwen3:8b; deployment planning works from real numbers rather than aspirational SLAs.
- **Population A coverage is graded per F3 framing-audit.** Tier A (default-accommodation), Tier B (operator-tuning-required), and Tier C (structurally outside scope) capture the operational reality without binary-framing distortion. Cline operators see the tuning path; Cursor operators see the structural limitation.
- **The tuning playbook is operator-actionable.** Four named mechanisms (classifier pre-filter; cached planner decisions; smaller faster planner model; streaming synthesizer) + the tier escalation policy give operators concrete levers. Cycle 7 BUILD ships streaming as the load-bearing default; the others land as deployment evidence warrants.
- **Tier escalation for direct-completion is structurally named.** Spike ε' A2's "Urga / Khovd" data point motivates the escalation policy; operators with reasoning-heavy direct-completion patterns can configure the escalation target.
- **Streaming is a load-bearing default.** First-token latency for streaming-default Tier A clients drops to synthesizer-first-token time (~2-3 s); the empirical 36-s floor is the full-response time, not the perceived first-byte time.

### Negative

- **The empirical floor is high for Cline default-configuration users.** Operators deploying for Cline users without tuning experience timeout errors on every chat-completions request. Cline issue #4308's documented unreliability of `requestTimeoutMs` means even tuned operators may experience intermittent timeout aborts. The smoke-test recommendation is a partial mitigation; the structural limitation in Cline's client surfaces here.
- **Cursor agentic-path coverage is unavailable.** Cursor users on the agentic path get no llm-orc agentic-serving coverage; only plan-mode chat-panel use reaches llm-orc. The Population A coverage gap is real; ADR-031 names it.
- **The Cycle 7 BUILD default does not include the classifier pre-filter or planner-decision caching mechanisms.** The full latency floor applies to every chat-completions request that doesn't hit the streaming mask. Operators with strict latency budgets implement the tuning mechanisms or accept the floor.
- **Tier escalation policy for direct-completion is operator-configured per deployment.** The default cheap-tier behavior may produce direct-completion responses with domain-specific training-data errors (per Spike ε' A2). Operators bearing the configuration burden may not configure escalation for non-obvious failure surfaces until production evidence surfaces them.
- **The streaming mechanism's latency benefit applies only to streaming-default clients.** Aider in non-streaming mode (reasoning-model forced) and operators disabling streaming pay the full floor; streaming is a configuration-dependent improvement, not universal.

### Neutral

- **The ADR does not commit to specific SLA bounds.** Deployment SLAs are operator-specific; the cycle's commitment is the empirical floor + tuning playbook + coverage tier classification.
- **The tier escalation policy reuses existing infrastructure** (Calibration Gate per ADR-014, Tier-Escalation Router per ADR-015, Audit Dispatch per ADR-018). No new architecture; the escalation triggers extend existing mechanisms.
- **The Population A coverage tiers may evolve.** Cline issue #4308 may be resolved upstream; Cursor's base-URL override may extend to agentic paths; new Population A clients may emerge. ADR-031's tier classification is a Cycle 7-empirical reading; future cycles update as evidence warrants.
- **The latency floor is a function of model selection.** Operators substituting faster cheap-tier models for qwen3:8b reduce the floor; substituting slower models grow it. The floor is empirically anchored, not architecturally fixed.

## Provenance check

- **Empirical latency floors (~36 s single-step; ~64 s chained at qwen3:8b)**: Spike ε research log `cycle-7-spike-epsilon-pipeline.md` (driver). Driver chain: same-cycle empirical spike.
- **Population A tool-family timeout behavior (OpenCode, Aider, Cline, Cursor)**: Tranche 1 research note `cycle-7-oq-20-population-a-timeouts.md` (driver). Driver chain: same-cycle DECIDE-entry research.
- **Tranche 2 framing audit F3 (graded framing not binary)**: cycle-status §"Framing-audit findings carried forward to DECIDE/ADR-drafting/gate consideration" F3 entry (driver). Driver chain: same-cycle Tranche 2 audit.
- **Cline issue #4308 unreliability of `requestTimeoutMs`**: OQ #20 research note Cline section + issue #4308 cited (driver). Driver chain: same-cycle DECIDE-entry research + external documentation.
- **Cursor structural limitation (base-URL override is plan-mode-only)**: OQ #20 research note Cursor section + Cursor forum threads cited (driver). Driver chain: same-cycle DECIDE-entry research + external documentation.
- **Tuning axes (classifier pre-filter, cached planner decisions, smaller faster planner model, streaming synthesizer)**: DISCOVER 2026-05-21 framing recorded in cycle-status §"Key conversational reframings to preserve" (driver). Driver chain: same-cycle DISCOVER + cycle-status.
- **Spike ν amendment (classifier pre-filter + caching elevated optional→recommended for adversarial-exposed deployments)**: Spike ν research log `cycle-7-spike-nu-long-horizon-ceiling.md` Surface 3 + Findings ν.2/ν.3 (driver). Driver chain: same-cycle architect→build boundary spike.
- **Spike ε' A2 "Urga / Khovd" data point motivating tier escalation policy for direct-completion**: Spike ε' research log Finding ε'.A2 (driver). Driver chain: same-cycle empirical spike.
- **Calibration Gate / Tier-Escalation Router / Audit Dispatch infrastructure reuse**: ADR-007, ADR-014, ADR-015, ADR-018 (drivers). Driver chain: prior-ADRs.
- **Streaming as load-bearing surface for Tier A clients**: OQ #20 streaming-default observations per OpenCode + Aider + Cline (driver) + drafting-time synthesis (first-token latency calculation). Driver chain: same-cycle research + drafting-time engagement.
- **Rejected alternative — binary framing**: Tranche 2 framing audit F3 (driver, the audit's specific concern). Driver chain: same-cycle audit.
- **Rejected alternative — mandate single deployment latency floor**: drafting-time synthesis weighing SLA-mandate cost against cost-distribution value proposition (per OQ #18). Driver chain: same-cycle research + drafting-time engagement.
- **Rejected alternative — defer all tuning to follow-on cycle**: drafting-time synthesis weighing streaming inclusion cost against Tier A user-experience benefit. Driver chain: drafting-time engagement.
- **Rejected alternative — Cursor as informational not deployment commitment**: drafting-time synthesis on operator-discovery cost. Driver chain: drafting-time engagement.
