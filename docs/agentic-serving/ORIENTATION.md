# Agentic Serving -- Orientation

## What this system is

Agentic serving extends llm-orc (a declarative DAG-based LLM orchestration engine) to serve as the backend for agentic coding tools via OpenAI-compatible endpoints. An orchestrator agent sits behind `/v1/chat/completions`, running a ReAct loop that delegates to llm-orc ensembles as tools. The Plexus knowledge graph, when available, provides cross-session memory and design knowledge accumulation, converting the orchestrator's routing decisions from reasoning tasks into retrieval tasks over time. The baseline product ships without Plexus; the learning-system value is an upgrade.

## Who it serves

- **Tool user** — developer using an agentic coding tool pointed at llm-orc. Cares about quality, speed, cost. Reading path: `ORIENTATION.md` → `product-discovery.md` (Jobs and Mental Models) → `interaction-specs.md` (Tool User stakeholder) → essay abstract.
- **Ensemble author / operator** — creates ensembles, profiles, scripts; runs the server. Often the same person as the tool user. Cares about visibility as tinkering, organic stabilization, cost control. Reading path: `ORIENTATION.md` → `system-design.md` → `roadmap.md` → `product-discovery.md` → `interaction-specs.md` (Ensemble Author / Operator stakeholder) → `domain-model.md` → essay (full).
- **Orchestrator LLM** — the agent behind the endpoint. Needs the full composition palette, knowledge graph access, and budget constraints. Reading path: `system-design.md` (Orchestrator Runtime, Orchestrator Tool Dispatch modules) → `domain-model.md` → essay (Orchestrator Agent, Context Management sections) → `interaction-specs.md` (Orchestrator LLM stakeholder).

## Key constraints

Drawn from `domain-model.md` (AS-1 through AS-8) and project-level Invariants 1-14. The constraints below most shape downstream decisions:

- **Plexus is optional (AS-8).** The orchestrator works statelessly — serving layer, ReAct loop, budget enforcement, result summarization, and ensemble composition all function without Plexus. Plexus is an upgrade to a learning system, not a prerequisite. The system design enforces this through the Plexus Adapter's no-op fallbacks (FC-7).
- **Result summarization is a correctness requirement (AS-7).** Full ensemble result dictionaries are summarized before entering the orchestrator's context by a Result Summarizer Harness interposed on Tool Dispatch's `invoke_ensemble` return path (system-design Amendment #3). The Runtime is unaware of the Harness — summarization is a Tool-Dispatch-side concern. FC-8 is structurally enforced via a strict AST dominance check (`test_fc8_summarizer_bypass.py`): every successful `invoke_ensemble` result lives inside the match on the summarize result, so a bypass path fails the test before merge. ADR-004's raw-output escape hatch is the only way to return the raw dict, and it is opt-in per ensemble (`raw_output: true`), not a default.
- **Budget enforcement is a control plane concern (AS-3).** Turn limits and token budgets are enforced at the session level, checked at each iteration of the ReAct loop, regardless of what the orchestrator LLM decides. The Budget Controller is an L1 policy module the Orchestrator Runtime calls pre-iteration (FC-10).
- **Dynamic invocations are outside the ensemble reference graph (AS-1).** Project Invariant 7 governs static YAML composition; it does not govern the orchestrator's runtime tool-mediated invocations. Orchestrator-composed ensembles must still satisfy Invariant 7 internally (AS-2); the Composition Validator enforces this by sharing a single public routine with the load path (FC-6).
- **The orchestrator's internal action space is exactly five tools (ADR-003).** `invoke_ensemble`, `compose_ensemble`, `list_ensembles`, `query_knowledge`, `record_outcome`. Client-declared tools (bash, file_edit, etc. from OpenCode / Roo Code / Cline) flow through the orchestrator's *response surface* via turn-boundary delegation, not through its internal action space. WP-F is scenario-gated against this boundary.

## How the artifacts fit together

**Tier 1 — Entry point:**
- `ORIENTATION.md` (this document) — what this is, who it serves, where to look next.

**Tier 2 — Primary readables:**
- `product-discovery.md` — stakeholder needs, value tensions, assumption inversions. Read for product context.
- `system-design.md` — module decomposition, responsibility matrix, dependency graph, integration contracts, fitness criteria, test architecture. Read for technical context during build.
- `roadmap.md` — work packages, dependency classification (hard / implied / open choice), transition states, open decision points. Read for sequencing context.

**Tier 3 — Supporting material:**
- `domain-model.md` — scoped vocabulary (concepts, actions, invariants AS-1..AS-8, open questions); project-level vocabulary remains in force.
- `decisions/adr-001..adr-011-*.md` — architectural decisions with context, decision, and consequences.
- `scenarios.md` — behavior scenarios in Given/When/Then form, derived from ADRs. Drive BDD acceptance tests during build.
- `interaction-specs.md` — stakeholder-by-stakeholder task decomposition at the interaction level. Informs UX and observable-behavior decisions.
- `field-guide.md` — module-to-code map derived from the system design at the current implementation state. Read when navigating the codebase in relation to the architecture.
- `essays/001-agentic-serving-architecture.md` — research essay: API surface, orchestrator agent, self-building ensembles, Plexus integration.
- `essays/research-logs/` — research logs for both cycles (Q1-Q4 and Q5-Q6).
- `essays/reflections/001-agentic-serving-architecture.md` — post-research reflections.
- `housekeeping/cycle-status.md` — active cycle state, phase tracking, feed-forward signals.
- `housekeeping/audits/` — citation, argument, susceptibility-snapshot, and conformance audit reports.
- `housekeeping/gates/` — gate reflection notes capturing phase-boundary commitment gating.

## Current state

**RESEARCH, DISCOVER, MODEL, DECIDE, and ARCHITECT phases complete. BUILD in progress — WP-A, WP-B, WP-C, WP-D, and WP-E done. Five of ten work packages complete.**

The essay investigates six questions across two research cycles and concludes with a four-layer architecture. Product discovery surfaced seven value tensions and six assumption inversions. The domain model establishes 8 scoped invariants (AS-1 through AS-8), 17 concepts, 13 actions, and 7 open questions. DECIDE produced 11 accepted ADRs, 29 behavior scenarios, and stakeholder interaction specifications. ARCHITECT produced a 12-module system design across 4 dependency layers, a 13-criterion fitness-criteria set, an 18-edge boundary integration test plan, and a 10-WP roadmap with 3 classified transition states.

**BUILD progress.** **WP-A** extracted `validate_ensemble_reference_graph` as a public function with three call sites, satisfying FC-6. **WP-B** stood up the serving foundation: `/v1/chat/completions` (streaming + non-streaming SSE), `/v1/models`, Session Registry, Orchestrator Configuration, and the typed `resolve_session_start_context` function, satisfying FC-9 behaviorally and structurally. **WP-C** landed the ReAct core end-to-end: three new L1/L2 modules (Budget Controller, Orchestrator Tool Dispatch, Orchestrator Runtime); extended `ModelInterface` with tool-calling support on `OpenAICompatibleModel`; wired Serving Layer to the real Runtime; added `llm-orc serve` command. Verified against local Ollama with `mistral-nemo:12b` in two live runs. FC-4 structurally enforced via AST scan. **WP-D** closed AS-7's structural enforcement: new Result Summarizer Harness module, `raw_output` flag on `EnsembleConfig` for the ADR-004 escape hatch, Tool Dispatch interposition of the Harness on `invoke_ensemble`'s return path, default `agentic-result-summarizer` ensemble shipped, system-design Amendment #3 moving the `RSH` edge from Runtime to Tool Dispatch, and FC-8 enforced via a strict AST dominance check with an adversarial self-test. **WP-E** landed Autonomy Policy: new L1 module with two Phase-1 levels (`operator-as-tool-user` baseline, `pure-tool-user-visible` tightened); interposition at Tool Dispatch between the unknown-tool filter and routing; `denied_by_autonomy` error kind; `VisibilityEvent` chunk variant rendered as inline `[composition: {json}]` narration on `delta.content` (OQ #2 resolved — narration reaches the tool user in vanilla OpenAI-compat clients); FC-11 enforced via strict AST dominance with an adversarial self-test. Test suite: 2257 passed, 91.48% coverage.

**Key architectural posture:** design for stateless operation (AS-8); Plexus is an upgrade to a learning system, not a prerequisite. The orchestrator's internal action space is fixed at five tools (ADR-003), structurally enforced via `match-case` dispatch + FC-4 static import check. Tool calling is opt-in per provider via `ModelInterface.supports_tool_calling`; `OpenAICompatibleModel` covers Ollama / OpenAI / OpenRouter / LM Studio / vLLM. Tool Dispatch delegates to `OrchestraService` — no parallel ensemble-execution path. Result summarization is a Tool-Dispatch-side concern; the Runtime never imports the Harness (FC-4 forbids it; FC-8 proves no bypass). Autonomy Policy is interposed on the dispatch path; every committed tool passes through `self._autonomy_policy.decide` before routing (FC-11). Client-declared tools delegate at turn boundaries (scenario-gated commitment; lands in WP-F). Phase 2 Plexus context injection is reserved via a typed function signature in Serving Layer (system design amendment #1, 2026-04-20).

**TS-1 remaining:** WP-F (client-tool turn-boundary delegation) is the last item before TS-1 is reached. Until it lands, the orchestrator can list and invoke ensembles but cannot delegate client-side tools (bash, file_edit) back to the coding client.

**Open going into next WP:** client-tool BDD scenarios (gates WP-F); Calibration N; `record_outcome` payload schema; summarizer-quality echo-back risk (carried forward as a forward signal for WP-H calibration scope — see `housekeeping/cycle-status.md` §Feed-Forward from WP-D). WP-G / WP-I can land in parallel (both depend only on WP-C). See `roadmap.md` §Open Decision Points for full list.
