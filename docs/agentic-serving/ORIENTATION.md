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
- **Result summarization is a correctness requirement (AS-7).** Full ensemble result dictionaries must be summarized before entering the orchestrator's context. The Result Summarizer Harness is structurally interposed between `invoke_ensemble` and the Runtime (FC-8).
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
- `essays/001-agentic-serving-architecture.md` — research essay: API surface, orchestrator agent, self-building ensembles, Plexus integration.
- `essays/research-logs/` — research logs for both cycles (Q1-Q4 and Q5-Q6).
- `essays/reflections/001-agentic-serving-architecture.md` — post-research reflections.
- `housekeeping/cycle-status.md` — active cycle state, phase tracking, feed-forward signals.
- `housekeeping/audits/` — citation, argument, susceptibility-snapshot, and conformance audit reports.
- `housekeeping/gates/` — gate reflection notes capturing phase-boundary commitment gating.

## Current state

**RESEARCH, DISCOVER, MODEL, DECIDE, and ARCHITECT phases complete. BUILD in progress — WP-A and WP-B done.**

The essay investigates six questions across two research cycles and concludes with a four-layer architecture. Product discovery surfaced seven value tensions and six assumption inversions. The domain model establishes 8 scoped invariants (AS-1 through AS-8), 17 concepts, 13 actions, and 7 open questions. DECIDE produced 11 accepted ADRs, 29 behavior scenarios, and stakeholder interaction specifications. ARCHITECT produced a 12-module system design across 4 dependency layers, a 13-criterion fitness-criteria set, an 18-edge boundary integration test plan, and a 10-WP roadmap with 3 classified transition states.

**BUILD progress.** Two of ten work packages complete. **WP-A** extracted `validate_ensemble_reference_graph` as a public function in `core/config/ensemble_config.py` with three call sites (load path, listing path, MCP/web validate path), satisfying FC-6 and unblocking WP-G (composition). **WP-B** stood up the serving foundation: `/v1/chat/completions` (streaming and non-streaming SSE), `/v1/models`, Session Registry with identity resolution across three derivation methods, Orchestrator Configuration with per-session resolution and allowlist, and the typed `resolve_session_start_context` function with cache — satisfying FC-9 both behaviorally (once-per-session cache) and structurally (AST-enforced single production call path). The orchestrator-handoff stubs are in place; WP-C replaces them with the real ReAct loop. Test suite: 2151 passed, 91.21% coverage.

**Key architectural posture:** design for stateless operation (AS-8); Plexus is an upgrade to a learning system, not a prerequisite. The orchestrator's internal action space is fixed (5 tools, ADR-003); client-declared tools delegate at turn boundaries (scenario-gated commitment). Phase 2 Plexus context injection is reserved via a typed function signature in Serving Layer rather than a standalone module (system design amendment #1, 2026-04-20).

**Open going into WP-C:** client-tool BDD scenarios (blocks WP-F); visibility form (OQ #2); Budget specific numbers; Calibration N; Session identity mechanism; `record_outcome` payload schema. See `roadmap.md` §Open Decision Points for full list and which WPs they gate. At WP-C start, FC-4 (Runtime imports only Budget Controller, Orchestrator Tool Dispatch, Result Summarizer Harness — no Plexus, no config, no Autonomy, no Calibration) becomes the load-bearing fitness criterion.
