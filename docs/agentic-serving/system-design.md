# System Design: Agentic Serving

**Version:** 1.0
**Status:** Current
**Last amended:** 2026-04-20
**Scope:** Scoped RDD cycle at `docs/agentic-serving/`. Inherits the project-level domain model (Invariants 1-14) and existing system architecture.

---

## Architectural Drivers

| Driver | Type | Provenance |
|--------|------|------------|
| Stateless-first operability — baseline product (Layers 1-3) runs with no Plexus dependency | Quality Attribute (primary) | AS-8; ADR-002; cycle-status §FF 15-16 |
| Deterministic control plane — Budget, Tool Surface, and Primitive set are harness-level circuit breakers, not model-level choices | Quality Attribute | AS-3, AS-6; ADR-003, ADR-005, ADR-008 |
| Orchestrator reasoning quality across long sessions | Quality Attribute | AS-7; ADR-004, ADR-005; essay §Context Management |
| Swappable orchestrator LLM via Model Profile | Quality Attribute | ADR-011; OQ #1 (knowledge-compensated model selection) |
| Observability of orchestrator activity (visibility *form* unresolved — OQ #2) | Quality Attribute | Product discovery tensions #2 and #5 |
| Auditability — closed tool-call vocabulary | Quality Attribute | ADR-003 |
| OpenAI-compatible protocol: `/v1/chat/completions` + `/v1/models`, SSE streaming with tool-call round-trips | Constraint | Essay §API Surface; interaction specs |
| Phase 2 Plexus injection hook point must be structurally reserved | Constraint | ADR-009 (post-gate reframe) |
| Ensemble Engine (Layer 3) unchanged | Constraint | ADR-001, ADR-002 |
| Project-level Invariants 1-14 remain in force; AS-1 through AS-8 layered on top | Constraint | Project domain model; agentic-serving domain model |
| Push-model Plexus ingestion; source-material ingestion boundary; async enrichment | Constraint | AS-4; ADR-010 |
| Orchestrator profile change is a session-boundary event | Constraint | ADR-011 |
| Existing FastAPI server and MCP handlers are extended, not replaced | Integration | Retrofit reconnaissance (2026-04-20) |
| Plexus (external lib) is optional; two code paths — with and without Plexus — must maintain feature parity on Layers 1-3 | Integration | ADR-002; AS-8 |
| Client-declared tools (OpenCode, Roo Code, etc.) flow through turn-boundary delegation, not through the orchestrator's internal action space | Integration | This document, §Client Tool Surface Commitment |
| Session sized for sustained agentic coding comparable to an RDD phase; multi-LLM-call-per-turn token accounting | Scale | ADR-005 |

---

## Client Tool Surface Commitment

**Decision.** The orchestrator's **internal** tool surface is exactly the five ADR-003 tools and no others. Client-declared tools (the `tools[]` array on a `/v1/chat/completions` request) become the orchestrator's **response surface**:

- The orchestrator's ReAct iterations call only the five internal tools.
- When a task step requires a client-side action (bash, file edit, etc.), the orchestrator closes the current turn with `finish_reason: tool_calls` and emits one or more client-tool `tool_calls[]` in the completion response.
- The client executes the tools and sends `role: tool` messages back in the next `/v1/chat/completions`. The orchestrator resumes the same Session's ReAct loop with those messages as observations.

**Provenance.** Committed in ARCHITECT 2026-04-20 on user direction. Honors ADR-003 strictly ("no others" refers to *internal* action surface; delegation at the turn boundary is response-surface behavior, not internal action). Supersedes the `interaction-specs.md` open-boundary note for the Tool User stakeholder. Future finding — ensembles needing to reach the client's environment *during* a tool execution rather than between turns — would reopen this as Option D territory and require amendment.

**Scenario-gated.** The commitment is the current architectural answer, but WP-F (client-tool delegation) does not start until stress scenarios are written into `scenarios.md` that exercise the turn-boundary vs. mid-execution distinction. If any of those scenarios require mid-execution callback to client tools, Option C is insufficient and the commitment is amended (Option C + hybrid, or full Option D).

---

## Module Decomposition

Twelve modules plus one typed extension-point function. Existing modules are marked **(existing)**; everything else is net-new surface area for agentic serving.

### Module: Serving Layer

**Purpose:** Translates the OpenAI-compatible wire protocol into Session-scoped orchestrator interactions.
**Provenance:** AS-8; ADR-001, ADR-002 (Layer 1); ADR-009 (Phase 2 injection reservation — function-level); Essay §API Surface; Client Tool Surface Commitment.
**Owns:** Serving Layer (concept); SSE streaming; tool-call formatting; `/v1/chat/completions` and `/v1/models` endpoints; response-surface tool delegation; session-start flow including the typed `resolve_session_start_context` function (Phase 1 returns `[]`; Phase 2 reads from Plexus Adapter).
**Depends on:** Session Registry, Orchestrator Configuration, Orchestrator Runtime. In Phase 2 only: Plexus Adapter.
**Depended on by:** (external clients — FastAPI app)
**Phase 2 hook point.** ADR-009 requires the Phase 2 injection point to be structurally reserved. This is satisfied by the typed function `resolve_session_start_context(session: SessionContext) -> list[PromptFragment]` at a single call site in the session-start flow. Phase 1 returns `[]` unconditionally; Phase 2 populates the body by reading from Plexus Adapter. The contract is the load-bearing part of the reservation — signature and call site commit the interface now so Phase 2 is a function-body change, not a structural change.
**Inversion note:** The Serving Layer boundary serves the Tool User's "the endpoint is a model" mental model (they see only the HTTP surface) and the Operator's "I start the server and point a client at it" mental model. Both converge at this boundary, so the boundary serves both users.

### Module: Session Registry

**Purpose:** Identifies and continues a multi-request Session by reconstructing orchestrator state from the conversation history.
**Provenance:** Session (concept); ADR-005 (Budget is Session-scoped); ADR-008 (Autonomy is Session-scoped); ADR-011 (Model Profile is Session-scoped); Client Tool Surface Commitment (Session spans client-tool round trips).
**Owns:** Session identity; Session lookup by request; cumulative-turn-count and cumulative-token-spend derivation; persistence of Session state across HTTP requests when persistence is required by Autonomy Level or Calibration state.
**Depends on:** Ensemble Engine (for profile resolution — optional).
**Depended on by:** Serving Layer, Budget Controller, Autonomy Policy, Calibration Gate.
**Inversion note:** The Operator's mental model is "a Session is the thing with Budget, Autonomy, and orchestrator profile; requests are how clients interact with it." The boundary aligns.

### Module: Budget Controller

**Purpose:** Enforces turn and token limits at each ReAct iteration boundary.
**Provenance:** AS-3; ADR-005; domain concept Budget.
**Owns:** Budget (concept); per-iteration circuit-breaker check; graceful termination with explicit exhaustion signaling.
**Depends on:** Session Registry.
**Depended on by:** Orchestrator Runtime.
**Inversion note:** Operator's mental model — "Budget is a thing I set; its enforcement is automatic and never negotiable by the LLM." The boundary is thin but load-bearing (AS-3 says control plane, not model plane). Kept separate from Session Registry because its change rate is different (Budget semantics will shift during rollout; Session identity rarely changes).

### Module: Orchestrator Runtime

**Purpose:** Runs the ReAct loop that delegates to llm-orc operations via a fixed tool surface.
**Provenance:** ADR-001; domain concepts Orchestrator Agent, Routing Decision, Conversation Compaction.
**Owns:** Orchestrator Agent (concept); Routing Decision (generation); Conversation Compaction (concept and action); Route, Invoke (Dynamic), Query, Record, Calibrate (as actor).
**Depends on:** Budget Controller, Orchestrator Tool Dispatch, Result Summarizer Harness.
**Depended on by:** Serving Layer.
**Inversion note:** The Orchestrator LLM's mental model is "I reason, I emit tool calls, I observe results." The Runtime boundary aligns with that mental model — it does not expose Session bookkeeping, Plexus awareness, or Autonomy gating to the LLM's reasoning context.

### Module: Orchestrator Tool Dispatch

**Purpose:** Defines the fixed five-tool surface and dispatches each orchestrator tool call to its downstream service.
**Provenance:** ADR-003; AS-6; domain concepts Orchestrator Tool, Dynamic Invocation.
**Owns:** Orchestrator Tool (concept); tool-name allowlist enforcement; routing of `invoke_ensemble`, `compose_ensemble`, `list_ensembles`, `query_knowledge`, `record_outcome` to their downstream services; rejection of any other tool name as a tool error.
**Depends on:** Ensemble Engine, Composition Validator, Plexus Adapter, Autonomy Policy, Calibration Gate.
**Depended on by:** Orchestrator Runtime.
**Rationale for separate module:** Keeping dispatch separate from the Runtime's reasoning loop makes the closed-set property of ADR-003 structurally enforceable — a code path that bypasses the dispatch to do something tool-like is mechanically excluded, not merely proscribed.

### Module: Composition Validator

**Purpose:** Validates a proposed ensemble against the existing reference graph using the same routine as load-time validation.
**Provenance:** AS-2, AS-6; ADR-006; Invariant 5 (cross-ensemble acyclicity); Invariant 7 (static references); Invariant 8 (depth limit); Invariant 11 (extras forbidden); scenarios.md refactor 1-3; cycle-status §FF 21.
**Owns:** Composition (concept); composition-time validation routine shared with `EnsembleLoader`.
**Depends on:** Ensemble Engine (shared validator routine lives in `core/config/ensemble_config.py` as a public function after the refactor).
**Depended on by:** Orchestrator Tool Dispatch.
**Retrofit debt:** Existing `EnsembleLoader._validate_cross_ensemble_cycles` and `_build_reference_graph` are private. They must be extracted to a public function (`core/config/ensemble_config.py`) that both the loader and the composition path call. `ValidationHandler._collect_validation_errors` must be wired through the same function with real `search_dirs`, and `list_ensembles` must pass its directory as `search_dirs` (scenarios refactor 1-3).

### Module: Ensemble Engine **(existing)**

**Purpose:** Executes ensembles per the existing declarative DAG engine.
**Provenance:** Entire project-level domain model; ADR-001, ADR-002 (Layer 3 unchanged).
**Owns:** Ensemble, Agent, AgentConfig (LLM/Script/Ensemble), Model Profile, Inline Model, Dependency, Phase, Fan-Out, Input Key, Ensemble Reference, Ensemble Reference Graph, Depth, Depth Limit, Artifact, Child Executor, Immutable Infrastructure, Mutable State, Agent Discriminator; actions Load, Validate, Discriminate, Execute, Dispatch, Recurse, Fan Out, Gather, Select, Detect Cycles, Check Depth, Merge Profile.
**Depends on:** (project-level dependencies unchanged)
**Depended on by:** Orchestrator Tool Dispatch (via invoke_ensemble), Composition Validator (shared validator), Result Summarizer Harness (invokes a summarizer ensemble), Bootstrapping Pipeline (reads library), Plexus Adapter (persistence of Routing Decisions derived from executions).

### Module: Result Summarizer Harness

**Purpose:** Positions a summarizer between ensemble completion and the orchestrator's context so unsummarized results never reach reasoning.
**Provenance:** AS-7; ADR-004; domain concepts Result Summarization, Summarize action.
**Owns:** Result Summarization (concept); Summarize (action); raw-output escape-hatch dispatch.
**Depends on:** Ensemble Engine (invokes the summarizer ensemble).
**Depended on by:** Orchestrator Tool Dispatch (interposed on `invoke_ensemble`'s return path).
**Rationale for separate module:** The summarizer itself is an ensemble (a primitive — configured, not coded). What this module owns is the *harness position* — the unskippable interposition between ensemble completion and tool-call result return. The Runtime is not aware of the summarizer; the summarizer is not aware of the Runtime. The harness makes the enforcement of AS-7 structural rather than conventional.

### Module: Autonomy Policy

**Purpose:** Gates orchestrator actions against the Session's Autonomy Level.
**Provenance:** ADR-008; AS-6 (hard limit: no configuration permits primitive authorship); domain concept Autonomy Level.
**Owns:** Autonomy Level (concept); per-action gate resolution; visibility surfacing of composition events when a tightened level requires it.
**Depends on:** Session Registry.
**Depended on by:** Orchestrator Tool Dispatch.
**Note on the pure-tool-user default.** Cycle-status §FF 25 flags that the default baseline Autonomy Level is calibrated for the operator-as-tool-user persona. Pure tool-user deployments (FF-2) may warrant a tighter default that surfaces composition events. The Autonomy Policy module exposes this as a configuration surface rather than a code change.

### Module: Calibration Gate

**Purpose:** Tracks Calibration state and runs quality checks on a composed ensemble's first N invocations.
**Provenance:** ADR-007; AS-5; domain concepts Calibration, Quality Signal.
**Owns:** Calibration (concept); Quality Signal (concept and generation in stateless mode; Plexus Adapter persists when active); per-ensemble calibration counter and transition-to-trusted logic; session-scoped state in stateless mode (ADR-007 clause 4).
**Depends on:** Ensemble Engine (invokes a check mechanism, which is itself an ensemble); Plexus Adapter (persistence when Plexus is active).
**Depended on by:** Orchestrator Tool Dispatch (interposed on `invoke_ensemble` for ensembles in calibration).

### Module: Plexus Adapter

**Purpose:** Mediates all Plexus interaction with no-op fallbacks when Plexus is absent.
**Provenance:** ADR-009, ADR-010; AS-4, AS-8; domain concepts Ingestion, Enrichment, Context Injection (data flow), Routing Decision (persistence), Quality Signal (persistence).
**Owns:** Ingestion (push to Plexus); Enrichment (invocation; Plexus performs the actual enrichment); Query (knowledge graph query mechanics); Record (outcome persistence); no-op fallback semantics when Plexus is absent.
**Depends on:** (external — Plexus lib)
**Depended on by:** Orchestrator Tool Dispatch (query_knowledge, record_outcome), Bootstrapping Pipeline (ingestion), Serving Layer's `resolve_session_start_context` (Phase 2 only), Calibration Gate (persistence of Quality Signals).
**Inversion note:** The Operator's mental model is "Plexus is a lib I enable; llm-orc pushes to it." The boundary preserves that — the Adapter is the single place Plexus-aware code lives, so the rest of the system sees a tool interface regardless of Plexus state. This is what makes AS-8 structurally enforceable.

### Module: Bootstrapping Pipeline

**Purpose:** Pushes library source material into Plexus as a deliberate operator operation.
**Provenance:** AS-4; ADR-010; DISCOVER FF #9 and #14; domain concept Bootstrapping.
**Owns:** Bootstrapping (concept and action); operator-triggered batch ingestion flow.
**Depends on:** Plexus Adapter, Ensemble Engine (reads library via existing config manager).
**Depended on by:** (operator — CLI/web trigger)

### Module: Orchestrator Configuration

**Purpose:** Loads and resolves the orchestrator's per-session configuration surface.
**Provenance:** ADR-005 (Budget defaults), ADR-008 (Autonomy defaults), ADR-009 (Plexus enablement), ADR-011 (Orchestrator Model Profile).
**Owns:** Per-session config resolution; operator-set bounds on per-request overrides.
**Depends on:** (project config manager — existing)
**Depended on by:** Serving Layer.

---

## Responsibility Matrix

Every concept and action from the agentic-serving domain model and the touched project-level concepts maps to exactly one owning module. Inherited project-level concepts that are not touched by agentic serving live with the existing Ensemble Engine (listed once at the bottom).

| Domain Concept / Action | Owning Module | Provenance |
|------------------------|---------------|------------|
| Orchestrator Agent | Orchestrator Runtime | Domain model; ADR-001 |
| Session | Session Registry | Domain model; ADR-005, ADR-008, ADR-011 |
| Serving Layer | Serving Layer | Domain model; ADR-002 |
| Orchestrator Tool | Orchestrator Tool Dispatch | Domain model; ADR-003 |
| Routing Decision | Orchestrator Runtime (generation); Plexus Adapter (persistence) | Domain model; ADR-009 |
| Dynamic Invocation | Orchestrator Tool Dispatch | AS-1; domain model |
| Composition | Composition Validator | Domain model; AS-2, ADR-006 |
| Primitive | Ensemble Engine (existing — role played by existing concepts) | AS-6; ADR-006 |
| Library | Ensemble Engine (existing — config manager) | Domain model |
| Budget | Budget Controller | AS-3; ADR-005 |
| Result Summarization | Result Summarizer Harness | AS-7; ADR-004 |
| Conversation Compaction | Orchestrator Runtime | Essay §Context Management; AS-7 |
| Context Injection | Serving Layer (function `resolve_session_start_context`) | ADR-009 (structurally reserved via typed function signature) |
| Ingestion | Plexus Adapter | AS-4; ADR-010 |
| Enrichment | Plexus Adapter (invocation); Plexus lib (performance) | AS-4; ADR-010 |
| Quality Signal | Calibration Gate (generation); Plexus Adapter (persistence when active) | AS-5; ADR-007 |
| Stabilization | Plexus Adapter (emergent; surfaced via queries) | AS-5; ADR-007 |
| Bootstrapping | Bootstrapping Pipeline | ADR-010; DISCOVER FF #9, #14 |
| Autonomy Level | Autonomy Policy | ADR-008 |
| Calibration | Calibration Gate | ADR-007 |
| Route (action) | Orchestrator Runtime | Domain model |
| Compose (action) | Orchestrator Tool Dispatch → Composition Validator | ADR-006 |
| Invoke (Dynamic) (action) | Orchestrator Tool Dispatch → Ensemble Engine | AS-1 |
| Summarize (action) | Result Summarizer Harness | ADR-004 |
| Compact (action) | Orchestrator Runtime | Essay §Context Management |
| Inject (action) | Serving Layer (function) | ADR-009 |
| Ingest (action) | Plexus Adapter | ADR-010 |
| Enrich (action) | Plexus Adapter (invocation) | ADR-010 |
| Query (action) | Orchestrator Tool Dispatch → Plexus Adapter | ADR-009 |
| Record (action) | Orchestrator Tool Dispatch → Plexus Adapter | ADR-009 |
| Calibrate (action) | Calibration Gate | ADR-007 |
| Stabilize (action) | (emergent — not owned by a single module) | AS-5 |
| Bootstrap (action) | Bootstrapping Pipeline | ADR-010 |
| Ensemble, Agent, AgentConfig, Model Profile, Inline Model, Dependency, Phase, Fan-Out, Input Key, Ensemble Reference, Ensemble Reference Graph, Depth, Depth Limit, Artifact, Child Executor, Immutable Infrastructure, Mutable State, Agent Discriminator; Load, Validate, Discriminate, Execute, Dispatch, Recurse, Fan Out, Gather, Select, Detect Cycles, Check Depth, Merge Profile | Ensemble Engine (existing) | Project-level domain model |

**Coverage check:** Every scoped concept (19) and action (13) is allocated. Every touched project-level concept stays with the existing Ensemble Engine. Stabilize is marked emergent — it is not owned by a module because AS-5 defines it as an emergent property of accumulated Quality Signals, not an explicit action.

---

## Dependency Graph

**Directed edges (A → B means A imports/calls B):**

```
Serving Layer ──────────────────────────────────▶ Session Registry
Serving Layer ──────────────────────────────────▶ Orchestrator Configuration
Serving Layer ──────────────────────────────────▶ Orchestrator Runtime
Serving Layer ───── (Phase 2 only) ─────────────▶ Plexus Adapter
Session Registry ───────────────────────────────▶ Ensemble Engine (profile lookup)
Orchestrator Runtime ───────────────────────────▶ Budget Controller
Orchestrator Runtime ───────────────────────────▶ Orchestrator Tool Dispatch
Orchestrator Runtime ───────────────────────────▶ Result Summarizer Harness
Budget Controller ──────────────────────────────▶ Session Registry
Orchestrator Tool Dispatch ─────────────────────▶ Ensemble Engine
Orchestrator Tool Dispatch ─────────────────────▶ Composition Validator
Orchestrator Tool Dispatch ─────────────────────▶ Plexus Adapter
Orchestrator Tool Dispatch ─────────────────────▶ Autonomy Policy
Orchestrator Tool Dispatch ─────────────────────▶ Calibration Gate
Result Summarizer Harness ──────────────────────▶ Ensemble Engine
Composition Validator ──────────────────────────▶ Ensemble Engine (shared public validator)
Autonomy Policy ────────────────────────────────▶ Session Registry
Calibration Gate ───────────────────────────────▶ Ensemble Engine
Calibration Gate ───────────────────────────────▶ Plexus Adapter
Bootstrapping Pipeline ─────────────────────────▶ Plexus Adapter
Bootstrapping Pipeline ─────────────────────────▶ Ensemble Engine
Plexus Adapter ─────────────────────────────────▶ (external — Plexus lib)
```

**Layering (inner → outer):**

| Layer | Modules | Rule |
|-------|---------|------|
| L0 — Core (existing) | Ensemble Engine | May not depend on any agentic-serving module |
| L1 — Domain Policy | Composition Validator, Budget Controller, Autonomy Policy, Calibration Gate, Plexus Adapter | May depend on L0 only |
| L2 — Runtime | Result Summarizer Harness, Orchestrator Tool Dispatch, Orchestrator Runtime | May depend on L0 and L1 |
| L3 — Entry | Serving Layer, Session Registry, Bootstrapping Pipeline, Orchestrator Configuration | May depend on L0, L1, and L2 |

**No cycles.** Verified by static inspection: every edge points from a higher layer to a same-or-lower layer. The only intra-layer dependencies are within L3 (Serving Layer → Session Registry; Serving Layer → Orchestrator Configuration) and do not form cycles.

**Fan-out warnings.** Orchestrator Tool Dispatch has five outbound edges (one per tool call class) — intentional per ADR-003; the fan-out *is* the closed tool set. Ensemble Engine is the highest-fan-in module (five agentic-serving modules depend on it), consistent with Layer 3 being the single shared execution substrate.

---

## Integration Contracts

### Serving Layer → Session Registry

**Protocol:** Synchronous function call at request entry.
**Shared types:** `SessionIdentity` (derivation-method-agnostic: may be client-supplied `user` field, hash of initial message prefix, or explicit session id header); `SessionState` (current Budget state, Autonomy Level, Calibration state if required).
**Error handling:** A request that fails Session identity resolution is treated as a new Session (cold start). Identity-resolution failures are not client errors.
**Owned by:** Session Registry defines the contract.

### Serving Layer → Orchestrator Runtime

**Protocol:** Asynchronous streaming; the Runtime yields SSE chunks.
**Shared types:** `SessionContext` (messages, tools array, session state); `OrchestratorChunk` (one of: content delta, internal tool call invocation-in-flight, internal tool call result, client tool call in final turn, completion).
**Error handling:** An Orchestrator Runtime exception becomes an SSE `error` chunk. The Runtime guarantees no partial state persists to the Session after a thrown exception unless explicitly committed.
**Owned by:** Orchestrator Runtime defines the contract.

### Orchestrator Runtime → Budget Controller

**Protocol:** Synchronous pre-iteration check.
**Shared types:** `BudgetCheck` (pass, or a typed exhaustion reason: turn or token).
**Error handling:** A failed check raises a typed `BudgetExhausted` event that the Runtime converts into a graceful session termination with explicit exhaustion signaling (ADR-005).
**Owned by:** Budget Controller defines the contract.

### Orchestrator Runtime → Orchestrator Tool Dispatch

**Protocol:** Synchronous tool invocation; returns the summarized/gated result.
**Shared types:** `InternalToolCall` (tool name from the fixed five, arguments); `ToolCallResult` (summarized result or typed error).
**Error handling:** A tool name outside the fixed five returns a `ToolCallResult` error (not an exception). The Runtime passes the error back to the orchestrator LLM as an observation (scenarios §Invocation outside the tool set).
**Owned by:** Orchestrator Tool Dispatch defines the contract.

### Orchestrator Tool Dispatch → Autonomy Policy

**Protocol:** Synchronous gate check before every tool dispatch.
**Shared types:** `AutonomyGateInput` (tool name, tool arguments, current Session Autonomy Level, tool-user persona flag); `AutonomyGateOutput` (allow, require_approval, or deny).
**Error handling:** `require_approval` surfaces an event to the operator via the visibility surface; `deny` is returned as a tool error to the orchestrator.
**Owned by:** Autonomy Policy defines the contract.

### Orchestrator Tool Dispatch → Ensemble Engine

**Protocol:** Existing `EnsembleExecutor.execute` (Layer 3 API, unchanged). Wrapped by Result Summarizer Harness on the return path.
**Shared types:** (existing Layer 3 types)
**Error handling:** Invariant 14 applies — runtime errors recorded; structural errors raised at load time.
**Owned by:** Ensemble Engine (existing).

### Orchestrator Tool Dispatch → Composition Validator

**Protocol:** Synchronous validation at composition time.
**Shared types:** `CompositionRequest` (proposed ensemble config, library search_dirs); `CompositionResult` (accepted and stored to local tier, or typed validation error naming the specific invariant violated).
**Error handling:** Validation failures are returned to the orchestrator as tool errors. No partial or pending ensemble state persists (ADR-006; scenarios §Composition that would introduce a reference-graph cycle).
**Owned by:** Composition Validator defines the contract.

### Composition Validator ↔ Ensemble Engine (shared validator routine)

**Protocol:** Shared public function call (`validate_ensemble_reference_graph` — new public API in `core/config/ensemble_config.py`).
**Shared types:** (existing `EnsembleConfig`, `AgentConfig` union, search_dirs list).
**Error handling:** Raises `ValueError` with cycle description on failure; returns `None` on success.
**Owned by:** Ensemble Engine owns the shared routine after extraction from private helpers. Both load-time (`EnsembleLoader.load_from_file`, `list_ensembles`) and composition-time callers use the same function (scenarios refactor 1-3; regression scenario "shared single routine").

### Orchestrator Tool Dispatch → Plexus Adapter

**Protocol:** Synchronous for `query_knowledge`; asynchronous-with-immediate-ack for `record_outcome`.
**Shared types:** `QueryRequest`, `QueryResult` (possibly empty — AS-8); `OutcomeRecord`, `RecordAck`.
**Error handling:** When Plexus is absent, both tools return well-formed empty/ack responses — no exception surfaces (scenarios §query_knowledge returns empty gracefully, §record_outcome writes asynchronously).
**Owned by:** Plexus Adapter defines the contract.

### Orchestrator Tool Dispatch → Calibration Gate

**Protocol:** Synchronous pre-invoke check and post-invoke Quality Signal attachment; interposed transparently on `invoke_ensemble` for ensembles in calibration.
**Shared types:** `CalibrationState` (in_calibration, trusted); `QualitySignal` (positive, negative, absent).
**Error handling:** A failing calibration check does not prevent invocation (ADR-007 clause 2) — it attaches the signal and returns the result normally.
**Owned by:** Calibration Gate defines the contract.

### Result Summarizer Harness → Ensemble Engine

**Protocol:** The Harness invokes a summarizer ensemble (configured primitive) via `EnsembleExecutor.execute`.
**Shared types:** (existing Layer 3 types)
**Error handling:** A summarizer failure is a tool failure — the original ensemble result is still persisted to its artifact (Invariant 9), but the orchestrator receives a typed summarization error as a tool result. Raw-output escape hatch (ADR-004) bypasses the Harness entirely for flagged ensembles.
**Owned by:** Result Summarizer Harness defines the contract.

### Serving Layer → `resolve_session_start_context` (internal function)

**Protocol:** Synchronous session-start hook; returns optional system-prompt augmentation. The hook is a typed function, not a module; call site and signature are the structural reservation.
**Shared types:** `SessionContext` (Session identity and state at start); `list[PromptFragment]` (empty in Phase 1; populated from Plexus Adapter in Phase 2).
**Error handling:** Injection failure in Phase 2 falls through to no injection rather than failing the session start.
**Owned by:** Serving Layer owns the function and its contract. **Phase 1 status:** returns `[]` unconditionally.

### Bootstrapping Pipeline → Plexus Adapter, Ensemble Engine

**Protocol:** Batch operation: reads from the library via the config manager, pushes file content to Plexus via the Adapter's ingestion path.
**Shared types:** `LibraryArtifactStream`, `IngestionAck` (AS-4: source material only — never LLM summaries).
**Error handling:** Ingestion is non-blocking; per-artifact failures are logged and the batch continues.
**Owned by:** Bootstrapping Pipeline defines the contract.

### Autonomy Policy → Session Registry; Calibration Gate → Session Registry; Budget Controller → Session Registry

**Protocol:** Read-only synchronous queries for Session state.
**Shared types:** `SessionState` (subsets scoped to consumer).
**Error handling:** Missing Session (identity unresolved) is treated as cold-session defaults for every consumer.
**Owned by:** Session Registry defines the contract.

---

## Fitness Criteria

| # | Criterion | Measure | Threshold | Derived From |
|---|-----------|---------|-----------|-------------|
| FC-1 | No module owns more than 5 scoped glossary entries as primary owner | Count rows per module in the Responsibility Matrix | ≤ 5 | God-class prevention (Essay §Guardrails; ARCHITECT principle) |
| FC-2 | Dependency edges point from higher layer to same-or-lower layer only | Static inspection of module imports against L0-L3 assignment | 0 violations | Layering rule (Dependency Graph) |
| FC-3 | No cycles in the dependency graph | Static cycle detection over the dependency edge list | 0 cycles | ARCHITECT principle |
| FC-4 | Orchestrator Runtime imports only Budget Controller, Orchestrator Tool Dispatch, and Result Summarizer Harness — no Plexus, no config, no Autonomy, no Calibration | Static import check | Exact match | Orchestrator LLM's mental model alignment; structural ADR-003 enforcement |
| FC-5 | Orchestrator Tool Dispatch has exactly five public entry points — one per committed tool | Static count of public dispatch methods | = 5 | ADR-003 (closed tool set) |
| FC-6 | Composition Validator and Ensemble Engine's load path call the same public validator function | Static check: single definition, two call sites | 1 definition, 2+ call sites | Scenarios refactor 1-3; ADR-006 negative consequence |
| FC-7 | Every Plexus-facing code path has a no-op fallback exercised by a stateless-mode test | Per-edge coverage of the stateless branch | 100% | AS-8 |
| FC-8 | `unsummarized-result` cannot reach the Orchestrator Runtime's context | Static check: Runtime imports `ToolCallResult`; no path from `EnsembleExecutor` to Runtime bypasses the Harness | 0 bypass paths | AS-7; ADR-004 |
| FC-9 | Session-start flow calls `resolve_session_start_context` exactly once; the function has a typed signature returning `list[PromptFragment]` | Static inspection | Exactly 1 call; signature present | ADR-009 (structural reservation via typed function) |
| FC-10 | Budget check executes before every ReAct iteration begins | Integration test covering the iteration-boundary contract | 100% of iterations | AS-3; ADR-005 |
| FC-11 | Autonomy Policy check executes before every Orchestrator Tool Dispatch | Integration test | 100% of dispatches | ADR-008 |
| FC-12 | Composed ensembles enter Calibration Gate transparently on `invoke_ensemble` during calibration | Integration test | 100% of in-calibration invocations | ADR-007 |
| FC-13 | Changing the orchestrator Model Profile requires touching only Orchestrator Configuration and Session start-logic — not Runtime, not Tool Dispatch | Diff inspection on a profile-swap change | No edits to Runtime or Tool Dispatch | ADR-011 |

All criteria are automatable via a combination of static import analysis, test coverage, and dependency-graph reconstruction. FC-1 is the only criterion that already passes for every module (maximum in the Responsibility Matrix is 5, held by Ensemble Engine for its cluster of project-level concepts — which are out-of-scope inheritance, not agentic-serving allocations).

---

## Test Architecture

### Boundary Integration Tests

Every dependency edge must have at least one integration test that exercises real data flow with real types on both sides. No mocking at the boundary under test.

| Edge | Integration Test | Verifies |
|------|-----------------|----------|
| Serving Layer → Session Registry | `test_serving_resolves_session_identity` | HTTP request with/without session continuity correlates to correct SessionState |
| Serving Layer → Orchestrator Runtime | `test_serving_streams_runtime_output` | SSE chunks flow end-to-end; client-tool round trip resumes same Session |
| Orchestrator Runtime → Budget Controller | `test_runtime_honors_budget_at_iteration_boundary` | Turn-limit and token-limit exhaustion both terminate at an iteration boundary |
| Orchestrator Runtime → Orchestrator Tool Dispatch | `test_runtime_dispatches_internal_tools_only` | Any tool name outside the five returns a tool error observation, not an exception |
| Orchestrator Runtime → Result Summarizer Harness | `test_runtime_never_sees_unsummarized_result` | `invoke_ensemble` returning a large dict reaches the Runtime as a summary; escape-hatch flag bypasses |
| Orchestrator Tool Dispatch → Ensemble Engine | `test_invoke_ensemble_executes_real_ensemble` | End-to-end ensemble execution with real `EnsembleExecutor` |
| Orchestrator Tool Dispatch → Composition Validator | `test_compose_ensemble_rejects_cycle` | Cyclic reference graph rejected at composition time |
| Orchestrator Tool Dispatch → Plexus Adapter | `test_query_knowledge_and_record_outcome_round_trip` | Real Plexus-active path; also run in stateless mode to verify no-op returns |
| Orchestrator Tool Dispatch → Autonomy Policy | `test_autonomy_gate_fires_before_every_dispatch` | Every tool path passes through the gate; baseline level allows invoke, composes, denies promotion |
| Orchestrator Tool Dispatch → Calibration Gate | `test_calibration_interposes_on_in_calibration_ensembles` | First N invocations run the check; (N+1)th does not |
| Composition Validator ↔ Ensemble Engine (shared) | `test_shared_validator_same_result_both_paths` | `validate_ensemble_reference_graph` returns identical outcome when called from load path and composition path on the same input (scenarios regression) |
| Result Summarizer Harness → Ensemble Engine | `test_summarizer_failure_preserves_artifact` | Summarizer exception: original artifact persists on disk, Runtime receives typed summarization error |
| Serving Layer's `resolve_session_start_context` | `test_session_start_context_is_empty_in_phase_1` | Function called once per session start, returns `[]`, never touches Plexus Adapter in Phase 1 |
| Bootstrapping Pipeline → Plexus Adapter | `test_bootstrap_pushes_source_material_not_summaries` | Ingested content is file content; no LLM-generated summaries in the push stream (AS-4) |
| Bootstrapping Pipeline → Ensemble Engine | `test_bootstrap_reads_library_via_config_manager` | Pipeline uses the existing config-manager path; respects tiered storage |
| Autonomy Policy → Session Registry | `test_autonomy_reads_session_state` | Autonomy Level resolved from per-session config, not global |
| Budget Controller → Session Registry | `test_budget_derives_cumulative_spend_from_session` | Token and turn spend sums are session-scoped, not request-scoped |
| Calibration Gate → Plexus Adapter | `test_calibration_persists_across_sessions_when_plexus_active` | Session 2 sees cleared calibration from Session 1 when Plexus is active; not when absent |

### Invariant Enforcement Tests

| Invariant | Enforcement Module | Test |
|-----------|--------------------|------|
| AS-1 (dynamic invocations outside reference graph) | Orchestrator Tool Dispatch | `test_invoke_ensemble_does_not_register_reference` — an invoke_ensemble call leaves the static reference graph unchanged |
| AS-2 (composed ensembles validated before loading) | Composition Validator | `test_compose_ensemble_validates_before_write` — no file written until validation passes |
| AS-3 (Budget is control-plane) | Budget Controller | `test_orchestrator_llm_cannot_observe_budget_state_in_context` |
| AS-4 (ingestion boundary is source material) | Plexus Adapter + Bootstrapping Pipeline | `test_ingestion_rejects_llm_summary_marker` (source-material assertion by type) |
| AS-5 (quality signals govern stabilization, not frequency) | Calibration Gate | `test_frequency_without_quality_does_not_trust` |
| AS-6 (compose from existing primitives only) | Composition Validator + Orchestrator Tool Dispatch | `test_compose_ensemble_rejects_new_script_or_profile` |
| AS-7 (result summarization is a correctness requirement) | Result Summarizer Harness | `test_runtime_never_sees_unsummarized_result` (also FC-8) |
| AS-8 (Plexus is optional) | Plexus Adapter + Serving Layer session-start function | `test_all_operations_work_with_plexus_absent` — full coverage of every Plexus-facing edge, including the session-start function in Phase 2 |
| Invariant 5 (cross-ensemble acyclicity) | Composition Validator + Ensemble Engine | `test_compose_ensemble_rejects_cycle` (also FC-6 regression) |
| Invariant 7 (static ensemble references) | Orchestrator Tool Dispatch | `test_invoke_ensemble_uses_existing_name_no_template_expression` |
| Invariant 9 (child artifacts nested) | Ensemble Engine (existing) + Result Summarizer Harness | `test_summarizer_failure_preserves_full_artifact` |
| Invariant 13 (execution resilient) | Ensemble Engine (existing) | (existing test suite) |

### Test Layers

- **Unit.** Verify logic within a single module. Mocks are acceptable for neighbor modules. Every module has unit tests for its core state transitions (e.g., Budget Controller's exhaustion check; Calibration Gate's N-invocation transition; Composition Validator's cycle detection on synthetic graphs).
- **Integration.** Verify real data flow across module boundaries. No mocks at the boundary under test. All 18 edges in the Boundary Integration Tests table are integration tests.
- **Acceptance.** Verify scenarios end-to-end using real wiring. The scenarios in `scenarios.md` map to acceptance tests. The scenario "Tool user completes a task against the stateless orchestrator" is the happy-path acceptance test; "Session terminates gracefully on turn limit exhaustion" and "Composition with ensemble-to-ensemble reference passes validation" are the representative stress paths.

---

## Roadmap

See [`./roadmap.md`](./roadmap.md) for the current roadmap — work packages, classified dependencies, transition states, and open decision points.

---

## Design Amendment Log

| # | Date | What Changed | Trigger | Provenance | Status |
|---|------|-------------|---------|------------|--------|
| — | 2026-04-20 | Initial system design | ARCHITECT phase | All inputs | Current |
| 1 | 2026-04-20 | Demote Context Injection Stage from module to typed function `resolve_session_start_context` owned by Serving Layer | ARCHITECT reflection-time Grounding Reframe (susceptibility-snapshot-agentic-serving-architect.md Item 1) | ADR-009; single-agent-paradigm survey (Claude Code, OpenCode, claw-code, OpenHands) | Current |
| 2 | 2026-04-20 | Mark Client Tool Surface Commitment (Option C) as scenario-gated — WP-F does not start until stress scenarios are written and C is shown to handle them | ARCHITECT reflection-time Grounding Reframe (Item 2) | Interaction specs open boundary; user direction "a step that direction" of OpenCode support | Current |
