# Interaction Specifications: Agentic Serving

**Derived from:** `product-discovery.md` (Stakeholder Map, Jobs and Mental Models, Product Vocabulary)

**Complements:** `scenarios.md` (business-rule behavior)

**Method caveat.** The derivation from stakeholder models to task decomposition is an open design problem in RDD v0.7.3. These specifications are a best-effort interpretation of the stakeholder jobs and mental models from product discovery, not a systematic derivation. They are concrete enough to create a playable surface for `/rdd-play` and abstract enough to survive UI or protocol refinements in ARCHITECT and BUILD.

**Open boundary.** One interaction boundary is not yet pinned: how the Orchestrator Agent handles the *client's* tool definitions (the tools declared by an agentic coding tool in its `/v1/chat/completions` request). The current ADRs specify the orchestrator's internal tool surface (ADR-003) but do not commit on whether client-declared tools are passed through to the orchestrator's LLM, held by the orchestrator for delegation to invoked ensembles, or rejected. This affects the Tool User interaction below — the placeholder text reflects the ambiguity rather than hiding it. Decision needed in ARCHITECT or a follow-up DECIDE mini-cycle.

---

## Stakeholder: Tool User

**Super-Objective:** Complete coding work through an agentic coding tool without knowing or caring what sits behind the endpoint, while retaining autonomy from a single provider.

### Task: Configure the agentic coding tool to use llm-orc as its backend

**Interaction mechanics:** The operator (or the tool user themselves when they are also the operator) sets the tool's API endpoint to the llm-orc serving layer URL and selects a model from the `/v1/models` list. The listed model names are the orchestrator's configured profile (ADR-011). From this point forward, all requests from the coding tool flow through the Orchestrator Agent behind the endpoint. The tool user does not configure ensembles, profiles, or Plexus at this layer.

### Task: Complete a coding task through the tool's normal workflow

**Interaction mechanics:** The tool user interacts with their agentic coding tool exactly as they would with any other LLM backend — asking questions, requesting code changes, reviewing suggestions, iterating. The Orchestrator Agent, behind the endpoint, runs its ReAct loop and delegates to ensembles via `invoke_ensemble` and (where appropriate) `compose_ensemble` while the tool user waits. The tool user observes responses, tool calls, and streamed content in the shape their tool expects.

*Open: the orchestrator's interaction with the client's tool definitions (bash, file-edit, etc.) is not pinned by current ADRs. This interaction's exact mechanics resolve once that boundary is decided.*

### Task: Observe what the orchestrator is doing (visibility-configurable)

**Interaction mechanics:** When the operator has configured visibility surfacing (see Open Question #2 in the domain model — visibility format is unresolved), the tool user sees composition events, ensemble invocations, or other orchestrator activity inline in the tool's response stream. When visibility is not surfaced, these events are invisible to the tool user and only visible to the operator through whatever visibility surface the operator has enabled. For the tool user, this ranges from "endpoint is a model" (no visibility) to "endpoint is an orchestrator" (composition events visible) depending on the Autonomy Level and visibility configuration.

### Task: Experience budget exhaustion cleanly

**Interaction mechanics:** When the Session reaches its turn limit or token limit, the tool user receives a final response that names budget exhaustion as the termination cause. The tool user can then start a new session (which gets a fresh Budget) or request that the operator adjust Budget defaults.

---

## Stakeholder: Ensemble Author / Operator

**Super-Objective:** Maintain a library of ensembles, profiles, and scripts that the orchestrator uses effectively, while observing how the system uses that library and improving it through tinkering and organic stabilization.

### Task: Bootstrap Plexus from the existing library

**Interaction mechanics:** The operator runs an llm-orc bootstrapping command that pushes the contents of the library — ensemble YAML files, script source, profile configurations, execution artifacts — into Plexus as source material (AS-4, ADR-010). Ingestion runs; enrichment runs asynchronously. The operator does not wait for enrichment to complete before running sessions; the knowledge graph's value at session start grows as enrichment catches up. Bootstrapping is a push operation, not a Plexus-pull.

### Task: Run the serving layer for client consumption

**Interaction mechanics:** The operator starts the llm-orc server with serving layer enabled, chooses the orchestrator's Model Profile (ADR-011), sets Budget defaults (ADR-005), and sets the default Autonomy Level (ADR-008). One or more agentic coding tools can then connect via the endpoint. The operator's responsibility after startup is observation, not per-request intervention (subject to Autonomy Level).

### Task: Observe orchestrator behavior during sessions

**Interaction mechanics:** The operator watches sessions through visibility surfaced inline in the tool user's response stream — composition events, calibration outcomes, and other orchestrator-internal signals render as `[kind: {json}]` narration on `delta.content` so vanilla OpenAI-compat clients (OpenCode, Roo Code, Cline) display them inline in the assistant message (OQ #2 resolved during WP-E build, 2026-04-22). What the interaction commits to is that these events are *observable in the conversation*, not opaque. Operator-only tooling surfaces (SSE comments, dedicated events endpoint) can layer on as a second audience-specific surface without changing the inline narration shape.

### Task: Review local-tier compositions and decide on promotion

**Interaction mechanics:** At default Autonomy Level (ADR-008), the Orchestrator Agent produces composed ensembles in the local tier automatically, subject to calibration (ADR-007). The operator reviews these compositions and their accumulated Quality Signals through a library-browsing interface (CLI, web UI, or equivalent — existing llm-orc `list_ensembles` primitives extend to this). Promotion from local to global or library tier requires explicit operator approval — the orchestrator cannot self-promote at default Autonomy Level. The operator's decision is informed by Quality Signals, not frequency (AS-5).

### Task: Tune Budget, Autonomy Level, and orchestrator profile

**Interaction mechanics:** The operator adjusts serving layer configuration — default Budget sizes, default Autonomy Level, orchestrator Model Profile — through configuration files or CLI commands. Changes take effect on new Sessions; active Sessions continue under their existing configuration (ADR-011 session-boundary discipline). The operator can also set per-session overrides within configured bounds (e.g., allowing a specific tool user a tighter Autonomy Level for production work).

### Task: Tinker — adjust system behavior through experimentation

**Interaction mechanics:** The operator uses their own agentic coding tool pointed at their own llm-orc server, collapsing the tool-user and operator roles into one person (product discovery assumption inversion #3). They observe orchestrator decisions in the tool's response, note patterns they want to change, edit ensemble YAML or profile configurations outside the agentic flow, and run the next session to see the effect. Tinkering is a cycle of observation → edit → re-run, not a one-shot configuration task.

---

## Stakeholder: Orchestrator LLM

**Super-Objective:** Resolve the tool user's task effectively within the Session Budget by routing to existing ensembles or composing new ones from the library's primitives, informed by the knowledge graph when available.

### Task: Receive a task and orient on it

**Interaction mechanics:** At the start of each ReAct iteration, the Orchestrator Agent considers the incoming messages, consults the library (`list_ensembles` when it needs to know what is available), and, when Plexus is active, may call `query_knowledge` to check whether similar tasks have been handled before (ADR-009 Phase 1 tool-first). Orientation is done through tool calls, not through context injection in this phase (injection is deferred per ADR-009).

### Task: Route an existing ensemble to a task

**Interaction mechanics:** When the orchestrator identifies an ensemble from the library whose Routing Decision history or stated capability matches the task, it emits an `invoke_ensemble` tool call with the task input. The Ensemble Engine runs the ensemble; the Result Summarizer condenses the output (ADR-004); the summary returns to the orchestrator's context. The orchestrator uses that summary to continue reasoning.

### Task: Compose a new ensemble when nothing existing fits

**Interaction mechanics:** When no existing ensemble fits the task, the orchestrator emits a `compose_ensemble` tool call specifying an ensemble configuration drawn from existing primitives (profiles, scripts, ensembles — ADR-006 full palette). Composition-time validation runs (AS-2, ADR-006); if validation fails, the orchestrator receives an error observation and may retry with a different composition, invoke an existing ensemble despite imperfect fit, or escalate via the response. The newly composed ensemble is written to the local tier and enters calibration (ADR-007). The orchestrator may then immediately invoke it.

### Task: Record outcomes as work proceeds

**Interaction mechanics:** At decision points (after an ensemble invocation completes, after a calibration check produces a Quality Signal, at the end of a task), the orchestrator emits `record_outcome` calls that write Routing Decisions, Quality Signals, and outcome metadata to the knowledge graph (ADR-009 — a no-op when Plexus is absent, not an error). Recording is non-blocking; the orchestrator does not wait for enrichment (ADR-010).

### Task: Stay within Budget

**Interaction mechanics:** The orchestrator does not directly observe the Budget's remaining capacity — the control plane enforces it at each ReAct iteration boundary (AS-3, ADR-005). The orchestrator's interaction with Budget is observational through prior-outcome records accessible via `query_knowledge` (costs of similar sessions), not negotiable. When Budget is exhausted, the control plane terminates the Session — the orchestrator does not emit a "final" turn; its last complete turn becomes the response.

### Task: Terminate the task cleanly

**Interaction mechanics:** When the orchestrator determines the task is complete, it emits a final message with `finish_reason: stop` (OpenAI-compatible protocol). No tool call is in flight at termination. If the orchestrator reasons itself into a dead end, it emits a message explaining the impasse and stops, rather than looping until Budget exhaustion — this is a quality behavior the orchestrator's Model Profile should support, not a guarantee of the framework.
