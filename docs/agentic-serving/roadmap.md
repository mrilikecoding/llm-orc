# Roadmap: Agentic Serving

**Generated:** 2026-04-20
**Derived from:** `system-design.md` (v1.0), ADRs 001-011, scenarios.md, interaction-specs.md

This roadmap expresses the sequencing landscape for building agentic serving — what depends on what, where the builder has a choice, and which coherent intermediates are worth pausing at. It does not prescribe a build order. Work package order within each dependency band is a build-time decision.

---

## Work Packages

### WP-A: Cycle-validator extraction (retrofit debt)

**Objective:** Extract `EnsembleLoader._validate_cross_ensemble_cycles` and `_build_reference_graph` to a public function so composition-time and load-time validation share a single routine.

**Changes:**
- `core/config/ensemble_config.py` — add public `validate_ensemble_reference_graph(name, agents, search_dirs) -> None`; rewrite `EnsembleLoader.load_from_file` to call it; rewrite `EnsembleLoader.list_ensembles` to pass its directory as `search_dirs` so the check actually fires.
- `services/handlers/validation_handler.py` — wire `_collect_validation_errors` through the public function with real `search_dirs` from the config manager.

**Scenarios covered:** refactor scenarios 1, 2, and regression 3 (scenarios.md §Structural Debt Remediation).

**Dependencies:** None.

**Participating modules:** Ensemble Engine (existing). Consistent with WP scope.

---

### WP-B: Serving foundation — endpoints, Session identity, configuration surface

**Objective:** Stand up the OpenAI-compatible serving surface and the per-Session state that every downstream module reads. Not yet a ReAct loop — this is the plumbing the loop slots into.

**Changes:**
- New **Serving Layer** module: `/v1/chat/completions` (streaming and non-streaming), `/v1/models`, SSE chunking, tool-call formatting.
- New **Session Registry** module: Session identity resolution from request features (Phase 1: message-history derivation with optional client-supplied correlation via OpenAI `user` field); cumulative turn and token accounting.
- New **Orchestrator Configuration** module: per-session resolution of orchestrator Model Profile, Budget defaults, Autonomy default, Plexus enablement flag; operator-set bounds on per-request overrides.
- Typed session-start function `resolve_session_start_context(session: SessionContext) -> list[PromptFragment]` in Serving Layer, called exactly once at session start (ADR-009 Phase 2 reservation; Phase 1 body returns `[]`). Replaces former WP-J.
- Integrates with existing FastAPI app in `web/server.py` via router inclusion.

**Scenarios covered:** foundation for everything; direct coverage of `/v1/models` listing (scenarios.md §Orchestrator tool set is exactly the committed set — the `/v1/models` side of it). FC-9 (`resolve_session_start_context` call-site check) satisfied on completion.

**Dependencies:** None.

**Participating modules:** Serving Layer, Session Registry, Orchestrator Configuration. Consistent with WP scope.

---

### WP-C: ReAct core — Runtime, Tool Dispatch, Budget Controller

**Objective:** Minimum viable orchestrator loop behind the serving surface. Wires in the `invoke_ensemble` and `list_ensembles` tools with Budget enforcement at the iteration boundary. Leaves `compose_ensemble`, `query_knowledge`, and `record_outcome` as named entry points that return typed "not-yet-wired" errors — enough to satisfy the closed-set property (ADR-003 and FC-5) from day one.

**Changes:**
- New **Orchestrator Runtime** module: ReAct loop, Conversation Compaction, Routing Decision generation.
- New **Orchestrator Tool Dispatch** module: five-entry public surface, allowlist rejection, routing of `invoke_ensemble` and `list_ensembles` to Ensemble Engine; the other three entries return typed not-yet-wired errors.
- New **Budget Controller** module: per-iteration check; turn-count and token-spend derivation via Session Registry; graceful termination signalling.
- Serving Layer → Orchestrator Runtime handoff wired.

**Scenarios covered:** scenarios.md §Session Lifecycle (Tool user completes a task against the stateless orchestrator; turn-limit graceful termination; token-limit graceful termination); §Orchestrator Tool Surface (Orchestrator tool set is exactly the committed set; Invocation outside the tool set is rejected).

**Dependencies:**
- WP-B (hard) — Runtime receives its Session from the Serving Layer handoff.

**Participating modules:** Orchestrator Runtime, Orchestrator Tool Dispatch, Budget Controller. Session Registry (existing WP-B) is read-only here.

---

### WP-D: Result Summarizer Harness

**Objective:** Structurally interpose the summarizer on the `invoke_ensemble` return path so unsummarized ensemble results can never reach the Orchestrator Runtime's context.

**Changes:**
- New **Result Summarizer Harness** module.
- Orchestrator Tool Dispatch's `invoke_ensemble` path wrapped by the Harness on the return leg.
- Raw-output escape-hatch flag honored on the ensemble config (ADR-004).
- A default summarizer ensemble configured and promoted to the library tier (primitive, not code).

**Scenarios covered:** scenarios.md §Orchestrator Tool Surface (Ensemble result is summarized before entering orchestrator context; Raw-output escape hatch is explicit).

**Dependencies:**
- WP-C (hard) — interposes on Tool Dispatch.

**Participating modules:** Result Summarizer Harness, Orchestrator Tool Dispatch (edit). Consistent with WP scope.

---

### WP-E: Autonomy Policy

**Objective:** Interpose the Autonomy Policy gate before every Orchestrator Tool Dispatch. Enforces the baseline level, exposes tighter and looser configurations, and surfaces composition events when the configured level requires it.

**Changes:**
- New **Autonomy Policy** module.
- Orchestrator Tool Dispatch edit: every dispatch call passes through Autonomy Policy.
- Visibility surfacing hook (Phase 1 form: structured events in the SSE stream; OQ #2 leaves final form open).
- AS-6 hard rule: no configuration level can enable script or profile authorship.

**Scenarios covered:** scenarios.md §Autonomy and Promotion (Default Autonomy Level permits invocation, permits composition, gates promotion; Tool user without operator role observes composition events; Pure tool-user session at default Autonomy Level experiences silent composition; Script authorship is never permitted at any Autonomy Level).

**Dependencies:**
- WP-C (hard) — interposes on Tool Dispatch.

**Participating modules:** Autonomy Policy, Orchestrator Tool Dispatch (edit). Consistent with WP scope.

---

### WP-F: Client-tool turn-boundary delegation

**Objective:** Implement the Client Tool Surface Commitment — the orchestrator closes turns with `finish_reason: tool_calls` when a task step needs a client-side action, and the next `/v1/chat/completions` resumes the same Session with the client's `tool`-role messages as observations.

**Changes:**
- Serving Layer edit: translate client-declared `tools[]` into a response-surface set that the Orchestrator Runtime can emit against; emit `finish_reason: tool_calls` with `tool_calls[]` for client tools; accept `role: tool` messages in subsequent requests and feed them into Session Registry's continuation logic.
- Session Registry edit: Session continuity across client-tool round trips (budget continues to accumulate; Autonomy state persists).
- Orchestrator Runtime edit: recognize when its turn produces a client-tool request (distinct from an internal tool call) and emit the final-turn signal.

**Scenarios covered:** this work package *generates new BDD scenarios* that must be added to scenarios.md before implementation (see Open Decision Points §1). Target behaviors: OpenCode sends `tools[bash, file_read, file_edit]`; orchestrator routes a task, needs a file read, emits final turn with tool_calls for file_read; client executes; next request resumes the same Session; turn count and token spend carry across.

**Dependencies:**
- WP-B (hard) — Serving Layer and Session Registry own the boundaries.
- WP-C (hard) — Orchestrator Runtime owns the final-turn emission decision.

**Participating modules:** Serving Layer (edit), Session Registry (edit), Orchestrator Runtime (edit). Consistent with WP scope.

---

### WP-G: Composition — compose_ensemble, Composition Validator

**Objective:** Wire the `compose_ensemble` tool through the Composition Validator to the shared cycle-validator routine and the Ensemble Engine's local-tier write path.

**Changes:**
- New **Composition Validator** module.
- Orchestrator Tool Dispatch edit: `compose_ensemble` entry point is now wired (previously returned not-yet-wired error).
- Local-tier write path for composed ensembles (Ensemble Engine local tier, via existing config manager).

**Scenarios covered:** scenarios.md §Ensemble Composition with Validation (six scenarios) plus the integration scenario (composed ensemble validates using the same logic as the load path — closes FC-6 as a regression test).

**Dependencies:**
- WP-A (hard) — needs `validate_ensemble_reference_graph` to exist as a public function.
- WP-C (hard) — needs the dispatch entry point to exist.

**Participating modules:** Composition Validator, Orchestrator Tool Dispatch (edit), Ensemble Engine (existing, called through). Consistent with WP scope.

---

### WP-H: Calibration Gate

**Objective:** Interpose the Calibration Gate on `invoke_ensemble` for composed ensembles in their first N invocations. Generate Quality Signals; attach to Routing Decisions; transition to trusted on clear.

**Changes:**
- New **Calibration Gate** module.
- Orchestrator Tool Dispatch edit: `invoke_ensemble` wrapped by Calibration Gate for in-calibration ensembles.
- Session-scoped state in stateless mode (lives in Session Registry's Session state).
- Default check mechanism (configurable; default implementation deferred to build — may itself be an ensemble).

**Scenarios covered:** scenarios.md §Calibration of Composed Ensembles (first N invocations result-checked; transition to trusted on positive signals; fails-to-clear on negative signals; session-scoped when Plexus absent; persists across sessions when Plexus active).

**Dependencies:**
- WP-G (implied) — calibration is only meaningful for composed ensembles; a pre-composed test fixture ensemble could unblock parallel work.
- WP-I (implied) — cross-session persistence requires Plexus; stateless calibration works without it.

**Participating modules:** Calibration Gate, Orchestrator Tool Dispatch (edit), Session Registry (edit for session-scoped state). Consistent with WP scope.

---

### WP-I: Plexus Adapter (tool-first)

**Objective:** Wire `query_knowledge` and `record_outcome` through the Plexus Adapter with graceful no-op fallbacks when Plexus is absent. Every Plexus-facing path tested in stateless mode (FC-7).

**Changes:**
- New **Plexus Adapter** module.
- Orchestrator Tool Dispatch edit: `query_knowledge` and `record_outcome` entry points are now wired.
- Integration with Plexus lib via MCP (plexus MCP server is available — schemas are deferred).

**Scenarios covered:** scenarios.md §Plexus Integration (query_knowledge returns empty gracefully; query_knowledge returns enriched content; record_outcome writes asynchronously; Ingestion accepts source material; ReAct loop remains responsive while enrichment lags); §Session Lifecycle (Four-layer stack operates with Plexus present).

**Dependencies:**
- WP-C (hard) — needs the dispatch entry points to exist.

**Participating modules:** Plexus Adapter, Orchestrator Tool Dispatch (edit). Consistent with WP scope.

---

### WP-J: Bootstrapping Pipeline

**Objective:** Operator-triggered batch ingestion of the library (ensemble YAML, scripts, profiles, execution artifacts) into Plexus as source material (AS-4).

**Changes:**
- New **Bootstrapping Pipeline** module.
- CLI command for triggering bootstrap.
- Uses Plexus Adapter's ingestion path.

**Scenarios covered:** scenarios.md §Cost and Quality Experimentation (Bootstrapped graph shortens time-to-first-useful-query — testable OQ #4).

**Dependencies:**
- WP-I (hard) — needs Plexus Adapter.

**Participating modules:** Bootstrapping Pipeline, Plexus Adapter (called through), Ensemble Engine (reads library via existing config manager). Consistent with WP scope.

---

## Dependency Graph

```
WP-A (extract cycle validator)
   │
   └─ hard ─▶ WP-G

WP-B (serving foundation; includes resolve_session_start_context)
   │
   ├─ hard ─▶ WP-C
   └─ hard ─▶ WP-F

WP-C (ReAct core)
   │
   ├─ hard ─▶ WP-D
   ├─ hard ─▶ WP-E
   ├─ hard ─▶ WP-F
   ├─ hard ─▶ WP-G
   ├─ implied ─▶ WP-H
   └─ hard ─▶ WP-I

WP-G (composition) ─ implied ─▶ WP-H
WP-I (Plexus adapter) ─ implied ─▶ WP-H (for cross-session calibration persistence)
WP-I ─ hard ─▶ WP-J
```

**Classification key:**

- **Hard dependency:** structural necessity — the downstream WP's code imports, extends, or requires the upstream WP's output. The builder has no choice.
- **Implied logic:** suggested ordering — building the upstream first is simpler because the downstream references concepts it defines, but a skilled builder could stub the references and fill in later.
- **Open choice:** genuinely independent — build either first. (WP-A and WP-B have no dependency between them; either can start. Similarly WP-D, WP-E, WP-F, WP-I are all mutually independent once their hard deps are met.)

---

## Transition States

### TS-1: Stateless orchestrator serving OpenCode (after WP-A, WP-B, WP-C, WP-D, WP-E, WP-F)

A complete, useful intermediate. An operator points OpenCode at the llm-orc endpoint and runs an RDD phase through it. The orchestrator routes tasks to existing library ensembles, summarizes results, enforces Budget, and delegates client-side actions (bash, file edits) to OpenCode at turn boundaries. No self-composition, no Plexus, no calibration. This is the minimum coherent system that satisfies the vision named in ARCHITECT: *"I can use OpenCode and run a version of this RDD pipeline with it."*

At TS-1, the fitness criteria satisfied are: FC-1 through FC-5, FC-8, FC-10, FC-11, FC-13. FC-6 (shared validator), FC-7 (Plexus no-op coverage), FC-9 (injection stage exists), and FC-12 (calibration interposition) are satisfied later.

### TS-2: Stateless baseline complete (after TS-1 + WP-G + WP-H)

The orchestrator can now compose new ensembles from existing library primitives, validate them, and calibrate them within the session. Still no Plexus — calibration is session-scoped; cross-session trust does not persist. This is the complete **baseline product** as defined by ADR-002 Layer 1-3 and AS-8. Fitness criteria FC-6 and FC-12 now satisfied.

### TS-3: Four-layer stack with Phase 1 Plexus integration (after TS-2 + WP-I + WP-J)

Full architecture live. `query_knowledge` and `record_outcome` flow to Plexus; library bootstrapping is available. Calibration now persists across sessions. Fitness criterion FC-7 satisfied (FC-9 was satisfied at WP-B via `resolve_session_start_context`). Phase 2 injection remains deferred per ADR-009 — when it lands, the change is a function body, not a structural addition.

---

## Open Decision Points

1. **Client-tool delegation scenarios in `scenarios.md`** *(Option C/D grounding — must resolve before WP-F)*. The Client Tool Surface Commitment in the system design resolves an open boundary from interaction-specs but is **scenario-gated**: WP-F does not start until stress scenarios are written into `scenarios.md` that exercise the turn-boundary vs. mid-execution distinction. Target scenarios include: (a) OpenCode sends `tools[bash, file_read, file_edit]` and the orchestrator emits a final-turn delegation; (b) Session turn count and token spend accumulate across a client-tool round trip; (c) an ensemble whose first agent needs a file from the client's filesystem before its second agent proceeds — does C handle this, or does mid-execution callback become necessary?; (d) a compose-then-invoke flow where the composed ensemble, mid-execution, needs a client-tool result the orchestrator didn't know to request in advance. If any scenario requires mid-execution callback, the Commitment is amended (C + hybrid, or Option D). Builder should request a short DECIDE mini-cycle or inline scenario-write before starting WP-F.

2. **Visibility form (OQ #2).** WP-E's composition-event surfacing currently defaults to structured SSE events. If the operator audience decides visibility belongs in a dashboard or structured log instead (or in addition), WP-E's surface changes. Decision can land during build — surfacing the events structurally is the necessary condition; the presentation format is a swap.

3. **Budget specific numbers (ADR-005 defers to build).** WP-C defaults need concrete turn and token limits. The outer anchor is "comparable to running an RDD phase." Concrete numbers are a tuning decision informed by observed rollout, not an architecture decision.

4. **Calibration N (ADR-007 defers to build).** WP-H needs a default N. The check mechanism's cost and value drive the number; no architectural constraint.

5. **Session identity mechanism.** WP-B defaults to message-history-derivation with optional client-supplied correlation via the OpenAI `user` field. If Autonomy tightening or multi-client deployments make this insufficient, a custom header or session-id cookie becomes necessary. Build-time decision; the Session Registry contract accommodates either.

6. **`record_outcome` payload schema.** WP-I writes outcomes to Plexus but ADR-009 defers the tool's argument schema. Builder should pick a minimum payload (routing decision + quality signal + free-form context) that Phase 2 injection can later read without breaking.

7. **Visibility surface for conductor-ceiling observations (OQ #6).** Not a decision point for any WP directly, but an observability requirement that WP-E and WP-I should consider together — the orchestrator's routing-decision stream is a window into whether orchestration depth is reachable by smaller models.

---

## Completed Work Log

### WP-A: Cycle-validator extraction (retrofit debt) — 2026-04-20

**Commits:**
- `8a0f5d6` refactor: extract validate_ensemble_reference_graph to public function
- `0980323` fix: surface cross-ensemble cycles through list_ensembles and ValidationHandler

**Outcome.** Public `validate_ensemble_reference_graph(name, agents, search_dirs)` now lives in `core/config/ensemble_config.py`. Three call sites share it: `EnsembleLoader.load_from_file`, `EnsembleLoader.list_ensembles` (via `search_dirs=[directory]`), and `ValidationHandler._collect_validation_errors` (via `config_manager.get_ensembles_dirs()`). `EnsembleLoader._find_ensemble_in_dirs` retained as a thin delegate to the module-level helper so `core/execution/ensemble_execution.py` continues to resolve through the single shared implementation.

**Scenarios covered:** scenarios.md §Structural Debt Remediation refactor 1, refactor 2, and the regression scenario (shared single routine).

**Fitness criteria status:** FC-6 satisfied — 1 definition, 3 call sites; load-time and MCP/web validate-time behavior cannot diverge.

**Unblocks:** WP-G (compose_ensemble wires in as the fourth call site).

**Debt surfaced (not addressed in WP-A scope):** `core/execution/ensemble_execution.py:808` reaches into `EnsembleLoader._find_ensemble_in_dirs` (still underscore-prefixed). The delegate preserves the call; a later cleanup can rewire the executor to the module-level helper directly if the underscore leak becomes a problem.

### WP-B Group 5: SSE streaming skeleton + tool-call formatting — 2026-04-21

**Commit:** `3db8eb3` feat: add SSE streaming skeleton and OrchestratorChunk types (WP-B Group 5)

**Outcome.** `/v1/chat/completions` with `stream=true` now returns a `StreamingResponse` with `text/event-stream` media type. The stream opens with the OpenAI role-delta convention, forwards chunks from a stubbed `_orchestrator_stream_handoff`, and terminates with `data: [DONE]\n\n`. The stub yields a single `Completion(finish_reason="stop")` — the minimum chunk sequence that satisfies the Serving Layer → Orchestrator Runtime integration contract. WP-C replaces the stub with the real ReAct loop.

**New modules.**
- `src/llm_orc/agentic/orchestrator_chunk.py` — typed integration contract between Orchestrator Runtime and Serving Layer. Six frozen-dataclass variants: `ContentDelta`, `Completion`, `ClientToolCall` (+ `ToolCallInvocation`), `InternalToolCallInFlight`, `InternalToolCallResult`, `ErrorChunk`, joined in the `OrchestratorChunk` union alias.
- `src/llm_orc/web/api/sse_format.py` — `OpenAiSseFormatter` class. `start_assistant_turn()` emits the role-delta opener; `format(chunk)` dispatches per variant to framed OpenAI `chat.completion.chunk` bytes (or `b""` for deferred-visibility internal tool-call chunks per OQ #2); `done()` emits `data: [DONE]\n\n`.

**Edits.**
- `v1_chat_completions.py` — removed the Group 4 HTTP 400 rejection of `stream=true`. Extracted `_resolve_context(request)` so streaming and non-streaming share pre-handoff work (identity resolution, session-start cache). Added `_stream_completion` async generator and `_orchestrator_stream_handoff` stub. Router gets `response_model=None` to permit the `dict | StreamingResponse` return.

**Scenarios covered.** `scenarios.md` does not explicitly claim Group 5; the work is integration-contract plumbing. FC-9 preservation under streaming is the load-bearing fitness criterion, covered by two new tests (`test_streaming_request_fires_session_start_exactly_once_per_session`, `test_streaming_and_non_streaming_share_session_start_cache`).

**Fitness criteria status.**
- FC-4 (Runtime import surface): new chunk types and formatter add zero imports that would leak into a future Runtime dependency tree. Runtime will import from `orchestrator_chunk` (neutral types) only.
- FC-9 (`resolve_session_start_context` called exactly once per session): preserved under streaming via `_resolve_context` + the existing cache.
- FC-5 (exactly five dispatch entry points): unchanged — Runtime isn't built yet.

**Test coverage delta.** +14 tests (9 new SSE formatter unit tests, 4 streaming endpoint tests, 2 FC-9-under-streaming integration tests, −1 Group 4 rejection test superseded). Full suite: 2141 passed, 91.21% coverage.

**Unblocks:** WP-B Group 6 (integration verification — session identity across requests, full FC-9 static inspection pass); WP-C (`_orchestrator_stream_handoff` stub is the body swap point); WP-F (`ClientToolCall` + `ToolCallInvocation` types and their formatter case already exist).
