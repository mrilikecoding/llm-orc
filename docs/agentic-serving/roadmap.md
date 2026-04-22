# Roadmap: Agentic Serving

**Generated:** 2026-04-20
**Derived from:** `system-design.md` (v1.0), ADRs 001-011, scenarios.md, interaction-specs.md

This roadmap expresses the sequencing landscape for building agentic serving — what depends on what, where the builder has a choice, and which coherent intermediates are worth pausing at. It does not prescribe a build order. Work package order within each dependency band is a build-time decision.

---

## Work Packages

> **WP-A, WP-B, WP-C, and WP-D are complete.** See [Completed Work Log](#completed-work-log) at the end of this document for scope, commits, and outcomes. The active section below lists only upcoming or in-progress work.

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

### WP-B Group 6: Integration verification — Serving Layer → Session Registry edge + FC-9 static inspection — 2026-04-21

**Commits:** (this change)

**Outcome.** WP-B closes out with verification-only work. No new production code — two test surfaces added:

1. **`TestServingResolvesSessionIdentity`** (5 integration tests in `tests/unit/web/test_api_v1_chat_completions.py`). Covers the Test Architecture table's `Serving Layer → Session Registry` edge — `test_serving_resolves_session_identity`:
   - Same `user` field across two requests resolves to a single `SessionState` in the registry.
   - Mutation through the retained `SessionState` between requests is visible to the follow-up request (the lifecycle-sequence check at the HTTP boundary — mirrors the unit-level `test_caller_mutation_visible_through_subsequent_lookup` at the integration tier).
   - Distinct `user` fields resolve to distinct `SessionState` instances.
   - When `user` is absent, the message-prefix derivation path kicks in and groups requests by first user message.
   - Cold-start requests (no user field, no user-role message) each get a fresh identity — they do not collapse into a shared cold bucket.

2. **`test_fc9_session_start_contract.py`** (5 static inspection tests). Covers the structural half of FC-9:
   - `resolve_session_start_context` has signature `(context: SessionContext) -> list[PromptFragment]` verified via `inspect.signature` + `typing.get_type_hints`.
   - The function is defined at module level (not nested), consistent with ADR-009's reservation shape.
   - AST scan over `src/llm_orc/` finds exactly one `FunctionDef` with that name (in `agentic/session_start.py`).
   - AST scan over `src/llm_orc/` finds exactly one `ast.Name` reference outside the definition — the default-resolver binding in `SessionStartCache.__init__`. Every runtime invocation flows through `self._resolver(context)`, not through the bare name, so FC-9's "exactly 1 call" holds structurally, not only behaviorally.
   - `SessionStartCache()` with no argument resolves to the module-level function by identity — confirms the counted reference is the default wiring, not a leftover.

**Scenarios covered.** Group 6 does not claim scenarios; it closes FC-9 (both halves — behavioral via existing tests, structural via AST). WP-B's roadmap claim — "FC-9 satisfied on completion" — is now honored.

**Fitness criteria status.**
- FC-9 (`resolve_session_start_context` called exactly once; signature present): **fully satisfied** at WP-B close — behavioral tests (`test_session_start_fires_exactly_once_per_session`, `test_streaming_request_fires_session_start_exactly_once_per_session`, `test_streaming_and_non_streaming_share_session_start_cache`) plus new structural tests (signature match, single production reference).

**Test coverage delta.** +10 tests (5 session-identity integration + 5 FC-9 static). Full suite: 2151 passed, 91.21% coverage, lint clean (mypy + ruff + bandit + vulture).

**Unblocks:** **WP-B complete.** TS-1 advances to WP-C (Orchestrator Runtime — ReAct loop, Tool Dispatch, Budget Controller). The `_orchestrator_handoff` and `_orchestrator_stream_handoff` stubs in `v1_chat_completions.py` are the body-swap points for WP-C.

### WP-C: ReAct core + real LLM adapter — 2026-04-21

**Commits (in order):**
- `790f596` feat: add Budget Controller with per-iteration exhaustion check (Group 1)
- `927f513` refactor: correct scenario wording — tool surface lives in Tool Dispatch, not /v1/models
- `07032a9` feat: add Orchestrator Tool Dispatch with five-entry closed set (Group 2)
- `b4e6f43` feat: add Orchestrator Runtime ReAct loop with Budget enforcement (Group 3)
- `90df826` refactor: delegate Tool Dispatch to OrchestraService instead of reimplementing invoke/list
- `061312e` feat: extend ModelInterface with tool-calling surface (Group 4a)
- `e48c7b8` feat: implement generate_with_tools on OpenAICompatibleModel (Group 4b)
- `7339eac` feat: wire Serving Layer to OrchestratorRuntime (Group 4c)
- `8227dc0` docs: add WP-C manual verification guide for Ollama end-to-end (Group 4d)
- `65b1334` feat: add llm-orc serve command for agentic-serving deployments
- `bb7b466` refactor: wire HTTP request timeout through performance config
- `22deeaf` fix: raise default HTTP read timeout to 180s for local tool-calling
- `bab8e1d` docs: correct WP-C manual verification findings (serve command, provider key, timeout)
- `12c19ac` docs: record re-verification pass and clarify session-cumulative Budget counter
- (this change) test: add FC-4 static check and Tool Dispatch → Ensemble Engine boundary tests

**Outcome.** The orchestrator runs end-to-end behind `/v1/chat/completions` against any OpenAI-compat backend (Ollama local, OpenAI proper, OpenRouter, LM Studio, vLLM, Anthropic-via-OpenAI-compat proxy). Verified against `mistral-nemo:12b` on local Ollama in two live runs — see `housekeeping/wp-c-manual-verification.md`.

Three new modules landed in `src/llm_orc/agentic/`:

- **`budget_controller.py`** — `BudgetController.check(turn_count, token_spend) -> BudgetCheckPass | BudgetCheckExhausted`. Return semantics (not raise). Deterministic turn-limit-first precedence. Zero agentic imports.
- **`orchestrator_tool_dispatch.py`** — Five-method closed set (FC-5). `invoke_ensemble` / `list_ensembles` delegate to `OrchestraService` via the `EnsembleOperations` Protocol (collapsed the parallel find-and-execute path introduced in Group 2 before the refactor in `90df826`). `compose_ensemble` / `query_knowledge` / `record_outcome` return typed `not_yet_wired` tool errors so the closed-set property holds from day one.
- **`orchestrator_runtime.py`** — ReAct loop. Budget check before every iteration (FC-10). `OrchestratorLLM` Protocol satisfied by any `ModelInterface` that overrides `generate_with_tools`. `ToolDispatcher` Protocol satisfied by `OrchestratorToolDispatch`. Tool results flow back as `role: tool` messages; LLM errors surface as observations, not exceptions.

Type unification: the Runtime's tool-calling response types (`ToolCallingResponse`, `ToolCall`, `ToolCallUsage`) moved to `models/base.py` and are shared by `ModelInterface.generate_with_tools` and the Runtime's `OrchestratorLLM` Protocol. No parallel data model.

Tool-calling surface added to the existing multi-provider infrastructure:

- `ModelInterface.generate_with_tools` default raises `ToolCallingNotSupportedError`; providers opt in by overriding and setting `supports_tool_calling = True`.
- `OpenAICompatibleModel` implements the default case for OpenAI-compat endpoints. Anthropic-native and Google-native wait for follow-up WPs that override on those provider classes.
- Session start fails loudly if the resolved orchestrator Model Profile does not support tool calling.

Serving Layer body-swap: `_orchestrator_handoff` and `_orchestrator_stream_handoff` in `v1_chat_completions.py` now construct and drive a real Runtime per request. `ModelFactory.load_model_from_agent_config({"model_profile": ...})` supplies the LLM; `BudgetController` is built from `OrchestratorConfig.budget`; Tool Dispatch is the shared process-scoped instance. Factories are `monkeypatch`-overridable from tests following the WP-B pattern.

`llm-orc serve` command added as a sibling of `llm-orc web`. Both commands start the same FastAPI app; `serve` is the natural name for agentic-client deployments, `web` remains the framing for "I want the browser UI." `llm-orc mcp serve` is unrelated (MCP server, direct-tool surface).

HTTP read timeout refactored: `HTTPConnectionPool` now reads `connect` / `read` / `write` / `pool` from `performance.concurrency.request_timeout` with per-field defaults. Default read raised from 30 to 180 seconds for local tool-calling models.

**Scenarios covered:**

- §Session Lifecycle: *Tool user completes a task against the stateless orchestrator* (end-to-end, verified in both automated tests and manual Ollama run); *Session terminates gracefully on turn limit exhaustion*; *Session terminates gracefully on token limit exhaustion*.
- §Orchestrator Tool Surface (retitled *tool surface* in `927f513`): *Orchestrator tool surface is exactly the committed set* (FC-5 structurally enforced); *Invocation outside the tool set is rejected* (Runtime-level integration verified via `test_runtime_propagates_tool_error_as_observation`).

**Fitness criteria status.**

- FC-4 (Runtime import surface): satisfied. `test_fc4_runtime_import_surface.py` walks `orchestrator_runtime.py` imports and fails closed on any `llm_orc.agentic.*` import outside the explicit allow list or on any match to the forbidden set (`orchestrator_config`, `session_registry`, `plexus_adapter`, `autonomy_policy`, `calibration_gate`). The last three do not yet exist — fails closed when they land.
- FC-5 (exactly five dispatch entry points): satisfied. `test_tool_dispatch_exposes_exactly_five_tool_methods` enumerates public async methods whose names are in `TOOL_NAMES`.
- FC-10 (Budget check before every iteration): satisfied. `test_turn_limit_exhausted_before_first_iteration`, `test_token_limit_exhausted_before_first_iteration`, and `test_runtime_terminates_mid_loop_when_budget_exhausted_between_iterations` exercise the control-plane property at all iteration positions.
- FC-8 (unsummarized result unreachable from Runtime context): **partial pending WP-D**. Current tool-result summarization is a trivial JSON-dump placeholder in `_tool_result_message`; WP-D's Result Summarizer Harness replaces it and closes the static no-bypass check.
- FC-13 (orchestrator Model Profile swap touches only config + session start): satisfied by construction — Runtime takes an `OrchestratorLLM` at construction; profile swap routes through `OrchestratorConfigResolver` + `ModelFactory` in `_build_runtime`, never touching Runtime internals.

**Test coverage delta.** +74 tests (Budget Controller 5, Tool Dispatch unit 10, Orchestrator Runtime 7, ModelInterface tool-calling base 2 + HTTP timeout config 3, OpenAICompatibleModel tool-calling 7, Serving Layer wiring 2 acceptance + 24 pre-existing still green after rewire, serve CLI 5, FC-4 static 2, boundary integration 3, timeout config tests 3; includes 5 tests that changed semantics during the refactor). Full suite: 2197 passing, 91.41% coverage.

**Unblocks TS-1 (stateless orchestrator serving OpenCode).** The intermediate transition state in this roadmap is *"I can use OpenCode and run a version of this RDD pipeline with it."* The orchestrator is live end-to-end. WP-F (client-tool turn-boundary delegation) remains the final TS-1 item — until WP-F lands, the orchestrator can list and invoke ensembles but cannot delegate client-side tools (bash, file_edit) at turn boundaries.

**Design Amendment candidate logged for WP-D start** (see `housekeeping/cycle-status.md` §Feed-Forward From BUILD). The system design has the Runtime depending on Result Summarizer Harness, but the module's own rationale states the Runtime is not aware of the summarizer — the harness is interposed by Tool Dispatch on the `invoke_ensemble` return path. WP-D should land the Design Amendment alongside RSH itself: remove `Runtime → RSH` from the dependency graph, add `Tool Dispatch → RSH`, update FC-4 to omit RSH from Runtime's import set.

**Debt surfaced (not addressed in WP-C scope).**

- Conversation Compaction is named in the Runtime's ownership list (system design §Orchestrator Runtime) but not implemented. The WP-C scenarios did not require it (turn/token exhaustion precedes compaction's utility). Can land in a follow-up mini-cycle or alongside another WP that touches the Runtime.
- Per-request usage accounting: the `/v1/chat/completions` response's `usage.completion_tokens` reports the per-request delta in `SessionState.token_spend`; `prompt_tokens` is hardcoded to 0. Fine-grained prompt-vs-completion accounting requires accumulating each iteration's `LLMUsage.prompt_tokens` separately, which the Runtime currently collapses into `total_tokens` on Session state. A follow-up can split the accounting without architectural change.
- Routing Decision generation (for `record_outcome` in WP-I) is named in the Runtime's ownership but only materializes when Plexus lands. WP-I generates the Routing Decision objects; Runtime emits them when `record_outcome` is no longer `not_yet_wired`.

### WP-D: Result Summarizer Harness — 2026-04-21

**Commits (in order):**

*Groups 0-4 (structural change):*
- `a15aa30` docs: Design Amendment #3 — move RSH dependency from Runtime to Tool Dispatch
- `326a36f` feat: add Result Summarizer Harness module with typed result variants
- `188f65f` feat: add raw_output flag to EnsembleConfig for ADR-004 escape hatch
- `9a0fea2` feat: interpose Result Summarizer Harness on invoke_ensemble return path
- `3e7c897` feat: ship default agentic-result-summarizer ensemble and profile

*Groups 5-6 (verification and closeout):*
- `4261238` refactor: tighten FC-4 forbidden list for Amendment #3
- `903833e` test: add strict FC-8 static no-bypass check for invoke_ensemble
- `03885f8` test: add raw-output escape-hatch acceptance scenario at Serving Layer
- `2f0f660` test: add Tool Dispatch → Harness → Ensemble Engine summarize boundary
- (this change) docs: close WP-D in field-guide, ORIENTATION, cycle-status, roadmap

**Outcome.** AS-7 ("Result summarization is a correctness requirement") is now structurally enforced. The Runtime never sees raw ensemble output: FC-4 forbids RSH from Runtime's import set; FC-8's strict AST dominance check proves Tool Dispatch cannot construct a successful `invoke_ensemble` result without routing through the Harness; boundary integration proves the real wiring produces summaries end-to-end. ADR-004's raw-output escape hatch is honored and opt-in, not a default.

**New module.** `src/llm_orc/agentic/result_summarizer_harness.py` — `ResultSummarizerHarness` class with `summarize(raw_result, *, raw_output) -> SummarizationSuccess | RawOutputPassthrough | SummarizationFailure`. Takes a `SummarizerInvoker` Protocol (shape: `async def invoke(arguments) -> dict`) so it is decoupled from `OrchestraService`; the Serving Layer wires them together in `get_orchestrator_tool_dispatch`. `_extract_summary` uses a synthesis → single-agent `response` fallback so the default single-agent summarizer ensemble works without requiring a synthesis pass (llm-orc's dependency-based execution model leaves `synthesis` unpopulated for single-agent ensembles).

**New primitive (library content, not code).** `.llm-orc/ensembles/agentic-result-summarizer.yaml` — single-agent default summarizer ensemble. `.llm-orc/config.yaml` gains a `summarizer` model profile. Operators override via `agentic_serving.orchestrator.summarizer_ensemble` in `config.yaml`.

**Edits.**
- `EnsembleConfig` gains a `raw_output: bool = False` field; YAML loader threads the flag through to `invoke_ensemble`'s return path.
- `OrchestratorConfig` gains `summarizer_ensemble: str` so the Harness's configured target is operator-visible.
- `OrchestratorToolDispatch.invoke_ensemble` calls `await self._harness.summarize(result, raw_output=...)` on every return, pattern-matches the three outcome variants, and emits either `ToolCallSuccess({"summary": <str>})`, `ToolCallSuccess(<raw dict>)`, or `ToolCallError(kind="summarization_failed")`. New `ToolErrorKind` literal: `summarization_failed`.
- `system-design.md` Amendment #3: Dependency Graph `Orchestrator Runtime → Result Summarizer Harness` moved to `Orchestrator Tool Dispatch → Result Summarizer Harness`; FC-4 wording amended to exclude RSH from Runtime's allow list; Responsibility Matrix and Test Architecture rows updated in sync.

**Scenarios covered.**
- `scenarios.md` §Ensemble result is summarized before entering orchestrator context — boundary integration via `tests/integration/test_tool_dispatch_summarizer_boundary.py`.
- `scenarios.md` §Raw-output escape hatch is explicit — Serving Layer acceptance via `tests/unit/web/test_api_v1_chat_completions.py::TestRawOutputEscapeHatchAcceptance`.

**Fitness criteria status.**
- FC-4 (Runtime import surface): **strengthened**. `result_summarizer_harness` now explicitly forbidden from Runtime imports.
- FC-8 (unsummarized result unreachable): **fully satisfied**. `test_fc8_summarizer_bypass.py` parses `orchestrator_tool_dispatch.py` and proves three properties on `invoke_ensemble`: the Harness is called; every `ToolCallSuccess` constructor is dominated by the match on the summarize result; lexical ordering is consistent. An adversarial self-test parses a synthetic bypass fixture and verifies the detector catches it.

**Test coverage delta.** +15 tests (Harness unit 10 pre-closeout + FC-8 static 3 + adversarial self-test 1 + raw-output acceptance 2 + summarize boundary 2; baseline at WP-C close was 2197, close at WP-D Group 4 was 2213, close at WP-D Group 6 is 2221 as some pre-existing tests adapted to the Amendment #3 wiring). Full suite: **2221 passing, 91.44% coverage, lint clean** (mypy + ruff check + ruff format).

**Decisions made during build.**
- **Strict-over-loose FC-8 formulation** (Group 5). The strict AST dominance check carries a legibility cost but catches the class of regressions (early-return fast paths, short-circuit branches) a "harness is mentioned somewhere" check would miss. Adversarial self-test in the same file makes the detection logic itself load-bearing. Chosen deliberately: robustness traded for legibility, with the expectation that future agentic work on this code benefits from the stronger convention.
- **Three-test coverage for the `test_runtime_never_sees_unsummarized_result` Test Architecture row** (Groups 5-6). The table row names a single test; post-WP-D the coverage is distributed across FC-8 static dominance, raw-output acceptance, and summarize-boundary integration. Worth a future system-design edit to point the table row at all three; deferred.

**Forward-carrying concerns** (not addressed in WP-D scope).
- **Summarizer-quality echo-back risk → WP-E / WP-H calibration scope.** FC-8 proves the Harness is always interposed; it does not prove the Harness's output is substantively a summary. A weak or compromised summarizer ensemble could return a JSON-encoded raw dict in its `response` field, and the Harness would return it as-is — the raw-dict leak would arrive through the summarizer's legitimate output channel rather than by bypassing. This is a quality property of the configured summarizer, not a structural bypass; Calibration Gate (ADR-007) is designed exactly for this class of problem. Failure mode is visible (weird summaries in the orchestrator's context, observable via SSE and artifacts) and recoverable (swap `summarizer_ensemble` via `config.yaml`). Deliberately deferred to WP-E / WP-H rather than adding a mechanism now. See `housekeeping/cycle-status.md` FF #81.

**Unblocks.** WP-E (Autonomy Policy), WP-G (Composition), and WP-I (Plexus Adapter) all depend only on WP-C and can land in parallel. WP-F (client-tool delegation) remains scenario-gated. TS-1's remaining gap is WP-F.
