# Field Guide: Agentic Serving

**Generated:** 2026-04-24
**Derived from:** `system-design.md` v1.0 + amendments #1–#4, current implementation at WP-H close (TS-2 reached).

## How to use this guide

This document maps each module in `system-design.md` to its current
implementation state. It is a reference — consult the entry for the
module you are working in or exploring. For the overall architecture,
read `system-design.md`. For routing to the right document, read
`ORIENTATION.md`.

State vocabulary:

- **Complete** — all named responsibilities implemented; production
  wiring in place; tests exercise the boundary.
- **Partial** — skeleton or subset implemented; WP still open.
- **Planned** — not yet implemented; design is stable; the WP that
  lands it is named.

Stability vocabulary:

- **Settled** — unlikely to change outside a named follow-up. Invest
  understanding here confidently.
- **In flux** — under active development or pending an adjacent
  change that will touch it.
- **Design-only** — system-design has the contract; no code yet.

---

## Module: Serving Layer

**Implementation state:** Complete for Phase 1 (Plexus injection reserved but not populated). Client Tool Surface Commitment (Option C) wired end-to-end at WP-F — client-declared tools flow through `message.tool_calls` / `delta.tool_calls` with `finish_reason: tool_calls`; `role: tool` + `tool_call_id` messages are accepted on subsequent requests.
**Code location:** `src/llm_orc/web/api/v1_chat_completions.py`, `src/llm_orc/web/api/v1_models.py`, `src/llm_orc/web/api/sse_format.py`, `src/llm_orc/agentic/session_start.py`.
**Stability:** Settled on structure; body of `resolve_session_start_context` changes at Phase 2 (ADR-009).

### Domain concepts in code

| Concept | Code manifestation | Location |
|---------|-------------------|----------|
| Serving Layer | `chat_completions` endpoint + `list_models` endpoint | `v1_chat_completions.py`, `v1_models.py` |
| SessionContext | `SessionContext` dataclass | `agentic/session_start.py` |
| ChatMessage (contract type) | `ChatMessage` dataclass — role, content, tool_call_id, tool_calls | `agentic/session_start.py` |
| PromptFragment | `PromptFragment` dataclass (Phase 2) | `agentic/session_start.py` |
| SSE streaming | `_stream_completion` + `OpenAiSseFormatter` | `v1_chat_completions.py`, `web/api/sse_format.py` |
| Client-tool delegation (non-streaming) | `_NonStreamingResult` + `_build_completion_body` | `v1_chat_completions.py` |
| Client-tool delegation (streaming) | `ClientToolCall` case in `OpenAiSseFormatter.format` | `web/api/sse_format.py` |
| Tool-call wire encoder (shared) | `encode_tool_call_for_message` | `web/api/sse_format.py` |
| Reserved-name guard | `_reject_reserved_tool_names` | `v1_chat_completions.py` |
| Context Injection (action) | `resolve_session_start_context` | `agentic/session_start.py` |
| Session-start cache (FC-9 single call) | `SessionStartCache` | `agentic/session_start.py` |
| Runtime construction per session | `_build_runtime` | `v1_chat_completions.py` |

### Design rationale

The endpoint is the OpenAI-compat entry point; Session identity resolution
happens here before the Runtime is constructed so the Runtime sees a
clean `SessionContext`. Streaming and non-streaming share
`_resolve_context` so session-start cache semantics hold under both
paths (FC-9 boundary-integration tested).

**Client Tool Surface Commitment (WP-F).** Client-declared `tools[]`
ride through the response surface under Option C. When the Runtime
yields a `ClientToolCall` chunk (pure client-declared tool batch), the
streaming formatter emits `delta.tool_calls` + `finish_reason:
tool_calls`; the non-streaming body path shapes
`message.tool_calls` + `finish_reason: tool_calls`. Both use the
shared `encode_tool_call_for_message` helper so the OpenAI entry
format stays in sync. `_resolve_context` rejects requests that
declare client tools colliding with `TOOL_NAMES` via HTTP 400
(`_reject_reserved_tool_names`) — silent misrouting would be worse
than an immediate operator-visible error.

**ChatMessage lives in `session_start`, not `session_registry`.** It
is a contract type on the Serving Layer → Runtime edge (it rides
inside `SessionContext`), not Session Registry internals. Locating it
in this module keeps FC-4 intact: the Runtime imports ChatMessage
from `session_start` (allow-listed) rather than reaching into
`session_registry` (forbidden).

Phase 2 Plexus injection is structurally reserved via the typed
function `resolve_session_start_context(context) -> list[PromptFragment]`
at a single call site inside `SessionStartCache.resolve`. ADR-009's
reservation is satisfied by signature + call site; Phase 1's empty
return is a body-only change.

### Key integration points

- **→ Session Registry:** `SessionRegistry.resolve_identity` + `get_or_create_state` via `_resolve_context`.
- **→ Orchestrator Configuration:** `get_orchestrator_config_resolver().resolve_validated()` via `_build_runtime`. Reads Budget, Model Profile, autonomy default, summarizer ensemble, and `orchestrator_system_prompt` (WP-F).
- **→ Orchestrator Runtime:** `runtime.run(context)` via `_collect_non_streaming` / `_stream_completion`. `_build_runtime` passes `config.orchestrator_system_prompt` into Runtime construction.
- **→ Orchestrator Tool Dispatch:** indirect — Runtime holds the dispatch; Serving Layer wires the shared instance via `get_orchestrator_tool_dispatch`.

---

## Module: Session Registry

**Implementation state:** Complete for Phase 1 (in-process, no cross-process persistence).
**Code location:** `src/llm_orc/agentic/session_registry.py`.
**Stability:** Settled. Persistence surface is available on the contract but not yet activated — adds when Autonomy Level / Calibration state demands it.

### Domain concepts in code

| Concept | Code manifestation | Location |
|---------|-------------------|----------|
| Session | `SessionState` dataclass | `session_registry.py:47` |
| SessionIdentity | `SessionIdentity` dataclass | `session_registry.py:33` |
| Identity-derivation method | `Literal["user_field", "message_prefix", "cold_start"]` | `session_registry.py:17` |
| Cumulative turn count + token spend | `SessionState.record_iteration` | `session_registry.py:61` |
| Per-process registry | `SessionRegistry` class | `session_registry.py:67` |

### Design rationale

Identity is derived from request features (user field first, then
message-prefix hash, then cold-start UUID) so the same agentic client
can continue a Session across requests without a bespoke header. The
mutable `SessionState` is returned by identity; callers mutate it,
retained references are shared by design (lifecycle-sequence test in
`tests/unit/web/test_api_v1_chat_completions.py`).

### Key integration points

- **← Serving Layer:** per-request identity resolution + state retrieval.
- **→ nothing outbound in L1 or L2 modules** — Budget / Autonomy / Calibration take Session state via plain args, not by importing the registry. This is a deliberate departure from the stated dependency graph (see cycle-status FF #64).

---

## Module: Orchestrator Configuration

**Implementation state:** Complete for Phase 1 (per-request overrides not yet applied).
**Code location:** `src/llm_orc/agentic/orchestrator_config.py`.
**Stability:** Settled on the resolved-config shape. Per-request override application is expressible but inert until a scenario requires it.

### Domain concepts in code

| Concept | Code manifestation | Location |
|---------|-------------------|----------|
| OrchestratorConfig (immutable per-session) | `OrchestratorConfig` dataclass | `orchestrator_config.py` |
| BudgetDefaults | `BudgetDefaults` dataclass | `orchestrator_config.py` |
| OverrideBounds (expressible) | `OverrideBounds` dataclass | `orchestrator_config.py` |
| Orchestrator system prompt | `OrchestratorConfig.orchestrator_system_prompt` field | `orchestrator_config.py` |
| Default prompt content | `DEFAULT_ORCHESTRATOR_SYSTEM_PROMPT` module constant | `orchestrator_config.py` |
| Fail-fast Model Profile validation | `resolve_validated` | `orchestrator_config.py` |
| Allowlist for `/v1/models` | `list_allowed_model_profile_ids` | `orchestrator_config.py` |
| `ModelProfileNotFoundError` | `ModelProfileNotFoundError` | `orchestrator_config.py` |

### Design rationale

`resolve_validated` raises before session start if the configured
orchestrator profile is absent from the library — `/v1/models` shop
window is silent-drop (ADR-011 posture). Immutable config matches the
session-boundary discipline in ADR-011: changes apply to new
sessions, not active ones.

**Orchestrator system prompt (WP-F Group 3).**
`orchestrator_system_prompt` is a required config field with a
`DEFAULT_ORCHESTRATOR_SYSTEM_PROMPT` default that teaches the
five internal tools (ADR-003), Option C's one-kind-per-turn
discipline (system-design §Client Tool Surface Commitment), and the
`needs_client_tool` retry convention (roadmap ODP #8 mechanism i).
Operators override via `agentic_serving.orchestrator.system_prompt`
in `config.yaml`. The Serving Layer passes the resolved prompt into
Runtime construction at `_build_runtime`; Runtime always prepends it
as a leading `role: system` message so orchestrator discipline sits
ahead of any client-supplied system guidance.

### Key integration points

- **→ ConfigurationManager** (project-level, not scoped).
- **← Serving Layer** at session start for Model Profile + Budget defaults.

---

## Module: Budget Controller

**Implementation state:** Complete.
**Code location:** `src/llm_orc/agentic/budget_controller.py`.
**Stability:** Settled.

### Domain concepts in code

| Concept | Code manifestation | Location |
|---------|-------------------|----------|
| Budget | `BudgetController` class | `budget_controller.py:55` |
| BudgetCheck (union) | `BudgetCheckPass \| BudgetCheckExhausted` | `budget_controller.py:30-51` |
| ExhaustionReason | `Literal["turn_limit", "token_limit"]` | `budget_controller.py:27` |
| Per-iteration check | `check(turn_count, token_spend)` | `budget_controller.py:62` |
| Deterministic precedence | Turn-limit checked before token-limit | `budget_controller.py:64-71` |

### Design rationale

Pure value-comparison, zero agentic-module imports. Takes plain
integers via method args — Runtime passes them in. Return semantics
(not raise); the Runtime converts `BudgetCheckExhausted` into a
graceful `Completion(finish_reason="length")` termination with an
explicit `ContentDelta` naming the exhaustion.

AS-3: Budget is a control-plane concern — checked each iteration,
independent of the orchestrator LLM's reasoning. The Runtime never
exposes Budget state to the LLM.

### Key integration points

- **← Orchestrator Runtime:** one call per iteration in `OrchestratorRuntime.run`.

---

## Module: Orchestrator Runtime

**Implementation state:** Complete for WP-C / WP-D / WP-F scope. Conversation Compaction named but not implemented (no scenario has required it). Routing Decision generation materializes at WP-I.
**Code location:** `src/llm_orc/agentic/orchestrator_runtime.py`.
**Stability:** Settled. Runtime receives already-summarized `ToolCallResult` values from Tool Dispatch (Amendment #3); summarization is a Tool-Dispatch-side concern the Runtime stays unaware of. Client Tool Surface Commitment (Option C) wired at WP-F — `run` splits each tool-calls batch by `TOOL_NAMES` membership and routes accordingly.

### Domain concepts in code

| Concept | Code manifestation | Location |
|---------|-------------------|----------|
| Orchestrator Agent | `OrchestratorRuntime` class | `orchestrator_runtime.py` |
| ReAct loop | `run` async generator | `orchestrator_runtime.py` |
| System prompt injection | `OrchestratorRuntime.__init__(..., system_prompt=...)` + prepend in `run` | `orchestrator_runtime.py` |
| Internal dispatch sub-generator | `_dispatch_internal_calls` | `orchestrator_runtime.py` |
| Batch classifier | `_split_tool_calls(tool_calls, client_tool_names)` | `orchestrator_runtime.py` |
| Client-tool names extractor | `_client_tool_names(context.tools)` | `orchestrator_runtime.py` |
| Mixed-batch rejection | `_record_mixed_batch_rejection` + `_mixed_batch_error_observation` | `orchestrator_runtime.py` |
| Client delegation chunk builder | `_client_delegation_chunk` | `orchestrator_runtime.py` |
| ChatMessage → LLM dict | `_session_message_to_llm` | `orchestrator_runtime.py` |
| OrchestratorLLM Protocol | `OrchestratorLLM` | `orchestrator_runtime.py` |
| ToolDispatcher Protocol | `ToolDispatcher` | `orchestrator_runtime.py` |
| Internal tool schemas | `_build_tool_schemas` (five schemas, assertion-verified) | `orchestrator_runtime.py` |
| Exhaustion formatting | `_format_exhaustion_message` | `orchestrator_runtime.py` |

### Design rationale

The Runtime imports only Budget Controller, Tool Dispatch (via
`ToolDispatcher` Protocol), and neutral contract types — FC-4
structurally enforced by `test_fc4_runtime_import_surface.py`, which
post-Amendment #3 explicitly forbids `result_summarizer_harness`
alongside Plexus / Autonomy / Calibration / config / Session
Registry. This preserves the orchestrator LLM's mental model ("I
reason, I emit tool calls, I observe results"): Session bookkeeping,
Plexus awareness, Autonomy gating, Calibration state, and result
summarization all live on the other side of Tool Dispatch.

**Client Tool Surface Commitment — Option C (WP-F).** `run` advertises
the union of the five internal schemas and the request's
client-declared `tools[]` to the LLM. On each iteration, the tool-calls
batch is split by `TOOL_NAMES` membership via `_split_tool_calls`.
Mixed batches (internal + client in one response) are rejected with
per-call `mixed_batch` error observations and the LLM retries on the
next iteration — no silent data loss. Pure-client batches yield a
`ClientToolCall` chunk (built via `_client_delegation_chunk`) and the
generator returns, closing the turn with `finish_reason: tool_calls`.
Pure-internal batches flow through `_dispatch_internal_calls` as
before. Names outside both sets route through Tool Dispatch, which
surfaces `unknown_tool` — keeping AS-6 closure structural regardless
of client-declared tools.

**System prompt injection (WP-F Group 3).** `OrchestratorRuntime` takes
a `system_prompt: str = ""` construction kwarg. When non-empty, `run`
always prepends it as a leading `role: system` message on every LLM
iteration — ahead of any client-supplied system message so the
orchestrator's discipline (five internal tools, turn-boundary
discipline, `needs_client_tool` retry convention) survives competing
client guidance. Empty string is a no-op. Tests and deployments that
want no orchestrator-side prompt pass `""`.

Scripted LLM fakes (in tests) satisfy `OrchestratorLLM` structurally
without subclassing `ModelInterface`; production wiring passes the
loaded `ModelInterface` directly (one shared type for tool-calling
responses across Runtime and `ModelInterface.generate_with_tools`).

### Key integration points

- **→ Budget Controller:** `check(turn_count, token_spend)` per iteration.
- **→ Tool Dispatch (via `ToolDispatcher` Protocol):** `dispatch(InternalToolCall)` per tool emission.
- **→ LLM (via `OrchestratorLLM` Protocol):** `generate_with_tools(messages, tools)` per iteration.
- **→ orchestrator_chunk:** yields `ContentDelta` / `Completion` / `InternalToolCallInFlight` / `InternalToolCallResult` / `ClientToolCall` to the Serving Layer.

---

## Module: Orchestrator Tool Dispatch

**Implementation state:** Complete for WP-C + WP-D + WP-E + WP-G wiring. `invoke_ensemble` interposes the Result Summarizer Harness on every return (Amendment #3). `list_ensembles` delegates to `OrchestraService.read_ensembles`. `compose_ensemble` delegates to `CompositionValidator` then writes via `LocalEnsembleWriter` on accept (WP-G). `query_knowledge` / `record_outcome` return typed `not_yet_wired` errors pending WP-I.
**Code location:** `src/llm_orc/agentic/orchestrator_tool_dispatch.py`.
**Stability:** Settled on the closed-set structure and the Harness/Autonomy/Composition interpositions. Two handler bodies change when WP-I wires them.

### Domain concepts in code

| Concept | Code manifestation | Location |
|---------|-------------------|----------|
| Orchestrator Tool | Five methods named after `TOOL_NAMES` | `orchestrator_tool_dispatch.py:162-265` |
| TOOL_NAMES (closed set) | `frozenset` | `orchestrator_tool_dispatch.py:45` |
| Dispatch-by-name | `dispatch` method with `match-case` over the five names | `orchestrator_tool_dispatch.py:132` |
| Allowlist rejection | `case _:` arm → `ToolCallError(kind="unknown_tool")` | `orchestrator_tool_dispatch.py:151` |
| InternalToolCall | `InternalToolCall` dataclass | `orchestrator_tool_dispatch.py:66` |
| ToolCallResult (union) | `ToolCallSuccess \| ToolCallError` | `orchestrator_tool_dispatch.py:80-103` |
| ToolErrorKind | `Literal["unknown_tool", "not_yet_wired", "invocation_failed", "invalid_arguments", "summarization_failed", "denied_by_autonomy"]` | `orchestrator_tool_dispatch.py:57` |
| EnsembleOperations Protocol | narrow facade over `OrchestraService` | `orchestrator_tool_dispatch.py:106` |
| AutonomyGate Protocol | narrow surface Tool Dispatch consults; AutonomyPolicy satisfies structurally | `orchestrator_tool_dispatch.py:124` |
| Gate interposition | `decision = self._autonomy_policy.decide(tool_name=..., arguments=...)` + Deny short-circuit + event attachment | `orchestrator_tool_dispatch.py:149-174` |
| Tool-routing indirection | `_route(call)` routes to the matched tool method — keeps the gate/route lexical ordering checkable by FC-11 | `orchestrator_tool_dispatch.py:181` |
| Event attachment | `_with_events(result, events)` — returns the result untouched when events are empty, else rebuilds with events | `orchestrator_tool_dispatch.py:280` |
| Dynamic Invocation (action) | `invoke_ensemble` delegates to `operations.invoke`, then interposes the Harness on the return | `orchestrator_tool_dispatch.py:162` |
| Summarize interposition | `summarization = await self._harness.summarize(result, raw_output=...)` + match on `SummarizationSuccess` / `RawOutputPassthrough` / `SummarizationFailure` | `orchestrator_tool_dispatch.py:202-221` |

### Design rationale

The five method names are the closed set (ADR-003). Dispatch uses
`match-case` rather than `getattr` so the closed set is visible at
the dispatch site and mypy preserves return types through the
branch. FC-5 static check counts public async methods whose names
are in `TOOL_NAMES`.

Delegation to `OrchestraService` via the `EnsembleOperations` Protocol
(commit `90df826`) collapsed a parallel find-and-execute path.
`OrchestraService` satisfies `EnsembleOperations` structurally. Future
changes to ensemble name resolution, tier discovery, or status
normalization propagate through one code path.

**Result Summarizer Harness interposition (Amendment #3).**
`invoke_ensemble` hands the raw ensemble result to the Harness before
constructing a `ToolCallSuccess`. The Harness decides between three
outcomes — summarize, raw-output pass-through (ADR-004 escape hatch
keyed on the ensemble's `raw_output` flag), or typed failure. The
Runtime receives only the post-Harness result; it never imports the
Harness itself. FC-8's static AST dominance check
(`test_fc8_summarizer_bypass.py`) enforces this mechanically — every
`ToolCallSuccess(..., name="invoke_ensemble", ...)` construction in
the method is nested inside the match on the summarize result, so a
future bypass fails the test before merge.

**Autonomy Policy gate (WP-E).** `dispatch` runs a three-step flow:
(1) unknown-tool filter — AS-6 closure via `TOOL_NAMES`; (2) Autonomy
gate — `self._autonomy_policy.decide(tool_name, arguments)` returns
`Allow(events)` or `Deny(reason)`; Deny short-circuits as
`ToolCallError(kind="denied_by_autonomy")` without routing; (3) route
to the matched tool method via `_route`, then attach decision events
via `_with_events`. FC-11's static AST dominance check
(`test_fc11_autonomy_gate.py`) enforces that every `await self._route
(...)` in `dispatch` is lexically after the first
`self._autonomy_policy.decide` call; an adversarial self-test proves
the detector catches a synthetic fast-path bypass.

### Key integration points

- **← Orchestrator Runtime:** `dispatch(InternalToolCall)` per LLM tool emission.
- **→ OrchestraService (via `EnsembleOperations`):** `invoke` for `invoke_ensemble`, `read_ensembles` for `list_ensembles`.
- **→ Result Summarizer Harness:** `summarize(raw_result, raw_output=...)` on every `invoke_ensemble` return. The Harness in turn calls back through `EnsembleOperations` to invoke the configured summarizer ensemble.
- **→ Autonomy Policy (via `AutonomyGate` Protocol):** `decide(tool_name, arguments)` on every dispatch for a committed tool name.
- **→ Composition Validator (via `CompositionGate` Protocol):** `validate(CompositionRequest)` on every `compose_ensemble` dispatch. Rejection → `ToolCallError(kind="invocation_failed")`; acceptance → hand the config to the writer.
- **→ Local ensemble writer (via `LocalEnsembleWriter` Protocol):** `write(EnsembleConfig)` on composition accept. Write failure (`EnsembleWriteError`) → `ToolCallError(kind="invocation_failed")`; success → `ToolCallSuccess({"name", "path"})`.
- **WP-I will add:** `query_knowledge` + `record_outcome` via Plexus Adapter.
- **WP-H will interpose:** Calibration Gate on in-calibration ensemble invocations.

---

## Module: Ensemble Engine (existing)

**Implementation state:** Unchanged by WP-C (retrofit mode per ADR-001 / ADR-002).
**Code location:** `src/llm_orc/core/execution/ensemble_execution.py` (plus runners, phases, fan-out, artifact_manager — the full existing subsystem).
**Stability:** Settled. Project-level concern, not scoped.

### Access from agentic-serving

Tool Dispatch reaches the Ensemble Engine through `OrchestraService`:

- `OrchestraService.invoke({ensemble_name, input})` → `ExecutionHandler.invoke` → `EnsembleLoader.find_ensemble` → `EnsembleExecutor.execute`.
- `OrchestraService.read_ensembles()` → `ResourceHandler.read_ensembles` → walks every ensembles dir → returns normalized metadata.

No agentic-serving module imports `EnsembleExecutor` directly —
everything flows through the `EnsembleOperations` Protocol on Tool
Dispatch.

---

## Module: Result Summarizer Harness

**Implementation state:** Complete (WP-D).
**Code location:** `src/llm_orc/agentic/result_summarizer_harness.py`.
**Stability:** Settled. The `_extract_summary` fallback order (synthesis → single-agent `response`) is the stable contract; the quality-of-summarizer concern defers to Calibration Gate (WP-E / WP-H).

### Domain concepts in code

| Concept | Code manifestation | Location |
|---------|-------------------|----------|
| Result Summarization (concept) | `ResultSummarizerHarness` class | `result_summarizer_harness.py:81` |
| Summarize (action) | `summarize(raw_result, *, raw_output)` async method | `result_summarizer_harness.py:88` |
| Summarizer invocation facade | `SummarizerInvoker` Protocol (shape: `async def invoke(arguments) -> dict`) | `result_summarizer_harness.py:34` |
| Summarization result variants | `SummarizationSuccess \| RawOutputPassthrough \| SummarizationFailure` | `result_summarizer_harness.py:46-78` |
| Raw-output escape hatch (ADR-004) | `raw_output: bool` keyword on `summarize`; set on `EnsembleConfig.raw_output` | `result_summarizer_harness.py:95`; `core/config/ensemble_config.py` |
| Summary extraction (synthesis + single-agent fallback) | `_extract_summary` module function | `result_summarizer_harness.py:118` |

### Design rationale

**Amendment #3 places the Harness on the Tool Dispatch side, not the
Runtime side.** The Runtime stays unaware of summarization; the
orchestrator LLM's reasoning surface remains "I emit tool calls and
observe results." Tool Dispatch invokes the Harness on every
`invoke_ensemble` return path; the Harness invokes one ensemble (the
configured summarizer) and translates the outcome into a typed
result variant.

The three-variant return is the contract Tool Dispatch pattern-matches
on: `SummarizationSuccess.summary` becomes the orchestrator's tool-
result content (wrapped as `{"summary": <str>}`); `RawOutputPassthrough
.content` is the raw dict pass-through (ADR-004 escape hatch for
ensembles whose operator has opted in via `raw_output: true`);
`SummarizationFailure.reason` becomes a typed `ToolCallError(kind=
"summarization_failed")` — never a raw-dict leak.

The `_extract_summary` fallback is deliberately forgiving: prefer
`synthesis` when non-empty, otherwise accept a single-agent ensemble's
`results[agent]["response"]`. llm-orc's dependency-based execution
model leaves `synthesis` unpopulated for single-agent ensembles, and
the default summarizer (`.llm-orc/ensembles/agentic-result-summarizer
.yaml`) is single-agent — the fallback keeps the default working
without requiring a synthesis pass.

**FC-8 enforcement is three-sided.** FC-4 isolates the Runtime from
ensemble-execution surfaces; the Harness is the only producer of
summarized content on the `invoke_ensemble` path; FC-8's strict AST
dominance check proves Tool Dispatch cannot construct a successful
`invoke_ensemble` result without routing through the Harness. An
adversarial self-test (`test_detection_logic_rejects_synthetic_bypass`)
proves the detector catches simulated regressions.

### Key integration points

- **← Orchestrator Tool Dispatch:** `harness.summarize(raw_result, raw_output=...)` per `invoke_ensemble` return.
- **→ EnsembleOperations (satisfied by `OrchestraService`):** `invoke({ensemble_name, input})` to run the configured summarizer ensemble. The Harness does not import `OrchestraService` directly — it takes a `SummarizerInvoker` whose shape matches `EnsembleOperations.invoke`.

### Scenarios covered

- `scenarios.md` §Ensemble result is summarized before entering orchestrator context — boundary integration via `tests/integration/test_tool_dispatch_summarizer_boundary.py`.
- `scenarios.md` §Raw-output escape hatch is explicit — acceptance via `tests/unit/web/test_api_v1_chat_completions.py::TestRawOutputEscapeHatchAcceptance`.

### Configured summarizer

**Ensemble:** `.llm-orc/ensembles/agentic-result-summarizer.yaml` — single-agent, configurable. Operators override via `agentic_serving.orchestrator.summarizer_ensemble` in `config.yaml`; the serving layer's Tool Dispatch factory reads this into `OrchestratorConfig.summarizer_ensemble` and passes it to the Harness at construction.

---

## Module: Autonomy Policy

**Implementation state:** Complete for Phase 1 (WP-E). Two named levels ship: `operator-as-tool-user` (baseline, silent) and `pure-tool-user-visible` (surfaces composition events). `Deny` is a first-class decision variant reserved for WP-H tighter-level semantics; Phase 1 never returns it but the dispatch-side handling is tested.
**Code location:** `src/llm_orc/agentic/autonomy_policy.py`.
**Stability:** Settled on decision shape and interposition. Level set expands when WP-H lands approve-before-uncalibrated semantics.

### Domain concepts in code

| Concept | Code manifestation | Location |
|---------|-------------------|----------|
| Autonomy Level | module-level constants `BASELINE_LEVEL`, `PURE_TOOL_USER_VISIBLE_LEVEL` | `autonomy_policy.py:30` |
| AutonomyPolicy | `AutonomyPolicy` class | `autonomy_policy.py:73` |
| AutonomyDecision (union) | `Allow \| Deny` | `autonomy_policy.py:39-68` |
| Allow (with events) | `Allow` frozen dataclass with `events: tuple[VisibilityEvent, ...]` | `autonomy_policy.py:44` |
| Deny | `Deny` frozen dataclass with `reason: str` | `autonomy_policy.py:57` |
| Level provider | `level_provider: Callable[[], str]` injected at construction | `autonomy_policy.py:76` |
| Composition event builder | `_composition_event(arguments) -> VisibilityEvent` | `autonomy_policy.py:87` |

### Design rationale

L1 Domain Policy. Takes plain values via `level_provider` so the module
never imports `ConfigurationManager` or `SessionState` (same posture as
Budget Controller, FF #64). The Serving Layer captures the operator-
configured level in a closure so a `config.yaml` edit takes effect on
the next request without a restart — and so a future WP with per-
session overrides can widen the provider's signature without
rewriting policy code.

The policy's scope is *decision about in-surface tool calls*, not
*closure of the surface*. AS-6 closure lives in `TOOL_NAMES` (FC-5);
dispatch's unknown-tool filter short-circuits before consulting
Autonomy. This means the gate never sees names outside the five
committed tools and cannot be a source of AS-6 leakage regardless of
level configuration.

Unknown levels fall back to baseline-silent rather than raising —
safer against future level names leaking into config ahead of policy
code. The operator sees missing surfacing rather than a locked-out
session.

**Visibility form (OQ #2 resolution).** `pure-tool-user-visible`
emits `VisibilityEvent(kind="composition", payload=...)` for
`compose_ensemble`. The SSE formatter renders the event as
`[composition: {json}]` narration on `delta.content`, and the non-
streaming response-body collector renders it identically. Chosen over
SSE comment lines because vanilla OpenAI-compat clients (OpenCode /
Roo Code / Cline) ignore comments per spec — the tool user's
conversation thread is the only place narration is actually
observable. The `[kind: {json}]` shape is generic across future event
kinds.

### Key integration points

- **← Orchestrator Tool Dispatch:** `decide(tool_name, arguments)` per dispatch for a committed tool name.
- **→ Level provider (closure):** operator-configured Autonomy Level via `OrchestratorConfigResolver.resolve().autonomy_level`.
- **→ `orchestrator_chunk.VisibilityEvent`:** the event type emitted when a tightened level surfaces a composition.

### Scenarios covered

- `scenarios.md` §Default Autonomy Level permits invocation, permits composition, gates promotion — acceptance in `test_api_v1_chat_completions.py::TestAutonomyAndPromotionAcceptance`; structural assertion that `"promote_ensemble" not in TOOL_NAMES`.
- `scenarios.md` §Tool user without operator role observes composition events when configured — acceptance same class; narration appears in `choices[0].message.content` between assistant turn segments.
- `scenarios.md` §Pure tool-user session at default Autonomy Level experiences silent composition — same class; no `[composition:` substring in response content.
- `scenarios.md` §Script authorship is never permitted at any Autonomy Level — same class, parametrized over `[BASELINE, TIGHTENED, synthetic-future]`; AS-6 closure via `TOOL_NAMES` unknown-tool filter.

### FC-11 enforcement

`tests/unit/agentic/test_fc11_autonomy_gate.py` — strict AST dominance
check. Three properties on `OrchestratorToolDispatch.dispatch`:

1. At least one `self._autonomy_policy.decide` call exists.
2. The `_route` indirection is used (routing goes through a single
   call site whose lexical order is checkable).
3. Every `await self._route(...)` call in `dispatch` is lexically
   after the first `decide` call — a fast-path bypass (cached-result
   early-return that skips the gate) trips the check.

Adversarial self-test verifies the detector catches a synthetic
bypass. Mirrors FC-8's template (`test_fc8_summarizer_bypass.py`).

### Boundary integration

`tests/integration/test_tool_dispatch_autonomy_policy.py` — real
`AutonomyPolicy` behind a delegating spy that records calls. Covers
(a) gate consulted for every tool in `TOOL_NAMES` at baseline;
(b) unknown-tool short-circuits before the gate; (c) tightened level
attaches the composition event to `compose_ensemble`'s result end-to-
end while leaving `invoke_ensemble` silent.

### Configured level

Operators set the default via `agentic_serving.autonomy.default_level`
in `config.yaml` (default: `operator-as-tool-user`). The Serving
Layer's `get_orchestrator_tool_dispatch` factory reads this per
decision so a `config.yaml` edit takes effect on the next request.
Per-session overrides land in a future WP.

---

## Module: Composition Validator

**Implementation state:** Complete (WP-G, 2026-04-22).
**Code location:** `src/llm_orc/agentic/composition_validator.py`. Depth helper at `src/llm_orc/core/config/ensemble_config.py:compute_reference_graph_depth`.
**Stability:** Settled. Six rejection branches + accept path exercised by 13 unit tests; boundary integration at `tests/integration/test_tool_dispatch_composition.py` covers the real wiring through `ConfigurationManager` + `OrchestraService` + file system.

### Domain Concepts in Code

| Concept | Code Manifestation | Location |
|---------|-------------------|----------|
| Composition | `CompositionValidator` class; `CompositionRequest`, `CompositionAccepted`, `CompositionRejected` result variants | `composition_validator.py` |
| Validation routine (shared) | `validate_ensemble_reference_graph` — public function, one definition, four call sites | `core/config/ensemble_config.py:309` |
| Depth check (shared graph walk) | `compute_reference_graph_depth` — sibling of the cycle validator, reuses `_build_reference_graph` | `core/config/ensemble_config.py:compute_reference_graph_depth` |
| Primitive Registry (production) | `ConfigManagerPrimitiveRegistry` — wraps `ConfigurationManager` + `ScriptResolver` + ensemble directory discovery | `composition_validator.py` |
| Primitive Registry (test surface) | `PrimitiveRegistry` Protocol — handwritten doubles in `tests/unit/agentic/test_composition_validator.py::_StubRegistry` | `composition_validator.py` |
| Local-tier write (action) | `ConfigManagerEnsembleWriter.write` — persists accepted config to `.llm-orc/ensembles/{name}.yaml` with collision rejection | `composition_validator.py` |
| Write failure | `EnsembleWriteError(ValueError)` — single-exception narrowing for dispatch | `composition_validator.py` |

### Rejection kinds in code

| Kind | Trigger | Dispatch surface |
|------|---------|-----------------|
| `invalid_agent_schema` | Pydantic rejects the agent dict | `ToolCallError(kind="invocation_failed")` |
| `missing_dependency` | agent `depends_on` a sibling not in the ensemble | same |
| `internal_dependency_cycle` | intra-ensemble dep cycle | same |
| `invalid_fan_out` | `fan_out: true` without `depends_on` | same |
| `missing_primitive` | model_profile / script / ensemble reference does not exist (AS-6) | same |
| `cross_ensemble_cycle` | proposed ensemble closes a cycle with existing ensembles (Invariant 5) | same |
| `depth_limit_exceeded` | proposed reference graph walks deeper than `depth_limit` (Invariant 8) | same |

Malformed LLM-emitted arguments (missing `name`, wrong `description` type, non-list `agents`) short-circuit before the validator and surface as `ToolCallError(kind="invalid_arguments")`.

### Design Rationale

- **L1 placement.** System design assigns Composition Validator to L1 (Domain Policy). The module imports from L0 (`core/config/ensemble_config.py`, `core/execution/scripting/resolver.py`, `schemas/agent_config.py`) only. Tool Dispatch (L2) depends on this module through the `CompositionGate` Protocol.
- **Shared validator routine (FC-6).** `validate_ensemble_reference_graph` lives in `ensemble_config.py` as a public function and is called from the load path (`EnsembleLoader.load_from_file` via `search_dirs`), the list path (`list_ensembles` via `search_dirs=[directory]`), the MCP validate path (`ValidationHandler._collect_validation_errors`), and composition (`CompositionValidator.validate`). Any regression would affect all paths identically.
- **Composition-time strictness (AS-6).** Load-time tolerates dangling ensemble references silently (they are caught at execution by `EnsembleAgentRunner` when the parent tries to resolve). Composition-time enforces existence strictly because the orchestrator must not be able to create an ensemble that names a missing primitive. The stricter check lives only on the composition path; load-time behavior is unchanged.
- **Depth-left shift (Invariant 8).** `EnsembleAgentRunner` enforces depth at runtime by comparing `child_depth` to `depth_limit` before creating a child executor. Composition-time depth check uses the same arithmetic (depth 0 = leaf, N-edge chain = depth N) so a composed ensemble cannot sneak past runtime enforcement. Load-time is not changed — depth remains a runtime concern for existing library content.
- **Write collision discipline (AS-2).** The writer rejects if `.llm-orc/ensembles/{name}.yaml` already exists. This mirrors `EnsembleCrudHandler.create_ensemble`'s contract and prevents a composition from silently overwriting operator-authored content. No partial state on failure: the writer is only reached after `CompositionAccepted`.

### Key Integration Points

- **Orchestrator Tool Dispatch** — holds a `CompositionGate` (validator) and a `LocalEnsembleWriter` (writer). `compose_ensemble` parses arguments → validates → writes on accept. Dispatch-level tests pass scripted doubles; boundary integration passes the real classes.
- **Ensemble Engine** — via `validate_ensemble_reference_graph` and `compute_reference_graph_depth`. Composition Validator is the fourth call site of the shared validator (FC-6 regression test asserts the identity).
- **Configuration Manager** — `ConfigManagerPrimitiveRegistry` reads `get_model_profiles()` and `get_ensembles_dirs()`; `ConfigManagerEnsembleWriter` reads `get_ensembles_dirs()` to locate the local tier. A `config.yaml` edit takes effect on the next request.
- **ScriptResolver** — `ConfigManagerPrimitiveRegistry.script_exists` delegates to `resolve_script_path`, catching `ScriptNotFoundError` / `FileNotFoundError` as "does not exist."

### Wiring site

`src/llm_orc/web/api/v1_chat_completions.py:get_orchestrator_tool_dispatch` constructs `ConfigManagerPrimitiveRegistry(service.config_manager)` → `CompositionValidator(primitives=registry)` → `ConfigManagerEnsembleWriter(service.config_manager)` and hands them to `OrchestratorToolDispatch`. Tests override the factory via `monkeypatch.setattr` to inject a scripted dispatch.

---

## Module: Calibration Gate

**Implementation state:** Complete at TS-2 (WP-H, 2026-04-24). Cross-session persistence via Plexus is the WP-I extension.
**Code location:** `src/llm_orc/agentic/calibration_gate.py`. Default checker ensemble at `.llm-orc/ensembles/agentic-calibration-checker.yaml`.
**Stability:** Settled for the stateless baseline; the Plexus-backed store for cross-session trust lands behind the same public surface in WP-I.

### Domain Concepts in Code

| Concept | Code Manifestation | Location |
|---------|-------------------|----------|
| Calibration | `CalibrationRecord` (per-(session, ensemble) state — status, signals, invocations_seen) | `calibration_gate.py:49` |
| Quality Signal | `QualitySignal = Literal["positive", "negative", "absent"]` | `calibration_gate.py:41` |
| Calibration Status | `CalibrationStatus = Literal["in_calibration", "trusted"]` | `calibration_gate.py:47` |
| Calibrate (action) | `CalibrationGate.check_and_record` | `calibration_gate.py:169` |
| Checker Protocol | `CalibrationChecker.check` | `calibration_gate.py:67` |
| Production Checker | `EnsembleBackedChecker` — invokes a checker ensemble, parses `signal:` from response | `calibration_gate.py:232` |
| Composition-time registration | `CalibrationGate.mark_composed` | `calibration_gate.py:124` |

### Design Rationale

- **L1 without reaching into L3.** The gate takes a plain `session_id: str` on every call rather than importing `SessionIdentity` from Session Registry. Same pattern as Budget Controller (plain integers for cumulative accounting) — L1 policy modules stay layering-clean.
- **Per-session store inside the gate.** Unlike calibration state living on `SessionState`, the records live in `CalibrationGate._sessions` keyed by session-id string. Session Registry stays agnostic of calibration vocabulary; WP-I swaps an in-memory store for a Plexus-backed one behind the same surface without touching the Session Registry contract.
- **ADR-007 clause 2 enforced at the dispatch call site.** The Tool Dispatch helper `_calibration_check_safe` swallows any exception the gate or checker raises; `invoke_ensemble` continues to the summarizer and returns the success path. A crashed checker never becomes a tool error.
- **Trust transition uses a sliding last-N window of positives.** Scenario 3 ("calibration period extends") is satisfied structurally — a negative or absent signal in the last-N window keeps the ensemble `in_calibration` even past `invocations_seen >= N`. Recovery requires a clean run of N positives in a row.
- **Unparseable checker responses yield `absent`, never raise.** Honest about evaluability rather than silently assuming positive. An `absent` signal keeps the ensemble in calibration, which matches the "not evaluable → not yet earning trust" semantics.
- **Config-driven tuning.** `default_n` and `checker_ensemble` both live in `OrchestratorConfig.calibration` and are resolved from `config.yaml` per session start. Default of `N = 3` balances check cost against single-reading noise; default checker is `agentic-calibration-checker` (shipped YAML). Operators point at a stricter, hallucination-detecting, or rubric-based checker without code changes.

### Key Integration Points

- **Orchestrator Tool Dispatch** (L2). Dispatch's `invoke_ensemble` calls `_calibration_check_safe(session_id, name, raw_result)` after `operations.invoke` and before `harness.summarize`. Dispatch's `compose_ensemble` calls `gate.mark_composed(session_id, name)` after a successful local-tier write — validation failure and write failure do not register.
- **Session Registry** (L3) — indirectly. Tool Dispatch threads `state.identity.value` through `dispatch(call, *, session_id)`, and the Runtime reads the identity from `SessionContext.state`. The gate never imports Session Registry.
- **Ensemble Engine** (L0) — via the `EnsembleBackedChecker`'s `CheckerInvoker` Protocol (structurally satisfied by `OrchestraService`). The checker invokes the configured checker ensemble through the same invocation path `invoke_ensemble` uses for library ensembles.
- **Plexus Adapter** (L1) — reserved for WP-I. Cross-session trust persistence (scenario §Calibration persists across sessions when Plexus is active) layers behind the current public API without changing Tool Dispatch or Session Registry call sites.

---

## Module: Plexus Adapter

**Implementation state:** Planned (WP-I).
**Code location:** Not yet created.
**Stability:** Design-only. `query_knowledge` and `record_outcome` currently return `ToolCallError(kind="not_yet_wired")`.

---

## Module: Bootstrapping Pipeline

**Implementation state:** Planned (WP-J).
**Code location:** Not yet created.
**Stability:** Design-only. Depends on WP-I landing first.

---

## Cross-cutting: HTTP infrastructure

**Implementation state:** WP-C extended `HTTPConnectionPool` to read timeout values from `performance.concurrency.request_timeout.{connect,read,write,pool}`. Default read timeout raised from 30 to 180 seconds for local tool-calling models.
**Code location:** `src/llm_orc/models/base.py:17-81`.

---

## Cross-cutting: Tool-calling model surface

**Implementation state:** WP-C added `generate_with_tools` to `ModelInterface`; `OpenAICompatibleModel` is the first implementor. Anthropic-native and Google-native are future WPs.
**Code locations:**
- `src/llm_orc/models/base.py:91-186` — `ToolCallingResponse`, `ToolCall`, `ToolCallUsage`, `ToolCallingNotSupportedError`, `ModelInterface.generate_with_tools`, `supports_tool_calling` class flag.
- `src/llm_orc/models/openai_compat.py:30-191` — `supports_tool_calling = True`; `generate_with_tools` implementation parses OpenAI tool-calling response shape.

Any provider that overrides `generate_with_tools` and sets
`supports_tool_calling = True` can drive the orchestrator — the
Serving Layer's session start checks the flag and fails loudly
otherwise.

---

## Cross-cutting: `llm-orc serve` command

**Implementation state:** Complete.
**Code location:** `src/llm_orc/cli.py:390-420`.

Alias for `llm-orc web` — starts the same FastAPI app with
agentic-serving-oriented CLI framing. Use `serve` for agentic-client
deployments, `web` for browser UI. `llm-orc mcp serve` is unrelated
(MCP server for direct tool use).
