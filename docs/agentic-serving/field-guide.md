# Field Guide: Agentic Serving

**Generated:** 2026-04-21
**Derived from:** `system-design.md` v1.0 + amendments #1–#3, current implementation at WP-D close.

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

**Implementation state:** Complete for Phase 1 (Plexus injection reserved but not populated).
**Code location:** `src/llm_orc/web/api/v1_chat_completions.py`, `src/llm_orc/web/api/v1_models.py`, `src/llm_orc/web/api/sse_format.py`, `src/llm_orc/agentic/session_start.py`.
**Stability:** Settled on structure; body of `resolve_session_start_context` changes at Phase 2 (ADR-009).

### Domain concepts in code

| Concept | Code manifestation | Location |
|---------|-------------------|----------|
| Serving Layer | `chat_completions` endpoint + `list_models` endpoint | `v1_chat_completions.py`, `v1_models.py` |
| SessionContext | `SessionContext` dataclass | `agentic/session_start.py:42` |
| PromptFragment | `PromptFragment` dataclass (Phase 2) | `agentic/session_start.py:29` |
| SSE streaming | `_stream_completion` + `OpenAiSseFormatter` | `v1_chat_completions.py:207`, `web/api/sse_format.py` |
| Context Injection (action) | `resolve_session_start_context` | `agentic/session_start.py:56` |
| Session-start cache (FC-9 single call) | `SessionStartCache` | `agentic/session_start.py:68` |
| Runtime construction per session | `_build_runtime` | `v1_chat_completions.py:176` |

### Design rationale

The endpoint is the OpenAI-compat entry point; Session identity resolution
happens here before the Runtime is constructed so the Runtime sees a
clean `SessionContext`. Streaming and non-streaming share
`_resolve_context` so session-start cache semantics hold under both
paths (FC-9 boundary-integration tested).

Phase 2 Plexus injection is structurally reserved via the typed
function `resolve_session_start_context(context) -> list[PromptFragment]`
at a single call site inside `SessionStartCache.resolve`. ADR-009's
reservation is satisfied by signature + call site; Phase 1's empty
return is a body-only change.

### Key integration points

- **→ Session Registry:** `SessionRegistry.resolve_identity` + `get_or_create_state` via `_resolve_context`.
- **→ Orchestrator Configuration:** `get_orchestrator_config_resolver().resolve_validated()` via `_build_runtime`.
- **→ Orchestrator Runtime:** `runtime.run(context)` via `_collect_non_streaming` / `_stream_completion`.
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
| OrchestratorConfig (immutable per-session) | `OrchestratorConfig` dataclass | `orchestrator_config.py:63` |
| BudgetDefaults | `BudgetDefaults` dataclass | `orchestrator_config.py:42` |
| OverrideBounds (expressible) | `OverrideBounds` dataclass | `orchestrator_config.py:50` |
| Fail-fast Model Profile validation | `resolve_validated` | `orchestrator_config.py:124` |
| Allowlist for `/v1/models` | `list_allowed_model_profile_ids` | `orchestrator_config.py:140` |
| `ModelProfileNotFoundError` | `ModelProfileNotFoundError` | `orchestrator_config.py:23` |

### Design rationale

`resolve_validated` raises before session start if the configured
orchestrator profile is absent from the library — `/v1/models` shop
window is silent-drop (ADR-011 posture). Immutable config matches the
session-boundary discipline in ADR-011: changes apply to new
sessions, not active ones.

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

**Implementation state:** Complete for WP-C / WP-D scope. Conversation Compaction named but not implemented (no WP-C/WP-D scenario required it). Routing Decision generation materializes at WP-I.
**Code location:** `src/llm_orc/agentic/orchestrator_runtime.py`.
**Stability:** Settled. Runtime receives already-summarized `ToolCallResult` values from Tool Dispatch (Amendment #3); summarization is a Tool-Dispatch-side concern the Runtime stays unaware of.

### Domain concepts in code

| Concept | Code manifestation | Location |
|---------|-------------------|----------|
| Orchestrator Agent | `OrchestratorRuntime` class | `orchestrator_runtime.py:134` |
| ReAct loop | `run` async generator | `orchestrator_runtime.py:151` |
| Route (action) | LLM emits tool calls; Runtime dispatches | `orchestrator_runtime.py:182-194` |
| Invoke (Dynamic) (action) | `invoke_ensemble` tool call dispatched through Tool Dispatch | same |
| OrchestratorLLM Protocol | `OrchestratorLLM` | `orchestrator_runtime.py:50` |
| ToolDispatcher Protocol | `ToolDispatcher` | `orchestrator_runtime.py:66` |
| Tool schemas | `_build_tool_schemas` (five schemas, assertion-verified) | `orchestrator_runtime.py:73` |
| Exhaustion formatting | `_format_exhaustion_message` | `orchestrator_runtime.py:130` |

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

Scripted LLM fakes (in tests) satisfy `OrchestratorLLM` structurally
without subclassing `ModelInterface`; production wiring passes the
loaded `ModelInterface` directly (one shared type for tool-calling
responses across Runtime and `ModelInterface.generate_with_tools`).

### Key integration points

- **→ Budget Controller:** `check(turn_count, token_spend)` per iteration.
- **→ Tool Dispatch (via `ToolDispatcher` Protocol):** `dispatch(InternalToolCall)` per tool emission.
- **→ LLM (via `OrchestratorLLM` Protocol):** `generate_with_tools(messages, tools)` per iteration.
- **→ orchestrator_chunk:** yields `ContentDelta` / `Completion` / `InternalToolCallInFlight` / `InternalToolCallResult` to the Serving Layer.

---

## Module: Orchestrator Tool Dispatch

**Implementation state:** Complete for WP-C + WP-D wiring. `invoke_ensemble` interposes the Result Summarizer Harness on every return (Amendment #3). `list_ensembles` delegates to `OrchestraService.read_ensembles`. `compose_ensemble` / `query_knowledge` / `record_outcome` return typed `not_yet_wired` errors pending WP-G / WP-I.
**Code location:** `src/llm_orc/agentic/orchestrator_tool_dispatch.py`.
**Stability:** Settled on the closed-set structure and the Harness interposition. Three handler bodies change when WP-G / WP-I wires them.

### Domain concepts in code

| Concept | Code manifestation | Location |
|---------|-------------------|----------|
| Orchestrator Tool | Five methods named after `TOOL_NAMES` | `orchestrator_tool_dispatch.py:162-265` |
| TOOL_NAMES (closed set) | `frozenset` | `orchestrator_tool_dispatch.py:45` |
| Dispatch-by-name | `dispatch` method with `match-case` over the five names | `orchestrator_tool_dispatch.py:132` |
| Allowlist rejection | `case _:` arm → `ToolCallError(kind="unknown_tool")` | `orchestrator_tool_dispatch.py:151` |
| InternalToolCall | `InternalToolCall` dataclass | `orchestrator_tool_dispatch.py:66` |
| ToolCallResult (union) | `ToolCallSuccess \| ToolCallError` | `orchestrator_tool_dispatch.py:80-103` |
| ToolErrorKind | `Literal["unknown_tool", "not_yet_wired", "invocation_failed", "invalid_arguments", "summarization_failed"]` | `orchestrator_tool_dispatch.py:57` |
| EnsembleOperations Protocol | narrow facade over `OrchestraService` | `orchestrator_tool_dispatch.py:106` |
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

### Key integration points

- **← Orchestrator Runtime:** `dispatch(InternalToolCall)` per LLM tool emission.
- **→ OrchestraService (via `EnsembleOperations`):** `invoke` for `invoke_ensemble`, `read_ensembles` for `list_ensembles`.
- **→ Result Summarizer Harness:** `summarize(raw_result, raw_output=...)` on every `invoke_ensemble` return. The Harness in turn calls back through `EnsembleOperations` to invoke the configured summarizer ensemble.
- **WP-G will add:** composition validation path (shared `validate_ensemble_reference_graph`).
- **WP-I will add:** `query_knowledge` + `record_outcome` via Plexus Adapter.
- **WP-E will interpose:** Autonomy Policy gate on every dispatch call.
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

**Implementation state:** Planned (WP-E).
**Code location:** Not yet created.
**Stability:** Design-only.

### Scenarios WP-E covers

- `scenarios.md` §Default Autonomy Level permits invocation, permits composition, gates promotion.
- `scenarios.md` §Tool user without operator role observes composition events when configured.
- `scenarios.md` §Pure tool-user session at default Autonomy Level experiences silent composition.
- `scenarios.md` §Script authorship is never permitted at any Autonomy Level.

### Architecture

Interposed before every Tool Dispatch call. Session state read via
plain args (not by importing Session Registry — same posture as
Budget Controller).

---

## Module: Composition Validator

**Implementation state:** Planned (WP-G).
**Code location:** Not yet created. WP-A landed the prerequisite: `validate_ensemble_reference_graph` is now a public function in `core/config/ensemble_config.py`.
**Stability:** Design-only; prerequisite unblocked.

---

## Module: Calibration Gate

**Implementation state:** Planned (WP-H).
**Code location:** Not yet created.
**Stability:** Design-only.

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
