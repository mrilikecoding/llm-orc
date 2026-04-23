# Active RDD Cycle: Agentic Serving

**Started:** 2026-03-20
**Current phase:** BUILD (in progress — WP-A, WP-B, WP-C, WP-D, WP-E, WP-F, WP-G complete 2026-04-22. **TS-1 reached at WP-F close.** WP-H (Calibration Gate) / WP-I (Plexus Adapter) remain on the TS-2 / TS-3 path)
**Artifact base:** `docs/agentic-serving/`
**Essay:** `../essays/001-agentic-serving-architecture.md`

## Phase Status

| Phase | Status | Artifact | Key Epistemic Response |
|-------|--------|----------|----------------------|
| RESEARCH | ✅ Complete | `../essays/001-agentic-serving-architecture.md` | The DAG engine and ReAct loop are complementary, not competing. Plexus as knowledge graph doesn't just add memory -- it lowers the capability threshold for the orchestrator LLM, converting reasoning into retrieval. The cost framing may matter more than the capability framing. |
| DISCOVER | ✅ Complete | `../product-discovery.md` | Plexus as lib (push model) means ingestion boundary is file content, not LLM output -- quality emerges from enrichment, not curation. Tool user and ensemble author often the same person -- visibility is tinkering, not just debugging. Conductor ceiling unknown; orchestration may require frontier models regardless of graph population. Bootstrapping pipeline has a shape but quality gate is the enrichment layer, not upstream curation. |
| MODEL | ✅ Complete | `../domain-model.md` | Plexus should be optional (AS-8) -- design for stateless, benefit from Plexus when available. Enrichment pipeline maturity is an open question that determines whether the learning-system value proposition is real. Two-tier architecture: stateless orchestrator as baseline, Plexus as upgrade to learning system. |
| DECIDE | ✅ Complete | `../decisions/adr-001..011-*.md`, `../scenarios.md`, `../interaction-specs.md` | Plexus's more compelling frame is intra-session multi-agent substrate via consumer-registered lens grammars, not only cross-session memory. Per-ensemble lens registration would make the orchestrator's access polyglot. AS-4 preserved (lens is query-surface grammar applied during enrichment). Reframe is forward signal, not a current-cycle driver -- Plexus's lens design is in-progress. Captured as OQ #8 and essay reflection; folds back in a later cycle. |
| ARCHITECT | ✅ Complete | `../system-design.md`, `../roadmap.md`, `../ORIENTATION.md` (regenerated) | Retrofit mode: ensemble engine stays Layer 3 unchanged; 12 modules across 4 layers plus typed `resolve_session_start_context` function in Serving Layer; 13 fitness criteria. Client tool surface: Option C (turn-boundary delegation) is the commitment, scenario-gated — WP-F does not start until stress scenarios exercise the C/D distinction. Context Injection demoted from module to typed function (ADR-009 reservation is satisfied by signature + call site, not by a module). Consolidations of Orchestrator Configuration into Serving Layer and Calibration Gate into Runtime rejected: the former would invert layering; the latter would break FC-4. Roadmap has 10 WPs, 3 classified transition states; TS-1 (stateless orchestrator serving OpenCode) is the vision-named intermediate target. |
| BUILD | ▶ In Progress | WP-A..WP-G complete — see roadmap Completed Work Log. **WP-G complete** — 4 commits (`32d2dd3`, `9972ed3`, `e5f8ea0`, `804aeb7`, this commit). Next: WP-H (Calibration Gate) / WP-I (Plexus Adapter). | WP-G closed. `compose_ensemble` is fully wired. FC-6 structurally satisfied — one public definition of `validate_ensemble_reference_graph`; four call sites (load path, `list_ensembles`, `ValidationHandler`, Composition Validator). Composition-time validation enforces AS-6 primitive existence strictly (load-time tolerates dangling refs silently) and moves Invariant 8 depth enforcement left of the runtime check. Test suite: 2297 passing, 91.51% coverage, lint clean. TS-2 remaining: WP-H only. |
| DECIDE (mini-cycle from BUILD) | ✅ Complete 2026-04-22 | `../scenarios.md` §Client Tool Surface Commitment (5 scenarios), `../system-design.md` §Client Tool Surface Commitment Amendment #4, `../roadmap.md` ODP #1 closed + ODP #8 added, gate reflection at `gates/wpf-decide-gate.md`, audits at `audits/argument-audit-decide-005-wpf.md` (4 rounds: 5, 5b, 5c, 5d — all closed clean) + `audits/susceptibility-snapshot-wpf-decide.md` | WP-F scenario gate resolved — Option C confirmed, retry pattern is the mechanism for scenario (d) with a negative-path scenario covering silent quality degradation. Commitment to retry pattern is **conditional** — committed-to insofar as it delivers more capable agentic serving; revisitable if WP-F or post-WP-F reveals capability benefit is not borne out. Option D out of scope (requires ADR-001/002 amendment). Five scenarios are WP-F acceptance criteria. |
| PLAY | ☐ Optional | -- | -- |
| SYNTHESIZE | ☐ Optional | -- | -- |

## Feed-Forward Signals

### From RESEARCH
1. The orchestrator agent pattern (ReAct loop with ensemble invocation as primary tool) is the pragmatic entry point -- not the hybrid model
2. Invariant 7 (static ensemble references) does not govern orchestrator tool-call invocations, but orchestrator-created ensembles must still satisfy it. Validation mechanism needed at creation time
3. Plexus integration is the differentiator: no comparable system provides persistent structured memory across agentic sessions
4. The knowledge-compensated model selection hypothesis (populated graph enables cheaper models) is commercially significant and testable but unvalidated
5. Context management (summarization, compaction, Plexus offloading) is a correctness requirement, not an optimization
6. Budget enforcement (turn limits, token budgets) is a control plane concern validated by both OpenHands and claw-code

### From RESEARCH (reflections)
7. The cost framing (economics over capability) may be the stronger value proposition -- "80% quality at 10% cost" resonates more than "marginally better quality"
8. Open question for domain model: can well-orchestrated smaller models + populated knowledge graph compete with frontier models on quality while winning on cost?

### From DISCOVER
9. Plexus operates as a lib (push model), not a server -- the client drives ingestion of file content, enrichments extract signal. Quality gate is the enrichment pipeline, not upstream curation
10. The orchestrator needs the full composition palette (ensembles + profiles + scripts), including ensemble-to-ensemble references -- the "restrict to profile-and-script only" fallback from the essay is insufficient
11. Tool user and ensemble author are often the same person -- visibility is part of the value proposition (tinkering), not just an operational concern
12. Organic stabilization over explicit curation -- repeated patterns should surface on their own, but quality signals (not just frequency) must inform what stabilizes
13. The conductor skill's ceiling is unknown -- orchestration is broad AND deep, and may require frontier models regardless of graph population. The knowledge-compensated model selection hypothesis remains unvalidated
14. Bootstrapping pipeline: llm-orc artifacts → Plexus ingestion (file content) → enrichment → queryable graph. Background/async ingestion is practical. "Garbage in" concern is addressed by ingesting source material, not LLM summaries

### From MODEL
15. AS-8: Plexus is optional -- the orchestrator works statelessly without it. Design for stateless, benefit from Plexus when available
16. Two-tier architecture: stateless orchestrator (serving layer + ReAct loop + ensemble engine) as baseline product; Plexus as upgrade to learning system
17. Enrichment pipeline maturity (open question #7) determines whether the learning-system value proposition is real. Not a blocker but load-bearing for AS-4 and AS-5
18. 8 invariants (AS-1 through AS-8), 17 concepts, 13 actions, 7 open questions in the scoped model

### From DECIDE
19. 11 ADRs accepted, cycle-scoped at `../decisions/`. All post-audit revisions applied; three argument-audit rounds closed clean
20. Budget (ADR-005) sized for long agentic sessions — outer bound of "running an RDD phase within a session." Specific turn/token numbers deferred to build; enforcement mechanism fixed
21. Full composition palette with validation (ADR-006) overrides the essay's "restrict to profile-and-script" fallback. Conformance debt: cross-ensemble cycle validator is currently private in `EnsembleLoader` — must be extracted to a public function before `compose_ensemble` is built. Captured as refactor scenarios 1-3 in `scenarios.md`
22. Fixed orchestrator tool surface of exactly 5 tools (ADR-003). No dynamic tool extension
23. Result summarization is mandatory and has a Plexus-active vs. stateless split (ADR-004). In stateless mode, lost summarization detail is unrecoverable by the orchestrator
24. Calibration is session-scoped when Plexus is absent; persists across sessions when Plexus is active (ADR-007)
25. Autonomy Level baseline is calibrated for the operator-as-tool-user persona from DISCOVER. Pure tool-user deployments (FF-2) may warrant a tighter default that surfaces composition events — add as architectural configuration surface
26. Plexus integration phased (ADR-009): Phase 1 tool-first; Phase 2 context injection deferred. **Phase 2 hook point is structurally reserved** (post-gate reframe applied): ARCHITECT's session-start flow design must include a pre-orchestration stage where injection can be inserted without modifying the ReAct loop (ADR-001) or tool surface (ADR-003). Phase 1 leaves the stage empty; Phase 2 populates it. Technical rationale for Phase 1 sufficiency: linked to orchestrator profile capability (ADR-011) and OQ #1 (knowledge-compensated model selection)
27. Orchestrator LLM is a Model Profile (ADR-011); no hard-coded tiered fallback. Tiered routing is expressible as a composed ensemble — but only once such an ensemble has been composed and promoted
28. Client tool surface boundary is an open decision discovered while writing interaction specs. How the orchestrator handles client-declared tools (bash, file-edit, etc.) vs. its internal tool surface (ADR-003) needs resolution in ARCHITECT or a follow-up DECIDE mini-cycle

### From DECIDE (reflection gate)
29. Plexus's more compelling frame (per user, 2026-04-19) is intra-session multi-agent substrate via consumer-registered lens grammars. Per-ensemble lens registration would make orchestrator access polyglot — each subagent writes and queries through its own grammar over a shared enrichment. AS-4 is preserved under this reading (lens is enrichment-time query grammar, not LLM-summary ingestion). In-design on the Plexus side; captured as OQ #8 and essay reflection; folds back in a later cycle. Not a driver for current-cycle ADR changes

### From BUILD
39. **Budget defaults sized for local-orchestration value prop** (commit `b1e3c54`, 2026-04-20). Default `turn_limit=500`, `token_limit=10_000_000`; override bounds raised proportionally. Reframing: llm-orc's value proposition is that orchestration with local-hardware compute trades tokens-for-quality against a single frontier-API call. The token ceiling is a pathology circuit breaker for the local-orchestration-heavy case, not a cost ceiling for frontier-API pricing. Frontier-mix deployments tighten via `config.yaml`. ADR-005 flagged this as a build-phase tuning decision; the new numbers track cost-framing reflection (FF #7) and OQ #1 (knowledge-compensated model selection). ADR-005's sizing framing could be amended later to make the local-first default explicit, but the numbers speak for themselves for now
40. **`DEFAULT_MODEL_PROFILE = "default"` is a placeholder in WP-B** (`orchestrator_config.py:22`). Resolves to the literal string "default" until WP-B Group 3 (`/v1/models`) wires it against the actual Model Profile library via `ConfigurationManager.get_model_profiles()`. At that point the resolver either lands on a real profile or raises "no default configured" — Group 3's decision. **Resolved at Group 3** (commit `86ed0be`): `resolve()` stays pure translation; new `resolve_validated()` method raises typed `ModelProfileNotFoundError` when the configured profile is absent from the library. Session-start (WP-C) calls `resolve_validated`; `/v1/models` calls `list_allowed_model_profile_ids` which silently intersects with the library
41. **WP-B Phase 1 defers per-request override *application*.** The `OverrideBounds` config surface is in place (operators can express bounds); the mechanism that applies request overrides and clamps/rejects against bounds is deferred until a concrete scenario demands it. `OrchestratorConfigResolver.resolve()` is parameterless in Phase 1
42. **Session Registry is in-process memory for WP-B.** No cross-process persistence. Sufficient while Autonomy Level and Calibration do not require it (baseline deployment). Persistence is added when AS-3/AS-5/AS-7 enforcement demands it — the contract already accommodates the change (system-design §Session Registry "persistence of Session state across HTTP requests when persistence is required by Autonomy Level or Calibration state")
43. **Tier 1 stewardship clean at WP-B Groups 1-2 boundary.** Dependency direction respects L3 → L0 layering; FC-1/FC-2/FC-3 pass; lifecycle-composition test added for SessionRegistry's shared-reference pattern (caller mutation visible through subsequent lookup by the same identity — commit `59b9053`). Boundary integration tests deferred to Group 4+ (Serving Layer construction). No ADR or domain-model amendments triggered
44. **Allowlist shape decided at Group 3** (commit `86ed0be`). New operator config key `agentic_serving.orchestrator.allowed_profiles: [...]` governs what `/v1/models` exposes. Defaults to `[model_profile]` when unset, so single-profile deployments work without additional configuration. Extensible if tiered-routing deployments want to expose multiple orchestrator profiles. `OrchestratorConfig.allowed_profiles: tuple[str, ...]` is the typed surface
45. **`/v1/models` intersects allowlist with `ConfigurationManager.get_model_profiles()` and returns the OpenAI-compatible shape** (`{"object": "list", "data": [{"id", "object":"model", "created":0, "owned_by":"llm-orc"}, ...]}`). Absent profiles silently drop out; the endpoint is a shop window, not a validator — missing-profile errors surface at session start via `resolve_validated`. `created` and `owned_by` are hard-coded (no metadata field on model profiles tracks them today)
46. **Config-staleness posture: per-request re-read.** `list_allowed_model_profile_ids` calls `resolve()` on every request, which re-reads `config.yaml` through the existing ConfigurationManager cache. Cheap today and lets operators tune `config.yaml` without restart. Worth revisiting when session handoff lands in WP-C and the session needs a consistent config snapshot across turns
47. **UX caveat: silent-drop on empty `/v1/models`.** A naive operator running the server for the first time with no profiles configured will see `/v1/models` return an empty `data` array with no hint why. Session start raises (`ModelProfileNotFoundError`) but the browse-before-session flow is quiet. Worth a future health/status endpoint that surfaces configuration problems before the first chat request. Not in WP-B scope
48. **Scenario-phrasing drift noted.** `scenarios.md` §"Orchestrator tool set is exactly the committed set" reads "the `/v1/models` endpoint is queried for the orchestrator's available tools". The roadmap splits this correctly (`/v1/models` side = WP-B Group 3; five-tool side = WP-C via Orchestrator Tool Dispatch), but the scenario wording conflates models and tools. Minor edit worth making when scenarios are next touched

49. **FC-9 lands via `SessionStartCache` at Group 4.** `resolve_session_start_context` has one definition (`session_start.py:56`) and one live invocation — `self._resolver(context)` inside `SessionStartCache.resolve`. The once-per-session invariant is a testable property of the cache class, not a comment-level invariant on the Serving Layer. Initial draft placed the "was session-start run?" state on `SessionState` via a `TYPE_CHECKING` import of `PromptFragment`; tidy-first reflection moved it into `SessionStartCache` in `session_start.py`, restoring Session Registry to minimal accounting and keeping the once-per-session discipline next to the resolver it guards. Phase 2 (Plexus injection) and any future mid-session refresh logic live inside the cache class rather than rippling through `SessionState`

50. **Group 4 module layout.** Group 4 introduces two new files: `llm_orc/agentic/session_start.py` (typed `PromptFragment`, `SessionContext`, `resolve_session_start_context`, `SessionStartCache`) and `llm_orc/web/api/v1_chat_completions.py` (POST endpoint with factory-based dependency injection, matching the `v1_models.get_orchestrator_config_resolver` pattern). `server.py` wires the router. `stream=true` returns HTTP 400 with a Group-5-deferred message — silent downgrade would mis-frame responses to SSE-expecting clients

51. **Orchestrator Runtime handoff is stubbed.** `v1_chat_completions._orchestrator_handoff(context) -> str` returns `""`. WP-C replaces the stub with the real ReAct loop. The placeholder response body carries `finish_reason: "stop"`, empty content, and zero `usage` — an honest OpenAI-shaped reply with nothing to say yet. The structural handoff edge exists so WP-C is a body swap, not a call-site addition

52. **Group 4 test coverage.** 19 new tests: `tests/unit/agentic/test_session_start.py` (12 — types, Phase 1 resolver behavior, `SessionStartCache` once-per-identity / per-identity resolution / empty-list caching invariant / default resolver wiring); `tests/unit/web/test_api_v1_chat_completions.py` (14 — response shape, request parsing, streaming rejection, session-start integration including the named `test_session_start_context_is_empty_in_phase_1` from the system-design Test Architecture table). Full suite 2127 passed, 91.14% coverage, lint clean

53. **OrchestratorChunk is six variants, neutrally located** (Group 5). `src/llm_orc/agentic/orchestrator_chunk.py` defines `ContentDelta`, `Completion`, `ClientToolCall` (+ inner `ToolCallInvocation`), `InternalToolCallInFlight`, `InternalToolCallResult`, `ErrorChunk` as frozen dataclasses joined under a `OrchestratorChunk` union alias. Placement is neutral (under `agentic/`) rather than under `web/` because the Runtime — a future L2 module — also imports these types. The formatter lives in `web/api/` because OpenAI protocol translation is Serving-Layer-specific; keeping them split preserves FC-4's spirit (Runtime never imports SSE-specific code)

54. **Internal tool-call chunks silent-drop in Phase 1** (Group 5, formatter). `InternalToolCallInFlight` and `InternalToolCallResult` exist in the Runtime's emission surface but the formatter returns `b""` for both — the Runtime yields them so the Serving Layer can choose what to surface, and Phase 1's choice is "nothing" pending OQ #2 (visibility form). Documented in `sse_format.py` source. WP-E can change the formatter case without touching the Runtime's emission contract

55. **Stream opener is a Serving-Layer convention, not a Runtime chunk** (Group 5). `OpenAiSseFormatter.start_assistant_turn()` emits OpenAI's first-chunk `delta.role` convention. Rejecting the alternative of a `RoleDelta` chunk variant keeps OpenAI-protocol specifics out of the Runtime ↔ Serving Layer contract. If a future protocol needs a different opener (e.g., Anthropic-style), the formatter swaps; the chunk vocabulary is unchanged

56. **Client tool-call emission is single-chunk for the skeleton** (Group 5). `ClientToolCall(tool_calls=(...))` emits one framed chunk carrying all tool calls with a synthetic `index` per call and `finish_reason: "tool_calls"`. Chunked argument streaming (OpenAI allows multi-chunk reconstruction by index) is deferred to WP-F — Group 5 gives WP-F the single-chunk baseline to extend from. `arguments` is stored as a pre-encoded JSON string, matching OpenAI's wire format and avoiding re-encoding inside the formatter

57. **`_resolve_context` extracted so streaming and non-streaming share pre-handoff work** (Group 5). Identity resolution, state retrieval, and session-start cache resolution happen synchronously before the response is shaped — so errors in any of those steps surface as proper HTTP errors rather than as SSE errors mid-stream. FC-9 under streaming is covered by two new tests (`test_streaming_request_fires_session_start_exactly_once_per_session`, `test_streaming_and_non_streaming_share_session_start_cache`). Cache is the same singleton used by the non-streaming path

58. **`response_model=None` is required when returning `dict | StreamingResponse`** (Group 5). Without it, FastAPI tries to build a Pydantic response model from the union and raises at import time. Documented inline at the decorator. Future streaming endpoints (e.g., a Phase 2 visibility-events stream) will need the same opt-out

59. **Group 5 test coverage.** +14 tests net (9 SSE formatter unit tests, 4 streaming endpoint tests, 2 FC-9-under-streaming integration tests, −1 superseded Group 4 `stream=true` rejection test). Full suite 2141 passed, 91.21% coverage, lint clean

60. **In-session ensemble design needs in-stream visibility** (Group 5 gate, 2026-04-21). The llm-conductor pattern — user + orchestrator composing ensembles mid-session and watching them invoke and calibrate — only closes its feedback loop when composition, invocation, and quality-signal events are observable on the same SSE stream the user is conversing through. A separate dashboard or events surface forks attention away from the conversation, breaking the "did this work?" loop that makes live ensemble design viable at all. Shifts OQ #2 toward "multiplex visibility into the existing SSE response" rather than "separate events surface." Directional lean (not committed): SSE comment lines (``: {json}\n\n``) over extended `data:` JSON — OpenAI-compat clients (OpenCode, Roo Code, Cline) ignore SSE comments per the W3C spec; llm-orc operator tooling parses them. Strict compat preserved; one surface. WP-E lands the implementation when it resolves OQ #2. **Validation deferred to `/rdd-play`:** stakeholder inhabitation tests whether in-stream events actually close the feedback loop in practice; field notes categorize findings (missing events, surplus noise, form-of-rendering problems); feedback recycles into DISCOVER or a mini-DECIDE cycle if the direction needs adjustment before WP-E commits. `sse_format.py` source comment on the `InternalToolCallInFlight | InternalToolCallResult` case carries the directional pointer so WP-C does not re-anchor on silent-drop

61. **FC-9 closes structurally at WP-B Group 6** (2026-04-21). Behavioral FC-9 tests (once-per-session caching under streaming and non-streaming) have existed since Groups 4-5. Group 6 adds the structural complement: AST scan over `src/llm_orc/` verifies exactly one `FunctionDef` for `resolve_session_start_context` and exactly one `ast.Name` reference outside that definition (the `SessionStartCache.__init__` default-resolver binding). All production invocations flow through `self._resolver(context)` inside the cache, not through the bare name — so a future WP adding a second direct caller would be caught by the static test before it reached review. The structural invariant is what makes ADR-009's Phase 2 reservation load-bearing: Phase 2 can land as a function-body change because the single call path is mechanically enforced, not convention-enforced.

62. **`test_serving_resolves_session_identity` lands the Serving Layer → Session Registry boundary** (Group 6). Five integration tests in `TestServingResolvesSessionIdentity` exercise the edge with a real `SessionRegistry` (no mocks at the boundary): same `user` field resolves to one `SessionState`; mutation-between-requests is visible to the next request through the retained reference (HTTP-level lifecycle-sequence check); distinct `user` fields resolve distinctly; message-prefix derivation groups by first user message when `user` is absent; cold-start requests each get a fresh identity. The mutation test is structurally the same check as the unit-level `test_caller_mutation_visible_through_subsequent_lookup` in `test_session_registry.py`, lifted to the integration tier to confirm the production call chain (FastAPI → `_resolve_context` → `SessionRegistry.get_or_create_state`) preserves the shared-reference lifecycle contract.

63. **WP-B complete; WP-C unblocked.** The roadmap's TS-1 vision advances next through WP-C (Orchestrator Runtime, Orchestrator Tool Dispatch, Budget Controller). The existing `_orchestrator_handoff` and `_orchestrator_stream_handoff` stubs in `v1_chat_completions.py` are the body-swap points — structural handoff edges exist so WP-C is function-body replacement, not call-site addition. FC-4 (Runtime imports only Budget Controller, Orchestrator Tool Dispatch, Result Summarizer Harness — no Plexus, no config, no Autonomy, no Calibration) becomes the load-bearing fitness criterion at WP-C start.

64. **Budget Controller is layering-clean by construction** (WP-C Group 1, commit `790f596`). System design has `Budget Controller → Session Registry` in the dependency graph (L1 → L3, which violates the stated layering rule L1 may depend only on L0). The implementation sidesteps: `BudgetController.check(turn_count, token_spend)` takes plain integers via method args; the module has zero `llm_orc.agentic.*` imports. The Runtime (L2) reads `context.state.turn_count` / `token_spend` from `SessionContext` and passes them in. The `Budget Controller → Session Registry` edge can be removed from the dependency graph at the next system-design amendment pass. Same posture applies to Autonomy Policy and Calibration Gate when they land (WP-E, WP-H) — take plain args for Session state rather than importing `SessionRegistry`.

65. **Tool Dispatch delegates to OrchestraService; no parallel ensemble-execution path** (WP-C Groups 2+refactor, commits `07032a9`, `90df826`). Group 2 shipped with a parallel find-and-execute path (iterated `config_manager.get_ensembles_dirs()`, called `EnsembleLoader.find_ensemble`, invoked an injected executor) that duplicated `ExecutionHandler.invoke`. The refactor (`90df826`, same-day) collapsed it: `OrchestratorToolDispatch` takes an `EnsembleOperations` Protocol that `OrchestraService` satisfies structurally; `invoke_ensemble` delegates to `operations.invoke({ensemble_name, input})`, `list_ensembles` delegates to `operations.read_ensembles()`. Observable change: the LLM now sees the normalized `{results, synthesis, status}` shape rather than the raw executor dict, and `list_ensembles` entries include `source` / `relative_path` / `agent_count` from `ResourceHandler`. Future changes to ensemble name resolution propagate through one code path.

66. **Runtime uses `match-case` dispatch over `getattr`** (WP-C Group 2). `getattr(self, call.name)` loses mypy's return-type tracking (Any leaks through). The match-case version is more verbose but makes the five-tool closed set visible at the dispatch site and keeps typing intact. FC-5's static check counts public async methods whose names are in `TOOL_NAMES`; the match-case form satisfies it and is the natural expression of ADR-003's closure.

67. **OrchestratorLLM Protocol uses `Awaitable` return rather than `async def`** (WP-C Group 3, mypy quirk). `async def` in a Protocol degrades the inferred return type to `Any` (a known mypy gap). `def ... -> Awaitable[...]` preserves the narrowed return type through `await`. Applied consistently on `OrchestratorLLM.generate_with_tools` and `EnsembleRuntimeExecutor.execute` (before the delegation refactor dropped the latter). Worth the same workaround on future tool-calling-related Protocols.

68. **Tool-calling is opt-in per provider via `ModelInterface.supports_tool_calling`** (WP-C Group 4a, commit `061312e`). `ModelInterface` grew `generate_with_tools(messages, tools) -> ToolCallingResponse` with a default body that raises `ToolCallingNotSupportedError`. Providers override the method and set the class-level `supports_tool_calling = True` flag. `OpenAICompatibleModel` is the first implementor (WP-C Group 4b, commit `e48c7b8`) — covers Ollama, OpenAI, OpenRouter, LM Studio, vLLM. Anthropic-native and Google-native opt in via follow-up WPs; the orchestrator's session start verifies `supports_tool_calling` and fails loudly otherwise. One tool-calling contract, one provider surface.

69. **`OrchestratorLLM` Protocol is satisfied by `ModelInterface` directly** (WP-C Group 4a). The earlier Runtime-local `LLMResponse` / `LLMToolCall` / `LLMUsage` types were replaced by `ToolCallingResponse` / `ToolCall` / `ToolCallUsage` in `models/base.py`. Serving Layer hands a loaded `ModelInterface` directly to the Runtime — no adapter. Session start's `_default_orchestrator_llm_loader` resolves the model via existing `ModelFactory.load_model_from_agent_config`.

70. **WP-C acceptance verified end-to-end against Ollama with `mistral-nemo:12b`** (2026-04-21). Two full runs documented in `housekeeping/wp-c-manual-verification.md` — initial run surfaced three gaps (serve command, provider key, HTTP timeout), all addressed in follow-up commits (`65b1334`, `bb7b466`, `22deeaf`, `bab8e1d`). Re-verification (`12c19ac`) passed all four acceptance checks: `/v1/models` → list, non-streaming completion with real tool-calling (65 ensembles rendered, 2m49s), streaming SSE (22s), budget exhaustion graceful. The counter-display observation `(2/1)` vs `(1/1)` was documentation (session is cumulative, not per-request), not a code bug.

71. **`llm-orc serve` alias for `llm-orc web`** (commit `65b1334`). Same FastAPI app; different name for the agentic-serving deployment context. `serve` omits `--open`; `web` keeps it. No `--no-ui` flag yet — add when production deployments want minimal attack surface.

72. **HTTP read timeout wired through performance config; default raised to 180s** (commits `bb7b466`, `22deeaf`). `performance.concurrency.request_timeout.{connect,read,write,pool}` overrides per-field defaults. Sized for local tool-calling models (30-80s per iteration observed on mistral-nemo / qwen2.5 / llama3.1). Remote providers tune down; local deployments work out of the box.

73. **WP-D Design Amendment candidate** — system design amendment backlog. The `Orchestrator Runtime → Result Summarizer Harness` dependency in the system design contradicts the Harness module's own "Runtime is not aware of the summarizer" rationale. Correct reading: RSH is interposed by Tool Dispatch on `invoke_ensemble`'s return path, not imported by Runtime. WP-D should land the amendment alongside the Harness: remove `Runtime → RSH` from Dependency Graph; add `Tool Dispatch → RSH`; update FC-4 to drop RSH from Runtime's import set (Runtime imports only Budget Controller + Tool Dispatch at WP-C close anyway — FC-4's list is already met by construction). The FC-4 static test (`test_fc4_runtime_import_surface.py`) enforces the corrected reading today.

74. **FC-4 structurally enforced** (WP-C Group 5, this change). AST scan over `orchestrator_runtime.py` asserts zero imports of `orchestrator_config`, `session_registry`, and the future `plexus_adapter` / `autonomy_policy` / `calibration_gate` modules. The three unborn modules named in the forbidden list fail the test closed when they land, forcing an explicit allow/deny decision.

75. **Boundary integration: Tool Dispatch → Ensemble Engine** (WP-C Group 5, this change). `tests/integration/test_tool_dispatch_ensemble_engine.py` exercises the full production call chain with real types at every boundary: `OrchestratorToolDispatch` → `OrchestraService` → `ExecutionHandler` → `EnsembleLoader.find_ensemble` → `EnsembleExecutor.execute` → `MockModel`. Closes the Test Architecture table row `test_invoke_ensemble_executes_real_ensemble` and verifies `list_ensembles`' ResourceHandler-shaped content passes through untouched.

76. **Debt surfaced, not addressed in WP-C scope** — carried to follow-ups.
    - Conversation Compaction (Runtime-owned per system design; not needed by WP-C scenarios; can land alongside any future Runtime edit).
    - Per-request prompt-vs-completion token split in `/v1/chat/completions` response body (currently the endpoint reports `completion_tokens = turn_delta` and `prompt_tokens = 0`; Runtime accumulates `total_tokens` on Session state but does not preserve the split).
    - Routing Decision generation (Runtime-owned per system design; materializes when WP-I wires `record_outcome`).
    - `core/execution/ensemble_execution.py:808` still reaches `EnsembleLoader._find_ensemble_in_dirs` with the underscore prefix (WP-A debt note). Not a WP-C blocker.

### From BUILD (WP-D Groups 5-6)

77. **FC-4 explicitly forbids `result_summarizer_harness` from Runtime imports** (commit `4261238`). Amendment #3 moved RSH to a Tool-Dispatch-side concern; the FC-4 test previously neither allowed nor forbade RSH (ambient). Now it is on the forbidden list alongside Plexus / Autonomy / Calibration / config / Session Registry. Summarization-side code cannot leak into the Runtime's reasoning surface even by accident.

78. **FC-8 structurally enforced via strict AST dominance check** (commit `903833e`, `tests/unit/agentic/test_fc8_summarizer_bypass.py`). Three properties hold on `OrchestratorToolDispatch.invoke_ensemble`: (a) it calls `await self._harness.summarize(...)` at least once; (b) every `ToolCallSuccess(...)` constructor in the method body is nested inside a `match` block whose subject is the summarize-result binding; (c) the summarize call lexically precedes the match. An adversarial self-test parses a synthetic bypass fixture and asserts the detector flags it — without the self-test, a bug in the detection logic would silently let real regressions through. The strict-over-loose choice was deliberate: legibility cost is paid in exchange for a test that catches the class of regressions (early-return fast paths, short-circuit branches) a "harness is mentioned somewhere" check would miss.

79. **Raw-output escape-hatch acceptance scenario passes at Serving Layer** (commit `03885f8`, `tests/unit/web/test_api_v1_chat_completions.py::TestRawOutputEscapeHatchAcceptance`). Real `OrchestratorToolDispatch` + real `ResultSummarizerHarness` wired over a stub `EnsembleOperations`; scripted LLM drives a full POST `/v1/chat/completions` round trip. Proves (a) `raw_output=True` bypasses the summarizer (observable: operations.invoke called exactly once for the analysis ensemble, never for the summarizer); (b) `raw_output` absent routes through the summarizer (observable: two invoke calls, second is the summarizer). The flag is opt-in per ADR-004 — confirmed end-to-end, not just in Harness unit tests.

80. **Tool Dispatch → RSH → Ensemble Engine summarize-path boundary integration** (commit `2f0f660`, `tests/integration/test_tool_dispatch_summarizer_boundary.py`). Complements the pre-existing `test_tool_dispatch_ensemble_engine.py` (which exercised only the raw_output=True pass-through branch). Real `OrchestraService` + real Tool Dispatch + real Harness + real `EnsembleExecutor` against MockModel; library has an analysis ensemble and a single-agent `test-summarizer` ensemble. Asserts the orchestrator receives a `{"summary": <str>}` payload carrying MockModel's signature phrase, with `raw_output` / `results` keys explicitly absent from the content. Second test points the Harness at a non-existent summarizer and asserts `ToolCallError(kind="summarization_failed")` — never a raw-dict leak on failure.

81. **Summarizer-quality echo-back risk carried forward to Calibration Gate (WP-E / WP-H)**. The FC-8 structural enforcement proves the Harness is always interposed; it does not prove the Harness's output is substantively a summary rather than an echo of the raw dict. A weak or compromised summarizer ensemble could place a JSON-encoded raw dict in its `response` field, and the Harness would return it as-is — the raw-dict leak would arrive through the summarizer's legitimate output channel rather than by bypassing the Harness. This is a **quality property of the configured summarizer**, not a structural bypass, and is precisely what Calibration Gate (ADR-007) is designed to govern: the first N invocations of any composed ensemble are result-checked for Quality Signals. **Decision**: leave WP-D as-is; the mechanism lives in WP-E / WP-H calibration scope. The failure mode is visible (weird summaries in the orchestrator's context, observable via SSE and artifacts) and recoverable (swap the configured `summarizer_ensemble` via `config.yaml`). This note is the forward signal; WP-E / WP-H should consider summarizer quality in their calibration check-mechanism scope.

82. **`test_runtime_never_sees_unsummarized_result` row in the Test Architecture table is now covered end-to-end.** The system-design Test Architecture table names this test for the Orchestrator Tool Dispatch → Result Summarizer Harness edge. Post-WP-D, three complementary tests share the coverage: (a) FC-8 static dominance (`test_fc8_summarizer_bypass.py`); (b) raw-output acceptance scenario at the Serving Layer; (c) summarize-branch boundary integration at Tool Dispatch → Ensemble Engine. No single test renaming; the table row's intent is distributed across the three. Consider tightening the Test Architecture table row in a future edit to point at the three tests explicitly.

### From BUILD (WP-E)

83. **OQ #2 resolved in favor of tool-user-visible `delta.content` narration** (2026-04-22, WP-E planning). The llm-conductor tinkering loop — user observing composition, editing YAML, re-running — only closes when composition is observable in the same conversation thread the tool user converses through. SSE comment lines (FF #60's earlier directional lean) would serve operator tooling but vanilla OpenAI-compat clients (OpenCode / Roo Code / Cline) ignore comments per spec. Structured non-standard `data:` fields risk strict clients dropping the stream. `delta.content` narration in the shape `[kind: {json}]` reaches every compliant client inline. Operator-parseable surfaces (SSE comments, dedicated events endpoint) can layer on as a second surface later without changing the `delta.content` emission.

84. **Events-on-result over DispatchOutcome wrapper** (WP-E Group 2 design choice). Adding `events: tuple[VisibilityEvent, ...] = ()` to both `ToolCallSuccess` and `ToolCallError` kept the `ToolDispatcher` Protocol signature unchanged; a `DispatchOutcome(result, events)` wrapper would have rippled across ~15 existing call sites for the same semantic payload. The events tuple defaults to empty — the common Allow-no-events path has no allocation overhead.

85. **`_route` factoring to make FC-11 lexically checkable** (WP-E Group 2). The match-case in `dispatch` was split into a private `_route` method so FC-11's AST dominance check has a single `await self._route(...)` call site whose lexical order vs. `self._autonomy_policy.decide` is trivial to verify. Regression protection: `test_dispatch_routes_exactly_via_self_route` asserts `_route` is the only routing hop; an inlined regression trips the test.

86. **AS-6 closure stays in `TOOL_NAMES`, not Autonomy.** The unknown-tool filter in `dispatch` short-circuits before the gate is consulted. Rationale: AutonomyPolicy would need `TOOL_NAMES` imported from Tool Dispatch (L2) to validate membership — an L1 → L2 dependency that violates layering. Delegating closure to the closed set keeps the layering clean and makes AS-6 a structural property of the tool surface, not a policy-code property.

87. **Unknown-level fallback to baseline-silent** (WP-E Group 1). An operator typo or a future level name leaking into config ahead of policy code falls back to baseline rather than raising or locking sessions out. The missing surfacing is a visible hint to the operator that the configured level isn't recognized. Alternative (raise at session start via `resolve_validated`) was considered but rejected — the gate fires per dispatch, not per session, and a hard raise would be disproportionate to "tightening didn't take effect."

88. **Per-session Autonomy Level overrides deferred.** Phase 1 `level_provider: Callable[[], str]` reads config on every `decide` call. Future per-session overrides land by widening the signature to `Callable[[SessionState], str]` without touching policy-code internals. No current scenario requires per-session tightening.

89. **Summarizer-quality echo-back risk carried to WP-H.** WP-D FF #81 flagged this; WP-E did not address it because summarizer quality is a calibration property (ADR-007) not an autonomy property. When WP-H lands alongside WP-E (or after), the Calibration Gate's check mechanism is the natural place to detect echo-back. WP-E's event-on-result shape accommodates future visibility events tied to calibration outcomes.

### From DECIDE mini-cycle (WP-F scenario gate, 2026-04-22)

93. **WP-F scenario gate resolved — Option C stands, retry pattern is the mechanism for scenario (d).** Five scenarios landed in `scenarios.md` §Client Tool Surface Commitment: four stress scenarios per Open Decision Point #1 (a: turn-boundary delegation; b: Session continuity; c: pre-invoke delegation when orchestrator can predict; d: retry pattern when ensemble mid-execution need is un-predicted) plus one negative-path scenario (composed ensemble without the structured signal silently degrades to a quality failure). All argument-audit rounds (5, 5b, 5c, 5d) closed clean. Susceptibility snapshot's Grounding Reframe on missing negative-path coverage was acted on. Option D is out of scope for this cycle — would require amending ADR-001/002 and adding suspend/resume to the DAG engine's synchronous phase loop. See `system-design.md` Amendment #4 and `roadmap.md` ODP #1 (closed) + ODP #8 (retry-enforcement mechanism layering carried forward).

94. **Retry-pattern commitment is conditional.** User's commitment language at the gate: "the retry pattern should be used if we think it will lead to more capable agentic serving." Review criterion is capability improvement — the retry pattern stands so long as it makes agentic serving more capable. If WP-F build or post-WP-F observation reveals that retry overhead dominates, silent-hallucination rates are unacceptable, or integration quality with agentic tools does not benefit, the commitment is revisitable. Named explicitly in the gate reflection note so future revisit is legitimate rather than a reversal. The measurement framework for "more capable agentic serving" is itself an open question — candidate measures include accuracy on mid-execution-need cases, user perception during live sessions, and integration smoothness with OpenCode / Roo Code / Cline.

95. **WP-F BUILD acceptance criteria.** All five scenarios in `scenarios.md` §Client Tool Surface Commitment must pass. Minimum-viable retry-enforcement stack per ODP #8: orchestrator system prompt (mechanism i) + composed-ensemble prompt convention (mechanism ii). Both are explicitly overridable without a new ADR — harder mechanisms (script-agent precondition guards, structural detection in Tool Dispatch) can be introduced as follow-on work if reliability gaps surface.

96. **Open questions carried into WP-F BUILD from DECIDE mini-cycle:**
    - **Responsibility Matrix gap for "emit retry signal."** Where does this responsibility live — Composition Validator (enforce at composition time), Orchestrator Tool Dispatch (detect at dispatch time), or convention-only? To be resolved before WP-G begins. Argument audit FI-2 + susceptibility snapshot feed-forward.
    - **`list_ensembles` output schema.** Scenario (c)'s pre-invoke delegation assumes a schema rich enough for the orchestrator to infer input-data format. ADR-003 does not specify the schema. A WP-F build-time decision.
    - **Summarizer transparency for structured JSON.** Scenario (d) depends on the Summarizer Harness preserving the `needs_client_tool` signal rather than prose-compressing it. Build-time configuration constraint, not guaranteed by ADR-004 alone.
    - **Calibration Gate coverage for silent quality failures.** The negative-path scenario documents the failure mode but does not test Calibration Gate's detection — that belongs to WP-H. No companion calibration scenario added at WP-F acceptance; the decision is to stay scope-narrow at this mini-cycle boundary.

### From BUILD (WP-E close / WP-F handoff)

90. **WP-F is a DECIDE mini-cycle, not BUILD — mode shift required.** Per Open Decision Point #1, WP-F does not start until four stress scenarios are written into `scenarios.md` that exercise the turn-boundary vs. mid-execution distinction. Writing these is adversarial scenario exploration ("what case would break Option C?"), not implementation flow. A BUILD-mode session carrying TDD/commit-loop attention is the wrong frame for this work — start a fresh session and state the mode explicitly. If any written scenario requires mid-execution callback, the Client Tool Surface Commitment is amended (C + hybrid, or Option D) before WP-F code begins. *(2026-04-22 update: this FF is superseded by FF #93–#96 — the mini-cycle was run in-session, all four original stress scenarios plus one negative-path scenario landed, Option C stood, and the retry pattern was accepted as the mechanism for scenario (d) with conditional commitment.)*

91. **Scenarios (c) and (d) require DAG engine semantics as context.** The two load-bearing stress scenarios probe agent-to-agent state flow within a multi-agent ensemble — (c) first agent needs a client-filesystem file before the second agent proceeds; (d) composed ensemble mid-execution needs a client-tool result the orchestrator didn't know to request. Writing these faithfully requires reading `src/llm_orc/core/execution/ensemble_execution.py` for how agents sequence and how their inputs are resolved. No WP touching the DAG engine's internals has landed in the agentic-serving cycle yet — the existing field guide's Ensemble Engine section describes the access surface, not the sequencing semantics. The fresh WP-F session should load this explicitly during orientation, not rely on system-design summaries.

92. **Opportunistic conformance tidying.** Four deferred v0.7.3 items from this corpus's migration notes were flagged as cleanup-bundleable at any natural boundary. Status after WP-F mini-cycle close (2026-04-22):
    - ~~`domain-model.md` §Concepts column header "Avoid (synonyms)" → "Related Terms" per v0.7.3 template.~~ **Fixed 2026-04-22** in the WP-F tidy commit.
    - ~~First-person plural in `essays/research-logs/001b-agentic-serving-architecture.md:3,47` — "we" in question titles.~~ **Fixed 2026-04-22** in the WP-F tidy commit (question headings recast to third-person "What lessons does X's architecture offer?").
    - **Still deferred:** Framing audit on `essays/001-agentic-serving-architecture.md` — `housekeeping/audits/argument-audit-001.md` has argument-audit only; v0.7.3 dispatch format adds framing audit. Pick up if framing tension surfaces.
    - **Still deferred:** `product-discovery.md` §Value Tensions phrased as declarative prose rather than open questions per v0.7.3 discover template. Editorial work, not mechanical tidy — defer to a session that has the attentional mode for it.

### From BUILD (WP-F Groups 2-3 / close, 2026-04-22)

102. **WP-F closed — TS-1 reached.** Eight tests in `TestClientToolSurfaceCommitment` cover all five scenarios of §Client Tool Surface Commitment plus the two refinement guards (mixed-batch reject-and-retry, TOOL_NAMES collision). Five commits on branch `agentic-serving` from Group 1 close through Group 3 close:
    - Group 1: `93e1229`, `61a6c40`, `b29a3b3`, `5d13e50` (see FF #97)
    - Group 2: `813bf60` (scenario c — pre-invoke delegation)
    - Group 3: `f3b9253` (retry pattern + orchestrator system prompt + summarizer YAML), plus this closeout docs commit.
    **State at WP-F close: 2270 tests passing, 91.52% coverage, lint clean** (mypy strict + ruff + format + bandit + vulture + complexipy). FC-4 / FC-5 / FC-8 / FC-9 / FC-11 static checks all pass; no fitness regression, no architectural drift. TS-1 (stateless orchestrator serving OpenCode) is now end-to-end reachable.

103. **Scenario (c) needed no schema change to `list_ensembles`.** `ResourceHandler.read_ensembles` already surfaces `description` on each entry (resource_handler.py:90). The WP-F build-time decision on schema enrichment turned out to be "not needed — the existing schema is already rich enough." `_StubEnsembleOperations` gained an optional `library_entries` kwarg so tests can verify orchestrator inference from description without reaching into the project's live config.

104. **Orchestrator system prompt — always-prepend strategy (Group 3).** Runtime accepts `system_prompt: str = ""` at construction; when non-empty, prepends as leading `role: system` on every LLM iteration. Sits ahead of any client-supplied system message so the orchestrator's discipline (ADR-003 internal tools, Option C turn-boundary, `needs_client_tool` retry convention) survives competing client guidance. Empty string is a no-op for tests and for deployments that want no orchestrator-side prompt. Default content lives in `DEFAULT_ORCHESTRATOR_SYSTEM_PROMPT` at `orchestrator_config.py`; operators override via `agentic_serving.orchestrator.system_prompt` in `config.yaml`.

105. **Summarizer transparency — YAML prompt, not code (Group 3).** Chose Path A (update `.llm-orc/ensembles/agentic-result-summarizer.yaml`'s `default_task` to instruct the agent to echo `needs_client_tool` JSON verbatim) over Path B (code-level signal preservation in the Harness). Keeps the Harness generic — it does not know about retry-pattern vocabulary. Production verification awaits `/rdd-play` stakeholder inhabitation; if prose LLMs don't honor the convention in practice, Path B (structural detection) can land as a follow-up without an ADR amendment.

106. **Retry-pattern acceptance verified structurally via stubbed summarizers (Group 3).** Scenarios (d) and (negative) use a stubbed `SummarizerInvoker` that returns either a JSON-shaped `needs_client_tool` signal or prose. The test proves the *plumbing* — does the orchestrator observe the signal when the summarizer preserves it? Does the orchestrator correctly not-retry when the signal is absent? It does not prove the production summarizer LLM honors the convention under real load — that's a quality-of-summarizer concern for WP-H Calibration Gate or post-TS-1 `/rdd-play`.

107. **Scenario (negative)'s failure mode is quality-class, not correctness-class.** When a composed ensemble skips the `needs_client_tool` convention, the orchestrator receives a normal prose summary, accepts it as final, and emits stop with (possibly hallucinated) content. Session's structural behavior is correct: no crash, Budget enforces, no spurious ClientToolCall, `turn_count` and `token_spend` advance normally. The answer is wrong; the Session dynamics are right. WP-H Calibration Gate is the designed backstop (ADR-007) — catching drift when a weak summarizer starts producing echo-back or a composed ensemble regresses on the convention.

### From BUILD (WP-F Group 1, 2026-04-22)

97. **WP-F Group 1 landed — Option C turn-boundary delegation for scenarios (a) and (b).** Three commits on branch `agentic-serving`:
    - `93e1229` refactor: relocate ChatMessage to session_start and extract tool-call encoder
    - `61a6c40` feat: route client-declared tools through turn-boundary delegation (WP-F Group 1)
    - `b29a3b3` feat: tighten mixed-batch discipline and reserve TOOL_NAMES (WP-F Group 1)
    Five acceptance tests at the serving-layer boundary cover scenario (a) non-streaming + streaming, scenario (b) session continuity, mixed-batch rejection, and name-collision rejection. Test suite: 2262 passing, 91.53% coverage, lint clean. FC-4 / FC-5 / FC-8 / FC-9 / FC-11 static checks all pass. Groups 2 (pre-invoke delegation) and 3 (retry pattern + system prompt + summarizer transparency) remain.

98. **Mixed-batch policy: reject-and-retry.** When the orchestrator LLM emits a single batch mixing internal (`TOOL_NAMES`) and client-declared tool calls, the Runtime rejects the entire batch with a per-call `mixed_batch` tool error and the LLM retries on the next iteration. Option C's one-kind-per-turn discipline is preserved without silent data loss. The orchestrator system prompt (Group 3) will teach the LLM to avoid mixing in the first place; this runtime rejection is the fallback for LLMs that ignore the guidance.

99. **Name collision: `TOOL_NAMES` reserved.** Client-declared tools whose `function.name` matches any of the five internal tool names are rejected at the Serving Layer with HTTP 400 and a `reserved_tool_name` error. Chosen over drop-with-warning because silent misrouting on a name collision is worse than an immediate actionable error for operators. The five names — `invoke_ensemble`, `compose_ensemble`, `list_ensembles`, `query_knowledge`, `record_outcome` — are llm-orc's contract per ADR-003.

100. **Domain-model open question — AS-6 and authorship.** The user surfaced (2026-04-22) that "eventually I'd want the orchestrator to create executable scripts as well as model profiles." AS-6 currently prohibits orchestrator authorship of scripts and model profiles on conservative grounds: (a) scripts are an RCE vector; (b) operator curation is deliberate; (c) Calibration Gate checks output plausibility, not code safety. Relaxing AS-6 would require a sandboxed execution model, an autonomy-level approval gate, and a calibration extension for code-quality signals. **Not load-bearing for WP-F or TS-1.** Revisit post-TS-1 as a standalone DECIDE mini-cycle.

101. **ChatMessage relocated to `session_start.py`.** ChatMessage is a *contract type* on the Serving Layer → Orchestrator Runtime edge, not Session Registry internals. Moving it into `session_start.py` (alongside `SessionContext`) keeps FC-4 intact when the Runtime imports ChatMessage directly — `session_start` is on the allow list, `session_registry` is forbidden. Circular import at the type level is guarded with `TYPE_CHECKING`.

### From ARCHITECT
30. Retrofit mode — llm-orc has existing FastAPI server, MCP handlers (ExecutionHandler, ValidationHandler, ensemble_crud_handler, promotion_handler, validation_handler, script_handler), ensemble engine, config manager, auth, and artifact system. Agentic serving is additive; Layer 3 (Ensemble Engine) stays unchanged per ADR-001/002
31. 12-module decomposition across 4 layers (L0 Core / L1 Domain Policy / L2 Runtime / L3 Entry) plus typed `resolve_session_start_context` function in Serving Layer. Originally 13 modules; Context Injection Stage demoted to function per ADR-066 gate-reflection amendment #1
32. FC-4 is load-bearing: Orchestrator Runtime imports *only* Budget Controller, Tool Dispatch, and Summarizer Harness. No Plexus, Autonomy, or Calibration leak into the reasoning space. This preserves the orchestrator LLM's mental model ("I emit tool calls and observe results") and structurally enforces ADR-003's closed tool-set property
33. Client tool surface commitment: Option C — turn-boundary delegation via `finish_reason: tool_calls`. Internal action space stays at 5 tools (ADR-003); client tools (bash, file_edit, etc. from OpenCode / Roo Code / Cline) flow through the response surface. Commitment is **scenario-gated** — WP-F does not start until stress scenarios (4 targets in roadmap Open Decision Point #1) test the C vs. D distinction. If any requires mid-execution callback, C is insufficient
34. Phase 2 Plexus context injection is reserved via typed function signature `resolve_session_start_context(session: SessionContext) -> list[PromptFragment]` in Serving Layer — not a module. ADR-009's "structurally reserved" clause satisfied by signature + call site; Phase 1 returns `[]`; Phase 2 is a function-body change. This matches the single-agent paradigm practice (Claude Code, OpenCode, claw-code)
35. Retrofit debt: WP-A extracts `_validate_cross_ensemble_cycles` + `_build_reference_graph` from `EnsembleLoader` private helpers to public `validate_ensemble_reference_graph` in `core/config/ensemble_config.py`. Both load-time and composition-time validation share the single routine (FC-6). Hard-blocks WP-G (composition)
36. Vision-named transition state TS-1 (WP-A + WP-B + WP-C + WP-D + WP-E + WP-F) is the stateless orchestrator that serves OpenCode — the intermediate target for "I can use OpenCode and run a version of this RDD pipeline with it"
37. Roadmap has 10 WPs; 9 hard deps, 4 implied; 3 transition states; 7 open decision points carried forward as build-time latitude
38. Consolidations probed at gate: Orchestrator Configuration → Serving Layer rejected (inverts layering — L1/L3 config is read by L1 modules like Budget Controller and Autonomy Policy via Session Registry). Calibration Gate → Orchestrator Runtime rejected (breaks FC-4; calibration is ensemble-state not runtime-state, and requires Plexus Adapter which Runtime must not import)

### From BUILD (WP-G, 2026-04-22)
101. **WP-G closed.** `compose_ensemble` is fully wired through the Composition Validator to the local-tier writer. Seven scenarios in `scenarios.md` §Ensemble Composition with Validation have explicit coverage across unit, boundary integration, and Serving Layer acceptance layers. Commits in order: `32d2dd3` (depth helper), `9972ed3` (Group 1 validator), `e5f8ea0` (Group 2 dispatch wiring), `804aeb7` (Group 3 integration + acceptance tests), this commit (Group 4 docs).

102. **FC-6 structurally satisfied** (this commit). `validate_ensemble_reference_graph` has exactly one public definition at `core/config/ensemble_config.py:309` and four call sites: `EnsembleLoader.load_from_file`, `EnsembleLoader.list_ensembles` (via `search_dirs=[directory]`), `ValidationHandler._collect_validation_errors`, and `CompositionValidator.validate` (new). Regression test in `test_tool_dispatch_composition.py::TestSharedValidatorSameBothPaths` verifies both load path and composition path surface identical cycle errors on the same input and confirms the composition validator imports the routine from its canonical module (not from a re-export).

103. **Composition-time validation is stricter than load-time by design.** Load path tolerates dangling ensemble references silently; composition path enforces AS-6 "compose from existing primitives only" by checking `model_profile_exists`, `script_exists`, `ensemble_exists` against the real registries. Moving Invariant 8 depth enforcement from the runtime (`EnsembleAgentRunner`) to composition-time follows the same principle: shrink time-to-error, keep load-time behavior unchanged. Captured in `composition_validator.py` docstring.

104. **`CompositionGate` and `LocalEnsembleWriter` are Protocols on Tool Dispatch's surface.** The concrete `CompositionValidator` pulls in a primitive registry (config manager + script resolver + ensemble directory discovery) which would bloat dispatch-level tests. The Protocol lets scripted doubles assert on dispatch behavior without constructing the production validator's dependency graph. Tests that exercise the full stack use the real classes (`test_tool_dispatch_composition.py`, `TestEnsembleCompositionWithValidationAcceptance`).

105. **Test-scope default validator rejects, does not fail loudly.** Many existing dispatch tests dispatch `compose_ensemble` incidentally (autonomy-gate coverage, visibility-event routing). The default helper in test files wires a `_RejectingValidator` so those tests continue passing; the writer is `_UnusedWriter`-equivalent and fails loudly if a non-compose test path reaches it. Tests that assert on composition behavior pass scripted validators explicitly. Single exception: `test_compose_rejects_malformed_arguments_without_calling_validator` uses `_UnusedValidator` to prove short-circuit on bad input.

106. **Overwrite semantics.** `ConfigManagerEnsembleWriter.write` rejects on file collision. If a future workflow wants composition-driven refinement (e.g., calibration-driven replacement), an explicit `overwrite=True` argument or a separate tool is needed. No scenario exercises this today.

107. **Hierarchical name collision edge case.** `EnsembleLoader.find_ensemble` supports hierarchical names (`examples/foo/foo`), but `ConfigManagerEnsembleWriter` always targets a flat `{name}.yaml` in the local tier. A composed ensemble whose name collides with a hierarchical library entry will not be caught by the writer's collision check. Low priority — composed ensembles use simple names.

108. **Raw-output composition is plumbed but unexercised.** `CompositionRequest.raw_output` flows through the writer's YAML serialization. No scenario drives the orchestrator to compose a raw-output ensemble today. If WP-H surfaces the need (e.g., calibration-scope composed ensembles that emit structured signals), add a scenario; otherwise the `False` default is structurally fine.

109. **TS-2 remaining work.** With WP-G closed, TS-2 (stateless baseline complete per ADR-002 Layer 1-3 and AS-8) needs only WP-H (Calibration Gate). WP-I (Plexus Adapter) begins the TS-3 path and can land in parallel since its hard deps are only WP-C.

## Context for Resumption

**Post-WP-G resumption pointer (2026-04-22, TS-2 one WP away).** WP-G is complete on branch `agentic-serving` across four commits:

- `32d2dd3` refactor: add compute_reference_graph_depth helper for composition-time depth check
- `9972ed3` feat: add Composition Validator module (WP-G Group 1)
- `e5f8ea0` feat: wire compose_ensemble through Composition Validator (WP-G Group 2)
- `804aeb7` test: add composition boundary integration + acceptance coverage (WP-G Group 3)
- (this commit) docs: close WP-G in roadmap, cycle-status, field guide, ORIENTATION

**State at WP-G close: 2297 tests passing, 91.51% coverage, lint clean.** Tier 1 stewardship clean. FC-6 structurally satisfied — one public definition of the shared cycle validator, four call sites, composition path and load path surface identical cycle errors on the same input.

**Next WP candidates:**

1. **WP-H: Calibration Gate.** Completes TS-2 (stateless baseline). Interpose Calibration Gate on `invoke_ensemble` for composed ensembles in their first N invocations. Four scenarios in `scenarios.md` §Calibration of Composed Ensembles. Depends on WP-G (now complete — composed-ensemble code path is live). Open decisions: default N, check mechanism (may itself be an ensemble).
2. **WP-I: Plexus Adapter (tool-first).** Begins TS-3. `query_knowledge` and `record_outcome` wiring with graceful no-op fallbacks. Five scenarios across §Plexus Integration and §Session Lifecycle. Depends only on WP-C.

**Post-WP-F resumption pointer (2026-04-22, TS-1 reached).** WP-F is complete on branch `agentic-serving` across six commits:

- `93e1229` refactor: relocate ChatMessage to session_start and extract tool-call encoder
- `61a6c40` feat: route client-declared tools through turn-boundary delegation (WP-F Group 1)
- `b29a3b3` feat: tighten mixed-batch discipline and reserve TOOL_NAMES (WP-F Group 1)
- `5d13e50` docs: record WP-F Group 1 feed-forward signals in cycle-status
- `813bf60` test: add scenario (c) pre-invoke delegation acceptance (WP-F Group 2)
- `f3b9253` feat: land retry pattern and orchestrator system prompt (WP-F Group 3)
- (this commit) docs: close WP-F in roadmap, cycle-status, field guide, ORIENTATION

**State at WP-F close: 2270 tests passing, 91.52% coverage, lint clean.** FC-4 / FC-5 / FC-8 / FC-9 / FC-11 static checks all pass. TS-1's vision — *"I can use OpenCode and run a version of this RDD pipeline with it"* — is end-to-end reachable: an operator deploys llm-orc with the Serving Layer, points OpenCode (or any OpenAI-compat agentic coding tool) at `/v1/chat/completions`, and the orchestrator routes tasks to library ensembles, summarizes results, enforces Budget, delegates client-side actions at turn boundaries, and retries composed ensembles whose mid-execution needs are un-predicted.

**Next WPs — two parallel candidates, both depend only on WP-C:**

1. **WP-G: Composition + Composition Validator.** Seven scenarios in `scenarios.md` §Ensemble Composition with Validation. Prerequisite (shared `validate_ensemble_reference_graph`) already in place from WP-A. Before starting: resolve the Responsibility Matrix gap for "emit retry signal" allocation (roadmap ODP #8 item iii — Composition Validator is the natural owner if enforcement lives in composition).

2. **WP-I: Plexus Adapter (tool-first).** Five scenarios across §Plexus Integration and §Session Lifecycle. Open decision: `record_outcome` payload schema (build-time).

Picking between them is a value-sequencing judgment; both lead toward TS-2 (stateless baseline complete after WP-G + WP-H) and TS-3 (four-layer Plexus stack after WP-I + WP-J).

**Forward-carrying concerns from WP-F:**
- **Silent quality failures when retry convention not honored** → WP-H calibration scope. Scenario (negative) documents the failure mode structurally; the Calibration Gate is the designed backstop.
- **AS-6 authorship open question** — the orchestrator currently cannot author scripts or model profiles (safety-conservative). User flagged this as eventually desirable. Revisit as a standalone DECIDE mini-cycle post-TS-1. See FF #100.
- **`list_ensembles` description richness.** The existing schema (name, source, relative_path, agent_count, description) works for scenario (c). Production deployments with many composed ensembles may need richer metadata (agent input expectations, tier, freshness). Defer until a real use case surfaces.

**Suggested fresh-session handoff prompt for WP-G or WP-I:**

> Continue the agentic-serving scoped cycle. WP-F closed 2026-04-22; TS-1 reached. Pick WP-G (Composition + Validator) or WP-I (Plexus Adapter, tool-first) — both depend only on WP-C. Read `docs/agentic-serving/cycle-status.md` §Context for Resumption and the chosen WP's entry in `docs/agentic-serving/roadmap.md`. For WP-G, resolve the Responsibility Matrix gap for "emit retry signal" (roadmap ODP #8 item iii) before implementing. For WP-I, the `record_outcome` payload schema is a build-time decision.

---

**Post-WP-F-mini-cycle resumption pointer (2026-04-22, superseded by WP-F close).** The WP-F scenario gate was resolved via in-session DECIDE mini-cycle; all five acceptance scenarios now pass. See the post-WP-F pointer above.

---

**Post-WP-E resumption pointer (2026-04-22).** WP-E is complete on branch `agentic-serving` across seven commits:

- `f07f64b` feat: add AutonomyPolicy module and VisibilityEvent chunk type (WP-E Group 1)
- `b2a1c88` refactor: carry VisibilityEvent tuple on ToolCallSuccess and ToolCallError
- `6c168da` feat: interpose Autonomy Policy gate before every Tool Dispatch (WP-E Group 2)
- `536f952` feat: render VisibilityEvent as delta.content narration (WP-E Group 3)
- `8ca482a` test: add autonomy and promotion acceptance scenarios (WP-E Group 5)
- `29fb4c0` test: add FC-11 static gate check and boundary integration (WP-E Group 6)
- (this commit) docs: close WP-E in field guide, ORIENTATION, cycle-status, roadmap

**State at WP-E close: 2257 tests passing, 91.48% coverage, lint clean.** Tier 1 stewardship clean. FC-11 structurally enforced via strict AST dominance + adversarial self-test. OQ #2 resolved to `delta.content` narration. Two Phase-1 levels ship; `Deny` is first-class for WP-H expansion.

**Next WPs — two parallel candidates, both depend only on WP-C:**

1. **WP-G: Composition + Composition Validator.** Seven scenarios in `scenarios.md` §Ensemble Composition with Validation. Prerequisite (shared `validate_ensemble_reference_graph`) already in place from WP-A.
2. **WP-I: Plexus Adapter (tool-first).** Five scenarios across §Plexus Integration and §Session Lifecycle. Open decision: `record_outcome` payload schema.

**TS-1 gap:** WP-F (client-tool turn-boundary delegation). Scenario-gated — Open Decision Point #1 in `roadmap.md` lists four stress scenarios that must be written before WP-F starts. If any requires mid-execution callback, the Option C commitment is amended.

**WP-F handoff guidance (fresh session recommended):**

WP-F is a DECIDE mini-cycle, not BUILD — a different attentional mode than the one that drove WP-A through WP-E. Start a fresh session and state the mode explicitly in the opening prompt. See FF #90–#92 for the mode shift rationale, the DAG engine reading directive (scenarios c and d probe agent-to-agent state flow in `src/llm_orc/core/execution/ensemble_execution.py`), and opportunistic conformance tidying that can be bundled at any natural boundary.

Suggested fresh-session handoff prompt:

> Continue the agentic-serving scoped cycle. WP-E closed (commit `368384f`). Resume on WP-F scenario writing per Open Decision Point #1 in `docs/agentic-serving/roadmap.md`. This is a DECIDE mini-cycle, not BUILD — write four stress scenarios into `scenarios.md` that exercise the Client Tool Surface Commitment's turn-boundary-vs-mid-execution distinction. If any requires mid-execution callback, Option C is insufficient and the Commitment is amended. Read `src/llm_orc/core/execution/ensemble_execution.py` for DAG semantics; agent-to-agent state flow determines scenarios (c) and (d).

**Forward-carrying concerns from WP-E:**
- Summarizer-quality echo-back risk → WP-H calibration scope (carried from WP-D FF #81, not addressed in WP-E).
- Per-session Autonomy Level overrides deferred (FF #88).
- Operator-tooling visibility surface (SSE comments or dedicated endpoint) deferable as a second audience-specific surface.

---

**Post-WP-D resumption pointer (2026-04-21).** WP-D is complete on branch `agentic-serving` across nine commits:

*WP-D Groups 0-4 (structural change):*
- `a15aa30` — docs: Design Amendment #3 to `system-design.md` (Runtime → RSH edge moved to Tool Dispatch → RSH; FC-4 amended; Dependency Graph + Responsibility + Test Architecture rows updated).
- `326a36f` — feat: `src/llm_orc/agentic/result_summarizer_harness.py` with `SummarizerInvoker` Protocol, typed `SummarizationSuccess | RawOutputPassthrough | SummarizationFailure` result variants.
- `188f65f` — feat: `EnsembleConfig.raw_output: bool = False` field + YAML loader wiring. ADR-004 escape hatch.
- `9a0fea2` — feat: Tool Dispatch interposes the Harness on `invoke_ensemble`'s return. New `ToolErrorKind` "summarization_failed". `OrchestratorConfig.summarizer_ensemble` field. Serving Layer builds the Harness.
- `3e7c897` — feat: default `agentic-result-summarizer` ensemble YAML + `summarizer` model profile. Harness extraction gains a single-agent `results[agent][response]` fallback.

*WP-D Groups 5-6 (verification and closeout):*
- `4261238` — refactor: tighten FC-4 forbidden list for Amendment #3 (RSH explicitly forbidden from Runtime imports).
- `903833e` — test: strict FC-8 static no-bypass check for `invoke_ensemble` (AST dominance + adversarial self-test).
- `03885f8` — test: raw-output escape-hatch acceptance scenario at Serving Layer.
- `2f0f660` — test: Tool Dispatch → Harness → Ensemble Engine summarize-path boundary integration.
- (this commit) — docs: regenerate field-guide + ORIENTATION + roadmap + cycle-status for WP-D close.

**State at WP-D close: 2221 tests passing, 91.44% coverage, lint clean.** Tier 1 stewardship clean. AS-7 structurally enforced; FC-8 holds via strict AST dominance; the Runtime has no path to raw ensemble output.

**Next WP — three parallel candidates, all depend only on WP-C:**

1. **WP-E: Autonomy Policy.** Interpose the Autonomy gate before every Tool Dispatch call. Four scenarios in `scenarios.md` §Autonomy and Promotion. Blocks on visibility-form decision (OQ #2) — structured SSE events is the default posture; operator-dashboard surface is possible. Carries forward the WP-D summarizer-quality echo-back risk as a calibration-scope concern if WP-H expands alongside.
2. **WP-G: Composition — `compose_ensemble` + Composition Validator.** Wire the composition path through the shared `validate_ensemble_reference_graph` routine WP-A extracted. Seven scenarios in `scenarios.md` §Ensemble Composition with Validation (six composition + one regression).
3. **WP-I: Plexus Adapter (tool-first).** Wire `query_knowledge` and `record_outcome` with graceful no-op fallbacks when Plexus is absent. Five scenarios across §Plexus Integration and §Session Lifecycle. Open decision: `record_outcome` payload schema (build-time choice).

**TS-1 gap:** WP-F (client-tool turn-boundary delegation). Scenario-gated — Open Decision Point #1 in `roadmap.md` lists four stress scenarios that must be written before WP-F starts. If any requires mid-execution callback, the Option C commitment is amended.

**Forward-carrying concerns from WP-D:**
- Summarizer-quality echo-back risk → WP-E / WP-H calibration scope (FF #81).
- Test Architecture table row `test_runtime_never_sees_unsummarized_result` is covered by three tests, not one — worth tightening the row in a future system-design edit (FF #82).

A fresh session picking up should read this cycle-status file (§Phase Status + this resumption pointer), `ORIENTATION.md`, and the chosen WP's entry in `roadmap.md`. The field guide is current at WP-D close for module-to-code navigation.

This is a scoped RDD cycle for the agentic-serving feature of llm-orc, a declarative DAG-based LLM orchestration engine. The research phase investigated whether llm-orc can serve as the backend for agentic coding tools via OpenAI-compatible endpoints. Six questions were explored across two research cycles: API surface requirements, DAG-to-ReAct mapping, self-building ensembles, Plexus as memory layer, OpenHands architecture, and claw-code architecture.

The essay concludes with a four-layer architecture: API surface, orchestrator agent, ensemble engine (existing), and knowledge graph (Plexus). Product discovery surfaced seven value tensions and six assumption inversions. Key product insight: Plexus operates as a push-model lib where the client drives ingestion of file content (not LLM summaries), and enrichments extract signal -- quality emerges from the enrichment pipeline, not upstream curation. The tool user and ensemble author are often the same person, making visibility a product feature (tinkering) not just an operational concern.

Citation and argument audits have been run. All P1 issues were remediated in the essay. Some P2 issues remain unresolved. Domain model establishes 8 invariants (AS-1 through AS-8) with AS-8 (Plexus is optional) as the key architectural constraint: design for stateless operation, benefit from Plexus when available. The next phase is DECIDE -- ADRs and behavior scenarios using the domain vocabulary.

## Conformance Notes

**Corpus migrated from RDD v0.4.2 to v0.7.3 on 2026-04-17.** See `.migration-version` and `.migration-rollback.json` for migration details.

**Retroactive Tier 1 audits are absent.** RESEARCH, DISCOVER, and MODEL phases were produced under v0.4.2 which did not have the following v0.7.x unconditional mechanisms: research-methods-reviewer dispatch (ADR-060), per-phase susceptibility-snapshot-evaluator dispatches, and gate reflection notes (ADR-066). These are not being fabricated retrospectively. Tier 1 mechanisms will fire properly starting at DECIDE.

**Deferred conformance items** (to be picked up opportunistically):
- **Framing audit on essay.** v0.7.3 argument-audit dispatches include a framing-audit section; `housekeeping/audits/argument-audit-001.md` has argument-audit only. Pick up in continued work if framing tension surfaces.
- ~~**Concepts table column header.**~~ **Fixed 2026-04-22** — `domain-model.md` §Concepts column 4 renamed from "Avoid (synonyms)" to "Related Terms" per v0.7.3 template.
- ~~**First-person plural in research log.**~~ **Fixed 2026-04-22** — question titles in `essays/research-logs/001b-agentic-serving-architecture.md` lines 3 and 47 recast to third-person.
- **Value tensions phrasing.** `product-discovery.md` §Value Tensions stated as declarative prose rather than open questions per v0.7.3 discover template. Editorial conversion; defer to a session with the attentional mode for it.
