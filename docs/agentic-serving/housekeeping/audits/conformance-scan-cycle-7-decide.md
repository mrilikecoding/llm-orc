# Conformance Scan Report

**Scanned against:** ADR-026, ADR-027, ADR-028, ADR-029, ADR-030, ADR-031, ADR-032 (Cycle 7 DECIDE set); ADR-021, ADR-022 (partial-update scope); ADR-007, ADR-014, ADR-015, ADR-018, ADR-023, ADR-024, ADR-025 (prior authoritative ADRs)
**Codebase:** `/Users/nathangreen/Development/eddi-lab/llm-orc/`
**Date:** 2026-05-22

---

## Summary

- **ADRs checked:** 13 (7 new Cycle 7 + 6 prior authoritative)
- **Conformant:** 9 findings
- **Violations found:** 13

**Category breakdown:**

| Category | Count |
|---|---|
| Conformant | 9 |
| Debt — `refactor:` commit (small, scoped cleanup) | 2 |
| Debt — new BUILD work (the pipeline itself) | 8 |
| Debt — deferred to ARCHITECT | 3 |

---

## Conformance Debt Table

| # | ADR | Violation | Type | Location | Current State | Resolution |
|---|-----|-----------|------|----------|---------------|------------|
| 1 | ADR-027 §Decision | Chat-completions surface routes every request through `OrchestratorRuntime` ReAct loop — the architecture ADR-027 supersedes on this surface | exists (wrong structure) | `v1_chat_completions.py:411-451`, `_build_runtime()` at line 564-594 | `chat_completions()` calls `_build_runtime()` which constructs `OrchestratorRuntime`; all requests flow through `runtime.run(context)` | new BUILD work — introduce the three-stage pipeline handler (`plan → dispatch → synthesize`) as the primary path for POST `/v1/chat/completions`; keep `OrchestratorRuntime` for future non-chat-completions surfaces |
| 2 | ADR-027 §Decision (preservation clause) | `OrchestratorRuntime` is **only** wired from `v1_chat_completions.py`; no other production surface imports or constructs it | wrong structure | `orchestrator_runtime.py:204`, `v1_chat_completions.py:62-64` | The `llm-orc invoke` CLI (`cli_commands.py:249`) calls `OrchestraService` directly via its executor — it does not use `OrchestratorRuntime`. ADR-027 names the CLI as the surface where `OrchestratorRuntime` should be *preserved*, but there is currently nothing preserving: it is only used on the surface it is supposed to be *replaced*. | deferred to ARCHITECT — before ADR-027 BUILD ships, the CLI's "preserve the ReAct loop for `llm-orc invoke`" path must be designed. The ReAct loop is not currently reachable from the CLI; the BUILD should introduce a CLI-facing `OrchestratorRuntime` wiring rather than stranding the class after the chat-completions pipeline moves |
| 3 | ADR-028 §Decision, §Ensemble structure | Routing-planner production ensemble (`agentic-routing-planner.yaml`) does not exist under `.llm-orc/ensembles/agentic-serving/` | missing | `.llm-orc/ensembles/agentic-serving/` directory | Only the spike artifact `spike-cycle7-zeta-routing-planner.yaml` exists at `.llm-orc/ensembles/` root; no production-promoted ensemble exists | new BUILD work — promote Spike ζ scratch artifact to `agentic-routing-planner.yaml` under `agentic-serving/`, integrating with loaded-ensemble registry and adding production `topaz_skill: tool_use` |
| 4 | ADR-028 §Output contract | Spike ζ routing-planner output schema is missing the `input` field | wrong structure | `.llm-orc/ensembles/spike-cycle7-zeta-routing-planner.yaml:12-16` | The spike's output contract specifies `{"action", "ensemble", "rationale"}` only; ADR-028 §Output contract adds `"input": "<input string for the dispatched ensemble; required when action=dispatch>"` as a fourth required field | `refactor:` commit — extend the routing-planner spike's system prompt to include the `input` field in its output schema before the production promotion; or defer to the BUILD-phase promotion where the production ensemble is authored against the full ADR-028 contract |
| 5 | ADR-029 §Decision | Response-synthesizer production ensemble (`agentic-response-synthesizer.yaml`) does not exist under `.llm-orc/ensembles/agentic-serving/` | missing | `.llm-orc/ensembles/agentic-serving/` directory | Only the spike artifact `spike-cycle7-epsilon-response-synthesizer.yaml` exists at `.llm-orc/ensembles/` root | new BUILD work — promote Spike ε scratch artifact to `agentic-response-synthesizer.yaml` under `agentic-serving/`; the spike's five-rule system prompt is the BUILD starting point per ADR-029 §Decision |
| 6 | ADR-029 §Strict-fidelity rule set (Rule 6) | Spike ε synthesizer system prompt is missing Rule 6 (framework-convention enumeration) | missing | `.llm-orc/ensembles/spike-cycle7-epsilon-response-synthesizer.yaml:37-118` | The spike's system prompt encodes Rules 1-5 (confirmed by file read); Rule 6 was codified from Spike μ.1 after the spike was authored and is absent | `refactor:` commit — add Rule 6 to the spike's system prompt as a pre-promotion fix, or include in BUILD-phase promotion; scored as refactor because the spike YAML requires only a system-prompt addition, no structural change |
| 7 | ADR-030 §Bridge advisory specification | `_ChatCompletionsRequest` does not declare a `tool_choice` field; Pydantic silently strips it | exists (wrong structure — the wrong structure is the status quo the ADR prescribes a bridge mechanism to replace) | `v1_chat_completions.py:401-408` | Pydantic model declares `model`, `messages`, `stream`, `tools`, `user` with no `tool_choice`; `extra="ignore"` strips it silently — the Spike κ D0 finding confirmed in ADR-030 §Context | new BUILD work — extend `_ChatCompletionsRequest` to accept and observe `tool_choice` without routing it (the bridge: detect presence, emit three-layer advisory signal: `X-LLM-Orc-Tool-Choice-Handling: deferred` header, `metadata.tool_choice_handling: "deferred"` body field, Rule 5-adjacent content on `action: "direct"` responses only per ADR-030 §Bridge advisory specification) |
| 8 | ADR-031 §Streaming as load-bearing surface | Streaming is currently wired through `OrchestratorRuntime.run()` SSE stream; when ADR-027 pipeline replaces the runtime, the streaming path must be re-wired to the response-synthesizer's output | wrong structure (pre-migration; streaming itself exists and is correct) | `v1_chat_completions.py:434-450`, `_stream_completion()` at line 646-660 | The handler correctly returns a `StreamingResponse` when `request.stream=True`; streaming drives `runtime.run(context)` through the SSE formatter. The mechanism is correct for the current architecture but structurally coupled to the runtime loop | deferred to ARCHITECT — the streaming re-wire is a consequence of the ADR-027 pipeline BUILD; ARCHITECT phase should specify how the response-synthesizer's token stream surfaces to the SSE formatter, reusing the existing `OpenAiSseFormatter` |
| 9 | ADR-032 §Sub-promise 1 (honest response labeling) | No response headers, body metadata, or Rule 5 content framing for served-by path exists on any code path | missing | `v1_chat_completions.py:762-801`, `_build_completion_body()` | The response body is a plain OpenAI-shaped dict with no `served_by`, `X-LLM-Orc-Served-By`, `metadata.served_by`, or any framing marker; response headers carry only FastAPI defaults | new BUILD work — introduce the three-layer labeling mechanism: response header (e.g., `X-LLM-Orc-Served-By`), body metadata field, and Rule 5 synthesizer framing on direct-completion responses |
| 10 | ADR-032 §Capability-list discovery | `/v1/models` endpoint advertises Model Profile IDs (orchestrator LLM profiles), not capability ensemble identifiers | wrong structure | `v1_models.py:41-57` | `GET /v1/models` calls `resolver.list_allowed_model_profile_ids()` which returns orchestrator model profile names (e.g., `"default"`, `"agentic-tier-cheap-general"`); capability ensemble names (`"web-searcher"`, `"code-generator"`, etc.) are not surfaced | new BUILD work — per ADR-032 §Capability-list discovery, extend `/v1/models` or add a sibling endpoint (`/v1/ensembles`) to advertise capability ensembles; the existing model-profile listing is the wrong data; the BUILD phase picks which of the three candidate surfaces to ship |
| 11 | ADR-027 §Dispatch stage, ADR-030 §Disposition (i) implementation scope | `OrchestratorToolDispatch.dispatch()` is callable independently of the ReAct loop (Protocol-defined; `InternalToolCall` argument) but its dispatch routing entry point requires an `InternalToolCall` typed object — not a plain `(ensemble_name, input)` tuple | wrong structure | `orchestrator_tool_dispatch.py:612-664` | `dispatch(call: InternalToolCall, *, session_id: str)` routes through unknown-tool filter, autonomy gate, then to named methods. ADR-027 §Dispatch stage says "the framework invokes the named ensemble via the existing `OrchestratorToolDispatch` machinery" — the machinery is there, but the pipeline's Plan stage produces `{"action": "dispatch", "ensemble": "<name>", "input": "..."}` not an `InternalToolCall`. The pipeline needs a translation layer to call dispatch from a plan | deferred to ARCHITECT — the ARCHITECT phase should specify the adapter: plan-output (`action`, `ensemble`, `input`) → `InternalToolCall(id=..., name="invoke_ensemble", arguments={"name": ensemble, "input": input})`. This is a clean translation; the dispatch machinery is structurally usable |
| 12 | ADR-027 §Relationship to existing infrastructure (dispatch event substrate) | `DispatchEventSubstrate` has no event types for pipeline-stage events (`plan-emitted`, `dispatch-fired`, `synthesizer-completed`) or degradation signaling events (`direct_completion_fallback`, `direct_completion_rate`) named in ADR-027 and ADR-032 | missing | `dispatch_event_substrate.py:30-38` (`__all__`) | `__all__` exports `DispatchEvent`, `DispatchEventSubstrate`, `DispatchPhase`, `DispatchTiming`, `ExitStatus`, `EventSink`, `new_dispatch_id`. The substrate's Protocol-based `DispatchEvent` is duck-typed (any object with `dispatch_id: str | None` attribute) so new event types can be added without restructuring the substrate itself | new BUILD work — define and register new event dataclasses for the pipeline-stage and degradation signals; the substrate's `emit()` fan-out mechanism and `events_for()` post-hoc query are already extensible (Protocol-based sink; duck-typed event) so no structural refactor is needed, only new type definitions |
| 13 | ADR-023 §Observability (ADR-032 extension) | Operator-terminal sink and orchestrator-context sink consume `TierSelection`, `CalibrationVerdictEvent`, `AuditDiagnostic`, `CalibrationSignal` events; no consumer for `direct_completion_fallback` or `direct_completion_rate` events exists | missing | `operator_terminal_event_sink.py:33-37`, `orchestrator_context_event_sink.py:43-49` | The existing sinks `isinstance`-discriminate on the four known event types; new pipeline/degradation events would pass through as unhandled (no-op consume) until a consumer is added | new BUILD work — extend the operator-terminal sink to consume and log/emit the two new ADR-032 degradation event types; this is additive to the existing sink, not a restructure |

---

## Conformant Findings

The following scan areas found current implementation conformant with the ADR prescriptions:

| # | ADR | Area | Location | Assessment |
|---|-----|------|----------|------------|
| C1 | ADR-030 §Context | `tool_choice` zero-handling (Spike κ D0) — the ADR explicitly names the current state as the footgun to replace, not an error to fix separately | `v1_chat_completions.py:401-408` | Conformant: current state matches ADR-030's documented starting point |
| C2 | ADR-022 §Amendment (ADR-027 scope update) | System-prompt amendment in `DEFAULT_ORCHESTRATOR_SYSTEM_PROMPT` is present and operative; scope narrowing for chat-completions is correctly framed as pending ADR-027 BUILD | `orchestrator_config.py:77-149` | Conformant: amendment is present; the ADR-027 partial-update header correctly records that the amendment is moot for chat-completions once BUILD lands |
| C3 | ADR-024 §Contract | `DispatchEnvelope` typed contract (`primary`, `structured`, `diagnostics`, `errors`, `artifacts`) is implemented per the ADR specification | `dispatch_envelope.py:74-160` | Conformant: all required fields present with correct types and semantics |
| C4 | ADR-025 §Scope | Capability ensembles in `agentic-serving/` declare `output_substrate: artifact` and `topaz_skill` metadata; `artifacts[0]` reference is named in comments | All six capability ensembles in `.llm-orc/ensembles/agentic-serving/` | Conformant: `web-searcher`, `code-generator`, `claim-extractor`, `text-summarizer`, `argument-mapper`, `prose-improver` all carry `output_substrate: artifact` and `topaz_skill` |
| C5 | ADR-007, ADR-014 | `CalibrationGate` operates within dispatched ensembles via `OrchestratorToolDispatch`; not at orchestrator-runtime level | `orchestrator_tool_dispatch.py:246-272`, `calibration_gate.py:1-100` | Conformant: gate is per-dispatch (L1 domain policy), not per-request at the serving layer |
| C6 | ADR-015, ADR-018 | `TierRouter` and `TierEscalationAuditor` operate within dispatched ensembles; per-skill taxonomy (8 Topaz skills) is implemented | `tier_router.py:64-93`, `v1_chat_completions.py:228-272` | Conformant: tier escalation is ensemble-scoped; the router's `select_tier()` is a stateless pure function |
| C7 | ADR-023 §Event substrate | `DispatchEventSubstrate` fan-out is extensible (Protocol-based `EventSink`; duck-typed `DispatchEvent`) without requiring architectural restructuring to add new event types | `dispatch_event_substrate.py:95-209` | Conformant: the substrate is structurally open for extension; new event types (ADR-027, ADR-032) can be added as new dataclasses |
| C8 | ADR-028 §Spike artifacts | `spike-cycle7-epsilon-response-synthesizer.yaml` and `spike-cycle7-zeta-routing-planner.yaml` both exist at `.llm-orc/ensembles/` root and are retained per the spike-artifact-retention practitioner directive | `.llm-orc/ensembles/spike-cycle7-epsilon-response-synthesizer.yaml`, `.llm-orc/ensembles/spike-cycle7-zeta-routing-planner.yaml` | Conformant: both spike artifacts retained; Spike ε synthesizer carries Rules 1-5; Spike ζ planner carries the 20-prompt battery structure |
| C9 | ADR-031 §Streaming | Streaming (SSE) is implemented and correctly wired in the current handler; `stream: true` returns a `StreamingResponse` via `OpenAiSseFormatter` | `v1_chat_completions.py:433-450`, `_stream_completion()` at line 646-660 | Conformant: streaming infrastructure exists and is operational; the ADR-031 commitment to ship streaming as a load-bearing default is met by the current infrastructure (re-wire to synthesizer output is a BUILD-phase consequence, tracked as Finding 8) |

---

## Notes

**Priority ordering for Scenario Writing (Tranche 5)**

The findings above map to two distinct scenario tracks:

**Track A — Refactor commits (should precede BUILD):**
- Finding 4: extend routing-planner spike output schema to include `input` field. This is a one-field system-prompt edit on the Spike ζ YAML; it unblocks production promotion without adding functionality.
- Finding 6: add Rule 6 to synthesizer spike system prompt. Same pattern — system-prompt addition on Spike ε YAML before promotion.

**Track B — BUILD work (the pipeline itself):**
The eight BUILD findings (1, 3, 5, 7, 9, 10, 12, 13) form the core ADR-027 BUILD scope. Their natural sequencing:
1. Introduce the three-stage pipeline handler in `v1_chat_completions.py` (Finding 1) — this is the structural entry point for all other BUILD work.
2. Promote routing-planner and response-synthesizer spike artifacts to production system ensembles (Findings 3, 5).
3. Add the `tool_choice` bridge advisory to `_ChatCompletionsRequest` (Finding 7) — independently testable.
4. Add honest response labeling (Finding 9) — deliverable alongside the pipeline.
5. Extend `/v1/models` or add `/v1/ensembles` for capability-list discovery (Finding 10) — the ADR-032 first-order requirement.
6. Define pipeline-stage and degradation event types + extend sinks (Findings 12, 13) — observability layer, can follow core pipeline.

**Track C — ARCHITECT-phase design (before BUILD starts):**
- Finding 2: specify how `OrchestratorRuntime` is preserved for the CLI surface (`llm-orc invoke`). Currently the CLI bypasses `OrchestratorRuntime` entirely; ADR-027's "preserve for non-chat-completions surfaces" framing requires a concrete wiring plan before the class is stranded.
- Finding 8: specify how streaming re-wires to the response-synthesizer's token stream.
- Finding 11: specify the Plan → `InternalToolCall` adapter that bridges the routing-planner's JSON output to `OrchestratorToolDispatch.dispatch()`.

**Structural note on Finding 2 (CLI and OrchestratorRuntime)**

This finding is worth surfacing prominently. ADR-027 says "The `OrchestratorRuntime` ReAct loop is preserved for non-chat-completions surfaces — the `llm-orc invoke` CLI surface." But `llm-orc invoke` currently does not use `OrchestratorRuntime` at all — it calls `OrchestraService` directly via its executor. The "preserve" framing in ADR-027 presupposes that the CLI is an existing user of `OrchestratorRuntime` that should not be disrupted by the ADR-027 migration. That assumption is false: the CLI is architecturally separate from the ReAct loop. The consequence is that after ADR-027 BUILD ships, `OrchestratorRuntime` would only exist as a class definition with no production caller. The ARCHITECT phase should determine whether (a) the CLI should grow a ReAct-backed path using `OrchestratorRuntime`, or (b) `OrchestratorRuntime` should be understood as a future non-chat-completions surface component (not currently wired to the CLI, but preserved for that purpose). Option (b) is acceptable but should be made explicit in an ADR annotation.

**The `/v1/models` gap (Finding 10)**

The current `/v1/models` endpoint surfaces orchestrator Model Profile IDs (e.g., `"default"`, `"agentic-tier-cheap-general"`), not capability ensemble identifiers. From a Population A client's perspective this makes `/v1/models` a model-picker list, not a capability-discovery surface. ADR-032 names capability-list discovery as a first-order requirement; the existing endpoint satisfies the protocol contract but not the discovery function. The BUILD phase must either extend `/v1/models` with ensemble entries (with a `capability_marker` distinguishing them from model profiles) or add a sibling endpoint. Both paths coexist with the existing model-profile listing without breaking it.

**Event substrate extensibility confirmation**

The `DispatchEventSubstrate` is Protocol-based: `EventSink` consumers receive any object via `consume(event: object)` and discriminate by `isinstance` at consumption time. The substrate's `emit()` logs events with a string `dispatch_id` attribute. This means new event dataclasses for ADR-027 pipeline stages and ADR-032 degradation signals can be defined anywhere and emitted through the existing substrate without touching the substrate module itself. Findings 12 and 13 are additive definitions, not structural refactors.

**Capability ensemble envelope compliance**

All six capability ensembles in `.llm-orc/ensembles/agentic-serving/` (`web-searcher`, `code-generator`, `claim-extractor`, `text-summarizer`, `argument-mapper`, `prose-improver`) declare `output_substrate: artifact` and `topaz_skill`. The `agentic-calibration-checker` and `agentic-result-summarizer` system ensembles do not declare `topaz_skill` or `output_substrate`, which is correct (they are system ensembles, not capability ensembles, and the synthesizer reads capability ensemble output; system ensembles are not dispatched through the pipeline). ADR-029's requirement that "the synthesizer reads envelope content (`primary` + `artifacts[0]` summary fields)" is satisfied structurally by the existing WP-D/WP-E envelope contract.
