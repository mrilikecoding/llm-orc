# Conformance Scan Report — Cycle 7 Loop-back #9

**Scanned against:** `docs/agentic-serving/decisions/adr-043-collapse-dual-serving-surfaces-to-one-loop.md`
**Context ADRs (cross-reference only):** ADR-027, ADR-033
**Codebase:** `src/llm_orc/`, `tests/`
**Date:** 2026-06-18

## Summary

- **ADRs checked:** 1 (ADR-043)
- **Conforming items:** 4 (the loop-driver Terminal is already the correct universal caller; `context.tools` is already available in `decide()`; `_delegation_tools()` + `_seat_filler_messages()` are already conditioned on `self._capabilities`; FC-58 invariant structure is preserved)
- **Violations (permanent contradiction of an accepted ADR):** 3
- **BUILD-WORK items (expected-temporary: code that must be removed or added to reach the prescribed state):** 14

The Dispatch Pipeline and its supporting code exist entirely as production callers and hard imports — not scaffolding or stubs. The two-surface discriminator is live on every request. All fourteen BUILD-WORK items are subtractive (remove or replace) except two: the new F-ι.1 gate (a small additive change inside `loop_driver.py`) and two new tests. No new module or dependency edge is required; the practitioner's "BUILD Design Amendment — subtractive change" routing is confirmed.

---

## Conformance Debt Table

| ID | ADR | Violation / Gap | Type | Location | Classification | Resolution |
|----|-----|-----------------|------|----------|----------------|------------|
| V-01 | ADR-043 §Decision 1 | `_is_tool_driven` still exists and is still called — the discriminator's split is the live routing path, not the universal Terminal path. All no-tools requests currently route to `get_dispatch_pipeline()`, which ADR-043 retires. | exists | `v1_chat_completions.py:568–587` (`_is_tool_driven`), `v1_chat_completions.py:622–625` (dispatch-vs-terminal branch) | VIOLATION | Remove `_is_tool_driven`, the conditional branch, and the `get_dispatch_pipeline()` call from `chat_completions()`. Every request routes to `get_client_tool_action_terminal()` unconditionally. |
| V-02 | ADR-043 §Decision 2 | `DispatchPipeline` and `ensemble_backed_roles` are hard imports in `v1_chat_completions.py` with live call sites (`get_dispatch_pipeline` builds and returns a `DispatchPipeline` on every non-tool request). These modules are designated for removal by ADR-043 §Decision 2. | exists | `v1_chat_completions.py:55–59` (imports), `v1_chat_completions.py:375–400` (`get_dispatch_pipeline` function) | VIOLATION | Remove the `DispatchPipeline` + `EnsembleResponseSynthesizer` + `EnsembleRoutingPlanner` imports, the `get_dispatch_pipeline` factory, and the `ROUTING_PLANNER_ENSEMBLE` / `RESPONSE_SYNTHESIZER_ENSEMBLE` constants once no longer referenced. Delete `src/llm_orc/agentic/dispatch_pipeline.py` (335 lines) and `src/llm_orc/agentic/ensemble_backed_roles.py` (171 lines). |
| V-03 | ADR-043 §Decision 3 (F-ι.1) | No F-ι.1 gate exists. `_delegation_tools()` returns `[invoke_ensemble_tool_def(...)]` whenever `self._capabilities` is non-empty, regardless of whether the client request carries any write-capable tool. `_seat_filler_messages()` similarly composes delegation guidance whenever `self._capabilities` is present. ADR-043 requires the gate to condition both the tool offering and the guidance on client-tool presence. | missing | `loop_driver.py:821–831` (`_delegation_tools`), `loop_driver.py:860–869` (`_seat_filler_messages`) | VIOLATION | Add a `_has_write_capable_tool(tools: list[dict[str,Any]]) -> bool` predicate (or inline check) in `_delegation_tools` and `_seat_filler_messages`. Both must inspect `context.tools` — already available as a parameter in `decide()` at `loop_driver.py:640` — and suppress `invoke_ensemble` + guidance when the client sent no tools. `context.tools` is the `list[dict[str,Any]]` from `SessionContext` (defined in `session_start.py:76`); it is already in scope in `decide()` which calls both helpers. |

---

## BUILD-WORK Items

| ID | ADR | Gap | Type | Location | Resolution |
|----|-----|-----|------|----------|------------|
| BW-01 | ADR-043 §Decision 2 | `dispatch_pipeline.py` has no production caller once V-01/V-02 are resolved, but the file and its 335 lines still exist. | exists | `src/llm_orc/agentic/dispatch_pipeline.py` (335 lines) | Delete file after removing all call sites (BW-05, BW-07, BW-08, BW-09, V-01, V-02). |
| BW-02 | ADR-043 §Decision 2 | `ensemble_backed_roles.py` (171 lines; `EnsembleRoutingPlanner`, `EnsembleResponseSynthesizer`) has no production caller once V-02 is resolved; it imports from `dispatch_pipeline.py` (`DispatchPlan` at line 25). | exists | `src/llm_orc/agentic/ensemble_backed_roles.py` (171 lines) | Delete file after V-02 is resolved. |
| BW-03 | ADR-043 §Decision 1 | Module docstring at `v1_chat_completions.py:1–25` describes the endpoint as using a `DispatchPipeline` and lists `DispatchPipeline` as step 4 of the serving contract. | wrong-structure | `v1_chat_completions.py:1–25` | Rewrite the docstring to describe the universal loop-driven Terminal surface (ADR-043 §Decision 1). Remove all `DispatchPipeline` references. |
| BW-04 | ADR-043 §Decision 1 | `chat_completions()` docstring at `v1_chat_completions.py:594–617` explains the two-surface split and says non-tool requests use "ADR-027's single-turn Dispatch Pipeline." The per-request context-sink comment at line 614 says "its turn-boundary observation injection is moot because the pipeline is single-turn." | wrong-structure | `v1_chat_completions.py:594–617` | Update the handler docstring to reflect the single universal surface. Remove the two-surface paragraph and the stale pipeline-specific context-sink note. |
| BW-05 | ADR-043 §Decision 2 | `loop_driver.py:435` (`GenerationTarget` Protocol docstring) references `DispatchPipeline.run()` as the wrapper-contingency target and states it "depends on the single-turn Dispatch Pipeline (WP-B/C)." That pipeline is now retired; the wrapper-contingency fallback path it describes is no longer a live option. | wrong-structure | `loop_driver.py:435`, `loop_driver.py:520` | Update the `GenerationTarget` docstring to note that the wrapper-contingency path (second-order fallback) is no longer available because the Dispatch Pipeline was retired in ADR-043. Remove the "depends on the single-turn Dispatch Pipeline (WP-B/C)" language. The FC-52 swappability contract can stay for forward compatibility; the concrete fallback target changes. |
| BW-06 | ADR-043 §Decision 2 | `test_dispatch_pipeline.py` is a 300+ line test file exercising the now-retired `DispatchPipeline` module directly. It imports from `dispatch_pipeline.py`. | exists | `tests/unit/agentic/test_dispatch_pipeline.py` (full file; imports at lines 21–24) | Delete the file when `dispatch_pipeline.py` is deleted (BW-01). |
| BW-07 | ADR-043 §Decision 2 | `test_ensemble_backed_roles.py` is a test file exercising the now-retired `EnsembleRoutingPlanner` and `EnsembleResponseSynthesizer` adapters. It imports from both `dispatch_pipeline.py` (line 15) and `ensemble_backed_roles.py` (line 16). | exists | `tests/unit/agentic/test_ensemble_backed_roles.py` (full file; imports at lines 15–16) | Delete the file when `ensemble_backed_roles.py` is deleted (BW-02). |
| BW-08 | ADR-043 §Decision 2 | `test_api_v1_chat_completions.py` imports `DispatchPipeline` and `DispatchPlan` (line 34) and `get_dispatch_pipeline` (line 366 monkeypatch). The `TestDispatchPipelineHandlerSwap` class (lines 1340–1424) exercises a real `DispatchPipeline` through the HTTP boundary. The `_build_client` helper (line 332) accepts a `pipeline` parameter, wires `get_dispatch_pipeline`, and documents itself as an ADR-027 pipeline fixture. | exists | `tests/unit/web/test_api_v1_chat_completions.py:34` (imports), `tests/unit/web/test_api_v1_chat_completions.py:327–399` (`_build_client`), `tests/unit/web/test_api_v1_chat_completions.py:1340–1424` (`TestDispatchPipelineHandlerSwap`) | Remove the `DispatchPipeline` / `DispatchPlan` import, the `pipeline` parameter from `_build_client`, the `get_dispatch_pipeline` monkeypatch from `_build_client`, and the `TestDispatchPipelineHandlerSwap` class. |
| BW-09 | ADR-043 §Decision 1 | `test_fc2_layering.py:68–69` records `"llm_orc.agentic.dispatch_pipeline": 2` and `"llm_orc.agentic.ensemble_backed_roles": 2` in the layer map. When the modules are deleted, the test will fail with a missing-module error rather than passing cleanly. | exists | `tests/unit/agentic/test_fc2_layering.py:68–69` | Remove these two entries from the layer map when the modules are deleted. |
| BW-10 | ADR-043 §Decision 1 | `TestSurfaceModeDiscrimination.test_non_tool_request_uses_single_turn_pipeline` (line 554) asserts `pipeline_caller.contexts` is truthy and `terminal_caller.contexts` is falsy for a no-tools request — exactly the two-surface behavior ADR-043 retires. `test_non_tool_request_emits_no_turn_decision` (line 686) asserts no `TurnDecision` events for a no-tools request, with the docstring explicitly naming this as "the non-tool surface routes to the pipeline, not the loop-driver." | exists | `tests/unit/web/test_api_v1_chat_completions.py:554–574` (`test_non_tool_request_uses_single_turn_pipeline`), `tests/unit/web/test_api_v1_chat_completions.py:686–708` (`test_non_tool_request_emits_no_turn_decision`) | Remove both tests. `test_non_tool_request_uses_single_turn_pipeline` asserts the retired pipeline path; `test_non_tool_request_emits_no_turn_decision` asserts absence of `TurnDecision` which is now incorrect — the loop-driver fires on all requests and emits `TurnDecision` on all of them (including no-tools, which finish immediately). |
| BW-11 | ADR-043 §Decision 3 (F-ι.1) | `TestSurfaceModeDiscrimination` class docstring (line 510–518) and `test_tool_request_engages_loop_driver` describe a two-surface discriminator. These tests verify pipeline vs loop-driver routing by client-tool presence. The discriminator's *split* is retired (both paths now reach the loop-driver) though the gate logic (F-ι.1) distinguishes tool vs no-tools at the delegation layer, not the handler level. | wrong-structure | `tests/unit/web/test_api_v1_chat_completions.py:509–609` (`TestSurfaceModeDiscrimination` — the class and its three tool-request tests) | Rename the class to `TestRequestRouting`. Remove or update the "two-surface" language in the docstring. The tool-request tests (`test_tool_request_engages_loop_driver`, `test_tool_request_engages_loop_driver_on_streaming_path`) remain structurally valid — a tools-bearing request still reaches the terminal — but their docstrings/assertions need updating so they verify "loop-driver Terminal is engaged" rather than "pipeline is not engaged" (because the pipeline comparison is no longer meaningful). |
| BW-12 | ADR-043 §Decision 1 | `system-design.agents.md` FC-42 entry (line ~1348) defines "Surface-mode discrimination" as: "a chat-completions request carrying client `tools[]` engages the Loop Driver; a request carrying no client tools engages the Dispatch Pipeline (single-turn); the two surfaces are mutually exclusive per request." The Dispatch Pipeline clause contradicts ADR-043. The `Serving Layer` module entry (line ~623–626) describes `_is_tool_driven` routing to the Dispatch Pipeline for non-tool requests. | wrong-structure | `docs/agentic-serving/system-design.agents.md` — FC-42 table row; Serving Layer `Fitness:` paragraph (~line 626); Serving Layer `Owns (additions):` (~line 623) | Update FC-42 to the ADR-043 FC (one-surface routing): every request engages the loop-driven Terminal. Remove the "engages the Dispatch Pipeline" clause. Add the new F-ι.1 fitness entry. Update the Serving Layer module description to remove the `_is_tool_driven` discriminator. Add a changelog entry for this update. |
| BW-13 | ADR-043 §Decision 1 | `ORIENTATION.md` (line ~7 preamble and line ~46 roadmap section) states "Cycle 7 has two surfaces after the loop-back" and lists "single-turn surface (first-pass WP-A landed; WP-B..E pending)" as Cycle 7 BUILD scope. `roadmap.md` (line ~8 and the WP-B/C/D/E descriptions) still lists WP-B through WP-E as pending work for the single-turn pipeline. These are now retired work packages. | wrong-structure | `docs/agentic-serving/ORIENTATION.md` (lines ~5–7, ~46–50); `docs/agentic-serving/roadmap.md` (lines ~8, ~404, WP-B section ~462–542) | Add a dated note to the top of each file recording that ADR-043 collapses the two-surface architecture; WP-B/C/D/E are formally superseded and do not need to be built. Update the `ORIENTATION.md` "next phase" paragraph and the `roadmap.md` status entries for WP-B/C/D/E to "Retired — ADR-043 (2026-06-18)". |
| BW-14 | ADR-043 §Decision 3 (F-ι.1) | No test exists for the F-ι.1 gate: (a) a no-tools request must not offer `invoke_ensemble` to the seat-filler (gate suppresses delegation tool); (b) a no-tools request must not carry delegation guidance in the seat-filler messages; (c) a write-capable-tools request must offer `invoke_ensemble` (gate permits delegation). FC-58's "guidance never references a tool not offered" invariant is only CI-verifiable if (a) and (b) are tested together. | missing | `tests/unit/agentic/test_loop_driver.py` — no `TestDelegationGate` class | Add a `TestDelegationGateF_iota1` class: (a) `test_no_tools_request_suppresses_invoke_ensemble` — construct the driver with capabilities, call `decide()` with `context.tools=[]`; assert `invoke_ensemble` is absent from `seat_tools` (the list passed to `generate_with_tools`); (b) `test_no_tools_request_suppresses_delegation_guidance` — same setup; assert `_DELEGATION_GUIDANCE` is absent from the seat-filler messages; (c) `test_write_capable_tools_request_offers_invoke_ensemble` — `context.tools` carries a `write` tool; assert `invoke_ensemble` is present in `seat_tools`. |

---

## ADR Cross-Reference Disposition Map

| ADR | Current header | Required action |
|-----|----------------|-----------------|
| ADR-027 | `> Updated by ADR-033 on 2026-06-01.` | Add: `> Superseded by ADR-043 on 2026-06-18. The plan → dispatch → synthesize pipeline is retired; all chat-completions requests now route through the loop-driven Terminal per ADR-043. Body preserved as architecture-of-record.` |
| ADR-033 | No supersession header | Add: `> Updated by ADR-043 on 2026-06-18. §Decision 1 (surface-mode discrimination — two-surface split) is superseded: the discriminator's split is removed and all requests engage the loop-driven Terminal. §Decisions 2–6 and the four fitness criteria are unchanged and now govern all requests, not only tool-driven ones.` |
| ADR-028 (Routing-Planner Ensemble) | `**Status:** Proposed` | Add a dated note: `> Production caller retired by ADR-043 on 2026-06-18. The Dispatch Pipeline that invoked this ensemble is removed; this ADR is dormant architecture-of-record. Body preserved; no code change required.` Do not touch the body. |
| ADR-029 (Response-Synthesizer Ensemble) | `**Status:** Proposed` | Same as ADR-028: add a dated dormant-architecture note. |
| ADR-031 (Latency and Timeout Policy) | `**Status:** Proposed` | ADR-031 governs the pipeline's latency posture. With the pipeline retired, its policy has no live surface. Add a dated dormant note. The loop-driver latency characteristics differ from the pipeline's; if a latency ADR is warranted for the loop, it is a new ADR (out of scope here). |
| ADR-032 (Fallback Shape and Transparent-Endpoint Split) | `**Status:** Proposed` | ADR-032 governs the pipeline's direct-completion fallback and response labeling. With the pipeline retired, the `DirectCompletionFallback` event and the three-layer honest labeling mechanism have no production caller. Add a dated dormant note. WP-E work packages listed in `roadmap.md` are formally retired (see BW-13). |
| ADR-021 (Per-Capability Dispatch) | Has ADR-022 + ADR-024 + ADR-025 update headers; actor-shift note refers to the routing-planner ensemble | No additional header required. The per-capability dispatch contract (one capability sub-task per request) continues to govern the loop-driver's callee delegation path. The actor-shift note naming the routing-planner ensemble as the new actor is stale (the planner is retired); a clarifying note is desirable but not a blocking BUILD item. Record as follow-up. |
| ADR-022 (Routing Surface Behavior) | Has `Updated by ADR-027` header; notes amendment is "structurally moot for chat-completions" | No further header required. ADR-027 is now superseded, but ADR-022's mootness conclusion strengthens rather than weakens under ADR-043 — the loop still doesn't route through `OrchestratorRuntime`. No change needed. |

---

## Conforming Items — What BUILD Inherits Without Modification

**C-01 — The loop-driven Terminal is already the universal caller for tool-driven requests.**
`get_client_tool_action_terminal()` at `v1_chat_completions.py:623` already constructs and returns the Terminal unconditionally on the tool-driven branch. Promoting it to the universal branch requires only removing the conditional at line 622 — no Terminal change, no factory change.

**C-02 — `context.tools` is already in scope in `decide()` via `seat_tools` at line 640.**
`loop_driver.py:640` reads `list(context.tools)` and passes it to `generate_with_tools`. The F-ι.1 gate needs to inspect this same field; `context` is the `decide()` parameter. No new plumbing.

**C-03 — `_delegation_tools()` and `_seat_filler_messages()` already have a capabilities-present guard.**
Lines 829–831 (`_delegation_tools`) and lines 860–861 (`_seat_filler_messages`) both short-circuit on `not self._capabilities`. The F-ι.1 gate is a second condition on the same short-circuit paths — adding `or not context.tools` (or a named predicate) at these guards is a one-liner per method. The FC-58 invariant ("guidance never references a tool not offered") is structurally preserved because both methods suppress the tool offering and the guidance under the same condition.

**C-04 — The Terminal's un-executable-tool_call path is already structurally unreachable on a no-tools request with the F-ι.1 gate in place.**
ADR-043 §Relationship to ADR-034 notes that the gate closes the F-ι.1 gap upstream; the Terminal's `_emit_apply_work` path requires a `ClientToolCall` which requires an `invoke_ensemble` which is suppressed by the gate. No Terminal change is needed.

---

## One-Logical-Unit Assessment

The change is one logical BUILD unit: a subtractive surface collapse with a small additive gate. Specifically:

- **Subtractive core:** remove `_is_tool_driven`, the handler branch, `get_dispatch_pipeline`, the imports, and the two retired source modules + their tests.
- **Additive gate:** add the F-ι.1 predicate in two method bodies (~3 lines total) and the `TestDelegationGateF_iota1` test class.
- **Doc/header sweep:** dated headers on ADRs + doc updates.

None of these require a new module, a new dependency edge, or a new protocol interface. The "architect folded as a BUILD Design Amendment" routing holds. There is no finding that argues for a separate ARCHITECT pass.

The only judgment call is whether to split the subtractive deletion (dispatch pipeline + tests) from the additive F-ι.1 gate into separate commits. That split aligns with the project's "structure vs behavior" commit discipline and is recommended: delete first (structure, trivially reversible), gate second (behavior, needs tests). Both commits are small and independently reviewable.

---

## Test-Gap and Risk Findings

**TG-01 — `TestSurfaceModeDiscrimination.test_non_tool_request_uses_single_turn_pipeline` will pass until BUILD lands (BW-10).**
This test currently passes by confirming the pipeline is driven on a no-tools request. After the pipeline is removed, it becomes a dead import error; before removal it is actively misleading CI. BUILD must touch this test in the same commit as V-01.

**TG-02 — `test_non_tool_request_emits_no_turn_decision` asserts the wrong invariant post-ADR-043 (BW-10).**
Under ADR-043, a no-tools request routes through the loop-driver and the driver emits a `TurnDecision` with `action="finish"` (same as the tool-present case at line 684). This test passes today but would flip to failing on V-01. The expected behavior after BUILD is a `TurnDecision(action="finish")` — the test should be replaced or rewritten to assert that, not to assert absence.

**TG-03 — No test covers F-ι.1 gate suppression (BW-14).**
This is the highest-risk gap. The ADR-043 §Decision 3 FC states: a no-tools request whose response emits any `tool_call` violates the gate. Without TG-03's test class, the gate can be missing from the implementation and CI is green. BUILD must add `TestDelegationGateF_iota1` (BW-14) in the same commit as V-03.

**TG-04 — `test_fc2_layering.py:68–69` will fail (KeyError or assertion error) when the modules are deleted (BW-09).**
This is a hard CI failure — not a latent risk. It must be fixed in the same commit as BW-01/BW-02.

---

## Notes

### Scope of the "write-capable tool" predicate (F-ι.1 boundary)

ADR-043 §Consequences §Negative names an edge case: a client offering only read-class tools (no write target). The current implementation already has the right structure: `seat_tools = self._delegation_tools() + list(context.tools)` at line 640 builds from both. The simplest F-ι.1 predicate is `bool(context.tools)` (any client tool present), which ADR-043's Spike ι arm B validated (27/30 finish-with-text at N=10/cell; no over-delegation). The write-capable narrowing is the documented future option if the read-only-tools edge case surfaces. BUILD should implement `bool(context.tools)` (not a tool-name inspection) for now and note the boundary in the gate's docstring.

### ADR-028/029/031/032 have no live production code to remove

These four ADRs governed the now-retired Dispatch Pipeline's internals (planner ensemble contract, synthesizer ensemble contract, latency policy, response labeling). None of them have standalone production modules — `EnsembleRoutingPlanner`, `EnsembleResponseSynthesizer`, and `DispatchPipeline` are all in the two files being deleted. The routing-planner YAML (`spike-cycle7-zeta-routing-planner.yaml`) and response-synthesizer YAML may still be present in the ensembles directory; BUILD should decide whether to archive or delete them. This is a follow-up item, not a blocking conformance violation.

### `GenerationTarget` Protocol and the wrapper-contingency comment (BW-05)

The wrapper-contingency fallback was always a recorded contingency, not a built path. With the Dispatch Pipeline retired, the fallback target it named (`DispatchPipeline.run()`) no longer exists. FC-52's swappability contract can be kept as a forward-compatibility affordance for future substitution; the comment about `WP-B/C` dependency needs to be removed. This is a comment-only cleanup.

### ADR-021 actor-shift note

ADR-021's `Updated by ADR-027` header records the actor shift from orchestrator-LLM to routing-planner ensemble. The routing-planner is now retired. The per-capability dispatch contract itself is unchanged and continues to govern the loop-driver's callee delegation. A follow-up note to ADR-021 recording that the planner actor is retired (and the loop-driver's delegation gate is now the actor for capability selection on the universal surface) would complete the provenance chain but is not a BUILD blocker.
