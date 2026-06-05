# Roadmap: Agentic Serving

**Generated:** 2026-04-20; **last amended:** 2026-06-02 (Cycle 7 loop-back ARCHITECT close)
**Derived from:** `system-design.md` (v6.0), ADRs 001-034, scenarios.md (Cycle 7 + loop-back additions), interaction-specs.md (Cycle 7 + loop-back additions)

This roadmap expresses the sequencing landscape for building agentic serving — what depends on what, where the builder has a choice, and which coherent intermediates are worth pausing at. It does not prescribe a build order. Work package order within each dependency band is a build-time decision.

> **Cycle 7 has two surfaces after the loop-back.** The first-pass Cycle 7 ARCHITECT designed the **single-turn (answer-a-question) surface** (Dispatch Pipeline plan → dispatch → synthesize). The BUILD reflection-gate then surfaced that this surface cannot deliver work to a tool-driven client's filesystem, triggering the loop-back. The loop-back ARCHITECT (this close) adds the **tool-driven multi-turn surface** (layer-A Loop Driver + client-tool-action Terminal + Artifact Bridge). **Both surfaces are Cycle 7 BUILD scope.** The loop-back tool-driven WPs are in the new section immediately below; the single-turn WPs are the section after it (first-pass WP-A landed; WP-B/C/D/E remain pending and valid). A surface-mode discriminator on the Serving Layer routes each request to the right surface; the **Capability List Builder (single-turn WP-D) is shared by both surfaces**.

---

## Work Packages — Cycle 7 loop-back (active; tool-driven multi-turn surface)

> **Six loop-back work packages (WP-LB-A through WP-LB-F).** They build the ADR-033 layer-A Loop Driver + ADR-034 client-tool-action Terminal + Artifact Bridge surface. Identifiers are `WP-LB-*` to avoid collision with the first-pass Cycle 7 `WP-A..E` (single-turn surface, below). **BUILD-mode declaration:** to be set at BUILD entry per ADR-091; recommended **gated** — the loop-back carries the cycle's load-bearing open risk (axis-2 long-horizon driver coherence, ADR-033 §Decision ¶5) and a drafting-time-synthesized surface-mode discriminator (validate-not-assume), both of which want practitioner-in-the-loop stewardship; auto mode is appropriate only once the surface-mode shape + Loop Driver structure are in place and the remaining WPs reduce to mechanical wiring.

### Track A-LB — `refactor:` commit (precedes WP-LB-A)

- **A-LB.1 — Remove the stale `ClientToolCall` docstring** (conformance Finding 5 refactor-now; advisory #6). The comment at `v1_chat_completions.py:581-583` documents `ClientToolCall` chunks as "not part of this surface's vocabulary under ADR-027" — it now contradicts ADR-034 and would mislead BUILD implementors re-introducing the terminal. Remove it as a standalone `refactor:` commit before the loop-back feature work begins (the one refactor-now item from the loop-back conformance scan; scenarios.md §"Loop-back Structural Debt Remediation"). Behavior-neutral; precedes WP-LB-A.

### WP-LB-A: Surface-mode discrimination (ADR-033 §Decision 1; D1)

**Objective:** Add the surface-mode discriminator to the Serving Layer so tool-driven requests (client `tools[]` present) engage the Loop Driver and non-tool requests continue through the Dispatch Pipeline. This re-scopes the first-pass WP-A handler swap (`0a7a822`), which currently routes every request through the pipeline.

**Changes:**
- Serving Layer extension at `web/api/v1_chat_completions.py`: a named discriminator (`len(request.tools) > 0`) selecting the surface; tool-driven requests delegate to the Loop Driver (WP-LB-B), non-tool requests continue to `dispatch_pipeline.run(context)`. **D1 resolution:** the branch lives in the Serving Layer as a named function (not a separate module — one branch coupled to request parsing the Serving Layer already owns).
- **Validate-not-assume (advisory #2):** the discriminator's signal validity — whether `tools[]` presence is the right surface discriminator (a tool-capable client might send `tools[]` for bookkeeping without expecting an agentic loop) — is flagged for production-traffic confirmation, not assumed from one spike client. The safe edge case (a tool-capable client wanting a plain answer) is handled by the Loop Driver finishing with text.

**Scenarios covered:** scenarios.md §"Layer-A Loop-Driver and Surface-Mode Discrimination (ADR-033)" — Surface engages the loop-driver when the request carries client tools; Surface uses the single-turn pipeline when no client tools are present; Preservation: the single-turn pipeline surface (ADR-027) is unchanged for non-tool requests.

**Participating modules:** Serving Layer (extended — surface-mode discriminator). FC-42.

**Dependencies:** Hard on first-pass **WP-A** (the handler the discriminator branches; landed `0a7a822`). Hard on **Track A-LB.1** (remove the stale docstring first). Gates **WP-LB-B** (the tool-driven branch needs a Loop Driver to delegate to) — but the branch can land routing tool-driven requests to a stub Loop Driver first.

---

### WP-LB-B: Loop Driver + Single-Step Enforcer + per-turn callee + axis-2 diagnostics (ADR-033)

**Objective:** Land the layer-A control structure — the Loop Driver (per-turn next-action decision, tool-mapping, swappable seat-filler LLM, per-turn callee delegation to a single capability ensemble) + the Single-Step Enforcer (batch-truncation grounding guarantee) + per-turn `TurnDecision` diagnostic events for axis-2 split-vs-callee diagnosis.

**Changes:**
- New module `agentic/loop_driver.py` (L2) owning the multi-turn control structure; invokes the seat-filler LLM (injected by the Serving Layer from the resolved Model Profile, ADR-011 — swappable; cheap-tier default, frontier-tier the named axis-2 fallback); maps the deliverable to a client tool (`write`/`edit`/`bash`/`read`); delegates per-turn generation to a single capability ensemble via `OrchestratorToolDispatch.dispatch()` (the callee — not the pipeline); emits `TurnDecision` events through the Dispatch Event Substrate.
- New module `agentic/single_step_enforcer.py` (L2) — a stateless batch-truncation policy (D2-selected technique; the only candidate with τ′ evidence + model-independence). The two untested candidates (re-planning prompt; one-tool `tool_choice`) remain tunable behind this boundary.
- New `TurnDecision` event dataclass (turn index, action, delegated-ensemble, grounded-carry-held, re-plan-after-truncation) — additive to the Dispatch Event Substrate per FC-24 extensibility.
- The Loop Driver delegates capability selection using the **Capability List Builder** (single-turn WP-D — shared).

**Scenarios covered:** scenarios.md §ADR-033 — Loop-driver delegates per-turn generation to a single capability ensemble (callee, not the pipeline); Single-action-per-turn enforcement truncates a driver batch; Grounded carry — an action depending on a prior observed result uses the observed value; Loop-driver finishes with a text completion when no further action is needed; Preservation: AS-10 capability matching from request content alone is unchanged. FC-43, FC-44, FC-45, FC-46, FC-51.

**Participating modules:** Loop Driver (new), Single-Step Enforcer (new), Orchestrator Tool Dispatch (existing — callee dispatch chokepoint), Capability List Builder (single-turn WP-D — shared), Budget Controller (existing — per-turn enforcement), Dispatch Event Substrate (existing — `TurnDecision` events), Serving Layer (seat-filler injection).

**Dependencies:** Hard on **WP-LB-A** (the surface-mode branch engages the Loop Driver). Implied on single-turn **WP-D** (Capability List Builder — shared; the Loop Driver is testable against a stub list until WP-D lands). Open choice with **WP-LB-C** / **WP-LB-D** internals, but the end-to-end loop needs all three.

---

### WP-LB-C: Client-Tool-Action Terminal + multi-turn loop participation (ADR-034)

**Objective:** Land the terminal that emits `finish_reason: "tool_calls"` carrying the deliverable, and the multi-turn loop participation (consume the incoming `role: "tool"` result, route it to the Loop Driver). Re-introduce the four `ClientToolCall` pieces commit `0a7a822` removed, on the tool-driven path.

**Changes:**
- New module `agentic/client_tool_action_terminal.py` (L3) — emits the streamed `ClientToolCall` (OpenAI streaming tool-call delta shape) via the existing `OpenAiSseFormatter` (the formatter's `ClientToolCall` case was retained by `0a7a822` — reused, not rebuilt); for the non-streaming path, re-introduces the `tool_calls` field on `_NonStreamingResult`, the `isinstance(chunk, ClientToolCall)` branch in `_collect_non_streaming`, and the `message["tool_calls"]` + `content: null` shaping in `_build_completion_body`.
- **Loop participation:** consumes the incoming request's trailing `role: "tool"` message and surfaces it to the Loop Driver. The current `_extract_request` (`dispatch_pipeline.py:325-335`) reads only the last `role: "user"` message and drops tool results — the terminal's message routing must not call it or must extend it (conformance Finding 6).
- Re-add the `ClientToolCall` import to `v1_chat_completions.py` (removed by `0a7a822`; conformance Finding 10).

**Scenarios covered:** scenarios.md §"Client-Tool-Action Terminal and Artifact-Bridge (ADR-034)" — Terminal emits finish_reason=tool_calls carrying the deliverable; Surface consumes the tool-result follow-up and continues the loop; Terminal never writes to the client's filesystem directly; Preservation: the SSE formatter's existing ClientToolCall handling is reused, not rebuilt. FC-47, FC-48, FC-50.

**Participating modules:** Client-Tool-Action Terminal (new), Serving Layer (composes the terminal), Artifact Bridge (WP-LB-D — resolves deliverable content), `OpenAiSseFormatter` (existing — reused).

**Dependencies:** Hard on **WP-LB-B** (the terminal emits the Loop Driver's decision). Hard on **WP-LB-D** (the terminal calls the Artifact Bridge to resolve deliverable content). Hard on **Track A-LB.1** (the stale docstring removed first).

---

### WP-LB-D: Artifact Bridge + `SessionArtifactStore.read_deliverable()` (ADR-034 §Decision 3)

**Objective:** Land the artifact-bridge marshalling step + the first read-side API on the Session Artifact Store. This is the highest-priority BUILD design dependency (conformance scan: the terminal chain cannot close without `read_deliverable`).

**Changes:**
- Session Artifact Store extension: `read_deliverable(reference: ArtifactReference) -> str | bytes` — resolves `reference.path` relative to the store root and returns content. First read-side API on the formerly write-only store (advisory #4).
- New module `agentic/artifact_bridge.py` (L2) — reads the deliverable (substrate-routed via `read_deliverable`; inline via `envelope.primary`) and marshals content into the tool-call `content` argument with fidelity (not a summary). Deterministic framework code, not an LLM generation.
- **Fidelity-at-scale (advisory #4 scope):** spike evidence used trivially small content; BUILD adds a large-deliverable integration test (scenarios.md Cycle 7 loop-back Acceptance Criteria Table row 3).

**Scenarios covered:** scenarios.md §ADR-034 — Artifact-bridge reads the substrate-routed deliverable and marshals it into tool-call content; Inline-substrate deliverable skips the bridge step; Preservation: ADR-025 artifact-as-substrate routing is unchanged. FC-49.

**Participating modules:** Artifact Bridge (new), Session Artifact Store (extended — `read_deliverable`), Client-Tool-Action Terminal (consumer from WP-LB-C).

**Dependencies:** Open choice with **WP-LB-B** (the bridge is testable in isolation against a fixture artifact). Hard prerequisite for **WP-LB-C**'s end-to-end terminal (the terminal calls the bridge).

---

### WP-LB-E: Wrapper-contingency accessibility + fallback ordering (ADR-033 §Rejected; advisory #1)

**Objective:** Make the per-turn generation delegation target a swappable strategy so the wrapper-contingency fallback is architecturally accessible without re-architecture, and record the fallback ordering. This is a structural-affordance WP, not a feature the callee design exercises — it keeps F3-1's recorded concession honest.

**Changes:**
- Structure the Loop Driver's per-turn delegation (WP-LB-B) so the generation target is a strategy: single-ensemble (callee — default) or `DispatchPipeline.run()` (wrapper — second-order fallback). The switch requires no change to the control structure, Single-Step Enforcer, or Terminal (FC-52).
- Document the fallback ordering in code + the field guide: **(1)** frontier-tier driver (a Model Profile swap, FC-46 — preferred first fallback if axis-2 validation shows the cheap driver cannot hold the horizon); **(2)** wrapper reversion (second-order, only if a frontier driver also fails).

**Scenarios covered:** No new behavior scenario (the callee path is the active behavior; this WP is the accessibility affordance). FC-52 verifies the swap touches only the delegation-target selection.

**Participating modules:** Loop Driver (the delegation-target strategy seam).

**Dependencies:** Hard on **WP-LB-B** (the delegation path it makes swappable). Low-cost — largely a structural-seam discipline applied when WP-LB-B lands; can fold into WP-LB-B if the builder prefers.

---

### WP-LB-F: Axis-2 diagnostic instrumentation surfacing (advisory #5; OQ #27 axis 2) — **FOLDED into WP-LB-J (2026-06-04)**

**Status: folded, not dropped.** ADR-036's delegation-rate instrumentation needs exactly this surfacing — the `TurnDecision` sink branch is simultaneously the axis-2 split-vs-callee diagnostic (FC-51, this WP's original objective) and the delegation-rate numerator surface (FC-59). The WP-LB-H feed-forward pre-named the fold ("F's TurnDecision surfacing is the standing instrument for exactly this measurement"). All WP-LB-F changes are carried verbatim inside **WP-LB-J** below; the trajectory-reconstruction helper rides along. Original spec retained below for the trail.

**Objective:** Surface the per-turn `TurnDecision` events (emitted by WP-LB-B) through the operator-terminal + orchestrator-context sinks so a failing long-horizon (axis-2) run in PLAY/first-deployment is diagnosable as split-incorrect vs. callee-incorrect. The recorded load-bearing risk (ADR-033 §Decision ¶5) gets its observation surface.

**Changes:**
- Operator-Terminal Event Sink extension: `isinstance(event, TurnDecision)` arm emitting a line-oriented `key=value` entry (turn index, action, delegated-ensemble, grounded-carry-held, re-plan-after-truncation).
- A trajectory-reconstruction helper (offline or dashboard) that distinguishes a wrong-action turn (driver/split) from a wrong-content turn (callee) from the event stream.

**Scenarios covered:** scenarios.md Cycle 7 loop-back Acceptance Criteria Table row 4 (axis-2 long-horizon coherence — PLAY/first-deployment surface). FC-51.

**Participating modules:** Operator-Terminal Event Sink (extended), Loop Driver (the `TurnDecision` producer from WP-LB-B).

**Dependencies:** Hard on **WP-LB-B** (the `TurnDecision` events it surfaces). Open choice with WP-LB-C/D. This is the BUILD-side preparation for the PLAY/first-deployment axis-2 validation — it does not itself validate axis 2 (no synthetic test reaches it; ADR-097 Conditional Acceptance).

---

### WP-LB-G: Offer the seat-filler `invoke_ensemble` + the capability list (surfaced by WP-LB-C validation; Finding B)

**Surfaced by:** the WP-LB-C real-OpenCode validation (2026-06-02; research log `essays/research-logs/cycle-7-wp-lb-c-opencode-validation.md`). A real OpenCode → real `llm-orc serve` → real qwen3:14b round-trip produced the parity mechanism (OpenCode executed the synthesized `write`) but **zero ensemble dispatches** — `LoopDriver.decide` passes only the client's tools to the seat-filler and never offers `invoke_ensemble`, so a real seat-filler cannot delegate. The callee → bridge → ApplyWork path (WP-LB-B/C/D) is unreachable in production until this lands.

**Objective:** Make ensemble delegation reachable by a real seat-filler — offer `invoke_ensemble` alongside the client tools, fed by the capability list, with guidance on when to delegate vs. act directly. The *design* is settled (ADR-033 callee + `invoke_ensemble`; the single-turn `_build_capability_names` is the reusable capability source); the gap is integration + prompt-shaping.

**Changes:**
- `LoopDriver.decide` augments the seat-filler's `tools` with an `invoke_ensemble` tool definition (args: capability `name`, `input`, client `filePath` — the shape `_delegate_generation` already parses) whose enumerated capabilities come from the capability list (reuse the single-turn `_build_capability_names`; thread the capability source into the Loop Driver at construction, no new L3 import).
- A seat-filler system prompt (or message preamble) that frames the delegate-vs-act-directly decision: delegate generation of substantive deliverables to a capability ensemble; carry literal/observed values directly (grounded carry).
- The Serving Layer threads the capability source into `get_loop_driver()`.

**Scenarios covered (new):** the seat-filler offered `invoke_ensemble` delegates a generation sub-task to a capability ensemble; the delegated deliverable marshals through the Artifact Bridge into the client `write` (the callee → bridge → ApplyWork path, now reachable). Harness tests assert the augmented tool list + the delegation branch; the **load-bearing acceptance gate is a real OpenCode session** whose serve log shows a real `invoke_ensemble` dispatch + a `code-generator` artifact marshalled into the client write — not a scripted test.

**Participating modules:** Loop Driver (tool-list augmentation + capability-source dependency), Serving Layer (threads the capability source), Capability List Builder (the reused source).

**Dependencies:** Hard on **WP-LB-B/C/D** (the delegation path it activates). **Sequenced before WP-LB-E/F** (both are downstream of delegation working at all). **Loop-back trigger:** if the real-OpenCode run shows the seat-filler will not reliably delegate (the cheap-driver-skips-delegation tension), loop back to DECIDE on the delegation-decision mechanism (`tool_choice` forcing / routing pre-filter / a different driver-vs-delegation split) — decided on the run's evidence.

**Status: ✅ Closed 2026-06-02** — delegation fired end-to-end against real OpenCode (seat-filler chose `invoke_ensemble` → `code-generator` → bridge → client `write`; Finding B resolved). The run surfaced **Finding D** (the marshalled deliverable is the raw envelope, not usable content) → WP-LB-H.

---

### WP-LB-H: Client-tool deliverable form contract + D1 extraction (ADR-035; Finding D)

**Status: ✅ Closed 2026-06-03** — Finding D refuted at the real-client layer (TS-14 reached): a real OpenCode session executed a delegated, form-contracted `write`; the landed file is bare, fence-free, parsing Python. Commits `f57b61e` (deliverable contract, FC-56) / `030723f` (contract consumers) / `7c14c94` (synthesis excision) / `db09a48` (form directive, FC-53/54) / `9303b0e` (FormRefusedError channel) / `612aa6d` (FormGate seam, FC-57) / `545c1b7` (critic depends_on fix). Suite 2914 green. LB-4 resolved executor-side; LB-5 resolved pass-through; LB-6 resolved to the χ wording (see system-design Amendment #15). **Surfaced Finding E** — delegation under the client's system prompt is ~coin-flip (2 delegated / 2 carried across all real-OpenCode runs; the direct-endpoint probe delegated immediately, isolating the suppressor to the client system prompt). Practitioner disposition: the pre-named loop-back trigger fires → **loop-back #3 to DECIDE on the delegation-decision mechanism, grounded by Spike ψ** (vary-and-measure: baseline rate / guidance variants / server-side `tool_choice` forcing on Ollama+qwen3 / structural pre-filter split). WP-LB-E/F resume after loop-back #3 (F's TurnDecision surfacing is the standing instrument for exactly this measurement — consider folding into the loop-back's BUILD).

**Surfaced by:** the WP-LB-G real-OpenCode run (2026-06-02): delegation fired end-to-end but the client `write` carried the raw ensemble result envelope (`{"results": {coder, critic, synthesizer}, …}`) — unusable as file content (Finding D). Spike φ split the problem into D1 (extraction) + D2a (declared contracts inert at execution) + D2b (deliverable form targets a human reader); Spike χ/χ.2 grounded the mechanism (boundary directive reliable at n=4 single-deliverable types; deterministic shaping fragile at 2/3 multi-fence ambiguity; multi-file-in-one-dispatch breaks).

**Objective:** A delegated client-tool deliverable lands as **usable bare content** — the north-star loop produces a runnable file, not prose-wrapped markdown or a JSON envelope.

**Changes:**
- **Loop Driver:** `compose_form_directive(tool)` stateless helper (`write` → bare file bytes; `bash` → bare command; `edit` → bare replacement content) + injection into the callee `invoke_ensemble` dispatch input (FC-53 presence; FC-54 keying). No ensemble YAML changes (destination-agnostic, ADR-035 decision 2).
- **Artifact Bridge:** `marshal()` gains `destination_tool` (the Terminal threads it); named **FormGate** seam, default pass-through (FC-57). Conservative single-fence normalization is an open choice (LB-5).
- **Orchestrator Tool Dispatch:** D1 extraction fix — store the last *successful* agent's output; fall back across failed terminal nodes; never `json.dumps(raw_result)` when an agent succeeded (FC-56). Extraction locus is an open choice (LB-4).
- **Client-Tool-Action Terminal:** passes the turn's destination tool to the bridge.

**Scenarios covered:** scenarios.md §Client-Tool Deliverable Form Contract (ADR-035, Finding D) — the behavior + D1 + preservation scenarios and the Cycle Acceptance Criteria (Finding D) table.

**Participating modules:** Loop Driver, Artifact Bridge, Client-Tool-Action Terminal, Orchestrator Tool Dispatch.

**Dependencies:** Hard on **WP-LB-B/C/D/G** (all landed — the delegation + marshalling + emission path the directive rides). **WP-LB-E/F resume after WP-LB-H** (the Finding D blocker gates the surface's usability; E/F refine a working surface). **Acceptance gates (load-bearing):** suite green + the **$0 real-OpenCode smoke test re-run showing a runnable file landing** (the Finding D refutation — the WP-LB-G rig re-run with the directive in effect; the layer-match "no" row in the acceptance criteria table is closed at this layer). **Escalation trigger:** if the real run still produces non-bare content, escalate per ADR-035's Conditional Acceptance order (detect-and-refuse gate → schema-retry → frontier seat-filler) on the run's evidence — not a redesign.

---

### WP-LB-I: User-turn delegation guidance composition (ADR-036; Finding E — the mechanism) — ✅ CLOSED 2026-06-04

**Outcome:** Landed (commits `863fb5d` feat + `0f9d48d` refactor; suite 2917 green). **Acceptance gate MET:** the $0 real-OpenCode run delegated under natural phrasing (4× serve-log `dispatch start`; bare `ast.parse`-clean file landed via the client `write`) — ADR-036's Conditional Acceptance gating condition. The same run surfaced **Finding F** (termination suppression on no-new-task tails); Spike ψ″ characterized it as the two-sided F-ψ″.3 tension → **loop-back #5: DECIDE on the termination mechanism** (see cycle-status §Finding F). The ADR-036 ≥0.9 soak read is deferred until termination is decided (Finding F inflates the turn stream).

**Objective (as specified):** Delegation fires reliably on generation-shaped turns under the real client — the seat-filler guidance relocates from the losing system slot (baseline 0/10) to the user-turn region (55/55 measured), per ADR-036 Decision 1.

**Changes:**
- **Loop Driver:** `_seat_filler_messages` composes the delegation guidance into the user-turn region — first turn: attached to the user task; trailing tool-result tails: standalone trailing user-role message (C3 preferred form; C1/C2 equally measured, not violations). The composed request carries **no framework-authored system message** (FC-58). Module + guidance docstrings updated (the conformance F-1 "System guidance" framing predates the ADR).
- **Single-Step Enforcer:** docstring stops calling `tool_choice` BUILD-tunable — the family is empirically closed (Spike ψ.3 third negative); re-planning prompt remains the only untested candidate (conformance F-4; `refactor:` commit).
- **Tool-list completeness constraint (FC-62):** assert the composition path cannot construct a narrowed tool list (ψ.4c empty-response break).

**Scenarios covered:** scenarios.md §Delegation-Decision Mechanism — first-turn composition, trailing-turn C3 form, client-invisibility, the generation-shaped delegation integration scenario, carry-side + verbatim grounded-carry preservations, the docstring refactor scenario.

**Participating modules:** Loop Driver; Single-Step Enforcer (docstring only).

**Dependencies:** None hard (the composition path landed with WP-LB-B/G). **Acceptance gate (load-bearing — ADR-036's Conditional Acceptance gating condition):** the $0 real-OpenCode run with delegation **verified fired** (serve-log `dispatch start` / `TurnDecision` — a passing-looking run can be model-direct; the WP-A scar). Carry-side preservation asserted in the same run shape (FC-61).

---

### WP-LB-J: Delegation-rate instrumentation (ADR-036 Decision 3; absorbs WP-LB-F) — ⏸ HELD pending loop-back #5

*Held 2026-06-04: the Finding F termination mechanism (DECIDE loop-back #5) plausibly extends the `TurnDecision` event shape this WP consumes (finish-policy event / work-remaining field) — build once against the settled shape. Advisory C (verify the WP-LB-F fold preserves FC-51 axis-2 intent) still applies at entry.*

**Objective:** The delegation rate is computable from events alone and operator-visible — the regression-visibility mechanism for the stack-scoped win (the meter is the safety net; Spike ψ′ Arm D: the lever does not transfer across models).

**Changes:**
- **Delegation Rate Meter (new module, `src/llm_orc/agentic/delegation_rate_meter.py`):** graduate the Spike ψ.4a rule from `scratch/spike-psi-delegation-rate/psi4a_prefilter.py` (`classify_turn` over raw inputs; labeled set ships as regression fixtures — 0/12 clear-case errors) + `delegation_rate(events)` over a narrow structural protocol (imports neither Loop Driver nor sinks; conformance F-3).
- **Loop Driver:** per-turn `classify_turn` call; `TurnDecision` gains `turn_shape` (FC-59 denominator).
- **Operator-Terminal Event Sink:** `TurnDecision` consume branch — one INFO line per turn (turn index, action, shape, delegated ensemble, carry-held, replanned) — the former WP-LB-F objective (FC-51) and the rate numerator surface (FC-59) in one branch; rate surfacing cadence (per-N-turns / session-close / on-demand) is a BUILD open choice. The WP-LB-F trajectory-reconstruction helper rides along.
- **Profile-swap re-validation discipline (FC-60):** recorded-run requirement documented in the field guide at the profile-swap touchpoint.

**Scenarios covered:** scenarios.md §Delegation-Decision Mechanism — TurnDecision surfacing, classifier graduation + labeled-set regression, boundary-exclusion observability; plus the original WP-LB-F axis-2 acceptance row (FC-51).

**Participating modules:** Delegation Rate Meter (new), Loop Driver, Operator-Terminal Event Sink.

**Dependencies:** Hard on **WP-LB-B** (landed — the TurnDecision events). **Open choice with WP-LB-I** (classification stamping is independent of guidance placement) — but ADR-036's trailing confirmation (the ≥25 generation-shaped-turn soak ≥0.9) needs both landed, so finishing both before the soak reading is the natural order.

---

## Dependency Graph (Cycle 7 loop-back)

```
Track A-LB.1 (remove stale ClientToolCall docstring)
   │
   └─ hard ─▶ WP-LB-A, WP-LB-C (terminal re-introduction)

WP-LB-A (surface-mode discrimination)
   │
   └─ hard ─▶ WP-LB-B (tool-driven branch engages the Loop Driver)

WP-LB-B (Loop Driver + Single-Step Enforcer + per-turn callee + TurnDecision events)
   │
   ├─ hard ─▶ WP-LB-C (terminal emits the Loop Driver's decision)
   ├─ hard ─▶ WP-LB-E (wrapper-contingency seam on the delegation path)
   └─ hard ─▶ WP-LB-F (surfaces the TurnDecision events)

WP-LB-D (Artifact Bridge + read_deliverable)
   │
   └─ hard ─▶ WP-LB-C (terminal calls the bridge to resolve deliverable content)

WP-LB-B ─ implied ─▶ single-turn WP-D (Capability List Builder — shared; Loop Driver selects from it)
WP-LB-D ─ open choice with WP-LB-B (bridge testable in isolation)

WP-LB-G (delegation reachable; ✅ closed — surfaced Finding D)
   │
   └─ hard ─▶ WP-LB-H (the form contract rides the delegation path G activated)

WP-LB-H (form contract + D1; ✅ closed — surfaced Finding E)
   │
   └─ hard ─▶ WP-LB-I (the guidance composition rides the delegation path H made form-correct)

WP-LB-I (user-turn guidance; the ADR-036 mechanism) ─ open choice with ─ WP-LB-J (instrumentation)
   (the Conditional Acceptance soak reading needs both; WP-LB-J absorbs WP-LB-F)

WP-LB-B ─ hard ─▶ WP-LB-J (the TurnDecision events it instruments; landed)

WP-LB-I, WP-LB-J ─ implied ─▶ WP-LB-E (wrapper-contingency seam — resumes once the delegation surface is reliable + measured)
```

**Classification key:** Hard = structural necessity; Implied = simpler-first but stub-able; Open choice = genuinely independent.

---

## Transition States (Cycle 7 loop-back)

### TS-12: Tool-driven surface delivers a parity round-trip (after WP-LB-A + WP-LB-B + WP-LB-C + WP-LB-D)

When the surface-mode discriminator, Loop Driver + Single-Step Enforcer, Client-Tool-Action Terminal, and Artifact Bridge land, a tool-driven client (e.g. OpenCode) driven against agentic-serving gets a single-turn parity round-trip: the loop-driver decides an action, delegates generation to a capability ensemble, the artifact-bridge marshals the deliverable, the terminal emits `finish_reason: "tool_calls"`, the client executes the write locally, and the tool result feeds back. This is the **north-star "delegate work, apply locally" surface** the loop-back exists to deliver. The multi-turn loop is observable from a real client; long-horizon coherence (axis 2) is not yet validated (PLAY/first-deployment, ADR-097 Conditional Acceptance).

### TS-14: Form-contracted parity — the north-star loop produces runnable files (after WP-LB-H)

TS-12's parity round-trip with ADR-035's form contract in effect: the delegated deliverable landing via the client's `write` is bare, runnable file content (the Finding D refutation), and the D1 extraction stores the terminal agent's output rather than the raw envelope. After this state, WP-LB-E/F resume and the remaining loop-back work is refinement + instrumentation, not surface correctness. Trajectory-scale form compliance remains the PLAY target (ADR-097 Conditional Acceptance).

### TS-15: Delegation reliable and measured (after WP-LB-I + WP-LB-J)

TS-14's form-contracted parity with ADR-036's mechanism and meter in effect: generation-shaped turns delegate reliably under the real client (the V3 composition; gating condition = the delegation-verified real-OpenCode run), and the delegation rate is computable from events alone with the boundary-excluded share observable (the trailing ≥25-generation-shaped-turn soak reads ≥0.9 or refutes). The win is stack-scoped (composition × qwen3:14b × OpenCode 1.15.5) — the meter is what makes losing it visible; profile swaps re-validate (FC-60). After this state only WP-LB-E remains in the loop-back set.

### TS-13: Cycle 7 fully complete — both surfaces (after both surface WP sets)

Both Cycle 7 surfaces shipped: the single-turn answer-a-question surface (first-pass WP-A landed + WP-B/C/D/E) and the tool-driven multi-turn surface (WP-LB-A..E + G..J; F folded into J). The surface-mode discriminator routes each request; the Capability List Builder serves both. Axis-2 long-horizon driver coherence is carried into PLAY/first-deployment as the recorded load-bearing risk under ADR-097 Conditional Acceptance, with the seat-filler swappable (cheap ↔ frontier) as the testable lever and FC-51 instrumentation making a failure diagnosable.

---

## Open Decision Points (Cycle 7 loop-back)

### LB-1. Single-step enforcement technique tuning (D2 selected: batch-truncation)

ARCHITECT selected **batch-truncation** (the only candidate with τ′ evidence + model-independence). FC-43 constrains the observable single-action behavior, not the technique, so BUILD may tune to a re-planning prompt or a one-tool `tool_choice` constraint behind the Single-Step Enforcer boundary if truncation surfaces issues. Open decision deferred to BUILD signal (`tool_choice` is the weakest candidate — Spike κ showed the framework does not forward it and MiniMax did not honor it).

### LB-2. Seat-filler driver model (cheap-tier default; the load-bearing axis-2 bet)

The Loop Driver's seat-filler is a swappable Model Profile (FC-46). Cheap-tier is the default (preserving the cost-distribution value proposition); frontier-tier is the named axis-2 fallback. The cheap-vs-capable bet is the recorded load-bearing risk (ADR-033 §Decision ¶5), resolved by outcome in PLAY/first-deployment via a config change, not re-architecture. Open decision deferred to axis-2 evidence.

### LB-3. `edit`/`bash`/multi-file/streaming-token coverage (ADR-034 §Negative)

Spike π/ρ validated the `write` round-trip for a single new file. `edit`-in-place (needs a `read` first), `bash` command execution, multi-file deliverables, and token-streaming the synthesized content are BUILD scope. Open decision: which of these land in Cycle 7 loop-back BUILD vs. defer to a follow-on cycle — the file-action `write` path is the BUILD focus; the richer surface is north-star context (ADR-033 §Decision 6(c)). *(Loop-back #2 note: multi-file is now design-resolved as across-turn decomposition — ADR-035 §Decision 3 / FC-55; what remains open is which destinations beyond `write` get directive coverage exercised in BUILD.)*

### LB-4. D1 extraction locus — ✅ resolved executor-side (WP-LB-H gate, 2026-06-03)

Resolved at the WP-LB-H scenario-group gate on the practitioner's contract framing ("the ensemble abstraction presents a single output"): `resolve_deliverable` computes the unique terminal node from `depends_on` at finalize and populates `ExecutionResult.deliverable`; the dispatch layer reads the contract instead of reconstructing it. The legacy `synthesis` field was excised wholesale. See system-design Amendment #15.

### LB-5. Conservative normalization — ✅ resolved pass-through (WP-LB-H, 2026-06-03)

No normalization ships: χ-P3/P4/P5 produced zero fences under the directive, so the strip would be day-one dead code. The FormGate seam exists (FC-57) with a pre-built `FormRefusedError` refusal channel the Terminal already degrades — the detect-and-refuse escalation installs at the seam with literal zero Terminal edits on PLAY evidence.

### LB-6. Directive wording — ✅ resolved to the χ-probe wording (WP-LB-H, 2026-06-03)

"Output ONLY {the exact raw bytes of the file | the exact shell command | the exact replacement content}. No markdown fences, no prose, no explanations, no example blocks." — appended after the generation task in the dispatch input. Held through the dependency wrapper's "provide your own analysis" framing on all delegating runs. Remains tunable within FC-53/54; if PLAY shows wording-sensitive drift, the dependency wrapper's framing is a named suspect.

### LB-7. Delegation-decision mechanism (Finding E; loop-back #3 → DECIDE behind Spike ψ)

Delegation under the client's system prompt is ~coin-flip (2/2 carried on natural phrasing; the direct probe delegated immediately). Candidate levers for Spike ψ to vary and measure ($0 local, replaying a captured real-OpenCode request shape): (ψ.1) baseline delegation rate under the current nudge; (ψ.2) guidance wording/position variants; (ψ.3) server-side `tool_choice` forcing on the seat-filler call (Ollama+qwen3 — distinct surface from Spike κ's Zen/MiniMax negative); (ψ.4) structural pre-filter split (the framework decides delegate-vs-carry; the seat-filler decides only the action shape — model-independent, consistent with the framework-guarantees-the-contract thesis). DECIDE picks the mechanism on measured rates.

---

## Work Packages — Cycle 7 single-turn surface (active; carried across the loop-back)

> **Status at loop-back close (2026-06-02):** First-pass **WP-A landed** (commits `e538264` + `0a7a822`; closed — status updated in-place below); the loop-back **WP-LB-A re-scopes its handler** via the surface-mode branch (WP-A made the pipeline the only surface; WP-LB-A adds the discriminator so tool-driven requests route to the Loop Driver instead). **WP-B / WP-C / WP-D / WP-E remain pending and valid** — they build the single-turn (non-tool) answer-a-question surface, which ADR-027 is now scoped to. The **Capability List Builder (WP-D) is shared** with the loop-back tool-driven surface (the Loop Driver selects per-turn capabilities from it). Sequence these alongside or after the loop-back tool-driven WPs at the builder's discretion; both surfaces are Cycle 7 BUILD scope. Track A.1 / A.2 (single-turn spike refactors) and Spike ν (A.3, complete) are unchanged.

## Work Packages — Cycle 7 (active)

> **Cycle 7 BUILD shapes as 5 work packages (WP-A through WP-E) + 2 Track A `refactor:` commits.** Identifiers reset for the new active cycle per the methodology — Cycle 6 BUILD's WP-A/WP-B/WP-D closed; WP-C/WP-E carry-forward with status updates noted below.
>
> **Cycle 7 BUILD-mode declaration**: to be set at BUILD entry per ADR-091. Recommended **gated** given the central architectural pivot character (replacing `OrchestratorRuntime` as the chat-completions caller; AS-9 + AS-10 universally satisfied as constitutional commitments; design-alternative surfaces in `tool_choice` disposition + multi-step composition mechanism + capability-list discovery surface choice). Auto mode appropriate only after WP-A structural shape is in place + the remaining WPs reduce to mechanical wiring.

### Track A — `refactor:` commits + architect-build boundary spike (precede WP-A / WP-B / WP-C)

The first two entries are `refactor:` commits that extend Spike artifacts to match the production contracts ADR-028 + ADR-029 specify. The third entry is an architect-build boundary spike that gates WP-A entry — per the EPISTEMIC GATE conversation (architect→build boundary, 2026-05-23), the long-horizon capability ceiling question identified at gate is load-bearing for the structural-bounding generalization AS-9 + ADR-027 rest on. The spike's results determine whether WP-A proceeds as designed or triggers architecture revision via Design Amendment.

- **A.1 — Routing-planner spike output schema `input` field** (conformance Finding 4). Edit `spike-cycle7-zeta-routing-planner.yaml` system prompt to specify the `"input"` field per ADR-028 §Output contract. The 20-prompt battery continues to pass with the added field populated. Single YAML system-prompt edit; precedes WP-B.
- **A.2 — Response-synthesizer spike Rule 6 codification** (conformance Finding 6). Edit `spike-cycle7-epsilon-response-synthesizer.yaml` system prompt to add Rule 6 (framework-convention enumeration in direct-completion mode) per ADR-029 §"Strict-fidelity rule set". Single YAML system-prompt addition; precedes WP-C.
- **A.3 — Spike ν: long-horizon capability ceiling probe** (architect→build boundary; **gates WP-A entry**). Cheap-tier qwen3:8b via local Ollama; $0 cost (per `feedback_free_options_preference`). Tests the three surfaces the cycle's existing Spike battery (ζ, ε, ε', μ) did not exercise — multi-step composition (per OQ #21), production-scale numerical content (per ADR-027 §Negative plausible-but-untested), adversarial routing (per OQ #25 production traffic diversity).

  **Pre-specified qualitative criteria** (per MODEL snapshot Advisory A — pre-specify before running):

  | Surface | Pass criterion (supports ADR-027 commitment) | Fail criterion (triggers architecture revision) | Intermediate (caveat-with-deployment-policy) |
  |---|---|---|---|
  | Multi-step composition (2-step + 3-step chains) | ≥80% successful end-to-end completion; structural-bounding holds at each step | <50% successful completion; structural-bounding does not extend to multi-step shapes | 50-80% completion → operator-deployment tier-escalation policy for multi-step workloads (extend ADR-031) |
  | Production-scale numerical content (100-figure + structured tables) | ≥95% fidelity (no rounding drift exceeding Spike ε' Mode 1/2 baseline) | <80% fidelity (cheap-tier numerical-fidelity floor too low for production) | 80-95% fidelity → runtime fidelity check load-bearing in ADR-029 §"Rounding-drift mitigation playbook" |
  | Adversarial routing (40-prompt adversarial battery extending Spike ζ's 20-prompt battery) | 100% JSON conformance + ≥80% defensible-judgment-match | <80% JSON conformance (planner reliability collapses under adversarial pressure) | 80-95% conformance → classifier pre-filter + caching tuning axes elevated from optional to recommended (ADR-031) |

  **Trigger conditions:**
  - **All three surfaces Pass** → WP-A proceeds as designed; spike findings recorded as empirical-floor strengthening; ADR-027 commitment stands.
  - **Any surface Fail** → trigger Design Amendment process per system-design.md §Design Amendment Log; architecture re-opens between ADR-027 framework-driven pipeline as PRIMARY vs. Tier 1 hybrid as PRIMARY vs. multi-mechanism architecture; user authorizes the next step before WP-A resumes.
  - **Single-surface Intermediate** → record finding as caveat-with-deployment-policy; update ADR-029 or ADR-031 per the relevant playbook; WP-A proceeds with the updated deployment policy as a constraint.
  - **Two or three surfaces Intermediate simultaneously** *(per architect-snapshot Advisory B carry-forward, 2026-05-23)* → treat as a candidate ceiling signal. Each surface in isolation is non-fatal, but two or three surfaces landing in the Intermediate band collectively suggest the structural-bounding generalization may not hold at the breadth ADR-027 commits to. Action: present the Intermediate cluster to the practitioner before WP-A proceeds; the practitioner authorizes one of (a) WP-A proceeds with the cluster of caveats stacked and an explicit Cycle 7 BUILD/PLAY ceiling-validation focus; (b) trigger Design Amendment per the Fail path; (c) run Spike ν follow-up tests narrowing the Intermediate surfaces to Pass or Fail before WP-A. The rule prevents the "many non-fatal results collectively mask a ceiling" failure mode that single-surface evaluation does not catch.

  **Methodology note (per MODEL snapshot Advisory A):** the pre-specified criteria above lock in *before* the spike runs, so the analysis avoids the Spike μ.1 procedural gap where qualitative criteria were articulated only after the pattern detector flagged a fail. The post-spike writeup records the pre-specified criteria + the actual outcome + the analysis of any criterion-boundary edge cases.

  **Empirical scope honesty:** Spike ν cannot exercise production-scale traffic volume or operator-deployment hardware diversity. The spike is a structural-bounding generalization test at qwen3:8b across the three named surfaces. Production-scale validation remains PLAY-phase + first-deployment-evidence territory.

  Writeup target path: `essays/research-logs/cycle-7-spike-nu-long-horizon-ceiling.md`.

---

### WP-A: Dispatch Pipeline + Plan→InternalToolCall adapter + pipeline-stage event types (ADR-027) — ✅ **Closed 2026-05-24** (commits `e538264` feat + `0a7a822` handler swap)

> **Loop-back re-scoping (2026-06-02):** WP-A landed the pipeline as the *only* chat-completions surface (the handler swap routed every request through it). The loop-back scopes the pipeline to the single-turn (non-tool) surface; **WP-LB-A** adds the surface-mode discriminator above it so tool-driven requests route to the Loop Driver instead. The `ClientToolCall` plumbing WP-A removed in `0a7a822` returns under the loop-back terminal (WP-LB-C). WP-A stands as a faithful, green, reversible increment of ADR-027 as written; the objective and changes below are the as-built record.

**Objective:** Land the framework-driven dispatch pipeline (plan → dispatch → synthesize) as the chat-completions caller. Introduce the new Dispatch Pipeline module at L2, the Plan→`InternalToolCall` adapter inside the pipeline (per ARCHITECT Finding 11 disposition), and the four new pipeline-stage event types (`PlanEmitted`, `DispatchFired`, `SynthesizerCompleted`, `DirectCompletionFallback`) emitted through the existing Dispatch Event Substrate.

**Changes:**
- New module `agentic/dispatch_pipeline.py` owning the three-stage orchestration. The pipeline's public surface yields the existing chunk vocabulary (`ContentDelta | VisibilityEvent | ClientToolCall | Completion`) so `OpenAiSseFormatter` consumes it unchanged per ARCHITECT Finding 8 disposition. The Plan → `InternalToolCall` adapter lives inside the pipeline module as a pure function.
- New event dataclasses for pipeline-stage events (Cycle 7 additions to the event vocabulary): `PlanEmitted(dispatch_id, plan)`, `DispatchFired(dispatch_id, ensemble_name)`, `SynthesizerCompleted(dispatch_id, finish_reason)`, `DirectCompletionFallback(dispatch_id, request_shape_category, planner_rationale)`. The Dispatch Event Substrate is structurally extensible (Protocol-based per FC-24) — additive new event types only.
- Serving Layer extension: `chat_completions()` handler at `src/llm_orc/web/api/v1_chat_completions.py` replaces the `_build_runtime()` + `runtime.run(context)` path with `dispatch_pipeline.run(context)`. The existing heartbeat scheduler + orchestrator-context sink lifecycle per Cycle 6 WP-B / WP-C is preserved structurally; both continue to register with the substrate.
- `OrchestratorRuntime` class remains in the codebase per ARCHITECT Finding 2 disposition (a) — preserved as architectural option; no production caller on chat-completions surface post-BUILD; ADR-001 + ADR-011 + ADR-022 system-prompt amendment retained as architectural commitments.

**Scenarios covered:** scenarios.md §"Framework-Driven Dispatch Pipeline (ADR-027)" — Chat-completions request flows through plan → dispatch → synthesize pipeline; No-capability-match request flows through plan → direct-completion synthesize path; (integration) Plan-stage output is a typed `InternalToolCall`-compatible shape; `OrchestratorRuntime` is not invoked on the chat-completions surface; Preservation: `OrchestratorToolDispatch.dispatch()` contract is unchanged; Preservation: ADR-021's per-capability dispatch contract structural commitments unchanged; Preservation: `llm-orc invoke` CLI surface is unaffected by ADR-027.

**Participating modules:** Dispatch Pipeline (new), Serving Layer (extended — handler invokes Dispatch Pipeline; preserves existing heartbeat + context-sink lifecycle), Orchestrator Tool Dispatch (existing — Stage 2 dispatch via Plan→InternalToolCall adapter; same chokepoint preserves calibration gate + tier router + autonomy interpositions), Dispatch Event Substrate (existing — extended with four new event types).

**Dependencies:** None — WP-A is the structural entry point. WP-B and WP-C depend on WP-A (the pipeline must exist before it can invoke planner and synthesizer). WP-D and WP-E have implied-logic dependencies on WP-A (their work landing without the pipeline has no host).

---

### WP-B: Routing Planner production ensemble + Routing Planner module (ADR-028)

**Objective:** Land the routing-planner system ensemble (`agentic-routing-planner.yaml`) as a production ensemble + the Routing Planner module that wraps invocation through `OrchestratorToolDispatch.dispatch()`. The 20-prompt regression battery from Spike ζ becomes the integration test fixture per FC-31.

**Changes:**
- New production ensemble YAML `.llm-orc/ensembles/agentic-serving/agentic-routing-planner.yaml` (promoted from `spike-cycle7-zeta-routing-planner.yaml` after Track A.1 refactor). `topaz_skill: tool_use` per ADR-028 §"Ensemble structure" (classification pragmatic, not load-bearing per ADR-028); cheap-tier model profile (qwen3:8b empirical baseline per Spike ζ).
- New module `agentic/routing_planner.py` owning the planner invocation surface — constructs `InternalToolCall(name="invoke_ensemble", arguments={"ensemble_name": "agentic-routing-planner", "input": <serialized REQUEST + CAPABILITY LIST>})`, calls `OrchestratorToolDispatch.dispatch()`, parses JSON output into typed `DispatchPlan(action, ensemble, input, rationale)` dataclass.
- 20-prompt regression battery (from Spike ζ) ported as integration test fixture.
- Bonus-path hooks (explicit-naming extractor pre-filter; `tool_choice` interception pre-filter) defined as optional pre-stage in Dispatch Pipeline — Cycle 7 ships the planner-only path; bonus paths layer in follow-on cycles per ADR-028.

**Scenarios covered:** scenarios.md §"Routing-Planner Ensemble (ADR-028)" — Explicit-naming NL request produces dispatch action; NL clear-match request produces dispatch; No-capability-match produces direct action; Adversarial/ambiguous request produces defensible-judgment dispatch; (integration) Routing-planner output feeds OrchestratorToolDispatch with real types; Preservation: planner operates within AS-9's structural-bounding property; Preservation: planner operates within AS-10's request-content-alone scope. Plus Track A.1 scenario: planner output schema includes `input` field.

**Participating modules:** Routing Planner (new), Orchestrator Tool Dispatch (existing — dispatch chokepoint), Capability List Builder (new — supplies CAPABILITY LIST input; ships in WP-D but the planner is testable in isolation against a stub list until WP-D's builder is wired).

**Dependencies:** Hard on **WP-A** (Dispatch Pipeline invokes the planner). Hard on **Track A.1** (planner spike YAML's output schema must include the `input` field before promotion). Implied logic with **WP-D** — Capability List Builder supplies CAPABILITY LIST input; the planner is testable with a stub list, but production wiring needs the builder.

---

### WP-C: Response Synthesizer production ensemble + Response Synthesizer module (ADR-029)

**Objective:** Land the response-synthesizer system ensemble (`agentic-response-synthesizer.yaml`) as a production ensemble + the Response Synthesizer module that wraps invocation. The 13-test + 4-confabulation-mode battery from Spike ε + ε' + μ becomes the integration test fixture per FC-32. Streaming support — synthesizer LLM streams tokens via the existing `OpenAiSseFormatter` per ADR-031 §"Streaming as a load-bearing surface".

**Changes:**
- New production ensemble YAML `.llm-orc/ensembles/agentic-serving/agentic-response-synthesizer.yaml` (promoted from `spike-cycle7-epsilon-response-synthesizer.yaml` after Track A.2 refactor adds Rule 6). `topaz_skill: summarization` per ADR-029 §"Neutral consequences"; cheap-tier model profile (qwen3:8b empirical baseline per Spike ε + ε' + μ); strict-fidelity Rules 1-6 in the ensemble's system prompt.
- New module `agentic/response_synthesizer.py` owning the synthesizer invocation surface — constructs `InternalToolCall` for the synthesizer ensemble; serializes structured input (`ORIGINAL REQUEST` + `PLAN` + `DISPATCH RESULTS`); calls `OrchestratorToolDispatch.dispatch()`; for streaming requests, dispatches in streaming mode and yields `ContentDelta` chunks.
- Calibration Gate Reflect-trigger criteria extension (per ADR-029 §"Calibration Gate integration"): Rule 5 framing absence on direct-completion responses; Rule 4 rounding-drift via runtime fidelity check (BUILD-phase mitigation playbook design); Rule 1 fabrication signal via post-hoc DISPATCH RESULTS-vs-content cross-check.
- Spike ε + Spike ε' + Spike μ battery (13 tests total + 4 confabulation modes) ported as integration test fixture (regression battery).
- Multi-turn continuity: pipeline serializes `messages[]` into the synthesizer's ORIGINAL REQUEST input per Spike ε' C1/C2.

**Scenarios covered:** scenarios.md §"Response-Synthesizer Ensemble (ADR-029)" — Synthesizer reads structured input; Rule 1 uses only DISPATCH RESULTS; Rule 2 reports Planned-but-not-run honestly; Rule 4 cites figures verbatim; Rule 5 honest direct-completion framing; Rule 6 framework-convention enumeration; Multi-turn continuity preserved; Calibration Gate Reflect fires on Rule 5 absence; Preservation: synthesizer prevents C4 substrate-path-file-read failure mode; Preservation: AS-7 amended summarization rules unchanged. Plus Track A.2 scenario: synthesizer YAML carries Rule 6.

**Participating modules:** Response Synthesizer (new), Orchestrator Tool Dispatch (existing — dispatch chokepoint), Calibration Gate (extended — three new Reflect-trigger criteria), `OpenAiSseFormatter` (existing — consumes synthesizer's `ContentDelta` chunks unchanged).

**Dependencies:** Hard on **WP-A** (Dispatch Pipeline invokes the synthesizer). Hard on **Track A.2** (synthesizer spike YAML must carry Rule 6 before promotion). Open choice with **WP-B** — planner and synthesizer can be built in either order once WP-A's pipeline is in place (the pipeline calls both modules independently).

---

### WP-D: Capability List Builder + Capability Discovery Endpoint (ADR-026 + ADR-028 + ADR-032)

**Objective:** Land the canonical Capability List Builder at L1 + the Capability Discovery Endpoint at L3. The builder produces the single source of truth for capability ensembles per AS-10 (no client-side opt-in mechanism per ADR-026); the discovery endpoint advertises capabilities via OpenAI-protocol-compatible mechanism per ADR-032 §Capability-list discovery.

**Changes:**
- New module `agentic/capability_list.py` owning the `CapabilityRegistry` Protocol + `CapabilityListBuilder.list_capabilities() -> list[CapabilityEntry]`. Filter logic: include ensembles declaring `output_substrate: artifact` + `topaz_skill` per ADR-019 + ADR-025; exclude system ensembles (`agentic-routing-planner`, `agentic-response-synthesizer`, `agentic-result-summarizer`, `agentic-calibration-checker`). New `CapabilityEntry` dataclass (`name`, `description`, `topaz_skill`).
- Concrete `CapabilityRegistry` implementation reads through the validate-once-at-load library cache established Cycle 6 (FC-27); existing `EnsembleLoader` cached `EnsembleConfig` collection is the data source.
- **BUILD picks one of three candidate surfaces per ADR-032 §Capability-list discovery** (Open Decision Point — see below): (a) extend `/v1/models` to list capability ensembles with a `type: "ensemble"` marker distinguishing them from model profiles; (b) add sibling endpoint `/v1/ensembles`; (c) response-metadata-only (capability list in chat-completion response metadata under a request flag). Recommended starting point: (a) `/v1/models` extension as lowest-cost candidate; (b) and (c) can layer later if operator evidence warrants.
- Dispatch Pipeline wires `CapabilityListBuilder.list_capabilities()` for the Routing Planner's CAPABILITY LIST input (per WP-A integration).

**Scenarios covered:** scenarios.md §"Honest Response Labeling and Capability-List Discovery (ADR-032)" — `/v1/models` advertises capability ensembles with topaz_skill metadata; Capability list updates reflect ensemble add/remove events. §"Capability Matching from Request Content Alone (ADR-026 / AS-10)" — Routing decision uses only request body + capability list; Population B accommodation via alternative surfaces, not via chat-completions opt-in; Preservation: `llm-orc invoke` CLI accepts explicit capability identifiers; Preservation: ADR-019 skill-framework-agnostic commitment unchanged.

**Participating modules:** Capability List Builder (new), Capability Discovery Endpoint (new), Ensemble Engine (existing — Protocol consumer reads cached registry per FC-27), Dispatch Pipeline (extended — calls builder for planner input — already covered in WP-A).

**Dependencies:** Open choice with **WP-B** and **WP-C** (capability list source is independent of planner/synthesizer modules). Hard on **WP-A** for Dispatch Pipeline integration (the pipeline wires the builder's output to the planner's input).

---

### WP-E: Response Labeling + `tool_choice` bridge + degradation event sink consumption (ADR-030 + ADR-032)

**Objective:** Land the three-layer honest response labeling mechanism at L3 (headers + body metadata; content-layer Rule 5 framing delegated to Response Synthesizer per WP-C) + the `tool_choice` bridge advisory per ADR-030 + the sink consumers for `DirectCompletionFallback` event including the `direct_completion_rate` rolling-window aggregator per ADR-032 §"Operator-observable degradation signaling".

**Changes:**
- New module `agentic/response_labeling.py` owning the three-layer decoration logic. Stateless decorator function `decorate(response_body, response_headers, plan, envelope, tool_choice_present) -> (response_body, response_headers)`. Headers: `X-LLM-Orc-Served-By: ensemble:<name> | direct | direct_fallback | tool_choice:<ensemble>` per ADR-032 §Sub-promise (1); `X-LLM-Orc-Tool-Choice-Handling: deferred` on `tool_choice`-bearing requests per ADR-030 §Bridge advisory specification. Body metadata: `metadata.served_by`, `metadata.tool_choice_handling`, `metadata.population_b_advisory`, `metadata.dispatch_failed` (when relevant).
- `_ChatCompletionsRequest` Pydantic model extension at `src/llm_orc/web/api/v1_chat_completions.py`: add `tool_choice: dict | str | None = None` field. Pydantic no longer silently strips the parameter; handler propagates it to Response Labeling for bridge advisory emission.
- Population B structured advisory content design: BUILD picks careful wording for the `metadata.population_b_advisory` field on `action: "direct"` responses with Population-B-style request patterns. The advisory is universally safe to send (Population A clients that don't surface metadata are unaffected per ADR-032).
- Operator-Terminal Event Sink extension: `isinstance(event, DirectCompletionFallback)` arm emitting line-oriented `key=value` log entry; rolling-window aggregator (`direct_completion_rate`) producing the metric over a configurable window (default 24-hour per ADR-032 §"Refutation threshold"); refutation-threshold detection (sustained `direct_completion_rate` > ~15pp above operator-estimated baseline) fires operator notification.
- Orchestrator-Context Event Sink extension: `isinstance(event, DirectCompletionFallback)` arm appending the event to the structured-observation block. Note (per ADR-027 + ARCHITECT disposition (a)): on the chat-completions surface, the orchestrator-context destination's consumer (`OrchestratorRuntime`) is structurally dormant — the sink consumes without error; the observation surface is reserved for future surfaces adopting `OrchestratorRuntime`.
- Three-layer signaling regression test: each response path (dispatch, direct, dispatch_fallback, tool_choice) produces expected three-layer signaling per FC-38; tool_choice bridge advisory conditional content-layer per FC-39.

**Scenarios covered:** scenarios.md §"`tool_choice` Disposition with Bridge Mechanism (ADR-030)" — Bridge advisory at three layers; `tool_choice: "auto"` treated as absent; Preservation: requests without `tool_choice` flow unchanged; Preservation: ADR-001 + ADR-011 ReAct model remains operative. §"Honest Response Labeling and Capability-List Discovery (ADR-032)" — Dispatch response declares served_by:ensemble at three layers; Direct-completion declares served_by:direct + Rule 5; Dispatch failure declares served_by:direct_fallback; action=direct emits direct_completion_fallback event; direct_completion_rate computable from events; Population B advisory at metadata layer; Preservation: OpenAI chat-completions API contract unchanged.

**Participating modules:** Response Labeling (new), Serving Layer (extended — Pydantic model adds `tool_choice` field; composes Response Labeling decoration), Operator-Terminal Event Sink (extended — `DirectCompletionFallback` consumer + rolling aggregator), Orchestrator-Context Event Sink (extended — `DirectCompletionFallback` isinstance arm), Response Synthesizer (existing from WP-C — content-layer Rule 5 framing).

**Dependencies:** Hard on **WP-A** (Response Labeling decorates pipeline output; the pipeline's `DispatchPlan` and `DispatchEnvelope` outputs drive the decoration). Hard on **WP-C** (content-layer Rule 5 framing is the synthesizer's commitment). Implied logic with **WP-D** (Population B advisory content; the same capability-list source the discovery endpoint exposes informs the advisory). Open choice with **WP-B** (Response Labeling does not depend directly on planner module internals).

---

## Dependency Graph (Cycle 7)

```
Track A.1 (planner schema input field refactor)
   │
   └─ hard ─▶ WP-B (Routing Planner promotion)

Track A.2 (synthesizer Rule 6 refactor)
   │
   └─ hard ─▶ WP-C (Response Synthesizer promotion)

WP-A (Dispatch Pipeline + adapter + pipeline-stage events)
   │
   ├─ hard ─▶ WP-B (pipeline invokes planner)
   ├─ hard ─▶ WP-C (pipeline invokes synthesizer)
   ├─ implied ─▶ WP-D (pipeline wires capability list to planner; builder works without pipeline but needs it for production integration)
   └─ hard ─▶ WP-E (Response Labeling decorates pipeline output; depends on DispatchPlan + DispatchEnvelope shapes the pipeline produces)

WP-C (Response Synthesizer)
   │
   └─ hard ─▶ WP-E (content-layer Rule 5 framing is the synthesizer's commitment)

WP-B ─ open choice with WP-C (planner and synthesizer build in either order once WP-A is in place)
WP-D ─ open choice with WP-B + WP-C (capability list source is independent of planner/synthesizer modules)
WP-D ─ implied with WP-E (Population B advisory content; the same capability list source informs the advisory)
```

**Classification key:**
- **Hard dependency:** structural necessity — the downstream WP's code imports, extends, or requires the upstream WP's output.
- **Implied logic:** suggested ordering — building the upstream first is simpler, but a skilled builder can stub the references.
- **Open choice:** genuinely independent — build either first.

---

## Transition States (Cycle 7)

### TS-10: Dispatch Pipeline operational on chat-completions surface (after WP-A + WP-B + WP-C)

When WP-A + WP-B + WP-C land, the framework-driven dispatch pipeline is operational on the chat-completions surface. Every request flows through plan → dispatch (when applicable) → synthesize; the orchestrator-LLM is removed from the routing-decision and post-dispatch-synthesis surfaces; AS-9 is satisfied universally on the chat-completions surface. This is the **central architectural pivot of Cycle 7** — the configuration the corpus has been engineering toward across Cycles 4-7. Capability list (WP-D) and honest response labeling (WP-E) can ship before or after; TS-10 is independently meaningful because the architectural state is observable from any request the operator sends.

### TS-11: Cycle 7 complete (after all 5 WPs)

All Cycle 7 BUILD deliverables shipped: Dispatch Pipeline + Plan→InternalToolCall adapter + pipeline-stage events; Routing Planner production ensemble + module; Response Synthesizer production ensemble + module; Capability List Builder + Capability Discovery Endpoint; Response Labeling + `tool_choice` bridge + degradation event sink consumption. The `direct_completion_rate` rolling metric is operational; honest response labeling fires three layers per response; bridge advisory addresses the AS-10 configuration-honesty footgun without blocking on follow-on cycle's full disposition (i) implementation per ADR-030. The 13 conformance findings from the Cycle 7 DECIDE scan are remediated (2 Track A refactors + 8 BUILD findings + 3 ARCHITECT-phase deferrals).

---

## Open Decision Points (Cycle 7)

### 1. Capability Discovery Endpoint surface choice (ADR-032 §Capability-list discovery)

ADR-032 names three candidate surfaces; BUILD picks one (or more). The trade-offs:

- **(a) `/v1/models` extension** — lowest-cost candidate; reuses existing endpoint. Capability ensembles appear alongside model profiles with a `type: "ensemble"` marker. Risk: clients that strictly enforce OpenAI schema may reject the `type` field on response entries; OpenAI's actual response shape includes other free-form fields so this is unlikely but possible.
- **(b) Sibling endpoint `/v1/ensembles`** — dedicated endpoint; new HTTP surface. Cleaner separation; capability ensembles' metadata can be richer (`calibration status`, `topaz_skill`, ensemble-specific properties). Cost: new endpoint + client documentation; clients have to know about it.
- **(c) Response metadata** — capability list in chat-completion response metadata under a request flag. Lowest bandwidth; relevant for clients that don't pre-discover. Cost: requires per-request flag detection; the discovery surface is post-hoc.

Recommended starting point: **(a)** as lowest-cost. (b) and (c) can layer later as deployment evidence warrants — multiple surfaces may coexist.

### 2. `direct_completion_rate` refutation-threshold default

ADR-032 §"Refutation threshold for the cost-distribution accountability sub-promise" names a starting heuristic — sustained `direct_completion_rate` > ~15 percentage points above operator-estimated baseline over a 24-hour rolling window. The threshold is a starting point; production evidence calibrates. BUILD picks the default + makes the threshold operator-configurable. Open decision: should the default fire an operator-notification at threshold (push) or only surface in dashboard logs (pull)?

### 3. Population B advisory wording

ADR-032 §"Mechanism: Structured advisory for Population B" names that the advisory is universally safe to send but BUILD picks the wording. The advisory must (a) be useful to Population B (clear pointer to `llm-orc invoke` and direct ensemble HTTP API), (b) be non-confusing to Population A users whose deployment configured the chat-completions surface for capability dispatch — Population A users may not need or want the advisory rendered, but it appears in metadata they don't necessarily surface. BUILD picks careful wording.

### 4. Multi-step composition mechanism (OQ #21)

Cycle 7 BUILD adopts single-step planner + framework chain-heuristic per Spike δ pattern as default per ADR-027 §"Open as downstream-phase design questions". Production traffic diversity may surface composition shapes the heuristic does not handle; BUILD/PLAY characterization informs whether multi-step planner output or planner-loops-with-context architecture warrants. Open decision deferred to BUILD/PLAY findings.

### 5. Rounding-drift mitigation playbook (OQ #24)

ADR-029 §"Rounding-drift mitigation playbook" names the three-mechanism hierarchy (system-prompt sharpening → tier escalation → runtime fidelity check). BUILD's initial default is system-prompt sharpening + tier escalation per Calibration Gate Reflect; runtime fidelity check is the load-bearing fallback if system-prompt + tier-escalation does not bound drift to acceptable rates. The fidelity check threshold (exact match vs. tolerance for last-digit rounding vs. semantic equivalence) is BUILD-phase design. Open decision deferred to BUILD signal.

### 6. Operator-deployment tier escalation policy for direct-completion (ADR-031 §"Tier escalation policy for direct-completion")

Cheap-tier qwen3:8b has documented domain-specific training-data error patterns (Spike ε' A2 "Urga / Khovd" data point); operators with reasoning-heavy direct-completion patterns may want a higher-tier escalation target configured. BUILD ships the default cheap-tier behavior with operator-configurable escalation target; operators bear the configuration burden for non-obvious failure surfaces. Open decision: does the default operator-config include a recommended escalation target (e.g., `agentic-tier-escalated-general` profile) or stay empty?

---

## Work Packages — Cycle 6 (status update at Cycle 7 entry)

> **Cycle 6 BUILD shipped 3 of 5 WPs.** WP-A, WP-B, WP-D closed (see commits referenced inline below — sketches preserved here pending migration to Completed Work Log).
> **Carry-forward to Cycle 7:** WP-C (Orchestrator-Context Event Sink) — surface defined per ADR-023; under ADR-027 + ARCHITECT disposition (a) the orchestrator-context destination's consumer (`OrchestratorRuntime`) is structurally dormant on chat-completions; the sink consumes without error; observation surface reserved for future surfaces. Production work to fully wire the sink remains a BUILD opportunity if a future surface adopts `OrchestratorRuntime`.
> **Carry-forward to Cycle 7:** WP-E (Session Artifact Store + AS-7 amendment + ADR-022 system-prompt amendment) — partially superseded. ADR-022 system-prompt amendment is structurally moot for chat-completions under ADR-027 (per ADR-022 partial-update header + ADR-027 §"Relationship to ADR-022"). Session Artifact Store + AS-7 amendment in code remain meaningful BUILD work for capability ensembles' substrate routing — these survive Cycle 7's pivot and warrant BUILD continuation per the capability-ensemble substrate-routing scope per ADR-025.

### WP-A: Dispatch Event Substrate + `dispatch_id` correlation (ADR-023 emission substrate) — ✅ **Closed 2026-05-15** (commits `b944f0f` refactor + `0dc8b7f` feat)

> **Cycle 6 BUILD shapes as 5 work packages (WP-A through WP-E).** Identifiers reset for the new active cycle per the methodology — Cycle 4 BUILD's WP-A4..WP-H4 closures and Cycle 5's library-reshape work are documented in system-design.md Amendment Log entries #7 (2026-05-08) and #8 (2026-05-12); the Cycle 4 (deferred) section below preserves WP-K and WP-J as deferred carry-forwards.
>
> **Cycle 6 BUILD-mode declaration**: to be set at BUILD entry per ADR-091 (gated recommended given the design-alternative examination character of ADR-022 routing-surface work + the cross-surface `dispatch_id` coupling per DECIDE snapshot Finding 2 advisory; auto mode appropriate only if BUILD reduces to mechanical wiring after the four-module structural shape is in place).

### WP-A: Dispatch Event Substrate + `dispatch_id` correlation (ADR-023 emission substrate) — ✅ **Closed 2026-05-15** (commits `b944f0f` refactor + `0dc8b7f` feat)

**Carry-forward to WP-B:** `CalibrationSignal` substrate emission deferred from WP-A producer migration. The signal originates at L0 (Ensemble Engine dispatch outputs) and flows through `CalibrationSignalChannel`'s read-only L0→L1 channel per ADR-016; wiring substrate emission requires channel-side plumbing that scopes naturally with WP-B's sink work. WP-B should land `CalibrationSignalChannel.record_signal(...)` substrate emission (with the additive `dispatch_id` field already on `CalibrationSignal`) so the operator-terminal sink observes calibration signals at DEBUG level per ADR-023 §Destination 1.

**Objective:** Land the unified event-emission substrate per Inversion N+2 — one substrate that fans out to two destinations. Add `DispatchTiming` event type + `dispatch_id` correlation identifier across the four existing event types.

**Changes:**
- New module `agentic/dispatch_event_substrate.py` with `DispatchTiming` event type (`phase: Literal["start","end"]`, `dispatch_id`, `ensemble_name`, `model_profile`, `timestamp`; `phase="end"` adds `duration_seconds` + `exit_status`); `new_dispatch_id(session_id) -> str` generator; `emit(event)` fan-out dispatcher; `register_sink(sink_instance)` registration; `events_for(dispatch_id) -> list[Event]` post-hoc query
- Extend `TierSelection`, `CalibrationVerdict`, `AuditDiagnostic`, `CalibrationSignal` with optional `dispatch_id: str | None` field (additive, `None` allowed during transition)
- Producer-side migration — Calibration Gate, Tier-Escalation Router, Tier-Router-Audit, Calibration Signal Channel emit through `dispatch_event_substrate.emit()`
- Orchestrator Tool Dispatch — call `new_dispatch_id` at `invoke_ensemble` entry; emit `DispatchTiming(phase="start")` before tier selection; emit `DispatchTiming(phase="end")` after harness/substrate-write; substrate-emit happens regardless of sink registration

**Scenarios covered:** scenarios.md §Observability Event Routing — `DispatchTiming` event carries start and end phases; `dispatch_id` correlation joins events; preservation scenarios for ADR-018 / ADR-016 (additive `dispatch_id` does not change semantics).

**Participating modules:** Dispatch Event Substrate (new), Calibration Gate (extended), Tier-Escalation Router (extended), Tier-Router-Audit (extended), Calibration Signal Channel (extended), Orchestrator Tool Dispatch (extended).

**Dependencies:** Hard on **WP-A4** (typed errors substrate from Cycle 4 BUILD — already shipped).

---

### WP-B: Operator-Terminal Event Sink + Liveness signals + Validate-once-at-load (ADR-023 destination 1) — ✅ **Closed 2026-05-15** (commits `62dccbf` piece 1 + `cb9142a` piece 2 + `6bad830` fix + `f1a2d16` piece 3 + `bb04dbb` piece 4 + `8fe63b0` piece 5)

**Status:**

- ✅ **Piece 1 — Operator-Terminal Event Sink module** (commit `62dccbf` + `6bad830` fix) — new L3 module `agentic/operator_terminal_event_sink.py` with per-event format strings + log-level discrimination (`CalibrationSignal` at DEBUG, others at INFO) + action surfaces (`emit_tool_call_log`, `emit_heartbeat`, `emit_validation_warning`); registers with substrate via `register_with()`; 13 unit tests
- ✅ **Piece 2 — CalibrationSignal substrate emission** (commit `cb9142a`) — closes the WP-A carry-forward; `CalibrationSignalChannel.record_signal` emits validated signals through optional `event_substrate`; 3 unit tests
- ✅ **Piece 3 — Validate-once-at-load** (commit `f1a2d16`) — `EnsembleLoader` gains `prime(directory)`, `reload(directory)`, `validation_results()`; per-directory cache keyed by resolved path; un-primed callers (CLI / MCP) keep on-demand validation with `Skipping invalid ensemble` log line preserved; `OperatorTerminalEventSink.report_validation_results(results)` drains failures into one `WARN` per invalid YAML; 11 unit tests for the loader, 2 sink tests
- ✅ **Piece 4 — Tool-call-emit logging** (commit `bb04dbb`) — `OrchestratorToolDispatch` gains optional `ToolCallEmitLogger` Protocol slot; the logger fires `emit_tool_call_log(tool_name="invoke_ensemble", dispatch_id=...)` between `new_dispatch_id` allocation and `DispatchTiming(start)` emission inside `_open_dispatch_event` — the FC-23 chronological-ordering property. L2 declares the Protocol locally so it does not import the L3 sink (FC-4 layering preserved); 5 unit tests + FC-23 integration anchor at `tests/integration/test_tool_call_emit_log_precedes_dispatch_start.py`
- ✅ **Piece 5 — Inference-wait heartbeat + serve-layer wiring** (commit `8fe63b0`) — new L3 module `agentic/inference_wait_heartbeat.py` implements both `EventSink` (substrate `DispatchTiming` events filtered by session_id) and `ToolCallEmitLogger` (forwards to sink, resets activity timer); `DispatchEventSubstrate.unregister_sink` added for per-request scheduler lifecycle; `OrchestratorConfig.observability.heartbeat_interval_seconds` (default 30s); `chat_completions` constructs a per-request scheduler, registers it with the substrate, starts an asyncio task, and cancels+unregisters in try/finally around both streaming and non-streaming response paths; serve-layer factories `get_dispatch_event_substrate()` and `get_operator_terminal_event_sink()` wire piece 3's prime+report flow at first construction and pass substrate+sink to `OrchestratorToolDispatch`; 10 unit tests + FC-27 integration anchor at `tests/integration/test_validate_once_at_load.py`

**Cycle Acceptance Criterion (Step 5.5).** The WP-B-applicable criterion *"Liveness signals fire during in-flight states before completion events"* (Layer-match `no` in the Cycle 6 acceptance criteria table) is verified at the layers BUILD can address: the FC-23 anchor verifies the tool-call-emit-precedes-dispatch-start chronological ordering against the real operator-terminal sink + real dispatch event substrate composition; the unit-level heartbeat scheduler tests verify the heartbeat-interval-counter logic under a controllable clock. The live timed-dispatch leg of the fitness (heartbeat fires under a >30s real-clock cloud-LLM inference wait) is appropriately PLAY-phase observational territory.

**Objective:** Land the operator-terminal destination as a registered sink consuming from the Dispatch Event Substrate. Ship the line-oriented log surface, tool-call-emit logging, inference-wait heartbeats, and the noise-floor remediation (validate-once-at-load).

**Changes:**
- New module `agentic/operator_terminal_event_sink.py` with per-event format strings (one line per event type at INFO; `CalibrationSignal` at DEBUG); registers with Dispatch Event Substrate at serve startup
- Tool-call-emit logging: Serving Layer detects `invoke_ensemble` tool-call structure in the SSE stream and emits `INFO: tool-call emit: tool=invoke_ensemble dispatch_id=<id>` before dispatching
- Inference-wait heartbeat: open-request inactivity timer in Serving Layer fires `INFO: inference wait: elapsed=<seconds> session_id=<id>` after `heartbeat_interval_seconds` (default 30s) and recurs at the interval while inactivity continues
- Library validation moves to startup (or library reload via `SIGHUP` / admin endpoint / restart) — `EnsembleLoader.load_from_file` invoked once per ensemble at startup; `list_ensembles()` returns the cached validated subset without re-validation
- Operator-Terminal Event Sink consumes Ensemble Engine's `validation_results()` at startup and emits one `WARN` line per invalid YAML
- The existing `INFO: tool dispatch: result kind=success` line is **replaced** by the new per-event lines carrying ensemble identification, duration, verdict, and `dispatch_id` correlation

**Scenarios covered:** scenarios.md §Observability Event Routing — Operator-terminal per-event INFO lines; Tool-call-emit log precedes dispatch; Inference-wait heartbeat fires after `heartbeat_interval_seconds`; Validate-once-at-load eliminates per-enumeration noise; preservation scenarios for ADR-019 library schema.

**Participating modules:** Operator-Terminal Event Sink (new), Serving Layer (extended for tool-call-emit + heartbeat timer), Ensemble Engine L0 (extended for validate-once-at-load library cache).

**Dependencies:** Hard on **WP-A** (sink consumes from substrate; sink registration requires substrate's `register_sink()` API). Open choice with **WP-C** at architecture level (the two sinks are independent consumers of the substrate).

---

### WP-C: Orchestrator-Context Event Sink (ADR-023 destination 2)

**Objective:** Land the orchestrator-context destination as a registered sink — structured observations between turns + end-of-session summary.

**Changes:**
- New module `agentic/orchestrator_context_event_sink.py` with structured-observation construction (JSON-shaped block per `dispatch_id`); `consume(event)` (substrate-registered) + `observations_for(dispatch_id) -> Observation` (query surface for Runtime); end-of-session summary writing the `dispatch_log` key to `execution.json`
- `CalibrationSignal` excluded from orchestrator-context routing by default; operator opt-in via `agentic_serving.observability.orchestrator_context_routes_calibration_signal: true`
- Orchestrator Runtime extension: at each turn boundary (after a dispatch returns control and before assembling the next turn's context), call `orchestrator_context_event_sink.observations_for(last_dispatch_id)` and prepend the returned observation block to the next turn's messages
- Session Registry → Orchestrator-Context Event Sink: end-of-session hook triggers the `dispatch_log` write to the session's `execution.json`
- Final-dispatch-before-session-close handling: in-turn routing is skipped (no next turn); end-of-session summary captures the events

**Scenarios covered:** scenarios.md §Observability Event Routing — Orchestrator-context destination prepends structured observation between turns; Final dispatch routes to end-of-session summary; preservation scenarios for `execution.json` existing fields.

**Participating modules:** Orchestrator-Context Event Sink (new), Orchestrator Runtime (extended for turn-boundary query), Session Registry (extended for end-of-session callback to write `dispatch_log`).

**Dependencies:** Hard on **WP-A**. Open choice with **WP-B** (sinks are mutually independent).

---

### WP-D: ADR-024 typed `DispatchEnvelope` + `output_schema:` per-ensemble declaration — ✅ **Closed 2026-05-16** (commits `18a3230` refactor + `98696a4` feat)

**Objective:** Codify the `invoke_ensemble` response shape as the typed `DispatchEnvelope` dataclass; add optional per-ensemble `output_schema:` declaration; populate `structured` payload when declared.

**Changes:**
- New shared type `agentic/dispatch_envelope.py` with `DispatchEnvelope` frozen dataclass (`status`, `primary`, `structured?`, `diagnostics`, `errors?`, `artifacts?`) — lives alongside `LlmOrcStructuralError`
- Orchestrator Tool Dispatch: `invoke_ensemble` returns `DispatchEnvelope` instead of the existing `ToolCallResult` shape; envelope construction reads dispatch events from Dispatch Event Substrate via `events_for(dispatch_id)` to populate `diagnostics`
- Ensemble Engine L0 extension: optional `output_schema: dict | None` YAML field on `EnsembleConfig`; when declared, the synthesizer agent (or post-dispatch processing) populates `envelope.structured` with the typed payload
- Migration of representative ensembles to declare `output_schema:` — recommended starters: `claim-extractor`, `text-summarizer`, `web-searcher` (their `default_task` already specifies structured output)
- The Cycle 5 `execution.json` artifact retains `metadata` field name; the rename to `diagnostics` is at the envelope layer only (Cycle 7+ artifact-shape ADR territory)

**Scenarios covered:** scenarios.md §Common I/O Envelope — `invoke_ensemble` returns typed `DispatchEnvelope`; `output_schema:` populates `envelope.structured`; capability ensemble without `output_schema:` produces `envelope.structured = None`; `errors[]` populated on partial-failure; `diagnostics.dispatch_id` correlates envelope to ADR-023 events; preservation scenarios for ADR-021 + `execution.json`.

**Participating modules:** Orchestrator Tool Dispatch (extended, owns the shared type as producer), Ensemble Engine L0 (extended for `output_schema:` field), Dispatch Event Substrate (envelope construction reads events).

**Dependencies:** Hard on **WP-A** (envelope's `diagnostics` populated from substrate events). Open choice with **WP-B** + **WP-C** (envelope construction does not depend on sink registration; sinks observe the events the substrate emits).

---

### WP-E: ADR-022 system-prompt amendment + ADR-025 Session Artifact Store + AS-7 amendment

**Objective:** Land the routing-surface intervention (ADR-022 system-prompt amendment) **plus** the artifact-as-substrate always-scope (ADR-025) **plus** the AS-7 amendment that ties them together. Capability ensembles substrate-route; system ensembles remain inline; the result-summarizer ensemble is not invoked for substrate-routed dispatches.

**Changes:**
- ADR-022 system-prompt amendment in `agentic/orchestrator_config.py::DEFAULT_ORCHESTRATOR_SYSTEM_PROMPT` — insert the new paragraph between the existing "Do not pick a client-declared tool for questions about llm-orc's own state" paragraph and the "When you need a client-declared tool, emit it alone in a single assistant turn" paragraph; verbatim per ADR-022 §"Amendment to the system prompt"
- New module `agentic/session_artifact_store.py` owning the `.llm-orc/agentic-sessions/<session_id>/<dispatch_id>/<deliverable>.<ext>` path layout; `<session_id>` format (`<iso-8601-datetime>-<short-uuid>`); retention semantics enforcement (`session` / `durable` / `ephemeral`); `write_deliverable(dispatch_id, deliverable_name, content, content_type) -> ArtifactReference` API
- Ensemble Engine L0 extension: optional `output_substrate: Literal["artifact","inline"] | None` (default — capability ensembles `artifact`, system ensembles `inline`); optional `output_retention` field (default `session`); optional `calibration_substrate_access` field (default `summary`)
- Orchestrator Tool Dispatch substrate-routing path: when dispatched ensemble's `output_substrate == "artifact"`, write deliverable via Session Artifact Store; construct `envelope.primary` as summary line; `envelope.artifacts[0]` as typed `ArtifactReference`; **skip Result Summarizer Harness** per AS-7 amended
- Result Summarizer Harness — no harness-internal changes; the skip is enforced at Tool Dispatch's interposition order; ADR-004's `raw_output=True` escape hatch operates unchanged within the inline-response scope
- Calibration Gate extension: three evaluation surfaces — summary-only (default — critic agents receive `envelope.primary` + `artifacts[0].summary`); `structured`-augmented (when `output_schema:` declared); artifact-content (when `calibration_substrate_access: artifact` declared — critic agents receive an `ArtifactReadTool`)
- Session Registry → Session Artifact Store: `on_session_close(session_id)` callback for `retention: session` cleanup; orphaned-session warnings at next startup
- Capability ensemble migration: 6 capability ensembles default `output_substrate: artifact` in Cycle 6 BUILD (with `code-generator` declaring `calibration_substrate_access: artifact`); `web-searcher` migrated **early** per DECIDE snapshot Finding 1 advisory to expose Indicator 1 (latency overhead) and Indicator 4 (opt-out count)
- Backward-deprecated `.llm-orc/artifacts/agentic-serving/<ensemble>/<timestamp>/` path: new dispatches write under the new structure; the old tree is deprecated but not actively removed
- AS-7 amendment: already recorded in `domain-model.md` Amendment Log entry #11 at DECIDE close; this WP operationalizes the amendment in code

**Scenarios covered:** scenarios.md §Routing Surface Behavior — NL request matching capability ensemble dispatches via `invoke_ensemble`; Client-tool verb-match does not displace capability-match; Direct completion residual when no capability match; ADR-022 effectiveness is per-orchestrator-profile-conditional; preservation scenarios for ADR-021 + ADR-003. scenarios.md §Artifact-as-Substrate — Capability ensemble writes deliverable to session-dir artifact path; System ensemble produces inline-response envelope; Substrate-routed dispatch's envelope is not passed through `agentic-result-summarizer`; Inline-response dispatch retains `agentic-result-summarizer`; Calibration gate evaluators receive `primary` + `artifacts[0].summary` by default; Calibration gate reads artifact content for `code-generator` (opt-in); Session-close cleanup removes `retention: session` artifacts; Dial-back falsification indicator fires; preservation scenarios for ADR-007 / ADR-014 / ADR-004 / ADR-006.

**Participating modules:** Session Artifact Store (new), Orchestrator Configuration (extended), Orchestrator Tool Dispatch (extended — substrate-routing path + envelope construction), Result Summarizer Harness (extended — substrate-conditional skip enforced upstream), Calibration Gate (extended — three evaluation surfaces), Ensemble Engine L0 (extended — three new YAML fields), Session Registry (extended — `on_session_close` callback for substrate cleanup).

**Dependencies:** Hard on **WP-A** (uses `dispatch_id` for path construction). Hard on **WP-D** (envelope shape is the substrate's structural home for `artifacts[]`). Implied logic with **WP-B + WP-C** — substrate-routed dispatches still emit events through the substrate; observability operates with or without artifact-routing, so WP-B + WP-C can ship first or after WP-E.

**Cross-surface `dispatch_id` consistency** (per DECIDE snapshot Finding 2 advisory): BUILD includes `test_dispatch_id_consistency_across_events_envelope_artifact_path` as an integration test asserting the same `dispatch_id` value across the event stream, the envelope's `diagnostics.dispatch_id`, and the artifact path's `<dispatch_id>` segment. FC-22 is the verification anchor.

---

## Dependency Graph (Cycle 6)

```
WP-A (Dispatch Event Substrate + dispatch_id correlation)
   │
   ├─ hard ─▶ WP-B (Operator-Terminal Event Sink + liveness + validate-once-at-load)
   ├─ hard ─▶ WP-C (Orchestrator-Context Event Sink)
   ├─ hard ─▶ WP-D (DispatchEnvelope + output_schema)
   └─ hard ─▶ WP-E (Session Artifact Store + AS-7 amendment + ADR-022 system-prompt amendment)

WP-D (DispatchEnvelope)
   │
   └─ hard ─▶ WP-E (envelope's artifacts[] is substrate's structural home)

WP-B ─ open choice with WP-C (mutually independent — both consume from substrate)
WP-B ─ open choice with WP-D (envelope construction does not depend on sink registration)
WP-C ─ open choice with WP-D (same rationale)
WP-B + WP-C ─ implied with WP-E (substrate-routed dispatches still emit events; observability operates independently — but operators benefit from observability while migrating ensembles to substrate-routing)
```

**Classification key:**
- **Hard dependency:** structural necessity — the downstream WP's code imports, extends, or requires the upstream WP's output.
- **Implied logic:** suggested ordering — building the upstream first is simpler, but a skilled builder can stub the references.
- **Open choice:** genuinely independent — build either first.

---

## Transition States (Cycle 6)

### TS-8: Dispatch event substrate operational with both destinations (after WP-A + WP-B + WP-C)

The unified event-emission substrate fans out to operator-terminal and orchestrator-context destinations; operators see per-event log lines with `dispatch_id` correlation; the orchestrator's reasoning surface receives structured observations between turns answering PLAY note 12's load-bearing dispatch-graph questions; liveness signals fire during in-flight states; validate-once-at-load eliminates per-enumeration noise. The typed envelope and substrate-routing have not yet shipped; `invoke_ensemble` continues to return its v3.0 shape. This is a coherent intermediate where the Cycle 6 observability story stands on its own.

### TS-9: Cycle 6 complete (after all 5 WPs)

Typed `DispatchEnvelope` is the `invoke_ensemble` response shape; capability ensembles substrate-route; system ensembles remain inline; AS-7 amendment operates — `agentic-result-summarizer` skipped for substrate-routed dispatches, mandatory for inline. The ADR-022 system-prompt amendment is active; capability-matched NL framing routes to `invoke_ensemble` under the MiniMax M2.5-free profile (effectiveness under other profiles characterized at post-BUILD PLAY per ADR-022 disposition (iii)). The cycle's structural ship is complete; per-profile effectiveness, dial-back falsification indicators (ADR-025), and qwen3:14b over-delegation remain PLAY-phase observational concerns.

---

## Open Decision Points (Cycle 6)

**C6-1. `dispatch_id` generation strategy** — monotonic counter (per-session) vs. UUID4. ADR-023 §"Event-emission substrate" leaves this implementation-level. Default for BUILD: monotonic counter (simpler; observably ordered; aligns with the `<dispatch_id>` filesystem path's lexicographic-sortability for operator review). Affects WP-A.

**C6-2. Heartbeat timer mechanism** — separate thread vs. async background task. The serving layer's existing async/await structure may make async background task the natural choice; thread-based avoids interaction with the serving layer's request loop. Default: async background task tied to the open-request lifetime; auto-cancelled on request close. Affects WP-B.

**C6-3. `web-searcher` migration sequencing** — per DECIDE snapshot Finding 1 advisory carry-forward: migrate `web-searcher` to `output_substrate: artifact` **early** in WP-E's per-ensemble migration so that Indicator 1 (latency overhead for deliverables under 1 KB) and Indicator 4 (`output_substrate: inline` opt-out count) are testable before the migration commits. The recommendation is structural (sequencing within WP-E), not a separate decision point — but flagged here so BUILD does not deprioritize it.

**C6-4. Orchestrator-context routing default** — ADR-023 specifies default `enabled` for `agentic_serving.observability.orchestrator_context_routing`. Context-budget impact on long-dispatch-count sessions is bounded by ADR-012 compaction but operationally untested. Default holds; operators with strict context budgets may disable via config. Revisit at post-BUILD PLAY if context-rot patterns surface.

**C6-5. Old artifact path (`.llm-orc/artifacts/agentic-serving/<ensemble>/<timestamp>/`) deprecation pace** — ADR-025 declares the old tree deprecated-but-not-actively-removed. BUILD does not ship cleanup tooling for the old tree. Operators decide when to remove the old tree manually; the `llm-orc agentic-sessions prune` command (mentioned out-of-scope in ADR-025) is operator-tooling territory for a follow-on cycle.

**C6-6. ADR-022 effectiveness across orchestrator profiles** — ADR-022 disposition (iii) defers per-profile characterization to BUILD or follow-on PLAY. The cycle's post-BUILD PLAY re-runs the spike γ probe across at least Cells A (MiniMax M2.5-free) and B (qwen3:14b local via `agentic-orchestrator-offline-tools`) with the amended prompt active; if qwen3:14b continues to over-delegate, per-profile system-prompt overrides become Cycle 7+ territory.

---

## Work Packages — Cycle 4 (deferred carry-forwards)

> **Closed Cycle 4 work (WP-A4 through WP-H4) is recorded in `system-design.md` Amendment Log entries #7 (Cycle 4 ARCHITECT, 2026-05-08) and Cycle 4 BUILD subsequent closure dates.** The two carry-forwards below remain deferred from Cycle 1.

> **Cycle 1 WPs (WP-A through WP-I) are complete and migrated to the Completed Work Log.** TS-1 (stateless orchestrator serving OpenCode) reached at WP-F close (2026-04-22); TS-2 (stateless baseline) reached at WP-H close (2026-04-24); Plexus Adapter skeleton landed at WP-I close (2026-04-24). The active section below lists Cycle 4 work plus deferred Cycle 1 work (WP-K, WP-J).
>
> **Cycle 4 BUILD comprises eight new WPs (WP-A4 through WP-H4)** integrating ADRs 012–017 into the codebase per the conformance-scan-recommended sequence. Identifiers reset for the new active cycle (per skill methodology: "Reset identifiers for the next active cycle — don't accumulate escalating letters across cycles").

### WP-A4: Shared `LlmOrcStructuralError` base class — *T1 prerequisite* — ✅ **Closed 2026-05-11** — see Completed Work Log

---

### WP-B4: FC-2 automated import layering check + FC-3 automated cycle detection — *T1 prerequisite* — ✅ **Closed 2026-05-11** — see Completed Work Log

---

### WP-C4: ADR-017 — Tool-call structural validation guard — ✅ **Closed 2026-05-11** — see Completed Work Log

---

### WP-D4: ADR-013 — Session Registry structured-handoff artifacts + write-gate validation + cluster determination

**Objective:** Extend Session Registry with the three adoption-derived components (feature-list, append-only progress log, init.sh deterministic bootstrap) plus the novel-design write-gate validation surface plus cluster determination at session-start.

**Changes:**
- Extend `session_registry.py` with `StructuredHandoffArtifactSet` dataclass and `cluster: Cluster` field on `SessionState`
- New module `session_artifacts.py` (or sub-module of session_registry) with the three write-gate validation classes — JSON schema validation for feature-list, append-only constraint enforcement for progress-log, signed-script integrity verification for init.sh
- Hash-rotation workflow tooling (CLI command or library function for operators)
- Cluster determination at session-start with disposition (i) default-to-required for cross-cluster ambiguity
- New `write_gate_rejection` typed error (uses `LlmOrcStructuralError` base)
- Session-start hook updated to invoke init.sh with hash verification

**Scenarios covered:** scenarios.md §Session Registry Initializer-then-Resume — all 8 scenarios (Cluster 2 activates artifact set; Cluster 1 opts out; monotonic passes constraint; append-only rejection; init.sh hash mismatch; operator hash rotation; cross-cluster session defaults; preservation of existing identification).

**Participating modules:** Session Registry (extended), session_artifacts (new sub-module), models/structural_errors (uses base class), CLI for hash rotation.

**Dependencies:** Hard on **WP-A4** (uses `LlmOrcStructuralError`). Open choice with **WP-C4** — mutually independent.

---

### WP-E4: ADR-012 — Conversation Compaction five-layer pipeline

**Objective:** Land the cheapest-first compaction pipeline as a new L2 module the Runtime invokes at turn boundaries.

**Changes:**
- New module `conversation_compaction.py` with the five layers, four thresholds, circuit-breaker state, nine-section session-notes template
- Layer 0 filesystem persistence (operator-configurable root)
- Layer 4 invokes a configured summarizer ensemble via `EnsembleExecutor.execute` (distinct from Result Summarizer Harness)
- Circuit-breaker auto-reset at session start
- New `compaction_layer_4_failure` typed error with `recovery_action_required="operator_intervention_required"`
- Extend `OrchestratorConfig` with the four threshold defaults
- Extend `orchestrator_runtime.py` to invoke compaction at turn boundaries (FC-4 amendment: `conversation_compaction` is added to the Runtime's allowed-import set)
- New default `agentic-context-summarizer.yaml` ensemble (Layer 4 summarizer, distinct from `agentic-result-summarizer.yaml`)

**Scenarios covered:** scenarios.md §Conversation Compaction Five-Layer Pipeline — all 8 scenarios.

**Participating modules:** Orchestrator Runtime (extended), Conversation Compaction (new), models/structural_errors (uses base class), Ensemble Engine (Layer 4 invocation), filesystem.

**Dependencies:** Hard on **WP-A4** (uses `LlmOrcStructuralError`). Open choice with **WP-D4** at the architecture level (Layer 3 session-notes template stays in-memory by default; storage-coupling is a BUILD-time decision per Open Decision Point #1 below).

---

### WP-F4: ADR-014 — Calibration Gate trajectory-level extension

**Objective:** Extend the existing Calibration Gate with in-process trajectory-level calibration (AUQ + HTC + verdict trichotomy + time-decay windowing).

**Changes:**
- Extend `calibration_gate.py` with verdict trichotomy (`Proceed | Reflect | Abstain`)
- AUQ verbalized-confidence consumption (System 1 attention-soft, System 2 binary gate at default 0.85 within 0.8–1.0 range)
- HTC trajectory feature extraction (token-level entropy, attention-weight distributions, decision-confidence trajectories)
- Three Abstain criteria (entropy collapse > 1.5σ; post-hoc result-check hard failure; multiple drift criteria simultaneously exceeding thresholds)
- Time-decay windowing (60-min/100-signal dual-bound, linear decay)
- New `calibration_abstain` typed error
- Verdict surface published via `verdict_for(session_id, ensemble_name, dispatch_context)`
- Conditional signal-channel consumption: if Calibration Signal Channel is registered, gate reads windowed signals; otherwise operates on L1-internal trajectory data only

**Scenarios covered:** scenarios.md §Calibration Verdict Trichotomy — all 6 scenarios.

**Participating modules:** Calibration Gate (extended), models/structural_errors.

**Dependencies:** Hard on **WP-A4** (typed errors). Implied ordering before **WP-G4** (Tier-Escalation Router consumes the verdict surface) — a skilled builder could stub the verdict surface and run G4 in parallel.

---

### WP-G4: ADR-015 + ADR-018 — Per-role tier-escalation router + (d)-analog audit dispatch

**Objective:** Land the new Tier-Escalation Router L2 module + Topaz skill metadata schema migration on existing library ensembles + per-skill tier-defaults configuration. **Per ADR-018 (added at architect-gate close 2026-05-11):** also land the (d)-analog audit dispatch — periodic out-of-band audit on the verdict→router edge analogous to ADR-016 mechanism (d).

**Changes (WP-G4-1 — core router, per ADR-015):**
- New module `tier_router.py` with verdict→tier mapping (Proceed → cheap; Reflect → escalated; Abstain → `escalation_bypass`)
- Extend `EnsembleConfig` with optional `topaz_skill: TopazSkill` field
- One-time migration: add `topaz_skill` field to all existing ensembles in `.llm-orc/ensembles/*.yaml` (Spike α 2026-05-11 confirmed all classifiable ensembles have a clean primary — the migration's classification choices are operator judgment; the 21-of-21 spike result validates that operator-authored classifications will not produce systemically-ambiguous primaries)
- Extend `OrchestratorConfig` with `per_skill_tier_defaults` configuration surface (8 skills × 2 tiers)
- Tool Dispatch interposition: route `invoke_ensemble` through tier_router before `EnsembleExecutor.execute`
- New `escalation_bypass` and `missing_skill_metadata` typed errors
- **Operator documentation:** the WP-G4 operator-facing docs should surface ADR-015 §Consequences §Negative's coverage hedge as load-bearing (Spike α distribution finding: 4 actively-used Topaz skills + 3 single-instance + `mathematical_reasoning` unused on the existing library; operators may legitimately collapse unused skill slots to shared Model Profiles)

**Changes (WP-G4-2 — (d)-analog audit dispatch, per ADR-018):**
- Extend `tier_router.py` with the (d)-analog audit dispatch (or a sibling module `tier_router_audit.py` if responsibility footprint warrants separation — judgment at BUILD-time)
- Three drift criteria at quantitative-threshold level: verdict-distribution shift (±15% between consecutive windows); escalation-vs-outcome correlation drift (default: escalation must produce at least +5pp outcome improvement over the audit window to be interpretable as a tier-configuration signal — this is the Sub-Q6 evidence surface); bypass-rate trend (default: +25% relative-rate increase per window)
- Audit verdict trichotomy: no drift / advisory / severe drift
- Severe-drift response: route-all-to-escalated fail-safe mode + operator notification
- Asynchronous-operator-review dynamic for advisory drift (diagnostics accumulate in operator-facing storage; operator reviews at session boundary)
- Extend `OrchestratorConfig` with audit-dispatch trigger thresholds (count + wall-clock) and drift-criteria thresholds — all operationally tunable
- Outcome observer wiring: the Router observes the dispatched ensemble's outcome (already available at the interposition point) for the escalation-vs-outcome correlation criterion
- **Sub-Q6 downstream consequence (per ADR-018 + ADR-015 §Consequences §Neutral coupling note):** the (d)-analog audit's escalation-vs-outcome correlation drift criterion's first-deployment evidence on the cycle's North-Star benchmark structurally closes Sub-Q6 (autonomous-routing evidence gap). OQ #14 partial closure for the L1→L2 stage is the inline-grounding deliverable; Sub-Q6 closure is the empirical-validation deliverable

**Scenarios covered:** scenarios.md §Per-Role Tier-Escalation Router — all 6 scenarios. **(d)-analog audit dispatch scenarios:** added in roadmap.md as design drivers; scenario candidates for the audit dispatch's trigger, drift criteria, and severe-drift fail-safe are Cycle 5+ scenario authorship territory (the audit dispatch is structurally specified by ADR-018, but scenario-level test specification can be deferred to the BUILD work or to a follow-up scenarios update).

**Participating modules:** Tier-Escalation Router (new + extended), Orchestrator Tool Dispatch (extended), Orchestrator Configuration (extended), Calibration Gate (verdict consumer), Ensemble Engine (metadata source + outcome observer for the (d)-analog audit), models/structural_errors.

**Dependencies:** Hard on **WP-A4** (typed errors). Hard on **WP-F4** (consumes verdict surface from extended Calibration Gate). WP-G4-2 ((d)-analog audit dispatch) has implied ordering after WP-G4-1 (core router) — a skilled builder could land them together or sequence them at BUILD-time judgment.

**Falsification trigger (per ADR-018, inherits ADR-016's elaboration-by-evidence discipline):** if BUILD finds that the (d)-analog audit dispatch cannot be operationalized within the Tier-Escalation Router module's responsibility (e.g., requires its own top-level module orthogonal to L0–L3, or requires bidirectional coupling with Calibration Gate that violates the read-only verdict-consumption contract), the elaboration-by-evidence framing commitment is invalidated for WP-G4-2; ADR-018 re-deliberates, OQ #14 partial closure reverts to "BUILD evidence will inform", and Sub-Q6 re-opens. Pause BUILD and escalate to practitioner.

---

### WP-H4: ADR-016 — Cross-layer calibration channel — *CONDITIONAL ACCEPTANCE*

**Objective:** Land the Calibration Signal Channel L1 module with the five bounding mechanisms; satisfy ADR-016's first-deployment evidence trigger.

**Changes:**
- New module `calibration_signal_channel.py` with the read-only L0→L1 signal channel
- The five bounding mechanisms — (a) fresh-context isolation in consumer, (b) time-decay windowing on cross-layer signals (60-min/100-signal dual-bound, linear decay), (c) categorical anchors via deterministic-tool-output (when ensemble has script-model slot), (d) periodic out-of-band audit dispatch (every 100 verdicts or 24 hours, whichever first), (e) read-only structural validation at the consumer
- Audit verdict trichotomy: no drift / advisory / severe drift
- Severe drift triggers fail-safe mode (verdicts default to Reflect-or-Abstain); operator notification
- Update FC-2 layer map to recognize the L0→L1 read-only annotated exception
- Update FC-3 cycle detection to account for the new edge
- New `malformed_signal` typed error (mechanism (e); internal — not raised to orchestrator)
- Calibration Gate (extended in WP-F4) gains conditional signal-channel consumption — when WP-H4 is active, gate reads windowed signals; HTC features extracted once at L0 and propagated upward

**Scenarios covered:** scenarios.md §Cross-Layer Calibration Channel — all 11 scenarios.

**Participating modules:** Calibration Signal Channel (new), Calibration Gate (extends consumption), Ensemble Engine (emits signals via registered hook), models/structural_errors.

**Dependencies:** Hard on **WP-A4** (typed errors), Hard on **WP-F4** (Calibration Gate's verdict surface). **Last in BUILD sequence** per the conformance scan and ADR-016's conditional-acceptance status.

**Conditional-acceptance handling.** WP-H4 is the only WP whose acceptance is conditional on first-deployment evidence. Per ADR-016 §"Concrete monitoring specification": the trigger artifact is either (i) a BUILD-phase research log entry recording the cross-layer channel's first dispatch outcome on a non-trivial fixture, or (ii) a PLAY-phase field note recording the channel's behavior on the cycle's North-Star benchmark. The cycle-status table for any cycle that touches ADR-016 includes a row noting the channel's status (conditional / fully accepted / superseded).

**Falsification trigger.** If BUILD or first-deployment evidence finds that mechanism (b) windowing or mechanism (d) audit dispatch cannot be operationalized within ADR-002's L0-L3 structure (e.g., they require a top-level module orthogonal to the four-layer architecture), the elaboration-by-evidence framing commitment is invalidated; the reorganization branch re-opens; ADR-016 is re-deliberated with reorganization on the table.

---

### WP-K: Plexus Integration (Plexus-active paths) — *deferred from Cycle 1*

**Objective:** Replace the Adapter's no-op bodies with real plexus MCP client calls; land the cross-session calibration persistence edge so composed ensembles' trust survives Session boundaries when Plexus is active.

**Status:** **Deferred.** Candidate triggers for un-deferring: (a) `/rdd-play` surfaces a concrete need that integration would address, (b) production deployments accumulate enough composition activity that cross-session trust matters, or (c) the Plexus enrichment pipeline matures enough to make AS-4 / AS-5 substantively load-bearing (cycle-status OQ #7).

**Changes:**
- Replace `PlexusAdapter` no-op method bodies with real plexus MCP client calls.
- New `Calibration Gate → Plexus Adapter` edge: extract a `CalibrationStore` Protocol behind the gate's per-session record store, with an `InProcessCalibrationStore` default and a `PlexusBackedCalibrationStore` for Plexus-active deployments.
- Plexus-active branch of `record_outcome` writes asynchronously; the Adapter's read of recent outcomes returns enriched content.

**Scenarios covered:**
- §query_knowledge returns enriched content when Plexus is populated
- §record_outcome writes asynchronously without blocking the ReAct loop *(Plexus-active branch — write-through plus eventual consistency)*
- §Calibration persists across sessions when Plexus is active *(the scenario WP-H deferred)*
- §Session Lifecycle: Four-layer stack operates with Plexus present
- §Cost and Quality Experimentation: Same task runs with and without Plexus context across Model Profiles *(testable OQ #1)*

**Dependencies:**
- WP-I (hard) — Adapter surface and Tool Dispatch wiring already in place.

**Participating modules:** Plexus Adapter (replace bodies), Calibration Gate (extract `CalibrationStore` Protocol), Plexus lib (external).

---

### WP-J: Bootstrapping Pipeline

**Objective:** Operator-triggered batch ingestion of the library (ensemble YAML, scripts, profiles, execution artifacts) into Plexus as source material (AS-4).

**Changes:**
- New **Bootstrapping Pipeline** module.
- CLI command for triggering bootstrap.
- Uses Plexus Adapter's ingestion path.

**Scenarios covered:** scenarios.md §Cost and Quality Experimentation (Bootstrapped graph shortens time-to-first-useful-query — testable OQ #4).

**Dependencies:**
- WP-K (hard) — needs Plexus-active Adapter paths (was WP-I; updated 2026-04-24 when WP-I split to skeleton + WP-K).

**Participating modules:** Bootstrapping Pipeline, Plexus Adapter (called through), Ensemble Engine (reads library via existing config manager). Consistent with WP scope.

---

## Dependency Graph (Cycle 4)

```
WP-A4 (LlmOrcStructuralError base class)
   │
   ├─ hard ─▶ WP-C4 (ADR-017 phantom_tool_call guard)
   ├─ hard ─▶ WP-D4 (ADR-013 Session Registry artifacts)
   ├─ hard ─▶ WP-E4 (ADR-012 Conversation Compaction)
   └─ hard ─▶ WP-F4 (ADR-014 Calibration Gate verdict trichotomy)

WP-B4 (FC-2 + FC-3 automated checks)
   │
   └─ open choice with WP-A4 — no hard dependency in either direction

WP-C4 (ADR-017) ─ open choice with WP-D4 (mutually independent)
WP-D4 (ADR-013) ─ open choice with WP-E4 (mutually independent at architecture level)
WP-E4 (ADR-012) ─ open choice with WP-D4

WP-F4 (ADR-014)
   │
   ├─ hard ─▶ WP-G4 (ADR-015 router consumes verdict)
   └─ hard ─▶ WP-H4 (ADR-016 channel composes with verdict computation)

WP-G4 (ADR-015) ─ implied ─▶ no downstream WP (terminal in this cycle's WP set)
WP-H4 (ADR-016) ─ conditional acceptance — first-deployment evidence is the validation trigger

(deferred Cycle 1 WPs)
WP-I (Adapter skeleton, complete) ─ hard ─▶ WP-K (replaces no-op bodies with real Plexus client)
WP-K (Plexus integration, deferred) ─ implied ─▶ cross-session calibration persistence
WP-K ─ hard ─▶ WP-J (Bootstrapping pipeline, deferred)
```

**Classification key:**

- **Hard dependency:** structural necessity — the downstream WP's code imports, extends, or requires the upstream WP's output. The builder has no choice.
- **Implied logic:** suggested ordering — building the upstream first is simpler because the downstream references concepts it defines, but a skilled builder could stub the references and fill in later.
- **Open choice:** genuinely independent — build either first.

---

## Transition States (Cycle 4)

### TS-1: Stateless orchestrator serving OpenCode — **reached 2026-04-22 (Cycle 1)**

See Completed Work Log. An operator points OpenCode at the llm-orc endpoint and runs an RDD phase through it; the orchestrator routes tasks to existing library ensembles, summarizes results, enforces Budget, and delegates client-side actions at turn boundaries. No self-composition, no Plexus, no calibration.

### TS-2: Stateless baseline complete — **reached 2026-04-24 (Cycle 1)**

See Completed Work Log. The orchestrator composes new ensembles from existing library primitives, validates them, and calibrates them within the session. Still no Plexus.

### TS-3: Four-layer stack with Phase 1 Plexus integration — *deferred*

Reached via WP-I (skeleton, complete) + WP-K (deferred — un-defers when `/rdd-play` surfaces concrete needs, when production deployments accumulate composition activity, or when the Plexus enrichment pipeline matures sufficiently) + WP-J (deferred until WP-K).

### TS-4: Typed-error infrastructure + structural fitness checks (after WP-A4 + WP-B4) — *Cycle 4*

A coherent intermediate where the typed-error base class lives, FC-2 and FC-3 run automated, and the codebase's layering discipline is mechanically verified. No new behavior shipped, but the substrate for ADRs 012–017 is in place. Foundational; unblocks all subsequent Cycle 4 WPs.

### TS-5: Independent ADR completions (after TS-4 + WP-C4 + WP-D4 + WP-E4) — *Cycle 4*

Three of the six new ADRs (017, 013, 012) are landed independently. The orchestrator now has the structural validation guard (phantom_tool_call detection), the structured-handoff artifact set with write-gate validation (Cluster 2 sessions), and the conversation compaction pipeline (long-horizon coherence). No tier escalation, no cross-layer calibration. This is a usable Cluster-2-aware long-horizon orchestrator.

### TS-6: In-process calibration + tier escalation (after TS-5 + WP-F4 + WP-G4) — *Cycle 4*

The verdict trichotomy and per-role tier-escalation router are landed. Dispatches now route to per-skill tier defaults based on calibration verdicts. ADR-014 + ADR-015 compose to form the in-process calibration-and-escalation system. Still no cross-layer calibration channel; in-process layer operates on L1-internal trajectory data only.

### TS-7: Full cross-layer calibration system (after TS-6 + WP-H4 conditional acceptance) — *Cycle 4*

The Calibration Signal Channel is active; HTC trajectory features extracted at L0 and propagated upward through the read-only channel; bounding mechanisms (a)–(e) operational; periodic audit dispatch detecting drift. **Conditional on first-deployment evidence on the cycle's North-Star benchmark.** This is the cycle's most novel architectural territory — the moment the elaboration-by-evidence framing commitment is empirically tested.

---

## Open Decision Points (Cycle 4)

### Cycle 4 build-time decision points

**C4-1. Layer 3 session-notes template storage.** ADR-012 specifies "continuously-maintained at zero LLM cost"; storage is implementation-tunable. Build-time decision: in-memory (simpler) vs. filesystem-resident (operator-readable, can compose with structured-handoff artifact set per ADR-013). Default: in-memory; promote to filesystem if BUILD evidence shows operators want to read the template. Affects WP-D4 / WP-E4 coupling.

**C4-2. Topaz skill metadata migration order.** Existing library ensembles need `topaz_skill` field (FC-18). Decision: migrate all at once (single PR) vs. incremental with default-to-`tool_use` fallback. Default: migrate all at once; absent-skill produces explicit error per ADR-015. Affects WP-G4.

**C4-3. Layer 4 summarizer ensemble — separate from `agentic-result-summarizer`?** ADR-012 §Consequences §Neutral says "Layer 4's LLM-summary semantics are a Conversation Compaction concern, distinct from AS-7's Result Summarization." Decision: ship separate `agentic-context-summarizer.yaml` for Cycle 4. Default: separate. Affects WP-E4.

**C4-4. OQ #14 grounding-mechanism asymmetry follow-up.** The decide-gate finding flagged five other cross-layer stages with less rigor than ADR-016. ARCHITECT's responsibility-allocation choices either (a) surface gaps for Cycle 5+ research, (b) propose grounding mechanisms inline as drivers, or (c) note that BUILD evidence will inform what grounding the other stages need. **Per the cycle status: choice (c) is the practitioner's selection** — first-deployment evidence is the natural validation surface for the asymmetric-rigor concern.

**C4-5. Sub-Q6 routing-reliability evidence gap (ADR-015 carry-forward).** Multi-iteration routing reliability at North-Star benchmark session length is empirically open. ARCHITECT records this as a deployment-evidence carry-forward — operators interpreting escalation-rate calibration evidence may be reading routing-noise rather than tier-configuration mismatches until first-deployment evidence resolves Sub-Q6. No architectural action; the responsibility is Calibration Gate's audit dispatch (mechanism (d)) detecting routing-quality patterns over time. Affects WP-H4 audit-verdict diagnostic content.

### Carry-forward Cycle 1 decision points (preserved for posterity / unresolved deferred WPs)

1. **Client-tool delegation scenarios in `scenarios.md`** *(resolved 2026-04-22 via DECIDE mini-cycle)*. The four stress scenarios are written into `scenarios.md` §Client Tool Surface Commitment. All four carried by Option C: (a)/(b) via intended turn-boundary delegation and Session continuity; (c) via pre-invoke delegation (orchestrator reads file at prior turn boundary, folds content into `input_data`); (d) via the **retry pattern** (ensemble runs atomically, agent emits structured `needs_client_tool`, Result Summarization preserves signal, orchestrator re-invokes with client-tool result folded into `input_data`). Option D (mid-execution callback) is out of scope for this cycle — it would require amending ADR-001/ADR-002 and adding suspend/resume to the DAG engine's synchronous phase loop — so scenario (d) could not reopen the Commitment as an Option-D question — only as a retry-viability question. Retry is viable; Commitment stands. See `system-design.md` Amendment #4. WP-F is now unblocked. The retry pattern's conditional dependence on a composed-ensemble convention for emitting structured un-met-dependency signals carries forward as Open Decision Point #8.

2. **Visibility form (OQ #2).** ~~WP-E's composition-event surfacing currently defaults to structured SSE events.~~ **Resolved during WP-E build (2026-04-22):** visibility renders as `[composition: {json}]` narration on `delta.content` so vanilla OpenAI-compat clients (OpenCode / Roo Code / Cline) surface the event inline in the assistant message. Chosen over SSE comment lines (invisible to spec-compliant clients) and structured non-standard `data:` fields (risks strict clients dropping the stream). Operator-only tooling surfaces can layer on later without changing WP-E's emission shape.

3. **Budget specific numbers (ADR-005 defers to build).** WP-C defaults need concrete turn and token limits. The outer anchor is "comparable to running an RDD phase." Concrete numbers are a tuning decision informed by observed rollout, not an architecture decision.

4. **Calibration N (ADR-007 defers to build)** *(resolved 2026-04-24 at WP-H close)*. Default `N = 3` — balances check cost against single-invocation noise tolerance. Operators tune via `agentic_serving.orchestrator.calibration.default_n`. Check mechanism is an LLM-based ensemble (`agentic-calibration-checker`, shipped) that parses `signal: positive|negative|absent` from the agent's response; operators point at a domain-specific checker via `agentic_serving.orchestrator.calibration.checker_ensemble`. No architectural constraint — both numbers and mechanism are runtime-configurable.

5. **Session identity mechanism.** WP-B defaults to message-history-derivation with optional client-supplied correlation via the OpenAI `user` field. If Autonomy tightening or multi-client deployments make this insufficient, a custom header or session-id cookie becomes necessary. Build-time decision; the Session Registry contract accommodates either.

6. **`record_outcome` payload schema** *(resolved 2026-04-24 at WP-I close)*. Minimum payload `{ensemble_name: str, quality_signal: "positive"|"negative"|"absent", context: str}`, composing with WP-H's `QualitySignal` vocabulary. The Plexus Adapter passes the dict through unchanged today; WP-K extends if Plexus enrichment requires richer fields. The orchestrator LLM is not bound to this schema in WP-I — Tool Dispatch forwards arguments verbatim — but the orchestrator system prompt's `record_outcome` description recommends this shape.

7. **Visibility surface for conductor-ceiling observations (OQ #6).** Not a decision point for any WP directly, but an observability requirement that WP-E and WP-I should consider together — the orchestrator's routing-decision stream is a window into whether orchestration depth is reachable by smaller models.

8. **Retry-signal enforcement mechanism for composed ensemble un-met dependencies** *(build-time decision, introduced 2026-04-22 via scenario (d) of the Client Tool Surface Commitment)*. Scenario (d) carries Option C via the retry pattern, but its viability is conditional on composed ensembles emitting a structured `needs_client_tool` signal when an agent lacks a required input. The failure mode when the convention isn't honored is a *quality* failure (agent hallucinates plausible-looking output), not a correctness/safety failure — the Session doesn't crash, Budget still enforces. Several layered mechanisms could ensure retry: (i) orchestrator system prompt instructing the Orchestrator LLM to recognize `needs_client_tool` in ensemble summaries and delegate at the turn boundary (soft, LLM compliance); (ii) composed-ensemble prompt convention for emitting the structured signal (soft, LLM compliance); (iii) deterministic script-agent precondition guard at phase 0 of composed ensembles (hard, script deterministic); (iv) structural detection in Orchestrator Tool Dispatch that recognizes the schema and emits a `ClientToolCall` chunk directly (hard, code-enforced, adds protocol surface); (v) Calibration Gate quality-check at first N invocations treating silent hallucination as a calibration failure (WP-H territory — catches drift, not first-invocation). Minimum viable stack for WP-F: (i) + (ii). This is a build-time default, not an architectural commitment; if WP-F reveals measurable reliability gaps, mechanisms (iii) or (iv) can be introduced as follow-on work without requiring a new ADR. (v) is WP-H backstop against drift. Specific stack is a build-time decision informed by observed WP-F behavior; not an architectural decision.

---

## Completed Work Log

### Cycle 4: Cheap-orchestrator + ensembles support — in progress

**Derived from:** ADRs 012-018, Essay 005 (`005-layer-conditional-composition.md`)

| WP | Title | Closed | Commits | Status |
|----|-------|--------|---------|--------|
| WP-A4 | Shared `LlmOrcStructuralError` base class | 2026-05-11 | `cc0d94f`, `7c2f64e` | Complete |
| WP-B4 | FC-2 layering + FC-3 cycle-detection automated checks | 2026-05-11 | `1701a22` | Complete |
| WP-C4 | ADR-017 tool-call structural validation guard | 2026-05-11 | `9116793` | Complete |

#### WP-A4 detail

**Objective:** Land the typed-error base class that ADRs 012, 013, 014, 015, 016, and 017 all depend on. Migrate the existing `ToolCallingNotSupportedError` (commit `9f86d0b`) as the first concrete subclass.

**Commits (in order):**

- `cc0d94f feat: add LlmOrcStructuralError base class for typed-error pipeline (WP-A4)` — new module `src/llm_orc/models/structural_errors.py` with the four common fields per ADR-017 §"Shared typed-error base class" and FC-17; `RecoveryAction` literal finalized as `"reformulate" | "escalate" | "abstain" | "operator_intervention_required"` per the architect-close decomposition. New test file `tests/unit/models/test_structural_errors.py` (8 base-class tests).
- `7c2f64e refactor: migrate ToolCallingNotSupportedError to LlmOrcStructuralError subclass` — re-parented in `src/llm_orc/models/base.py`; `error_kind="tool_call_rejected_per_model"` and `recovery_action_required="reformulate"` fixed by construction; existing call sites unchanged. `NotImplementedError` lineage dropped (verified no caller depends on it). 4 new tests confirming subclass shape.

**Outcome:** FC-17 coverage at 1 of 8 typed-error surfaces; full test suite 2363 passing; mypy strict + ruff + complexipy + bandit + vulture all clean. Tier 1 stewardship check clean — no responsibility, dependency, cohesion, size, or test-quality flags. No undecided territory surfaced.

**Participating modules:** `models/structural_errors` (new), `models/base.py` (existing — `ToolCallingNotSupportedError` re-parented).

#### WP-B4 detail

**Objective:** Land the static fitness checks the conformance scan flagged as missing prerequisites. Both checks recognize the ADR-016 calibration-channel exception via an annotated allowed-edge in the layer map.

**Commits (in order):**

- `1701a22 test: add FC-2 layering and FC-3 cycle-detection checks (WP-B4)` — AST-based static scan with layer-map registry covering all 12 existing agentic modules plus the Ensemble Engine; ADR-016 read-only L0→L1 signal-channel exception pre-declared in `_ALLOWED_UPWARD_EDGES`; not-yet-landed Cycle 4 modules (`conversation_compaction`, `tier_router`, `calibration_signal_channel`) pre-declared so the test is ready when subsequent WPs ship; fail-closed coverage test catches new agentic modules without an explicit layer assignment; FC-3 uses iterative tri-color DFS cycle detection with a complementary non-empty-graph assertion preventing vacuous pass.

**Design notes:**

- Contract modules (`orchestrator_chunk`, `session_start`) are layer-neutral and exempt from FC-2; they participate in FC-3 as ordinary graph nodes.
- Imports inside `if TYPE_CHECKING:` blocks are excluded from both scans — they are a Python idiom for breaking circular type-annotation imports and do not execute at module load. `session_registry`'s `TYPE_CHECKING` import of `session_start.ChatMessage` exercises this exclusion.
- The dep-graph in system-design.agents.md notes 28 architectural edges; this test exercises the subset that manifest as actual Python imports (~11 edges). Logical "calls" edges (e.g., Budget Controller → Session Registry, where the dependency is parameter-injected rather than imported) are not in scope — adding them would be a deferred amendment.

**Outcome:** Four new tests, all passing; full suite 2367 passing; mypy strict + ruff + complexipy + bandit + vulture all clean. Tier 1 stewardship check clean.

**Participating modules:** test files only.

#### WP-C4 detail

**Objective:** Land the structural validation guard in Tool Dispatch as the most-bounded ADR with the most direct codebase precedent. First behavioral typed-error producer using `LlmOrcStructuralError` (WP-A4 base class).

**Commits:**

- `9116793 feat: add ADR-017 tool-call structural validation guard (WP-C4)` — new module `src/llm_orc/agentic/tool_call_validation_guard.py` with `PhantomToolCallError` (subclass of `LlmOrcStructuralError`, fixed `error_kind="phantom_tool_call"` + `recovery_action_required="reformulate"`), nine default assertion-pattern regexes per ADR-017 §Detection (deliberately excluding future-intent patterns per rejected alternative (f)), and the pure scanner function `scan_response_for_phantom_claims`. Tool Dispatch (`orchestrator_tool_dispatch.py`) gains a `validate_response(response_text, tool_call_names, *, session_id)` method and a `tool_call_validation_patterns` constructor argument; the typed error is re-exported through Tool Dispatch as the public API surface. Runtime (`orchestrator_runtime.py`) calls the dispatcher's `validate_response` after each LLM response; on detection a new `_record_phantom_tool_call_rejection` helper appends the rejected assistant turn plus a structural-feedback `role:user` diagnostic before continuing the loop (orchestrator reformulates on the next iteration; rejected response prose is not surfaced to the client). To stay under the complexipy ceiling after adding the new branch, two existing-logic helpers are extracted from `run`: `_chunks_for_response` (post-response routing — content delta + tool_calls split + mixed batch + client batch + internal dispatch) and `_budget_exhaustion_chunks`. `OrchestratorConfig` (`orchestrator_config.py`) gains a `tool_call_validation_patterns: tuple[str, ...] = ()` field with a malformed-input-tolerant resolver helper; `v1_chat_completions._build_orchestrator_tool_dispatch` threads the operator pattern set through to the dispatcher. `tool_call_validation_guard` added to `_LAYER_MAP` at L2 in the FC-2 test (FC-4's narrow Runtime allowlist preserved — Runtime imports only the typed-error class, not the guard module).

**Design notes:**

- **Scope of structural correspondence check.** The guard treats any non-empty `tool_calls` list as the structural anchor — a response with prose claiming a tool was called and an emitted tool-call structure passes regardless of name correspondence. Per-name correspondence (prose claims `invoke_ensemble`, structure names `compose_ensemble` → partial mismatch) is a future enhancement; the four ADR-017 scenarios as written are satisfied at the minimum-viable level and the conservative false-positive discipline supports the simpler check.
- **Default pattern set source.** Patterns derive from the spike's observed text (essay 005 Wave 3.A Trial 3: *"the tool call has been made and the result is displayed above"*) plus the explicit examples in ADR-017 §Detection. The set is **minimal rather than calibrated** — the spike evidence does not support a richer default per ADR-017 §"Minimal default pattern set". Operator-extensibility under `orchestrator.tool_call_validation_patterns` in config is the operational refinement surface.
- **Phantom rejection wiring.** The runtime's `_record_phantom_tool_call_rejection` helper appends a `role:user` diagnostic (not `role:tool` — the rejected response had no tool-call structure to attach a `tool_call_id` to) carrying the JSON payload `{error: "phantom_tool_call", reason: ..., detected_prose_claim: ..., recovery_action_required: "reformulate"}`. The orchestrator's next iteration sees this in its message history and reformulates per ADR-017 §Rejection ("the orchestrator's reasoning surface receives the structural feedback").
- **FC-4 architectural preservation.** Tool Dispatch owns the guard responsibility (per system-design.agents.md L107-119); the Runtime calls through to Tool Dispatch's `validate_response` rather than importing the guard module directly. This preserves FC-4's narrow Runtime allowlist (the guard module is not in `_ALLOWED_AGENTIC_IMPORTS`) and matches the architectural intent: the guard is Tool Dispatch's interposition step (1) on `invoke_ensemble`.

**Outcome:** 27 new guard-module tests + 4 new runtime guard tests + 3 new config tests + parametrized future-intent and assertion-pattern coverage; full suite 2400 passing; mypy strict + ruff + complexipy + bandit + vulture all clean. Tier 1 stewardship check clean. FC-17 coverage now 2 of 8 typed-error surfaces.

**Participating modules:** `agentic/tool_call_validation_guard` (new), `agentic/orchestrator_tool_dispatch` (extended), `agentic/orchestrator_runtime` (extended), `agentic/orchestrator_config` (extended), `web/api/v1_chat_completions` (wiring).

---

### Cycle 1: Stateless agentic serving baseline (closed 2026-04-29)

**Derived from:** ADRs 001-011, Essay 001 (`001-agentic-serving-architecture.md`), Essay 002 (`002-capability-floor-and-observability.md`)

| WP | Title | Closed | Status |
|----|-------|--------|--------|
| WP-A | Cycle-validator extraction (retrofit debt) | 2026-04-20 | Complete |
| WP-B | Serving foundation + session-start | 2026-04-21 | Complete |
| WP-C | ReAct core + real LLM adapter | 2026-04-21 | Complete |
| WP-D | Result Summarizer Harness | 2026-04-21 | Complete |
| WP-E | Autonomy Policy | 2026-04-22 | Complete |
| WP-F | Client-tool turn-boundary delegation | 2026-04-22 | Complete |
| WP-G | Composition + Composition Validator | 2026-04-22 | Complete |
| WP-H | Calibration Gate | 2026-04-24 | Complete |
| WP-I | Plexus Adapter skeleton (no-op fallbacks) | 2026-04-24 | Complete |

**Summary:**
- TS-1 (stateless orchestrator serving OpenCode) reached at WP-F close (2026-04-22)
- TS-2 (stateless baseline complete per ADR-002 Layer 1-3 and AS-8) reached at WP-H close (2026-04-24)
- Plexus Adapter skeleton landed at WP-I close (2026-04-24) — FC-7 stateless coverage complete; WP-K (Plexus-active body-swap) and WP-J (Bootstrapping pipeline) deferred
- 13 fitness criteria (FC-1 through FC-13) defined and verified or in-place; 18 boundary integration tests; 12 modules + 1 typed extension function across 4 dependency layers
- Test suite at Cycle 1 close: 2347 passing, 91.56% coverage, lint clean (mypy strict + ruff + bandit + vulture + complexipy)

**Dependency graph (as-built; preserved for posterity):**

```
WP-A (extract cycle validator) ─ hard ─▶ WP-G (composition)
WP-B (serving foundation) ─ hard ─▶ {WP-C, WP-F}
WP-C (ReAct core) ─ hard ─▶ {WP-D, WP-E, WP-F, WP-G, WP-I}; implied ─▶ WP-H
WP-G (composition) ─ implied ─▶ WP-H (calibration of composed ensembles)
WP-I (Adapter skeleton) ─ hard ─▶ WP-K (deferred Plexus body-swap)
```

**Per-WP detail follows below.** Migrated unchanged from prior roadmap structure.

---

### WP-I: Plexus Adapter skeleton (no-op fallbacks) — 2026-04-24

**Commits (in order):**

- `<TBD>` docs: split Plexus integration into WP-I (skeleton) + WP-K (deferred)
- `<TBD>` feat: add Plexus Adapter skeleton with no-op fallbacks (WP-I Group 1)
- `<TBD>` feat: wire Plexus Adapter into Tool Dispatch (WP-I Group 2)

**Outcome.** Tool Dispatch's `query_knowledge` and `record_outcome` switch from `not_yet_wired` errors to delegating through the Plexus Adapter. With Plexus absent (the WP-I shipping configuration), `query_knowledge` returns `{"results": [], "context": ""}` and `record_outcome` returns `{"acknowledged": True}` — both flow through the dispatch's match-case routing as `ToolCallSuccess`. FC-7 stateless coverage complete; WP-K is body-swap territory — replacing the Adapter's method bodies with real plexus MCP client calls does not require changes to Tool Dispatch, Runtime, or any tool-call shape.

**New module.** `src/llm_orc/agentic/plexus_adapter.py` (L1) — `PlexusAdapter` class with two async methods (`query`, `record`) holding no-op fallback bodies. Class-shaped (rather than module-level functions) so WP-K injects the plexus MCP client through `__init__` without touching call sites. Constructor takes no parameters in WP-I; WP-K extends the signature when the client surface is committed.

**Tool Dispatch.** New `PlexusAccess` Protocol (narrow surface — `query` and `record` only) lets tests substitute recording doubles. `OrchestratorToolDispatch.__init__` gains an optional `plexus_adapter: PlexusAccess | None = None` parameter; production wiring always passes one, the absent-adapter path returns the existing `not_yet_wired` typed error as a defensive fallback. The Adapter is constructed in `v1_chat_completions.get_orchestrator_tool_dispatch` alongside the other process-scoped dispatch dependencies.

**`record_outcome` payload schema decision (Open Decision Point #6 resolved).** Minimum payload is `{ensemble_name, quality_signal, context}` composing with WP-H's `QualitySignal` vocabulary. Tool Dispatch forwards arguments verbatim to the Adapter — no schema validation in dispatch, no rejection of richer payloads. The orchestrator LLM gets the recommendation through the system prompt; WP-K extends if Plexus enrichment requires richer fields.

**Test coverage.** 4 unit tests in `test_plexus_adapter.py` (no-op contract for query and record + argument-insensitivity); 4 wiring tests in `test_orchestrator_tool_dispatch.py::TestPlexusToolWiring` (dispatch delegates with recording double + dispatch delegates with real Adapter); 3 boundary integration tests in `test_tool_dispatch_plexus_boundary.py::TestQueryKnowledgeAndRecordOutcomeRoundTrip` (production-shaped wiring with real PlexusAdapter, covering scenarios.md §query_knowledge returns empty gracefully + §record_outcome writes asynchronously + §Orchestrator's ReAct loop remains responsive while enrichment lags). Existing `TestNotYetWiredTools` renamed to `TestPlexusToolsRequireAdapter` and asserts the absent-adapter fallback. Full suite **2347 passing, 91.56% coverage, lint clean**.

**Forward-carrying to WP-K.**
- The `PlexusAccess` Protocol is the seam — WP-K replaces `PlexusAdapter`'s method bodies and any consumer that holds a `PlexusAccess` reference (Tool Dispatch, future Calibration store) keeps working.
- `record` exception handling in Tool Dispatch is intentionally bare (no try/except) — the no-op never raises. WP-K decides whether real-Plexus failures should degrade to empty or surface as `ToolCallError`; the right answer there is contextual to the actual plexus MCP client behavior, so committing to either now would be premature.
- The `not_yet_wired` `ToolErrorKind` is retained for the absent-adapter fallback. After WP-I, no production code path produces it; it remains a defensive shape for misconfigured deployments and for tests that don't bother passing an Adapter.

---

### WP-H: Calibration Gate — 2026-04-24

**Commits (in order):**

- `3ab6f27` feat: add Calibration Gate module (WP-H Group 1)
- `9caa4b4` feat: interpose Calibration Gate on compose/invoke (WP-H Group 2)
- `d3da9d8` test: add Tool Dispatch → Calibration Gate boundary integration (WP-H Group 3)
- (this change) docs: close WP-H in roadmap, cycle-status, field guide, ORIENTATION

**Outcome.** Every composed ensemble enters calibration at compose time and the first `N = 3` invocations run a result-checker ensemble that produces a Quality Signal (`positive` / `negative` / `absent`). Three positives in the most-recent-N window transition the ensemble to `trusted`; a negative or absent signal keeps it in calibration indefinitely. **TS-2 reached — stateless baseline complete** per ADR-002 Layer 1-3 and AS-8. Calibration is session-scoped while Plexus is absent; cross-session persistence lands with WP-I via `Calibration Gate → Plexus Adapter`.

**New module.** `src/llm_orc/agentic/calibration_gate.py` (L1) — `CalibrationGate.{mark_composed, status, check_and_record, record_for}` with per-session records indexed by `(session_id, ensemble_name)`. `QualitySignal = Literal["positive", "negative", "absent"]` per system-design §Integration Contracts. `DEFAULT_CALIBRATION_N = 3` (ODP #4 resolution). A `CalibrationChecker` Protocol narrows the checker surface so tests pass scripted doubles; `EnsembleBackedChecker` is the production implementation that invokes a configured checker ensemble and parses `signal: <value>` from the response. The gate is stateful per-process — L3 callers pass plain session-id strings so the L1 module stays free of L3 imports (layering-clean, same pattern as Budget Controller).

**Interposition on Tool Dispatch.** `OrchestratorToolDispatch` accepts an optional `CalibrationGate` and a new `session_id` kwarg on `dispatch()`. On successful `compose_ensemble` the gate is notified via `mark_composed`; on `invoke_ensemble` the raw result is handed to the gate via `check_and_record` before summarization. Calibration failures are swallowed (`_calibration_check_safe`) — ADR-007 clause 2: the check never prevents invocation. The `ToolDispatcher` Protocol in the Runtime widened with `session_id: str = ""` so the Runtime threads `state.identity.value` to dispatch; test doubles and existing call sites carry the default and need no churn.

**Default checker ensemble.** `.llm-orc/ensembles/agentic-calibration-checker.yaml` ships as the default — a single-agent ensemble that asks the LLM "Does this output look like a plausible, on-task response?" and returns `signal: positive|negative|absent`. Uses the same `summarizer` model profile as the Result Summarizer Harness (small, fast; operators swap via `config.yaml` when domain-specific checking is needed). The parser tolerates case variation and surrounding prose; unparseable responses yield `absent`, never raise.

**Config surface.** `OrchestratorConfig.calibration: CalibrationDefaults(default_n, checker_ensemble)`. Operators override via `agentic_serving.orchestrator.calibration.{default_n, checker_ensemble}` in `config.yaml`. Invalid `default_n` (zero, negative, non-integer) falls back to the shipped default rather than failing session start.

**Fitness Criteria touched.**
- **FC-12** (integration — "composed ensembles enter Calibration Gate transparently on invoke_ensemble during calibration") — satisfied by `tests/integration/test_tool_dispatch_calibration_boundary.py::TestCalibrationInterposesOnInCalibrationEnsembles::test_calibration_interposes_on_in_calibration_ensembles`. Real Tool Dispatch + real Calibration Gate + scripted checker + real `OrchestraService` → `ExecutionHandler` → `EnsembleExecutor` → `MockModel`.
- **FC-4** unchanged — `calibration_gate` was already on the Runtime's forbidden-import list (WP-D); the new module's arrival did not require a code change to the static check.
- **AS-5** (quality signals govern stabilization, not frequency) enforced by `test_frequency_without_quality_does_not_trust` — ten invocations with mixed signals never transition the ensemble to trusted.

**Scenarios covered (scenarios.md §Calibration of Composed Ensembles):**
- First N invocations result-checked — unit + integration layers
- Transition to trusted with sufficient positive signals — unit + integration
- Fails to clear with negative signals — unit (period extends after a negative; a clean run of positives later transitions)
- Session-scoped when Plexus absent — unit + integration

Scenario §Calibration persists across sessions when Plexus is active is deferred to WP-I.

**Feed-forward to WP-I.**
- Calibration Gate persistence layer lands alongside the Plexus Adapter. The gate's `mark_composed` / `status` / `check_and_record` surface is the contract WP-I preserves; a Plexus-backed store is injected behind it without changing Tool Dispatch's call sites.
- `CalibrationRecord.signals` is currently `tuple[QualitySignal, ...]`. Plexus persistence may introduce richer structure (timestamps, evidence) — the record is a `@dataclass`, so additive fields are non-breaking for existing callers.
- The checker currently runs synchronously and blocks `invoke_ensemble`. A future optimization (async/background) is possible but out of scope for TS-2. Calibration adds ~one LLM-call's worth of latency per in-calibration invocation.

**Test count and quality.** 18 unit tests in `test_calibration_gate.py`, 6 Tool Dispatch interposition tests in `test_orchestrator_tool_dispatch.py`, 4 boundary integration tests in `test_tool_dispatch_calibration_boundary.py`, 2 config tests in `test_orchestrator_config.py`, 1 Runtime plumbing assertion in `test_orchestrator_runtime.py`. Full suite **2336 passing, 91.56% coverage, lint clean** (mypy strict + ruff + format + complexipy + bandit + vulture).

**Summarizer-quality echo-back (WP-D FF #81) carried forward.** A weak summarizer that echoes a JSON-encoded raw dict in its `response` field remains a quality risk the structural Harness cannot detect. The Calibration Gate is the designed backstop and now runs end-to-end for composed ensembles. A richer checker ensemble that specifically detects echo-back — rather than just plausibility — is a follow-up left to operator tuning (swap `checker_ensemble` via config.yaml) or a future WP if empirical observation warrants.

---

### WP-G: Composition + Composition Validator — 2026-04-22

**Commits (in order):**

- `32d2dd3` refactor: add compute_reference_graph_depth helper for composition-time depth check
- `9972ed3` feat: add Composition Validator module (WP-G Group 1)
- `e5f8ea0` feat: wire compose_ensemble through Composition Validator (WP-G Group 2)
- `804aeb7` test: add composition boundary integration + acceptance coverage (WP-G Group 3)
- (this change) docs: close WP-G in roadmap, cycle-status, field guide, ORIENTATION

**Outcome.** `compose_ensemble` is fully wired. The orchestrator can now assemble a new ensemble from existing primitives, have it validated against AS-2, AS-6, Invariant 5 (cross-ensemble acyclicity), Invariant 7 (static reference resolution), and Invariant 8 (depth limit), and — on accept — persist the ensemble to the local tier at `.llm-orc/ensembles/{name}.yaml`. Composition-time validation is stricter than load-time: AS-6 existence checks (profile, script, ensemble) reject dangling references that the load path tolerates silently, and Invariant 8 is enforced before disk write instead of deferring to the runtime. AS-2 is structurally enforced — the writer is only reached after `CompositionAccepted`.

**New module.** `src/llm_orc/agentic/composition_validator.py` (L1) — `CompositionValidator.validate(request)` returns `CompositionAccepted(config)` or `CompositionRejected(kind, reason)` across seven outcomes:

- `invalid_agent_schema` — Pydantic rejects the agent dict
- `missing_dependency` — sibling `depends_on` not present
- `internal_dependency_cycle` — intra-ensemble dep cycle
- `invalid_fan_out` — `fan_out: true` without `depends_on`
- `missing_primitive` — profile/script/ensemble does not exist in the library (AS-6)
- `cross_ensemble_cycle` — delegates to `validate_ensemble_reference_graph` (FC-6)
- `depth_limit_exceeded` — `compute_reference_graph_depth` > configured limit (Invariant 8)

Production adapters live in the same module: `ConfigManagerPrimitiveRegistry` wraps `ConfigurationManager` + `ScriptResolver` + ensemble directory discovery; `ConfigManagerEnsembleWriter` persists an accepted config to the local tier with collision rejection (mirrors `EnsembleCrudHandler.get_local_ensembles_dir`). `EnsembleWriteError` inherits `ValueError` so Tool Dispatch narrows on a single exception type for the whole validation-plus-write surface.

**Edits.**

- `OrchestratorToolDispatch.__init__` takes two new kwargs: `composition_validator: CompositionGate` and `local_ensemble_writer: LocalEnsembleWriter` (both Protocols — tests substitute scripted doubles without constructing the production validator's registry dependency). `compose_ensemble` parses arguments, delegates to the validator, and hands the accepted config to the writer; validation rejection and write failure both surface as `ToolCallError(kind="invocation_failed", reason=...)` so the ReAct loop continues with a typed observation. Malformed arguments (missing name, wrong description type, non-list agents) surface as `ToolCallError(kind="invalid_arguments", reason=...)` without touching the validator.
- `v1_chat_completions.get_orchestrator_tool_dispatch` constructs the real registry + validator + writer from the shared `ConfigurationManager` so a `config.yaml` edit takes effect on the next request without restart.
- `core/config/ensemble_config.py` gains `compute_reference_graph_depth(name, agents, search_dirs)` — sibling of `validate_ensemble_reference_graph`, reusing the existing `_build_reference_graph` helper. Depth 0 is a leaf; an N-edge chain returns N, matching the runtime depth counter in `EnsembleAgentRunner`.

**Scenarios covered.** `scenarios.md` §Ensemble Composition with Validation — all seven scenarios have explicit coverage:

- §Composition with only profiles and scripts succeeds → `TestAcceptance::test_accept_with_only_profiles_and_scripts` (unit) + `TestEnsembleCompositionWithValidationAcceptance::test_compose_happy_path_writes_new_ensemble_and_reports_to_llm` (Serving Layer)
- §Composition with ensemble-to-ensemble reference passes → `TestAcceptance::test_accept_with_existing_ensemble_reference` (unit)
- §Composition that would introduce a cycle fails → `TestCrossEnsembleCycle::test_reject_cycle_through_existing_ensembles` (unit) + `TestComposeEnsembleRejectsCycle::test_compose_ensemble_rejects_cycle` (boundary) + `TestEnsembleCompositionWithValidationAcceptance::test_compose_rejects_cycle_and_leaves_local_tier_untouched` (Serving Layer)
- §Composition referencing a non-existent primitive fails → `TestPrimitiveExistence` class (unit — profile, script, ensemble)
- §Composition exceeds depth limit → `TestDepthLimit::test_reject_when_proposed_graph_exceeds_depth_limit` + boundary accept at limit
- §Composition never authors scripts or profiles → `TestComposeEnsembleNeverAuthorsPrimitives` (boundary, structural) + existing `TestAutonomyAndPromotionAcceptance::test_script_authorship_never_permitted_at_any_level`
- §(integration) shared single routine → `TestSharedValidatorSameBothPaths::test_shared_validator_same_result_both_paths` (boundary FC-6 regression)

**Fitness criteria status.**

- FC-5 (exactly five public dispatch entry points): unchanged.
- FC-6 (Composition Validator and Ensemble Engine's load path call the same public validator function): **fully satisfied**. One definition at `core/config/ensemble_config.py:309`; four call sites (load path, `list_ensembles`, `ValidationHandler`, Composition Validator). The regression test verifies both paths return identical outcomes on the same input and that the composition validator imports the routine from its canonical module.
- FC-4 (Runtime import surface): unchanged — no new imports into `orchestrator_runtime.py`.
- FC-11 (Autonomy gate fires before every dispatch): unchanged.

**Decisions made during build.**

- **Shared helper in `ensemble_config.py`, not inline in the validator.** `compute_reference_graph_depth` lives alongside `validate_ensemble_reference_graph` so both graph-walking routines stay in one module. Depth detection reuses `_build_reference_graph` — the private helper already walks search dirs. Composition-time is the only caller today, but the placement keeps the option open for a future load-time depth check without another extraction pass.
- **Primitive existence is composition-time strict, not load-time strict.** Load-time's tolerance of dangling ensemble references (Invariant 7 is enforced at execution via `child_executor`, not at load) preserves test fixture flexibility in the existing suite. Composition-time follows AS-6's literal reading — "compose from existing primitives only" — so the orchestrator cannot create an ensemble that names a missing profile/script/ensemble.
- **Depth check is composition-time only.** Moving `EnsembleAgentRunner`'s runtime depth enforcement into load-time would be scope creep and a behavior change on existing ensembles. Composition-time depth check is an additive composition-level discipline that does not alter load-path behavior.
- **`CompositionGate` Protocol on Tool Dispatch.** Concrete `CompositionValidator` has a deeper dependency surface (primitive registry + depth limit); dispatch-level tests would either construct the full stack or duck-type. The Protocol formalizes the duck-typed surface (one `validate` method) so test doubles pass mypy strict without pulling the registry.
- **Test-default `_rejecting_validator`, not `_UnusedValidator`.** Many existing dispatch-scope tests dispatch `compose_ensemble` incidentally (autonomy-gate coverage, visibility-event routing). A default that raises on consult (`_UnusedValidator`) would break those tests; a default that rejects cheaply (`_rejecting_validator`) keeps them passing while preserving loud failure on the write path if a test reaches it by mistake. Tests that assert on composition behavior pass scripted validators explicitly.

**Test coverage delta.** +20 tests (13 unit validator + 5 boundary integration + 2 Serving Layer acceptance). Full suite: **2297 passing, 91.51% coverage, lint clean** (mypy strict + ruff + format + bandit + vulture + complexipy).

**Unblocks.** WP-H (Calibration Gate) now has a composed-ensemble code path to calibrate against — scenarios under §Calibration of Composed Ensembles can be exercised end-to-end once the Calibration Gate is wired.

**Forward-carrying concerns** (not addressed in WP-G scope).

- **Overwrite semantics.** `ConfigManagerEnsembleWriter.write` rejects collision with the existing-file message. A future workflow where the orchestrator wants to update an existing composition (e.g., after calibration-driven refinement) would need an explicit `compose_ensemble(overwrite=True)` argument or a separate tool — not the default behavior.
- **Hierarchical ensemble names.** `EnsembleLoader.find_ensemble` supports `examples/neon-shadows/neon-shadows` style names, but the writer always targets a flat `{name}.yaml` in the local tier. If composition-time names collide with a hierarchical library entry, the writer's collision check will not see it. Low priority — composed ensembles use simple names today.
- **Raw-output escape hatch via composition.** `CompositionRequest.raw_output` is plumbed through the writer, but no scenario exercises the orchestrator composing a raw-output ensemble. If WP-H/WP-I work surfaces the need, add a scenario; otherwise the `raw_output: false` default is structurally fine.

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

### WP-F: Client-tool turn-boundary delegation — 2026-04-22

**Objective delivered.** The Client Tool Surface Commitment (Option C) is implemented end-to-end. The orchestrator closes turns with `finish_reason: tool_calls` when a task step needs a client-side action, and the next `/v1/chat/completions` resumes the same Session with the client's `role: tool` messages as observations. TS-1 (stateless orchestrator serving OpenCode) is reached.

**Commits (in order):**

*Group 1 — Turn-boundary mechanics (scenarios a + b):*
- `93e1229` refactor: relocate ChatMessage to session_start and extract tool-call encoder
- `61a6c40` feat: route client-declared tools through turn-boundary delegation (WP-F Group 1)
- `b29a3b3` feat: tighten mixed-batch discipline and reserve TOOL_NAMES (WP-F Group 1)
- `5d13e50` docs: record WP-F Group 1 feed-forward signals in cycle-status

*Group 2 — Pre-invoke delegation (scenario c):*
- `813bf60` test: add scenario (c) pre-invoke delegation acceptance (WP-F Group 2)

*Group 3 — Retry pattern + system prompt (scenarios d + negative):*
- `f3b9253` feat: land retry pattern and orchestrator system prompt (WP-F Group 3)
- (this change) docs: close WP-F in roadmap, cycle-status, field guide, ORIENTATION

**Outcome.** The orchestrator Runtime accepts the union of the closed internal five tools (ADR-003) and client-declared `tools[]` in each session, classifies each LLM-emitted tool call by `TOOL_NAMES` membership, and routes accordingly: internal calls dispatch in-process through Tool Dispatch; client-declared calls yield a `ClientToolCall` chunk and terminate the generator. The Serving Layer shapes `ClientToolCall` into `finish_reason: tool_calls` on both the streaming and non-streaming paths and accepts `role: tool` + `tool_call_id` on subsequent requests so Session continuity survives the round trip. The orchestrator system prompt (roadmap ODP #8 mechanism i) teaches the LLM the turn-boundary discipline, the one-kind-per-turn rule, and the `needs_client_tool` retry convention; the default summarizer YAML (ODP #8 mechanism ii) preserves structured signals verbatim.

**New modules/fields.**
- `ChatMessage` relocated to `agentic/session_start.py` with optional `tool_call_id` and `tool_calls` fields so `role: tool` messages and echoed `role: assistant` with `tool_calls` parse through the request schema.
- `OrchestratorRuntime` gains a `system_prompt` constructor kwarg — always prepended as `role: system` on every LLM iteration when non-empty.
- `OrchestratorConfig.orchestrator_system_prompt` field with `DEFAULT_ORCHESTRATOR_SYSTEM_PROMPT`; operators override via `agentic_serving.orchestrator.system_prompt` in config.yaml.
- `_NonStreamingResult` dataclass in `v1_chat_completions.py` collects content + finish_reason + optional tool_calls for the non-streaming response body.
- `encode_tool_call_for_message` helper in `sse_format.py` shared between streaming (`_encode_tool_calls` adds `index`) and non-streaming paths.
- `_reject_reserved_tool_names` guard in the Serving Layer — HTTP 400 if client declares a tool whose name is in `TOOL_NAMES`.

**Edits.**
- `OrchestratorRuntime.run` splits the LLM's tool-calls batch by `TOOL_NAMES` membership. Mixed batches (internal + client in one response) feed a `mixed_batch` error observation per call and the LLM retries on the next iteration — no silent data loss. Pure-client batches yield `ClientToolCall` and terminate. Pure-internal batches dispatch as before. `_dispatch_internal_calls` was extracted from `run` for complexity-ceiling compliance.
- `session_registry.ChatMessage` moved to `session_start` (contract type on the Serving Layer → Runtime edge, not Session Registry internals). Session Registry uses TYPE_CHECKING forward ref to avoid circular import. Keeps FC-4 intact when Runtime imports ChatMessage.
- `_ChatCompletionMessage` Pydantic model gains optional `tool_call_id` and `tool_calls` fields; `content` is now nullable. `_resolve_context` threads these into the `ChatMessage` dataclass.
- `.llm-orc/ensembles/agentic-result-summarizer.yaml` `default_task` teaches the summarizer to echo `needs_client_tool` JSON verbatim when present; production deployments inherit the correct default.

**Scenarios covered.** `scenarios.md` §Client Tool Surface Commitment — all five scenarios pass via eight tests in `TestClientToolSurfaceCommitment`:
- `test_orchestrator_delegates_client_tool_at_turn_boundary` — scenario (a), non-streaming
- `test_session_continuity_across_client_tool_round_trip` — scenario (b)
- `test_streaming_client_tool_delegation_yields_tool_calls_chunk` — scenario (a), streaming
- `test_mixed_batch_rejected_and_retried_without_silent_loss` — mixed-batch discipline
- `test_client_tool_shadowing_internal_name_is_rejected` — collision guard
- `test_pre_invoke_delegation_reads_file_before_invoking_ensemble` — scenario (c)
- `test_retry_pattern_resolves_mid_execution_client_tool_need` — scenario (d)
- `test_composed_ensemble_without_retry_signal_silently_degrades` — scenario (negative)

**Fitness criteria status.** No new FCs introduced by WP-F. Existing FC-4, FC-5, FC-8, FC-9, FC-11 all continue to pass (verified via static-inspection tests).

**Decisions made during build.**
- **Mixed-batch reject-and-retry** (Group 1 refinement). When the LLM emits internal + client in one batch, feed a typed `mixed_batch` error per call and loop — never silent drop. Recorded in cycle-status FF #98.
- **Name-collision guard** (Group 1 refinement). Client tools whose names match `TOOL_NAMES` are rejected with HTTP 400. Alternative (drop-with-warning) was considered and rejected because silent misrouting on collision is worse than immediate actionable error. Recorded in cycle-status FF #99.
- **System prompt always prepends** (Group 3). Chosen over skip-when-client-has-system because the orchestrator's discipline is load-bearing exactly for deployments that send their own system message (agentic coding clients). Two `role: system` messages in sequence is awkward but the orchestrator's guidance wins.
- **Summarizer transparency via YAML prompt, not code** (Group 3). Keeps the Harness generic — it does not know about the `needs_client_tool` vocabulary. Tests drive the production path with stubbed summarizers. Recorded at cycle-status FF (Group 3).

**Test coverage delta.** +13 tests net (5 WP-F acceptance from Group 1 + 1 from Group 2 + 2 from Group 3 — all in `TestClientToolSurfaceCommitment`; 3 Runtime system-prompt unit tests; 2 OrchestratorConfig tests). Full suite: **2270 passing, 91.52% coverage**, lint clean (mypy strict + ruff + format + bandit + vulture + complexipy).

**Unblocks.** **TS-1 reached.** The stateless orchestrator can serve OpenCode: list ensembles, invoke them, summarize results, enforce Budget, delegate client-side actions (file_read, bash, file_edit) at turn boundaries, and retry composed ensembles with client-tool results folded into input_data. Next parallel candidates: WP-G (Composition + Validator) and WP-I (Plexus Adapter).

**Forward-carrying concerns** (not addressed in WP-F scope).
- **Silent quality failures when retry convention not honored.** Scenario (negative) documents the failure mode structurally; catching it belongs to WP-H's Calibration Gate quality-signal check at first N invocations. Cycle-status FF #81 carries this from WP-D.
- **AS-6 authorship open question.** The user flagged that the orchestrator should eventually be able to create scripts and model profiles. AS-6 currently prohibits both on conservative safety grounds. Revisit as a standalone DECIDE mini-cycle post-TS-1. Cycle-status FF #100.
- **`list_ensembles` description richness.** Scenario (c) works with the current description field, but production deployments may need richer metadata (agent input expectations, tier, freshness) as composed ensembles proliferate. Not blocking; defer until a real use case surfaces.

### WP-E: Autonomy Policy — 2026-04-22

**Commits (in order):**
- `f07f64b` feat: add AutonomyPolicy module and VisibilityEvent chunk type (WP-E Group 1)
- `b2a1c88` refactor: carry VisibilityEvent tuple on ToolCallSuccess and ToolCallError
- `6c168da` feat: interpose Autonomy Policy gate before every Tool Dispatch (WP-E Group 2)
- `536f952` feat: render VisibilityEvent as delta.content narration (WP-E Group 3)
- `8ca482a` test: add autonomy and promotion acceptance scenarios (WP-E Group 5)
- `29fb4c0` test: add FC-11 static gate check and boundary integration (WP-E Group 6)
- (this change) docs: close WP-E in field guide, ORIENTATION, cycle-status, roadmap

**Outcome.** ADR-008's per-session Autonomy Level gate is interposed before every Orchestrator Tool Dispatch (FC-11). Two Phase-1 levels ship: `operator-as-tool-user` (baseline, silent) and `pure-tool-user-visible` (surfaces composition events). The composition event renders as `[composition: {json}]` narration on `delta.content` — OQ #2's resolution favors tool-user-visible inline narration over operator-only SSE comments so the llm-conductor tinkering loop closes in the same conversation thread the tool user interacts with.

**New module.** `src/llm_orc/agentic/autonomy_policy.py` — `AutonomyPolicy.decide(tool_name, arguments)` returns `Allow(events)` or `Deny(reason)`. Deny is first-class for WP-H's future approve-before-uncalibrated semantics; Phase 1 never returns it. `VisibilityEvent(kind, payload)` is a neutral chunk variant in `orchestrator_chunk.py` — future event kinds (routing, calibration) reuse the same shape without changing the chunk contract. The SSE formatter's `render_visibility_narration` helper is shared between the streaming path and the non-streaming response-body collector so transport does not change what the tool user sees.

**Edits.**
- `OrchestratorToolDispatch.__init__` takes `autonomy_policy: AutonomyGate`; `dispatch()` runs a three-step flow (unknown-tool filter, gate, route) and attaches decision events to the result via `_with_events`. `_route` factored from the old dispatch match-case body so FC-11's lexical ordering check has one call site to reason about.
- `ToolErrorKind` gains `denied_by_autonomy`.
- `OrchestratorRuntime.run` iterates `result.events` and yields each as a `VisibilityEvent` chunk between `InternalToolCallInFlight` and `InternalToolCallResult`.
- `v1_chat_completions.get_orchestrator_tool_dispatch` constructs `AutonomyPolicy` with `level_provider=lambda: resolver.resolve().autonomy_level` so `config.yaml` edits take effect on the next request.
- SSE formatter renders `VisibilityEvent` as `delta.content`; non-streaming `_collect_non_streaming` does the same via the shared helper.

**Scenarios covered.**
- `scenarios.md` §Default Autonomy Level permits invocation, permits composition, gates promotion — acceptance at the Serving Layer via `tests/unit/web/test_api_v1_chat_completions.py::TestAutonomyAndPromotionAcceptance`. Structural check: `"promote_ensemble" not in TOOL_NAMES`.
- `scenarios.md` §Tool user without operator role observes composition events when configured — acceptance same class; `[composition:` narration appears in `choices[0].message.content` between turn segments at the tightened level.
- `scenarios.md` §Pure tool-user session at default Autonomy Level experiences silent composition — acceptance same class; no narration substring at baseline.
- `scenarios.md` §Script authorship is never permitted at any Autonomy Level — acceptance same class, parametrized over `[BASELINE, TIGHTENED, synthetic-future]`; AS-6 closure via the `TOOL_NAMES` unknown-tool filter.

**Fitness criteria status.**
- FC-11 (Autonomy Policy check executes before every Tool Dispatch): **fully satisfied**. `test_fc11_autonomy_gate.py` proves three AST properties on `dispatch`: decide is called at least once; every `await self._route(...)` is lexically after the first decide call; an adversarial synthetic bypass (route-before-gate) trips the detector. Boundary integration at `tests/integration/test_tool_dispatch_autonomy_policy.py` verifies the real `AutonomyPolicy` fires for every committed tool and stays silent on unknown names.

**Test coverage delta.** +36 tests (AutonomyPolicy unit 14; dispatch gate unit 7; SSE formatter visibility 2; acceptance scenarios 6; FC-11 static 4; boundary integration 3). Full suite: **2257 passing, 91.48% coverage, lint clean** (mypy strict + ruff + format + bandit + vulture).

**Decisions made during build.**
- **Events-on-result over DispatchOutcome wrapper.** Adding `events: tuple[VisibilityEvent, ...] = ()` to `ToolCallSuccess` and `ToolCallError` kept the `ToolDispatcher` Protocol signature unchanged and let existing tests pass without modification; a `DispatchOutcome(result, events)` wrapper would have rippled across ~15 call sites for the same semantic payload.
- **`_route` factoring.** Split from the old match-case body in `dispatch` so FC-11's lexical ordering check has a single callable to reason about. A future regression that inlined `_route` back into `dispatch` would trip `test_dispatch_routes_exactly_via_self_route`.
- **Visibility narration form (OQ #2).** `[kind: {json}]` is generic across event kinds, greppable by operators, and survives JSON's newline escaping so the narration stays single-line. Chosen for tool-user-visible observability (vanilla clients show `delta.content` inline); operator-parseable SSE comments can be a future additive surface without changing the emission shape.
- **Unknown-level fallback to baseline-silent.** An operator typo or a future level name leaking into config ahead of policy code falls back to baseline rather than locking sessions out; the missing surfacing is a visible hint.

**Unblocks.** TS-1 remaining work: WP-F (client-tool turn-boundary delegation, scenario-gated) is the only TS-1 item left. WP-G (composition) and WP-I (Plexus Adapter tool-first) both depend only on WP-C and can land in parallel.

**Forward-carrying concerns** (not addressed in WP-E scope).
- **Summarizer-quality echo-back risk → WP-H calibration scope.** Carried forward from WP-D FF #81; WP-E did not address it because summarizer quality is a calibration property, not an autonomy property.
- **Per-session Autonomy Level overrides.** Phase 1 operates at operator-configured level; a future WP with per-session overrides can widen `level_provider`'s signature without rewriting policy code.
- **Operator-tooling visibility surface.** SSE comment lines or a structured events endpoint can be added as a second audience-specific surface without changing WP-E's `delta.content` emission.

---

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
