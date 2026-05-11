# Roadmap: Agentic Serving

**Generated:** 2026-04-20; **last amended:** 2026-05-08 (Cycle 4 ARCHITECT close)
**Derived from:** `system-design.md` (v3.0), ADRs 001-017 + adr-deferred-005, scenarios.md, interaction-specs.md

This roadmap expresses the sequencing landscape for building agentic serving — what depends on what, where the builder has a choice, and which coherent intermediates are worth pausing at. It does not prescribe a build order. Work package order within each dependency band is a build-time decision.

---

## Work Packages — Cycle 4 (active)

> **Cycle 1 WPs (WP-A through WP-I) are complete and migrated to the Completed Work Log.** TS-1 (stateless orchestrator serving OpenCode) reached at WP-F close (2026-04-22); TS-2 (stateless baseline) reached at WP-H close (2026-04-24); Plexus Adapter skeleton landed at WP-I close (2026-04-24). The active section below lists Cycle 4 work plus deferred Cycle 1 work (WP-K, WP-J).
>
> **Cycle 4 BUILD comprises eight new WPs (WP-A4 through WP-H4)** integrating ADRs 012–017 into the codebase per the conformance-scan-recommended sequence. Identifiers reset for the new active cycle (per skill methodology: "Reset identifiers for the next active cycle — don't accumulate escalating letters across cycles").

### WP-A4: Shared `LlmOrcStructuralError` base class — *T1 prerequisite*

**Objective:** Land the typed-error base class that ADRs 012, 013, 014, 015, 016, and 017 all depend on. Migrate the existing `ToolCallingNotSupportedError` (commit `9f86d0b`) as the first concrete subclass.

**Changes:**
- New module `models/structural_errors.py` with `LlmOrcStructuralError` base class and the four common fields (`error_kind`, `dispatch_context`, `recovery_action_required`, `operator_diagnostic`)
- Migrate `ToolCallingNotSupportedError` → `LlmOrcStructuralError(error_kind="tool_call_rejected_per_model", ...)`; preserve existing exception chain via subclassing if call sites depend on the old type
- Update `recovery_action_required` literal type to `Literal["reformulate", "escalate", "abstain", "operator_intervention_required"]`

**Scenarios covered:** none directly — infrastructure prerequisite. Unblocks the eight `error_kind` types in ADRs 012–017.

**Dependencies:** None. **First WP.**

**Participating modules:** `models/structural_errors` (new), `models/base.py` (existing — preserves precedent).

---

### WP-B4: FC-2 automated import layering check + FC-3 automated cycle detection — *T1 prerequisite*

**Objective:** Land the static fitness checks the conformance scan flagged as missing prerequisites. Both checks recognize the ADR-016 calibration-channel exception via an annotated allowed-edge in the layer map.

**Changes:**
- New `tests/unit/agentic/test_fc2_layering.py` — AST-based per-module import walk; layer-map registry; assertion that every import edge respects "edges point higher → same-or-lower" with the calibration-channel exception annotated. Pattern follows existing `test_fc4_runtime_import_surface.py`.
- New `tests/unit/agentic/test_fc3_no_cycles.py` — directed graph construction over agentic-module imports + annotated logical edges; cycle detection via DFS or topological sort; assertion of zero cycles.

**Scenarios covered:** none directly — these enforce architectural fitness criteria. They constrain all subsequent BUILD work.

**Dependencies:** Open choice with WP-A4 — mutually independent.

**Participating modules:** test files only.

---

### WP-C4: ADR-017 — Tool-call structural validation guard

**Objective:** Land the structural validation guard in Tool Dispatch as the most-bounded ADR with the most direct codebase precedent.

**Changes:**
- Extend `orchestrator_tool_dispatch.py` with response-text scan for default phantom_tool_call assertion patterns
- New `phantom_tool_call` typed error (uses `LlmOrcStructuralError` base from WP-A4)
- Operator-extensible pattern set surface in `OrchestratorConfig`
- Pattern-matching logic produces typed error on mismatch; orchestrator gets the error as a tool-call observation

**Scenarios covered:** scenarios.md §Tool-Call Structural Validation Guard — Match (validation passes); Mismatch (phantom_tool_call produced); Future-intent patterns not flagged; Pattern set is operator-extensible.

**Participating modules:** Orchestrator Tool Dispatch (extended), models/structural_errors (uses base class), Orchestrator Configuration (extends pattern-set surface).

**Dependencies:** Hard on **WP-A4** (uses `LlmOrcStructuralError`).

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
