# System Design Companion: Agentic Serving — Agent Context

**Version:** 3.0
**Status:** Current
**Last amended:** 2026-05-08
**Companion to:** [system-design.md](./system-design.md) (slim human-facing surface)
**Scope:** Scoped RDD cycle at `docs/agentic-serving/`. Inherits the project-level domain model (Invariants 1-14) and existing system architecture.

---

## How to read this file

This is the **agent-context companion** to `system-design.md`. It carries the dense reference material that benefits agent context-construction for code work — full driver tables, module decomposition with Owns/Depends/Inversion notes, the responsibility matrix, the per-edge integration contracts, fitness criteria, test architecture, and the per-phase susceptibility-snapshot briefs (Appendix A).

The split is the **companion-file pattern** (ADR-084 Pattern B): the slim sibling at [system-design.md](./system-design.md) is for first-encounter human readers; this file is read end-to-end (or by section) when an agent needs full architectural detail. Both files share the same Version field; they are amended together.

The contracts and matrices in this file are the load-bearing artifacts FC-1 through FC-13 verify against. Appendix A is the canonical brief source for the **Phase Boundary: Susceptibility Snapshot Dispatch** subsection of every phase skill executing within this cycle's scope.

---

## Architectural Drivers

| Driver | Type | Provenance |
|--------|------|------------|
| Stateless-first operability — baseline product (Layers 1-3) runs with no Plexus dependency | Quality Attribute (primary) | AS-8; ADR-002; cycle-status §FF 15-16 |
| Deterministic control plane — Budget, Tool Surface, and Primitive set are harness-level circuit breakers, not model-level choices | Quality Attribute | AS-3, AS-6; ADR-003, ADR-005, ADR-008 |
| Orchestrator reasoning quality across long sessions | Quality Attribute | AS-7; ADR-004, ADR-005; essay 001 §Context Management |
| Swappable orchestrator LLM via Model Profile | Quality Attribute | ADR-011; OQ #1 (knowledge-compensated model selection) |
| Observability of orchestrator activity | Quality Attribute | Product discovery tensions #2 and #5; essay 002 (capability floor + observability) |
| Auditability — closed tool-call vocabulary | Quality Attribute | ADR-003 |
| OpenAI-compatible protocol: `/v1/chat/completions` + `/v1/models`, SSE streaming with tool-call round-trips | Constraint | Essay 001 §API Surface; interaction specs |
| Phase 2 Plexus injection hook point must be structurally reserved | Constraint | ADR-009 (post-gate reframe) |
| Ensemble Engine (Layer 3) unchanged | Constraint | ADR-001, ADR-002 |
| Project-level Invariants 1-14 remain in force; AS-1 through AS-8 layered on top | Constraint | Project domain model; agentic-serving domain model |
| Push-model Plexus ingestion; source-material ingestion boundary; async enrichment | Constraint | AS-4; ADR-010 |
| Orchestrator profile change is a session-boundary event | Constraint | ADR-011 |
| Existing FastAPI server and MCP handlers are extended, not replaced | Integration | Retrofit reconnaissance (2026-04-20) |
| Plexus (external lib) is optional; two code paths — with and without Plexus — must maintain feature parity on Layers 1-3 | Integration | ADR-002; AS-8 |
| Client-declared tools (OpenCode, Roo Code, etc.) flow through turn-boundary delegation, not through the orchestrator's internal action space | Integration | system-design.md §Client Tool Surface Commitment |
| Session sized for sustained agentic coding comparable to an RDD phase; multi-LLM-call-per-turn token accounting | Scale | ADR-005 |
| Orchestrator coherence across long-horizon sessions (50K-token degradation point per Chroma 2025; Khanal 2026 universal-non-improvement under episodic memory); requires automatic compaction at turn boundaries | Quality Attribute | ADR-012; essay 005 §"Long-Horizon Reliability"; Chroma 2025; Liu et al. TACL 2023; Khanal arXiv:2603.29231 |
| Externalized structured state for Cluster 2 sessions (BUILD/ARCHITECT/DEBUG/REFACTOR) — schema-level non-regression, narrative continuity, deterministic environment bootstrap | Quality Attribute | ADR-013; Confucius Code Agent arXiv:2512.10398; essay 005 §"Long-Horizon Reliability"; Anthropic published initializer schema |
| In-process trajectory-level calibration (AUQ verbalized confidence + HTC trajectory features) producing dispatch-time verdict trichotomy (Proceed / Reflect / Abstain) | Quality Attribute | ADR-014; AUQ arXiv:2601.15703; HTC arXiv:2601.15778; OI-MAS arXiv:2601.04861 |
| Cost-discriminate dispatch via confidence-gated tier escalation (OI-MAS pattern at +12.88% accuracy, −17–78% cost; SC-MAS, MasRouter heterogeneous-staffing evidence base) | Quality Attribute | ADR-015; OI-MAS arXiv:2601.04861; SC-MAS; MasRouter; Topaz paper |
| Cross-layer calibration as the cycle's most novel territory — bias-bounded by 5 mechanisms (a)–(e) per cycle/scale risk from feedback-bias compounding (Khanal universal non-improvement; CAAF prompt-engineering-artifact) | Quality Attribute (load-bearing; conditional acceptance) | ADR-016; essay 005 §"ADR candidate #6"; Khanal arXiv:2603.29231; CAAF arXiv:2604.17025; Wisdom and Delusion of LLM Ensembles arXiv:2510.21513 |
| Class (a) deterministic-override against phantom tool-call confabulation observed at cheap-cloud tier (Wave 3.A Trial 3 spike evidence; codebase precedent commit `9f86d0b`) | Quality Attribute | ADR-017; essay 005 §"Behavioral Spike"; CAAF arXiv:2604.17025 |
| Read-only upward signal channel from L0 to L1 (calibration-only; structurally typed at boundary; gated by 5 bounding mechanisms) — one narrow exception to ADR-002's "edges never upward" layering rule | Constraint (newly established Cycle 4) | ADR-016; ADR-002 partial-update header |
| Shared `LlmOrcStructuralError` typed-error base class binding eight `error_kind` surfaces (`tool_call_rejected_per_model` from commit `9f86d0b`; `phantom_tool_call`, `compaction_layer_4_failure`, `write_gate_rejection`, `calibration_abstain`, `escalation_bypass`, `missing_skill_metadata`, `malformed_signal`) with common fields (`error_kind`, `dispatch_context`, `recovery_action_required`, `operator_diagnostic`) | Integration | ADR-017 §"Shared typed-error base class"; codebase precedent commit `9f86d0b` |
| Topaz eight-skill role profiling (`code_generation`, `tool_use`, `mathematical_reasoning`, `logical_reasoning`, `factual_knowledge`, `writing_quality`, `instruction_following`, `summarization`) declared per-ensemble in YAML metadata | Constraint | ADR-015; Topaz paper |
| Per-skill tier defaults (8 skills × 2 tiers = 16 Model Profile slots) configured at deployment; defaults shareable across skills | Constraint | ADR-015 |

---

## Module Decomposition

Twelve modules plus one typed extension-point function. Existing modules are marked **(existing)**; everything else is net-new surface area for agentic serving.

### Module: Serving Layer

**Purpose:** Translates the OpenAI-compatible wire protocol into Session-scoped orchestrator interactions.
**Provenance:** AS-8; ADR-001, ADR-002 (Layer 1); ADR-009 (Phase 2 injection reservation — function-level); Essay 001 §API Surface; Client Tool Surface Commitment.
**Owns:** Serving Layer (concept); SSE streaming; tool-call formatting; `/v1/chat/completions` and `/v1/models` endpoints; response-surface tool delegation; session-start flow including the typed `resolve_session_start_context` function (Phase 1 returns `[]`; Phase 2 reads from Plexus Adapter).
**Depends on:** Session Registry, Orchestrator Configuration, Orchestrator Runtime. In Phase 2 only: Plexus Adapter.
**Depended on by:** (external clients — FastAPI app)
**Phase 2 hook point.** ADR-009 requires the Phase 2 injection point to be structurally reserved. This is satisfied by the typed function `resolve_session_start_context(session: SessionContext) -> list[PromptFragment]` at a single call site in the session-start flow. Phase 1 returns `[]` unconditionally; Phase 2 populates the body by reading from Plexus Adapter. The contract is the load-bearing part of the reservation — signature and call site commit the interface now so Phase 2 is a function-body change, not a structural change.
**Inversion note:** The Serving Layer boundary serves the Tool User's "the endpoint is a model" mental model (they see only the HTTP surface) and the Operator's "I start the server and point a client at it" mental model. Both converge at this boundary, so the boundary serves both users.

### Module: Session Registry *(extended in Cycle 4 per ADR-013)*

**Purpose:** Identifies and continues a multi-request Session by reconstructing orchestrator state from the conversation history; maintains the structured-handoff artifact set for Cluster 2 sessions; enforces write-gate validation on artifact writes.
**Provenance:** Session (concept); ADR-005 (Budget is Session-scoped); ADR-008 (Autonomy is Session-scoped); ADR-011 (Model Profile is Session-scoped); Client Tool Surface Commitment (Session spans client-tool round trips); ADR-013 (structured-handoff artifact set + write-gate validation + cluster determination); Confucius Code Agent arXiv:2512.10398 (cross-session note-taking pattern empirical evidence); Anthropic published initializer schema (feature-list / progress-log / init.sh structure); OpenDev arXiv:Bui 2026 (class (a)/(c) hybrid schema-level enforcement); Mnemonic sovereignty arXiv:2604.16548 (memory-poisoning attack surface motivating write-gate validation).
**Owns:** Session identity; Session lookup by request; cumulative-turn-count and cumulative-token-spend derivation; persistence of Session state across HTTP requests when persistence is required by Autonomy Level or Calibration state; **Cycle 4 additions:** Structured-handoff artifact (concept) — operator-facing name for the three-component artifact set; the three artifact components — feature-list-with-monotonic-passes (JSON schema with monotonic `passes` field; structural non-regression at schema level), append-only progress log (free-text narrative continuity; filesystem-level append-only constraint), init-sh-style deterministic environment bootstrap (operator-authored shell script; hash-recorded at authoring time); cluster determination at session-start (concept) — Cluster 1 / Cluster 2 / Cluster 3 classification with disposition (i) default-required for cross-cluster ambiguity; write-gate validation (concept) with the three validation classes — (i) JSON schema validation for feature-list (rejects monotonicity violations without audit-logged operator override), (ii) append-only constraint enforcement for progress-log (rejects overwrite, truncate, mid-file edit), (iii) signed-script tamper-detection for init.sh (hash-rotation workflow for legitimate modifications); the typed `write_gate_rejection` error.
**Depends on:** Ensemble Engine (for profile resolution — optional); filesystem (artifact persistence); cryptographic hash routine (init.sh integrity check).
**Depended on by:** Serving Layer, Budget Controller, Autonomy Policy, Calibration Gate.
**Inversion note:** The Operator's mental model is "a Session is the thing with Budget, Autonomy, and orchestrator profile; requests are how clients interact with it." For Cluster 2 sessions, the model extends: "I work with `feature_list.json`, `claude-progress.txt`, `init.sh` by name; I declare cluster at session-start; the system prevents structural regressions and tampering." The methodology-voice term *externalized structured state* (per domain-model §Methodology Vocabulary) does not appear in operator-facing surfaces — operators work with the artifact set's named components.

**Cluster-conditional applicability (ADR-013):** the artifact set is **required** for Cluster 2 phase contexts (long-horizon continuous routing — BUILD/ARCHITECT/DEBUG/REFACTOR territory) and **supported but optional** for Cluster 1 (specialist-dispatch — RESEARCH/DECIDE/SYNTHESIZE) and Cluster 3 (conversational/exploratory — DISCOVER/PLAY) contexts. Cross-cluster sessions default to required per disposition (i). The cluster declaration is operator-set at session-start; mid-session reclassification (disposition (ii)) is a refinement available if BUILD evidence shows reclassification is operationally useful.

**Fitness:**
- **Fitness:** The feature_list.json's `passes` field is monotonic — a write transitioning a feature from `passes: true` to `passes: false` is rejected with `write_gate_rejection` typed error unless the write carries an audit-logged operator override field — verified by `test_write_gate_rejects_passes_regression_without_override` (unit).
- **Fitness:** The progress-log file rejects all non-append operations (overwrite, truncate, mid-file edit) with `write_gate_rejection` typed error; the file on disk is unchanged — verified by `test_progress_log_rejects_non_append_writes` (unit).
- **Fitness:** init.sh execution is gated on a hash match between the operator-recorded hash and the file's current content hash; mismatch produces `write_gate_rejection` typed error and execution is blocked — verified by `test_init_sh_hash_mismatch_blocks_execution` (unit).
- **Fitness:** Session creation honors the operator's cluster declaration; Cluster 2 default-required-artifact-set behavior is the default for ambiguous declarations per disposition (i) — verified by `test_cross_cluster_session_defaults_to_required_artifacts` (integration).

**Direction-not-constraint note (per ADR-076):** "Narrative continuity" of the progress log is a *direction* the artifact set optimizes toward — the log preserves text, but whether it actually serves narrative continuity for the operator is a UX-shaped property that resists single-test decomposition. The append-only constraint is the testable mechanism; the narrative-continuity outcome is the direction.

### Module: Budget Controller

**Purpose:** Enforces turn and token limits at each ReAct iteration boundary.
**Provenance:** AS-3; ADR-005; domain concept Budget.
**Owns:** Budget (concept); per-iteration circuit-breaker check; graceful termination with explicit exhaustion signaling.
**Depends on:** Session Registry.
**Depended on by:** Orchestrator Runtime.
**Inversion note:** Operator's mental model — "Budget is a thing I set; its enforcement is automatic and never negotiable by the LLM." The boundary is thin but load-bearing (AS-3 says control plane, not model plane). Kept separate from Session Registry because its change rate is different (Budget semantics will shift during rollout; Session identity rarely changes).

### Module: Orchestrator Runtime *(extended in Cycle 4 per ADR-012)*

**Purpose:** Runs the ReAct loop that delegates to llm-orc operations via a fixed tool surface; invokes Conversation Compaction at each turn boundary.
**Provenance:** ADR-001; domain concepts Orchestrator Agent, Routing Decision; ADR-012 (Cycle 4: invokes Conversation Compaction at turn boundaries; the *Conversation Compaction* concept ownership re-allocated from Runtime to the new Conversation Compaction module).
**Owns:** Orchestrator Agent (concept); Routing Decision (generation); Route, Invoke (Dynamic), Query, Record, Calibrate (as actor). **No longer owns** Conversation Compaction (re-allocated to the new Conversation Compaction module — Runtime now invokes it the same arms-length way it invokes Tool Dispatch and Budget Controller).
**Depends on:** Budget Controller, Orchestrator Tool Dispatch, **Conversation Compaction** *(new in Cycle 4)*. FC-4 amendment: the Runtime's allowed import set extends from `{Budget Controller, Orchestrator Tool Dispatch}` to `{Budget Controller, Orchestrator Tool Dispatch, Conversation Compaction}`.
**Depended on by:** Serving Layer.
**Inversion note:** The Orchestrator LLM's mental model is "I reason, I emit tool calls, I observe results." The Runtime boundary aligns with that mental model — it does not expose Session bookkeeping, Plexus awareness, Autonomy gating, summarization, calibration internals, tier escalation, or compaction internals to the LLM's reasoning context. Result summarization is interposed by Orchestrator Tool Dispatch on the `invoke_ensemble` return path; Conversation Compaction is invoked by the Runtime at turn boundaries with the messages array as parameter — the Runtime is unaware of compaction internals just as it is unaware of summarizer ensemble selection. The Runtime continues to own Routing Decisions and the Route/Invoke (Dynamic)/Query/Record/Calibrate actions; ownership of Conversation Compaction (the concept) moves to the new module per ADR-012's substantial owned surface.

### Module: Orchestrator Tool Dispatch *(extended in Cycle 4 per ADR-015 + ADR-017)*

**Purpose:** Defines the fixed five-tool surface, dispatches each orchestrator tool call to its downstream service, and runs the structural validation guard on orchestrator responses to detect phantom tool-call confabulation.
**Provenance:** ADR-003; AS-6; domain concepts Orchestrator Tool, Dynamic Invocation; **Cycle 4 additions:** ADR-015 (Tier-Escalation Router interposition on `invoke_ensemble`); ADR-017 (structural validation guard); commit `9f86d0b` typed-error precedent; CAAF arXiv:2604.17025 (class (a) deterministic-override intervention class for confabulation).
**Owns:** Orchestrator Tool (concept); tool-name allowlist enforcement; routing of `invoke_ensemble`, `compose_ensemble`, `list_ensembles`, `query_knowledge`, `record_outcome` to their downstream services; rejection of any other tool name as a tool error; interposition order on `invoke_ensemble` (post-Cycle 4): (1) structural validation guard scans the orchestrator's response text for assertion patterns → `phantom_tool_call` rejection if mismatch; (2) Autonomy Policy gate; (3) Tier-Escalation Router selects Model Profile via Calibration Gate verdict; (4) `EnsembleExecutor.execute` runs the dispatched ensemble; (5) Calibration Gate `check_and_record` (post-result, ADR-007 substrate); (6) Result Summarizer Harness (post-result; intercepts before Runtime sees the dict). **Cycle 4 additions:** Tool-call structural validation guard (concept); Phantom tool-call (concept; the failure mode the guard detects); the operator-extensible default pattern set (assertion patterns: *"the tool returned ..."* / *"I called X and the result was ..."* / *"the result of X is displayed above"* / etc.; future-intent patterns *"I will call X"* are not flagged per the conservative false-positive discipline); the typed `phantom_tool_call` error.
**Depends on:** Ensemble Engine, Composition Validator, Plexus Adapter, Autonomy Policy, Calibration Gate, Result Summarizer Harness, **Tier-Escalation Router** *(new in Cycle 4 per ADR-015)*.
**Depended on by:** Orchestrator Runtime.
**Rationale for separate module:** Keeping dispatch separate from the Runtime's reasoning loop makes the closed-set property of ADR-003 structurally enforceable — a code path that bypasses the dispatch to do something tool-like is mechanically excluded, not merely proscribed. ADR-015's router and ADR-017's structural validation guard inherit this property — they are interposed on the dispatch path, not on the Runtime's reasoning loop.

**Fitness (additions per Cycle 4):**
- **Fitness:** The structural validation guard scans every orchestrator response for the configured assertion patterns; a prose claim of tool-call occurrence with no corresponding structurally-valid tool-call structure in the same response produces `phantom_tool_call` typed error — verified by `test_phantom_tool_call_detected_when_prose_claim_lacks_structure` (integration).
- **Fitness:** Future-intent patterns ("I will call X", "I am going to invoke X") are not flagged — verified by `test_future_intent_patterns_not_flagged` (unit).
- **Fitness:** The pattern set is operator-extensible at deployment configuration — operator-added patterns are scanned alongside default patterns, and operator-added patterns trigger `phantom_tool_call` errors when matched without a corresponding structure — verified by `test_operator_extended_patterns_trigger_validation` (integration).

**Direction-not-constraint note (per ADR-076):** "The pattern-set adequacy as deployment evidence accumulates" is a *direction* the operator-extensibility surface optimizes toward; no automated mechanism verifies pattern-set adequacy. The operator-extension surface IS the operational discovery path — adequacy is what BUILD-time and deployment-time refinement establish.

### Module: Composition Validator

**Purpose:** Validates a proposed ensemble against the existing reference graph using the same routine as load-time validation.
**Provenance:** AS-2, AS-6; ADR-006; Invariant 5 (cross-ensemble acyclicity); Invariant 7 (static references); Invariant 8 (depth limit); Invariant 11 (extras forbidden); scenarios.md refactor 1-3; cycle 1 §FF 21.
**Owns:** Composition (concept); composition-time validation routine shared with `EnsembleLoader`.
**Depends on:** Ensemble Engine (shared validator routine lives in `core/config/ensemble_config.py` as a public function after the refactor).
**Depended on by:** Orchestrator Tool Dispatch.
**Retrofit debt resolved at WP-A:** `_validate_cross_ensemble_cycles` and `_build_reference_graph` were extracted from `EnsembleLoader` private helpers to public `validate_ensemble_reference_graph` in `core/config/ensemble_config.py`. Both load-time and composition-time validation share the single routine (FC-6).

### Module: Ensemble Engine **(existing)**

**Purpose:** Executes ensembles per the existing declarative DAG engine.
**Provenance:** Entire project-level domain model; ADR-001, ADR-002 (Layer 3 unchanged).
**Owns:** Ensemble, Agent, AgentConfig (LLM/Script/Ensemble), Model Profile, Inline Model, Dependency, Phase, Fan-Out, Input Key, Ensemble Reference, Ensemble Reference Graph, Depth, Depth Limit, Artifact, Child Executor, Immutable Infrastructure, Mutable State, Agent Discriminator; actions Load, Validate, Discriminate, Execute, Dispatch, Recurse, Fan Out, Gather, Select, Detect Cycles, Check Depth, Merge Profile.
**Depends on:** (project-level dependencies unchanged)
**Depended on by:** Orchestrator Tool Dispatch (via invoke_ensemble), Composition Validator (shared validator), Result Summarizer Harness (invokes a summarizer ensemble), Bootstrapping Pipeline (reads library), Plexus Adapter (persistence of Routing Decisions derived from executions).

### Module: Result Summarizer Harness

**Purpose:** Positions a summarizer between ensemble completion and the orchestrator's context so unsummarized results never reach reasoning.
**Provenance:** AS-7; ADR-004; domain concepts Result Summarization, Summarize action.
**Owns:** Result Summarization (concept); Summarize (action); raw-output escape-hatch dispatch.
**Depends on:** Ensemble Engine (invokes the summarizer ensemble).
**Depended on by:** Orchestrator Tool Dispatch (interposed on `invoke_ensemble`'s return path).
**Rationale for separate module:** The summarizer itself is an ensemble (a primitive — configured, not coded). What this module owns is the *harness position* — the unskippable interposition between ensemble completion and tool-call result return. The Runtime is not aware of the summarizer; the summarizer is not aware of the Runtime. The harness makes the enforcement of AS-7 structural rather than conventional.

### Module: Autonomy Policy

**Purpose:** Gates orchestrator actions against the Session's Autonomy Level.
**Provenance:** ADR-008; AS-6 (hard limit: no configuration permits primitive authorship); domain concept Autonomy Level.
**Owns:** Autonomy Level (concept); per-action gate resolution; visibility surfacing of composition events when a tightened level requires it.
**Depends on:** Session Registry.
**Depended on by:** Orchestrator Tool Dispatch.
**Note on the pure-tool-user default.** Cycle 1 §FF 25 flags that the default baseline Autonomy Level is calibrated for the operator-as-tool-user persona. Pure tool-user deployments (FF-2) may warrant a tighter default that surfaces composition events. The Autonomy Policy module exposes this as a configuration surface rather than a code change.

### Module: Calibration Gate *(extended in Cycle 4 per ADR-014)*

**Purpose:** Tracks Calibration state, runs quality checks on a composed ensemble's first N invocations (ADR-007 substrate), and produces a dispatch-time calibration verdict (Proceed / Reflect / Abstain trichotomy) for every dispatch decision.
**Provenance:** ADR-007; AS-5; domain concepts Calibration, Quality Signal; **Cycle 4 additions:** ADR-014 (verdict trichotomy + AUQ + HTC + time-decay windowing); AUQ arXiv:2601.15703 (System 1 + System 2 dual-process structure); HTC arXiv:2601.15778 (process-level trajectory features); OI-MAS arXiv:2601.04861 (confidence-gated dispatch pattern that ADR-015's router consumes the verdict from); Chuang et al. arXiv:2502.04428 (uncertainty-quantification method choice dominates threshold choice); essay 005 §"ADR candidate #3" + §"Long-Horizon Reliability Infrastructure".
**Owns:** Calibration (concept); Quality Signal (concept and generation in stateless mode; Plexus Adapter persists when active); per-ensemble calibration counter and transition-to-trusted logic; session-scoped state in stateless mode (ADR-007 clause 4); **Cycle 4 additions:** Calibration verdict (concept; trichotomy *Proceed / Reflect / Abstain*); In-process trajectory-level calibration (concept); Trajectory feature (concept); AUQ verbalized-confidence consumption (System 1 attention-soft + System 2 binary gate at default 0.85 within 0.8–1.0 range; operationally tunable per deployment); HTC trajectory feature extraction (token-level entropy patterns, attention-weight distributions over tool-call sequences, decision-confidence trajectories across reasoning steps); the three Abstain criteria (entropy collapse — token-level entropy in the most recent N tokens drops > 1.5σ below trajectory's running mean; post-hoc result-check hard failure with non-recoverable error class; multiple drift-detection criteria simultaneously exceeding thresholds when ADR-016 mechanism (d) is active and reporting drift); time-decay windowing on trajectory features (60-minute / 100-signal dual-bound, linear decay 1.0 → 0.0 from signal-emission to window-edge; the in-layer instance of ADR-016 mechanism (b)); the typed `calibration_abstain` error; the verdict surface published via `verdict_for(session_id, ensemble_name, dispatch_context)`.
**Depends on:** Ensemble Engine (invokes a check mechanism, which is itself an ensemble); Plexus Adapter (persistence when Plexus is active); **Calibration Signal Channel** *(new conditional dependency in Cycle 4)* — when ADR-016 channel is active, gate reads windowed signals (HTC features extracted once at L0 and propagated upward); when ADR-016 is rejected, gate operates on L1-internal trajectory data only (per ADR-014 §"Feature-extraction location" — the single-extraction-point property is load-bearing for performance and consistency).
**Depended on by:** Orchestrator Tool Dispatch (interposed on `invoke_ensemble` for ensembles in calibration; ADR-007 post-hoc check); **Tier-Escalation Router** *(new in Cycle 4)* — consumes the verdict surface directly with no LLM-mediated translation.

**Composition with ADR-007 (preserved per ADR-014):** the post-hoc layer (ADR-007's first-N output-level mechanism) and the in-process layer (ADR-014's dispatch-time trajectory-level mechanism) compose additively. Post-hoc tracks *whether an ensemble can be trusted*; in-process tracks *whether a specific dispatch should proceed right now*. Both signals govern stabilization (per AS-5, when Plexus is active).

**Fitness:**
- **Fitness:** The verdict trichotomy (Proceed / Reflect / Abstain) is structurally exhaustive: every dispatch produces exactly one verdict; the verdict's value depends only on AUQ confidence, post-hoc result-check signal, and trajectory feature anomaly criteria — verified by `test_verdict_is_exhaustive_and_deterministic_given_inputs` (unit).
- **Fitness:** Time-decay windowing operates only on signals within the dual-bound (60-minute / 100-signal, whichever shorter); signals outside the window contribute weight 0 to the verdict; signals within the window contribute linearly-decaying weight from 1.0 (most recent) to 0.0 (window edge) — verified by `test_time_decay_windowing_dual_bound_linear` (unit).
- **Fitness:** ADR-007's first-N post-hoc result-check continues to fire on every first-N invocation under ADR-014's extension; the existing quality-signal accumulation and trusted-status transition logic continue unchanged — verified by `test_adr_007_post_hoc_calibration_unchanged_under_adr_014` (preservation).
- **Fitness:** When ADR-016 is rejected, ADR-014's in-process layer operates on L1-internal trajectory data only; the verdict trichotomy continues to function with the in-layer feature set — verified by `test_verdict_computation_works_without_signal_channel` (unit).

### Module: Plexus Adapter

**Purpose:** Mediates all Plexus interaction with no-op fallbacks when Plexus is absent.
**Provenance:** ADR-009, ADR-010; AS-4, AS-8; domain concepts Ingestion, Enrichment, Context Injection (data flow), Routing Decision (persistence), Quality Signal (persistence).
**Owns:** Ingestion (push to Plexus); Enrichment (invocation; Plexus performs the actual enrichment); Query (knowledge graph query mechanics); Record (outcome persistence); no-op fallback semantics when Plexus is absent.
**Depends on:** (external — Plexus lib)
**Depended on by:** Orchestrator Tool Dispatch (query_knowledge, record_outcome), Bootstrapping Pipeline (ingestion), Serving Layer's `resolve_session_start_context` (Phase 2 only), Calibration Gate (persistence of Quality Signals).
**Inversion note:** The Operator's mental model is "Plexus is a lib I enable; llm-orc pushes to it." The boundary preserves that — the Adapter is the single place Plexus-aware code lives, so the rest of the system sees a tool interface regardless of Plexus state. This is what makes AS-8 structurally enforceable.

### Module: Bootstrapping Pipeline

**Purpose:** Pushes library source material into Plexus as a deliberate operator operation.
**Provenance:** AS-4; ADR-010; DISCOVER FF #9 and #14; domain concept Bootstrapping.
**Owns:** Bootstrapping (concept and action); operator-triggered batch ingestion flow.
**Depends on:** Plexus Adapter, Ensemble Engine (reads library via existing config manager).
**Depended on by:** (operator — CLI/web trigger)

### Module: Orchestrator Configuration

**Purpose:** Loads and resolves the orchestrator's per-session configuration surface.
**Provenance:** ADR-005 (Budget defaults), ADR-008 (Autonomy defaults), ADR-009 (Plexus enablement), ADR-011 (Orchestrator Model Profile), ADR-012 (Conversation Compaction thresholds), ADR-015 (per-skill tier defaults), ADR-017 (structural validation guard pattern set).
**Owns:** Per-session config resolution; operator-set bounds on per-request overrides; the four Conversation Compaction thresholds (50K-character Layer 0 trigger, 60-minute Layer 2 idle window, 12,288-token Layer 3 cap, 3-failure Layer 4 circuit-breaker); per-skill tier-defaults configuration (8 skills × 2 tiers = 16 Model Profile slots, with sharing permitted); structural validation guard pattern set (default + operator extensions).
**Depends on:** (project config manager — existing)
**Depended on by:** Serving Layer; values flow downward at construction time to L2 modules (Conversation Compaction thresholds; Tier-Escalation Router tier defaults; Tool Dispatch pattern set).

### Module: Conversation Compaction *(new in Cycle 4 per ADR-012)*

**Purpose:** Runs the five-layer cheapest-first compaction pipeline at orchestrator turn boundaries.
**Provenance:** ADR-012; essay 005 §"Long-Horizon Reliability Infrastructure"; Khanal arXiv:2603.29231; Chroma 2025; Liu et al. TACL 2023; Anthropic published Claude Code specification; codebase precedent commit `9f86d0b` for typed-error coupling.
**Owns:** Conversation Compaction (concept; re-allocated from Orchestrator Runtime ownership in v2.0); the five compaction layers — Layer 0 persist-large-tool-results (>50K char to filesystem with 2 KB preview), Layer 1 cache-edit, Layer 2 idle-expiry (60-minute default), Layer 3 free summary via nine-section session-notes template (12,288-token cap; deterministic logic; zero LLM cost), Layer 4 LLM-summary via configured summarizer ensemble; the four operationally-tunable thresholds; circuit-breaker state (per-session; auto-reset at session start); the nine-section template structure; the typed `compaction_layer_4_failure` error.
**Depends on:** Ensemble Engine (Layer 4 invokes summarizer ensemble via existing `EnsembleExecutor.execute`; distinct from Result Summarizer Harness's AS-7 ensemble-output summarization); filesystem (Layer 0 persistence root; operator-configurable). No L3 dependencies — receives messages array as parameter from Runtime.
**Depended on by:** Orchestrator Runtime (invokes compaction at turn boundaries; FC-4 amendment adds `conversation_compaction` to Runtime's allowed import set alongside `budget_controller` and `orchestrator_tool_dispatch`).
**Inversion note:** Operator's mental model — "the orchestrator stays coherent over long sessions because something compacts the conversation." Module name and owned thresholds match; layer ordering is internal mechanics. The five-layer cheapest-first ordering is the load-bearing design property (per ADR-012's published reference; reversal produces the failure mode the pattern was designed to replace).

**Fitness:**
- **Fitness:** Across a multi-turn fixture session that exceeds the configured context threshold by ≥ 200%, the orchestrator's per-turn LLM call receives a context whose token-count stays at or below threshold for ≥ 95% of iterations — verified by `test_compaction_holds_context_below_threshold_across_long_session` (integration).
- **Fitness:** Each compaction invocation applies the five layers in order (0 → 1 → 2 → 3 → 4) and short-circuits on the first layer that brings context below threshold; Layer N+1 fires only if Layers 0..N together cannot — verified by static AST inspection of the compaction main loop plus `test_layer_4_fires_only_after_layers_0_3_attempted` (integration).
- **Fitness:** Layer 4 LLM-summary failure increments a per-session circuit-breaker; the third consecutive failure suspends Layer 4 for the rest of the session and emits a typed `compaction_layer_4_failure` error to operator-readable storage with `recovery_action_required="operator_intervention_required"` — verified by `test_layer_4_circuit_breaker_after_three_consecutive_failures` (unit).
- **Fitness:** Layer 4 circuit-breaker state resets to active at session-start without operator intervention — verified by `test_circuit_breaker_resets_at_session_start` (unit).

### Module: Tier-Escalation Router *(new in Cycle 4 per ADR-015; extended in Cycle 4 architect-gate close per ADR-018)*

**Purpose:** Selects per-dispatch Model Profile (cheap-tier or escalated-tier) for `invoke_ensemble` based on the dispatched ensemble's Topaz skill metadata and the Calibration Gate's verdict. **Per ADR-018:** also fires a periodic out-of-band audit dispatch (analog of ADR-016 mechanism (d)) on the verdict→router edge to detect routing-vs-tier-config drift.
**Provenance:** ADR-015; ADR-018 (amending ADR — (d)-analog audit dispatch responsibility); ADR-014 (verdict producer); ADR-011 (compatibility envelope — orchestrator's own LLM session-boundary scope preserved); ADR-016 mechanism (d) (structural precedent for the audit dispatch via Spike β analytical transfer audit, research log `005h-`); essay 005 §"ADR candidate #4"; OI-MAS arXiv:2601.04861; Topaz paper; SC-MAS, MasRouter literature evidence base.
**Owns:** Tier-escalation router (concept); Per-skill tier default (concept; the operator-configured pair of Model Profiles per Topaz skill); Topaz skill profile (schema owner — the YAML metadata field `topaz_skill: TopazSkill`); the verdict→tier mapping (Proceed → cheap-tier; Reflect → escalated-tier; Abstain → typed `escalation_bypass` error); the typed `missing_skill_metadata` error; the typed `escalation_bypass` error. **Per ADR-018:** the (d)-analog audit dispatch — trigger (every 100 verdict consumptions or 24 wall-clock hours, whichever first; operationally tunable); three drift criteria at quantitative-threshold level — verdict-distribution shift (±15% between consecutive windows), escalation-vs-outcome correlation drift (default: escalation must produce at least +5pp outcome improvement over the audit window to be interpretable as a tier-configuration signal rather than routing-noise — this is the Sub-Q6 evidence surface), bypass-rate trend (default: +25% relative-rate increase per window); typed verdict trichotomy (no drift / advisory / severe drift); operator action surface for advisory drift (diagnostic + parameter-tuning recommendation against the 16-Model-Profile-slot configuration surface); severe-drift fail-safe mode (route-all-to-escalated until operator review); asynchronous-operator-review dynamic.
**Depends on:** Calibration Gate (consumes calibration verdict via `verdict_for(session_id, ensemble_name, dispatch_context)`); Ensemble Engine (reads ensemble YAML metadata via existing config manager; observes dispatched ensemble's outcome for the (d)-analog audit's escalation-vs-outcome correlation criterion). Per-skill tier defaults and (d)-analog audit thresholds flow from Orchestrator Configuration at construction time (no L3 import).
**Depended on by:** Orchestrator Tool Dispatch (interposed on `invoke_ensemble` between dispatch entry and `EnsembleExecutor.execute` — selected tier flows into ensemble execution).
**Inversion note:** Operator's mental model — "I configure per-skill tier defaults at deployment; ensembles declare their primary skill; the system routes by skill+confidence; an audit periodically checks whether routing matches tier-configuration intent." Module name and configuration surface match. Per-ensemble tier alternatives — rejected per ADR-015 (and confirmed by Spike α, 2026-05-11) — would have served *developer convenience* (granularity) over operator's mental model (per-skill defaults).
**Inherited bounding-mechanism properties (per ADR-018, from Spike β analytical transfer audit 2026-05-11):** Three of ADR-016's five bounding mechanisms hold for the verdict→router edge by inheritance from existing infrastructure rather than by new specification: **(a) fresh-context isolation** holds by construction (`select_tier` is a stateless pure function); **(b) time-decay windowing** is operationalized one layer upstream at the Calibration Gate per ADR-014 (the bias-bound for the verdict→router feedback path exists in the producer); **(e) read-only structural validation** is satisfied by the FC-17 typed-error infrastructure (`escalation_bypass`, `missing_skill_metadata`). One mechanism — **(c) categorical anchors** — is structurally inapplicable at this edge (the Router consumes verdicts, not deterministic-tool outputs; (c)'s anchor substrate is at L0). The (d)-analog audit dispatch is the load-bearing addition.
**Falsification trigger (per ADR-018, inherits ADR-016's elaboration-by-evidence discipline):** if WP-G4's BUILD work finds that the (d)-analog audit dispatch cannot be operationalized within the Tier-Escalation Router module's responsibility (e.g., it requires its own top-level module orthogonal to L0–L3, or it requires bidirectional coupling with the Calibration Gate that violates the read-only verdict-consumption contract), the elaboration-by-evidence framing commitment is invalidated for ADR-018; the (d)-analog spec re-opens, and OQ #14 partial closure for the L1→L2 stage reverts to "BUILD evidence will inform" with Sub-Q6 also re-opening.

**Fitness:**
- **Fitness:** The orchestrator's own Model Profile remains constant for the entire Session under any verdict-driven tier escalation; only the dispatched task's tier varies — verified by `test_orchestrator_profile_unchanged_under_tier_escalation` (preservation of FC-13).
- **Fitness:** The orchestrator's tool-call API (`invoke_ensemble`) signature is unchanged under tier escalation; the orchestrator's reasoning surface receives the same shape of tool-call result regardless of which tier was dispatched — verified by `test_invoke_ensemble_api_unchanged_under_tier_escalation` (integration).
- **Fitness:** The verdict-to-action mapping is deterministic: Proceed → cheap-tier dispatch; Reflect → escalated-tier dispatch; Abstain → `escalation_bypass` typed error — verified by `test_verdict_to_tier_mapping_is_deterministic` (unit).
- **Fitness (per ADR-018):** The Router consumes each verdict in a fresh stateless context; no prior verdict state is carried forward through `select_tier` — verified by `test_select_tier_is_stateless_pure_function` (unit). (Inherits ADR-016 mechanism (a) by construction.)
- **Fitness (per ADR-018):** All Router error surfaces (`escalation_bypass`, `missing_skill_metadata`) derive from `LlmOrcStructuralError` with the four common fields per FC-17 — verified by FC-17's existing static class-hierarchy walk. (Inherits ADR-016 mechanism (e) by infrastructure reuse.)
- **Fitness (per ADR-018):** The (d)-analog audit dispatch fires at the configured trigger frequency (default every 100 verdict consumptions or 24 wall-clock hours, whichever first); on severe drift the audit's verdict triggers fail-safe mode (route-all-to-escalated until operator review) plus operator notification — verified by `test_d_analog_audit_dispatch_fires_at_trigger_and_severe_drift_activates_fail_safe` (integration).

**Direction-not-constraint note (per ADR-076, extended for ADR-018):** "Routing-vs-tier-config drift detection" is a *direction* the (d)-analog audit dispatch collectively optimizes toward — the three drift criteria (verdict-distribution shift, escalation-vs-outcome correlation drift, bypass-rate trend) are individually testable at unit/integration level, but whether the aggregate drift-detection property holds under real-deployment dynamics is exactly what first-deployment evidence answers. Spike β's analytical transfer audit established the structural properties; first-deployment evidence on the cycle's North-Star benchmark closes Sub-Q6 in conjunction with closing this audit dispatch's operational-validation gate.

### Module: Calibration Signal Channel *(new in Cycle 4 per ADR-016; CONDITIONAL ACCEPTANCE — first-deployment evidence pending)*

**Purpose:** Carries read-only calibration data upward from L0 (Ensemble Engine outputs) to L1 (Calibration Gate dispatch decisions), enforcing the five bounding mechanisms.
**Provenance:** ADR-016; essay 005 §"ADR candidate #6"; Khanal arXiv:2603.29231 (cycle/scale risk that motivates bounding mechanisms (b) and (d)); CAAF arXiv:2604.17025 (prompt-engineering-artifact finding); Wisdom and Delusion of LLM Ensembles arXiv:2510.21513 (mechanism (c) precedent); RDD methodology susceptibility-snapshot pattern (mechanism (d) structural transfer); architectural-isolation pattern from RDD methodology tooling (mechanism (a) precedent — methodology-level rather than codebase-internal); commit `9f86d0b` typed-error pattern (mechanism (e) precedent).
**Owns:** the read-only signal channel boundary (the only edge in the dependency graph that points L0→L1 — the explicit narrow exception ADR-016 amends ADR-002 to permit); the typed signal-data schema (trajectory features per ADR-014's HTC specification, dispatch outcomes, deterministic-tool-output anchors per mechanism (c)); the five bounding mechanisms — (a) fresh-context isolation in the consumer, (b) time-decay windowing (60-min/100-signal dual-bound, linear decay; smaller-window configurations track better than default per spike (b) finding — operational tuning territory), (c) categorical anchors via deterministic-tool-output (when ensemble has script-model slot; ensemble-composition-conditional; anchor strength scales with output verifiability — binary-verifiable strongest, interpretable-prose weakest), (d) periodic out-of-band audit dispatch (default every 100 verdicts or 24 wall-clock hours; three drift-detection criteria — verdict skew, outcome divergence, signal-to-verdict correlation drift — at quantitative-threshold level), (e) read-only structural validation at the consumer (schema validation; mismatch produces typed `malformed_signal` error and is dropped from verdict computation); the audit verdict trichotomy (no drift / advisory / severe drift) — severe drift triggers fail-safe mode (calibration verdicts default to Reflect-or-Abstain) plus operator notification; the operator action surface for advisory drift (diagnostic + parameter-tuning recommendation; operator approves or overrides at next session boundary); the asynchronous-operator-review dynamic for runtime audit verdicts; the conditional-acceptance status's first-deployment-evidence trigger (per ADR-016 §"Concrete monitoring specification") with sweep responsibility at each cycle that exercises the channel.
**Depends on:** Ensemble Engine (consumes typed signal data from L0 dispatch outputs via observer-pattern callback registration — *this is the upward edge ADR-002 amendment permits, read-only, calibration-data-only, structurally typed at boundary*).
**Depended on by:** Calibration Gate (the L1 consumer reads channel signals — within the consumer's fresh evaluation context per mechanism (a)).
**Inversion note:** The channel is intrinsically an architectural concept (the upward-edge exception ADR-016 amends ADR-002 to permit). The operator-facing surface is audit verdicts and parameter-tuning recommendations — operator interacts with the audit verdict, not the channel internals. Naming the module *Signal Channel* leans architectural; the operator-facing visibility comes through the audit dispatch's diagnostic plus the periodic operator review of advisory drift recommendations.
**Falsification trigger (load-bearing carry-forward):** if BUILD or first-deployment evidence finds that mechanism (b) windowing or mechanism (d) audit dispatch cannot be operationalized within ADR-002's L0-L3 structure (e.g., they require a top-level module orthogonal to the four-layer architecture), the elaboration-by-evidence framing commitment is invalidated, the reorganization branch re-opens, and ADR-016 is re-deliberated with reorganization on the table.

**Fitness:**
- **Fitness:** Every signal traversing the channel is structurally typed at the channel boundary; signals that fail schema validation produce `malformed_signal` typed error and are dropped from the verdict computation — verified by `test_channel_validates_signal_schema_at_boundary` (unit).
- **Fitness:** Each calibration verdict computation runs in a fresh evaluation context — no signal data from prior verdicts is carried forward through the consumer's context; influence on the next verdict is only through the time-decay-windowed feature aggregation — verified by `test_consumer_runs_in_fresh_context_no_carryover` (unit).
- **Fitness:** The channel rejects any upward write attempt at the structural level; no L0 state is mutated through the channel boundary by L1 callers — verified by `test_channel_is_read_only_no_l1_to_l0_writes` (structural).
- **Fitness:** Mechanism (d)'s periodic out-of-band audit dispatch fires at the configured trigger frequency (default every 100 verdicts or 24 wall-clock hours, whichever first); on severe drift the audit's verdict triggers fail-safe mode (calibration verdicts default to Reflect-or-Abstain) — verified by `test_audit_dispatch_fires_at_trigger_and_severe_drift_activates_fail_safe` (integration).

**Direction-not-constraint note (per ADR-076):** "Load-bearing bounding of feedback-bias compounding" is a *direction* the bounding mechanisms collectively optimize toward — bounded under synthetic-data spike (b)/(d), but whether the bound holds under real-deployment dynamics is exactly what first-deployment evidence answers (the conditional-acceptance status's load-bearing question). The five mechanisms are individually testable; the *aggregate* bias-bound property is testable only by first-deployment evidence on the cycle's North-Star benchmark.

---

## Responsibility Matrix

Every concept and action from the agentic-serving domain model and the touched project-level concepts maps to exactly one owning module. Inherited project-level concepts that are not touched by agentic serving live with the existing Ensemble Engine (listed once at the bottom).

| Domain Concept / Action | Owning Module | Provenance |
|------------------------|---------------|------------|
| Orchestrator Agent | Orchestrator Runtime | Domain model; ADR-001 |
| Session | Session Registry | Domain model; ADR-005, ADR-008, ADR-011 |
| Serving Layer | Serving Layer | Domain model; ADR-002 |
| Orchestrator Tool | Orchestrator Tool Dispatch | Domain model; ADR-003 |
| Routing Decision | Orchestrator Runtime (generation); Plexus Adapter (persistence) | Domain model; ADR-009 |
| Dynamic Invocation | Orchestrator Tool Dispatch | AS-1; domain model |
| Composition | Composition Validator | Domain model; AS-2, ADR-006 |
| Primitive | Ensemble Engine (existing — role played by existing concepts) | AS-6; ADR-006 |
| Library | Ensemble Engine (existing — config manager) | Domain model |
| Budget | Budget Controller | AS-3; ADR-005 |
| Result Summarization | Result Summarizer Harness | AS-7; ADR-004 |
| Conversation Compaction | Orchestrator Runtime | Essay 001 §Context Management; AS-7 |
| Context Injection | Serving Layer (function `resolve_session_start_context`) | ADR-009 (structurally reserved via typed function signature) |
| Ingestion | Plexus Adapter | AS-4; ADR-010 |
| Enrichment | Plexus Adapter (invocation); Plexus lib (performance) | AS-4; ADR-010 |
| Quality Signal | Calibration Gate (generation); Plexus Adapter (persistence when active) | AS-5; ADR-007 |
| Stabilization | Plexus Adapter (emergent; surfaced via queries) | AS-5; ADR-007 |
| Bootstrapping | Bootstrapping Pipeline | ADR-010; DISCOVER FF #9, #14 |
| Autonomy Level | Autonomy Policy | ADR-008 |
| Calibration | Calibration Gate | ADR-007 |
| Route (action) | Orchestrator Runtime | Domain model |
| Compose (action) | Orchestrator Tool Dispatch → Composition Validator | ADR-006 |
| Invoke (Dynamic) (action) | Orchestrator Tool Dispatch → Ensemble Engine | AS-1 |
| Summarize (action) | Result Summarizer Harness | ADR-004 |
| Compact (action) | Orchestrator Runtime | Essay 001 §Context Management |
| Inject (action) | Serving Layer (function) | ADR-009 |
| Ingest (action) | Plexus Adapter | ADR-010 |
| Enrich (action) | Plexus Adapter (invocation) | ADR-010 |
| Query (action) | Orchestrator Tool Dispatch → Plexus Adapter | ADR-009 |
| Record (action) | Orchestrator Tool Dispatch → Plexus Adapter | ADR-009 |
| Calibrate (action) | Calibration Gate | ADR-007 |
| Stabilize (action) | (emergent — not owned by a single module) | AS-5 |
| Bootstrap (action) | Bootstrapping Pipeline | ADR-010 |
| Ensemble, Agent, AgentConfig, Model Profile, Inline Model, Dependency, Phase, Fan-Out, Input Key, Ensemble Reference, Ensemble Reference Graph, Depth, Depth Limit, Artifact, Child Executor, Immutable Infrastructure, Mutable State, Agent Discriminator; Load, Validate, Discriminate, Execute, Dispatch, Recurse, Fan Out, Gather, Select, Detect Cycles, Check Depth, Merge Profile | Ensemble Engine (existing) | Project-level domain model |
| **Conversation Compaction** *(re-allocated from Orchestrator Runtime)* | **Conversation Compaction** *(new in Cycle 4)* | ADR-012; essay 005 §"Long-Horizon Reliability" |
| Calibration verdict | Calibration Gate (extended) | ADR-014; domain-model §Concepts |
| Tier-escalation router | Tier-Escalation Router (new) | ADR-015; domain-model §Concepts |
| Per-skill tier default | Tier-Escalation Router (new) | ADR-015; domain-model §Concepts |
| Topaz skill profile | Tier-Escalation Router (schema owner); Ensemble Engine (existing — YAML metadata field parsing via existing config manager) | ADR-015; domain-model §Methodology Vocabulary |
| Trajectory feature | Calibration Gate (extended) | ADR-014; domain-model §Methodology Vocabulary |
| In-process trajectory-level calibration | Calibration Gate (extended) | ADR-014; domain-model §Methodology Vocabulary |
| Structured-handoff artifact | Session Registry (extended) | ADR-013; domain-model §Concepts |
| Write-gate validation | Session Registry (extended) | ADR-013; domain-model §Concepts |
| Cluster determination at session-start | Session Registry (extended) | ADR-013; domain-model §Concepts |
| Tool-call structural validation guard | Orchestrator Tool Dispatch (extended) | ADR-017; domain-model §Concepts |
| Phantom tool-call (failure mode the guard detects) | Orchestrator Tool Dispatch (extended) | ADR-017; domain-model §Concepts |
| Cross-layer calibration channel | Calibration Signal Channel (new; conditional acceptance) | ADR-016; domain-model §Methodology Vocabulary |
| Bounding mechanisms (a)–(e) | Calibration Signal Channel (new; conditional acceptance) — owns mechanism implementations | ADR-016; domain-model §Methodology Vocabulary |

**Coverage check (post-Cycle 4):** Every scoped concept (19 v2.0 + 12 Cycle 4 additions across §Concepts and §Methodology Vocabulary = 31 total) is allocated. Every action (13; unchanged in Cycle 4) is allocated to its existing owner. Every touched project-level concept stays with the existing Ensemble Engine. Stabilize is marked emergent — it is not owned by a module because AS-5 defines it as an emergent property of accumulated Quality Signals, not an explicit action. The Cycle 4 concept additions add 12 rows; the *Conversation Compaction* concept moves from Orchestrator Runtime to the new Conversation Compaction module. Bounding mechanisms (a)–(e) are treated as a coherent set per ADR-016's framing rather than five separate matrix entries.

---

## Dependency Graph

**Directed edges (A → B means A imports/calls B):**

```
Serving Layer ──────────────────────────────────▶ Session Registry
Serving Layer ──────────────────────────────────▶ Orchestrator Configuration
Serving Layer ──────────────────────────────────▶ Orchestrator Runtime
Serving Layer ───── (Phase 2 only) ─────────────▶ Plexus Adapter
Session Registry ───────────────────────────────▶ Ensemble Engine (profile lookup)
Orchestrator Runtime ───────────────────────────▶ Budget Controller
Orchestrator Runtime ───────────────────────────▶ Orchestrator Tool Dispatch
Orchestrator Runtime ───────────────────────────▶ Conversation Compaction          (Cycle 4 — new)
Budget Controller ──────────────────────────────▶ Session Registry
Orchestrator Tool Dispatch ─────────────────────▶ Ensemble Engine
Orchestrator Tool Dispatch ─────────────────────▶ Composition Validator
Orchestrator Tool Dispatch ─────────────────────▶ Plexus Adapter
Orchestrator Tool Dispatch ─────────────────────▶ Autonomy Policy
Orchestrator Tool Dispatch ─────────────────────▶ Calibration Gate
Orchestrator Tool Dispatch ─────────────────────▶ Result Summarizer Harness
Orchestrator Tool Dispatch ─────────────────────▶ Tier-Escalation Router            (Cycle 4 — new)
Result Summarizer Harness ──────────────────────▶ Ensemble Engine
Conversation Compaction ────────────────────────▶ Ensemble Engine                   (Cycle 4 — new; Layer 4 summarizer)
Composition Validator ──────────────────────────▶ Ensemble Engine (shared public validator)
Autonomy Policy ────────────────────────────────▶ Session Registry
Calibration Gate ───────────────────────────────▶ Ensemble Engine
Calibration Gate ───────────────────────────────▶ Plexus Adapter
Calibration Gate ───────────────────────────────▶ Calibration Signal Channel        (Cycle 4 — new; conditional on ADR-016)
Tier-Escalation Router ─────────────────────────▶ Calibration Gate                  (Cycle 4 — new; verdict consumer + ADR-018 (d)-analog audit edge: composes stateless consumer (inherited a) + upstream windowing (inherited b via ADR-014) + typed validation (inherited e via FC-17) + periodic audit (novel d-analog))
Tier-Escalation Router ─────────────────────────▶ Ensemble Engine                   (Cycle 4 — new; Topaz skill metadata reader + ADR-018 outcome observer for the (d)-analog escalation-vs-outcome correlation criterion)
Bootstrapping Pipeline ─────────────────────────▶ Plexus Adapter
Bootstrapping Pipeline ─────────────────────────▶ Ensemble Engine
Plexus Adapter ─────────────────────────────────▶ (external — Plexus lib)

╔══════════════════════════════════════════════════════════════════════════════════╗
║  Ensemble Engine ────read-only────▶ Calibration Signal Channel                  ║
║  (Cycle 4 — new; the load-bearing upward exception ADR-016 amends ADR-002 to    ║
║   permit; read-only, calibration-data-only, structurally typed at boundary,     ║
║   gated by 5 bounding mechanisms specified in ADR-016)                          ║
╚══════════════════════════════════════════════════════════════════════════════════╝
```

**Layering (inner → outer; post-Cycle 4):**

| Layer | Modules | Rule |
|-------|---------|------|
| L0 — Core (existing) | Ensemble Engine | May not depend on any agentic-serving module — **with one exception per ADR-016:** the Ensemble Engine emits read-only calibration signals to the Calibration Signal Channel (L1) via observer-pattern callback registration. This is the upward edge ADR-016 amends ADR-002 to permit. |
| L1 — Domain Policy | Composition Validator, Budget Controller, Autonomy Policy, Calibration Gate, Plexus Adapter, **Calibration Signal Channel** *(new in Cycle 4; conditional acceptance)* | May depend on L0 only |
| L2 — Runtime | Result Summarizer Harness, Orchestrator Tool Dispatch, Orchestrator Runtime, **Conversation Compaction** *(new in Cycle 4)*, **Tier-Escalation Router** *(new in Cycle 4)* | May depend on L0 and L1 |
| L3 — Entry | Serving Layer, Session Registry, Bootstrapping Pipeline, Orchestrator Configuration | May depend on L0, L1, and L2 |

**No cycles.** Verified by static inspection over all 28 edges (21 v2.0 + 7 Cycle 4 additions): every edge points from a higher layer to a same-or-lower layer, except the single ADR-016 exception (Ensemble Engine → Calibration Signal Channel; read-only). The only intra-layer dependencies are within L3 (Serving Layer → Session Registry; Serving Layer → Orchestrator Configuration) and within L2 (Runtime → Conversation Compaction; Tool Dispatch → Tier-Escalation Router) — none form cycles. The Calibration Signal Channel is a new sink for L0 signals; the verdict produced by Calibration Gate flows downward into Tier-Escalation Router and Tool Dispatch — no signal path returns to L0.

**Fan-out warnings (post-Cycle 4).** Orchestrator Tool Dispatch has seven outbound edges: five correspond to the closed tool set (Ensemble Engine, Composition Validator, Plexus Adapter, Autonomy Policy, Calibration Gate) — intentional per ADR-003; the fan-out *is* the closed tool set. The sixth edge (Result Summarizer Harness) is a cross-cutting interposition on the `invoke_ensemble` return path — orthogonal to the closed set, structurally placed on the Tool Dispatch side rather than the Runtime side so the Runtime stays unaware of summarization (per Amendment #3). The seventh edge (Tier-Escalation Router; new in Cycle 4) is another orthogonal interposition on the `invoke_ensemble` path — selecting Model Profile pre-dispatch based on calibration verdict per ADR-015. Ensemble Engine remains the highest-fan-in module (six agentic-serving modules depend on it post-Cycle 4: the existing five plus the new Tier-Escalation Router and Conversation Compaction reading via callback for signal channel emission). The pattern stays consistent with Layer 3 (Ensemble Engine) being the single shared execution substrate.

**FC-2 / FC-3 amendment for ADR-016 exception.** When FC-2 is implemented as an automated AST-based per-module import layering check (per WP-B4), it must recognize the `Ensemble Engine → Calibration Signal Channel` edge as an annotated allowed exception. The annotation is "read-only, calibration-data-only" and is enforced via the channel's owned schema validation (mechanism (e)) at runtime. FC-3 (cycle detection) computes over the full edge set; the new edge does not introduce a cycle, so FC-3 passes without amendment.

---

## Integration Contracts

### Serving Layer → Session Registry

**Protocol:** Synchronous function call at request entry.
**Shared types:** `SessionIdentity` (derivation-method-agnostic: may be client-supplied `user` field, hash of initial message prefix, or explicit session id header); `SessionState` (current Budget state, Autonomy Level, Calibration state if required).
**Error handling:** A request that fails Session identity resolution is treated as a new Session (cold start). Identity-resolution failures are not client errors.
**Owned by:** Session Registry defines the contract.

### Serving Layer → Orchestrator Runtime

**Protocol:** Asynchronous streaming; the Runtime yields SSE chunks.
**Shared types:** `SessionContext` (messages, tools array, session state); `OrchestratorChunk` (one of: content delta, internal tool call invocation-in-flight, internal tool call result, client tool call in final turn, completion).
**Error handling:** An Orchestrator Runtime exception becomes an SSE `error` chunk. The Runtime guarantees no partial state persists to the Session after a thrown exception unless explicitly committed.
**Owned by:** Orchestrator Runtime defines the contract.

### Orchestrator Runtime → Budget Controller

**Protocol:** Synchronous pre-iteration check.
**Shared types:** `BudgetCheck` (pass, or a typed exhaustion reason: turn or token).
**Error handling:** A failed check raises a typed `BudgetExhausted` event that the Runtime converts into a graceful session termination with explicit exhaustion signaling (ADR-005).
**Owned by:** Budget Controller defines the contract.

### Orchestrator Runtime → Orchestrator Tool Dispatch

**Protocol:** Synchronous tool invocation; returns the summarized/gated result.
**Shared types:** `InternalToolCall` (tool name from the fixed five, arguments); `ToolCallResult` (summarized result or typed error).
**Error handling:** A tool name outside the fixed five returns a `ToolCallResult` error (not an exception). The Runtime passes the error back to the orchestrator LLM as an observation (scenarios §Invocation outside the tool set).
**Owned by:** Orchestrator Tool Dispatch defines the contract.

### Orchestrator Tool Dispatch → Autonomy Policy

**Protocol:** Synchronous gate check before every tool dispatch.
**Shared types:** `AutonomyGateInput` (tool name, tool arguments, current Session Autonomy Level, tool-user persona flag); `AutonomyGateOutput` (allow, require_approval, or deny).
**Error handling:** `require_approval` surfaces an event to the operator via the visibility surface; `deny` is returned as a tool error to the orchestrator.
**Owned by:** Autonomy Policy defines the contract.

### Orchestrator Tool Dispatch → Ensemble Engine

**Protocol:** Existing `EnsembleExecutor.execute` (Layer 3 API, unchanged). Wrapped by Result Summarizer Harness on the return path.
**Shared types:** (existing Layer 3 types)
**Error handling:** Invariant 14 applies — runtime errors recorded; structural errors raised at load time.
**Owned by:** Ensemble Engine (existing).

### Orchestrator Tool Dispatch → Composition Validator

**Protocol:** Synchronous validation at composition time.
**Shared types:** `CompositionRequest` (proposed ensemble config, library search_dirs); `CompositionResult` (accepted and stored to local tier, or typed validation error naming the specific invariant violated).
**Error handling:** Validation failures are returned to the orchestrator as tool errors. No partial or pending ensemble state persists (ADR-006; scenarios §Composition that would introduce a reference-graph cycle).
**Owned by:** Composition Validator defines the contract.

### Composition Validator ↔ Ensemble Engine (shared validator routine)

**Protocol:** Shared public function call (`validate_ensemble_reference_graph` — public API in `core/config/ensemble_config.py`).
**Shared types:** (existing `EnsembleConfig`, `AgentConfig` union, search_dirs list).
**Error handling:** Raises `ValueError` with cycle description on failure; returns `None` on success.
**Owned by:** Ensemble Engine owns the shared routine after extraction from private helpers (WP-A). Both load-time (`EnsembleLoader.load_from_file`, `list_ensembles`) and composition-time callers use the same function (scenarios refactor 1-3; regression scenario "shared single routine").

### Orchestrator Tool Dispatch → Plexus Adapter

**Protocol:** Synchronous for `query_knowledge`; asynchronous-with-immediate-ack for `record_outcome`.
**Shared types:** `QueryRequest`, `QueryResult` (possibly empty — AS-8); `OutcomeRecord`, `RecordAck`.
**Error handling:** When Plexus is absent, both tools return well-formed empty/ack responses — no exception surfaces (scenarios §query_knowledge returns empty gracefully, §record_outcome writes asynchronously).
**Owned by:** Plexus Adapter defines the contract.

### Orchestrator Tool Dispatch → Calibration Gate

**Protocol:** Synchronous pre-invoke check and post-invoke Quality Signal attachment; interposed transparently on `invoke_ensemble` for ensembles in calibration.
**Shared types:** `CalibrationState` (in_calibration, trusted); `QualitySignal` (positive, negative, absent).
**Error handling:** A failing calibration check does not prevent invocation (ADR-007 clause 2) — it attaches the signal and returns the result normally.
**Owned by:** Calibration Gate defines the contract.

### Result Summarizer Harness → Ensemble Engine

**Protocol:** The Harness invokes a summarizer ensemble (configured primitive) via `EnsembleExecutor.execute`.
**Shared types:** (existing Layer 3 types)
**Error handling:** A summarizer failure is a tool failure — the original ensemble result is still persisted to its artifact (Invariant 9), but the orchestrator receives a typed summarization error as a tool result. Raw-output escape hatch (ADR-004) bypasses the Harness entirely for flagged ensembles.
**Owned by:** Result Summarizer Harness defines the contract.

### Serving Layer → `resolve_session_start_context` (internal function)

**Protocol:** Synchronous session-start hook; returns optional system-prompt augmentation. The hook is a typed function, not a module; call site and signature are the structural reservation.
**Shared types:** `SessionContext` (Session identity and state at start); `list[PromptFragment]` (empty in Phase 1; populated from Plexus Adapter in Phase 2).
**Error handling:** Injection failure in Phase 2 falls through to no injection rather than failing the session start.
**Owned by:** Serving Layer owns the function and its contract. **Phase 1 status:** returns `[]` unconditionally.

### Bootstrapping Pipeline → Plexus Adapter, Ensemble Engine

**Protocol:** Batch operation: reads from the library via the config manager, pushes file content to Plexus via the Adapter's ingestion path.
**Shared types:** `LibraryArtifactStream`, `IngestionAck` (AS-4: source material only — never LLM summaries).
**Error handling:** Ingestion is non-blocking; per-artifact failures are logged and the batch continues.
**Owned by:** Bootstrapping Pipeline defines the contract.

### Autonomy Policy → Session Registry; Calibration Gate → Session Registry; Budget Controller → Session Registry

**Protocol:** Read-only synchronous queries for Session state.
**Shared types:** `SessionState` (subsets scoped to consumer).
**Error handling:** Missing Session (identity unresolved) is treated as cold-session defaults for every consumer.
**Owned by:** Session Registry defines the contract.

### Orchestrator Runtime → Conversation Compaction *(new in Cycle 4 per ADR-012)*

**Protocol:** Synchronous turn-boundary call. Runtime invokes `compact(messages, session_state) -> CompactedContext` at each iteration boundary before passing messages to the LLM.
**Shared types:** `ConversationMessages` (the orchestrator's accumulated messages); `SessionState` (subset: turn count, token spend; needed for token-budget targeting); `CompactedContext` (the message array possibly with persisted-tool-result references substituted for raw payloads, possibly with cache-edits applied, possibly with a session-notes block prepended).
**Error handling:** Layer 4 LLM-summary failure does not raise to the Runtime — the circuit-breaker tracks failures and produces a typed `compaction_layer_4_failure` error in operator-readable diagnostics (logged); compaction returns the best-available reduction from Layers 0–3. After three consecutive Layer 4 failures, Layer 4 is suspended for the rest of the session; circuit-breaker auto-resets at session start.
**Owned by:** Conversation Compaction defines the contract.

### Conversation Compaction → Ensemble Engine *(new in Cycle 4 per ADR-012)*

**Protocol:** Layer 4 invokes a configured summarizer ensemble via `EnsembleExecutor.execute` with a context-summarization prompt. Distinct from Result Summarizer Harness's invocation (which targets ensemble outputs, AS-7) — this targets the conversation history.
**Shared types:** existing Layer 3 types (`EnsembleConfig`, ensemble result dict).
**Error handling:** ensemble-execution exception → Layer 4 failure (counted toward circuit-breaker). The conversation context is returned at the best Layer 0–3 reduction.
**Owned by:** Conversation Compaction defines the contract; Ensemble Engine API is unchanged.

### Orchestrator Tool Dispatch → Tier-Escalation Router *(new in Cycle 4 per ADR-015)*

**Protocol:** Synchronous interposition on `invoke_ensemble`. Tool Dispatch passes the dispatched ensemble's name and the calibration verdict (already retrieved from Calibration Gate) to `select_tier(ensemble_name, verdict) -> TierSelection`.
**Shared types:** `TierSelection` (one of: `CheapTier(model_profile)`, `EscalatedTier(model_profile)`, `Bypass(reason)`); `CalibrationVerdict` (`Proceed | Reflect | Abstain` per ADR-014).
**Error handling:** missing Topaz skill metadata produces `LlmOrcStructuralError(error_kind="missing_skill_metadata", recovery_action_required="reformulate")`. Abstain verdict produces `LlmOrcStructuralError(error_kind="escalation_bypass", recovery_action_required="reformulate")`.
**Owned by:** Tier-Escalation Router defines the contract.

### Tier-Escalation Router → Calibration Gate *(new in Cycle 4 per ADR-014 + ADR-015)*

**Protocol:** Synchronous read of calibration verdict. Router calls `verdict_for(session_id, ensemble_name, dispatch_context) -> CalibrationVerdict`.
**Shared types:** `CalibrationVerdict`; `DispatchContext` (the dispatch-time inputs needed for verdict computation: orchestrator's recent trajectory features when ADR-016 is active, or empty when the channel is not active).
**Error handling:** verdict computation does not raise; if ADR-016 is rejected (no cross-layer signals available), the verdict computation operates on L1-internal trajectory data only (per ADR-014 §"Feature-extraction location").
**Owned by:** Calibration Gate defines the contract; the verdict surface is the public extension to ADR-007's `QualitySignal` API.

### Tier-Escalation Router → Ensemble Engine *(new in Cycle 4 per ADR-015)*

**Protocol:** Synchronous read of ensemble metadata. Router reads the dispatched ensemble's Topaz skill via the existing config manager surface.
**Shared types:** existing `EnsembleConfig` extended with optional `topaz_skill: TopazSkill` field; `TopazSkill = Literal["code_generation", "tool_use", "mathematical_reasoning", "logical_reasoning", "factual_knowledge", "writing_quality", "instruction_following", "summarization"]`.
**Error handling:** missing or invalid `topaz_skill` field → `missing_skill_metadata` typed error (raised before dispatch — the orchestrator gets the error as a tool-call observation; the ReAct loop continues).
**Owned by:** Ensemble Engine owns the YAML schema extension; Tier-Escalation Router owns the metadata-read contract.

### Calibration Gate → Calibration Signal Channel *(new in Cycle 4 per ADR-016; conditional)*

**Protocol:** Synchronous read of windowed signal data. Gate calls `recent_signals(window) -> SignalWindow`.
**Shared types:** `SignalWindow` (the time-decay-windowed signals — only the most recent window; per mechanism (b) 60-min/100-signal dual-bound, linear decay); `TrajectorySignal` (typed shape per ADR-016 §"The signal channel": trajectory features per ADR-014's HTC specification, dispatch outcomes, deterministic-tool-output anchors when applicable).
**Error handling:** when ADR-016 is rejected, Calibration Signal Channel returns an empty window; the Gate's verdict computation operates on L1-internal trajectory data only (per ADR-014 §"Feature-extraction location").
**Owned by:** Calibration Signal Channel defines the contract.

### Ensemble Engine → Calibration Signal Channel *(new in Cycle 4 per ADR-016 — THE LOAD-BEARING UPWARD EXCEPTION)*

**Protocol:** Observer-pattern callback. Ensemble Engine invokes `emit_signal(signal: TrajectorySignal)` after each dispatch completes. Read-only — the channel does not mutate ensemble state through this edge.
**Shared types:** `TrajectorySignal` (typed schema validated at the channel boundary by mechanism (e)); `SignalEmitter` (Protocol; the registration interface the Calibration Signal Channel exposes for L0 to call into).
**Error handling:** mechanism (e) — read-only structural validation at the consumer — rejects malformed signals with `LlmOrcStructuralError(error_kind="malformed_signal", recovery_action_required=internal)`. Malformed signals are logged but do not influence verdicts; the verdict computation skips them as if outside the time window. This typed error is internal — not raised to the orchestrator's reasoning surface.
**Constraints:**
1. **Read-only** — no upward writes; ADR-002's write-path layering rule is preserved.
2. **Calibration-only** — the channel rejects non-calibration data via mechanism (e)'s structural validation guard.
3. **Bounded by mechanisms (a)–(e)** — the L1 consumer reads in fresh evaluation context (mechanism (a)), with time-decay-windowed feature aggregation (mechanism (b)), with deterministic-tool-output anchors when the dispatched ensemble has script-model slots (mechanism (c)), with periodic out-of-band audit dispatch detecting drift (mechanism (d)), with read-only structural validation rejecting malformed signals (mechanism (e)).
**Owned by:** Calibration Signal Channel defines the upward-data-flow contract; the L0-side hook registration is implemented by Ensemble Engine but the schema and discipline live with the channel.

### Cross-cutting integration: shared `LlmOrcStructuralError` typed-error infrastructure *(new in Cycle 4 per ADR-017 §"Shared typed-error base class")*

**Not a single edge — a cross-module integration contract.**

**Base class** (lives in `models/structural_errors.py` or extends `models/base.py` to preserve the codebase precedent at commit `9f86d0b`):

```python
class LlmOrcStructuralError(Exception):
    error_kind: str
    dispatch_context: dict
    recovery_action_required: Literal[
        "reformulate", "escalate", "abstain", "operator_intervention_required"
    ]
    operator_diagnostic: str
```

**Eight initial `error_kind` subclasses (one per Cycle 4 typed-error surface):**

| `error_kind` | Producer | `recovery_action_required` | ADR |
|--------------|----------|---------------------------|-----|
| `tool_call_rejected_per_model` | Provider adapter (existing — migrated as first concrete subclass) | `reformulate` | commit `9f86d0b` |
| `phantom_tool_call` | Tool Dispatch (structural validation guard) | `reformulate` | ADR-017 |
| `compaction_layer_4_failure` | Conversation Compaction (Layer 4 circuit-breaker) | `operator_intervention_required` (after 3 consecutive failures) | ADR-012 |
| `write_gate_rejection` | Session Registry (write-gate validation; three classes) | `reformulate` | ADR-013 |
| `calibration_abstain` | Calibration Gate (Abstain verdict) | `reformulate` | ADR-014 |
| `escalation_bypass` | Tier-Escalation Router (Abstain verdict produces typed bypass) | `reformulate` | ADR-015 |
| `missing_skill_metadata` | Tier-Escalation Router (ensemble lacks `topaz_skill` field) | `reformulate` | ADR-015 |
| `malformed_signal` | Calibration Signal Channel (mechanism (e) — schema validation; internal — not raised to orchestrator) | (internal — verdict computation skips signal) | ADR-016 |

**Provenance:** ADR-017 §"Shared typed-error base class" + commit `9f86d0b` typed-error precedent. Naming and field finalization is BUILD-time work (WP-A4); ARCHITECT commits to the structural commitment.

---

## Fitness Criteria

| # | Criterion | Measure | Threshold | Derived From |
|---|-----------|---------|-----------|-------------|
| FC-1 | No module owns more than 5 scoped glossary entries as primary owner | Count rows per module in the Responsibility Matrix | ≤ 5 | God-class prevention (Essay §Guardrails; ARCHITECT principle) |
| FC-2 | Dependency edges point from higher layer to same-or-lower layer only | Static inspection of module imports against L0-L3 assignment | 0 violations | Layering rule (Dependency Graph) |
| FC-3 | No cycles in the dependency graph | Static cycle detection over the dependency edge list | 0 cycles | ARCHITECT principle |
| FC-4 | Orchestrator Runtime imports only Budget Controller and Orchestrator Tool Dispatch — no Result Summarizer Harness, no Plexus, no config, no Autonomy, no Calibration. Result Summarizer Harness is interposed on the `invoke_ensemble` return path by Orchestrator Tool Dispatch (Amendment #3). | Static import check | Exact match | Orchestrator LLM's mental model alignment; structural ADR-003 enforcement |
| FC-5 | Orchestrator Tool Dispatch has exactly five public entry points — one per committed tool | Static count of public dispatch methods | = 5 | ADR-003 (closed tool set) |
| FC-6 | Composition Validator and Ensemble Engine's load path call the same public validator function | Static check: single definition, two call sites | 1 definition, 2+ call sites | Scenarios refactor 1-3; ADR-006 negative consequence |
| FC-7 | Every Plexus-facing code path has a no-op fallback exercised by a stateless-mode test | Per-edge coverage of the stateless branch | 100% | AS-8 |
| FC-8 | `unsummarized-result` cannot reach the Orchestrator Runtime's context | Static check: Runtime imports `ToolCallResult`; no path from `EnsembleExecutor` to Runtime bypasses the Harness | 0 bypass paths | AS-7; ADR-004 |
| FC-9 | Session-start flow calls `resolve_session_start_context` exactly once; the function has a typed signature returning `list[PromptFragment]` | Static inspection | Exactly 1 call; signature present | ADR-009 (structural reservation via typed function) |
| FC-10 | Budget check executes before every ReAct iteration begins | Integration test covering the iteration-boundary contract | 100% of iterations | AS-3; ADR-005 |
| FC-11 | Autonomy Policy check executes before every Orchestrator Tool Dispatch | Integration test | 100% of dispatches | ADR-008 |
| FC-12 | Composed ensembles enter Calibration Gate transparently on `invoke_ensemble` during calibration | Integration test | 100% of in-calibration invocations | ADR-007 |
| FC-13 | Changing the orchestrator Model Profile requires touching only Orchestrator Configuration and Session start-logic — not Runtime, not Tool Dispatch | Diff inspection on a profile-swap change | No edits to Runtime or Tool Dispatch | ADR-011 |
| FC-14 | Conversation Compaction's five-layer pipeline applies layers in cheapest-first order; Layer 4 LLM-summary fires only after Layers 0–3 have been attempted | Static inspection of `compact()` implementation; integration test exercising each layer's trigger | Layer ordering preserved in code + test demonstrates Layer 4 only fires after Layers 0–3 reduction attempted | ADR-012 §Decision (load-bearing design property); essay 005 §"Long-Horizon Reliability" |
| FC-15 | Every `invoke_ensemble` dispatch passes through the Tier-Escalation Router; the Tool Dispatch cannot reach `EnsembleExecutor.execute` for an `invoke_ensemble` call without passing through tier selection | Static AST dominance check on `orchestrator_tool_dispatch.py::invoke_ensemble`; pattern follows FC-8 | 0 bypass paths | ADR-015 (router operates inside Tool Dispatch's `invoke_ensemble` interposition) |
| FC-16 | Calibration Signal Channel boundary is read-only at the L1 consumer; no upward writes from L1 to L0 through the channel | Static check: channel module exposes only read-side methods to L1 callers; no write hooks register at L0; mechanism (e) rejects malformed signals | 0 write hooks; all upward write attempts produce `malformed_signal` rejection at boundary | ADR-016 §"The signal channel" (read-only constraint) |
| FC-17 | All eight Cycle 4 typed errors derive from `LlmOrcStructuralError` base class with the four common fields (`error_kind`, `dispatch_context`, `recovery_action_required`, `operator_diagnostic`) | Static class-hierarchy walk + AST inspection of error construction sites | 100% of Cycle 4 typed errors are `LlmOrcStructuralError` subclasses with all four fields populated | ADR-017 §"Shared typed-error base class" |
| FC-18 | Every ensemble in the library declares `topaz_skill` metadata; dispatch of an ensemble lacking the field produces `missing_skill_metadata` typed error before tier selection runs | Static check across `.llm-orc/ensembles/*.yaml` + integration test on metadata-missing case | All library ensembles have `topaz_skill`; `missing_skill_metadata` raised pre-dispatch when absent | ADR-015 (Topaz skill metadata required) |
| FC-19 | Tier-Escalation Router's `select_tier` is a stateless pure function — each verdict consumption runs in fresh context with no prior verdict state carried forward through the call | Static AST inspection of `tier_router.select_tier` (no instance state mutated; no module-level state read) + unit test exercising stateless property under sequential dispatches | 0 statefulness violations; sequential `select_tier` calls produce identical output for identical inputs regardless of call history | ADR-018 (inherits ADR-016 mechanism (a) by construction) |
| FC-20 | Tier-Escalation Router's (d)-analog audit dispatch fires at the configured trigger frequency (default every 100 verdict consumptions or 24 wall-clock hours, whichever first); severe-drift verdict activates route-all-to-escalated fail-safe mode plus operator notification | Integration test exercising the trigger boundary + a separate test exercising severe-drift fail-safe mode activation; pattern follows FC-14 (cheapest-first-order audit) | Audit fires within ±10% of configured trigger frequency; severe-drift triggers fail-safe within one dispatch cycle | ADR-018 §"Trigger" + §"Severe-drift response" |

All criteria are automatable via a combination of static import analysis, test coverage, and dependency-graph reconstruction. **Three structural prerequisites must land before any ADR-by-ADR BUILD work begins** (per the Cycle 4 conformance scan): (1) FC-2 and FC-3 as automated tests (currently specified but not implemented as test artifacts; pre-existing gap); (2) the shared `LlmOrcStructuralError` base class (FC-17 prerequisite — without it, the eight typed errors would diverge as separate classes and re-merging post-hoc is BUILD-burden); (3) ensemble-YAML migration to add `topaz_skill` metadata field (FC-18 prerequisite). FC-1 (≤ 5 entries per module) holds with Cycle 4 additions — Calibration Gate is at the threshold (5 entries) but the entries all compose into the verdict and share the same change-rate; splitting would not be load-bearing.

---

## Test Architecture

### Boundary Integration Tests

Every dependency edge must have at least one integration test that exercises real data flow with real types on both sides. No mocking at the boundary under test.

| Edge | Integration Test | Verifies |
|------|-----------------|----------|
| Serving Layer → Session Registry | `test_serving_resolves_session_identity` | HTTP request with/without session continuity correlates to correct SessionState |
| Serving Layer → Orchestrator Runtime | `test_serving_streams_runtime_output` | SSE chunks flow end-to-end; client-tool round trip resumes same Session |
| Orchestrator Runtime → Budget Controller | `test_runtime_honors_budget_at_iteration_boundary` | Turn-limit and token-limit exhaustion both terminate at an iteration boundary |
| Orchestrator Runtime → Orchestrator Tool Dispatch | `test_runtime_dispatches_internal_tools_only` | Any tool name outside the five returns a tool error observation, not an exception |
| Orchestrator Tool Dispatch → Result Summarizer Harness | `test_runtime_never_sees_unsummarized_result` | `invoke_ensemble` returning a large dict reaches the Runtime as a summary via the Harness; escape-hatch flag bypasses. Interposition lives on Tool Dispatch per Amendment #3 — the Runtime is unaware of the Harness |
| Orchestrator Tool Dispatch → Ensemble Engine | `test_invoke_ensemble_executes_real_ensemble` | End-to-end ensemble execution with real `EnsembleExecutor` |
| Orchestrator Tool Dispatch → Composition Validator | `test_compose_ensemble_rejects_cycle` | Cyclic reference graph rejected at composition time |
| Orchestrator Tool Dispatch → Plexus Adapter | `test_query_knowledge_and_record_outcome_round_trip` | Real Plexus-active path; also run in stateless mode to verify no-op returns |
| Orchestrator Tool Dispatch → Autonomy Policy | `test_autonomy_gate_fires_before_every_dispatch` | Every tool path passes through the gate; baseline level allows invoke, composes, denies promotion |
| Orchestrator Tool Dispatch → Calibration Gate | `test_calibration_interposes_on_in_calibration_ensembles` | First N invocations run the check; (N+1)th does not |
| Composition Validator ↔ Ensemble Engine (shared) | `test_shared_validator_same_result_both_paths` | `validate_ensemble_reference_graph` returns identical outcome when called from load path and composition path on the same input (scenarios regression) |
| Result Summarizer Harness → Ensemble Engine | `test_summarizer_failure_preserves_artifact` | Summarizer exception: original artifact persists on disk, Runtime receives typed summarization error |
| Serving Layer's `resolve_session_start_context` | `test_session_start_context_is_empty_in_phase_1` | Function called once per session start, returns `[]`, never touches Plexus Adapter in Phase 1 |
| Bootstrapping Pipeline → Plexus Adapter | `test_bootstrap_pushes_source_material_not_summaries` | Ingested content is file content; no LLM-generated summaries in the push stream (AS-4) |
| Bootstrapping Pipeline → Ensemble Engine | `test_bootstrap_reads_library_via_config_manager` | Pipeline uses the existing config-manager path; respects tiered storage |
| Autonomy Policy → Session Registry | `test_autonomy_reads_session_state` | Autonomy Level resolved from per-session config, not global |
| Budget Controller → Session Registry | `test_budget_derives_cumulative_spend_from_session` | Token and turn spend sums are session-scoped, not request-scoped |
| Calibration Gate → Plexus Adapter | `test_calibration_persists_across_sessions_when_plexus_active` | Session 2 sees cleared calibration from Session 1 when Plexus is active; not when absent |
| Orchestrator Runtime → Conversation Compaction *(Cycle 4)* | `test_runtime_invokes_compaction_at_turn_boundary` | Compaction runs at every turn boundary; the resulting `CompactedContext` is what flows into the next LLM call |
| Conversation Compaction → Ensemble Engine *(Cycle 4)* | `test_compaction_layer_4_invokes_summarizer_ensemble` | When Layers 0–3 cannot reduce context below threshold, Layer 4 dispatches a summarizer ensemble via real `EnsembleExecutor`; circuit-breaker counts failures |
| Conversation Compaction → filesystem *(Cycle 4)* | `test_compaction_layer_0_persists_large_tool_results` | Tool results above 50K chars persist to disk; orchestrator context receives 2 KB preview + path; path queryable later |
| Orchestrator Tool Dispatch → Tier-Escalation Router *(Cycle 4)* | `test_invoke_ensemble_routes_through_tier_router` | Every `invoke_ensemble` dispatch passes through tier selection; verdict input flows from Calibration Gate; selected tier flows into `EnsembleExecutor` |
| Tier-Escalation Router → Calibration Gate *(Cycle 4)* | `test_router_consumes_calibration_verdict` | Router reads the verdict directly (no LLM-mediated translation); the verdict's three values map deterministically to router actions (cheap-tier / escalated-tier / `escalation_bypass` typed error) |
| Tier-Escalation Router → Ensemble Engine *(Cycle 4)* | `test_router_reads_topaz_skill_from_ensemble_yaml` | Router resolves the dispatched ensemble's `topaz_skill` field via existing config manager; ensembles without the field produce `missing_skill_metadata` typed error |
| Calibration Gate → Calibration Signal Channel *(Cycle 4; conditional)* | `test_gate_reads_windowed_signals_in_fresh_context` | When ADR-016 active, gate reads windowed signal data within a fresh evaluation context (mechanism (a)); when ADR-016 rejected, gate operates on L1-internal data only |
| Ensemble Engine → Calibration Signal Channel *(Cycle 4 — read-only upward exception)* | `test_ensemble_emits_typed_signals_to_channel_read_only` | L0 emits typed signals through the registered hook; channel validates schema (mechanism (e)) and rejects malformed signals with `malformed_signal` error; channel is read-only — no upward write paths from L1 to L0 |

### Invariant Enforcement Tests

| Invariant | Enforcement Module | Test |
|-----------|--------------------|------|
| AS-1 (dynamic invocations outside reference graph) | Orchestrator Tool Dispatch | `test_invoke_ensemble_does_not_register_reference` — an invoke_ensemble call leaves the static reference graph unchanged |
| AS-2 (composed ensembles validated before loading) | Composition Validator | `test_compose_ensemble_validates_before_write` — no file written until validation passes |
| AS-3 (Budget is control-plane) | Budget Controller | `test_orchestrator_llm_cannot_observe_budget_state_in_context` |
| AS-4 (ingestion boundary is source material) | Plexus Adapter + Bootstrapping Pipeline | `test_ingestion_rejects_llm_summary_marker` (source-material assertion by type) |
| AS-5 (quality signals govern stabilization, not frequency) | Calibration Gate | `test_frequency_without_quality_does_not_trust` |
| AS-6 (compose from existing primitives only) | Composition Validator + Orchestrator Tool Dispatch | `test_compose_ensemble_rejects_new_script_or_profile` |
| AS-7 (result summarization is a correctness requirement) | Result Summarizer Harness | `test_runtime_never_sees_unsummarized_result` (also FC-8) |
| AS-8 (Plexus is optional) | Plexus Adapter + Serving Layer session-start function | `test_all_operations_work_with_plexus_absent` — full coverage of every Plexus-facing edge, including the session-start function in Phase 2 |
| Invariant 5 (cross-ensemble acyclicity) | Composition Validator + Ensemble Engine | `test_compose_ensemble_rejects_cycle` (also FC-6 regression) |
| Invariant 7 (static ensemble references) | Orchestrator Tool Dispatch | `test_invoke_ensemble_uses_existing_name_no_template_expression` |
| Invariant 9 (child artifacts nested) | Ensemble Engine (existing) + Result Summarizer Harness | `test_summarizer_failure_preserves_full_artifact` |
| Invariant 13 (execution resilient) | Ensemble Engine (existing) | (existing test suite) |
| AS-7 generalization to conversation context (per ADR-012) | Conversation Compaction | `test_orchestrator_context_stays_below_threshold_across_long_session` — multi-turn fixture exceeding threshold; compaction maintains coherence |
| AS-3 (control-plane discipline preserved) under tier escalation | Budget Controller, Tier-Escalation Router | `test_tier_escalation_preserves_budget_enforcement_at_iteration_boundary` — escalated dispatches do not bypass turn/token limits |
| AS-1 (dynamic invocations outside reference graph) under tier escalation | Orchestrator Tool Dispatch + Tier-Escalation Router | `test_tier_routed_invocation_does_not_register_reference` — same as existing AS-1 test plus tier-routed dispatches |
| ADR-016 read-only constraint on signal channel | Calibration Signal Channel | `test_signal_channel_rejects_upward_write_attempts` — any write toward L0 through the channel boundary is structurally rejected |
| ADR-016 calibration-only constraint on signal channel | Calibration Signal Channel (mechanism (e)) | `test_signal_channel_rejects_non_calibration_data` — non-calibration data (arbitrary upward import) is rejected via structural validation |
| ADR-002 layering rule preserved for write paths and non-calibration upward signaling | (cross-cutting) | `test_fc2_no_upward_imports_except_calibration_channel` — FC-2 prerequisite from WP-B4 |

### Test Layers

- **Unit.** Verify logic within a single module. Mocks are acceptable for neighbor modules. Every module has unit tests for its core state transitions (e.g., Budget Controller's exhaustion check; Calibration Gate's N-invocation transition; Composition Validator's cycle detection on synthetic graphs).
- **Integration.** Verify real data flow across module boundaries. No mocks at the boundary under test. All 18 edges in the Boundary Integration Tests table are integration tests.
- **Acceptance.** Verify scenarios end-to-end using real wiring. The scenarios in `scenarios.md` map to acceptance tests. The scenario "Tool user completes a task against the stateless orchestrator" is the happy-path acceptance test; "Session terminates gracefully on turn limit exhaustion" and "Composition with ensemble-to-ensemble reference passes validation" are the representative stress paths.

---

## Appendix A: Per-Phase Susceptibility Snapshot Briefs

This appendix holds the canonical brief content for each phase skill's `Phase Boundary: Susceptibility Snapshot Dispatch` subsection when operating within this scoped cycle. The briefs are phase-specific per ADR-056's non-formulaic requirement and reference concrete prior-cycle findings where applicable. Briefs may need updating as future cycles accumulate evidence; the `/rdd-conform` dispatch prompt format audit verifies structural compliance with the canonical skeleton but does not audit brief content substance.

**Structural wrapping (applied identically to every phase skill that runs in this cycle):**

```
## Phase Boundary: Susceptibility Snapshot Dispatch

Before completing this phase, dispatch the susceptibility-snapshot-evaluator subagent with the following brief:

<phase-specific brief content from Appendix A>

Output path: <canonical path from Appendix A>
```

The appendix records the **post-v0.8.5 canonical paths** at `housekeeping/audits/...` (this scoped corpus uses ADR-070 housekeeping placement; if `/rdd-conform migrate-to-rdd` is run later, paths swap to `.rdd/audits/...`).

### A.1 Research phase (`skills/research/SKILL.md`)

> This is the research → discover boundary for the agentic-serving cycle. Cycle 1's research phase produced two essays: 001 (architecture, pre-build) and 002 (capability floor and observability, backward loop from PLAY). Essay 002 surfaced a confabulation gap in tool-calling benchmarks and a hybrid-deployment latency-closing finding. Cycle 2's research enters the multi-turn + live-composition territory the prior cycle named but did not exercise; the structural machinery (`compose_ensemble`, Composition Validator, Calibration Gate, retry pattern) is in place but unentered. The primary risk at this boundary is framing adoption at essay crystallization moments — particularly any drift from "what does multi-turn composition look like in practice?" toward "the architecture handles it" without the empirical entry. Evaluate whether the essay that is about to enter downstream phases has been shaped by Cycle 1's structural commitments without those commitments being independently re-tested for the Cycle 2 question. Apply constraint-removal per ADR-082 at the research-entry moment: was *"what would we build if compose_ensemble were not available?"* posed against the prior cycle's machinery, and did the answer move the framing or merely confirm it?
>
> Output path: housekeeping/audits/susceptibility-snapshot-{cycle-id}-research.md

### A.2 Discover phase (`skills/discover/SKILL.md`)

> This is the discover → model boundary for the agentic-serving cycle. Cycle 1's product-discovery established stakeholder maps (Tool User, Operator, Pure Tool User, Orchestrator LLM), seven value tensions, and six assumption inversions. Cycle 1's PLAY findings (FF #128–#135) routed bilateral observability gap, capability-floor experience, and experience-conditional visibility back to discover as candidate updates. Discover for Cycle 2 will run in update mode against the existing artifact; the canonical failure mode at this boundary is research-phase framings propagating into product vocabulary without attribution, and value tensions surfacing as spectra but collapsing into binary framings without alternatives examined. Evaluate the attached AID signals for two specific patterns: (a) Cycle 2's research-essay framings inherited into product-facing language without being tested against user voice, and (b) llm-conductor-pattern framings (composition + multi-turn) that may have been adopted from the user's explicit framing without belief-mapping against alternatives like Cycle 1's "stateless orchestrator suffices" position.
>
> Output path: housekeeping/audits/susceptibility-snapshot-{cycle-id}-discover.md

### A.3 Model phase (`skills/model/SKILL.md`)

> This is the model → decide boundary for the agentic-serving cycle. Cycle 1's domain-model layered AS-1 through AS-8 onto the project's Invariants 1-14, with AS-8 (Plexus is optional) as the load-bearing architectural constraint. Cycle 2 may need to amend invariants if multi-turn + composition surfaces tensions with AS-1 (dynamic invocations outside the reference graph) or AS-6 (compose from existing primitives only — the user has flagged eventual orchestrator authorship of scripts and model profiles as desirable). Invariant amendments are the highest-stakes commitment type in the methodology. The canonical failure mode is preference-accelerated commitment: user-stated preference precedes implications analysis; alternatives not engaged at comparable depth. Evaluate the attached AID signals for warrant-elicitation failures and preference-accelerated commitments. For any invariant amendment or consequential concept addition, check whether belief-mapping was performed before adoption — not after. The test is whether the user could name what they would need to believe for a different commitment to be right.
>
> Output path: housekeeping/audits/susceptibility-snapshot-{cycle-id}-model.md

### A.4 Decide phase (`skills/decide/SKILL.md`)

> This is the decide → architect boundary for the agentic-serving cycle. Cycle 1 produced 11 ADRs (adr-001..011); Cycle 2 may extend or amend several — particularly ADR-006 (composition palette), ADR-007 (calibration gate), ADR-008 (autonomy levels), and ADR-011 (orchestrator as Model Profile) — depending on what multi-turn + live composition surfaces. The primary risk is that ADR framings originate from agent synthesis during drafting rather than from architectural drivers (research findings, domain model concepts, prior ADRs). Cycle 1's WP-F mini-cycle produced a strong example of provenance discipline (the retry pattern was committed as conditional on capability evidence, with explicit reopen conditions). Evaluate the attached AID signals for rebuttal-elicitation failures on rejected alternatives, and for cross-ADR compositions where one ADR's framing was adopted by another within the same cycle without being independently tested. Check whether each ADR's core framing traces to its driver chain or to drafting-time composition.
>
> Output path: housekeeping/audits/susceptibility-snapshot-{cycle-id}-decide.md

### A.5 Architect phase (`skills/architect/SKILL.md`)

> This is the architect → build boundary for the agentic-serving cycle. Cycle 1's architecture composed twelve modules across four layers plus the typed `resolve_session_start_context` function reservation; Cycle 4 ARCHITECT integrates six new ADRs (012-017) by adding three new modules (Conversation Compaction L2; Tier-Escalation Router L2; Calibration Signal Channel L1 — conditional acceptance), extending four existing modules (Session Registry, Orchestrator Runtime, Orchestrator Tool Dispatch, Calibration Gate), and amending ADR-002's layering rule for the load-bearing read-only L0→L1 calibration signal channel exception. The specific risk at this boundary is that the architectural integration encodes unexamined assumptions inherited from DECIDE-phase ADRs — particularly the elaboration-by-evidence framing commitment (research-gate Grounding Action 2) which holds that bounding mechanisms operationalize within ADR-002's existing layer structure rather than as a cross-cutting module. The Inversion Principle check at architecture is the designed counterweight but can be performed pro forma. **Asymmetric-grounding finding from decide-gate (OQ #14):** the cycle's L0→L1 calibration channel received 5 bounding mechanisms with operational grounding while five other cross-layer stages (L1→L2 verdict→router; L3 cross-session artifact set; intra-L2 conversation-history boundary; orchestrator-response→tool-dispatch boundary; L1→L4 Plexus integration) received less rigor — the asymmetry was carried forward as Cycle 5+ research territory; ARCHITECT decided to defer to BUILD evidence per the cycle status. Evaluate the attached AID signals for: (a) solution-space narrowing on the new module boundaries — were Conversation Compaction's separation from Runtime, Tier-Escalation Router's separation from Tool Dispatch, and Calibration Signal Channel's separation from Calibration Gate examined as alternatives or accepted from ADR framings? (b) framing adoption from DECIDE-phase ADRs — the system design inherits ADR-016's "five mechanisms within L0-L3" framing; was the inheritance examined or automatic? (c) whether the proposed module boundaries would survive a product-facing reading — do they track the operator's mental model (compose, calibrate, sustain, escalate by skill, persist artifacts) or only developer convenience? (d) whether the cross-layer-stage asymmetric-rigor concern (OQ #14) shows up in the dependency graph or fitness criteria — should other cross-layer edges receive grounding-mechanism analogs of (a)-(e)? The Pattern A vs Pattern B (ADR-084) decision was made at Cycle 2 (slim human-facing + companion-file split); Cycle 4 inherits the pattern.
>
> Output path: housekeeping/audits/susceptibility-snapshot-cycle-4-architect.md

### A.6 Build phase (`skills/build/SKILL.md`)

> This is the build phase boundary for the agentic-serving cycle (typically build → play or build → next-WP). Cycle 1's build (WP-A through WP-I plus four post-PLAY production changes) was strongly empirically grounded: each WP closed with a green test suite, fitness-criteria static checks, and stewardship-clean reports. Cycle 2's build extends this surface to multi-turn + live composition. Build is the most empirically grounded phase per the sycophancy gradient — test execution grounds commitments in observable outcomes. The residual risks are in the spaces tests do not reach: stewardship-checkpoint commitments that adopt rejected-alternative framings without surfacing them, debug hypotheses that absorb a "this must be because X" framing without belief-mapping X, and mode-shift transitions where the user's mental model may drift without being tested. Evaluate the attached AID signals with build's empirical grounding as the baseline — concerning signals here are patterns the tests did not catch.
>
> Output path: housekeeping/audits/susceptibility-snapshot-{cycle-id}-build.md

### A.7 Play phase (`skills/play/SKILL.md`)

> This is the play → synthesize boundary (or play → next-cycle if synthesize is deferred) for the agentic-serving cycle. Cycle 1's PLAY produced field-notes that surfaced the bilateral observability gap, capability-floor experience verdict, and the deferred Note 6 / Note 7 reframings. Cycle 2's play will enter the live-composition territory: stakeholder inhabitation against an orchestrator actually composing and using ensembles in real time. The specific risk at this boundary is that field notes flatten from observation into advocacy — discoveries that "confirm the design" crowd out discoveries that challenge it. Two signal patterns matter most: (a) selection bias in the six-category classification, and (b) gamemaster/player role blur under task load (especially likely under multi-turn fatigue). Evaluate the attached AID signals for framing adoption in field-note language (user voice vs. methodology voice) and for declining alternative engagement on discoveries that would challenge the design. The field notes entering synthesize should read as observation, not endorsement. Cycle 1's Grounding Reframe items 6 and 7 (deferral named, practitioner prior restored) are the reference for honest reframing under play.
>
> Output path: housekeeping/audits/susceptibility-snapshot-{cycle-id}-play.md

### A.8 Synthesize phase (`skills/synthesize/SKILL.md`)

> This is the synthesize phase boundary for the agentic-serving cycle (typically terminal; or synthesize → research re-entry if structural experiments surface new questions). Cycle 1 did not run synthesis — the cycle declared complete without it. Cycle 2 may run synthesis if multi-turn + composition produces a publishable finding (the conductor pattern's empirical character would be a candidate). The canonical failure mode at synthesis is the "softer than sycophancy but real" framing-adoption pattern where narrative framings composed at synthesis moments shape what the cycle will be remembered as. The four-dimension framing navigation provides structural scaffolding, but emerging framings can still be shaped by synthesis-moment adoption. Evaluate the attached AID signals with particular emphasis on framing adoption and declining alternative engagement during the framing conversation. The outline about to become the writer's essay should carry framings the writer owns, not framings they inherited from agent composition or from the artifact trail without examining the alternatives.
>
> Output path: housekeeping/audits/susceptibility-snapshot-{cycle-id}-synthesize.md

### Maintenance note for future cycles

Each brief references concrete prior-cycle findings where applicable (Cycle 1's two essays, eight-phase progression, eleven WPs, four production changes from the backward research loop). These references are anchored in time. As Cycle 2 and subsequent cycles accumulate evidence, briefs may need updating. Brief refresh is a deliberate cycle operation, not automated.
