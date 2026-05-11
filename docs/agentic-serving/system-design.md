# System Design: Agentic Serving

**Version:** 3.0
**Status:** Current
**Last amended:** 2026-05-08
**Scope:** Scoped RDD cycle at `docs/agentic-serving/`. Inherits the project-level domain model (Invariants 1-14) and existing system architecture.

---

## What this document is, who it's for, where to read what

This is the technical perspective on **agentic serving** — how llm-orc serves as the orchestrator backend for agentic coding tools (OpenCode, Roo Code, Cline, etc.) via OpenAI-compatible endpoints. It describes the four-layer system, the closed-set tool surface, the client-tool delegation boundary, and the supporting modules that govern Budget, Composition, Calibration, Autonomy, and the (optional) Plexus knowledge-graph integration.

**First-encounter readers** orient through the diagram, brief module summaries, and the Client Tool Surface Commitment below. **Agents constructing context** for code work read the companion file at [system-design.agents.md](./system-design.agents.md), which carries the full architectural drivers table, module decomposition (Owns/Depends/Inversion notes), responsibility matrix, dependency graph, integration contracts, fitness criteria, test architecture, and Appendix A per-phase susceptibility-snapshot briefs in the form best suited to that work.

The split between this artifact and `system-design.agents.md` is the **companion-file pattern** (ADR-084 Pattern B): a single human-facing surface here, with a parallel-sibling agent-context file at a predictable path. The diagram below retains its load-bearing role for human orientation; the dense reference material has been relocated rather than removed.

**Sequencing** lives in [roadmap.md](./roadmap.md). **Decisions** live in [decisions/](./decisions/). **Vocabulary** lives in [domain-model.md](./domain-model.md). **Behavior** lives in [scenarios.md](./scenarios.md) and [interaction-specs.md](./interaction-specs.md). If you are coming to this corpus with no prior context, read [ORIENTATION.md](./ORIENTATION.md) first.

---

## Architecture at a glance

```
                       OpenAI-compatible client
                       (OpenCode, Roo Code, Cline, ...)
                                  │
                                  ▼  /v1/chat/completions, /v1/models
┌─────────────────────────────────────────────────────────────────────┐
│  L3 — Entry Layer                                                    │
│  Serving Layer · Session Registry (extended w/ structured-handoff   │
│  artifacts + write-gate validation per ADR-013) · Bootstrapping     │
│  Pipeline · Orchestrator Configuration                               │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  L2 — Runtime                                                        │
│  Orchestrator Runtime ──── ReAct loop, fixed 5-tool action surface   │
│  Orchestrator Tool Dispatch (extended w/ structural validation guard │
│  per ADR-017) · Result Summarizer Harness · Conversation Compaction  │
│  (5-layer pipeline per ADR-012) · Tier-Escalation Router (per-role   │
│  via OI-MAS per ADR-015)                                             │
└─────────────────────────────────────────────────────────────────────┘
                                  │     ▲
                                  ▼     │ (read-only signal channel —
┌─────────────────────────────────────────────────────────────────────┐
│  L1 — Domain Policy                  │ ADR-016 narrow exception)    │
│  Composition Validator · Budget Controller · Autonomy Policy ·       │
│  Calibration Gate (extended w/ verdict trichotomy + AUQ + HTC +      │
│  time-decay windowing per ADR-014) · Plexus Adapter (optional,       │
│  no-op when absent) · Calibration Signal Channel (per ADR-016;       │
│  conditional acceptance — first-deployment evidence pending)         │
└─────────────────────────────────────────────────────────────────────┘
                                        │
                                  ▼     │
┌─────────────────────────────────────────────────────────────────────┐
│  L0 — Core (existing)                                                │
│  Ensemble Engine ───── DAG executor (Layer 3 unchanged per ADR-002)  │
└─────────────────────────────────────────────────────────────────────┘
```

**Layering rule.** Edges point from a higher layer to a same-or-lower layer; never upward — **with one narrow exception per ADR-016 (conditional acceptance; first-deployment evidence pending):** a read-only signal channel may flow from L0 (Ensemble Engine dispatch outputs) to L1 (Calibration Signal Channel module), restricted to calibration data and gated by five bounding mechanisms specified in ADR-016. The exception is signal-channel-specific (calibration only; not a general upward import permission) and read-only (no upward writes; ADR-002's layering rule for write paths is unchanged). All other layer pairs remain prohibited. FC-2 (static import check) and FC-3 (cycle detection) recognize the calibration-channel exception via an annotated allowed-edge in their layer map. Intra-L3 edges (Serving Layer ↔ Session Registry, Orchestrator Configuration) do not form cycles. Cycle 4's seven new dependency edges (Runtime → Conversation Compaction; Compaction → Ensemble Engine; Tool Dispatch → Tier-Escalation Router; Tier-Escalation Router → Calibration Gate; Tier-Escalation Router → Ensemble Engine; Calibration Gate → Calibration Signal Channel; Ensemble Engine → Calibration Signal Channel as the upward exception) are all verified cycle-free.

**Falsification trigger (ADR-016).** If BUILD or first-deployment evidence finds that mechanism (b) time-decay windowing or mechanism (d) periodic out-of-band audit dispatch cannot be operationalized within ADR-002's L0-L3 structure (e.g., the mechanisms require a top-level module orthogonal to the four-layer architecture), the elaboration-by-evidence framing commitment is invalidated, the reorganization branch re-opens, and ADR-016 is re-deliberated with reorganization on the table.

**Five-tool internal action surface (ADR-003).** The orchestrator's ReAct iterations call only: `invoke_ensemble`, `compose_ensemble`, `list_ensembles`, `query_knowledge`, `record_outcome`. No others. Client-declared tools flow through the response surface (turn-boundary delegation), not the action surface.

**Plexus is optional (ADR-002, AS-8).** L1's Plexus Adapter has no-op fallbacks exercised by stateless-mode tests (FC-7). The orchestrator works statelessly without Plexus; Plexus upgrades it to a learning system.

---

## Modules — brief

### L3 — Entry layer

| Module | One-line purpose |
|---|---|
| **Serving Layer** | Translates OpenAI-compatible wire protocol (`/v1/chat/completions`, `/v1/models`, SSE) into Session-scoped orchestrator interactions. Owns session-start flow including the typed `resolve_session_start_context` extension point (Phase 1 returns `[]`; Phase 2 reads from Plexus). |
| **Session Registry** *(extended in Cycle 4 per ADR-013)* | Identifies and continues a multi-request Session; maintains the structured-handoff artifact set (feature-list-with-monotonic-passes, append-only progress log, init.sh deterministic environment bootstrap) for Cluster 2 sessions; runs three-class write-gate validation (JSON schema, append-only constraint, signed-script tamper-detection) on artifact writes; resolves cluster declaration at session-start (default-required for cross-cluster ambiguity per disposition (i)). |
| **Bootstrapping Pipeline** | Operator-triggered batch ingestion of the library into Plexus as source material per AS-4. Currently deferred (WP-J). |
| **Orchestrator Configuration** | Loads and resolves per-session config: Budget defaults, Autonomy defaults, Plexus enablement, orchestrator Model Profile (ADR-011), Conversation Compaction thresholds (ADR-012), per-skill tier defaults for the eight Topaz skills (ADR-015), structural validation guard pattern set (ADR-017). |

### L2 — Runtime

| Module | One-line purpose |
|---|---|
| **Orchestrator Runtime** *(extended in Cycle 4 per ADR-012)* | Runs the ReAct loop. Imports only Budget Controller, Orchestrator Tool Dispatch, and Conversation Compaction (FC-4 amendment). Invokes compaction at each turn boundary; unaware of compaction internals. Aware of Routing Decisions and Conversation Compaction; unaware of summarization, Plexus, Autonomy, Calibration, or tier escalation. |
| **Orchestrator Tool Dispatch** *(extended in Cycle 4 per ADR-015 + ADR-017)* | Defines the closed five-tool surface (FC-5) and dispatches each call to its downstream service. Interposes (in order on `invoke_ensemble`): structural validation guard (response-text scan for phantom_tool_call assertion patterns); Autonomy Policy gate; Tier-Escalation Router (verdict→tier selection); `EnsembleExecutor.execute`; Calibration Gate post-result; Result Summarizer Harness. |
| **Result Summarizer Harness** | Unskippable interposition between ensemble completion and the orchestrator's context. Configured summarizer ensemble; raw-output escape hatch (ADR-004). |
| **Conversation Compaction** *(new in Cycle 4 per ADR-012)* | Runs the five-layer cheapest-first compaction pipeline at orchestrator turn boundaries. Layer 0 persist-large-tool-results (>50K char to disk); Layer 1 cache-edit; Layer 2 idle-expiry (60-min default); Layer 3 free summary via nine-section session-notes template (12K-token cap; zero LLM cost); Layer 4 LLM semantic summary with three-failure circuit-breaker (auto-resets at session start). |
| **Tier-Escalation Router** *(new in Cycle 4 per ADR-015)* | Selects per-dispatch Model Profile (cheap-tier or escalated-tier) for `invoke_ensemble` based on the dispatched ensemble's Topaz skill metadata and the Calibration Gate's verdict. Verdict→tier mapping: Proceed → cheap; Reflect → escalated; Abstain → typed `escalation_bypass` error. Honors ADR-011's session-boundary scope for the orchestrator's own LLM. |

### L1 — Domain policy

| Module | One-line purpose |
|---|---|
| **Composition Validator** | Validates a proposed composed ensemble against the existing reference graph using the same routine as load-time validation (FC-6). Stricter than load-time: enforces "compose from existing primitives only" (AS-6). |
| **Budget Controller** | Enforces turn and token limits at each ReAct iteration boundary (FC-10). Control plane, not model plane (AS-3). |
| **Autonomy Policy** | Gates orchestrator actions against the Session's Autonomy Level (FC-11). Surfaces composition events when configuration requires it. |
| **Calibration Gate** *(extended in Cycle 4 per ADR-014)* | Tracks Calibration state and produces dispatch-time calibration verdict (Proceed / Reflect / Abstain trichotomy). Verdict composition: AUQ verbalized confidence (System 1 + System 2 at default 0.85 threshold within 0.8–1.0 range), HTC trajectory features (token-level entropy, attention-weight distributions, decision-confidence trajectories), and ADR-007's existing post-hoc result-check signal. Time-decay windowing on trajectory features (60-min/100-signal dual-bound, linear decay). Three Abstain criteria (entropy collapse > 1.5σ; post-hoc hard failure; multiple drift criteria simultaneously). When ADR-016 channel is active, HTC features extracted once at L0 and propagated upward; otherwise extracted in-layer. |
| **Plexus Adapter** | Single place Plexus-aware code lives. No-op fallbacks when Plexus is absent (FC-7). WP-K (Plexus-active paths) deferred; WP-I shipped the skeleton. |
| **Calibration Signal Channel** *(new in Cycle 4 per ADR-016; conditional acceptance)* | Carries read-only calibration data upward from L0 (Ensemble Engine outputs) to L1 (Calibration Gate dispatch decisions), enforcing the five bounding mechanisms — (a) fresh-context isolation in the consumer, (b) time-decay windowing for the bias-compounding horizon, (c) categorical anchors via deterministic-tool-output (when ensemble has script-model slot), (d) periodic out-of-band audit dispatch (every 100 verdicts or 24 hours; severe drift triggers fail-safe), (e) read-only structural validation. Owns the upward L0→L1 read-only edge — the single narrow exception ADR-016 amends ADR-002 to permit. **Conditional acceptance:** synthetic-data validation (mechanism b) and structural-transfer validation (mechanism d) completed; first-deployment evidence on the cycle's North-Star benchmark closes the conditionality (or fires the falsification trigger). |

### L0 — Core (existing, unchanged per ADR-002)

| Module | One-line purpose |
|---|---|
| **Ensemble Engine** | Executes ensembles per the existing declarative DAG engine. Cross-ensemble references, depth limits, fan-out gather, child executor, immutable-vs-mutable state separation. |

---

## Client Tool Surface Commitment

**Decision.** The orchestrator's **internal** tool surface is exactly the five ADR-003 tools and no others. Client-declared tools (the `tools[]` array on a `/v1/chat/completions` request) become the orchestrator's **response surface**:

- The orchestrator's ReAct iterations call only the five internal tools.
- When a task step requires a client-side action (bash, file edit, etc.), the orchestrator closes the current turn with `finish_reason: tool_calls` and emits one or more client-tool `tool_calls[]` in the completion response.
- The client executes the tools and sends `role: tool` messages back in the next `/v1/chat/completions`. The orchestrator resumes the same Session's ReAct loop with those messages as observations.

**Provenance.** Committed in ARCHITECT 2026-04-20 on user direction. Honors ADR-003 strictly ("no others" refers to *internal* action surface; delegation at the turn boundary is response-surface behavior, not internal action). Supersedes the `interaction-specs.md` open-boundary note for the Tool User stakeholder.

**Scenario gate resolved (2026-04-22; Amendment #4).** The four stress scenarios from roadmap Open Decision Point #1 are written into `scenarios.md` §Client Tool Surface Commitment. All four are carried by Option C:

- Scenarios (a) and (b) — turn-boundary delegation path, Session continuity across a client-tool round trip.
- Scenario (c) — first-agent client-filesystem-file need carried by *pre-invoke* delegation: orchestrator reads at the prior turn boundary, folds into `invoke_ensemble`'s `input_data`.
- Scenario (d) — composed ensemble's un-predicted mid-execution client-tool need carried by the **retry pattern**: ensemble runs to completion; agent emits structured `needs_client_tool` signal; Result Summarization preserves it; orchestrator observes, closes next turn with client-tool delegation, re-invokes with client-tool result folded into `input_data`. The DAG engine never suspends; Layer 3 is unchanged.

Option D (mid-execution callback) would require amending ADR-001/ADR-002 and adding suspend/resume to the synchronous DAG phase loop. Out of scope for Cycle 1; reopens only at the architectural ADR level. The retry pattern is conditional on composed ensembles following a `needs_client_tool` convention; the enforcement mechanism is a build-time layering decision (roadmap Open Decision Point #8).

---

## Roadmap

See [`./roadmap.md`](./roadmap.md) for active work packages, classified dependencies, transition states, completed work log, and open decision points.

**Cycle 1 status (closed 2026-04-29):** TS-1 (stateless orchestrator serving OpenCode) reached at WP-F close; TS-2 (stateless baseline) reached at WP-H close; WP-I (Plexus Adapter skeleton) shipped with no-op fallbacks. WP-K (Plexus-active) and WP-J (Bootstrapping) deferred.

**Cycle 4 ARCHITECT close (2026-05-08):** Six new ADRs (012-017) integrated into the system design. Three new modules (Conversation Compaction L2; Tier-Escalation Router L2; Calibration Signal Channel L1 conditional). Four module extensions (Session Registry; Orchestrator Runtime; Orchestrator Tool Dispatch; Calibration Gate). Five new system-level fitness criteria (FC-14 through FC-18); ADR-076 qualitative-claim decomposition complete; design audit clean. Cycle 4 BUILD comprises eight WPs (WP-A4 through WP-H4) per the conformance-scan-recommended sequence: shared `LlmOrcStructuralError` base class first; FC-2/FC-3 automated checks; ADR-017 → ADR-013 → ADR-012 → ADR-014 → ADR-015 → ADR-016 (last; conditional on first-deployment evidence per ADR-016 §"Concrete monitoring specification").

---

## Design Amendment Log

| # | Date | What Changed | Trigger | Status |
|---|------|-------------|---------|--------|
| — | 2026-04-20 | Initial system design | ARCHITECT phase | Superseded by v2.0 |
| 1 | 2026-04-20 | Demote Context Injection Stage from module to typed function `resolve_session_start_context` owned by Serving Layer | ARCHITECT reflection-time Grounding Reframe (Item 1) | Current |
| 2 | 2026-04-20 | Mark Client Tool Surface Commitment scenario-gated; WP-F blocked until stress scenarios written | ARCHITECT reflection-time Grounding Reframe (Item 2) | Resolved by Amendment #4 |
| 3 | 2026-04-21 | Move Result Summarizer Harness dependency edge from Orchestrator Runtime to Orchestrator Tool Dispatch; FC-4 amended to exclude RSH from Runtime's allow list | WP-C close detected `Runtime → RSH` was inconsistent with RSH's own rationale; structural test already enforced corrected reading | Current |
| 4 | 2026-04-22 | Resolve Amendment #2 scenario gate. Four WP-F stress scenarios written; Option C carries all four (turn-boundary, session continuity, pre-invoke delegation, retry pattern). Option D out of scope this cycle | WP-F scenario-gate resolution (DECIDE mini-cycle) | Current |
| 5 | 2026-04-29 | Restructure per ADR-083 (F-pattern orientation lead) and ADR-084 (Pattern B companion-file split). Architectural drivers, full module decomposition, responsibility matrix, dependency graph, integration contracts, fitness criteria, test architecture, and new Appendix A per-phase susceptibility-snapshot briefs relocated to `system-design.agents.md`. Slim human-facing surface here. | Cycle 2 entry conformance to v0.8.5 (ADR-083, ADR-084); CHANGELOG note "system-design.md restructured to slim human-facing v14.0 with companion `system-design.agents.md`" applied to scoped corpus | Current |
| 6 | 2026-05-06 | Layering rule amended: read-only signal channel from L0 to L1 permitted (calibration-only), gated by five bounding mechanisms per ADR-016. All other layer pairs remain prohibited. Six new ADRs accepted (Conversation Compaction five-layer pipeline ADR-012; Session Registry initializer-then-resume schema ADR-013; Calibration Gate trajectory-level extension ADR-014; per-role tier-escalation router ADR-015; upward L0→L1 signal channel ADR-016 conditional acceptance; tool-call structural validation guard ADR-017). One ADR candidate deferred (ADR-004 amendment for Result Summarizer Harness reconsideration — below evidentiary threshold). Implementation gap inventory recorded in conformance-scan-cycle-4-decide.md; full BUILD work pending. | Cycle 4 DECIDE close (Mode B+ → DECIDE close shape); cross-references ADR-016 (load-bearing decision), conformance-scan-cycle-4-decide.md, argument-audit-cycle-4-decide.md (round 2 clean) | Current |
| 7 | 2026-05-08 | Cycle 4 ARCHITECT integration of ADRs 012-017. Three new modules: Conversation Compaction (L2; ADR-012 five-layer pipeline + circuit-breaker + nine-section session-notes template); Tier-Escalation Router (L2; ADR-015 verdict→tier mapping + Topaz skill metadata + per-skill tier defaults); Calibration Signal Channel (L1; ADR-016 read-only L0→L1 channel + five bounding mechanisms; conditional acceptance). Four module extensions: Session Registry (ADR-013 structured-handoff artifacts + write-gate validation + cluster determination); Orchestrator Runtime (ADR-012 turn-boundary compaction invocation; FC-4 amendment to add `conversation_compaction` to allowed imports); Orchestrator Tool Dispatch (ADR-017 structural validation guard + ADR-015 router interposition); Calibration Gate (ADR-014 verdict trichotomy + AUQ + HTC + time-decay windowing). Seven new dependency edges including the load-bearing read-only L0→L1 upward exception. Five new fitness criteria (FC-14 cheapest-first compaction; FC-15 every dispatch through tier router; FC-16 signal channel read-only; FC-17 typed errors derive from `LlmOrcStructuralError` base; FC-18 every ensemble declares Topaz skill). ADR-076 qualitative-claim decomposition: 24 fitness properties recorded inline; 3 direction-not-constraint annotations (ADR-013 narrative continuity; ADR-016 aggregate bias-bound; ADR-017 pattern-set adequacy). | Cycle 4 ARCHITECT phase (re-scoped 2026-05-08 from Mode B+ → DECIDE close to Mode A — extended through ARCHITECT and BUILD); cross-references conformance-scan-cycle-4-decide.md (BUILD sequencing recommendation honored), cycle-4-decide-gate.md (asymmetric-grounding finding carried forward to OQ #14), ADR-016 §"Concrete monitoring specification" (conditional-acceptance trigger artifact + sweep responsibility recorded). | Current |

---

## Provenance

The full architectural-drivers table, module decomposition, responsibility matrix, dependency graph, integration contracts, fitness criteria, test architecture, and Appendix A per-phase susceptibility-snapshot briefs are in [system-design.agents.md](./system-design.agents.md). That companion file is the verification surface for FC-1 through FC-13 and the canonical brief source for every phase boundary's susceptibility-snapshot dispatch.
