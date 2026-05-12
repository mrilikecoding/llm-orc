# RDD Cycle Status — Agentic Serving (Scoped)

**Artifact base:** `docs/agentic-serving/`
**Plugin version at last cycle close:** v0.8.5
**Migration version:** 0.8.5 (`housekeeping/.migration-version`)

## Cycle Stack

### Active: Cycle 4 — Supported design methods for cheap-orchestrator + ensembles (long-session agentic coding) — extended Mode A

**Cycle number:** 4
**Started:** 2026-05-04
**Re-scoped:** 2026-05-08 (from Mode B+ → DECIDE close to Mode A — extended through ARCHITECT and BUILD)
**Current phase:** build
**In-progress phase:** build
**Cycle type:** standard
**Plugin version:** v0.8.5
**Artifact base:** `docs/agentic-serving/`
**BUILD mode:** gated (declared at re-scoping; can be revised to `auto` at BUILD entry)

**Original close shape (now superseded by re-scoping):** Mode B+ → DECIDE close (practitioner decision 2026-05-04). Path: research → discover (update mode) → model (update mode) → decide on the seven ADR candidates.

**Re-scoped close shape (2026-05-08):** **Mode A — extended through ARCHITECT and BUILD.** Practitioner re-scoped at decide-gate close on the rationale that the cycle's *elaboration-by-evidence framing commitment* is empirically-grounded by design — closing at DECIDE makes the framing a deferral rather than a test. ARCHITECT and BUILD provide the practice-based evidence the framing commitment named as the falsification surface. PLAY is open (decision deferred to BUILD close — practitioner will assess whether to run PLAY in this cycle or as a follow-up).

**Re-scoping rationale (recorded 2026-05-08):** the decide-gate finding (asymmetric grounding-mechanism rigor across cross-layer stages — OQ #14) is *better grounded by first-deployment evidence than by pre-BUILD theoretical research*. The cycle's elaboration-by-evidence framing commitment said practice-based evidence is what would change the framing; closing at DECIDE deferred that test. Re-scoping to extend through BUILD honors the commitment by making the framing testable in the cycle that produced it.

## Phase Status

| Phase | Status | Artifact | Key Epistemic Response |
|-------|--------|----------|----------------------|
| RESEARCH | ✅ Complete (essay 005 audit-clean at round-5; gate completed with belief-mapping on ADR candidate #6; Grounding Reframe triggered per ADR-059 with two grounding actions recorded for DECIDE entry; reflections written; research log archived as `005-layer-conditional-composition.md`) | essay `005-layer-conditional-composition.md` + reflection `reflections/005-layer-conditional-composition.md` + lit-reviews `005a/b/c-` + spike `005d-` + archived research log `005-` + 5 audit rounds + susceptibility snapshot + gate reflection note | Practitioner caught content-selection sycophancy gap at gate (cycle/feedback-loop concern on ADR candidate #6 not surfaced by audit apparatus despite literature evidence in context) — produced load-bearing constraint that ADR candidate #6's drafting in DECIDE includes five bounding mechanisms with explicit asymmetric implementation-readiness, and that the elaboration-vs-reorganization architectural choice be posed as DECIDE-entry belief-mapping rather than inherited from essay 005's Conclusion. |
| DISCOVER | ✅ Complete (update mode 2026-05-05) | updated `product-discovery.md` + susceptibility snapshot + gate reflection | Practitioner rejected the alternative reading on tension #8; introduced calibration-baseline ensemble at install/startup as third design surface; capability-floor specification has two compatible design surfaces (static spec + runtime probe), both as DECIDE-phase scenario candidates. |
| MODEL | ✅ Complete (update mode 2026-05-05) | updated `domain-model.md` (§Methodology Vocabulary added; OQ #9–#13 added; Amendment Log entries #2–#4) + companion edit to `product-discovery.md` vocabulary table + susceptibility snapshot + gate reflection | Practitioner adjudicated the "externalized structured state" relocation with operator-experience warrant; two of three borderline terms (`tier escalation`, `initializer-then-resume`) retained in product vocabulary as operator voice; one (`externalized structured state`) relocated. |
| DECIDE | ✅ Complete (2026-05-08; Mode A re-scoping declared at gate close) | 6 new ADRs (012-017) + deferred candidate #5 document + ADR-002 partial-update header + scenarios.md (29 new scenarios + Cycle Acceptance Criteria Table) + interaction-specs.md (10 new tasks) + 2 DECIDE-phase spike research logs (005e, 005f) + argument audit (clean at round 2) + conformance scan + susceptibility snapshot (Grounding Reframe applied in-cycle) + gate reflection | DECIDE-entry Grounding Action 2 resolved with reading (a) elaboration-by-evidence. Decide-gate pre-mortem (practitioner verbatim: *"we missed clear mechanisms by which to ground various cross-layer stages"*) surfaced asymmetric grounding-mechanism rigor — ADR-016 received 5 bounding mechanisms with operational grounding while ADR-014→ADR-015, ADR-013, ADR-012, ADR-017, and Plexus integration boundary received less rigor. Logged as domain-model OQ #14. The decide-gate finding is the *load-bearing motivation for re-scoping* — first-deployment evidence is the natural validation surface for the asymmetric-rigor concern. |
| ARCHITECT | ✅ Complete (gate closed 2026-05-11; deliverables completed 2026-05-08; pre-BUILD spikes α and β completed 2026-05-11; full integration of spike outcomes per practitioner disposition) | system-design.md v3.0 + architect-gate-close extension (Tier-Escalation Router module ADR-018 extension; FC-19 and FC-20; L1→L2 dependency-graph annotation update); system-design.agents.md v3.0 (Cycle 4 architectural drivers; 3 new module decompositions; 4 module extensions; 13+ responsibility-matrix rows; 7 new dependency edges; 7 new integration contracts plus shared-typed-error infrastructure; FC-14 through FC-20; 8 new boundary integration tests; 6 new invariant enforcement tests; updated A.5 architect-phase brief in Appendix A; ADR-076 qualitative-claim decomposition); roadmap.md (Cycle 1 WPs migrated to Completed Work Log; Cycle 4 WP-A4 through WP-H4 added with classified dependencies; WP-G4 restructured into WP-G4-1 + WP-G4-2 at architect-gate close per ADR-018; transition states TS-4 through TS-7); susceptibility-snapshot-cycle-4-architect.md (low-to-moderate susceptibility; no Grounding Reframe triggered); ADR-018 added at architect-gate close (Tier-Escalation Router periodic audit dispatch — ADR-016 mechanism (d) analog); ADR-015 partial-update header pointing to ADR-018; domain-model OQ #14 Amendment Log entry #8 (partial closure for L1→L2 stage); gate reflection note `housekeeping/gates/cycle-4-architect-gate.md`; ORIENTATION.md regenerated; Spike α research log `essays/research-logs/005g-spike-topaz-skill-classification.md` + companion ensemble `.llm-orc/ensembles/spike-005g-skill-classifier.yaml`; Spike β research log `essays/research-logs/005h-spike-bounding-mechanism-transfer-l1-l2.md` | Gate engagement spanned two sessions 2026-05-11. **First session (pre-spike):** constraint-removal question composed on the L1→L2 verdict→router stage — *what would the architecture look like if the Tier-Escalation Router → Calibration Gate edge couldn't defer to BUILD evidence and needed a grounding mechanism inline now?* Practitioner approved running both Spike α and Spike β. **Second session (post-spike):** both spikes ran via `rdd:spike-runner` parallel dispatch; Spike β disposition partial transfer with (d)-analog as load-bearing addition (three mechanisms inherited; one inapplicable; one transfers cleanly with novel design work); Spike α disposition classification clean (21 of 21 ensembles satisfy clean-primary criterion). Practitioner approved Full Integration (all 7 actions) at gate close. **Sub-Q6 structurally closed** by coupling to ADR-018's (d)-analog audit's escalation-vs-outcome correlation drift criterion. **OQ #14 partially closed** for the L1→L2 stage; four other cross-layer stages remain Cycle 5+ research territory. **Surprise finding:** OQ #14 and Sub-Q6 are addressed by the same mechanism — a coupling not visible from the system-design alone. |
| BUILD | ▶ In Progress (entered 2026-05-11; **WP-A4 + WP-B4 + WP-C4 + WP-D4 + WP-E4 + WP-F4 closed 2026-05-11**) | WP-A4 commits `cc0d94f` + `7c2f64e`; WP-B4 commit `1701a22`; WP-C4 commit `9116793`; WP-D4 commit `ded9e2d`; WP-E4 commits `e1ff875` + `31e261f` + `b574689` + `36f540c`; WP-F4 commit `35f48ce`. WP-A4 through WP-E4 as previously recorded. **WP-F4:** `src/llm_orc/agentic/calibration_gate.py` (extended — added `CalibrationVerdict` Literal, `DriftLevel` Literal, `AbstainCriterion` Literal, `TrajectoryFeatures` frozen dataclass, `DispatchContext` frozen dataclass, `_WindowedDispatchSample` frozen dataclass, `WallClock` Protocol + `_SystemWallClock`, `CalibrationAbstainError` typed-error subclass, four operationally-tunable defaults `DEFAULT_AUQ_CONFIDENCE_THRESHOLD=0.85` + `DEFAULT_TRAJECTORY_WINDOW_MINUTES=60.0` + `DEFAULT_TRAJECTORY_WINDOW_DISPATCHES=100` + `DEFAULT_ENTROPY_COLLAPSE_SIGMA=1.5`, `CalibrationGate.verdict_for(session_id, ensemble_name, dispatch_context) -> CalibrationVerdict`, side-effect-free `abstain_criterion_for` for consumer-side typed-error construction, internal `_prune_trajectory_window` + `_windowed_running_entropy` + `_entropy_collapse_triggers` + `_abstain_criterion` helpers, `_RunningEntropy` frozen dataclass); `tests/unit/models/test_structural_errors.py` (added `TestCalibrationAbstainErrorAsConcreteSubclass` — 4 tests); `tests/unit/agentic/test_calibration_gate.py` (added 36 tests across `TestADR014Defaults`, `TestCalibrationGateAdr014ConstructorValidation`, `TestVerdictTrichotomyProceed`, `TestVerdictTrichotomyReflect`, `TestVerdictTrichotomyAbstain`, `TestCalibrationGateAbstainCriterionExtraction`, `TestVerdictIsExhaustiveAndDeterministicGivenInputs`, `TestTimeDecayWindowingDualBoundLinear`, `TestVerdictComputationWorksWithoutSignalChannel`, `TestAdr007PostHocCalibrationUnchangedUnderAdr014`, `TestCalibrationAbstainErrorRaiseAndDispatchContext`). Full suite 2516 passing; mypy strict + ruff + complexipy + bandit + vulture all clean. | **WP-A4** (shared `LlmOrcStructuralError` base class) — first concrete subclass migrated with no-regression behavior; FC-17 coverage 1 of 8 typed-error surfaces. **WP-B4** (FC-2 layering + FC-3 cycles automated checks) — AST-based static scan over `src/llm_orc/agentic/*.py` + Ensemble Engine; ADR-016 read-only L0→L1 signal-channel exception pre-declared in `_ALLOWED_UPWARD_EDGES` for WP-H4; layer map fail-closes on unclassified new modules; TYPE_CHECKING-guarded imports excluded. **WP-C4** (ADR-017 phantom_tool_call guard) — first behavioral typed-error producer using `LlmOrcStructuralError`; FC-17 coverage now 2 of 8 typed-error surfaces. The structural validation guard is interposed by **Orchestrator Tool Dispatch** per system-design.agents.md L107-119 (Runtime calls `tool_dispatch.validate_response(response.content, tool_call_names)`; FC-4 narrow Runtime imports preserved — guard module is not in Runtime's allowlist). Default pattern set is **minimal rather than calibrated** per ADR-017 §"Minimal default pattern set" (spike evidence does not support a richer default); operator-extensibility surface lives in `OrchestratorConfig` (L3) as `orchestrator.tool_call_validation_patterns`. Conservative false-positive discipline: future-intent patterns ("I will call X") are NOT in defaults; rejected alternative (f). On phantom detection the rejected response's prose is NOT surfaced to the client (would mislead); the orchestrator's reasoning surface receives a structural-feedback diagnostic via injected `role:user` message and reformulates on the next iteration. **WP-D4** (ADR-013 Session Registry structured-handoff artifacts) — three artifact components (`FeatureListStore` with monotonic-passes JSON schema; `ProgressLog` with append-only constraint; `InitScriptGate` with SHA-256 hash gate) plus the `write_gate_rejection` typed error (FC-17 coverage now **3 of 8 typed-error surfaces**). The artifact components live in a new `session_artifacts` module at L3 alongside Session Registry; responsibility split is **Session Registry owns the session-shape decisions** (cluster determination at session-start, artifact-set activation), the **artifact module owns the structural validation logic**. Validation classes are sub-discriminated via `dispatch_context["validation_class"]` (`feature_list_schema` | `progress_log_append_only` | `init_sh_integrity`). Schema and append-only violations are `reformulate`; init.sh hash mismatch is `operator_intervention_required` because hash rotation is structurally outside the orchestrator's tool surface. `InitScriptGate.rotate_hash()` is a pure computation returning the new hash — it does NOT mutate the gate's recorded hash in place, preventing a hostile orchestrator from unlocking execution through the rotation method. Cluster determination implements ADR-013 disposition (i): cross-cluster ambiguity, multi-cluster declarations, `None`, empty lists, and unrecognized cluster names ALL default to `cluster_2` so the artifact set is active. FC-2 layer map adds `session_artifacts` at L3; FC-4 unaffected (Runtime imports neither module). Three T1 prerequisites + two ADR surfaces complete; ready for **WP-E4** (ADR-012 Conversation Compaction five-layer pipeline). **WP-E4** (ADR-012 Conversation Compaction five-layer pipeline) — new `conversation_compaction.py` module at L2 implements the cheapest-first pipeline (Layer 0 persist-large-tool-results > 50K chars; Layer 1 cache-edit structural placeholder; Layer 2 idle-expiry 60-min default; Layer 3 free-summary-via-nine-section-session-notes that replaces older history with synthetic system message; Layer 4 LLM semantic summary with three-failure circuit-breaker). Fourth `LlmOrcStructuralError` subclass `CompactionLayer4FailureError` (`error_kind="compaction_layer_4_failure"`, `recovery_action_required="operator_intervention_required"`); FC-17 coverage now **4 of 8 typed-error surfaces**. `CompactionDefaults` value-type owned by L2 module per FC-2 layering; `orchestrator_config.py` (L3) imports and composes (downward dependency). FC-4 amended to allow `conversation_compaction` in Runtime's import set; Runtime invokes `compact()` at every turn boundary before `generate_with_tools` per system-design.agents.md L612. Serving Layer wires `get_conversation_compaction()` singleton into `_build_runtime` with persistence root under operator's global config dir; Layer 4 summarizer left unconfigured by default (operator opts in via `orchestrator.compaction.summarizer_ensemble`). Four undecided-territory dispositions recorded: (a) Layer 1 cache-edit is structural placeholder until llm-orc grows a provider-cache abstraction; (b) Layer 3 nine-section template update logic is minimal worklog-only heuristic; (c) aggregate `trigger_token_count` defaults to 100,000; (d) Layer 4 `summarizer_ensemble` defaults to `None`. Cycle-acceptance criterion (Step 5.5 row 5) verified at integration layer by `test_compaction_holds_context_below_threshold_across_long_session` — 10-turn fixture with 5K-char tool result per turn; ≥ 95% of LLM-input contexts stay at or below trigger threshold (FC-14 fitness criterion). **WP-F4** (ADR-014 Calibration Gate trajectory-level extension) — verdict producer extends Calibration Gate (L1) with the dispatch-time trichotomy `CalibrationVerdict` = `proceed` \| `reflect` \| `abstain`. Fifth `LlmOrcStructuralError` subclass `CalibrationAbstainError` (`error_kind="calibration_abstain"`, `recovery_action_required="abstain"` — the four-value Literal entry that pre-existed for this surface); FC-17 coverage now **5 of 8 typed-error surfaces**. AUQ verbalized-confidence drives the System 2 binary threshold (default 0.85, operationally tunable within the literature-supported 0.8-1.0 range per Chuang et al. arXiv:2502.04428's UQ-method-dominates-threshold finding); HTC-style trajectory features (token-level entropy) feed the entropy-collapse Abstain criterion against a time-decay-windowed running mean. Three Abstain criteria per ADR-014 §"Calibration verdict": entropy-collapse (recent_token_entropy < running_mean − 1.5σ); post-hoc result-check hard failure (consumer-supplied bool); severe drift verdict from ADR-016 mechanism (d) (None when channel inactive, per ADR-014 §"Feature-extraction location"). Time-decay windowing dual-bound — 60-minute time window OR 100-dispatch count cap (whichever shorter), with linear-decay weights from 1.0 (most recent) to 0.0 (window edge). **Zero-stdev windows do not form a basis** for the 1.5-sigma comparison ("more than 1.5 standard deviations below mean" is statistically undefined when σ=0); the criterion degrades to Proceed in that case rather than collapsing to a strict mean comparison. **ADR-007 first-N post-hoc mechanism preserved unchanged** — `check_and_record`/`mark_composed`/`status` semantics identical; `verdict_for` composes additively. Side-effect split: `verdict_for` records into the trajectory window; `abstain_criterion_for` is side-effect-free for consumers that need the criterion to construct `CalibrationAbstainError`. The verdict-producer surface is consumed by **Tier-Escalation Router (WP-G4-1)**, which lands next; Router-side integration verification of cycle-criterion #4 (verdict trichotomy → router action) is deferred to WP-G4-1's BUILD scope. Operator-tunable parameters validated at construction: AUQ threshold must be in [0.0, 1.0]; window minutes > 0; window dispatches ≥ 1; entropy-collapse sigma > 0. |
| PLAY | ☐ Open — decision deferred to BUILD close | — | — |
| SYNTHESIZE | ☐ Optional | — | — |

## ARCHITECT-entry context (the carry-forwards a fresh session inherits)

The fresh session entering ARCHITECT inherits the DECIDE-phase deliverable corpus plus the following load-bearing carry-forwards:

### Settled premises going into ARCHITECT

1. **Six accepted ADRs** form the cycle's design-decision deliverable: ADR-012 (Conversation Compaction five-layer pipeline), ADR-013 (Session Registry initializer-then-resume schema with write-gate validation), ADR-014 (Calibration Gate trajectory-level extension), ADR-015 (Per-role tier-escalation router via OI-MAS), ADR-016 (Upward L0→L1 read-only signal channel — *conditional acceptance*), ADR-017 (Tool-call structural validation guard).

2. **One deferred ADR** (`adr-deferred-005-summarizer-harness-reconsideration.md`) awaits a Cycle 5+ spike; ARCHITECT does not need to resolve this.

3. **ADR-002 partial-update header** records the layering-rule amendment per ADR-016.

4. **Domain-model and system-design downstream sweep complete** at DECIDE close: vocabulary entries promoted, layering rule amended, ORIENTATION.md updated. ARCHITECT inherits the current-state artifacts as the substrate.

5. **Argument audit clean at round 2; conformance scan zero structural violations.** No structural debt to resolve before ARCHITECT.

6. **DECIDE-phase spike validations complete** for ADR-016 mechanisms (b) and (d) at structural/logical level; first-deployment evidence remains the operational validation surface.

### Open commitments ARCHITECT must honor

1. **ADR-016 conditional acceptance.** Per its concrete monitoring specification, the cycle's BUILD work is what produces first-deployment evidence. ARCHITECT's responsibility allocation must keep mechanism (b) and mechanism (d) within ADR-002's L1 layer (the elaboration-by-evidence falsification trigger fires if the mechanisms require module-shape orthogonal to L0–L3). If ARCHITECT discovers the mechanisms need a top-level module outside L0–L3, the falsification trigger fires — the reorganization branch re-opens; ADR-016 is re-deliberated.

2. **OQ #14 (asymmetric grounding-mechanism rigor across cross-layer stages).** ARCHITECT may surface concrete grounding-mechanism gaps as it allocates responsibilities — five cross-layer stages were flagged at the decide gate as having less rigor than ADR-016's stage. ARCHITECT can either (a) surface the gaps for Cycle 5+ research, (b) propose grounding mechanisms inline as architectural-driver entries, or (c) note that BUILD evidence will inform what grounding the other stages need.

3. **BUILD sequencing recommendation from conformance scan** (informs ARCHITECT's responsibility allocation): ADR-017 first (most bounded; codifies typed-error pattern from commit `9f86d0b`); shared `LlmOrcStructuralError` base class second (unlocks all five typed-error producers); FC-2/FC-3 automated checks third (currently absent as test artifacts); then per-ADR work in dependency order — ADR-013 → ADR-012 → ADR-014 → ADR-015 → ADR-016 (last; conditional on first-deployment evidence per monitoring specification).

4. **Three structural prerequisites for BUILD** to be specified in ARCHITECT's responsibility matrix: FC-2 automated test (AST-based per-module import layering check), FC-3 automated test (dependency-graph cycle detection), shared `LlmOrcStructuralError` base class with the four common fields specified in ADR-017 (`error_kind`, `dispatch_context`, `recovery_action_required`, `operator_diagnostic`).

### Grounding Reframe action carried forward (from decide gate)

**ADR-015 autonomous-routing evidence gap** — Sub-Q6 from essay 005 (multi-iteration routing reliability at North-Star benchmark session length is empirically unvalidated) was carried into ADR-015 as a Consequences §Neutral entry. ARCHITECT should attend: when allocating responsibilities for the per-role tier-escalation router, the routing-quality assumption is empirically open. Operators interpreting escalation-rate calibration evidence may be reading routing-noise rather than tier-configuration mismatches until first-deployment evidence resolves Sub-Q6.

### Advisory carry-forwards from decide-gate susceptibility snapshot

- **ADR-016 cross-session value is Plexus-conditional.** Operators evaluating the cross-layer calibration channel's deployment cost should scope the value claim to their Plexus-activation status — full value (cross-session calibration stabilization under AS-5) requires Plexus; in-session value is preserved without Plexus.
- **ADR-015 Attention-MoA orchestrator-as-aggregator dependency.** Deployment evidence should track whether escalation gains are concentrated where member-model quality is the bottleneck (where ADR-015 helps) versus where orchestrator-aggregation is the binding constraint (where ADR-015 may not help; the orchestrator's own Model Profile becomes the design surface, which is ADR-011 territory).

## Pre-BUILD spike plan (architect-gate continuation; ✅ completed 2026-05-11)

The architect-gate engagement on 2026-05-11 produced practitioner approval to run pre-BUILD spike work before closing the architect gate's commitment gating. Two spikes were approved; both ran in parallel via `rdd:spike-runner` subagent dispatch on the same day. Outcomes recorded below; full integration of spike outcomes is captured in `housekeeping/gates/cycle-4-architect-gate.md`.

**Spike outcomes summary (2026-05-11):**

| Spike | Method | Disposition | Integration |
|-------|--------|-------------|-------------|
| α — Topaz skill classification | Cheap-orchestrator dispatch (local Ollama `qwen3:8b`; $0) | Classification is clean (21 of 21 production-style ensembles satisfy clean-primary criterion; max 2nd-ranked 40%) | ADR-015 primary-skill framing stands; no amendment to that aspect. Distribution finding informs WP-G4-1 operator docs (coverage hedge load-bearing) |
| β — Bounding-mechanism transfer audit | Analytical only (no LLM dispatch) | Partial transfer; (d)-analog audit dispatch is the load-bearing addition; three mechanisms inherited; one inapplicable | ADR-018 records the (d)-analog audit dispatch as amendment to ADR-015; OQ #14 partial closure for L1→L2 stage; Sub-Q6 structural closure via coupling |

### Spike α — Topaz skill classification adequacy on the existing ensemble library

**Question:** Do the eight Topaz skills (`code_generation`, `tool_use`, `mathematical_reasoning`, `logical_reasoning`, `factual_knowledge`, `writing_quality`, `instruction_following`, `summarization`) cleanly partition the existing library's ensembles, or do many ensembles obviously span 2+ skills?

**Why it matters at BUILD entry:** ADR-015's primary-skill framing is load-bearing for the per-role tier-escalation router (Tier-Escalation Router module per system-design.agents.md v3.0). ADR-015's rejected alternative (b) was per-ensemble tier alternatives, rejected on configuration-burden grounds. If many ensembles don't have a clean primary skill, the primary-skill framing is brittle and per-ensemble alternatives may have been the right call — ADR-015 would need amendment before WP-G4 BUILD work begins.

**Spike shape.** Walk `.llm-orc/ensembles/*.yaml` (small set; mostly WP-shipped defaults plus the spike artifacts retained per the corpus-close retention directive). For each ensemble, dispatch the cheap-orchestrator (free-tier model per practitioner cost preference) to declare a *primary* Topaz skill and rank the other seven by relevance. Flag ensembles where the second-ranked skill is ≥ 80% as relevant as the first (no clean primary).

**Cost.** ~30 minutes; free-tier model; pure orchestration; no codebase change.

**Disposition options when the spike completes:**
- **Classification is clean (most ensembles have an obvious primary):** ADR-015 stands; WP-G4 proceeds as designed in roadmap.md.
- **Many non-clean-primary cases:** amend ADR-015 to permit per-ensemble override of per-skill tier defaults; WP-G4 design changes (configuration surface grows; primary-skill metadata becomes a hint rather than a constraint).
- **Mixed (some clean, some span):** scope-condition the amendment; document the boundary in ADR-015's Consequences.

**Spike artifact location:** `essays/research-logs/005g-spike-topaz-skill-classification.md` (per corpus spike-retention directive — retained until corpus close).

### Spike β — Bounding-mechanism transfer audit for L1→L2 verdict→router stage

**Question:** Can the bounding-mechanism pattern (a)–(e) from ADR-016 actually transfer to the Tier-Escalation Router → Calibration Gate edge (the L1→L2 verdict→router stage), or does it not apply structurally?

**Why it matters at BUILD entry:** the asymmetric-grounding finding (OQ #14, decide-gate 2026-05-08) flagged five cross-layer stages with less rigor than ADR-016. Practitioner chose option (c) — defer to BUILD evidence — uniformly across all five. If one stage transfers analytically to (a)–(e)-style grounding (without first-deployment evidence), that stage's grounding moves from "BUILD evidence will inform" to "inline grounding mechanism" — partially closing OQ #14 ahead of BUILD rather than deferring entirely. Methodologically analogous to spike (d) on mechanism (d) at DECIDE close (research log `005f-spike-adr016-d-structural-transfer-audit.md`) which was largely-clean transfer with three specification gaps. **The verdict→router stage is the highest-priority OQ #14 candidate** because Sub-Q6's routing-reliability evidence gap (ADR-015 §Consequences §Neutral) operates exactly here.

**Spike shape.** Property-by-property analytical audit of mechanisms (a)–(e) against the L1→L2 verdict→router edge:
- **(a) fresh-context isolation in consumer** — does the router need fresh context for each verdict consumption? Likely yes; trivially transfers.
- **(b) time-decay windowing** — does verdict history influence current routing? If yes, windowing applies; if no, mechanism doesn't transfer.
- **(c) categorical anchors via deterministic-tool-output** — does the router consume deterministic signals? Not directly; the router consumes calibration verdicts. Mechanism is ensemble-composition-conditional and may not transfer.
- **(d) periodic out-of-band audit** — should an auditor check routing-vs-tier-config decoupling over time? Plausibly yes; transfers cleanly.
- **(e) read-only structural validation** — does the router validate verdict schema before acting? Already part of FC-17's typed-error discipline (`escalation_bypass` typed error pattern).

**Cost.** ~45–60 minutes; analytical work (no LLM dispatch needed — free-tier compatible by construction); writes a research log entry.

**Disposition options when the spike completes:**
- **Transfer is clean:** propose grounding mechanisms inline as drivers for WP-G4 (router module); WP-G4 design changes to include an analog audit dispatch for routing-vs-tier-config decoupling. Partial close on OQ #14 for the L1→L2 stage.
- **Transfer is partial:** the partial-transfer findings inform which mechanism-analogs the router needs vs which are mechanism (d)'s territory in the audit dispatch ADR-016 already specifies.
- **Transfer fails:** confirms choice (c) deferral was correct for at least this stage; documents the negative result; OQ #14 remains Cycle 5+ research territory for this stage.

**Spike artifact location:** `essays/research-logs/005h-spike-bounding-mechanism-transfer-l1-l2.md` (per corpus spike-retention directive).

### Architect-gate continuation after spikes

When both spikes are complete (or one, if practitioner elects partial scope), the architect gate resumes with:

1. **Spike outcomes integrated** — any ADR amendments (e.g., ADR-015 amendment from spike α) recorded as supersession events with downstream sweep; any new grounding mechanisms from spike β proposed as drivers in WP-G4 + roadmap update.
2. **Commitment-gating outputs finalized** — settled premises and open questions going into BUILD are recorded in the architect gate reflection note based on the spike outcomes.
3. **Gate reflection note written** to `housekeeping/gates/cycle-4-architect-gate.md` (per ADR-070 placement; ADR-085 `.rdd/` migration is deferred per migration-window allowance).
4. **ORIENTATION.md regenerated** if the spike outcomes change Cycle 4's design surface materially.
5. **Phase advance** to BUILD: `**Current phase:**` set to `build`; `**In-progress phase:**` set to `build`; `**In-progress gate:**` field removed (gate complete).

## Suggested fresh-session handoff prompt for WP-G4-1 entry (2026-05-11)

> Continue Cycle 4 BUILD at **WP-G4-1 — ADR-015 per-role tier-escalation router (core)**. WP-A4 + WP-B4 + WP-C4 + WP-D4 + WP-E4 + WP-F4 closed 2026-05-11 (commits `cc0d94f`, `7c2f64e`, `1701a22`, `9116793`, `ded9e2d`, `e1ff875`, `31e261f`, `b574689`, `36f540c`, `35f48ce`). Full suite 2516 passing; all linters clean. **BUILD mode: gated** (per-scenario-group EPISTEMIC GATES with AID cycle).
>
> **Read in this order before opening WP-G4-1:** (1) `housekeeping/cycle-status.md` for the cycle's current state; (2) `decisions/adr-015-per-role-tier-escalation-router.md` in full (verdict→tier mapping; Topaz skill metadata schema; per-skill tier-defaults configuration model); (3) `decisions/adr-018-tier-escalation-router-d-analog-audit.md` (the ADR-018 (d)-analog audit dispatch responsibility — note this is **WP-G4-2** territory, not WP-G4-1; WP-G4-1 ships the core router); (4) `scenarios.md` §Per-Role Tier-Escalation Router (ADR-015 scenarios) — Code-generation cheap-tier, Reflect escalated-tier, Abstain escalation-bypass, missing-skill rejection, per-skill not-per-ensemble defaults, ADR-011 orchestrator-profile preservation; (5) `system-design.agents.md` §Module: Tier-Escalation Router (L107-119 interposition order; L214-229 module spec; L487-497 Tool Dispatch → Tier-Escalation Router and Tier-Escalation Router → Calibration Gate integration contracts); (6) `src/llm_orc/agentic/calibration_gate.py` — the WP-F4 `verdict_for(session_id, ensemble_name, dispatch_context) -> CalibrationVerdict` surface that the router consumes deterministically, plus `CalibrationAbstainError` for the consumer-side typed-error pattern; (7) `src/llm_orc/agentic/orchestrator_tool_dispatch.py` for the interposition site (router fires between Autonomy Policy gate and `EnsembleExecutor.execute`).
>
> **Settled premises going into WP-G4-1:**
>
> 1. **The `LlmOrcStructuralError` base class** is the typed-error parent for both new error_kinds: `escalation_bypass` (`recovery_action_required="reformulate"` — router transforms calibration Abstain into this when consuming via the router edge per ADR-015 §Decision) and `missing_skill_metadata` (`recovery_action_required="reformulate"` — explanatory diagnostic listing valid Topaz skill values).
> 2. **FC-17 coverage advances 5 of 8 → 7 of 8.** Five concrete `LlmOrcStructuralError` subclasses now exist (`ToolCallingNotSupportedError`, `PhantomToolCallError`, `WriteGateRejectionError`, `CompactionLayer4FailureError`, `CalibrationAbstainError`); WP-G4-1 adds two more (`escalation_bypass`, `missing_skill_metadata`); WP-H4 / ADR-016 adds the final `malformed_signal` to reach 8 of 8.
> 3. **Verdict trichotomy is load-bearing (FC-19):** Proceed → cheap-tier dispatch; Reflect → escalated-tier dispatch; Abstain → `escalation_bypass` typed error. No LLM-mediated translation between Calibration Gate and Tier-Escalation Router per system-design.agents.md L484 — `CalibrationVerdict` flows as a typed value.
> 4. **Per-skill (not per-ensemble) tier defaults** per ADR-015 rejected alternative (b) — confirmed by Spike α empirically (research log `005g-` 2026-05-11) showing clean primary-skill partition on 21 of 21 production-style ensembles. Two ensembles declaring the same Topaz skill share the operator's per-skill cheap-tier/escalated-tier Model Profile pair.
> 5. **The router's existing `select_tier` is a stateless pure function** per ADR-018 §"Inherited bounding mechanism (a)" (Spike β analytical transfer audit 2026-05-11). FC-2 layering places `llm_orc.agentic.tier_router` at L2; `_LAYER_MAP` in `tests/unit/agentic/test_fc2_layering.py` is already pre-declared.
> 6. **WP-G4-2 (ADR-018 (d)-analog audit dispatch) is a separate work package** landing after WP-G4-1. WP-G4-1's scope is the core router (verdict consumption, per-skill mapping, typed errors); WP-G4-2 adds the periodic out-of-band audit dispatch responsibility.
>
> **Honor at WP-G4-1 entry:**
>
> 1. **FC-13 preservation (ADR-011):** The orchestrator's own Model Profile remains constant for the entire Session under any verdict-driven tier escalation; only the dispatched task's tier varies. The router does NOT touch the orchestrator's reasoning-loop Model Profile. Verified by `test_orchestrator_profile_unchanged_under_tier_escalation` (preservation).
> 2. **`invoke_ensemble` API signature is unchanged** under tier escalation — the orchestrator's reasoning surface receives the same tool-call result shape regardless of which tier was dispatched. Verified by `test_invoke_ensemble_api_unchanged_under_tier_escalation` (integration).
> 3. **Spike α distribution finding informs operator docs (WP-G4-1 scope):** Surface ADR-015 §Consequences §Negative's coverage hedge as load-bearing — operators may collapse unused skill slots (e.g., `mathematical_reasoning` exercises zero on existing library) to shared Model Profiles. Spike α's cheap-classifier specific labels are NOT recommended for adoption (the classifier exhibits bias toward structured-output-shape skills); use operator judgment for `topaz_skill` field values during the migration.
> 4. **Sub-Q6 routing-quality assumption remains empirically open.** ADR-015 §Consequences §Neutral records that operators interpreting escalation-rate calibration evidence may be reading routing-noise rather than tier-configuration mismatches until first-deployment evidence resolves Sub-Q6. WP-G4-2's (d)-analog audit dispatch is the structural follow-up — WP-G4-1 BUILD should keep the router's interface clean for that audit to attach.
>
> **Next steps after WP-G4-1 closes** (in conformance-scan order): WP-G4-2 (ADR-018 d-analog audit dispatch) → WP-H4 (ADR-016 — conditional on first-deployment evidence per ADR-016 §"Concrete monitoring specification").

## Suggested fresh-session handoff prompt for WP-F4 entry (2026-05-11; superseded — preserved for cycle continuity)

> Continue Cycle 4 BUILD at **WP-F4 — ADR-014 Calibration Gate trajectory-level extension**. WP-A4 + WP-B4 + WP-C4 + WP-D4 + WP-E4 closed 2026-05-11 (commits `cc0d94f`, `7c2f64e`, `1701a22`, `9116793`, `ded9e2d`, `e1ff875`, `31e261f`, `b574689`, `36f540c`). Full suite 2476 passing; all linters clean. **BUILD mode: gated** (per-scenario-group EPISTEMIC GATES with AID cycle).
>
> **Read in this order before opening WP-F4:** (1) `housekeeping/cycle-status.md` for the cycle's current state; (2) `decisions/adr-014-calibration-gate-trajectory-extension.md` in full — including the verdict trichotomy (Proceed / Reflect / Abstain), the time-decay windowing dual-bound (60 min / 100 dispatches), and AUQ confidence threshold (default 0.85); (3) `scenarios.md` §Calibration Verdict Trichotomy (ADR-014 scenarios) — particularly the "Proceed routes dispatch as-is", "Reflect routes to escalated tier", "Abstain produces typed error", "Time-decay windowing", and "Calibration verdict feeds router input" scenarios; (4) the existing `src/llm_orc/agentic/calibration_gate.py` to understand where the trajectory-level extension interposes alongside the current ADR-007 first-N post-hoc mechanism; (5) `src/llm_orc/agentic/conversation_compaction.py` for the most recent `LlmOrcStructuralError` subclass pattern (WP-E4's `CompactionLayer4FailureError` is the most recent precedent; WP-F4's `calibration_abstain` is the fifth subclass).
>
> **Settled premises going into WP-F4:**
>
> 1. **The `LlmOrcStructuralError` base class is the typed-error parent** for the new `calibration_abstain` error_kind. `recovery_action_required` should be `abstain` per the four-value Literal in `models/structural_errors.py` — the orchestrator must take a different action (reformulate, dispatch elsewhere, or abstain entirely) per ADR-014 §Decision.
> 2. **FC-17 coverage advances 4 of 8 → 5 of 8.** Four concrete `LlmOrcStructuralError` subclasses now exist (`ToolCallingNotSupportedError`, `PhantomToolCallError`, `WriteGateRejectionError`, `CompactionLayer4FailureError`); WP-F4 adds the fifth.
> 3. **The Calibration Gate already exists at L1** (`src/llm_orc/agentic/calibration_gate.py`) implementing ADR-007's first-N post-hoc result-check. ADR-014's trajectory-level extension composes additively — the existing first-N mechanism is preserved per the §Preservation scenario.
> 4. **The verdict produced by Calibration Gate flows downward** into Tier-Escalation Router (WP-G4-1) and Tool Dispatch per system-design.agents.md §Dependency Graph. WP-F4 produces the verdict; WP-G4-1 consumes it.
> 5. **Time-decay windowing is dual-bound:** 60 minutes OR 100 dispatches, whichever shorter; weighted linearly from 1.0 (most recent) to 0.0 (window edge). Per ADR-014 §Decision and the DECIDE-phase spike research log `005e-spike-adr014-b-windowing.md`.
>
> **Honor at WP-F4 entry:**
>
> 1. **Verdict trichotomy is load-bearing** (FC-19): Proceed / Reflect / Abstain are the three values the router consumes deterministically — no LLM-mediated translation step between Calibration Gate and Tier-Escalation Router.
> 2. **AUQ confidence threshold (default 0.85)** is operator-tunable per ADR-014; below threshold AND no trajectory anomaly → Reflect; below threshold AND trajectory entropy drop > 1.5σ → Abstain.
> 3. **ADR-007 first-N post-hoc mechanism continues unchanged** — the existing quality-signal accumulation and trusted-status transition logic compose additively with ADR-014's in-process trajectory-level layer.
> 4. **The bias-bound from mechanism (b) windowing** lives at the producer (Calibration Gate per ADR-018 inherited windowing transfer); the router does not duplicate windowing.
>
> **Next steps after WP-F4 closes** (in conformance-scan order): WP-G4-1 (ADR-015 per-role tier-escalation router) → WP-G4-2 (ADR-018 d-analog audit dispatch) → WP-H4 (ADR-016 — conditional on first-deployment evidence).

## Suggested fresh-session handoff prompt for WP-E4 entry (2026-05-11; superseded — preserved for cycle continuity)

> Continue Cycle 4 BUILD at **WP-E4 — ADR-012 Conversation Compaction five-layer pipeline**. WP-A4 + WP-B4 + WP-C4 + WP-D4 closed 2026-05-11 (commits `cc0d94f`, `7c2f64e`, `1701a22`, `9116793`, `ded9e2d`). Full suite 2448 passing; all linters clean. **BUILD mode: gated** (per-scenario-group EPISTEMIC GATES with AID cycle).
>
> **Read in this order before opening WP-E4:** (1) `housekeeping/cycle-status.md` for the cycle's current state; (2) `decisions/adr-012-conversation-compaction-five-layer-pipeline.md` in full — including the cheapest-first layer ordering (Layers 0–4) and the Layer 4 circuit-breaker disposition; (3) `scenarios.md` §Conversation Compaction (all Cycle 4 ADR-012 scenarios); (4) the existing `src/llm_orc/agentic/orchestrator_runtime.py` to understand where the compaction module is invoked at turn boundaries; (5) `src/llm_orc/agentic/session_artifacts.py` for the most recent `LlmOrcStructuralError` subclass pattern (WP-D4 added `compaction_layer_4_failure`'s sibling `write_gate_rejection`; the construction shape is the same).
>
> **Settled premises going into WP-E4:**
>
> 1. **The `LlmOrcStructuralError` base class is the typed-error parent** for the new `compaction_layer_4_failure` error_kind. Per the system-design.agents.md error pathway table line 546, `recovery_action_required="operator_intervention_required"` after 3 consecutive Layer 4 failures (circuit-breaker disposition).
> 2. **FC-2 and FC-3 are enforced as automated tests** — any new module under `src/llm_orc/agentic/` (likely `conversation_compaction.py` per the WP-E4 spec) must be added to `_LAYER_MAP` in `tests/unit/agentic/test_fc2_layering.py` (Conversation Compaction sits at **L2**, pre-declared in `_LAYER_MAP` at WP-B4 time).
> 3. **FC-4 amendment is in play.** Orchestrator Runtime's allowed import set extends from `{Budget Controller, Orchestrator Tool Dispatch}` to `{Budget Controller, Orchestrator Tool Dispatch, Conversation Compaction}` per ADR-012. Update `tests/unit/agentic/test_fc4_runtime_import_surface.py` accordingly.
> 4. **WP-C4 + WP-D4's typed-error pattern is the precedent.** Three concrete `LlmOrcStructuralError` subclasses now exist (`ToolCallingNotSupportedError` from WP-A4, `PhantomToolCallError` from WP-C4, `WriteGateRejectionError` from WP-D4). FC-17 coverage is 3 of 8; WP-E4's `compaction_layer_4_failure` brings it to 4.
> 5. **The Compaction owns Conversation Compaction (concept).** Ownership re-allocated from Orchestrator Runtime per ADR-012 (Cycle 4) — Runtime now invokes Compaction the same arms-length way it invokes Tool Dispatch.
>
> **Honor at WP-E4 entry:**
>
> 1. **Layer ordering is load-bearing** (FC-14): Layer 4 (LLM summary via summarizer ensemble) fires only after Layers 0–3 (filesystem persistence; conversation-history pruning; tool-result truncation; structured-message compaction) have been attempted.
> 2. **Layer 4 circuit-breaker.** After 3 consecutive Layer 4 failures, the typed error `compaction_layer_4_failure` is raised with `recovery_action_required="operator_intervention_required"` and the orchestrator yields.
> 3. **Compaction is invoked at every turn boundary** per FC-14; the resulting `CompactedContext` is what flows into the next LLM call.
> 4. **AS-7 unchanged** — Result Summarizer Harness continues to run on ensemble outputs independently of Conversation Compaction operating on conversation history.
>
> **Next steps after WP-E4 closes** (in conformance-scan order): WP-F4 (ADR-014 Calibration Gate trajectory-level extension) → WP-G4-1 + WP-G4-2 (ADR-015 + ADR-018) → WP-H4 (ADR-016 — conditional on first-deployment evidence).

## Suggested fresh-session handoff prompt for WP-D4 entry (2026-05-11; superseded — preserved for cycle continuity)

> Continue Cycle 4 BUILD at **WP-D4 — ADR-013 Session Registry structured-handoff artifacts + write-gate validation + cluster determination**. WP-A4 + WP-B4 + WP-C4 closed 2026-05-11. Full suite 2400 passing; all linters clean. **BUILD mode: gated** (per-scenario-group EPISTEMIC GATES with AID cycle).
>
> **Read in this order before opening WP-D4:** (1) `housekeeping/cycle-status.md` for the cycle's current state; (2) `decisions/adr-013-session-registry-initializer-then-resume-schema.md` in full — including the write-gate validation surface and the cluster determination logic; (3) `scenarios.md` §Session Registry Initializer-then-Resume — all 8 scenarios (Cluster 2 activates artifact set; Cluster 1 opts out; monotonic passes constraint; append-only rejection; init.sh hash mismatch; operator hash rotation; cross-cluster session defaults; preservation of existing identification); (4) the existing `src/llm_orc/agentic/session_registry.py` to understand where the structured-handoff extension interposes; (5) `src/llm_orc/models/structural_errors.py` + `src/llm_orc/agentic/tool_call_validation_guard.py` for the `LlmOrcStructuralError` subclass pattern WP-D4's `write_gate_rejection` subclass should follow (WP-C4's `PhantomToolCallError` is the most recent precedent).
>
> **Settled premises going into WP-D4:**
>
> 1. **Shared `LlmOrcStructuralError` base class is the typed-error parent** for the new `write_gate_rejection` error_kind. Construction shape: `error_kind="write_gate_rejection"`, `recovery_action_required="reformulate"` (orchestrator can rephrase the write attempt) or `"operator_intervention_required"` (init.sh hash mismatch — operator hash rotation needed).
> 2. **FC-2 and FC-3 are enforced as automated tests** — any new module under `src/llm_orc/agentic/` (likely `session_artifacts.py` per the WP-D4 spec) must be added to `_LAYER_MAP` in `tests/unit/agentic/test_fc2_layering.py` (Session Registry sits at L3; the new sub-module shares that layer or sits at L1 depending on its dependencies — judgment at BUILD time).
> 3. **WP-C4's `validate_response` interposition pattern is the precedent** for write-gate validation: Tool Dispatch (or in this case, Session Registry's write surface) interposes structural validation BEFORE the side-effect; produces typed `LlmOrcStructuralError` subclass on rejection; the orchestrator's reasoning surface receives the structural feedback.
> 4. **WP-C4 added 2 of 8 typed-error surfaces** (`tool_call_rejected_per_model` from WP-A4 precedent + `phantom_tool_call` from WP-C4). WP-D4 adds the third: `write_gate_rejection`.
>
> **Honor at WP-D4 entry:**
>
> 1. **The append-only progress log is structural enforcement, not advisory.** Write-gate rejection on non-append writes is a typed error producer per ADR-013's design — no silent fallback, no retry-with-warning.
> 2. **init.sh hash mismatch is `operator_intervention_required`**, not `reformulate`. The orchestrator cannot rotate the hash itself; the operator's hash-rotation workflow is the recovery path.
> 3. **Cluster determination at session-start defaults to required-artifact-set** for ambiguous declarations (disposition (i) per ADR-013 §"Cross-cluster session defaults"). North-Star-benchmark sessions that straddle RESEARCH and BUILD get the artifact set active.
> 4. **Per-session state in stateless mode** is unchanged — ADR-013 extends Session Registry additively; existing `SessionIdentity` derivation, `SessionState` tracking, `turn_count`, and `token_spend` bookkeeping continue to operate exactly as before.
>
> **Next steps after WP-D4 closes** (in conformance-scan order): WP-E4 (ADR-012 Conversation Compaction five-layer pipeline) → WP-F4 (ADR-014 Calibration Gate trajectory-level extension) → WP-G4-1 + WP-G4-2 (ADR-015 + ADR-018) → WP-H4 (ADR-016 — conditional on first-deployment evidence).

## Suggested fresh-session handoff prompt for BUILD entry

> Continue Cycle 4 of the agentic-serving scoped corpus. ARCHITECT phase is **complete** (gate closed 2026-05-11). The cycle is now in BUILD. Cycle 4 was re-scoped 2026-05-08 from Mode B+ → DECIDE close to **Mode A — extended through ARCHITECT and BUILD** (PLAY open; decision deferred to BUILD close).
>
> **Cycle 4 BUILD work** comprises eight WPs (WP-A4 through WP-H4), per the conformance-scan-recommended sequence with ADR-018 inline at WP-G4. WP-G4 is **restructured into WP-G4-1 + WP-G4-2** at architect-gate close per ADR-018:
>
> 1. **WP-A4: Shared `LlmOrcStructuralError` base class** — T1 prerequisite. Codifies the typed-error pattern from commit `9f86d0b`. Unlocks all five typed-error producers downstream.
> 2. **WP-B4: FC-2 + FC-3 automated tests** — T1 prerequisite. AST-based per-module import layering check + dependency-graph cycle detection.
> 3. **WP-C4: ADR-017** — Tool-call structural validation guard (class (a) deterministic-override against phantom tool-call confabulation).
> 4. **WP-D4: ADR-013** — Session Registry structured-handoff artifacts + write-gate validation + cluster determination.
> 5. **WP-E4: ADR-012** — Conversation Compaction five-layer pipeline.
> 6. **WP-F4: ADR-014** — Calibration Gate trajectory-level extension.
> 7. **WP-G4-1: ADR-015** — Per-role tier-escalation router (core); Topaz skill metadata migration; per-skill tier-defaults configuration.
> 8. **WP-G4-2: ADR-018** — (d)-analog audit dispatch (periodic out-of-band audit on the L1→L2 verdict→router edge; three drift criteria including the Sub-Q6-coupling escalation-vs-outcome correlation criterion).
> 9. **WP-H4: ADR-016** — Cross-layer calibration channel; **CONDITIONAL ACCEPTANCE** — depends on first-deployment evidence per ADR-016 §"Concrete monitoring specification". Land last.
>
> **Honor at BUILD entry:**
>
> 1. **ADR-016 falsification trigger** remains live for WP-H4. If responsibility allocation surfaces a need for a module outside L0–L3 for mechanism (b) or (d), pause BUILD, re-open ADR-016 deliberation, escalate to practitioner.
>
> 2. **ADR-018 falsification trigger** is live for WP-G4-2. If BUILD finds the (d)-analog audit dispatch cannot be operationalized within the Tier-Escalation Router module's responsibility (e.g., requires its own top-level module orthogonal to L0–L3, or bidirectional coupling with Calibration Gate that violates read-only verdict-consumption), pause BUILD, ADR-018 re-deliberates, OQ #14 partial closure reverts, Sub-Q6 re-opens.
>
> 3. **Spike α distribution finding informs WP-G4-1 operator docs.** Surface ADR-015 §Consequences §Negative's coverage hedge as load-bearing — operators may collapse unused skill slots (e.g., `mathematical_reasoning` exercises zero on existing library) to shared Model Profiles. Spike α's cheap-classifier specific labels are NOT recommended for adoption (the classifier exhibits bias toward structured-output-shape skills); use operator judgment for `topaz_skill` field values during the migration.
>
> 4. **Spike β inheritance discipline informs WP-G4-2 BUILD.** Three of ADR-016's bounding mechanisms hold for the verdict→router edge by inheritance: (a) Router stateless `select_tier`; (b) upstream windowing at Calibration Gate per ADR-014; (e) FC-17 typed-error infrastructure. WP-G4-2 BUILD should NOT add Router-side windowing (b is upstream) or Router-side anchors (c is structurally inapplicable); these would be defensive duplication.
>
> 5. **PLAY decision deferred to BUILD close.** Assess at BUILD close whether to run `/rdd-play` in this cycle or schedule as a follow-up. The decision will be informed by first-deployment evidence accumulating during BUILD.
>
> 6. **Free-tier preference for the agentic-serving corpus** still applies. BUILD work may necessarily incur some token cost (LLM-bearing tests, integration verification), but spike-style exploratory dispatches should continue to prefer local Ollama profiles.
>
> **Read in this order before opening BUILD:** (1) `housekeeping/cycle-status.md` for the cycle's current state (this document); (2) `roadmap.md` for the WP-by-WP plan with WP-G4 split into WP-G4-1 + WP-G4-2; (3) `system-design.agents.md` (Cycle 4 + architect-gate-close extension) — responsibility matrix, dependency graph, fitness criteria FC-14 through FC-20; (4) the seven new ADRs (012-018) in `decisions/`; (5) `housekeeping/gates/cycle-4-architect-gate.md` for the gate's specific commitments going into BUILD; (6) `housekeeping/audits/conformance-scan-cycle-4-decide.md` for the BUILD sequencing recommendation that informed WP ordering.

## Suggested fresh-session handoff prompt for ARCHITECT entry (superseded — preserved for cycle continuity)

> Continue Cycle 4 of the agentic-serving scoped corpus. The cycle was re-scoped on 2026-05-08 from Mode B+ → DECIDE close to Mode A — extended through ARCHITECT and BUILD (PLAY is open, deferred to BUILD close). RESEARCH, DISCOVER, MODEL, DECIDE all closed; the cycle's six new ADRs (012-017) plus deferred candidate #5 are the DECIDE-phase deliverable. Next phase: `/rdd-architect` against the current `system-design.md` + `system-design.agents.md` + the six new ADRs.
>
> **ARCHITECT-phase work:** integrate the six new ADRs into the system design's responsibility matrix and dependency graph. Per the conformance-scan recommendation, BUILD sequencing is ADR-017 → shared `LlmOrcStructuralError` base class → FC-2/FC-3 automated checks → ADR-013 → ADR-012 → ADR-014 → ADR-015 → ADR-016 (conditional on first-deployment evidence). ARCHITECT's responsibility matrix should surface this sequencing.
>
> **Honor at ARCHITECT entry:**
>
> 1. **ADR-016 conditional-acceptance discipline** — keep mechanism (b) time-decay windowing and mechanism (d) periodic out-of-band audit dispatch within ADR-002's L1 layer. If responsibility allocation surfaces a need for a module outside L0–L3, the falsification trigger fires — pause ARCHITECT, re-open ADR-016 deliberation, escalate to practitioner.
>
> 2. **OQ #14 (asymmetric grounding-mechanism rigor across cross-layer stages)** — ARCHITECT may surface grounding-mechanism gaps as it allocates responsibilities for L1→L2 verdict-router stage, L3 cross-session artifact set stage, intra-L2 conversation-history boundary, orchestrator-response→tool-dispatch boundary, and L1→L4 Plexus integration. ARCHITECT decides whether to (a) surface gaps for Cycle 5+ research, (b) propose grounding mechanisms inline as drivers, or (c) note that BUILD evidence will inform.
>
> 3. **Three structural prerequisites must appear in the responsibility matrix:** FC-2 automated test, FC-3 automated test, shared `LlmOrcStructuralError` base class.
>
> 4. **Grounding Reframe carry-forward (Sub-Q6 routing reliability)** — when allocating the tier-escalation router's responsibilities, attend to Sub-Q6's evidence gap. The router's value assumes routing accuracy that is empirically open at multi-iteration scale.
>
> Read in this order before opening ARCHITECT: (1) `cycle-status.md` for the cycle's current state and ARCHITECT-entry context (this document); (2) `housekeeping/gates/cycle-4-decide-gate.md` for the decide-gate close findings; (3) `essays/005-layer-conditional-composition.md` for the cycle's design-method posture; (4) the six new ADRs in `decisions/`; (5) `housekeeping/audits/conformance-scan-cycle-4-decide.md` for the BUILD sequencing recommendation; (6) the current `system-design.md` and `system-design.agents.md` as the substrate ARCHITECT extends.

## Pause Log

(No pauses on the corpus. Re-scoping 2026-05-08 is not a pause — it's a scope expansion.)

## Cycle History

| Cycle | Started | Closed | Shape | Archive |
|-------|---------|--------|-------|---------|
| 1 | 2026-03-20 | 2026-04-29 | Standard pipeline through PLAY + backward research loop | `../cycle-archive/cycle-1-agentic-serving.md` |
| 2 | 2026-04-29 | 2026-05-01 | Mode B (Research Only) — closed at research-phase end | `../cycle-archive/cycle-2-multi-turn-and-composition.md` |
| 3 | 2026-05-01 | 2026-05-01 | Mode B (Research Only) — closed at research-phase end with five research logs + audit trail as the deliverable. **Retroactive essay 004** added 2026-05-04 (essay numbering now continuous: 001-005 without skip) | `../cycle-archive/cycle-3-agent-design-script-models-orchestrator.md` |

(Cycle 4 row pending — will be added when Cycle 4 closes through ARCHITECT and BUILD per re-scoped close shape.)

## Conformance Notes

**Corpus is on RDD v0.8.5.** Cycle 4's audit corpus and gates follow the existing ADR-070 housekeeping placement convention (`housekeeping/audits/`, `housekeeping/gates/`); ADR-085 `.rdd/` migration target applies but is deferred per the migration-window allowance.

**Cycle 4 supersession events at DECIDE close (2026-05-08, per ADR-016 partial-update of ADR-002):**
- ADR-002 has dated update header (`> Updated by ADR-016 on 2026-05-06.`) and Status field updated to `Updated by ADR-016`
- Four-artifact downstream sweep complete: ADR-002 partial-update header; `domain-model.md` (Methodology Vocabulary entries promoted from "proposed pending DECIDE" to "conditionally accepted via ADR-016"; new vocabulary entries added; Amendment Log entries 5, 6, 7); `system-design.md` (layering rule amended for read-only signal-channel exception; Design Amendment Log entry 6); `ORIENTATION.md` (Cycle 4 outcomes section added)
- `field-guide.md` unchanged (no direct ADR-002 layering-rule references that needed update)

**Cycle 4 supersession events at ARCHITECT close (2026-05-11, per ADR-018 partial-update of ADR-015):**
- ADR-015 has dated update header (`> Updated by ADR-018 on 2026-05-11.`) and Status field updated to `Updated by ADR-018`
- ADR-018 added as new ADR (Tier-Escalation Router periodic audit dispatch — ADR-016 mechanism (d) analog); spike-empirically anchored by research log `005h-`
- Six-artifact downstream sweep complete: ADR-015 partial-update header + §Consequences §Neutral Sub-Q6 coupling note; ADR-018 new file; `system-design.agents.md` (Tier-Escalation Router module extended with ADR-018 responsibilities + Falsification trigger + Direction-not-constraint note; FC-19 and FC-20 added; L1→L2 dependency-graph annotation updated with four-property composition callout); `roadmap.md` (WP-G4 restructured into WP-G4-1 + WP-G4-2; falsification trigger added); `domain-model.md` (OQ #14 partial closure recorded inline + Amendment Log entry #8); `ORIENTATION.md` (Cycle 4 outcomes section updated through architect close — seven ADRs, FC-14 through FC-20, WP-G4 restructure)
- `system-design.md` v3.0 unchanged at architect-gate close (the human-facing surface; ADR-018 details surface in system-design.agents.md per the agent-facing/human-facing surface split)
- `field-guide.md` unchanged (no module-to-code mapping changes; the field guide is reflexively maintained as implementation lands during BUILD)
- Methodological observation recorded in gate reflection note: the Spike β / Spike α coupling was emergent — parallel-dispatch of conceptually-related spikes produced coupling findings (OQ #14 + Sub-Q6 closed by the same mechanism) the individual spike specifications did not anticipate

**Deferred conformance items carried forward** (low priority; pick up opportunistically):

- ADR Rejected Alternatives + Provenance Check sections — 11 prior ADRs (001–011) lack discrete headers (alternatives discussed inline in Context); the 6 new Cycle 4 ADRs use the v0.8.5 template with discrete headers. Format alignment for prior ADRs matters only when those ADRs are re-audited.
- Value tensions phrasing — `product-discovery.md` §Value Tensions stated declaratively rather than as open questions per v0.8.5 discover template.
- Essay 001 framing-audit dispatch — `housekeeping/audits/argument-audit-001.md` is argument-only; v0.8.5 dispatches combine argument + framing audits.
- Field-guide path — currently at `docs/agentic-serving/field-guide.md`; canonical is `references/field-guide.md`. ORIENTATION links current location; navigability preserved.
- ~~Scenarios cycle-acceptance-criteria table~~ — *resolved 2026-05-08:* `scenarios.md` now includes the Cycle Acceptance Criteria Table at the top per v0.8.5 decide Step 4 (5 emergent/aggregate criteria from Cycle 4).
- Housekeeping placement (`docs/agentic-serving/housekeeping/` per ADR-070) — ADR-085 supersedes with `.rdd/` placement during the transition window.

**Spike artifacts retention (Cycle 3 directive, applies to corpus until close):**

Cycle 3 spikes (retained):
- `scratch/spike-a-cycle3-*`, `scratch/spike-b-cycle3-*`, `scratch/spike-c-cycle3-architecture-comparison/`, `scratch/spike-d-cycle3-multi-ensemble-pilot/`
- `.llm-orc/ensembles/spike-c-code-review.yaml`, `.llm-orc/ensembles/spike-d-fix-verifier.yaml`
- `.llm-orc/scripts/spike_c_diff_analyzer.py`, `.llm-orc/scripts/spike_d_fix_verifier.py`

Cycle 4 spikes (retained, added 2026-05-06):
- `scratch/spike-cycle4-research-loop-dogfood/` (research-phase Wave 3.A behavioral spike)
- `scratch/spike-cycle4-adr016-b-windowing/` (DECIDE-phase synthetic-data spike on mechanism (b))

Cycle 4 architect-gate-continuation spikes (retained, added 2026-05-11):
- `essays/research-logs/005g-spike-topaz-skill-classification.md` (Spike α — cheap-orchestrator dispatch via `qwen3:8b`)
- `.llm-orc/ensembles/spike-005g-skill-classifier.yaml` (companion ensemble for Spike α; retained per spike-artifact retention directive)
- `essays/research-logs/005h-spike-bounding-mechanism-transfer-l1-l2.md` (Spike β — analytical-only structural transfer audit; no scratch directory; the research log is the deliverable)
- `.llm-orc/scripts/spike_adr016_b_time_decay_windowing.py`
- (Spike (d) on mechanism (d) was an analytical structural-transfer audit; outputs are in `essays/research-logs/005f-` rather than scratch/)

These are retained per practitioner directive overriding standard rdd-research delete-after-recording discipline; the directive applies until the agentic-serving corpus closes.
