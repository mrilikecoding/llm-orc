# ADR-045: Clean-slate collapse of the imperative serving layer (executes AS-11)

**Status:** Accepted (2026-07-01)

## Context

ADR-044 codified AS-11: agentic serving is declarative-ensemble-native; where the engine is inadequate, extend it, never add a parallel orchestration layer. The Cycle-7 serving behavior lives in a bespoke imperative layer, `src/llm_orc/agentic/` (~12.7K LOC across 32 files: Loop Driver, single-step enforcer, tier router, orchestrator tool dispatch, calibration gates, session records), whose architecture the ADR-033 loop-driver chain (ADR-033 through ADR-043) established. AS-11 contradicts that architecture directly: those modules are the parallel orchestration layer AS-11 forbids.

The practitioner chose **clean-slate over strangler** (2026-07-01): the Ω-spikes and the bespoke code are learning, not a substrate to harden. Grafting a production system onto spike scaffolding is the worst of both.

## Decision

Collapse the imperative serving layer clean-slate:

1. **Supersede by mechanism, not by ADR-number.** The clean-slate removes the entire imperative serving layer `src/llm_orc/agentic/`; this supersedes the decided *imperative-orchestration mechanism* of every ADR realized there. **The confident cases, superseded here (headers stamped):** the loop-driver chain **ADR-033 through ADR-043**, plus **ADR-015** (tier router as a dedicated L2 Python layer) and **ADR-021** (`OrchestratorRuntime` per-capability ReAct dispatch). **ADR-027**'s dispatch mechanism was already superseded via ADR-043 (chain 027→043→045); its framework-driven *direction* survives. **Every other ADR with an implementation in the removed layer is classified by the Cycle-8 target-architecture ADR**, which performs the full per-ADR mechanism audit against AS-11. That deferred set is **definitionally complete** — every `agentic/`-implemented ADR minus the confident set — and **enumerated from the source tree** (each module's ADR attribution; see the Appendix inventory) rather than hand-picked, because hand-picking kept omitting members (argument-audit R2 missed ADR-008/009/010; R5 missed ADR-003/011) and mislabeling others (R5: `orchestrator_tool_dispatch.py` self-attributes to the ADR-003 closed-tool-surface *contract*, not an orchestration mechanism; ADR-012 is deterministic compaction, a `script:`-node survivor; ADR-016 has its own cycle-status-row governance). The deferred set: **ADR-001, 003, 004, 005, 006, 007, 008, 009/010, 011, 012, 013, 014, 016, 017, 018, 023, 024, 025** (ADR-018 deferred despite extending the superseded ADR-015, because its audit-dispatch derives from the deferred ADR-016). **Guidance carried to that classification:** (a) the test is *imperative-orchestration mechanism* → superseded vs *contract / invariant / deterministic-work / direction* → survives (implementation re-homed, decision intact); (b) an extension is not classified more definitively than its base; (c) ADR-016's absence-of-cycle-status-row governance is honored; (d) ADR-001 is classified on the loop-*ownership* axis (OpenCode owns the loop) not only the control-flow-shape axis. **Decisions with no implementation in the removed layer are untouched and survive cleanly:** ADR-019 (skill-framework-agnostic), ADR-026/AS-10 (capability-matching-from-content). **In the deferred set but expected to survive as contracts/invariants** (the target-arch ADR confirms, does not pre-judge): ADR-004/AS-7, ADR-005/AS-3, ADR-006/AS-2, ADR-024 (envelope), ADR-025 (artifact-substrate), ADR-003 (closed tool surface), ADR-011 (orchestrator-is-a-model-profile), ADR-008 (autonomy), ADR-009/010 (Plexus/AS-8). **This ADR is the manifest of record; the authoritative full classification is the target-architecture ADR's first task.**
2. **Remove `src/llm_orc/agentic/`** and its 37 test files. Removal *follows* the declarative serving form reaching behavioral parity (the declarative product is built fresh, per AS-11); it does not precede it. But the code is destined for deletion, not incremental adaptation — no strangler on spike code.
3. **Carry forward the validated behavioral requirements, not the implementations,** as design inputs to the Cycle-8 declarative target architecture. The ADR-033 chain discovered real requirements the declarative form must still satisfy (table below). The clean-slate deletes the HOW; the WHAT is re-decided declaratively.
4. **Retain the Ω-spike artifacts** (`scratch/spike-omega-*`, 48 dirs) until corpus close per the retention discipline; they are the research the Cycle-8 design draws on. Retire at corpus graduation, not now.

The removal **execution** (code deletion + the downstream current-state doc sweep) happens in Cycle-8 BUILD, when the declarative form ships. This ADR records the decision and the supersession; BUILD sequences the mechanics.

**Carry-forward requirements** (what survives; the mechanism is re-homed declaratively):

| Superseded ADR | Imperative mechanism (removed) | Behavioral requirement (carried forward) | Declarative home (Cycle-8 target) |
|---|---|---|---|
| 033 | Loop Driver + single-action-per-turn enforcer | multi-turn loop participation; one grounded action per turn — **axis-2 long-horizon sequential-composition drift remains OPEN** (AS-9 scope boundary; a BUILD/PLAY validation target, not a settled property) | bounded `loop:` + per-turn node structure |
| 034 | Client-Tool-Action Terminal + Artifact Bridge | deliverable arrives as a client tool_call the client executes; artifact→tool_call bridge (parity) | terminal/marshal script node |
| 035 | deliverable form contract (imperative FormGate) | bare, fence-free, form-valid tool_call content | form gate as a script/guard node |
| 036 | delegation-decision mechanism | reliable delegate-vs-carry per turn — **validated for one composition × qwen3:14b only; the V3 guidance lever is profile-bound and does NOT transfer across seat-filler models** (Spike ψ′ Arm D) | guard/branch |
| 037 | two-call termination composition | terminate when work is complete, continue when not | loop `until:` |
| 038 | remaining-work anchor | advance across multiple deliverables (multi-file progress) | loop `carry:` |
| 039 (042 reverted) | content anchor — **unbounded, all-prior-siblings selection** (ADR-042's K=8 bound was REVERTED per Spike τ′; the validated end-state is ADR-039's unbounded form) | cross-file content coherence | script node injecting sibling content |
| 040 | deterministic completeness gate | completeness check independent of the builder's claim | guard/script node (feeds grounded acceptance) |
| 041 | destination-validity gate | destination validity before marshalling — **protects (0 invalid files reached the client); convergence/recovery is NOT guaranteed** (Conditional Acceptance; "protects-but-does-not-recover") | guard/script node |
| 043 | collapse to one loop-driven surface | one serving surface, not two | one declarative serving ensemble |

## Rejected alternatives

- **Strangler / incremental peel** — route production traffic through the declarative form while peeling `agentic/` module by module. Rejected by the practitioner: no strangler on spike code; incremental peel keeps the parallel layer alive during the transition, which is the AS-11 violation prolonged.
- **Parity-then-delete-in-place** — adapt `agentic/` toward the declarative form. Rejected: it hardens scaffolding and preserves the parallel layer as the substrate, contradicting the clean-slate intent.
- **Three scoping errors the manifest-of-record + deferral avoids.** (i) *Proximity-based supersession* — the initial draft's "supersede ADR-033–043" line — under-scopes: it silently left ADR-015/021/027, whose decided mechanisms are the exact imperative shape AS-11 forbids and which live in the removed layer, un-superseded (argument-audit P1-1/P1-2/P1-5, R1). (ii) *Blanket-supersede everything implemented in `agentic/`* over-scopes: it would retire surviving contracts and invariants (envelopes, budget/AS-3, summarization/AS-7, capability-matching/AS-10) whose decisions AS-11 does not touch. (iii) *Hasty full-classification via the mechanism test alone* also misfires — R2/R3 caught it over-classifying deterministic-work ADR-012 (a `script:`-node survivor) and governance-protected ADR-016. The mechanism test states the principle, but per-ADR application requires deciding the declarative form, so the peer ADRs are deferred to the target-architecture ADR rather than classified from this manifest.

## Consequences

**Positive:**
- One architecture. The validated requirements are preserved as explicit design inputs, so the Cycle-7 learning is not lost with the code.
- The carry-forward table is the bridge from Cycle-7's hard-won requirements to the Cycle-8 declarative design — it prevents clean-slate from becoming amnesia.

**Negative:**
- Large deletion (12.7K LOC + 37 tests) in BUILD.
- The current-state docs (system-design.md, ORIENTATION.md, roadmap.md, field-guide.md) describe the Cycle-7 architecture and need a downstream sweep. **Deferred to Cycle-8 BUILD when the collapse ships** (recorded in cycle-status), because rewriting them before the declarative form exists would document an unbuilt architecture. This is the Step-2.5 deferral-with-rationale.

**Neutral:**
- The superseded ADRs (the 033–043 chain plus ADR-015/021, with ADR-027 already chained via 043) remain as historical records (body-immutable). Headers are stamped on the clearest cases per the manifest-of-record approach; the remaining `agentic/`-implemented ADRs are classified by the target-architecture ADR. The corpus is an isolated branch that will graduate or archive, so exhaustive per-ADR header bookkeeping is not warranted.
- ADR-044/AS-11 is the driver; this ADR is its execution record.

## Provenance check

- **Driver-derived:** the clean-slate choice and the rejected strangler are practitioner direction 2026-07-01; the carry-forward requirements are the ADR-033-chain's own validated findings (Spikes ρ/θ/ξ/ι + the real-OpenCode acceptance runs recorded across the Cycle-7 cycle-status). AS-11 (ADR-044) is the governing driver.
- **Drafting-time synthesis (flagged for the auditor):** the mechanism→requirement→declarative-home mapping in the carry-forward table is drafting-time synthesis; each row's "declarative home" is a candidate for the Cycle-8 target-architecture ADR, not a settled allocation. The **"behavioral requirement" column's settledness also varies by source ADR** — rows 033 (axis-2 open), 036 (profile-bound), and 041 (protects-not-recovers) carry the inline caveats above; all requirements are carried forward as inputs to **re-validate** on the declarative form, not as settled guarantees.
- **Empirical-Grounding Filter (ADR-097):** the carried-forward requirements are PLAY/BUILD-validated (paths 1/3); the removal is structural, not a research-surfaced feature.

## Appendix: `src/llm_orc/agentic/` ADR inventory (source-derived, 2026-07-01)

The complete module → primary-ADR attribution (from each module's docstring), the authoritative input for the target-architecture ADR's per-ADR classification. This inventory makes the deferred set complete by construction — argument-audit R2/R5 caught the hand-picked lists omitting ADR-003/008/009/010/011 and mislabeling `orchestrator_tool_dispatch.py`; a source-derived inventory cannot omit a module that exists.

| module(s) | primary ADR | disposition |
|---|---|---|
| `loop_driver.py`, `single_step_enforcer.py` | 033 | confident-superseded (header) |
| `client_tool_action_terminal.py`, `artifact_bridge.py` | 034 | confident-superseded (header) |
| `delegation_rate_meter.py` | 036 | confident-superseded (header) |
| `session_action_record.py` | 037 | confident-superseded (header) |
| `sibling_interface_extractor.py` | 039 | confident-superseded (header) |
| `tier_router.py` | 015 | confident-superseded (header) |
| `orchestrator_runtime.py` / `orchestrator_tool_dispatch.py` (021 mechanism, spread) | 021 (via 017/001/003 refs) | confident-superseded (021 header) |
| `tier_router_audit.py` | 018 | deferred |
| `orchestrator_tool_dispatch.py` | 003 (closed tool surface — contract) | deferred |
| `orchestrator_chunk.py` | 003 | deferred |
| `orchestrator_config.py` | 011 | deferred |
| `orchestrator_runtime.py` | 017 (+001 ReAct loop) | deferred |
| `calibration_gate.py` | 007 (+014) | deferred |
| `calibration_signal_channel.py` | 016 | deferred |
| `conversation_compaction.py` | 012 | deferred |
| `dispatch_envelope.py` | 024 | deferred |
| `dispatch_event_substrate.py`, `operator_terminal_event_sink.py`, `orchestrator_context_event_sink.py` | 023 | deferred |
| `session_artifact_store.py` | 025 | deferred |
| `session_artifacts.py`, `session_registry.py` | 013 | deferred |
| `result_summarizer_harness.py` | 004 | deferred |
| `autonomy_policy.py` | 008 | deferred |
| `plexus_adapter.py`, `session_start.py` | 009/010 | deferred |
| `tool_call_validation_guard.py` | 017 | deferred |
| `budget_controller.py` | 005 (no docstring ADR; inferred) | deferred |
| `composition_validator.py` | 006 (no docstring ADR; inferred) | deferred |
| `inference_wait_heartbeat.py` | liveness (Cycle-6; no ADR) | deferred |

Some modules touch several ADRs (e.g. `loop_driver.py` references 011/023/024/025/027/033); the primary docstring ADR is used for disposition, and the target-architecture ADR resolves any multi-ADR module. `orchestrator_tool_dispatch.py` appears twice deliberately: it houses both the ADR-003 closed-tool-surface contract and part of the ADR-021 dispatch mechanism — the confident supersession applies to the 021 portion, the ADR-003 contract is deferred (expected to survive).
