# Gate Reflection: Cycle 7 (Framework-Driven Orchestration: Routing as Code) DECIDE → ARCHITECT

**Date:** 2026-05-22
**Phase boundary:** DECIDE → ARCHITECT
**Cycle:** Cycle 7 — Framework-driven orchestration: routing as code

## Belief-mapping question composed for this gate

The question was composed as a pre-mortem against the seven new ADRs' rejected alternatives, with the susceptibility-snapshot's advisory #5 as the framing anchor ("the audit loop substituted for per-ADR practitioner challenge; the 7-ADR set is well-audited but not fully deliberated"). The composed question:

> "Pre-mortem: imagine ADR-027 BUILD ships and PLAY surfaces that the architectural commitment was wrong on some dimension. Which dimension is most likely to break, and which rejected alternative would you reach for first? The seven new ADRs name these rejected alternatives at the load-bearing-decision layer: [enumeration of rejected alternatives across ADR-027/028/029/030/032 — Tier 1 hybrid as primary; preserving OrchestratorRuntime with stronger amendments; frontier-tier orchestrator-LLM; rule-based classifier; embedding-similarity router; LLM-with-tools planner; no-planner; orchestrator-LLM-as-synthesizer; deterministic template synthesis; Rule 5 in headers/metadata only; not codifying Rule 6; disposition (ii) 4xx error; disposition (iii) silent strip; Cycle 7 immediate implementation; routing-planner-internal `tool_choice` honoring; single bundled transparent-endpoint promise; configuration-honesty as only commitment; 4xx on `action: 'direct'`]. If you were to name a single most-likely failure mode that would cause Cycle 7 BUILD to need rework, which would it be? And if that failure surfaces in PLAY, is the corresponding rejected alternative the natural reach — or does the cycle hold a different escape hatch?"

## User's response

> "If we are still not getting the work delegated to ensembles by default that would be a real failure."

The follow-up exchange engaged the response substantively:

- Agent: mapped the failure mode to the C1 failure surface ADR-027 is engineered to dissolve; named the cycle's operational signal (`direct_completion_rate` rolling metric per ADR-032); enumerated preserved escape hatches (routing-planner model profile override per ADR-028; classifier pre-filter; caching; embedding-similarity router as the natural reach if tuning is insufficient); enumerated deliberately-not-preserved escape hatches (Tier 1 hybrid as primary mitigates only explicit-`tool_choice` subset; frontier-tier orchestrator-LLM closed by AS-9 structural finding); surfaced that the cycle never named a refutation threshold distinguishing "expected operational baseline" from "real failure" and asked whether the cycle should commit to one now even rough or leave it as operator-deployment-specific judgment.

- Practitioner: "Sure that's likely wise"

- Agent: applied a refutation-threshold amendment to ADR-032 §"Refutation threshold for the cost-distribution accountability sub-promise" — deployment-relative structural relationship (expected baseline ≈ 100% - operator-estimated-capability-coverage); rough threshold of ~15 percentage points above baseline over 24-hour rolling window as evidence the C1 failure signal is firing; investigation pathway (model profile substitution → classifier pre-filter → caching → capability library expansion); falsification escape hatch (embedding-similarity router as the natural reach if tuning insufficient; AS-9 closes the frontier-tier-LLM path); PLAY-phase refinement acknowledgment. The amendment triggered round 6 + round 7 of the argument-audit loop; both round 6's P2 fix and round 7's clean verdict landed before the gate closed.

## Pedagogical move selected

**Probe** (then iterative). Referenced specific ADRs and their rejected alternatives via pre-mortem; engaged the practitioner's response by mapping the named failure mode to architectural surfaces; surfaced a follow-up question (refutation threshold) that produced a substantive ADR amendment rather than letting the gate close with the question deferred. The iteration moved from probe → probe → small targeted contingent question, ending with a gate-time ADR amendment that strengthened the cycle's commitment-vs-operator-signal coupling.

## Commitment gating outputs

### Settled premises (the practitioner is building on these going into ARCHITECT)

1. **ADR-027 framework-driven dispatch pipeline (plan → dispatch → synthesize) as PRIMARY direction for the chat-completions surface.** The 7-ADR set operationalizes this commitment. `OrchestratorRuntime` becomes unused production code on the chat-completions surface after ADR-027 BUILD ships; the ARCHITECT phase resolves the disposition (a/b/c).

2. **AS-9 + AS-10 as constitutional invariants.** AS-9 (structurally-bounded LLM roles produce reliable output on single-decision-shaped tasks; codified at MODEL boundary). AS-10 (capability matching on the agentic-serving chat-completions surface works from request content alone; codified in ADR-026). Both bind downstream artifacts.

3. **C1 failure mode is the cycle's primary risk surface.** The practitioner-named failure mode ("we're still not getting work delegated to ensembles by default") is the C1 surface ADR-027 is engineered to dissolve. The cycle's operational signal is `direct_completion_rate` per ADR-032's degradation signaling mechanism.

4. **Refutation threshold for the cost-distribution accountability sub-promise** (gate-time amendment to ADR-032). Deployment-relative structural relationship — expected baseline ≈ 100% minus operator-estimated-capability-coverage; sustained `direct_completion_rate` >~15 percentage points above baseline over 24-hour rolling window is evidence the C1 failure signal is firing. The ~15pp threshold and 24-hour window are starting heuristics; PLAY refines.

5. **Investigation pathway hierarchy when the threshold trips** (per ADR-032 round-7 amendment): (1) routing-planner model profile substitution; (2) classifier pre-filter; (3) caching planner decisions; (4) capability library expansion. Reordered from ADR-028 + ADR-031's latency-optimization sequence because the refutation threshold is a reliability/coverage signal, not a latency signal. Falsification escape hatch: embedding-similarity router (rejected in ADR-028 as primary; revisitable if LLM-reasoning planner is the production bottleneck). Frontier-tier orchestrator-LLM is explicitly NOT the natural reach per AS-9.

6. **Population A vs. Population B partition** (per ADR-026 + ADR-032). Cycle 7's principal stakeholder is Population A (tool-call-aware OpenAI-family clients without alternative llm-orc surface access); Population B is important-but-not-Cycle-7-focus, accommodated via structured advisory toward `llm-orc invoke` and direct ensemble HTTP API.

7. **Configuration honesty + cost-distribution accountability as two distinct sub-promises** (per ADR-032). Configuration honesty has Population A direct corroboration (Cline #10551 + OpenCode #20859); delivered by honest response labeling at three layers. Cost-distribution accountability is Population A silent; project-developer-lens grounded; delivered by strict-dispatch-when-capability-matched.

### Open questions (the practitioner is holding these open going into ARCHITECT)

1. **`OrchestratorRuntime` codebase-disposition** (ADR-027 §Decision §"OrchestratorRuntime status under ADR-027" defers to ARCHITECT). Three candidates: (a) preserve as architecture-for-future-surfaces; (b) wire `llm-orc invoke` to use `OrchestratorRuntime`; (c) mark for removal as unused code. Per susceptibility-snapshot Advisory #1 (NF2), ARCHITECT should name this as a required deliverable rather than a deferred deliberation. Per the round-4 conformance-scan Finding 2 amendment, the current codebase state is: `OrchestratorRuntime` is instantiated only by the chat-completions handler being replaced by ADR-027.

2. **Multi-step composition mechanism** (OQ #21). Initial BUILD defaults to single-step planner + framework chain-heuristic per Spike δ's pattern; production traffic diversity may warrant alternatives (multi-step planner; planner-loops-with-context). Carried forward as design question.

3. **Rule 5 framing requirement scope** (OQ #23). Load-bearing-default for BUILD with named falsification trigger to migrate the signal to headers/metadata if production evidence warrants. Spike ε' established the synthesizer systematically omits Rule 5 framing; BUILD treats prompt sharpening for Rule 5 as first-priority test per susceptibility-snapshot Advisory #3.

4. **Rounding-drift mitigation playbook** (OQ #24). Three-mechanism shape named (system-prompt sharpening → tier escalation → runtime fidelity check); specific thresholds + mechanism choices are BUILD-phase design.

5. **Routing-planner reliability under production traffic diversity** (OQ #25). 20-prompt battery at qwen3:8b is the empirical floor; production-scale characterization is BUILD/PLAY work. The gate-time refutation-threshold amendment specifies the operational signal that surfaces failure of this open question.

6. **ADR-030 disposition (i) full implementation timing** (`tool_choice` honoring; bridge mechanism is provisional). Deferred to follow-on cycle after ADR-027 reaches production; commitment itself is not evidence-conditional (configuration honesty is the structural commitment); sprint-scoping priority IS evidence-conditional. Per susceptibility-snapshot Advisory #4, the next DECIDE gate should place disposition-(i) implementation as a named opening precondition.

7. **Seven framing observations carried forward to ARCHITECT + BUILD** (F1, F2, F3, NF1, NF2, NF3, NF4). Not resolved at the gate per skill text framing-observation discipline; available to ARCHITECT for content-selection judgment.

### Specific commitments carried forward to ARCHITECT

1. **`OrchestratorRuntime` disposition as required ARCHITECT deliverable** (per susceptibility-snapshot Advisory #1 / NF2). ARCHITECT names disposition (a/b/c) before BUILD ships ADR-027.

2. **System-design + roadmap rewrite for the chat-completions surface.** Per cycle-status §"ARCHITECT handling for Cycle 7": chat-completions handler refactor is non-trivial; new modules for routing-planner integration + synthesizer integration + plan-execution loop; fitness criteria for the new pipeline's invariants. ADR-021 + ADR-022 partial-update headers + ADR-026/027/028/029/030/031/032 must be reflected in updated system-design.md + ORIENTATION.md provenance chains.

3. **Track A `refactor:` commits to apply before BUILD starts** (per conformance-scan): Spike ζ routing-planner output schema (add `input` field per ADR-028); Spike ε response-synthesizer (add Rule 6 per ADR-029). One YAML edit each.

4. **Track B BUILD work items scoped via the Cycle 7 Acceptance Criteria Table** (per conformance-scan + scenarios.md): chat-completions pipeline handler; production routing-planner + response-synthesizer ensembles under `agentic-` prefix; `_ChatCompletionsRequest` extension for bridge mechanism; honest response labeling at three layers; capability-list discovery via `/v1/models` extension or sibling endpoint; dispatch event substrate extensions (pipeline-stage events + degradation events); event consumers for degradation events. Estimated ~16 person-days median per OQ #19; BUILD scope is realistic per susceptibility-snapshot's positive evaluation.

5. **Confabulation modes validation quality differentiation in BUILD regression suite** (per susceptibility-snapshot Advisory #2 / F3). PLAY-note-22 case (directly witnessed, large-n) is qualitatively different from Spike μ cases (constructed fixtures, n=1 per mode); BUILD regression suite should distinguish.

6. **Cline integration smoke test before deploying to Tier B clients** (per ADR-031 §Tier B). Single-capability NL request within `requestTimeoutMs` − 5s headroom; chained-composition request within `requestTimeoutMs` − 10s headroom; failure investigation pathway via Cline issue #4308.

7. **PLAY-phase refinement of the refutation threshold** (per ADR-032 round-7 amendment). The ~15pp threshold + 24-hour rolling window are starting heuristics; PLAY field notes are the empirical surface that refines.

8. **Backward propagation sweep results** (per Step 3.7): AS-10 propagation found zero contradicting language in prior ADRs; Amendment Log entry #14 records the codification + propagation list. Bridge mechanism §Concepts entry preserves supersession discipline (in-place amendment + Amendment Log entry when disposition (i) lands; not deletion).

## Snapshot summary

**Susceptibility snapshot verdict:** **No Grounding Reframe warranted.** Per `docs/agentic-serving/housekeeping/audits/susceptibility-snapshot-cycle-7-decide.md`, DECIDE is the strongest phase in the Cycle 7 corpus from a susceptibility standpoint. The 5-round audit loop is disciplined closure, not compounding correction churn. All 7 ADRs include honest Provenance check sections labeling driver-derived vs. drafting-time content. Cross-ADR AS-10 inheritance is not blind adoption — each downstream ADR engages AS-10 substantively in its rejected-alternatives or carved-out scope.

**Five advisory carry-forwards** (named in §"Specific commitments carried forward to ARCHITECT" above): NF2 OrchestratorRuntime disposition as required deliverable; F3 confabulation modes validation quality; Rule 5 BUILD execution risk; ADR-030 disposition-(i) enforceability; per-ADR rejected-alternatives revisitability.

**Audit loop outcome:** 7 rounds; round 7 clean at P1/P2/P3. Gate threshold per skill text met. Seven framing observations carry forward to the practitioner gate (F1, F2, F3, NF1, NF2, NF3, NF4); ARCHITECT and BUILD inherit them for content-selection and design-rationale judgment.
