# ADR-016: Upward L0→L1 Read-Only Signal Channel for Cross-Layer Calibration

**Status:** Proposed (conditional acceptance — synthetic-data and structural-transfer validation completed 2026-05-06; first-deployment evidence pending for full operational validation; see §"Empirical validation pathway")

**Date:** 2026-05-05 (drafted); 2026-05-06 (spike validation incorporated)

**Amends:** ADR-002 (Four-Layer Architecture with Plexus Optional) — narrow exception to the "edges never upward" layering rule

**Composes with:** ADR-014 (Calibration Gate trajectory-level extension)

---

## Context

### Layer-numbering convention used in this ADR

This ADR uses **essay 005's L0–L3 numbering** consistently throughout (the same numbering established in `domain-model.md` §Methodology Vocabulary entry "Cross-layer calibration channel"):

- **L0** — Ensemble Engine (corresponds to ADR-002 "Layer 3")
- **L1** — Calibration Gate / dispatch decision logic (a sub-module of ADR-002 "Layer 2" Orchestrator Agent)
- **L2** — Orchestrator Runtime / Tool Dispatch (the broader portion of ADR-002 "Layer 2" Orchestrator Agent)
- **L3** — Session Registry (a sub-module spanning ADR-002 "Layer 2" Orchestrator Agent, with persistence concerns extending toward ADR-002 "Layer 4" Knowledge Graph when Plexus is active)

The L0–L3 numbering names the cycle's relevant *sub-architecture* of the orchestration tier, where the cross-layer calibration question lives. ADR-002's Layer 1–4 numbering names the *overall four-layer system* (Serving Layer / Orchestrator Agent / Ensemble Engine / Knowledge Graph) and remains in force unchanged. The amendment proposed in this ADR specifically updates the "edges never upward" rule that lives in `system-design.md` and applies at the L0→L1 boundary in essay 005's numbering — equivalently, at the boundary between ADR-002's "Layer 3 Ensemble Engine" outputs and the Calibration Gate sub-module within ADR-002's "Layer 2 Orchestrator Agent."

### The layering-rule discipline

ADR-002 establishes the four-layer architecture (Serving Layer / Orchestrator Agent / Ensemble Engine / Knowledge Graph). ADR-002's layering rule, made explicit in `system-design.md` and verified by FC-2 (static import check) and FC-3 (cycle detection), states that *edges never go upward* — lower layers do not import from or signal to higher layers. ADR-002 itself acknowledges the cost it pays for the architecture's value: *"two code paths to maintain for orchestrator behavior: with-Plexus and without-Plexus."* The layering rule is the structural discipline that keeps those paths bounded.

ADR-014 extends ADR-007's Calibration Gate (L1) with in-process trajectory-level calibration. ADR-014's specification is in-layer (L1 → L1) and operates on trajectory data the Calibration Gate itself collects from its own dispatch decisions. The cycle's research surfaced a stronger composition: cross-layer calibration where L0 ensemble outputs flow signals upward to inform L1 dispatch decisions. This composition is the cycle's most novel territory — the components exist (AUQ for in-process confidence; HTC for trajectory features; OI-MAS for tier-escalation) but the cross-layer composition is unstudied in the literature reviewed (essay 005 §"Long-Horizon Reliability Infrastructure").

Making calibration a cross-layer primitive requires a read-only signal path from L0 (ensemble outputs) to L1 (Calibration Gate dispatch decisions). The current layering rule prohibits this. Essay 005 §"ADR candidate #6" frames the amendment: a small-scope exception to ADR-002's layering rule — **read-only** (no upward writes; the rule's structural integrity for write paths is preserved) and **signal-channel-specific** (calibration only; not a general upward import permission).

### Cycle/scale risk and the bounding-mechanism load-bearing question

The signal path creates a feedback shape: Calibration Gate (L1) gates dispatch → Ensemble Engine (L0) runs → signals flow back to L1 → Calibration Gate uses signals for the next dispatch. **Feedback paths can compound bias**, and the cycle's literature evidence makes this risk concrete:

- **Khanal et al.** (arXiv:2603.29231) found *universal non-improvement* from episodic memory augmentation across ten tested models at long horizons — six models showed negative effects, four were neutral, none improved. The mechanism is feedback-bias compounding when stale signals influence current decisions.
- **CAAF** (arXiv:2604.17025) flagged that *"apparent LLM reliability in safety-critical domains is often a prompt engineering artifact"*. Calibration signals are themselves model outputs; gating on biased outputs entrenches bias.
- **Li et al.** (ICLR 2026) documented the trigger-vulnerability finding that injecting *objective* context into a debate accelerates polarization rather than moderating it — the directional caution transfers from debate-shape coordination to feedback-shape signal flow but the original study's domain is multi-agent debate, not unidirectional read-only calibration channels. Cycle 2's lit-review explicitly flagged this transfer as empirically open.

The amendment's drafting must therefore specify five bounding mechanisms as **load-bearing design surface, not decoration**. Per the cycle's research-gate Grounding Action 1 (recorded 2026-05-04), the five mechanisms have asymmetric implementation-readiness:

| Mechanism | Readiness class |
|-----------|-----------------|
| (a) Fresh-context isolation in the consumer | **Methodology-tooling precedent** (the RDD methodology's audit-subagent dispatch infrastructure uses fresh-context isolation; not yet present as llm-orc codebase infrastructure — BUILD ports the pattern) |
| (b) Time-decay windowing for the bias-compounding horizon | **Novel design work without reference implementation** |
| (c) Categorical anchors via deterministic-tool-output | **Direct literature precedent** (Wisdom and Delusion of LLM Ensembles, arXiv:2510.21513), but **ensemble-composition-conditional** |
| (d) Periodic out-of-band audit dispatch | **Novel design work without reference implementation** (the susceptibility-snapshot pattern from RDD methodology, applied to architectural calibration rather than to research framing) |
| (e) Read-only structural validation at the consumer | **Direct codebase precedent** (typed-error pattern from commit `9f86d0b`; ADR-017's structural validation guard) |

Argument-audit and conformance-audit on this ADR concentrate on (b) and (d) where the load-bearing question of whether the mechanism can be operationalized is genuinely open. (a), (c), (e) are specified as elaborations of existing patterns. **The "five mechanisms as a coherent operationalized set" framing has been replaced with the "two-elaboration + one-conditional + two-novel-design" classification** (research-gate Grounding Action 1, recorded 2026-05-04).

### Framing commitment from research-gate Grounding Action 2

The practitioner's framing commitment, recorded at DECIDE entry on 2026-05-05, is **elaboration-by-evidence** (reading (a)): the seven ADR candidates as currently scoped constitute the long-horizon strategy; module-shape inheritance from essay 005's verdict is held; ADR-002's four-layer frame is retained; bounding mechanisms operationalize within ADR-002's L1 layer rather than as a cross-cutting module. The architectural reorganization reading is held in reserve, conditional on practice-based evidence that the as-designed system fails for long-horizon tasks.

ADR-016 honors this framing commitment by specifying the bounding mechanisms within ADR-002's existing layer structure. The falsification trigger applies: if BUILD or first-deployment evidence finds that (b) or (d) cannot be operationalized within L1 (e.g., they require a cross-cutting module shape orthogonal to L0–L3), the elaboration-by-evidence framing commitment is invalidated, the reorganization branch re-opens, and ADR-016 is re-deliberated with reorganization on the table.

### Practitioner guidance on unsupported assumptions

The practitioner's gate-conversation guidance on 2026-05-05 (Tranche B close): *"I'd rather be explicit about potential decisions that need more support. What I don't want is to codify an unsupported assumption without evidence. Rather we should do spikes or experiments to validate our hunches."* This guidance is load-bearing for ADR-016's drafting. The conditional-acceptance status, the empirical validation pathway specifications below, and the explicit marking of mechanisms (b) and (d) as novel-design-pending-validation are all applications of this guidance.

---

## Decision

ADR-016 amends ADR-002's layering rule with a **single narrow exception**: a read-only signal channel from L0 (Ensemble Engine outputs) to L1 (Calibration Gate dispatch decisions), gated by five bounding mechanisms.

### The exception

The amendment to ADR-002 reads:

> *Edges never go upward, with one exception: a read-only signal channel may flow from L0 to L1, restricted to calibration data and gated by the five bounding mechanisms specified in ADR-016. The exception is signal-channel-specific (calibration only; not a general upward import permission) and read-only (no upward writes; ADR-002's layering rule for write paths is unchanged). All other layer pairs (L1→L2 upward, L2→L3 upward, L1→L3 upward) remain prohibited.*

The amendment is the smallest scope exception consistent with operationalizing cross-layer calibration. The Plexus integration boundary (L4) is unchanged — Plexus remains optional per ADR-002's AS-8.

### The signal channel

The signal channel carries **calibration data** from L0 to L1. The data shape includes:

- Ensemble output trajectory features (extracted at L0 per ADR-014's HTC specification, available for L1's calibration verdict)
- Ensemble dispatch outcomes (success/failure structural signals; not LLM-judgment summaries)
- Deterministic-tool-output signals (where the ensemble has script-model slots; categorical anchor signals per mechanism (c))

Data is **structurally typed** at the channel boundary — the L1 consumer reads typed signal data, not arbitrary ensemble state. The channel is **read-only at the L1 consumer**: L1 cannot write back to L0 through the channel.

Per the practitioner's no-token-limit-pre-optimization guidance (recorded 2026-05-05 Tranche-A close), the signal-channel data shape is generous rather than minimized — token economy on free local models is not a binding constraint for the channel's deployment.

### The five bounding mechanisms (asymmetric specification per readiness class)

#### (a) Fresh-context isolation in the consumer — *direct architectural precedent*

The L1 calibration consumer reads signals in a context that does not carry prior signal forward. Each calibration verdict computation runs in a fresh evaluation context; prior signals influence the next verdict only through the time-decay-windowed feature aggregation specified by mechanism (b), not through context accumulation in the L1 consumer.

**Architectural precedent (refined per conformance scan 2026-05-06).** This is the same architectural-isolation pattern the RDD methodology uses for audit-subagent dispatches (citation-auditor, argument-auditor, susceptibility-snapshot-evaluator). The pattern is established in the **RDD methodology's tooling infrastructure** — the rdd plugin's audit-subagent dispatch mechanism (the same dispatch infrastructure that produced this ADR's argument-audit and conformance-scan reports) — and in methodology documents (ADR-058 of the rdd plugin establishes Architectural Isolation as the Tier 1 mechanism class). The pattern is **not** present as coded infrastructure in the llm-orc `src/` codebase; BUILD will implement mechanism (a) by porting the architectural-isolation pattern from its current methodology-tooling form into llm-orc infrastructure. Specification: ADR-applicable as elaboration of *methodology-precedent* pattern; codebase implementation is BUILD-time work, not pre-existing.

#### (b) Time-decay windowing for the bias-compounding horizon — *novel design work without reference implementation*

Only signals within a bounded recent window influence the current calibration verdict. Older signals decay out of the verdict computation; their influence on current decisions falls to zero outside the window.

**Specification (synthetic-data validated 2026-05-06; operational tuning pending):**

- **Window shape: dual-bound with the shorter taking precedence.** The window is bounded by the lesser of (i) a wall-clock time bound (default 60 minutes; operationally tunable) or (ii) a signal-count bound (default last 100 signals; operationally tunable). Spike (b) results suggest *smaller* window parameters consistently track better than the default across tested synthetic bias trajectories — operational deployment should test smaller defaults (e.g., 30 / 50) against deployment-realistic dispatch frequencies.
- **Decay function: linear within the window.** Signals within the window contribute to the verdict with a weight that linearly decreases from 1.0 at signal-emission to 0.0 at window-edge. Spike (b) confirmed linear decay outperforms hard cutoff (uniform weight inside window) and is comparable to exponential decay; the linear-decay specification holds rather than relaxing to "any window-shape inside the dual-bound."
- **Storage discipline.** Historical signals outside the window remain in the artifact record (and, when Plexus is active, in the knowledge graph) for analysis purposes; they do not influence current verdicts. The windowing applies at feature-aggregation time, not at signal-storage time.

**Validation status (updated 2026-05-06).** The windowing's structural bias-bound property is validated by spike (b) — synthetic-data simulation showed the windowing fully eliminates stale-signal contribution across three bias-trajectory scenarios (slow drift, step change, periodic oscillation). The default parameters produce positive tracking-error reduction in all scenarios; smaller-window configurations track better, indicating the default may be too large for deployment-realistic bias dynamics. See research log `005e-spike-adr016-b-time-decay-windowing.md` for full method and findings. Operational validation (whether the windowing's bias-bound holds in real deployments) remains pending first-deployment evidence on the cycle's North-Star benchmark.

**Why structural validation matters but is not the whole story.** Khanal et al.'s universal non-improvement finding is the cycle's load-bearing literature evidence on feedback-bias compounding. The synthetic-data spike establishes that the *logical* mechanism produces the bias-bound property. Whether the bound is *operationally sufficient* under real-deployment conditions — where bias dynamics may differ from synthetic trajectories and where multi-iteration accumulation effects may surface — remains the question first-deployment evidence will answer. Mechanism (b) without empirical operational validation is logically sound; mechanism (b) with first-deployment evidence is empirically grounded.

#### (c) Categorical anchors via deterministic-tool-output — *direct literature precedent (ensemble-composition-conditional)*

Where the ensemble has script-model slots, deterministic outputs anchor the feedback loop against probabilistic drift. Deterministic outputs cannot be argued away by LLM consensus, so the feedback loop cannot drift on probabilistic noise.

**Literature precedent.** Wisdom and Delusion of LLM Ensembles (arXiv:2510.21513, October 2025) on CrossHair counterexample feedback embedded in code-generation ensembles. Spike A3 (Cycle 2) instantiates the same pattern at the script-member-alongside-LLM stage. The mechanism is categorical, not probabilistic.

**Anchor-strength scope condition (argument-audit P3.4 finding 2026-05-06).** The "cannot be argued away by LLM consensus" property is strongest for **binary-verifiable deterministic outputs** — the formal counterexample shape from CrossHair (output is structurally pass/fail; no interpretation latitude). Deterministic outputs that are interpretable (structured reports, numerical scores, multi-field analyses) provide a weaker anchor: the LLM consensus can frame, dismiss, or recontextualize an interpretable deterministic output even when it cannot directly contradict its values. The mechanism's anchor strength scales with output verifiability — binary-verifiable is the strongest anchor; interpretable-numerical is moderate; interpretable-prose is weakest. Mechanism (c)'s value in any specific ensemble configuration depends on which class of deterministic output the script-model slot produces.

**Ensemble-composition-conditional applicability.** The mechanism is available only when the ensemble has script-model slots. **LLM-only ensembles cannot use this anchor.** ADR-016 specifies the conditionality explicitly: cross-layer calibration on LLM-only ensembles operates with bounding mechanisms (a), (b), (d), (e) only — mechanism (c) is not available in those configurations. The other four mechanisms remain load-bearing in LLM-only configurations; (c) is a strengthening anchor when available.

#### (d) Periodic out-of-band audit dispatch for architectural calibration drift — *novel design work, structural-transfer validated 2026-05-06*

A periodic auditor checks whether the calibration system itself is drifting. The auditor runs in a fresh context (consistent with mechanism (a)'s isolation pattern), reviews recent calibration verdicts and their outcomes, and produces a typed audit verdict (consistent with (e)'s structural validation pattern).

**Specification (refined 2026-05-06 from spike (d) structural transfer audit):**

- **Trigger frequency.** The audit dispatches periodically — default every 100 calibration verdicts or every 24 wall-clock hours of session activity, whichever comes first; operationally tunable. The trigger is structural (count-based or time-based), not LLM-mediated.

- **Audit scope.** The auditor reads the time-windowed signal record and the corresponding calibration verdicts produced over the audit window. The auditor does not re-run the calibration computation — it audits the calibration *system's* behavior. Three drift-detection criteria, each at quantitative-threshold level (operationally tunable; specification establishes criterion shape):

  - **Verdict skew.** Distributional shift in the (Proceed / Reflect / Abstain) trichotomy frequency relative to the prior audit window. Threshold: relative-frequency change exceeding the configured tolerance (default ±15% per verdict class). The criterion shape is concrete (a frequency-distribution comparison test); thresholds are tunable.
  - **Outcome divergence.** The verdicts' predictive accuracy for downstream dispatch outcomes drops over consecutive audit windows. Threshold: predictive-accuracy decline exceeding the configured tolerance (default −10 percentage points per audit window). Outcomes are recorded by the post-hoc result-check from ADR-007's existing first-N calibration mechanism plus dispatch-completion signals from the Calibration Gate.
  - **Signal-to-verdict correlation drift.** The relationship between input signal trajectory features (HTC features per ADR-014) and output calibration verdicts shifts. Threshold: correlation-coefficient change exceeding the configured tolerance (default ±0.20 per audit window).

  The drift surfaces also explicitly include detection of mechanism (b) parameter drift — if smaller-window configurations track better than the current default consistently, the audit produces a parameter-tuning recommendation (per the (b)/(d) coupling identified in spike (b)).

- **Audit verdict shape.** Three classes parallel to the susceptibility-snapshot's clean / advisory / Grounding-Reframe-trigger:

  - *No drift detected* — all three drift-detection criteria within thresholds; calibration system continues operating; no diagnostic produced.
  - *Drift detected* (advisory) — at least one drift-detection criterion exceeds threshold; the audit produces an operator-readable diagnostic *plus a parameter-tuning recommendation* identifying which threshold was exceeded and which mechanism (b) parameter adjustment may correct it; calibration system continues operating with the diagnostic logged. The three-property test for advisory: specific (names concrete criteria-and-thresholds) + actionable (specific operator action implied by the recommendation) + **operationally applicable** (the deployed calibration system can act on the recommendation before the next audit fires; or the operator's near-term review can act).
  - *Severe drift* — multiple drift-detection criteria simultaneously exceed thresholds, or any criterion exceeds severe-threshold (default 2× advisory threshold); calibration system enters fail-safe mode (calibration verdicts default to Reflect-or-Abstain); operator notified through the operator-notification mechanism specified below.

- **Operator action surface.** The advisory level produces *diagnostic + parameter-tuning recommendation* (option (ii) per spike (d) audit). The operator's expected action is review-and-approve-or-override, not author-from-scratch. The audit produces specific candidate parameter adjustments based on the drift criteria; the operator approves to apply at the next session boundary or overrides with explicit rationale.

- **Asynchronous-operator-review dynamic.** The susceptibility-snapshot pattern depends on the user being present at the gate. Mechanism (d) operates during deployed-system runtime, where the operator may be absent. Implications:

  - **Advisory drift** is handled passively — diagnostic logged to operator-facing storage; calibration continues operating; operator reviews at their cadence.
  - **Severe drift** is handled actively — fail-safe mode triggers automatically (no operator presence required); operator-notification mechanism fires (log entry with severity tag; optional webhook/email per operator deployment configuration).
  - **Failure mode under operator absence.** If the operator never reviews advisory diagnostics, the calibration system continues operating with degraded calibration accuracy that the audit has flagged but the operator has not addressed. If drift escalates from advisory to severe (the same criteria worsen across consecutive audit windows), the system enters fail-safe mode without operator intervention. Sustained fail-safe mode persists until operator review.

- **Audit logic provenance.** The pattern is the susceptibility-snapshot pattern from RDD methodology (which audits research framing for content-selection sycophancy), applied to *architectural calibration* drift rather than *research framing* drift. **The transfer is structural** — the same dispatch-fresh-context-evaluator-produces-typed-verdict pattern, with different content (calibration data instead of research-framing data). Spike (d) validated the structural transfer at the analytical level; the transfer is largely clean with three specification gaps (drift-criteria specificity, operationally-applicable test, asynchronous-operator dynamic) addressed in the refined specification above.

**Validation status (updated 2026-05-06).** Spike (d) — structural transfer audit — validated the pattern's transfer from RDD methodology to architectural calibration drift detection at the analytical level. The transfer is largely clean (properties 1–5, 7, 12 transfer without modification); three specification gaps were identified and have been addressed in the refined specification above. See research log `005f-spike-adr016-d-structural-transfer-audit.md` for the full property-by-property audit. Operational validation (whether the audit produces useful drift detection in real deployments; whether the operator workflow is workable; whether the (b)/(d) coupling functions in practice) remains pending first-deployment evidence.

**Mechanism (b)/(d) coupling.** Spike (d) identified that mechanisms (b) and (d) are more tightly coupled than ADR-016's initial drafting suggested: (b) is the bias-bound mechanism, (d) is the *parameter-drift detector* that informs when (b)'s parameters need tuning. The drift criteria above (especially the verdict-skew and signal-to-verdict-correlation surfaces) explicitly include detection of (b)'s parameter drift; the parameter-tuning recommendation in the advisory-verdict's diagnostic is the natural consumer of this coupling. Without mechanism (d), mechanism (b)'s parameters drift silently; without mechanism (b), mechanism (d) has no parameters to audit. The two compose into a self-correcting bias-bound system.

#### (e) Read-only structural validation at the consumer — *direct codebase precedent*

The L1 consumer validates the schema and type of incoming signal data before acting on its content. Malformed signals are rejected at the channel boundary; they do not influence verdicts.

**Codebase precedent.** Maps to the existing typed-error path established by commit `9f86d0b feat: raise typed error when provider rejects tool calling per-model`. ADR-017's tool-call structural validation guard is the parallel-by-construction pattern at a different surface. Specification: ADR-applicable as elaboration of existing pattern.

**Specification.** Schema validation runs at the channel boundary: incoming signals must match the typed schema (trajectory feature schema, dispatch outcome schema, deterministic-output schema). Schema mismatch produces a typed `malformed_signal` error consistent with the typed-error pattern. The L1 consumer does not act on the signal's content; the error is logged and the verdict computation skips the malformed signal (treating it as if outside the time window).

---

## Rejected alternatives

**(a) No upward signal channel (preserve ADR-002's layering rule unchanged).** Rejected: the cycle's research established that cross-layer calibration is the most novel territory the architecture can occupy at the cycle's deployment shape, and the components exist (AUQ, HTC, OI-MAS) but the composition does not. Without the upward signal channel, ADR-014's in-process calibration is L1-internal-only — it cannot incorporate L0 ensemble-output signals. The cycle's design-method posture (essay 005's composite framing element D — experience accumulation as the progression arc) loses its operational instantiation. The cost (preserving ADR-002 unchanged) is the value (cross-layer calibration). Not preserving the layering rule discipline is the cost ADR-016 actually proposes; preserving it absolutely loses the value.

**(b) Cross-cutting module orthogonal to L0–L3 (the reorganization branch from research-gate Grounding Action 2).** Rejected at DECIDE entry per the practitioner's framing commitment to elaboration-by-evidence. The framing commitment is conditional on practice-based evidence; if BUILD or first-deployment finds (b) or (d) cannot be operationalized within L1, the reorganization branch re-opens. ADR-016 honors the framing commitment by specifying within-L1 operationalization; the falsification trigger is explicit.

**(c) Bidirectional channel (allow both upward signal flow and downward control flow).** Rejected: the read-only constraint is load-bearing. ADR-002's layering rule is preserved for write paths because write paths carry the structural integrity guarantees that make the architecture analyzable. A bidirectional channel would erode the rule's structural integrity beyond what the cross-layer calibration value justifies. The read-only restriction limits the amendment to exactly what's needed.

**(d) Open-scope upward channel (allow any upward signal, not calibration-only).** Rejected: the calibration-only restriction is load-bearing. An open-scope upward channel would be a general upward import permission, which would erode ADR-002's layering rule fully. The amendment's value is the contained exception, not the principle of upward signaling.

**(e) Reduce the bounding-mechanism set to the three with direct precedent ((a), (c), (e); drop (b) and (d) as overengineering).** Rejected: the cycle's literature evidence on feedback-bias compounding (Khanal et al.'s universal non-improvement; CAAF's prompt-engineering-artifact; Li et al.'s trigger-vulnerability) makes time-decay windowing and out-of-band audit dispatch load-bearing risk-bounding mechanisms, not overengineering. Direct-precedent mechanisms (a), (c), (e) bound *known* failure modes (context accumulation, probabilistic drift, malformed signals); novel-design mechanisms (b), (d) bound *the cycle's load-bearing risk* (feedback-bias compounding from cross-layer signal flow). Dropping (b) and (d) leaves the load-bearing risk unbounded; per essay 005, *"absent operational bounding mechanisms, the precedent erodes the layering rule's discipline rather than carving a contained exception."*

**(f) Conditional acceptance only on (a), (c), (e); defer (b), (d) to a separate ADR after spike validation.** Rejected for cohesion: the bounding mechanisms compose as a single set. Splitting (b), (d) into a separate ADR creates the surface where the layering-rule amendment ships without its load-bearing bounding mechanisms — exactly the failure mode essay 005 warns against. The conditional-acceptance approach for the *whole* ADR (status: proposed pending operational validation of (b) and (d)) is the cleaner application of the practitioner's "be explicit about decisions needing more support" guidance.

**(g) Synthetic-data spike on (b) windowing dynamics in DECIDE before closing the ADR.** Considered, deferred decision pending practitioner input (see §"Empirical validation pathway"). The spike could simulate calibration data flow with synthetic signals at varying time-decay rates and measure whether the windowing prevents bias compounding. The argument for running it in DECIDE: empirical validation of (b) before ADR closure honors the practitioner's "validate via spike" guidance directly. The argument for deferring it: the cycle's close shape is Mode B+ → DECIDE; running spikes overflows into RESEARCH territory. ADR-016's drafting completes; whether to run the spike in DECIDE before closure is presented as a Tranche-C-summary decision to the practitioner.

**(h) Structural transfer audit on (d) susceptibility-snapshot-pattern to architectural-calibration in DECIDE before closing the ADR.** Considered, deferred decision pending practitioner input (see §"Empirical validation pathway"). The audit could examine whether the susceptibility-snapshot pattern's properties (clear evaluation criteria; clear consumer; typed verdict shape) transfer cleanly to architectural calibration. Same decision-class as (g); presented as a Tranche-C-summary decision.

---

## Consequences

**Positive:**
- Operationalizes cross-layer calibration — the cycle's most novel territory — within the smallest layering-rule exception consistent with the value
- Preserves ADR-002's layering rule for write paths (the structural integrity guarantee for analyzability)
- The exception is bounded structurally (read-only, signal-channel-specific) rather than by convention
- Composes with ADR-014 (cross-layer signal input) and ADR-015 (downstream verdict consumer) to form the cycle's calibration-and-escalation system
- Bounding mechanisms (a), (c), (e) ship with direct precedent; (b), (d) ship with rigorous specifications and explicit empirical validation pathways
- The conditional-acceptance status makes the unsupported-by-evidence portions visible at the architecture-decision level rather than burying them in BUILD work
- The falsification trigger is explicit; if practice-based evidence invalidates the elaboration-by-evidence framing commitment, the reorganization branch re-opens with the falsification recorded

**Negative:**
- ADR-002's layering rule now has an exception that future ADRs and FCs must understand — the static import check (FC-2) and cycle detection (FC-3) need updating to recognize the calibration-channel exception
- Bounding mechanisms (b) and (d) ship with provisional specifications; if BUILD or first-deployment evidence finds the specifications inadequate, the ADR re-opens
- Five bounding mechanisms add coordination complexity — the calibration-channel boundary plus four mechanisms (excluding (c) for LLM-only ensembles) operate together on every L1 verdict
- Time-decay windowing's storage-vs-aggregation distinction adds operator-understandable subtlety — historical signals are kept (artifact record, Plexus when active) but do not influence current verdicts; operator surfaces must communicate this clearly to avoid confusion
- The audit-dispatch mechanism (d) is novel work; the susceptibility-snapshot transfer is structurally clear but the empirical clarity (what counts as drift, what diagnostic is actionable) requires deployment evidence
- The conditional acceptance status defers the cycle's load-bearing decision's full closure to operational validation — a real cost the cycle's close shape (Mode B+ → DECIDE) accepts but does not eliminate

**Neutral:**
- Plexus integration is unchanged (L4 boundary remains optional per AS-8)
- The signal-channel data shape is generous per the practitioner's no-token-limit-pre-optimization guidance; cost-bearing tier deployment may need scoping
- The five-mechanism set with asymmetric implementation-readiness mapping is the cycle's load-bearing structural framing; future cycles inheriting this work read mechanisms (a), (c), (e) as adoption-discipline territory and (b), (d) as ongoing-validation territory
- The exception is the only amendment ADR-002 takes from this cycle; ADR-016 does not amend the rest of ADR-002's substance
- **Plexus-conditional value of cross-session calibration stabilization (advisory, per round-1 framing audit P2 + decide-phase susceptibility snapshot 2026-05-06).** ADR-014's post-hoc promotion tracking (the cross-session signal-accumulation mechanism that determines trusted-status transition under AS-5) requires Plexus to be active for cross-session value. In Plexus-absent stateless mode, the cross-layer calibration channel still operates within a session — bias-bound trajectory features inform per-dispatch verdicts — but the cross-session stabilization value (calibration signals accumulating across sessions to inform future dispatch decisions) is unavailable. The ADR's value proposition is partially Plexus-conditional: full value requires Plexus activation per AS-8's optionality; in-session value is preserved without Plexus. Operators evaluating ADR-016's deployment cost should scope the value claim to their Plexus-activation status

---

## Empirical validation pathway

The conditional-acceptance status names two mechanisms — (b) time-decay windowing and (d) periodic out-of-band audit dispatch — as novel design work without reference implementation. **DECIDE-phase spikes (per practitioner Path-2 authorization 2026-05-06) completed validation at the structural/logical level.** Remaining validation surfaces are pre-BUILD/first-deployment work:

### Mechanism (b) — time-decay windowing

**Validation status (updated 2026-05-06).** Spike (b) — synthetic-data simulation — was completed at DECIDE close per practitioner Path-2 authorization. The spike validated the windowing's structural bias-bound property (windowing fully eliminates stale-signal contribution across three bias-trajectory scenarios) and the parametric tracking property (default parameters produce positive tracking-error reduction in all scenarios). Operational tuning territory surfaced: smaller window configurations consistently track better than the default, indicating the default may be too large for deployment-realistic bias dynamics. See research log `005e-spike-adr016-b-time-decay-windowing.md` for the full method and findings.

**Remaining validation pathway:**

1. ~~**Synthetic-data spike**~~ — completed 2026-05-06. Logic of the windowing specification validated.

2. **Pre-BUILD spike on real data at small scale (medium cost, Cycle 5+).** Run windowed calibration on a small ensemble dispatch corpus (e.g., the spike-cycle4-research-loop fixture extended); measure verdict distribution under windowed vs. non-windowed configurations. Closer to operational validation but smaller scope than full BUILD.

3. **First-deployment evidence (high cost, post-BUILD).** Operationalize windowing in the actual implementation; measure long-horizon session calibration trajectory; surface bias-compounding evidence (or absence) over multi-iteration runs. The natural validation surface for the cycle's North-Star benchmark.

### Mechanism (d) — periodic out-of-band audit dispatch

**Validation status (updated 2026-05-06).** Spike (d) — structural transfer audit — was completed at DECIDE close per practitioner Path-2 authorization. The audit examined the susceptibility-snapshot pattern's structural properties property-by-property and identified which transfer cleanly to architectural calibration drift detection. The transfer is largely clean (properties 1–5, 7, 12 transfer without modification). Three specification gaps were identified and have been addressed in the refined mechanism (d) specification (drift-criteria specificity raised to quantitative-threshold level; the in-cycle-applicability test reframed as operationally-applicable; asynchronous-operator-review dynamic specified explicitly). See research log `005f-spike-adr016-d-structural-transfer-audit.md` for the full audit.

**Remaining validation pathway:**

1. ~~**Structural transfer audit**~~ — completed 2026-05-06. Pattern transfer validated at the analytical level.

2. **Manual audit-dispatch trial (medium cost, available in Cycle 5+).** Manually run an audit dispatch on a synthetic or small-scale calibration record; check whether the dispatch produces useful drift diagnostics under the refined specification. Validates the audit mechanism's logic at small scale.

3. **First-deployment evidence (high cost, post-BUILD).** Operationalize the audit dispatch; measure whether it produces drift diagnostics that lead to operator action that actually corrects drift. The natural validation surface for the cycle's North-Star benchmark.

### Falsification trigger

If any validation pathway produces evidence that mechanism (b) or (d) cannot be operationalized within the Calibration Gate's existing or extended class structure without introducing a new top-level module (e.g., the windowing's bias-compounding bound fails empirically; the audit dispatch's drift criteria don't transfer; either mechanism requires hosting in a module that is not within ADR-002's existing four-layer structure), the elaboration-by-evidence framing commitment is invalidated. The reorganization branch re-opens; ADR-016 is re-deliberated with reorganization on the table; ADR-002's layering rule amendment is re-evaluated.

The falsification trigger is **specific** (named mechanisms; named failure modes; BUILD-concrete module-structure criterion), **actionable** (re-open reorganization branch; re-deliberate ADR-016), and **operationally applicable** (validation pathways 1 for both mechanisms have been completed in DECIDE; pathways 2 and 3 are pre-BUILD/first-deployment territory).

### Concrete monitoring specification (post-spike-validation)

The conditional-acceptance status's load-bearing question (per argument-audit P1.2 finding 2026-05-06) is whether the conditionality functions as a structural gate or only as a monitoring aspiration. The following specifies what artifact, which phase, and which human action constitutes first-deployment-evidence receipt and triggers re-deliberation:

**Trigger artifact.** First-deployment evidence is captured in either (i) a BUILD-phase research log entry recording the cross-layer channel's first dispatch outcome on a non-trivial fixture, or (ii) a PLAY-phase field note recording the channel's behavior on the cycle's North-Star benchmark.

**Trigger phase.** The earlier of: (a) the first BUILD-phase test dispatch that exercises the cross-layer channel end-to-end (mechanism (b) windowing on real signals + mechanism (d) audit dispatch on real verdicts); (b) the first PLAY-phase scenario that involves multi-iteration session continuation under the cross-layer channel.

**Trigger conditions for re-deliberation.** Any of the following observed in the trigger artifact escalates the conditional-acceptance status:

- Mechanism (b) windowing produces residual bias accumulation that the parameter-tuning recommendation surface cannot correct
- Mechanism (d) audit dispatch fails to produce actionable drift diagnostics (operator cannot determine what action to take from the diagnostic)
- Mechanism (b) or (d) requires a top-level module outside ADR-002's L0–L3 structure to implement (the BUILD-concrete module-shape criterion)
- The (b)/(d) coupling fails — drift detection does not surface windowing-parameter problems

**Trigger action.** The practitioner reviews the trigger artifact and decides: (a) the conditionality moves to full acceptance (both mechanisms operationalized successfully); (b) the conditionality is preserved with deployment-tuning notes (operationalization works but requires parameter adjustments not yet documented); (c) the falsification trigger fires (a superseding ADR is filed; the reorganization branch is re-opened in the next cycle's RESEARCH territory).

**Sweep responsibility.** Per ADR-068's pattern for monitoring-conditional decisions, the practitioner reviews this status at the end of each cycle that exercises the cross-layer channel. The cycle-status `Phase Status` table for any cycle that touches ADR-016 includes a row noting the channel's status (conditional / fully accepted / superseded). Absence of the row constitutes structural evidence that the cycle has not exercised the channel.

This monitoring specification is what makes the conditional-acceptance status structurally meaningful rather than an aspirational label. The conditionality is bounded by a concrete review obligation at each cycle boundary that exercises the channel.

---

## Provenance check

- **Driver-derived content (the upward signal channel amendment).** The amendment's specification (read-only, signal-channel-specific, calibration-only) is essay-derived from essay 005 §"ADR candidate #6". The driver chain runs essay 005 → cycle 4 research-gate Grounding Action 1 (asymmetric implementation-readiness mapping) → cycle 4 research-gate Grounding Action 2 (elaboration-by-evidence framing commitment).

- **Driver-derived content (the five bounding mechanisms).** The mechanism set is essay-derived from essay 005's §"ADR candidate #6, gate-conversation refinement, 2026-05-04". The asymmetric readiness classification is from research-gate Grounding Action 1 (recorded 2026-05-04). The literature drivers for each mechanism are direct (Wisdom and Delusion of LLM Ensembles for (c); commit `9f86d0b` for (e); RDD methodology susceptibility-snapshot pattern for (d)'s structural transfer; the architectural-isolation pattern for (a); Khanal et al. and CAAF for the cycle/scale risk that motivates (b) and (d)).

- **Drafting-time synthesis (mechanism (b) specification).** The dual-bound (60-minute / 100-signal) and linear-decay specifications were drafting-time synthesis. Essay 005 specifies time-decay windowing as a load-bearing mechanism but does not specify the window shape or decay function. **Spike (b) (2026-05-06) validated the windowing's structural bias-bound property and the linear-decay-over-hard-cutoff parametric advantage**; the dual-bound default values remain operationally tunable, with the spike's parameter-sensitivity finding (smaller windows track better in tested scenarios) noted in the mechanism (b) specification.

- **Drafting-time synthesis (mechanism (d) specification).** The trigger frequency, audit scope, and audit-verdict shape were drafting-time synthesis. Essay 005 specifies the susceptibility-snapshot-pattern transfer at the structural level; the operational specification was drafting-time application of the structural pattern. **Spike (d) (2026-05-06) validated the structural transfer property-by-property and surfaced three specification gaps that have been addressed in the refined mechanism (d) specification**: drift-criteria specificity raised to quantitative-threshold level; the in-cycle-applicability test reframed as operationally-applicable; asynchronous-operator-review dynamic specified explicitly. The mechanism (b)/(d) coupling — (d) detecting (b)'s parameter drift — was identified by spike (d) and integrated into mechanism (d)'s drift-detection criteria.

- **Drafting-time synthesis (the conditional-acceptance status).** The status `Proposed (conditional acceptance)` is a drafting-time application of the practitioner's gate-conversation guidance on unsupported assumptions. Essay 005 specifies that the layering-rule amendment is conditional on the bounding mechanisms being load-bearing in implementation; the conditional-acceptance status formalizes the conditionality at the ADR-status level. **Post-spike-validation update**: the conditionality scope has narrowed from "operational validation of bounding mechanisms (b) and (d)" (pre-spike) to "first-deployment evidence pending for full operational validation" (post-spike) — structural and logical validation completed by the DECIDE-phase spikes.

- **Drafting-time synthesis (validation pathway specifications).** The three-option validation pathways for (b) and (d) are drafting-time synthesis. Essay 005 names BUILD-time and conformance-audit pressure-testing as the validation venues; the three-option structure (synthetic-data spike → small-scale real-data spike → first-deployment evidence) is drafting-time judgment about cost-vs-evidence trade-offs in the validation pipeline.

- **Drafting-time synthesis (LLM-only-ensemble conditionality treatment for mechanism (c)).** Essay 005 flags (c) as ensemble-composition-conditional. The explicit specification that LLM-only ensembles operate with mechanisms (a), (b), (d), (e) only — without (c) — is drafting-time synthesis. The alternative (require (c) for all ensembles, blocking LLM-only configurations from cross-layer calibration) was rejected for being inconsistent with ADR-002's AS-8 (Plexus optionality) parallel: configuration optionality is established architecture in this corpus.

- **Vocabulary impact.** ADR-016 introduces five terms candidate for domain-model addition at Tranche-C close, all currently in §Methodology Vocabulary as proposed-pending-DECIDE entries (per Amendment Log entry #2):
  - **Cross-layer calibration channel** — promote from "proposed; pending DECIDE" to confirmed methodology vocabulary
  - **Bounding mechanisms (a–e)** — promote, with mechanism specifications visible
  - The conditionality of mechanisms (b), (d) is preserved in the vocabulary entry text

- **Asymmetric DECIDE budget per research-gate carry-forward #4.** ADR-016 is the cycle's load-bearing decision and warrants the heaviest argument-audit and conformance-audit budget. Argument-audit on this ADR should pressure-test:
  - **(i) The mechanism (b) and (d) specifications** — are they rigorous enough to be operationalizable, or are they too thin to distinguish from decoration?
  - **(ii) The conditional-acceptance status** — is the conditionality structurally meaningful, or does it shipping the ADR functionally accept the unsupported portions?
  - **(iii) The falsification trigger** — is the trigger specific enough to actually fire, or is it formal language that won't fire in practice?
  - **(iv) The composition with ADR-014** — does the data-flow assumption (ADR-014's input expansion under ADR-016 acceptance) cohere with ADR-014's L1-internal-default specification?
  - **(v) The amendment scope** — is read-only-and-calibration-only narrow enough to preserve ADR-002's discipline, or does the precedent erode the rule meaningfully?

  Conformance-audit on this ADR should examine the codebase for the precedent patterns the mechanisms reference: fresh-context isolation in audit-subagent dispatches (mechanism (a)); typed-error path at commit `9f86d0b` (mechanism (e)); Spike A3's script-member-alongside-LLM pattern (mechanism (c)). The codebase precedents must be operational, not only documented.
