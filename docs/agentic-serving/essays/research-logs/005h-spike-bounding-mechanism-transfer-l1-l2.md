# Research Log 005h — Spike β: Bounding-Mechanism Transfer Audit for the L1→L2 Verdict→Router Stage

**Cycle:** 4
**Phase:** ARCHITECT (architect-gate continuation, per practitioner approval 2026-05-11)
**Date:** 2026-05-11
**Type:** Analytical spike — property-by-property structural transfer audit on ADR-016 mechanisms (a)–(e) against the Tier-Escalation Router → Calibration Gate edge
**Edge under audit:** L1 → L2 verdict→router stage (`Tier-Escalation Router → Calibration Gate` per system-design.agents.md v3.0 §Dependency Graph)

---

## Spike question

Can the bounding-mechanism pattern (a)–(e) from ADR-016 actually transfer to the Tier-Escalation Router → Calibration Gate edge (the L1→L2 verdict→router stage), or does it not apply structurally?

The OQ #14 finding (decide-gate 2026-05-08) flagged five cross-layer stages with grounding-mechanism rigor lower than ADR-016. Practitioner chose option (c) — defer to BUILD evidence — uniformly. The susceptibility-snapshot's F6 finding (architect-phase audit 2026-05-11) named the verdict→router edge as the highest-priority OQ #14 carry-forward because Sub-Q6's routing-reliability evidence gap (ADR-015 §Consequences §Neutral) operates exactly at this edge: if the verdict is unreliable, the Router's escalation decisions are noise, and no grounding mechanism is currently specified to detect that.

If the (a)–(e) pattern transfers analytically — without first-deployment evidence — the verdict→router stage's grounding moves from "BUILD evidence will inform" to "inline grounding mechanism specifiable as a WP-G4 driver", partially closing OQ #14 ahead of BUILD. Methodologically this audit follows the pattern of spike (d) at DECIDE close (research log 005f), which audited the susceptibility-snapshot's structural transfer to mechanism (d).

---

## The pattern under audit's structural properties

ADR-016 mechanisms (a)–(e) bound a **read-only signal channel from L0 (Ensemble Engine outputs) to L1 (Calibration Gate dispatch decisions)**. Their structural properties:

### Pattern shape

The five mechanisms together bound a **feedback path** in which a *consumer* repeatedly reads *typed signals* from a *producer* across a layer boundary, and the consumer's output influences the producer's next emission. The risk class the pattern bounds is feedback-bias compounding (Khanal et al. universal non-improvement; CAAF prompt-engineering-artifact; Li et al. trigger-vulnerability).

### Per-mechanism structural properties

- **(a) Fresh-context isolation in the consumer.** The consumer reads each signal in a context that does not carry prior consumed signal forward. Influence on the next consumption is only through the windowed feature aggregation specified by mechanism (b) — never through context accumulation in the consumer.
- **(b) Time-decay windowing for the bias-compounding horizon.** Only signals within a bounded recent window influence the current consumer output. Older signals decay out of the computation; their influence on current decisions falls to zero outside the window. Dual-bound (wall-clock + signal-count); linear decay validated by spike (b).
- **(c) Categorical anchors via deterministic-tool-output.** Where the producer has script-model slots, deterministic outputs anchor the loop against probabilistic drift. Anchor strength scales with verifiability — binary-verifiable strongest, interpretable-prose weakest. Ensemble-composition-conditional.
- **(d) Periodic out-of-band audit dispatch.** A periodic auditor checks whether the consumer-producer feedback system is drifting. Three drift criteria (verdict skew; outcome divergence; signal-to-verdict correlation drift); typed verdict trichotomy (no drift / advisory / severe); operator action surface for advisory; fail-safe mode for severe.
- **(e) Read-only structural validation at the consumer.** The consumer validates schema and type of incoming signal data before acting on its content. Malformed signals produce a typed `malformed_signal` error and are dropped from the consumer's output computation.

### Aggregate properties

- The mechanism set is **load-bearing as a coherent whole**: (a) bounds context-accumulation; (b) bounds stale-signal-influence; (c) bounds probabilistic-consensus-drift; (d) bounds parameter drift in (b) and the consumer; (e) bounds malformed-signal contamination. Mechanism (b) is the bias-bound; (d) is the parameter-drift detector; the two compose into a self-correcting system per spike (d) §"(b)/(d) coupling".
- The pattern targets a **continuous-runtime feedback loop**, not a one-shot data flow.

---

## The target edge's analogous structure

The **Tier-Escalation Router → Calibration Gate** edge per system-design.agents.md v3.0:

- **Consumer:** Tier-Escalation Router (L2). On each `invoke_ensemble` dispatch, the Router calls `verdict_for(session_id, ensemble_name, dispatch_context) -> CalibrationVerdict`.
- **Producer:** Calibration Gate (L1). Produces a verdict in the trichotomy `Proceed | Reflect | Abstain` per ADR-014.
- **Consumer's output:** a `TierSelection` — one of `CheapTier(model_profile)` (on Proceed), `EscalatedTier(model_profile)` (on Reflect), or `Bypass(reason)` raising `LlmOrcStructuralError(error_kind="escalation_bypass")` (on Abstain).
- **Verdict-to-tier mapping:** deterministic (per fitness criterion `test_verdict_to_tier_mapping_is_deterministic`).

### Whether the edge is a feedback loop

This is the load-bearing structural question for the audit. The L1→L2 edge is a **read-only verdict consumption** — the Router does not write back into the Calibration Gate. But the *system* the edge participates in **does form a feedback loop at multi-iteration scale**:

1. Router selects tier T₁ for dispatch D₁ based on verdict V₁.
2. Dispatched ensemble at tier T₁ produces an output O₁.
3. O₁ flows back into the Calibration Gate's trajectory data (and, when ADR-016 is active, through the cross-layer signal channel from L0 to L1).
4. Calibration Gate produces verdict V₂ for the next dispatch D₂, partially determined by O₁'s trajectory features.
5. Router selects tier T₂ based on V₂.

The feedback loop runs **through the calibration-and-dispatch substrate**, not through a direct upward edge from Router to Gate. The Router's *direct* output (a tier selection) does not flow back to the Gate. But the *consequences of that tier selection* (the dispatched ensemble's behavior at the chosen tier) become the next round's verdict input.

This is the structural shape the audit must hold against the (a)–(e) pattern: the same *risk class* (feedback-bias compounding when consumer choices influence the next producer signal) applies, but the *coupling shape* is mediated through the dispatched ensemble's runtime behavior, not through a direct consumer-to-producer edge.

### The Sub-Q6 routing-reliability gap as the load-bearing risk

ADR-015 §Consequences §Neutral records the autonomous-routing evidence gap: *"multi-iteration routing reliability at the North-Star benchmark's session length is empirically unvalidated."* Operators interpreting the Router's escalation-rate calibration evidence may be reading routing-noise rather than tier-configuration mismatches until first-deployment evidence resolves Sub-Q6. This is the specific failure mode the (a)–(e) pattern would bound *if* it transfers — runtime drift in the verdict→tier mapping's interpretability.

---

## Property-by-property transfer audit

| # | Mechanism | Property tested | Transfer status | Notes |
|---|-----------|------------------|-----------------|-------|
| 1 | (a) Fresh-context isolation in the consumer | Does the Router need fresh context for each verdict consumption? | **Transfers cleanly — already structurally satisfied** | The Router's `select_tier(ensemble_name, verdict)` call is a pure function of two inputs: the ensemble's static `topaz_skill` metadata and the current verdict. The Router does not retain prior verdict state; each call is structurally context-free. The fresh-context property is satisfied **by construction** of the Router's deterministic-mapping design (FC fitness `test_verdict_to_tier_mapping_is_deterministic`). No new mechanism is needed; the pattern's property holds without architectural addition. |
| 2 | (b) Time-decay windowing for the bias-compounding horizon | Does verdict history influence current routing? If yes, windowing applies; if no, mechanism does not transfer. | **Does NOT transfer at the Router edge — but transfers one layer up at the verdict-producer** | The Router consumes one verdict per dispatch; verdict *history* is not an input to `select_tier`. Windowing on the Router's side has nothing to window. **However**, the verdict-producer (Calibration Gate) already has time-decay windowing applied to its own trajectory-feature inputs (ADR-014 §"Time-decay windowing on trajectory features", 60-min/N-dispatch window). The bias-bound for the verdict→router feedback path is *already operationalized one layer upstream*. Mechanism (b) at the Router edge would be redundant; the bias-bound exists in the producer. **Transfer status:** mechanism is achieved structurally upstream rather than at the audited edge. |
| 3 | (c) Categorical anchors via deterministic-tool-output | Does the Router consume deterministic signals? | **Does NOT transfer — wrong layer for the anchor** | The Router consumes calibration verdicts — model-derived classifications, not deterministic-tool outputs. Mechanism (c) anchors the *signal-emission* side of a feedback loop where the producer has script-model slots; the Router is on the *consumption* side and consumes verdicts whose probabilistic content is set upstream. Mechanism (c)'s anchor (when available) operates at the ensemble level inside L0, surfaces through the cross-layer signal channel (ADR-016 mechanism (c)), and reaches the Router only as part of the verdict's already-anchored content. Adding a (c)-analog at the Router edge would have no substrate to anchor against. **Transfer status:** structurally inapplicable at this edge. |
| 4 | (d) Periodic out-of-band audit dispatch | Should an auditor check routing-vs-tier-config decoupling over time? | **Transfers cleanly with novel design work — strongest single transfer** | The audit's structural shape (periodic dispatch in fresh context; reads recent signal record; produces typed verdict trichotomy; advisory/severe distinction with operator action surface; asynchronous-operator-review dynamic) maps directly to the verdict→router edge. The drift surfaces are different from ADR-016 mechanism (d) but structurally analogous: (i) **Verdict-to-tier-mapping consistency** — does the deterministic mapping produce stable distributional results, or does the *upstream verdict distribution* shift in ways that change the downstream tier mix without operator awareness? (ii) **Escalation-rate-vs-outcome correlation** — do escalated dispatches actually outperform cheap-tier dispatches on the same skill, or has the cheap-tier capability drifted (model upgrade, capability-floor regression) such that escalation no longer produces measurable gains? (iii) **Bypass-rate trend** — is the rate of `escalation_bypass` Abstain verdicts trending up over consecutive audit windows, indicating either Calibration Gate pessimism drift or routing-substrate degradation? Each criterion has the same structure as ADR-016 (d)'s drift criteria — quantitative thresholds, audit-window comparison, typed verdict output. **The (d)-analog is the load-bearing transfer for this audit.** It is also the directly-applicable mechanism for closing the Sub-Q6 gap: the audit's "escalation-rate-vs-outcome correlation" surface is exactly what distinguishes routing-noise from tier-configuration signal. |
| 5 | (e) Read-only structural validation at the consumer | Does the Router validate verdict schema before acting? | **Already satisfied — mechanism is pre-existing as part of FC-17 typed-error infrastructure** | The Router already produces a typed `escalation_bypass` error on Abstain verdicts (system-design.agents.md §Integration Contracts: Tool Dispatch → Tier-Escalation Router) and a typed `missing_skill_metadata` error when the dispatched ensemble lacks the `topaz_skill` field. Both errors derive from `LlmOrcStructuralError` per FC-17. The verdict shape (`CalibrationVerdict = Proceed \| Reflect \| Abstain`) is enumerated; any value outside the literal would fail Python's type system at the boundary. The structural validation is satisfied by the typed-error pattern from commit `9f86d0b` — the same pattern mechanism (e) elaborates at the L0→L1 edge. **Transfer status:** mechanism is achieved by inheritance from the FC-17 typed-error discipline; no new specification needed. |

---

## Transfer summary

| Mechanism | Transfer status | Action implied |
|-----------|-----------------|----------------|
| (a) Fresh-context isolation | Already satisfied by Router's stateless `select_tier` design | No new work; document the property in the Router module's fitness criteria |
| (b) Time-decay windowing | Achieved upstream at the Calibration Gate (ADR-014) — does not apply at the audited edge | No new work; the pattern's property holds via upstream mechanism |
| (c) Categorical anchors | Structurally inapplicable — Router does not consume deterministic-tool outputs | No new work; document the inapplicability |
| (d) Periodic out-of-band audit | **Clean transfer with novel design work** — the load-bearing transfer for this audit | **Specify a new periodic audit dispatch for the verdict→router edge** as a WP-G4 design driver |
| (e) Read-only structural validation | Already satisfied by FC-17 typed-error infrastructure (`escalation_bypass`, `missing_skill_metadata`) | No new work; document the property in the Router module's fitness criteria |

**Aggregate finding: partial transfer, with mechanism (d) as the load-bearing addition.**

Three of five mechanisms are achieved without new work — (a) by the Router's stateless design, (b) by upstream operationalization at the Calibration Gate, (e) by the FC-17 typed-error pattern. One mechanism (c) is structurally inapplicable. The remaining mechanism — (d) periodic out-of-band audit dispatch — transfers cleanly and is the load-bearing addition for the verdict→router edge. Mechanism (d) is also the precise mechanism that addresses the Sub-Q6 routing-reliability gap that the susceptibility snapshot's F6 finding flagged as the highest-priority concern.

The pattern's *aggregate* bias-bounding property at this edge is therefore composed of: stateless-consumer-by-construction + upstream-windowing + typed-validation + periodic-audit. Mechanism (c)'s absence is consequential only when the dispatched ensemble's tier configuration depends on script-model anchors — which is upstream territory, not Router territory.

---

## Disposition

**Disposition selected: Transfer is partial — three mechanisms satisfied without new work, one inapplicable, one transfers cleanly with novel design work.**

The disposition is closest to the cycle-status's "transfer is partial" option. The single mechanism that requires new specification — the (d)-analog audit dispatch on the verdict→router edge — is operationally significant because it closes the Sub-Q6 gap that the architect-phase susceptibility snapshot flagged as F6's highest-priority risk.

### Recommended actions for the architect-gate continuation

**1. Specify a (d)-analog audit dispatch as a WP-G4 design driver.** The Router module specification gains a periodic out-of-band audit responsibility paralleling ADR-016 mechanism (d) but operating on the verdict→router edge's drift surfaces:

- **Trigger:** every 100 verdict consumptions or every 24 wall-clock hours of session activity, whichever first; operationally tunable. Structural trigger, not LLM-mediated.
- **Drift criteria** (each at quantitative-threshold level; thresholds operationally tunable):
  - **Verdict-distribution shift** — relative frequency of `Proceed | Reflect | Abstain` verdicts changes more than ±15% between consecutive audit windows (mirrors ADR-016 mechanism (d) verdict-skew criterion).
  - **Escalation-vs-outcome correlation drift** — escalated-tier dispatches' downstream success rate diverges from cheap-tier dispatches' success rate by less than the configured tolerance over the audit window (default: escalation must produce ≥ +5 percentage points of outcome improvement to be interpretable as a tier-configuration signal rather than routing-noise). **This is the Sub-Q6 evidence surface.**
  - **Bypass-rate trend** — rate of `escalation_bypass` typed errors per dispatch trends up across consecutive audit windows beyond the configured tolerance (default: relative-rate increase exceeding +25% per window).
- **Verdict shape:** parallel-by-construction to ADR-016 mechanism (d): no drift / advisory / severe.
- **Operator action surface for advisory:** diagnostic + parameter-tuning recommendation (per spike (d) §Gap #3 option (ii)) — concretely, recommendations to adjust per-skill tier defaults (the 16-Model-Profile-slot configuration surface from ADR-015).
- **Severe-drift response:** Router enters fail-safe mode — all dispatches routed to escalated-tier regardless of verdict, until operator review. (Conservative default: prefers capability over economy under detected drift.)
- **Asynchronous-operator-review dynamic:** specified explicitly per spike (d) §Gap #4 — advisory diagnostics accumulate in operator-facing storage; severe drift triggers operator notification.

**2. Document the inherited-mechanism properties in the Router module's fitness criteria.** Add fitness entries acknowledging that mechanisms (a) and (e)'s properties hold for the Router edge by construction:

- *Fitness:* The Router consumes each verdict in a fresh stateless context; no prior verdict state is carried forward through `select_tier` — verified by `test_select_tier_is_stateless_pure_function` (unit).
- *Fitness:* All Router error surfaces (`escalation_bypass`, `missing_skill_metadata`) derive from `LlmOrcStructuralError` with the four common fields per FC-17 — verified by FC-17's existing static class-hierarchy walk.

**3. Update the L1→L2 dependency-graph entry.** The current annotation reads "Cycle 4 — new; verdict consumer". Replace with an annotation paralleling the L0→L1 edge's bounding-mechanism callout, naming the four-property composition (stateless consumer + upstream windowing + typed validation + periodic audit) and the (d)-analog audit's operator-facing surface.

**4. Propose ADR-015 amendment with partial-update header.** The (d)-analog audit responsibility is a material extension to the Router module and warrants ADR-015's first amendment. Pattern follows ADR-016's amendment of ADR-002: dated update header, Status field updated to "Updated by ADR-015a" (or similar; naming per practitioner), rejected-alternative for "no audit dispatch" added, Consequences §Positive entry for the Sub-Q6 partial-closure noted.

**5. Roadmap update for WP-G4.** Add the (d)-analog audit dispatch as a sub-WP under WP-G4, with the Sub-Q6 evidence-gap closure noted as a downstream consequence. The audit mechanism's first-deployment evidence becomes the natural validation surface for both the (d)-analog's effectiveness *and* the Sub-Q6 routing-reliability question that ADR-015's Consequences §Neutral flagged.

### What this disposition closes

- **OQ #14 partial closure for the L1→L2 verdict→router stage.** The stage moves from "BUILD evidence will inform" to "inline grounding mechanism (d-analog audit dispatch) specified as a WP-G4 driver". Four of five mechanism-analogs are achieved without new architectural work; the fifth is specified as a load-bearing addition.
- **Sub-Q6 routing-reliability evidence-gap structural closure.** The (d)-analog audit's escalation-vs-outcome correlation drift criterion is the specific surface that distinguishes routing-noise from tier-configuration signal. First-deployment evidence on this drift criterion is what Sub-Q6 needs.
- **F6 advisory carry-forward partial closure.** The architect-phase susceptibility snapshot's F6 finding (highest-priority OQ #14 carry-forward — verdict→router edge has no grounding-mechanism analog) is partially addressed. The dependency-graph annotation update plus the WP-G4 audit-dispatch specification produce the analog the F6 finding said was missing.

### What this disposition does NOT close

- **The other four under-grounded cross-layer stages** named at the decide gate (L3 cross-session artifact set; intra-L2 conversation-history boundary; orchestrator-response→tool-dispatch boundary; L1→L4 Plexus integration) remain Cycle 5+ research territory. This audit is scoped to the L1→L2 stage only.
- **First-deployment validation** of the (d)-analog audit dispatch's operational effectiveness. Like ADR-016 mechanism (d) itself, the (d)-analog inherits a "structurally validated, operationally pending" status — first-deployment evidence on the cycle's North-Star benchmark is the natural validation surface.
- **The aggregate bias-bound property** at this edge (composed of stateless-consumer + upstream-windowing + typed-validation + periodic-audit) is testable only by first-deployment evidence; the individual properties are testable analytically and at unit/integration level.

### Falsification trigger for this audit's recommendation

If WP-G4's BUILD work finds that the (d)-analog audit dispatch cannot be operationalized within the Tier-Escalation Router module's responsibility (e.g., it requires its own top-level module orthogonal to L0–L3, or it requires bidirectional coupling with the Calibration Gate that violates the read-only verdict-consumption contract), the elaboration-by-evidence framing commitment is invalidated for this audit's recommendation. The (d)-analog spec re-opens; the Router→Gate edge re-deliberates; the susceptibility snapshot's F6 advisory re-instates with first-deployment evidence as Cycle 5+ research surface. This trigger parallels ADR-016's existing falsification trigger and inherits the same elaboration-by-evidence discipline.

---

## Surprises beyond the cycle-status's pre-stated hypotheses

Three findings exceed the cycle-status's framing of the spike β shape:

**1. Mechanism (b) does not need to transfer because it is already operationalized one layer upstream.** The cycle-status hypothesis was *"if [verdict history influences routing], windowing applies; if no, mechanism does not transfer."* The reality is more interesting: verdict history *does* influence the system the verdict→router edge participates in, but the influence is mediated through the Calibration Gate's *own* time-decay windowing on its trajectory inputs (ADR-014 §"Time-decay windowing on trajectory features", explicitly the in-layer instance of ADR-016 mechanism (b)). The bias-bound the (b)-analog would provide already exists upstream. This is not a transfer failure — it is structural redundancy detection. The same pattern element appears at multiple layers and only needs to be operationalized once for its bias-bound property to hold for downstream consumers.

**2. The Router's verdict→router edge does not have a direct feedback loop, but the system it participates in does — through the dispatched ensemble's runtime behavior at the chosen tier.** The cycle-status framed the L1→L2 edge as a verdict-consumption boundary; the audit surfaced that the *feedback-bias-compounding risk* (Khanal et al. universal non-improvement) operates through the indirect loop: Router selects tier → dispatched ensemble at that tier produces output → output flows into Calibration Gate trajectory data → next verdict shifts. This indirection means mechanism (d)'s drift criteria for the Router edge must measure *outcome* divergence (escalation produces measurably better outcomes vs. cheap-tier on the same skill), not just *signal* divergence. The escalation-vs-outcome correlation criterion is the specific shape this indirection demands. ADR-016 mechanism (d) at the L0→L1 edge can use signal-to-verdict correlation drift directly; the L1→L2 (d)-analog must use a more multi-step measurement because the loop is multi-step.

**3. The (d)-analog audit naturally produces the Sub-Q6 evidence surface as a by-product.** The cycle-status framed Sub-Q6's evidence-gap as a separately-tracked routing-reliability concern flagged in ADR-015 §Consequences §Neutral. The audit surfaced that the (d)-analog's escalation-vs-outcome correlation drift criterion *is* the Sub-Q6 evidence surface — it is the specific operational measurement that distinguishes "routing is reliable; escalations track real tier-capability differences" from "routing is noise; escalation rate reflects calibration noise rather than tier-configuration signal." Closing OQ #14 partially for this stage produces a structural answer to Sub-Q6 as a coupled outcome rather than as a separate research question. The two evidence gaps are addressed by the same mechanism. This was not visible from the system-design alone — it surfaced only when the (d)-analog's drift criteria were specified concretely enough to map to an operational measurement.

A fourth observation, more about audit method than findings: the property-by-property structure inherited from spike (d) at DECIDE close worked well for an edge where transfer is *not* uniform across the five mechanisms. The audit table's value is making the asymmetric transfer status visible per-mechanism rather than burying it in prose. Spike (d) found largely-clean transfer with three specification gaps; this spike found three "already satisfied", one "structurally inapplicable", one "transfers cleanly with new work". Both are partial-transfer outcomes but at different points on the spectrum — the per-property structure surfaces this difference cleanly.

---

## Spike findings recap

- The pattern's transfer to the Tier-Escalation Router → Calibration Gate edge is **partial: three mechanisms satisfied without new work, one structurally inapplicable, one transfers cleanly with novel design work.**
- The load-bearing transfer is **mechanism (d) — periodic out-of-band audit dispatch.** The (d)-analog has three drift criteria adapted to the verdict→router edge: verdict-distribution shift; escalation-vs-outcome correlation drift; bypass-rate trend.
- The (d)-analog's escalation-vs-outcome correlation criterion is the specific operational surface that closes the Sub-Q6 routing-reliability evidence gap — these two evidence concerns are addressed by the same mechanism.
- Recommended actions for architect-gate continuation: WP-G4 design driver for the (d)-analog audit dispatch; ADR-015 amendment with partial-update header; dependency-graph annotation update for the L1→L2 edge; roadmap update with audit dispatch as a sub-WP under WP-G4; fitness criteria additions documenting the (a) and (e) properties as inherited.
- **OQ #14 partial closure for the L1→L2 verdict→router stage** is the principal architectural-decision deliverable; the four other under-grounded cross-layer stages remain Cycle 5+ research territory.
- The falsification trigger for this audit's recommendation parallels ADR-016's: if BUILD finds the (d)-analog cannot be operationalized within the Router module, the elaboration-by-evidence framing commitment is invalidated for this finding and the recommendation re-opens.

---

## Spike artifact retained per corpus retention policy

Per cycle-status §"Conformance Notes" — Spike artifacts retention (Cycle 3 directive, applies to corpus until close):

- This research log itself (analytical spike output)

No scratch directory was created — the spike is analytical and the artifact is the deliverable.
