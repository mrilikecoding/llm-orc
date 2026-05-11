# ADR-018: Tier-Escalation Router Periodic Audit Dispatch (ADR-016 Mechanism (d) Analog)

**Status:** Accepted (amends ADR-015)

**Date:** 2026-05-11

---

## Context

ADR-015 specifies the per-role tier-escalation router as a Tool Dispatch (L2) interposition consuming Calibration Gate (L1) verdicts. The verdict→router edge (L1→L2) is one of five cross-layer stages OQ #14 (domain-model Open Question #14, recorded at the decide gate 2026-05-08) flagged as carrying less grounding-mechanism rigor than ADR-016's L0→L1 channel. Practitioner direction at decide-gate close was to defer all five stages to BUILD evidence.

Spike β at the architect-gate continuation (research log `005h-spike-bounding-mechanism-transfer-l1-l2.md`, 2026-05-11) audited whether ADR-016's bounding mechanisms (a)–(e) transfer analytically to the verdict→router edge. The audit's finding: partial transfer — three mechanisms are achieved without new architectural work ((a) fresh-context isolation by Router's stateless `select_tier` design; (b) time-decay windowing operationalized one layer upstream at the Calibration Gate per ADR-014; (e) read-only structural validation by the FC-17 typed-error infrastructure), one is structurally inapplicable ((c) categorical anchors — wrong layer for the anchor substrate), and one transfers cleanly with novel design work — **mechanism (d) periodic out-of-band audit dispatch**.

The architect-phase susceptibility snapshot's F6 finding (2026-05-11) named the verdict→router edge as the highest-priority OQ #14 carry-forward because Sub-Q6's routing-reliability evidence gap (ADR-015 §Consequences §Neutral) operates exactly at this edge: if the verdict is unreliable, the Router's escalation decisions are noise, and no grounding mechanism is currently specified to detect that. Spike β surfaced that the (d)-analog's *escalation-vs-outcome correlation* drift criterion is the specific operational measurement that distinguishes routing-noise from tier-configuration signal — closing OQ #14 (partially, for this stage) and Sub-Q6 (structurally) with the same mechanism.

This ADR records the (d)-analog audit dispatch as an amendment to ADR-015's Tier-Escalation Router responsibilities. It does not change ADR-015's primary-skill framing, the verdict-to-tier mapping, or the rejected alternatives — those remain unchanged. It adds a new periodic out-of-band audit dispatch responsibility to the Router module, paralleling ADR-016 mechanism (d) but operating on the verdict→router edge's drift surfaces.

---

## Decision

The Tier-Escalation Router module (L2 per ADR-015) gains a **periodic out-of-band audit dispatch responsibility** structurally analogous to ADR-016 mechanism (d), operating on the L1→L2 verdict→router edge.

### Trigger

Every 100 verdict consumptions or every 24 wall-clock hours of session activity, whichever comes first; operationally tunable via `OrchestratorConfig`. Structural trigger (count + wall-clock), not LLM-mediated.

### Drift criteria

Each at quantitative-threshold level; thresholds operationally tunable via `OrchestratorConfig`:

- **Verdict-distribution shift.** Relative frequency of `Proceed | Reflect | Abstain` verdicts changes by more than ±15% between consecutive audit windows. Mirrors ADR-016 mechanism (d)'s verdict-skew criterion.
- **Escalation-vs-outcome correlation drift.** Escalated-tier dispatches' downstream success rate diverges from cheap-tier dispatches' success rate by less than the configured tolerance over the audit window (default: escalation must produce at least +5 percentage points of outcome improvement to be interpretable as a tier-configuration signal rather than routing-noise). **This is the Sub-Q6 evidence surface — the criterion that distinguishes routing-noise from tier-configuration signal.**
- **Bypass-rate trend.** Rate of `escalation_bypass` typed errors per dispatch trends up across consecutive audit windows beyond the configured tolerance (default: relative-rate increase exceeding +25% per window).

### Verdict shape

Parallel-by-construction to ADR-016 mechanism (d): `no drift | advisory | severe`.

### Operator action surface for advisory drift

Diagnostic plus parameter-tuning recommendation — concretely, recommendations to adjust per-skill tier defaults (the 16-Model-Profile-slot configuration surface from ADR-015). Diagnostics accumulate in operator-facing storage (asynchronous-operator-review dynamic per ADR-016 mechanism (d)'s precedent).

### Severe-drift response

Router enters fail-safe mode: all dispatches routed to escalated-tier regardless of verdict, until operator review. Conservative default: prefers capability over economy under detected drift. Operator notification fires at severe-drift detection.

### Asynchronous-operator-review dynamic

Specified explicitly per spike (d) §Gap #4 precedent — advisory diagnostics do not block dispatch flow; they accumulate for operator review at the next session boundary or via explicit operator-initiated audit-log review.

### Inherited mechanism properties (no new work)

ADR-018 records that the four other ADR-016 mechanisms hold for the verdict→router edge by inheritance, not by addition:

- **(a) Fresh-context isolation:** the Router's `select_tier(ensemble_name, verdict)` is a stateless pure function of two inputs; no prior verdict state is carried forward through the call. Property holds by construction of ADR-015's deterministic-mapping design.
- **(b) Time-decay windowing:** operationalized one layer upstream at the Calibration Gate (ADR-014 §"Time-decay windowing on trajectory features", 60-min/N-dispatch window). The bias-bound for the verdict→router feedback path exists in the producer; no Router-side windowing is needed.
- **(c) Categorical anchors via deterministic-tool-output:** structurally inapplicable at this edge — the Router consumes verdicts (model-derived classifications), not deterministic-tool outputs. Mechanism (c)'s anchor (when available) operates at the ensemble level inside L0 and surfaces through ADR-016's signal channel; the Router consumes its already-anchored content.
- **(e) Read-only structural validation:** satisfied by FC-17 typed-error infrastructure — `escalation_bypass` and `missing_skill_metadata` typed errors derive from `LlmOrcStructuralError` per FC-17.

### What this does NOT change in ADR-015

- **Primary-skill framing stands.** Spike α (research log `005g-spike-topaz-skill-classification.md`, 2026-05-11) confirmed that 21 of 21 classified library ensembles satisfy the clean-primary criterion. The per-skill tier router design is well-founded for the actual library; rejected alternative §(b) (per-ensemble overrides) remains unwarranted.
- **Verdict-to-tier mapping stands.** Proceed → cheap-tier; Reflect → escalated-tier; Abstain → `escalation_bypass` typed error. Unchanged.
- **The 16-Model-Profile-slot configuration surface stands.** Operators retain the per-skill defaults per ADR-015's friction-trades-for-discovery posture.

---

## Rejected alternatives

**(a) No audit dispatch (deferred to BUILD evidence per decide-gate disposition).** Rejected: Spike β surfaced that the (d)-analog transfers cleanly with novel design work and is the specific mechanism that closes Sub-Q6's routing-reliability evidence gap. Deferring entirely would leave the L1→L2 verdict→router stage without any inline grounding mechanism, perpetuating the asymmetric-rigor concern OQ #14 records. The audit dispatch's structural shape inherits from ADR-016 mechanism (d)'s precedent; first-deployment evidence on the audit's effectiveness is the remaining validation gate.

**(b) Run audit on the Calibration Gate side rather than the Router side.** Rejected: the audit consumes verdict-to-outcome correlation data which is naturally available at the Router's interposition point (the Router observes both the verdict it consumed and the dispatched ensemble's outcome). Placing the audit on the Calibration Gate side would require the Gate to receive outcome data from L2 — which would create an upward edge from L2 to L1 outside ADR-016's narrow exception. The Router's L2 placement keeps the audit within the existing dependency-graph rules.

**(c) Synchronous severe-drift response (block dispatches until operator clears).** Rejected: would couple operator availability to dispatch availability. The fail-safe mode (route-all-to-escalated until operator review) preserves dispatch availability while degrading to maximum-capability behavior under detected drift — an asynchronous-tolerable failure shape.

**(d) Audit dispatched ensemble's success rate without the cheap-vs-escalated comparison.** Rejected: the load-bearing question for routing-noise detection is whether escalation produces measurably better outcomes than cheap-tier on the same skill. A success-rate-only audit cannot distinguish "routing is reliable; escalation tracks tier-capability differences" from "routing is noise; escalation rate reflects calibration noise." The escalation-vs-outcome correlation criterion is the specific operational measurement that disambiguates.

---

## Consequences

**Positive:**
- Closes OQ #14 partially for the L1→L2 verdict→router stage — the stage moves from "BUILD evidence will inform" to "inline grounding mechanism specified as a WP-G4 driver"
- Closes Sub-Q6 (routing-reliability evidence gap, ADR-015 §Consequences §Neutral) structurally: the (d)-analog's escalation-vs-outcome correlation criterion is the operational surface that distinguishes routing-noise from tier-configuration signal
- Inherits four of ADR-016 mechanism (d)'s structural properties (trigger pattern, verdict shape, asynchronous-operator-review dynamic, fail-safe mode) — reduces drafting-time synthesis to the three drift criteria specific to this edge
- Composes with existing FC-17 typed-error infrastructure — `escalation_bypass` and `missing_skill_metadata` errors are already in scope for the audit's validation surface

**Negative:**
- Adds a periodic background dispatch to the Tier-Escalation Router module — the module's responsibility footprint grows beyond ADR-015's original verdict→tier mapping
- Audit-dispatch effectiveness inherits the same first-deployment-evidence dependency as ADR-016 mechanism (d) itself — the (d)-analog is structurally validated by Spike β's analytical audit but operationally pending until BUILD work and first-deployment evidence
- Operator-facing surface grows: per-skill tier defaults (16 slots from ADR-015) plus audit-dispatch trigger thresholds (default values; operationally tunable) plus drift criteria thresholds (default values; operationally tunable). The increase in surface area is bounded but non-zero

**Neutral:**
- The audit dispatch operates inside the Tier-Escalation Router module per ADR-015's Tool Dispatch (L2) placement — no new top-level module orthogonal to L0–L3 is introduced. The elaboration-by-evidence framing commitment from research-gate Grounding Action 2 holds for ADR-018
- Composition with ADR-014 is unchanged: the Router still consumes the verdict trichotomy via `verdict_for(session_id, ensemble_name, dispatch_context)`. The audit reads outcome data from the dispatched ensemble's runtime behavior (already available at the Router's interposition point), not new data from the Calibration Gate
- The four other under-grounded cross-layer stages OQ #14 names (L3 cross-session artifact set per ADR-013; intra-L2 conversation-history boundary per ADR-012; orchestrator-response→tool-dispatch boundary per ADR-017; L1→L4 Plexus integration) remain Cycle 5+ research territory. ADR-018 closes OQ #14 partially for the L1→L2 stage only

---

## Falsification trigger

If WP-G4's BUILD work finds that the (d)-analog audit dispatch cannot be operationalized within the Tier-Escalation Router module's responsibility (e.g., it requires its own top-level module orthogonal to L0–L3, or it requires bidirectional coupling with the Calibration Gate that violates the read-only verdict-consumption contract), the elaboration-by-evidence framing commitment is invalidated for ADR-018. The (d)-analog spec re-opens; the Router→Gate edge re-deliberates; the susceptibility snapshot's F6 advisory re-instates with first-deployment evidence as Cycle 5+ research surface. This trigger parallels ADR-016's existing falsification trigger and inherits the same elaboration-by-evidence discipline.

---

## Provenance check

- **Driver-derived content (audit-dispatch structural shape).** ADR-016 mechanism (d) is the structural precedent — periodic dispatch in fresh context, typed verdict trichotomy, advisory/severe distinction with operator action surface, asynchronous-operator-review dynamic. Spike (d) at decide close (research log `005f-spike-adr016-d-structural-transfer-audit.md`, 2026-05-06) established the pattern's structural properties; Spike β at architect-gate continuation (research log `005h-spike-bounding-mechanism-transfer-l1-l2.md`, 2026-05-11) audited the pattern's transfer to the verdict→router edge.

- **Driver-derived content (drift criteria).** The three drift criteria are adapted from ADR-016 mechanism (d)'s drift criteria with edge-specific substitutions: verdict-distribution shift (mirrors verdict-skew); escalation-vs-outcome correlation drift (the multi-step measurement Spike β surfaced as load-bearing for the indirect feedback loop the verdict→router edge participates in); bypass-rate trend (the drift surface specific to the typed `escalation_bypass` error pattern from ADR-015).

- **Drafting-time synthesis (escalation-vs-outcome correlation as Sub-Q6 evidence surface).** Spike β surfaced the coupling between OQ #14 partial closure and Sub-Q6 structural closure: the same criterion addresses both. The framing of the criterion as the specific operational measurement that distinguishes routing-noise from tier-configuration signal is drafting-time synthesis composing Spike β's analytical finding with ADR-015's §Consequences §Neutral Sub-Q6 entry.

- **Drafting-time synthesis (severe-drift fail-safe mode default).** Routing all dispatches to escalated-tier under severe drift is drafting-time judgment. Alternatives — block dispatches synchronously; route to cheap-tier (preserve economy under drift); raise on every drift detection — were rejected for availability and conservatism reasons documented above.

- **Inherited properties (FC-17 typed-error discipline + ADR-014 upstream windowing + Router stateless design).** Three of ADR-016's bounding mechanisms hold for the verdict→router edge by inheritance from existing infrastructure; ADR-018 records the inheritance explicitly rather than re-specifying.

- **Vocabulary impact.** ADR-018 does not introduce new domain vocabulary. The (d)-analog audit's structural shape uses the existing ADR-016 mechanism (d) vocabulary. The "escalation-vs-outcome correlation" phrase is descriptive (a drift criterion's name), not a domain concept warranting glossary promotion.

- **Practitioner direction.** Spike β was approved by the practitioner on 2026-05-11 as one of two pre-BUILD spikes (the other being Spike α). Spike outcomes were integrated at the architect-gate continuation by practitioner-approved disposition (full-integration scope, all 7 actions). ADR-018 records the spike-derived amendment to ADR-015 with the partial-update header pattern from ADR-002/ADR-016 precedent.
