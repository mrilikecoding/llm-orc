---
title: WP-H4 first-deployment evidence — ADR-016 cross-layer calibration channel
date: 2026-05-12
cycle: 4
phase: build
work-package: WP-H4
status: trigger-artifact-(i) per ADR-016 §"Concrete monitoring specification"
purpose: |
  Record the cross-layer channel's first BUILD-phase dispatch outcome on a
  non-trivial fixture so the practitioner can apply the conditional-
  acceptance trigger action per ADR-016. Pairs with the gate reflection
  note and the cycle-status closure entry as the WP-H4 evidence corpus.
---

# WP-H4 first-deployment evidence — ADR-016 cross-layer calibration channel

## Purpose

ADR-016 ships with **conditional acceptance** — first-deployment evidence
is the validation gate. Per ADR-016 §"Concrete monitoring specification",
the trigger artifact is either *(i) a BUILD-phase research log entry
recording the cross-layer channel's first dispatch outcome on a non-
trivial fixture*, or *(ii) a PLAY-phase field note recording the
channel's behavior on the cycle's North-Star benchmark*.

This log is the **trigger artifact (i)**. The cycle's PLAY decision is
open (deferred to BUILD close per cycle-status); whether to additionally
produce trigger artifact (ii) is a practitioner decision at the WP-H4
close gate.

## What the fixture exercises

The cycle's BUILD-phase fixture covers the integration of all five
bounding mechanisms (a)–(e) end-to-end across the L0 → channel → L1
path:

- **L0 emission** — `EnsembleExecutor` constructed with an optional
  `_calibration_signal_channel` parameter; at the end of `execute()`
  the executor calls `channel.record_signal(...)` with a typed
  `CalibrationSignal`. Mechanism (e) wrapping catches
  `MalformedSignalError` at the L0 boundary.
- **Channel buffer (mechanism (b))** — signals accumulate in the
  dual-bound time-decay window (60-min / 100-signal default per
  ADR-016 §"Mechanism (b)"; operationally tunable). The
  `_prune_window` step at every `record_signal` enforces both bounds;
  the linear-decay weighting applies at every `windowed_features`
  read.
- **L1 read (mechanism (a))** — Calibration Gate's `verdict_for` reads
  aggregated `WindowedSignalFeatures` from the channel; no signal-by-
  signal data is exposed; consecutive reads return value-equal but
  distinct-instance views (the structural enforcement of mechanism
  (a)'s fresh-context isolation).
- **Categorical anchor (mechanism (c))** — `CalibrationSignal`
  carries an optional `deterministic_anchor: bool | None` field;
  signals from script-model-slot dispatches set the field; LLM-only
  ensembles produce signals with `deterministic_anchor=None` and the
  aggregated view's `deterministic_anchor_count` is zero in that
  configuration.
- **Audit dispatch (mechanism (d))** — the channel's internal audit
  fires at every 100th verdict consumption (configurable; the
  BUILD-fixture uses smaller trigger counts to keep tests deterministic
  per the WP-G4-2 precedent). Three drift criteria evaluate
  parallel-by-construction to the `TierEscalationAuditor`:
  verdict-distribution shift, outcome-divergence, signal-to-verdict
  correlation drift. Severe drift activates `fail_safe_active`; the
  gate's `verdict_for` then defaults to Reflect.
- **Schema validation (mechanism (e))** — `MalformedSignalError`
  (FC-17's eighth and final typed-error surface; coverage now 8 of
  8) is raised at the channel boundary on schema mismatch and caught
  inside `EnsembleExecutor._emit_calibration_signal` per ADR-016
  §"Mechanism (e)" §"internal".

## Dispatch outcome (BUILD-phase fixture)

The BUILD test corpus exercises the integration along five surfaces:

| Surface | Test | Outcome |
|---|---|---|
| L0 → channel | `TestLifecycleCompositionL0ToL1::test_executor_emits_signal_then_gate_reads_features` | ✅ Signal flows L0 → channel → L1 aggregated view; success-rate aggregation reflects dispatch outcomes. |
| Channel → gate | `TestGateConsumesChannel::test_gate_records_verdict_to_channel_audit` | ✅ Verdict consumption flows from `verdict_for` into the channel's audit window. |
| Fail-safe propagation | `TestGateConsumesChannel::test_fail_safe_from_channel_forces_reflect_verdict` | ✅ Channel fail-safe → gate defaults to Reflect (verified with AUQ confidence 0.99 — Proceed-eligible under L1-internal logic). |
| Schema validation | `TestMechanismEMalformedSignalProducesTypedError::test_malformed_signal_dropped_from_verdict_computation` | ✅ Malformed signal rejected at boundary; in-window count unchanged. |
| Audit dispatch | `TestFitnessCriteria::test_audit_dispatch_fires_at_trigger_and_severe_drift_activates_fail_safe` | ✅ Audit fires at trigger; severe-drift window (100% Proceed → 100% Reflect) activates fail-safe within one dispatch cycle. |

Full test suite at WP-H4 close: **2656 passing** (+50 from 2606 baseline,
WP-G4-2 close). All linters clean (mypy strict + ruff + complexipy +
bandit + vulture).

## Falsification-trigger check

Per ADR-016 §"Falsification trigger" — *the elaboration-by-evidence
framing commitment is invalidated if BUILD or first-deployment evidence
finds that mechanism (b) windowing or mechanism (d) audit dispatch
cannot be operationalized within ADR-002's L0–L3 structure*.

**Result at WP-H4 close: the falsification trigger has NOT fired.**

- **Mechanism (b)** is implemented inside `src/llm_orc/agentic/
  calibration_signal_channel.py` at L1 (per FC-2's `_LAYER_MAP`). No
  top-level module outside L0–L3 was needed.
- **Mechanism (d)** is implemented inside the same L1 module as an
  internal audit accumulator (`_ChannelAuditWindow`). The audit
  dispatch fires from within the channel's `record_verdict_outcome`
  call — same-module, same-layer. No bidirectional coupling with the
  Calibration Gate was needed; the gate calls into the channel via
  the same single API used to read features. No top-level module
  outside L0–L3 was needed.

The elaboration-by-evidence framing commitment **holds at BUILD-phase
evidence level**. The reorganization branch remains held in reserve;
ADR-016 retains its conditional-acceptance status pending first-
deployment evidence at the cycle's North-Star benchmark (PLAY decision
or post-cycle deployment).

## Drift-criteria operational sufficiency (initial signal)

The BUILD-phase fixture's drift-criteria exercise is **structural** —
the criteria fire deterministically on synthetic verdict distributions
(100% Proceed → 100% Reflect produces verdict-distribution shift of
1.0, well past the 2× severe threshold). The question ADR-016 §
"Concrete monitoring specification" expects first-deployment evidence
to answer — *whether the drift criteria produce useful diagnostics on
real deployments where verdict distributions shift more gradually* —
remains open. PLAY-phase field-note territory if the practitioner
elects.

## Asymmetric grounding-mechanism rigor (OQ #14) — status update

OQ #14 was logged at the decide-gate close (2026-05-08, cycle-status
§DECIDE row) flagging five cross-layer stages with less grounding-
mechanism rigor than ADR-016's stage:

1. L1→L2 verdict→router stage *(partially closed at architect-gate
   close 2026-05-11 via ADR-018; spike β disposition partial transfer
   with (d)-analog as load-bearing addition — cycle-status §ARCHITECT
   Key Epistemic Response)*.
2. L3 cross-session artifact set stage *(unchanged; Cycle 5+ research
   territory)*.
3. Intra-L2 conversation-history boundary *(unchanged; Cycle 5+
   research territory)*.
4. Orchestrator-response → tool-dispatch boundary *(unchanged; Cycle
   5+ research territory)*.
5. L1→L4 Plexus integration boundary *(unchanged; Cycle 5+ research
   territory)*.

**OQ #14 partial-closure status at WP-H4 close:** the L1→L2 stage's
(d)-analog audit dispatch is implemented (WP-G4-2). The ADR-016
stage's full grounding mechanism set is now operational in code (this
WP, WP-H4). Four cross-layer stages remain Cycle 5+ research territory;
the asymmetric-rigor question for those stages does not move at this
close.

## Trigger-action surface at WP-H4 close

Per ADR-016 §"Concrete monitoring specification" §"Trigger action" the
practitioner reviews the trigger artifact and decides between three
courses:

- **(a) Conditionality moves to full acceptance** — both mechanisms
  operationalized successfully.
- **(b) Conditionality preserved with deployment-tuning notes** —
  operationalization works but requires parameter adjustments not yet
  documented.
- **(c) Falsification trigger fires** — a superseding ADR is filed;
  the reorganization branch is re-opened.

Option (c) is **closed by BUILD-phase evidence.** The falsification
trigger — as specifically written in ADR-016 §"Falsification trigger"
— names structural-operationalization failure as its criterion: *if
any validation pathway produces evidence that mechanism (b) or (d)
cannot be operationalized within the Calibration Gate's existing or
extended class structure without introducing a new top-level module*.
Both mechanisms operationalized inside L1 of ADR-002's four-layer
structure. The trigger did not fire. Option (c) is structurally
foreclosed at this gate.

The choice between (a) and (b) is a practitioner judgment that the
BUILD-phase evidence informs but does not resolve. Both readings are
defensible; the two-question test below names the load-bearing
distinction.

### Two-question test for (a) vs. (b)

The practitioner should form an independent judgment via this
two-question test before adopting either disposition:

**Q1.** Does ADR-016's falsification trigger — as specifically written
— name *operational-validation failure* or *structural-operationalization
failure* as its criterion?

- If **structural-operationalization**, the trigger's non-firing IS
  the conditionality's primary criterion. Option (a) full acceptance
  may be earned at this gate; operational validation becomes a natural
  post-deployment learning surface that the ADR-016 §"Sweep
  responsibility" clause already provides for.
- If **operational-validation**, the trigger's non-firing is necessary
  but not sufficient. Option (b) preserved-conditional is the
  appropriate disposition until first-deployment evidence at the
  cycle's North-Star benchmark resolves the operational question.

**Q2.** Is the remaining operational-validation question (drift
diagnostics produce useful action on real deployments; operator
workflow handles advisory diagnostics productively; (b)/(d) coupling
tunes parameters under deployment dynamics) a *condition of the
ADR-016 amendment's acceptance*, or a *natural post-deployment learning
surface regardless of acceptance status*?

- If **condition of acceptance**, option (b) preserved-conditional is
  the appropriate disposition; the ADR-016 amendment is not yet fully
  accepted.
- If **post-deployment learning surface**, option (a) is the appropriate
  disposition; the ADR-016 amendment is accepted; the operational
  questions are tracked as Cycle 5+ research / observation territory
  not as conditional-acceptance status.

### The case for option (a) — full acceptance at this gate

ADR-016 §"Falsification trigger" was written with structural-
operationalization as its specific concrete criterion. The trigger
did not fire. FC-17 typed-error coverage closed at 8 of 8. All five
bounding mechanisms operationalize inside L1. The static FC-2
layering check accepts the single pre-declared upward edge and
rejects all others. The conditional-acceptance status's gate
criterion is structurally satisfied.

The operational validation questions (drift diagnostics on real
deployments; operator workflow; (b)/(d) coupling under deployment
dynamics) are real and important — but they are the *next learning
surface*, not the *conditionality's gate criterion*. ADR-016's
sweep-responsibility clause already requires a cycle-status row for
any future cycle that exercises the channel; that mechanism handles
ongoing operational learning regardless of conditional-vs-full
acceptance status.

### The case for option (b) — preserved-conditional

ADR-016's conditional-acceptance status was written to track *all*
the open evidence surfaces, not only structural operationalization.
The practitioner-guidance recorded at decide-gate (*"I'd rather be
explicit about potential decisions that need more support. What I
don't want is to codify an unsupported assumption without evidence.
Rather we should do spikes or experiments to validate our hunches"*,
ADR-016 §"Practitioner guidance on unsupported assumptions") frames
the conditionality broadly: it captures the gap between *structural
validation* and *operational sufficiency*, not only structural
failure.

Preserved-conditional acknowledges that mechanism (b)'s windowing
parameters were operationally tuned by Spike (b) under synthetic-data
conditions, not real-deployment conditions. Mechanism (d)'s drift
criteria thresholds were drafting-time synthesis. Until first-
deployment evidence resolves whether these tunings hold, preserved-
conditional reflects the actual evidence state more accurately than
full acceptance.

### The case is genuinely open

Both readings are defensible. This research log does not pre-frame
the practitioner's choice. The trigger-action review is a practitioner
judgment about how strictly to read ADR-016's conditional-acceptance
language — broadly (everything operational) or narrowly (just the
falsification trigger).

If the practitioner reads broadly: option (b) is earned; the
preserved-conditional status carries to Cycle 5+ with explicit
operational-evidence sweep at every cycle that exercises the channel.

If the practitioner reads narrowly (per ADR-016 §"Falsification
trigger" as the specific gate criterion): option (a) is earned; the
operational questions become Cycle 5+ research territory tracked via
the §"Sweep responsibility" clause but not via conditional-acceptance
status.

The practitioner's disposition is recorded at the WP-H4 close gate
in the gate reflection note and the cycle-status §BUILD row.

## Implementation provenance

| Artifact | Path | Cycle role |
|---|---|---|
| Channel module (L1) | `src/llm_orc/agentic/calibration_signal_channel.py` | WP-H4 new |
| `MalformedSignalError` (FC-17 8 of 8) | same file | WP-H4 new |
| L0 emission wiring | `src/llm_orc/core/execution/ensemble_execution.py` | WP-H4 modification (3 changes: import, constructor, `_emit_calibration_signal`) |
| Gate channel consumption | `src/llm_orc/agentic/calibration_gate.py` | WP-H4 modification (3 changes: TYPE_CHECKING import, constructor param, `verdict_for` channel-read + audit-feedback) |
| Channel unit tests | `tests/unit/agentic/test_calibration_signal_channel.py` | WP-H4 new (44 tests covering 11 scenarios + 4 fitness criteria + constructor/threshold validation + lifecycle composition) |
| `MalformedSignalError` typed-error tests | `tests/unit/models/test_structural_errors.py` | WP-H4 extension (`TestMalformedSignalErrorAsConcreteSubclass`, 6 tests) |
| FC-2 layer map pre-declaration | `tests/unit/agentic/test_fc2_layering.py` | WP-B4 pre-declared; WP-H4 activated (the upward edge tuple `(ensemble_execution, calibration_signal_channel)` was already in `_ALLOWED_UPWARD_EDGES`) |

## What this log does NOT establish

Per ADR-016 §"Mechanism (b)" §"Why structural validation matters but
is not the whole story" — *whether the bias-compounding bound holds
operationally* under real-deployment conditions is the load-bearing
question that pre-BUILD spikes and BUILD-phase fixtures cannot answer.
Mechanism (b) without empirical operational validation is *logically
sound*; mechanism (b) with first-deployment evidence is *empirically
grounded*. This log establishes the former; the latter remains open.

Same caveat for mechanism (d) per ADR-016 §"Mechanism (d)" §"Validation
status" — structural transfer is validated (spike (d), research log
`005f-`); operational sufficiency requires deployment evidence.
