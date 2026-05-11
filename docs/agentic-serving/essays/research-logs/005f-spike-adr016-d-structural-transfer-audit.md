# Research Log 005f — Spike on ADR-016 Bounding Mechanism (d): Structural Transfer Audit

**Cycle:** 4
**Phase:** DECIDE (Tranche-C ADR-016 validation, per practitioner Path-2 authorization 2026-05-06)
**Date:** 2026-05-06
**Type:** Analytical spike — structural transfer audit on the susceptibility-snapshot pattern
**Mechanism under test:** ADR-016 bounding mechanism (d), periodic out-of-band audit dispatch for architectural calibration drift

---

## Spike question

ADR-016 mechanism (d) proposes applying the susceptibility-snapshot pattern from RDD methodology to architectural calibration drift detection. Essay 005 names the transfer at the structural level: same dispatch-fresh-context-evaluator-produces-typed-verdict pattern, with different content (calibration data instead of research-framing data).

The question: do the susceptibility-snapshot pattern's structural properties — what makes it work in RDD methodology — transfer cleanly to architectural calibration drift detection? Where the transfer is clean, mechanism (d) inherits the established pattern's properties. Where the transfer is *not* clean, mechanism (d) requires drafting-time synthesis to fill in the gaps; argument-audit must verify the synthesis is rigorous.

This is an analytical audit, not an empirical one. The audit specifies both patterns' structural properties and identifies transfer status property-by-property.

---

## The susceptibility-snapshot pattern's structural properties (as established in RDD methodology)

The pattern is defined across rdd plugin ADRs 058, 065, 067, and the orchestrator skill text (susceptibility snapshot dispatch protocol). Its properties:

### Trigger
- **Pre-specified structural trigger.** Fires at phase boundaries (research → discover, discover → model, model → decide, etc.). The trigger is structural (phase position in the pipeline), not LLM-mediated.
- **Periodicity.** Fires once per phase boundary. The trigger is non-overlapping; one snapshot completes before the next is dispatched.

### Context
- **Fresh-context isolation (Architectural Isolation per ADR-058).** The evaluator runs in a fresh context separate from the agent's main conversation. This is the load-bearing property — it exploits the Self-Correction Blind Spot, where a fresh context treats prior output as external input and reactivates dormant correction.

### Inputs
- **Phase artifacts** (essay, ADRs, gate exchange transcripts) — concrete, structurally-typed inputs.
- **AID signals** (engagement signals + susceptibility signals from the conversation) — recorded during the phase, dispatched alongside artifacts.

### Logic
- **Established evaluation criteria.** Content-selection sycophancy signals: assertion density, framing adoption, confidence markers, declining alternative engagement, architectural-verdict signals. The criteria are well-specified — an evaluator agent has concrete signals to test against.
- **Single-evaluator dispatch.** One evaluator runs in the isolated context; produces one verdict.

### Output
- **Typed verdict shape.** Three classes: clean / advisory finding / Grounding Reframe trigger.
- **Structured findings.** Specific findings name concrete artifacts, decisions, or commitments (the "specific + actionable + in-cycle applicable" three-property test for Grounding Reframe trigger per ADR-068).

### Consumer
- **Multi-level consumer logic.** The agent's gate-handling logic reads the verdict.
  - **Clean:** no action; phase advances.
  - **Advisory finding:** record as feed-forward signal in cycle-status; phase advances.
  - **Grounding Reframe trigger:** present the reframe to the user; user decides whether to pursue grounding actions or proceed; the decision is recorded visibly.
- **User-in-the-loop dynamic.** The user is present and active during gate exchanges; the consumer's actions interact with user choices.

### Effectiveness conditions
- **Well-defined evaluation surface.** Research framing has a well-defined surface (the artifacts produced are concrete; sycophancy signals are concrete).
- **Repeatable across boundaries.** Each phase boundary fires the same dispatch shape; the pattern's properties accumulate evidence over multiple firings.
- **Verifiable outcomes.** The methodology's reasoning improves with snapshots (per the rdd plugin's own evolution evidence).

---

## The analogous structure for architectural calibration drift detection (ADR-016 mechanism (d))

ADR-016's drafted specification:

### Trigger
- **Pre-specified structural trigger.** Fires every 100 calibration verdicts OR every 24 wall-clock hours of session activity, whichever comes first. The trigger is structural (count-based or time-based), not LLM-mediated.
- **Periodicity.** Fires periodically. Non-overlapping (one audit completes before the next is dispatched).

### Context
- **Fresh-context isolation (consistent with mechanism (a) — cited as the architectural-isolation pattern).**

### Inputs
- **Time-windowed signal record** (the signals that flowed through mechanism (b)'s window during the audit interval).
- **Corresponding calibration verdicts** (the verdicts ADR-014's Calibration Gate produced from those signals).

### Logic
- **Drafted evaluation criteria.** Audits the calibration system's behavior over the audit window for systematic patterns: verdict skew (distributional shift in verdict trichotomy), outcome divergence (verdicts predict outcomes worse over time), signal-to-verdict correlation drift (relationship between inputs and outputs changes).
- **Single-evaluator dispatch.**

### Output
- **Typed verdict shape.** Three classes: no drift / drift detected / severe drift.
- **Structured findings.** Operator-readable diagnostic on drift detection.

### Consumer
- **Drafted consumer logic.**
  - **No drift:** no action; calibration system continues operating.
  - **Drift detected:** operator-readable diagnostic produced; calibration system continues operating with diagnostic logged.
  - **Severe drift:** calibration system enters a fail-safe mode (calibration verdicts default to Reflect-or-Abstain until operator review); operator notified.
- **Operator action surface.** Operator reads diagnostic and acts.

### Effectiveness conditions
- **Whether the calibration system's behavior has a well-defined drift surface.**
- **Whether the audit's findings produce actionable operator response.**
- **Whether the drift criteria can be operationalized with codebase-implementable evaluators.**

---

## Property-by-property transfer audit

| # | Property | Transfer status | Notes |
|---|----------|-----------------|-------|
| 1 | Pre-specified structural trigger | **Transfers cleanly** | The count-based / time-based trigger is structurally analogous to the phase-boundary trigger. The shape is identical. |
| 2 | Periodicity (non-overlapping) | **Transfers cleanly** | The periodic dispatch property is preserved. |
| 3 | Fresh-context isolation | **Transfers cleanly** | Mechanism (a) provides the isolation; it is the same architectural-isolation pattern the methodology uses. ADR-058's load-bearing property is preserved. |
| 4 | Typed verdict shape (multi-level) | **Transfers cleanly** | The three-class trichotomy (no drift / drift detected / severe drift) is parallel-by-construction to (clean / advisory finding / Grounding Reframe trigger). |
| 5 | Structured inputs | **Transfers with specification work** | Time-windowed signal record + calibration verdicts are structurally typed (per ADR-014's data shape). The input shape is concrete; the *content* differs from research-framing artifacts but the structural property is preserved. |
| 6 | Evaluation criteria specificity | **Transfers with significant specification work** | The susceptibility-snapshot's content-selection-sycophancy criteria are well-developed; the calibration-drift criteria (verdict skew, outcome divergence, signal-to-verdict correlation drift) are named at the right level but **not specified concretely enough for a codebase evaluator to apply uniformly**. See gap #1 below. |
| 7 | Single-evaluator dispatch | **Transfers cleanly** | Same dispatch shape. |
| 8 | Structured findings ("specific + actionable + in-cycle applicable") | **Transfers with question-mark on "in-cycle applicable"** | Susceptibility-snapshot's three-property test (ADR-068) is methodology-specific. Architectural calibration audit's analogous three-property test would be: specific (names concrete signals/verdicts) + actionable (specific operator action implied) + **operationally applicable (the deployed calibration system can act on it during operation, not only during retrospective review)**. The in-cycle vs operationally-applicable distinction may be material. See gap #2 below. |
| 9 | Multi-level consumer logic | **Transfers with specification work on operator action surface** | The clean / advisory / severe levels parallel cleanly. The *consumer actions* per level are partially specified (severe = fail-safe mode; advisory = log diagnostic; clean = no action) but the operator workflow for "drift detected" requires fuller specification. See gap #3 below. |
| 10 | User-in-the-loop dynamic | **Does NOT transfer cleanly** | Susceptibility-snapshot's user is present and active during gate exchanges; operator's relationship to the running calibration system is different — the operator may be absent during real-time drift detection. Asynchronous-review-by-operator is a meaningfully different consumer dynamic. See gap #4 below. |
| 11 | Effectiveness verifiability | **Transfers with question-mark** | Susceptibility-snapshot's effectiveness is assessed via methodology-outcomes (does reasoning improve?). Calibration-audit's effectiveness assessment requires measuring whether drift detection leads to calibration system improvement — which is a multi-step empirical chain (audit fires → diagnostic produced → operator acts → calibration improves). See gap #5 below. |
| 12 | Repeatability across boundaries | **Transfers cleanly** | The pattern's accumulating-evidence property is preserved. |

---

## Gaps surfaced by the audit

### Gap #1 — Evaluation criteria specificity

ADR-016's drafted criteria for drift (verdict skew, outcome divergence, signal-to-verdict correlation drift) are named at the right level of abstraction but lack concrete specification. The susceptibility-snapshot's criteria — "did the user's declarative conclusions increase while questions decreased?", "did alternatives drop away without examination?", "did the agent adopt the user's framing without surfacing alternatives?" — are operationalized at a level a codebase evaluator can apply.

The calibration-drift parallel needs:

- **Concrete drift detection thresholds.** What count of verdicts within an audit window constitutes "verdict skew"? At what magnitude? The drafted criterion is qualitative; operationalization requires quantitative thresholds.
- **Concrete signal-to-verdict correlation tests.** What statistical test? What baseline correlation to compare against?
- **Concrete outcome divergence measurement.** What outcome signal? Calibration verdicts' predictiveness over what window?

**Proposed action for ADR-016:** specify the drift-detection criteria at quantitative-threshold level. The thresholds themselves are operationally tunable (parallel to mechanism (b)'s window parameters), but the *criterion shape* must be concrete enough that BUILD can implement an evaluator.

### Gap #2 — "In-cycle applicable" → "operationally applicable" reframe

ADR-068's three-property test for Grounding Reframe trigger requires findings to be "in-cycle applicable" — meaning the action can be taken at the current phase boundary, not only downstream. The methodology gate boundary creates the moment where an in-cycle action is possible.

The architectural calibration audit fires periodically during a deployed system's runtime, not at a methodology phase boundary. There is no equivalent "in-cycle" moment. The analogous property is **operationally applicable** — meaning the deployed calibration system can act on the finding during operation, not only during retrospective review.

**Proposed action for ADR-016:** the three-property test for "drift detected" trigger (the level analogous to Grounding Reframe trigger) becomes specific + actionable + operationally applicable. Operationally applicable means the calibration system's automated response (or operator's near-term review) can act before the next audit fires.

### Gap #3 — Operator action surface for "drift detected"

ADR-016 specifies that "drift detected" produces an operator-readable diagnostic with the calibration system continuing to operate. What the operator does with the diagnostic is unspecified. The susceptibility-snapshot's analogue (advisory finding) feeds into cycle-status as feed-forward signal — there is a defined consumer (subsequent phase reads the cycle-status). The calibration-audit advisory has no defined consumer.

**Proposed action for ADR-016:** specify the operator action surface concretely. Three options:

- **(i) Diagnostic logged to operator-facing surface only** (current drafted behavior). Operator reviews logs at their cadence; no specific action expected. Lowest friction; risk that diagnostics accumulate unread.
- **(ii) Diagnostic logged + parameter-tuning recommendation.** The audit produces not just "drift detected" but specific tuning recommendations (e.g., "window size 100 → suggested 50 based on verdict-tracking-error trend"). Operator action is "review and apply or override."
- **(iii) Diagnostic + automated parameter-adjustment proposal.** The audit produces a proposed parameter change; the operator approves/rejects; on approval, the calibration system applies the change at the next session boundary. Highest friction; closes the loop on drift correction.

The choice depends on operator expectations and the deployment shape. Drafting-time judgment suggests (ii) — diagnostic plus recommendation, with operator action being approval rather than authoring. (i) is too passive; (iii) introduces more architectural complexity than the cycle's evidence justifies.

### Gap #4 — Asynchronous-operator-review dynamic (the key non-transferring property)

The susceptibility-snapshot's effectiveness depends on the user being present at the gate. The Grounding Reframe action surface is real-time conversation; the user can immediately decide whether to pursue grounding.

The architectural calibration audit fires during runtime. The operator may be absent or may discover the audit log hours/days later. The "operator action" is asynchronous review.

This is a real difference. Three implications:

- **The advisory/severe distinction matters more.** Severe drift cannot wait for asynchronous review — fail-safe mode is the right response (calibration system enters a defensive state where its verdicts default to Reflect-or-Abstain). Advisory drift can wait, because asynchronous review is appropriate for it.
- **The audit's failure mode under operator absence must be specified.** What if the operator never reviews the diagnostic? ADR-016 should specify that the calibration system continues operating in advisory-drift mode without operator intervention; the diagnostic accumulates in operator-facing storage; if drift escalates from advisory to severe, the system enters fail-safe automatically.
- **Operator notification mechanisms become load-bearing.** Asynchronous review depends on the operator being notified at all. The operator-readable surface must include a notification mechanism (e.g., log file with timestamps; optional email/webhook for severe drift).

**Proposed action for ADR-016:** specify the asynchronous-operator-review dynamic explicitly in the consumer section. Differentiate advisory-drift handling (passive, log-based, operator reviews at their cadence) from severe-drift handling (active, fail-safe-mode-trigger, operator-notified). Document that mechanism (d)'s effectiveness depends on operator presence in a way the susceptibility-snapshot's does not — and the operator-notification mechanism is part of the design.

### Gap #5 — Effectiveness verifiability

Susceptibility-snapshot's effectiveness is assessable via methodology outcomes; the rdd plugin's own evolution provides multi-cycle evidence. Calibration-audit's effectiveness is harder to assess — the multi-step chain (audit fires → diagnostic produced → operator acts → calibration improves) has multiple failure surfaces.

This gap does not block ADR-016 mechanism (d). The methodology has a precedent: introducing the mechanism, deploying it, and measuring effectiveness over time. ADR-016's first-deployment evidence on the cycle's North-Star benchmark is the natural validation surface.

**Proposed action for ADR-016:** acknowledge the effectiveness-verifiability gap honestly in the empirical validation pathway. The mechanism's effectiveness is provisional pending real-deployment evidence over multiple audit cycles.

---

## Implications for ADR-016

### Mechanism (d) status: provisionally validated at the structural-transfer level, with three identified specification gaps

The structural transfer is largely clean. Properties 1–5, 7, 12 transfer without modification. Properties 6, 8, 9 transfer with specification work. Properties 10, 11 surface real differences (gap #4 most consequentially).

**The conditional-acceptance status remains appropriate** — the structural transfer is established, but the specification gaps require ADR-016 revision before the mechanism is operationally implementable.

### Three specific revisions to ADR-016 mechanism (d)

1. **Specify drift-detection criteria at quantitative-threshold level** (gap #1). Verdict skew, outcome divergence, signal-to-verdict correlation drift each get concrete threshold specification or explicit operational-tuning territory designation.

2. **Reframe the in-cycle-applicability test as operationally-applicable** (gap #2). The three-property test for "drift detected" trigger becomes specific + actionable + operationally applicable.

3. **Specify the asynchronous-operator-review dynamic** (gap #4). Differentiate advisory and severe drift handling; specify operator-notification mechanisms; acknowledge the operator-presence dependency that distinguishes mechanism (d) from the susceptibility-snapshot's dynamic.

The operator action surface choice (gap #3) is drafting-time territory — recommend specifying option (ii): diagnostic + parameter-tuning recommendation.

### Composition with mechanism (b)

The audit (#5e) revealed mechanisms (b) and (d) are more tightly coupled than ADR-016's drafting suggested: (b) is the bias-bound mechanism, (d) is the parameter-drift detector. Mechanism (d)'s drift criteria should explicitly include detection of (b)'s parameters drifting out of effectiveness — e.g., "if smaller-window configurations track better than the current default consistently, recommend adjusting the default."

This coupling sharpens (d)'s specification: one of the drift surfaces (d) audits is whether (b)'s parameters need tuning.

### Falsification trigger status

The structural transfer audit does not trigger the falsification clause in ADR-016. The pattern transfers; the specification gaps are addressable through ADR revision rather than re-design. The elaboration-by-evidence framing commitment from research-gate Grounding Action 2 remains in force.

---

## Spike artifact retained per corpus retention policy

Per cycle-status §"Conformance Notes" — Spike artifacts retention (Cycle 3 directive, applies to corpus until close):

- This research log itself (analytical spike output)
