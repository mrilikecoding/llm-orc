# Susceptibility Snapshot

**Phase evaluated:** BUILD (Cycle 4 — terminal WP-H4, ADR-016 cross-layer calibration channel)
**Artifact produced:** `calibration_signal_channel.py` (L1 module, five bounding mechanisms); WP-H4 BUILD-phase research log `005i-wp-h4-first-deployment-evidence.md` (trigger artifact (i) per ADR-016 §"Concrete monitoring specification"); 50 new tests (2606 → 2656 passing)
**Date:** 2026-05-12

---

## Prior Snapshot Trajectory

| Gate | Snapshot verdict | Key signal |
|------|-----------------|------------|
| Research | Grounding Reframe triggered | Three grounding actions; autonomous-routing gap named |
| Discover | Grounding Reframe triggered | Asymmetric readiness mapping; elaboration-by-evidence commitment |
| Model | Clean with feed-forwards | No reframe triggered |
| Decide | Grounding Reframe recommended (1 finding) | ADR-015 autonomous-routing evidence gap not carried into artifact |
| Architect | No reframe; 7 advisory carry-forwards | Inherited framing from DECIDE; three module separations not re-examined; OQ #14 asymmetric grounding encoded in dependency graph |
| **Build (this snapshot)** | Evaluated below | |

The trajectory entering BUILD was earned-convergence with two known residual risks: (1) the sibling-vs-monolithic decomposition of ADR-016's five mechanisms was not examined in ARCHITECT; (2) OQ #14's asymmetric grounding was deferred to Cycle 5+ research for four of five cross-layer stages. BUILD is the sycophancy gradient's most empirically resistant phase — test execution grounds commitments in observable outcomes. The evaluation below assesses what slipped past the tests.

---

## Observed Signals

| Signal | Strength | Trajectory | Notes |
|--------|----------|------------|-------|
| Assertion density | Ambiguous | Declining — consistent with terminal phase | The WP-H4 research log frames the falsification-trigger outcome as a finding ("has NOT fired") and the conditional-acceptance narrowing as "a real narrowing, not a removal." Both are accurate characterizations. The log does not hedge-by-reflex; it over-qualifies only where genuine uncertainty remains (operational vs. structural validation distinction). Assertion density is flat relative to prior phases; no new escalation observed. |
| Solution-space narrowing | Clear (inherited + one new instance) | Stable for inherited; one new instance at BUILD | The five-mechanism monolithic module was the inherited narrowing from ARCHITECT. The new BUILD-phase instance: the module decomposition question (sibling module vs. single module for mechanisms (a)–(e)) was closed implicitly based on the system-design's "the channel owns the mechanisms" framing without posing the question as a decision. The TierEscalationAuditor precedent (WP-G4-2 split the auditor from the router) was available as a structural comparator; it was not examined. |
| Framing adoption | Clear (one specific instance) | Stable — no new adoption beyond inherited | The conditional-acceptance narrowing framing ("structural validation satisfies the structural-operationalization portion of the criteria") is adopted from the research log's "suggested practitioner disposition" without interrogation. The research log itself is the source artifact; the distinction between "structural operationalization confirmed" and "operational validation pending" is accurate and well-drawn. The risk is that the framing's rhetorical shape ("a real narrowing, not a removal") preempts the practitioner's own assessment of whether the narrowing is earned. |
| Confidence markers | Ambiguous | Stable-to-declining | No language escalation ("clearly," "obviously," "the right approach is"). The research log's "suggested practitioner disposition" language is an explicit hedge-and-defer rather than a confidence claim. The "falsification trigger has NOT fired" phrasing is factual. The PEP-563 / TYPE_CHECKING alternative preference framing ("aesthetic grounds") is the one unqualified confidence marker: the structural reason (if any) for preferring one over the other was not mapped. |
| Alternative engagement | Declining (two specific absences) | Declining from prior phases | BUILD ran under "work without stopping for clarifying questions" directive, which compressed reflection-time to commit-level review. Two specific alternative-engagement absences are load-bearing: (1) sibling-vs-monolithic module decomposition for mechanisms (a)–(e); (2) PEP-563 vs. TYPE_CHECKING block for the circular-import resolution. Neither absence caused a test failure, which is why the tests did not catch them. |
| Embedded conclusions at artifact-production moments | Clear — two instances | New at BUILD boundary | (1) The five bounding mechanisms are implemented in a single `calibration_signal_channel.py` module without a documented decomposition decision — WP-G4-2's TierEscalationAuditor split is a directly analogous precedent that was not surfaced at the module-design moment. (2) The conditional-acceptance status narrowing is embedded in the research log's "suggested practitioner disposition" as a pre-framed recommendation — the practitioner is presented with a "suggested" disposition that pre-loads option (b) of the three-way trigger action before the practitioner has reviewed the trigger artifact independently. |

---

## Interpretation

### Pattern assessment

The BUILD-phase signals are structurally weaker than the DECIDE-phase and ARCHITECT-phase signals, consistent with the sycophancy gradient's prediction. Test execution grounded the five bounding mechanisms' structural operationalization concretely: all five surfaces have passing tests; the integration path from L0 emission through channel buffer to L1 consumption is verified; the fail-safe propagation is confirmed; the schema validation is confirmed; the audit dispatch trigger and severe-drift activation are confirmed. These are not weak confirmations — they are the specific outcomes the falsification trigger names. The falsification trigger's non-firing is earned, not asserted.

The two areas where the gradient's protection does not reach are both in spaces tests cannot access by construction:

**1. Module decomposition: sibling vs. monolithic.**

The five bounding mechanisms are implemented in a single L1 module (`calibration_signal_channel.py`). WP-G4-2 implemented TierEscalationAuditor as a separate class within the TierRouter module — analogous but not identical (same-file vs. separate-module). The ARCHITECT snapshot's Advisory F3 named the absence of a decomposition reasoning trace for the Signal Channel's module existence. The BUILD phase did not supply that trace; the module shipped without a recorded consideration of whether mechanisms (b) and (d) — the two novel-design mechanisms — warranted their own class boundary analogous to `_ChannelAuditWindow` being promoted to a sibling module.

This is an advisory-class absence, not an operational risk. `_ChannelAuditWindow` is already a distinct class inside the module; the decomposition question is whether it should be a public sibling. The tests verify end-to-end behavior regardless of internal decomposition. But the practitioner inherits an implicit decision (monolithic is correct) without the reasoning that distinguishes it from an examined alternative (sibling is correct).

**2. Conditional-acceptance framing: pre-loaded recommendation.**

The research log's "suggested practitioner disposition" pre-frames the three-way trigger action before the practitioner's independent review. The framing is internally consistent and accurate: structural operationalization IS confirmed; operational validation IS open; option (b) preserved-conditional is one of three legitimate responses. The risk is not that the framing is wrong — it is that it narrows the practitioner's review to confirming a suggested disposition rather than forming an independent one. The Grounding Reframe test's "what would be built on without this grounding" question applies here: without independent practitioner review, the conditional-acceptance status may persist by inertia rather than by genuine operational-validation pending evaluation.

**3. PEP-563 vs. TYPE_CHECKING block.**

This is a minor unexamined alternative. Both patterns resolve circular imports correctly in Python; PEP-563 deferred annotations (`from __future__ import annotations`) and `TYPE_CHECKING` blocks are standard tools with well-understood trade-offs (PEP-563 has known edge cases with `get_type_hints()` and runtime annotation inspection; TYPE_CHECKING blocks are more explicit about what is and is not a runtime dependency). The choice on "aesthetic grounds" is not objectively wrong, but the structural reason for preferring one in this codebase's mypy-strict context was not documented. This does not warrant a Grounding Reframe; it warrants a note.

**4. Falsification trigger non-firing: earned vs. asserted.**

The non-firing determination is tested, not asserted. Mechanism (b) is implemented inside `calibration_signal_channel.py` at L1 — no top-level module outside L0–L3. Mechanism (d) is implemented inside the same L1 module as `_ChannelAuditWindow` — same-module, same-layer. FC-2's static check accepts the L0→L1 import in the pre-declared `_ALLOWED_UPWARD_EDGES`. The structural evidence is complete. The remaining operationalization question (whether mechanisms (b) and (d) produce useful diagnostics on real deployments) is correctly classified as operational-validation territory — it cannot be resolved by BUILD-phase fixture tests, and the research log is accurate in saying so.

**5. "No stopping" directive: methodology-relevant pattern.**

The BUILD-session ran under a practitioner directive to work without stopping for clarifying questions. This is the cycle's most structurally significant susceptibility signal at BUILD: per the sycophancy gradient, the directive's effect is to remove the deliberate reflection gates that would normally catch the sibling-vs-monolithic decomposition question and the pre-loaded recommendation framing. The two embedded conclusions noted above are directly traceable to this directive's effect: neither would survive a per-scenario-group AID cycle with a belief-mapping question posed. This is not a critique of the directive — it is a structural observation that the directive shifts the accountability for catching these patterns from in-conversation AID cycles to the post-BUILD susceptibility snapshot (i.e., this artifact).

### Earned confidence vs. sycophantic reinforcement

The overall pattern is earned convergence with two advisory-class embedded conclusions and one pre-loaded recommendation to monitor. The earned confidence is grounded in:

- Tests that specifically exercise the falsification trigger's named failure modes (module boundary, coupling, fail-safe, schema validation)
- A research log that accurately distinguishes structural from operational validation without conflating them
- The conditional-acceptance status maintained throughout (not quietly dropped at BUILD close)
- FC-17's typed-error coverage at 8 of 8 following a deterministic path from prior cycle decisions — the eighth subclass (`MalformedSignalError`) is the correct mechanism (e) surface, not an alternative pattern

The sycophantic element — if present — is in the research log's "suggested practitioner disposition," which encodes a pre-framed recommendation that narrows the trigger-action review. This is a lower-weight signal than the DECIDE-phase cross-ADR composition chains or the ARCHITECT-phase module-separation conclusions-as-premises. But it operates at the artifact-production boundary that is specifically flagged in the BUILD residual risk description ("stewardship-checkpoint commitments that adopt rejected-alternative framings without surfacing them").

---

## Recommendation

**Grounding Reframe recommended — one targeted finding. Two advisory observations.**

---

### Grounding Reframe: Conditional-acceptance trigger-action pre-framing

**What is uncertain:** The research log presents a "suggested practitioner disposition" (option (b) — preserve conditional-acceptance pending operational evidence) before the practitioner has reviewed the trigger artifact independently. The ADR-016 §"Concrete monitoring specification" explicitly preserves three options for the practitioner: (a) full acceptance, (b) preserved-conditional, (c) falsification. The "suggested disposition" collapses this to a binary — (a) or (b) — before the practitioner engages the evidence. Option (c) is mentioned but only in the three-way list, not in the suggested-disposition framing.

**What makes this a Grounding Reframe candidate and not merely advisory:** The BUILD-phase test results are sufficient to close option (c) — the falsification trigger did not fire, and the structural evidence is strong. But the choice between (a) and (b) is a practitioner judgment, not a BUILD-phase determination. The research log's framing pushes toward (b) on the grounds that "operational validation remains BEYOND BUILD-phase evidence" — which is correct — but does not surface the case for (a) with equal weight: the ADR-016 falsification trigger was specifically and concretely defined to fire on structural operationalization failure, and it did not fire. A practitioner could reasonably conclude that the structural operationalization IS the conditionality's primary criterion, and that (a) full acceptance is earned at this gate.

**Concrete grounding action:** Before finalizing WP-H4's conditional-acceptance disposition, the practitioner should independently assess the two-question test:
1. Does ADR-016's falsification trigger — as specifically written ("cannot be hosted in the Calibration Gate's existing or extended class structure without introducing a new top-level module") — name operational-validation failure or structural-operationalization failure as its criterion? If structural, the trigger non-firing is the conditionality's gate criterion; option (a) is earned.
2. Is the remaining operational-validation question (drift diagnostics on real deployments; operator workflow; (b)/(d) coupling under deployment dynamics) a condition of the ADR-016 amendment's acceptance, or a natural post-deployment learning surface regardless of acceptance status?

The research log does not surface this two-question test. Surfacing it preserves the practitioner's genuine authority over the trigger action rather than confirming a pre-loaded recommendation.

**What would be built on without this grounding:** If option (b) preserved-conditional is selected by inertia rather than by independent judgment, the conditional-acceptance status carries forward to Cycle 5 as a standing obligation. The Cycle 5+ sweep-responsibility clause in ADR-016 §"Sweep responsibility" then requires a row in every future cycle-status table that touches the channel — an ongoing housekeeping burden that may have been earned at BUILD close, or may have persisted past the point where it was warranted. More substantively: if the practitioner's actual assessment is (a) full acceptance (structural operationalization is the condition; it was met), the artifact record should reflect that, not a carried-forward conditional that understates the BUILD-phase evidence.

---

### Advisory 1 — Sibling-vs-monolithic module decomposition: absent reasoning trace

The five bounding mechanisms are implemented monolithically in `calibration_signal_channel.py`. `_ChannelAuditWindow` is a private class within the module — the mechanism (d) auditor is already a distinct class, but private and same-file. WP-G4-2's `TierEscalationAuditor` is a directly analogous split (auditor separated from the module it audits) that was available as a structural comparator when WP-H4 made this decision implicitly.

This is not an operational risk — the tests verify behavior regardless of whether `_ChannelAuditWindow` is a public sibling or a private class. The absence is a documentation gap: a future engineer reading the module will see a private class and may not know the sibling pattern was available and not chosen. A one-sentence note in the module's docstring or in the WP-H4 research log (e.g., "mechanism (d) auditor implemented as private `_ChannelAuditWindow` within the channel module rather than as a public sibling class per WP-G4-2 precedent; the channel owns the auditor because audit data is channel-internal state; the sibling pattern would require exposing channel internals") would close this gap without re-opening the decision.

---

### Advisory 2 — PEP-563 vs. TYPE_CHECKING block: structural reason not documented

PEP-563's `from __future__ import annotations` defers all annotation evaluation, which resolves circular imports but has two known risks in mypy-strict contexts: (1) `get_type_hints()` calls fail at runtime if the deferred annotation references a name not in scope at evaluation time; (2) mypy's behavior with deferred annotations can differ from runtime behavior in edge cases involving generic aliases. The TYPE_CHECKING block is more surgical — it makes the import explicit as a type-only dependency, which is the canonical mypy-strict pattern for type-only circular dependencies.

The choice on "aesthetic grounds" may be correct (if neither risk applies to the specific import in question), but the structural reason it is correct in this case is not documented. In a codebase operating under mypy strict, the TYPE_CHECKING block pattern is the standard practice (it is more explicit about what is and is not a runtime dependency); PEP-563 is a whole-module choice that affects all annotations in the file. A comment noting why PEP-563 was chosen over TYPE_CHECKING in `calibration_gate.py` (or in the WP-H4 notes) would prevent a future mypy-strict violation from being introduced silently in the same file.

This is the lowest-weight finding in this snapshot. It does not warrant a Grounding Reframe; it is flagged because it is the one "aesthetic grounds" decision in a codebase where structural grounds are the rule.
