# Argument Audit Report — Round 4

**Audited document:** `docs/agentic-serving/essays/005-layer-conditional-composition.md`
**Source material:**
- `docs/agentic-serving/essays/research-logs/research-log.md`
- `docs/agentic-serving/essays/research-logs/005a-lit-review-long-horizon-reliability.md`
- `docs/agentic-serving/essays/research-logs/005b-lit-review-composition-shapes-per-layer.md`
- `docs/agentic-serving/essays/research-logs/005c-lit-review-long-horizon-reliability-infrastructure.md`
- `docs/agentic-serving/essays/research-logs/005d-spike-research-loop-dogfood.md`
- `docs/agentic-serving/essays/research-logs/003a-lit-review-multi-turn-and-composition.md` (Cycle 2, Li et al. provenance)
- `docs/agentic-serving/housekeeping/audits/argument-audit-cycle-4.md` (round-1 findings)
- `docs/agentic-serving/housekeeping/audits/argument-audit-cycle-4-round2.md` (round-2 findings)
**Date:** 2026-05-04

---

## Prior Audit Status Note

The audit trail on disk runs: round 1 (`argument-audit-cycle-4.md`) and round 2 (`argument-audit-cycle-4-round2.md`). No written `argument-audit-cycle-4-round3.md` file exists. The task's reference to "round 3" corresponds to a gate conversation that approved the essay without producing a written audit artifact — the verdict (gate-ready; 0 P1, 0 P2, 2 P3 carry-forwards) is consistent with the current essay text, where the round-2 P2-NEW-1 scope condition has been added to the Conclusion and the two round-2 P3 carry-forwards remain unaddressed by design. This round-4 audit is the next written artifact.

---

## Round-4 Scope

This audit is scoped to four questions:

1. Is the ADR candidate #6 revision internally consistent with the rest of the essay — specifically: literature attribution accuracy, "load-bearing not decorative" alignment with prior framing, and conditionality framing coexistence with read-only scope?
2. Do the round-4 revisions introduce new framing issues, including proportionality?
3. Do the revisions aggravate either round-2/3 P3 carry-forward?
4. Pipeline readiness assessment.

---

## Section 1: Argument Audit

### Summary

- **Argument chains mapped:** 14 (unchanged from prior rounds; round-4 revision adds substantive detail to chain 10 — the ADR #6 architectural extension justification — without adding new logical chains)
- **Issues found:** 2 (0 P1, 1 P2, 1 P3)
- **Prior carry-forwards aggravated:** 0

---

### P1 — Must Fix

No P1 issues found. The round-4 revision does not introduce any logical gaps where conclusions fail to follow from evidence, internal contradictions, or claims demonstrably wrong given the evidence trail.

---

### P2 — Should Fix

**P2-R4-1 — Khanal et al. memory-scaffold attribution overstates the finding by one grade**

- **Location:** "Seven ADR Candidates" section, ADR candidate #6, "Cycle/scale risk" paragraph: "Khanal et al. (arXiv:2603.29231) found *universal negative effects across all ten tested models* from episodic memory augmentation at long horizons"
- **Claim:** The essay asserts universal negative effects across all ten models.
- **Evidence gap:** The source material (005a-lit-review-long-horizon-reliability.md, FQ1 synthesis) states: "episodic memory augmentation *universally hurts* long-horizon performance across all 10 models tested. Six models show negative effects; four are neutral." Six negative plus four neutral is not uniform negative across all ten — it is universal non-improvement (none improved), with graduated impact. The essay's phrasing "universal negative effects across all ten tested models" conflates the aggregate direction (no model benefits) with the per-model severity (four models were neutral, not harmed). The source's own language — "universally hurts" — is also slightly imprecise (the source immediately qualifies it as "six negative, four neutral"), but the essay inherits that imprecision and compounds it with "across all ten tested models," which implies all ten exhibited negative effects rather than no-benefit-or-harm.
- **Why this matters for ADR #6:** The revision uses Khanal et al.'s memory finding as one of three literature anchors making the cycle/scale risk concrete. Overstating it by a grade does not change the direction of the argument (the risk is real), but it weakens the evidentiary precision in a paragraph whose stated purpose is to name bounding mechanisms that must be "load-bearing, not decoration." An argument for stringent bounding mechanisms is most credible when its risk evidence is precisely stated.
- **Recommendation:** Replace "universal negative effects across all ten tested models" with "no benefit across all ten tested models, with negative effects in six of ten — none improved" or with the source's own qualified formulation "universal non-improvement (six negative, four neutral across all ten models)." This preserves the directional force of the finding without overstating severity.

---

### P3 — Consider

**P3-R4-1 — Li et al. (ICLR 2026) trigger-vulnerability finding applied outside its source domain without scope condition**

- **Location:** "Seven ADR Candidates" section, ADR candidate #6, "Cycle/scale risk" paragraph: "Li et al. (ICLR 2026) documented the trigger-vulnerability finding that injecting *objective* context into a debate accelerates polarization rather than moderating it"
- **Claim:** The essay cites Li et al. as evidence that feedback paths can compound bias in a calibration context.
- **Tightening opportunity:** Li et al.'s finding (source: 003a-lit-review-multi-turn-and-composition.md, RQ-3 failure modes section) is specifically about structured multi-agent debate workflows — "structured workflows act as echo chambers, amplifying minor stochastic biases into systemic polarization." The Trigger Vulnerability finding — that injecting objective context accelerates polarization — is a property of debate topologies where agents exchange and respond to each other's outputs iteratively. The L0→L1 calibration signal channel being described in ADR #6 is not a debate topology: it is a read-only unidirectional signal from ensemble outputs to the Calibration Gate's dispatch decisions. The feedback shape is present (Gate dispatches → Ensemble runs → signals return to Gate), but the mechanism by which Li et al. documented polarization (iterative agent-to-agent refinement in a debate loop) is architecturally distinct from a unidirectional calibration signal read by a downstream gate. The Cycle 2 source itself flags this applicability question: "Whether the trigger-vulnerability + echo-chamber findings translate to the actual coordination shape is empirically open. RQ-3 + RQ-3c." The essay applies Li et al. without carrying this scope condition, which was explicitly recorded in the lit-review.
- **Impact:** This is P3 rather than P2 because (a) the directional concern is still valid (feedback paths generally can compound bias), (b) the other two literature anchors in the paragraph — Khanal et al. on memory scaffolds and CAAF on prompt-engineering artifacts — provide more direct support for the calibration-bias risk, and (c) the bounding mechanisms the paragraph proposes are well-motivated regardless of which specific literature piece motivates them. The Li et al. citation adds rhetorical weight but is not the load-bearing evidential anchor.
- **Recommendation:** Add a scope condition clause: "Li et al. (ICLR 2026) documented in multi-agent debate settings that injecting *objective* context accelerates polarization rather than moderating it — the debate topology differs from the calibration channel's unidirectional read-only structure, but the directional risk (feedback shapes can entrench rather than correct bias) is transferable as a design caution." Alternatively, move Li et al. to a supporting position after the more directly applicable Khanal et al. and CAAF citations, with a brief note that bias compounding in debate topologies is an additional literature anchor for the general caution.

---

### Cleanliness Inheritance: Round-2/3 P3 Carry-Forwards

**P3-R2-1 (spike section "partially closes" language):** The essay still contains "the autonomous-routing positive observation (N=1) partially closes Cycle 3 grounding action 2's evidence gap" in the spike section's summary paragraph. This phrasing predates the more careful earlier-in-section language establishing that the Sub-Q6 transfer-test remains entirely open. The round-4 revision does not aggravate this finding — the ADR #6 expansion is in a separate section and does not touch the spike summary paragraph.

**P3-R2-2 (ADR-004 implications paragraph without pending-spike caveat):** The implications section's ADR-004 paragraph still says "suggests 'mandatory' should become 'default-with-conditional-skip'" without the pending-spike caveat language. The round-4 revision does not aggravate this finding.

Both P3 carry-forwards are confirmed unaddressed by design (DECIDE-phase carry-forwards per practitioner approval), and neither is made worse by the round-4 revision.

---

### Internal Consistency Verification: Round-4 Revision Specific Checks

**Check 1 — Literature attribution accuracy**

Three literature sources are invoked in the ADR #6 "cycle/scale risk" paragraph. Attribution accuracy by source:

- *Khanal et al. (arXiv:2603.29231):* Invoked for "universal negative effects across all ten tested models from episodic memory augmentation." Source: 005a FQ1 synthesis. Attribution is directionally accurate but overstates by conflating "universal non-improvement" with "universal negative effect across all ten." See P2-R4-1 above.

- *CAAF (arXiv:2604.17025):* Invoked for "calibration signals are themselves model outputs, and gating on biased outputs entrenches bias." The source (005a FQ2) documents: "apparent LLM reliability in safety-critical domains is often a prompt engineering artifact — removing semantic hints collapses monolithic models from 90% to 0%." The essay's application to calibration signals extends CAAF's finding by one inferential step: CAAF demonstrates that LLM outputs that appear reliable are prompt-conditioned artifacts; the essay then applies this to the specific case of calibration signals, inferring that gating on such outputs can entrench bias. This is a sound inference but is not directly stated in CAAF. The arXiv citation is correctly assigned. The extension is reasonable; it is an authorial inference, not a misattribution.

- *Li et al. (ICLR 2026):* Attribution to the trigger-vulnerability finding is correct — the source material (003a, RQ-3 section) confirms this finding at ICLR 2026, OpenReview mo7u21GoQv. The accuracy issue is scope applicability (see P3-R4-1), not citation accuracy.

**Check 2 — "Load-bearing not decorative" vs. "small in scope but consequential in principle"**

The essay's earlier formulation ("The amendment is small in scope but consequential in principle") appears in the ADR #6 description before the new paragraph. The new paragraph states the bounding mechanisms "must be load-bearing in the implementation, not decoration" and that "the amendment to ADR-002's layering rule is conditional on these bounding mechanisms being load-bearing." These framings are consistent: an amendment small in scope (read-only channel, signal-only, not general upward import) can impose large implementation requirements as the price of the exception's contained scope. "Small in scope" describes what the amendment permits; "load-bearing bounding mechanisms" describes what the amendment requires. No contradiction.

**Check 3 — "Conditional on bounding mechanisms" vs. "read-only and signal-channel-specific"**

The new paragraph states the amendment "is conditional on these bounding mechanisms being load-bearing in the implementation" while the earlier text describes the amendment as "read-only (no upward writes; the layering rule's structural integrity for write paths is preserved) and signal-channel-specific (calibration only; not a general upward import permission)." These hold simultaneously without contradiction. The scope description characterizes the amendment's permissions; the conditionality characterizes the prerequisites for exercising those permissions. The logical form is: "the exception to the layering rule is permitted if and only if these five mechanisms are operationalized." The scope condition does not widen what the amendment permits — it narrows the conditions under which the permission is activated.

**Check 4 — "Whether the cycle/scale risk can be bounded effectively is the load-bearing question DECIDE must answer"**

The concluding sentence of the new paragraph is an honest scope statement — it does not assert the risk is bounded, only that DECIDE must answer whether it can be. This is consistent with the essay's prior treatment of DECIDE-phase questions (numerous other candidates explicitly defer to DECIDE). No issue.

---

## Section 2: Framing Audit

The framing audit evaluates whether the round-4 revision introduces new framing choices that exclude available material or overreach the evidence.

### Question 1: What alternative framings does the round-4 revision support?

The round-4 revision is scoped to ADR candidate #6's "cycle/scale risk and required bounding mechanisms" paragraph. It does not shift the essay's primary framing (layer-conditional cross-layer composition) or its architectural verdict (operationalizable within existing layers with one amendment). The three alternative framings from prior rounds (script-models-as-primary; inadequate-empirical-base; pre-specifiable-routing-as-bet-against-capability-trajectory) remain valid against the revised text without change. No new alternative framings are opened or closed by the revision.

### Question 2: What truths were available but not featured in the revision?

The revision draws on three pieces of literature to motivate the five bounding mechanisms. One available truth not featured:

**Available truth: The scope-condition caveat on Li et al.'s applicability was explicitly recorded in the Cycle 2 lit-review.** The 003a source notes at RQ-3+RQ-3c: "Whether the trigger-vulnerability + echo-chamber findings translate to the actual coordination shape is empirically open." This caveat — that Li et al.'s debate-topology finding has unverified applicability to llm-orc's coordination architecture — was available in the source material and is not carried into the essay's use of Li et al. as a motivating citation for ADR #6's bounding mechanisms. Inclusion would strengthen rather than weaken the paragraph: a reader who sees the scope condition acknowledged will be more confident that the bounding mechanisms are motivated by careful risk assessment rather than conservative borrowing from adjacent literature. This is the substance of P3-R4-1 above.

### Question 3: What would change if the dominant framing were inverted?

The dominant framing of the round-4 revision is: feedback paths in calibration architectures are a concrete risk requiring explicit bounding mechanisms. The inversion: the feedback path is architecturally benign and the five bounding mechanisms are precautionary over-specification.

Under the inverted framing:
- The literature evidence becomes weaker: Khanal et al.'s memory-scaffold finding is about episodic augmentation loops, not read-only calibration channels; CAAF's prompt-engineering-artifact finding is about monolithic LLM reliability, not gated calibration systems; Li et al.'s debate finding is domain-specific. None directly studies the L0→L1 read-only signal shape.
- The "load-bearing not decorative" framing becomes a design-cost risk rather than a protective requirement: DECIDE might over-specify the bounding mechanism surface, making ADR #6's implementation heavier than the evidence warrants.
- The conditionality language becomes a potential blocker rather than a quality gate: if the five mechanisms are "required," any DECIDE-phase finding that one mechanism is unimplementable blocks the amendment even if the risk is lower than the citation pattern implies.

The essay's current framing is defensible against the inversion because: (a) the essay's own concluding sentence explicitly leaves "whether the risk can be bounded effectively" to DECIDE rather than asserting the risk is catastrophic, and (b) the five bounding mechanisms are individually lightweight (fresh-context isolation, time-decay windowing, categorical anchors, periodic audit, structural validation), not prohibitively expensive. The framing is precautionary rather than alarmist, which is appropriate given the novel architectural territory.

### Framing Issues — Round 4

**Framing proportionality — not a gate issue, acknowledged**

ADR candidate #6's description is now approximately three times the length of any other candidate. This reflects the genuinely elevated load-bearing nature of the candidate: it amends a layering rule, touches FC-2 and FC-3 test constraints, and carries a feedback risk requiring explicit bounding. The proportionality difference is substantively justified rather than editorially inconsistent. The closing paragraph of the "Seven ADR Candidates" section already contains language distinguishing the epistemic profiles and deliberation budgets appropriate to each candidate, so a reader is primed to expect unequal depth. No framing issue.

**P2 — Underrepresented alternative**

**FP2-R4-1 — Bounding mechanism (c) categorical anchors via deterministic-tool-output may require more infrastructure than the framing implies**

- **Location:** ADR candidate #6, "Cycle/scale risk" paragraph, bounding mechanism (c): "categorical anchors via deterministic-tool-output where possible (Wisdom and Delusion of LLM Ensembles, arXiv:2510.21513, on CrossHair counterexample feedback — deterministic outputs cannot be argued away by LLM consensus, so the feedback loop cannot drift on probabilistic noise)"
- **Underrepresented alternative:** The essay cites this as a mechanism available within the L0→L1 calibration channel. But the CrossHair counterexample pattern (from the Wisdom and Delusion paper) requires a CrossHair-style deterministic verification tool embedded as a script member of the ensemble — a specific ensemble composition requirement, not a general property of any L0 ensemble output. Not all ensembles in llm-orc's library will have deterministic-tool members whose outputs can serve as categorical anchors. The bounding mechanism is deployable only for ensembles that include script-model members; for LLM-only ensembles, this bounding mechanism is unavailable. The paragraph's "where possible" qualifier is present but may read as an implementation detail rather than a scope condition on the mechanism's applicability to a substantial fraction of the ensemble library.
- **Impact on ADR #6's conditional framing:** If bounding mechanism (c) is unavailable for LLM-only ensembles, then the conditionality statement ("the amendment is conditional on these bounding mechanisms being load-bearing") must specify whether all five mechanisms are required for all ensemble types or whether a subset suffices. The paragraph does not specify substitution logic.
- **Recommendation:** Add a scope condition: "bounding mechanism (c) is available only for ensembles with deterministic-tool (script-model) members; for LLM-only ensembles, the Calibration Gate consumer must rely on mechanisms (a), (b), (d), and (e) — DECIDE must specify whether the full five-mechanism set is required or whether a documented subset satisfies the conditionality for LLM-only configurations." This surfaces a genuine DECIDE-phase scope question without blocking the amendment.

**P3 — Minor framing choices**

**FP3-R4-1 (= P3-R4-1 restated as framing finding)**

The Li et al. trigger-vulnerability citation, applied without the source's own scope-condition caveat on applicability to non-debate topologies, creates a minor framing choice that underrepresents the available qualification. See P3-R4-1 in Section 1 above. The same finding is operationally a P3 in both the argument audit and the framing audit.

**FP3-R4-2 — Carry-forward framing issues from prior rounds remain operative**

The four carry-forward framing issues from round 2 (FP2-ROUND2-1 on five-tool surface prompt-structure conditioning; FP2-ROUND2-2 on DeliberationBench's boundary condition on the unifying frame; FP3-ROUND2-1 on adoption-vs-novel-architectural distinction among the seven candidates; FP3-ROUND2-2 on write-gate validation readiness) remain unaddressed by the round-4 revision. The revision does not aggravate any of them. They are recorded here for completeness; their resolution status (DECIDE-phase carry-forwards) is unchanged.

---

## Pipeline Readiness Assessment

**The essay is gate-ready.** The round-4 revision is internally consistent. Its three specific internal-consistency checks (literature attribution alignment, "load-bearing not decorative" alignment with prior framing, and conditionality coexistence with read-only scope) all pass. No P1 issues are present.

The one P2 finding (P2-R4-1, Khanal et al. overstated by one grade from "universal non-improvement" to "universal negative effects") is correctable with a single sentence substitution and does not block gate passage — the directional argument is sound and the bounding mechanisms are well-motivated without it. The P2 should be resolved before the ADR candidate #6 text is carried into DECIDE-phase drafting to ensure the risk evidence is precisely stated in a decision document.

The P3 findings (P3-R4-1 on Li et al. scope condition; FP2-R4-1 on mechanism (c) LLM-only ensemble applicability) are genuine tightening opportunities that would strengthen DECIDE's deliberation surface but are not gate blockers.

The round-2/3 P3 carry-forwards (spike section "partially closes" language; ADR-004 implications without pending-spike caveat) are confirmed unaffected by the revision.

**Verdict: Gate-ready. Resolve P2-R4-1 before DECIDE-phase drafting of ADR candidate #6. P3-R4-1 and FP2-R4-1 are recommended tightening for the DECIDE argument-audit gate on ADR #6 specifically.**
