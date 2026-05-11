# Argument Audit Report — Round 2

**Audited document:** `docs/agentic-serving/essays/005-layer-conditional-composition.md`
**Source material:**
- `docs/agentic-serving/essays/research-logs/research-log.md`
- `docs/agentic-serving/essays/research-logs/005a-lit-review-long-horizon-reliability.md`
- `docs/agentic-serving/essays/research-logs/005b-lit-review-composition-shapes-per-layer.md`
- `docs/agentic-serving/essays/research-logs/005c-lit-review-long-horizon-reliability-infrastructure.md`
- `docs/agentic-serving/essays/research-logs/005d-spike-research-loop-dogfood.md`
- `docs/agentic-serving/housekeeping/audits/argument-audit-cycle-4.md` (round-1 findings)
**Date:** 2026-05-04

---

## Round-1 Finding Resolution Status

### Resolved — All 13 round-1 findings addressed

The following table summarizes resolution status for each finding before the detailed analysis.

| Finding | Status | Notes |
|---------|--------|-------|
| P1-1 (spike-as-routing-comparison evidence) | Resolved | Language accurately scopes the spike's evidential role |
| P1-2 (ADR-004 amendment evidence threshold) | Resolved | Reframed as candidate amendment territory pending spike |
| P2-1 (B+D tension) | Resolved | Tension surfaced explicitly with deployment-state condition |
| P2-2 (MOP paradox cohort scope) | Resolved | Scope condition added; alternative reading recorded |
| P2-3 (Wave 2.B responsibility-concentration caveat) | Resolved | Caveat present in architectural verdict introduction |
| P2-4 (phantom tool-call novelty + Trial 3 prompt confound) | Resolved | Class of failure correctly scoped; confound flagged |
| P2-5 (Trial 1 not partial Sub-Q6 transfer-test answer) | Resolved | Reframed as necessary precondition only; transfer-test stated as entirely open |
| P3-1 (heterogeneity-vs-Self-MoA cluster-conditional) | Resolved | Cluster-conditional resolution present |
| P3-2 (Attention-MoA aggregator-quality scope condition) | Resolved | Benchmark scope condition added |
| P3-3 (74% MemoryAgentBench clarification) | Resolved | Reworded to plain-filesystem absolute score |
| P3-4 (250K API calls specificity) | Resolved | Claude Code-specific parenthetical added |
| P3-5 (Sub-Q6 resolution bridge) | Resolved | Bridge named explicitly |
| P3-6 (Qwen3-Coder-Next category shift) | Resolved | Category-shift note present |

---

## Section 1: Argument Audit

### Summary

- **Argument chains mapped:** 14 (same chains as round 1; re-verified against revised text)
- **Issues found:** 3 (0 P1, 1 P2, 2 P3)
- **Round-1 issues remaining:** 0

---

### P1 — Must Fix

No P1 issues found in the revised essay.

---

### P2 — Should Fix

**P2-NEW-1 — Scope condition added in one section but not carried into the Conclusion's invocation of the same claim**

- **Location:** "Long-Horizon Reliability" section (MOP paradox paragraph) versus Conclusion paragraph 2
- **Claim:** The "Long-Horizon Reliability" section now correctly adds the open-source-cohort scope condition: "The 'frontier' in Khanal et al. is the frontier of an open-source models cohort... not proprietary frontier models... extrapolation from open-source-frontier meltdown rates to proprietary frontier behavior on the North-Star benchmark is directionally plausible but not directly evidenced." It also records the capability-bounded-failure alternative reading.
- **Residual gap:** The Conclusion paragraph 2 still reads: "The MOP-paradox finding from the literature reframes the cycle's local-first commitment as well-calibrated rather than as a compromise: frontier-bare exhibits highest meltdown rates on long-horizon, and cheap-orchestrator + good-architecture may be more reliable than frontier-bare on the North-Star benchmark's task category." This sentence in the Conclusion drops the scope condition — a reader who reads only the Conclusion will not see that "frontier" here refers to the open-source cohort and that the proprietary-tier extrapolation is directionally plausible but not directly evidenced. The scope condition was the substance of P2-2; the Conclusion's summary-level invocation of the same claim does not carry it.
- **Recommendation:** Add a parenthetical or subordinate clause to the Conclusion's MOP-paradox sentence: "...frontier-bare (within Khanal et al.'s open-source cohort) exhibits highest meltdown rates on long-horizon, and cheap-orchestrator + good-architecture may be more reliable than frontier-bare on the North-Star benchmark's task category — a directionally plausible extrapolation, not a directly evidenced comparison."

---

### P3 — Consider

**P3-NEW-1 — "The Behavioral Spike" section's final paragraph presents four spike findings in a list that includes one claim revised away in earlier text**

- **Location:** "The Behavioral Spike" section, final paragraph ("The spike's findings refined the cycle's synthesis in three specific places. First... Second... Third... Fourth...")
- **Claim:** "Third, the autonomous-routing positive observation (N=1) partially closes Cycle 3 grounding action 2's evidence gap."
- **Tightening opportunity:** This wording predates P2-5's revision. The Sub-Q6 reformulation section and the preceding spike analysis now correctly state that Trial 1 "establishes a necessary precondition for the transfer-test (the path must work before degradation can be observed) but does not itself constitute partial evidence on the transfer question" and that "the Sub-Q6 transfer-test remains entirely open at cycle close." But the final summary paragraph still uses "partially closes Cycle 3 grounding action 2's evidence gap" — which is narrower than Sub-Q6 (grounding action 2 is about autonomous routing reliability, not the context-growth degradation question), so the phrasing is technically defensible, but it is in tension with the earlier, more careful reframing. Grounding action 2 asked whether the cheap-orchestrator dispatches autonomously at known transition points; Trial 1 gives a positive N=1 answer on that narrower question. The distinction between that narrower question (does it fire at all) and Sub-Q6's broader question (does judgment degrade as context grows) is precise but non-obvious, and the final-paragraph phrasing does not carry the precision established earlier.
- **Recommendation:** Tighten the third item to match the earlier reframing: "the dispatch path firing autonomously on one fixture (N=1) answers whether autonomous routing is *possible* at the cheap-cloud tier (yes) but does not establish its reliability at multi-iteration scale — Cycle 3 grounding action 2's full evidence gap remains open."

**P3-NEW-2 — ADR-004 implications paragraph still uses "amendment territory" language that the ADR candidate #5 revision softened**

- **Location:** "Implications for the Architecture" section, ADR-004 paragraph
- **Claim:** "ADR-004 (Result summarization mandatory) is the candidate amendment from ADR candidate #5. The cycle's evidence (Wave 3.A Trial 2's empirical specificity-loss; DeliberationBench's 6× gap) suggests 'mandatory' should become 'default-with-conditional-skip'."
- **Tightening opportunity:** ADR candidate #5 in the "Seven ADR Candidates" section correctly reframes this as "candidate amendment territory pending the targeted spike" with explicit acknowledgment that "the single-trial empirical motivation is below the evidentiary threshold for amending ADR-004's mandatory commitment in DECIDE." The "Implications for the Architecture" section's ADR-004 paragraph restates the amendment direction ("should become 'default-with-conditional-skip'") without carrying the pending-spike caveat. A reader referencing the Implications section without reading the ADR candidate #5 description in full will see an unqualified amendment direction. The ADR-004 implications paragraph and the ADR candidate #5 description are in mild tension; the candidate description is the more accurate statement.
- **Recommendation:** Add the pending-spike caveat to the Implications section's ADR-004 paragraph: append "— pending the targeted follow-up spike on diverse output sizes and ensemble configurations that the ADR candidate #5 description makes a prerequisite" after the "default-with-conditional-skip" phrasing.

---

## Section 2: Framing Audit

The framing audit makes the negative space of content selection visible. Round 1's framing section produced four framing issues (FP1-1, FP2-1, FP2-2, FP3-1 through FP3-3). The argument-audit revisions addressed some of these indirectly; this section evaluates which round-1 framing findings remain operative for the gate.

### Round-1 Framing Findings: Operative Status

**FP1-1 — Closed-five-tool-surface adequacy depends on conditions the cycle has not specified**

Status: **Remains operative, partially addressed.**

The revised essay does not add the specific scope condition FP1-1 recommended: "The closed-five-tool-surface finding depends on the orchestrator reliably emitting tool-call structures when dispatching; Trial 3's confabulation observation suggests this reliability is conditional on the prompt's task-structure implying the dispatch rather than demanding the tool call directly." The P2-4 revision correctly notes that "Trial 3 used an adversarial diagnostic prompt... rather than conditions representative of operational agentic sessions; the 1-of-3 confabulation rate is confounded by the prompt design." This partially addresses FP1-1 by flagging the prompt confound. But the inverse of the confound — that the success in Trials 1–2 is also conditioned on the prompt's task-structure implying the dispatch — is not stated. The essay's "closed five-tool surface sufficed" claim in the Conclusion remains unqualified by the conditions that made it true. FP1-1 survives at P2 level for the gate.

**FP2-1 — DeliberationBench selection-quality versus error-distribution-heterogeneity tension**

Status: **Remains operative.**

No revision addresses FP2-1. The unifying frame ("composition of components whose error distributions are different enough") is carried through the revised essay without noting that DeliberationBench's best-single-selection finding suggests selection quality may matter more than heterogeneity for tasks where the best response is identifiable. The essay treats DeliberationBench as a topology recommendation (no deliberation, parallel concatenation) but not as a challenge to the error-distribution-heterogeneity frame. This tension is relevant at the gate: the DECIDE phase's ADR candidates on ensemble design should be flagged that the unifying frame may not hold uniformly across the cycle's task class.

**FP2-2 — Capability-bounded-failure alternative reading of MOP paradox**

Status: **Resolved by P2-2 revision.**

The revised essay explicitly adds: "An alternative reading available in the same data is worth recording: the cheap orchestrator's reliability advantage on long-horizon may be partly due to capability-bounded failure (failing before the meltdown threshold) rather than architectural reliability compensation. Khanal et al. characterize weaker models as failing 'earlier and more uniformly' — distinct from failing *less*. The architectural-reliability and capability-bounded-failure readings coexist in the data; neither is settled." This directly addresses FP2-2's recommended alternative reading. Resolved.

**FP3-1 — Seven ADR candidates' provenance distinction**

Status: **Remains operative at P3 level.**

The essay still presents all seven candidates as "the cycle's research surfaces seven concrete design decisions as candidate ADRs." FP3-1's point — that ADR candidates #1 (Claude Code five-layer pattern) and #2 (Anthropic initializer schema) are adoption decisions with minimal uncertainty while candidates #3–#7 involve novel architectural territory — is not addressed in the revision. This framing choice affects DECIDE-phase resource allocation. Operative at P3 for the gate.

**FP3-2 — ADR-082 reference without content**

Status: **Remains operative at P3 level, low severity.**

The reference to "ADR-082 (of the RDD plugin methodology)" still does not name the constraint it represents. A reader unfamiliar with the RDD methodology's ADR numbering remains in the dark. Operative at P3.

**FP3-3 — Terminology consistency: "posture" versus "design-method" versus "design-formulation deliverable"**

Status: **Remains operative at P3 level.**

No revision addresses the terminology consistency observation. "Posture," "design-method posture," "design-method," and "design-formulation deliverable" continue to co-exist as framings for the same output across different sections of the essay.

---

### Question 1: What alternative framings does the revised essay support?

The three alternative framings from round 1 (script-models-as-primary-layer; inadequate-empirical-base; pre-specifiable-routing-as-bet-against-capability-trajectory) remain valid against the revised text. No revision shifted the essay's primary framing in a way that reduces the force of any of these alternatives. The revisions addressed precision of specific claims rather than framing posture.

### Question 2: What truths remain available but not featured?

From round 1:

- **Absent finding A** (7B-14B local small model boundary connection to spike's 0.6B validation) — **remains absent.** The revised essay does not connect the lit-review's 7B-14B reliability boundary to the 0.6B models used in the spike. The spike's "closed five-tool surface sufficed" finding remains unconditioned on the specific bounded-single-call roles the 0.6B models played. This is operationally relevant for anyone reading the spike validation as evidence that very small local models are adequate for the ensemble role generally.

- **Absent finding B** (four-priorities frame measured-divergence test unaddressed) — **remains absent.** No revision introduces this carry-forward.

- **Absent finding C** (write-gate validation is current open research, not a deployable pattern) — **remains absent.** ADR candidate #2's description of write-gate validation does not carry the lit-review's "not operationalized in any reviewed system" caveat into the essay.

- **Absent finding D** (tool-consensus conflict failure mode as a Cycle 5 spike candidate) — **partially addressed.** The revised essay mentions the open question once in the "Composition Shapes Per Layer" section ("it surfaces an open question the literature does not address: what happens when deterministic tool output conflicts with LLM consensus?") but does not elevate it as a Cycle 5 spike candidate or as scope-of-claim discipline for ADR candidates that rely on the script-alongside-LLM pattern.

### Question 3: What would change if the dominant framing were inverted?

The three inversions from round 1 remain unchanged in their force against the revised text. The revisions tightened specific inferential chains without shifting the essay's architectural verdict ("operationalizable within existing layers"), its primary frame ("layer-conditional cross-layer composition"), or its design-axis conclusion ("pre-specifiable routing as primary axis within each layer"). The inversions remain as the gate's framing-alternative surface for the practitioner.

---

### Framing Issues — Round 2

**P1 — Consequential omissions**

No new P1 framing issues. The round-1 FP1-1 remains operative at P2 level (see above).

**P2 — Underrepresented alternatives**

**FP2-ROUND2-1 (= FP1-1 from round 1, downgraded)**

- **Location:** Conclusion paragraph 3; "The Behavioral Spike" section's "What this spike establishes" subsection
- **Claim:** "The dispatch path itself works — autonomous routing fired correctly in one trial; the closed five-tool surface sufficed throughout."
- **Underrepresented condition:** The spike log explicitly records: "the autonomous-vs-explicit comparison (trial 1 vs trial 2) shows both fired correctly when the *task structure implies the dispatch*; the diagnostic failure (trial 3) suggests the cheap-cloud model is more reliable when the task structure implies the dispatch than when the tool call itself is the demand." The essay's revised framing correctly notes that Trial 3's prompt was adversarial and confounds the failure rate — but does not state the positive implication: that Trials 1 and 2's success is also conditioned on task-structure-implying-dispatch. The "closed five-tool surface sufficed" claim in the Conclusion is unqualified by this conditioning. A practitioner reading the Conclusion as a standalone finding will not see that the adequacy claim holds for task-structured prompts and has a documented failure mode for tool-demand prompts. This is now P2 rather than P1 because the Trial 3 prompt confound is flagged; what is missing is the symmetric condition on Trials 1–2.
- **Recommendation:** Add a parenthetical to the Conclusion's dispatch claim: "...autonomous routing fired correctly in one trial; the closed five-tool surface sufficed throughout — on task-structured prompts where the work to be done implies the dispatch (Trials 1–2); the adversarial diagnostic prompt (Trial 3) produced the one confabulation, suggesting adequacy is prompt-structure-conditional."

**FP2-ROUND2-2 (= FP2-1 from round 1, unchanged)**

- **Location:** "Composition Shapes Per Layer" section; "RDD-Cycle Decomposition" section's unifying frame
- **Claim:** Composition of components with different error distributions is the unifying frame across the cycle's empirical evidence.
- **Underrepresented alternative:** DeliberationBench's best-single-selection finding (82.5% win rate versus 13.8% for best deliberation protocol) suggests that for tasks where a best response is identifiable, selection quality may dominate error-distribution heterogeneity as the mechanism. The essay treats this as a topology recommendation but not as a boundary condition on the unifying frame. For DECIDE-phase ADR deliberation, this boundary matters: ensemble designs relying on heterogeneity-uncorrelated-errors are most defensible for verification and documentation-review task classes; for task classes where a best response is identifiable in advance (bounded analytical questions), the frame's prescriptive force is weaker.
- **Recommendation:** Name the boundary condition on the unifying frame in the section that introduces it: "The unifying frame holds most directly for tasks where no single best response is identifiable in advance — the cycle's verification and documentation-review task classes. For task classes where a best response is identifiable (bounded analytical questions with well-defined answers), DeliberationBench's selection-quality finding suggests a simpler 'select the best' pattern may outperform heterogeneous composition."

**P3 — Minor framing choices**

**FP3-ROUND2-1 (= FP3-1 from round 1, unchanged)**

- **Location:** "Seven ADR Candidates" section
- **Minor framing choice:** ADR candidates #1 and #2 are adoption decisions with canonical external reference implementations (Claude Code five-layer pattern; Anthropic initializer schema). The essay's framing presents all seven as cycle-surfaced design decisions without distinguishing adoption decisions from novel architectural choices. DECIDE-phase resource allocation would benefit from this distinction being explicit.

**FP3-ROUND2-2 (= absent finding C from round 1)**

- **Location:** "Seven ADR Candidates" section, ADR candidate #2
- **Minor framing choice:** Write-gate validation is listed as a component of ADR candidate #2's responsibilities but the source material (005c) flags it explicitly as "not operationalized in any reviewed system" — recommended in the literature but without a reference implementation. The essay's presentation of write-gate validation as part of the candidate's "design decision" surface implies it is adoptable at the same level of readiness as the feature-list schema, append-only log, and init-sh bootstrap — all of which have canonical Anthropic reference implementations. The distinction is material for DECIDE-phase scope estimation.
