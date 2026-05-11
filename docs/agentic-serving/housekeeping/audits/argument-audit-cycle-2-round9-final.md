# Argument Audit Report

**Audited document:** `docs/agentic-serving/essays/003-multi-turn-orchestration-and-the-four-axis-frame.md`
**Source material:**
- `docs/agentic-serving/housekeeping/audits/argument-audit-cycle-2-round8.md` (prior cleared state; six deferred framing items)
- `docs/agentic-serving/essays/research-logs/lit-review-cycle-2-multi-turn-and-composition.md` (tau-bench citation cross-check)
- `docs/agentic-serving/essays/research-logs/lit-review-cycle-2-ensemble-design-principles.md` (Yao et al. 2025 sequential panel discussion cross-check)
**Date:** 2026-04-29
**Round:** 9 (final verification — confirming six framing-audit amendments hold; verifying no new issues; cycle-close readiness assessment)

---

## Section 1: Argument Audit

### Summary

- **Argument chains mapped:** 13 (unchanged from round-8; six amendments are additions and qualifiers within existing chains, not new chains)
- **Issues found:** 0

### Amendment verification

Each of the six targeted amendments was located and verified against the essay text and supporting source material.

**Amendment 1 (A2 + script input alternative, §"What this means for the cycle") — PRESENT, internally consistent.**

The paragraph names the untested configuration explicitly ("call it 'A2 + script input': a prompt-steered single cloud orchestrator receiving the script-agent's deterministic report as additional input context"), identifies it as logically available from A3's data, names the alternative policy direction it would imply ("augment prompt-steering with deterministic tool outputs" rather than "use ensemble topology"), and closes with "The cycle does not refute this alternative; the next cycle should test it directly."

The paragraph does not claim A3's ensemble topology is superior to A2 + script input — it acknowledges the spike does not isolate which of A3's three design changes is load-bearing. This is logically consistent with §"Spike A3"'s description of A3 mixing three simultaneous changes, and with the §"Implications for the Architecture" ADR-011 refinement that preserves both paths. No contradiction with any other section.

**Amendment 2 (tau-bench citation, §"What the Literature Settled" / multi-turn dynamics subsection) — PRESENT, citation cross-checked.**

The paragraph cites "Yao et al., 2024" for tau-bench, consistent with the lit-review source table (entry 21: Yao, Shinn, Razavi, Narasimhan, 2024, arXiv:2406.12045). The quantitative claims — "GPT-4o achieves under 50% task success" and "pass^8 (consistency across eight runs) drops below 25% even for frontier models" — match the lit-review summary exactly ("GPT-4o <50% task success, <25% pass^8"). Year and numbers are correct.

The scoping is well-drawn: the paragraph names that the cycle's three spikes were single-ask tasks that do not reach the sustained multi-turn regime tau-bench measures, and explicitly limits the cycle's empirical strengthening of ADR-011 to single-ask task classes. This scoping is consistent with the Abstract's characterization of the spikes and with the §"Conclusion" which does not claim multi-turn generalization.

One internal terminological note: the same §"Loop 4" section cites "Yao et al. (2025)" for sequential panel discussion (panel-discussion precision 72.6% vs majority voting 77.3%). This is a distinct paper by a different Yao et al. group (meta-judge literature, arXiv listed in the ensemble design principles lit-review), and the two citations have different years. The essay does not disambiguate the two "Yao et al." citations to avoid reader confusion — a reader scanning §"Conclusion" (which references "Yao et al.'s sequential-panel-discussion diversity destruction") and the tau-bench paragraph in isolation could momentarily conflate them. This is a P3 presentation note, addressed below.

**Amendment 3 (A2 latency variance overlap, §"Spike A3") — PRESENT, correctly hedged.**

The relevant text: "The 4× framing is median-based; A2's actual range (15–70s, with one trial at 70s attributable to OpenCode Zen scheduling variance) overlaps A3's lower bound — on a slow-Zen-day for A2 and a fast-Zen-day for A3, the two configurations' wall-clocks are comparable. The reliability of the latency advantage is therefore stronger on the median comparison than on a worst-case-A2 versus best-case-A3 comparison."

This is correctly scoped. The abstract's "4× A2" formulation survives because the abstract says "4× A2 but bounded under A1's 145s," and the §"Spike A3" qualifier provides the supporting granularity without contradicting the abstract's shorthand. The essay carries the 4× figure in two other locations (§"What this means for the cycle": "A3's latency is 4× A2's median"; §"Conclusion": "at a 4× latency cost that is workable for batch/asynchronous review but not for chat-loop interaction") — both of these omit the overlap qualifier, but both also refer only to the median, which is consistent with §"Spike A3"'s framing. No internal contradiction; the abstract and body are aligned.

**Amendment 4 (realignment-as-correction treatment, §"The Design Priorities the Cycle Is Actually Navigating") — PRESENT. This is the substantively new claim; detailed analysis follows.**

The paragraph reads: "The four-priorities frame was adopted at the Loop 1 lit-review synthesis exchange, in response to practitioner pushback on an initial agent synthesis that overweighted performance findings and underweighted the practitioner's named optimization scope. Recording this honestly: the frame is one valid choice among alternatives, not the only defensible reading of the literature. A performance-only frame would reach similar configuration recommendations on the spikes that ran (A2 wins on performance, environmental cost, and token cost simultaneously on Spike A; A3's added value is shape-conditional rather than priority-conditional), but would close the novel-ensemble research program faster than the four-priorities frame did. The four-priorities frame's directional work — preserving the next-cycle territory rather than closing it — is what justifies its adoption here, with the caveat that 'we adopted this frame because it serves the cycle's research program' is a different claim than 'this frame is the most evidence-supported reading of the literature.'"

The audit brief asked specifically whether this admission is internally consistent with the conclusion's claim that "the framing realignment... remains the right frame."

The two claims are logically compatible but require care to distinguish. The §"Design Priorities" paragraph says the frame is justified because it serves the cycle's research program (a pragmatic justification). The §"Conclusion" says the frame "remains the right frame" after having characterized how priorities distribute across three configurations (an empirical-validation justification — the frame proved predictively useful). These are two different justifications for the same conclusion. The essay does not treat them as identical, and neither one undermines the other: a frame can be justified both by its research-program utility at the point of adoption and by its subsequent empirical productivity.

There is no logical inconsistency. The §"Design Priorities" paragraph's honesty disclosure ("we adopted because it serves the program") does not self-undermine the conclusion — it is transparent about why the frame was *chosen*, while the conclusion reflects on what the frame *produced*. The distinction is real and the essay preserves it. Landing as honest disclosure, not as self-undermining.

**Amendment 5 (A1-clean methodology purpose, §"Spike A" opening) — PRESENT, correctly framed.**

The methodological note explains: A1-clean was added to rule out a confound (MiniMax preferring OpenCode's mixed tool surface), names the confound precisely (cascade might not have run in via-opencode trials), and reports the resolution (server-side artifacts confirmed cascade ran regardless of client tool surface). This is internally consistent with the spike's qualitative findings and does not introduce any tension with the performance comparison between A1 and A2 — the confound-ruling was about whether the cascade ran at all, not about quality measurement.

**Amendment 6 (R1-Hunyuan format severity / recommendation count, Abstract) — PRESENT, internally consistent with body.**

The abstract reads: "~22 recommendations across three sections is the structural total, but R1-Hunyuan's chain-of-thought leakage means roughly half of those (R1's ~7–8) are accessible only after extraction work from ~18,000 characters of reasoning narration. The directly-consumable count on first read is closer to ~10 (R2-Kimi's ~6–8 cleanly numbered + the script's 3 factual sections); R1's content-equivalent specificity is real but format-degraded."

The body's §"Spike A3" carries the same information: "its recommendations were content-equivalent to A2's per-recommendation specificity... but its format required the user to extract recommendations from 18,000+ characters of reasoning narration." The §"What this means for the cycle" also records "R1-Hunyuan's chain-of-thought leakage taught the cycle that model-selection inside heterogeneity slots matters as much as family-distinctness." The abstract, §"Spike A3," and §"What this means for the cycle" are mutually consistent.

### New issues from the round-9 amendments

The six amendment locations and their surrounding prose were read for hidden assumptions, overstatements, contradictions, and scope-accuracy concerns.

### P1 — Must Fix

None.

### P2 — Should Fix

None.

### P3 — Consider

**P3-R9-1 — Two distinct "Yao et al." citations without disambiguation**

- **Location:** §"What the Literature Settled" (multi-turn dynamics subsection, line 54) and §"Loop 4" (line 170) and §"Conclusion" (line 216)
- **Claim:** The essay cites "Yao et al., 2024" for tau-bench and "Yao et al. (2025)" for sequential panel discussion. These are different papers by different first-author groups.
- **Evidence gap:** No ambiguity in meaning — the context clearly differentiates them (multi-turn tool tasks vs. judge-panel aggregation). The §"Conclusion"'s reference to "Yao et al.'s sequential-panel-discussion diversity destruction" could momentarily be read against the tau-bench citation if a reader moves non-linearly through the essay.
- **Recommendation:** Add a disambiguating phrase to one of the two citations on first use (e.g., "Yao et al. (2025, meta-judge framework)" vs. "Yao et al. (2024, tau-bench)"). This is a cosmetic presentation issue; the logical content is unambiguous.

**Verdict: clear.** Zero P1 or P2 issues. One P3 presentation note (two "Yao et al." citations without disambiguation; trivially fixable and does not affect argument soundness). All six amendments hold. No new issues introduced.

---

## Section 2: Framing Audit

The six deferred framing items have been addressed by targeted essay amendments and are confirmed integrated. This section verifies the integration holds, checks whether the amendments introduced new framing concerns, and provides the three structural framing questions against the essay's final state.

### Deferred items — resolved status

| Prior ID | Prior Severity | Amendment | Status |
|----------|---------------|-----------|--------|
| Framing P1-2 | P1 (gate) | Amendment 2: tau-bench paragraph added in §"What the Literature Settled" | Resolved — the omission no longer exists; the paragraph names the gap, cites numbers correctly, and scopes the cycle's empirical claims accordingly |
| Framing P2-1 | P2 (gate) | Amendment 4: realignment-as-correction paragraph in §"The Design Priorities" | Resolved — the framing choice is now transparent about its pragmatic justification, with the "serves the cycle's research program" / "most evidence-supported reading" distinction made explicit |
| Framing P2-R5-1 | P2 (gate) | Amendment 3: A2 latency variance qualifier in §"Spike A3" | Resolved — the 4× median framing is now qualified with the overlap disclosure |
| Framing P3-R5-1 | P3 (gate) | Amendment 5: A1-clean methodology note at the start of §"Spike A" | Resolved — the arm's purpose is explained before the narrative proceeds |
| (Amendment 1 per brief) | — | Amendment 1: A2 + script input paragraph in §"What this means for the cycle" | Resolved — the untested alternative is named, its logical availability is stated, and the next-cycle direction is explicit |
| (Amendment 6 per brief) | — | Amendment 6: Recommendation count scoping caveat in Abstract | Resolved — the ~22 / ~10 distinction is in the abstract and consistent with the body |

### Question 1: What alternative framings did the evidence support?

The source material supports three alternative framings not adopted by the essay.

**Alternative A: performance-first with enumerated priority trade-offs.** The essay acknowledges this framing directly in the §"Design Priorities" paragraph (amendment 4). Under this framing, A2 wins the spike battery on the dominant axis (performance), and the four-priorities frame adds research-program preservation rather than evidence weight. The essay names this alternative and distinguishes it from the adopted frame. No new framing concern; it is now explicitly visible.

**Alternative B: "single-task-class" limitation as the primary finding.** The essay's three spikes all used the same task class (code review on the project README). An alternative framing of the same evidence would foreground the extreme task-class specificity of the findings as the primary takeaway rather than the ensemble-design characterization. Under this framing, the essay's boundary characterization ("ensembles whose topology adds structural capability prompt-steering structurally lacks") would be a provisional hypothesis rather than a characterized boundary. The essay acknowledges task-class specificity in several places but does not foreground it as the primary lens.

Belief-mapping: "What would a reader need to believe for this framing to be right?" They would need to believe that code-review-on-README is insufficiently representative to support the essay's general characterization of ADR-011's boundary. The essay's treatment of this as a scope condition rather than a primary framing is a reasonable choice — the cycle was never positioned to generalize beyond its task class, and the essay's boundary claim is explicitly scoped to "task classes where factual grounding via deterministic checks is part of the success criterion." The alternative framing does not invalidate the essay's conclusions; it frames them more tentatively. This sits at P3 as a framing observation, not a framing defect.

**Alternative C: "model-selection dominates topology selection."** R1-Hunyuan's format degradation means that model selection inside the heterogeneity slot mattered as much as the topology (MARG concatenation) or the family-distinctness. An alternative framing would read A3's moderate pass as primarily a finding about model format quality rather than about topology validity. The essay acknowledges this ("model-selection inside heterogeneity slots matters as much as family-distinctness") but does not carry it through to the ADR-011 boundary refinement, which focuses on topology ("factual grounding via deterministic checks"). Under this alternative framing, the boundary refinement would be "use deterministic tool outputs, and ensure LLM slots have format-compliant models" — a model-selection lesson alongside a topology lesson. The essay's topology-focused framing is a reasonable simplification; the model-selection lesson is present in the body and in the abstract.

### Question 2: What truths were available but not featured?

The six deferred items are now integrated. Checking for residual gaps after integration:

**Tau-bench pass^k context.** The essay reports pass^8 below 25% for frontier models. Pass^k is a compound consistency metric — it measures whether a model can repeat a correct answer k times, not just once. The essay does not explain what pass^8 measures to a reader unfamiliar with the metric. The body paragraph parenthetically defines it as "consistency across eight runs," which is correct but minimal. This is a presentation adequacy note, not an evidence gap; the number is accurate and the source is cited.

**A3 trial-count basis.** A3's findings rest on three trials. The essay acknowledges "the test was on a single task class" but does not flag the three-trial basis for the specific quantitative claims (62–128s range, 5–8 distinct findings per reviewer, 1–2 overlap). Three trials is a thin basis for numerical ranges. The essay's characterization of A3 as a "moderate pass with caveats" implicitly carries this epistemic condition, and the absence of a confidence caveat on the per-trial measurements is consistent with the essay's voice elsewhere (Spike A and Spike B also used small N). This is a pre-existing condition, not introduced by the round-9 amendments. Naming it here as a residual framing observation.

### Question 3: What would change if the dominant framing were inverted?

The dominant framing after amendment 4's integration: "the four-priorities frame justifies investigation of ensemble territory because it preserves the research program, and that choice is distinct from the claim that the frame is the most evidence-supported reading of the literature."

Inverted framing: "the four-priorities frame is a practitioner preference that shaped the research program more than the evidence warranted; the most evidence-supported reading would have closed the ensemble investigation after Spike A."

Under the inverted framing:
- The §"Conclusion"'s "remains the right frame" claim becomes a preference assertion rather than an evidence-supported conclusion.
- Amendment 4's "serves the cycle's research program" admission becomes the primary finding rather than a transparency disclosure.
- A3's moderate pass with caveats would read as a weak result that justifies closing the line of inquiry rather than as empirical territory-mapping.
- The ADR-011 boundary refinement ("defensible as default, not as ceiling") would be read as a weak empirical result at three trials on one task class rather than as a characterized operational boundary.

What the essay would need to address if it took the inverted framing seriously: it would need to distinguish more sharply between "the four-priorities frame helped us run a productive cycle" and "the four-priorities frame is descriptively accurate of the deployment's actual constraints." The essay treats these as aligned; the inverted framing would stress-test whether environmental cost and local-first preference are genuine binding constraints on the practitioner's actual decision-making, or aspirational design values that rationalize a preference for ensemble investigation. Amendment 4 now makes this tension legible — but the §"Conclusion"'s "remains the right frame" does not engage with the inverted framing directly.

This is not a new issue introduced by the amendments; it was the original Framing P2-1 deferred item. Amendment 4's integration makes the frame's justification transparent. The conclusion's "remains the right frame" does not need to refute the performance-only alternative in order to be defensible — the essay's position is that both frames are available and the four-priorities frame was chosen for its research-program utility. The reader now has what they need to evaluate that choice.

### Framing Issues

**No new P1 framing issues.** The tau-bench omission (prior Framing P1-2) is integrated.

**No new P2 framing issues.** The realignment-as-correction framing (prior Framing P2-1) and the A2 latency variance gap (prior Framing P2-R5-1) are integrated.

**No new P3 framing issues beyond the cosmetic Yao disambiguation already noted in Section 1.**

The two framing observations from Question 2 (pass^8 explanation thinness; three-trial basis for A3 quantitative claims) are pre-existing conditions not introduced by the round-9 amendments, and both sit below the threshold for flagging as new findings. They are named here for completeness.

---

**Overall verdict: clear. Ready for cycle-research-phase close.** Argument audit is clean (zero issues at any severity beyond one P3 presentation note). All six targeted amendments are confirmed present, internally consistent, and logically sound. The amendment-4 tension (pragmatic adoption justification vs. conclusion's "remains the right frame") resolves as honest disclosure, not self-undermining. The six prior deferred framing items are resolved. No new framing concerns at P1 or P2.

The essay is ready for: gate reflection note, susceptibility snapshot, reflections, research log archive, and cycle-status update.
