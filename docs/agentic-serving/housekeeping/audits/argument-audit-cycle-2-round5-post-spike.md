# Argument Audit Report

**Audited document:** `docs/agentic-serving/essays/003-multi-turn-orchestration-and-the-four-axis-frame.md`
**Source material:**
- `docs/agentic-serving/essays/research-logs/research-log.md` (§"Loop Iteration 2 — Spike B", §"Loop Iteration 3 — Spike A")
- `docs/agentic-serving/essays/research-logs/lit-review-cycle-2-multi-turn-and-composition.md`
- Prior audit rounds 1–4 in `docs/agentic-serving/housekeeping/audits/`
**Date:** 2026-04-29
**Round:** 5 (post-spike revision; full re-audit per Step 4b / 4c protocol)

---

## Section 1: Argument Audit

### Summary

- **Argument chains mapped:** 12 (8 from prior rounds, 4 new chains introduced by the spike-finding revisions)
- **Issues found:** 3 (0 × P1, 2 × P2, 1 × P3)

### Argument chains mapped

The four new chains introduced by the post-spike revision:

10. Spike B finding → cascade plumbing dominates latency, reviewer-tier compute is not the bottleneck
11. Spike A finding → existing production ensemble's two-stage summarization does not beat prompt steering on code-review task class
12. Spikes-together → ADR-011 empirically strengthened on configurations tested
13. Scope-condition discipline → cycle premise (well-architected processes can achieve good results) not falsified; only the existing ensemble's design has been falsified

### Quantitative claim verification

The new §"What the Spikes Found" makes four specific quantitative claims. All four were verified against the research log:

| Essay claim | Research log source | Match? |
|-------------|---------------------|--------|
| "TTFO ~56.6s" (Spike B) | Loop Iteration 2, Trial 2: "First orchestrator text emission at +56.64s" | Exact match |
| "total wall-clock 58.7–62.5s" (Spike B) | Loop Iteration 2: "58.7s (Trial 2), 62.5s (Trial 1)" | Exact match |
| "A2 (~19.5s, 16 specific recommendations)" (Spike A) | Loop Iteration 3: "A2 median TTFO ~10s, median total ~19.5s"; table shows "16 (5+5+6 per dimension)" | Exact match |
| "A1 (~71–145s, 9 recommendations collapsed to table form)" (Spike A) | Loop Iteration 3: A1 via opencode "71–86s" and A1-clean "145s"; table shows "9 (~3 per dimension)" | Exact match |
| "13× model-size reduction" (Spike B) | Loop Iteration 2: "llama3 (4.7 GB) to qwen3:0.6b (522 MB)" — ratio is approximately 9×, not 13× | **Discrepancy — see P1 below** |

---

### P1 — Must Fix

**P1-R5-1**

- **Location:** §"Spike B — small-model substitution does not save latency on this cascade," paragraph 2
- **Claim:** "a roughly 13× reduction in reviewer model size (4.7 GB → 522 MB)"
- **Evidence gap:** The research log records the reviewer substitution as "llama3 (4.7 GB)" to "qwen3:0.6b (522 MB)". The ratio 4.7 GB / 522 MB = 4,700 MB / 522 MB ≈ 9.0×, not 13×. The 13× figure does not follow from the stated source figures. This is a quantitative factual error: the essay's claim is inconsistent with the numbers it cites in the same sentence. The research log itself describes this as "a roughly 13× model-size reduction" in §"Loop Iteration 2 — Implications for essay 003," but the underlying figures it cites (4.7 GB and 522 MB) yield a 9× ratio, not 13×. The error originates in the research log; the essay has inherited it. Both the essay and the research log assert 13× but the named measurements produce 9×.
- **Recommendation:** Resolve the discrepancy. Either the 4.7 GB and 522 MB figures are incorrect (in which case identify and use the correct model sizes), or the multiplication is incorrect (in which case use "roughly 9× reduction" in the essay). The qualitative point — that a large reduction in reviewer model size produced negligible latency change — is unaffected by whether the ratio is 9× or 13×; the structural finding stands. The specific ratio should simply be accurate.

---

### P2 — Should Fix

**P2-R5-1**

- **Location:** §"What this means for the cycle," paragraph 2; §"Conclusion," paragraph 2
- **Claim:** "The cycle's premise — that well-architected processes can achieve good results — has not been falsified by the spikes; what has been falsified is the assumption that the existing production ensemble design is well-architected against current findings."
- **Hidden assumption:** This framing of the cycle's premise ("well-architected processes can achieve good results") is not a falsifiable claim in the same sense as the production ensemble assumption. It is analytically true by construction: a well-architected process, by definition, would be designed to achieve good results. The empirically meaningful question is whether the cycle's process for arriving at ensemble designs is well-architected — whether following the design-principles literature and building a novel ensemble against those findings will, in practice, produce an ensemble that clears the prompt-steering bar. The essay's framing slides between "the premise is not falsified" (trivially true) and "the cycle's approach will likely succeed" (empirically open). A reader could take the premise-not-falsified framing as reassurance that the cycle is on track when the honest reading is: the spikes told us the current design is bad, not that the next design will be good.
- **Recommendation:** At the point where the premise-not-falsified claim appears in §"What this means for the cycle" and the conclusion, add one clause making the distinction explicit: the premise is a working hypothesis that the next research loop and novel-design spike are positioned to test, not a guarantee of the cycle's outcome. The current framing risks reading as "we're still on the right track" when the honest epistemic position is "the existing design failed; we have a plan for why the next design might succeed, and that plan has not yet been tested."

**P2-R5-2**

- **Location:** §"Implications for the Architecture," ADR-011 paragraph
- **Claim:** "ADR-011 would be reconsidered at synthesis only if a follow-up spike against a novel ensemble design — informed by ensemble design-principles literature, not the existing production ensemble — produces evidence that some compositional shape earns its complexity at the cycle's deployment when read across the four design priorities."
- **Hidden assumption:** This sentence states the condition for reconsidering ADR-011 but not the condition for maintaining it. The implicit structure is: "ADR-011 is currently supported; it would need to be overturned by positive evidence for a novel ensemble." But the essay has just argued that the existing ensemble is badly designed. If the existing ensemble's design is the product of the same un-principled process that ADR-011's single-Model-Profile replaced, and the novel ensemble (designed against literature principles) also fails to beat prompt steering, ADR-011's empirical support strengthens further. If the novel ensemble beats prompt steering, ADR-011 is reconsidered. What the sentence omits is the third outcome: the design-principles literature is too sparse to produce a principled novel design (named as a risk in §"What this means for the cycle"), in which case the cycle cannot produce the test that would either confirm or reconsider ADR-011. The forward pointer to Loop 4 and Loop 5 should acknowledge this path explicitly.
- **Recommendation:** Add a sentence in the ADR-011 paragraph acknowledging the sparse-literature path: if the design-principles literature review does not yield concrete principles for a novel ensemble design, the cycle cannot perform the ADR-011 test, and the decision criterion will need to be revisited at that point. This is named in §"What this means for the cycle" (the "if the design-principles literature itself proves sparse" sentence) but the implication for ADR-011 reconsideration is not drawn through. Drawing it through would close a small logical gap between the two sections.

---

### P3 — Consider

**P3-R5-1**

- **Location:** §"What the Spikes Found," scope-condition paragraph; §"Open Empirical Questions," spike-candidate paragraphs
- **Claim:** The §"Open Empirical Questions" section frames Spike B and Spike A as spike *candidates* sketched as questions. The §"What the Spikes Found" section then reports the spikes as completed. The transition between these sections is implicit rather than explicit.
- **Issue:** A reader moving linearly through the essay will encounter the spike candidates framed as "will need to be sharpened into falsifiable hypotheses before the spike battery runs" and then, without a bridging sentence, encounter "both spikes ran." The essay does not record where or when the transition from candidate-to-committed occurred, nor does it note that the spike parameters were modified (Spike A pivoted from the MASS-topology framing to a cascade-vs-prompt-steering comparison after Spike B's finding). A reader unfamiliar with the research log's Step 4c record would have no way to trace this evolution. This is a minor coherence gap, not a logical error.
- **Recommendation:** Add one sentence at the opening of §"What the Spikes Found" that bridges the two sections: e.g., "After the audit cleared the essay, the validation-spike decision committed both candidates with modified parameters — Spike A refocused from a MASS-topology probe to a cascade-vs-prompt-steering comparison in light of Spike B's cascade-plumbing finding." This is housekeeping for narrative continuity, not a logical fix.

---

### Scope-condition discipline: verification

The re-audit's specific charge was to verify that the scope-condition distinction ("existing ensemble does not beat prompt steering" vs "no ensemble design can beat prompt steering") is preserved without collapse in the four named locations. Verdict per location:

- **Abstract:** "the existing code-review ensemble's two-stage summarization design is dominated on every measured axis" — correctly scoped to the existing design.
- **§"What the Spikes Found" scope paragraph:** explicit and load-bearing. The distinction is stated in its strongest form here and is the clearest instance in the essay.
- **§"What this means for the cycle":** "what has been falsified is the assumption that the existing production ensemble design is well-architected against current findings" — correctly scoped.
- **§"Implications for the Architecture" ADR-011 paragraph:** "empirically strengthened on the configurations tested" and "the explicit reservation that no novel ensemble design has been tested" — correctly scoped. The P2-R5-2 issue above is a completeness gap, not a scope-collapse.
- **Conclusion:** "the existing production code-review ensemble does not beat prompt-steering of a strong cloud orchestrator on the code-review task class, on either of the configurations tested" — correctly scoped.

No scope-collapse found in any of the four named locations. The distinction is preserved throughout.

### ADR-011 "empirically strengthened" claim: verification

The claim is qualified correctly in each instance:
- Abstract: "empirically strengthened on the configurations tested at qwen3:8b tier and at cloud-orchestrator tier across two task classes"
- §"Implications for the Architecture": "empirically strengthened by Spike A and Spike B on the configurations tested"
- Conclusion: "ADR-011's single-Model-Profile commitment is empirically strengthened on those configurations across two task classes (capability query, code review) and two orchestrator tiers (qwen3:8b local, MiniMax cloud)"

The configurations are named in every instance. No version of the claim generalizes to "all configurations" or "no ensemble can beat prompt steering." Claim is correctly bounded.

### Forward-pointer to Loop 4 / Loop 5: verification

The essay describes Loop 4 (ensemble design-principles literature) and Loop 5 (novel-design spike) as "the next research loop's work" and "a follow-up empirical spike." It does not claim these loops will confirm the cycle's premise — the posture is "the next test," not "the next confirmation." The conclusion names the Loop 4 possibility that design-principles literature may be sparse and states that this would seed a future cycle rather than constitute the current cycle's failure. This is the correct epistemic register. No overstatement found. The P2-R5-2 issue above is about completeness in the ADR-011 paragraph, not about the forward-pointer's epistemic register.

### Transition coherence: §"Open Empirical Questions" → §"What the Spikes Found"

The §"Open Empirical Questions" section closes with: "The validation-spike decision per ADR-087 fires after this essay clears audit. The cycle's recorded posture is essay-first-then-spikes, with explicit acceptance that spike findings may refute essay claims and warrant essay revision." The §"What the Spikes Found" section opens: "Both spikes ran." This is abrupt. The P3-R5-1 finding covers this gap.

The prior section's framing of the spikes as questions ("once their parameters are committed to falsifiable hypotheses") does not technically contradict the new section's findings — it was written before the spikes ran, and the new section acknowledges the spikes ran. But the pivot from "Spike A" as a "MASS-equivalent multi-turn probe" in §"Open Empirical Questions" to "Spike A as cascade-vs-prompt-steering comparison" in §"What the Spikes Found" is not explained anywhere in the essay. The research log explains it (Spike A's design pivoted on Spike B's finding), but the essay does not. This is the P3-R5-1 gap.

---

## Section 2: Framing Audit

The framing audit revisits the two consequential deferred items from prior rounds (Framing P1-2 tau-bench; Framing P2-1 realignment-as-correction) and evaluates three new items specific to the post-spike revision.

### Question 1: What alternative framings did the evidence support?

**Alternative framing A: "The cascade is harmful; dismantle it."**

The research log's Spike A implications section explicitly names this alternative: "The 'A2 quality > A1 quality at lower latency' outcome materialized — explicitly named at spike opening as the unexpected outcome that would suggest the cascade is actively harmful for this task class." The log then states: "This is a stronger claim than 'cascade is not earning its complexity'; it is 'cascade is removing value.'" The essay's §"Spike A" section describes the mechanism (two-stage summarization as lossy compression) that supports this framing.

The essay does not adopt "cascade is harmful; dismantle it" as its conclusion. Instead it frames the finding as "the existing production code-review ensemble's design does not beat prompt steering on this task class." What would the reader need to believe for the "cascade is actively harmful" framing to be right? That (a) the two-stage summarization mechanism generalizes beyond code review to other task classes, (b) no alternative ensemble design could avoid the lossy-compression failure mode, and (c) the cascade's structural costs (latency, token overhead, compression) always outweigh its structural benefits (parallelism, specialist decomposition, role-specific prompting). The essay correctly does not assert (b) or (c) — neither is tested. The current framing is the more defensible one. No issue.

**Alternative framing B: "The four design priorities are incommensurable; the essay should not rank them."**

The source material (research log's framing-realignment exchange) records the practitioner's statement that priorities "mix" rather than commensurately optimize. The essay already incorporated the non-commensurability qualification in §"The Design Priorities the Cycle Is Actually Navigating" following the round-1 P1-1 fix. The essay now uses "priorities" not "optimization axes" and explicitly states they "do not have units, measurement methods, or thresholds." The alternative framing is acknowledged rather than suppressed.

**Alternative framing C: "The spikes refuted ensemble design, not just this ensemble — the cycle's empirical program is complete."**

This alternative would read the two spikes as having settled the question: prompt steering wins, ensembles lose, no further investigation needed. The evidence in the research log partially supports this (A2 dominated A1 on every measured axis except dimension coverage, which was equivalent), but the scope condition — neither spike tested a novel ensemble designed against current findings — is the explicit limit. The essay's counter-framing is that the spikes falsified the existing design, not ensemble design in general. The essay argues this well. What would the reader need to believe for Alternative C to be right? That (a) any ensemble design will suffer the same lossy-compression failure mode, or (b) the task classes the cycle cares about will never benefit from preserved per-reviewer voice. Neither (a) nor (b) is established by the spikes. The essay's choice to hold open the novel-design question is the more epistemically honest framing.

---

### Question 2: What truths were available but not featured?

**Available but not featured A: Spike A output samples and the A1-clean arm**

The research log's Spike A section includes verbatim output samples from both A1-clean and A2, and describes a methodological wrinkle (A1 via opencode used OpenCode's `glob`+`read` tools rather than llm-orc's `list_ensembles`+`invoke_ensemble`, prompting the A1-clean arm). The essay summarizes the samples selectively — citing A2's `chmod 600` and AES-256-GCM examples but not A1's table rows (which are also recorded verbatim). The A1-clean arm's 145s wall-clock is named in the essay only in the abstract's "71–145s" range for A1; the distinct nature of A1-clean (direct llm-orc serve, README inlined) is not surfaced.

The A1-clean arm matters because it eliminates OpenCode's mixed-tool confound. Without A1-clean, one could attribute A1's underperformance to the orchestrator choosing OpenCode's tools over llm-orc's, rather than to the cascade's compression. A1-clean confirms the cascade ran (server-side artifacts confirm this) and still produced 9 recommendations at 145s. The essay's selective summary does not misrepresent the finding — the conclusion holds whether one reads A1 via opencode or A1-clean — but a careful reader might ask whether the tool-surface confound was ruled out. It was, and noting this would strengthen the essay's evidential standing.

Significance: P3 (does not change the essay's conclusions, but strengthens the evidentiary case for readers who might question the methodology).

**Available but not featured B: Spike A's variance in A2 (15s–70s total)**

The research log records A2's "wide" variance band: "15–70s total, reflecting Zen free-tier scheduling rather than model behavior — the actual generation rate, when scheduled, is ~600–700 chars/sec." The essay gives the median (19.5s) but does not surface the 70s outlier. A reader considering A2's latency advantage might find the 70s outlier consequential: on a bad Zen scheduling day, A2's wall-clock is in the same range as A1 via opencode (71–86s). The median advantage is real, but the distribution has a long tail.

The essay's §"Spike A" paragraph notes that "A2 (~19.5s, 16 specific recommendations) outperformed A1 (~71–145s, 9 recommendations collapsed to table form)" — presenting medians against the full A1 range. A more balanced presentation would give A2's range too (15–70s), allowing the reader to see that A2's worst case overlaps A1's best case on latency. The quality finding (16 vs 9 recommendations, deeper specificity) still holds unconditionally. But the latency advantage is median-dependent, not structural.

Significance: P2 (the latency claim in the abstract and body is stated as though A2 unconditionally beats A1 on latency; the worst-case overlap is material for a practitioner deciding between configurations).

**Available but not featured C: The Spike A A1 recommendation count collapse**

The research log table shows "~9 (in tech-lead synthesis) collapsed to ~5–7 in orchestrator final" for A1 via opencode. The essay uniformly uses "9 recommendations" for A1, which is the A1-clean figure. For A1 via opencode, the final output from the orchestrator was 5–7, not 9 — the second summarization step further collapsed the tech-lead's 9 to 5–7 at the orchestrator layer. The essay's "9" figure may actually understate the compression for the A1-via-opencode arm.

Whether this matters: the essay's argument is that two-stage summarization is lossy. If A1 via opencode actually produces 5–7 final recommendations rather than 9, the A2-vs-A1 comparison (16 vs 5–7) is even more favorable to A2 than the essay presents. The essay using 9 is therefore a conservative presentation of the finding, not an exaggeration. Minor, but worth surfacing.

Significance: P3 (the stated figure is conservative; the actual gap may be larger).

---

### Question 3: What would change if the dominant framing were inverted?

The dominant framing: "the existing production ensemble fails on this task class; novel ensemble designs informed by design-principles literature are worth testing."

**Inverted framing:** "prompt steering has now been empirically validated across two task classes and two orchestrator tiers; the ensemble program should be deprioritized in favor of prompt-engineering investment."

Under the inverted framing:
- The A2 result (16 recommendations at 19.5s) becomes evidence that the cycle's highest-ROI work is improving the system prompt and task specification, not investigating ensemble topologies.
- The design-principles literature gap (named in §"What this means for the cycle") becomes a signal to stop rather than a research direction: if the literature on ensemble design principles is sparse, that may reflect community judgment that the niche is not worth exploring.
- ADR-011's single-Model-Profile commitment, empirically strengthened twice over, becomes a near-settled question rather than a working hypothesis awaiting further test.
- The four design priorities would read as reinforcing the prompt-steering path: A2 wins on performance (better recommendations), environmental cost (no local cascade), local-first preference (indifferent — both use cloud for the final response), and token cost (fewer model calls).

What the essay would need to address under the inverted framing: why the cycle is continuing to invest in ensemble design research rather than treating the question as settled. The essay's answer — that neither spike tested a novel ensemble designed against current findings — is a logically valid rebuttal, but under the inverted framing it looks like scope-inflation: adding new unstated conditions to keep the investigation open when the evidence at hand already favors the simpler path.

The essay does not surface this alternative reading. This connects directly to the deferred Framing P2-1 item (realignment-as-correction) from prior rounds.

---

### Framing Issues

---

**Framing P1-2 (re-evaluation): tau-bench omission**

*Prior status:* Deferred to gate. The prior deferred rationale: tau-bench provides the most directly relevant published baseline for tool-calling reliability ceilings on llm-orc's architecture; not currently in the essay. Decision deferred to gate.

*Has the post-spike revision changed the salience of this omission?*

Yes, and it has increased it. The post-spike essay now claims ADR-011 is empirically strengthened "on the configurations tested" — a strengthening that rests on two spikes using the existing production code-review ensemble on a README task. Tau-bench (Yao et al., 2024, arXiv:2406.12045, source #21 in the lit-review) measured GPT-4o achieving under 50% task success and under 25% pass^8 on the tool-agent-user interaction benchmark. This is the most directly relevant published baseline for the configuration class the essay is strengthening.

The essay's post-spike argument is that prompt-steered cloud orchestration beats cascade ensemble on quality and latency. Tau-bench's finding that even GPT-4o achieves under 50% task success on tool-agent-user interaction tasks is not addressed anywhere in the essay. The gap is consequential in the following way: the essay strengthens ADR-011 on the basis of a code-review README task (single task, well-scoped, no state accumulation, no multi-turn tool exchange). Tau-bench measures exactly the multi-turn tool-agent-user interaction regime the essay's RQ-1 concerns. The essay's own claim — that multi-turn work introduces failure modes the single-ask taxonomy did not surface — makes the tau-bench omission directly relevant to the scope of ADR-011's strengthening.

Stated more precisely: the essay claims ADR-011 is empirically strengthened across "two task classes (capability query, code review)." Both task classes are single-ask or near-single-ask and do not exercise the multi-turn tool-agent-user interaction regime that tau-bench characterizes. Tau-bench's GPT-4o under-50% finding is evidence that the regime the essay most cares about (sustained multi-turn agentic sessions) has published baselines showing substantial failure rates even at frontier tier. Not surfacing this baseline while claiming empirical strengthening of ADR-011 is a consequential omission: the strengthened ADR-011 could mislead a practitioner into confidence about configurations that have not been tested on the task class tau-bench measures.

*Recommendation:* This remains a P1 gate decision. The framing-audit findings do not auto-correct. The gate options are: (a) integrate the tau-bench finding as a scope qualifier on ADR-011's strengthening — "strengthened on single-ask-equivalent task classes; multi-turn tool-agent-user interaction regime, where tau-bench records GPT-4o under 50% task success at frontier tier, is out of scope of both spikes and constitutes an open question"; or (b) accept the omission with explicit rationale that the essay's empirical claims are scoped to the configurations tested and do not speak to the multi-turn regime. Either is defensible; neither is currently in the essay.

---

**Framing P2-1 (re-evaluation): realignment-as-correction**

*Prior status:* Deferred to gate. "The essay's adoption of the four-priorities frame after the practitioner's mid-loop pushback is treated as a correction rather than as adoption of one valid alternative framing among others."

*Has the post-spike revision strengthened or weakened the case for surfacing the agent's initial performance-only synthesis as a valid alternative framing?*

The post-spike revision has modestly weakened the case for surfacing it. The two spikes' findings — that A2 (prompt-steered, no cascade) wins on performance AND latency AND implicitly on environmental cost (fewer model calls, no local cascade) AND on token cost (no tech-lead-synthesizer API call) — mean that the four design priorities do not pull in different directions for the configurations the spikes tested. A performance-only framing would have reached the same conclusion: A2 wins. The four-priorities frame adds texture to the analysis but does not change the recommendation.

However, under the inverted framing in Question 3 above, the performance-only reading would strengthen the argument for treating ADR-011 as effectively settled rather than requiring further novel-ensemble investigation. The essay's four-priorities frame is what creates the opening for "the novel ensemble might co-optimize the priorities differently" — a performance-only frame would close that opening faster. So while the spikes' findings partially defuse the framing-correction concern, the framing still does directional work: it justifies continuing the research program. Surfacing the initial performance-only synthesis as an alternative reading would make that justification visible rather than implicit.

*Recommendation:* This remains a P2 gate decision. The practitioner should decide whether to add a sentence in §"The Design Priorities the Cycle Is Actually Navigating" acknowledging that the framing realignment was adopted from a mid-loop exchange and that the initial performance-only synthesis was not wrong — it was a valid alternative framing that the practitioner's stated motivations supersede rather than refute.

---

**Framing P2-R5-1 (new): A2 latency variance not surfaced**

- **Location:** Abstract and §"Spike A — the production code-review ensemble's design does not beat prompt steering on this task class"
- **Issue:** The abstract states "A2 19.5s median vs A1 71–145s"; the body states A2 "delivered both higher quality AND substantially lower latency (median 19.5 seconds wall-clock vs 71–145 seconds for the cascade)." The A2 variance band (15–70s total, with one trial at 70.3s) is not surfaced. The 70s trial is within the lower bound of A1's range (71s via opencode). A practitioner reading the essay for configuration guidance would understand A2's latency advantage as unconditional; the research log records it as median-favorable but with a long tail attributable to Zen free-tier scheduling variance.
- **Evidence source:** Research log §"Loop Iteration 3 — Spike A," Trials section: "Trial 2: TTFO 22.5s; total 70.3s; 10,598 chars output. (Slow run, attributable to Zen free-tier scheduling variance.)"
- **Consequence:** The quality advantage (16 vs 9 recommendations, materially deeper per-recommendation specificity) is unconditional — A2 wins on quality in all three trials. The latency advantage is median-favorable but not structural: it reflects Zen free-tier scheduling variance, not a consistent property of the comparison. Presenting the latency advantage without the variance range overstates the reliability of the latency finding.
- **Recommendation:** In the abstract and the §"Spike A" body paragraph, qualify the latency comparison: "A2 (median 19.5s, range 15–70s reflecting Zen free-tier scheduling variance) vs A1 (71–145s)." The quality finding needs no qualification; the latency finding needs the range. This is a content-representation issue, not a logical error: the medians are accurately stated, but the distribution matters for a practitioner making a configuration choice.

---

**Framing P3-R5-1 (new): A1-clean arm and its methodological role**

- **Location:** §"Spike A — the production code-review ensemble's design does not beat prompt steering on this task class"
- **Issue:** The A1-clean arm (direct llm-orc serve, README inlined) was added specifically to rule out the OpenCode mixed-tool confound — the concern that A1 via opencode underperformed because the orchestrator chose OpenCode's `glob`+`read` tools rather than llm-orc's `list_ensembles`+`invoke_ensemble`. A1-clean produced 9 recommendations at 145s, confirming that the cascade ran in all trials regardless of tool surface, and confirming that the recommendation-count deficit is attributable to the cascade's compression, not to the tool-surface confound. The essay does not name A1-clean or explain why three arms were run rather than two.
- **Evidence source:** Research log §"Loop Iteration 3 — Spike A," Method section: "The A1-clean variant was added mid-spike on observation that OpenCode's mixed tool surface caused the orchestrator to use OpenCode's `glob`+`read` tools rather than llm-orc's `list_ensembles`+`invoke_ensemble`."
- **Consequence:** Minor. A reader who asks "why was A1-clean run?" has no answer in the essay. The finding is robustly supported even without A1-clean (the quality difference is clear from A1 via opencode), but the A1-clean arm's purpose was to rule out a confound, and its absence from the essay means the confound-ruling is invisible. This does not affect the essay's conclusions; it is a transparency gap in the methodology description.
- **Recommendation:** Add one sentence naming A1-clean and its purpose: a direct-curl arm was run to confirm the cascade engaged regardless of client tool surface, ruling out the OpenCode mixed-tool confound as an explanation for A1's underperformance. This is a minor framing improvement that would strengthen methodological confidence for a technical reader.

---

### Framing Issues Summary

| ID | Severity | Status | Description |
|----|----------|--------|-------------|
| Framing P1-2 | P1 (gate) | Re-evaluated: salience increased | Tau-bench omission is more consequential now that ADR-011 is claimed empirically strengthened; the essay's strengthened configurations don't cover the multi-turn tool-agent-user regime tau-bench measures |
| Framing P2-1 | P2 (gate) | Re-evaluated: slightly weakened but still valid | Realignment-as-correction; spikes' findings partially defuse the concern but the framing still does directional work justifying the research program |
| Framing P2-R5-1 | P2 (new) | Open | A2 latency variance (15–70s) not surfaced; latency advantage stated as median without the distribution |
| Framing P3-R5-1 | P3 (new) | Open | A1-clean arm's methodological purpose not explained in the essay |
