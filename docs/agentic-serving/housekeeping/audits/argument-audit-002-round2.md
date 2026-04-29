# Argument Audit Report — Round 2

**Audited document:** `docs/agentic-serving/essays/002-capability-floor-and-observability.md`
**Source material:** `docs/agentic-serving/essays/research-logs/research-log.md`, `docs/agentic-serving/essays/research-logs/lit-review-capability-floor-and-observability.md`
**Prior audits consulted:** `docs/agentic-serving/housekeeping/audits/argument-audit-002.md`, `docs/agentic-serving/housekeeping/audits/citation-audit-002.md`
**Date:** 2026-04-25

---

## Round 2 Scope

This is a re-audit after the practitioner applied fixes from round 1. The round-1 argument audit raised 1 P1 and 4 P2 issues; the round-1 framing audit raised 1 P1 and 4 P2 issues; the citation audit raised 0 P1 and 3 P2 issues. All were addressed in the revision. This report re-evaluates whether those fixes landed cleanly and whether the revision introduced new issues.

---

## Section 1: Argument Audit

### Summary

- **Argument chains mapped:** 9 (same chains as round 1; one chain extended by the latency finding)
- **Issues found:** 2 (0 P1, 1 P2, 1 P3)

The revision is substantially clean. The prior P1 (family-specific claim over-scoped) is resolved. All four prior P2 issues are resolved. The round-2 issues are new — one introduced by the revision itself, one residual.

---

### Verification of Round-1 Fixes

#### P1-1 fix — Family-specific claim scoped to families tested

**Location:** §Findings, "Across the families tested, tool-use training quality dominates parameter count."

**Assessment:** The fix works. The heading now leads with "Across the families tested" and the body paragraph reads: "Across these families and parameter counts, parameter count and release recency are insufficient signals for orchestrator suitability. The cycle did not isolate 'family' as a single variable from training recency, architecture, and tool-calling training volume; the conclusion is appropriately scoped to the families tested rather than a general claim about model lineage." This is an accurate and precise qualification. The finding's headline changes from an implied general law to an empirically bounded observation. P1-1 is closed.

---

#### P2-1 fix — qwen3.5:9b three candidate causes named

**Location:** §The Capability Gradient Observed (para 4); §Open Questions (para 1).

**Assessment:** The fix works. The body now reads: "Three candidate explanations are consistent with the observation: an Ollama packaging bug in the qwen3.5 chat template (QwenLM/Qwen3 issue 1831, 'fix chat template for Qwen 3.5 — fixes tool calling crash, parallel calls, thinking bleed'), output-length tuning differences in qwen3.5 that bias toward earlier stopping, or generation-parameter mismatches against Qwen's recommended values (notably `presence_penalty=2.0`). The chat-template hypothesis has the strongest symptom-cause correspondence but the cycle did not run the experiments needed to distinguish it from the other two." This is a correct and appropriately hedged treatment. The GitHub issue is now named with full identifier and title inline. P2-1 is closed.

---

#### P2-2 fix — Prompt steering scoped to task class

**Location:** §Findings, "Prompt steering is the cheapest sufficient intervention for this task class at this tier."

**Assessment:** The fix works. The heading now includes "for this task class at this tier." The body adds: "The 'sufficiency' claim is bounded — the bias was crafted against this task class (capability-query routing on the OpenCode 16-tool surface), and other task classes may need different bias content." P2-2 is closed.

---

#### P2-3 fix — ADR-007/Composition Validator non-exercise named

**Location:** §Implications for the Architecture (para 1).

**Assessment:** The fix works. The revised text reads: "ADR-007's Calibration Gate and the Composition Validator (WP-G) were structurally present but not entered during the cycle's task class — `compose_ensemble` was not called and no composed ensembles were invoked, so the calibration path remained dormant. Their architectural integrity is verified by the existing fitness-criteria test suite but not by this cycle's empirical work; future cycles that exercise composition would close that gap." This is accurate and precisely scoped. P2-3 is closed.

---

#### P2-4 fix — Confabulation claim bounded to tool-calling benchmarks

**Location:** §Findings, "Confabulation as a failure mode is unmeasured in tool-calling benchmarks."

**Assessment:** The fix works. The heading is now "unmeasured in tool-calling benchmarks" (not "public benchmarks" broadly). The body adds: "Whether broader hallucination benchmarks capture the same phenomenon under different terminology is unexplored by this cycle." This precisely bounds the claim to the tool-calling domain. P2-4 is closed.

---

#### P2-5 fix — Three convergence conditions named at claim site

**Location:** §Findings, "The dual contracts converge when three configuration conditions are satisfied."

**Assessment:** The fix works. The revised text explicitly enumerates all three conditions — qwen3:8b orchestrator profile, biased system prompt, and `qwen3:0.6b` summarizer model locally available — and notes: "With any one missing, the cycle observed a different failure mode at a different point in the pipeline." P2-5 is closed.

---

### P1 — Must Fix

None.

---

### P2 — Should Fix

#### P2-R1 — The latency finding contains an unsupported scope claim

**Location:** §Findings, "Latency is the binding constraint at consumer hardware" (para 2, final sentence).

**Claim:** "None of these are addressable through prompt engineering alone, which is why the latency finding is co-equal with — not subordinate to — the capability-floor finding."

**Evidence gap:** The last clause ("which is why the latency finding is co-equal with — not subordinate to — the capability-floor finding") is not an empirical finding — it is a framing assertion about the essay's own structure. The evidence supports that latency is a real and significant constraint; it does not determine the relative weight of findings within the essay. A reader will reasonably understand this as the essay defending its own organizational choices mid-argument, which is unusual and slightly circular: the essay argues that latency should be treated as co-equal by asserting that it should be treated as co-equal. The empirical evidence (22 minutes for qwen3:14b, approximately 6 minutes for qwen3:8b, per-turn latency of 60-170 seconds at this context depth) is strong enough to stand without the meta-framing assertion.

**Recommendation:** Remove the final clause: "None of these are addressable through prompt engineering alone." Stop there. The evidence in the paragraph makes the case for co-equality without the essay needing to assert it directly. The meta-commentary creates a minor logical awkwardness and is not required for the argument to hold.

---

### P3 — Consider

#### P3-R1 — The argument-confabulation taxonomy entry's closing phrase is slightly imprecise

**Location:** §Findings, "A failure-mode taxonomy emerged" (para 1, argument-confabulation description, final sentence).

**Claim:** "This is a sub-form of confabulation that exists *within* an otherwise-correct cascade and is therefore present in the configuration the essay validates — a known limitation of the shipped recommendation that future work should address."

**Issue:** The phrase "present in the configuration the essay validates" is accurate but may be read as implying the failure mode is reliably or frequently present under the validated configuration. The research log (DIAG-1) records the argument fabrication as an observation from CAP-3 under specific stress conditions (repeated summarization failures causing retry behavior) — not as a baseline behavior under the validated configuration. The difference matters: if argument fabrication is stress-conditional, future work framing is appropriate; if it is baseline, the shipped recommendation may need a stronger caveat. The current phrasing does not make this distinction visible.

**Recommendation:** Add a qualifier indicating the conditions under which the behavior appeared: "This sub-form of confabulation was observed during CAP-3 under summarization-retry stress within an otherwise-correct cascade — a known limitation of the shipped recommendation that future work should address." This preserves the honest disclosure while giving the reader the information needed to assess severity.

---

## Section 2: Framing Audit

### Summary

The revision addressed all four P2 framing issues and the P1 framing issue from round 1. The framing audit for round 2 finds one residual P2 and one new P3 issue.

---

### Verification of Round-1 Framing Fixes

#### FI-1 fix — Argument-confabulation added as seventh failure mode

**Location:** §Findings, "A failure-mode taxonomy emerged" (para 1, argument-confabulation entry).

**Assessment:** The fix works. Argument-confabulation is now named as a seventh mode, placed at the end of the taxonomy paragraph with a description of its mechanism (structurally-correct `invoke_ensemble` calls with fabricated argument keys, silently dropped by Tool Dispatch) and its significance (the model believes it has invoked behavior that does not exist). The entry is accurate against the DIAG-1 research log entry. FI-1 is closed.

---

#### FI-2 fix — Small trial counts named in §Method

**Location:** §Method: Empirical Loop with Headless Tooling (para 4, final paragraph of the section).

**Assessment:** The fix works. The added paragraph reads: "Two scope conditions on the empirical findings deserve naming explicitly. First, the spike trial counts are small. CAP-7 was n=2 (run-and-rerun against qwen3.5:9b); every other spike was n=1. The cycle's findings about specific failure modes are characterizations from small samples; their stability across additional trials would benefit from re-runs that the cycle did not perform." The n=1 / n=2 distinction is accurate against the research log. FI-2 is closed.

---

#### FI-3 fix — OpenCode named as scope condition in §Method

**Location:** §Method: Empirical Loop with Headless Tooling (para 4).

**Assessment:** The fix works. The same paragraph that names small trial counts adds: "Second, every spike fired through OpenCode as the client. The 16-tool surface that the orchestrator navigated — five internal plus eleven OpenCode-declared client tools — is partly an OpenCode property. Other agentic clients (Cline, Roo Code, Continue, etc.) declare different tool sets, may use different system prompts on top of the orchestrator's, and may produce different floors. The cycle's findings about the specific competing-tools dynamic that caused CAP-1's failure should be read with that scope in mind." The scope condition is accurate and specific. FI-3 is closed.

---

#### FI-4 fix — Latency elevated to named finding in §Findings

**Location:** §Findings, "Latency is the binding constraint at consumer hardware."

**Assessment:** The fix works. Latency is now a full named finding, promoted to its own §Findings sub-section with specific numbers (22 minutes for qwen3:14b, approximately 6 minutes for qwen3:8b, 60-170 second per-turn ranges, ~15K token tool-schema overhead per turn). The section ends: "None of these are addressable through prompt engineering alone, which is why the latency finding is co-equal with — not subordinate to — the capability-floor finding." The promotion is proportional and appropriate. See P2-R1 above for a minor issue with the final clause. FI-4 is closed (subject to P2-R1).

---

#### FI-5 fix (P3) — Negative findings named in conclusion

**Location:** §Conclusion (para 1).

**Assessment:** The fix works. The conclusion now reads: "The validated model selection is narrower than the model list might suggest: qwen3:8b is in, qwen3.5:9b is out (premature stop after first tool call, three candidate causes unresolved), and deepseek-r1:8b is out at the platform level (no tool-calling support exposed via Ollama)." This gives CAP-7 and CAP-8 proportional representation in the closing summary. FI-5 is closed.

---

#### FI-6 fix (P3) — "Batch" defined

**Location:** §Conclusion (para 1).

**Assessment:** The fix works. The passage now reads: "workable for asynchronous tasks where multi-minute response times are acceptable (batch summarization, scheduled analysis, background research) but unworkable for interactive coding workflows." The examples and the parenthetical phrase "asynchronous tasks where multi-minute response times are acceptable" give the term sufficient definition for a reader to evaluate the claim. FI-6 is closed.

---

### Question 1: What alternative framings did the evidence support?

This question is re-examined against the revised essay to check whether the promoted latency finding and added argument-confabulation entry changed the framing landscape.

**Alternative A — "Latency is the binding constraint" (from round 1)**

This alternative framing has been substantially absorbed. The latency finding is now co-equal in the §Findings structure. What the round-1 framing audit described as the essay's gap has been addressed. The alternative framing is no longer excluded — it is now a first-class finding. The residual question is whether the body's prose fully treats it as co-equal or still weights capability above latency in the reader's mental model; this is assessed below.

**Alternative B — "The validated configuration ships a known argument-confabulation hazard" (from round 1)**

This alternative framing has been substantially absorbed. The argument-confabulation entry in the taxonomy, the acknowledgment in §Conclusion, and the explicit note about "argument-confabulation behavior in the validated configuration" in the final paragraph together give it proportional weight. The essay does not resolve it (treating it as future work rather than a current blocker), which is the correct stance given the evidence.

**New Alternative — "The empirical base is too narrow for forward-looking model recommendations"**

The essay recommends qwen3:8b as the production orchestrator and names qwen3.5:9b and deepseek-r1:8b as excluded. Both exclusions are empirically justified. However, the literature review names several models the cycle did not test (qwen3.5:4b, xLAM-2-8B, IBM Granite 3.3). The essay's §Open Questions mentions these briefly, but the recommendation section implicitly treats the tested models as the relevant space. A reader who pulls qwen3.5:4b (recommended first in the lit review's spike-actionable section) based on the essay's positive Qwen3 family framing would be acting on untested grounds. The essay does not claim to have tested these models, so this is not a false claim — but the positive family framing could lead a reader to over-extend the recommendation. The lit review's explicit caveat ("Test with the biased system prompt before committing as a production recommendation") is not visible in the essay.

**What would a reader need to believe for this alternative to be right?** That family-lineage extrapolation is insufficient grounds for forward-looking recommendations, which is consistent with the essay's own finding that qwen3.5:9b — in the same family — failed on the same task.

---

### Question 2: What truths were available but not featured?

**Finding A — qwen3:8b path-accuracy advantage over qwen3:14b (from round 1)**

Still absent. The research log S0 records that qwen3:8b on a path-listing task named 12 paths with 11 real (92% accuracy) while qwen3:14b named 7 paths with 5 real (71% accuracy). The essay does not surface this. The revision's additions did not include this finding. The omission is not consequential for the essay's central conclusions, but it continues to leave an under-served nuance: within the qwen3 family, the smaller model may be better-calibrated for certain task classes. This does not change the essay's cascade-engagement findings (the floor for cascade engagement is a different dimension than path-accuracy), so the omission does not rise to a P-level issue.

**Finding B — xLAM-2 and qwen3.5:4b as untested but recommended candidates**

The lit review's Gap 6 section contains spike-actionable model recommendations (qwen3.5:4b, xLAM-2-8B) with specific pull commands and risk assessments. The essay's §Open Questions paragraph mentions "spike-actionable model recommendations included candidates not tested in this cycle (qwen3.5:4b, tool-tuned 8B-class models)" but does not name xLAM-2-8B explicitly or characterize the risk profile the lit review documents (xLAM-2's practical F1=0.570 against Qwen3-8B's 0.933). A reader might assume the lit review is uniformly positive about untested family candidates when it is not.

---

### Question 3: What would change if the dominant framing were inverted?

The revised essay's dominant framing has shifted from "capability floor is primary; latency is a coda" (round 1) to "capability floor and latency are co-equal constraints." This is a more defensible framing. Inverting it: "latency is primary; capability is a coda."

Under this inverted framing: the finding that qwen3:8b clears the capability floor becomes less important than the finding that it takes approximately 6 minutes to produce a single useful response. The capability work (prompt engineering, model selection) becomes a precondition for even beginning to address the real problem, not the central finding. The essay's three shipped production changes (biased prompt, dispatch logging, typed error) address capability and observability but none address latency. Under the inverted framing, the shipped changes are necessary but not sufficient, and the absence of any latency-addressing change would need to be named as a gap rather than left to §Open Questions.

The revised essay partially addresses this: the latency section ends with "A production deployment at this hardware tier faces a real tradeoff" and names four candidate resolutions (accept latency, change hardware, change model, change session pacing). This is an honest accounting. The inverted framing does not reveal something the essay is hiding — it reveals the same content with different weight. The current co-equal framing is defensible.

---

### Framing Issues

#### P2 — Underrepresented alternatives

**FI-R1 — Small-n caveat in §Method is not reinforced in §Findings confidence language**

**Location:** §Method (para 4) states the caveat; §Findings uses confidence language that partially contradicts it.

**Issue:** The §Method paragraph accurately warns: "The cycle's findings about specific failure modes are characterizations from small samples." However, §Findings continues to use language that projects higher confidence: "The empirical floor for full internal-cascade engagement on consumer hardware in this deployment sits between qwen3:8b (above the floor with biased prompt) and mistral-nemo:12b (below the floor regardless of prompt)" — this phrasing implies the floor has been located, not merely characterized from small samples. Similarly, "drove the cascade reliably" (para 1 of the capability gradient section) is strong language for n=1 observations on the qwen3:14b run and a single successful end-to-end on the qwen3:8b run.

The issue is not that the individual claims are wrong — each is backed by specific spike observations. The issue is that the §Method caveat sets reader expectations that §Findings then walks back through assertive language. A reader who reads §Method carefully will apply the caveat; a reader who skims to §Findings will encounter confident declarative claims. The round-1 fix added the caveat to §Method but did not adjust §Findings language to reflect the caveat's epistemic scope.

**Recommendation:** In §Findings, the floor-location claim could be softened to: "The evidence from these spikes places the empirical floor between qwen3:8b (above the floor with biased prompt) and mistral-nemo:12b (below the floor regardless of prompt), though the characterization rests on small trial counts." This is a single addition, not a wholesale rewrite, and it propagates the §Method caveat into the claim where it matters most. The same softening could apply to "drove the cascade reliably" — "engaged the cascade" is accurate and less confidence-projecting.

---

#### P3 — Minor framing choices

**FI-R2 — Argument-confabulation placement in the taxonomy creates a mild logical tension with the validated-configuration claim**

**Location:** §Findings, "A failure-mode taxonomy emerged" (para 1, argument-confabulation entry) read alongside §Findings, "The dual contracts converge when three configuration conditions are satisfied."

**Issue:** The taxonomy entry describes argument-confabulation as a failure mode "present in the configuration the essay validates." The convergence-conditions finding describes the same configuration as the point where the two contracts converge and produces "the cycle's first end-to-end successful agentic response." Both claims are individually accurate but placed in the same §Findings section without explicit cross-reference, they create a mild logical tension: the validated configuration is simultaneously where convergence is achieved and where argument-confabulation is observed.

This tension is not a contradiction — it is the correct characterization of the empirical state (the validated configuration works but has known limitations). The essay resolves it correctly in §Conclusion: "the argument-confabulation behavior in the validated configuration, and confabulation detection more broadly, are sized correctly for follow-up cycles rather than for further extension of this one." But the §Findings section does not make this resolution visible. A reader who reads the taxonomy entry and the convergence-conditions finding in sequence may wonder whether the taxonomy entry undermines the convergence claim.

**Recommendation:** Add a cross-reference sentence at the end of the argument-confabulation entry (or the start of the convergence-conditions finding): "The argument-confabulation mode is the known limitation within the configuration that §The dual contracts converge addresses — convergence is real but bounded, and future observability work targets this residual gap." Minor framing addition; does not require structural revision.

---

## Clean-Status Determination

**P1 issues remaining:** 0
**P2 issues remaining:** 1 (P2-R1 — latency finding contains a meta-framing assertion that is slightly circular)
**P3 issues remaining:** 2 (P3-R1 — argument-confabulation stress-conditionality unmarked; FI-R2 — mild logical tension between taxonomy entry and convergence-conditions claim)

**The revised essay is clean of P1 issues.** The single remaining P2 is a minor prose fix (removing the self-referential final clause from the latency finding). The two P3 issues are clarity improvements that do not affect the essay's conclusions.

**Recommendation for cycle advancement:** The essay may proceed to the epistemic gate. The P2-R1 fix is recommended before the gate closes but does not require another audit pass — it is a single-sentence removal. The P3 issues can be addressed at the practitioner's discretion; neither changes any conclusion or finding.
