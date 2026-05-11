# Argument Audit Report

**Audited document:** `docs/agentic-serving/essays/003-multi-turn-orchestration-and-the-four-axis-frame.md`
**Source material:**
- `docs/agentic-serving/essays/research-logs/research-log.md` (§"Loop Iteration 2 — Spike B", §"Loop Iteration 3 — Spike A")
- Prior audit: `docs/agentic-serving/housekeeping/audits/argument-audit-cycle-2-round5-post-spike.md`
**Date:** 2026-04-29
**Round:** 6 (verification audit — confirming four round-5 fixes hold; checking for new issues introduced by the revisions)

---

## Section 1: Argument Audit

### Summary

- **Argument chains mapped:** 13 (all chains from prior rounds, no new chains introduced)
- **Issues found:** 0

### Fix verification

Each of the four round-5 fixes was verified against the current essay text and the research log.

**P1-R5-1 (13× ratio arithmetically wrong) — FIXED, holds cleanly.**

The essay now reads: "a roughly 9× reduction in reviewer model size on disk (4.7 GB → 522 MB; equivalently a ~13× reduction in parameter count from 8B to 0.6B)." Both figures are present, correctly labeled with their respective units (disk size vs parameter count), and the parenthetical "(4.7 GB → 522 MB)" matches the 9× disk-size ratio rather than the 13× parameter-count ratio. The research log carries the same corrected text at §"Loop Iteration 2 — Implications for essay 003." The two figures are now internally consistent and the distinction between them is explicit. No residual discrepancy.

**P2-R5-1 ("premise not falsified" framing) — FIXED at both appearance sites, holds cleanly, no new tension between them.**

First appearance (§"What this means for the cycle"): "well-architected processes achieve good results" is closer to definitionally true than empirically meaningful. The empirically meaningful question is whether the cycle's process for arriving at ensemble designs is well-architected — whether design-principles-informed novel designs will actually clear the bar. The honest position is therefore "the existing design failed empirically; whether a design-principles-informed alternative succeeds is an untested hypothesis," not "the cycle is on track."

Second appearance (§"Conclusion"): "the analytic form of the premise is close to definitional. The empirically meaningful version is whether the cycle's process for arriving at ensemble designs is well-architected — whether the existing production design's failure here generalizes to novel designs informed by current findings, or whether design-principles-informed alternatives can clear the bar. The existing design has failed empirically; the alternative is an untested hypothesis."

Both instances carry the same epistemic content: analytic form is near-definitional, empirically meaningful form is untested, honest position is "existing design failed; alternative is untested hypothesis." The phrasing differs slightly between the two but the substance is identical and non-contradictory. The abstract carries a shorter version ("has not been falsified by the spikes — what has been falsified is the assumption that the existing production ensemble design is well-architected against current findings") which is compressed appropriately for an abstract — it does not carry the definitional/empirical qualifier but does not contradict it.

**P2-R5-2 (ADR-011 reconsideration criterion omits sparse-literature path) — FIXED, holds cleanly.**

The §"Implications for the Architecture" ADR-011 paragraph now states the criterion explicitly as two named branches. Branch one: literature is rich enough + spike produces positive evidence → criterion fires → ADR-011 reconsidered at synthesis. Branch two: literature too sparse to inform a principled novel design → criterion stays unfired, cycle's empirical evidence is bounded by what was actually tested, open territory graduates to a future cycle on test-and-evaluate methodology. The §"Conclusion" and §"What this means for the cycle" both corroborate this framing (the "if the design-principles literature itself proves sparse" sentences in both sections). No gap between the ADR-011 paragraph and the forward-pointer sections.

**P3-R5-1 (Spike A pivot from MASS-topology probe to cascade-vs-prompt-steering unexplained) — FIXED, more fully than required.**

The Spike A subsection now opens with a full bridging paragraph explaining: (1) the original framing as a MASS-equivalent topology-delta probe at qwen3:8b on multi-turn coding sessions; (2) Spike B's finding that cascade plumbing dominates total wall-clock regardless of inner-model choice; (3) the consequence — the relevant comparison shifted from "does topology help inside the cascade" to "does the cascade itself help compared to no cascade at all"; (4) Spike A's refocused question. This is coherent with the §"Open Empirical Questions" framing and closes the narrative gap identified in P3-R5-1. A reader moving linearly through the essay now has a complete account of why Spike A ran differently from how it was specified.

### New issues from the round-5 revisions

The two premise-qualifier instances, the two-branch ADR-011 criterion, and the Spike A bridging paragraph were read for internal consistency and for any hidden assumptions introduced by the new prose.

No new issues found. The two premise-qualifier appearances are consistent and non-redundant: the §"What this means for the cycle" instance names concrete alternative ensemble designs that have not been tested (preserved per-reviewer voice, alternative coordination protocol, design-principles-informed role decomposition); the §"Conclusion" instance sets up the load-bearing distinction for what follows. Neither instance overstates what the cycle has established, and neither understates the epistemic openness of the untested-hypothesis path.

The two-branch ADR-011 criterion does not introduce hidden assumptions. The preconditions for each branch are stated explicitly and are internally consistent with the essay's forward pointers in §"What this means for the cycle" and §"Conclusion."

### P1 — Must Fix

None.

### P2 — Should Fix

None.

### P3 — Consider

None.

**Verdict: clear.** Zero issues at any severity. The four round-5 fixes hold cleanly and no new issues were introduced by the revisions.

---

## Section 2: Framing Audit

The four framing-audit items deferred from prior rounds (Framing P1-2, Framing P2-1, Framing P2-R5-1, Framing P3-R5-1) were declared as practitioner-gate decisions, not unresolved errors. Per the round-6 audit brief, they are not re-flagged here. This section checks only whether the round-5 revisions introduced new framing concerns.

### Question 1: What alternative framings did the round-5 revisions support that the essay did not adopt?

The round-5 revisions added: (a) the 9×/13× disambiguation paragraph; (b) the two premise-qualifier passages; (c) the two-branch ADR-011 criterion; (d) the Spike A bridging paragraph. None of these additions introduce new framing choices — they are scope qualifications on existing claims, not reframings of the essay's central argument.

One implicit framing choice in the premise-qualifier prose is worth naming as a structural observation, not a finding. Both qualifier passages frame the untested-hypothesis path as "the next research loop's question" — they invite the reader toward continued investigation. An alternative framing would be to read the same qualifier as a signal to pause the ensemble-investigation program until the design-principles literature is confirmed non-sparse. The essay does not adopt that alternative, and the research log's Spike A implications (which explicitly distinguish "the cascade is actively harmful for this task class" from "no ensemble design succeeds") support the essay's more open framing. No issue.

### Question 2: What truths available in the round-5 source material are absent from the revised essay?

The four deferred framing items (tau-bench, realignment-as-correction, A2 latency variance, A1-clean methodology) remain absent and remain gate decisions. No new absences introduced by the round-5 revisions.

One observation: the two-branch ADR-011 criterion names "future cycle on test-and-evaluate methodology" as the Branch Two outcome, but the essay does not name what that methodology would look like. The research log does not supply this either — it is genuinely open. This is not a truth available but not featured; it is an open question the essay correctly scopes as future work.

### Question 3: What would change if the dominant framing of the round-5 additions were inverted?

The premise-qualifier additions establish the dominant framing of the round-5 revisions: "existing design failed; alternative is untested hypothesis; the next research loop tests it." The inverted framing would read: "both spikes found prompt steering wins; the alternative-design hypothesis has no tested support; the investigation is speculative at this stage."

Under the inverted framing, the two-branch ADR-011 criterion's Branch One would read as conditional on an investigation the current evidence does not justify. The essay's case against this inversion rests on the scope-condition it maintains throughout — neither spike tested a novel ensemble designed against current findings. This is a logically sound rebuttal, and the round-5 premise-qualifier additions strengthened it rather than weakened it by being explicit that the alternative-design path is "an untested hypothesis" rather than a likely success.

### Framing Issues

No new P1, P2, or P3 framing issues specific to the round-5 revisions.

**The four deferred gate items remain open gate decisions:**

| ID | Severity | Status |
|----|----------|--------|
| Framing P1-2 | P1 (gate) | Deferred — tau-bench omission; salience increased post-spike (noted in Round 5 audit) |
| Framing P2-1 | P2 (gate) | Deferred — realignment-as-correction framing |
| Framing P2-R5-1 | P2 (gate) | Deferred — A2 latency variance (15–70s) not surfaced in abstract or body |
| Framing P3-R5-1 | P3 (gate) | Deferred — A1-clean arm's methodological purpose not explained |

These are carried forward as gate decisions, not audit findings. The round-6 audit introduces no new framing items.

---

**Overall verdict: ready for the epistemic gate.** Argument audit is clear (0 issues). Framing audit introduces no new items. The four deferred gate decisions are the remaining open questions for practitioner resolution before the cycle closes.
