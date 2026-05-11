# Argument Audit Report

**Audited document:** `docs/agentic-serving/essays/003-multi-turn-orchestration-and-the-four-axis-frame.md`
**Source material:** Round-7 findings as specified in audit brief (round-7 audit file not present in filesystem; five fixes reconstructed from task description)
**Prior clean audit:** `docs/agentic-serving/housekeeping/audits/argument-audit-cycle-2-round6.md`
**Date:** 2026-04-29
**Round:** 8 (verification audit — confirming five round-7 fixes hold; checking for new issues introduced by revisions)

---

## Section 1: Argument Audit

### Summary

- **Argument chains mapped:** 13 (no new chains introduced by round-7 revisions)
- **Issues found:** 0

### Fix verification

Each of the five round-7 fixes was verified against the current essay text.

**P2-R7-1 (MQ-1 Branch One overstatement) — FIXED, holds cleanly, new prose internally consistent.**

The phrase "MQ-1's Branch One has now fired in a partial form" is absent. The replacement paragraph (§"Implications for the Architecture," the ADR-011 block) reads correctly: "MQ-1's strong-pass condition (A3 ≥ A2 on specificity AND adds factual grounding AND latency acceptable for chat-loop interaction) did not fully fire in Spike A3 — A3's latency is 4× A2's median and R1-Hunyuan's format penalty meant per-recommendation specificity was at content parity rather than format parity, so the Branch One reconsideration trigger is not unambiguously activated."

The three-condition AND-conjunction logic is correctly applied. The paragraph identifies which conditions failed (latency; format-level specificity parity) and which fired (factual grounding), and the "not unambiguously activated" hedge accurately represents that mixed result without understating it.

The new prose then claims three specific capabilities A3 demonstrated: script-agent factual grounding (verified link counts, confirmed section presence, code-block parseability); heterogeneity-uncorrelated-errors producing 5–8 distinct findings per reviewer with low overlap; and the ADR-011 ceiling refinement. All three claims were verified against §"Spike A3" — they match exactly. The paragraph introduces no new claims beyond what the spike sections establish.

The paragraph's conclusion ("ADR-011 should therefore be read as the right default for tasks where prompt-steering of a single capable orchestrator suffices, but not as an upper bound on what the architecture can deliver") is consistent with the Abstract, §"What this means for the cycle," and §"Conclusion." No contradiction between the MQ-1 paragraph and any other section.

**P2-R7-2 (prospective-tense dissonance in §"Capability-Tier Gap" and §"Open Empirical Questions") — FIXED, transitional notes coherent with surrounding text.**

Both notes are present and correctly placed. §"Capability-Tier Gap" opens with: "*Note: this section was written before the cycle's spikes ran; the spike findings appear in §'What the Spikes Found' below. The framing here is preserved as the essay's pre-spike orientation toward the territory.*" §"Open Empirical Questions" opens with: "*Note: this section was written before the cycle's spikes ran. It records the empirical questions the literature did not reach and the spike candidates' pre-run framing. The actual spike outcomes are reported in §'What the Spikes Found' below.*"

Both notes are set off in italics and do not insert themselves into the analytical flow. A reader moving linearly now has explicit orientation before encountering prospective-framed prose. The §"Capability-Tier Gap" note is coherent with the section's closing sentence ("The cycle's recorded posture is essay-first-then-spikes, with explicit acceptance that spike findings may refute essay claims and warrant essay revision"), which retrospectively confirms the posture was held. No incoherence with surrounding voice.

**P2-R7-3 (unattributed "wild ideas" framing) — FIXED, attribution clean.**

The §"Conclusion" paragraph now reads: "The candidate compositional shapes named at cycle entry as speculative seeds (recorded in `housekeeping/cycle-status.md` §'Compositional shape axis': small-model swarms, semantic-routed ensembles, biologically-inspired collective-intelligence patterns, ensembles-of-ensembles) pointed at the territory the published literature does not yet describe." The attribution is explicit, the list of four candidates is named, and the "speculative seeds" framing is less dismissive and more accurate than the prior colloquial label. Clean.

**P3-R7-1 (URL count parenthetical) — FIXED, internally consistent.**

§"Spike A3" now reads: "a verified count of 17 external URLs all returning 2xx/3xx (plus one localhost loopback URL flagged separately at line 526 as a documentation example, not counted toward the 17)." The 17-figure is now unambiguously scoped to external URLs, the localhost URL is correctly excluded with an explanation, and the parenthetical is self-contained. No residual ambiguity.

**P3-R7-2 (recursive composition qualifier) — FIXED, qualifier well-scoped.**

The §"Conclusion" ensembles-of-ensembles forward-pointer now reads: "...without re-introducing the collapse problem at the meta level (provided the meta-aggregation step also avoids a synthesizer collapse, which is a design choice not an automatic property of the recursive structure)." The qualifier correctly names the risk, frames it as a design choice, and distinguishes it from an automatic property of the topology. It does not overstate the risk or the mitigation. Clean.

### New issues from the round-7 revisions

The five fix locations and their surrounding prose were read for hidden assumptions, overstatements, and contradictions with the established evidence base.

No new issues found. The MQ-1 paragraph — the substantive new prose of round-7 — is internally consistent, corroborated by §"Spike A3" at every factual claim, and consistent with all prior characterizations of ADR-011's status in the Abstract, §"What this means for the cycle," and §"Conclusion."

### P1 — Must Fix

None.

### P2 — Should Fix

None.

### P3 — Consider

None.

**Verdict: clear.** Zero issues at any severity. All five round-7 fixes hold cleanly. No new issues introduced by the revisions.

---

## Section 2: Framing Audit

Per the audit brief, the framing-audit deferred items from prior rounds are gate decisions and are not re-flagged in this round. This section confirms only that the round-7 revisions introduced no new framing concerns.

The five round-7 changes are scope qualifications and attribution corrections, not reframings of the essay's central argument. None of them introduce new framing choices that would require framing-audit consideration. The MQ-1 paragraph's "not unambiguously activated" language, if anything, slightly narrows the essay's positive framing of A3 relative to prior draft states — it is a conservative framing move, not an expansive one.

### Framing Issues

No new P1, P2, or P3 framing issues from the round-7 revisions.

**The four deferred gate items from prior rounds remain open gate decisions and are not re-examined here:**

| ID | Severity | Status |
|----|----------|--------|
| Framing P1-2 | P1 (gate) | Deferred — tau-bench omission |
| Framing P2-1 | P2 (gate) | Deferred — realignment-as-correction framing |
| Framing P2-R5-1 | P2 (gate) | Deferred — A2 latency variance (15–70s) not surfaced |
| Framing P3-R5-1 | P3 (gate) | Deferred — A1-clean arm's methodological purpose |

---

**Overall verdict: clear. Ready for the epistemic gate.** Argument audit is clean (0 issues at any severity). All five round-7 fixes verified. No new issues introduced. The four deferred framing items remain the only open questions and are practitioner-gate decisions, not audit findings.
