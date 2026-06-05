# Argument Audit Report — R2

**Audited document:** `docs/agentic-serving/decisions/adr-037-session-termination-two-call-composition.md`
**Partial-update header also audited:** `docs/agentic-serving/decisions/adr-036-delegation-decision-mechanism.md` (top block + Status)
**Domain model block audited:** `docs/agentic-serving/domain-model.md` (AS-9 full block incl. §Termination-judgment instance annotation + two new §Methodology Vocabulary rows + Amendment Log #18)
**Source material read:**
- `docs/agentic-serving/essays/research-logs/cycle-7-spike-theta-termination-mechanism.md` (F-θ.2 paragraph + composed-estimate block)
- R1 report: `docs/agentic-serving/housekeeping/audits/argument-audit-decide-cycle-7-loopback5.md`

**Genre:** ADR
**Date:** 2026-06-05

---

## Section 1: Argument Audit

### Summary

- **Genre:** ADR
- **Argument chains mapped:** 11 (same as R1; no new chains)
- **Issues found:** 2 (0 P1, 1 P2, 1 P3)
- **Pyramid coverage map:** N/A
- **Expansion-fidelity findings:** N/A

---

### R1 Correction Verification

Each of the six R1 P2/P3 corrections was checked against the current document state.

| R1 finding | Status | Notes |
|------------|--------|-------|
| P2-1 (AS-9 class claim re-scoped) | **Fixed** | §Decision now reads "new instance AS-9's recorded empirical basis does not cover (Spike θ is its instance-level evidence; the domain model records the extension — see Amendment Log)." Three-layer Layer 2 correctly characterizes the judgment as "a new instance of that shape, evidenced by Spike θ itself, not by AS-9's recorded basis." Domain-model §Termination-judgment instance annotation added (confirmed). Amendment Log #18 present and consistent. |
| P2-2 (composed-estimate label in Consequences) | **Fixed** | §Consequences Positive now reads "composed estimate from independently-measured n=10 arms, not an end-to-end measurement" — full-form label matching §Decision ¶5 precision. |
| P2-3 (geometric-decay claim carries arithmetic caveat at claim site) | **Fixed** | §Consequences Positive now reads "geometric decay, which is drafting-time arithmetic from a single-base n=10 observation (the multi-turn decay itself is unmeasured)". The provenance caveat is present at the claim site. |
| P2-4 (judgment-seat re-validation FC operationally defined) | **Fixed** | FC (judgment-seat re-validation) now names the instruments explicitly: "a run of the θ-harness judgment arms (the θ.3b/θ.4b/θ.4b′ compositions)"; the combined-event definition is present ("one recorded run covering both instruments (ADR-036's ψ′-harness delegation arms AND the θ-harness judgment arms)"); the per-seat-when-split case is named. |
| P3-1 (implicit-variant rejection clarifies E4a's 6/10 are all inline writes) | **Fixed** | §Rejected alternatives (Implicit judgment) now explicitly states: "Note E4a's 6/10 continues are themselves all inline writes (0/10 delegation) — the ~0.54 is the rate after call 2 re-shapes them, not the implicit dispatch's own delegation rate, which is zero." |
| P3-2 (Form A fallback notes reintroduced client-prompt cost) | **Fixed** | §Rejected alternatives (Form A-enriched) now reads "knowingly reintroducing the client-prompt processing cost on every judgment call (the reason B won the tiebreak)." |
| P3-3 (DECLINED — ADR-036 body-immutable rule) | **Disposition verified** | ADR-036's §Consequences Neutral text is unchanged (body-immutable). The partial-update header carries the narrowing. The declination is sound: ADR-036's body-immutable rule is a corpus discipline established to preserve accepted-ADR integrity (the same rule that governs ADR-027's immutability). A skimming reader encounter with the Consequences Neutral text is adequately mitigated by the header's placement above the Context section — on any canonical reading of the ADR, the header is encountered first. No new finding warranted. |
| P3-4 (19/60 figure corrected in ADR Context and F-θ.2) | **Fixed** | §Context now reads "Form A 19/30, failing toward confabulated COMPLETE; Form B 20/30 where all 20 came from degenerate uniform-REMAINING." The F-θ.2 paragraph in the spike θ log has a visible correction note. The "19/60 → 59/60" phrase in F-θ.2's final sentence remains but it is now sequentially downstream of the per-form correction and reads as a shorthand for the "information-starved baseline vs. round 2" contrast, not as the Form A raw count. Residual: see P2-R2-1 below. |
| P3-5 (hosted latency corrected to 0.7–3.0s/call) | **Partially fixed** | §Context's Portability annotation now reads "0.7–3.0s/call" — correct. §Consequences Negative still reads "Hosted seating reduces this to ~1–2s at ~$0.0015/call." This is a new inconsistency introduced by a partial correction. See P3-R2-1 below. |
| P2-F1 (hosted annotation placement + "annotation only" prefix) | **Fixed** | §Context's Portability annotation now opens "annotation only — the local arms carry the verdict" before stating the 20/20 result. The disciplined reading order is restored. |
| P2-F2 (Layer 2 separates primary evidence from hosted annotation) | **Fixed** | Layer 2 now reads "29/30 on qwen3:14b (the verdict-bearing primary evidence); the one-pair hosted annotation (20/20) is supporting context at a different evidence level, not a second reliability measurement." The "two measured models" conflation is gone. |

---

### P1 — Must Fix

None.

---

### P2 — Should Fix

**P2-R2-1 — The "19/60 → 59/60" shorthand in F-θ.2's closing sentence survives the per-form correction, creating a latent ambiguity**

- **Location:** ADR-037 §Context, F-θ.2 bullet's last sentence: "the same model, same bases, same verdict format moved 19/60 → 59/60 on digest + standard alone"
- **Claim:** Round 1 produced 19/60; round 2 produced 59/60.
- **Evidence gap:** The per-form correction earlier in F-θ.2 (Form A 19/30, Form B 20/30) correctly breaks out the round-1 per-form picture. The closing sentence then reverts to "19/60" without qualification. A reader who skims the closing sentence will encounter the original discrepant figure. The research log now carries a correction note explaining that "19/60" was the Form A subset conflated with the full sixty; the ADR carries the same shorthand without the research log's footnote. The "19/60 → 59/60" contrast is narratively useful (same-model, same-bases change), but the "19" now references "Form A's 19/30 on the subset that couldn't discriminate" rather than any aggregate count — a meaning that should be made explicit rather than inherited from context.
- **Recommendation:** Qualify the closing sentence: e.g., "the same model, same bases moved from no usable form (Form A 19/30 confabulating, Form B 20/30 degenerate-REMAINING) to 59/60 on digest + standard alone." Alternatively: "the same model, same bases moved from no discrimination (round 1: see per-form above) to 59/60 on digest + standard alone." Either phrasing closes the "19/60" shorthand that the per-form fix was intended to retire.

---

### P3 — Consider

**P3-R2-1 — §Consequences Negative retains "~1–2s" after §Context was corrected to "0.7–3.0s/call"**

- **Location:** §Consequences Negative: "Hosted seating reduces this to ~1–2s at ~$0.0015/call if adopted."
- **Claim:** Hosted latency is ~1–2s/call.
- **Evidence gap:** P3-5 corrected §Context to "0.7–3.0s/call" (the source log's range). §Consequences Negative was not updated in the same pass, leaving the two sections reporting different upper bounds (2s vs 3s) for the same measured quantity. The inconsistency is minor — the figures overlap — but the Consequences section is the practitioner's decision-support surface; understating the upper bound by 1s in a call that runs on every trailing turn is worth correcting for completeness.
- **Recommendation:** Update §Consequences Negative to "~0.7–3.0s/call" or "~1–3s/call" to match §Context. One-line fix.

---

## Section 2: Framing Audit

The framing audit for R2 focuses on whether the R1 corrections introduced new framing choices that should be examined, and whether the pre-existing framing issues were honestly addressed.

### Question 1: What alternative framings did the evidence support?

No new alternative framings surfaced by the R2 corrections. The three alternatives examined in R1 (narrower guidance-fix frame, Form A as primary, three-layer taxonomy as evidence organization) remain validly addressed by the document's current framing. The P2-F1 and P2-F2 corrections tightened the hosted-annotation framing; neither introduced a new framing option that should be audited here.

### Question 2: What truths were available but not featured?

The R2 corrections themselves are transparent about their provenance — the corrected F-θ.2 text in the spike log carries a visible correction note, and Amendment Log #18 names the R1 argument audit as the source of the annotation additions. No previously available truth was suppressed or obscured by the revision pass.

One small new observation: the "19/60 → 59/60" shorthand that P2-R2-1 flags was introduced into the ADR during original drafting and was not retired by the per-form correction. It now sits alongside a more precise per-form breakdown without being explicitly superseded, which means a reader of the ADR (not the spike log) sees both a precise per-form picture and an unexplained aggregate shorthand. This is the evidence-presentation gap the P2-R2-1 finding addresses; no separate framing finding warranted.

### Question 3: What would change if the dominant framing were inverted?

The dominant framing and its inversion were examined in R1. The R2 corrections do not alter the inversion's force: the COMPLETE verdict is a deliverable-production certificate, not a work-correctness certificate, and the AS-3 cap is the hard backstop beneath the mechanism. Neither correction touches these structural framing choices; no new inversion finding warranted.

### Framing Issues

**P3-F-R2-1 — The "annotation only — the local arms carry the verdict" phrase now front-loads the Portability annotation correctly, but the sentence that follows still places the 20/20 number before the scope hedge**

- **Location:** §Context Portability annotation: "annotation only — the local arms carry the verdict: the identical composition scored 20/20 on `zen:minimax-m2.7` at 0.7–3.0s/call (vs 7–19s local). One pair does not establish portability; it establishes the composition is not qwen-idiosyncratic."
- **Claim:** The "annotation only" prefix adequately signals the evidential weight before the number is stated.
- **Evidence gap:** The fix is an improvement over R1 (where the number appeared with no prefix). The phrase now signals subordinate status before the result. Reading order: prefix → number → hedge sentence. The hedge follows the number. This is the correct order for a practitioner reading carefully; it is adequate. The concern is lower than P2 — not a framing error, but a note that the Portability annotation's prose structure front-loads the conclusion ("annotation only") while placing the methodological limitation ("One pair does not establish portability") in a second sentence. This is standard prose rhythm and the fix landed cleanly.
- **Recommendation:** No change required. Noting this as P3 for completeness; the current form is adequate.

---

## Convergence-Saturation Signal (ADR-094)

**Convergence-Saturation Signal:** TRIGGERED

- Round number: R2
- P1 count this round: 0 (Section 1: 0; Section 2: 0)
- P2 count this round (new, non-carry-over): 1 (P2-R2-1 — "19/60" shorthand residual in F-θ.2 closing sentence)
- New framings or claim-scope expansions: none. The R2 corrections tightened existing framings (AS-9 instance scoping, hosted-annotation evidential weight) without introducing new warrants or claim-scope characterizations that R1 did not name.
- Recommendation: **STOP at this round.**

*Single-purpose re-audits (dispatched per the re-audit-after-revision rule) omit this section. Form-change events reset the round-count baseline — the first audit on a new form is its R1.*
