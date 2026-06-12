# Argument Audit Report — Round 2

**Audited document:** docs/agentic-serving/decisions/adr-041-destination-validity-gate.md
**Source material:** docs/agentic-serving/essays/research-logs/cycle-7-spike-pi-form-adequacy-gate.md; docs/agentic-serving/decisions/adr-035-client-tool-deliverable-form-contract.md; docs/agentic-serving/decisions/adr-040-deterministic-completeness-gate.md
**Prior audit (R1):** docs/agentic-serving/housekeeping/audits/argument-audit-decide-cycle-7-loopback8.md
**Genre:** ADR
**Date:** 2026-06-11

---

## R2 Verification Map

Before the section-by-section audit, each R1 finding is assessed against the revised ADR.

| R1 Finding | Status | Notes |
|---|---|---|
| P1-1 (protection discharge / install conflation) | **RESOLVED** — with one residual (see new P2-N1) | Status block, §"What this discharges," §Consequences Positive now correctly distinguish design-discharge from install-discharge. A minor internal inconsistency remains (see below). |
| P1-2 (ADR-040 non-overlap incomplete — recovered case) | **RESOLVED** — with a new concern (see new P2-N2) | All three cases (recovered / valid-first-try / cap-exhausted) are now traced. The recovered-case argument is sound but introduces a new claim that requires scrutiny (see below). |
| P2-1 (pre-registered rule transparency) | **RESOLVED** | The live-arm section now explicitly states the literal rule returns "recovers" (B=5/5) and names the converged reading (B=3/5) as the honest application of the rule's spirit. The spike log's transparency is reproduced faithfully. |
| P2-2 (Arm E "closes the residual" overreach) | **MOSTLY RESOLVED** — one carry-over fragment | Three of the four target locations are now correctly scoped. One fragment remains (see carry-over P2-C1 below). |
| P2-3 (degradation-independence qualifier) | **RESOLVED** | Added to §Consequences Positive and appears consistently across Status, §What this discharges, and §Consequences. |
| P3-1 (both-seams uneven evidence) | **RESOLVED** | §Consequences Positive now explicitly notes live-arm evidence covers form-bleed, corpus arm covers wrong-language. |
| P3-2 (destination_path full-path vs. extension) | **RESOLVED** | Decision 2 now states: "The field is the full path, not a pre-extracted extension, so the same thread serves any future extension-keyed check." |
| P2-F1 (ADR-035 "not required" tension) | **RESOLVED** — with a new concern (see new P3-N1) | The "Reconciling with ADR-035" paragraph explicitly resolves the tension via reading (a): the "not required" was conditional on the gate being available as escalation, which Spike π now makes it. The argument is logically sound. |
| P2-F2 (experiential PLAY scope) | **RESOLVED** | The PLAY-observation-target paragraph now names the short-session-vs-broken-file trade-off alongside the semantic-detection residual. |
| P3-F1 (unchanged → narrowed) | **RESOLVED** | The "narrower than what ADR-035 disclaimed to PLAY" language replaces the imprecise "unchanged" framing. |

---

## Section 1: Argument Audit

### Summary

- **Genre:** ADR
- **Argument chains mapped:** 8 (protection discharge, recovery mechanism, ADR-040 three-case non-overlap, protects-but-does-not-recover reading, coder-tier redirection, heuristic alternatives rejection, unbounded-retry rejection, ADR-035 reconciliation)
- **Issues found:** 4 (P1: 0, P2: 2 [1 carry-over, 1 new], P3: 2 [0 carry-over, 2 new])
- **Pyramid coverage map:** N/A
- **Expansion-fidelity findings:** N/A

---

### P1 — Must Fix

No P1 findings. Both R1 P1 findings are resolved. The design-discharge / install-discharge distinction is now present in all three locations cited in R1 (Status block, §"What this discharges," §Consequences Positive), and the ADR-040 three-case interaction is now fully traced.

---

### P2 — Should Fix

**P2-C1 (carry-over, partially resolved). Arm E "closes the residual" — one residual fragment remains in the research log language imported into the ADR.**

- **Location:** §"What the spike established," Arm E bullet, last sentence: "This completes the escalation picture the live arm opened: **deterministic gate (protection, tier-independent) + cheap-tier server-side recovery (rescues intermittent bleeds) + coder-tier escalation (the lever for persistent bleeds, confirmed in isolation)**."
- **Claim:** The phrase "confirmed in isolation" is now present, but the preceding word "completes" reads as completing the escalation picture rather than as completing the *evidence* picture. Combined with "the lever for persistent bleeds," a reader not holding the Arm E paragraph closely will still read this as "all rungs are grounded and the residual is only PLAY territory."
- **Issue:** The two prior occurrences R1 flagged (Decision 5, §"Why the lever") are now correctly scoped: Decision 5 says "confirmed in isolation (Arm E, n=6 on the hardest file) but not yet wired into a converging session," and §"Why the lever" says "that is enough to establish *where* the lever is … it is not a wired-session convergence test." The summary sentence in §"What the spike established" should match that precision. "Completes the escalation picture" without qualification implies the picture is fully validated, not partially grounded. The spike log itself says "This completes the escalation picture the live arm opened" — but that is the spike log describing what the spike accomplished provisionally; an ADR converts provisional spike language into a decision record and should apply the same scoping the Decision section gets.
- **Recommendation:** Add "in design" or "provisionally" to "completes the escalation picture": "This completes the escalation picture *in design*: gate (protection, tier-independent) + cheap-tier recovery (rescues intermittent bleeds) + coder-tier escalation (the lever for persistent bleeds, confirmed in isolation, n=6; session-level wiring is BUILD)." This matches the Decision 5 and §"Why the lever" language already in the ADR.

---

**P2-N1 (new). Status block introduces a minor internal inconsistency between "design-discharged" and the ADR-035 update header's stronger language.**

- **Location:** Status block (ADR-041) vs. ADR-035's `> Updated by ADR-041` header (first paragraph of ADR-035).
- **Claim:** ADR-041's Status block correctly states: "the protection design is structural and discharged *in principle* … but discharge of ADR-035's form-seam Conditional Acceptance is itself contingent on the production install." That is the precise split the R1 correction introduced and it is correct.
- **Issue:** ADR-035's own update header says: "The form-seam **protection** Conditional Acceptance is *discharged*: a wrong-form deliverable is now caught structurally (the gate inspects bytes), not relied upon not to occur." It does not add the "contingent on the production install" qualifier. The two documents now say different things about the same discharge event: ADR-041 says the protection Conditional Acceptance is design-discharged but install-contingent; ADR-035 says it is simply discharged. A reader consulting only ADR-035's update header will not see the install-contingency.
- **Recommendation:** Add a brief qualifier to ADR-035's update header: "discharged *in design* (the install is BUILD — env-gated spike code; de-gate and thread `destination_path`)." This is a one-phrase addition to ADR-035's header that aligns the two documents. Without it, the careful distinction introduced in ADR-041 is invisible at the ADR-035 reading surface.

---

### P3 — Consider

**P3-N1 (new). The ADR-035 reconciliation paragraph carries a mild self-undermining move.**

- **Location:** §Context, "Reconciling with ADR-035's 'a hard form-guarantee is neither available nor required'" paragraph, third sentence.
- **Claim:** "The 'not required' claim was conditional on the gate being *available as escalation*, which Spike π now makes it — committed and deterministic — at a cost (one additive `destination_path` thread) low enough that holding it in reserve buys nothing."
- **Issue:** The reconciliation argument is logically valid and this audit endorses the reasoning. But the phrase "holding it in reserve buys nothing" is stronger than the paragraph needs and slightly undercuts the ADR-035 decision record. ADR-035's "not required" was explicitly grounded in the bounded-failure-cost argument (the client's permission gate + diff makes wrong-form deliverables rejectable). That argument is not overturned by ADR-041 — the paragraph itself says "the bounded-failure-cost argument is not overturned, it is the reason the gate could stay lighter than schema-retry." Saying the reserve position "buys nothing" when the bounded-failure-cost argument is simultaneously affirmed is slightly contradictory: if the bounded-failure-cost argument still holds, then holding the gate in reserve had a coherent rationale, even if the practitioner's directive and the low cost of the gate now make committing it the better call.
- **Recommendation:** Replace "holding it in reserve buys nothing" with "holding it in reserve is no longer the better choice given the low cost of commitment." This preserves the decision logic without implying the ADR-035 reserve position was incoherent.

---

**P3-N2 (new). The recovered-case ADR-040 compose-argument introduces a claim that is stated as verified but was a unit-test-only validation path.**

- **Location:** §Relationship to prior ADRs, ADR-040 entry, case (i) recovered: "the spike's recovered runs converged through ADR-040's completeness check with no double-count (research log §'Recovery built + validated live')."
- **Claim:** The ADR asserts the recovered runs converged through ADR-040's completeness check. This implies ADR-040's completeness gate fired correctly on the turns following a recovered delivery.
- **Issue:** The spike log §"Recovery built + validated live" reports the re-smoke session (turn 3 cli.py recovered → loop continued → "session converged 5/5"). The full n=5×2 live arm shows runs 1/2/5 converged. What the spike log does not explicitly trace is whether ADR-040's `_completeness` function fired, saw the recovered file in the `produced` set, and correctly advanced the anchor. The ADR-041 clause "re-dispatch reuses the delegation path (not a full re-decision), so the action is not double-recorded" is a design assertion grounded in the code reading; it is plausible, but the spike log does not show a `completeness:` trace for the recovered runs comparable to the ADR-040 discharge runs. ADR-040 §"Empirical grounding" notes that "what the live runs validate vs. what only the unit test validates" is a meaningful distinction — the same standard applies here. The assertion that the recovered runs composed cleanly with ADR-040's completeness check is likely correct but is not directly evidenced in the spike log beyond the fact that the sessions converged.
- **Recommendation:** Downgrade the assertion slightly: replace "the spike's recovered runs converged through ADR-040's completeness check with no double-count" with "the spike's recovered runs converged (research log §'Recovery built + validated live'), consistent with ADR-040's completeness check composing correctly; explicit ADR-040 trace on a recovered session is a BUILD verification item (already listed in §What this discharges)." The BUILD verification item is already listed; this just aligns the body claim with its actual evidence level.

---

## Section 2: Framing Audit

### Framing audit scope at R2

The R1 framing audit surfaced two P2 framing issues (P2-F1, P2-F2) and one P3 (P3-F1). All three are resolved in the revised ADR. The two alternative framings held at the gate (gate-as-routing-signal-primary; lever-redirection-as-primary) were explicitly not auto-corrected by design and are confirmed HELD — this audit does not penalize their absence.

### Question 1: Residual alternative framings after revision

The R1 audit surfaced two alternative framings. Both are now partially addressed:

- **Gate-as-routing-signal framing:** The PLAY observation target paragraph now acknowledges the short-session-vs-broken-file trade-off, which is the experiential face of the routing-signal framing. The ADR does not adopt the routing-signal framing as primary but names it as a PLAY question — appropriate.
- **Lever-redirection-as-primary framing:** The §"Why the lever is the coder tier" subsection preserves this as a full-section treatment. It remains secondary to the protection argument, which is a coherent authorial choice.

No new alternative framings emerge from the revision. The revisions are targeted repairs, not a reframing.

### Question 2: Truths available but not featured — any new gaps from revisions?

The R1 framing audit noted three underrepresented truths (corpus narrow scope; recovery-loop trace P2-C gap; baseline 2/5 non-trivial). The revisions do not address these (they were P2/P3 framing observations, not required fixes). They remain as they were — underrepresented but not consequential for the ADR's core decisions. One revision introduces a new framing-adjacent issue:

The "Reconciling with ADR-035" paragraph now says the reconciliation "is a framing the argument audit surfaced; it is recorded here and flagged for the gate, where the practitioner can confirm or re-weigh the priority call." This is honest and appropriate — the paragraph correctly attributes the framing obligation and defers the practitioner priority call. No new framing gap is introduced.

### Question 3: Inverted framing — does the revision change the inverted reading?

The R1 inverted framing was: the gate converts a client-visible, rejectable problem into a server-side invisible one (short session vs. broken file). The PLAY observation target paragraph now directly addresses this: "Whether a short session is a better or worse *experience* than a broken-file diff is an experiential question, not a detection question, and it is not settled by this spike." This is an honest, direct engagement with the inverted framing, and it is appropriately routed to PLAY with FC-51 instrumentation. The inverted framing is no longer an unaddressed gap — it is a named PLAY question.

### Framing Issues

No new P1 or P2 framing findings. The R1 P2-F1, P2-F2, and P3-F1 are resolved. The only residual framing consideration is the ADR-035 update-header precision issue, which is captured as P2-N1 above (an argument-audit finding, not a framing finding — it is an internal inconsistency between two documents).

One minor framing note without severity rating: The "Reconciling with ADR-035" paragraph says the reconciliation "is flagged for the gate." It is not clear from the ADR what "the gate" refers to here — the DECIDE gate, a conformance gate, or the epistemic gate. This is a small referential ambiguity, not a logical issue. It could be clarified as "flagged for the DECIDE gate" or "flagged for practitioner review at the gate" if precision matters downstream.

---

## Convergence-Saturation Signal (ADR-094)

**Convergence-Saturation Signal:** TRIGGERED

- Round number: R2
- P1 count this round: 0 (Section 1: none; Section 2: none)
- P2 count this round (new, non-carry-over): 1 (P2-N1 — Status block / ADR-035 header inconsistency). P2-C1 is a carry-over from R1 P2-2, not counted as new.
- New framings or claim-scope expansions: none. The revisions are in-place corrections that do not introduce new warrant structures or claim-scope characterizations not present in R1.
- Recommendation: **STOP at R2.** P1 count = 0; new P2 count = 1 (within the ≤1 threshold); no new framings. The signal triggers. The three P3 issues (P3-N1, P3-N2, and the carry-over P2-C1 now at carry-over P2 status) are optional-fix territory; they do not block the gate.

**Summary of open items at gate:**

| Finding | Severity | Status | Gate action |
|---|---|---|---|
| P2-C1 (carry-over) | P2 | Partially resolved; one summary-sentence fragment remains | Should fix before production handoff; does not block |
| P2-N1 | P2 | New; ADR-041 / ADR-035 header inconsistency on install-contingency qualifier | Should fix; one-phrase addition to ADR-035 header |
| P3-N1 | P3 | New; "buys nothing" slightly contradicts the bounded-failure-cost affirmation | Consider |
| P3-N2 | P3 | New; ADR-040 compose-claim slightly overstates live evidence | Consider |
| Gate-as-routing-signal framing | HELD | Practitioner decision; not penalized | Review at gate |
| Lever-redirection-as-primary framing | HELD | Practitioner decision; not penalized | Review at gate |

*Single-purpose re-audits (dispatched per the re-audit-after-revision rule) omit this section. Form-change events reset the round-count baseline — the first audit on a new form is its R1.*
