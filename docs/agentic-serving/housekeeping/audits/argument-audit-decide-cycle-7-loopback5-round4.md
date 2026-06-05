# Argument Audit Report — R4 (scoped post-gate-revision verification)

**Audited document:** `docs/agentic-serving/decisions/adr-037-session-termination-two-call-composition.md`
**Source material read:**
- ADR-037 full text (current, post-gate insertion)
- R2 report: `docs/agentic-serving/housekeeping/audits/argument-audit-decide-cycle-7-loopback5-round2.md`
- R3 report: `docs/agentic-serving/housekeeping/audits/argument-audit-decide-cycle-7-loopback5-round3.md`

**Genre:** ADR
**Scope:** Single insertion in §Consequences Negative — the gate-attributed bullet beginning "The digest's expressiveness is the mechanism's reliability ceiling" (lines 327-337, 2026-06-05). All prior argument chains are clean per R2/R3; this report verifies only the insertion and its immediate context.
**Date:** 2026-06-05

---

## Section 1: Argument Audit

### Summary

- **Genre:** ADR
- **Argument chains mapped:** 1 (insertion site only; full chains audited R2, R3 clean)
- **Issues found:** 1 (0 P1, 1 P2, 0 P3)
- **Pyramid coverage map:** N/A
- **Expansion-fidelity findings:** N/A

---

### Insertion Text Under Audit

§Consequences Negative, the new bullet (lines 327-337):

> **The digest's expressiveness is the mechanism's reliability ceiling** (practitioner pre-mortem at the gate, 2026-06-05): a judgment over a meta-record that cannot represent what completeness means for a complex session — multi-part asks, mid-session intent refinement, deliverables that are not file writes — degrades exactly the way round 1 measured. The framework records; the model reasons over the record; neither substitutes for the other. The committed digest (action kind + path + result) is the **first increment of an extensible meta-record seam, not its final form**; the false-stop share (termination-observability FC) is the extend-on-evidence trigger for digest enrichment, and the digest's home and shape are an ARCHITECT allocation question.

---

### Check 1: Internal consistency with existing ADR claims

**§Decision 2 (framework-owned digest).** Decision 2 pins the digest as deriving from "the framework's own records — the client-tool calls it emitted... joined with the client's per-call tool results — never reconstructed from client-serialized messages alone." The insertion's description of the committed digest as "action kind + path + result" is consistent with this; neither expands nor contracts the defined content.

**§Decision 5 (scope of the validated claim).** Decision 5 records that "tasks whose deliverables are not write-shaped (explanation, command sequences) are outside the measured scope." The insertion's list of complex-session failure modes ("multi-part asks, mid-session intent refinement, deliverables that are not file writes") overlaps with Decision 5's boundary but frames it as a degradation risk rather than a scope exclusion. This is not a contradiction — the insertion correctly characterizes what happens when the mechanism is applied beyond its validated scope — but the overlap is worth noting: the insertion's "multi-part asks" and "mid-session intent refinement" extend slightly beyond Decision 5's list (which names only non-write deliverables), characterizing additional digest-expressiveness limits that Decision 5 does not explicitly name. The insertion is making a forward-looking architectural observation; Decision 5 is recording a measured scope boundary. These are compatible; the insertion does not contradict Decision 5.

**Three-layer scope-of-claim.** Layer 1 lists "the digest's provenance and content" as a hard framework guarantee. The insertion names the committed digest's content as "action kind + path + result" — this is descriptively consistent with the digest-provenance FC, which governs how records are sourced (framework dispatch records joined with client results) rather than what semantic categories they cover. No contradiction.

**Non-write-shaped deliverable bullet (the existing Negative bullet immediately preceding the insertion, lines 322-326).** The insertion follows "Non-write-shaped deliverables are outside the measured scope; the deliverable-accounting standard may not bound the judgment there. The boundary is recorded; widening it is future spike work, watched by the termination-observability events in the meantime." The insertion generalizes this to expressiveness limits more broadly ("multi-part asks, mid-session intent refinement") and gives the mechanism for remediation (digest enrichment triggered by false-stop share). The two bullets are complementary; the insertion adds a mechanism the prior bullet left implicit. No contradiction.

**§Rejected alternatives — Deterministic framework termination policy.** That section includes "The framework records; the model reasons" as an implicit principle (consequence-enforcement is what the framework can compute). The insertion makes this principle explicit: "The framework records; the model reasons over the record; neither substitutes for the other." This is a direct echo of the rejected-alternative analysis, not an independent claim — the insertion anchors to established ADR reasoning rather than introducing new premises. Consistent.

**Verdict on Check 1:** Internally consistent across all named targets.

---

### Check 2: Attribution

The insertion carries explicit attribution: "(practitioner pre-mortem at the gate, 2026-06-05)." The pre-mortem concern is characterized as a limit condition (what the mechanism degrades to when the digest cannot represent completeness), not as a measured failure. The round-1 citation ("degrades exactly the way round 1 measured") grounds the degradation mode in spike evidence without claiming that round 1 measured the complex-session cases specifically — it claims only that the failure mode is the same kind (impoverished digest → impoverished judgment). This framing is accurate: round 1's failure was exactly impoverished evidence base, and the insertion correctly identifies the analog risk for more complex sessions.

The insertion does not present the pre-mortem observation as spike-measured. The attribution is proper.

**Verdict on Check 2:** Attribution clean.

---

### Check 3: "First increment of an extensible meta-record seam, not its final form" — scope overreach check

The FC (digest provenance) governs sourcing, not content coverage. The "extensible meta-record seam" language introduces an architectural framing — the digest as a seam that can be enriched — that does not appear elsewhere in the ADR or in the Fitness Criteria. The insertion acknowledges this explicitly: "the digest's home and shape are an ARCHITECT allocation question." This is accurate: no digest-shape design has been committed in the ADR or the Fitness Criteria. The FC governs provenance (how records are joined), not coverage (what semantic categories are represented). The seam framing therefore does not contradict the FC.

The phrase "extensible meta-record seam" does quietly signal an architectural direction — enrichment as the remediation path — without that direction being a measured or decided property. The insertion handles this correctly by delegating the design to ARCHITECT. However, the phrase "first increment" implies a planned sequence, which is a stronger commitment than "current form." Whether this creates a commitment ARCHITECT cannot refuse depends on how the ADR corpus reads: the sentence immediately delegates shape and home to ARCHITECT, which is a sufficient limiting clause. The implication of sequence is a mild framing risk, not an overreach that commits ARCHITECT to a design.

One wrinkle: the "extend-on-evidence trigger" framing — using the false-stop share (termination-observability FC) as the signal for digest enrichment — does pre-select the evidence trigger for future enrichment decisions. This is an architectural preference embedded in a Consequences Negative bullet, without a corresponding FC or explicit note that this is drafting-time synthesis rather than a decided criterion. This is the kind of forward-looking framing that should be labeled, consistently with the Provenance Check's discipline (which explicitly labels drafting-time synthesis as such). The false-stop-share-as-trigger reading is reasonable and probably correct, but it is not derived from spike evidence — it is an inference about how to operate the observability mechanism introduced in Decision 6. It belongs in the Provenance Check's drafting-time synthesis list alongside the geometric-decay characterization.

**Verdict on Check 3:** The insertion does not overreach on the digest FC or commit ARCHITECT to a design. One unlabeled drafting-time synthesis element: the false-stop share as the extend-on-evidence trigger. See P2 finding below.

---

### Check 4: "Framework records; model reasons" — consistency with rejected framework-termination-policy alternative

The rejected-alternative section explicitly grounds the framework-records/model-reasons principle: "What survives is consequence-enforcement — the framework enforces which call gets guidance and which response returns — and that is exactly where this decision places the framework." The insertion rephrases this as "The framework records; the model reasons over the record; neither substitutes for the other" — a direct parallel.

The rejected-alternative's argument was that the framework cannot compute task-completeness, so framework termination policy reduces to consequence-enforcement. The insertion applies this same principle to the expressiveness domain: the framework cannot represent completeness for complex sessions; extending the digest's expressiveness is the remediation path, but the judgment still belongs to the model. This is consistent; the insertion does not reopen the rejected alternative.

**Verdict on Check 4:** Consistent with the rejected-alternative analysis.

---

### P1 — Must Fix

None.

---

### P2 — Should Fix

**P2-R4-1 — "False-stop share as extend-on-evidence trigger" is unlabeled drafting-time synthesis**

- **Location:** §Consequences Negative, insertion, penultimate clause: "the false-stop share (termination-observability FC) is the extend-on-evidence trigger for digest enrichment"
- **Claim:** The termination-observability FC's false-stop share is the designated trigger for deciding when to enrich the digest.
- **Evidence gap:** This framing is not derived from spike evidence; it is an inference about how to operate Decision 6's observability mechanism. It is arguably the correct inference, but it is a synthesis choice, not a measured property. The Provenance Check lists other drafting-time synthesis items (geometric-decay characterization, three-layer taxonomy, VERDICT-line-stripping detail) by name. This one is absent from that list. Consistent with that discipline, it should be named there.
- **Recommendation:** Add a sentence to §Provenance Check's drafting-time synthesis paragraph: "the false-stop-share-as-trigger reading for digest enrichment (operational inference from Decision 6's observability mechanism; the enrichment criterion itself is an ARCHITECT question)." No change to the insertion text is required — the "ARCHITECT allocation question" clause already limits the design commitment. The Provenance Check entry closes the labeling gap.

---

### P3 — Consider

None.

---

## Section 2: Framing Audit

The insertion captures a pre-mortem concern raised at the gate and folds it into §Consequences Negative as a risk that lives within the mechanism's current scope boundaries. No new framing choice is introduced — the insertion extends the existing "boundary is recorded, not guessed across" pattern that governs Decision 5 and the non-write-shaped bullet. The characterization of the digest as an extensible seam is the only potentially new framing; Check 3 above establishes it does not overreach.

### Question 1: Alternative framings available

The insertion could have been framed as a new Fitness Criterion (an FC governing digest coverage requirements). This alternative framing is correctly not chosen: the digest's home and shape are unresolved ARCHITECT questions, so a coverage FC would be premature. The chosen framing — a Consequences Negative risk with an extend-on-evidence trigger and ARCHITECT delegation — is the scope-appropriate form.

### Question 2: Truths available but not featured

The round-1 evidence base directly demonstrates the degradation mode the insertion describes (impoverished digest → impoverished judgment). The insertion cites this correctly. No additional evidence in the available material would sharpen the insertion.

### Question 3: Inverted framing

Inverted: the digest is already expressive enough for the committed scope, and enrichment would introduce scope creep. Under this inversion, the insertion's risk is overstated and the false-stop-share trigger is a premature response. This inversion is consistent with Decision 5's scope-bounding discipline — and the insertion handles it by delegating enrichment to ARCHITECT rather than prescribing it. The framing is not overreaching; the inversion is already partially absorbed.

### Framing Issues

None beyond P2-R4-1 above (the false-stop-share synthesis label, which is a labeling gap rather than a framing overreach).

---

## Convergence-Saturation Signal (ADR-094)

**Convergence-Saturation Signal:** NOT TRIGGERED
- Round number: R4
- P1 count this round: 0
- P2 count this round (new, non-carry-over): 1 (P2-R4-1 — unlabeled drafting-time synthesis in Provenance Check)
- New framings or claim-scope expansions: none (the "extensible meta-record seam" framing was audited and found to be properly delegated to ARCHITECT; no new warrant or claim-scope characterization introduced)
- Recommendation: CONTINUE to R5 — P2 count = 1, which meets the ≤1 threshold, but R2 previously declared TRIGGERED (STOP) for the pre-insertion document; this is R1 for the insertion as a new element. Treating this as the insertion's first audit, the signal requires one more clean round (zero new P2) before STOP is warranted. If the P2-R4-1 repair (Provenance Check entry) is made and a scoped R5 confirms the repair, STOP is appropriate.

*Form-change baseline note: this is not a form-change event; the insertion is an in-place revision of an accepted ADR bullet. The R2 TRIGGERED verdict applies to the pre-insertion document; this round audits the new element only. Treat as R1 for the insertion's own saturation signal.*
