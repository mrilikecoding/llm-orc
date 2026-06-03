# Argument Audit Report — R2 Verification

**Audited document:** docs/agentic-serving/decisions/adr-035-client-tool-deliverable-form-contract.md
**Source material:**
- docs/agentic-serving/essays/research-logs/cycle-7-spike-phi-deliverable-shape.md
- docs/agentic-serving/essays/research-logs/cycle-7-spike-chi-deliverable-shaping.md
- docs/agentic-serving/decisions/adr-024-common-io-envelope.md
- docs/agentic-serving/decisions/adr-034-client-tool-action-terminal-artifact-bridge.md
**Prior audit (R1):** docs/agentic-serving/housekeeping/audits/argument-audit-decide-cycle-7-loopback2.md
**Genre:** ADR
**Date:** 2026-06-03

---

## Section 1: Argument Audit

### Summary

- **Genre:** ADR
- **Round:** R2 — verification of R1 P2 corrections, no full re-derivation
- **R1 findings under review:** P2-1 (structural overclaim), P2-2 (ADR-024/Spike β correction overstated), P2-3 (ADR-034 mitigation misattribution); P3-1 (F3-1 as driver vs. concession), P3-2 (n=4 caveat scope)
- **Argument chains mapped:** 6 (unchanged from R1)
- **Issues found:** 0 P1, 1 P2 (new), 1 P3 (carry-over from R1, P3-3 — not corrected per dispatch brief; noted as still open)
- **Pyramid coverage map:** N/A
- **Expansion-fidelity findings:** N/A

---

### R1 Correction Verification

#### P2-1 — "structural in spirit" overclaim

**R1 finding:** The Positive consequence bullet labeled the mechanism "structural in spirit" while the Negative section qualified it as model-compliance-dependent, creating a contradictory framing across consequence buckets.

**R1 recommendation:** Revise the Positive bullet to distinguish what the framework structurally guarantees (directive presence) from what it does not (deliverable form).

**Verification:** The correction held. The revised Positive bullet 3 (§Consequences) now reads:

> "Lighter than schema-retry; the structural guarantee is bounded and explicit. What the framework structurally guarantees is the *presence* of the destination-keyed directive in every client-tool dispatch (the controlled lever ADR-024 anticipated). It does *not* structurally guarantee the deliverable's *form* — that rests on documented model compliance (Spike χ, n=4 first-try). Directive-presence is enforced; form-compliance is relied upon. No retry loop in the common path."

The AS-9 relationship paragraph (§Relationship to prior ADRs) correspondingly reads:

> "AS-9 / single-step enforcement — consistent in spirit. The framework *guarantees the directive is present* (structural), while relying on documented model compliance to produce the form (Spike χ, n=4). This is lighter than hard schema-retry enforcement and structurally framework-owned, but it is model-compliance-dependent, not a hard structural guarantee — hence Conditional Acceptance."

The distinction between what is structurally guaranteed (directive presence) and what is model-compliance-dependent (form) is now explicit in the Positive consequence itself, not only in the Negative. The tension R1 identified is resolved. **VERIFIED.**

---

#### P2-2 — ADR-024/Spike β "correction" framing overstated

**R1 finding:** The D2a section framed the Spike φ finding as "correcting the empirical premise of ADR-024 / Spike β," which overstated the contradiction. Spike β's headline finding (composition assumptions live in the orchestrator's reasoning surface, not the typed contract) survives the Spike φ finding intact; the Spike φ finding adds mechanistic precision, not a contradiction.

**R1 recommendation:** Replace "corrects the empirical premise" with framing that acknowledges Spike β's headline survives and that Spike φ adds precision.

**Verification:** The correction held. The revised D2a section reads:

> "This **refines the mechanism Spike β identified, without disturbing its headline finding.** Spike β's headline — that composition assumptions live in the orchestrator's reasoning surface, not the typed contract — survives intact. ADR-024 described the drift mechanism as 'the orchestrator hand-writes `input.data`, overriding `default_task`'; Spike φ adds the mechanistic precision that `default_task` does not reach the model at all, so it is not 'overridden' so much as absent — the dispatch input is the *only* contract surface that reaches the model. That precision is what makes boundary-injection the natural lever."

The Provenance check also records this accurately ("corrects Spike β premise" language does not appear; it reads "corrects ADR-024 / Spike β's mechanism without disturbing its headline finding" in the relevant bullet context). The "correction" framing has been replaced with "refines ... without disturbing" throughout. **VERIFIED.**

Note: One minor residual in the Provenance check (line: "`default_task` inert; dispatch input is the only contract surface (corrects Spike β premise)") still uses "corrects Spike β premise" as a parenthetical label. This parenthetical is in tension with the ADR body, which now correctly says "refines." The parenthetical is in the provenance metadata rather than the argument, so it is not a logical error, but it is imprecise. This is a P3 observation, logged below as NEW-P3-1.

---

#### P2-3 — ADR-034 "store-read mitigation incomplete" misattribution

**R1 finding:** The D2b section described ADR-034 as having a "store-read mitigation" that was "incomplete," attributing to ADR-034 a promise it did not make. ADR-034 promised faithful marshalling (fidelity FC), not a fix for the form of whatever was stored.

**R1 recommendation:** Reframe to say ADR-034 left the form question open (promised fidelity, not form) and ADR-035 fills that open question.

**Verification:** The correction held. The revised D2b section reads:

> "**ADR-034 already named this exact risk** when it rejected synthesizer-as-terminal ('a corruption risk for a tool-call content argument that must be exactly the deliverable'). ADR-034 did not, however, offer a mitigation for the deliverable's *form* — it promised faithful *marshalling* (the artifact-bridge-fidelity FC: 'marshal exactly what is stored') and left open the question of what form the stored content should take. Finding D shows the store holds the same prose-framed content, so fidelity alone delivers prose; ADR-035 fills the open question (the form contract), upstream of the marshalling ADR-034 specified."

The §Relationship to prior ADRs — ADR-034 bullet confirms this:

> "ADR-034 (client-tool-action terminal + artifact-bridge) — completes, does not change. ADR-034's decisions stand: the bridge reads the server-side artifact and marshals it faithfully. ADR-034's synthesizer-as-terminal rejection *named* the prose-framing risk but specified only faithful marshalling, leaving open what form the stored content should take. ADR-035 fills that open question — the deliverable is produced in client-tool form at the source (boundary directive), so the content the bridge faithfully marshals is already bare."

This is accurate and precisely sourced against ADR-034's actual FC language ("marshal exactly what is stored"). The misattribution is resolved. **VERIFIED.**

---

#### P3-1 — F3-1 characterization (granularity invariant provenance)

**R1 finding:** The Provenance check listed ADR-033 F3-1 as a "driver" for the granularity invariant, where F3-1 is explicitly a "recorded concession" in ADR-033, not a fitness criterion. The spike evidence (χ-P6) is the real driver.

**R1 recommendation:** Characterize ADR-033 F3-1 as corroborating prior ADR (recorded concession) rather than "driver, prior ADR."

**Verification:** The correction held. The revised Provenance check reads:

> "**Granularity invariant (one dispatch → one deliverable; multi-file across turns)**: Spike χ-P6 (driver) + ADR-033 F3-1 across-turn composition (a *recorded concession* / wrapper-residual watch point in ADR-033, not a driver finding — the granularity invariant is an inference consistent with F3-1, and the structured-multi-file alternative was not probed; see Framing note for the gate). Driver chain: spike + prior-ADR concession."

The framing is now precise: Spike χ-P6 is the driver; F3-1 is a recorded concession consistent with (not independently establishing) the granularity invariant. The provenance distinguishes evidentiary weight correctly. **VERIFIED.**

---

#### P3-2 — n=4 caveat scope (pipeline-narrow, code-generator-heavy sample)

**R1 finding:** The n=4 characterization implied compliance breadth across task types but all four samples ran through code-generator's agent pipeline. Other capability ensembles' agent stacks were not tested.

**R1 recommendation:** Note in Conditional Acceptance that directive reliability was tested only through code-generator's synthesizer agent pipeline; other ensemble agent stacks are PLAY targets.

**Verification:** The correction held. The revised §Consequences Negative bullet 3 reads:

> "**Grounding is n=4 single-deliverable, cheap-tier, and pipeline-narrow.** Three of the four compliant samples ran through `code-generator`'s pipeline (the fourth, φ Run 2, through claim-extractor); breadth across other capability ensembles, long trajectories, escalated tiers, and `edit`/`bash` at scale is PLAY/first-deployment work, not settled here."

This is accurate: the φ Run 2 sample (claim-extractor, structured prose) is the outlier; the other three are code-generator. The "pipeline-narrow" label is appropriate. The Conditional Acceptance section also notes the n=4 covers single dispatches, not trajectories. **VERIFIED.**

---

### P1 — Must Fix

No P1 findings. The five R1 argument-P2 and P3 corrections are all verified above. No new P1 issues found.

---

### P2 — Should Fix

**NEW-P2-1: The Provenance check "corrects Spike β premise" parenthetical contradicts the ADR body**

- **Location:** §Provenance check, bullet: "`default_task` inert; dispatch input is the only contract surface (corrects Spike β premise): Spike φ Part 3 grep + path trace (driver)."
- **Claim:** The parenthetical labels the Spike φ finding as "corrects Spike β premise."
- **Evidence gap:** The ADR body (D2a section) was correctly revised at P2-2 to say Spike φ "refines the mechanism Spike β identified, without disturbing its headline finding." The Provenance check parenthetical was not updated to match — it still says "corrects Spike β premise," which is the overclaiming framing the P2-2 correction was intended to fix. A reader cross-referencing the Provenance check against the body will encounter contradictory characterizations: the body says "refines, headline survives"; the Provenance check says "corrects premise." Since the Provenance check is precisely the metadata a future auditor or BUILD reader will scan for sourcing, the inconsistency is exposed exactly where it matters.
- **Recommendation:** Update the parenthetical to: "(refines Spike β mechanism; headline finding survives intact — the dispatch input is the sole contract surface, consistent with Spike β's 'orchestrator reasoning surface is the composition substrate' conclusion)." This aligns the Provenance check with the argument the body now makes.

---

### P3 — Consider

**NEW-P3-1 (same as NEW-P2-1 at P3 severity if the contradiction is treated as minor):** The Provenance check parenthetical is imprecise but located in metadata, not the argument chain. If the practitioner reads this as a residual editing gap rather than a logical issue, P3 is the appropriate severity. The P2 reading (above) is correct if provenance metadata is considered part of the argument (which it is in RDD methodology — provenance is a first-class accountability structure, not decoration). The practitioner should confirm severity.

**Carry-over P3-3 (from R1, not corrected per dispatch brief):** The "marshalling boundary" / "Loop Driver / Client-Tool-Action Terminal" terminology interchangeability in Decision §1 was noted in R1 as a minor BUILD-clarity issue. It was not in scope for the P3 corrections applied. Still open; no change.

---

## Section 2: Framing Audit

The R1 framing audit found two P2 framing issues (P2-F1: AS-9 analogy invoked favorably when its logic points toward the opposite default; P2-F2: granularity invariant forecloses structured-multi-file framing without testing it) and one P3 (P3-F1: Conditional Acceptance does not name delegation-fires as a PLAY precondition). Per the dispatch brief, the practitioner held these for the gate rather than applying corrections — they should not be re-flagged as failures. They are acknowledged here for completeness.

### Held R1 Framing Issues (not re-flagged)

- **P2-F1** (AS-9 analogy tension) — held for practitioner gate. The ADR's AS-9 relationship paragraph now correctly qualifies the mechanism as "model-compliance-dependent, not a hard structural guarantee — hence Conditional Acceptance," which partially surfaces the tension. The full AS-9 analogy inversion (R1 recommended noting the analogy's logic supports the opposite default; the empirical basis for taking the lighter path is the compliance track record) is not added. Held.
- **P2-F2** (granularity invariant / structured-multi-file alternative not probed) — held for practitioner gate. The Provenance check now accurately notes the structured-multi-file alternative was not probed ("the structured-multi-file alternative was not probed; see Framing note for the gate"), so the evidentiary boundary is flagged. The Decision §3 does not add the parenthetical R1 recommended. Held.
- **P3-F1** (delegation-fires precondition not named in Conditional Acceptance) — held per dispatch brief. Not corrected; not re-flagged.

### Question 1: Did the R2 corrections introduce any new framing issues?

The P2-1 correction (bounded structural claim made explicit in Positive consequences) is accurate: the ADR now honestly distinguishes directive-presence (structural) from form-compliance (model-reliant). This narrowing does not introduce a new framing issue — it removes one.

The P2-2 correction (Spike β headline framing) is accurate and does not foreground any new framing alternative that was available in the source material but excluded. The Spike φ finding and its relationship to Spike β are now correctly characterized.

The P3-1 correction (F3-1 as concession not driver) surfaces an honest acknowledgment that the structured-multi-file alternative was not probed — the Provenance check now explicitly notes this. This is the correct framing given the source material (Spike χ-P6 tested only implicit multi-file, not a JSON-array structured format). No new framing issue introduced.

**No new framing issues found.** The corrections improved framing accuracy without introducing new gaps.

### Question 2: Any truths available in source material now underrepresented by the corrections?

Checking the three changes against the spike logs:

- The Spike χ-P6 finding is now more accurately characterized (granularity invariant as inference, not established finding; alternative not probed). This is more honest, not less. No truth suppressed.
- The Spike φ / Spike β relationship is now "refines, headline survives" — this matches what the spike logs actually say (Spike φ's D2a section ends: "This is BUILD-adjacent (like D1: dead config that should reach the model) but the *how* (wire `default_task` through vs. adopt a different contract surface) is a DECIDE policy question" — no claim that Spike β's mechanism was wrong, only that the code path is more precisely understood). Accurate.
- The ADR-034 relationship is now "left open the form question" — this matches ADR-034's actual FC language ("marshal exactly what is stored") and the Consequences §Negative ("edit, bash, multi-file, and streaming-token synthesis are unbuilt") which does not claim the form question was answered. Accurate.

No truths underrepresented by the corrections.

### Framing Issues

No new framing issues found this round. The two held P2-F1 and P2-F2 from R1 remain open for practitioner gate decision; they are not re-rated.

---

## Convergence-Saturation Signal (ADR-094)

**Convergence-Saturation Signal:** TRIGGERED

- Round number: R2 (on the same form; R1 was the reset baseline)
- P1 count this round: 0 (Section 1: 0; Section 2: 0)
- P2 count this round (new, non-carry-over): 1 (NEW-P2-1 — Provenance check parenthetical contradicts the body's P2-2 correction)
- New framings or claim-scope expansions: none — the corrections improved characterization accuracy without surfacing new warrants or claim-scope expansions not present in R1
- Recommendation: **STOP at this round.** All three signal conditions hold: P1 = 0; new P2 = 1 (at threshold); no new framings. The residual is a Provenance check inconsistency (editing gap, not a structural argument failure) and two held framing P2s at the practitioner gate. Further audit rounds are unlikely to surface new structural issues on this document.

*The two held framing P2s (P2-F1, P2-F2) and carry-over P3-3 from R1 are practitioner-gate items, not argument-audit blockers. TRIGGERED does not mean these are resolved — it means the audit round-count discipline is satisfied. The practitioner decides whether to accept the framing concessions or reopen.*
