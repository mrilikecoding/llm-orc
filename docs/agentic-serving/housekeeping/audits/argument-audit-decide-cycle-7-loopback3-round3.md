# Argument Audit Report — R3

**Audited document:** docs/agentic-serving/decisions/adr-036-delegation-decision-mechanism.md
**Source material:**
- docs/agentic-serving/essays/research-logs/cycle-7-spike-psi-delegation-rate.md
- docs/agentic-serving/housekeeping/audits/research-methods-spike-psi-prime.md
- docs/agentic-serving/decisions/adr-033-layer-a-loop-driver-multi-turn-agentic-surface.md
- docs/agentic-serving/decisions/adr-032-fallback-shape-and-transparent-endpoint-split.md
- docs/agentic-serving/decisions/adr-035-client-tool-deliverable-form-contract.md
- docs/agentic-serving/domain-model.md §Methodology Vocabulary + §Invariants
**Prior round reports read:**
- docs/agentic-serving/housekeeping/audits/argument-audit-decide-cycle-7-loopback3.md (R1)
- docs/agentic-serving/housekeeping/audits/argument-audit-decide-cycle-7-loopback3-round2.md (R2)
**Genre:** ADR
**Date:** 2026-06-03

---

## R2 Finding Status Summary

| R2 finding | Applied before R3? | R3 status |
|------------|-------------------|-----------|
| P2-1 (soak-window cross-context equivalence — qualify ≥25 figure as provisional with cross-context non-equivalence stated explicitly) | Yes | **Verified held** |
| P3-1 (0.85 sub-band provenance — label as drafting-time synthesis in Provenance check, parallel to 0.9 figure) | Yes | **Verified held** |
| P3-2 (profile-swap soak-window cross-reference — specify ≥25 minimum applies to profile-swap case via Decision 3 cross-reference) | Yes | **Verified held** |
| P2-F1 (ψ.4c empty-response tool-list design implication) | Not applied (held for practitioner) | Carry-over |
| P2-F2 (portability failure boundary uncharacterized) | Not applied (held for practitioner) | Carry-over |
| P3-F2 ("won, not coerced" framing parenthetical) | Not applied (held for practitioner) | Carry-over |

---

## Section 1: Argument Audit

### Summary

- **Genre:** ADR
- **Argument chains mapped:** 5 (Decisions 1–5)
- **Issues found:** 2 (0 P1, 0 new P2, 0 new P3, 3 carry-over framing items held for practitioner)
- **Pyramid coverage map:** N/A
- **Expansion-fidelity findings:** N/A

### P1 — Must Fix

None.

### P2 — Should Fix

None new this round.

### P3 — Consider

None new this round.

---

### R2 Correction Verification Notes

**R2 P2-1 (soak-window cross-context equivalence).** The §Empirical grounding paragraph
now reads: "a qualifying first soak window is a provisional minimum of 25
generation-shaped turns reading ≥0.9 — ψ′ Arm A's n is the starting reference, not a
claim that 25 controlled replay turns and 25 live-traffic turns carry equivalent
epistemic weight (live traffic adds unknown phrasing variation, client-prompt
micro-versions, and classifier error; the practitioner revises the window at the gate for
the deployment context)." The cross-context non-equivalence is stated explicitly; the
figure is labeled provisional; the soak is positioned as trailing confirmation. **Held.**

**R2 P3-1 (0.85 sub-band provenance).** The §Provenance check now includes: "the ~0.85
retry-vs-diagnose sub-band in decision 5 (synthesized from the threshold shape, same
provisional status as the 0.9 figure)." The sub-band is labeled as drafting-time
synthesis in the provenance section, parallel to the treatment of the 0.9 figure.
**Held.**

**R2 P3-2 (profile-swap soak-window cross-reference).** Decision 4 now reads: "trust in
the swapped configuration is an empirical property (a recorded re-validation run, or the
production meter under decision 3 watched through a soak window — qualifying window per
decision 3's ≥25 generation-shaped-turn minimum)." The FC (profile-swap re-validation)
text is unchanged in the current artifact but the cross-reference in Decision 4's body
addresses the ambiguity the P3-2 finding identified. **Held.**

---

## Section 2: Framing Audit

The three framing items carried from R1 and R2 are reviewed below. The status of each
is unchanged: the practitioner elected at R1 not to apply them, and the R2 audit
reconfirmed them as carry-overs. This review checks whether the R2 revisions introduced
any interaction with the carry-over findings that would change their status.

### Question 1: What alternative framings did the evidence support?

The R2 revisions did not introduce new evidence or new framings. The three alternative
framings surfaced at R1 and confirmed at R2 remain available:

**Alternative framing A (portability as latent design risk):** The ADR scopes the claim
to qwen3:14b and requires re-validation on profile swap, but does not characterize the
portability failure boundary (Arm D: qwen3.5:9b 1/5, mistral-nemo:12b 2/5 with no
identified cause). The framing is that re-validation is uninformative diagnosis without a
failure model. Nothing in the R2 revisions changes this.

**Alternative framing B (denominator brittleness over time):** The classifier's
repair-shaped and capability-domain exclusions are permanent structural exclusions whose
production incidence is unknown. The ADR's Decision 3 boundary-degradation signal
addresses rate accuracy, not coverage representativeness as the capability library grows.
Nothing in the R2 revisions changes this.

**Alternative framing C (inverted: environment-conditional win):** The "won, not coerced"
framing presents the mechanism positively; the inversion — "fragile, not robust" — is
available from the same evidence. The R2 revisions did not add any language that would
settle the P3-F2 parenthetical finding.

### Question 2: What truths were available but not featured?

No new omissions were introduced by the R2 revisions. The role-vs-adjacency mechanism
isolation gap, the B4 repair-boundary first-turn implication for denominator design, and
the context-growth delegation persistence gap remain available truths from the source
material that the ADR acknowledges at varying levels but does not foreground. None of
these changed in the current revision.

### Question 3: What would change if the dominant framing were inverted?

Unchanged from R2. The R2 revisions improved the provisional-status labeling of the 0.9
and 0.85 figures and tightened the soak-window cross-context qualification, which
slightly narrows the gap between the ADR's framing and the inverted framing — but the
structural observation (that the instrumentation is doing more load-bearing work than the
headline framing implies) stands.

### Framing Issues

**P2-F1 (CARRY-OVER — practitioner-held, not applied):** The ψ.4c empty-response
finding constrains the detect-and-retry escalation path in Decision 5: a retry that
restricts the tool list would reproduce ψ.4c's empty-response failure. The ADR names
detect-and-retry as "architecturally available at the same composition point" without
noting that the tool-list-restriction shape is measured to break the turn. The R2
revisions did not touch Decision 5's detect-and-retry text. Status: available for the
gate.

**P2-F2 (CARRY-OVER — practitioner-held, not applied):** The portability failure
boundary is uncharacterized. The R2 revisions did not add portability-failure-mechanism
language. The FC (profile-swap re-validation) remains a correct operational response to
an unknown failure mechanism, but re-validation results are uninterpretable without a
theory of what properties a model must have for V3 to work. Status: available for the
gate.

**P3-F2 (CARRY-OVER — practitioner-held, not applied):** The "won, not coerced" framing
in the Decision statement does not carry a parenthetical scoping the win to the validated
stack. The R2 revisions improved scope labeling elsewhere (Decision 2, §Empirical
grounding) but left the Decision statement's opening clause unchanged. Status: available
for the gate; low-stakes.

---

## Convergence-Saturation Signal (ADR-094)

**Convergence-Saturation Signal:** TRIGGERED

- **Round number:** R3
- **P1 count this round:** 0 (Section 1: 0; Section 2: 0)
- **P2 count this round (new, non-carry-over):** 0 (P2-F1 and P2-F2 are carry-overs
  from R1, not new findings; they were not applied by practitioner election and were
  present in both prior round reports)
- **New framings or claim-scope expansions:** none — the R2 revisions tightened existing
  scope qualifications (soak-window provisional status; sub-band provenance) without
  introducing new warrants or claim-scope characterizations not present in prior rounds
- **Recommendation:** STOP at R3

All three signal conditions hold:
1. P1 count = 0
2. New (non-carry-over) P2 count = 0
3. No new framings or claim-scope expansions surfaced this round

The three practitioner-held items (P2-F1, P2-F2, P3-F2) are carry-overs recorded in
both R1 and R2; they are not counted as new P2 findings for saturation-signal purposes.
The signal triggers. The document is ready for the gate with the three carry-over framing
items remaining explicitly available for practitioner evaluation.

*Single-purpose re-audits omit this section. Form-change events reset the round-count
baseline — the first audit on a new form is its R1.*
