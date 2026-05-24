# Argument Audit Report — Cycle 7 DECIDE ADR Set (Round 7)

**Audited document:** `docs/agentic-serving/decisions/adr-032-fallback-shape-and-transparent-endpoint-split.md` (focus: §"Refutation threshold" investigation pathway bullet rewrite from round 7 in response to round-6 P2-R6-1)

**Trigger:** Round-6 P2-R6-1 auto-correction per skill text (P2 issues are auto-corrected by the agent; re-audit after revision is mandatory).

**Source material:**
- Round-6 audit: `docs/agentic-serving/housekeeping/audits/argument-audit-decide-cycle-7-round6.md`
- ADR-028 §"Latency posture" (tuning axes; framing as "each independent; pursued separately based on operational evidence")
- ADR-031 §"Tuning playbook" (same four axes; framing as "operators apply them based on their deployment profile and Population A coverage requirements")

**Genre:** ADR (focused re-audit of single-bullet rewrite)
**Date:** 2026-05-22

---

## Section 1: Argument Audit

### Summary

- **Argument chains re-examined:** 1 (the investigation pathway bullet in its round-7-amended form)
- **Issues found:** 0 P1, 0 P2, 0 P3
- **Gate threshold met.**

### Round-6 P2 fix verification

The round-7 amendment to ADR-032 §"Refutation threshold" investigation pathway bullet:

- **"In order" qualifier dropped:** confirmed.
- **Reordering rationale is explicit:** the amendment names the contrast: ADR-028 + ADR-031 sequence for latency reduction; the refutation threshold sequences for routing-decision quality (reliability/coverage diagnostic frame). The reason the sequences differ is stated in the text.
- **Four-step sequence with per-step rationale preserved:** (1) model profile substitution — addresses planner reliability directly; (2) classifier pre-filter — reduces planner load on covered cases; (3) caching planner decisions — reduces invocations on repeated shapes; (4) capability library expansion — upstream root cause if planner decisions correctly reflect a library scoping mismatch.
- **No residual "in order" language or hierarchy misrepresentation:** confirmed.

### Reordering rationale consistency check

**Claim A (ADR-028 + ADR-031 sequence for latency reduction with classifier pre-filter first):** Verified against both source ADRs. The list order in both ADRs puts classifier pre-filter first; the framing is "each independent; pursued separately based on operational evidence" (ADR-028) — not a strict ordered prescription, but the list order is what an inheriting reader observes.

**Claim B (refutation-threshold context warrants reordering because it's a reliability/coverage signal):** Logically sound. The same tuning axis (model substitution) targets a different symptom in each diagnostic frame — in latency context it means a faster model; in reliability/coverage context it means a more capable model. The mechanism is the same (operator override of the planner's model profile per ADR-028 §"Model profile and tier"); the frame is different.

**Claim C (step 4 capability library expansion is last because it's the upstream root cause if planner decisions correctly reflect a library scoping mismatch):** Coherent. Steps (1)-(3) are interventions on the planner; step (4) is an intervention on the planner's inputs. You would only reach step (4) after confirming steps (1)-(3) failed to resolve the trip.

No new logical gaps introduced.

### P1 — Must Fix

None.

### P2 — Should Fix

None.

### P3 — Consider

None new.

---

## Section 2: Framing Audit

### Carry-Forward Status

All six prior framing observations (F1, F2, F3, NF1, NF2, NF3) carry forward to the practitioner gate. The round-7 amendment touched only the investigation pathway ordering rationale; it did not modify the surfaces the prior framing observations addressed.

**F3 partial-resolution note:** the round-7 amendment's closing-paragraph sentence "The ~15 percentage points threshold is a starting heuristic..." applies the "rough starting heuristic" qualifier explicitly to the ~15pp threshold. The 24-hour rolling window still lacks an equivalent qualifier — NF3's primary observation (asymmetry in epistemic transparency) thus partially resolves; the residual asymmetry remains for practitioner consideration.

### New Framing Observations

**NF4 (P3, round 7 — Minor Precision).** The round-7 amendment's reordering rationale states that ADR-028 + ADR-031 "sequence the tuning axes for latency reduction (classifier pre-filter first, then caching, then smaller-faster planner model)." Both source ADRs frame the axes as "each independent; pursued separately based on operational evidence" (ADR-028) / "operators apply them based on their deployment profile" (ADR-031) — not as a strict ordered prescription. The list order in both ADRs does put classifier pre-filter first, but neither ADR explicitly prescribes a mandatory sequential ordering. The ADR-032 contrast (reliability frame vs. latency frame) is correct in its diagnostic logic; the characterization of the source ADRs' ordering as prescriptive slightly overstates. Carry-forward to practitioner gate as a minor precision observation.

---

## Round 7 Verdict

- **Zero P1.**
- **Zero P2.** Round-6 P2-R6-1 resolved.
- **Zero new P3 in argument-audit section.**
- **One new framing observation NF4** (minor precision; source-ADRs' ordering characterized as prescriptive when source ADRs frame it as "each independent").
- **Six carry-forward framing observations confirmed:** F1, F2, F3, NF1, NF2, NF3 (NF3 partially resolved on the ~15pp side; 24-hour window side persists).
- **No re-emergence of any round-1 through round-6 finding.**

**Gate threshold met.** Per /rdd-decide skill text "Advance only when the most recent audit found no unaddressed issues." The audit loop closes. Cycle 7 DECIDE → ARCHITECT phase boundary is cleared for advancement.

**Framing observations surfaced to the practitioner gate (seven total):** F1, F2, F3, NF1, NF2, NF3, NF4. Per /rdd-decide skill text these are not auto-corrected; they are judgment calls about decision framings for practitioner consideration during gate close.
