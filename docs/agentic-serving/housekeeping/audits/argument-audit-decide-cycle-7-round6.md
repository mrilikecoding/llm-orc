# Argument Audit Report — Cycle 7 DECIDE ADR Set (Round 6)

**Audited document:** `docs/agentic-serving/decisions/adr-032-fallback-shape-and-transparent-endpoint-split.md` (focus: §"Refutation threshold for the cost-distribution accountability sub-promise" — gate-time amendment)

**Trigger:** Gate-time practitioner direction during EPISTEMIC GATE conversation 2026-05-22 ("Sure that's likely wise" — endorsing the suggestion to commit to a refutation threshold for the C1 failure signal). Skill text frames framing-discussion-driven amendments as triggering re-audit per the mandatory re-audit-after-revision discipline.

**Source material:**
- `docs/agentic-serving/decisions/adr-027-framework-driven-dispatch-pipeline.md` (C1 failure surface)
- `docs/agentic-serving/decisions/adr-028-routing-planner-ensemble.md` (tuning axes; embedding-similarity router rejection)
- `docs/agentic-serving/decisions/adr-031-latency-timeout-policy.md` (tuning playbook hierarchy)
- `docs/agentic-serving/domain-model.md` §AS-9 (structural-bounding property)
- `docs/agentic-serving/housekeeping/audits/argument-audit-decide-cycle-7-round5.md` (round-5 verdict)

**Genre:** ADR (focused re-audit of gate-time amendment to ADR-032 only)
**Date:** 2026-05-22

---

## Section 1: Argument Audit

### Summary

- **Amendment elements verified:** 5 of 5 (deployment-relative structural relationship; refutation threshold; investigation pathway; falsification escape hatch; PLAY-phase refinement acknowledgment)
- **New issues found:** 0 P1, 1 P2, 0 P3
- **Round-5 verdict (clean at P1/P2/P3 for the seven-ADR set):** confirmed; the amendment does not re-introduce any prior finding

### P1 — Must Fix

None.

### P2 — Should Fix

**P2-R6-1. ADR-032 §"Refutation threshold" investigation pathway hierarchy claims to follow "ADR-028 + ADR-031 in order" but lists an order that differs from both source ADRs.**

- **Location:** ADR-032 §"Refutation threshold for the cost-distribution accountability sub-promise," third bullet: "follow the tuning playbook hierarchy from ADR-028 + ADR-031 in order — (1) routing-planner model profile substitution; (2) classifier pre-filter; (3) caching planner decisions; (4) capability library expansion."
- **Claim:** the hierarchy follows ADR-028 + ADR-031's ordering.
- **Evidence gap:** ADR-028 lists tuning axes as: classifier pre-filter (first), cached planner decisions (second), smaller faster planner model (third), streaming synthesizer response (fourth). ADR-031 lists the same four axes in the same order. ADR-032 sequences model profile substitution (ADR-028/031's third axis) first — before classifier pre-filter. The "in order" qualifier is inaccurate.
- **Recommendation:** Either (a) remove the "in order" qualifier and add a one-sentence rationale for the reordering ("model profile substitution is prioritized first because the refutation threshold is a reliability signal — a more capable cheap-tier planner model addresses routing-decision quality directly; classifier pre-filter and caching address latency, which is a secondary concern at threshold-trip time"), or (b) align the ordering with ADR-028/031 and add a note that capability library expansion is appended as the upstream intervention if planner tuning is insufficient.

### P3 — Consider

None new.

### Element verification details

- **Element 1 (deployment-relative structural relationship):** verified correctly applied. The formula `expected_baseline ≈ (100% - operator-estimated-capability-coverage)` is internally consistent; worked example matches.
- **Element 2 (refutation threshold):** verified correctly applied. The phrasing links the threshold to the C1 failure surface explicitly; uses "evidence" rather than "proof"; qualifies the figure as a starting heuristic.
- **Element 3 (investigation pathway hierarchy):** see P2-R6-1 above.
- **Element 4 (falsification escape hatch):** verified correctly applied. The embedding-similarity router is named as the natural reach if tuning is insufficient; the AS-9 reference correctly closes the frontier-tier-LLM path.
- **Element 5 (PLAY-phase refinement acknowledgment):** verified correctly applied.

---

## Section 2: Framing Audit

### Carry-Forward Confirmation (F1, F2, F3, NF1, NF2)

The amendment modifies only ADR-032 §"Refutation threshold." None of the ADRs housing the five carry-forward framing observations are modified. All five carry forward unmodified.

### New Framing Observations

**NF3 (P3). The 24-hour rolling window is introduced without the same "rough starting heuristic" qualifier applied to the ~15pp threshold, creating a slight asymmetry in epistemic transparency.**

- **Location:** ADR-032 §"Refutation threshold," second bullet: "sustained `direct_completion_rate` more than ~15 percentage points above the deployment's expected baseline over a **24-hour rolling window**..."
- **Observation:** the ~15pp figure is explicitly named as a "rough starting heuristic" in the closing paragraph; the 24-hour window appears without qualification. Both are equally without evidentiary basis in this cycle; the asymmetry could mislead a reader into treating the window as more principled than it is.
- **Recommendation (gate-surfacing only; not auto-corrected):** extend the closing-paragraph qualifier to cover both figures.

---

## Round 6 Verdict

- **Zero P1.**
- **One P2 (P2-R6-1):** investigation pathway "in order" claim misrepresents the ordering relative to ADR-028 + ADR-031.
- **One P3 framing observation (NF3, new):** 24-hour rolling window lacks the "rough starting heuristic" qualifier given to the ~15pp threshold.
- **Five carry-forward framing observations confirmed unmodified:** F1, F2, F3, NF1, NF2.
- **No re-emergence of any round-1 through round-5 finding.**
- **Gate threshold:** One P2 requires editorial fix; framing observation NF3 surfaces to practitioner gate.
