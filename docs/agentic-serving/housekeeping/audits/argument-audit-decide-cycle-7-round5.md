# Argument Audit Report — Cycle 7 DECIDE ADR Set (Round 5)

**Audited documents (revised):**
- `docs/agentic-serving/decisions/adr-021-skill-orchestration-via-per-capability-dispatch.md` (partial-update header from ADR-027, rewritten per round-4 P3-2)
- `docs/agentic-serving/decisions/adr-022-routing-surface-behavior.md` (partial-update header, rewritten per round-4 P1-1 Location 1)
- `docs/agentic-serving/decisions/adr-027-framework-driven-dispatch-pipeline.md` (§Relationship to ADR-022 + §Relationship to AS-9 first bullet, per round-4 P1-1 Location 2 + P3-1)

**Source material:**
- `docs/agentic-serving/housekeeping/audits/argument-audit-decide-cycle-7-round4.md` (round-4 findings being verified)

**Genre:** ADR (focused re-audit of round-4 corrections)
**Date:** 2026-05-22

---

## Section 1: Argument Audit

### Summary

- **Round-4 corrections verified:** 4 of 4 correctly applied
- **New issues found:** 0 P1, 0 P2, 0 P3
- **Gate threshold:** Met — zero unaddressed P1/P2 findings

### Round-4 Correction Verification

**Correction 1 (P1-1 Location 1 — ADR-022 partial-update header).** Verified correctly applied. The header now names: amendment remains operative for any future surface adopting `OrchestratorRuntime`; Tranche 4 Finding 2 acknowledgment that CLI uses `OrchestraService` directly; disposition-(c) consequence (dormant code preserved in version history). The previous claim naming CLI surface + Population B HTTP API as live amendment surfaces has been dropped.

**Correction 2 (P1-1 Location 2 — ADR-027 §Relationship to ADR-022).** Verified correctly applied. The paragraph now mirrors the corrected ADR-022 header with the same Tranche 4 Finding 2 acknowledgment and disposition-(a/b/c) framing. The previous stale CLI-as-live-surface claim has been replaced.

**Correction 3 (P3-1 — ADR-027 §Relationship to AS-9 first bullet parenthetical).** Verified correctly applied. Parenthetical updated from "(preserved on other surfaces per scope)" to "(the role remains available as an architectural option for future surfaces per ADR-001 + ADR-011)."

**Correction 4 (P3-2 — ADR-021 partial-update header symmetric staleness).** Verified correctly applied. Header now names `OrchestratorRuntime` as available architectural option + Tranche 4 Finding 2 acknowledgment + CLI + REST endpoints use `OrchestraService` directly + non-chat-completions surfaces continue per ADR-021's original dispatch shapes via `OrchestraService`.

### Cross-Document Consistency Check

**Disposition-(c) consequence framing — three-location audit:** ADR-022 header, ADR-027 §Relationship to ADR-022, ADR-027 Provenance check all state the disposition-(c) consequence consistently (dormant code, preserved in version history per body-immutable record). ADR-022 adds the "until a future cycle re-introduces a ReAct-loop component" clause as clarifying extension; not a contradiction.

**`OrchestratorRuntime` production-caller framing — three-location audit:** ADR-021 header, ADR-022 header, ADR-027 §Decision, ADR-027 §Relationship to ADR-022 all state the same factual claim ("no production caller other than the chat-completions handler"). Minor phrasing variation ("outside" vs "other than") is not semantic.

**`OrchestratorRuntime` as future-option framing — four-location audit:** ADR-021 header, ADR-022 header, ADR-027 §Relationship to ADR-022, ADR-027 §Relationship to AS-9 all consistent in substance. "Operative" (ADR-022 + ADR-027 §Relationship to ADR-022) and "available as an architectural option" (ADR-021 + ADR-027 §Relationship to AS-9) are equivalent for this context.

### Re-emergence Check (Rounds 1-3)

All prior P1/P2/P3 findings confirmed resolved at round-3 close. Round-4 corrections touched only ADR-021, ADR-022, and ADR-027 — none of the 7 new ADRs (ADR-026 through ADR-032) modified beyond ADR-027. No prior finding has re-emerged.

### P1 — Must Fix

None. Round-4 P1-1 resolved.

### P2 — Should Fix

None. No new P2 findings.

### P3 — Consider

None new. Round-4 P3-1 + P3-2 resolved.

### Gate-Readiness Verdict

Round 5 finds **zero P1 or P2 findings**. The cycle is cleared to advance. Open items are acknowledged framing observations (F1, F2, NF2 at P2; F3, NF1 at P3) carried forward to the practitioner gate, not modified by the round-4 corrections.

---

## Section 2: Framing Audit

### Carry-Forward Confirmation

- **F1 (P2)** — ADR-030 hybrid-as-orthogonal abstract framing. ADR-030 not modified. Carried forward unmodified.
- **F2 (P2)** — ADR-031 tier classification's active-maintenance framing. ADR-031 not modified. Carried forward unmodified.
- **F3 (P3)** — "4 confabulation modes" shorthand in ADR-027 + ADR-028. §Scope-of-claim partition §Settled section not among the corrected sections; ADR-028 not modified. Carried forward unmodified.
- **NF1 (P3)** — ADR-030 follow-on trigger asymmetric quantification. ADR-030 not modified. Carried forward unmodified.
- **NF2 (P2)** — Disposition selection should be a required ARCHITECT deliverable, not a deferred deliberation. ADR-027 §Consequences §Negative still reads "The `OrchestratorRuntime` ReAct loop's disposition after ADR-027 is ARCHITECT-phase work" without naming it as a required deliverable. Carried forward unmodified per skill text (framing observations are surfaced to practitioner gate; not auto-corrected).

### New Framing Observations from Round-4 Corrections

None. The round-4 corrections are surgical text replacements that restore factual accuracy. No new content selections, scope decisions, or alternative-framing exclusions are introduced.

---

## Round 5 Verdict

- **Zero P1.**
- **Zero P2.**
- **Zero new P3.**
- **Framing carry-forwards confirmed unmodified:** F1, F2, F3, NF1, NF2.
- **Gate threshold met.** The Cycle 7 DECIDE ADR set is clear to advance to Tranche 5 (scenarios + interaction specifications + DECIDE phase boundary).
