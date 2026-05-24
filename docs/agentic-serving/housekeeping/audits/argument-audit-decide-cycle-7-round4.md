# Argument Audit Report — Cycle 7 DECIDE ADR Set (Round 4)

**Audited document:** `docs/agentic-serving/decisions/adr-027-framework-driven-dispatch-pipeline.md` (round-4 amendment: §Context, §Decision, §Consequences §Negative, §Provenance check)
**Cross-document audit:** `docs/agentic-serving/decisions/adr-022-routing-surface-behavior.md` (partial-update header); `docs/agentic-serving/decisions/adr-021-skill-orchestration-via-per-capability-dispatch.md` (partial-update header)
**Trigger:** Round-4 amendment was applied in response to Tranche 4 conformance-scan Finding 2 (codebase fact: `OrchestratorRuntime` only used by chat-completions handler; CLI uses `OrchestraService` directly). The amendment touched four sections of ADR-027.
**Source material:** `docs/agentic-serving/housekeeping/audits/argument-audit-decide-cycle-7-round3.md` (round 3 clean verdict for the ADR set); `docs/agentic-serving/housekeeping/audits/conformance-scan-cycle-7-decide.md` (Finding 2 trigger)
**Genre:** ADR (focused re-audit of four amended sections + cross-document headers)
**Date:** 2026-05-22

---

## Section 1: Argument Audit

### Summary

- **Genre:** ADR focused re-audit (round 4)
- **Amendments verified:** 4 of 4 correctly applied to ADR-027 (Context codebase-finding sentence; Decision OrchestratorRuntime status paragraph with three dispositions; Consequences Negative bullet; Provenance check two new entries)
- **New issues found:** 1 P1 (coordinated cross-document fix needed); 0 P2; 2 P3
- **Round-3 verdict (clean at P1/P2 for ADR set):** confirmed; round-4 P1 is a *new* finding introduced by the amendment's incomplete propagation, not a re-emergence of any round-1/2/3 finding
- **Pyramid coverage map:** N/A (ADR genre)

### P1 — Must Fix

**P1-1 (round 4). ADR-022 partial-update header and ADR-027 §Relationship to ADR-022 paragraph both state the ADR-022 system-prompt amendment remains operative for the `llm-orc invoke` CLI surface — a claim the round-4 amendment to ADR-027 §Context + §Decision establishes as factually incorrect. Coordinated update required across both documents.**

- **Location 1:** ADR-022 partial-update header, final sentence: "The amendment **remains operative** for the `OrchestratorRuntime` ReAct loop on non-chat-completions surfaces — the `llm-orc invoke` CLI surface, the direct ensemble HTTP API (where surfaced for Population B), and any future ReAct-loop surfaces; the system prompt's 'prefer `invoke_ensemble` when capability match exists' commitment continues to apply there."
- **Location 2:** ADR-027 §Relationship to ADR-022 paragraph (which was NOT in the four named round-4 amendment targets): "The amendment remains operative for the `OrchestratorRuntime` ReAct loop on non-chat-completions surfaces (`llm-orc invoke` CLI; future surfaces)."
- **Claim:** The ADR-022 amendment applies to `llm-orc invoke` CLI surface via `OrchestratorRuntime`.
- **Logical gap:** The round-4 amendment to ADR-027 §Context + §Decision establishes that the CLI does NOT use `OrchestratorRuntime` (it routes through `OrchestraService` directly per `cli_commands.py:28`). The header in ADR-022 and the §Relationship paragraph in ADR-027 both encode the stale framing.
- **Body-mutability:** Per Step 2.5, partial-update headers are body-mutable (they ARE the mutable annotation; the underlying ADR-022 body remains immutable). The fix is appropriate.
- **Recommendation:** Coordinated update to both locations:
  1. ADR-022 header: drop the "`llm-orc invoke` CLI surface, the direct ensemble HTTP API" specifics; replace with language reflecting that the amendment remains operative for *any future surface that adopts `OrchestratorRuntime`*, and that per the Tranche 4 conformance-scan Finding 2, `OrchestratorRuntime` currently has no production caller other than the chat-completions handler being replaced by ADR-027.
  2. ADR-027 §Relationship to ADR-022: same correction.

### P2 — Should Fix

None new. The four named amendments are internally consistent (within ADR-027) and externally consistent with ADR-001 + ADR-011 + AS-9 + AS-10.

### P3 — Consider

**P3-1 (round 4). ADR-027 §Relationship to AS-9 first bullet's parenthetical "(preserved on other surfaces per scope)" implies current non-chat-completions surfaces exist — the Tranche 4 finding shows no such current surface uses the orchestrator-LLM-as-decider.**

- **Location:** ADR-027 §Relationship to AS-9 and AS-10, first bullet: "the orchestrator-LLM-as-decider is removed from this surface (preserved on other surfaces per scope)"
- **Recommendation:** Update parenthetical to "available as an architectural option for future surfaces per ADR-001 + ADR-011" to remove the implication of current usage.

**P3-2 (round 4 cross-document). ADR-021's partial-update header (the `> Updated by ADR-027 on 2026-05-22.` block) penultimate sentence carries the symmetric factual staleness as P1-1: "The `OrchestratorRuntime` ReAct loop continues to govern routing on the `llm-orc invoke` CLI surface..." — also factually incorrect.**

- **Location:** ADR-021 partial-update header from ADR-027.
- **Severity:** P3 (the ADR-021 header's claim is narrower than ADR-022's — ADR-021's claim is that `OrchestratorRuntime` governs routing on the CLI, not that the ADR-022 amendment specifically applies; the consequence is smaller). Still factually stale and should be corrected.
- **Recommendation:** Update ADR-021 partial-update header sentence to: "The `OrchestratorRuntime` ReAct loop is available as an architectural option (per ADR-001 + ADR-011) for future non-chat-completions surfaces; per the Tranche 4 conformance scan (Finding 2), it currently has no production caller other than the chat-completions handler — the `llm-orc invoke` CLI and other REST endpoints route through `OrchestraService` directly. The actor-shift described in this header is chat-completions-surface scoped; non-chat-completions surfaces continue to operate per ADR-021's original dispatch shapes."

### Carry-forward confirmation (round 3 verdict preserved)

The round-3 audit cleared the seven Cycle 7 ADRs at the P1/P2 level. The round-4 amendments to ADR-027 + the round-4 findings above are scoped to:

- The four amended sections of ADR-027 (verified correctly applied; no new P2 issues introduced within ADR-027).
- Cross-document staleness in ADR-022 + ADR-021 partial-update headers (the new P1 + P3-2; not present at round 3 because the headers' factual claims about CLI usage were not contradicted by ADR-027's text at that time — the round-4 amendment to ADR-027 made the contradiction visible).
- A residual parenthetical in ADR-027 §Relationship to AS-9 (the new P3-1).

No round-3 finding is re-emergent. The round-3 carry-forward framing findings (F1, F2, F3, NF1) are confirmed unmodified by the round-4 amendments and continue to be carried forward to the practitioner gate.

### Disposition (a/b/c) coherence check

The three candidate dispositions are coherently distinguished and the framing leaves room for ARCHITECT to select without ADR-027 forcing a direction:

- **(a) preserve as architecture-for-future-surfaces** — zero-effort default; correctly framed as "architectural-option preservation."
- **(b) wire `llm-orc invoke` to use `OrchestratorRuntime`** — net-new work; correctly framed (the description accurately notes the CLI's current `OrchestraService`-direct path).
- **(c) mark for removal as unused code** — `refactor:` commit after BUILD ships; ADR-022's amendment becomes dormant code (preserved in version history per body-immutable record); ADR-001 + ADR-011 remain operative as architectural commitments regardless.

All three are coherently distinguished. ADR-027 does not foreclose any disposition.

### AS-9 / AS-10 compatibility

Unaffected. AS-9's structural-bounding property is about role-shape, not codebase class instantiation. AS-10's request-shape commitment is unaffected by `OrchestratorRuntime`'s codebase status.

---

## Section 2: Framing Audit

### Carry-forward confirmation — round 3 framing findings

The round-4 amendments did not modify the sections housing F1, F2, F3, or NF1. All four carry forward unmodified to the practitioner gate:

- **F1 (P2)** — ADR-030 hybrid-as-orthogonal abstract framing.
- **F2 (P2)** — ADR-031 tier classification's active-maintenance framing.
- **F3 (P3)** — "4 confabulation modes" shorthand in ADR-027 + ADR-028.
- **NF1 (P3)** — ADR-030 follow-on trigger asymmetric quantification.

### New framing observations from round-4 amendments

**NF2 (P2). The three dispositions (a/b/c) for `OrchestratorRuntime` are presented as fully symmetric ARCHITECT choices, but the actual default-by-inertia is disposition (a) — if ARCHITECT does not deliberate, the class remains stranded. ADR-027 currently positions the question as a "deferred deliberation" rather than a "required ARCHITECT output."**

- **Location:** ADR-027 §Decision §"OrchestratorRuntime status under ADR-027" + §Consequences §Negative bullet.
- **Observation:** The conformance-scan Finding 2 made the question structural; ADR-027 correctly defers to ARCHITECT but does not make the disposition selection a named ARCHITECT deliverable. Without explicit naming, ARCHITECT may treat it as an optional annotation; the class then remains as unused production code by default.
- **Recommendation:** Add one sentence to the §Consequences §Negative bullet naming the disposition selection as a *required* ARCHITECT design output, not an optional annotation. The cycle's commitment becomes: ARCHITECT names the disposition before BUILD ships ADR-027.

**Framing observation 3 (P3) — coupled with P1-1.** The §Relationship to ADR-022 section's stale sentence (named in P1-1 Location 2) is the same finding from a framing lens as from an argument lens. They land in the same coordinated fix. No separate framing recommendation beyond P1-1.

---

## Round 4 verdict

- **One P1** (coordinated ADR-022 header + ADR-027 §Relationship to ADR-022 update required).
- **Zero new P2** findings within the four targeted amendments.
- **Two new P3** findings (ADR-027 §Relationship to AS-9 parenthetical; ADR-021 partial-update header symmetric staleness).
- **One new framing observation (NF2, P2)** — disposition selection should be a required ARCHITECT deliverable, not a deferred deliberation.
- **Round-3 carry-forward findings** (F1, F2, F3, NF1) confirmed unmodified.

**Gate threshold:** Not met at round 4 — the P1 must be addressed before advancement. The fix is small (coordinated text updates in two documents) and should not introduce new substantive content.
