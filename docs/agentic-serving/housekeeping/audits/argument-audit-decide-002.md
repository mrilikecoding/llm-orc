# Argument Audit Report — Round 2 (Re-audit)

**Audited documents:** ADR-001 through ADR-011 (`docs/agentic-serving/decisions/`)
**Source material:** `essays/001-agentic-serving-architecture.md`, `product-discovery.md`, `domain-model.md` (scoped), `domain-model.md` (project-level)
**Prior audit:** `housekeeping/audits/argument-audit-decide-001.md`
**Date:** 2026-04-17

---

## Section 1: Resolution Status

**P1-A** (ADR-004, recovery path conditional on Plexus) — **Resolved.**
The negative consequence was restructured into a two-part statement. The Plexus-active case (recovery via `query_knowledge` / `record_outcome` / ADR-010 ingestion) is separated from the Plexus-absent case (summary is the only form the orchestrator retains; recovering lost detail requires operator intervention outside the session). The conditionality is now explicit and accurate.

**P1-B** (ADR-007, contradiction between session-scoped re-calibration and local-tier metadata persistence) — **Resolved.**
Decision point 4 was rewritten to state that calibration runs within the session, quality signals accumulate in session artifacts, and trust may transition within the session — but does not persist across sessions without Plexus. The neutral consequence was aligned: calibration state lives in Plexus when active, in session memory when absent, and is rebuilt each session in stateless mode. The earlier local-tier metadata claim is gone. The two claims now say the same thing.

**P2-A** (ADR-001, summarization framed as a differentiator between models when both require it) — **Resolved.**
The context section was substantially rewritten. It leads with the explicit statement that summarization is not what distinguishes the options, that ADR-004 makes it mandatory for the internal model too, and that the genuine differentiator is the Plexus/memory/self-building integration path unavailable to the external model without outer-tool mediation. The argument is now grounded in an actual asymmetry.

**P2-B** (ADR-008, baseline autonomy uncalibrated for non-operator tool users) — **Resolved.**
The negative consequences section now explicitly names the "endpoint is a model" persona from product discovery, identifies that silent ensemble composition may be surprising to that user, and calls out that deployments targeting pure tool-user sessions should configure a tighter default deliberately. The Autonomy Level configuration surface is noted as the mechanism. The ADR does not pretend the baseline fits both personas equally.

**P2-C** (ADR-011, negative consequence implies mid-session escalation is unavailable when it is available via ensemble composition) — **Resolved.**
The negative consequence was extended with a parenthetical that directly addresses this: tiered routing of individual tasks via the triage-route ensemble pattern remains fully available; what is session-scoped is only the orchestrator's own LLM, not the library's capacity to route across tiers. The distinction between orchestrator-level swap and library-level routing is now explicit.

**P3-A** (ADR-003, fixed tool surface implies more restriction than it delivers) — **Resolved.**
The negative consequence was revised to include the clarification parenthetically: the effective capability surface is the library, not the five tools, and composition via `compose_ensemble` extends that scope at runtime within operator-curated primitives. The fixed surface is now accurately scoped to orchestrator-level actions, not system capability.

**P3-B** (ADR-009, "must think to query" interacts with ADR-011 profile selection) — **Resolved.**
The negative consequence was extended to name the interaction: smaller orchestrator profiles targeted by the knowledge-compensated model selection hypothesis (OQ #1) may be less reliable at deciding when to call `query_knowledge`, making Phase 1 performance across model tiers partially a function of each model's tool-selection discipline. The cross-ADR dependency is now visible.

**P3-C** (ADR-005, budget sizing guidance does not account for sub-invocation multiplier) — **Resolved.**
The negative consequences section was extended with an explicit statement: a single orchestrator turn may trigger an ensemble execution + mandatory result summarization (ADR-004) + calibration check (ADR-007), each counting against the session token budget. Turn-count sizing assumes a single LLM call per turn; token-ceiling sizing must account for this multiplier. The cross-ADR cost interaction is now named.

---

## Section 2: New Issues Introduced by Revisions

### P2 — Should Fix

**N-P2-A**
- **Location:** ADR-011, Consequences → Negative, parenthetical extension
- **Claim:** "Tiered *routing* of individual tasks — the essay's local-triage-then-escalate pattern — remains fully available through the triage-route ensemble pattern invoked by the orchestrator; what is session-scoped is only the orchestrator's own LLM, not the library's capacity to route across tiers."
- **Evidence gap:** This claim subtly reframes the "session-scoped profile" decision as having no effective cost relative to a built-in tiered fallback, because the triage-route ensemble covers it. But the triage-route ensemble itself must be authored and registered in the library before it can be invoked — it is not automatic. The parenthetical implies triage routing is immediately available, when it is actually available only if an operator has composed and promoted a triage-route ensemble. For a fresh deployment with an empty library, the negative consequence of blocking mid-session LLM swap is the full cost, not a cost softened by an available alternative.
- **Recommendation:** Qualify the parenthetical: "...remains fully available *once a triage-route ensemble has been composed and promoted to the library*." One clause, honest about the precondition.

### P3 — Consider

**N-P3-A**
- **Location:** ADR-001, Context, paragraph 2 (the revised section)
- **Claim:** "The external model defers all of that to whichever outer tool is in use, and no outer tool currently provides it."
- **Interaction note:** The phrase "currently" encodes a temporal claim that may become stale. The conductor skill is named as an existing outer tool in the same context section. The statement is accurate at the time of writing but could mislead a future reader who encounters it post-conductor-skill enhancement. The claim is not wrong, but its temporal dependency is unmarked.
- **Recommendation:** Rephrase to "no outer tool in the current stack provides it" or move the claim into a parenthetical that makes the scope explicit.

**N-P3-B**
- **Location:** ADR-004, Consequences → Negative, the revised two-part consequence
- **Claim:** The revision states that in the Plexus-active case, "full artifacts on disk are reachable through the same pathway via ingestion (ADR-010)." This is accurate but introduces a new implicit assumption: that the orchestrator's `query_knowledge` call can reach specific artifact content that was ingested from disk, rather than only structured outcome signals.
- **Interaction note:** ADR-010 commits to ingesting source material including "agent output artifacts, execution logs." But the query interface of `query_knowledge` is not specified in any ADR — whether it supports retrieving ingested artifact content versus returning structured Plexus graph nodes is left to implementation. The recovery claim is architecturally sound but depends on a query interface capability not committed to in the ADR chain.
- **Recommendation:** Qualify the artifact-retrieval recovery path as dependent on the `query_knowledge` schema (not yet specified). Low urgency — this is a forward reference to build-phase design, but the recovery claim should not overreach what is currently committed.

---

## Section 3: Fresh Framing Pass

The revisions tightened argument quality without materially changing what the ADR chain emphasizes or omits. Two new framing observations emerge from the revised text.

**FF-1 (P3 — user judgment)**
- **Location:** ADR-001, Context (revised), and ADR-004 (revised)
- **Observation:** The revised ADR-001 now correctly positions Plexus integration as the genuine differentiator for the internal model. This makes the Plexus-absent case's value proposition structurally thinner than the revised framing acknowledges. In stateless mode: summarization is mandatory (ADR-004), calibration re-runs each session (ADR-007), there is no cross-session memory (ADR-002), knowledge-compensated model selection is not testable (ADR-011), and the triage-route ensemble must be authored before tiered routing works (ADR-011, revised). The internal model in stateless mode is the external model plus one additional maintenance burden (the ReAct loop) but without the external model's advantage of living in Claude Code's already-mature harness. The revised ADR-001 framing describes the stateless case as a "pragmatic entry point" — that framing is coherent only if stateless deployment is expected to be short-lived. If Plexus integration is delayed or stalls (OQ #7), the internal model's standing case weakens in proportion. This is not new evidence — OQ #3 and FI-1 from round 1 address the same structural uncertainty — but the revision's success in clarifying the Plexus-dependent differentiator also makes the stateless case's thinness more visible than before.

**FF-2 (P3 — user judgment)**
- **Location:** ADR-008, Consequences → Negative (revised)
- **Observation:** The revision explicitly names the "endpoint is a model" persona and calls out that deployments targeting pure tool-user sessions need deliberate configuration of a tighter default. This is a genuine improvement. However, the ADR still does not characterize what a "pure tool-user" serving configuration would look like — tighter to what degree? What Autonomy Level expression disables silent composition? The framing now acknowledges a gap but does not point toward a resolution. For a scenario-writing phase this could become load-bearing: the scenario set should include a pure tool-user session to exercise the uncalibrated-composition surprise path. The ADR's acknowledgement is now visible enough to surface this need clearly.

---

## Section 4: Advance Recommendation

**The ADR chain is clean enough to advance to scenario-writing.**

All five non-trivial issues (P1-A, P1-B, P2-A, P2-B, P2-C) are genuinely resolved — not merely reworded. The resolutions address the underlying logical gaps: the Plexus conditionality is now correctly split; the calibration contradiction is gone; the differentiator argument is grounded in an actual asymmetry; the non-operator persona is acknowledged; and the mid-session escalation negative consequence is accurate without being misleading.

The two new P2/P3 issues (N-P2-A, N-P3-A, N-P3-B) are introduced by the revisions but are narrow. N-P2-A is the most substantive: the triage-route availability claim overstates immediacy and should be qualified before the scenarios depend on it. N-P3-A and N-P3-B are clarifications the revision created by tightening adjacent claims.

Recommended before closing the DECIDE gate: apply the one-clause fix to ADR-011's triage-route parenthetical (N-P2-A). The other two P3 notes can be deferred to the build phase.

The three FI observations from round 1 remain open for user judgment at the gate, as intended. FF-1 above sharpens FI-1's concern — the revised framing makes the Plexus-dependency more explicit, which is correct, and the stateless value proposition is now more visibly thin. That is the user's gate decision, not a defect in the ADR chain.
