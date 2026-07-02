# ADR-001: Internal ReAct Loop as Execution Model

> **Superseded by ADR-046 on 2026-07-01.** The internal ReAct loop is superseded: under the orchestrator-actor dissolution, the client (OpenCode / any caller) owns the loop and each request is one declarative classify→seat→marshal ensemble pass (ADR-046 §1) — no persistent internal orchestrator. This is the reversal of ADR-001's internal-model choice, reached via OQ #26 (callee / client-owned loop → ADR-033). **AS-1 survives independently:** dynamic invocations by a composing role sit outside the ensemble reference graph (Invariant 7) — that framing does not depend on the internal-loop mechanism. Body immutable.

**Status:** Superseded by ADR-046 on 2026-07-01 (was: Accepted)

**Date:** 2026-04-17

---

## Context

Essay §Orchestrator Agent surveyed three architectural options for where the agentic loop sits: external (loop stays in the outer tool, llm-orc serves as MCP tool provider), internal (llm-orc runs its own ReAct loop behind the serving layer), and hybrid (declarative DAG edges supplemented by LLM-driven routing edges inside the ensemble engine). The external model already exists via the conductor skill. The hybrid model requires modifications to the existing ensemble execution engine.

Summarization is not what distinguishes the options. ADR-004 makes result summarization mandatory for the internal model too; the external model would need it equally. The genuine differentiator is the integration path for cross-session memory and self-building ensembles: the internal model sits at a code location that can naturally integrate Plexus as a tool (ADR-009) and author new compositions (ADR-006) without requiring the outer tool to mediate. The external model defers all of that to whichever outer tool is in use, and no outer tool currently provides it.

AS-1 establishes that dynamic invocations by an orchestrator agent sit outside the ensemble reference graph; Invariant 7 still governs static YAML composition. Whichever execution model is adopted must preserve this separation.

Open Question #3 in the domain model remains open: whether the genuine gap is the execution model or summarization quality. Framing audit FI-1 on this cycle notes the external+summarization option is underengaged relative to its "dramatically simpler architecture" framing in the essay's own assumption inversion #5. This ADR does not close OQ #3 — it commits to the internal model as the pragmatic entry point while leaving the question available to a future cycle to reopen if the self-building and cross-session memory benefits turn out not to materialize.

---

## Decision

The agentic serving feature implements an **internal ReAct loop** inside the llm-orc server. An orchestrator agent sits behind `/v1/chat/completions`, receives requests from clients, and delegates to llm-orc operations via tool calls (`invoke_ensemble`, `compose_ensemble`, `list_ensembles`, `query_knowledge`, `record_outcome`). The ensemble execution engine (Layer 3 of the four-layer architecture) is unchanged.

The external MCP model remains available for clients that prefer to run the loop themselves; it is not replaced. The hybrid model is deferred — it is not attempted in this cycle.

---

## Consequences

**Positive:**
- Reuses the existing ensemble execution engine unchanged. New surface area is bounded to the orchestrator agent + serving layer
- Preserves Invariant 7 for the ensemble reference graph — orchestrator tool calls are not static YAML references (AS-1)
- Matches the convergent pattern of mature open-source agent platforms (OpenHands, claw-code)
- Enables Plexus integration as Layer 4 without modifying Layer 3

**Negative:**
- Adds a second execution model (ReAct loop) alongside the existing DAG engine. Both must be maintained
- Does not close OQ #3 — if summarization quality turns out to be the actual gap, this ADR's scope of effort was larger than necessary
- New surface area for failure modes: context rot, unbounded tool-call loops, orchestrator LLM capability limits

**Neutral:**
- The external MCP model continues to exist via the conductor skill. Agentic serving is additive
- Event-sourced vs. direct-loop implementation of the orchestrator remains undecided (flagged in the essay, not a blocker for the entry point)
