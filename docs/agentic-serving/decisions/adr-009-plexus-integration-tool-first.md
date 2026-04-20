# ADR-009: Plexus Integration — Tool-First, Injection Later

**Status:** Accepted

**Date:** 2026-04-17

---

## Context

Essay §Integration Architecture identifies two integration modes for Plexus: as tools the orchestrator calls (`query_knowledge`, `record_outcome`) and as context injected into the orchestrator's system prompt at session start. The essay notes both are needed — pre-loaded context answers "what do I already know about this kind of task?" while tool access answers "let me look up the specifics."

Open Question #4 (bootstrapping quality) asks how well the knowledge graph can be pre-populated from existing context versus learning through use. Open Question #7 (enrichment pipeline maturity) asks whether Plexus's enrichment is strong enough to make ingested content queryable in a way that enables stabilization. Both are unresolved — context injection's value depends on the graph being populated with useful, retrievable content at session start, which depends on both open questions.

Tool-based access is less sensitive to either. If the graph is sparse, `query_knowledge` returns nothing and the orchestrator falls back on its own reasoning. If the graph is rich, `query_knowledge` is precise because it queries for what the orchestrator has identified as a gap.

---

## Decision

Plexus integration ships in two phases:

**Phase 1 (this cycle):** Plexus-as-tool. The orchestrator has `query_knowledge` and `record_outcome` as tool calls (per ADR-003). The knowledge graph is queried when the orchestrator identifies a specific knowledge gap. Outcomes, quality signals, and routing decisions are recorded at each completed step that warrants it.

**Phase 2 (deferred):** Context injection at session start. Before the orchestrator processes a request, the serving layer queries Plexus for relevant context and injects it into the orchestrator's system prompt.

Phase 2 is not implemented in this cycle. It is deferred until:
1. Bootstrapping (OQ #4) is understood well enough to know what context is worth injecting
2. The enrichment pipeline (OQ #7) produces content whose quality is high enough to be worth paying the session-start latency and token cost

The Phase 2 architecture does not require changes to the ReAct loop (ADR-001) or the tool surface (ADR-003). Adding it later modifies only the session-start flow.

**Phase 2 hook point is structurally reserved.** The ARCHITECT-phase design for the session-start flow shall include a pre-orchestration stage where context injection can be inserted without modifying the ReAct loop or the orchestrator tool surface. Phase 1 leaves this stage empty (or a no-op pass-through); Phase 2 populates it. This prevents silent deletion — the deferral is a configuration of a reserved stage, not an absence of structure.

**Technical rationale for Phase 1 sufficiency.** Phase 1's tool-first integration is sufficient when the orchestrator's Model Profile is capability-adequate for reasoning about *when* to emit a `query_knowledge` tool call — that is, the orchestrator can recognize knowledge gaps on its own. Phase 2 becomes necessary when orchestrator profile capability is limited and query load is high (the knowledge-compensated model selection hypothesis, OQ #1), because a smaller model relies more on session-start context to compensate for weaker in-loop tool selection. Sequencing Phase 1 before Phase 2 is therefore consistent with the rollout of profile experimentation: first validate the system with a capable profile, then add injection as smaller-profile experimentation begins.

---

## Consequences

**Positive:**
- Simpler first integration — the entry point is a single tool-call pattern
- Orchestrator behavior is insensitive to Plexus population quality early on — sparse graph degrades gracefully to unused-tool behavior
- Phase 2 can be added without modifying Phase 1 components
- Consistent with AS-8 — the baseline product ships without Plexus; Phase 1 with Plexus is the next step

**Negative:**
- Every Plexus consultation incurs tool-call latency. Context injection, once added, could pre-empt many of these
- The orchestrator's baseline awareness of "what the system knows" is zero at session start — it must think to query. For cold-start sessions, this leaves value on the floor. This interacts with ADR-011: smaller orchestrator profiles (the "knowledge-compensated model selection" hypothesis, OQ #1) may be less reliable at deciding when to call `query_knowledge`. Phase 1 performance across model tiers is partially a function of each model's tool-selection discipline, not only its reasoning capability
- Phase 2's eventual value depends on OQ #4 and OQ #7 resolving favorably; if they do not, Phase 2 may never be worth implementing

**Neutral:**
- Phase 2 is a sequencing decision, not a permanent rejection. A future ADR supersedes this one if the sequencing changes
- The `query_knowledge` / `record_outcome` schemas are defined here as tool-level contracts; their shape can evolve in future ADRs without changing this one
