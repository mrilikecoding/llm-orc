# ADR-003: Fixed Orchestrator Tool Surface

**Status:** Accepted

**Date:** 2026-04-17

---

## Context

AS-6 establishes that the orchestrator composes from existing primitives only — it cannot author arbitrary scripts or create new model profiles. The domain concept Orchestrator Tool defines the orchestrator's action space as a fixed set of operations it can invoke via tool call.

The essay lists five candidate operations: `invoke_ensemble`, `compose_ensemble`, `list_ensembles`, `query_knowledge`, `record_outcome`. These map directly to the orchestrator's jobs in product discovery (§Orchestrator LLM): match tasks to ensembles, compose from primitives when nothing fits, know what has worked before.

A counterfactual is worth surfacing: extending the tool surface dynamically at runtime (e.g., letting the orchestrator author new tools, as explored in *LLM Agents Making Agent Tools*, ACL 2025). This is rejected at the entry point because it enlarges the audit surface and undermines predictability, which the operator stakeholder's visibility and cost-control jobs explicitly require.

---

## Decision

The orchestrator agent exposes exactly these tools, and no others:

| Tool | Purpose |
|------|---------|
| `invoke_ensemble` | Execute an existing ensemble with given input |
| `compose_ensemble` | Create a new ensemble at runtime from existing primitives (per ADR-006), producing a named ensemble that can then be invoked |
| `list_ensembles` | Enumerate the library — ensembles, profiles, scripts available for composition and invocation |
| `query_knowledge` | Read from Plexus (no-op if Plexus is absent, per ADR-002) |
| `record_outcome` | Write routing decisions, quality signals, and outcomes to Plexus (no-op if Plexus is absent) |

The set is closed. Adding a new orchestrator tool requires a new ADR. The orchestrator cannot generate, register, or invoke tools outside this set.

---

## Consequences

**Positive:**
- Predictable orchestrator action space — the operator knows what the orchestrator can do
- Supports the operator's visibility and cost-control jobs from product discovery
- Preserves AS-6: composition is dynamic, but the composable *primitives* (scripts, profiles) remain operator-curated
- Auditable — every orchestrator turn's tool call falls within a known vocabulary

**Negative:**
- Loses the flexibility of dynamic tool creation demonstrated in agent-building-agent literature (the effective capability surface is the library, not the five tools — library scope determines the orchestrator's task space, and composition via `compose_ensemble` extends that scope at runtime within operator-curated primitives)
- Adding new capabilities requires an ADR + implementation cycle, not runtime agent action

**Neutral:**
- The set is minimal by design. It can be extended in future ADRs if a concrete need surfaces that existing tools cannot serve
