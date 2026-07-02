# ADR-002: Four-Layer Architecture with Plexus Optional

> **Updated by ADR-046 on 2026-07-01 (supersedes the ADR-016 amendment below).** Two changes follow from the orchestrator-actor dissolution (ADR-046 §3):
> 1. **Layering-rule revert.** ADR-016 is superseded, so its "edges never upward, with one exception (the L0→L1 calibration channel)" amendment is **struck — the rule reverts to no exception, edges never upward.** (The literal rule text in `system-design.md` reverts with the deferred Cycle-8 BUILD current-state sweep, when the `calibration_signal_channel` code is removed.)
> 2. **Layer-2 definition superseded.** Decision item 2 below — "Orchestrator Agent — ReAct loop powered by a model profile, exercising a fixed tool surface (ADR-003)" — no longer describes the architecture. Under ADR-046 the orchestrator actor is dissolved: the client owns the loop, and the per-turn serving handler is **one declarative ensemble (classify → seat → marshal)** — typed seats joined by the common I/O envelope (ADR-024), the capability seat filled by runtime dynamic dispatch, with no persistent internal ReAct loop and no fixed LLM-facing tool surface. The four-layer split, Plexus optionality (AS-8), and Layers 1 / 3 / 4 remain current.
>
> Body immutable.
>
> **Updated by ADR-016 on 2026-05-06 — SUPERSEDED by ADR-046 (see above).** The layering rule's "edges never upward" property was amended for a single narrow exception — a read-only signal channel from L0 to L1, gated by five bounding mechanisms specified in ADR-016. This amendment is now reverted.

**Status:** Updated by ADR-046 on 2026-07-01 (Layer-2 definition superseded; ADR-016 layering-rule exception reverted) — core four-layer split + Plexus optionality remain Accepted

**Date:** 2026-04-17

---

## Context

Essay §What Emerges defines a four-layer system: Layer 1 (API Surface / serving layer), Layer 2 (Orchestrator Agent), Layer 3 (Ensemble Engine — existing DAG engine), Layer 4 (Knowledge Graph — Plexus).

AS-8 establishes that Plexus is optional: the orchestrator agent, serving layer, budget enforcement, result summarization, conversation compaction, and ensemble composition all function without Plexus. When Plexus is absent, the orchestrator operates statelessly — no cross-session memory, no stabilization, no bootstrapping. Plexus transforms a stateless orchestrator into a learning one, but is not a prerequisite for agentic serving.

Open Question #7 (enrichment pipeline maturity) and #4 (bootstrapping quality) concern Plexus's readiness as a learning substrate. The layered architecture must not bet the baseline product on those questions being answered favorably.

---

## Decision

The agentic serving feature is built as four independently operable layers:

1. **Serving Layer** — OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/models`) on the existing FastAPI server
2. **Orchestrator Agent** — ReAct loop powered by a model profile, exercising a fixed tool surface (ADR-003)
3. **Ensemble Engine** — the existing DAG execution engine, unchanged
4. **Knowledge Graph** — Plexus, providing cross-session memory, provenance, and stabilization

**Layers 1-3 constitute the baseline product.** They must compose into a functioning system with no Plexus dependency. Code paths that integrate with Plexus must be guarded such that their absence causes no functional degradation of Layers 1-3.

**Layer 4 is an additive upgrade.** Its presence converts a stateless orchestrator into a learning one.

---

## Consequences

**Positive:**
- Baseline product does not depend on resolving OQ #4 or OQ #7 — agentic serving ships whether or not Plexus's enrichment pipeline matures
- Operators without Plexus deployed still get value from agentic serving
- Plexus integration can iterate at its own pace without blocking serving layer releases
- Layer 3 is not modified — existing llm-orc deployments are not disrupted

**Negative:**
- Two code paths to maintain for orchestrator behavior: with-Plexus and without-Plexus. Feature parity between them is a constraint on new orchestrator capabilities
- Stateless mode has no cross-session learning — the "economics over capability" value proposition (essay reflections) only fully applies once Plexus is active
- Integration boundaries must be explicit enough that a missing Plexus does not silently degrade the system

**Neutral:**
- The distinction between "stateless baseline" and "learning upgrade" must be communicated clearly in operator-facing docs; users deploying without Plexus should not experience it as a degraded mode
