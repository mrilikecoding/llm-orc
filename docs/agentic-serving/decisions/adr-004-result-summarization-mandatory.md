# ADR-004: Result Summarization Is Mandatory Before Orchestrator Context

**Status:** Accepted

**Date:** 2026-04-17

---

## Context

AS-7 establishes that result summarization is a correctness requirement, not an optimization. Essay §Context Management cites Chroma research showing that LLM performance degrades as context fills even within technical token limits. An orchestrator that accumulates full ensemble result dictionaries across multiple tool calls will degrade in quality over the course of a session — the "context rot" problem.

Ensemble result dictionaries carry per-agent outputs, intermediate reasoning, artifact references, and metadata. The full dictionary is essential to the ensemble's artifact record (Invariant 9 — child artifacts are nested within parents) but is counterproductive when fed back into the orchestrator's reasoning context.

---

## Decision

Every ensemble invocation triggered by the orchestrator agent MUST produce a summarized result before that result enters the orchestrator's conversation context. The full ensemble result dictionary is retained in the execution artifact (unchanged — Invariant 9 is preserved); only the summary enters the orchestrator's context.

**Default implementation:** a dedicated summarizer — itself an ensemble invokable by the orchestrator control plane, not by the orchestrator LLM. Summarization runs between the ensemble completing and the orchestrator receiving the tool-call result.

**Escape hatch:** a per-ensemble or per-invocation flag may indicate that the raw output is small enough to pass through directly (e.g., a classifier returning a single label). The default behavior is summarization; opting out is explicit.

Unsummarized ensemble results MUST NOT reach the orchestrator's context by any path.

---

## Consequences

**Positive:**
- Protects orchestrator reasoning quality across long sessions (AS-7)
- Summarization is an ensemble — reuses the existing execution engine, benefits from profile/model flexibility
- Full detail is preserved in artifacts for audit, debugging, and Plexus ingestion
- Escape hatch keeps simple cases simple

**Negative:**
- Every orchestrated invocation incurs summarization cost (tokens + latency)
- Summary quality depends on the summarizer's own capability — an inadequate summarizer becomes the bottleneck
- Detail lost at summarization is recoverable only through explicit retrieval. When Plexus is active (ADR-002, ADR-009), `record_outcome` writes structured signal into the graph and `query_knowledge` retrieves it — full artifacts on disk are reachable through the same pathway via ingestion (ADR-010). When Plexus is absent (stateless mode), the summary is the only form of the result the orchestrator retains. Direct artifact file inspection is not in the orchestrator's tool surface (ADR-003); recovering lost detail in stateless mode requires operator intervention outside the session

**Neutral:**
- Summarization is a natural Plexus ingestion point for the ensemble's *full* output (raw artifact), independent of what the orchestrator sees. Per AS-4, Plexus ingestion boundary is source material — the full artifact is the source, the orchestrator's summary is not
