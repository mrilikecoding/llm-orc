# ADR-010: Plexus Ingestion Boundary — Source Material, Push Model

> **Partial update by ADR-046 on 2026-07-01.** The push-model ingestion boundary survives unchanged (source-material-only, client-driven, non-blocking; AS-4). Only the `record_outcome`-tool-call phrasing in Decision item 2 is stale — under ADR-046 the KG-write access re-homes from an orchestrator tool call to an engine/script operation (see the ADR-009 update). The push-model, source-material, enrichment-as-quality-gate, and non-blocking properties are unaffected. Body immutable.

**Status:** Accepted; partial update by ADR-046 on 2026-07-01 (write-access phrasing re-homed)

**Date:** 2026-04-17

---

## Context

AS-4 establishes that the ingestion boundary is source material: the knowledge graph ingests file content (source material), not LLM-generated summaries or interpretations. Quality emerges from the enrichment pipeline, not from upstream curation.

DISCOVER feed-forward signals #9 and #14 clarified the operational shape: Plexus operates as a lib (push model), not a server — the client (llm-orc) drives ingestion. The bootstrapping pipeline is llm-orc artifacts → Plexus ingestion (file content) → enrichment → queryable graph. The "garbage in" concern is addressed by ingesting source material, not LLM summaries.

Open Question #7 (enrichment pipeline maturity) flags that AS-4 and AS-5 depend on enrichment being effective. This ADR commits to the ingestion-boundary invariant; it does not resolve the open question about whether enrichment works well enough in practice.

---

## Decision

Plexus ingestion, when active, obeys these rules:

1. **Source material only.** What is ingested is the file content itself — ensemble YAML, script source, agent output artifacts, execution logs, user-provided documents. LLM-generated summaries, interpretations, or abstractions are not ingested. If the operator wants an LLM-generated summary in the graph, it is written to disk as a file first, then ingested as source — making the provenance (which LLM, when, under what prompt) part of the source material itself.

2. **Push model.** The client (llm-orc, via its bootstrapping and `record_outcome` paths) drives ingestion. Plexus does not pull. This makes ingestion a deliberate llm-orc operation, auditable and configurable on the client side.

3. **Enrichment is the quality gate.** Extraction of signal from ingested sources happens asynchronously via Plexus's enrichment pipeline. Quality is asserted by enrichment, not by curation of what gets ingested. Declarative adapters on the llm-orc side and core enrichments on the Plexus side reinforce stronger signals in the graph.

4. **Ingestion is non-blocking.** A session does not wait for ingestion to complete before proceeding. Recorded outcomes and provenance become queryable after enrichment, not immediately.

---

## Consequences

**Positive:**
- Provenance is preserved — what is queryable traces back to its source, not to a lossy interpretation
- Compatible with AS-8 — Plexus ingestion is a llm-orc-driven operation, optional, does not change baseline serving behavior
- Push model matches Plexus's current operational shape (DISCOVER #9)
- The "garbage in" concern is localized to enrichment quality, not to ingestion gatekeeping

**Negative:**
- Querying the graph for a recent routing decision may return stale results while enrichment is still running — orchestrator must tolerate eventual consistency
- System value is gated by enrichment pipeline maturity (OQ #7). If enrichment is weak, AS-4 and AS-5 are hollow in practice
- Ingestion of source files may grow the graph rapidly — enrichment cost scales with ingested volume, not with query load

**Neutral:**
- The specific set of llm-orc artifacts that are ingestible (ensemble YAML, profiles, scripts, outcome records, conversation transcripts, etc.) is a build-time decision. What the ADR commits to is the shape of the boundary (source, push, async), not the contents
