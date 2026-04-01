# Active RDD Cycle: Agentic Serving

**Started:** 2026-03-20
**Current phase:** MODEL (next)
**Artifact base:** `docs/agentic-serving/`
**Essay:** `essays/001-agentic-serving-architecture.md`

## Phase Status

| Phase | Status | Artifact | Key Epistemic Response |
|-------|--------|----------|----------------------|
| RESEARCH | ✅ Complete | `essays/001-agentic-serving-architecture.md` | The DAG engine and ReAct loop are complementary, not competing. Plexus as knowledge graph doesn't just add memory -- it lowers the capability threshold for the orchestrator LLM, converting reasoning into retrieval. The cost framing may matter more than the capability framing. |
| DISCOVER | ✅ Complete | `product-discovery.md` | Plexus as lib (push model) means ingestion boundary is file content, not LLM output -- quality emerges from enrichment, not curation. Tool user and ensemble author often the same person -- visibility is tinkering, not just debugging. Conductor ceiling unknown; orchestration may require frontier models regardless of graph population. Bootstrapping pipeline has a shape but quality gate is the enrichment layer, not upstream curation. |
| MODEL | ☐ Pending | -- | -- |
| DECIDE | ☐ Pending | -- | -- |
| ARCHITECT | ☐ Pending | -- | -- |
| BUILD | ☐ Pending | -- | -- |
| INTEGRATE | ☐ Pending | -- | -- |
| PLAY | ☐ Optional | -- | -- |
| SYNTHESIZE | ☐ Optional | -- | -- |

## Feed-Forward Signals

### From RESEARCH
1. The orchestrator agent pattern (ReAct loop with ensemble invocation as primary tool) is the pragmatic entry point -- not the hybrid model
2. Invariant 7 (static ensemble references) does not govern orchestrator tool-call invocations, but orchestrator-created ensembles must still satisfy it. Validation mechanism needed at creation time
3. Plexus integration is the differentiator: no comparable system provides persistent structured memory across agentic sessions
4. The knowledge-compensated model selection hypothesis (populated graph enables cheaper models) is commercially significant and testable but unvalidated
5. Context management (summarization, compaction, Plexus offloading) is a correctness requirement, not an optimization
6. Budget enforcement (turn limits, token budgets) is a control plane concern validated by both OpenHands and claw-code

### From RESEARCH (reflections)
7. The cost framing (economics over capability) may be the stronger value proposition -- "80% quality at 10% cost" resonates more than "marginally better quality"
8. Open question for domain model: can well-orchestrated smaller models + populated knowledge graph compete with frontier models on quality while winning on cost?

### From DISCOVER
9. Plexus operates as a lib (push model), not a server -- the client drives ingestion of file content, enrichments extract signal. Quality gate is the enrichment pipeline, not upstream curation
10. The orchestrator needs the full composition palette (ensembles + profiles + scripts), including ensemble-to-ensemble references -- the "restrict to profile-and-script only" fallback from the essay is insufficient
11. Tool user and ensemble author are often the same person -- visibility is part of the value proposition (tinkering), not just an operational concern
12. Organic stabilization over explicit curation -- repeated patterns should surface on their own, but quality signals (not just frequency) must inform what stabilizes
13. The conductor skill's ceiling is unknown -- orchestration is broad AND deep, and may require frontier models regardless of graph population. The knowledge-compensated model selection hypothesis remains unvalidated
14. Bootstrapping pipeline: llm-orc artifacts → Plexus ingestion (file content) → enrichment → queryable graph. Background/async ingestion is practical. "Garbage in" concern is addressed by ingesting source material, not LLM summaries

## Context for Resumption

This is a scoped RDD cycle for the agentic-serving feature of llm-orc, a declarative DAG-based LLM orchestration engine. The research phase investigated whether llm-orc can serve as the backend for agentic coding tools via OpenAI-compatible endpoints. Six questions were explored across two research cycles: API surface requirements, DAG-to-ReAct mapping, self-building ensembles, Plexus as memory layer, OpenHands architecture, and claw-code architecture.

The essay concludes with a four-layer architecture: API surface, orchestrator agent, ensemble engine (existing), and knowledge graph (Plexus). Product discovery surfaced seven value tensions and six assumption inversions. Key product insight: Plexus operates as a push-model lib where the client drives ingestion of file content (not LLM summaries), and enrichments extract signal -- quality emerges from the enrichment pipeline, not upstream curation. The tool user and ensemble author are often the same person, making visibility a product feature (tinkering) not just an operational concern.

Citation and argument audits have been run. All P1 issues were remediated in the essay. Some P2 issues remain unresolved. The next phase is MODEL -- domain vocabulary extraction from the essay and product discovery artifact.
