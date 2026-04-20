# Gate Reflection: Agentic Serving DECIDE → ARCHITECT

**Date:** 2026-04-19
**Phase boundary:** DECIDE → ARCHITECT
**Cycle:** agentic-serving (scoped cycle)

## Belief-mapping question composed for this gate

> FF-1 sharpens FI-1. Together they point at something real. ADR-001's Plexus-as-differentiator framing is correct (ADR-004 removes summarization as the differentiator), but it also makes the entry-point cost/benefit depend on Plexus eventually working. If enrichment maturity (OQ #7) stalls, the stateless internal orchestrator offers: cross-session memory? no. Self-building? yes, but without cross-session retention. Vs. the conductor-skill external path which offers neither but needs no new maintenance.
>
> What would you need to believe about Plexus's enrichment pipeline (OQ #7) — or about the standalone value of runtime ensemble composition without cross-session memory — for ADR-001's commitment to the internal ReAct loop to be right even if enrichment stays weak for an extended period?

## User's response

User's first response:

> There are a few layers to the Plexus thinking here that I want to untangle. One is the idea of cross-session memory. But the other, and I think more compelling frame, is that Plexus enables the multi-agent orchestration to be more effective. In Plexus, what's taking shape is the idea of consumers registering their own lens for how that consumers content is ingested into the graph and how the graph is queried back using the grammar of that lens. I'm thinking out loud here so-to-speak, so take this as just an idea. But say an llm-orc model driving one subagent ensemble process here is registering a lens grammar that enriches the graph and another subagent ensemble process can query the full graph using its own lens grammar. So shared context (the context being 1:1 with the llm-orc agentic session) between subagents is queryable using the grammar of the specific subagent. What the subagents encode into the graph is enriched by each other agent's lens. So Plexus here enables that kind of cross-talk between agents for intelligence sharing. On the one hand, the research shows that agents talking with one another can reinforce agent biases. But a hypothesis here could be that shared KG intelligence like this doesn't have the same effect.

User's second response (clarifying after agent tensioned "lens" against AS-4 and asked which of two readings applied):

> The lens for a consumer is about the query surface — if we ingest content into Plexus we need to specify how we encode in such away that we can query it back. There are many strategies there, but it's the former of your two statements. I was pointing out though that different agents or ensembles may have their own registered spec that define its own lens. So rather than llm-orc agentic having one adapter with one lens, perhaps instead within the ensembles there are independent lenses. The bias question I'm not sure about. I'm also not sure this is a blocking concern. This likely needs to fold back into another cycle later when the Plexus design is more settled. Just wanted to call out how Plexus's query surface is taking shape.

## Pedagogical move selected

Challenge (belief-mapping), with an inline Grounding Reframe invoked after the user's first response when the reframe rested on in-design Plexus work the agent could not verify.

## Commitment gating outputs

**Settled premises (the user is building on these going into ARCHITECT):**

- The 11 ADRs as drafted (including the post-audit revisions) — scope, numbering (cycle-local), decisions, consequences
- Budget sizing anchor: "dozens of turns or more," with outer bound of "running an RDD phase within a session." Specific numbers deferred to build; enforcement mechanism fixed (ADR-005)
- Full composition palette with validation (ADR-006) — essay's "restrict to profile-and-script" fallback is superseded
- Plexus optional (AS-8) — baseline product does not depend on Plexus or on OQ #4 / OQ #7 resolving favorably
- AS-4 preserved under the multi-lens reframe — a lens is a query-surface grammar applied during enrichment, not an LLM-summary ingestion path
- Conformance debt on ADR-006 is a structural refactor prerequisite, captured as `refactor:` scenarios in scenarios.md

**Open questions (the user is holding these open going into ARCHITECT):**

- OQ #2: visibility format unresolved. Shape of visibility surface (events, logs, dashboard, inline) is not committed
- OQ #3: internal ReAct loop vs. external with better summarization — ADR-001 commits to internal for the entry point but does not close the question
- OQ #4, #7: Plexus bootstrapping quality and enrichment pipeline maturity — not blocking (AS-8), but load-bearing for the learning-system value proposition
- OQ #8 (new, added at this gate): per-ensemble lens registration and whether KG-mediated cross-talk avoids the agent-bias-reinforcement effect — folds back in a later cycle when Plexus's lens design is more settled
- Event-sourced vs. direct-loop orchestrator implementation — flagged in the essay, not decided, not a blocker
- Client tool surface boundary — how the orchestrator handles client-declared tools (bash, file-edit, etc.) vs. its own internal tool surface (ADR-003). Discovered while writing interaction specs. Decision needed in ARCHITECT or a follow-up DECIDE mini-cycle
- FF-2 scenario: pure tool-user session exercising uncalibrated-composition surprise path — not yet in scenarios.md

**Specific commitments carried forward to ARCHITECT:**

- The four-layer architecture (serving, orchestrator, ensemble engine, knowledge graph) with independent operability per layer (ADR-002)
- Fixed orchestrator tool surface of five tools (ADR-003)
- Session-boundary budget enforcement with configurable defaults (ADR-005)
- Composition-time validation reusing the load-path validator (not a separate implementation) — ADR-006
- Orchestrator LLM is a Model Profile, no hard-coded tiered fallback (ADR-011)
- Plexus integration phased: tool-first in Phase 1, context injection deferred (ADR-009). **Phase 2 hook point is structurally reserved** — ARCHITECT's session-start flow design must include a pre-orchestration stage where injection can be inserted without modifying the ReAct loop or tool surface. Phase 1 leaves the stage empty; Phase 2 populates it (added in response to Grounding Reframe #1)
- Conformance debt remediation as architectural tidying (refactor scenarios 1-3 in scenarios.md) before `compose_ensemble` is built

## Post-gate revisions (in response to susceptibility snapshot Grounding Reframes)

**Reframe 1 (FI-2 — ADR-009 sequencing provenance).** Applied. ADR-009's Decision section now commits to a Phase 2 hook point in the ARCHITECT-phase session-start flow and names a technical rationale for Phase 1 sufficiency (linked to orchestrator profile capability via ADR-011 and OQ #1).

**Reframe 2 (FF-2 — pure tool-user session coverage).** Applied. scenarios.md's Autonomy and Promotion feature now includes a scenario "Pure tool-user session at default Autonomy Level experiences silent composition" — exercising the surprise path explicitly as a refutable behavior so ARCHITECT's serving-layer configuration surface can decide whether to tighten defaults for non-operator deployments.

**Fourth argument-audit round:** clean, no new issues. Advance recommendation: clear to ARCHITECT.
