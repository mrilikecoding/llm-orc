# Reflections: Agentic Serving Architecture
*2026-04-01*

## Complementary, not competing

The finding that DAG execution and ReAct loops are complementary rather than competing was surprising. The initial assumption was that making llm-orc "agentic" would require rethinking the declarative model. Instead, the declarative model becomes a tool — a high-efficiency, parallel execution primitive that the agentic loop invokes. The DAG engine doesn't need to become agentic; it needs to be *available to* an agent. This reframing made the architecture feel tractable rather than aspirational.

## Knowledge graph as capability reducer

The most significant insight from the epistemic gate was a reframing of Plexus's role. The essay positions Plexus as a differentiator for cross-session memory and provenance debugging. The reflection surfaced a stronger claim: externalizing knowledge into a graph reduces the reasoning load on the orchestrator LLM, which directly lowers the model-tier required, which translates to cost savings.

If the orchestrator can look up "this ensemble has worked well for code review tasks" rather than reasoning about task-ensemble fit from scratch, the routing decision becomes a retrieval task — fundamentally easier than a reasoning task. Easier tasks run on smaller, cheaper models.

This reframes the value proposition: the system doesn't just get *smarter* over time (better routing decisions) — it gets *cheaper* over time (accumulated knowledge means smaller models can make equivalent-quality decisions). An all-day agentic session against a 7B model with frontier models called only at capability boundaries could be dramatically cheaper than continuous frontier usage.

## Open question: knowledge-compensated model selection

Can well-orchestrated smaller models + a populated knowledge graph compete with frontier models on quality while winning on cost? This is not a question the essay answers, and it may be the most commercially relevant question for the project. It's testable: run the same task routing with and without Plexus context, compare decision quality across model tiers.

This question should appear in the domain model as an open question for future research/experimentation.

## The cost framing

The user's framing of cost (comparing all-day local model usage against Claude Max subscription pricing) suggests the value proposition may be more about economics than capability. If llm-orc + Plexus can deliver 80% of a frontier model's agentic quality at 10% of the cost, that's a compelling pitch even if it never matches frontier quality on hard tasks. The question becomes: how much of agentic work is "hard" vs. "routine with good memory"?

## Plexus as intra-session multi-agent substrate (DECIDE gate, 2026-04-19)

Surfaced at the DECIDE reflection-time gate: Plexus's value is not only cross-session memory. Its query surface takes the shape of *lens grammars* — consumer-registered specifications that govern how content is encoded into the graph during enrichment, and how it is queried back. AS-4 is preserved under this framing: the lens is a schema applied during enrichment, not an LLM-summary ingestion path.

The sharper observation: rather than llm-orc registering a *single* adapter/lens against Plexus, individual ensembles or agents within the orchestration could each register their *own* lenses over the same Plexus context. The orchestrator's access to the graph becomes polyglot — each subagent writes into the shared enrichment through its own grammar and queries the graph through its own lens. This makes Plexus a medium for intra-session cross-talk between subagents, not only a cross-session memory layer.

If this pattern holds, the essay's framing of Plexus-as-memory undersells what Plexus enables. The value appears at the first session, not only after accumulation, because the orchestrator and its subagents can share structured intelligence through the graph within a single session. The OQ #7 dependency weakens: enrichment pipeline maturity still matters, but "wait for the graph to populate" is no longer the dominant constraint.

An open hypothesis travels with this framing: does KG-mediated communication between subagents (via lens grammars) avoid the bias reinforcement documented in direct agent-to-agent communication? A counter: lens grammars could *amplify* bias if a lens privileges some signals over others — bias need not live in the message; it can live in the lens.

The lens-grammar design is in-progress on the Plexus side; this reflection is forward-signal, not a change driver for current ADRs. Captured here and as domain model OQ #8. Folds back into a later cycle when Plexus's lens specification is more settled.
