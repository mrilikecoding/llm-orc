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
