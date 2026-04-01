# Product Discovery: Agentic Serving

*2026-04-01*

## Stakeholder Map

### Direct Stakeholders

**Tool user.** A developer using an agentic coding tool (OpenCode, Roo Code, Aider, Cursor, Cline) configured to point at an llm-orc endpoint. Interacts with the system through their tool's interface. Cares about response quality, speed, and cost. Does not need to know about ensembles, DAGs, or Plexus -- but may choose to look under the hood if they are also an ensemble author.

**Ensemble author / operator.** Creates and maintains ensemble YAML configs, model profiles, and scripts. Runs the llm-orc server. Wants the orchestrator to leverage the library of capabilities they have built. Cares about visibility -- not just for debugging, but as part of the experience of tinkering with and improving the system. May also be a tool user, collapsing the "black box" and "visibility" perspectives into one person.

**Orchestrator LLM.** The agent behind the endpoint. Not a person, but an actor with needs: access to the full palette of composition primitives (ensembles, profiles, scripts), knowledge of what has worked before, budget constraints, and a way to surface its reasoning for inspection. Its effectiveness directly shapes the tool user's experience.

### Indirect Stakeholders

**Plexus (knowledge graph).** The system's memory layer. Its population over time affects cost, quality, and routing effectiveness for all stakeholders. Its query surface is actively evolving, which constrains what the orchestrator can lean on in early iterations.

## Jobs and Mental Models

### Tool User

**Jobs:**
- "I want to use my coding tool the same way I would with Claude or GPT, but backed by my own orchestration" -- autonomy from a single provider
- "I want complex tasks handled well without knowing what happens behind the endpoint" -- the endpoint is a model
- "I want it fast enough that it does not break my flow" -- speed matters
- "I want it cheap enough to leave running all day" -- cost matters more than peak capability for sustained use

**Mental model:** The endpoint is a model. The tool user does not distinguish between "a model answered" and "an ensemble of models answered." Quality, speed, and cost are the only axes they evaluate on.

### Ensemble Author / Operator

**Jobs:**
- "I want my ensembles to get used -- the orchestrator should route to the right one" -- the work I have built has value
- "I want to see what the orchestrator is doing -- which ensembles it picks, what it creates, why it fails" -- visibility for debugging and understanding
- "I want to tinker -- adjust behavior, experiment with configurations, turn the tool on itself" -- visibility as play, not just audit
- "I want the system to get better over time without me hand-tuning everything" -- organic stabilization of patterns through use
- "I want to control cost and resource boundaries" -- predictable spend
- "I want to bootstrap the knowledge graph from what already exists" -- time-to-value from session one

**Mental model:** The system is a library of capabilities (ensembles, profiles, scripts) that the orchestrator draws from. The orchestrator is a smart dispatcher that learns through use. The knowledge graph is what makes it learn. Over time, repeated patterns stabilize organically -- the operator does not have to curate them explicitly, but should be able to see what has stabilized and intervene when needed.

### Orchestrator LLM

**Jobs:**
- "Match the task to the right ensemble from the library -- or compose one from available primitives if nothing fits" -- effective task resolution
- "Know what has worked before for similar tasks" -- retrieval over reasoning, the cost/quality lever
- "Compose from the full palette: other ensembles, model profiles, and scripts" -- not restricted to a subset of primitives
- "Stay within budget -- turns, tokens, wall-clock time" -- resource discipline
- "Surface what I am doing and why in a way that is inspectable" -- the visibility contract with the operator

**Mental model:** The orchestrator sees a toolkit of primitives and a knowledge graph. It does not distinguish between pre-built and self-composed ensembles at execution time -- both are just ensembles. The distinction is in the lifecycle (validation, calibration, trust, promotion). A well-populated knowledge graph converts routing from reasoning to retrieval, enabling cheaper model tiers to make effective decisions.

## Value Tensions

1. **Quality vs. cost vs. speed.** The classic triangle, with a twist: the knowledge graph shifts the frontier over time. Early sessions may be expensive (reasoning-heavy, frontier models); later sessions are cheaper (retrieval-heavy, smaller models). The tension is not static -- it evolves as the graph matures. If bootstrapping works, the expensive early phase may shrink or disappear.

2. **Autonomy vs. visibility.** The tool user wants a black box that works. The operator wants to see inside it. When these are the same person (which may be the common case), the tension becomes: how much of the orchestrator's reasoning is surfaced, where, and when? Too little and the system is opaque; too much and it is noisy.

3. **Find vs. build.** The orchestrator can invoke existing ensembles or compose new ones. Finding is cheaper and safer; building is more capable but riskier. Over time, building should feed back into finding -- a composed ensemble that works well should stabilize into a library entry organically.

4. **Static composition vs. dynamic composition.** Invariant 7 says ensemble references are static string literals resolved at load time. The orchestrator needs to compose ensembles from all three primitives at runtime, including other ensembles. The validation path (checking references against the existing ensemble reference graph before loading) is the technical resolution, but the conceptual tension remains: how much of the ensemble graph is knowable at load time vs. runtime?

5. **Visibility: what form?** Flagged as important but unresolved. Possibilities include structured events, logs, a dashboard, or information surfaced in the tool user's output. The tension is between "enough to debug and trust" and "not so much overhead that it slows the system or overwhelms the operator." Tinkering as a use case suggests visibility should be interactive, not just retrospective.

6. **Cold start vs. warm start.** Does the knowledge graph have to learn through use, or can it be bootstrapped from existing context? The bootstrapping pipeline has a shape: llm-orc artifacts → Plexus ingestion (background) → enrichment fills the knowledge graph → queries and analysis results feed back. Quality gate concern at the ingestion boundary -- "garbage in, garbage out." The quality of bootstrapping determines how quickly the system delivers on its value proposition.

7. **Organic stabilization vs. explicit curation.** Repeated patterns should surface and stabilize on their own rather than requiring the operator to hand-tag them. But how does the system distinguish a genuinely good pattern from one that was merely used often? The knowledge graph needs to track quality signals, not just frequency.

## Assumption Inversions

| Assumption | Inverted Form | Implications |
|------------|--------------|-------------|
| The orchestrator needs a frontier model for complex routing | What if a small model + populated knowledge graph routes just as well? | The cost model inverts -- the knowledge graph is the primary cost lever, not a nice-to-have. Testable: run identical tasks with and without Plexus context across model tiers. |
| The knowledge graph starts empty and learns through use | What if it can be bootstrapped from existing context (ensemble library, artifacts, conductor patterns)? | The cold-start problem may not exist. The economic value proposition applies from day one. Bootstrapping quality becomes a first-order design concern. |
| The tool user does not know or care about ensembles | What if some tool users are ensemble authors, using their own orchestration from their own coding tool? | The "black box" mental model breaks down. These users want both the seamless experience and the ability to peek inside. The autonomy/visibility tension collapses into one stakeholder. |
| The orchestrator should try to find an existing ensemble before building a new one | What if building-first produces better results because composed-to-fit ensembles outperform general-purpose library entries? | The library becomes a set of primitives for composition rather than a set of ready-made solutions. The "find vs. build" priority inverts. |
| Agentic serving requires an internal ReAct loop | What if the external model (MCP tool provider) with better result summarization is sufficient? | Dramatically simpler architecture. The conductor skill already works this way. The gap may be summarization quality, not a missing execution mode. |
| Visibility is primarily for debugging and auditing | What if visibility is part of the value proposition -- tinkering, experimentation, turning the tool on itself? | Visibility features are not overhead -- they are product. The system's inspectability is a differentiator, not a cost center. |

## Product Vocabulary

| User Term | Stakeholder | Context | Notes |
|-----------|-------------|---------|-------|
| ensemble | author/operator | The unit of orchestration -- a named YAML config of agents | Domain model term, also user-facing |
| profile | author/operator | A named model configuration (model + provider + defaults) | Short for "model profile" |
| script | author/operator | An executable that an agent runs as a subprocess | One of the three composition primitives |
| library | author/operator | The collection of available ensembles, profiles, scripts | The menu the orchestrator draws from |
| artifact | operator | A persisted record of an execution's results | What Plexus could ingest for bootstrapping |
| context | operator/Plexus | A Plexus workspace scoped to a set of sources | The boundary for ingestion and querying |
| routing | operator | The orchestrator choosing which ensemble handles a task | The decision the knowledge graph improves over time |
| tinkering | author/operator | Adjusting the system's behavior through visibility and experimentation | Play and iteration, not just audit |
| warm start / cold start | operator | Whether the knowledge graph has useful data from session one | Determines time-to-value |
| stabilize | operator | When a repeated pattern becomes a reliable library entry | Organic, not curated -- but quality-gated |
| enrichment | operator/Plexus | Background process that fills the knowledge graph from ingested sources | Part of the bootstrapping pipeline |
