# Domain Model: Agentic Serving

*Scoped to the agentic-serving feature. Extends the project-level domain model (`docs/domain-model.md`), whose vocabulary and invariants (1-14) are assumed.*

## Concepts (Nouns)

| Term | Definition | Product Origin | Avoid (synonyms) |
|------|-----------|----------------|-------------------|
| **Orchestrator Agent** | A ReAct-loop agent that sits behind the serving layer, receives requests from clients, and delegates to llm-orc operations as tool calls. Powered by a configurable LLM via model profile. | "orchestrator" (operator) | "controller", "router" (too narrow -- it also composes and reasons) |
| **Session** | A stateful conversation between a client and the orchestrator agent, bounded by budget constraints. Begins when a client connects; ends when the client disconnects or the budget is exhausted. | -- | "conversation" (ambiguous with LLM conversation), "connection" |
| **Serving Layer** | The OpenAI-compatible endpoint surface (`/v1/chat/completions`, `/v1/models`) added to the existing FastAPI server. Handles protocol translation: SSE streaming, tool-call formatting, model listing. | -- | "API", "server" (the server already exists; this is the new surface) |
| **Orchestrator Tool** | An operation the orchestrator agent can invoke via tool call: `invoke_ensemble`, `compose_ensemble`, `list_ensembles`, `query_knowledge`, `record_outcome`. A fixed set, not dynamically extensible. | -- | "action", "function" |
| **Routing Decision** | The orchestrator agent's choice of which ensemble handles a task. Recorded in the knowledge graph with provenance. Over time, shifts from reasoning (LLM figures it out) to retrieval (knowledge graph provides the answer). | "routing" (operator) | "dispatch" (already used for agent dispatch in project model) |
| **Dynamic Invocation** | Tool-mediated ensemble invocation by the orchestrator agent, outside the ensemble reference graph. Analogous to a user invoking an ensemble via the CLI. Not governed by Invariant 7. | -- | "dynamic reference" (misleading -- it is not a reference in the Invariant 7 sense) |
| **Composition** | The orchestrator agent creating a new ensemble at runtime from available primitives: model profiles, scripts, and other ensembles. The primitives are fixed; only their arrangement is dynamic. | "compose" (operator) | "generation", "creation" (too broad -- composition specifically means assembling from existing primitives) |
| **Primitive** | A building block the orchestrator uses to compose ensembles: a model profile, a script, or an existing ensemble. The set of primitives is determined by what exists in the library. | "primitives" (operator) | "component" (used generically elsewhere) |
| **Library** | The collection of available ensembles, model profiles, and scripts across all tiers (local, global, library). The menu the orchestrator draws from for routing and composition. | "library" (operator) | "registry", "catalog" |
| **Budget** | The resource constraints on a session: turn limit (maximum iterations of the ReAct loop) and token limit (cumulative token spend). Enforced at the control plane level, not the model level. Circuit breakers that prevent runaway sessions. | -- | "limit" (too vague), "quota" |
| **Result Summarization** | Condensing ensemble output before it enters the orchestrator agent's context. A correctness requirement, not an optimization -- full ensemble result dictionaries would cause context rot. May itself be implemented as an ensemble. | -- | "compression" (ambiguous with conversation compaction) |
| **Conversation Compaction** | Compressing prior turns in the orchestrator's conversation when context exceeds a threshold. Preserves tool-call/result correlations to maintain reasoning coherence. | -- | "pruning" (implies deletion; compaction preserves meaning) |
| **Context Injection** | Pre-loading knowledge from Plexus into the orchestrator agent's system prompt at session start. Provides baseline awareness without burning tool-call turns. | -- | "pre-fetch", "priming" |
| **Ingestion** | Loading source material (file content, not LLM-generated summaries) into the knowledge graph. The quality gate: what enters the graph is the actual source, not an interpretation of it. | -- | "import" |
| **Enrichment** | Background process that extracts signal from ingested sources. Runs asynchronously after ingestion. Declarative adapters on the llm-orc side and core enrichments on the Plexus side reinforce stronger signals in the graph. | "enrichment" (operator) | "analysis", "processing" |
| **Quality Signal** | A measurement of ensemble effectiveness attached to a routing decision or execution outcome. Distinguishes "this worked well" from "this was used often." Governs stabilization. | -- | "score" (too narrow), "rating" |
| **Stabilization** | The organic process by which a repeated, quality-gated pattern becomes a reliable library entry. Not curated -- emerges from accumulated quality signals over multiple executions. | "stabilize" (operator) | "promotion" (promotion is an explicit operator action; stabilization is organic) |
| **Bootstrapping** | Pre-populating the knowledge graph from existing context (library contents, execution artifacts) so the system starts warm instead of cold. The bootstrapping pipeline: artifacts → ingestion → enrichment → queryable graph. | "warm start" (operator) | "seeding" |
| **Autonomy Level** | A per-session configuration controlling how much the orchestrator can do without operator approval. Ranges from fully autonomous (within guardrails) to fully supervised. | -- | "permission level" |
| **Calibration** | The evaluation period for a newly composed ensemble. During calibration (first N invocations), results are always checked. The ensemble must demonstrate quality before earning trust. | -- | "trial", "probation" |

## Actions (Verbs)

| Action | Actor | Subject | Description |
|--------|-------|---------|-------------|
| **Route** | Orchestrator Agent | Task → Ensemble | Select an ensemble from the library to handle a task, informed by knowledge graph context and quality signals |
| **Compose** | Orchestrator Agent | Primitives → Ensemble | Create a new ensemble at runtime from available model profiles, scripts, and other ensembles |
| **Invoke (Dynamic)** | Orchestrator Agent | Ensemble | Execute an ensemble via tool call, outside the ensemble reference graph. Not governed by Invariant 7 |
| **Summarize** | Result Summarizer | Ensemble Output | Condense ensemble results before they enter the orchestrator's context window |
| **Compact** | Session | Conversation History | Compress prior turns when the orchestrator's context exceeds a threshold, preserving tool-call correlations |
| **Inject** | Serving Layer | Knowledge Graph → System Prompt | Query Plexus at session start and load relevant context into the orchestrator's system prompt |
| **Ingest** | Plexus (lib) | Source Material | Load file content into the knowledge graph. Driven by the client (push model) |
| **Enrich** | Plexus | Ingested Content | Extract signal from source material asynchronously, reinforcing strong patterns in the graph |
| **Query** | Orchestrator Agent | Knowledge Graph | Ask Plexus for knowledge during reasoning -- specific lookups when the orchestrator identifies a gap |
| **Record** | Orchestrator Agent | Routing Decision / Outcome | Store a routing decision, quality signal, or execution outcome in the knowledge graph with provenance |
| **Calibrate** | Orchestrator Agent | New Ensemble | Evaluate a newly composed ensemble over its first N invocations, checking results before the ensemble earns trust |
| **Stabilize** | Knowledge Graph | Pattern | Organically promote a repeated, quality-gated pattern to a reliable library entry -- not an explicit action but an emergent property |
| **Bootstrap** | Operator | Knowledge Graph | Pre-populate the graph from existing context (library contents, execution artifacts) via ingestion and enrichment |

## Relationships

- An **Orchestrator Agent** *runs within* exactly one **Session**
- A **Session** *has* one **Budget** (turn limit + token limit)
- An **Orchestrator Agent** *uses* **Orchestrator Tools** (fixed set)
- An **Orchestrator Agent** *routes* tasks to **Ensembles** via **Routing Decisions**
- An **Orchestrator Agent** *composes* new **Ensembles** from **Primitives** (profiles, scripts, ensembles)
- A **Routing Decision** *is recorded in* the **Knowledge Graph** with provenance
- A **Quality Signal** *is attached to* a **Routing Decision** or execution outcome
- **Result Summarization** *condenses* **Ensemble** output before it enters the **Session** context
- **Conversation Compaction** *compresses* prior turns within a **Session**
- **Context Injection** *loads* **Knowledge Graph** data into the **Orchestrator Agent's** system prompt
- **Ingestion** *loads* source material into the **Knowledge Graph** (push model, client-driven)
- **Enrichment** *extracts signal from* ingested content asynchronously
- **Stabilization** *emerges from* accumulated **Quality Signals** over multiple **Routing Decisions**
- **Dynamic Invocation** *bypasses* the **Ensemble Reference Graph** (project-level Invariant 7 does not govern it)
- A composed **Ensemble** that includes ensemble-to-ensemble references *must satisfy* project-level Invariant 7 at load time
- The **Orchestrator Agent** *is parameterized by* a **Model Profile** (the system orchestrates itself)
- **Bootstrapping** *pre-populates* the **Knowledge Graph** via **Ingestion** + **Enrichment**
- The **Library** *is the source of* **Primitives** the orchestrator draws from
- **Calibration** *gates* a composed **Ensemble** before it earns trust

## Invariants

*Project-level invariants 1-14 remain in force. The following are specific to agentic serving:*

**AS-1. Dynamic invocations are outside the ensemble reference graph.** The orchestrator agent's tool-mediated ensemble invocations are analogous to CLI invocations -- they are not ensemble references in the Invariant 7 sense. Invariant 7 governs static YAML composition, not the orchestrator's runtime tool calls.

**AS-2. Composed ensembles must be validated before loading.** When the orchestrator composes a new ensemble that includes ensemble-to-ensemble references, those references must be validated against the existing ensemble reference graph before the ensemble is loaded. This satisfies project-level Invariant 7 for the composed ensemble's internal structure.

**AS-3. Budget enforcement is a control plane concern.** Turn limits and token limits are enforced at the session level, checked at each iteration of the ReAct loop, regardless of what the orchestrator LLM decides to do. These are harness-level circuit breakers, not model-level parameters.

**AS-4. Ingestion boundary is source material.** *(Applies when Plexus is active.)* The knowledge graph ingests file content (source material), not LLM-generated summaries or interpretations. Quality emerges from the enrichment pipeline, not from upstream curation of what gets ingested.

**AS-5. Quality signals govern stabilization, not frequency.** *(Applies when Plexus is active.)* A pattern does not stabilize into a reliable library entry by being used often. Stabilization requires accumulated quality signals demonstrating effectiveness. Frequency without quality is noise.

**AS-6. The orchestrator composes from existing primitives only.** The orchestrator agent can compose new ensembles from model profiles, scripts, and other ensembles that exist in the library. It cannot author arbitrary scripts or create new model profiles. The composable surface is fixed; only the arrangement is dynamic.

**AS-7. Result summarization is a correctness requirement.** Full ensemble result dictionaries must be summarized before entering the orchestrator's context. This is not an optimization -- unsummarized results cause context rot that degrades orchestrator quality over the course of a session.

**AS-8. Plexus is optional.** The orchestrator agent, serving layer, budget enforcement, result summarization, conversation compaction, and ensemble composition all function without Plexus. When Plexus is absent, the orchestrator operates statelessly -- routing by reasoning rather than retrieval, with no cross-session memory, no stabilization, and no bootstrapping. Plexus transforms a stateless orchestrator into a learning one, but is not a prerequisite for agentic serving.

## Open Questions

1. **Knowledge-compensated model selection.** Can well-orchestrated smaller models + a populated knowledge graph compete with frontier models on routing quality while winning on cost? This is the most commercially relevant question and is testable: run identical tasks with and without Plexus context across model tiers. *(Source: reflections, product discovery tension #1)*

2. **What form does visibility take?** The operator wants to see what the orchestrator is doing -- for debugging, understanding, and tinkering. Structured events? Logs? Interactive dashboard? Information surfaced in the tool's output? The answer shapes the serving layer's event model. *(Source: product discovery tension #5)*

3. **Internal ReAct loop vs. external MCP model.** The essay chose the internal orchestrator as the pragmatic entry point. The conductor skill already implements the external model. Is the gap genuinely the execution model, or is it summarization quality? If fixing summarization is cheaper than building an internal agent loop, the architecture simplifies dramatically. *(Source: product discovery assumption inversion #5)*

4. **Bootstrapping quality.** How well does the knowledge graph bootstrap from existing context vs. learning through use? Structural bootstrapping (what exists) is straightforward. Behavioral bootstrapping (what works) requires execution history. Pattern bootstrapping (why things work) requires the conductor's accumulated knowledge -- which may not be ingestible. *(Source: product discovery tension #6)*

5. **Organic stabilization mechanism.** How does the knowledge graph distinguish a genuinely good pattern from one that was merely used often? What constitutes a quality signal in this context -- execution success, user satisfaction, task completion rate, something else? *(Source: product discovery tension #7)*

6. **Orchestration depth.** Is orchestration invariably a broad AND deep task requiring frontier models, or does accumulated knowledge eventually bring it within reach of consumer-hardware models for routine decisions? The answer determines the long-term cost model. *(Source: product discovery feed-forward signal #13)*

7. **Enrichment pipeline maturity.** The model currently asks enrichment to carry two loads: quality gate for ingestion (AS-4) and signal source for stabilization (AS-5). Plexus's enrichment model is under active investigation. If enrichment is weak, both invariants are hollow -- the graph fills with undifferentiated content and stabilization degrades to frequency-counting. This is not a blocker (AS-8 means the system works without Plexus) but determines whether the learning-system value proposition is real. *(Source: MODEL reflection gate)*

## Amendment Log

| # | Date | Invariant | Change | Propagation |
|---|------|-----------|--------|-------------|
| -- | -- | -- | Initial scoped model, no prior version | -- |
| 1 | 2026-04-17 | AS-8 (Plexus is optional) | Added. Plexus transforms stateless orchestrator into learning one, but is not prerequisite. AS-4 and AS-5 scoped to "when Plexus is active." | No prior ADR contradictions. Essay's four-layer architecture (Layer 4 = Knowledge Graph) should note optionality. |
