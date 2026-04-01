# Research Log: Agentic Serving for llm-orc

## Question 1: What OpenAI-compatible API surface is needed?

**Method:** Web search + codebase analysis

**Findings:**

The minimum viable endpoint set is **POST `/v1/chat/completions`** + **GET `/v1/models`**. Every agentic coding tool (OpenCode, Roo Code, Cursor, Continue.dev, Aider, Cline) converges on this pair.

For agentic use, the chat completions endpoint must support:

- **Request:** `tools` array (function definitions with JSON Schema parameters), `tool_choice` (`"auto"` | `"none"` | `"required"` | specific function object), `stream: true`, `messages` with roles `system`, `user`, `assistant`, and `tool` (the last carrying `tool_call_id` for correlating results back)
- **Non-streaming response:** `choices[].message.tool_calls[]` with `id`, `type: "function"`, `function.name`, `function.arguments` (JSON string). `finish_reason` must be `"tool_calls"` when the model requests tool execution.
- **Streaming (SSE):** `Content-Type: text/event-stream`, `data: {json}\n\n` per chunk, `data: [DONE]` to terminate. Tool calls stream via `delta.tool_calls[]` where each chunk has an `index` field; the first chunk carries `id`, `type`, `function.name`; subsequent chunks carry `function.arguments` fragments to be concatenated.

The `/v1/models` endpoint returns `{"object": "list", "data": [{"id": "model-name", "object": "model", "created": timestamp, "owned_by": "org"}]}`.

**Current llm-orc gap:** The existing FastAPI server (`web/server.py`) serves a management UI — ensemble CRUD, profiles, scripts, artifacts. No OpenAI-compatible endpoints exist. The `ModelInterface` base class has only `generate_response(message, role_prompt) -> str` — no tool use, no streaming, no multi-turn conversation management. The `OpenAICompatibleModel` is a *client* that calls external OpenAI endpoints, not a *server* that exposes them.

**Tool-specific notes:**
- OpenCode requires both tool calling and streaming (uses Vercel AI SDK's `streamText()`). Config via `opencode.json` with `@ai-sdk/openai-compatible` package, `baseURL` ending in `/v1`.
- Roo Code has **no XML fallback** — native tool calling is the only path.
- Aider is most forgiving (has XML fallback for models without tool calling).
- Known bug in OpenCode: custom provider options may not be properly passed (GitHub issue #5674).

**Implications:** Building an OpenAI-compatible serving layer is a well-defined engineering task. The spec is stable, the client expectations are documented, and reference implementations exist (Ollama, vLLM, LiteLLM). The interesting question is not "what API shape?" but "what happens behind the endpoint?" — which is Q2.

## Question 2: How does a declarative DAG engine become an agentic loop?

**Method:** Web search + framework analysis + codebase analysis

**Findings:**

### The canonical agentic loop

The standard pattern (ReAct) is: observe → think → act → observe, with `stop_reason` / `finish_reason` as the sole loop control signal. When the model returns `tool_use` / `tool_calls`, the loop continues. When it returns `end_turn` / `stop`, the loop terminates. Both Claude's agent SDK and OpenAI's Codex use this pattern. The model decides when it's done — no natural language parsing or iteration caps as primary control.

### Where does the loop live?

Three architectural options emerged:

**Option 1 — External (current model):** An outer agent (Claude Code, OpenCode, etc.) uses llm-orc's MCP tools to invoke ensembles. The agentic loop lives in the outer agent; llm-orc is a pure tool provider. This already works today via the conductor skill. The gap is result summarization for the outer agent's context management.

**Option 2 — Internal (new capability):** llm-orc adds its own agentic loop. A new execution mode where an LLM agent receives the user's request, has access to llm-orc tools (invoke ensemble, create ensemble, query plexus, etc.), and runs a ReAct loop until done. This is what serving via OpenAI-compatible endpoints would require — the server needs an agent behind the endpoint.

**Option 3 — Hybrid (CrewAI model):** The ensemble DAG remains declarative (the "Flow"), but individual agent slots can optionally run agentic loops (the "Crews"). LLM-driven routing edges are added alongside static `depends_on` edges.

### Industry convergence

The industry is converging on the hybrid model (option 3), but the pragmatic entry point is option 2 — an orchestrator agent behind the API that delegates to ensembles as tools:

```
OpenCode → /v1/chat/completions → llm-orc server
                                      ↓
                               Orchestrator Agent (LLM)
                               ↕ tool_calls (ReAct loop)
                         invoke_ensemble, create_ensemble,
                         query_plexus, list_ensembles, etc.
                               ↓
                    Ensemble Execution (existing DAG engine)
```

### Key patterns from the landscape

- **LLMCompiler** (ICML 2024): Separates planning from execution. A planner generates a full task DAG, an executor runs it with parallelism, and a "joiner" decides whether to replan or finish. Achieves up to 3.7x latency speedup over sequential ReAct. llm-orc's existing DAG execution already provides this benefit — wrapping it as a tool gives the outer agent LLMCompiler-like efficiency for free.

- **CrewAI Flows + Crews:** The Flow is the deterministic backbone; Crews are the agentic moments. Maps to: ensemble = Flow, orchestrator agent = the meta-Flow that dynamically invokes ensembles.

- **Google ADK `AgentTool`:** Wraps sub-agents as tools. The parent agent treats an entire workflow as a single function call. This is exactly what the orchestrator agent would do with ensembles.

- **Context rot** (Chroma research): LLM performance degrades sharply as context fills, even within technical token limits. Ensemble results need summarization before entering the orchestrator's context — returning full result dicts (as EnsembleAgentRunner does for inner composition) would saturate the outer loop.

- **Codex compaction:** When tokens exceed a threshold, the conversation is compressed into a smaller representation. The orchestrator agent would need similar compaction for long-running sessions.

**Implications:** The orchestrator agent is conceptually straightforward — it's a ReAct loop where the tools are llm-orc operations. The hard problems are context management (summarizing ensemble results), deciding which LLM powers the orchestrator (could be cloud or local), and defining the tool interface cleanly. The conductor skill has already solved many of these problems at the Claude Code layer — the question is how much of that intelligence transfers to a server-side agent.

## Question 3: Self-building ensembles — the agent as ensemble designer

**Method:** Web search + conductor skill analysis

**Findings:**

### Prior art for self-modifying agents

- **Voyager** (Minecraft, 2023): Creates executable skills at runtime, stores them in a library indexed by embedding similarity, retrieves them for future tasks. Achieves 3.3x more unique items and 15.3x faster milestone completion. Three components: automatic curriculum (what to learn next), skill library (what was learned), iterative prompting (self-verification + debugging).

- **LLM Agents Making Agent Tools** (ACL 2025, arXiv 2502.11705): Demonstrates that agents can autonomously convert GitHub repositories into LLM-compatible tools via closed-loop self-correction, effectively expanding the agent's action space at runtime.

- **Declarative pipeline DSL** (arxiv 2512.19769): Expresses agent workflows in under 50 lines of DSL vs 500+ lines of imperative code, with 60% development time reduction.

### Guardrails for self-modification

The field converges on a **control plane architecture** — hard-coded, deterministic logic gates that evaluate proposed actions regardless of LLM reasoning:

1. **Action allowlists:** Only create tools from predefined primitives. In llm-orc terms: the orchestrator can create ensembles using existing profiles and script templates, but cannot create arbitrary Python scripts.
2. **Depth/budget limits:** Bound self-modification by token spend, wall-clock time, and recursion depth. llm-orc already has configurable depth limits (default: 5).
3. **Human-in-the-loop gates:** Self-created tools require approval before entering the permanent library. Voyager's model — skills are verified before committed.
4. **Sandboxing:** New ensembles execute in isolated environments before being trusted.
5. **Versioned libraries:** Each created ensemble is versioned; rollback is always possible. llm-orc's tiered storage (local → global → library) already provides a natural promotion path.

### Translation from the conductor skill

The conductor skill already has a mature framework for this:

- **Ensemble lifecycle:** Design → Calibrate → Establish → Trust → Promote. New ensembles start in calibration (first 5 invocations always evaluated).
- **Separation of concerns:** The conductor routes and evaluates; the ensemble designer composes. In the server-side agent, these could be two modes or two tool sets.
- **"User always decides" invariant:** In the conductor, the user approves ensemble creation, routing decisions, and promotions. When the "user" is OpenCode, this invariant needs translation. Options:
  - The orchestrator agent acts autonomously within guardrails (budget limits, action allowlists)
  - The orchestrator surfaces decisions to the OpenCode user via its response text ("I'd like to create an ensemble for this task type. Proceed?")
  - A configuration flag sets the autonomy level per-session

### Natural guardrails in llm-orc

llm-orc already has several guardrails that translate directly:
- Ensemble YAML is inspectable and version-controllable
- Script agents run as subprocesses (process isolation)
- Depth limits prevent unbounded recursion
- Cycle detection catches circular references at load time
- Pydantic validation with `extra="forbid"` catches malformed configs
- Tiered storage means a self-created ensemble starts local and must be explicitly promoted

**Implications:** Self-building ensembles are architecturally feasible. The conductor's lifecycle model provides the framework. The key design decision is the autonomy boundary: how much can the orchestrator agent do without human approval? A reasonable starting point: the agent can invoke existing ensembles freely, can create new ensembles with existing profiles/templates (subject to a calibration period), but cannot create arbitrary scripts or promote ensembles without approval.

## Question 4: What role does Plexus play in agentic orchestration?

**Method:** Codebase analysis + ADR-020 analysis

**Findings:**

### What Plexus provides

Plexus is a knowledge graph engine with:
- **Graph storage:** Entities, relations, observations in SQLite
- **Provenance tracking:** Evidence trails showing where knowledge originated and how it changed
- **Context management:** Isolated workspaces (contexts) with file/directory sources
- **Clawmarks:** Annotated bookmarks with types (decision, question, change_needed, reference, alternative, dependency) and cross-references
- **MCP interface:** `set_context`, `annotate`, `evidence_trail`, context CRUD, clawmark CRUD

### ADR-020's vision

The conductor skill's ADR-020 already targets Plexus as the memory layer, identifying three categories of data:
- **Provenance:** Ensemble specs (Entities), calibration runs (Activities), routing decisions (Activities using specific configurations). Maps to W3C PROV.
- **Design knowledge:** DAG shapes, profile pairings, verification technique effectiveness, anti-patterns
- **Operational data:** Routing logs, evaluation records, task profiles, token usage

The ADR notes that flat files (JSONL/YAML) fail at the queries that matter: "What led to this poor result?" "What has worked for similar tasks?" "Which ensembles are ready for promotion?" These are graph traversal problems.

### Integration modes for the agentic server

Three ways Plexus could integrate with the orchestrator agent:

**As a tool the orchestrator calls:** The orchestrator has MCP tools like `query_plexus("what ensembles have worked for code review tasks?")` or `record_outcome(ensemble, score, context)`. This is the simplest integration — Plexus is just another tool in the ReAct loop.

**As context injected into prompts:** Before the orchestrator processes a request, the server queries Plexus for relevant context (prior routing decisions for similar tasks, ensemble quality history, known patterns) and injects it into the system prompt. This gives the orchestrator "memory" without burning tool-call turns.

**Both (the likely answer):** Pre-loaded context for baseline awareness, plus tool access for specific queries during reasoning. The pre-loaded context answers "what do I already know about this kind of task?" The tools answer "let me look up the specifics."

### Concrete use cases

1. **Cross-session memory:** User asks the orchestrator to review code. Plexus knows that the `code-review` ensemble scored "good" 8/10 times in prior sessions — route to it with high confidence.
2. **Failure analysis:** An ensemble scores "poor." Plexus records the failure with provenance (input, output, ensemble config). Future similar tasks can check: "has this ensemble failed on similar inputs before?"
3. **Design knowledge accumulation:** Each ensemble creation and calibration generates design knowledge. Plexus stores which DAG shapes work for which task types, enabling the designer agent to make better choices over time.
4. **Provenance for debugging:** When a user asks "why did you use that ensemble?", the orchestrator can trace through Plexus: routing decision → task profile → evaluation history → ensemble creation rationale.

### Integration architecture

```
Orchestrator Agent
  ├── Tool: invoke_ensemble(name, input)  → Ensemble Executor
  ├── Tool: create_ensemble(spec)         → Ensemble CRUD
  ├── Tool: query_knowledge(question)     → Plexus (graph query)
  ├── Tool: record_outcome(data)          → Plexus (write)
  └── Context: plexus_context_injection   → Plexus (pre-loaded)
         ↓
    Plexus Knowledge Graph
    ├── Routing decisions (provenance)
    ├── Ensemble quality history
    ├── Design patterns
    └── Task type profiles
```

**Implications:** Plexus integration is the differentiator. Without it, the orchestrator agent is just another agentic framework wrapping LLM calls. With it, the system accumulates knowledge across sessions, projects, and users — the orchestrator gets smarter over time. The integration should start simple (Plexus as a tool) and add context injection as the knowledge graph matures.
