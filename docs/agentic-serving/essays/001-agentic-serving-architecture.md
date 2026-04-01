# Agentic Serving: From Declarative DAG Engine to Autonomous Orchestrator
*2026-03-20*

## Abstract

This essay investigates whether llm-orc, a declarative DAG-based LLM orchestration engine, can serve as the backend for agentic coding tools via OpenAI-compatible endpoints. Six questions were explored through web search, framework analysis, and codebase analysis: (1) the API surface required for tool-calling agents, (2) how a DAG execution engine maps to an agentic ReAct loop, (3) whether the orchestrator can build its own ensembles at runtime, (4) how the Plexus knowledge graph integrates as a memory layer, and (5-6) what architectural patterns transfer from OpenHands and claw-code, two open-source agent harnesses that have independently converged on similar designs. The central finding is that the architecture is feasible and well-validated by the landscape, but introduces a structural tension with llm-orc's current invariant that ensemble references are static string literals resolved at load time. A second finding emerged from reflection: a populated knowledge graph does not merely provide memory — it lowers the capability threshold for the orchestrator LLM, converting reasoning tasks into retrieval tasks that smaller, cheaper models can handle. The orchestrator agent pattern — a ReAct loop with ensemble invocation as its primary tool — emerges as the pragmatic entry point, with Plexus-backed memory as the differentiator that no comparable system currently provides.

## The Problem: Bridging Two Worlds

llm-orc executes ensembles: declarative YAML configurations of agents with dependency relationships, run in topological phase order with parallel execution within phases. This model is powerful for structured multi-model workflows but inert — ensembles do not reason about themselves, select themselves, or compose themselves. They execute when told to, in the shape they were given.

Agentic coding tools (OpenCode, Roo Code, Cursor, Aider, Cline) expect something different: a chat endpoint that reasons, decides, and acts. They send messages with tool definitions and expect the model to call those tools when needed, continuing in a loop until the task is complete. The canonical pattern is ReAct — observe, think, act, observe — with `finish_reason` as the sole loop termination signal.

The question is not whether these two models are compatible but how they compose. A DAG engine and a ReAct loop are complementary: the engine provides structured, parallel, multi-model execution; the loop provides dynamic reasoning about when and how to invoke that execution. OpenHands, the most architecturally mature open-source agent platform (65K+ stars, $18.8M Series A), draws a useful distinction: *agentic AI* is the LLM-driven perceive-reason-act-observe loop; *AI orchestration* is the layer managing lifecycle, resource boundaries, and sandboxing. Production systems require both. The architecture this essay describes places a ReAct loop *in front of* the DAG engine, using ensembles as the loop's primary tools — agentic AI backed by AI orchestration.

## The API Surface

The minimum viable endpoint set is POST `/v1/chat/completions` and GET `/v1/models`. Every agentic coding tool in the current landscape converges on this pair, which is sufficient for both streaming and non-streaming interactions with full tool-calling support.

For agentic use, the chat completions endpoint must handle:

- A `tools` array carrying function definitions with JSON Schema parameters
- `tool_choice` controlling whether and how the model uses tools (`"auto"`, `"none"`, `"required"`, or a specific function)
- Streaming via Server-Sent Events, where tool calls arrive as `delta.tool_calls[]` fragments indexed for reassembly
- Messages with roles `system`, `user`, `assistant`, and `tool`, the last carrying `tool_call_id` for correlating results

This surface is stable and well-documented. Reference implementations exist in Ollama, vLLM, and LiteLLM. The engineering task of implementing the endpoint is bounded; the design challenge lies behind it.

llm-orc's existing FastAPI server handles ensemble management — CRUD operations on ensembles, profiles, scripts, and artifacts. The `ModelInterface` base class exposes only `generate_response(message, role_prompt) -> str`, with no tool use, no streaming, and no multi-turn conversation. The `OpenAICompatibleModel` is a *client* consuming external APIs, not a server exposing them. Building the serving layer means adding new endpoints alongside the existing management API, not replacing it.

## The Orchestrator Agent

Three architectural options exist for where the agentic loop lives:

**External.** The loop stays in the outer tool (Claude Code, OpenCode). llm-orc remains a pure tool provider, invoked via MCP. This already works via the conductor skill, which routes tasks to ensembles through Claude Code's own reasoning loop. The gap is result summarization — full ensemble output dictionaries are too large for the outer agent's context.

**Internal.** llm-orc runs its own ReAct loop. An orchestrator agent receives requests via the OpenAI-compatible endpoint, reasons about them, and calls llm-orc operations as tools — `invoke_ensemble`, `create_ensemble`, `list_ensembles`, `query_plexus`. This is what agentic serving requires: the server needs an agent behind the endpoint.

**Hybrid.** The ensemble DAG remains declarative, but individual agent slots can run agentic sub-loops. LLM-driven routing edges supplement static `depends_on` edges. This is where the industry is converging (CrewAI's Flows + Crews model), but it requires deeper changes to the ensemble execution engine.

The pragmatic entry point is the internal model — it reuses the existing ensemble engine unchanged, whereas the hybrid model would require adding LLM-driven routing edges to the execution engine itself. An orchestrator agent sits behind `/v1/chat/completions` and delegates to ensembles:

```
Client → /v1/chat/completions → llm-orc server
                                     ↓
                              Orchestrator Agent (LLM)
                              ↕ tool_calls (ReAct loop)
                        invoke_ensemble, create_ensemble,
                        query_plexus, list_ensembles, etc.
                              ↓
                   Ensemble Execution (existing DAG engine)
```

This pattern maps directly to Google ADK's `AgentTool` concept, where a parent agent treats an entire sub-workflow as a single function call. OpenHands implements a similar pattern via `AgentDelegateAction` — delegation is a typed action in the event stream, not a framework-level graph construct. It also benefits from the LLMCompiler insight (ICML 2024): that architecture separates planning from execution using a planner, a parallel executor, and a joiner that decides whether to replan — achieving up to 3.7x latency speedup over sequential ReAct. llm-orc's DAG engine provides the executor-phase benefit: within-ensemble parallelism that reduces execution latency compared to sequential tool calls. The planner and joiner components would require additional implementation to capture the full LLMCompiler architecture.

OpenHands' implementation offers an additional architectural lesson: its agent loop is built on an event-sourced `EventStream` — an append-only, immutable log of typed events. State is derived from the log, not mutated in place. This enables deterministic replay of any agent session, a property valuable for debugging and auditability. Whether the llm-orc orchestrator adopts event sourcing or a simpler direct-loop implementation is a design decision, but the replay and auditability benefits are worth noting — particularly since ensemble execution artifacts already provide a structured record of the DAG layer's work.

### Context Management

The hardest operational problem is context rot. Research from Chroma demonstrates that LLM performance degrades as context fills, even within technical token limits. An orchestrator agent that accumulates full ensemble results across multiple invocations will degrade in quality over the course of a session.

Three mitigation strategies apply:

1. **Result summarization.** Ensemble outputs are summarized before entering the orchestrator's context. The summarization itself can be an ensemble — a dedicated summarizer agent that distills results into what the orchestrator needs to know.
2. **Conversation compaction.** When the orchestrator's context exceeds a threshold, prior turns are compressed into a shorter representation. The key is preserving tool call/result correlations so the agent's reasoning chain remains coherent. Claw-code (a clean-room rewrite of Claude Code's architecture) treats compaction as a first-class concern: its `TranscriptStore` triggers compaction at a configurable turn threshold and maintains a synchronized transcript log. This is not an optimization — it is a correctness requirement for long-running sessions.
3. **Plexus offloading.** Rather than keeping everything in context, the orchestrator records key decisions and outcomes in Plexus and retrieves them on demand. This converts context memory into external memory, trading token cost for tool-call latency.

The orchestrator also needs loop-level budget enforcement. Claw-code implements this with `max_turns` (default: 8) and `max_budget_tokens` (default: 2000) as control plane constraints checked at each iteration. These are not model-level limits — they are harness-level circuit breakers that prevent runaway sessions regardless of what the LLM decides to do.

### Which LLM Powers the Orchestrator?

The orchestrator agent needs an LLM to reason with. This is itself a routing decision:

- A cloud model (Claude, GPT-4) provides the strongest reasoning for complex task decomposition and tool selection, at the cost of latency and API spend.
- A local model (via Ollama or vLLM) provides fast, free inference suitable for simpler routing decisions.
- A tiered approach — local model for initial triage, cloud model when the task exceeds local capability — mirrors the conductor skill's existing routing logic.

The LLM powering the orchestrator should be configurable via a model profile, reusing llm-orc's existing profile infrastructure. This means the orchestrator is itself parameterized by the same system it orchestrates. OpenHands validates this pattern: it uses LiteLLM for provider abstraction across 100+ providers, including local models via Ollama, vLLM, and SGLang. Both OpenHands and claw-code treat provider-agnostic design as a first-class requirement, not an afterthought.

The cost implications are significant. If the orchestrator can lean on Plexus for accumulated routing knowledge — which ensembles work for which task types, what has failed before, what patterns are effective — then many routing decisions become retrieval tasks rather than reasoning tasks. Retrieval tasks require less model capability. A well-populated knowledge graph could enable a smaller, cheaper model (MiniMax, GLM, Qwen, or a local 7B model) to make routing decisions that would otherwise require a frontier model reasoning from scratch. The system doesn't just get smarter over time — it gets cheaper, because accumulated knowledge reduces the capability threshold for the orchestrator LLM. An all-day agentic session against local or low-cost models, with frontier models called only at capability boundaries, could be dramatically less expensive than continuous frontier model usage.

## Self-Building Ensembles

The conductor skill already implements a mature framework for agent-driven ensemble creation. Its lifecycle model — Design, Calibrate, Establish, Trust, Promote — provides the template for a server-side equivalent.

### Prior Art

Voyager (2023) demonstrated that agents can create executable skills at runtime and store them in a retrievable library, achieving 3.3x more unique items and 15.3x faster milestone completion in its domain. Three components drove this: an automatic curriculum (what to learn next), a skill library (what was learned), and iterative prompting (self-verification and debugging).

LLM Agents Making Agent Tools (ACL 2025) extended this direction, showing that agents can autonomously generate and integrate new tools from existing code repositories — effectively expanding their own action space at runtime. The conceptual framing — tool creation as a valid action that expands the agent's capabilities — applies at a general level, though the specific mechanism (converting GitHub repositories into LLM tools) differs from YAML ensemble composition.

The key lesson from both: self-modification works when constrained by a control plane — hard-coded, deterministic gates that evaluate proposed actions regardless of LLM reasoning.

### Guardrails

Five guardrail categories emerge from the literature and the conductor skill:

1. **Action allowlists.** The orchestrator can create ensembles from existing profiles and script templates but cannot author arbitrary scripts. The composable primitives are fixed; only their arrangement is dynamic.
2. **Budget limits.** Token spend, wall-clock time, and recursion depth are bounded. llm-orc already enforces depth limits (default: 5).
3. **Human-in-the-loop gates.** Self-created ensembles require approval before entering the permanent library. The conductor skill's "user always decides" invariant translates to: the orchestrator can create and invoke local ensembles freely, but promoting them to global or library tier requires explicit approval.
4. **Sandboxing.** New ensembles execute in an evaluation mode during their first N invocations. Results are always checked before the ensemble earns trust.
5. **Versioned storage.** llm-orc's tiered storage (local, global, library) provides a natural promotion path. Self-created ensembles start local and must be explicitly promoted.

### The Autonomy Boundary

When the "user" is OpenCode rather than a human at a terminal, the conductor's "user always decides" invariant needs translation. Three options:

- The orchestrator acts autonomously within guardrails — budget limits and action allowlists constrain behavior without requiring approval for each action.
- The orchestrator surfaces decisions to the end user via response text — "I'd like to create an ensemble for this task type. Proceed?" — and waits for confirmation.
- A configuration flag sets the autonomy level per-session, ranging from fully autonomous (within guardrails) to fully supervised.

A reasonable starting position: the orchestrator can invoke existing ensembles freely, can create new ensembles with existing profiles and templates (subject to calibration), but cannot create scripts or promote ensembles without approval.

## Tension: Static References vs. Dynamic Invocation

The existing domain model states: *"Ensemble agent references are string literals resolved at load time. No template expressions or dynamic resolution"* (Invariant 7). An orchestrator agent that creates and invokes ensembles at runtime directly contradicts this invariant.

However, the tension is narrower than it first appears. Invariant 7 governs the *ensemble reference graph* — the static composition of ensembles referencing other ensembles via `EnsembleAgentConfig`. The orchestrator agent does not compose ensembles this way. It sits *outside* the ensemble execution model, invoking ensembles as tools through the execution API rather than through static YAML references. The relationship is analogous to a user invoking ensembles via the CLI — the CLI is not subject to Invariant 7, and neither would the orchestrator be.

That said, if the orchestrator creates new ensembles that themselves reference other ensembles, those references *are* subject to Invariant 7. The invariant still holds for the ensemble reference graph; it simply does not govern the orchestrator's tool-call-level invocations. A practical constraint follows: orchestrator-created ensembles should be restricted to pure profile-and-script compositions (no ensemble-to-ensemble references), or alternatively, the creation path must include an explicit validation step that checks references against the existing ensemble reference graph before loading. Without one of these guards, the orchestrator could produce ensembles that fail Invariant 7 at load time.

This distinction should be made explicit in the domain model. The orchestrator introduces a new kind of ensemble invocation — dynamic, tool-mediated, outside the reference graph — that coexists with but does not replace static composition.

## Plexus as Memory Layer

Without a memory layer, the orchestrator agent is stateless across sessions. Each conversation starts from zero. This is not a hypothetical problem — it is the current state of the art. OpenHands' agents re-discover codebases every session; a MemU blog post identifies this directly as a structural weakness. Claw-code has no persistent memory beyond its `TranscriptStore`. The community workaround in both ecosystems is flat-file memory (`AGENTS.md` files, `CLAUDE.md` instructions) — useful but not queryable, not structured, and not capable of answering relational questions like "what has worked for similar tasks?" or "why did this ensemble fail last time?"

Stateless operation is adequate for simple tool use but inadequate for the self-building ensemble vision, where the agent needs to know what ensembles exist, how they have performed, and what patterns have worked for similar tasks.

Plexus, the sibling knowledge graph engine, provides the infrastructure. ADR-020 in the conductor skill already targets Plexus for this role, identifying three categories of data: provenance (ensemble specs, calibration runs, routing decisions), design knowledge (DAG shapes, profile pairings, anti-patterns), and operational data (routing logs, evaluation records, token usage).

### Integration Architecture

Two integration modes compose naturally:

**As a tool the orchestrator calls.** The orchestrator has tools like `query_knowledge("what ensembles work for code review?")` and `record_outcome(ensemble, score, context)`. This is the simplest integration — Plexus is another tool in the ReAct loop, called when the orchestrator needs to look something up or record a result.

**As context injected into prompts.** Before the orchestrator processes a request, the server queries Plexus for relevant context — prior routing decisions for similar tasks, ensemble quality history, known patterns — and injects it into the system prompt. This gives the orchestrator baseline awareness without burning tool-call turns.

Both modes are needed. Pre-loaded context answers "what do I already know about this kind of task?" Tool access answers "let me look up the specifics." The pre-loaded context is cheap (one query at session start); the tools are precise (queries during reasoning when the orchestrator identifies a specific knowledge gap).

### What Plexus Enables

1. **Cross-session memory.** The orchestrator routes to the `code-review` ensemble because Plexus records that it has scored well in prior sessions.
2. **Failure analysis.** An ensemble scores poorly. Plexus records the failure with provenance. Future similar tasks check: "has this ensemble failed on similar inputs?"
3. **Design knowledge accumulation.** Each ensemble creation and calibration generates design knowledge. Over time, the orchestrator (or its designer mode) makes better choices because it has seen what works.
4. **Provenance for debugging.** When a user asks "why did you use that ensemble?", the orchestrator traces through Plexus: routing decision, task profile, evaluation history, ensemble creation rationale.

The integration should start with Plexus as a tool and add context injection as the knowledge graph accumulates enough data to be useful at session start.

## What Emerges

The architecture that emerges from this research is a layered system:

**Layer 1 — API Surface.** OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/models`) added to the existing FastAPI server. Handles protocol translation: SSE streaming, tool-call formatting, model listing.

**Layer 2 — Orchestrator Agent.** A ReAct loop powered by a configurable LLM (via model profile). Has access to llm-orc tools (invoke ensemble, create ensemble, list/manage ensembles) and Plexus tools (query knowledge, record outcomes). Manages its own context through summarization and compaction.

**Layer 3 — Ensemble Engine.** The existing DAG execution engine, unchanged. Receives invocations from the orchestrator, executes them, returns results. Provides parallel multi-model execution that the orchestrator treats as atomic tool calls.

**Layer 4 — Knowledge Graph.** Plexus, providing cross-session memory, provenance tracking, and design knowledge accumulation. Queried by the orchestrator both at session start (context injection) and during reasoning (tool calls).

The system's distinguishing property is that the orchestrator can extend Layer 3 at runtime — creating new ensembles from existing primitives, calibrating them through use, and accumulating design knowledge in Layer 4. Over time, the ensemble library grows through use, and the orchestrator's routing decisions improve through recorded experience.

This is not a novel architecture in isolation. The ReAct loop, the tool-calling pattern, the knowledge graph — all have precedent. OpenHands has the most mature event-sourced agent loop; claw-code demonstrates provider-agnostic design and budget enforcement; LiteLLM solves the provider abstraction problem. What none of them have is persistent, structured memory that accumulates across sessions and enables the system to improve through use.

That is the gap, and it is the gap that Plexus fills. The distinctive property of this architecture is that the knowledge graph converts the orchestrator's routing decisions from reasoning tasks into retrieval tasks — and retrieval tasks run on cheaper models. Over time, the ensemble library grows through use, the knowledge graph accumulates routing decisions and quality signals, and the cost of operating the system decreases as less model capability is needed for routine decisions. Each layer is designed to be independently operable — Layer 3 already runs standalone today; Layers 2 and 4 would be additive, not modifying existing behavior. The value is in their composition, and the economic argument strengthens as the knowledge graph matures.
