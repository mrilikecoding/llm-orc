# Research Log: Agentic Serving for llm-orc (Cycle 2)

## Question 5: What lessons does OpenHands' architecture offer?

**Method:** Web search + documentation analysis

**Findings:**

### What OpenHands is

OpenHands (formerly All-Hands-AI) is an open-source platform for building and running autonomous AI software engineering agents. 65K+ GitHub stars, $18.8M Series A (late 2025), Apache 2.0 licensed. Reports resolving 87% of bug tickets same-day at enterprise scale.

Three deployment modes: web UI, headless CLI (for CI/pipelines), and an SDK for programmatic agent definition.

### Architecture patterns

**Event-sourced ReAct loop.** All agent-environment interaction flows through an `EventStream` — an append-only, immutable log of typed events. The hierarchy splits into `ActionEvent` (run code, browse, delegate) and `ObservationEvent` (stdout, diffs, screenshots). State is derived from the log, not mutated in place. This enables deterministic replay — any failed run can be reproduced exactly.

This is architecturally distinct from a direct function-call loop. The event stream is the coordination primitive, not the call stack. Components publish and subscribe rather than calling each other.

**Tool system.** Tools follow an Action-Observation triad: Pydantic-validated actions, a dispatcher that enforces constraints, and structured observations. Tool arguments are validated at the schema boundary before execution. 19 permission-gated tools including file ops, bash, git, browser, LSP, notebook editing, and sub-agent spawning.

**Multi-agent delegation.** `AgentDelegateAction` lets a generalist agent spawn specialized sub-agents for subtasks. Delegation is a typed action in the event stream, not a framework-level construct. This is closer to Google ADK's `AgentTool` than to LangGraph's graph wiring.

**LLM routing via LiteLLM.** Provider abstraction across 100+ providers. Supports cloud (OpenAI, Anthropic, Google), local (Ollama, vLLM, SGLang, LM Studio), and proxy routing. Model selection is per-agent-session; no dynamic per-task routing in the core framework.

### Memory gap

OpenHands has a `Memory` class with keyword-triggered "microagents" that inject context into prompts, plus `ConversationMemory` for event-to-message conversion. But there is no persistent vector store, knowledge graph, or semantic retrieval in the core framework. A MemU blog post identifies this directly: "Coding Agents Without Project Memory Re-Discover Codebases Every Session." The community workaround is `AGENTS.md` files stored in repos.

### OpenHands' own distinction: agentic AI vs. AI orchestration

Their January 2026 blog post draws a deliberate line:
- **Agentic AI** = the LLM-driven Perceive-Reason-Act-Observe loop
- **AI orchestration** = the layer managing lifecycle, resource boundaries, timeouts, retries, sandboxing

Their thesis: small agents can run without orchestration, but production systems require both.

**Implications for llm-orc:**

1. The event-sourced architecture is the most mature open-source reference for production-scale agentic execution. The immutable event log enables replay, auditability, and horizontal scaling — properties the orchestrator agent would benefit from.
2. LiteLLM as the provider abstraction layer is a validated pattern — llm-orc's `openai-compatible` provider work is heading in this direction already.
3. The persistent memory gap is where llm-orc + Plexus could differentiate. OpenHands agents re-discover codebases every session; an llm-orc orchestrator with Plexus would not.
4. The agentic-vs-orchestration distinction maps to the essay's Layer 2 (orchestrator agent) vs Layer 1 (API surface + lifecycle management). llm-orc needs both.
5. Delegation as a typed action (not graph wiring) aligns with the essay's model: ensembles as tools, not as nodes in a fixed graph.

## Question 6: What lessons does claw-code's architecture offer?

**Method:** Web search + documentation analysis

**Findings:**

### What claw-code is

Claw-code is a clean-room rewrite of Claude Code's agent harness architecture, created by Sigrid Jin (@instructkr) after the March 2026 Claude Code source leak. Originally Python, now being rewritten in Rust (72.9% Rust, 27.1% Python). 117K+ GitHub stars. Built using oh-my-codex (orchestration layer on OpenAI Codex) with parallel code review.

### Architecture patterns

**Turn-based agentic loop.** `QueryEnginePort.submit_message()` implements the loop with:
- `max_turns` limit (default: 8)
- Cumulative token budget (`max_budget_tokens`: 2000)
- History compaction when threshold exceeded
- `TurnResult` with `stop_reason`

**Mirrored snapshot tool system.** 184 tools loaded from `tools_snapshot.json` as static entries. `MirroredTool` wrapper invokes `execute_tool(name, payload)`. Permission-based filtering via `ToolPermissionContext` with deny lists. Simple mode restricts to core file/shell operations. This is architecturally interesting — tools are a static, filterable catalog rather than dynamically discovered.

**Provider-agnostic LLM integration.** Supports Claude, OpenAI, and local models — a deliberate departure from Claude Code's Claude-exclusive design.

**Streaming.** `stream_submit_message()` generator yields dictionary payloads across phases: `message_start`, `tool_match`, `message_delta`, `message_stop`. Supports structured JSON output with retry logic.

**History compaction.** Triggers at `compact_after_turns` (default: 12). Prunes to retain only recent turns. `TranscriptStore` maintains synchronized log. Token tracking via whitespace-split heuristic.

**Sub-agent spawning ("swarms").** Parallel task decomposition via isolated sub-agents with shared memory access.

**Full MCP support.** 6 transport types: Stdio, SSE, HTTP, WebSocket, SDK, ClaudeAiProxy. Auto name normalization and OAuth authentication.

### What's architecturally interesting

1. **Static tool catalog with permission filtering.** Rather than dynamic tool discovery, tools are a known set that gets filtered per-context. This is simpler than MCP's dynamic discovery and may be sufficient for the orchestrator agent's tool set (which is the fixed set of llm-orc operations).
2. **Budget enforcement at the loop level.** Turn limits and token budgets are loop-level constraints, not model-level. The orchestrator agent would need the same — budget enforcement is a control plane concern, not a model concern.
3. **History compaction as a first-class concern.** The compaction strategy (trigger threshold, prune to recent, synchronized transcript) addresses the context rot problem the essay identifies.
4. **Provider agnosticism as a design goal.** The explicit departure from Claude-only shows that multi-provider support is a user expectation for agent harnesses.

**Implications for llm-orc:**

1. The turn-based loop with budget enforcement maps directly to the orchestrator agent pattern — the orchestrator needs both turn limits and token budgets as control plane constraints.
2. The static tool catalog approach may be simpler and sufficient for the orchestrator's tool set, which is the bounded set of llm-orc operations (invoke, create, list, query) rather than a dynamically extensible set.
3. History compaction validates the essay's context management concerns — claw-code treats it as a core architectural concern, not an optimization.
4. The Rust rewrite suggests performance matters for the harness layer. If llm-orc's orchestrator handles many concurrent sessions, the serving layer's performance profile matters.
