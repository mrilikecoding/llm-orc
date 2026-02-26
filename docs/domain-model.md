# Domain Model: Composable Ensemble Orchestration

## Concepts (Nouns)

| Term | Definition | Avoid (synonyms) |
|------|-----------|-------------------|
| **Ensemble** | A named configuration of agents with dependency relationships, loaded from a YAML file. The unit of orchestration. | "workflow", "pipeline" (use only informally) |
| **Agent** | A named participant in an ensemble that receives input and produces output. Has exactly one type: LLM, Script, or Ensemble. | "node", "step", "task" |
| **Agent Config** | The Pydantic model (`AgentConfig`) that validates and represents an agent's declaration within an ensemble YAML. A discriminated union of `LlmAgentConfig`, `ScriptAgentConfig`, and `EnsembleAgentConfig`. | "agent dict", "agent definition" |
| **LLM Agent** | An agent that sends input to a language model and returns its response. Identified by presence of `model_profile` or `model` key. | "model agent" |
| **Script Agent** | An agent that executes a script as a subprocess and returns its output. Identified by presence of `script` key. | "command agent" |
| **Ensemble Agent** | An agent that recursively executes another ensemble and returns its result. Identified by presence of `ensemble` key. The mechanism for ensemble composition. | "meta-agent", "sub-ensemble", "nested ensemble" |
| **Model Profile** | A named configuration combining model identifier, provider, and defaults (system prompt, timeout, temperature, provider-specific options). Stored in config YAML, referenced by name from LLM agents. | "model config" |
| **Inline Model** | An LLM agent specification using `model` + `provider` directly, bypassing the profile system. For experimentation. | "anonymous model", "direct model" |
| **Dependency** | A declared ordering relationship between agents (`depends_on`). Upstream agents must complete before downstream agents execute. | "prerequisite", "requirement" |
| **Phase** | A group of agents with no unresolved dependencies between them, eligible for parallel execution. Produced by topological sort of the dependency graph. | "stage", "level", "tier" |
| **Fan-Out** | Expansion of a single agent declaration into N parallel instances, one per item in an upstream array result. Declared with `fan_out: true`. | "map", "scatter" |
| **Input Key** | An optional field on any agent config that selects a specific key from the upstream agent's output before passing it as input. Enables routing patterns. | "output selector", "key filter" |
| **Ensemble Reference** | The static name string in an `EnsembleAgentConfig.ensemble` field, resolved to an ensemble config at load time. | "ensemble link", "ensemble pointer" |
| **Ensemble Reference Graph** | The directed graph of ensemble-to-ensemble references across all loaded ensembles. Used for cross-ensemble cycle detection. | "ensemble DAG", "reference DAG" |
| **Depth** | The nesting level of recursive ensemble execution. The top-level ensemble is depth 0; each ensemble agent increments depth by 1. | "nesting level", "recursion depth" |
| **Depth Limit** | A system-level maximum for ensemble nesting depth, configured in performance config. Prevents unbounded recursion. | "max depth", "recursion limit" |
| **Artifact** | A persisted record of an ensemble execution's results. One artifact per top-level execution; child ensemble results are nested within the parent artifact. | "execution log", "result file" |
| **Child Executor** | An `EnsembleExecutor` instance created by an `EnsembleAgentRunner` to execute a referenced ensemble. Shares immutable infrastructure with the parent; isolates mutable state. | "nested executor", "sub-executor" |
| **Immutable Infrastructure** | Shared components that are stateless per-call: config manager, credential storage, model factory. Safe to share between parent and child executors. | — |
| **Mutable State** | Per-execution components that track state: usage collector, event queue, streaming tracker. Must be isolated per executor instance. | — |
| **Agent Discriminator** | The mechanism that determines agent type from config. Currently runtime key inspection; after migration, Pydantic discriminated union resolved at parse time. | "type dispatch", "agent router" |

## Actions (Verbs)

| Action | Actor | Subject | Description |
|--------|-------|---------|-------------|
| **Load** | EnsembleLoader | Ensemble | Parse YAML file into `EnsembleConfig` with validated `AgentConfig` models |
| **Validate** | EnsembleLoader | Ensemble | Check for missing dependencies, fan-out requirements, intra-ensemble cycles, and cross-ensemble reference cycles |
| **Discriminate** | AgentConfig (Pydantic) | Agent dict | Parse a raw YAML dict into the correct `AgentConfig` subtype based on key presence |
| **Execute** | EnsembleExecutor | Ensemble | Run all agents in phase order, coordinating dependencies and fan-out |
| **Dispatch** | AgentDispatcher | Agent | Route an agent to its runner (`LlmAgentRunner`, `ScriptAgentRunner`, or `EnsembleAgentRunner`) based on config type |
| **Recurse** | EnsembleAgentRunner | Ensemble Agent | Load the referenced ensemble, create a child executor, execute it, and return the result as the agent's response |
| **Fan Out** | FanOutCoordinator | Agent | Expand a single agent config into N instances based on an upstream array |
| **Gather** | FanOutGatherer | Fan-out instances | Collect results from expanded instances into an ordered array under the original agent name |
| **Select** | DependencyResolver | Input Key | Extract a specific key from an upstream agent's output before passing it to the consuming agent |
| **Detect Cycles** | EnsembleLoader | Ensemble Reference Graph | Build directed graph of ensemble references and run DFS to find cycles at load time |
| **Check Depth** | EnsembleAgentRunner | Depth | Compare current depth against depth limit before recursive execution; raise error if exceeded |
| **Merge Profile** | LlmAgentRunner | LLM Agent | Combine profile defaults with agent-level overrides (`{**profile, **agent}`); agent fields win |

## Relationships

- An **Ensemble** *has many* **Agents** (ordered list)
- An **Agent** *belongs to* exactly one **Ensemble**
- An **Agent** *has exactly one* **Agent Config** type (LLM, Script, or Ensemble — mutually exclusive)
- An **LLM Agent** *references* zero or one **Model Profile** (or uses inline model + provider)
- An **Ensemble Agent** *references* exactly one **Ensemble** (by name)
- An **Agent** *depends on* zero or more other **Agents** (within the same ensemble)
- **Dependencies** *determine* **Phases** (via topological sort)
- **Agents** within a **Phase** *execute in parallel*
- A **Fan-Out** agent *expands into* N parallel instances
- An **Input Key** *selects from* an upstream agent's output
- The **Ensemble Reference Graph** *spans across* all loaded ensembles
- A **Child Executor** *shares* **Immutable Infrastructure** with its parent
- A **Child Executor** *isolates* its own **Mutable State**
- A child ensemble's results *nest within* the parent's **Artifact**

## Invariants

1. **Mutual exclusivity of agent type**: An agent config must contain exactly one of: (`model_profile` or `model`+`provider`), `script`, or `ensemble`. Never more than one. Never none.

2. **Inline model completeness**: If an LLM agent specifies `model`, it must also specify `provider`. `model` without `provider` is invalid.

3. **Profile XOR inline**: An LLM agent must specify either `model_profile` or (`model` + `provider`), not both.

4. **Intra-ensemble acyclicity**: The dependency graph within a single ensemble must be a DAG. Cycles in `depends_on` are rejected at load time.

5. **Cross-ensemble acyclicity**: The ensemble reference graph must be a DAG. If ensemble A contains an agent referencing ensemble B, and ensemble B contains an agent referencing ensemble A, this is rejected at load time.

6. **Fan-out requires dependency**: An agent with `fan_out: true` must have a non-empty `depends_on` list. The first dependency must produce an array result.

7. **Static ensemble references**: Ensemble agent references are string literals resolved at load time. No template expressions or dynamic resolution.

8. **Depth is bounded**: Recursive ensemble execution must not exceed the configured depth limit. The top-level ensemble is depth 0; each ensemble agent increments depth by 1.

9. **Child artifacts are nested**: A child ensemble execution does not produce its own artifact on disk. Its results are captured within the parent ensemble's artifact.

10. **Child state is isolated**: A child executor shares immutable infrastructure (config manager, credential storage, model factory) but must have its own mutable state (usage collector, event queue, streaming tracker).

11. **Extra fields are forbidden**: Agent configs reject unknown fields at parse time (`extra="forbid"`). This catches typos and invalid configurations at load time rather than runtime.

12. **Agent-level overrides win**: When an LLM agent specifies both a `model_profile` and an agent-level field (e.g., `system_prompt`, `temperature`, `options`), the agent-level value overrides the profile default. For the `options` dict, profile and agent options are deep-merged with agent keys winning on conflict.

13. **Execution is resilient**: An agent failure does not halt ensemble execution. Agents that do not depend on the failed agent continue to execute. Downstream agents that depend on the failed agent receive the failure status in their dependency context. The ensemble completes with partial results and a `has_errors` flag. This applies equally to LLM agents, script agents, and ensemble agents — a child ensemble failure is an agent failure, not an orchestration failure.

14. **Validation catches structural errors; execution tolerates runtime errors**: Structural problems (missing dependencies, cycles, unknown fields, invalid agent type) are caught at load time and prevent execution. Runtime problems (model unavailable, script timeout, child ensemble failure) are recorded but do not stop the ensemble.

## Amendment Log

| # | Date | Invariant | Change | Propagation |
|---|------|-----------|--------|-------------|
| — | — | — | Initial model, no prior version | — |
| 1 | 2026-02-20 | Invariant 13 (Execution is resilient) | Added. Agent failures do not halt ensemble execution; results are partial with `has_errors` flag. | No prior ADR contradictions found. Codebase audit "Silent Failure Philosophy" finding aligns in intent but current implementation suppresses errors silently rather than recording them — conformance debt. |
| 2 | 2026-02-20 | Invariant 14 (Validation vs. execution boundary) | Added. Structural problems caught at load time; runtime problems recorded but do not stop the ensemble. | No prior ADR contradictions found. ADR-007 validation layers are complementary, not conflicting. |
