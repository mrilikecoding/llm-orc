# Composable Ensemble Orchestration

## The Problem

llm-orc can orchestrate multiple agents within a single ensemble — LLM agents calling models and script agents running code, coordinated through dependency chains, parallel phases, and fan-out expansion. But it cannot compose ensembles together. A code analysis pipeline can't chain a security-scanning ensemble into a quality-review ensemble into a synthesis ensemble. A document processing workflow can't route PDFs through one extraction ensemble and audio files through another.

Issue #29 proposed a solution: a new "meta-ensemble" YAML format with its own execution strategies (sequential, parallel, conditional, iterative), template-based data flow, and a dedicated `MetaEnsembleExecutor`. This would work, but it introduces a parallel system — new config format, new executor, new validation, new CLI commands — when the existing ensemble format already handles agent coordination well.

The question is whether we can get ensemble composition without a new format.

## The Design: Ensembles as Agents

The existing agent dispatch works by key inspection. An agent with `model_profile` or `model` is an LLM agent. An agent with `script` is a script agent. The natural extension: an agent with `ensemble` is an ensemble agent — it runs another ensemble and returns the result.

```yaml
name: document-analysis-pipeline
agents:
  - name: scanner
    script: scripts/scan-directory.py

  - name: deep-analysis
    ensemble: detailed-code-review
    depends_on: [scanner]

  - name: synthesizer
    model_profile: research-synthesizer
    depends_on: [deep-analysis]
```

No new format. The ensemble agent participates in dependency chains, fan-out, and parallel phases exactly like any other agent. The `EnsembleExecutor` recurses: when it encounters an ensemble agent, it loads and executes the referenced ensemble, returning the result as the agent's response.

This is composition through uniformity. An ensemble doesn't need to know whether its agents are LLMs, scripts, or other ensembles. They're all agents with inputs and outputs.

### Routing by File Type

The motivating use case: a meta-ensemble receives a directory of mixed files. Different file types need different extraction ensembles. A classifier script groups files by type, and separate ensemble agents handle each group:

```yaml
name: semantic-extraction-pipeline
agents:
  - name: classifier
    script: scripts/classify-by-type.py
    # outputs: {pdfs: [...], audio: [...], images: [...]}

  - name: pdf-extractor
    ensemble: pdf-semantic-extraction
    depends_on: [classifier]
    input_key: pdfs
    fan_out: true

  - name: audio-extractor
    ensemble: audio-semantic-extraction
    depends_on: [classifier]
    input_key: audio
    fan_out: true

  - name: synthesizer
    model_profile: research-synthesizer
    depends_on: [pdf-extractor, audio-extractor]
```

This combines three orthogonal features:

1. **`ensemble:` agent type** — run another ensemble (the core addition)
2. **`input_key:`** — select a specific key from upstream output (new, works on any agent type)
3. **`fan_out: true`** — expand per array item (already exists)

Each is independently useful. An ensemble agent works without fan-out. `input_key` works without ensemble agents. They compose for the routing case but can be delivered as separate increments.

## Prerequisite: Pydantic Agent Config Models

Agent configurations are currently `list[dict[str, Any]]`. Type determination happens at runtime through key inspection. There is no load-time validation that an agent has the right fields for its type — a typo in `model_profile` silently produces an agent that fails at execution time.

Pydantic is already used elsewhere in the project: script agent I/O schemas, validation config models, execution results. Agent configs are the gap. Adding a third agent type to the untyped dict system would make the type-dispatch logic increasingly fragile.

The right sequence is: migrate agent configs to Pydantic first (structural change, no behavior change), then add the ensemble agent type (behavior change on clean structure). "Make the change easy, then make the easy change."

### The Models

A survey of 63 ensemble files containing 250 agent configurations identified 10 meaningful fields. No agent uses both `model_profile` and `script`. No exotic combinations exist. The data fits a clean discriminated union:

```python
class BaseAgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    depends_on: list[str] = []
    fan_out: bool = False

class LlmAgentConfig(BaseAgentConfig):
    model_profile: str | None = None    # path 1: named profile
    model: str | None = None            # path 2: inline model
    provider: str | None = None         # required with inline model
    system_prompt: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    timeout_seconds: int | None = None
    output_format: str | None = None
    fallback_model_profile: str | None = None
    # Validator: model_profile XOR (model + provider)

class ScriptAgentConfig(BaseAgentConfig):
    script: str
    parameters: dict[str, Any] = {}

class EnsembleAgentConfig(BaseAgentConfig):
    ensemble: str

AgentConfig = LlmAgentConfig | ScriptAgentConfig | EnsembleAgentConfig
```

`LlmAgentConfig` supports two mutually exclusive paths: a named profile reference (`model_profile`) or an inline model specification (`model` + `provider`). The inline path exists for experimentation — trying different models without creating a profile. A validator enforces that exactly one path is provided.

`extra="forbid"` catches typos and invalid fields at load time. The project is in beta; strict schemas are appropriate.

### Fields Dropped

- **`type`** — 4 agents used `type: script`, redundant with key-based discrimination. With Pydantic, `isinstance(agent, ScriptAgentConfig)` replaces runtime dispatch.
- **`conversation`** — dead code from an unfinished conversational ensemble system (see below).
- **`synthesis_timeout_seconds`** — one agent used it, no code reads it.
- **`cost_per_token`** — profile-level concern, no reason for per-agent override.

### Migration Blast Radius

The migration touches `EnsembleConfig`, `EnsembleLoader`, `AgentDispatcher`, `DependencyAnalyzer`, `DependencyResolver`, `FanOutCoordinator`, `LlmAgentRunner`, `ScriptAgentRunner`, and all tests that construct agent config dicts. Wide but shallow — each change is mechanical type narrowing, not logic change. The test suite verifies no behavior change.

## Ensemble Agent Runtime Behavior

### Black Box with Full Results

Ensemble agents are black boxes. The calling ensemble doesn't see the child's internal agents or phases. The child ensemble produces a result dict; the ensemble agent's "response" is that full result serialized as JSON.

This gives downstream consumers maximum flexibility. LLM agents receive the full context in their prompt. Script agents receive structured data they can parse. If a downstream agent only needs part of the result, `input_key` provides consumer-side selection — one mechanism instead of two.

If experience reveals that full results are too verbose for certain patterns, an `output_mode` field can be added to `EnsembleAgentConfig` later without breaking changes.

### Artifact Scoping

Child ensembles do not produce their own artifacts. The child's execution results are captured as a nested entry within the parent orchestrator's artifact. The orchestrator ensemble is the unit of work; its artifact represents the full execution tree.

The `EnsembleAgentRunner` creates the child executor with artifact saving disabled.

### Resource Sharing

The `EnsembleExecutor` holds immutable infrastructure (config manager, credential storage, model factory) and mutable per-execution state (usage collector, event queue, streaming tracker).

Child executors share the parent's immutable infrastructure — no redundant credential lookups or profile resolution. Mutable state is isolated per execution. Model instances are created per-agent by the factory, so sharing the factory creates no contention. The depth limit bounds total concurrent resource usage.

### Cycle Detection

Cross-ensemble cycles (A references B, B references A) must be caught at load time. The approach extends the existing DFS cycle detection to operate at the ensemble level: scan `ensemble:` keys across all loaded ensemble configs, build a directed graph of ensemble-to-ensemble references, and check for cycles.

This runs once during loading, not per-execution. The loader already has access to ensemble directories via `find_ensemble()`.

### Depth Limit

A configurable depth limit (default 4-8) prevents unbounded nesting. A `depth` counter passes through each recursive invocation; exceeding the limit produces a clear error. The limit lives in performance config as a system-level constraint.

## Cleanup: Conversational Ensemble System

Research uncovered that the conversational ensemble system — `ConversationalEnsembleExecutor`, `ConversationalAgent` schemas, and associated tests — is dead code. It is not imported by the CLI, MCP server, or any execution path. The `conversation:` key on agents appears in exactly one example ensemble file.

The regular execution path already supports user-directed flows through script primitives (`get_user_input.py`, `confirm_action.py`), the `ScriptUserInputHandler`, and conditional dependencies. The conversational system duplicates this with worse integration.

Removing it simplifies the codebase and eliminates a confusing parallel system before the Pydantic migration.

## Implementation Sequence

Four increments, each independently shippable:

1. **Cleanup**: Remove conversational ensemble dead code. Delete 4 files, verify tests pass.

2. **Structure (refactor)**: Introduce Pydantic agent config models. Migrate `EnsembleConfig.agents` from `list[dict[str, Any]]` to `list[AgentConfig]`. Replace `AgentDispatcher._determine_agent_type()` with `isinstance` checks. Remove `type` field from 4 agents in 2 YAML files. Drop `synthesis_timeout_seconds` from its one usage. All existing tests pass — no behavior change.

3. **Behavior**: Add `EnsembleAgentConfig` and `EnsembleAgentRunner`. Implement cross-ensemble cycle detection. Add depth limiting. Wire into `AgentDispatcher` and `EnsembleExecutor._execute_agent()`.

4. **Behavior**: Add `input_key` to `BaseAgentConfig`. Modify `DependencyResolver` to select the specified key from upstream results before building agent input. Works for all agent types.

Each increment is a single commit type: cleanup, refactor, feat, feat. Structure never mixes with behavior.
