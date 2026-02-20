# Research Log: Composable Ensemble Orchestration

## Question 1: What is the right shape for ensemble composition?

**Method:** Codebase analysis + design conversation

**Findings:**

Three candidate approaches for multi-ensemble orchestration were evaluated:

- **Option A (selected):** Static ensemble references with explicit routing via `input_key` selection. An agent config uses `ensemble:` key instead of `model_profile:` or `script:`. The ensemble name is resolved at load time, enabling static cycle detection. Routing between different ensembles per file type is handled by a classifier script producing keyed output, with separate ensemble agents each selecting their key via `input_key:`.

- **Option B (rejected):** Dynamic ensemble resolution via templates (`ensemble: "${route.ensemble}"`). More flexible but prevents static validation and makes cycle detection a runtime problem.

- **Option C (rejected):** Routing pushed into child ensembles via conditional agent execution. Requires a new `condition:` mechanism on agents and conflates routing with extraction logic.

**Key design insight:** The ensemble agent, `input_key`, and `fan_out` are three orthogonal, composable features — not one monolithic feature. An ensemble agent works standalone with no fan-out. `input_key` is useful for any agent type. They compose naturally for the file-routing use case but can be delivered independently.

**Implications:** Three separate increments rather than one large feature. Ensemble agent type is the core; `input_key` is an independent enhancement.

## Question 2: Should we use Pydantic for ensemble agent config validation?

**Method:** Codebase analysis

**Findings:**

Pydantic is already used in the project for:
- Script agent I/O schemas (`ScriptAgentInput`, `ScriptAgentOutput` in `schemas/script_agent.py`)
- Conversational agent schemas with field validators (`ConversationalAgent` in `schemas/conversational_agent.py`)
- Validation config models (`ValidationConfig` in `core/validation/models.py`)
- Execution result models

But ensemble agent configs remain `list[dict[str, Any]]` with runtime key inspection for type determination (`AgentDispatcher._determine_agent_type()`). No compile-time or load-time validation that an agent has correct fields for its type.

The `ConversationalAgent` schema already validates mutual exclusivity of `script` and `model_profile` — exactly the pattern needed, extended to include `ensemble` as a third variant.

**Decision:** Migrate agent configs to Pydantic as a structural change *before* adding the ensemble agent type. "Make the change easy, then make the easy change." This was always the long-term intent when Pydantic was introduced for script agents.

**Implications:** The work splits into three increments:
1. **Structure (refactor):** Introduce Pydantic agent config models (`LlmAgentConfig`, `ScriptAgentConfig`, shared `BaseAgentConfig`). Discriminated union via key presence. All existing tests pass — no behavior change.
2. **Behavior:** Add `EnsembleAgentConfig` variant and `EnsembleAgentRunner` with recursive execution + cross-ensemble cycle detection.
3. **Behavior:** Add `input_key` field to `BaseAgentConfig` for selective upstream output consumption.

## Question 3: How should ensemble agents behave at execution time?

**Method:** Design conversation + codebase analysis

**Findings across five sub-questions:**

### 3a. Result format — what does the parent see?

Ensemble agents are black boxes from the calling ensemble's perspective. The child ensemble produces a full result dict (with `results`, `metadata`, `status`). The question is what surfaces as the ensemble agent's "response" to downstream agents.

**Decision:** Default to returning the full result dict as JSON. This gives downstream LLM agents full context in their prompt and downstream script agents structured data to parse. Rather than adding `output_mode` to the ensemble agent config, rely on `input_key` on downstream agents to select what they need. One mechanism (consumer-side selection) instead of two (producer-side filtering + consumer-side selection).

If experience shows this is insufficient, `output_mode` can be added later without breaking changes.

### 3b. Artifact handling — nested, not separate

Child ensemble execution should not produce its own artifact on disk. The child's full results become a nested entry within the parent orchestrator's artifact. This means the `EnsembleAgentRunner` must either:
- Suppress artifact saving in the child executor, or
- Create the child executor with artifact saving disabled

The orchestrator ensemble is the unit of work; its artifact captures the full execution tree.

### 3c. Executor resource sharing

The `EnsembleExecutor` holds both immutable infrastructure (config manager, credential storage, model factory) and mutable per-execution state (usage collector, event queue, streaming tracker).

**Decision:** Share immutable infrastructure, isolate mutable state. The child `EnsembleAgentRunner` creates a child executor that receives the parent's config manager, credential storage, and model factory (the expensive pieces — credential lookups, profile resolution, all stateless per-call). But it gets its own usage collector, event queue, and streaming tracker. This avoids duplicating expensive setup while preventing state contamination.

System resources are the practical constraint. Model instances (Ollama connections, API clients) are created per-agent-execution by the model factory, so sharing the factory doesn't create contention. The depth limit (see 3e) bounds total concurrent resource usage.

### 3d. Cross-ensemble cycle detection

Cycles like A → B → A must be caught at load time, not runtime. The existing `detect_cycle()` only checks `depends_on` within a single ensemble's agent list.

**Approach:** When loading an ensemble, build a directed graph of ensemble-to-ensemble references by scanning all `ensemble:` keys in agent configs. Apply the same DFS cycle detection used for agent dependencies, but at the ensemble level. This runs once at load time in `EnsembleLoader._validate_dependencies()` (or a new `_validate_ensemble_references()` step).

Requires the loader to have access to the ensemble directory to resolve references — which it already does via `find_ensemble()`.

### 3e. Depth limit

Even without cycles, unbounded nesting is a practical risk (resource exhaustion, debugging difficulty). A configurable depth limit with a sensible default (e.g., 4-8 levels) serves as a safety net.

**Approach:** Pass a `depth` counter through the `EnsembleAgentRunner`. The parent starts at depth 0; each nested invocation increments. Exceeding the limit produces a clear error. The limit is configurable in performance config (not per-ensemble, since it's a system-level constraint).

**Implications:** The `EnsembleAgentRunner` is a thin adapter: resolve the ensemble name, create a child executor with shared infrastructure + isolated state + artifact saving disabled, pass input, return the result dict as the agent's response. Most of the complexity is in the structural Pydantic migration, not in the runner itself.

## Question 4: What is the blast radius of the Pydantic migration?

**Method:** Automated survey of all ensemble YAML files + codebase analysis

**Findings:**

Surveyed 63 ensemble files containing 250 agent configurations across `.llm-orc/ensembles/` and `llm-orchestra-library/ensembles/`.

### Field inventory (actual usage)

Only 12 unique keys appear across all 250 agents. After removing the `conversation` cruft (see Question 5), the clean inventory is 10 keys:

**Shared (`BaseAgentConfig`):**
- `name` (250 agents) — required
- `depends_on` (90 agents) — optional list
- `fan_out` (2 agents) — optional bool

**LLM agents (`LlmAgentConfig`):**
- `model_profile` (171 agents) — required for LLM type
- `system_prompt` (151 agents)
- `timeout_seconds` (6 agents)
- `output_format` (10 agents)
- `temperature` (0 in YAML, but fully supported in code path via `ModelFactory`)
- `max_tokens` (0 in YAML, but fully supported in code path via `ModelFactory`)
- `synthesis_timeout_seconds` (1 agent — one-off, may be LLM-specific)

**Script agents (`ScriptAgentConfig`):**
- `script` (79 agents) — required for script type
- `parameters` (75 agents)

### The `type` field

Used on 4 agents (2 ensembles), always `type: script` — redundant with the `script:` key. `AgentDispatcher._determine_agent_type()` checks it first but falls back to key inspection. With Pydantic discriminated unions, the model class *is* the type. The `type` field can be dropped; existing YAMLs that include it will still parse because `script:` key presence determines the variant.

### Agent type patterns

- `model_profile` only: 171 agents (LLM)
- `script` only: 75 agents (script, implicit type)
- `script` + `type: script`: 4 agents (script, explicit type)
- No agent uses both `model_profile` and `script`
- No agent uses bare `model` + `provider` (code supports it, nobody uses it)

### Backward compatibility

The project is in beta — strict schemas are appropriate. Pydantic's `model_config = ConfigDict(extra="forbid")` would catch typos and invalid fields at load time rather than silently ignoring them. Unknown keys in existing YAML would become validation errors, which is desirable.

The `type` field requires a decision: accept and ignore it (`extra="ignore"` just for that field), or add it as an optional literal field on the base. Recommendation: accept it as an optional field for now, mark deprecated, remove in next minor version.

### Blast radius assessment

The migration touches:
- `EnsembleConfig.agents` type changes from `list[dict[str, Any]]` to `list[AgentConfig]`
- `EnsembleLoader.load_from_file()` — parse agents through Pydantic
- `AgentDispatcher._determine_agent_type()` — replaced by isinstance checks
- `DependencyAnalyzer` — reads `depends_on` from configs
- `DependencyResolver` — checks `script` key to determine formatting
- `FanOutCoordinator/Expander` — reads `fan_out` and `depends_on`
- `LlmAgentRunner` — reads model_profile, system_prompt, etc.
- `ScriptAgentRunner` — reads script, parameters
- All tests that construct agent config dicts

The blast radius is wide but shallow — many files need to change from `dict[str, Any]` to typed models, but each change is mechanical. No logic changes, just type narrowing.

## Question 5: Conversational ensemble system — keep or remove?

**Method:** Codebase analysis — tracing imports, call sites, and YAML usage

**Findings:**

The conversational ensemble system is dead code:

- `ConversationalEnsembleExecutor` — exists in source, never imported by CLI, MCP server, or any real execution path
- `ConversationalAgent`, `ConversationalEnsemble`, `ConversationState` schemas — only imported by the dead executor and its tests
- `conversation:` key on agents — used by exactly one example ensemble (`neon-shadows-detective` in the library submodule), all 15 usages in that single file
- "Conversational" validation ensembles don't use the `conversation` key — they're regular dependency chains in a misleadingly named folder
- Only consumers: unit tests and one BDD test that exercise the executor in isolation

The regular execution path already supports user-directed flows via:
- `get_user_input.py` / `confirm_action.py` script primitives
- `ScriptUserInputHandler` — detects interactive ensembles at runtime
- `EnsembleExecutor.execute_with_user_input()` — handles input pausing/resuming
- Conditional `depends_on` (dict form with `condition`) — branching based on user responses

The `ConversationalEnsembleExecutor` duplicates this capability with worse integration.

**Decision:** Remove the conversational system.

**Files to remove:**
- `src/llm_orc/core/execution/conversational_ensemble_executor.py`
- `src/llm_orc/schemas/conversational_agent.py`
- `tests/unit/core/execution/test_conversational_ensemble_executor.py`
- `tests/bdd/test_adr_005_multi_turn_conversations.py`

**Files to keep:**
- User input primitives and `ScriptUserInputHandler` (wired into real execution)
- `neon-shadows-detective` ensemble (library submodule — `conversation:` keys are silently ignored by regular executor)

This cleanup removes the `conversation` field from the Pydantic migration scope and eliminates a confusing parallel system.

## Question 6: Complete agent field audit — what overrides are legitimate?

**Method:** Traced all field reads through `LlmAgentRunner`, `ModelFactory`, `ScriptAgentRunner`, `AgentDispatcher`, and profile configs

**Findings:**

### Profile keys (set on profiles, overridable at agent level via dict merge)

The `LlmAgentRunner._resolve_model_profile_to_config()` merges profile and agent config as `{**profile_config, **agent_config}` — so any profile key can technically be overridden. But not all overrides are intentional.

Profile keys in use: `model`, `provider`, `system_prompt`, `timeout_seconds`, `cost_per_token`, `fallback_model_profile`, `temperature`, `max_tokens`.

### Decisions on each field

**Keep on `LlmAgentConfig`:**
- `model_profile: str | None` — select a named profile (primary path)
- `model: str | None` — inline model ID for quick experimentation
- `provider: str | None` — required with inline `model`
- `system_prompt: str | None` — common override (151 agents use it)
- `temperature: float | None` — per-agent generation tuning
- `max_tokens: int | None` — per-agent generation tuning
- `timeout_seconds: int | None` — execution timeout
- `output_format: str | None` — output control (10 agents)
- `fallback_model_profile: str | None` — per-agent fallback chain

Validator: must have `model_profile` OR (`model` AND `provider`), not both paths, not neither.

**Drop:**
- `type` — redundant with Pydantic discriminated union. Remove from YAMLs that use it (4 agents in 2 files).
- `cost_per_token` — profile-only concern, no reason to override per-agent
- `synthesis_timeout_seconds` — dead field, no code reads it, single usage in one YAML
- `conversation` — dead code (see Question 5)

### Final Pydantic models

```python
class BaseAgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    depends_on: list[str] = []
    fan_out: bool = False

class LlmAgentConfig(BaseAgentConfig):
    model_profile: str | None = None
    model: str | None = None
    provider: str | None = None
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
```

`AgentConfig = LlmAgentConfig | ScriptAgentConfig | EnsembleAgentConfig`

Discriminator: presence of `model_profile` or `model` → LLM, `script` → Script, `ensemble` → Ensemble. Custom discriminator function needed since LLM detection checks two possible keys.
