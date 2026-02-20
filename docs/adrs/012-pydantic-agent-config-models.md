# ADR-012: Pydantic Agent Config Models

**Status:** Accepted

**Date:** 2026-02-20

**Extends:** ADR-001

---

## Context

Agent configurations are currently `list[dict[str, Any]]`. Type determination happens at runtime through key inspection in `AgentDispatcher._determine_agent_type()`. There is no load-time validation that an agent has the right fields for its type — a typo in `model_profile` silently produces an agent that fails at execution time.

Pydantic is already used in the project for script agent I/O schemas (ADR-001), validation config models, and execution results. Agent configs are the gap.

A survey of 63 ensemble files containing 250 agent configurations identified 10 meaningful fields. No agent uses both `model_profile` and `script`. The data fits a clean discriminated union.

Adding a third agent type (ensemble agents, ADR-013) to the untyped dict system would make the type-dispatch logic fragile. The right sequence: migrate agent configs to Pydantic first (structural change, no behavior change), then add the ensemble agent type (behavior change on clean structure).

---

## Decision

Introduce Pydantic agent config models as a discriminated union. The agent discriminator determines agent type from config based on key presence: `model_profile` or `model` indicates LLM, `script` indicates Script. A custom discriminator function handles the two-key LLM detection.

### Models

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

class ScriptAgentConfig(BaseAgentConfig):
    script: str
    parameters: dict[str, Any] = {}

AgentConfig = LlmAgentConfig | ScriptAgentConfig
```

### Invariants enforced

- **Mutual exclusivity** (Invariant 1): An agent config must contain exactly one of `model_profile`/`model`, `script`, or `ensemble` (ADR-013). The discriminated union enforces this at parse time.
- **Inline model completeness** (Invariant 2): If `model` is specified, `provider` must also be specified. A validator on `LlmAgentConfig` enforces this.
- **Profile XOR inline** (Invariant 3): An LLM agent must specify either `model_profile` or `model` + `provider`, not both. A validator enforces this.
- **Extra fields forbidden** (Invariant 11): `extra="forbid"` catches typos and invalid fields at parse time.

### Fields dropped

- `type` — redundant with key-based discrimination. Used on 4 agents in 2 YAML files; remove from those files.
- `conversation` — dead code, removed per ADR-011.
- `synthesis_timeout_seconds` — one agent used it, no code reads it. Remove from that YAML.

### Migration scope

`EnsembleConfig.agents` changes from `list[dict[str, Any]]` to `list[AgentConfig]`. `EnsembleLoader.load_from_file()` parses agents through Pydantic. `AgentDispatcher._determine_agent_type()` is replaced by `isinstance` checks. `DependencyAnalyzer`, `DependencyResolver`, `FanOutCoordinator`, `LlmAgentRunner`, `ScriptAgentRunner`, and all tests that construct agent config dicts are updated.

The blast radius is wide but shallow — many files change from `dict[str, Any]` to typed models, but each change is mechanical type narrowing. No logic changes.

---

## Consequences

**Positive:**
- Load-time validation catches typos and invalid configs before execution
- `isinstance` checks replace fragile key inspection for type dispatch
- Clean foundation for adding ensemble agent type (ADR-013)
- Agent-level overrides win over profile defaults is now enforceable via model structure

**Negative:**
- Wide blast radius across the codebase. Every file that reads agent config dicts needs updating.
- Existing ensemble YAML files with `type:`, `conversation:`, or `synthesis_timeout_seconds:` fields will fail validation until cleaned up.

**Neutral:**
- This is a structural change — all existing tests must pass with no behavior change. The commit type is `refactor:`.
