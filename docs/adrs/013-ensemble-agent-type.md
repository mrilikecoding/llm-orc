# ADR-013: Ensemble Agent Type

**Status:** Accepted

**Date:** 2026-02-20

**Depends on:** ADR-012

---

## Context

llm-orc orchestrates LLM agents and script agents within a single ensemble. But it cannot compose ensembles together. A code analysis pipeline can't chain a security-scanning ensemble into a quality-review ensemble. A document processing workflow can't route PDFs through one extraction ensemble and audio files through another.

Issue #29 proposed a meta-ensemble YAML format with its own execution strategies and a dedicated executor. That introduces a parallel system. The existing ensemble format already handles agent coordination well — the natural extension is to let an agent reference another ensemble.

The Pydantic agent config migration (ADR-012) provides the structural foundation: adding a third variant to the discriminated union is straightforward.

---

## Decision

Add `EnsembleAgentConfig` as a third variant in the agent config discriminated union, and `EnsembleAgentRunner` to handle execution.

### Config

```python
class EnsembleAgentConfig(BaseAgentConfig):
    ensemble: str  # ensemble reference — static name, resolved at load time

AgentConfig = LlmAgentConfig | ScriptAgentConfig | EnsembleAgentConfig
```

An agent with an `ensemble` key is an ensemble agent. It participates in dependency chains, fan-out, and parallel phases exactly like any other agent.

### Execution

The `EnsembleAgentRunner` is a thin adapter:
1. Resolve the ensemble reference to an ensemble config via `EnsembleLoader.find_ensemble()`
2. Create a child executor that shares the parent's immutable infrastructure (config manager, credential storage, model factory) but isolates mutable state (usage collector, event queue, streaming tracker) — per Invariant 10
3. Execute the child ensemble with artifact saving disabled — child results nest within the parent artifact per Invariant 9
4. Return the child's full result dict as the ensemble agent's response (JSON-serialized)

### Agent dispatcher

`AgentDispatcher` routes `EnsembleAgentConfig` to `EnsembleAgentRunner`, extending the existing `isinstance`-based dispatch from ADR-012.

### Cycle detection

Cross-ensemble cycles (A references B, B references A) are caught at load time per Invariant 5. The `EnsembleLoader` builds the ensemble reference graph — a directed graph of ensemble-to-ensemble references from all `ensemble:` keys — and runs DFS cycle detection. This extends the existing intra-ensemble cycle detection.

Ensemble references are static string literals resolved at load time per Invariant 7. No template expressions or dynamic resolution.

### Depth limiting

Recursive ensemble execution is bounded per Invariant 8. A `depth` counter passes through each recursive invocation. The top-level ensemble is depth 0; each ensemble agent increments depth by 1. Exceeding the configurable depth limit (in performance config) produces a clear error.

### Failure handling

A child ensemble failure is an agent failure, not an orchestration failure, per Invariant 13. The parent ensemble continues executing agents that do not depend on the failed ensemble agent. The failure is recorded in the parent's results with a `has_errors` flag.

---

## Consequences

**Positive:**
- Ensemble composition without a new format. The ensemble agent participates in all existing machinery (dependencies, phases, fan-out).
- Static cycle detection prevents reference loops at load time.
- Depth limiting prevents unbounded nesting.
- Recursive execution reuses the existing `EnsembleExecutor` — no parallel execution system.

**Negative:**
- Full result dict as response may be verbose for downstream LLM agents. Mitigated by `input_key` (ADR-014) for consumer-side selection. If that proves insufficient, an `output_mode` field can be added later.
- Depth limiting is a system-level constraint, not per-ensemble. All ensembles share the same limit.

**Neutral:**
- The `EnsembleAgentRunner` is thin. Most complexity is in the structural Pydantic migration (ADR-012), not in the runner itself.
- MCP server tools that call `EnsembleExecutor` (`invoke`, `validate`) should work without changes to their interfaces — ensemble agents are handled internally by the executor.
