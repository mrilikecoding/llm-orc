# ADR-014: Input Key for Selective Upstream Consumption

**Status:** Accepted

**Date:** 2026-02-20

---

## Context

When agents depend on an upstream agent, they receive the upstream's full output. For routing patterns — where a classifier script produces keyed output like `{pdfs: [...], audio: [...]}` and different downstream agents need different keys — there is no mechanism for a consumer to select a specific key from upstream output.

This is the third orthogonal feature for composable ensemble orchestration. The ensemble agent type (ADR-013) enables composition. Fan-out enables per-item expansion. Input key enables routing — selecting which slice of upstream output a downstream agent consumes.

---

## Decision

Add `input_key` to `BaseAgentConfig`. The `DependencyResolver` selects the specified key from the first upstream agent's output before passing it as input to the consuming agent.

### Config

```python
class BaseAgentConfig(BaseModel):
    name: str
    depends_on: list[str] = []
    fan_out: bool = False
    input_key: str | None = None  # new
```

### Behavior

When `input_key` is set on an agent:
1. The `DependencyResolver` takes the output of the first entry in the agent's `depends_on` list (order matters)
2. If the output is a dict (or JSON-parseable to a dict), the resolver selects `output[input_key]`
3. The selected value becomes the agent's input
4. If the key does not exist in the upstream output, the agent receives an error status — per Invariant 14, this is a runtime error, not a structural one
5. If the upstream output is not dict-shaped (e.g., a plain string), the agent receives an error status — `input_key` requires structured output from its upstream dependency

When `input_key` is not set, behavior is unchanged — the agent receives the full upstream output.

### Interaction with fan-out

Input key and fan-out compose naturally. When both are set:
1. `input_key` selects the value from upstream output
2. If the selected value is an array, fan-out expands the agent into N instances (one per item)

This is the routing pattern: a classifier produces `{pdfs: [a, b], audio: [c]}`, an ensemble agent selects `pdfs` via `input_key`, and fan-out expands to process each PDF.

### Applicability

Input key works on all agent types — LLM, script, and ensemble. It is a consumer-side concern, independent of the producer agent's type.

---

## Consequences

**Positive:**
- Enables the routing pattern (classify → select → fan-out) that motivates composable ensemble orchestration
- Consumer-side selection — one mechanism instead of separate producer-side filtering and consumer-side filtering
- Works with all agent types, not just ensemble agents
- Orthogonal to ensemble agents and fan-out; independently useful

**Negative:**
- Selects from the first dependency only. If an agent depends on multiple upstream agents and needs keys from different ones, this mechanism doesn't cover it. Sufficient for the routing use case; can be extended later if needed.
- A missing key is a runtime error, not a load-time error. The loader cannot validate key existence because output shapes are determined at execution time.

**Neutral:**
- The change is small — `input_key` field on `BaseAgentConfig`, selection logic in `DependencyResolver`. The commit type is `feat:`.
