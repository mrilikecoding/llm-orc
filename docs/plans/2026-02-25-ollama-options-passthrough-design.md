# Ollama Options Pass-Through

## Problem

OllamaModel only forwards `temperature` and `max_tokens` (as `num_predict`). Ollama's `/api/chat` accepts many more parameters (`num_ctx`, `top_k`, `top_p`, `repeat_penalty`, `seed`, etc.) that users can't configure.

## Design

Add a generic `options: dict[str, Any] | None` field that flows from profile YAML and/or agent config through to the Ollama provider.

### Profile YAML example

```yaml
name: analyst-qwen
provider: ollama
model: qwen3:8b
temperature: 0.6
max_tokens: 2000
options:
  num_ctx: 8192
  top_k: 20
  top_p: 0.8
```

### Merge precedence (lowest to highest)

```
profile.options -> agent.options -> explicit temperature/max_tokens
```

Profile and agent `options` dicts are deep-merged (`{**profile_options, **agent_options}`). Explicit `temperature` and `max_tokens` fields always win over anything in `options`.

### Touch points

1. **`LlmAgentConfig`** (`schemas/agent_config.py`) -- add `options: dict[str, Any] | None = None`
2. **`OllamaModel`** (`models/ollama.py`) -- accept `options` param, merge into Ollama options dict
3. **`ModelFactory.load_model_from_agent_config()`** (`core/models/model_factory.py`) -- extract and pass `options`
4. **`ModelFactory.load_model()`** (`core/models/model_factory.py`) -- thread `options` to OllamaModel
5. **`_resolve_model_profile_to_config()`** (`core/execution/runners/llm_runner.py`) -- deep-merge `options` dicts instead of last-wins replacement

### Backward compatibility

- Existing profiles/agents without `options` work unchanged
- Non-Ollama providers ignore the field
- `temperature` and `max_tokens` continue to work as top-level fields
