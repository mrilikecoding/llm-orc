# WP-C Manual Verification — Ollama

Quick-start for running the WP-C orchestrator end-to-end against a local Ollama
tool-calling model. Verifies the TS-1 vision (operator points an OpenAI-compat
client at `llm-orc serve` and gets real orchestrator responses) in a reproducible
local setup with no API keys.

## Prerequisites

1. **Ollama** — <https://ollama.com>. Version ≥ 0.3 for tool-calling support.
2. **Tool-calling model**, pulled locally. Known-good as of 2026-04:
   - `llama3.1:8b-instruct-q4_K_M` (4 GB, fast on Apple Silicon)
   - `qwen2.5:7b-instruct`
   - `mistral-nemo`
   One of these. Others may work; Ollama's model catalog tags tool-calling
   models as "Tools"-capable.
3. **At least one library ensemble** — anything the orchestrator can list and
   invoke. The existing `examples/` directory works.

## Setup

### 1. Start Ollama and pull the model

```bash
ollama serve &                          # runs http://localhost:11434
ollama pull llama3.1:8b-instruct-q4_K_M
```

### 2. Configure an orchestrator Model Profile

Add to `~/.config/llm-orc/config.yaml` (or project-local
`.llm-orc/config.yaml` — local takes precedence):

```yaml
model_profiles:
  orchestrator-local:
    model: llama3.1:8b-instruct-q4_K_M
    provider: openai_compat
    base_url: http://localhost:11434/v1
    # No api_key needed — Ollama accepts anonymous requests.

agentic_serving:
  orchestrator:
    model_profile: orchestrator-local
  budget:
    turn_limit: 20
    token_limit: 50000
```

### 3. Start the llm-orc server

```bash
uv run llm-orc serve --port 8000
```

## Verify

### Check /v1/models

```bash
curl -s http://localhost:8000/v1/models | jq
```

Expected: a `data` array containing `orchestrator-local` (or whatever profile
names you listed in `agentic_serving.orchestrator.allowed_profiles`).

### Simple orchestrator request (non-streaming)

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "orchestrator-local",
    "messages": [
      {"role": "user", "content": "What ensembles are available in my library?"}
    ]
  }' | jq
```

Expected: an OpenAI-shaped response body whose `choices[0].message.content`
describes your library. The model should have called `list_ensembles` internally;
the final content will paraphrase the result. `finish_reason: "stop"`.

### Streaming request

```bash
curl -N -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "orchestrator-local",
    "messages": [
      {"role": "user", "content": "List ensembles, then pick one and invoke it."}
    ],
    "stream": true
  }'
```

Expected: SSE frames — role-delta opener, content deltas interleaved with the
orchestrator's reasoning, a `stop` completion, then `data: [DONE]`.

### Budget exhaustion

Temporarily drop the Budget to force termination:

```yaml
agentic_serving:
  budget:
    turn_limit: 1
    token_limit: 100
```

The next request terminates after one iteration with `finish_reason: "length"`
and a content payload like `[Session budget exhausted: turn limit reached (1/1)]`.

### Point an agentic coding tool at it

OpenCode, Roo Code, Cline, etc. configured with:

- **Endpoint:** `http://localhost:8000/v1`
- **Model name:** `orchestrator-local` (whatever's in `/v1/models`)
- **API key:** any non-empty string (the server ignores it today)

The tool's client-declared tools (bash, file_edit) round-trip via the
Option C turn-boundary delegation path once WP-F lands. Until then,
the orchestrator can list/invoke ensembles but not delegate client
actions — the scenario is constrained to ensemble routing.

## What's covered

WP-C's acceptance scenarios exercise in this setup:

- *Tool user completes a task against the stateless orchestrator*
- *Session terminates gracefully on turn limit exhaustion*
- *Session terminates gracefully on token limit exhaustion*
- *Orchestrator tool surface is exactly the committed set* (LLM only ever
  emits one of the five tool names; any other returns an error observation)
- *Invocation outside the tool set is rejected*

## Troubleshooting

- **`ModelProfileNotFoundError` at request time** — the
  `agentic_serving.orchestrator.model_profile` name does not resolve
  to a Model Profile. Check `/v1/models` first.
- **`ToolCallingNotSupportedError`** — the resolved profile's provider
  is not `openai_compat`. Anthropic-native, Google/Gemini native, and
  Ollama's legacy `/api/chat` endpoint are not yet supported; those
  land in follow-up work. Use the `openai_compat` provider pointing at
  Ollama's `/v1` or any OpenAI-compat endpoint.
- **Model emits text but no tool calls** — the model isn't using the
  tool interface. Try a more capable tool-calling model (`qwen2.5:7b`,
  `llama3.1:8b` or larger). Smaller models sometimes ignore the tool
  schema.
- **Empty `/v1/models` response** — no Model Profiles configured, or
  the allowlist filters them all out. See WP-B Group 3 behavior: the
  endpoint is a shop window; session-start is where missing profiles
  raise.
