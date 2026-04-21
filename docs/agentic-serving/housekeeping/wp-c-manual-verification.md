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
    provider: openai-compatible
    base_url: http://localhost:11434/v1
    # No api_key needed — Ollama accepts anonymous requests.

agentic_serving:
  orchestrator:
    model_profile: orchestrator-local
  budget:
    turn_limit: 20
    token_limit: 50000
```

The canonical provider key is `openai-compatible` (hyphenated); see
`src/llm_orc/providers/registry.py`. Other spellings silently fall
through the factory.

### 3. Start the llm-orc server

```bash
uv run llm-orc serve --port 8000
```

`llm-orc serve` and `llm-orc web` mount the same FastAPI app — both
expose the OpenAI-compatible `/v1/...` surface plus the ensemble
management REST API. Use `serve` for agentic-client deployments and
`web` when you also want the browser UI. (`llm-orc mcp serve` is
unrelated — that one starts the MCP server for direct tool use.)

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

The next request terminates with `finish_reason: "length"` and a content
payload like `[Session budget exhausted: turn limit reached (2/1)]`.

The counter shows **cumulative session turn count**, not per-request.
Sessions persist across HTTP requests that share identity — same
`user` field, or (when `user` is absent) same first-user-message hash.
If you re-run the previous curl verbatim after lowering the turn
limit, the existing Session's turn_count is already > 1, so the first
budget check on the next iteration terminates immediately and the
counter reports the prior count.

To exercise a fresh session, pass a unique `user` field:

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "orchestrator-local",
    "user": "budget-test-'"$(date +%s)"'",
    "messages": [{"role": "user", "content": "any request"}]
  }' | jq
```

A fresh session yields `(1/1)` — the first iteration completes the
LLM call (turn_count → 1), then the check at the top of iteration 2
sees `turn_count >= turn_limit` and terminates.

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
  is not `openai-compatible`. Anthropic-native, Google/Gemini native,
  and Ollama's legacy `/api/chat` endpoint are not yet supported;
  those land in follow-up work. Use the `openai-compatible` provider
  pointing at Ollama's `/v1` or any OpenAI-compat endpoint.
- **Model emits text but no tool calls** — the model isn't using the
  tool interface. Try a more capable tool-calling model (`qwen2.5:7b`,
  `llama3.1:8b` or larger). Smaller models sometimes ignore the tool
  schema.
- **Empty `/v1/models` response** — no Model Profiles configured, or
  the allowlist filters them all out. See WP-B Group 3 behavior: the
  endpoint is a shop window; session-start is where missing profiles
  raise.

## Verification run — 2026-04-21

A live run against local Ollama with `mistral-nemo:12b` passed all
four acceptance checks (models list, non-streaming completion,
streaming SSE, budget exhaustion). Known-good models confirmed:

- `mistral-nemo:12b` (verified 2026-04-21)
- `llama3.1:8b-instruct-q4_K_M` (listed above)
- `qwen2.5:7b-instruct` (listed above)

The run surfaced three gaps that were addressed in follow-up commits:

1. **`llm-orc serve` did not exist.** Added as a command alias for
   `llm-orc web` — both start the same FastAPI app. The
   agentic-serving framing is the natural name for the deployment
   context; `web` remains for "I want the browser UI" use.
2. **`openai_compat` provider key was wrong.** Corrected to
   `openai-compatible` (the canonical registry key) throughout this
   guide.
3. **HTTP read timeout hardcoded at 30 s.** Local tool-calling models
   (`mistral-nemo:12b`, `qwen2.5:7b`, etc.) take 30–80 s per iteration,
   tripping `httpx.ReadTimeout`. Fixed by wiring the timeout through
   the performance config and raising the default read timeout.
   Operators deploying against slower models can tune further via
   `performance.concurrency.request_timeout.read`.

## Re-verification run — 2026-04-21 (post-fix)

Followed this guide verbatim after the three gaps above were
addressed. All four acceptance checks green against local Ollama with
`mistral-nemo:12b`:

| Check | Result | Wall clock |
|---|---|---|
| `uv run llm-orc serve --port 8000` starts | ✓ (new alias mounts same app) | — |
| `GET /v1/models` → `orchestrator-local` | ✓ | <1s |
| Non-streaming completion, `finish_reason: "stop"` | ✓ (65 ensembles rendered) | 2m49s |
| Streaming SSE: role-delta → content → stop → `[DONE]` | ✓ | 22s |
| `turn_limit: 1` → `finish_reason: "length"` + budget-exhausted content | ✓ | <1s |

Notes on what changed since the first run:

- `llm-orc serve` now has its own help text framing it for the
  agentic-serving deployment context — no longer falls through to
  "No such command."
- `provider: openai-compatible` resolves via the factory
  (`src/llm_orc/providers/registry.py:77`) as intended.
- Default read timeout is now 180 s, wired through
  `performance.concurrency.request_timeout.read`. First non-streaming
  call no longer 500s out with `httpx.ReadTimeout`.
- Stateless config resolution confirmed: lowering `turn_limit` to 1
  took effect on the next request without a server restart.

### Observation resolved — counter is cumulative per session

The re-run's budget-exhausted payload read `(2/1)` rather than the
`(1/1)` a naive reading of the example implied. This is not a bug:
the counter reports cumulative session turn_count, and Sessions
persist across HTTP requests that share identity (either a matching
`user` field or, when `user` is absent, a matching hash of the first
user message). The re-run reused the same first-user-message across
requests, so the Session's turn_count had already incremented during
earlier verification steps before the Budget-dropped request landed.

The §Budget exhaustion section now documents this explicitly and
provides a `"user": "budget-test-$(date +%s)"` recipe for exercising
a fresh session when the `(1/1)` case is desired.
