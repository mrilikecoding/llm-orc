# llama.cpp / llama-server as a local serving backend — tradeoffs vs Ollama

**Scope:** whether and how to serve llm-orc's local model tiers from
`llama-server` (llama.cpp, installed via Homebrew) instead of Ollama, and which
serving topology fits llm-orc's multi-model strategies. This is a
serving-backend choice, not an engine change: llm-orc already has the seam.

**Status:** as of 2026-07-01. Backend capability is evidence-backed (source
read). Topology recommendations are reasoned from the current tier profiles and
execution model; the "which wins on the 32GB rig" question is empirical and
**unmeasured** — see [What to measure](#what-to-measure).

---

## The one fact that drives everything

Ollama is *one endpoint that multiplexes many models*: it lazy-loads, keeps a
few resident under a memory budget, and evicts on idle. A single `llama-server`
process is the opposite: **one model, resident for the life of the process.**

llm-orc ensembles reference many model profiles, so you cannot swap Ollama's
single endpoint for a single llama-server. The real question is not "llama-server
yes/no," it is *which serving topology*, keyed to how many distinct models a
given ensemble actually touches.

---

## What llm-orc already supports (no code change needed)

| capability | mechanism | evidence |
|---|---|---|
| OpenAI-compatible client | `OpenAICompatibleModel` POSTs to `{base_url}/chat/completions` | `src/llm_orc/models/openai_compat.py:43-92` |
| Route a profile to it | `provider: openai-compatible` + `base_url:` on the profile | `core/models/model_factory.py:362-368`, base_url read at `:70` |
| No API key required | no-auth path builds the model straight from `base_url` | `core/models/model_factory.py:354-368` |
| Tool calling | `supports_tool_calling = True`; per-model, needs `--jinja` server-side | `models/openai_compat.py:19-23, 94-152` |
| Scoped providers | `openai-compatible/<name>` also matches (e.g. the Zen frontier profile) | `model_factory.py:255-259` |

`llama-server` speaks exactly this dialect (`/v1/chat/completions`, default port
8080). So a profile pointed at it is a config change, not a provider port.

---

## The models a real session touches

The tier profiles are the actual multi-model strategy. Local models in play:

| profile | model | ~Q4_K_M weights |
|---|---|---|
| `agentic-tier-cheap-summary` | `qwen3:1.7b` | ~1.4 GB |
| `agentic-tier-cheap-general` | `qwen3:8b` | ~5 GB |
| `agentic-tier-escalated-reasoning` | `deepseek-r1:8b` | ~5 GB |
| `agentic-tier-escalated-general` | `qwen3:14b` | ~9 GB |
| (`agentic-orchestrator-offline` reuses the 14b on purpose) | `qwen3:14b` | shared |

All four resident at once is ~20 GB of weights before KV caches. On the 32 GB
target rig that fits only with thin headroom, and once macOS starts compressing
memory, throughput collapses. **Memory is the master constraint**, and it is why
"just run llama-servers for everything" is not free.

---

## Three topologies, weighed

### A. One llama-server per model (a fleet)

Each tier profile gets its own port and `base_url`; every model stays hot.

- **For:** zero swap latency; genuinely-parallel fan-out *across* models
  (independent agents on different models run concurrently, not queued behind one
  runner).
- **Against:** you pay resident RAM for every model whether used or not. On 32 GB
  the full tier set leaves almost no headroom for KV caches + OS.
- **Fits:** an ensemble that fans out to a *small* model set you can afford to
  pin hot.

### B. `llama-swap` proxy in front (the Ollama-equivalent)

`llama-swap` presents one OpenAI-compatible endpoint, routes by the request's
`model` field, lazy-loads the target llama-server, unloads on a TTL, and can pin
a "group" co-resident.

- **For:** memory-bounded; single endpoint; near drop-in for the current Ollama
  setup. All tier profiles share one `base_url` and the `model:` field becomes
  the routing key again. The mental model transfers directly.
- **Against:** cold-model load stalls (seconds, worse for the 14b). Tier
  escalation crosses a model boundary (`cheap-general` 8b → `escalated-general`
  14b), so each escalation eats a load stall *unless both sit in the same
  resident group.*
- **Fits:** Ollama-like ergonomics with llama.cpp's control. Best default for the
  multi-tier setup.

### C. One llama-server, `--parallel N`, continuous batching (same-model fan-out)

Where llama.cpp genuinely beats Ollama. When multiple agents share a profile,
one server loads the weights *once* and continuous-batching interleaves the
concurrent decodes across slots.

- **For:** real concurrent throughput on shared weights. Maps directly to the
  dd-* spikes: `dd-seat-code-verified` runs `coder` **and** `reviewer` both on
  `agentic-tier-cheap-general` (qwen3:8b).
- **Against:** KV cache splits across slots, so per-slot context ≈
  `--ctx-size / --parallel`. You trade context depth for concurrency.
- **Fits:** any ensemble where several agents land on the *same* model.

### Summary

| | memory cost | cross-model parallelism | swap stalls | same-model throughput | ergonomics |
|---|---|---|---|---|---|
| **A. fleet** | highest (all resident) | best | none | needs `--parallel` too | N base_urls to manage |
| **B. llama-swap** | bounded (group) | limited to resident group | yes, off-group | needs `--parallel` too | ~Ollama, single endpoint |
| **C. one server** | lowest (one model) | none (single model) | none | best | trivial, one base_url |

These compose: run llama-swap (B) with `--parallel` on each underlying server
(C), and pin a fleet-like group (A) for the hot pair.

---

## Recommended shape for the 32 GB rig

A hybrid, because the workload is not one shape:

1. **`llama-swap` as the single endpoint** (B) so profiles keep pointing at one
   `base_url` and `model:` routes. Preserves the Ollama ergonomics the config
   already assumes.
2. **A co-resident group holding {qwen3:8b cheap, qwen3:14b escalated}** (~14 GB)
   so the tier-escalation hop does not stall. Let `qwen3:1.7b` summary and
   `deepseek-r1:8b` reasoning swap in on demand; both are used less and 1.7b
   loads fast.
3. **`--parallel` per server matched to same-model concurrency** (C) so
   shared-profile fan-out batches instead of serializing.
4. **Keep orchestrator and escalated-general on the same 14b.** Already done
   deliberately in `agentic-orchestrator-offline.yaml:5`; carry that into a
   llama-swap group so one resident 14b serves both roles.

### Server flags that matter, priority order

- `-ngl 999` — all layers on Metal; free on Apple unified memory.
- `--flash-attn` — cuts KV memory and speeds attention; buys headroom.
- Q4_K_M / Q5_K_M quant — the primary memory dial.
- continuous batching — default-on in recent builds; it is what makes
  `--parallel` worth anything.
- `--jinja` — real chat template + tool-call parsing; the agentic loop needs it.
- per-model TTL (via llama-swap) — long for the always-hot cheap tier, short for
  the rare reasoning tier.

---

## Gotchas that will bite specifically

- **Semaphore vs `--parallel` mismatch.** `max_concurrent_agents`
  (`ensemble_execution.py:359-366`) lets N agents fly, but if they land on a
  server with fewer `--parallel` slots they queue *at the server*:
  head-of-line blocking, and the concurrency is illusory. `--parallel` on a
  shared model must be ≥ the concurrent agents that hit it.
- **Escalation crosses a model boundary.** cheap→escalated is a different model,
  so under llama-swap it is a load stall unless both are group-resident.
- **Telemetry regression.** The `OpenAICompatibleModel` path harvests token
  counts from `usage` but drops Ollama's timing breakdown
  (`prompt_eval_count`, `eval_count`, `total_duration`/`eval_duration`/
  `load_duration`) that the benchmarks lean on (see `models/ollama.py:63-92`).
  llama-server *does* expose per-request `timings` and a `/metrics` endpoint,
  but the current openai_compat model does not read them. Recovering timing
  fidelity is a small code change to `OpenAICompatibleModel`, not just config.

---

## Decision rule

- Ensemble touches **one local model** → one llama-server + `--parallel`
  (topology C). Simplest, fastest, lowest memory.
- Ensemble **fans out across a small model set that fits hot** → fleet (A) or a
  pinned llama-swap group.
- Ensemble **spans the full tier ladder / uses dynamic dispatch** → llama-swap
  (B) with a resident group for the escalation pair.
- Need **Ollama's exact drop-in behavior with more control** → llama-swap (B) is
  the closest analog.

Ollama stays the low-ceremony default; reach for llama-server when you want
continuous-batching throughput on a shared model or finer control over
resident-set and KV memory.

---

## What to measure

Which topology wins on the 32 GB rig is empirical, not deducible. Cheapest
resolution: stand up llama-swap with the two-model group, point the existing
tier profiles at it, and run `dd-seat-code-verified` (two agents, same qwen3:8b)
both ways.

- wall-clock per turn (Ollama vs llama-swap)
- tokens/sec under 2 concurrent same-model agents (proves the `--parallel` win)
- escalation-hop latency with the 14b in-group vs cold
- peak resident memory vs the 32 GB ceiling

Note the telemetry regression above: for an apples-to-apples timing comparison
you may need the `OpenAICompatibleModel` timing patch first, or measure
wall-clock externally.

---

## Convenience note

Ollama itself also exposes an OpenAI-compatible endpoint at
`http://localhost:11434/v1`, so the same `openai-compatible` profile shape can
point at either backend by swapping only `base_url`. That makes the A/B above a
one-line profile change rather than two separate configs.
