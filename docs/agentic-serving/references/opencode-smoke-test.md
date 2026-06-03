# $0 OpenCode Smoke Test — tool-driven parity round-trip

**Scope:** the north-star validation the harness tests cannot reach — a *real*
OpenCode session driving a *real* `llm-orc serve`, with every generation
delegated to an ensemble and the deliverable applied locally through OpenCode's
own `write` tool. The WP-LB-C integration test
(`tests/integration/test_serving_surface_tool_round_trip.py`) drives the real
ASGI surface but doubles the model and the client; this procedure closes the
last gap that harness design cannot reach — the real client.

**Why manual:** the round-trip needs a real OpenCode install plus a local
Ollama model. It is **$0** (local inference only — confirm before any paid step
per the free-options preference), but it cannot run in CI.

**What it does NOT validate:** axis-2 long-horizon coherence (sustained
multi-turn trajectories under the cheap driver). That is a separate
PLAY / first-deployment validation (ADR-033 §6b, ADR-097); this smoke test
confirms the single delegated write round-trip closes with parity.

---

## Prerequisites

- A local Ollama with a tool-calling model pulled (Spike π verified `qwen3:8b`;
  any local model whose profile reports `supports_tool_calling`).
- A real OpenCode install (Spike π Phase 0 observed OpenCode 1.15.5).
- `llm-orc` on the same machine as OpenCode (co-location is the assumed
  deployment; the deliverable still returns through the client's tool, not a
  server-side write — the justification is OpenCode's *execution model*, not
  filesystem geography).

## Setup

1. **A capability ensemble that generates code.** Configure a local capability
   ensemble (e.g. `code-generator`) whose agent resolves to the local
   tool-calling model. It carries an ADR-015 `topaz_skill` marker so the
   capability-list builder registers it.

2. **The seat-filler model profile.** Set the orchestrator's `model_profile`
   (the loop-driver's "model" seat, ADR-033 §Decision 5) to a local
   tool-calling profile. A profile that resolves to a non-tool-calling model
   surfaces a clear startup error rather than a silent first-turn failure.

3. **Start the serving layer:**

   ```
   llm-orc serve            # binds 127.0.0.1:8765 by default
   ```

   The OpenAI-compatible surface is then at `http://127.0.0.1:8765/v1`.

4. **Point OpenCode at the endpoint.** Configure OpenCode's provider base URL
   to `http://127.0.0.1:8765/v1` and select any model name (the surface-mode
   discriminator keys on the request carrying client `tools[]`, not on the
   model name). OpenCode's build agent sends `tool_choice: "auto"`,
   `stream: true`, and declares `write`/`edit`/`bash`/`read` — the tool-driven
   surface.

## Run

In an OpenCode session pointed at the endpoint, issue a task that should
delegate generation and apply the result locally, for example:

> Write a quicksort function to `quicksort.py`.

## Parity checklist (pass vs. fail)

The round-trip **passes** when OpenCode behaves as it would against a normal
single-model provider:

- [ ] OpenCode shows a **`write` tool call** (a `tool_use` event), not only an
      assistant text message.
- [ ] OpenCode's **permission gate / diff** appears for the write (the client is
      driving its own filesystem).
- [ ] `quicksort.py` lands on disk with the **ensemble's content** — the full
      generated code, not a summary line.
- [ ] OpenCode sends the **tool result back** (`role: "tool"`) and the next turn
      continues or finishes coherently.

**Failure signatures** (each maps to a specific regression):

- **Assistant text only, zero tool events, file may still appear** → the C8
  regression (text-only terminal / a server-side direct write behind OpenCode's
  back). This is exactly the Spike π Phase A failure: the bytes can land, but
  OpenCode executed nothing, so a multi-turn session degrades.
- **`write` emitted but `content` is a summary / empty** → the Artifact Bridge
  did not marshal full-fidelity substrate content (FC-49), or degraded to a
  dispatch-failure completion (unresolvable reference / binary deliverable).
- **The follow-up turn ignores the tool result / re-issues the same write** →
  the `role: "tool"` follow-up did not reach the loop driver (FC-50).

## Recording the result

Capture the OpenCode session trace (the tool events and the written file) under
a scratch directory and note the verdict. A passing run is the WP-LB-C
north-star confirmation; a failing run names which FC regressed and feeds back
to the corresponding module (terminal emission, bridge marshalling, or loop
participation).
