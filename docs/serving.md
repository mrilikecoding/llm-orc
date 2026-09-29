# Agentic Serving

llm-orc serves as the backend for agentic coding tools (OpenCode, Aider, Cline,
Cursor — any OpenAI-compatible client) via `llm-orc serve`, which exposes
`/v1/models` and `/v1/chat/completions`.

Every request to `/v1/chat/completions` is handled by **one declarative
Serving Ensemble** — classify (route the turn) → seat (dispatch to the resolved
capability) → marshal (shape the deliverable, gate its validity, emit it) —
executed by the same L0 Ensemble Engine that runs any llm-orc ensemble, on
primitives that ship with the engine (guard/branch, bounded loop, dynamic
dispatch). There is no persistent internal orchestrator: the client owns the
multi-turn agentic loop, and each request is a single declarative pass.

The north star is **full model parity through composition** — the endpoint
should do everything a single model does behind a coding tool (explain, fix,
edit, run tests, build) with no capability loss, and composing ensembles should
widen what's possible, never narrow it.

## The per-turn pipeline

Defined in `.llm-orc/ensembles/agentic-serving/serving.yaml`; scripts in
`.llm-orc/scripts/agentic_serving/`.

| Node | Kind | Responsibility |
|------|------|----------------|
| `classify` | script | Routing decision `{target, kind, file, dispatch_input, build}`. Deterministic where the signal is structural (explain markers, build verbs, named files); emits `needs_decider` when not. |
| `decide` | model (guarded) | Runs only on the ambiguous path (`when: ${classify.needs_decider}`). Picks a target from a closed seat set. The model classifies; the control stays deterministic. |
| `resolve` | script | Merges the structural and model-backed decisions into the final routing. |
| `seat` | dynamic dispatch | Resolves `${resolve.target}` at the phase layer and runs the resolved capability ensemble as a child, passing it the clean turn. |
| `seat_contract` | script | Admits or rejects the seat's output against the resolved seat's own `seat_contract:` block via a wired `ValidationEvaluator` (seat-owned, black-box, deterministic-first). |
| `shape` | script | Reads the deliverable faithfully from the seat's I/O envelope (ADR-024) — content from the envelope, destination from the routing decision. |
| `form_gate` | script | Deterministic destination-validity check: refuses a deliverable that does not parse as its path claims (a `.py` must parse, a `.json` must load). |
| `emit` | script | Shapes the client-seam outcome: a `write` tool_call (`finish_reason: tool_calls`) for a valid build, a prose finish otherwise. |

`resolve`, `shape`, `form_gate`, and `emit` are each marked
`on_dependency_failure: run` (Invariant 13/14): a crashed `classify`, `seat`,
or any node earlier in this chain would otherwise cascade-skip everything
downstream, and the turn would lose its refusal along with the failure it
exists to report. Each reads its own crashed dependency defensively and
composes an honest `"Refused: serving pipeline error: ..."` instead of
vanishing. The top-level `status`/`has_errors` a client or operator reads
still say `error` — the CAUSE is `has_errors` (the crashed `classify`/`seat`
node is genuinely `failed`, and that alone makes `has_errors` true regardless
of what runs after it), not the `handled_failure` marking on the node that
ran to compose the refusal. `handled_failure` is a separate fact about that
node's own `outcome` (addendum 2026-09-23): it keeps the composed refusal
from being misread as a *succeeded* terminal by anything that consults it —
without it, a wrapping node could report success just because the handler
ran without incident, masking the real upstream failure. So the client's
HTTP response body carries real refusal content, while the top-level status
still honestly says `error`: two separate reads of the same underlying
failure, not one causing the other. See `docs/domain-model.md` Invariant 13
and the `on_dependency_failure` glossary entry for the general contract;
this is serving's own instance of it.

A `loop:` node's `has_errors` (round-4 amendment, 2026-09-23) reflects only
its FINAL iteration: earlier iterations are retry attempts the loop exists
to absorb, so a build-gated round that fails once and passes on retry must
not report `error` — the loop's own outcome is `succeeded`, has_errors is
`false`. Earlier iterations' blocking outcomes are never dropped, though:
each one lands in the loop's JSON response as `iteration_failures`, a list
of `{iteration, error}` entries naming which terminal failed and why. An
operator reading `turns.jsonl` sees both facts — the turn succeeded, and it
took more than one try.

Build turns route to the **gated build shape** (`build-gated.yaml`): test-writer
→ code-writer → deterministic executor (runs code + tests, sandboxed
subprocess) → isolated adequacy judge → accept gate (`accept = tests_pass AND
tests_adequate`). The two signals catch orthogonal failures — the executor
catches wrong code, the judge catches trivial or under-covering tests; neither
alone suffices. On reject, the client owns the retry loop (the response says
another round is needed and writes nothing).

## Key constraints and invariants

- **AS-11 — declarative-ensemble-native; extend the engine, never a parallel
  layer.** Control flow lives in ensemble DAGs and engine primitives, not
  bespoke Python. Where the engine is inadequate, add a primitive
  (guard/loop/dispatch were added this way), never a driver beside the engine.
- **AS-2 — validate-before-load is the registry's single admission gate.**
  Every capability part and composition shape passes a shared reference-graph
  check (no cycle, within depth, resolves) before it becomes dispatchable. One
  shared routine (`core/validation/composition_validator.py`).
- **The client owns the loop.** No internal ReAct loop, no runtime
  self-composition, no trust-promotion machinery. Cross-turn state, where it is
  needed, belongs in the session substrate (`core/session/`), not in a
  resident orchestrator actor.
- **Grounded acceptance is composed verification, independent of the
  builder.** A build deliverable is accepted only when the deterministic
  executor and the isolated judge both pass. The builder never grades itself.
- **Interactive latency on the 32GB rig is first-class.** Thinking-mode is a
  per-seat routing decision (`options.think`, sent to llama-server as
  `chat_template_kwargs.enable_thinking`): easy turns run
  thinking-off (~seconds), hard turns may route thinking-on.
- **Determinism over carve-outs.** Essential control (termination, routing,
  admission) is deterministic; model judgment is confined to guarded,
  closed-set decisions whose blast radius the deterministic surround bounds.

## Where things live

| Concern | Code |
|---------|------|
| HTTP endpoint / OpenAI compat | `src/llm_orc/web/api/v1_chat_completions.py`, `v1_models.py`, `sse_format.py` |
| Per-turn caller (endpoint → ensemble) | `src/llm_orc/web/serving/serving_ensemble_caller.py` |
| Chunk vocabulary / session-start contract | `src/llm_orc/web/serving/chunks.py`, `session_start.py` |
| Serving ensemble + seats | `.llm-orc/ensembles/agentic-serving/` (shipped in the wheel as `llm_orc/serving_project/`) |
| Registry: Topaz-keyed parts, shape catalog, admission | `src/llm_orc/core/serving/` |
| Seat contracts / validation framework | `src/llm_orc/core/validation/` |
| Session substrate (registry, artifacts, compaction, plexus adapter) | `src/llm_orc/core/session/` |
| I/O envelope (inter-seat seam) | `src/llm_orc/models/dispatch_envelope.py` |
| Engine primitives (guard, loop, dynamic dispatch) | `src/llm_orc/core/execution/` |
| Turn trace (per-turn introspection) | `src/llm_orc/web/serving/turn_trace.py` → `<state dir>/.serve-trace/turns.jsonl` |

## Decisions

The architectural decisions behind this design live in
[`docs/adrs/serving/`](adrs/serving/) — a separate numbering space from the
project-level ADRs in `docs/adrs/` (see that directory's README for the
namespace rule). Start with ADR-044 (the declarative-serving invariant),
ADR-046 (the target architecture and the orchestrator-actor dissolution),
ADR-047 (extensibility: registry + shape catalog), and ADR-048 (grounded
acceptance).

## Layers and state

Configuration is read from four tiers, highest precedence first:

| Tier | Where | Written by |
|------|-------|------------|
| project | `<checkout>/.llm-orc/` (discovered walking up from cwd) | CRUD with `scope: project` (the default), the orchestrator's composition writes |
| library | `LLM_ORC_LIBRARY_PATH`, else `<checkout>/llm-orchestra-library/` | nothing (templates) |
| global | `$XDG_CONFIG_HOME/llm-orc/`, default `~/.config/llm-orc/` | CRUD with `scope: global`, operator `*.local.yaml` overrides |
| packaged | `llm_orc/serving_project/` in the wheel; this repo's `.llm-orc/` in a checkout | nothing (read-only; `brew upgrade` changes it) |

Ensembles, profile listings and scripts merge all four (first match
wins). Runtime model profiles merge packaged → global → project (the
library is listed, never resolved). The serving ensemble, its
serve-owned scripts and the `serving:` config keys are read from the
**serving root**: the project's `.llm-orc/` when it carries
`ensembles/agentic-serving/serving.yaml`, else the packaged project. So a
serve started in an empty directory runs the shipped serving ensemble,
and a checkout of this repo shadows it. `LLM_ORC_SERVING_PROJECT_DIR`
points the packaged tier at another directory (empty disables it).

Runtime state (artifacts, the turn trace, the script cache, the rendered
router preset) is written to one **state dir**: `LLM_ORC_STATE_DIR` or
`llm-orc serve --state-dir`, else the project's `.llm-orc/` when there is
one, else `$XDG_STATE_HOME/llm-orc/` (default `~/.local/state/llm-orc/`).
Nothing is ever written under the packaged tier.

## Operator seat configuration

Seat models resolve through **tier profile names** (`agentic-tier-cheap-general`
and friends in `.llm-orc/profiles/`) — the tier name is the stable operator
surface; which model/provider backs it is deployment-specific. The shipped
defaults are all local (`provider: llama-server`). To back any tier with your
own provider — a paid API, a hosted endpoint, a bigger local model — create a
gitignored override:

```yaml
# .llm-orc/profiles/my-paid-seat.local.yaml   (never committed)
name: agentic-tier-cheap-general   # the tier name to override
model: your-hosted-model
provider: openai-compatible/yourprovider
cost_per_token: 0.0
```

`*.local.yaml` files load last (deterministically), so they win over the
checked-in profile of the same name. Nothing provider-specific belongs in
tracked config. Empirical note (2026-07-08 A/B): a hosted frontier seat did
not change the dominant failure class — reach for structure (retry rounds,
shapes) before bigger models.

## Local inference: the serve owns the router

`llm-orc serve` starts and supervises one `llama-server` process in router
mode (llama.cpp; the binary is the only external runtime dependency). At
start the serve renders `llama-server.ini` into the state dir from every
`provider: llama-server` profile: one section per distinct `model:` name,
its GGUF source from the profile's `hf_repo:` (fetched from Hugging Face on
first load), a 40960-token context (the window the truncation backstop
assumes; `options.num_ctx` overrides per model), and one resident model at
a time (`--models-max`). The router loads a model on the first request that
names it and evicts least-recently-used. Model names route exactly and must
not contain a colon (the router rewrites `name:tag`); the renderer refuses
one.

`LLAMA_SERVER_URL` (default `http://127.0.0.1:8080/v1`) is exported for the
life of the serve so the model factory reaches the router; `--no-backend`
skips ownership and uses whatever is already at that URL. `GET /api/models`
lists what the router serves with load status; `POST /api/models/{name}/pull`
loads (and downloads) one. Per-request thinking control is
`chat_template_kwargs.enable_thinking`; llama-server's `timings` land on the
usage record as `prompt_eval_count` / `eval_count`.

An embedding seat is a `provider: llama-server` profile whose `options` set
`embeddings: true` and a `pooling` mode (`local-nomic-embed-text.yaml` backs
`nomic-embed-text-v1.5`); the renderer passes both through to the model's
preset section, allowlisted the same way `num_ctx` is. `POST /v1/embeddings`
forwards an OpenAI-shaped request to the router's own `/v1/embeddings`:
`model` may be a llama-server profile id (resolved to its served model name)
or the router model name directly; `input` (string or list of strings),
`encoding_format`, and `dimensions` forward as given. A model that isn't in
the rendered preset 404s without reaching the router; a router that's down
503s.

## Conversation memory

The serve threads conversation context from the client-sent history into
generation seats (bounded render: last 8 user/assistant turns, ~4KB;
written-file bodies included so referents like "add tests for it" resolve).
classify composes it behind the deterministic `Current request:` marker;
routing and verifier seats read the clean latest turn only (the accept-gate
judge is input-scoped to its dependencies). Conversation-written files
materialize into the accept-gate sandbox, so follow-up builds can import the
modules the conversation created. Design and the scaling ladder (lossless
session record, plexus lenses):
`docs/plans/2026-07-08-serving-conversation-memory-design.md`.

## Current capability coverage

Build (accept-gated), explain, within-session conversation memory,
**client-file reads**, and **client-delegated test runs** are implemented
and grounded against a live endpoint and a literal `opencode run`
(multi-turn battery 2026-07-08: build → "did you see my previous query?"
→ "add tests for it", all green; existing-file battery 2026-07-09:
"write tests for existing storage.py" → read tool_call → gated test
deliverable running green against the real module, and an honest
one-round refusal when the read fails; run battery 2026-07-09: "run
test_calc.py" → bash tool_call → client-executed pytest → honest verdict,
green and red both). A turn naming a file the serve can't see delegates a
`read` through the permission seam
(`docs/plans/2026-07-09-client-file-reads-design.md`); the result
materializes into the gate sandbox and stays retrievable in later turns.
A run turn delegates one closed-template `pytest` command through the
same seam and replies with a deterministic verdict parsed from pytest's
own summary — zero model calls end to end
(`docs/plans/2026-07-09-client-run-delegation-design.md`). Remaining
frontier on the execution surface: discovery of files the turn doesn't
name, and chaining write → run inside one fix turn. Context older than
the render window is dropped until the lossless session record (design
§Rung 2′) lands.

## Roadmap

The staged path to the north star, with per-stage exit gates and the
ladder-based parity measurement: [`docs/serving-roadmap.md`](serving-roadmap.md).

## History

This design is the product of an 8-cycle research process (2026-04 → 2026-07).
The full research corpus — essays, research logs, spike records, the complete
ADR set including superseded decisions, scenarios, field notes, and audits —
is preserved on the `research/agentic-serving-corpus` branch under
`docs/agentic-serving/`. Any reference to a `docs/agentic-serving/...` path or
an ADR number not present in `docs/adrs/serving/` resolves there.
