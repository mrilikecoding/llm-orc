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
  per-seat routing decision (Ollama `think` param): easy turns run
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
| Serving ensemble + seats | `.llm-orc/ensembles/agentic-serving/` |
| Registry: Topaz-keyed parts, shape catalog, admission | `src/llm_orc/core/serving/` |
| Seat contracts / validation framework | `src/llm_orc/core/validation/` |
| Session substrate (registry, artifacts, compaction, plexus adapter) | `src/llm_orc/core/session/` |
| I/O envelope (inter-seat seam) | `src/llm_orc/models/dispatch_envelope.py` |
| Engine primitives (guard, loop, dynamic dispatch) | `src/llm_orc/core/execution/` |
| Turn trace (per-turn introspection) | `src/llm_orc/web/serving/turn_trace.py` → `.llm-orc/.serve-trace/turns.jsonl` |

## Decisions

The architectural decisions behind this design live in
[`docs/adrs/serving/`](adrs/serving/) — a separate numbering space from the
project-level ADRs in `docs/adrs/` (see that directory's README for the
namespace rule). Start with ADR-044 (the declarative-serving invariant),
ADR-046 (the target architecture and the orchestrator-actor dissolution),
ADR-047 (extensibility: registry + shape catalog), and ADR-048 (grounded
acceptance).

## Operator seat configuration

Seat models resolve through **tier profile names** (`agentic-tier-cheap-general`
and friends in `.llm-orc/profiles/`) — the tier name is the stable operator
surface; which model/provider backs it is deployment-specific. The shipped
defaults are all local (Ollama). To back any tier with your own provider —
a paid API, a hosted endpoint, a bigger local model — create a gitignored
override:

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

Build (accept-gated), explain, and within-session conversation memory are
implemented and grounded against a live endpoint and a literal
`opencode run` (multi-turn battery 2026-07-08: build → "did you see my
previous query?" → "add tests for it", all green). Fix and edit-existing for
files the conversation did not write are the named frontier — they need
client-file threading or client-delegated execution. Context older than the
render window is dropped until the lossless session record (design §Rung 2′)
lands.

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
