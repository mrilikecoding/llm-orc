# Spike — dynamic-dispatch serve skeleton

The remaining spike arm: assemble `classify -> seat -> marshal` as ONE
declarative ensemble on the shipped dynamic-dispatch primitive, and validate
swap-ability (#2) and a non-build turn (#3) with real local models.

## What it is

`dd-serve-skeleton` (`.llm-orc/ensembles/dd-serve-skeleton.yaml`) is the whole
serve turn as one ensemble:

- **classify** (`scripts/spike-dynamic-dispatch/classify.py`) — the decider.
  Emits `{target, kind, file, dispatch_input}`. Deterministic routing: a `seat`
  override wins; an explain-shaped turn routes to the explain seat; else the
  solo code seat.
- **seat** — `dispatch: "${classify.target}"`, the dynamic-dispatch node. The
  target ensemble name is resolved at the phase layer (guard-sibling) and run as
  a child. `input_key: dispatch_input` feeds it the classify payload.
- **marshal** (`scripts/spike-dynamic-dispatch/marshal.py`) — deterministic
  finalize. Code kind → file write; explanation → prose finish.

This reunifies what `spike-omega-dispatch/agent-turn-dispatch.yaml` had to split
into decide-ensemble + adapter-invoke + finalize (its docstring notes the engine
could not resolve a runtime-chosen `ensemble:` target). Now it can, so the turn
is one ensemble.

## Seats

- `dd-seat-code-solo` — strategy A, one coder node.
- `dd-seat-code-verified` — strategy B, coder + reviewer (structurally different,
  same code-seat contract). The swap-ability pair.
- `dd-seat-explain` — non-build capability, prose.
- `dd-echo-*` — deterministic stand-ins for the no-model smoke.

## Running

No-model smoke (proves the wiring, routing, swap, non-build shaping):

    uv run python scratch/spike-dynamic-dispatch/smoke_skeleton.py

Live-model drive (real local 8b seats; models must be pulled in Ollama):

    uv run python scratch/spike-dynamic-dispatch/drive_skeleton.py

Through real `opencode run`: serve `dd-serve-skeleton` behind the a2-proven
Ω-serve harness (`scratch/spike-omega-serve/serve_ensemble.py`), replacing its
imperative `handle_async`/`_produce` body with a single
`executor.execute(dd-serve-skeleton, turn)` and mapping marshal's outcome to the
OpenAI response. Transport (SSE, toolless aux short-circuit) is unchanged.
