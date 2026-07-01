# Spike Ω-loop — the bounded loop primitive as a real declarative flow

**Question (engine handoff #2):** can the shipped `loop:` combinator express a
real convergent agentic flow *as config*, not just pass unit tests? The Ω-E
spike already had the executable spec for one — `resolve_contract` (the
architect-repair cycle) over `coherence_gate.py` — as plain Python. This lifts
that same cycle into the engine as a declarative loop and validates it.

**Answer: yes.** Both control paths run through the real `EnsembleExecutor`.

## What was built (all config + two thin script shims)

- `.llm-orc/scripts/spike-omega-loop/coherence_gate_node.py` — the Ω-E gate
  lifted to a `script:` node. Reuses `scratch/spike-omega-e/coherence_gate.py`
  unchanged (no logic fork); the node is only an I/O shim. Reads the upstream
  architect's output from its `dependencies`, emits `{ok, reasons, contract,
  next_input}` as the body's terminal output. Task-aware (see the carry finding)
  and robust to real LLM output: extracts the JSON object even when fenced, and
  treats an unparseable/empty contract as a failure so the loop never
  false-passes on no contract.
- `.llm-orc/scripts/spike-omega-loop/fixture_architect.py` — a deterministic
  fixture-replay stand-in for the frontier architect, so the LOOP MECHANICS are
  validated without spending frontier tokens. It is a state machine: a fresh
  request emits the `fresh_fixture`; once the gate's rejection feedback (carrying
  `REJECTED by the coherence gate`) is fed into its input, it emits the
  `repair_fixture`. Its `{"deliverables": [...]}` output matches the real
  architect's contract (run_e.py), so the gate node and loop work unchanged when
  the live architect is swapped in.
- `.llm-orc/ensembles/spike-omega-loop-body-{converge,exhaust}.yaml` — the loop
  body: `architect -> gate`. The gate is the terminal node, so its output is the
  body's deliverable that `until`/`carry` read.
- `.llm-orc/ensembles/spike-omega-loop-{converge,exhaust}.yaml` — the loop nodes:
  `until: ${ok}`, `carry: ${next_input}`, `max_iterations: 3` (maps to
  `resolve_contract(max_repairs=2)` = 3 total attempts).
- `validate_loop.py` — the harness. Builds a root executor, runs both arms,
  asserts the LoopOutcome.

## Result (deterministic, free)

    uv run python scratch/spike-omega-loop/validate_loop.py

    [converge] terminated='until'     iterations=2 ok=True   -> PASS
    [exhaust]  terminated='exhausted' iterations=3 ok=False  -> PASS

- **converge:** incoherent contract on iteration 1 -> gate rejects (ok=false) ->
  `carry` threads the feedback into iteration 2 -> architect emits a coherent
  contract -> gate passes -> loop stops via `until` at iteration 2, within the
  bound of 3.
- **exhaust:** never-coherent contract -> `until` never holds -> loop stops
  "exhausted" at the bound (3), reporting 8 honest failure reasons.

The two arms share an identical loop spec and differ only in the architect's
repair fixture. The only per-iteration-varying input the architect sees is the
carried feedback, so convergence at iteration 2 is necessarily **carry-driven** —
this is the end-to-end proof of carry threading through the real script-stdin
nesting (engine handoff #3), previously only covered by `LoopController` /
`LoopAgentRunner` unit tests.

## Findings

- **The shipped `loop:` primitive expresses the Ω-E repair cycle as pure config.**
  Termination (`until`), the mandatory bound, and carry threading all work
  through the real engine. The Python `resolve_contract` driver is no longer
  needed to run this flow.
- **`carry` threads exactly one field as the next iteration's *whole* input —
  and that is enough, once the carried field carries the task too.** The body's
  terminal output is one dict; `carry: ${next_input}` replaces the base input, so
  the original task is NOT automatically preserved. The deterministic architect
  sidesteps this (it only needs "was I rejected?"), but a LIVE architect needs
  both task and feedback on a repair iteration. Resolved in the gate node: it
  composes `next_input = task + DELIM + feedback`, and recovers the pristine task
  by splitting its own `input_data` on `DELIM` (the whole input on iteration 1).
  No accumulation, single source of truth, and the terminal node owns the
  "compose the next prompt" logic. This is the general pattern for any convergent
  loop body that needs more than one field across iterations: fold them into the
  one carried string and re-split on the way in.

## Live (paid) arm — done

    uv run python scratch/spike-omega-loop/run_live.py   # PAID: real frontier architect

Swapped `fixture_architect.py` for the real frontier architect (model node
`agentic-orchestrator-qwen36-zen` = qwen3.6-plus via paid OpenCode Go), same loop
spec, in `spike-omega-loop-body-live.yaml` + `spike-omega-loop-live.yaml`.

**Result (2026-06-30):** `terminated="until"`, `iterations=1`, `ok=True`, ~88s.
The real architect emitted a coherent calc contract on the first try (tokenizer
defines `Token`; parser imports `Token` from tokenizer and defines `ASTNode`;
evaluator imports `ASTNode` from parser; cli imports all three; correct
dependency order, no orphan modules, no self-naming), the task-aware gate
accepted it, and the loop terminated via `until` at iteration 1. So a real
frontier architect plugs into the exact loop the deterministic arm validated and
converges through it; the robust gate accepts genuine frontier output.

**One path not exercised this run:** a *live carry-repair* — the real architect
erring, the gate rejecting, and the real architect repairing from the carried
feedback. The architect didn't err on calc this time (it did on 2026-06-30,
which motivated the gate; model nondeterminism). The deterministic arm proves the
carry-repair *mechanics* rigorously; whether qwen3.6-plus reliably *fixes* an
incoherent contract from the gate's specific feedback is a separate, still-open
empirical question (a follow-up paid run on a task that trips it, or a forced
seed, would close it).
