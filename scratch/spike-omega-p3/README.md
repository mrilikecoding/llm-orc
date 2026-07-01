# Spike Ω-P3 — the full serving flow as ONE declarative ensemble (Phase 3)

**Question (engine handoff #1):** can the whole agentic serving flow —
`resolve-contract → build → score` — be expressed as ONE declarative ensemble
(no Python driver), and where does the declarative vocabulary run out? The
handoff predicted the wall would be **dynamic dispatch** for the build step's
capability choice.

**Answer: the (b) generalist flow assembles end-to-end with the SHIPPED
primitives — there is no hard wall, and dynamic dispatch is NOT required for it.**
The predicted wall was wrong for this case: the build-step capability choice
(code vs prose) is a CLOSED set, which a guarded branch covers. Dynamic dispatch
is only needed for an OPEN capability library (the (c) research track).

## What runs (free, deterministic)

    uv run python scratch/spike-omega-p3/run_p3.py spike-omega-p3-full

    stage status: resolve-contract=success plan=success build=success score=success
    resolve-contract (loop): terminated=until iterations=2 ok=True
    score: 6 files built
       tokenizer.py  tier=code   fired=build-code   content=True
       parser.py     tier=code   fired=build-code   content=True
       evaluator.py  tier=code   fired=build-code   content=True
       cli.py        tier=code   fired=build-code   content=True
       test.py       tier=code   fired=build-code   content=True
       README.md     tier=prose  fired=build-prose  content=True

`spike-omega-p3-full.yaml` wires four nodes, every stage declarative:

- **resolve-contract** — `loop:` node, the validated Ω-loop (deterministic
  architect → coherence gate). Converges at iteration 2.
- **plan** — `script:` projecting `output.contract` into per-file build tasks,
  each a composite `{deliverable, contract}` (the full contract rides in every
  element so each fanned instance has its sibling APIs).
- **build** — `ensemble:` + `input_key: tasks` + `fan_out: true`: one
  `build-one` sub-ensemble per file. Inside it: `classify` → guarded
  `build-code` (a `loop:`, the build-with-retry) / `build-prose` branch →
  `marshal`. The closed capability set ⇒ a guard, not dynamic dispatch.
- **score** — `script:` gathering the fanned results (stand-in execution gate).

## What was proven, in order (free, script stand-ins)

| probe | question | result |
|---|---|---|
| 3a | `fan_out` over an `ensemble:` node? | YES — runs the sub-ensemble per element (routing-demo pattern); chunk delivered as `"...Chunk content:\n<json>"` |
| 3b | guard branch (code/prose) INSIDE a fanned sub-ensemble? | YES — code files → build-code, doc → build-prose; marshal joins the fired branch |
| 3c | a `loop:` INSIDE a fanned sub-ensemble? | YES — build-code loop ran at depth (top → fanned ensemble → loop → loop body), converged, marshal unwrapped it |
| full | all four stages as one ensemble? | YES — 6 files, routed 5 code / 1 prose, converged |

So: loop + fan-out + guard-branch + nested loop all compose, including deep
nesting, within the depth-5 limit.

## The real findings — ergonomic frictions, not missing primitives

Every wall I expected turned out to be a rough edge in data plumbing, each of
which names a future *convenience*, not a blocking primitive:

1. **Data threads through prose wrappers.** fan-out hands a chunk as
   `"Processing chunk N of M ... Chunk content:\n<json>"`; a node with deps gets
   the verbose "enhanced input" wrapper; a loop body inside a fanned sub-ensemble
   gets BOTH layers. A deterministic node must locate the marker and `raw_decode`
   the JSON out — a naive parse spans two JSON blobs plus prose and fails (this
   bit the code stand-in until fixed). For LLM nodes the prose is natural; for
   deterministic stages it is friction. Future: a clean structured-chunk channel.
2. **Per-instance context rides in the chunk.** fan-out gives an instance only
   its element + base_input, not the whole array — so the plan stage folds the
   full contract into each `{deliverable, contract}`. Future: a "broadcast"
   input alongside the per-instance chunk.
3. **Loop output is wrapped** `{output, iterations, terminated}`; downstream
   consumers (marshal, plan) must unwrap `output`. Future: an unwrap option.
4. **Bare-list script output trips the request processor.** A `script:` emitting
   a top-level JSON array for fan-out hits `process_script_output_with_requests`
   (it assumes dict-shaped output) → a logged `AttributeError` (non-fatal, the
   array is still used). Fix: emit a dict and select the array with `input_key`
   (the routing-demo convention) — adopted here, removes the error. Future:
   tolerate list-shaped script output.

## Bottom line for the a/b/c ladder

The (b) generalist serving flow is fully declarative on today's engine. Dynamic
dispatch (handoff #4 / Phase 4) is reserved for the OPEN-library (c) case, not
needed here. The friction-fixes above are the natural next engine conveniences if
this flow graduates from spike to product.

## Live checkpoint — DONE (PAID), and it surfaced the real quality wall

    uv run python scratch/spike-omega-p3/run_live_full.py spike-omega-p3-fanbuild-live  # local-only, free
    uv run python scratch/spike-omega-p3/run_live_full.py spike-omega-p3-full-live      # PAID, full

`spike-omega-p3-full-live.yaml` swaps the loop body to `spike-omega-loop-body-live`
(real frontier architect, qwen3.6-plus paid Go) and the build stand-ins to the
real cheap-local capabilities (`code-generator-omega` / `prose-generator-omega`,
qwen3:8b), with `score_live.py` as the real execution gate (write files + run the
test). The structure (Phase 3) is unchanged; only the models are real.

**It runs end to end as one declarative ensemble with real models.** Across runs:

- **Frontier architect converges live inside the loop** — iteration 1 on one run,
  and **iteration 2 with a real carry-repair on another** (the architect emitted
  an incoherent contract, the gate rejected it, the carried feedback drove a
  successful repair). That is the live carry-repair the Ω-loop live run never
  triggered.
- **Run-to-run variance in the cheap-local build.** One run: `build` came back
  `partial` with 0 usable files (flaky 6-way-concurrent local generation). Next
  run: all 6 instances succeeded, files written, execution gate ran.
- **The execution gate reported `rc=0` — and it was a FALSE PASS.** The generated
  package fails on *every* expression when run directly:

      ValueError: Unexpected token   # parser.py parse_factor

  Two real, layered defects, neither caught by the flow as built:

  1. **Cross-file coherence leak the contract did not pin.** `tokenizer.py` emits
     `Token.kind ∈ {"operator","number"}`; `parser.py` checks `kind == 'NUMBER'`
     / `in ('+','-')` / `== '('`. The two cheap builders invented incompatible
     `kind` vocabularies. Contract-first (Ω-E) froze signatures + field *names*
     (`Token.kind`, `.value`) but NOT the string discriminant *values*, so
     independent cheap tiers diverged on the convention. The Ω-E coherence gate
     would not catch this (imports/symbols resolve fine); the form gate would not
     (both files are structurally valid). Only running real expressions catches
     it.
  2. **A builder-written test can defeat the execution gate.** `test_calc.py`
     wraps its asserts in `try/except Exception: print(...)`, so the parser's
     exception is swallowed and the test passes vacuously → `rc=0`. The gate did
     not lie (the test passed); the test is the liar.

  The retained broken package is in `live_run_package/`.

### Framing correction — the terminal `score` node mis-modeled the outer loop

`score_live` here is wired as a TERMINAL verdict (run the test → pass/fail →
done). That is a spike artifact, not the serving architecture: OpenCode owns the
outer loop and the ensemble serves ONE turn per request, so "code is wrong"
should trigger another round, not a terminal failure. So do not read the `rc=0`
false pass as "the system failed." Read it as: an internal check false-passed,
and a gameable check does not disappear under the outer loop — it RELOCATES to
wherever the loop gets its correctness signal. Wrong code is safely "another
round" IFF that signal is independent of the builder (OpenCode running real tests
as a tool, held/property/golden checks, or the user's acceptance). It becomes a
wrong terminal state only when the loop trusts the builder's own test or a
self-report of "done, tested."

### What the live checkpoint teaches (refines the parent program)

- **Grounding must live at the loop's accept decision, not (necessarily) inside
  the ensemble's single turn.** The Ω-exec execution gate is a truth-teller only
  over an evaluation the builder couldn't weaken; a cheap-tier self-test that
  swallows exceptions false-passes it. The requirement is an INDEPENDENT signal
  at the accept/another-round boundary, not an infallible internal gate. This is
  the sharpest open (b) item now (parent §6.2b).
- Contract-first has a vocabulary limit: it pins interface shape, not the semantic
  conventions (string discriminants, enum values) cheap independent builders must
  agree on. Either the architect must pin them, or a stricter cross-file check
  (beyond import/symbol closure) is needed.
- The form-gate retry (omitted in this run to keep it tractable) + layer-aware
  escalation (§6.1) would not have caught the vocabulary leak — both files are
  structurally valid. This is an *executional* defect, so only an independent
  execution check finds it.

Bottom line: the declarative agent **serves end to end with real models** — the
Phase 3 structure holds — but the cheap-tier output quality, and trusting a
cheap-built test as the gate, are the real walls, not the engine vocabulary.

## Files

- Ensembles: `.llm-orc/ensembles/spike-omega-p3-{full,fanbuild,build-one,code-attempt}.yaml`
- Scripts: `.llm-orc/scripts/spike-omega-p3/{plan,plan_from_contract,classify,build_code,build_prose,code_attempt,form_gate,marshal_one,score,echo_chunk,collect}.py`
- Harness: `scratch/spike-omega-p3/run_p3.py`
