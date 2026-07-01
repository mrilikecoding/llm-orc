# Engine control-flow primitives: state + next steps (handoff)

**Status:** as of 2026-06-30. Resumption entry point for the L0 engine
control-flow work spun out of the "agent as ensemble" program. Companion docs:
`../references/engine-control-flow-primitives.md` (the WHAT — the declarative
vocabulary, shipped vs planned, with file evidence) and
`ensemble-agent-state-and-next-steps.md` (the parent program: questions a/b/c).
This doc is the WHERE-WE-ARE + WHAT'S-NEXT for the engine primitives.

---

## 1. The arc (why this exists)

The parent program's §2 finding was "the agent can BE an ensemble with **no L0
engine change**." That answered *is it possible* (yes, via scripts + a thin
adapter). This session deliberately **reopened that decision** on a different
axis — *is it right* — and concluded no: if control flow stays in adapter/
harness Python, you rebuild the bespoke LoopDriver (the orchestration graph
lives in Python, the engine is a leaf-caller). The corpus thesis ("the agent
*is* a declarative ensemble") is only honestly true if the agent's control flow
lives in the ensemble, not in a driver. So we are adding the minimal engine
primitives that let the serving flow externalize into config.

The engine was a **fire-everything acyclic DAG**: it fans out and gathers but
could not skip, branch, loop, or dispatch a runtime-chosen target. Evidence: it
already shipped an imperative `agentic/` layer (`tier_router.py` = route-and-
judge in Python) because the DAG couldn't express it. We extract that pattern
into primitives instead of hand-rolling it a third time.

Serves the parent ladder: declarative single-responsibility nodes are the only
form where "each component is independently optimizable" is literally true, and
they make (c) — comparing strategy ensembles head-to-head — possible as config
rather than bespoke harnesses.

## 2. What shipped this session (4 commits, branch `agentic-serving`)

- `cb87ded` **feat:** guard primitive — `when:` on a node; skip when the
  predicate is false or all deps skipped. Branch = mutually-exclusive guards;
  join runs on whichever branch fired.
- `9d1a619` **refactor:** extract shared `predicate` evaluator (the `${x}`
  truthiness / `== literal` grammar) so guard and loop share it.
- `c5059bf` **feat:** bounded loop combinator — `loop: {body, until,
  max_iterations, carry}`. Re-runs a body ensemble until `until` holds over its
  terminal output or the mandatory bound trips; `carry` threads a body-output
  field into the next iteration. Top-level graph stays acyclic; iteration is
  scoped in the node; termination guaranteed by the mandatory bound.
- `7ceab06` **docs:** reference doc marks the loop shipped.

**Verified:** full suite 3070 passed, coverage 92.61%, mypy/ruff/complexipy/
bandit/vulture clean via pre-commit. The one failing test
(`test_cache_configuration_from_performance_config`) is pre-existing and
unrelated — local `.llm-orc/config.yaml` sets `script_cache.enabled: false`.

**Primitive status:** shipped = guard/branch + bounded loop (plus the prior
DAG/fan-out/input_key/script-model-ensemble nodes). Planned = dynamic dispatch.
Future = stall-detect + wall-clock budget loop bounds. Full table with evidence
in the reference doc.

## 3. Next steps (prioritized)

1. **Phase 3 — assemble the full serving flow as ONE declarative ensemble —
   DONE (2026-06-30, spike Ω-P3, `scratch/spike-omega-p3/`).** Assembled
   `resolve-contract (loop) → plan (script) → build (ensemble + fan_out) →
   score (script)` as ONE ensemble (`spike-omega-p3-full.yaml`), no Python
   driver; runs free end-to-end (deterministic architect + stand-in builders):
   loop converges iter 2, build fans over the 6-file contract, routes 5→code /
   1→prose, score gathers all 6. **The predicted wall (dynamic dispatch) is NOT
   hit for the (b) flow.** The build-step capability choice (code vs prose) is a
   CLOSED set, so a guarded BRANCH covers it — dynamic dispatch is only the
   OPEN-library (c) lever. Proven to compose: `fan_out` over an `ensemble:` node
   (routing-demo pattern), a guard branch INSIDE a fanned sub-ensemble, and a
   `loop:` INSIDE a fanned sub-ensemble (deep nesting within the depth-5 limit).
   **The real walls were ergonomic, not missing primitives** (see Ω-P3 README):
   (a) data threads through prose wrappers — fan-out's "Chunk content:\n<json>"
   plus the dependency enhanced-input wrapper, so deterministic nodes must
   `raw_decode` the chunk out; (b) per-instance context must be folded into the
   chunk (fan-out hands an instance only its element); (c) loop output is wrapped
   `{output, iterations, terminated}` and must be unwrapped downstream; (d) a
   bare-list `script:` output trips `process_script_output_with_requests`
   (assumes a dict) — fix is a dict + `input_key` (routing-demo convention).
   These name future *conveniences*, not blockers. **Live checkpoint DONE (PAID,
   2026-06-30):** `spike-omega-p3-full-live.yaml` ran the whole flow with REAL
   models (frontier architect `spike-omega-loop-body-live` + cheap-local qwen3:8b
   builders + real execution gate). It serves end to end as one declarative
   ensemble: the architect converges live inside the loop (iter 1 one run; **iter
   2 with a real carry-repair** another — the live repair Ω-loop never triggered),
   the build fans the real generators, the execution gate runs. **But the run
   exposed the real quality wall, not an engine one:** the cheap-local one-pass
   build shipped a CROSS-FILE COHERENCE LEAK the contract did not pin —
   `tokenizer.py` emits `Token.kind ∈ {"operator","number"}` while `parser.py`
   checks `'NUMBER'`/`'+'`, so every expression raises — and the cheap-built
   `test_calc.py` SWALLOWED the exception (`try/except print`), so the execution
   gate `rc=0` was a FALSE PASS (package retained, `scratch/spike-omega-p3/
   live_run_package/`). Two sharpened findings: (i) the execution gate is a
   truth-teller only over a REAL test — a cheap-built self-test that swallows
   failures defeats it, so the gate must run an INDEPENDENT/author-controlled
   check, not the builder's own; (ii) contract-first pins interface shape, not the
   semantic conventions (string discriminant vocabularies) cheap independent
   builders must agree on. Neither the Ω-E coherence gate (imports resolve) nor a
   form gate (both files valid) nor builder-tier escalation (§6.1) catches an
   executional vocabulary leak — only an independent execution check does.
2. **Validate `resolve_contract` as a real declarative loop — FREE ARM DONE
   (2026-06-30, spike Ω-loop, `scratch/spike-omega-loop/`).** Lifted
   `scratch/spike-omega-e/coherence_gate.py` to a substrate `script:` node
   (reused unchanged, no fork); body ensemble = architect + gate emitting
   `{ok, reasons, contract, next_input}`; loop node `until: ${ok}`, `carry:
   ${next_input}`, `max_iterations: 3` (= `resolve_contract(max_repairs=2)`).
   Validated through the real `EnsembleExecutor` against the pinned calc
   fixtures with a *deterministic* fixture-replay architect (so the loop
   mechanics cost no frontier tokens): converge arm stops via `until` at
   iteration 2, exhaust arm stops "exhausted" at the bound (3) with honest
   reasons. Two changes from the original plan, both load-tested: (a) carry
   `${next_input}` (the *formatted* feedback) not `${reasons}` (a raw list) —
   the architect keys on the rejection-marker prose, and a live architect needs
   prose, not a list; (b) the free deterministic arm comes before the paid run.
   **PAID ARM DONE (2026-06-30):** swapped the fixture architect for the real
   frontier architect (model node `agentic-orchestrator-qwen36-zen` =
   qwen3.6-plus via paid Go) in `spike-omega-loop-body-live.yaml` +
   `spike-omega-loop-live.yaml`; `run_live.py` drove it. Result:
   `terminated="until"`, `iterations=1`, `ok=True`, ~88s — a real frontier
   architect converges through the exact loop the deterministic arm validated,
   and the task-aware robust gate accepts genuine frontier output. **#2 is now
   confirmed end-to-end (free + paid).** Still open (a model question, NOT a
   loop-primitive gap): whether qwen3.6-plus reliably *repairs* an incoherent
   contract from the gate's feedback — it didn't err on calc this run, so a live
   carry-repair wasn't observed (the mechanics are proven deterministically; the
   model behavior ties to §6.2's architect-as-single-point-of-failure).
3. **Carry threading end-to-end through a real convergent body — DONE.** The
   Ω-loop converge arm demonstrates it through the real script-stdin nesting (no
   longer only `LoopController`/`LoopAgentRunner` unit tests): the two arms share
   one loop spec and differ only in the architect's repair fixture, so
   convergence is necessarily carry-driven. **Constraint surfaced and resolved:**
   `carry` threads exactly one field as the next iteration's *whole* input — the
   original task is NOT auto-preserved. Resolved in the gate node (the terminal):
   it composes `next_input = task + DELIM + feedback` and recovers the pristine
   task by re-splitting its own `input_data` on `DELIM` (the whole input on
   iteration 1). No accumulation, single source of truth. General pattern: a loop
   body that needs more than one field across iterations folds them into the one
   carried string and re-splits on the way in. Confirmed by the live arm.
4. **Dynamic dispatch (Phase 4)** — runtime-resolved `ensemble:` target.
   **Confirmed by Ω-P3 to be the OPEN-library (c) lever, NOT a (b) blocker:** #1
   assembled the full (b) flow without it (closed code/prose set → guarded
   branch). Build it when a strategy needs a capability chosen from an open
   library at runtime, not for the generalist serving flow. The likelier near-
   term engine work is the Ω-P3 ergonomic conveniences (clean structured-chunk
   channel, broadcast input for fan-out, loop-output unwrap, list-shaped script
   output) if the flow graduates from spike to product.
5. **Stall-detect + budget** loop bounds (fixpoint detection, wall-clock) — when
   a real loop needs more than `max_iterations` + `until`.

## 4. Operational facts (so a fresh session does not relearn them)

- **Source map:** schema `src/llm_orc/schemas/agent_config.py` (`when` field,
  `LoopSpec`, `LoopAgentConfig`); `core/execution/phases/predicate.py` (shared
  grammar), `.../guard_evaluator.py`, `.../loop_controller.py`;
  `core/execution/runners/loop_runner.py`. Wiring in `ensemble_execution.py`
  (`_partition_by_guard` in the phase loop; `LoopAgentConfig` branch in
  `_execute_agent`; runner constructed in `__init__`) and
  `phases/agent_dispatcher.py` (`_determine_agent_type`).
- **Tests:** `tests/unit/core/execution/test_{predicate,guard_evaluator,
  guard_integration,loop_controller,loop_runner,loop_integration}.py` and the
  `Loop`/`when` cases in `tests/unit/schemas/test_agent_config.py`.
- **Running tests:** the repo's pytest `addopts` force `--cov ... --cov-fail-
  under=90`; for a single file use `uv run pytest <file> -o addopts=""
  -p no:cacheprovider -q`. Full suite + gate: `uv run pytest -p no:cacheprovider`.
- **Pre-commit hook** runs mypy `src tests`, ruff check, ruff format --check,
  complexipy (≤15), bandit, vulture — but **not** pytest. So mypy-check test
  files too (the hook caught a test-only `no-any-return` this session), and run
  `ruff format` on touched files before committing.
- **Loop semantics:** `until`/`carry` read the body's **terminal output**
  (the `deliverable` field of the child result, JSON-parsed) as a flat dict, so
  `${ok}` not `${body.ok}`. The body ensemble owns emitting `{ok, reasons, ...}`.
- **Commit discipline:** structure vs behavior in separate commits (the
  predicate extraction was its own `refactor:` before the loop `feat:`). No AI
  attribution.

## 5. Pointers

- Reference (the vocabulary): `../references/engine-control-flow-primitives.md`.
- The loop primitive validated as a declarative flow: `scratch/spike-omega-loop/`
  (README + `validate_loop.py`); config in `.llm-orc/ensembles/spike-omega-loop-*`
  and `.llm-orc/scripts/spike-omega-loop/`.
- The full serving flow as ONE declarative ensemble (Phase 3):
  `scratch/spike-omega-p3/` (README + `run_p3.py`); config in
  `.llm-orc/ensembles/spike-omega-p3-*` and `.llm-orc/scripts/spike-omega-p3/`.
- The gate to lift in #2: `scratch/spike-omega-e/coherence_gate.py` +
  `run_e.py`'s `resolve_contract` (the loop's reference spec) + `fixtures/`
  (now incl. `calc_coherent_contract.json`).
- Parent program + the a/b/c ladder: `ensemble-agent-state-and-next-steps.md`.
- Working preference: default to declarative engine nodes over large scripts;
  add the minimal primitive rather than growing an adapter.
