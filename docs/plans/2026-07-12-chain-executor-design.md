# Chain executor: one budgeted deterministic chain plan

**Status:** Designed 2026-07-12 (brainstormed with the practitioner;
approved). Not yet implemented. Roadmap WS-3 item 1 / issue #120, the
single most leveraged design in the plan: grep, surgical edits,
multi-file deliverables, and the WS-5 long-horizon plan all ride it.
Generalizes the four shipped ad-hoc chains (read->build, glob->read->build,
write->run->verdict, convergent re-fix) with all four in hand as evidence.

## Problem

The four chains are not a control structure. Each is a lifecycle
*implicitly* encoded in `classify._route`, a twelve-branch priority
cascade (`.llm-orc/scripts/agentic_serving/classify.py:625-684`, plus
`_fix_chain_route:604-622`). Every wire round, `main()` recomputes
signals from the wire (`has_run_block`, `wrote_path`, `needs_files`,
`failure_shape`, glob candidates) and `_route` first-matches them to a
target. The "current step" of a chain is just the first ladder branch
whose evidence is present. There is no server state; the wire is the
state, and the turn loop is client-owned (each step is a separate wire
round the client drives).

That works, but the plan is invisible and adding a chain means editing
control flow:

1. The chain structure lives in an if/elif ladder, not as data. You
   cannot see the plan, diagnose which step a multi-pass turn is on, or
   add a chain (grep->read) without touching the routing brain.
2. Termination is guaranteed only implicitly, by per-step idempotency
   (`has_run_block` flips need-run->verdict, `has_refixed` bounds re-fix,
   `_visibility` bounds reads). There is no explicit round ceiling.
3. The turn trace records nodes and snippets, not the plan, so a
   multi-pass turn is undiagnosable post-hoc (the gap the fix-execution
   design flagged).

## Decisions

- **A serving-layer chain-plan primitive, not an engine node.** The
  engine's `loop:` combinator is walled off from cross-turn iteration by
  design ("the outer turn loop is client-owned"); every chain step is a
  separate client-driven wire round. So the primitive is the cross-round
  cousin of `loop:`: same termination philosophy (a mandatory bound),
  opposite side of the client boundary. It lives in a new
  `chain_plan.py` sibling of `classify.py`, holding:
  - `Step = {target, kind, build, guard}` where `guard` is a predicate
    over the signal bundle. A step *is* one `_route` branch.
  - `Chain = {name, steps, max_rounds}`, an ordered labeled group of
    steps sharing a lifecycle.
  - `advance(signals) -> Decision`, a flat first-match scan over all
    steps in priority order, tagged with `(chain, step_index)`.

- **The advancer is `_route`, transposed to data.** `CHAINS` is the
  ladder linearized in the same order with the same conditions; each
  guard carries the *full* condition from `_route` (including the
  chain-selecting signal such as `fix_chain`/`run_signal`), so there is
  **no separate "chain applies" gate** that could reorder anything. The
  `chain` label drives the plan, budget, and trace only; it never affects
  which step fires. Same inputs -> same target, mechanically.

- **The signal bundle schema is `_route`'s own signature.** `_route`
  already takes every signal it reads as an explicit kwarg, so the bundle
  is literally that kwarg set (a dataclass built in `main()` from the
  existing extractors). No signal can be silently dropped.

- **An explicit per-chain round budget (defense in depth).** Each chain
  declares `max_rounds` (run: 2, build: ~3, fix-cont: ~4, explain: 1),
  the deepest legitimate wire-round depth for its lifecycle. `advance`
  derives `rounds_consumed` statelessly by counting this turn's
  continuation blocks in the post-boundary slice
  (`[read]`/`[ran]`/`[globbed]`/`[wrote]`); a chain over its ceiling
  terminates with an honest "chain budget exhausted" message. The
  existing per-step idempotency guards already guarantee termination, so
  **the budget never trips on today's chains** (a regression test pins
  that). It is the cross-round analogue of `loop:`'s mandatory
  `max_iterations`, earning its place the day a future chain or a bug
  would otherwise spin.

- **The trace records the plan.** `build_turn_trace` gains a structured,
  un-clipped `chain_plan` record `{chain, step, step_index,
  rounds_consumed, max_rounds}`, so a multi-pass turn is diagnosable
  ("fix-cont, step 2/3, round 2 of 4"). The separate 280-char snippet cap
  (`turn_trace.py:24`) stays out of scope (the #114 remainder).

- **Deliberate non-changes (keep the structural commit tight).** The
  caller's `_resumes_turn`/`_tool_result_ack` (ack-vs-resume), the
  downstream shape nodes, and `main()`'s per-target
  `dispatch_input`/`file`/`needs_run`/`not_grounded` composition all stay
  as-is. Folding the output composition into per-step definitions, and
  deduping the caller's resume predicates against `CHAINS`, are flagged
  fast-follows, not this arc.

## The table

Every `_route` branch is one labeled step, in the same priority order.
Rows 1-3 are `_fix_chain_route`; the rest are `_route` verbatim.

| # | chain | step (target) | guard (over the signal bundle) | build= |
|---|-------|---------------|--------------------------------|--------|
| 1 | fix-cont | `need-run` | `fix_chain and needs_another_run` | F |
| 2 | fix-cont | `re-fix` | `fix_chain and not has_refixed and failure_shape=="localized"` | T |
| 3 | fix-cont | `run-verdict` | `fix_chain` | F |
| 4 | run | `run-verdict` | `run_signal and has_run_block` | F |
| 5 | run | `need-run` | `run_signal` | F |
| 6 | explain | `not-grounded` | `is_explain and explain_ungrounded` | F |
| 7 | explain | `explainer` | `is_explain` | F |
| 8 | build | `need-glob` | `needs_glob or glob_failed` | F |
| 9 | build | `need-files` | `needs_files or read_failed` | F |
| 10 | build | `tests-seat` | `tests_primary` | T |
| 11 | build | `code-seat` | `has_build_signal` | T |
| 12 | decider | *(model decider)* | `True` (fallthrough -> `needs_decider`) | F |

Reading it top to bottom is reading `_route`. The four chains read off it
statelessly: the read->build turn sits at row 9 while its file is
invisible then falls to 10/11 once the `[read]` block appears;
glob->read->build walks 8->9->10/11 as `[globbed]` then `[read]` land;
write->run->verdict walks 1->3 as `[wrote]` then `[ran]` land; the
convergent re-fix takes 1->2->1->3 as the localized-red verdict routes to
`re-fix`, whose write then needs its own run before the terminal verdict.

## Components

| Change | Where |
|--------|-------|
| `Step`/`Chain`/`CHAINS` table, `advance`, `rounds_consumed` | new `chain_plan.py` |
| Signal bundle built from existing extractors; call `advance`; drop `_route`/`_fix_chain_route` | `classify.py` |
| `chain_plan` trace record (structured, un-clipped) | `turn_trace.py` |
| grep->read: `needs_grep` predicate, new row(s), `[grepped]` render + tool map | `classify.py`, `serving_ensemble_caller.py` (behavior arc) |

## Commit plan (structure then behavior, strictly separated)

1. **Structural, the migration.** Extract `chain_plan.py`; build the
   signal bundle in `main()`; re-express `_route`/`_fix_chain_route` onto
   `advance()`. Byte-identical: the existing routing suite passes
   unchanged.
2. **Structural, additive.** The budget backstop (+ its never-trips
   regression pin), then the trace `chain_plan` record. Each adds no
   change to today's observable routing.
3. **Behavioral, grep->read.** Wire-capture the grep tool first (the
   read/glob procedure), then TDD the `needs_grep` predicate, the
   `[grepped]` render + tool map, and the new row(s). Its own arc.

## Testing and validation

- **Parallel-run differential test (the byte-identical proof).** Keep
  `_route` in place, add `advance()`, assert identical output over a
  corpus of turns (existing fixtures plus generated permutations), then
  delete `_route` once green. Cover with a characterization test, refactor
  under it, remove the old path.
- **Existing routing suite green UNCHANGED** (~80 pinned tests across
  `test_serving_classify.py`, `test_serving_context_render.py`,
  `test_serving_resolve.py`, `test_serving_run_verdict.py`,
  `test_serving_refix_*.py`, `test_serving_emit.py`). Any test that needs
  editing is a behavior change and a red flag.
- **New `chain_plan` unit tests:** the table walk, first-match order,
  budget-never-trips-on-existing, budget-trips-on-synthetic-overflow.
- **Hermetic e2e through the real engine:** the four chains fire
  identically (read->build, glob->read->build, write->run->verdict,
  convergent re-fix).
- **Live real-OpenCode battery rerun as a no-op check** (WP-A lesson): the
  13-turn ladder scores within variance of 10/13, routing fires
  identically on every turn, zero new dishonest outcomes. Trajectory row
  appended.
- **Independent adversarial review with a wrong-accept hunt retargeted
  for a refactor:** did any routing decision silently drift, did the
  budget trip when it should not, did the bundle drop a signal. The
  author-independent reviewer's APPROVE is the merge gate.

**Exit gate (issue #120):** the existing chains re-expressed on the
executor with byte-identical battery behavior; then one new chain
(grep->read) lands as data. Meta-task probe for the grep arc: a question
about the llm-orc repo ("how does classify decide routing?") answered
through the serve via a grep->read chain, grounded and honest.

## Risks

1. **Silent routing drift** (dominant): a mis-transcribed guard, a
   reorder, a dropped signal. Killed by the parallel-run differential
   test, the unchanged fixtures, and the bundle-is-the-signature gift.
2. **Budget over-eager:** trips a legitimate deep chain. Ceilings set
   above measured depth; the never-trips-on-battery pin guards it.
3. **grep scope creep:** grep's discovery discipline (match rule,
   ambiguity, top-N read fan) is deferred to WS-3 item 2's own design;
   this arc fixes only its attachment shape.

## Named forward directions (not built here)

- **grep->read semantics** (WS-3 item 2): the match rule, refuse-with-
  candidates on ambiguity, the bounded read fan, and the `explain ->
  grep -> read` meta-task consumer the grounded-explain design named.
- **Caller resume dedup:** `_resumes_turn`/`_tool_result_ack` derive the
  resuming step-shapes from `CHAINS` instead of hard-coding read/run/glob,
  removing the duplicated knowledge. A structural fast-follow.
- **Per-step output composition:** fold `main()`'s per-target
  `dispatch_input`/`file` shaping into per-step definitions.
- **WS-5 plan substrate seam:** a long-horizon plan is a chain with many
  steps; `chain_plan.py` is where that lands, with a client-visible
  `todowrite` mirror as the open fork. Designed against, not into, this
  arc.
- **#106 single home for serving shapes:** `chain_plan.py` is a natural
  home; kept separate here.
