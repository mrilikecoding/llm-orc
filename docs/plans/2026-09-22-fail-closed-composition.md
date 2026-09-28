# Fail-closed composition: research-dossier on the mini

Date: 2026-09-22 · Source: `docs/composition-debugging-session.md` (mini, 2026-09-21)
and a local reproduction the same day.

## Measured (laptop, M-series, local llama-server router)

- Decomposer timeout on the mini is Qwen3 thinking. Same prompt, direct to the
  router: thinking on 295-893 generated tokens across 18 runs; off 64-83, valid
  JSON 3/3. At the mini's 4-6 gen tok/s that is ~60-225 s vs ~15 s.
- With `options: {think: false}` on the decomposer, `research-dossier` completed
  end to end locally in 128 s (decomposer 3.4 s, 5/5 searches). The compiler
  received 9,204 prompt tokens; 70% of the searcher payload is the child
  execution record, double-escaped (27,881 chars, 8,333 of them title/url/snippet).
- A variant on `glm-5.3-flash` via OpenCode Go "succeeded" with qwen3-0.6b output:
  Go 400s without `x-opencode-session` and on `chat_template_kwargs`; the runtime
  fallback substituted `micro-local` and reported `status: success`
  (`model_substituted: True`). The 0.6b decomposer fenced its JSON, `input_key`
  extraction failed, fan-out logged "zero instances — skipping", and the compiler
  wrote a dossier about JSON errors.

## Structure

One invariant, broken three ways: **a failed step never reaches a downstream
consumer as a success.**

### A. Fallback is explicit only (decided: explicit chains)

- Remove the implicit legacy fallback (`project.default_models.test` /
  `DEFAULT_LOCAL_MODEL`). A model load or runtime failure uses the
  `fallback_model_profile` chain (agent-level, then profile-level) and nothing else.
- The runtime-failure path honors the chain (today it passes no
  `original_profile`, so it always lands on the legacy default).
- No chain, or chain exhausted: the agent fails with the original error;
  dependents do not run (existing Invariant 13 behavior for failed agents).
- A chain hop still records `model_substituted: True` and the model that ran.

### B. Fan-out contract failure fails the agent

- `input_key` given and the upstream response does not parse, is not an object,
  or the key is missing / not a list: the fan-out agent fails with an error naming
  the upstream and the key. Dependents do not run.
- A parsed, genuinely empty list is a legitimate zero-instance result (success,
  empty). No lenient parsing (no fence stripping): the capture shows fences came
  from a substituted 0.6b, and A removes that path.

### C. Provider request shape

- `think` maps to `chat_template_kwargs.enable_thinking` for `llama-server` only.
  Setting `think` on any other provider is a load-time error, not a request field.
- OpenCode Go (`base_url` under `opencode.ai/zen/go`): send
  `User-Agent: llm-orc/<version>` and a stable `x-opencode-session` per top-level
  execution (child executors inherit it). Go is scoped to coding-agent traffic per
  its docs; research-dossier stays on local models.

### D. LLM consumers get the child's output, not its execution record (decided: consumer contract)

Extends #202's consumer dispatch in `DependencyResolver`. When an LLM agent
depends on an `ensemble:` agent (including gathered fan-out results), each child
result renders as its terminal agents' responses (one labeled block per terminal
agent), not the JSON-serialized result dict. Script and child-execution
consumers keep the full record, unchanged. This retires ADR-013's deferred
`output_mode` consequence; amend the ADR.

## Gates

- Hermetic suite green (baseline 4,398 passed; the one failure,
  `testget_available_providers_auth_only`, is a pre-existing leak from a live
  router on :8080, not in scope).
- Each pin goes red under a mutant reintroducing its defect (doctrine 11).
- Live: `research-dossier` end to end on the local router with the untracked
  ensemble config, compiler prompt tokens reported before/after D; fault
  injection for A (unreachable profile, no chain → agent fails, compiler does
  not run) and B (decomposer forced to emit non-JSON → fan-out fails).
- Independent adversarial review before merge.

## Addendum 2026-09-23: explicit node outcome (after review round 3)

Three review rounds each found a call site inferring success from overlapping
fields (`status`, presence of `reason`, `handled_failure`, `partial`, a merged
script payload). Doctrine 4: state the invariant once.

**One engine-owned field, `outcome`, stamped in one place per node:**

| outcome | when |
|---|---|
| `succeeded` | ran and succeeded |
| `partial` | fan-out gather with >=1 instance ok and >=1 instance blocking |
| `failed` | ran and failed: exception, script failure, input_key / fan-out contract failure, child/dispatch/loop with no ok terminal and >=1 blocking terminal |
| `skipped_by_failure` | no dependency ok, >=1 dependency blocking, node not `on_dependency_failure: run`; also a `when:`-false node whose dependencies include no ok one and >=1 blocking one |
| `skipped_by_guard` | `when:` false with >=1 ok dependency (or none), or every dependency `skipped_by_guard` |
| `handled_failure` | an `on_dependency_failure: run` node that executed where the cascade would have given `skipped_by_failure`, and its own execution succeeded |

Predicates (one module, the only readers): ok = `succeeded | partial`;
blocking = `failed | skipped_by_failure | handled_failure`; neutral =
`skipped_by_guard`.

- Nested (`ensemble:`/`dispatch:`/`loop:` final iteration): the agent is
  `failed` iff no terminal is ok and >=1 terminal is blocking; all-neutral
  terminals give `succeeded` with empty output.
- `has_errors` on every node result and every execution = any blocking
  outcome in its subtree (child executions included). Caller `status` is
  `error` iff the execution's `has_errors`. So a nested child with a failed
  intermediate is `succeeded` + `has_errors: true`, and the top level
  reports `error`: same facts, the thresholds the practitioner chose.
- `status` stays for compatibility, derived from `outcome`; `outcome` is
  authoritative and documented as such.
- Engine-owned keys are reserved. A script's structured payload is stored
  under `payload` and never merged over the record (`turn_trace` reads
  `payload.stderr`).

## Round 4 (2026-09-23, adversarial review round 4)

Five reviewer mutants (M1–M5), each pinned through the real executor
(doctrine 11) with a probe ensemble under `tests/fixtures/outcome_probes`,
demonstrated red under the mutant and green after revert:

- **M1** — `BLOCKING_OUTCOMES` excluding `handled_failure` (probes `hchain`,
  `hchainparent`): a run-marked node chained after another run-marked node
  must read its blocking (`handled_failure`) upstream as blocking, not ok.
  Already correct on this branch; the gap was the missing pin.
- **M3** — `loop_runner.py` dropping the `has_errors` key from its JSON
  response (probe `lhe`). Already correct; gap was the missing pin.
- **M4** — `turn_trace._engine_failure_fields` ignoring the nested `payload`
  key added by the outcome addendum. Already correct; the EXISTING pin
  hand-built a flat (pre-addendum) dict and stayed green under this exact
  mutant — the new pin drives a real executor record instead.
- **M5** — `_propagate_child_execution_errors`'s fan-out LIST branch
  (probe `efan`, new: an `ensemble:` fan-out over the existing `stderr`
  child, both instances succeeding outright but each with a failed
  non-terminal). Already correct; gap was the missing pin.
- Not a mutant, a genuine gap found while writing these pins: a **`partial`
  gathered fan-out's own `has_errors`** read `false` (`stamp_outcome`'s
  generic outcome-in-`BLOCKING_OUTCOMES` rule doesn't cover `partial`, an
  OK outcome). Fixed: `FanOutGatherer.gather_results` now stamps
  `has_errors: true` directly from the instance fail count whenever >=1
  instance failed. Pinned on probe `partfan`.

**Loop `has_errors`/`iteration_failures` (lead decision).** `has_errors` on
a loop agent's JSON response reflects the FINAL iteration only — earlier
iterations are retry attempts the loop exists to absorb, so a serving turn
that succeeds on retry must not report error. This was already the
behavior (row 6/8 of `docs/domain-model.md`'s Invariant 13 history). What
was missing: every OTHER iteration's blocking outcome was silently
dropped, recorded nowhere. `LoopAgentRunner.execute` now collects an
`iteration_failures` list (`{iteration, error}`) for every non-final
iteration whose terminal failed, added to the loop's JSON response when
non-empty. Pinned on new probe `lf2`/`lf2body` (flaky2.py fails iteration
1, passes iteration 2): loop succeeds, `has_errors` false,
`iteration_failures` names iteration 1.

**Deviation (a) — documentation only, no behavior change.**
`GuardEvaluator.should_run`'s cascade GATE (`cascades = agent_config.
on_dependency_failure != "run"`) is, and always was, unconditional for a
run-marked node: it short-circuits the "no dependency succeeded" check
entirely, so a run-marked node always executes regardless of whether its
dependencies are blocking, neutral, or ok. The addendum table above (and
domain-model row 8) only describe what OUTCOME gets stamped once the gate
lets the node through — `succeeded` when the cascade would not have been
`skipped_by_failure` (including the all-neutral case: probe
`runwhenskipped`), `handled_failure` otherwise. Neither previously stated
the gate/outcome distinction explicitly, which read as ambiguous on
re-read; `docs/domain-model.md`'s Invariant 13 history (row 9) now states
it.

**`ikdrop` — known pre-existing issue, deliberately not fixed this round.**
Probe `ikdrop`: `a` (succeeds, produces `{"q": ...}`), `b` (fails,
unrelated), `k` (`ensemble: child`, `depends_on: [a, b]`, `input_key: q`).
`k` runs and succeeds on `a`'s output alone — `b`'s failure is silently
never surfaced anywhere on `k`'s own result. Cause:
`GuardEvaluator.should_run`'s cascade rule only requires ONE dependency to
be ok (`a` is), and `_partition_by_input_key_contract` only checks the
`input_key` SOURCE (`depends_on[0]`, also `a`) — neither step consults
`b` at all. This is a real gap (an `input_key` child with extra
`depends_on` entries beyond its selection source can lose a co-dependency's
failure entirely) but is out of round-4 scope; left alone per lead
decision, recorded here so it isn't rediscovered as new.
