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
