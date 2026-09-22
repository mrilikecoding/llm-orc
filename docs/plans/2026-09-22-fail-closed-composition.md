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
