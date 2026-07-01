# Spike Ω-2 — Substrate-as-state, multi-turn

**Status:** GATE PASSED 2026-06-29 (not HARD PASS). See
`docs/agentic-serving/proposals/ensemble-spike-sequence.md` §4.

## Result

The multi-turn ensemble runs across 2–3 turns through the substrate
state mechanism with **no engine primitive added**. Per-turn latency
~70–80s with the lean single-LLM code-generator-omega fork, vs. 200s+
with the production 3-stage code-generator chain. Cross-file substrate
threading works (parse reads updated state, dispatch-shim uses the
correct next file). But the ensemble form lands only 1 of 3 requested
files on the strict-validate task — the LLM hits form-directive failures
(shell-usage examples, mixed indentation) that the bespoke's FormGate
+self-healing recovery (ADR-041) would have re-dispatched. The gate
passes on substrate threading, not on end-to-end completion parity.

## Shape (built)

- `.llm-orc/ensembles/spike-omega/agent-turn-omega2.yaml` — the ensemble
- `.llm-orc/ensembles/spike-omega/code-generator-omega.yaml` — lean single-coder fork
- `.llm-orc/scripts/spike-omega-2/parse.py` — substrate-aware parse script
- `.llm-orc/scripts/spike-omega-2/marshal.py` — substrate-updating marshal script
- `.llm-orc/scripts/spike-omega/dispatch_shim.py` — extended (parse-aware + deterministic next_file)
- `.llm-orc/scripts/spike-omega/validate.py` — extended (filename-strip + shell-prefix strip)
- `scratch/spike-omega-2/run_multi.py` — the multi-turn harness
- `scratch/spike-omega-2/out/session_state.json` — substrate state (updated each turn)
- `scratch/spike-omega-2/out/produced/converters.py` — landed deliverable from a passing turn

## Run

```
uv run python scratch/spike-omega-2/run_multi.py
```

Needs Ollama + `qwen3:14b` (planner) + `qwen3:8b` (code-generator coder).
The local config was extended (see `glue inventory` below): the default
`performance.script_cache.enabled` is now `false` (see Finding #2), and
`performance.execution.default_timeout` is now `300` (qwen3's thinking
variant needs longer than the default 60s on this rig).

## Gate — PASS (HARD PASS not reached)

- Ensemble-form multi-turn holds with no engine primitive added ✓
- Trajectories match bespoke partially — substrate state threading ✓
  (turn 2 correctly picks `cli.py` after `converters.py` lands), but
  end-to-end completion does not (the form-directive falls to D2b
  residuals; no recovery logic in the ensemble form). 1 of 3 files lands.
- The substrate-+-scripts reframe HOLDS — primitive 3 (cross-run state)
  collapses cleanly into parse/marshal scripts with filesystem access.
- The form-directive cruft (D2b) survives multi-turn but the ensemble form
  has no recovery path; recovery becomes a Ω-3+ consideration.

## Findings (six real ones)

### #1 — Substrate threading works; primitive 3 collapses

The clique primitive. parse-stage reads `session_state.json`, emits
plan_queue head as `next_file`; marshal writes the updated state file
with file-path removed from queue and appended to produced. The
harness loops calls to `executor.execute` with byte-identical input.

State threading is a parse-script-read + marshal-script-write pattern,
not an engine state-machine primitive. The §2 reframe bet HOLDS.

### #2 — The script cache default breaks substrate-driven multi-turn

A real, fresh Ω-2 finding. The engine's `ScriptCache` (`cache.py:71`)
keys on `script_content + input_data + parameters`. The harness sends
identical input across turns — `{task, substrate_path, last_tool_result}` —
but the substrate file CONTENT changes. Turn 2's parse py served turn 1's
cached output, returning the wrong next_file.

Workaround now in `.llm-orc/config.yaml`: `performance.script_cache.enabled: false`.
Two real fixes the engine could provide:
- **include substrate hash in the cache key** (an L0 extension; small but
  real surface), OR
- **per-ensemble cache opt-out** (a YAML flag `script_cache: false` on the
  ensemble or on specific script agents).

This is fresh engine consideration the §6 doc missed.

### #3 — Plan-stage queue deviance; pragmatic fix

The plan LLM ignored the parse-stage's `plan_input` ordering and
re-chose files on early runs. The bespoke LoopDriver's deterministic
completeness gate (ADR-040) picks the next file deterministically FROM
the plan_queue head; the LLM only elaborates the brief.

The pragmatic fix in the ensemble form: parse emits `next_file` explicitly,
and dispatch-shim uses parse's `next_file` overriding the plan LLM's
choice. This matches the bespoke's deterministic-first principle. It
costs one extra dependency edge (`dispatch-shim depends on [parse, plan]`)
and removes the planner's freedom to deviate.

Real Ω-2 finding: **the planner LLM cannot be trusted to pick the next
file when given partial freedom**; the ensemble form must structure it
out the same way the bespoke did. The cruft transfers cleanly.

### #4 — D2b residuals recur; the gate catches but no recovery path

The coder's output occasionally includes shell usage examples leading
the file (`python cli.py convert c-to-f <value>`), filenames on the
first line, mixed-indent Python. The bespoke's FormGate + self-healing
re-dispatch (ADR-041) recovers these by re-delegating on the next turn.
The ensemble form's `validate.py` is hardening (filename-strip,
shell-prefix-strip) but it does not auto-recover; the harness just stops.

Real Ω-2 finding: the ensemble form's recovery loop is not present yet.
This is "the third cruft layer" alongside D1 extraction and D2b
directive — bespoke's self-healing feature would translate to an inner
loop inside the agent-turn ensemble that re-invokes code-generator on
validate failure. Hard to express cleanly without either (a) a
conditional-branch primitive or (b) the adapter simulating it via a
retry wrapper that re-invokes the ensemble with a "production-rejected:
re-emit only this file" input. Adapter-side retry is the cheaper path.

### #5 — Latency with substrate threading

~70–80s per turn with the lean single-coder code-generator-omega. The
production code-generator (coder → critic → synthesizer chain) hits
200–220s per turn — over the bespoke's typical 5–30s per turn by a wide
margin. The substrate write/read adds negligible (<1%) overhead; latency
is dominated by the inner LLM chain.

Real Ω-2 finding: with the existing production code-generator ensemble,
multi-form composition per turn is 6× the bespoke's per-turn cost. The
bespoke uses the same code-generator via `invoke_ensemble` but only one
LLM call per turn for trivial work, the ensemble form runs the full
3-LLM pipeline. **Composition cost is the architectural tax** that the
form directive + recovery + final valid gate have to pay for reliability.

### #6 — Engine asymmetry (re-stated from Ω-1)

`_find_ensemble_in_dirs` (engine-internal, non-recursive) ≠
`EnsembleLoader.find_ensemble` (production, recursive). The symlink
workaround from Ω-1 still applies: top-level symlink to `agentic-serving/`
X.yaml AND `spike-omega/code-generator-omega.yaml`. For Ω-3+ this needs
proper resolution — either adapter-side (the harness resolves the path
and passes raw config) or engine-side (recursive walking). Cheaper than
the four §6 primitives.

## Glue inventory (bespoke shims; transferable to Ω-3+)

- `parse.py` — substrate reader + deterministic next_file emitter.
- `dispatch_shim.py` — deterministic next_file from parse, form directive append.
- `validate.py` — D1 extraction + FormGate (filename-strip, shell-prefix
  strip, fenced block extraction, fallback to `def`/`import` start).
- `marshal.py` — write tool_call emission + substrate state update.
- `.llm-orc/config.yaml` — `script_cache.enabled: false` (Finding #2
  workaround); `execution.default_timeout: 300` (latency budget).
- `.llm-orc/ensembles/code-generator.yaml` + `code-generator-omega.yaml`
  symlinks — Finding #6 workaround (asymmetric ensemble resolution).
- HTTP adapter — deferred; harness acts as adapter for Ω-2.
- Dynamic dispatch — sidestepped: code-generator-omega fixed-target.
- Recovery loop — not implemented; Ω-3 candidate (adapter-side retry).

## What this gate unblocked

- Ω-3 (library reflection + dynamic dispatch) can proceed. The library
  reflection primitive (§6 primitive 1, "library-as-data to a stage") is
  testable directly: parse script reads `.llm-orc/ensembles/` and
  produces a capabilities list — no engine primitive needed (confirmed
  by the §2 reframe).
- The script-cache finding (#2) becomes a candidate for either a small
  L0 extension (cache key includes substrate hash) or a per-ensemble
  opt-out YAML flag. Documented in the Ω-3 / Ω-5 candidate work.
- The plan-stage-determinism finding (#3) becomes a stable pattern: any
  selectable decision (which file, which capability) should be
  deterministic-from-substrate when an LLM might deviate; the LLM
  elaborates brevity only.

## What stays open for Ω-3

- Library reflection: parse script reads `.llm-orc/ensembles/`, scores
  capabilities against the task, picks the right capability ensemble.
  With ONE capability (code-generator-omega) the case is trivial; the
  Ω-3 spike adds a second capability (prose-improver for README.md) to
  test dynamic routing.
- Dynamic-dispatch fork: whether dispatch-via-adapter (the §8 boundary)
  stays clean OR whether the engine generalizes a script-initiated
  dispatch primitive. Ω-2 kept dispatch adapter-side via the static
  YAML reference; Ω-3 needs the choice between dynamic ensembles to be
  made at runtime.

## What stays open past Ω-3

- Recovery loop (Finding #4) — adapter-side retry on validate failure.
- Cross-file coherence ground truth — converters.py was produced on the
  first turn (no siblings) so the substrate-input siblings-text path
  never kicked in. Ω-3 with multi-turn ordering of dependent files
  would test it.