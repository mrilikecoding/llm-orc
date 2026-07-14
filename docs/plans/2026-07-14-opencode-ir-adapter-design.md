# WS-8 raw→IR adapter: opencode `--format json` → Transcript (#131)

**Status:** design, 2026-07-14. Arc B of the WS-8 baseline (Arc A, the
scorer, merged). Unblocked by the real captures
(`docs/plans/2026-07-13-opencode-run-captures/`).

## Problem

The scorer (`benchmarks/agentic_serving/{transcript,honesty,metrics}.py`)
scores the arm-agnostic `Transcript` IR. Nothing yet turns a real client
transcript into that IR. This is the one arm-specific seam; the scorer never
branches on arm. Build the OpenCode adapter first (it serves Arm 0 and
Arm 1, both driven through OpenCode); the Claude Code `--output-format json`
adapter for Arm 2 is a sibling, built when Arm 2 runs.

## Ground truth (the captures — do NOT guess the schema)

`opencode run --format json` emits JSONL, one event per line, keyed by
`type`. From the two captures:

- `step_start` — envelope; carries `timestamp` (ms). No content.
- `text` — assistant prose: `part.text`.
- `tool_use` — `part.tool` (name), `part.state.input` (args dict),
  `part.state.output` (result text, verbatim), `part.state.status`
  (`"completed"`).
- `step_finish` — `part.tokens.{input,output,...}`, `part.cost`,
  `part.reason` (`"tool-calls"` | `"stop"`). All tokens/cost are 0 for the
  local Arm-0 serve.

One `opencode run` invocation = one battery turn = one JSONL stream.

## Mapping to the IR

`turn_from_events(events, *, index, prompt) -> Turn`:

- **assistant_text**: concatenate every `text` event's `part.text`, in order,
  joined by `"\n"`.
- **tool_calls**: one `ToolCall` per `tool_use` event, in order:
  - `name` = `part.tool`
  - `command` = `part.state.input["command"]` when `name in {"bash","run"}`,
    else `None` (this is what `honesty._is_test_command` reads to detect a
    test run — the load-bearing field for the verification metric)
  - `path` = `input.get("filePath")` or `input.get("pattern")` or
    `input.get("path")` (read/write/glob), else `None`
  - `result_text` = `str(part.state.output)` (verbatim; never summarized —
    the honesty check pattern-matches it)
- **input_tokens / output_tokens**: sum `part.tokens.input` (fresh input) and
  `part.tokens.output + part.tokens.reasoning` (reasoning bills at the output
  rate) over all `step_finish` events. If BOTH sums are 0 (the local unbilled
  signal), set both to `None` so `metrics.turn_cost` returns `None` and Arm 0
  is $0 by construction; otherwise the summed counts. **`tokens.cache.*` is
  EXCLUDED** (cache-read/write bill at 0.1x/1.25x rates the flat `Pricing`
  can't express, and opencode's paid token shape isn't captured yet), so the
  cost figure is FRESH-token cost — a lower bound on a cache-heavy paid turn.
  Documented + pinned by test; close it in Arc D with a real paid capture
  (grow the IR/Pricing for cache).
- **tool-call dedup**: `tool_use` events are deduped by `callID` keeping the
  terminal state, so a paid stream emitting pending -> completed for one call
  is one round, not two.
- **wall_seconds**: `(max_timestamp - min_timestamp) / 1000.0` over all events
  carrying a `timestamp`; `None` when fewer than two timestamps.

`transcript_from_runs(arm, runs) -> Transcript` where `runs` is an ordered
list of `(prompt, jsonl_text)`: parse each, build a `Turn` with a 1-based
`index`, assemble.

Module: `benchmarks/agentic_serving/opencode_adapter.py`. Pure, deterministic,
no I/O in the mapping functions (a thin `parse_events(jsonl_text)` splits
lines and `json.loads` each; blank lines skipped).

## TDD fixtures

Against the REAL captures (read from
`docs/plans/2026-07-13-opencode-run-captures/`), plus synthetic edges:

- text-turn capture → a `Turn` with the prose, no tool calls, tokens `None`,
  wall ≈ 5.5s.
- tool-turn capture (glob→read→write→text) → 3 tool calls with the right
  name/path/result_text, `assistant_text == "Wrote test_metrics.py."`, tokens
  `None`.
- synthetic bash test-run event (`input.command == "pytest -q"`,
  `output == "3 passed"`) → `ToolCall.command` set, so
  `honesty.ran_verification` is True and `observed_test_result` is True (the
  adapter↔scorer wiring is the integration point that matters).
- synthetic paid-arm turn (nonzero `tokens`) → real `input_tokens` /
  `output_tokens`, so `metrics.turn_cost` is non-`None`.
- empty / blank-line-only stream → empty `Turn` (no crash).

## Exit

Adapter + tests green (mypy strict, ruff), the two real captures round-trip to
the expected IR, and one end-to-end assertion that a bash `pytest` tool event
drives `honesty.classify_turn` correctly. Then Arc C (runner + strict-score
table) can drive real Arm-0/Arm-2 sessions through adapter→scorer.
