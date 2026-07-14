# opencode run --format json captures (2026-07-13)

Real Arm-0 captures (llm-orc serve, qwen3:8b, opencode 1.17.15) for WS-8's
raw->IR adapter and the grep->read normalizer. The design doc said do NOT
guess this schema; these are the ground truth.

- `format-json-text-turn.jsonl` — a conceptual-explain turn (no tools). Shows
  the base envelope: `step_start` / `text` (assistant text in `part.text`) /
  `step_finish` (`part.tokens.{input,output}`, `part.cost`; all 0 for local).
- `format-json-tool-turn-glob-read-write.jsonl` — a discovery build
  ("write tests for the metrics module"): glob -> read -> write -> text.
  Shows `type:"tool_use"` -> `part.tool` (name), `part.state.input` (args),
  `part.state.output` (result_text verbatim). glob output = newline-joined
  absolute paths; read output = `<path>/<type>/<content>` tagged w/ `N: ` gutter.

Wedge note: opencode run wedges under the agent's Bash sandbox; captured with
`dangerouslyDisableSandbox` + nohup-detach. See memory `opencode-run-wedge`.

STILL TO CAPTURE: grep's `part.state.output` content format (the normalizer
target). grep isn't emitted by the serve yet, so it needs a throwaway probe
(serve emits a grep tool_call) — the one remaining unknown for grep->read.
