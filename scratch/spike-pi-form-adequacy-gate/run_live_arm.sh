#!/bin/bash
# Spike π live-arm runner (Fork 3) — N live OpenCode sessions against the
# running serve (port 8765). Adapted from the σ run_arm.sh proven driver.
#
# Per-session classifier: all-files-valid (validate.py — .py→ast.parse,
# .json→json.loads, .md→pass). Captures, per session, the serve.log slice
# (turn decisions, dispatch starts) + the opencode JSON output (where the
# `[dispatch failed: ...]` refusal text lands) + per-file validity — enough
# to reconstruct the recovery-loop trace (P2-C) and the re-dispatch-success
# control post-hoc.
#
# The parse-gate flag (LLMORC_SPIKE_PI_GATE=parse) lives on the SERVE
# process, NOT here. Cell A = serve launched without it; Cell B = with it.
# Usage: run_live_arm.sh <arm-label> <N>   e.g. run_live_arm.sh B_live 5
set -u
ARM="$1"; N="$2"
ROOT=/Users/nathangreen/Development/eddi-lab/llm-orc/scratch/spike-pi-form-adequacy-gate
LOG="$ROOT/serve.log"
OUT="$ROOT/$ARM"
mkdir -p "$OUT"
SUMMARY="$OUT/SUMMARY.tsv"
: > "$SUMMARY"

PROMPT='Build a small temperature-conversion library in this directory. Create these five files:
1. converters.py with three functions: celsius_to_fahrenheit, fahrenheit_to_celsius, celsius_to_kelvin
2. test_converters.py with unit tests for converters.py
3. cli.py, a command-line tool that imports converters.py and converts a value given as command-line arguments
4. test_cli.py with tests for cli.py
5. README.md documenting how to use the CLI

The tests must import the real module under test, cli.py must call the real functions in converters.py, and the README must document the real CLI usage.'

for i in $(seq 1 "$N"); do
  WS="$OUT/ws_$i"
  rm -rf "$WS"; mkdir -p "$WS"
  cat > "$WS/opencode.json" <<'JSON'
{ "$schema": "https://opencode.ai/config.json",
  "provider": { "llmorc": { "npm": "@ai-sdk/openai-compatible",
    "name": "llm-orc", "options": { "baseURL": "http://127.0.0.1:8765/v1", "apiKey": "sk-llmorc-local-dummy" },
    "models": { "agentic": { "name": "agentic" } } } } }
JSON
  # Unique per-session marker → unique serve session_id (the sha256 of the
  # first message). Without it every session collides on one process-scoped
  # action record, and runs 2..N inherit run 1's "produced" files and finish
  # immediately (the η harness artifact — `validate-against-real-client` lesson,
  # cycle-status Spike η §"LIVE BASELINE corrected"). The marker is inert: a
  # trailing note the task ignores.
  SESSION_PROMPT="$PROMPT

(Internal run label: $ARM run $i. This line is not part of the task; ignore it.)"
  start=$(wc -l < "$LOG")
  opencode run -m llmorc/agentic --format json --dir "$WS" "$SESSION_PROMPT" > "$OUT/run_$i.out" 2> "$OUT/run_$i.err"
  # serve.log slice for this run (turn decisions, dispatch starts, etc.)
  tail -n +$((start+1)) "$LOG" > "$OUT/run_$i.log"
  grep "turn decision:" "$OUT/run_$i.log" | sed -E 's/.*turn decision: //' > "$OUT/run_$i.turns"
  # gate refusals — the `[dispatch failed: <path>: not valid ...]` text lands
  # in the client stream (run_$i.out); also scan the serve slice defensively.
  refusals=$(cat "$OUT/run_$i.out" "$OUT/run_$i.log" 2>/dev/null \
    | grep -oE "dispatch failed: [^]\"]*(not valid|markdown fence)[^]\"]*" | sort -u | wc -l | tr -d ' ')
  # validity of produced files
  python3 "$ROOT/validate.py" "$WS" > "$OUT/run_$i.validity"
  sumline=$(grep '^SUMMARY' "$OUT/run_$i.validity" | sed -E 's/^SUMMARY\t//')
  printf '%s\trun=%d\t%s\trefusals=%s\n' "$ARM" "$i" "$sumline" "$refusals" | tee -a "$SUMMARY"
done

echo "=== $ARM DONE ===" | tee -a "$SUMMARY"
awk -F'\t' '/all_valid=/{for(j=1;j<=NF;j++){if($j ~ /^all_valid=/){split($j,a,"=");c[a[2]]++}}} END{for(k in c) printf "all_valid=%s: %d\n", k, c[k]}' "$SUMMARY" | tee -a "$SUMMARY"
