#!/bin/bash
# Spike σ runner — N live OpenCode sessions against the running serve (port 8765),
# classifying each by per-session completion. Usage: run_arm.sh <arm> <N>
set -u
ARM="$1"; N="$2"
ROOT=/Users/nathangreen/Development/eddi-lab/llm-orc/scratch/spike-sigma-premature-finish
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
  start=$(wc -l < "$LOG")
  opencode run -m llmorc/agentic --format json --dir "$WS" "$PROMPT" > "$OUT/run_$i.out" 2> "$OUT/run_$i.err"
  # turn decisions emitted during this run
  tail -n +$((start+1)) "$LOG" | grep "turn decision:" | sed -E 's/.*turn decision: //' > "$OUT/run_$i.turns"
  # classify
  files=$(ls "$WS" | grep -vE '^opencode.json$' | grep -cE '\.(py|md)$')
  writes=$(grep -c "action=write" "$OUT/run_$i.turns")
  anchors=$(grep -c "anchor=true" "$OUT/run_$i.turns")
  last=$(tail -1 "$OUT/run_$i.turns")
  if echo "$last" | grep -q "judgment_verdict=COMPLETE action=finish"; then
    if [ "$files" -ge 5 ]; then cls="COMPLETE"; else cls="EARLY_COMPLETE"; fi
  elif echo "$last" | grep -q "judgment_verdict=REMAINING action=finish"; then
    cls="PREMATURE"
  else
    cls="OTHER"
  fi
  printf '%s\trun=%d\tcls=%s\tfiles=%d\twrites=%s\tanchors=%s\tlast=[%s]\n' \
    "$ARM" "$i" "$cls" "$files" "$writes" "$anchors" "$last" | tee -a "$SUMMARY"
done

echo "=== $ARM DONE ===" | tee -a "$SUMMARY"
awk -F'\t' '/cls=/{split($3,a,"="); c[a[2]]++} END{for(k in c) printf "%s: %d\n", k, c[k]}' "$SUMMARY" | tee -a "$SUMMARY"
