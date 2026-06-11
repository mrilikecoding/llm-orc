#!/bin/bash
# Spike η live runner — N real-OpenCode sessions against a running serve (port 8765),
# on the UNNAMED temperature-library task (no filenames → regex gate empty → judge
# path, where arm D's enumerator seeds the checklist). Classify each session by
# per-session convergence to the intended 5-deliverable set.
#
# The ETA_ARM gate lives in the SERVE process env, not here:
#   arm A (baseline):  start serve with NO ETA_ARM      -> judge sees produced-only
#   arm D:             start serve with ETA_ARM=D       -> enumerator seeds the judge
#   control:           start serve with ETA_ARM=control -> decoy seeds the judge
#
# Usage: run_eta_arm.sh <arm-label> <N> <serve.log path>
set -u
ARM="$1"; N="$2"; LOG="$3"
ROOT=/Users/nathangreen/Development/eddi-lab/llm-orc/scratch/spike-eta-deliverable-enumerator
OUT="$ROOT/live_$ARM"
mkdir -p "$OUT"
SUMMARY="$OUT/SUMMARY.tsv"
: > "$SUMMARY"

# UNNAMED task: the faithful de-named transform of the σ named task (5 deliverables
# described, zero filenames). Pre-registered in the η research log.
PROMPT='Build a small temperature-conversion library in this directory. It needs: (1) a module with three conversion functions — celsius to fahrenheit, fahrenheit to celsius, and celsius to kelvin; (2) unit tests for those conversion functions; (3) a command-line tool that converts a value given as command-line arguments; (4) tests for the command-line tool; (5) documentation explaining how to use the command-line tool. The tests must import the real module under test, the CLI must call the real conversion functions, and the docs must describe the real CLI usage.'

for i in $(seq 1 "$N"); do
  WS="$OUT/ws_$i"
  rm -rf "$WS"; mkdir -p "$WS"
  cat > "$WS/opencode.json" <<'JSON'
{ "$schema": "https://opencode.ai/config.json",
  "provider": { "llmorc": { "npm": "@ai-sdk/openai-compatible",
    "name": "llm-orc", "options": { "baseURL": "http://127.0.0.1:8765/v1", "apiKey": "sk-llmorc-local-dummy" },
    "models": { "agentic": { "name": "agentic" } } } } }
JSON
  # llm-orc derives the session id from sha256(first user message). Identical
  # prompts across headless runs collide into ONE in-memory action record (the
  # close-callback never fires for `opencode run`), bleeding produced files
  # between runs. A unique inert suffix per run forces a distinct session id ->
  # a fresh record per run.
  RUNPROMPT="$PROMPT"$'\n\n'"(Harness reference ${ARM}-${i}-${RANDOM}${RANDOM}; ignore — not a deliverable.)"
  start=$(wc -l < "$LOG")
  opencode run -m llmorc/agentic --format json --dir "$WS" "$RUNPROMPT" > "$OUT/run_$i.out" 2> "$OUT/run_$i.err"
  # serve-log lines emitted during this run
  tail -n +$((start+1)) "$LOG" > "$OUT/run_$i.serve"
  grep "turn decision:" "$OUT/run_$i.serve" | sed -E 's/.*turn decision: //' > "$OUT/run_$i.turns"
  grep -E "eta-enumerate|completeness:" "$OUT/run_$i.serve" > "$OUT/run_$i.gate" 2>/dev/null

  # produced deliverables (code/doc files the session actually wrote)
  files=$(ls "$WS" | grep -vE '^opencode.json$' | grep -E '\.(py|md|txt|cfg|toml)$' | tr '\n' ',' )
  nfiles=$(ls "$WS" | grep -vE '^opencode.json$' | grep -cE '\.(py|md|txt|cfg|toml)$')
  writes=$(grep -c "action=write" "$OUT/run_$i.turns")
  last=$(tail -1 "$OUT/run_$i.turns")
  enum=$(grep "eta-enumerate" "$OUT/run_$i.gate" | tail -1 | sed -E 's/.*eta-enumerate/eta/')

  # per-session convergence: a clean finish with the full deliverable set
  if echo "$last" | grep -q "action=finish"; then
    if [ "$nfiles" -ge 5 ]; then cls="CONVERGE"; else cls="PREMATURE"; fi
  else
    cls="OTHER"  # cap-hit / no clean finish (overrun candidate)
  fi
  printf '%s\trun=%d\tcls=%s\tnfiles=%d\twrites=%s\tfiles=[%s]\tlast=[%s]\t%s\n' \
    "$ARM" "$i" "$cls" "$nfiles" "$writes" "$files" "$last" "$enum" | tee -a "$SUMMARY"
done

echo "=== $ARM DONE ===" | tee -a "$SUMMARY"
awk -F'\t' '/cls=/{split($3,a,"="); c[a[2]]++} END{for(k in c) printf "%s: %d\n", k, c[k]}' "$SUMMARY" | tee -a "$SUMMARY"
