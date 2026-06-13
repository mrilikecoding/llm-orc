#!/bin/bash
# Coder-tier escalation live validation (BUILD loop-back #8 step 4).
# A deliberately undersized 2B cheap coder reliably bleeds form -> the ADR-041
# coder-tier escalation fires (2B cheap recovery exhausts -> 8B escalated rung).
# Simple 2-file task so the 8B rung reliably closes what 2B bleeds, making the
# full escalate-and-converge path observable in serve.log.
# Usage: run.sh <label> <N>
set -u
LBL="$1"; N="$2"
ROOT=/Users/nathangreen/Development/eddi-lab/llm-orc/scratch/escalation-validation
LOG="$ROOT/serve.log"
VALIDATE=/Users/nathangreen/Development/eddi-lab/llm-orc/scratch/spike-pi-form-adequacy-gate/validate.py
OUT="$ROOT/$LBL"; mkdir -p "$OUT"

PROMPT='Create cli.py in this directory: a command-line tool. Use argparse to
accept a numeric temperature value as a positional argument and a mutually
exclusive flag group --to-fahrenheit / --to-celsius. Convert the value with the
appropriate formula and print the result formatted to two decimals. Define a
main() function, parse args inside it, and call it under an
if __name__ == "__main__" guard. Add a module docstring and inline comments.'

for i in $(seq 1 "$N"); do
  WS="$OUT/ws_$i"; rm -rf "$WS"; mkdir -p "$WS"
  cat > "$WS/opencode.json" <<'JSON'
{ "$schema": "https://opencode.ai/config.json",
  "provider": { "llmorc": { "npm": "@ai-sdk/openai-compatible",
    "name": "llm-orc", "options": { "baseURL": "http://127.0.0.1:8765/v1", "apiKey": "sk-llmorc-local-dummy" },
    "models": { "agentic": { "name": "agentic" } } } } }
JSON
  SP="$PROMPT

(Internal run label: $LBL run $i. Ignore this line.)"
  start=$(wc -l < "$LOG")
  opencode run -m llmorc/agentic --format json --dir "$WS" "$SP" > "$OUT/run_$i.out" 2> "$OUT/run_$i.err"
  tail -n +$((start+1)) "$LOG" > "$OUT/run_$i.log"
  # escalation evidence: tier selections (2b cheap vs 8b escalated), recovery, escalation
  esc=$(grep -c "form escalation:" "$OUT/run_$i.log" 2>/dev/null || echo 0)
  rec=$(grep -c "form recovery: re-dispatch" "$OUT/run_$i.log" 2>/dev/null || echo 0)
  python3 "$VALIDATE" "$WS" > "$OUT/run_$i.validity"
  sumline=$(grep '^SUMMARY' "$OUT/run_$i.validity" | sed -E 's/^SUMMARY\t//')
  printf '%s\trun=%d\t%s\tescalations=%s\tcheap_redispatch=%s\n' "$LBL" "$i" "$sumline" "$esc" "$rec" | tee -a "$OUT/SUMMARY.tsv"
done
echo "=== $LBL DONE ===" | tee -a "$OUT/SUMMARY.tsv"
