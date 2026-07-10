#!/bin/zsh
# The conversational ladder battery — the roadmap's per-stage exit gate,
# recorded so reruns are reproducible (first recorded run: 2026-07-09 vs
# v0.18.6; earlier trajectory rows were unrecorded manual drives).
#
# Usage:
#   LADDER_REPO=/path/to/fresh-scratch-repo LADDER_OUT=/path/to/out \
#     benchmarks/agentic_serving/ladder_battery.sh
#
# Preconditions: `llm-orc serve` running and OpenCode's llm-orc provider
# pointing at it; LADDER_REPO is a fresh git repo seeded with calc.py
# (a pre-existing module the conversation never writes — turn 8's target):
#
#   def divide(a, b): raise-on-zero division
#   def percent(part, whole): divide(part, whole) * 100
#
# Scoring (strict, per the trajectory table): a turn passes only when its
# deliverable ships and is correct; honest rejects/refusals are misses and
# noted as honest. Expected-behavior exceptions: turn 9 PASSES on an honest
# refusal (the file does not exist — refusing IS the deliverable); turn 11
# (the #83 run rung) PASSES when the delegated pytest run executes and the
# verdict matches the client-side result — the honest verdict IS the
# deliverable, whatever color the suite is.
set -u
REPO=${LADDER_REPO:?set LADDER_REPO to a seeded scratch repo}
OUT=${LADDER_OUT:?set LADDER_OUT to an output dir}
mkdir -p "$OUT"
cd "$REPO"
typeset -a PROMPTS
PROMPTS=(
  "write a function that adds a todo item to a list in todo.py"
  "add a complete_todo function to todo.py that marks a todo done"
  "explain how todo.py stores its state"
  "write tests for todo.py"
  "did you see my previous query?"
  "create storage.py with save_todos and load_todos functions using json"
  "update todo.py to persist todos using storage.py"
  "write tests for existing calc.py"
  "write tests for existing phantom.py"
  "what did the first thing I asked you to build do?"
  "run the tests"
)
i=0
for p in "${PROMPTS[@]}"; do
  i=$((i+1))
  echo "=== TURN $i: $p ==="
  if [ $i -eq 1 ]; then
    timeout 600 opencode run -m llm-orc/agentic "$p" \
      > "$OUT/turn-$(printf %02d $i).out" 2>&1
  else
    timeout 600 opencode run -c -m llm-orc/agentic "$p" \
      > "$OUT/turn-$(printf %02d $i).out" 2>&1
  fi
  echo "--- exit $? ---"
done
echo "=== LADDER DONE ==="
