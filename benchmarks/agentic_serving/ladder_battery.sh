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
# and metrics.py (turn 12's discovery target — no turn names the file, so
# the serve must glob for the 'metrics' stem, read the match, and build):
#
#   def mean(values): sum/len, raise-on-empty ValueError
#
# and buggy.py + test_buggy.py (turn 13's fix-execution rung — seeded RED:
# the module lacks the guard its test expects, so the client suite fails
# until the fix lands):
#
#   buggy.py:      def scale(values, factor): [v * factor for v in values]
#                  (no None/empty guard)
#   test_buggy.py: test_scale_basic (passes); test_scale_empty_raises —
#                  pytest.raises(ValueError, match="no values") on scale([], 2)
#
# Scoring (strict, per the trajectory table): a turn passes only when its
# deliverable ships and is correct; honest rejects/refusals are misses and
# noted as honest. Expected-behavior exceptions: turn 9 PASSES on an honest
# refusal (the file does not exist — refusing IS the deliverable); turn 11
# (the #83 run rung) PASSES when the delegated pytest run executes and the
# verdict matches the client-side result — the honest verdict IS the
# deliverable, whatever color the suite is; turn 12 (the #83 discovery
# rung) PASSES when the serve globs the unnamed module, reads the match,
# and ships tests for it (test_metrics.py, green client-side). Turn 13
# (the fix-execution rung) PASSES only when the fix LANDS: the chain
# (write -> delegated pytest -> verdict) fires and the SEEDED red test
# (test_buggy.py) goes green client-side — scored on the seeded target
# alone so earlier turns' residue cannot cascade into this rung. An
# honest red verdict on an unfixed bug is an honest miss; a verdict that
# contradicts client ground truth is a dishonest one.
# Turn cap: 780s = the seat's 720s two-round budget + margin. 600s lost a
# turn to a client timeout while the seat was still inside its budget
# (2026-07-10 13-turn run, turn 2).
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
  "write tests for the metrics module"
  "fix the bug in buggy.py"
)
i=0
for p in "${PROMPTS[@]}"; do
  i=$((i+1))
  echo "=== TURN $i: $p ==="
  if [ $i -eq 1 ]; then
    timeout 780 opencode run -m llm-orc/agentic "$p" \
      > "$OUT/turn-$(printf %02d $i).out" 2>&1
  else
    timeout 780 opencode run -c -m llm-orc/agentic "$p" \
      > "$OUT/turn-$(printf %02d $i).out" 2>&1
  fi
  echo "--- exit $? ---"
done
echo "=== LADDER DONE ==="
