#!/bin/zsh
# The conversational ladder battery — the roadmap's per-stage exit gate,
# recorded so reruns are reproducible (first recorded run: 2026-07-09 vs
# v0.18.6; earlier trajectory rows were unrecorded manual drives).
#
# WS-8 (#131): arm-parameterized and emitting `--format json` JSONL, so one
# script drives every OpenCode-hosted arm and `score_run.py` reads the result
# without regex-scraping human-formatted text. Arm 2 (Claude Code) is a
# DIFFERENT harness and gets its own driver; this script covers arms 0 and 1.
#
# Usage:
#   LADDER_REPO=/path/to/fresh-scratch-repo LADDER_OUT=/path/to/out \
#     [LADDER_MODEL=llm-orc/agentic] [LADDER_TIMEOUT=780] \
#     benchmarks/agentic_serving/ladder_battery.sh
#
#   LADDER_MODEL selects the arm behind the same OpenCode client:
#     arm 0 (the serve, free) : llm-orc/agentic          [default]
#     arm 1 (harness held constant, PAID): anthropic/claude-haiku-4-5
#                                          anthropic/claude-sonnet-5
#
# Output per turn N: `turn-NN.jsonl` (raw JSON events -> score_run.py),
# `turn-NN.err` (stderr, kept OUT of the JSONL so a warning can't corrupt the
# event stream), and `truth-NN.json` (workspace ground truth), plus
# `exits.tsv` (turn, exit code) for post-hoc triage.
#
# WHY ground truth is captured per turn, not derived from the transcript: the
# strict rule is that a turn passes only when its deliverable ships AND IS
# CORRECT (turn 11's verdict must match the client-side result; turns 12/13
# require green client-side). A transcript can only show that a `write`
# happened — never that what landed is right. So each turn is followed by a
# real pytest run against the real workspace, recorded as the arm-independent
# fact the strict table scores against. This is arm-BLIND by construction: it
# judges what reached disk, identically for every arm, instead of trusting
# what an arm's own prose claims — which is the whole bet under measurement.
# `seeded` runs test_buggy.py ALONE (turn 13 is scored on the seeded target
# only, so earlier turns' residue cannot cascade into that rung).
#
# A turn that produces NO events leaves NO `turn-NN.jsonl` behind: score_run's
# `_load_runs` reads an ABSENT file as a client-side death (`missing_turns`)
# and a present-but-empty one as an honest empty turn. A shell redirect
# creates the file on open even when the turn dies, so a file with no
# non-whitespace content is removed here — otherwise every death would silently
# read as honesty, which is the exact miscount `missing_turns` exists to
# prevent. Testing for non-whitespace rather than nonzero SIZE matters: a client
# that flushes a bare newline before dying would slip through a `[ -s ]` guard
# and score as an honest empty turn.
#
# Preconditions: GNU coreutils `timeout` on PATH (stock macOS has none, and
# Homebrew installs it as `gtimeout` unless the coreutils gnubin dir is on
# PATH). Without it every turn exits 127 immediately and all 13 turns register
# as client-side deaths — loud and self-diagnosing via `exits.tsv`, but check
# here first. For arm 0, `llm-orc serve` running and OpenCode's llm-orc
# provider pointing at it; LADDER_REPO is a fresh git repo seeded with calc.py
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
MODEL=${LADDER_MODEL:-llm-orc/agentic}
TIMEOUT=${LADDER_TIMEOUT:-780}
# The ground-truth runner. Must be able to run the fixture repo's suite; it is
# the same command for every arm, so the truth it records is arm-independent.
PYTEST=${LADDER_PYTEST:-pytest}
# llm-orc checkout root (this script lives in benchmarks/agentic_serving/), so
# the oracle module is importable while the battery's cwd is the fixture repo.
SRCROOT=${0:A:h:h:h}
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
echo "=== LADDER START: model=$MODEL repo=$REPO out=$OUT ==="
: > "$OUT/exits.tsv"
: > "$OUT/oracle-exits.tsv"
i=0
for p in "${PROMPTS[@]}"; do
  i=$((i+1))
  n=$(printf %02d $i)
  echo "=== TURN $i: $p ==="
  # Turn 1 opens the session; every later turn continues it (-c) — the
  # battery is ONE conversation, which is what the recall/memory rungs test.
  if [ $i -eq 1 ]; then
    timeout "$TIMEOUT" opencode run --format json -m "$MODEL" "$p" \
      > "$OUT/turn-$n.jsonl" 2> "$OUT/turn-$n.err"
  else
    timeout "$TIMEOUT" opencode run --format json -c -m "$MODEL" "$p" \
      > "$OUT/turn-$n.jsonl" 2> "$OUT/turn-$n.err"
  fi
  rc=$?
  # No events => client-side death: leave the file absent (see header). Tests
  # for non-whitespace content, not size: a bare newline flushed on death would
  # pass a `[ -s ]` guard and then score as an honest empty turn.
  grep -q '[^[:space:]]' "$OUT/turn-$n.jsonl" 2>/dev/null || rm -f "$OUT/turn-$n.jsonl"
  printf '%s\t%s\n' "$n" "$rc" >> "$OUT/exits.tsv"
  # Workspace ground truth AFTER this turn (see header): the full suite, the
  # seeded target alone, and the file manifest. Caches are excluded from the
  # manifest and swept so the probe leaves no trace the next turn can read.
  # Exit codes come from pytest itself: a pipeline's $? is the LAST command's
  # (tail always succeeds), so each run goes to a file and is tailed after.
  (cd "$REPO" && timeout 120 $PYTEST -q > "$OUT/.suite-$n.txt" 2>&1)
  suite_rc=$?
  (cd "$REPO" && timeout 120 $PYTEST -q test_buggy.py > "$OUT/.seeded-$n.txt" 2>&1)
  seeded_rc=$?
  suite=$(tail -12 "$OUT/.suite-$n.txt"); rm -f "$OUT/.suite-$n.txt"
  seeded=$(tail -8 "$OUT/.seeded-$n.txt"); rm -f "$OUT/.seeded-$n.txt"
  # One filename per LINE into a file, never a space-joined string: a model can
  # write "my notes.py", and a space-joined list round-tripped through
  # str.split() would silently corrupt it into two entries.
  (cd "$REPO" && git ls-files --others --cached --exclude-standard \
    | grep -vE '(^\.|/\.|__pycache__|\.pyc$)' | sort) > "$OUT/.files-$n.txt"
  # The turn's hidden correctness oracle, run NOW against the workspace this
  # turn produced. It cannot be deferred to the end of the run: later turns
  # mutate files (turn 13 rewrites buggy.py), so a post-hoc probe would judge a
  # turn against a workspace it never saw. The module prints `null` itself for a
  # turn with no oracle, so there is deliberately NO `|| echo null` fallback
  # here: that would make a real crash in the measurement instrument
  # indistinguishable from "no oracle by design". A crash leaves a non-JSON
  # stdout (recorded as `oracle: null` with the reason in oracle-NN.err) and a
  # nonzero code visible in oracle-exits.tsv.
  oracle=$(cd "$SRCROOT" && uv run python -m benchmarks.agentic_serving.oracles \
    "$i" "$REPO" 2> "$OUT/oracle-$n.err")
  printf '%s\t%s\n' "$n" "$?" >> "$OUT/oracle-exits.tsv"
  [ -s "$OUT/oracle-$n.err" ] || rm -f "$OUT/oracle-$n.err"
  (cd "$REPO" && rm -rf .pytest_cache && find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null)
  python3 - "$OUT/truth-$n.json" "$n" "$suite_rc" "$seeded_rc" \
    "$OUT/.files-$n.txt" "$suite" "$seeded" "$oracle" <<'PY'
import json, sys
path, turn, suite_rc, seeded_rc, files_path, suite, seeded, oracle = sys.argv[1:9]
with open(files_path) as handle:
    files = handle.read().splitlines()
try:
    oracle_verdict = json.loads(oracle)
except ValueError:
    # A crash in the oracle module itself, not "no oracle for this turn". The
    # reason is in oracle-NN.err and the code in oracle-exits.tsv.
    oracle_verdict = None
json.dump({
    "turn": int(turn),
    "files": files,
    "suite": {"rc": int(suite_rc), "tail": suite},
    "seeded": {"rc": int(seeded_rc), "tail": seeded},
    "oracle": oracle_verdict,
}, open(path, "w"), indent=1)
PY
  rm -f "$OUT/.files-$n.txt"
  echo "--- exit $rc | suite_rc $suite_rc | seeded_rc $seeded_rc ---"
done
echo "=== LADDER DONE ==="
