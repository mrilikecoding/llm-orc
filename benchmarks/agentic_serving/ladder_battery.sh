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
# `exits.tsv` (turn, exit code) for post-hoc triage. `truth-00.json` is the
# SEEDED-repo baseline captured before turn 1, so turn 1's manifest diff has
# something to diff against.
#
# WHY ground truth is captured per turn, not derived from the transcript: the
# strict rule is that a turn passes only when its deliverable ships AND IS
# CORRECT (turn 11's verdict must match the client-side result; turns 12/13
# require green client-side). A transcript can only show that a `write`
# happened — never that what landed is right. So each turn is followed by a
# real pytest run over the workspace's files, recorded as the arm-independent
# fact the strict table scores against. This is arm-BLIND by construction: it
# judges what reached disk, identically for every arm, instead of trusting
# what an arm's own prose claims — which is the whole bet under measurement.
# `seeded` runs test_buggy.py ALONE (turn 13 is scored on the seeded target
# only, so earlier turns' residue cannot cascade into that rung).
#
# The truth pytest runs execute in a THROWAWAY COPY of the workspace, never in
# the live one: they run arm-authored tests, which import arm-authored modules,
# which execute arm-authored module-level code — exactly what oracles.py
# forbids for its own probes. Run live, a test that writes todos.json (routine
# for storage-adjacent tests) would land in the next manifest attributed to
# the arm.
#
# The manifest is HASHED (path<TAB>sha256 per line): names alone cannot show
# that a turn EDITED an existing file, and the scorer derives "shipped" from
# the manifest diff — the only channel that means the same thing for every arm
# (a write tool, a bash heredoc, and a patch all land here identically).
#
# A turn that produces NO events (absent file, or present but eventless — a
# timeout can flush a partial line before dying) is a client-side death:
# score_run's `_load_runs` tests EVENTS, not bytes, so the file is left as
# captured rather than post-processed here.
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
# Absolutize BOTH: a relative $OUT fails loudly at turn 1, but a relative
# $REPO fails QUIETLY — the initial cd succeeds from the invocation cwd, then
# every truth subshell's cd resolves against the repo itself and records
# suite.rc 1 (indistinguishable from a red suite) through a whole paid run,
# while feeding the oracle a bad path.
REPO=${REPO:A}
OUT=${OUT:A}
# The manifest is git-derived; without a repo, `git ls-files` dies but the
# pipeline still emits an empty-yet-valid manifest, so every truth file
# records "manifest": {} and every oracled turn scores not_shipped through a
# whole paid run — the same quiet-corruption family as a relative $REPO.
git -C "$REPO" rev-parse --is-inside-work-tree > /dev/null 2>&1 \
  || { echo "LADDER_REPO must be a seeded GIT repo: $REPO"; exit 1; }
MODEL=${LADDER_MODEL:-llm-orc/agentic}
TIMEOUT=${LADDER_TIMEOUT:-780}
# The ground-truth runner. Must be able to run the fixture repo's suite; it is
# the same command for every arm, so the truth it records is arm-independent.
# Expanded with ${=PYTEST}: zsh does not word-split parameters, so a
# multi-word value ("uv run pytest") would otherwise exec a single
# 127-everywhere command named "uv run pytest".
PYTEST=${LADDER_PYTEST:-pytest}
# llm-orc checkout root (this script lives in benchmarks/agentic_serving/), so
# the oracle module is importable while the battery's cwd is the fixture repo.
SRCROOT=${0:A:h:h:h}
mkdir -p "$OUT"
cd "$REPO"

# ALL ground-truth capture (hashed manifest, truth pytest, oracle,
# contamination, truth-NN.json) lives in capture_truth.sh, SHARED with every
# other arm's driver so the truth substrate cannot drift between arms.
CAPTURE="$SRCROOT/benchmarks/agentic_serving/capture_truth.sh"
capture_truth() {
  TRUTH_REPO="$REPO" TRUTH_OUT="$OUT" TRUTH_PYTEST="$PYTEST" \
    zsh "$CAPTURE" "$1"
}
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
# Baseline manifest of the SEEDED repo, before any turn runs: turn 1's
# shipped-detection diffs against this.
capture_truth 0
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
  printf '%s\t%s\n' "$n" "$rc" >> "$OUT/exits.tsv"
  # Workspace ground truth AFTER this turn — the shared capture script (see
  # its header for the design rationale: throwaway-copy pytest, hashed
  # manifest, oracle-now-not-post-hoc, post-oracle contamination record).
  capture_truth "$i"
  echo "--- exit $rc ---"
done
echo "=== LADDER DONE ==="
