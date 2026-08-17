#!/bin/zsh
# The #138 volume ladder driver (design:
# docs/plans/2026-08-15-138-volume-instrument-design.md).
#
# One LEVEL per invocation of the client, each in a FRESH workspace holding
# only that level's modules and their seeded tests, each its own session (no
# -c). Levels are independent by construction: nothing an arm did at L2 can
# reach L3, and every level's baseline is the seed itself.
#
# Usage:
#   VOLUME_OUT=/path/to/out [VOLUME_MODEL=llm-orc/agentic] \
#     [VOLUME_LEVELS="1 2 3 5"] [VOLUME_DRY_RUN=1] \
#     benchmarks/agentic_serving/volume_battery.sh
#
#   VOLUME_MODEL selects the arm behind the same OpenCode client:
#     arm 0 (the serve, free) : llm-orc/agentic          [default]
#     arm 1 (harness held constant, PAID): anthropic/claude-haiku-4-5
#                                          anthropic/claude-sonnet-5
#   Arm 2 (Claude Code) is a different harness and follows the documented
#   per-level protocol instead; it feeds the SAME volume_truth capture.
#
# TIMEOUTS SCALE WITH LEVEL. The ladder's flat 780s cap was calibrated to one
# task; an L5 turn is five fix-and-test cycles. A SIGTERM mid-work flushes
# partial events, which the scorer would otherwise read as shipped-unverified
# at exactly the largest level — the direction that FABRICATES the hypothesis
# under measurement. Cap = BASE + PER_MODULE * level, and a level killed at
# the deadline (exit 124) is recorded so the scorer can censor it rather than
# score it.
#
# VOLUME_DRY_RUN=1 exercises every part of the harness except the client
# (fixture generation, timeout arithmetic, truth capture, exits.tsv) against
# an untouched workspace. Free, and the design requires it before any paid
# run: it proves the plumbing without spending a token.
#
# Preconditions: GNU coreutils `timeout` on PATH (stock macOS has none;
# Homebrew installs it as `gtimeout` unless the gnubin dir is on PATH).
# Without it every level exits 127 instantly and all levels register as
# client-side deaths — loud and self-diagnosing via exits.tsv, but check here
# first. For arm 0, `llm-orc serve` running with OpenCode's llm-orc provider
# pointing at it, and the serve started with the venv bin on PATH (its script
# agents run bare `python3`; see the loop protocol's instrument rules).
set -u
OUT=${VOLUME_OUT:?set VOLUME_OUT to an output dir}
OUT=${OUT:A}
MODEL=${VOLUME_MODEL:-llm-orc/agentic}
DRY_RUN=${VOLUME_DRY_RUN:-0}
BASE_TIMEOUT=${VOLUME_BASE_TIMEOUT:-300}
PER_MODULE_TIMEOUT=${VOLUME_PER_MODULE_TIMEOUT:-480}
typeset -a LEVELS
LEVELS=(${=VOLUME_LEVELS:-1 2 3 5})

# Prior artifacts are evidence. Refuse a dirty OUT; moving them aside must be
# a deliberate act, never an overwrite (the ladder's retention rule).
mkdir -p "$OUT"
[[ -z "$(ls -A "$OUT" 2>/dev/null)" ]] \
  || { echo "VOLUME_OUT is not empty; move the prior run's artifacts aside first: $OUT"; exit 1; }

command -v timeout > /dev/null 2>&1 \
  || { echo "GNU 'timeout' not on PATH (brew install coreutils, or use gnubin)"; exit 1; }
if [[ "$DRY_RUN" != "1" ]]; then
  command -v opencode > /dev/null 2>&1 \
    || { echo "'opencode' not on PATH"; exit 1; }
fi

# llm-orc checkout root (this script lives in benchmarks/agentic_serving/), so
# the fixture/truth modules are importable while the client's cwd is the
# level workspace.
SRCROOT=${0:A:h:h:h}
PY=${VOLUME_PYTHON:-python3}
# The ground-truth test runner, threaded into volume_truth. Same command for
# every arm, so the truth it records is arm-independent (the ladder's
# LADDER_PYTEST seam). Empty means "volume_truth's own interpreter -m pytest".
PYTEST=${VOLUME_PYTEST:-}

# A truth runner that cannot run pytest records NO verdict for any module, and
# the scorer marks those UNSCORED rather than broken — but a whole run of
# unscored cells is a wasted (possibly paid) run. Refuse up front. This is the
# ladder's quiet-corruption lesson: the failure was silent and survived a
# whole run before anyone noticed.
if [[ -n "$PYTEST" ]]; then
  ${=PYTEST} --version > /dev/null 2>&1 \
    || { echo "VOLUME_PYTEST cannot run ('$PYTEST --version' failed)"; exit 1; }
else
  "$PY" -c 'import pytest' > /dev/null 2>&1 \
    || { echo "'$PY' has no pytest; set VOLUME_PYTHON or VOLUME_PYTEST"; exit 1; }
fi

echo "=== VOLUME LADDER START: model=$MODEL levels=$LEVELS out=$OUT dry_run=$DRY_RUN ==="
: > "$OUT/exits.tsv"

for level in "${LEVELS[@]}"; do
  ws="$OUT/ws-L$level"
  cap=$((BASE_TIMEOUT + PER_MODULE_TIMEOUT * level))
  prompt=$(cd "$SRCROOT" && "$PY" -c "
from benchmarks.agentic_serving.volume_fixture import VOLUME_PROMPTS
print(VOLUME_PROMPTS[$level])
") || { echo "could not read the level-$level prompt"; exit 1; }

  (cd "$SRCROOT" && "$PY" -m benchmarks.agentic_serving.volume_fixture \
     --level "$level" --dest "$ws" --verify > "$OUT/seed-L$level.tsv") \
    || { echo "fixture generation failed for level $level"; exit 1; }

  echo "=== LEVEL $level (cap ${cap}s): $prompt ==="
  if [[ "$DRY_RUN" == "1" ]]; then
    : > "$OUT/turn-L$level.jsonl"
    : > "$OUT/turn-L$level.err"
    rc=0
  else
    (cd "$ws" && timeout "$cap" opencode run --format json -m "$MODEL" "$prompt") \
      > "$OUT/turn-L$level.jsonl" 2> "$OUT/turn-L$level.err"
    rc=$?
  fi
  printf '%s\t%s\n' "L$level" "$rc" >> "$OUT/exits.tsv"

  # Ground truth AFTER the level, from the SHARED capture (hashed manifest,
  # per-module seeded rc and hidden oracle, both in a throwaway copy).
  # A capture failure does not abort the run — the remaining levels are still
  # worth collecting, and every level's workspace persists under $OUT/ws-L* so
  # capture can be re-run post hoc — but it is recorded and the script exits
  # nonzero at the end, so it cannot be missed in a scrollback.
  (cd "$SRCROOT" && "$PY" -m benchmarks.agentic_serving.volume_truth \
     --workspace "$ws" --level "$level" --exit-code "$rc" --out "$OUT" \
     --pytest "$PYTEST") \
    || { echo "WARNING: truth capture failed for level $level"; \
         printf 'L%s\n' "$level" >> "$OUT/truth-failures.txt"; }
  echo "--- level $level exit $rc ---"
done
if [[ -s "$OUT/truth-failures.txt" ]]; then
  echo "=== VOLUME LADDER DONE, WITH TRUTH-CAPTURE FAILURES ==="
  echo "levels needing re-capture (workspaces kept under $OUT/ws-L*):"
  cat "$OUT/truth-failures.txt"
  exit 1
fi
echo "=== VOLUME LADDER DONE ==="
