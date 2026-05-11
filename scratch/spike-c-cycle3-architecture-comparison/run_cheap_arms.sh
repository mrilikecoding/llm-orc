#!/usr/bin/env bash
# Run cheap arms × 3 trials each, sequentially (avoid hammering free tier).
# Saves output per trial to scratch/spike-c-.../trials/.

set -e
cd "$(dirname "$0")/../.."

TRIALS_DIR=scratch/spike-c-cycle3-architecture-comparison/trials
DIFF=scratch/spike-c-cycle3-architecture-comparison/fixture/diff.patch

REVIEW_PROMPT="Please review the following diff and produce a structured code review covering: (1) Logic / semantic issues — boundary conditions, off-by-one, unreachable code, contract violations; (2) Security / sensitive-data issues — credentials in logs, missing auth, etc.; (3) Type-safety / API-contract issues — annotations, error handling; (4) Test coverage; (5) Cross-file consistency; (6) Other concerns. Format as Markdown with sections. Cite line numbers. Be specific — flag concrete issues, not generic advice.

DIFF UNDER REVIEW:

$(cat $DIFF)"

echo "=== Arm A (cheap-bare) — 3 trials ==="
for i in 1 2 3; do
    out=$TRIALS_DIR/arm-a-cheap-bare-trial${i}.txt
    echo "[$(date +%H:%M:%S)] arm-a trial $i starting → $out"
    opencode run -m llm-orc/orchestrator-minimax-m25-free --print-logs "$REVIEW_PROMPT" > "$out" 2>&1
    echo "[$(date +%H:%M:%S)] arm-a trial $i done ($(wc -c < $out) chars)"
done

echo ""
echo "=== Arm B (cheap-with-ensemble) — 3 trials ==="
for i in 1 2 3; do
    out=$TRIALS_DIR/arm-b-cheap-with-ensemble-trial${i}.json
    echo "[$(date +%H:%M:%S)] arm-b trial $i starting → $out"
    uv run llm-orc invoke spike-c-code-review -f $DIFF --output-format json --no-streaming > "$out" 2>&1
    echo "[$(date +%H:%M:%S)] arm-b trial $i done ($(wc -c < $out) chars)"
done

echo ""
echo "=== ALL CHEAP TRIALS COMPLETE ==="
ls -la $TRIALS_DIR/arm-a-* $TRIALS_DIR/arm-b-* 2>/dev/null
