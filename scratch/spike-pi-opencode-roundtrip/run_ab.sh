#!/usr/bin/env bash
# Spike pi Phase A or B. Usage: ./run_ab.sh A   |   ./run_ab.sh B
set -uo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
PHASE="${1:-A}"
PORT=8099
PROJECT="/Users/nathangreen/Development/eddi-lab/llm-orc"
WS="$DIR/workspace_$PHASE"
cd "$DIR"
rm -f "requests_$PHASE.jsonl" "server_$PHASE.out" "opencode_$PHASE.out"
rm -rf "$WS"; mkdir -p "$WS"

PORT=$PORT PHASE=$PHASE WORKSPACE="$WS" PROJECT_DIR="$PROJECT" \
  python3 server_ab.py > "server_$PHASE.out" 2>&1 &
SRV=$!
for _ in $(seq 1 30); do
  grep -q LISTENING "server_$PHASE.out" 2>/dev/null && break
  sleep 0.2
done
echo "server pid $SRV ($(head -1 "server_$PHASE.out")) ws=$WS"

OPENCODE_CONFIG="$DIR/opencode.json" opencode run \
  "create a file hello.py that prints hello world" \
  --format json --dir "$WS" -m spike/spike-model \
  --log-level INFO > "opencode_$PHASE.out" 2>&1 &
OC=$!
for _ in $(seq 1 150); do
  kill -0 "$OC" 2>/dev/null || break
  sleep 1
done
kill "$OC" 2>/dev/null; wait "$OC" 2>/dev/null; OC_RC=$?
kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null

echo "opencode rc: $OC_RC"
echo "=== requests captured ==="; wc -l "requests_$PHASE.jsonl" 2>/dev/null || echo NONE
echo "=== workspace $WS ==="; ls -la "$WS"
echo "--- hello.py ---"; cat "$WS/hello.py" 2>/dev/null || echo "(no hello.py written)"
