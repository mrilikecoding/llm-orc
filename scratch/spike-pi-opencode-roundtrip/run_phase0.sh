#!/usr/bin/env bash
# Spike pi Phase 0 - drive a headless OpenCode session at the observe server
# and capture what it sends. Self-limits the opencode run; no GNU `timeout` dep.
set -uo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
PORT=8099
cd "$DIR"
rm -f requests.jsonl server.out opencode.out
mkdir -p workspace
rm -rf workspace/* 2>/dev/null

python3 server.py "$PORT" > server.out 2>&1 &
SRV=$!
for _ in $(seq 1 30); do
  grep -q "LISTENING" server.out 2>/dev/null && break
  sleep 0.2
done
echo "server pid $SRV ($(head -1 server.out))"

OPENCODE_CONFIG="$DIR/opencode.json" opencode run \
  "create a file hello.py that prints hello world" \
  --format json --dir "$DIR/workspace" -m spike/spike-model \
  --log-level INFO > opencode.out 2>&1 &
OC=$!
for _ in $(seq 1 90); do
  kill -0 "$OC" 2>/dev/null || break
  sleep 1
done
kill "$OC" 2>/dev/null
wait "$OC" 2>/dev/null
OC_RC=$?

kill "$SRV" 2>/dev/null
wait "$SRV" 2>/dev/null

echo "opencode rc: $OC_RC"
echo "=== requests captured ==="
wc -l requests.jsonl 2>/dev/null || echo "NONE"
echo "=== workspace contents ==="
ls -la workspace/ 2>/dev/null
