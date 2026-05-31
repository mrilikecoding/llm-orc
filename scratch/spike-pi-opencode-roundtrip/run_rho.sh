#!/usr/bin/env bash
# Spike rho - planner-driven delegation + tool_calls terminal end-to-end.
set -uo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
PORT=8099
PROJECT="/Users/nathangreen/Development/eddi-lab/llm-orc"
WS="$DIR/workspace_rho"
cd "$DIR"
rm -f requests_rho.jsonl server_rho.out opencode_rho.out
rm -rf "$WS"; mkdir -p "$WS"

PORT=$PORT WORKSPACE="$WS" PROJECT_DIR="$PROJECT" \
  python3 server_rho.py > server_rho.out 2>&1 &
SRV=$!
for _ in $(seq 1 30); do
  grep -q LISTENING server_rho.out 2>/dev/null && break
  sleep 0.2
done
echo "server pid $SRV ($(head -1 server_rho.out)) ws=$WS"

OPENCODE_CONFIG="$DIR/opencode.json" opencode run \
  "create a file hello.py that prints hello world" \
  --format json --dir "$WS" -m spike/spike-model \
  --log-level INFO > opencode_rho.out 2>&1 &
OC=$!
for _ in $(seq 1 180); do
  kill -0 "$OC" 2>/dev/null || break
  sleep 1
done
kill "$OC" 2>/dev/null; wait "$OC" 2>/dev/null; OC_RC=$?
kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null

echo "opencode rc: $OC_RC"
echo "=== planner decision + dispatch (from server log) ==="
python3 - <<'PY'
import json
for l in open("requests_rho.jsonl"):
    r=json.loads(l)
    if r.get("stage")=="agent-turn": print("PLAN:", json.dumps(r.get("plan")))
    if r.get("event"): print("EVENT:", r.get("event"), r.get("named_ensemble",""), r.get("generator_used",""))
PY
echo "=== workspace ==="; ls -la "$WS"; echo "--- hello.py ---"; cat "$WS/hello.py" 2>/dev/null || echo "(no hello.py)"
