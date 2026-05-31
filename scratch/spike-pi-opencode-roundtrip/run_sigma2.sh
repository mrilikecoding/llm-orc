#!/usr/bin/env bash
# Spike sigma.2 - cheap layer-A loop-driver + layer-B ensemble delegation.
set -uo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
PORT=8099
PROJECT="/Users/nathangreen/Development/eddi-lab/llm-orc"
WS="$DIR/workspace_sigma2"
cd "$DIR"
rm -f requests_sigma2.jsonl server_sigma2.out opencode_sigma2.out
rm -rf "$WS"; mkdir -p "$WS"

PORT=$PORT WORKSPACE="$WS" PROJECT_DIR="$PROJECT" \
  python3 server_sigma2.py > server_sigma2.out 2>&1 &
SRV=$!
for _ in $(seq 1 30); do
  grep -q LISTENING server_sigma2.out 2>/dev/null && break
  sleep 0.2
done
echo "server pid $SRV ($(head -1 server_sigma2.out)) ws=$WS"

TASK="Create calc.py with a function add(a, b) that returns a + b. Then create test_calc.py that imports add from calc and asserts add(2, 3) == 5, printing PASS if correct. Then run test_calc.py with python3 and tell me whether it passed."

OPENCODE_CONFIG="$DIR/opencode.json" opencode run "$TASK" \
  --format json --dir "$WS" -m spike/spike-model \
  --log-level INFO > opencode_sigma2.out 2>&1 &
OC=$!
for _ in $(seq 1 340); do
  kill -0 "$OC" 2>/dev/null || break
  sleep 1
done
kill "$OC" 2>/dev/null; wait "$OC" 2>/dev/null; OC_RC=$?
kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null

echo "opencode rc: $OC_RC"
echo "=== turn-by-turn (server log: driver decisions + delegation) ==="
python3 - <<'PY'
import json
for l in open("requests_sigma2.jsonl"):
    r=json.loads(l)
    s=r.get("stage")
    if s=="turn": print(f"  turn: tools={r.get('tool_calls')} delegated_writes={r.get('delegated_writes')}")
    elif s=="finish": print(f"  finish: {r.get('text','')[:100]}")
    elif s in ("driver-error","delegate-error"): print(f"  {s}: {r.get('err','')[:150]}")
PY
echo "=== opencode tool trajectory ==="
python3 - <<'PY'
import json,collections
ev=[json.loads(l) for l in open("opencode_sigma2.out") if l.strip() and l.strip()[0]=="{"]
print("types:", dict(collections.Counter(e.get("type") for e in ev if isinstance(e,dict))))
for e in ev:
    if isinstance(e,dict) and e.get("type")=="tool":
        st=e.get("part",{}).get("state",{}); print(f"  TOOL {e['part'].get('tool')} status={st.get('status')}")
    if isinstance(e,dict) and e.get("type")=="text": print(f"  TEXT {e.get('part',{}).get('text','')[:120]}")
PY
echo "=== workspace ==="; ls -la "$WS"
for f in "$WS"/*.py; do [ -f "$f" ] && { echo "--- $f ---"; cat "$f"; }; done
