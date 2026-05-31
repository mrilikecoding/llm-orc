#!/usr/bin/env bash
# Spike sigma.1 - baseline: can a cheap local model (qwen3:14b) sustain
# OpenCode's multi-turn agentic loop on a real multi-step task? No stand-in;
# OpenCode points straight at Ollama's OpenAI-compatible endpoint.
set -uo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
WS="$DIR/workspace_sigma1"
cd "$DIR"
rm -f opencode_sigma1.out
rm -rf "$WS"; mkdir -p "$WS"

TASK="Create calc.py with a function add(a, b) that returns a + b. Then create test_calc.py that imports add from calc and asserts add(2, 3) == 5, printing PASS if correct. Then run test_calc.py with python3 and tell me whether it passed."

OPENCODE_CONFIG="$DIR/opencode_sigma.json" opencode run "$TASK" \
  --format json --dir "$WS" -m ollocal/qwen3:14b \
  --log-level INFO > opencode_sigma1.out 2>&1 &
OC=$!
for _ in $(seq 1 300); do
  kill -0 "$OC" 2>/dev/null || break
  sleep 1
done
kill "$OC" 2>/dev/null; wait "$OC" 2>/dev/null; OC_RC=$?

echo "opencode rc: $OC_RC"
echo "=== tool-call trajectory (from json events) ==="
python3 - <<'PY'
import json, collections
ev=[]
for l in open("opencode_sigma1.out"):
    l=l.strip()
    if l and l[0]=="{":
        try: ev.append(json.loads(l))
        except Exception: pass
print("event types:", dict(collections.Counter(e.get("type") for e in ev if isinstance(e,dict))))
for e in ev:
    if not isinstance(e,dict): continue
    if e.get("type")=="tool":
        st=e.get("part",{}).get("state",{})
        inp=st.get("input") or {}
        print(f"  TOOL {e['part'].get('tool')} status={st.get('status')} args={json.dumps(inp)[:90]}")
    elif e.get("type")=="text":
        print(f"  TEXT {e.get('part',{}).get('text','')[:120]}")
    elif e.get("type")=="error":
        print(f"  ERROR {json.dumps(e)[:200]}")
PY
echo "=== workspace ==="; ls -la "$WS"
for f in "$WS"/*.py; do [ -f "$f" ] && { echo "--- $f ---"; cat "$f"; }; done
