#!/usr/bin/env bash
# Spike upsilon - WRAPPER-shaped probe (OQ #26). Same task as sigma.2, but
# per-turn generation runs the FULL plan->dispatch->synthesize pipeline as a
# subroutine under the layer-A loop-driver (vs sigma.2's bare-ensemble callee
# call). Measures wrapper viability + per-turn latency compounding.
set -uo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
PORT=8099
PROJECT="/Users/nathangreen/Development/eddi-lab/llm-orc"
WS="$DIR/workspace_upsilon"
cd "$DIR"
rm -f requests_upsilon.jsonl server_upsilon.out opencode_upsilon.out
rm -rf "$WS"; mkdir -p "$WS"

PORT=$PORT PROJECT_DIR="$PROJECT" \
  python3 server_upsilon.py > server_upsilon.out 2>&1 &
SRV=$!
for _ in $(seq 1 30); do
  grep -q LISTENING server_upsilon.out 2>/dev/null && break
  sleep 0.2
done
echo "server pid $SRV ($(head -1 server_upsilon.out)) ws=$WS"

TASK="Create calc.py with a function add(a, b) that returns a + b. Then create test_calc.py that imports add from calc and asserts add(2, 3) == 5, printing PASS if correct. Then run test_calc.py with python3 and tell me whether it passed."

OPENCODE_CONFIG="$DIR/opencode_spike.json" opencode run "$TASK" \
  --format json --dir "$WS" -m spike/spike-model \
  --log-level INFO > opencode_upsilon.out 2>&1 &
OC=$!
for _ in $(seq 1 500); do
  kill -0 "$OC" 2>/dev/null || break
  sleep 1
done
kill "$OC" 2>/dev/null; wait "$OC" 2>/dev/null; OC_RC=$?
kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null

echo "opencode rc: $OC_RC"
echo "=== per-turn driver + pipeline-stage latencies (OQ #26 latency criterion) ==="
python3 - <<'PY'
import json
tot = 0.0
for l in open("requests_upsilon.jsonl"):
    r = json.loads(l)
    s = r.get("stage")
    if s == "pipeline":
        st = r.get("stages_s", {})
        tot += r.get("total_s", 0)
        print(f"  pipeline[{r.get('file')}] plan={st.get('plan_s')}s "
              f"dispatch={st.get('dispatch_s')}s synth={st.get('synth_s')}s "
              f"TOTAL={r.get('total_s')}s  action={r.get('plan',{}).get('action')} "
              f"ens={r.get('plan',{}).get('ensemble')}")
        print(f"      dispatched_len={r.get('dispatched_len')} "
              f"synth_len={r.get('synth_len')} final_len={r.get('final_len')} "
              f"synth_differs_from_dispatch={r.get('synth_differs_from_dispatch')}")
        print(f"      content_head={r.get('content_head')!r}")
    elif s == "turn":
        print(f"  turn: tools={r.get('tool_calls')} pipeline_writes={r.get('pipeline_writes')}")
    elif s == "finish":
        print(f"  finish: {r.get('text','')[:120]}")
    elif s in ("driver-error", "wrap-error"):
        print(f"  {s}: {r.get('err','')[:160]}")
print(f"  >>> total pipeline wall-clock across all writes: {round(tot,1)}s")
PY
echo "=== opencode tool trajectory ==="
python3 - <<'PY'
import json, collections
ev = [json.loads(l) for l in open("opencode_upsilon.out")
      if l.strip() and l.strip()[0] == "{"]
print("types:", dict(collections.Counter(e.get("type") for e in ev if isinstance(e, dict))))
for e in ev:
    if not isinstance(e, dict):
        continue
    if e.get("type") == "tool":
        st = e.get("part", {}).get("state", {})
        print(f"  TOOL {e['part'].get('tool')} status={st.get('status')}")
    elif e.get("type") == "text":
        print(f"  TEXT {e.get('part', {}).get('text', '')[:160]}")
PY
echo "=== workspace ==="; ls -la "$WS"
for f in "$WS"/*.py; do [ -f "$f" ] && { echo "--- $f ---"; cat "$f"; }; done
