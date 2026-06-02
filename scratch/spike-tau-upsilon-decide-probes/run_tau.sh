#!/usr/bin/env bash
# Spike tau (v2) - grounded-loop FALSIFICATION probe (OQ #27 axis 1).
#
# NON-COLLAPSIBLE task: a random 8-char code exists ONLY in step-1 stdout
# (not written to a file, not recomputable). To create code.txt correctly the
# driver MUST observe step-1's tool result and carry the value into a write
# call. There is no bash one-liner shortcut (the value is not in any file).
#   - grounded stepping (bash -> observe -> write) -> code.txt == printed code  (AXIS-1 PASS)
#   - batched guess (bash + write in one turn)      -> code.txt != printed code  (AXIS-1 FAIL)
# A clean PASS is an AXIS-1 pass ONLY; axis 2 (long-horizon error accumulation)
# is a BUILD-phase target this short probe cannot settle. $0 (local qwen3:14b).
set -uo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
PORT=8099
WS="$DIR/workspace_tau"
cd "$DIR"
rm -f requests_tau.jsonl server_tau.out opencode_tau.out
rm -rf "$WS"; mkdir -p "$WS"

PORT=$PORT SINGLE_STEP=${SINGLE_STEP:-0} python3 server_tau.py > server_tau.out 2>&1 &
SRV=$!
for _ in $(seq 1 30); do
  grep -q LISTENING server_tau.out 2>/dev/null && break
  sleep 0.2
done
echo "server pid $SRV ($(head -1 server_tau.out)) ws=$WS"

TASK='Do exactly two steps using your tools, then stop. Step 1: use the bash tool to run a Python command that prints a random 8-character code made of uppercase letters A-Z to standard output. In this step only print the code - do NOT write it to any file. Step 2: use the write tool to create a file named code.txt whose entire content is exactly the 8-character code that Step 1 printed to stdout. Do not regenerate the code; use the exact code you saw printed.'

OPENCODE_CONFIG="$DIR/opencode_spike.json" opencode run "$TASK" \
  --format json --dir "$WS" -m spike/spike-model \
  --log-level INFO > opencode_tau.out 2>&1 &
OC=$!
for _ in $(seq 1 360); do
  kill -0 "$OC" 2>/dev/null || break
  sleep 1
done
kill "$OC" 2>/dev/null; wait "$OC" 2>/dev/null; OC_RC=$?
kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null

echo "opencode rc: $OC_RC"
echo "=== per-turn: what the driver OBSERVED, then what it DECIDED ==="
python3 - <<'PY'
import json
for l in open("requests_tau.jsonl"):
    r = json.loads(l)
    s = r.get("stage")
    if s == "observed":
        print(f"  [turn {r.get('turn_index')}] OBSERVED tool results:")
        for tr in r.get("tool_results", []):
            print(f"        {tr!r}")
    elif s == "turn":
        print(f"  [turn {r.get('turn_index')}] DECIDED {r.get('n_tool_calls')} call(s):")
        for c in r.get("calls", []):
            print(f"        {c['name']}: {c['args']}")
    elif s == "finish":
        print(f"  [turn {r.get('turn_index')}] FINISH: {r.get('text','')[:160]}")
    elif s == "driver-error":
        print(f"  driver-error: {r.get('err','')[:160]}")
PY
echo "=== verdict ==="
python3 - <<'PY'
import json, re, os, glob

# what step 1 actually printed (from OpenCode's bash tool output)
printed = None
ev = [json.loads(l) for l in open("opencode_tau.out")
      if l.strip() and l.strip()[0] == "{"]
for e in ev:
    if isinstance(e, dict) and e.get("type") == "tool":
        st = e.get("part", {}).get("state", {})
        if e["part"].get("tool") == "bash" and st.get("status") == "completed":
            out = st.get("output") or ""
            m = re.search(r"\b([A-Z]{8})\b", out)
            if m:
                printed = m.group(1)
                break

# did the driver use the write tool, and in which turn relative to the bash?
turns = [json.loads(l) for l in open("requests_tau.jsonl")
         if json.loads(l).get("stage") == "turn"]
batched = any(
    {"bash", "write"} <= {c["name"] for c in t.get("calls", [])}
    for t in turns)
used_write = any(c["name"] == "write"
                 for t in turns for c in t.get("calls", []))

ws = "workspace_tau"
code_txt = None
p = os.path.join(ws, "code.txt")
if os.path.exists(p):
    code_txt = open(p).read().strip()

print(f"  step-1 printed code (from bash stdout): {printed!r}")
print(f"  code.txt content:                       {code_txt!r}")
print(f"  driver used write tool:                 {used_write}")
print(f"  driver batched bash+write in one turn:  {batched}")
print(f"  other files: {[os.path.basename(x) for x in glob.glob(ws+'/*')]}")
if printed and code_txt and printed == code_txt and not batched:
    print("  >>> AXIS-1 PASS: grounded stepping (observed value carried into write)")
elif printed and code_txt and printed == code_txt and batched:
    print("  >>> AMBIGUOUS: values match but batched - check whether write preceded observation")
elif printed and code_txt and printed != code_txt:
    print("  >>> AXIS-1 FAIL: code.txt does NOT match the observed code (presupposed unobserved state)")
else:
    print("  >>> INCONCLUSIVE: see trajectory (collapsed task / no write / timeout)")
PY
echo "=== workspace ==="; ls -la "$WS"
[ -f "$WS/code.txt" ] && { echo "--- code.txt ---"; cat "$WS/code.txt"; echo; }
