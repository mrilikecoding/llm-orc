#!/bin/bash
# Spike π Fork 3 full run — Cell A-live (flag off, baseline) then Cell B-live
# (parse gate + server-side recovery on). Manages the serve lifecycle across
# cells: the LLMORC_SPIKE_PI_GATE flag lives on the serve process, so each cell
# gets its own serve invocation (restart between cells = unambiguous flag
# state). serve.log is truncated at each cell start so the runner's line-range
# slicing stays sane within a cell. Usage: run_full.sh [N]   (N default 5)
set -u
ROOT=/Users/nathangreen/Development/eddi-lab/llm-orc
SPIKE="$ROOT/scratch/spike-pi-form-adequacy-gate"
N="${1:-5}"

start_serve() {  # $1 = flag value ("" = off, "parse" = on)
  if lsof -ti :8765 >/dev/null 2>&1; then kill "$(lsof -ti :8765)" 2>/dev/null; sleep 2; fi
  : > "$SPIKE/serve.log"
  cd "$ROOT"
  if [ -n "$1" ]; then
    LLMORC_SPIKE_PI_GATE="$1" uv run llm-orc serve --port 8765 >> "$SPIKE/serve.log" 2>&1 &
  else
    uv run llm-orc serve --port 8765 >> "$SPIKE/serve.log" 2>&1 &
  fi
  echo "$!" > "$SPIKE/serve.pid"
  for n in $(seq 1 30); do
    if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8765/v1/models 2>/dev/null | grep -q 200; then
      echo "serve READY (flag='${1:-OFF}') pid $(cat "$SPIKE/serve.pid")"; return 0
    fi
    sleep 1
  done
  echo "SERVE FAILED TO START (flag='${1:-OFF}')"; return 1
}

echo "######## CELL A-live (flag OFF — baseline, no gate, no recovery) — $(date +%H:%M:%S) ########"
start_serve "" || exit 1
cd "$SPIKE" && ./run_live_arm.sh A_live "$N"

echo ""
echo "######## CELL B-live (parse gate + server-side recovery ON) — $(date +%H:%M:%S) ########"
start_serve "parse" || exit 1
cd "$SPIKE" && ./run_live_arm.sh B_live "$N"

if lsof -ti :8765 >/dev/null 2>&1; then kill "$(lsof -ti :8765)" 2>/dev/null; fi
echo ""
echo "######## FULL RUN DONE — $(date +%H:%M:%S) ########"
echo "=== A_live SUMMARY ===" && cat "$SPIKE/A_live/SUMMARY.tsv"
echo "=== B_live SUMMARY ===" && cat "$SPIKE/B_live/SUMMARY.tsv"
