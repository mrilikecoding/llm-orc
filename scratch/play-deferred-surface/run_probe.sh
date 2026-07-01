#!/bin/bash
# Retry-on-wedge runner for the intermittent opencode bootstrap hang.
# Launch opencode, detect the first chat POST on the serve within BOOT_S;
# if none, the client wedged at bootstrap -> graceful TERM (never -9, which
# aggravated global state per the breadcrumb) and relaunch. Wedged attempts
# cost ~$0 (no POST ever reaches the seat).
# args: LABEL WS PROMPT OUT ERR STARTFILE
set -u
LABEL=$1; WS=$2; PROMPT=$3; OUT=$4; ERR=$5; STARTFILE=$6
LOG=/Users/nathangreen/Development/eddi-lab/llm-orc/scratch/play-deferred-surface/logs/play-serve.log
BOOT_S=35
MAX_ATTEMPTS=6

# pre-kill any stale opencode run for this workspace
pkill -TERM -f "opencode run .*--dir $WS" 2>/dev/null; sleep 1

for attempt in $(seq 1 $MAX_ATTEMPTS); do
  START=$(wc -l < "$LOG"); echo "$START" > "$STARTFILE"
  ts=$(date +%H:%M:%S)
  timeout 1800 opencode run -m llmorc/agentic --format json --dir "$WS" "$PROMPT" > "$OUT" 2>"$ERR" &
  CPID=$!
  booted=0
  for i in $(seq 1 $BOOT_S); do
    n=$(tail -n +$((START+1)) "$LOG" | grep -c "POST /v1/chat/completions")
    [ "${n:-0}" -ge 1 ] && { booted=1; break; }
    kill -0 $CPID 2>/dev/null || { booted=2; break; }   # exited fast (e.g. PONG)
    sleep 1
  done
  if [ "$booted" = "0" ]; then
    echo "$LABEL attempt=$attempt WEDGED (no POST in ${BOOT_S}s, started $ts) -> TERM + retry"
    kill -TERM $CPID 2>/dev/null; sleep 2
    pkill -TERM -f "opencode run .*--dir $WS" 2>/dev/null; sleep 2
    continue
  fi
  echo "$LABEL attempt=$attempt BOOTED (booted=$booted, started $ts) -> awaiting completion"
  wait $CPID; rc=$?
  echo "$LABEL EXIT rc=$rc attempt=$attempt at $(date +%H:%M:%S)"
  exit 0
done
echo "$LABEL FAILED to boot after $MAX_ATTEMPTS attempts"
exit 7
