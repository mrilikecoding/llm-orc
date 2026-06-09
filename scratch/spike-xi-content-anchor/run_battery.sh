#!/usr/bin/env bash
# Spike ξ primary battery — coder-layer qwen3:8b, $0 local.
# Base T: A/B/C + both controls. Base V: A/B/C (controls are Base-T only).
set -u
cd "$(dirname "$0")"
N="${1:-10}"
echo "=== Spike xi primary battery (n=$N per cell, qwen3:8b) ==="
for cell in \
  "T A_current" "T B_signatures" "T C_full" "T Control_decoy" "T Control_filler" \
  "V A_current" "V B_signatures" "V C_full" ; do
  set -- $cell
  echo ""
  echo ">>> $1 / $2"
  python3 probe.py "$1" "$2" "$N"
done
echo ""
echo "=== battery complete ==="
