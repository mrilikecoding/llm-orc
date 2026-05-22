#!/usr/bin/env bash
# Spike ζ runner — invoke the routing-planner ensemble against each test
# prompt and capture results in JSON.
#
# Cycle 7 DISCOVER Spike ζ (2026-05-21). Free-tier; local qwen3:8b.

set -euo pipefail

SPIKE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROMPTS_FILE="${SPIKE_DIR}/test_prompts.json"
RESULTS_FILE="${SPIKE_DIR}/results.json"
ENSEMBLE_NAME="spike-cycle7-zeta-routing-planner"

cd "$(git rev-parse --show-toplevel)"

NUM_PROMPTS=$(jq '.prompts | length' "$PROMPTS_FILE")
echo "Running Spike ζ: $NUM_PROMPTS prompts against $ENSEMBLE_NAME"
echo ""

jq -n --arg date "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --arg ensemble "$ENSEMBLE_NAME" \
  '{spike: "cycle-7-zeta-routing-planner", run_date: $date, ensemble: $ensemble, runs: []}' \
  > "$RESULTS_FILE"

for i in $(seq 0 $((NUM_PROMPTS - 1))); do
    PROMPT_ID=$(jq -r ".prompts[$i].id" "$PROMPTS_FILE")
    INPUT=$(jq -r ".prompts[$i].input" "$PROMPTS_FILE")
    SHAPE=$(jq -r ".prompts[$i].shape" "$PROMPTS_FILE")
    EXPECTED_ACTION=$(jq -r ".prompts[$i].expected_action" "$PROMPTS_FILE")
    EXPECTED_ENSEMBLE=$(jq -r ".prompts[$i].expected_ensemble // \"null\"" "$PROMPTS_FILE")

    echo "[$((i + 1))/$NUM_PROMPTS] $PROMPT_ID ($SHAPE)"

    START_TIME=$(date +%s.%N)

    # Use --input-data with proper escaping and --output-format json
    # Suppress streaming output; capture full JSON result
    OUTPUT=$(llm-orc invoke "$ENSEMBLE_NAME" --input-data "$INPUT" \
                              --output-format json --no-streaming 2>/dev/null \
                              || echo '{"error": "invoke_failed"}')

    END_TIME=$(date +%s.%N)
    LATENCY=$(echo "$END_TIME - $START_TIME" | bc)

    jq --arg id "$PROMPT_ID" \
       --arg shape "$SHAPE" \
       --arg input "$INPUT" \
       --arg expected_action "$EXPECTED_ACTION" \
       --arg expected_ensemble "$EXPECTED_ENSEMBLE" \
       --argjson output "$OUTPUT" \
       --arg latency "$LATENCY" \
       '.runs += [{
          id: $id,
          shape: $shape,
          input: $input,
          expected_action: $expected_action,
          expected_ensemble: $expected_ensemble,
          output: $output,
          latency_seconds: ($latency | tonumber)
       }]' \
       "$RESULTS_FILE" > "${RESULTS_FILE}.tmp" && mv "${RESULTS_FILE}.tmp" "$RESULTS_FILE"

    echo "  latency: ${LATENCY}s"
done

echo ""
echo "Done. Results at $RESULTS_FILE"
