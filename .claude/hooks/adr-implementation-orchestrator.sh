#!/bin/bash

# ADR Implementation Orchestrator Hook
# Detects ADRs ready for implementation and orchestrates the BDD→TDD→Code pipeline

set +e

echo "🎯 ADR Implementation Orchestrator"
echo ""

# Function to extract ADR number from filename
extract_adr_number() {
    local filename="$1"
    echo "$filename" | sed -E 's/^0*([0-9]+)-.*/\1/'
}

# Function to check if ADR has BDD mapping hints
has_bdd_mapping() {
    local adr_file="$1"
    grep -q "## BDD Mapping Hints" "$adr_file"
}

# Function to get ADR status
get_adr_status() {
    local adr_file="$1"
    grep "^## Status" -A1 "$adr_file" | tail -n1 | xargs
}

# Find all ADRs marked as "In Progress"
echo "🔍 Scanning for ADRs ready for implementation..."
READY_ADRS=$(find docs/adrs -name "*.md" -type f | while read -r adr; do
    status=$(get_adr_status "$adr")
    if [[ "$status" == "In Progress" ]]; then
        echo "$adr"
    fi
done)

if [ -z "$READY_ADRS" ]; then
    echo "📋 No ADRs currently marked 'In Progress'"
    exit 0
fi

# Process each ready ADR
echo "$READY_ADRS" | while read -r ADR_FILE; do
    if [ -z "$ADR_FILE" ]; then
        continue
    fi

    ADR_NAME=$(basename "$ADR_FILE")
    ADR_NUM=$(extract_adr_number "$ADR_NAME")

    echo ""
    echo "═══════════════════════════════════════════════════"
    echo "📄 Processing: $ADR_NAME"
    echo "═══════════════════════════════════════════════════"

    # Check for BDD mapping hints
    if ! has_bdd_mapping "$ADR_FILE"; then
        echo "⚠️  No BDD Mapping Hints found in ADR"
        echo "   Add '## BDD Mapping Hints' section to enable orchestration"
        continue
    fi

    # Check if BDD scenarios already exist
    BDD_PATTERN="tests/bdd/features/adr-$(printf "%03d" "$ADR_NUM")-*.feature"
    if ls $BDD_PATTERN >/dev/null 2>&1; then
        echo "✅ BDD scenarios already exist:"
        ls $BDD_PATTERN | sed 's/^/   /'

        # Check if tests exist
        UNIT_TESTS=$(find tests -name "*adr_${ADR_NUM}*.py" -o -name "*adr_$(printf "%03d" "$ADR_NUM")*.py" 2>/dev/null | wc -l)
        echo "📊 Test coverage: $UNIT_TESTS unit test files found"

        if [ "$UNIT_TESTS" -eq 0 ]; then
            echo "🔄 Triggering TDD specialist for test generation..."
            cat <<EOF

{
  "adrOrchestration": {
    "adr": "$ADR_FILE",
    "adrNumber": "$ADR_NUM",
    "phase": "tdd_generation",
    "action": "Run 'llm-orc-tdd-specialist' to create unit tests",
    "bddScenarios": "$BDD_PATTERN"
  }
}
EOF
        fi
    else
        echo "📝 BDD scenarios need to be generated"
        echo "🔄 Triggering ADR-to-BDD orchestrator..."

        # Extract BDD hints for the orchestrator
        echo ""
        echo "📋 BDD Mapping Hints from ADR:"
        sed -n '/## BDD Mapping Hints/,/^##[^#]/p' "$ADR_FILE" | head -n -1 | sed 's/^/   /'

        cat <<EOF

{
  "adrOrchestration": {
    "adr": "$ADR_FILE",
    "adrNumber": "$ADR_NUM",
    "phase": "bdd_generation",
    "action": "Run 'llm-orc-adr-to-bdd-orchestrator' to generate BDD scenarios",
    "command": "claude agent run llm-orc-adr-to-bdd-orchestrator --adr=$ADR_FILE"
  }
}
EOF
    fi
done

# Summary
echo ""
echo "═══════════════════════════════════════════════════"
echo "📊 Orchestration Summary"
echo "═══════════════════════════════════════════════════"

TOTAL_ADRS=$(echo "$READY_ADRS" | grep -c .)
ADRS_WITH_BDD=$(echo "$READY_ADRS" | while read -r adr; do
    [ -n "$adr" ] && has_bdd_mapping "$adr" && echo "1"
done | wc -l)

echo "📄 Total ADRs in progress: $TOTAL_ADRS"
echo "🎯 ADRs with BDD mapping: $ADRS_WITH_BDD"
echo "🔄 ADRs ready for orchestration: $ADRS_WITH_BDD"

if [ "$ADRS_WITH_BDD" -gt 0 ]; then
    echo ""
    echo "💡 Next steps:"
    echo "   1. Review the orchestration suggestions above"
    echo "   2. Run the suggested agent commands"
    echo "   3. Monitor progress with 'epic-progress-tracker.sh'"
fi

exit 0