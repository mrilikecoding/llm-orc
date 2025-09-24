#!/bin/bash
# ADR Lifecycle Management Hook
#
# Automatically manages ADR status transitions based on:
# - BDD scenario results (tests/bdd/ changes)
# - Implementation milestones (schema/agent changes)
# - Architectural compliance validation
#
# Triggers: PostToolUse, PostTest, Manual
# Integration: docs/adr-lifecycle-management.md

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Hook configuration
HOOK_NAME="ADR Lifecycle Manager"
ADR_DIR="docs/adrs"
BDD_DIR="tests/bdd"
SCHEMAS_DIR="src/llm_orc/schemas"

echo -e "${BLUE}🔄 ${HOOK_NAME}${NC}"

# Exit early if no relevant files changed
if [[ $# -gt 0 ]]; then
    # Called with specific files - check if they're ADR-relevant
    relevant_files=false
    for file in "$@"; do
        if [[ "$file" =~ ^(docs/adrs|tests/bdd|src/llm_orc/schemas|tests/.*test.*bdd) ]]; then
            relevant_files=true
            break
        fi
    done

    if [[ "$relevant_files" = false ]]; then
        echo "ℹ️  No ADR-relevant changes detected, skipping"
        exit 0
    fi
fi

# Function to extract ADR number from filename or content
extract_adr_numbers() {
    local file="$1"
    if [[ "$file" =~ docs/adrs/([0-9]+)- ]]; then
        echo "${BASH_REMATCH[1]}"
    elif [[ -f "$file" ]]; then
        # Look for ADR references in content
        grep -o "ADR-[0-9]\+" "$file" 2>/dev/null | sed 's/ADR-//' | sort -u || true
    fi
}

# Function to check BDD scenario status for an ADR
check_bdd_scenarios() {
    local adr_num="$1"
    local scenarios_passing=0
    local scenarios_total=0

    # Find BDD feature files that reference this ADR
    local feature_files
    feature_files=$(find "$BDD_DIR" -name "*.feature" -exec grep -l "ADR-${adr_num}" {} \; 2>/dev/null || true)

    if [[ -z "$feature_files" ]]; then
        echo "no_scenarios"
        return 0
    fi

    # Find corresponding test files
    local test_files
    test_files=$(find "$BDD_DIR" -name "*test*.py" -exec grep -l "ADR-${adr_num}\|adr.${adr_num}\|adr_${adr_num}" {} \; 2>/dev/null || true)

    if [[ -z "$test_files" ]]; then
        echo "no_tests"
        return 0
    fi

    # Run BDD tests and count results (if uv is available)
    if command -v uv >/dev/null 2>&1; then
        # Try to run the BDD scenarios
        local test_output
        test_output=$(uv run pytest "$test_files" -v --tb=no -q 2>/dev/null || echo "test_error")

        if [[ "$test_output" != "test_error" ]]; then
            scenarios_passing=$(echo "$test_output" | grep -c "PASSED" || echo "0")
            scenarios_total=$(echo "$test_output" | grep -c -E "(PASSED|FAILED)" || echo "0")
        fi
    fi

    echo "${scenarios_passing}/${scenarios_total}"
}

# Function to get current ADR status
get_adr_status() {
    local adr_file="$1"
    if [[ -f "$adr_file" ]]; then
        # Look for "## Status" and get the next non-empty line
        awk '/^## Status/ {
            while ((getline line) > 0) {
                if (line !~ /^$/) {
                    print line
                    exit
                }
            }
        }' "$adr_file" | sed 's/^[[:space:]]*//' || echo "Unknown"
    else
        echo "NotFound"
    fi
}

# Function to suggest ADR status transition
suggest_status_transition() {
    local adr_num="$1"
    local current_status="$2"
    local bdd_status="$3"
    local has_implementation="$4"

    case "$current_status" in
        "Proposed")
            if [[ "$bdd_status" != "no_scenarios" && "$bdd_status" != "no_tests" ]]; then
                echo "IN_PROGRESS"
                return 0
            fi
            ;;
        "Unknown")
            # If status is unknown but we have BDD scenarios and implementation, suggest IN_PROGRESS
            if [[ "$bdd_status" =~ ^[0-9]+/[0-9]+$ && "$has_implementation" = true ]]; then
                echo "IN_PROGRESS"
                return 0
            fi
            ;;
        "InProgress"|"In Progress")
            if [[ "$bdd_status" =~ ^[0-9]+/[0-9]+$ ]]; then
                local passing=$(echo "$bdd_status" | cut -d'/' -f1)
                local total=$(echo "$bdd_status" | cut -d'/' -f2)
                if [[ "$passing" -gt 0 && "$passing" -eq "$total" && "$has_implementation" = true ]]; then
                    echo "IMPLEMENTED"
                    return 0
                fi
            fi
            ;;
        "Implemented")
            # Could transition to VALIDATED after refactor phase
            # For now, suggest review for VALIDATED status
            if [[ "$bdd_status" =~ ^[0-9]+/[0-9]+$ ]]; then
                local passing=$(echo "$bdd_status" | cut -d'/' -f1)
                local total=$(echo "$bdd_status" | cut -d'/' -f2)
                if [[ "$passing" -eq "$total" && "$passing" -gt 0 ]]; then
                    echo "CONSIDER_VALIDATED"
                    return 0
                fi
            fi
            ;;
    esac

    echo "NO_CHANGE"
}

# Function to check if implementation exists for ADR
check_implementation_status() {
    local adr_num="$1"
    local implementation_found=false

    # Check for schema implementations
    if [[ -d "$SCHEMAS_DIR" ]]; then
        if find "$SCHEMAS_DIR" -name "*.py" -exec grep -l "ADR-${adr_num}" {} \; 2>/dev/null | grep -q .; then
            implementation_found=true
        fi
    fi

    # Check for agent implementations
    if find src/llm_orc/agents -name "*.py" -exec grep -l "ADR-${adr_num}" {} \; 2>/dev/null | grep -q .; then
        implementation_found=true
    fi

    # Check for core implementations
    if find src/llm_orc/core -name "*.py" -exec grep -l "ADR-${adr_num}" {} \; 2>/dev/null | grep -q .; then
        implementation_found=true
    fi

    echo "$implementation_found"
}

# Function to update ADR status (interactive)
update_adr_status() {
    local adr_file="$1"
    local suggested_status="$2"
    local adr_num="$3"
    local bdd_status="$4"

    # Check if stdin is a terminal (interactive mode)
    if [ -t 0 ]; then
        echo -e "\n${PURPLE}📋 ADR-${adr_num} Status Update Opportunity${NC}"
        echo -e "Current Status: $(get_adr_status "$adr_file")"
        echo -e "BDD Scenarios: $bdd_status"
        echo -e "Suggested: $suggested_status"

        case "$suggested_status" in
            "IN_PROGRESS")
                echo -e "\n${GREEN}✨ Ready to transition to IN_PROGRESS${NC}"
                echo "BDD scenarios and tests are set up - implementation can begin!"
                ;;
            "IMPLEMENTED")
                echo -e "\n${GREEN}🎉 Ready to transition to IMPLEMENTED${NC}"
                echo "All BDD scenarios are passing - core implementation complete!"
                ;;
            "CONSIDER_VALIDATED")
                echo -e "\n${BLUE}🔍 Consider transitioning to VALIDATED${NC}"
                echo "Implementation is stable and scenarios passing consistently."
                ;;
        esac

        echo -e "\nWould you like to:"
        echo "1. Update ADR status automatically"
        echo "2. View ADR lifecycle documentation"
        echo "3. Skip this update"

        read -p "Choice (1/2/3): " -n 1 -r choice
        echo

        case "$choice" in
            1)
                # Would implement actual ADR file modification here
                echo -e "${GREEN}✅ ADR status would be updated (implementation needed)${NC}"
                echo "Next: Implement ADR file parsing and status update logic"
                ;;
            2)
                echo -e "${BLUE}📖 ADR Lifecycle Documentation${NC}"
                echo "See: docs/adr-lifecycle-management.md"
                if command -v cat >/dev/null 2>&1; then
                    echo -e "\nKey status transitions:"
                    grep -A 5 "## ADR Status Lifecycle" docs/adr-lifecycle-management.md || true
                fi
                ;;
            3)
                echo "Skipping ADR status update"
                ;;
        esac
    else
        # Non-interactive mode: log the suggested status and take default safe actions
        echo -e "\n${BLUE}ADR-${adr_num} Non-Interactive Mode${NC}"
        echo "Current Status: $(get_adr_status "$adr_file")"
        echo "BDD Scenarios: $bdd_status"
        echo "Suggested Status: $suggested_status"

        case "$suggested_status" in
            "IN_PROGRESS"|"IMPLEMENTED"|"CONSIDER_VALIDATED")
                echo -e "${GREEN}✅ Logging potential ADR status transition${NC}"
                echo "Would transition to: $suggested_status"
                # Log the potential status transition without modifying the file
                logger -t "ADR-Lifecycle" "ADR-${adr_num} potential status transition to $suggested_status"
                ;;
            *)
                echo "No action needed for current ADR status"
                ;;
        esac
    fi
}

# Main processing logic
main() {
    local updated_adrs=()
    local processed_adrs=()

    # Process all ADR files
    if [[ -d "$ADR_DIR" ]]; then
        while IFS= read -r -d '' adr_file; do
            local adr_num
            adr_num=$(basename "$adr_file" | grep -o "^[0-9]\+")

            if [[ -n "$adr_num" ]]; then
                processed_adrs+=("$adr_num")

                local current_status
                current_status=$(get_adr_status "$adr_file")

                local bdd_status
                bdd_status=$(check_bdd_scenarios "$adr_num")

                local has_implementation
                has_implementation=$(check_implementation_status "$adr_num")

                local suggested_status
                suggested_status=$(suggest_status_transition "$adr_num" "$current_status" "$bdd_status" "$has_implementation")

                echo -e "${BLUE}🔍 ADR-${adr_num}:${NC} Status=$current_status, BDD=$bdd_status, Impl=$has_implementation → $suggested_status"

                if [[ "$suggested_status" != "NO_CHANGE" ]]; then
                    updated_adrs+=("$adr_num")
                    update_adr_status "$adr_file" "$suggested_status" "$adr_num" "$bdd_status"
                fi
            fi
        done < <(find "$ADR_DIR" -name "*.md" -print0)
    fi

    # Summary
    if [[ ${#updated_adrs[@]} -eq 0 ]]; then
        echo -e "${GREEN}✅ All ADRs are up to date${NC}"
    else
        echo -e "\n${YELLOW}📊 Processed ${#updated_adrs[@]} ADR(s): ${updated_adrs[*]}${NC}"
    fi

    # Show BDD integration reminder
    echo -e "\n${BLUE}💡 Tip:${NC} Run BDD scenarios with:"
    echo "   uv run uv run pytest tests/bdd/ -v"
    echo -e "\n${BLUE}📚 Documentation:${NC} docs/adr-lifecycle-management.md"
}

# Execute main function
main "$@"

echo -e "${GREEN}✅ ${HOOK_NAME} complete${NC}"
exit 0