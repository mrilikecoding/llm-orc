#!/bin/bash
# Testing Pyramid Validation Gate
#
# Ensures proper testing pyramid structure (Unit > Integration > BDD)
# Validates BDD → Unit Test → Implementation workflow
#
# Triggers: PreCommit, Manual
# Integration: Quality gate for testing discipline

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

HOOK_NAME="Testing Pyramid Gate"

echo -e "${BLUE}🔺 ${HOOK_NAME}${NC}"

# Function to count different types of tests
count_test_types() {
    local unit_tests=0
    local integration_tests=0
    local bdd_scenarios=0
    local step_definitions=0

    # Count unit tests (excluding BDD directory)
    if [[ -d "tests" ]]; then
        unit_tests=$(find tests -name "test_*.py" -not -path "*/bdd/*" -type f 2>/dev/null | wc -l)
    fi

    # Count integration tests
    if [[ -d "tests" ]]; then
        integration_tests=$(find tests -name "*integration*.py" -o -name "*e2e*.py" 2>/dev/null | wc -l)
    fi

    # Count BDD scenarios
    if [[ -d "tests/bdd/features" ]]; then
        bdd_scenarios=$(find tests/bdd/features -name "*.feature" 2>/dev/null | \
                       xargs grep -c "^[[:space:]]*Scenario:" 2>/dev/null | \
                       awk '{sum += $1} END {print sum}' 2>/dev/null || echo 0)
    fi

    # Count BDD step definitions
    if [[ -d "tests/bdd" ]]; then
        step_definitions=$(find tests/bdd -name "test_*.py" 2>/dev/null | wc -l)
    fi

    echo "$unit_tests,$integration_tests,$bdd_scenarios,$step_definitions"
}

# Function to validate pyramid ratios
validate_pyramid_ratios() {
    local counts="$1"
    IFS=',' read -r unit_tests integration_tests bdd_scenarios step_definitions <<< "$counts"

    echo -e "${PURPLE}📊 Current Testing Pyramid:${NC}"
    echo -e "  🧪 Unit Tests: $unit_tests"
    echo -e "  🔗 Integration Tests: $integration_tests"
    echo -e "  🎭 BDD Scenarios: $bdd_scenarios"
    echo -e "  📋 BDD Step Definitions: $step_definitions"

    local issues=()

    # Validate pyramid structure (Unit > Integration > BDD)
    if [[ $unit_tests -lt $integration_tests ]]; then
        issues+=("❌ Inverted pyramid: More integration ($integration_tests) than unit tests ($unit_tests)")
    fi

    if [[ $integration_tests -lt $bdd_scenarios ]]; then
        issues+=("❌ Too many BDD scenarios ($bdd_scenarios) for integration tests ($integration_tests)")
    fi

    # Validate minimum thresholds
    if [[ $unit_tests -eq 0 ]] && [[ -d "src" ]]; then
        issues+=("❌ No unit tests found - implementation needs unit-level validation")
    fi

    if [[ $bdd_scenarios -gt 0 ]] && [[ $step_definitions -eq 0 ]]; then
        issues+=("❌ BDD scenarios exist but no step definition files")
    fi

    # Calculate pyramid health ratio
    local total_tests=$((unit_tests + integration_tests + bdd_scenarios))
    if [[ $total_tests -gt 0 ]]; then
        local unit_percentage=$((unit_tests * 100 / total_tests))
        local integration_percentage=$((integration_tests * 100 / total_tests))
        local bdd_percentage=$((bdd_scenarios * 100 / total_tests))

        echo -e "${BLUE}📈 Test Distribution:${NC}"
        echo -e "  Unit: ${unit_percentage}% (ideal: 70%+)"
        echo -e "  Integration: ${integration_percentage}% (ideal: 20%+)"
        echo -e "  BDD: ${bdd_percentage}% (ideal: 10%+)"

        # Validate ideal ratios (70/20/10 pyramid)
        if [[ $unit_percentage -lt 60 ]] && [[ $unit_tests -gt 0 ]]; then
            issues+=("⚠️ Low unit test percentage: ${unit_percentage}% (recommended: 70%+)")
        fi
    fi

    if [[ ${#issues[@]} -eq 0 ]]; then
        echo -e "${GREEN}✅ Testing pyramid structure looks good!${NC}"
        return 0
    else
        echo -e "${RED}🚫 Testing pyramid issues:${NC}"
        printf '%s\n' "${issues[@]}"
        return 1
    fi
}

# Function to check for missing unit tests
check_missing_unit_tests() {
    echo -e "${BLUE}🔍 Checking for missing unit tests...${NC}"

    local missing_tests=()

    # Find source files that might need unit tests
    if [[ -d "src" ]]; then
        while IFS= read -r -d '' src_file; do
            # Skip __init__.py files
            if [[ "$(basename "$src_file")" == "__init__.py" ]]; then
                continue
            fi

            # Calculate expected test file path
            local rel_path="${src_file#src/llm_orc/}"
            local test_file="tests/${rel_path%.py}/test_$(basename "$rel_path")"

            # Alternative test file locations
            local alt_test_file1="tests/$(dirname "$rel_path")/test_$(basename "$rel_path")"
            local alt_test_file2="tests/test_${rel_path//\//_}"

            if [[ ! -f "$test_file" ]] && [[ ! -f "$alt_test_file1" ]] && [[ ! -f "$alt_test_file2" ]]; then
                # Check if file has substantial implementation (not just imports)
                local line_count=$(grep -c -v '^[[:space:]]*#\|^[[:space:]]*$\|^[[:space:]]*import\|^[[:space:]]*from' "$src_file" || echo 0)
                if [[ $line_count -gt 10 ]]; then
                    missing_tests+=("$src_file → $test_file")
                fi
            fi
        done < <(find src -name "*.py" -type f -print0 2>/dev/null)
    fi

    if [[ ${#missing_tests[@]} -eq 0 ]]; then
        echo -e "${GREEN}✅ All significant source files have corresponding tests${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠️ Source files that might need unit tests:${NC}"
        printf '  %s\n' "${missing_tests[@]}"
        echo ""
        echo -e "${BLUE}💡 Consider running: .claude/hooks/bdd-unit-test-generator.sh${NC}"
        return 1
    fi
}

# Function to validate BDD-Unit test relationship
validate_bdd_unit_relationship() {
    echo -e "${BLUE}🔗 Validating BDD ↔ Unit Test relationship...${NC}"

    local bdd_files=()
    local unit_files=()

    # Find BDD feature files
    if [[ -d "tests/bdd/features" ]]; then
        while IFS= read -r -d '' file; do
            bdd_files+=("$file")
        done < <(find tests/bdd/features -name "*.feature" -print0 2>/dev/null)
    fi

    # Find unit test files
    if [[ -d "tests" ]]; then
        while IFS= read -r -d '' file; do
            unit_files+=("$file")
        done < <(find tests -name "test_*.py" -not -path "*/bdd/*" -print0 2>/dev/null)
    fi

    local relationship_issues=()

    # Check if BDD scenarios exist without corresponding unit tests
    for bdd_file in "${bdd_files[@]}"; do
        # Extract issue number if present
        local issue_num=$(basename "$bdd_file" | grep -o 'issue-[0-9]\+' | grep -o '[0-9]\+' || echo "")

        if [[ -n "$issue_num" ]]; then
            local expected_unit_file="tests/test_issue_${issue_num}_units.py"
            if [[ ! -f "$expected_unit_file" ]]; then
                relationship_issues+=("BDD file $(basename "$bdd_file") missing unit tests: $expected_unit_file")
            fi
        fi
    done

    if [[ ${#relationship_issues[@]} -eq 0 ]]; then
        echo -e "${GREEN}✅ BDD scenarios have corresponding unit test structures${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠️ BDD-Unit relationship issues:${NC}"
        printf '  %s\n' "${relationship_issues[@]}"
        return 1
    fi
}

# Function to suggest improvements
suggest_improvements() {
    local counts="$1"
    IFS=',' read -r unit_tests integration_tests bdd_scenarios step_definitions <<< "$counts"

    echo -e "${BLUE}💡 Testing Pyramid Improvement Suggestions:${NC}"

    if [[ $unit_tests -eq 0 ]] && [[ -d "src" ]]; then
        echo "  1. Generate unit tests: .claude/hooks/bdd-unit-test-generator.sh"
    fi

    if [[ $bdd_scenarios -gt 0 ]] && [[ $unit_tests -lt $((bdd_scenarios * 3)) ]]; then
        echo "  2. Add more unit tests to support BDD scenarios (recommended ratio: 3:1)"
    fi

    if [[ $integration_tests -eq 0 ]] && [[ $unit_tests -gt 5 ]]; then
        echo "  3. Consider adding integration tests to bridge unit tests and BDD scenarios"
    fi

    echo "  4. Run test coverage: make test-coverage"
    echo "  5. Review testing strategy: docs/testing-strategy.md"
}

# Main validation function
main() {
    local test_counts
    test_counts=$(count_test_types)

    echo -e "${BLUE}🎯 Validating testing pyramid structure...${NC}"

    local pyramid_valid=true
    local unit_tests_valid=true
    local relationship_valid=true

    # Validate pyramid structure
    if ! validate_pyramid_ratios "$test_counts"; then
        pyramid_valid=false
    fi

    echo ""

    # Check for missing unit tests
    if ! check_missing_unit_tests; then
        unit_tests_valid=false
    fi

    echo ""

    # Validate BDD-Unit relationship
    if ! validate_bdd_unit_relationship; then
        relationship_valid=false
    fi

    echo ""

    # Overall assessment
    if [[ "$pyramid_valid" = true ]] && [[ "$unit_tests_valid" = true ]] && [[ "$relationship_valid" = true ]]; then
        echo -e "${GREEN}✅ Testing pyramid is well-structured${NC}"
        return 0
    else
        suggest_improvements "$test_counts"
        echo ""
        echo -e "${YELLOW}⚠️ Testing pyramid needs attention${NC}"
        echo -e "${BLUE}💡 Use --fix flag to generate missing tests automatically${NC}"
        return 1
    fi
}

# Check for --fix flag
if [[ "${1:-}" == "--fix" ]]; then
    echo -e "${BLUE}🔧 Running in fix mode - generating missing tests...${NC}"
    .claude/hooks/bdd-unit-test-generator.sh
    echo ""
fi

# Execute main function
main "$@"

echo -e "${GREEN}✅ ${HOOK_NAME} complete${NC}"
exit 0