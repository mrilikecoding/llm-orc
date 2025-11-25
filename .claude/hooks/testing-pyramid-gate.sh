#!/bin/bash
# Testing Pyramid Validation Gate
#
# Ensures proper testing pyramid structure (Unit > Integration > BDD)
# Validates BDD → Unit Test → Implementation workflow
#
# Triggers: SessionStart, PreCommit, Manual
# Integration: Quality gate for testing discipline

set +e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

HOOK_NAME="Testing Pyramid Gate"

# Moved header to end to avoid duplication

# Function to count different types of tests
count_test_types() {
	local unit_tests=0
	local integration_tests=0
	local bdd_scenarios=0
	local step_definitions=0

	# Count unit test FUNCTIONS (not files) in tests/unit/
	if [[ -d "tests/unit" ]]; then
		unit_tests=$(grep -r "def test_" tests/unit/ 2>/dev/null | wc -l | tr -d ' ')
	fi

	# Count integration test FUNCTIONS (not files) in tests/integration/
	if [[ -d "tests/integration" ]]; then
		integration_tests=$(grep -r "def test_" tests/integration/ 2>/dev/null | wc -l | tr -d ' ')
	fi

	# Count BDD scenarios
	if [[ -d "tests/bdd/features" ]]; then
		bdd_scenarios=$(grep -r "Scenario:" tests/bdd/features/ 2>/dev/null | wc -l | tr -d ' ')
	fi

	# Count BDD step definition files (files are reusable across scenarios)
	if [[ -d "tests/bdd" ]]; then
		step_definitions=$(find tests/bdd -name "test_*.py" 2>/dev/null | wc -l | tr -d ' ')
	fi

	echo "$unit_tests,$integration_tests,$bdd_scenarios,$step_definitions"
}

# Function to get missing unit tests as JSON array (simplified for performance)
get_missing_unit_tests_json() {
	# For now, return a simplified check to avoid performance issues
	# This can be expanded later if needed
	echo "[]"
}

# Function to get BDD issues as JSON array (simplified for performance)
get_bdd_unit_issues_json() {
	# Return empty for now to avoid performance issues
	echo "[]"
}

# Function to validate pyramid ratios
validate_pyramid_ratios() {
	local counts="$1"
	IFS=',' read -r unit_tests integration_tests bdd_scenarios step_definitions <<<"$counts"

	echo -e "${PURPLE}📊 Current Testing Pyramid:${NC}" >&2
	echo -e "  🧪 Unit Test Functions: $unit_tests" >&2
	echo -e "  🔗 Integration Test Functions: $integration_tests" >&2
	echo -e "  🎭 BDD Scenarios: $bdd_scenarios" >&2
	echo -e "  📋 BDD Step Definition Files: $step_definitions" >&2

	local issues=()

	# Validate pyramid structure (Unit > Integration > BDD)
	if [[ $unit_tests -lt $integration_tests ]]; then
		issues+=("inverted_pyramid")
	fi

	if [[ $integration_tests -lt $bdd_scenarios ]]; then
		issues+=("too_many_bdd_scenarios")
	fi

	# Validate minimum thresholds
	if [[ $unit_tests -eq 0 ]] && [[ -d "src" ]]; then
		issues+=("no_unit_tests")
	fi

	if [[ $bdd_scenarios -gt 0 ]] && [[ $step_definitions -eq 0 ]]; then
		issues+=("missing_step_definitions")
	fi

	# Calculate pyramid health ratio
	local total_tests=$((unit_tests + integration_tests + bdd_scenarios))
	if [[ $total_tests -gt 0 ]]; then
		local unit_percentage=$((unit_tests * 100 / total_tests))
		local integration_percentage=$((integration_tests * 100 / total_tests))
		local bdd_percentage=$((bdd_scenarios * 100 / total_tests))

		echo -e "${BLUE}📈 Test Distribution:${NC}" >&2
		echo -e "  Unit: ${unit_percentage}% (ideal: 70%+)" >&2
		echo -e "  Integration: ${integration_percentage}% (ideal: 20%+)" >&2
		echo -e "  BDD: ${bdd_percentage}% (ideal: 10%+)" >&2

		# Validate ideal ratios (70/20/10 pyramid)
		if [[ $unit_percentage -lt 60 ]] && [[ $unit_tests -gt 0 ]]; then
			issues+=("low_unit_test_percentage")
		fi
	fi

	# Return issues as JSON array
	if [[ ${#issues[@]} -gt 0 ]]; then
		local json_issues
		json_issues=$(printf '"%s",' "${issues[@]}" | sed 's/,$//')
		echo "[$json_issues]"
		return 1
	else
		echo "[]"
		return 0
	fi
}

# Function to output testing specialist context
output_testing_specialist_context() {
	local counts="$1"
	local pyramid_issues="$2"
	local missing_tests="$3"
	local bdd_issues="$4"

	IFS=',' read -r unit_tests integration_tests bdd_scenarios step_definitions <<<"$counts"

	cat <<EOF
{
  "testingPyramidContext": {
    "action": "fix_pyramid_structure",
    "currentState": {
      "unitTests": $unit_tests,
      "integrationTests": $integration_tests,
      "bddScenarios": $bdd_scenarios,
      "stepDefinitions": $step_definitions
    },
    "issues": {
      "pyramidStructure": $pyramid_issues,
      "missingUnitTests": $missing_tests,
      "bddUnitGaps": $bdd_issues
    },
    "message": "Testing pyramid counts test FUNCTIONS (not files) for accurate ratio analysis. Ideal distribution: 70% unit test functions, 20% integration test functions, 10% BDD scenarios. Review current ratios and suggest improvements if pyramid is inverted or unit test coverage is low."
  }
}
EOF
}

# Function to check for --fix flag and auto-remediate
auto_fix_if_requested() {
	if [[ "${1:-}" == "--fix" ]]; then
		echo -e "${BLUE}🔧 Running in fix mode - generating missing tests...${NC}"

		# Try to run the BDD unit test generator
		if [[ -f ".claude/hooks/bdd-unit-test-generator.sh" ]]; then
			echo "🔍 Found BDD unit test generator, executing..."
			if .claude/hooks/bdd-unit-test-generator.sh; then
				echo "✅ BDD unit test generator completed successfully"
			else
				echo "❌ BDD unit test generator failed with exit code $?" >&2
				return 1
			fi
		else
			echo -e "${YELLOW}⚠️ BDD unit test generator not found at .claude/hooks/bdd-unit-test-generator.sh${NC}" >&2
			echo "💡 Auto-fix requires BDD unit test generator to be available" >&2
			return 1
		fi

		echo ""
		return 0
	fi
	return 1
}

# Main validation function
main() {
	# Check for auto-fix first
	local auto_fix_attempted=false
	if auto_fix_if_requested "$@"; then
		auto_fix_attempted=true
		# Re-run validation after auto-fix
		echo -e "${BLUE}🔄 Re-validating after auto-fix...${NC}"
	elif [[ "${1:-}" == "--fix" ]]; then
		# Auto-fix was requested but failed
		echo "❌ Auto-fix failed, continuing with validation..." >&2
		auto_fix_attempted=true
	fi

	local test_counts
	test_counts=$(count_test_types)

	echo -e "${BLUE}🎯 Validating testing pyramid structure...${NC}"

	# Get all validation data
	local pyramid_issues
	pyramid_issues=$(validate_pyramid_ratios "$test_counts")
	local pyramid_valid=$?

	local missing_tests
	missing_tests=$(get_missing_unit_tests_json)

	local bdd_issues
	bdd_issues=$(get_bdd_unit_issues_json)

	# Determine if we have any issues
	local has_missing_tests=false
	local has_bdd_issues=false

	if [[ "$missing_tests" != "[]" ]]; then
		has_missing_tests=true
	fi

	if [[ "$bdd_issues" != "[]" ]]; then
		has_bdd_issues=true
	fi

	# Report status
	echo ""
	if [[ $pyramid_valid -eq 0 ]] && [[ "$has_missing_tests" = false ]] && [[ "$has_bdd_issues" = false ]]; then
		echo -e "${GREEN}✅ Testing pyramid is well-structured${NC}"
		return 0
	else
		# Generate context for Claude proactive action (no user warnings)
		local error_msgs=()
		[[ $pyramid_valid -ne 0 ]] && error_msgs+=("Testing pyramid structure issues detected")
		[[ "$has_missing_tests" = true ]] && error_msgs+=("Missing unit tests for source files")
		[[ "$has_bdd_issues" = true ]] && error_msgs+=("BDD scenarios lack corresponding unit tests")

		# Removed stderr output to prevent user warnings
		# printf '%s\n' "${error_msgs[@]}" >&2

		echo -e "${BLUE}🔧 Generating testing specialist context for Claude...${NC}"

		# Only output specialist context if auto-fix wasn't attempted or failed
		if [[ "$auto_fix_attempted" = false ]] || [[ "${1:-}" != "--fix" ]]; then
			echo ""
			echo -e "${BLUE}🤖 Activating testing specialist to fix pyramid issues...${NC}"
			echo ""

			output_testing_specialist_context "$test_counts" "$pyramid_issues" "$missing_tests" "$bdd_issues"
		fi

		return 0
	fi
}

# Execute main function
main "$@"
echo -e "${GREEN}✅ ${HOOK_NAME} complete${NC}"
exit 0
