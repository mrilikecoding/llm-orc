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
		bdd_scenarios=$(find tests/bdd/features -name "*.feature" 2>/dev/null |
			xargs grep -c "^[[:space:]]*Scenario:" 2>/dev/null |
			awk -F: '{sum += $2} END {print sum}' 2>/dev/null || echo 0)
	fi

	# Count BDD step definitions
	if [[ -d "tests/bdd" ]]; then
		step_definitions=$(find tests/bdd -name "test_*.py" 2>/dev/null | wc -l)
	fi

	echo "$unit_tests,$integration_tests,$bdd_scenarios,$step_definitions"
}

# Function to get missing unit tests as JSON array
get_missing_unit_tests_json() {
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
				local line_count
				line_count=$(grep -c -v '^[[:space:]]*#\|^[[:space:]]*$\|^[[:space:]]*import\|^[[:space:]]*from' "$src_file" || echo 0)
				if [[ $line_count -gt 10 ]]; then
					missing_tests+=("\"$src_file\"")
				fi
			fi
		done < <(find src -name "*.py" -type f -print0 2>/dev/null)
	fi

	# Output as JSON array
	if [[ ${#missing_tests[@]} -gt 0 ]]; then
		echo "[$(
			IFS=,
			echo "${missing_tests[*]}"
		)]"
	else
		echo "[]"
	fi
}

# Function to get BDD issues as JSON array
get_bdd_unit_issues_json() {
	local relationship_issues=()

	# Find BDD feature files
	if [[ -d "tests/bdd/features" ]]; then
		while IFS= read -r -d '' bdd_file; do
			# Extract issue number if present
			local issue_num
			issue_num=$(basename "$bdd_file" | grep -o 'issue-[0-9]\+' | grep -o '[0-9]\+' || echo "")

			if [[ -n "$issue_num" ]]; then
				local expected_unit_file="tests/test_issue_${issue_num}_units.py"
				if [[ ! -f "$expected_unit_file" ]]; then
					relationship_issues+=("\"$(basename "$bdd_file")\"")
				fi
			fi
		done < <(find tests/bdd/features -name "*.feature" -print0 2>/dev/null)
	fi

	# Output as JSON array
	if [[ ${#relationship_issues[@]} -gt 0 ]]; then
		echo "[$(
			IFS=,
			echo "${relationship_issues[*]}"
		)]"
	else
		echo "[]"
	fi
}

# Function to validate pyramid ratios
validate_pyramid_ratios() {
	local counts="$1"
	IFS=',' read -r unit_tests integration_tests bdd_scenarios step_definitions <<<"$counts"

	echo -e "${PURPLE}📊 Current Testing Pyramid:${NC}"
	echo -e "  🧪 Unit Tests: $unit_tests"
	echo -e "  🔗 Integration Tests: $integration_tests"
	echo -e "  🎭 BDD Scenarios: $bdd_scenarios"
	echo -e "  📋 BDD Step Definitions: $step_definitions"

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

		echo -e "${BLUE}📈 Test Distribution:${NC}"
		echo -e "  Unit: ${unit_percentage}% (ideal: 70%+)"
		echo -e "  Integration: ${integration_percentage}% (ideal: 20%+)"
		echo -e "  BDD: ${bdd_percentage}% (ideal: 10%+)"

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
    "message": "Use the llm-orc-testing-specialist agent to analyze the testing pyramid structure and generate missing unit tests. Focus on creating unit tests that support BDD scenarios and maintain proper pyramid ratios (70% unit, 20% integration, 10% BDD). Generate unit test stubs for source files that lack coverage and ensure BDD scenarios have corresponding unit-level validation."
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
		# Report specific issues to stderr for Claude Code
		local error_msgs=()
		[[ $pyramid_valid -ne 0 ]] && error_msgs+=("Testing pyramid structure issues detected")
		[[ "$has_missing_tests" = true ]] && error_msgs+=("Missing unit tests for source files")
		[[ "$has_bdd_issues" = true ]] && error_msgs+=("BDD scenarios lack corresponding unit tests")

		printf '%s\n' "${error_msgs[@]}" >&2

		echo -e "${YELLOW}⚠️ Testing pyramid needs attention${NC}"

		# Only output specialist context if auto-fix wasn't attempted or failed
		if [[ "$auto_fix_attempted" = false ]] || [[ "${1:-}" != "--fix" ]]; then
			echo ""
			echo -e "${BLUE}🤖 Activating testing specialist to fix pyramid issues...${NC}"
			echo ""

			output_testing_specialist_context "$test_counts" "$pyramid_issues" "$missing_tests" "$bdd_issues"
		fi

		return 1
	fi
}

# Execute main function
if main "$@"; then
	echo -e "${GREEN}✅ ${HOOK_NAME} complete${NC}"
	exit 0
else
	echo -e "${GREEN}✅ ${HOOK_NAME} complete${NC}"
	exit 1
fi
