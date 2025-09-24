#!/bin/bash

# BDD Development Gate Hook (Manual)
# Generates BDD scenarios from GitHub issues and validates implementation compliance

# Don't exit on error immediately - we want to provide useful feedback
set +e

echo "🎭 BDD Development Gate"
echo ""

# Function to check if we have a current issue context
get_issue_number_from_branch() {
	local current_branch
	current_branch=$(git branch --show-current 2>/dev/null)

	if [[ "$current_branch" =~ feature/([0-9]+) ]]; then
		echo "${BASH_REMATCH[1]}"
	elif [[ "$current_branch" =~ ([0-9]+)- ]]; then
		echo "${BASH_REMATCH[1]}"
	else
		echo ""
	fi
}

# Function to check if BDD scenarios exist for current issue
bdd_scenarios_exist() {
	local issue_number=$1
	local patterns=(
		"tests/bdd/features/issue-${issue_number}.feature"
		"tests/bdd/features/issue-${issue_number}-*.feature"
		"tests/bdd/features/adr-*-issue-${issue_number}.feature"
	)

	for pattern in "${patterns[@]}"; do
		if ls $pattern >/dev/null 2>&1; then
			return 0 # scenarios found
		fi
	done

	return 1 # scenarios not found
}

# Function to run BDD scenario validation
validate_bdd_scenarios() {
	echo "🧪 Running BDD scenario validation..."

	if [ ! -d "tests/bdd" ]; then
		echo "💡 No BDD scenarios directory found (tests/bdd/)"
		return 0
	fi

	if ! command -v uv &>/dev/null; then
		echo "⚠️ uv not found - install uv for scenario validation"
		return 0
	fi

	local result
	result=$(uv run pytest tests/bdd/ --tb=short -q 2>&1)
	local exit_code=$?

	if [ $exit_code -eq 0 ]; then
		echo "✅ All BDD scenarios passing"
		return 0
	else
		echo "❌ BDD scenarios failing"
		echo "   Error details:"
		echo "$result" | head -n 10 | sed 's/^/   /'
		return 1
	fi
}

# Function to output BDD context JSON
output_bdd_context() {
	local issue_number=$1
	local action=$2
	local message=$3

	cat <<EOF
{
  "bddContext": {
    "issueNumber": "$issue_number",
    "action": "$action",
    "message": "$message"
  }
}
EOF
}

# Function to handle missing BDD scenarios
handle_missing_scenarios() {
	local issue_number=$1

	echo "📝 No BDD scenarios found for Issue #${issue_number}"
	echo ""

	if ! command -v gh &>/dev/null; then
		echo "💡 Install 'gh' CLI for automated BDD scenario generation"
		return 0
	fi

	if [ ! -t 0 ]; then
		# Non-interactive mode
		echo "🤖 Non-interactive mode: BDD scenarios needed for Issue #${issue_number}"
		echo ""
		echo "💡 Suggested action: Generate BDD scenarios"
		echo ""

		output_bdd_context "$issue_number" "generate_scenarios" \
			"Use the llm-orc-bdd-specialist agent to analyze GitHub issue #$issue_number and generate comprehensive BDD scenarios. Create feature file at tests/bdd/features/issue-${issue_number}.feature with scenarios that validate both functionality and architectural compliance per relevant ADRs."
		return 0
	fi

	# Interactive mode
	echo "🤖 Generating BDD scenarios from issue context..."
	echo ""
	echo "Would you like to generate BDD scenarios? (y/n)"
	read -r response

	if [[ "$response" =~ ^[Yy] ]]; then
		echo ""
		echo "🎭 Activating BDD specialist to create scenarios..."
		echo ""

		output_bdd_context "$issue_number" "generate_scenarios" \
			"Use the llm-orc-bdd-specialist agent to analyze GitHub issue #$issue_number and generate comprehensive BDD scenarios. Create feature file at tests/bdd/features/issue-${issue_number}.feature with scenarios that validate both functionality and architectural compliance per relevant ADRs."
		exit 0
	else
		echo "⏭️ Skipping BDD scenario generation"
	fi
}

# Function to handle existing scenarios that need validation
handle_existing_scenarios() {
	local issue_number=$1

	echo "✅ BDD scenarios found for Issue #${issue_number}"

	if validate_bdd_scenarios; then
		return 0 # All good
	fi

	# Validation failed
	echo ""
	echo "🔧 BDD scenarios need attention"
	echo "   Run: uv run pytest tests/bdd/features/issue-${issue_number}.feature -v"
	echo ""

	if [ ! -t 0 ]; then
		return 0 # Non-interactive, just report the issue
	fi

	# Interactive mode - offer to update scenarios
	echo "Would you like to update scenarios for current implementation? (y/n)"
	read -r response

	if [[ "$response" =~ ^[Yy] ]]; then
		echo ""
		echo "🎭 Activating BDD specialist to update scenarios..."
		echo ""

		output_bdd_context "$issue_number" "update_scenarios" \
			"Use the llm-orc-bdd-specialist agent to analyze failing BDD scenarios for issue #$issue_number and update them to match current implementation while maintaining architectural compliance."
	fi
}

# Function to get issue number (from parameter or branch)
get_issue_number() {
	if [ -n "${ISSUE_NUMBER:-}" ]; then
		echo "$ISSUE_NUMBER"
	else
		get_issue_number_from_branch
	fi
}

# Function to handle no issue detected
handle_no_issue() {
	echo "🤔 No GitHub issue detected in branch name"
	echo "   Current branch: $(git branch --show-current 2>/dev/null || echo 'unknown')"
	echo "   Consider using format: feature/24-script-agents"
	echo "   Or run: .claude/hooks/bdd-development-gate.sh --issue 24"
	return 0
}

# Main workflow
main() {
	local issue_number
	issue_number=$(get_issue_number)

	if [ -z "$issue_number" ]; then
		handle_no_issue
		return 0
	fi

	echo "🎯 Working on Issue #${issue_number}"

	if bdd_scenarios_exist "$issue_number"; then
		handle_existing_scenarios "$issue_number"
	else
		handle_missing_scenarios "$issue_number"
	fi

	echo ""
	echo "🎭 BDD Development Gate complete"
	echo ""
	return 0
}

# Function to show help
show_help() {
	echo "BDD Development Gate Hook"
	echo ""
	echo "Usage:"
	echo "  $0                    # Auto-detect issue from branch name"
	echo "  $0 --issue <number>   # Specify issue number"
	echo "  $0 --validate         # Validate existing BDD scenarios"
	echo "  $0 --help             # Show this help"
	echo ""
	echo "This hook integrates BDD scenario generation and validation into"
	echo "the llm-orc development workflow, ensuring behavioral compliance"
	echo "with architectural decisions and providing LLM development guidance."
}

# Handle command line arguments
case "${1:-}" in
--issue)
	if [ -n "$2" ]; then
		ISSUE_NUMBER="$2"
		main
	else
		echo "Usage: $0 --issue <issue_number>"
		exit 1
	fi
	;;
--validate)
	validate_bdd_scenarios
	;;
--help)
	show_help
	;;
*)
	main
	;;
esac

# Always exit with success unless critical error
exit_code=$?
if [ $exit_code -ne 0 ]; then
	echo "⚠️  BDD Development Gate encountered issues but allowing workflow to continue"
	echo "   Exit code: $exit_code"
fi
exit 0
