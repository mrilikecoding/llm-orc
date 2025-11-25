#!/bin/bash
# BDD Test Migrator
#
# Migrates passing unit tests from issue-specific files to proper module locations
# Triggers after individual BDD scenarios pass to maintain clean test organization
#
# Triggers: PostToolUse (pytest/test/coverage), Manual
# Integration: Continuous test organization maintenance

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

HOOK_NAME="BDD Test Migrator"

echo -e "${BLUE}🔄 ${HOOK_NAME}${NC}"

# Function to extract current issue number from branch
get_current_issue() {
    local branch=$(git branch --show-current 2>/dev/null || echo "")
    echo "$branch" | grep -o '[0-9]\+' | head -1 || echo ""
}

# Function to identify passing unit tests
identify_passing_tests() {
    local issue_number="$1"
    local issue_test_file="tests/test_issue_${issue_number}_units.py"

    if [[ ! -f "$issue_test_file" ]]; then
        echo -e "${BLUE}ℹ️ No issue-specific test file found: $issue_test_file${NC}" >&2
        echo -e "${BLUE}ℹ️ Tests may already be migrated - this is expected after migration${NC}" >&2
        # Return with empty stdout - don't output anything for caller to process
        return 0
    fi

    echo -e "${BLUE}🔍 Identifying passing tests in: $(basename "$issue_test_file")${NC}" >&2

    # Run just the issue-specific tests to see which pass
    local test_output
    local test_exit_code

    # Run tests without coverage to avoid failing due to coverage thresholds
    if ! test_output=$(uv run python -m pytest "$issue_test_file" -v --tb=no --no-header --no-summary --no-cov 2>&1); then
        test_exit_code=$?
        echo -e "${RED}❌ pytest failed to run (exit code $test_exit_code):${NC}" >&2
        echo "$test_output" >&2
        return $test_exit_code
    fi

    local passing_tests=()

    while IFS= read -r line; do
        if [[ "$line" =~ PASSED ]]; then
            # Extract test method name more robustly
            local test_name=$(echo "$line" | sed -E 's/.*::([^:]*) PASSED.*/\1/' | tr -d ' ')
            if [[ -n "$test_name" && "$test_name" != "$line" ]]; then
                passing_tests+=("$test_name")
            fi
        fi
    done <<< "$test_output"

    if [[ ${#passing_tests[@]} -eq 0 ]]; then
        echo -e "${BLUE}ℹ️ No passing tests found - staying in Red/Green phase${NC}" >&2
        return 0
    fi

    echo -e "${GREEN}✅ Found ${#passing_tests[@]} passing tests:${NC}" >&2
    printf '%s\n' "${passing_tests[@]}" | sed 's/^/  • /' >&2

    # Output the test names to stdout for capture by caller
    echo "${passing_tests[@]}"
    return 0
}

# Function to determine target module for test migration
determine_target_module() {
    local test_name="$1"
    local issue_test_file="$2"

    # Analyze test content to determine which module it's testing
    local test_content=$(sed -n "/def ${test_name}/,/^    def\\|^$/p" "$issue_test_file")

    # Look for import patterns or module references
    if [[ "$test_content" =~ script_agent|ScriptAgent ]]; then
        if [[ "$test_content" =~ enhanced|Enhanced ]]; then
            echo "tests/agents/test_enhanced_script_agent.py"
        else
            echo "tests/agents/test_script_agent.py"
        fi
    elif [[ "$test_content" =~ script_resolver|ScriptResolver ]]; then
        echo "tests/core/execution/test_script_resolver.py"
    elif [[ "$test_content" =~ schema|Schema ]]; then
        echo "tests/schemas/test_script_agent.py"
    else
        # Default fallback - analyze imports in original file
        local imports=$(head -20 "$issue_test_file" | grep "from llm_orc" | head -1)
        if [[ "$imports" =~ agents ]]; then
            echo "tests/agents/test_script_agent.py"
        elif [[ "$imports" =~ schemas ]]; then
            echo "tests/schemas/test_script_agent.py"
        else
            echo "tests/test_generic_units.py"
        fi
    fi
}

# Function to migrate a single test
migrate_test() {
    local test_name="$1"
    local source_file="$2"
    local target_file="$3"

    echo -e "${BLUE}📦 Migrating ${test_name} → $(basename "$target_file")${NC}"

    # Extract the test method and its docstring
    local test_content=$(sed -n "/def ${test_name}/,/^    def\\|^class\\|^$/p" "$source_file" | head -n -1)

    # Ensure target directory exists
    local target_dir=$(dirname "$target_file")
    mkdir -p "$target_dir"

    # Create target file if it doesn't exist
    if [[ ! -f "$target_file" ]]; then
        local module_name=$(basename "$target_file" .py | sed 's/test_//')
        cat > "$target_file" << EOF
"""Unit tests for ${module_name}."""

import pytest


class Test$(echo "$module_name" | sed 's/_//g' | awk '{print toupper(substr($0,1,1)) tolower(substr($0,2))}'):
    """Unit tests for ${module_name}."""
    pass
EOF
    fi

    # Add the migrated test
    # Remove the "pass" placeholder if it exists
    if grep -q "pass$" "$target_file"; then
        sed -i '' '/pass$/d' "$target_file"
    fi

    # Append the test method
    echo "" >> "$target_file"
    echo "$test_content" >> "$target_file"

    # Remove test from source file
    # Use a more robust approach to remove just this test
    local temp_file=$(mktemp)
    awk "
        /def ${test_name}/ { skip=1; next }
        skip && /^    def / { skip=0 }
        skip && /^class / { skip=0 }
        skip && /^$/ && prev_empty { skip=0; next }
        !skip { print }
        { prev_empty = (\$0 == \"\") }
    " "$source_file" > "$temp_file"

    mv "$temp_file" "$source_file"

    echo -e "${GREEN}✅ Migrated ${test_name}${NC}"
}

# Function to update issue test file imports
update_issue_file_imports() {
    local issue_test_file="$1"

    # If file is now mostly empty (just imports and class definition), clean it up
    local content_lines=$(grep -c -v '^[[:space:]]*#\|^[[:space:]]*$\|^[[:space:]]*import\|^[[:space:]]*from\|^[[:space:]]*"""\|^[[:space:]]*class\|^[[:space:]]*pass' "$issue_test_file" 2>/dev/null || echo 0)

    if [[ $content_lines -eq 0 ]]; then
        echo -e "${BLUE}🗑️ Issue test file is now empty, removing...${NC}"
        rm "$issue_test_file"
        echo -e "${GREEN}✅ Cleaned up empty issue test file${NC}"
    else
        echo -e "${BLUE}ℹ️ Issue test file still has $(( content_lines )) test methods remaining${NC}"
    fi
}

# Main migration workflow
main() {
    local issue_number
    issue_number=$(get_current_issue)

    if [[ -z "$issue_number" ]]; then
        echo -e "${YELLOW}⚠️ Could not determine issue number from branch name${NC}" >&2
        echo "Please run from a feature branch like 'feature/24-script-agents'" >&2
        return 1
    fi

    echo -e "${BLUE}🎯 Processing Issue #${issue_number} test migration${NC}"

    local issue_test_file="tests/test_issue_${issue_number}_units.py"
    local passing_tests_result

    # Capture both output and exit code from identify_passing_tests
    if ! passing_tests_result=$(identify_passing_tests "$issue_number"); then
        local exit_code=$?
        echo -e "${RED}❌ Failed to identify passing tests (exit code: $exit_code)${NC}" >&2
        return $exit_code
    fi

    if [[ -z "$passing_tests_result" ]]; then
        return 0
    fi

    # Convert result to array
    local passing_tests=($passing_tests_result)

    # Check if we're in an interactive terminal (for manual runs vs hook runs)
    if [[ -t 0 && -t 1 ]]; then
        # Interactive mode - ask user what to do
        echo ""
        echo "Would you like to migrate these passing tests to their proper module locations?"
        echo "1. Yes - migrate all passing tests"
        echo "2. Select specific tests to migrate"
        echo "3. Skip migration (keep in issue file)"

        read -p "Choice (1/2/3): " -n 1 -r choice
        echo

        case "$choice" in
            1)
                for test_name in "${passing_tests[@]}"; do
                    local target_file=$(determine_target_module "$test_name" "$issue_test_file")
                    migrate_test "$test_name" "$issue_test_file" "$target_file"
                done
                ;;
            2)
                echo "Select tests to migrate (space-separated numbers):"
                for i in "${!passing_tests[@]}"; do
                    echo "$((i+1)). ${passing_tests[$i]}"
                done
                read -p "Selection: " -r selection

                for num in $selection; do
                    if [[ "$num" =~ ^[0-9]+$ ]] && [[ $num -le ${#passing_tests[@]} ]] && [[ $num -gt 0 ]]; then
                        local test_name="${passing_tests[$((num-1))]}"
                        local target_file=$(determine_target_module "$test_name" "$issue_test_file")
                        migrate_test "$test_name" "$issue_test_file" "$target_file"
                    fi
                done
                ;;
            3)
                echo "Skipping migration - tests remain in issue file"
                return 0
                ;;
            *)
                echo "Invalid choice, skipping migration"
                return 0
                ;;
        esac
    else
        # Non-interactive mode (called as hook) - report but don't migrate automatically
        echo -e "${BLUE}ℹ️ Running in non-interactive mode (as post-hook)${NC}"
        echo -e "${BLUE}ℹ️ Found passing tests that could be migrated:${NC}"
        printf '%s\n' "${passing_tests[@]}" | sed 's/^/  • /'
        echo ""
        echo -e "${YELLOW}💡 To migrate these tests, run manually:${NC}"
        echo "  .claude/hooks/bdd-test-migrator.sh"
        return 0
    fi

    # Update issue file (only in interactive mode)
    update_issue_file_imports "$issue_test_file"

    echo ""
    echo -e "${GREEN}✅ Test migration complete${NC}"
    echo -e "${BLUE}💡 Next steps:${NC}"
    echo "  1. Run full test suite to ensure migrations worked: make test"
    echo "  2. Continue with next BDD scenario"
    echo "  3. Commit the test organization improvements"
}

# Execute main function
main "$@"

echo -e "${GREEN}✅ ${HOOK_NAME} complete${NC}"
exit 0