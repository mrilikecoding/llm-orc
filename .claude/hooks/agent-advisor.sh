#!/bin/bash

# Agent Advisor Hook - Suggests relevant agents based on development context
# This hook analyzes what type of work is being done and suggests appropriate specialized agents

set -euo pipefail

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Check if we're in an interactive terminal
if [[ ! -t 0 || ! -t 1 ]]; then
    # Non-interactive mode - just log suggestions
    echo "ℹ️ Agent advisor running in non-interactive mode"
fi

# Function to suggest agents based on file patterns
suggest_agents() {
    local changed_files="$1"
    local suggestions=()

    # BDD/Testing related changes
    if echo "$changed_files" | grep -q "tests/bdd/\|\.feature$\|test_.*\.py$"; then
        suggestions+=("🧪 llm-orc-bdd-specialist: For BDD scenario creation and validation")
    fi

    # Core architecture changes
    if echo "$changed_files" | grep -q "core/execution/\|core/config/\|schemas/\|docs/adrs/"; then
        suggestions+=("🏗️ llm-orc-architecture-reviewer: For architectural alignment review")
    fi

    # TDD implementation work
    if echo "$changed_files" | grep -q "src/.*\.py$" && echo "$changed_files" | grep -q "test.*\.py$"; then
        suggestions+=("🔴🟢🔄 llm-orc-tdd-specialist: For TDD cycle discipline enforcement")
    fi

    # Pre-commit quality checks
    if echo "$changed_files" | grep -q "\.py$"; then
        suggestions+=("✅ llm-orc-precommit-specialist: For code quality and linting")
    fi

    # Performance-related changes
    if echo "$changed_files" | grep -q "execution/.*executor\|models/.*factory\|async"; then
        suggestions+=("⚡ llm-orc-performance-optimizer: For performance optimization")
    fi

    # Security-related changes
    if echo "$changed_files" | grep -q "auth\|credential\|api.*key\|security"; then
        suggestions+=("🔒 llm-orc-security-auditor: For security review")
    fi

    # UX-related changes
    if echo "$changed_files" | grep -q "cli\|error.*message\|user.*interface\|config.*management"; then
        suggestions+=("👤 llm-orc-ux-specialist: For user experience improvement")
    fi

    # Documentation updates
    if echo "$changed_files" | grep -q "README\|\.md$\|docs/"; then
        suggestions+=("📚 documentation-maintainer: For documentation maintenance")
    fi

    # Print suggestions
    if [ ${#suggestions[@]} -gt 0 ]; then
        echo -e "${BLUE}🤖 Agent Advisor Suggestions:${NC}"
        echo
        for suggestion in "${suggestions[@]}"; do
            echo -e "  ${GREEN}•${NC} $suggestion"
        done
        echo
        echo -e "${YELLOW}💡 Tip: Use these agents proactively for better development outcomes!${NC}"
        echo
    fi
}

# Get list of changed files since last commit
if command -v git &> /dev/null && git rev-parse --git-dir &> /dev/null; then
    # Get staged files
    staged_files=$(git diff --cached --name-only 2>/dev/null || echo "")

    # Get unstaged files
    unstaged_files=$(git diff --name-only 2>/dev/null || echo "")

    # Get untracked files
    untracked_files=$(git ls-files --others --exclude-standard 2>/dev/null || echo "")

    # Combine all changed files
    all_changed_files="$staged_files$unstaged_files$untracked_files"

    if [ -n "$all_changed_files" ]; then
        suggest_agents "$all_changed_files"
    fi
else
    echo -e "${YELLOW}ℹ️ Not in a git repository - agent suggestions limited${NC}"
fi

# Detect current development context
if [ -f "tests/bdd/test_adr_005_multi_turn_conversations.py" ] && grep -q "pytest.fail" tests/bdd/test_adr_005_multi_turn_conversations.py 2>/dev/null; then
    echo -e "${PURPLE}🎯 Context: ADR-005 TDD implementation in progress${NC}"
    echo -e "   ${GREEN}→${NC} Consider using llm-orc-tdd-specialist for next Red→Green cycle"
    echo
fi

# Check for common development scenarios
if [ -f ".git/COMMIT_EDITMSG" ]; then
    echo -e "${BLUE}📝 Pre-commit context detected${NC}"
    echo -e "   ${GREEN}→${NC} Consider using llm-orc-precommit-specialist for quality checks"
    echo
fi

exit 0