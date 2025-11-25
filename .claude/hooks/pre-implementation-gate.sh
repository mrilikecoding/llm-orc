#!/bin/bash

# Pre-Implementation Gate Hook (Manual)
# Ensures BDD scenarios exist before implementation begins
# Provides behavioral contracts to guide LLM development

set -e

echo "🚀 Pre-Implementation Gate"
echo ""

# Function to detect current issue context
get_issue_context() {
    local current_branch=$(git branch --show-current 2>/dev/null)
    
    if [[ "$current_branch" =~ feature/([0-9]+) ]]; then
        echo "${BASH_REMATCH[1]}"
    elif [[ "$current_branch" =~ ([0-9]+)- ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        echo ""
    fi
}

# Function to check implementation readiness
check_implementation_readiness() {
    local issue_number=$1
    local readiness_score=0
    local issues=()
    
    # Check 1: BDD scenarios exist
    if [ -f "tests/bdd/features/issue-${issue_number}.feature" ]; then
        echo "✅ BDD scenarios exist"
        readiness_score=$((readiness_score + 1))
    else
        echo "❌ BDD scenarios missing"
        issues+=("bdd_scenarios_missing")
    fi
    
    # Check 2: Issue has sufficient detail
    if command -v gh &>/dev/null; then
        local issue_body=$(gh issue view "$issue_number" --json body -q .body 2>/dev/null || echo "")
        if [ ${#issue_body} -gt 200 ]; then
            echo "✅ Issue has detailed requirements"
            readiness_score=$((readiness_score + 1))
        else
            echo "⚠️ Issue may need more detail"
            issues+=("issue_detail_sparse")
        fi
    else
        echo "⚠️ Cannot validate issue detail (gh CLI missing)"
    fi
    
    # Check 3: Relevant ADRs identified
    local adr_count=$(find docs/adrs/ -name "*.md" -type f 2>/dev/null | wc -l | tr -d ' ')
    if [ "$adr_count" -gt 0 ]; then
        echo "✅ ADRs available for architectural guidance"
        readiness_score=$((readiness_score + 1))
    else
        echo "⚠️ No ADRs found for architectural context"
    fi
    
    # Check 4: Tests directory structure exists
    if [ -d "tests/" ]; then
        echo "✅ Test infrastructure ready"
        readiness_score=$((readiness_score + 1))
    else
        echo "❌ Test directory missing"
        issues+=("test_infrastructure_missing")
    fi
    
    echo ""
    echo "📊 Implementation Readiness: ${readiness_score}/4"
    
    # Return readiness status
    if [ $readiness_score -ge 3 ]; then
        echo "ready"
    else
        echo "not_ready"
    fi
    
    # Store issues for resolution
    printf '%s\n' "${issues[@]}"
}

# Function to resolve readiness issues
resolve_readiness_issues() {
    local issue_number=$1
    shift
    local issues=("$@")
    
    echo "🔧 Resolving implementation readiness issues..."
    echo ""
    
    for issue in "${issues[@]}"; do
        case "$issue" in
            "bdd_scenarios_missing")
                echo "📝 Missing BDD scenarios for Issue #${issue_number}"
                echo ""
                echo "🤖 Activating BDD specialist to generate behavioral contracts..."
                echo ""
                
                # Generate comprehensive BDD scenarios
                cat << EOF
{
  "preImplementationContext": {
    "issueNumber": "$issue_number",
    "action": "generate_comprehensive_scenarios",
    "priority": "high",
    "message": "Use llm-orc-bdd-specialist to analyze GitHub issue #$issue_number and create comprehensive BDD scenarios. Focus on:
    1. Core functionality behavioral contracts
    2. ADR compliance validation scenarios  
    3. Error handling and edge cases
    4. Integration with existing architecture
    5. LLM development guidance in scenario documentation
    
    Create tests/bdd/features/issue-${issue_number}.feature with rich context for implementation guidance."
  }
}
EOF
                echo ""
                ;;
                
            "test_infrastructure_missing")
                echo "🧪 Setting up test infrastructure..."
                mkdir -p tests/bdd/features
                mkdir -p tests/bdd/steps
                
                # Create basic conftest.py for BDD
                cat > tests/bdd/conftest.py << 'EOL'
"""BDD test configuration for llm-orc."""
import pytest
from pytest_bdd import given, when, then

# Import existing fixtures from main test suite
pytest_plugins = ["tests.conftest"]

@pytest.fixture
def bdd_context():
    """Shared context for BDD scenarios."""
    return {}
EOL
                echo "✅ Created basic BDD test infrastructure"
                ;;
                
            "issue_detail_sparse")
                echo "📋 Issue may need more detail for comprehensive implementation"
                echo "   Consider adding:"
                echo "   - Specific acceptance criteria"
                echo "   - Integration requirements" 
                echo "   - Performance expectations"
                echo "   - Error handling requirements"
                ;;
        esac
        echo ""
    done
}

# Function to provide implementation guidance
provide_implementation_guidance() {
    local issue_number=$1
    
    echo "🎯 Implementation Guidance for Issue #${issue_number}"
    echo ""
    echo "Recommended development flow:"
    echo "1. 📖 Read BDD scenarios in tests/bdd/features/issue-${issue_number}.feature"
    echo "2. 🔴 TDD Red: Write failing tests based on BDD behavioral contracts"
    echo "3. 🟢 TDD Green: Implement minimal solution to satisfy behaviors"
    echo "4. 🔄 TDD Refactor: Improve structure while preserving behavior"
    echo "5. ✅ Validate: Run BDD scenarios to ensure architectural compliance"
    echo ""
    echo "Specialized agents available:"
    echo "• llm-orc-tdd-specialist: Enforces TDD discipline"
    echo "• llm-orc-architecture-reviewer: Validates ADR compliance" 
    echo "• llm-orc-performance-optimizer: Optimizes async patterns"
    echo "• llm-orc-security-auditor: Reviews security aspects"
    echo ""
    echo "Quality gates:"
    echo "• PostToolUse hooks: Automatic formatting and complexity checks"
    echo "• Implementation checkpoints: Continuous behavioral validation"
    echo "• BDD validation: uv run pytest tests/bdd/features/issue-${issue_number}.feature"
}

# Main workflow
main() {
    local issue_number=$(get_issue_context)
    
    if [ -z "$issue_number" ]; then
        echo "❓ No GitHub issue detected in branch name"
        echo "   Expected format: feature/24-script-agents"
        echo ""
        echo "Manual usage: $0 --issue <number>"
        exit 0
    fi
    
    echo "🎯 Preparing implementation for Issue #${issue_number}"
    echo ""
    
    # Check implementation readiness
    local readiness_output=$(check_implementation_readiness "$issue_number")
    local readiness_status=$(echo "$readiness_output" | grep -E "ready|not_ready")
    local issues=($(echo "$readiness_output" | grep -v -E "✅|❌|⚠️|📊|ready|not_ready"))
    
    if [ "$readiness_status" = "not_ready" ]; then
        echo ""
        resolve_readiness_issues "$issue_number" "${issues[@]}"
        
        echo ""
        echo "🔄 Run this hook again after resolving issues to proceed with implementation"
        exit 0
    else
        echo ""
        echo "🎉 Implementation ready!"
        echo ""
        provide_implementation_guidance "$issue_number"
    fi
    
    echo ""
    echo "🚀 Pre-implementation gate complete - ready to code!"
}

# Handle command line arguments
case "${1:-}" in
    --issue)
        if [ -n "$2" ]; then
            # Override issue detection
            ISSUE_NUMBER="$2"
            main
        else
            echo "Usage: $0 --issue <issue_number>"
            exit 1
        fi
        ;;
    --check)
        # Just check readiness without resolution
        issue_number=$(get_issue_context)
        if [ -n "$issue_number" ]; then
            readiness_output=$(check_implementation_readiness "$issue_number")
            echo "$readiness_output" | grep -v -E "ready|not_ready"
        else
            echo "No issue context available"
        fi
        ;;
    --help)
        echo "Pre-Implementation Gate Hook"
        echo ""
        echo "Ensures you have behavioral contracts (BDD scenarios) before"
        echo "starting implementation, providing clear guidance for LLM development."
        echo ""
        echo "Usage:"
        echo "  $0                    # Auto-detect issue and check readiness"
        echo "  $0 --issue <number>   # Check readiness for specific issue"
        echo "  $0 --check            # Quick readiness check only"
        echo "  $0 --help             # Show this help"
        echo ""
        ;;
    *)
        main
        ;;
esac

exit 0