#!/bin/bash

# BDD Development Gate Hook (Manual)
# Generates BDD scenarios from GitHub issues and validates implementation compliance

set -e

echo "🎭 BDD Development Gate"
echo ""

# Function to check if we have a current issue context
check_issue_context() {
    local current_branch=$(git branch --show-current 2>/dev/null)
    
    if [[ "$current_branch" =~ feature/([0-9]+) ]]; then
        echo "${BASH_REMATCH[1]}"
    elif [[ "$current_branch" =~ ([0-9]+)- ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        echo ""
    fi
}

# Function to check if BDD scenarios exist for current issue
check_bdd_scenarios() {
    local issue_number=$1
    local scenario_file="tests/bdd/features/issue-${issue_number}.feature"
    
    if [ -f "$scenario_file" ]; then
        echo "found"
    else
        echo "missing"
    fi
}

# Function to run BDD scenario validation
validate_bdd_scenarios() {
    echo "🧪 Running BDD scenario validation..."
    
    if [ -d "tests/bdd" ]; then
        if command -v uv &>/dev/null; then
            if uv run pytest tests/bdd/ --tb=short -q; then
                echo "✅ All BDD scenarios passing"
                return 0
            else
                echo "❌ BDD scenarios failing"
                return 1
            fi
        else
            echo "⚠️ uv not found - install uv for scenario validation"
            return 0
        fi
    else
        echo "💡 No BDD scenarios directory found (tests/bdd/)"
        return 0
    fi
}

# Main workflow
main() {
    local issue_number=$(check_issue_context)
    
    if [ -z "$issue_number" ]; then
        echo "🤔 No GitHub issue detected in branch name"
        echo "   Consider using format: feature/24-script-agents"
        echo "   Or run: .claude/hooks/bdd-development-gate.sh --issue 24"
        exit 0
    fi
    
    echo "🎯 Working on Issue #${issue_number}"
    
    local bdd_status=$(check_bdd_scenarios "$issue_number")
    
    if [ "$bdd_status" = "missing" ]; then
        echo "📝 No BDD scenarios found for Issue #${issue_number}"
        echo ""
        
        # Check if gh CLI is available for issue context
        if command -v gh &>/dev/null; then
            echo "🤖 Generating BDD scenarios from issue context..."
            echo ""
            echo "Would you like to generate BDD scenarios? (y/n)"
            read -r response
            
            if [[ "$response" =~ ^[Yy] ]]; then
                echo ""
                echo "🎭 Activating BDD specialist to create scenarios..."
                echo ""
                
                # Output context for Claude to use BDD specialist agent
                cat << EOF
{
  "bddContext": {
    "issueNumber": "$issue_number",
    "action": "generate_scenarios",
    "message": "Use the llm-orc-bdd-specialist agent to analyze GitHub issue #$issue_number and generate comprehensive BDD scenarios. Create feature file at tests/bdd/features/issue-${issue_number}.feature with scenarios that validate both functionality and architectural compliance per relevant ADRs."
  }
}
EOF
                exit 0
            else
                echo "⏭️ Skipping BDD scenario generation"
            fi
        else
            echo "💡 Install 'gh' CLI for automated BDD scenario generation"
        fi
    else
        echo "✅ BDD scenarios found for Issue #${issue_number}"
        
        # Validate existing scenarios
        if ! validate_bdd_scenarios; then
            echo ""
            echo "🔧 BDD scenarios need attention"
            echo "   Run: uv run pytest tests/bdd/features/issue-${issue_number}.feature -v"
            echo ""
            echo "Would you like to update scenarios for current implementation? (y/n)"
            read -r response
            
            if [[ "$response" =~ ^[Yy] ]]; then
                echo ""
                echo "🎭 Activating BDD specialist to update scenarios..."
                echo ""
                
                cat << EOF
{
  "bddContext": {
    "issueNumber": "$issue_number",
    "action": "update_scenarios",
    "message": "Use the llm-orc-bdd-specialist agent to analyze failing BDD scenarios for issue #$issue_number and update them to match current implementation while maintaining architectural compliance."
  }
}
EOF
            fi
        fi
    fi
    
    echo ""
    echo "🎭 BDD Development Gate complete"
    echo ""
}

# Handle command line arguments
case "${1:-}" in
    --issue)
        if [ -n "$2" ]; then
            # Override issue detection with provided number
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
        ;;
    *)
        main
        ;;
esac

exit 0