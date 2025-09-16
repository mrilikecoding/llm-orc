#!/bin/bash

# Implementation Checkpoint Hook (Manual/Auto)
# Continuous validation during development to catch architectural drift
# Validates implementation against BDD scenarios and ADR compliance

set -e

echo "🔄 Implementation Checkpoint"
echo ""

# Function to detect current development context
get_development_context() {
    local current_branch=$(git branch --show-current 2>/dev/null)
    local issue_number=""
    local commits_since_main=0
    local modified_files_count=0
    
    if [[ "$current_branch" =~ feature/([0-9]+) ]]; then
        issue_number="${BASH_REMATCH[1]}"
    fi
    
    commits_since_main=$(git rev-list --count HEAD ^main 2>/dev/null || echo "0")
    modified_files_count=$(git diff --name-only main...HEAD 2>/dev/null | wc -l | tr -d ' ')
    
    echo "$current_branch|$issue_number|$commits_since_main|$modified_files_count"
}

# Function to run BDD scenario validation
validate_bdd_scenarios() {
    local issue_number=$1
    local scenario_file="tests/bdd/features/issue-${issue_number}.feature"
    
    echo "🎭 Validating BDD scenarios..."
    
    if [ ! -f "$scenario_file" ]; then
        echo "⚠️ No BDD scenarios found for issue #${issue_number}"
        echo "   Consider running: .claude/hooks/pre-implementation-gate.sh"
        return 1
    fi
    
    if command -v uv &>/dev/null; then
        echo "  Running: uv run pytest $scenario_file"
        if uv run pytest "$scenario_file" --tb=short -q; then
            echo "  ✅ BDD scenarios passing"
            return 0
        else
            echo "  ❌ BDD scenarios failing"
            return 1
        fi
    else
        echo "  ⚠️ uv not available - install uv for validation"
        return 0
    fi
}

# Function to check architectural compliance
check_architectural_compliance() {
    echo "🏗️ Checking architectural compliance..."
    
    local compliance_issues=()
    local compliance_score=0
    local max_score=4
    
    # Check 1: Type annotations compliance
    if command -v mypy &>/dev/null; then
        if mypy src/llm_orc/ --strict --no-error-summary >/dev/null 2>&1; then
            echo "  ✅ Type annotations compliant (mypy strict)"
            compliance_score=$((compliance_score + 1))
        else
            echo "  ❌ Type annotation issues detected"
            compliance_issues+=("type_annotations")
        fi
    else
        echo "  ⚠️ mypy not available for type checking"
    fi
    
    # Check 2: Code formatting compliance
    if command -v ruff &>/dev/null; then
        if ruff check src/llm_orc/ >/dev/null 2>&1; then
            echo "  ✅ Code formatting compliant (ruff)"
            compliance_score=$((compliance_score + 1))
        else
            echo "  ❌ Code formatting issues detected"
            compliance_issues+=("code_formatting")
        fi
    fi
    
    # Check 3: Test coverage
    if command -v uv &>/dev/null && [ -f "pyproject.toml" ]; then
        local coverage_report=$(uv run pytest --cov=src/llm_orc --cov-report=term-missing --tb=no -q 2>/dev/null | grep "TOTAL" || echo "")
        if [[ "$coverage_report" =~ ([0-9]+)% ]]; then
            local coverage_percent="${BASH_REMATCH[1]}"
            if [ "$coverage_percent" -ge 95 ]; then
                echo "  ✅ Test coverage adequate (${coverage_percent}%)"
                compliance_score=$((compliance_score + 1))
            else
                echo "  ⚠️ Test coverage below threshold (${coverage_percent}% < 95%)"
                compliance_issues+=("test_coverage")
            fi
        fi
    else
        echo "  ⚠️ Cannot check test coverage (pytest/pyproject.toml missing)"
    fi
    
    # Check 4: No obvious anti-patterns in recent changes
    local recent_files=$(git diff --name-only HEAD~1 2>/dev/null | grep "\.py$" || echo "")
    if [ -n "$recent_files" ]; then
        local anti_patterns_found=0
        while IFS= read -r file; do
            if [ -f "$file" ]; then
                # Check for common anti-patterns
                if grep -q "except:" "$file" 2>/dev/null; then
                    echo "  ⚠️ Bare except clause found in $file"
                    anti_patterns_found=$((anti_patterns_found + 1))
                fi
                
                if grep -q "from.*import \*" "$file" 2>/dev/null; then
                    echo "  ⚠️ Star import found in $file"
                    anti_patterns_found=$((anti_patterns_found + 1))
                fi
            fi
        done <<< "$recent_files"
        
        if [ $anti_patterns_found -eq 0 ]; then
            echo "  ✅ No obvious anti-patterns detected"
            compliance_score=$((compliance_score + 1))
        else
            compliance_issues+=("anti_patterns")
        fi
    else
        echo "  ⚠️ No recent Python files to check for anti-patterns"
    fi
    
    echo "  📊 Architectural compliance: ${compliance_score}/${max_score}"
    
    # Return compliance status and issues
    if [ $compliance_score -ge 3 ]; then
        echo "compliant"
    else
        echo "non_compliant"
    fi
    
    printf '%s\n' "${compliance_issues[@]}"
}

# Function to analyze development velocity and patterns
analyze_development_patterns() {
    local commits_count=$1
    local files_count=$2
    
    echo "📈 Development pattern analysis..."
    
    # Commit frequency analysis
    if [ "$commits_count" -gt 10 ]; then
        echo "  ⚠️ High commit count (${commits_count}) - consider feature branch cleanup"
    elif [ "$commits_count" -eq 0 ]; then
        echo "  💡 No commits yet - remember to commit early and often"
    else
        echo "  ✅ Reasonable commit frequency (${commits_count} commits)"
    fi
    
    # File change analysis
    if [ "$files_count" -gt 20 ]; then
        echo "  ⚠️ Many files modified (${files_count}) - ensure changes are cohesive"
    elif [ "$files_count" -eq 0 ]; then
        echo "  💡 No files modified yet"
    else
        echo "  ✅ Focused changes (${files_count} files)"
    fi
    
    # Recent commit message analysis
    local recent_commit_msg=$(git log -1 --pretty=format:"%s" 2>/dev/null || echo "")
    if [[ "$recent_commit_msg" =~ ^(feat|fix|refactor|test|docs): ]]; then
        echo "  ✅ Recent commit follows conventional format"
    elif [ -n "$recent_commit_msg" ]; then
        echo "  💡 Consider conventional commit format: feat/fix/refactor/test/docs:"
    fi
}

# Function to provide actionable recommendations
provide_recommendations() {
    local context=$1
    local bdd_status=$2
    local compliance_status=$3
    shift 3
    local compliance_issues=("$@")
    
    local issue_number=$(echo "$context" | cut -d'|' -f2)
    
    echo ""
    echo "💡 Checkpoint Recommendations:"
    
    # BDD-specific recommendations
    if [ "$bdd_status" = "failing" ]; then
        echo ""
        echo "🎭 BDD Scenarios Need Attention:"
        echo "  1. Review failing scenarios: uv run pytest tests/bdd/features/issue-${issue_number}.feature -v"
        echo "  2. Update scenarios if requirements changed"
        echo "  3. Fix implementation if behavioral contracts are correct"
        echo ""
        echo "🤖 Consider activating: llm-orc-bdd-specialist for scenario analysis"
    fi
    
    # Compliance-specific recommendations  
    if [ "$compliance_status" = "non_compliant" ]; then
        echo ""
        echo "🏗️ Architectural Compliance Issues:"
        
        for issue in "${compliance_issues[@]}"; do
            case "$issue" in
                "type_annotations")
                    echo "  • Fix type annotations: mypy src/llm_orc/ --strict"
                    echo "    🤖 Consider: llm-orc-architecture-reviewer"
                    ;;
                "code_formatting")
                    echo "  • Fix formatting: ruff check --fix src/llm_orc/"
                    echo "  • Format code: ruff format src/llm_orc/"
                    ;;
                "test_coverage")
                    echo "  • Improve test coverage: uv run pytest --cov=src/llm_orc --cov-report=html"
                    echo "    🤖 Consider: llm-orc-tdd-specialist"
                    ;;
                "anti_patterns")
                    echo "  • Review recent changes for coding standards compliance"
                    echo "    🤖 Consider: llm-orc-architecture-reviewer"
                    ;;
            esac
        done
    fi
    
    # General development recommendations
    echo ""
    echo "🚀 Next Steps:"
    echo "  • Continue TDD cycle: Red → Green → Refactor"
    echo "  • Run hooks before commit: .claude/hooks/bdd-development-gate.sh"
    echo "  • Validate final implementation: uv run pytest tests/bdd/features/issue-${issue_number}.feature"
}

# Main checkpoint workflow
main() {
    local auto_mode="${1:-manual}"
    
    if [ "$auto_mode" = "--auto" ]; then
        echo "🤖 Automatic checkpoint triggered by file modifications"
    else
        echo "👤 Manual implementation checkpoint"
    fi
    
    echo ""
    
    # Get development context
    local context=$(get_development_context)
    local branch_name=$(echo "$context" | cut -d'|' -f1)
    local issue_number=$(echo "$context" | cut -d'|' -f2)
    local commits_count=$(echo "$context" | cut -d'|' -f3)
    local files_count=$(echo "$context" | cut -d'|' -f4)
    
    if [ -z "$issue_number" ]; then
        echo "❓ No issue context detected"
        echo "   Working on branch: $branch_name"
        echo "   Limited validation available without issue context"
        echo ""
    else
        echo "🎯 Issue Context: #${issue_number} on $branch_name"
        echo "📊 Changes: ${commits_count} commits, ${files_count} files modified"
        echo ""
    fi
    
    # Run validations
    local bdd_status="unknown"
    local compliance_output=""
    local compliance_status="unknown"
    local compliance_issues=()
    
    # BDD validation (if we have issue context)
    if [ -n "$issue_number" ]; then
        if validate_bdd_scenarios "$issue_number"; then
            bdd_status="passing"
        else
            bdd_status="failing"
        fi
        echo ""
    fi
    
    # Architectural compliance check
    compliance_output=$(check_architectural_compliance)
    compliance_status=$(echo "$compliance_output" | grep -E "compliant|non_compliant")
    compliance_issues=($(echo "$compliance_output" | grep -v -E "✅|❌|⚠️|📊|compliant|non_compliant" | grep -v "^$"))
    echo ""
    
    # Development pattern analysis
    analyze_development_patterns "$commits_count" "$files_count"
    
    # Provide actionable recommendations
    provide_recommendations "$context" "$bdd_status" "$compliance_status" "${compliance_issues[@]}"
    
    echo ""
    echo "🔄 Implementation checkpoint complete"
    
    # Exit with appropriate code for automated workflows
    if [ "$bdd_status" = "failing" ] || [ "$compliance_status" = "non_compliant" ]; then
        exit 1
    fi
}

# Handle command line arguments
case "${1:-}" in
    --auto)
        main --auto
        ;;
    --help)
        echo "Implementation Checkpoint Hook"
        echo ""
        echo "Provides continuous validation during development to catch"
        echo "architectural drift and ensure behavioral compliance."
        echo ""
        echo "Usage:"
        echo "  $0           # Manual checkpoint with full analysis"
        echo "  $0 --auto    # Automatic checkpoint (triggered by file changes)"
        echo "  $0 --help    # Show this help"
        echo ""
        echo "Validations performed:"
        echo "  • BDD scenario compliance"
        echo "  • Architectural pattern adherence"
        echo "  • Code quality metrics"
        echo "  • Development velocity analysis"
        ;;
    *)
        main
        ;;
esac

exit 0