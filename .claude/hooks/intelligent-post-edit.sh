#!/bin/bash

# Intelligent Post-Edit Hook (PostToolUse)
# Context-aware agent activation based on file patterns and changes
# Triggers specialized agents when their expertise is most needed

set -e

# Get the files that were modified from Claude Code
# Claude Code passes modified files as arguments to PostToolUse hooks
MODIFIED_FILES=("$@")

# If no files provided, exit gracefully
if [ ${#MODIFIED_FILES[@]} -eq 0 ]; then
    exit 0
fi

# Function to determine if we should trigger agents based on file patterns
analyze_modifications() {
    local files=("$@")
    local agents_to_trigger=()
    local architectural_changes=false
    local test_changes=false
    local core_changes=false
    
    echo "🔍 Analyzing modifications to ${#files[@]} files"
    
    for file in "${files[@]}"; do
        # Core execution components - architectural significance
        if [[ "$file" =~ ^src/llm_orc/core/ ]]; then
            echo "  📁 Core component: $file"
            agents_to_trigger+=("architecture-reviewer")
            core_changes=true
            
            # Specific performance-critical components
            if [[ "$file" =~ ensemble_execution|script_.*agent ]]; then
                agents_to_trigger+=("performance-optimizer")
            fi
        fi
        
        # Schema changes affect contracts and interfaces
        if [[ "$file" =~ ^src/llm_orc/schemas/ ]]; then
            echo "  📋 Schema change: $file"
            agents_to_trigger+=("architecture-reviewer")
            agents_to_trigger+=("bdd-specialist")
            architectural_changes=true
        fi
        
        # Test file modifications
        if [[ "$file" =~ ^tests/ ]]; then
            echo "  🧪 Test change: $file"
            agents_to_trigger+=("tdd-specialist")
            test_changes=true
            
            # BDD scenario changes
            if [[ "$file" =~ tests/bdd/features/ ]]; then
                agents_to_trigger+=("bdd-specialist")
            fi
        fi
        
        # CLI and user-facing components
        if [[ "$file" =~ ^src/llm_orc/cli/ ]]; then
            echo "  💻 CLI change: $file"
            agents_to_trigger+=("ux-specialist")
        fi
        
        # Security-sensitive areas
        if [[ "$file" =~ auth|credential|token|security ]]; then
            echo "  🔒 Security-sensitive: $file"
            agents_to_trigger+=("security-auditor")
        fi
        
        # ADR changes require BDD scenario updates
        if [[ "$file" =~ ^docs/adrs/ ]]; then
            echo "  📋 ADR change: $file"
            agents_to_trigger+=("bdd-specialist")
            agents_to_trigger+=("architecture-reviewer")
            architectural_changes=true
        fi
        
        # Configuration and hooks
        if [[ "$file" =~ \.claude/|\.yaml$|\.yml$ ]]; then
            echo "  ⚙️ Configuration change: $file"
            agents_to_trigger+=("automation-optimizer")
        fi
    done
    
    # Remove duplicates and return unique agents
    printf '%s\n' "${agents_to_trigger[@]}" | sort -u
    
    # Return analysis flags
    echo "---"
    echo "architectural_changes:$architectural_changes"
    echo "test_changes:$test_changes" 
    echo "core_changes:$core_changes"
}

# Function to trigger appropriate agents based on analysis
trigger_agents() {
    local analysis_output="$1"
    local agents=($(echo "$analysis_output" | grep -v "^---" | grep -v ":"))
    local flags=$(echo "$analysis_output" | grep "^---" -A 10)

    if [ ${#agents[@]} -eq 0 ]; then
        echo "✅ No specialized agents needed for these changes"
        exit 0
    fi

    echo ""
    echo "🤖 Recommended Agent Actions (maintain user control):"

    # Get current issue context for agent guidance
    local current_branch=$(git branch --show-current 2>/dev/null)
    local issue_number=""
    if [[ "$current_branch" =~ feature/([0-9]+) ]]; then
        issue_number="${BASH_REMATCH[1]}"
    fi

    # Provide actionable guidance for each required agent
    for agent in "${agents[@]}"; do
        case "$agent" in
            "architecture-reviewer")
                echo "  🏗️  llm-orc-architecture-reviewer"
                echo "     📋 Focus: ADR compliance, architectural patterns, integration points"
                echo "     📁 Files: $(printf '%s ' "${MODIFIED_FILES[@]}" | head -c 80)..."
                echo "     💡 Run: Use Task tool with llm-orc-architecture-reviewer"
                echo "     🎯 Goal: Validate core component changes maintain design consistency"
                ;;

            "bdd-specialist")
                echo "  🎭 llm-orc-bdd-specialist"
                echo "     📋 Focus: BDD scenario updates, behavioral contracts"
                echo "     📁 Files: $(printf '%s ' "${MODIFIED_FILES[@]}" | head -c 80)..."
                echo "     💡 Run: Use Task tool with llm-orc-bdd-specialist"
                echo "     🎯 Goal: Ensure scenarios match implementation changes"
                ;;

            "tdd-specialist")
                echo "  🧪 llm-orc-tdd-specialist"
                echo "     📋 Focus: TDD cycle discipline, test quality standards"
                echo "     📁 Files: $(printf '%s ' "${MODIFIED_FILES[@]}" | head -c 80)..."
                echo "     💡 Run: Use Task tool with llm-orc-tdd-specialist"
                echo "     🎯 Goal: Validate Red→Green→Refactor compliance"
                ;;

            "performance-optimizer")
                echo "  ⚡ llm-orc-performance-optimizer"
                echo "     📋 Focus: Ensemble execution, async performance implications"
                echo "     📁 Files: $(printf '%s ' "${MODIFIED_FILES[@]}" | head -c 80)..."
                echo "     💡 Run: Use Task tool with llm-orc-performance-optimizer"
                echo "     🎯 Goal: Identify optimization opportunities"
                ;;

            "security-auditor")
                echo "  🔒 llm-orc-security-auditor"
                echo "     📋 Focus: Credential handling, input validation, secure coding"
                echo "     📁 Files: $(printf '%s ' "${MODIFIED_FILES[@]}" | head -c 80)..."
                echo "     💡 Run: Use Task tool with llm-orc-security-auditor"
                echo "     🎯 Goal: Review security-sensitive changes"
                ;;

            "ux-specialist")
                echo "  💻 llm-orc-ux-specialist"
                echo "     📋 Focus: CLI interface, error messaging, developer ergonomics"
                echo "     📁 Files: $(printf '%s ' "${MODIFIED_FILES[@]}" | head -c 80)..."
                echo "     💡 Run: Use Task tool with llm-orc-ux-specialist"
                echo "     🎯 Goal: Improve user experience"
                ;;

            "automation-optimizer")
                echo "  🔧 automation-optimizer"
                echo "     📋 Focus: Development automation system effectiveness"
                echo "     📁 Files: $(printf '%s ' "${MODIFIED_FILES[@]}" | head -c 80)..."
                echo "     💡 Run: Use Task tool with automation-optimizer"
                echo "     🎯 Goal: Optimize automation workflows"
                ;;
        esac
        echo ""
    done

    echo "💡 To use agents: Call Task tool with subagent_type matching the agent name above"
    echo "🎯 Context: Issue #${issue_number:-"N/A"} - $(git branch --show-current)"
}

# Function to check if changes warrant continuous validation
check_continuous_validation_needed() {
    local files=("$@")
    local significant_changes=0
    
    for file in "${files[@]}"; do
        # Count significant changes
        if [[ "$file" =~ ^src/llm_orc/core/|^src/llm_orc/schemas/|^tests/bdd/ ]]; then
            significant_changes=$((significant_changes + 1))
        fi
    done
    
    # If we have multiple significant changes, trigger continuous validation
    if [ $significant_changes -ge 2 ]; then
        echo ""
        echo "🔄 Multiple significant changes detected - triggering implementation checkpoint"
        echo ""
        
        # Trigger implementation checkpoint hook
        if [ -x ".claude/hooks/implementation-checkpoint.sh" ]; then
            .claude/hooks/implementation-checkpoint.sh --auto
        else
            echo "💡 Consider running: .claude/hooks/implementation-checkpoint.sh"
        fi
    fi
}

# Main execution
main() {
    # Analyze the modifications
    local analysis=$(analyze_modifications "${MODIFIED_FILES[@]}")
    
    # Trigger appropriate agents
    trigger_agents "$analysis"
    
    # Check if continuous validation is needed
    check_continuous_validation_needed "${MODIFIED_FILES[@]}"
    
    echo ""
    echo "🔍 Intelligent post-edit analysis complete"
}

# Only run if we have files to analyze
if [ ${#MODIFIED_FILES[@]} -gt 0 ]; then
    main
fi

exit 0