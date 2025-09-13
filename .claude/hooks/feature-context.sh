#!/bin/bash

# Feature Context Hook (SessionStart)
# Provides feature development context when working on non-main branches

# Exit early if not in a git repository
if ! git rev-parse --is-inside-work-tree &>/dev/null; then
    exit 0
fi

# Get current branch
CURRENT_BRANCH=$(git branch --show-current 2>/dev/null)

# Exit if on main branch or no branch detected
if [ -z "$CURRENT_BRANCH" ] || [ "$CURRENT_BRANCH" = "main" ] || [ "$CURRENT_BRANCH" = "master" ]; then
    exit 0
fi

echo "🚀 Feature Development Context"
echo "Branch: $CURRENT_BRANCH"

# Try to extract issue number from branch name
ISSUE_NUMBER=""
if [[ "$CURRENT_BRANCH" =~ feature/([0-9]+) ]]; then
    ISSUE_NUMBER="${BASH_REMATCH[1]}"
elif [[ "$CURRENT_BRANCH" =~ ([0-9]+)- ]]; then
    ISSUE_NUMBER="${BASH_REMATCH[1]}"
elif [[ "$CURRENT_BRANCH" =~ -([0-9]+)$ ]]; then
    ISSUE_NUMBER="${BASH_REMATCH[1]}"
fi

# Get commit info since branching from main
COMMITS_SINCE_MAIN=$(git rev-list --count HEAD ^main 2>/dev/null || echo "0")
CHANGED_FILES=$(git diff --name-only main...HEAD 2>/dev/null | wc -l | tr -d ' ')

echo "📊 Branch Stats:"
echo "  • $COMMITS_SINCE_MAIN commits since main"
echo "  • $CHANGED_FILES files modified"

if [ -n "$ISSUE_NUMBER" ]; then
    echo "🎯 Linked Issue: #$ISSUE_NUMBER"
    
    # Check if gh CLI is available
    if command -v gh &>/dev/null; then
        echo ""
        echo "📋 Activating feature-progress-tracker agent to analyze progress..."
        echo ""
        
        # Export context for the agent
        export FEATURE_BRANCH="$CURRENT_BRANCH"
        export FEATURE_ISSUE="$ISSUE_NUMBER"
        export COMMITS_COUNT="$COMMITS_SINCE_MAIN"
        export FILES_CHANGED="$CHANGED_FILES"
        
        # Check if BDD scenarios exist for this issue
        BDD_SCENARIOS=""
        if [ -f "tests/bdd/features/issue-${ISSUE_NUMBER}.feature" ]; then
            BDD_SCENARIOS="present"
        else
            BDD_SCENARIOS="missing"
        fi
        
        # Output JSON context for Claude to use with feature agents
        cat << EOF
{
  "featureContext": {
    "branch": "$CURRENT_BRANCH",
    "issueNumber": "$ISSUE_NUMBER",
    "commitsSinceMain": $COMMITS_SINCE_MAIN,
    "filesChanged": $CHANGED_FILES,
    "bddScenarios": "$BDD_SCENARIOS",
    "shouldAnalyzeProgress": true,
    "message": "Use the feature-analyzer agent to analyze development progress for issue #$ISSUE_NUMBER on branch '$CURRENT_BRANCH'. If BDD scenarios are missing, use llm-orc-bdd-specialist to create behavioral contracts. Then use progress-updater agent to update CHANGELOG with findings."
  }
}
EOF
    else
        echo "💡 Install 'gh' CLI for GitHub integration and progress tracking"
    fi
else
    echo "💡 Branch name doesn't contain issue number - add issue number to branch name for progress tracking"
    echo "   Example: feature/24-script-agents or feature-24-new-feature"
fi

echo ""

exit 0