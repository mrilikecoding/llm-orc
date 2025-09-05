#!/bin/bash

# Complexity Prevention Hook
# Prevents overly complex functions from being committed

# Exit early if not in Python project
if ! command -v uv &> /dev/null || [ ! -f "pyproject.toml" ]; then
    exit 0
fi

# Get changed files
CHANGED_FILES=()
if [ -n "$CLAUDE_HOOK_FILE_PATH" ] && [[ "$CLAUDE_HOOK_FILE_PATH" == *.py ]]; then
    CHANGED_FILES=("$CLAUDE_HOOK_FILE_PATH")
else
    # Check git for recently modified Python files in src/
    mapfile -t CHANGED_FILES < <(git diff --name-only HEAD~1 2>/dev/null | grep '^src/.*\.py$' || echo "")
fi

if [ ${#CHANGED_FILES[@]} -eq 0 ]; then
    exit 0
fi

# Run complexity analysis (using project's limit of 15)
echo "🧠 Checking code complexity..."
COMPLEXITY_RESULT=$(uv run complexipy --max-complexity-allowed 15 "${CHANGED_FILES[@]}" 2>&1 || echo "")

# Check if there are complexity violations
if echo "$COMPLEXITY_RESULT" | grep -q "Complexity: [0-9]\+"; then
    HIGH_COMPLEXITY=$(echo "$COMPLEXITY_RESULT" | grep "Complexity: 1[6-9]\|Complexity: [2-9][0-9]" || echo "")
    
    if [ -n "$HIGH_COMPLEXITY" ]; then
        echo "🚨 High complexity functions detected:"
        echo "$HIGH_COMPLEXITY"
        echo ""
        echo "💡 Refactoring suggestions:"
        echo "  • Extract methods to reduce function size"
        echo "  • Simplify conditional logic with early returns"
        echo "  • Break complex loops into separate functions" 
        echo "  • Use guard clauses to reduce nesting"
        echo ""
        echo "Options:"
        echo "1. Continue anyway (not recommended) (c)?"
        echo "2. Show refactoring tips (r)?"
        echo "3. Abort to refactor first (any other key)?"
        read -r -n 1 response
        echo ""
        
        case $response in
            [Cc])
                echo "⚠️  Proceeding with high complexity (consider refactoring later)"
                ;;
            [Rr])
                echo ""
                echo "🔄 Complexity reduction techniques:"
                echo ""
                echo "1. Extract Method:"
                echo "   def complex_function():"
                echo "     # ... lots of code ..."
                echo "   →"
                echo "   def complex_function():"
                echo "     result1 = helper_method_1()"
                echo "     result2 = helper_method_2()" 
                echo ""
                echo "2. Early Returns:"
                echo "   if condition:"
                echo "     if other_condition:"
                echo "       # nested code"
                echo "   →"
                echo "   if not condition:"
                echo "     return"
                echo "   if not other_condition:"
                echo "     return"
                echo "   # main logic"
                echo ""
                exit 1  # Block the operation to encourage refactoring
                ;;
            *)
                echo "✋ Blocking operation - please refactor complex functions first"
                echo "Run 'make lint' to see detailed complexity report"
                exit 1
                ;;
        esac
    else
        echo "✅ Complexity within acceptable limits"
    fi
else
    echo "✅ No complexity issues detected"
fi

exit 0