#!/bin/bash

# Line Length Helper Hook  
# Detects long lines and suggests refactoring strategies

# Exit early if not in Python project
if ! command -v uv &> /dev/null || [ ! -f "pyproject.toml" ]; then
    exit 0
fi

# Get changed files
CHANGED_FILES=()
if [ -n "$CLAUDE_HOOK_FILE_PATH" ] && [[ "$CLAUDE_HOOK_FILE_PATH" == *.py ]]; then
    CHANGED_FILES=("$CLAUDE_HOOK_FILE_PATH")
else
    # Check git for recently modified Python files
    mapfile -t CHANGED_FILES < <(git diff --name-only HEAD~1 2>/dev/null | grep '\.py$' || echo "")
fi

if [ ${#CHANGED_FILES[@]} -eq 0 ]; then
    exit 0
fi

# Check for lines longer than 88 characters (ruff limit)
LONG_LINES=$(uv run ruff check "${CHANGED_FILES[@]}" --select E501 --output-format=text 2>/dev/null || echo "")

if [ -z "$LONG_LINES" ]; then
    exit 0
fi

echo "📏 Long lines detected (>88 chars):"
echo "$LONG_LINES"
echo ""
echo "💡 Refactoring suggestions:"
echo "  • Break long function calls across multiple lines"
echo "  • Extract long expressions into variables"
echo "  • Use parentheses for implicit line continuation"
echo "  • Consider shorter variable names for deeply nested code"
echo ""
echo "Options:"
echo "1. Open file for manual refactoring (o)?"
echo "2. Show refactoring examples (s)?" 
echo "3. Skip for now (any other key)?"
read -r -n 1 response
echo ""

case $response in
    [Oo])
        # Get the first file with long lines
        FIRST_FILE=$(echo "$LONG_LINES" | head -1 | cut -d: -f1)
        if command -v code &> /dev/null; then
            echo "🔧 Opening $FIRST_FILE in VS Code..."
            code "$FIRST_FILE"
        else
            echo "📝 File to edit: $FIRST_FILE"
        fi
        ;;
    [Ss])
        echo ""
        echo "🔄 Common refactoring patterns:"
        echo ""
        echo "Before: some_function(very_long_argument_name, another_long_argument, third_argument)"
        echo "After:  some_function("
        echo "            very_long_argument_name,"
        echo "            another_long_argument," 
        echo "            third_argument"
        echo "        )"
        echo ""
        echo "Before: result = some_object.very_long_method_name().another_method().final_method()"
        echo "After:  intermediate = some_object.very_long_method_name().another_method()"
        echo "        result = intermediate.final_method()"
        echo ""
        ;;
    *)
        echo "⏭️  Skipping line length refactoring"
        ;;
esac

exit 0