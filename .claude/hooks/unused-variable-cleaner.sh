#!/bin/bash

# Unused Variable Cleaner Hook
# Detects unused variables and offers interactive cleanup

# Exit early if not in Python project
if ! command -v uv &> /dev/null || [ ! -f "pyproject.toml" ]; then
    exit 0
fi

# Get changed files (focus on recently modified Python files)
CHANGED_FILES=()
if [ -n "$CLAUDE_HOOK_FILE_PATH" ] && [[ "$CLAUDE_HOOK_FILE_PATH" == *.py ]]; then
    CHANGED_FILES=("$CLAUDE_HOOK_FILE_PATH")
else
    # Check git for recently modified Python files
    mapfile -t CHANGED_FILES < <(git diff --name-only HEAD~1 2>/dev/null | grep '\.py$' || echo "")
fi

# If no Python files to check, exit
if [ ${#CHANGED_FILES[@]} -eq 0 ]; then
    exit 0
fi

# Run ruff to find unused variables
UNUSED_VARS=$(uv run ruff check "${CHANGED_FILES[@]}" --select F841 --output-format=text 2>/dev/null || echo "")

if [ -z "$UNUSED_VARS" ]; then
    exit 0
fi

echo "🧹 Unused variables detected:"
echo "$UNUSED_VARS"
echo ""
echo "Options:"
echo "1. Auto-remove unused variables (y/n)?"
echo "2. Show suggested fixes (s)?"
echo "3. Skip for now (any other key)?"
read -r -n 1 response
echo ""

case $response in
    [Yy])
        echo "🔧 Removing unused variables..."
        # Use ruff --fix with only F841 (unused variables)
        uv run ruff check "${CHANGED_FILES[@]}" --select F841 --fix --quiet 2>/dev/null || true
        echo "✅ Unused variables cleaned"
        ;;
    [Ss])
        echo "💡 Suggested fixes:"
        uv run ruff check "${CHANGED_FILES[@]}" --select F841 --output-format=text --show-fixes 2>/dev/null || echo "Run 'make lint-fix' for detailed suggestions"
        ;;
    *)
        echo "⏭️  Skipping unused variable cleanup"
        ;;
esac

exit 0