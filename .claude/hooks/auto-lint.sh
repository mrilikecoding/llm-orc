#!/bin/bash

# Auto-lint hook for llm-orc development
# Automatically formats and fixes common linting issues after Write/Edit operations

# Exit early if not in a Python project or if uv is not available
if ! command -v uv &> /dev/null; then
    exit 0
fi

if [ ! -f "pyproject.toml" ]; then
    exit 0
fi

# Get the file being modified from Claude's hook context
# For Write/Edit operations, Claude provides the file path in environment variables
FILE_PATH="${CLAUDE_HOOK_FILE_PATH:-}"

# If no specific file provided, format all Python files in src and tests
if [ -z "$FILE_PATH" ]; then
    echo "🔧 Auto-formatting Python code..."
    
    # Quick format and fix - these are fast operations
    uv run ruff check --fix src tests 2>/dev/null || true
    uv run ruff format src tests 2>/dev/null || true
    
    echo "✅ Code formatting complete"
else
    # Format specific file if it's a Python file
    if [[ "$FILE_PATH" == *.py ]]; then
        echo "🔧 Auto-formatting $FILE_PATH..."
        
        # Check if file exists and format it
        if [ -f "$FILE_PATH" ]; then
            uv run ruff check --fix "$FILE_PATH" 2>/dev/null || true
            uv run ruff format "$FILE_PATH" 2>/dev/null || true
            echo "✅ Formatted $FILE_PATH"
        fi
    fi
fi

# Exit successfully to avoid blocking the operation
exit 0