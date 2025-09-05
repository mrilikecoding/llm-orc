#!/bin/bash

# Manual Quality Check Helper
# Provides interactive prompts for issues that can't be auto-fixed

echo "🧹 Running manual quality checks..."

# Check for unused variables
if [ -f ".claude/hooks/unused-variable-cleaner.sh" ]; then
    .claude/hooks/unused-variable-cleaner.sh
fi

# Check for long lines  
if [ -f ".claude/hooks/line-length-helper.sh" ]; then
    .claude/hooks/line-length-helper.sh
fi

echo "✅ Manual quality checks complete"