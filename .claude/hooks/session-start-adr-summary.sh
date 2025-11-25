#!/bin/bash
# Claude Code hook: Generate ADR progress summary at session start

# Exit if not in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    exit 0
fi

# Check if this is llm-orc project
if [[ ! -f ".claude/hooks/session_start_adr_summary.py" ]]; then
    exit 0
fi

# Generate ADR summary
echo "🔄 Generating ADR progress summary..."
echo

if python .claude/hooks/session_start_adr_summary.py; then
    echo
    echo "💡 Use 'python .claude/hooks/validate_adr_consistency.py' to check status consistency"
    echo "💡 Use 'python .claude/hooks/check_bdd_coverage.py' to check BDD coverage"
else
    echo "⚠️  Failed to generate ADR summary"
fi

echo