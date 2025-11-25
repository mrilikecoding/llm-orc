#!/bin/bash

# Development Workflow Guide (Manual)
# Shows the complete BDD-driven development workflow for llm-orc

echo "🚀 LLM-Orc BDD-Driven Development Workflow"
echo ""

# Get current context
CURRENT_BRANCH=$(git branch --show-current 2>/dev/null)
ISSUE_NUMBER=""
if [[ "$CURRENT_BRANCH" =~ feature/([0-9]+) ]]; then
    ISSUE_NUMBER="${BASH_REMATCH[1]}"
fi

if [ -n "$ISSUE_NUMBER" ]; then
    echo "🎯 Current Context: Issue #${ISSUE_NUMBER} on ${CURRENT_BRANCH}"
else
    echo "🎯 Current Branch: ${CURRENT_BRANCH}"
fi
echo ""

echo "📋 Complete Development Workflow:"
echo ""

echo "1. 🚀 Pre-Implementation (Run before coding)"
echo "   .claude/hooks/pre-implementation-gate.sh"
echo "   • Ensures BDD scenarios exist for issue"
echo "   • Activates bdd-specialist if scenarios missing"
echo "   • Provides behavioral contracts for implementation"
echo ""

echo "2. 🔴 TDD Red Phase (Write failing tests)"
echo "   • Use BDD scenarios as behavioral specification"
echo "   • TDD specialist automatically activates during test writing"
echo "   • Write tests that validate scenario requirements"
echo ""

echo "3. 🟢 TDD Green Phase (Minimal implementation)"
echo "   • Implement just enough to pass tests"
echo "   • Architecture reviewer activates for core component changes"
echo "   • Other specialists activate based on file patterns"
echo ""

echo "4. 🔄 Continuous Validation (During development)"
echo "   .claude/hooks/implementation-checkpoint.sh"
echo "   • Validates implementation against BDD scenarios"
echo "   • Checks architectural compliance"
echo "   • Analyzes development patterns"
echo ""

echo "5. ♻️ TDD Refactor Phase (Improve structure)"
echo "   • Separate commits for structural changes"
echo "   • Architecture reviewer ensures pattern compliance"
echo "   • Implementation checkpoint validates no behavior changes"
echo ""

echo "6. ✅ Final Validation (Before commit/PR)"
echo "   .claude/hooks/bdd-development-gate.sh --validate"
echo "   • Runs all BDD scenarios for issue"
echo "   • Validates behavioral compliance"
echo "   • Ensures architectural adherence"
echo ""

echo "🤖 Available Specialized Agents:"
echo ""
echo "Strategic & Planning:"
echo "  • llm-orc-project-manager: Issue prioritization, roadmap guidance"
echo "  • llm-orc-dogfooding-advisor: Self-improvement opportunities"
echo ""
echo "Behavioral & Quality:"
echo "  • llm-orc-bdd-specialist: BDD scenarios and behavioral contracts"
echo "  • llm-orc-tdd-specialist: TDD discipline and test quality"
echo "  • llm-orc-architecture-reviewer: ADR compliance and patterns"
echo ""
echo "Technical Excellence:"
echo "  • llm-orc-performance-optimizer: Async performance and optimization"
echo "  • llm-orc-security-auditor: Security best practices"
echo "  • llm-orc-ux-specialist: CLI and developer experience"
echo ""
echo "Meta & Automation:"
echo "  • automation-optimizer: Hook and workflow optimization"
echo "  • documentation-maintainer: Keep docs current"
echo "  • branch-context-reviewer: Development context analysis"
echo ""

echo "🎛️ Hook Integration Points:"
echo ""
echo "Automatic Triggers:"
echo "  • SessionStart: Issue context and BDD scenario detection"
echo "  • PostToolUse: Intelligent agent activation by file patterns"
echo "  • File Changes: Continuous validation checkpoints"
echo ""
echo "Manual Workflow:"
echo "  • Pre-implementation gate: .claude/hooks/pre-implementation-gate.sh"
echo "  • BDD scenario management: .claude/hooks/bdd-development-gate.sh"
echo "  • Implementation checkpoints: .claude/hooks/implementation-checkpoint.sh"
echo ""

if [ -n "$ISSUE_NUMBER" ]; then
    echo "🎯 Next Steps for Issue #${ISSUE_NUMBER}:"
    echo ""
    
    # Check current status
    if [ -f "tests/bdd/features/issue-${ISSUE_NUMBER}.feature" ]; then
        echo "✅ BDD scenarios exist"
        echo "   Run: uv run pytest tests/bdd/features/issue-${ISSUE_NUMBER}.feature -v"
    else
        echo "❌ BDD scenarios missing"
        echo "   Run: .claude/hooks/pre-implementation-gate.sh"
    fi
    
    # Check if in middle of development
    local commits_count=$(git rev-list --count HEAD ^main 2>/dev/null || echo "0")
    if [ "$commits_count" -gt 0 ]; then
        echo "🔄 Development in progress (${commits_count} commits)"
        echo "   Run: .claude/hooks/implementation-checkpoint.sh"
    else
        echo "🚀 Ready to start development"
        echo "   Run: .claude/hooks/pre-implementation-gate.sh"
    fi
    
else
    echo "💡 To start development on an issue:"
    echo "   1. Create feature branch: git checkout -b feature/24-script-agents"
    echo "   2. Run pre-implementation gate: .claude/hooks/pre-implementation-gate.sh"
    echo "   3. Follow TDD cycle with BDD behavioral guidance"
fi

echo ""
echo "📚 For detailed information: .claude/hooks/README.md"
echo ""

exit 0