# LLM-Orc Development Hooks

This directory contains Claude Code hooks that automate quality checks and optimize development workflows for llm-orc.

## Hook Philosophy

The hook system follows a **progressive quality gates** approach:
- **Automated Fixes**: Handle routine formatting and simple issues automatically
- **Interactive Guidance**: Provide helpful prompts for issues requiring decisions
- **Preventive Gates**: Block problematic patterns before they become technical debt
- **Performance Focus**: Fast execution to avoid interrupting development flow

## Hook Categories

### Automated Quality (PostToolUse)

**auto-lint.sh**  
Automatically runs `ruff check --fix` and `ruff format` after Write/Edit operations. Handles basic formatting without interrupting workflow.

**complexity-gate.sh**  
Checks code complexity after changes and provides interactive refactoring guidance for functions exceeding complexity limits.

### Interactive Quality (Manual)

**unused-variable-cleaner.sh**  
Detects unused variables and offers interactive cleanup options. Provides safe auto-removal or detailed fix suggestions.

**line-length-helper.sh**  
Identifies long lines (>88 chars) and provides refactoring suggestions with examples of common patterns.

Users can run interactive quality helpers directly:
- `.claude/hooks/unused-variable-cleaner.sh`
- `.claude/hooks/line-length-helper.sh`

### Coverage & Testing

**test-coverage-gate.sh**  
Ensures 95% test coverage before commits. Provides detailed coverage reports and blocks commits with insufficient testing.

### Agent Loading

**load-agents.sh**  
Loads all specialized development agents on session start, making them available for proactive use throughout development.

## Hook Timing Strategy

### PostToolUse Hooks (After Write/Edit/MultiEdit)
- **auto-lint.sh**: Immediate formatting fixes
- **complexity-gate.sh**: Immediate complexity feedback

### Manual Execution
- **unused-variable-cleaner.sh**: Run before commits
- **line-length-helper.sh**: Run before commits  
- **manual-quality-checks.sh**: Combined pre-commit cleanup

### SessionStart Hooks
- **load-agents.sh**: Agent initialization

### Git Integration
- Existing git pre-commit hook runs `make lint-fix`
- **test-coverage-gate.sh** integrates with git workflow

## Hook Design Principles

### Fast & Non-Blocking
All hooks exit successfully to avoid blocking operations. Interactive prompts provide escape routes.

### Focused Responsibility
Each hook handles one specific category of issues rather than trying to solve everything.

### Error Resilience
Hooks gracefully handle missing tools, invalid files, and edge cases without failing.

### User Choice
Interactive hooks always provide options to skip, view details, or proceed differently.

## Integration with Existing Workflow

### With Make Commands
- `make lint`: Comprehensive linting including complexity, security, dead code
- `make lint-fix`: Auto-fixes issues that can be safely corrected
- `make pre-commit`: Full quality gate before commits

### With Git Workflow
- Pre-commit hook ensures `make lint-fix` passes
- Coverage gate integrates with commit process
- Manual hooks complement git pre-commit checks

### With Claude Code Features
- Hooks work with Write, Edit, MultiEdit tools
- Agent loading provides specialized development assistance
- Interactive prompts respect Claude Code's CLI environment

## Maintenance

### Adding New Hooks
1. Create executable shell script in `.claude/hooks/`
2. Add appropriate error handling and exit codes
3. Update `.claude/settings.json` with hook configuration
4. Test with various file types and edge cases

### Hook Optimization
- Monitor execution times to avoid workflow delays
- Use **development-flow-optimizer** agent for performance analysis
- Balance thoroughness with development velocity
- Gather developer feedback on hook usefulness

### Troubleshooting
- All hooks include descriptive output for debugging
- Interactive hooks show available options clearly
- Error messages guide users toward solutions
- Hooks can be bypassed when necessary (`git commit --no-verify`)

The hook system creates a seamless development experience where quality is maintained automatically while preserving developer autonomy and workflow velocity.