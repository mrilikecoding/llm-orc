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
- **intelligent-post-edit.sh**: Context-aware agent activation based on file patterns

### Development Workflow Hooks (Manual)
- **pre-implementation-gate.sh**: Ensures BDD scenarios exist before coding begins
- **bdd-development-gate.sh**: Generates/validates BDD scenarios for issues
- **implementation-checkpoint.sh**: Continuous validation during development

### Quality Maintenance (Manual)
- **unused-variable-cleaner.sh**: Run before commits
- **line-length-helper.sh**: Run before commits  
- **manual-quality-checks.sh**: Combined pre-commit cleanup

### SessionStart Hooks
- **feature-context.sh**: Provides issue context and BDD scenario status
- **load-agents.sh**: Agent initialization

### Git Integration
- Existing git pre-commit hook runs `make lint-fix`
- **test-coverage-gate.sh** integrates with git workflow

## Enhanced BDD-Driven Development Flow

### LLM Development Workflow
The hook system now supports **BDD-driven LLM development** with automatic agent activation:

1. **Pre-Implementation**: `pre-implementation-gate.sh` ensures BDD scenarios exist
2. **During Implementation**: `intelligent-post-edit.sh` activates agents based on file patterns
3. **Continuous Validation**: `implementation-checkpoint.sh` prevents architectural drift
4. **BDD Maintenance**: `bdd-development-gate.sh` keeps scenarios current

### Intelligent Agent Activation
**File Pattern-Based Triggers**:
- Core changes → `architecture-reviewer`, `performance-optimizer`
- Schema changes → `architecture-reviewer`, `bdd-specialist` 
- Test changes → `tdd-specialist`, `bdd-specialist`
- CLI changes → `ux-specialist`
- Security files → `security-auditor`
- ADR changes → `bdd-specialist`, `architecture-reviewer`

### BDD Integration Points
- **SessionStart**: Detects missing BDD scenarios for current issue
- **PreImplementation**: Generates behavioral contracts before coding
- **PostToolUse**: Updates scenarios when implementation changes
- **Continuous**: Validates implementation against behavioral contracts

## Hook Design Principles

### Fast & Non-Blocking
All hooks exit successfully to avoid blocking operations. Interactive prompts provide escape routes.

### Focused Responsibility
Each hook handles one specific category of issues rather than trying to solve everything.

### Error Resilience
Hooks gracefully handle missing tools, invalid files, and edge cases without failing.

### User Choice
Interactive hooks always provide options to skip, view details, or proceed differently.

### Context-Aware Intelligence
Hooks analyze file patterns and development context to activate the most relevant agents automatically.

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