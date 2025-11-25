# ADR Lifecycle Automation Scripts

This directory contains automation scripts for managing Architecture Decision Record (ADR) lifecycle in the llm-orc project.

## Scripts

### `validate_adr_consistency.py`
Validates that ADR status is consistent with BDD scenario state and implementation progress.

```bash
python .claude/hooks/validate_adr_consistency.py
```

**Features:**
- Checks if ADRs marked as IN_PROGRESS have BDD scenarios
- Validates that IMPLEMENTED/VALIDATED/ACCEPTED ADRs have passing BDD scenarios
- Reports status inconsistencies and suggests corrections

### `update_adr_status.py`
Updates ADR status following allowed transition rules and updates implementation checklists.

```bash
python .claude/hooks/update_adr_status.py <adr_number> <new_status>

# Examples:
python .claude/hooks/update_adr_status.py 001 in_progress
python .claude/hooks/update_adr_status.py 001 implemented
```

**Features:**
- Validates status transitions follow ADR lifecycle rules
- Updates implementation status checklist for new status
- Adds timestamped progress log entries
- Prevents invalid status transitions

### `check_bdd_coverage.py`
Ensures all ADRs have corresponding BDD scenarios and reports coverage statistics.

```bash
python .claude/hooks/check_bdd_coverage.py

# Generate BDD templates for missing scenarios:
python .claude/hooks/check_bdd_coverage.py --generate
```

**Features:**
- Reports ADR-BDD coverage percentage
- Identifies ADRs missing BDD scenarios
- Finds orphaned BDD files without corresponding ADRs
- Generates BDD feature file templates

### `session_start_adr_summary.py`
Generates ADR progress summary for session start automation and development planning.

```bash
python .claude/hooks/session_start_adr_summary.py

# JSON output for programmatic use:
python .claude/hooks/session_start_adr_summary.py --json
```

**Features:**
- Shows overall ADR progress statistics
- Identifies priority actions based on status and BDD health
- Displays next tasks for each ADR
- Integrates with Claude Code session start hooks

## Integration with Claude Code

These scripts are integrated with Claude Code hooks for automatic execution:

- **Session Start**: `session-start-adr-summary.sh` runs ADR summary automatically
- **Pre-commit**: ADR consistency validation (when implemented)
- **CI/CD**: BDD coverage and status validation

## ADR Status Lifecycle

```
PROPOSED → IN_PROGRESS → IMPLEMENTED → VALIDATED → ACCEPTED
    ↑           ↓               ↓           ↓
    ←───────────←───────────────←───────────←
```

**Status Requirements:**
- `IN_PROGRESS`: BDD scenarios must exist
- `IMPLEMENTED`: All BDD scenarios must pass
- `VALIDATED`: Post-refactor scenarios still pass
- `ACCEPTED`: Peer review and documentation complete

## Usage in Development Workflow

1. **Session Start**: Check ADR summary for priorities
2. **Before Implementation**: Validate ADR status consistency
3. **During Development**: Use update script to transition status
4. **Before Commit**: Check BDD coverage and consistency
5. **Regular Maintenance**: Generate missing BDD scenarios

## Example Workflow

```bash
# Check current state
python .claude/hooks/session_start_adr_summary.py

# Work on ADR-001, create BDD scenarios
python .claude/hooks/check_bdd_coverage.py --generate

# Update status when scenarios created
python .claude/hooks/update_adr_status.py 001 in_progress

# Validate implementation readiness
python .claude/hooks/validate_adr_consistency.py

# Update to implemented when all scenarios pass
python .claude/hooks/update_adr_status.py 001 implemented
```

This automation ensures ADR lifecycle discipline and maintains consistency between architectural decisions and their implementation validation through BDD scenarios.