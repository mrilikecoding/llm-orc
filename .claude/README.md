# Claude Code Automation Workflow

This directory contains the automation system for llm-orc development, designed to enforce architectural discipline and prevent scope drift.

## Workflow Architecture

### Core Principles
- **Issue-Driven Development**: All work aligns with specific GitHub issues
- **Branch Context Enforcement**: Branch names define scope boundaries
- **TDD Discipline**: Red→Green→Refactor cycle with agent handoff
- **BDD Guardrails**: Behavioral contracts guide implementation (ADR-004)
- **Testing Pyramid Health**: 70% unit, 20% integration, 10% BDD

### Hook System Architecture

#### Pre-Implementation Gates
- **`bdd-development-gate.sh`**: Ensures BDD scenarios exist before implementation
- **`testing-pyramid-gate.sh`**: Shows pyramid health and identifies imbalances
- **`pre-implementation-gate.sh`**: Validates architectural alignment

#### Post-Change Intelligence
- **`intelligent-post-edit.sh`**: Context-aware agent recommendations
- **`bdd-test-migrator.sh`**: Migrates passing tests to appropriate locations
- **`auto-lint.sh`**: Automated code quality fixes

#### Session Management
- **`load-agents.sh`**: Initializes available agents for session
- **`agent-advisor.sh`**: Provides targeted agent suggestions

### Agent Handoff Protocol

#### Scope Validation (Before Any Work)
1. **llm-orc-architecture-reviewer**: Validate alignment with issue/branch scope
2. **branch-context-reviewer**: Understand current development state
3. **llm-orc-project-manager**: Confirm priority and strategic alignment

#### BDD-Driven Development Flow
1. **llm-orc-bdd-specialist**: Create/validate BDD scenarios for feature
2. **llm-orc-tdd-specialist**: Guide Red→Green→Refactor cycles
3. **llm-orc-architecture-reviewer**: Review design compliance

#### Specialized Coordination
- **llm-orc-performance-optimizer**: For execution/async changes
- **llm-orc-security-auditor**: For auth/credential handling
- **llm-orc-ux-specialist**: For CLI/interface changes
- **llm-orc-precommit-specialist**: For quality assurance

## Current Branch Context

**Branch**: `feature/24-script-agents`
**Issue**: #24 - Script Agents Support
**ADR Focus**: ADR-001 (Pydantic Script Agent Interfaces)

### In-Scope Work
- **Script Discovery**: ScriptResolver finding scripts in `.llm-orc/scripts/`
- **JSON I/O Contracts**: Pydantic schema validation for script communication
- **Script Execution**: Process execution and result handling
- **Ensemble Integration**: Script agents working within ensemble flow
- **Dynamic Parameters**: Script agents generating parameters for other agents

### Out-of-Scope Work
- ❌ Provider authentication (different architectural concern)
- ❌ Model factory credential handling (separate system)
- ❌ OAuth flows or API key management
- ❌ Cloud provider integration
- ❌ CLI interface changes (separate issue/ADR)

## Testing Strategy

### Current Testing Pyramid Status
- **Unit Tests**: 93 (62%) - Target: 70%
- **Integration Tests**: 8 (5%) - Target: 20%
- **BDD Tests**: 50 (33%) - Target: 10%

**Issue**: Inverted pyramid - too many BDD scenarios, not enough integration tests

### Script Agent Testing Priorities
1. **Script Discovery Integration**: ScriptResolver ↔ file system
2. **JSON Contract Integration**: Schema validation in execution pipeline
3. **Agent Coordination Integration**: ScriptAgent ↔ EnsembleExecutor
4. **Dynamic Parameter Integration**: Script output → Agent input flows

## ADR Compliance Guidelines

### ADR-001: Pydantic Script Agent Interfaces
- All script I/O uses Pydantic schemas (`ScriptAgentInput`, `ScriptAgentOutput`)
- JSON contracts enforced at execution boundaries
- Type safety maintained through agent chains

### ADR-002: Composable Primitive System
- Scripts compose into larger workflows
- Type-safe primitive chaining
- Reusable script library management

### ADR-003: Testable Contract System
- Exception chaining with `from e` syntax
- Proper error context preservation
- Contract validation at component boundaries

### ADR-004: BDD Development Guardrails
- BDD scenarios guide implementation
- Prevent scope drift through behavioral contracts
- TDD discipline with Red→Green→Refactor cycles

### ADR-005: Multi-Turn Agent Conversations
- Agent memory and context management
- Conversation state persistence
- Multi-turn workflow coordination

## Hook Response Protocol

### When Hooks Provide Recommendations
1. **Read the output carefully** - hooks provide targeted guidance
2. **Follow agent suggestions** - use recommended agents proactively
3. **Validate scope alignment** - ensure work stays within branch context
4. **Document decisions** - explain any deviations from recommendations

### Warning Signs of Scope Drift
- Working on files outside issue scope
- Implementing features not in current ADR
- Ignoring hook recommendations without justification
- Testing components unrelated to branch context
- Mixing architectural concerns (e.g., auth + script execution)

## Commit Discipline

### Requirements for Any Commit
- All tests pass (`make test`)
- No linting violations (`make lint`)
- Single logical unit of work
- Aligns with current issue/ADR scope

### Commit Message Standards
- No AI attribution
- Natural, human-style messages
- Focus on "why" rather than "what"
- Reference issue numbers when relevant

## Quality Gates

### Before Implementation
- [ ] BDD scenarios exist for the feature
- [ ] Architecture review confirms scope alignment
- [ ] Testing pyramid shows healthy balance

### Before Commit
- [ ] All tests pass
- [ ] Linting clean
- [ ] Hook recommendations addressed
- [ ] Work aligns with branch context

### Before PR
- [ ] ADR compliance validated
- [ ] Integration tests cover cross-component flows
- [ ] Documentation updated for significant changes

## Emergency Workflow Recovery

### If You Go Off Track
1. **Stop immediately** when scope drift is identified
2. **Use llm-orc-architecture-reviewer** for immediate scope validation
3. **Consult branch context** - what does the branch name/issue require?
4. **Review ADR alignment** - does this work advance the current ADR?
5. **Pivot back to scope** - abandon out-of-scope work if necessary

### Preventing Future Drift
1. **Start every session** with architecture review
2. **Check branch context** before any new work
3. **Follow hook recommendations** proactively
4. **Use BDD scenarios** as implementation guardrails
5. **Validate scope frequently** during development

---

**Remember**: The automation system is designed to keep you on track. Trust the hooks, use the agents, and stay within the architectural boundaries defined by the current issue and ADR focus.