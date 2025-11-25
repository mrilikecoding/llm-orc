# LLM-Orc Development Agents

This directory contains specialized Claude Code sub-agents designed to enhance llm-orc development through focused expertise and proactive assistance.

## Agent Philosophy

These agents follow **2025 best practices for sub-agent design**:
- **Single Responsibility**: Each agent has one clear expertise domain
- **Proactive Language**: Descriptions use "PROACTIVELY" and "MUST BE USED" for better auto-selection
- **Tool Restrictions**: Limited tool access for security and performance
- **Cost Optimization**: Appropriate model selection (Sonnet vs Haiku) based on complexity

## Development Agents

### Core Development Support

**llm-orc-tdd-specialist** (`red`)  
Enforces strict TDD methodology (Red→Green→Refactor cycle). Ensures all code follows proper test-driven development practices.

**llm-orc-architecture-reviewer** (`green`)  
Reviews architectural decisions and code changes for alignment with llm-orc's multi-agent orchestration principles.

**llm-orc-performance-optimizer** (`orange`)  
Identifies and addresses performance bottlenecks in async execution, model coordination, and resource management.

**llm-orc-security-auditor** (`yellow`)  
Audits security aspects including credential management, API security, and secure coding practices.

**llm-orc-ux-specialist** (`cyan`)  
Improves user experience across CLI interface, configuration management, and developer ergonomics.

### Strategic & Meta Support

**llm-orc-project-manager** (`blue`)  
Manages development priorities, assesses GitHub issues, and provides strategic roadmap guidance.

**llm-orc-dogfooding-advisor** (`purple`)  
Identifies opportunities to use llm-orc ensembles to improve llm-orc development itself.

### Meta-Automation Agents

**automation-optimizer** (`magenta`)  
Optimizes the entire Claude Code automation system (agents + hooks + workflows) for maximum development effectiveness.

**branch-context-reviewer** (`blue`)  
Provides context on current development work by reviewing README, architecture docs, ADRs, and branch changes.

**documentation-maintainer** (`green`)  
Maintains project documentation when significant features are completed or architectural changes are finalized.

## Agent Orchestration Strategy

The agents follow an **orchestrator-worker pattern** with clear specializations:

```
Strategic Layer     │ project-manager, dogfooding-advisor
Code Quality        │ tdd-specialist, architecture-reviewer  
Technical Excellence │ performance-optimizer, security-auditor
User Experience     │ ux-specialist
Meta-Optimization   │ automation-optimizer
Context & Docs      │ branch-context-reviewer, documentation-maintainer
```

## Usage Patterns

### Automatic Selection
Claude Code automatically selects appropriate agents based on:
- Code being modified (triggers relevant specialists)
- Questions asked (routes to domain experts)
- Tasks being performed (activates workflow-specific agents)

### Proactive Activation
Agents are designed to activate proactively when:
- Writing tests → TDD specialist engages
- Modifying core components → Architecture reviewer activates  
- Performance-critical changes → Performance optimizer intervenes
- Security-sensitive code → Security auditor reviews

## Configuration

### Agent Loading
Agents are automatically loaded via `.claude/hooks/load-agents.sh` on session start.

### Model Selection
- **Sonnet**: Complex analysis agents (architecture, project management)
- **Haiku**: Focused task agents (performance, security, UX)

### Tool Permissions
Each agent has restricted tool access based on their needs:
- Development agents: Read, Write, Edit, Bash, Grep, Glob
- Strategic agents: Read, WebFetch, WebSearch, Task
- Meta agents: Read, Bash, Grep, Glob

## Maintenance

### Adding New Agents
1. Create markdown file with YAML frontmatter
2. Include "PROACTIVELY" and "MUST BE USED" in description
3. Specify appropriate tools and model
4. Test with `.claude/hooks/load-agents.sh`

### Agent Evolution
- Use **automation-ecosystem-auditor** to review agent effectiveness
- Monitor usage patterns and adjust descriptions for better selection
- Consolidate overlapping agents or split overly broad ones
- Update tool permissions based on actual usage needs

The system uses Claude Code's sub-agent capabilities to enhance llm-orc development, preparing the groundwork for eventual "dogfooding" where llm-orc could use its own ensemble orchestration to improve itself.