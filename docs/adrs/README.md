# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records documenting key architectural decisions for llm-orc.

## Active ADRs

| ADR | Title | Status | Completion |
|-----|-------|--------|------------|
| [001](001-pydantic-script-agent-interfaces.md) | Pydantic-Based Script Agent Interface System | Implemented | 100% |
| [002](002-composable-primitive-agent-system.md) | Composable Primitive Agent System | Implemented | 100% |
| [003](003-testable-script-agent-contracts.md) | Testable Script Agent Contracts | Implemented | 100% |
| [004](004-bdd-llm-development-guardrails.md) | BDD as LLM Development Guardrails | Implemented | 100% |
| [005](005-multi-turn-agent-conversations.md) | Multi-Turn Agent Conversations | Implemented | 70% |
| [006](006-library-based-primitives-architecture.md) | Library-Based Primitives Architecture | Implemented | 100% |

## ADR Overview

### Issue #24: Script-Based Agent Support

ADRs 001-006 collectively implement [Issue #24](https://github.com/anthropics/llm-orc/issues/24), adding script-based agent support to llm-orc.

**Core Architectural Principles:**
- **Pure Primitive-Driven**: Everything flows through `llm-orc invoke` with behavior emerging from primitive composition
- **Implicit Agent Typing**: Agents identified by fields present (`script` = script agent, `model_profile` = LLM agent)
- **Library-Based Content**: Primitives live in `llm-orchestra-library`, not in core application
- **Type-Safe Communication**: All agents use Pydantic schemas for JSON I/O

### Key Features Enabled

1. **Script Agents**: Deterministic processing via executable scripts
2. **Primitive Composition**: Complex workflows from simple building blocks
3. **Multi-Turn Conversations**: Agents interact across multiple turns with state accumulation
4. **Contract Validation**: Type-safe script interfaces with automated testing
5. **Human-in-the-Loop**: User interaction primitives for interactive workflows
6. **Research Infrastructure**: Statistical analysis, network science, experimental design

## Using ADRs

### For Developers

See [TEMPLATE-adr.md](TEMPLATE-adr.md) for creating new ADRs.

See [README-adr-lifecycle.md](README-adr-lifecycle.md) for ADR lifecycle management.

### BDD Integration

Each ADR includes BDD scenarios that serve as executable behavioral contracts. See:
- `tests/bdd/features/adr-001-*.feature`
- `tests/bdd/features/adr-002-*.feature`
- etc.

Run ADR tests:
```bash
# All ADR tests
uv run pytest tests/bdd/ -k adr

# Specific ADR
uv run pytest tests/bdd/ -k adr-001
```

## Architecture Documentation

See [docs/architecture.md](../architecture.md) for system architecture diagrams and detailed design documentation.
