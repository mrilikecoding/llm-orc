# llm-orc Research

This directory contains research studies, templates, and tooling documentation for conducting rigorous research with llm-orc ensembles.

## Directory Structure

```
docs/research/
├── README.md                 # This file
├── studies/                  # Individual research studies
│   └── STUDY-001-swarm-vs-foundation/
├── templates/                # Research templates
│   └── TEMPLATE-research-study.md
└── tooling/                  # Tooling analysis
    └── research-tooling-gaps.md
```

## Research Philosophy

llm-orc provides a platform for exploring multi-agent LLM orchestration. Our research aims to:

1. **Validate hypotheses** about ensemble architectures with empirical evidence
2. **Compare approaches** rigorously using statistical methods
3. **Identify patterns** that generalize across domains
4. **Publish findings** that advance the field

## Getting Started

### 1. Use the Template

Copy `templates/TEMPLATE-research-study.md` to start a new study:

```bash
cp docs/research/templates/TEMPLATE-research-study.md \
   docs/research/studies/STUDY-XXX-your-study/README.md
```

### 2. Check Tooling Gaps

Review `tooling/research-tooling-gaps.md` to understand:
- What scripts exist for your methodology
- What gaps you may need to work around
- What MCP capabilities are available

### 3. Design Your Methodology

Use the template sections to define:
- Clear, testable hypotheses
- Appropriate statistical methods
- Data collection procedures
- Validation approaches

### 4. Leverage Existing Ensembles

Research-relevant ensembles in the library:
- `literature-synthesizer` - For background research
- `human-in-loop-validation` - For expert validation gates
- `adr-swarm-review` - Example swarm architecture

## Current Studies

| Study ID | Title | Status |
|----------|-------|--------|
| STUDY-001 | Micro-Model Swarms vs Foundation Models | Draft |

## Research Scripts

Located in `llm-orchestra-library/scripts/specialized/research/`:

| Script | Purpose |
|--------|---------|
| `t_test.py` | Statistical t-test analysis |

See `tooling/research-tooling-gaps.md` for needed scripts.

## Contributing Research

1. Create a new directory under `studies/`
2. Use the template structure
3. Document your methodology clearly
4. Store artifacts and raw data
5. Ensure reproducibility

## Publication Pipeline

```
┌─────────────────┐
│  Research       │
│  Question       │
└────────┬────────┘
         ▼
┌─────────────────┐
│  Methodology    │  ← Use TEMPLATE-research-study.md
│  Design         │
└────────┬────────┘
         ▼
┌─────────────────┐
│  Data           │  ← Use research ensembles + scripts
│  Collection     │
└────────┬────────┘
         ▼
┌─────────────────┐
│  Analysis       │  ← Use statistical scripts
│                 │
└────────┬────────┘
         ▼
┌─────────────────┐
│  Validation     │  ← Human-in-loop, peer review
│                 │
└────────┬────────┘
         ▼
┌─────────────────┐
│  Publication    │  ← LaTeX export, formatting
│                 │
└─────────────────┘
```

## MCP Integration

Use Claude via MCP to:
- Invoke ensembles for data collection
- Analyze artifacts
- Identify implementation gaps
- Draft study sections

Example workflow:
```
1. mcp__llm-orc__set_project("/path/to/project")
2. mcp__llm-orc__invoke("experiment-ensemble", "input data")
3. mcp__llm-orc__analyze_execution("artifact-id")
4. [Manual analysis and writing]
```

## Contact

For questions about research methodology or tooling, see the project maintainers.
