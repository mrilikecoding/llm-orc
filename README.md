# LLM Orchestra

A multi-agent LLM communication system for ensemble orchestration and intelligent analysis.

## Overview

LLM Orchestra lets you coordinate multiple AI agents for complex analysis tasks. Run code reviews with security and performance specialists, analyze architecture decisions from multiple angles, or get systematic coverage of any multi-faceted problem.

Mix expensive cloud models with free local models - use Claude for strategic insights while Llama3 handles systematic analysis tasks.

## Key Features

- **Multi-Agent Ensembles**: Coordinate specialized agents for different aspects of analysis
- **Model Profiles**: Simplified configuration with named shortcuts for model + provider combinations
- **Cost Optimization**: Mix expensive and free models based on what each task needs
- **CLI Interface**: Simple commands with piping support (`cat code.py | llm-orc invoke code-review`)
- **Secure Authentication**: Encrypted API key storage with easy credential management
- **YAML Configuration**: Easy ensemble setup with readable config files
- **Usage Tracking**: Token counting, cost estimation, and timing metrics

## Installation

### Option 1: Homebrew (macOS - Recommended)
```bash
# Add the tap
brew tap mrilikecoding/llm-orchestra

# Install LLM Orchestra
brew install llm-orchestra

# Verify installation
llm-orc --version
```

### Option 2: pip (All Platforms)
```bash
# Install from PyPI
pip install llm-orchestra

# Verify installation
llm-orc --version
```

### Option 3: Development Installation
```bash
# Clone the repository
git clone https://github.com/mrilikecoding/llm-orc.git
cd llm-orc

# Install with development dependencies
uv sync --dev

# Verify installation
uv run llm-orc --version
```

### Updates
```bash
# Homebrew users
brew update && brew upgrade llm-orchestra

# pip users
pip install --upgrade llm-orchestra
```

## Quick Start

### 1. Set Up Authentication

Before using LLM Orchestra, configure authentication for your LLM providers:

```bash
# Interactive setup wizard (recommended for first-time users)
llm-orc auth setup

# Or add providers individually
llm-orc auth add anthropic --api-key YOUR_ANTHROPIC_KEY
llm-orc auth add google --api-key YOUR_GOOGLE_KEY

# OAuth for Claude Pro/Max users
llm-orc auth add anthropic-claude-pro-max

# List configured providers
llm-orc auth list

# Remove a provider if needed
llm-orc auth remove anthropic
```

**Security**: API keys are encrypted and stored securely in `~/.config/llm-orc/credentials.yaml`.

### 2. Configuration Options

LLM Orchestra supports both global and local configurations:

#### Global Configuration
Create `~/.config/llm-orc/ensembles/code-review.yaml`:

```yaml
name: code-review
description: Multi-perspective code review ensemble

agents:
  - name: security-reviewer
    model_profile: free-local     # Fast, free initial analysis
    system_prompt: "You are a security analyst. Focus on identifying security vulnerabilities, authentication issues, and potential attack vectors."
    timeout_seconds: 60

  - name: performance-reviewer
    model_profile: free-local     # Fast, free initial analysis
    system_prompt: "You are a performance analyst. Focus on identifying bottlenecks, inefficient algorithms, and scalability issues."
    timeout_seconds: 60

  - name: quality-reviewer
    model_profile: default-claude # High-quality cloud analysis
    system_prompt: "You are a code quality analyst. Focus on maintainability, readability, and best practices."
    timeout_seconds: 90

coordinator:
  model_profile: default-claude   # Best quality for synthesis
  synthesis_prompt: |
    You are a senior engineering lead. Synthesize the security, performance,
    and quality analysis into actionable recommendations.
  output_format: json
  timeout_seconds: 90
```

#### Local Project Configuration
For project-specific ensembles, initialize local configuration:

```bash
# Initialize local configuration in your project
llm-orc config init

# This creates .llm-orc/ directory with:
# - ensembles/   (project-specific ensembles)
# - models/      (shared model configurations)
# - scripts/     (project-specific scripts)
# - config.yaml  (project configuration)
```

#### View Current Configuration
```bash
# Show configuration paths and status
llm-orc config show
```

### 3. Using LLM Orchestra

#### Basic Usage
```bash
# List available ensembles
llm-orc list-ensembles

# List available model profiles
llm-orc list-profiles

# Get help for any command
llm-orc --help
llm-orc invoke --help
```

#### Invoke Ensembles
```bash
# Analyze code from a file (pipe input)
cat mycode.py | llm-orc invoke code-review

# Provide input directly
llm-orc invoke code-review --input "Review this function: def add(a, b): return a + b"

# JSON output for integration with other tools
llm-orc invoke code-review --input "..." --output-format json

# Use specific configuration directory
llm-orc invoke code-review --config-dir ./custom-config
```

#### Configuration Management
```bash
# Initialize local project configuration
llm-orc config init --project-name my-project

# Show current configuration status
llm-orc config show

```

## Use Cases

### Code Review
Get systematic analysis across security, performance, and maintainability dimensions. Each agent focuses on their specialty while synthesis provides actionable recommendations.

### Architecture Review  
Analyze system designs from scalability, security, performance, and reliability perspectives. Identify bottlenecks and suggest architectural patterns.

### Product Strategy
Evaluate business decisions from market, financial, competitive, and user experience angles. Get comprehensive analysis for complex strategic choices.

### Research Analysis
Systematic literature review, methodology evaluation, or multi-dimensional analysis of research questions.

## Model Support

- **Claude** (Anthropic) - Strategic analysis and synthesis
- **Gemini** (Google) - Multi-modal and reasoning tasks  
- **Ollama** - Local deployment of open-source models (Llama3, etc.)
- **Custom models** - Extensible interface for additional providers

## Configuration

### Model Profiles

Model profiles simplify ensemble configuration by providing named shortcuts for common model + provider combinations:

```yaml
# In ~/.config/llm-orc/config.yaml or .llm-orc/config.yaml
model_profiles:
  free-local:
    model: llama3
    provider: ollama
    cost_per_token: 0.0

  default-claude:
    model: claude-sonnet-4-20250514
    provider: anthropic-claude-pro-max
    # No cost_per_token: subscription-based

  high-context:
    model: claude-3-5-sonnet-20241022
    provider: anthropic-api
    cost_per_token: 3.0e-06

  small:
    model: claude-3-haiku-20240307
    provider: anthropic-api
    cost_per_token: 1.0e-06
```

**Profile Benefits:**
- **Simplified Configuration**: Use `model_profile: default-claude` instead of explicit model + provider
- **Consistency**: Same profile names work across all ensembles
- **Cost Tracking**: Built-in cost information for budgeting
- **Flexibility**: Local profiles override global ones

**Usage in Ensembles:**
```yaml
agents:
  - name: bulk-analyzer
    model_profile: free-local     # Fast, free bulk analysis
  - name: expert-reviewer
    model_profile: default-claude # High-quality cloud analysis
  - name: document-processor
    model_profile: high-context   # Large context processing
```

### Ensemble Configuration
Ensemble configurations support:

- **Model profiles** for simplified, consistent model selection
- **Agent specialization** with role-specific prompts
- **Timeout management** per agent and coordinator
- **Mixed model strategies** combining local and cloud models
- **Synthesis strategies** for combining agent outputs
- **Output formatting** (text, JSON) for integration

### Configuration Hierarchy
LLM Orchestra follows a configuration hierarchy:

1. **Local project configuration** (`.llm-orc/` in current directory)
2. **Global user configuration** (`~/.config/llm-orc/`)
3. **Command-line options** (highest priority)

### XDG Base Directory Support
Configurations follow the XDG Base Directory specification:
- Global config: `~/.config/llm-orc/` (or `$XDG_CONFIG_HOME/llm-orc/`)
- Automatic migration from old `~/.llm-orc/` location

## Cost Optimization

- **Local models** (free) for systematic analysis tasks
- **Cloud models** (paid) reserved for strategic insights
- **Usage tracking** shows exactly what each analysis costs
- **Intelligent routing** based on task complexity

## Development

```bash
# Run tests
uv run pytest

# Run linting and formatting
uv run ruff check .
uv run ruff format --check .

# Type checking
uv run mypy src/llm_orc
```

## Research

This project includes comparative analysis of multi-agent vs single-agent approaches. See [docs/ensemble_vs_single_agent_analysis.md](docs/ensemble_vs_single_agent_analysis.md) for detailed findings.

## Philosophy

**Reduce toil, don't replace creativity.** Use AI to handle systematic, repetitive analysis while preserving human creativity and strategic thinking.

## License

MIT License - see [LICENSE](LICENSE) for details.