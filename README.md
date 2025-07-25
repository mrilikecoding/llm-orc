# LLM Orchestra

[![PyPI version](https://badge.fury.io/py/llm-orchestra.svg)](https://badge.fury.io/py/llm-orchestra)
[![CI](https://github.com/mrilikecoding/llm-orc/workflows/CI/badge.svg)](https://github.com/mrilikecoding/llm-orc/actions)
[![codecov](https://codecov.io/gh/mrilikecoding/llm-orc/graph/badge.svg?token=FWHP257H9E)](https://codecov.io/gh/mrilikecoding/llm-orc)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://static.pepy.tech/badge/llm-orchestra)](https://pepy.tech/project/llm-orchestra)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/mrilikecoding/llm-orc)](https://github.com/mrilikecoding/llm-orc/releases)

A multi-agent LLM communication system for ensemble orchestration and intelligent analysis.

## Overview

LLM Orchestra lets you coordinate multiple AI agents for complex analysis tasks. Run code reviews with security and performance specialists, analyze architecture decisions from multiple angles, or get systematic coverage of any multi-faceted problem.

Mix expensive cloud models with free local models - use Claude for strategic insights while Llama3 handles systematic analysis tasks.

## Key Features

- **Multi-Agent Ensembles**: Coordinate specialized agents with flexible dependency graphs
- **Agent Dependencies**: Define which agents depend on others for sophisticated orchestration patterns
- **Model Profiles**: Simplified configuration with named shortcuts for model + provider combinations
- **Cost Optimization**: Mix expensive and free models based on what each task needs
- **Streaming Output**: Real-time progress updates during ensemble execution
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
    model_profile: free-local
    system_prompt: "You are a security analyst. Focus on identifying security vulnerabilities, authentication issues, and potential attack vectors."

  - name: performance-reviewer
    model_profile: free-local
    system_prompt: "You are a performance analyst. Focus on identifying bottlenecks, inefficient algorithms, and scalability issues."

  - name: quality-reviewer
    model_profile: free-local
    system_prompt: "You are a code quality analyst. Focus on maintainability, readability, and best practices."

  - name: senior-reviewer
    model_profile: default-claude
    depends_on: [security-reviewer, performance-reviewer, quality-reviewer]
    system_prompt: |
      You are a senior engineering lead. Synthesize the security, performance,
      and quality analysis into actionable recommendations.
    output_format: json
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
# Check configuration status with visual indicators
llm-orc config check
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

# Enable streaming for real-time progress (enabled by default)
llm-orc invoke code-review --streaming
```

#### Configuration Management
```bash
# Initialize local project configuration
llm-orc config init --project-name my-project

# Check configuration status with visual indicators
llm-orc config check                # Global + local status with legend
llm-orc config check-global        # Global configuration only  
llm-orc config check-local         # Local project configuration only

# Reset configurations with safety options
llm-orc config reset-global        # Reset global config (backup + preserve auth by default)
llm-orc config reset-local         # Reset local config (backup + preserve ensembles by default)

# Advanced reset options
llm-orc config reset-global --no-backup --reset-auth       # Complete reset including auth
llm-orc config reset-local --reset-ensembles --no-backup   # Reset including ensembles

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

Model profiles simplify ensemble configuration by providing named shortcuts for complete agent configurations including model, provider, system prompts, and timeouts:

```yaml
# In ~/.config/llm-orc/config.yaml or .llm-orc/config.yaml
model_profiles:
  free-local:
    model: llama3
    provider: ollama
    cost_per_token: 0.0
    system_prompt: "You are a helpful assistant that provides concise, accurate responses for local development and testing."
    timeout_seconds: 30

  default-claude:
    model: claude-sonnet-4-20250514
    provider: anthropic-claude-pro-max
    system_prompt: "You are an expert assistant that provides high-quality, detailed analysis and solutions."
    timeout_seconds: 60

  high-context:
    model: claude-3-5-sonnet-20241022
    provider: anthropic-api
    cost_per_token: 3.0e-06
    system_prompt: "You are an expert assistant capable of handling complex, multi-faceted problems with detailed analysis."
    timeout_seconds: 120

  small:
    model: claude-3-haiku-20240307
    provider: anthropic-api
    cost_per_token: 1.0e-06
    system_prompt: "You are a quick, efficient assistant that provides concise and accurate responses."
    timeout_seconds: 30
```

**Profile Benefits:**
- **Complete Agent Configuration**: Includes model, provider, system prompts, and timeout settings
- **Simplified Configuration**: Use `model_profile: default-claude` instead of explicit model + provider + system_prompt + timeout
- **Consistency**: Same profile names work across all ensembles with consistent behavior
- **Cost Tracking**: Built-in cost information for budgeting
- **Flexibility**: Local profiles override global ones, explicit agent configs override profile defaults

**Usage in Ensembles:**
```yaml
agents:
  - name: bulk-analyzer
    model_profile: free-local     # Complete config: model, provider, prompt, timeout
  - name: expert-reviewer
    model_profile: default-claude # High-quality config with appropriate timeout
  - name: document-processor
    model_profile: high-context   # Large context processing with extended timeout
    system_prompt: "Custom prompt override"  # Overrides profile default
```

**Override Behavior:**
Explicit agent configuration takes precedence over model profile defaults:
```yaml
agents:
  - name: custom-agent
    model_profile: free-local
    system_prompt: "Custom prompt"  # Overrides profile system_prompt
    timeout_seconds: 60            # Overrides profile timeout_seconds
```

### Ensemble Configuration
Ensemble configurations support:

- **Model profiles** for simplified, consistent model selection
- **Agent specialization** with role-specific prompts
- **Agent dependencies** using `depends_on` for sophisticated orchestration
- **Dependency validation** with automatic cycle detection and missing dependency checks
- **Timeout management** per agent with performance configuration
- **Mixed model strategies** combining local and cloud models
- **Output formatting** (text, JSON) for integration
- **Streaming execution** with real-time progress updates

#### Agent Dependencies

The new dependency-based architecture allows agents to depend on other agents, enabling sophisticated orchestration patterns:

```yaml
agents:
  # Independent agents execute in parallel
  - name: security-reviewer
    model_profile: free-local
    system_prompt: "Focus on security vulnerabilities..."

  - name: performance-reviewer  
    model_profile: free-local
    system_prompt: "Focus on performance issues..."

  # Dependent agent waits for dependencies to complete
  - name: senior-reviewer
    model_profile: default-claude
    depends_on: [security-reviewer, performance-reviewer]
    system_prompt: "Synthesize the security and performance analysis..."
```

**Benefits:**
- **Flexible orchestration**: Create complex dependency graphs beyond simple coordinator patterns
- **Parallel execution**: Independent agents run concurrently for better performance  
- **Automatic validation**: Circular dependencies and missing dependencies are detected at load time
- **Better maintainability**: Clear, explicit dependencies instead of implicit coordinator relationships

### Configuration Status Checking

LLM Orchestra provides visual status checking to quickly see which configurations are ready to use:

```bash
# Check all configurations with visual indicators
llm-orc config check
```

**Visual Indicators:**
- 🟢 **Ready to use** - Profile/provider is properly configured and available
- 🟥 **Needs setup** - Profile references unavailable provider or missing authentication

**Provider Availability Detection:**
- **Authenticated providers** - Checks for valid API credentials
- **Ollama service** - Tests connection to local Ollama instance (localhost:11434)
- **Configuration validation** - Verifies model profiles reference available providers

**Example Output:**
```
Configuration Status Legend:
🟢 Ready to use    🟥 Needs setup

=== Global Configuration Status ===
📁 Model Profiles:
🟢 local-free (llama3 via ollama)
🟢 quality (claude-sonnet-4 via anthropic-claude-pro-max)  
🟥 high-context (claude-3-5-sonnet via anthropic-api)

🌐 Available Providers: anthropic-claude-pro-max, ollama

=== Local Configuration Status: My Project ===
📁 Model Profiles:
🟢 security-auditor (llama3 via ollama)
🟢 senior-reviewer (claude-sonnet-4 via anthropic-claude-pro-max)
```

### Configuration Reset Commands

LLM Orchestra provides safe configuration reset with backup and selective retention options:

```bash
# Reset global configuration (safe defaults)
llm-orc config reset-global        # Creates backup, preserves authentication

# Reset local configuration (safe defaults)  
llm-orc config reset-local         # Creates backup, preserves ensembles

# Advanced reset options
llm-orc config reset-global --no-backup --reset-auth           # Complete global reset
llm-orc config reset-local --reset-ensembles --no-backup       # Complete local reset
llm-orc config reset-local --project-name "My Project"         # Set project name
```

**Safety Features:**
- **Automatic backups** - Creates timestamped `.backup` directories by default
- **Authentication preservation** - Keeps API keys and credentials safe by default
- **Ensemble retention** - Preserves local ensembles by default
- **Confirmation prompts** - Prevents accidental data loss

**Available Options:**

*Global Reset:*
- `--backup/--no-backup` - Create backup before reset (default: backup)
- `--preserve-auth/--reset-auth` - Keep authentication (default: preserve)

*Local Reset:*
- `--backup/--no-backup` - Create backup before reset (default: backup)
- `--preserve-ensembles/--reset-ensembles` - Keep ensembles (default: preserve)
- `--project-name` - Set project name (defaults to directory name)

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