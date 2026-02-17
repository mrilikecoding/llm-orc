# LLM Orchestra

[![PyPI version](https://badge.fury.io/py/llm-orchestra.svg)](https://badge.fury.io/py/llm-orchestra)
[![CI](https://github.com/mrilikecoding/llm-orc/workflows/CI/badge.svg)](https://github.com/mrilikecoding/llm-orc/actions)
[![codecov](https://codecov.io/gh/mrilikecoding/llm-orc/graph/badge.svg?token=FWHP257H9E)](https://codecov.io/gh/mrilikecoding/llm-orc)
[![Python 3.11-3.13](https://img.shields.io/badge/python-3.11--3.13-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Downloads](https://static.pepy.tech/badge/llm-orchestra)](https://pepy.tech/project/llm-orchestra)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/mrilikecoding/llm-orc)](https://github.com/mrilikecoding/llm-orc/releases)

Orchestrate ensembles of specialized models — local and cloud — to do real analytical work. Coordination over scale.

## Overview

A decent laptop can run multiple small language models simultaneously. What's missing is the coordination layer — the system that decomposes problems, routes them to specialized agents, manages dependencies between them, and synthesizes results. LLM Orchestra provides that layer.

The approach is architectural intelligence over brute-force scaling: instead of sending everything to one large model, decompose the problem and let specialized agents own their piece. Independent agents run in parallel. Dependent agents wait for what they need. Script agents handle data processing and analysis alongside LLM agents, enabling hybrid workflows that go beyond pure language model orchestration.

Mix expensive cloud models with free local models. Use Claude for strategic synthesis while local models handle systematic analysis at zero marginal cost.

## Key Features

- **Multi-Agent Ensembles**: Coordinate specialized agents with flexible dependency graphs
- **Agent Dependencies**: Define which agents depend on others for sophisticated orchestration patterns
- **Script Agent Integration**: Execute custom scripts alongside LLM agents with JSON I/O communication
- **Model Profiles**: Simplified configuration with named shortcuts for model + provider combinations
- **Cost Optimization**: Mix expensive and free models based on what each task needs
- **Streaming Output**: Real-time progress updates during ensemble execution
- **CLI Interface**: Simple commands with piping support (`cat code.py | llm-orc invoke code-review`)
- **Secure Authentication**: Encrypted API key storage with easy credential management
- **YAML Configuration**: Easy ensemble setup with readable config files
- **Usage Tracking**: Token counting, cost estimation, and timing metrics
- **Artifact Management**: Automatic saving of execution results with timestamped persistence

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

### Output Formats

LLM Orchestra supports three output formats for different use cases:

#### Rich Interface (Default)
Interactive format with real-time progress updates and visual dependency graphs:

```bash
llm-orc invoke code-review --input "def add(a, b): return a + b"
```

#### JSON Output
Structured data format for integration and automation:

```bash
llm-orc invoke code-review --output-format json --input "code to review"
```

Returns complete execution data including events, results, metadata, and dependency information.

#### Text Output  
Clean, pipe-friendly format for command-line workflows:

```bash
llm-orc invoke code-review --output-format text --input "code to review"
```

Plain text results perfect for piping and scripting: `llm-orc invoke ... | grep "security"`

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

### Script Management

LLM Orchestra includes powerful script agent integration for executing custom scripts alongside LLM agents:

```bash
# List available scripts in your project
llm-orc scripts list

# Show detailed information about a script
llm-orc scripts show file_operations/read_file.py

# Test a script with parameters
llm-orc scripts test file_operations/read_file.py --parameters '{"filepath": "example.txt"}'

# Scripts are discovered from .llm-orc/scripts/ directories
# Results are automatically saved to .llm-orc/artifacts/ with timestamps
```

Script agents use JSON I/O for seamless integration with LLM agents, enabling powerful hybrid workflows where scripts provide data and context for LLM analysis.

### MCP Server

LLM Orchestra includes a Model Context Protocol (MCP) server that exposes ensembles, artifacts, and metrics as MCP resources. This enables integration with MCP clients like Claude Code, Claude Desktop, and other tools.

#### Quick Start

1. Add `.mcp.json` to your project root:
```json
{
  "mcpServers": {
    "llm-orc": {
      "command": "uv",
      "args": ["run", "llm-orc", "mcp", "serve"]
    }
  }
}
```

2. Restart Claude Code - MCP tools appear as `mcp__llm-orc__*`

3. Try it:
```
mcp__llm-orc__get_help              # Get full documentation
mcp__llm-orc__get_provider_status   # Check which models are available
mcp__llm-orc__list_ensembles        # See available ensembles
```

#### Resources (Read-Only Data)

| Resource | Description |
|----------|-------------|
| `llm-orc://ensembles` | List all available ensembles with metadata |
| `llm-orc://ensemble/{name}` | Get specific ensemble configuration |
| `llm-orc://profiles` | List model profiles |
| `llm-orc://artifacts/{ensemble}` | List execution artifacts for an ensemble |
| `llm-orc://artifact/{ensemble}/{id}` | Get individual artifact details |
| `llm-orc://metrics/{ensemble}` | Get aggregated metrics (success rate, cost, duration) |

#### Tools (25 Total)

**Core Execution**
| Tool | Description |
|------|-------------|
| `invoke` | Execute ensemble with streaming progress, saves artifacts automatically |
| `list_ensembles` | List all ensembles from local/library/global sources |
| `validate_ensemble` | Check config validity, profile availability, and dependencies |
| `update_ensemble` | Modify ensemble config (supports dry-run and backup) |
| `analyze_execution` | Analyze execution artifact data |

**Provider Discovery** - Check what's available before running
| Tool | Description |
|------|-------------|
| `get_provider_status` | Show available providers and Ollama models |
| `check_ensemble_runnable` | Check if ensemble can run, suggest local alternatives |

**Ensemble Management**
| Tool | Description |
|------|-------------|
| `create_ensemble` | Create new ensemble from scratch or template |
| `delete_ensemble` | Delete ensemble (requires confirmation) |

**Profile Management**
| Tool | Description |
|------|-------------|
| `list_profiles` | List profiles with optional provider filter |
| `create_profile` | Create new model profile |
| `update_profile` | Update existing profile |
| `delete_profile` | Delete profile (requires confirmation) |

**Script Management**
| Tool | Description |
|------|-------------|
| `list_scripts` | List primitive scripts by category |
| `get_script` | Get script source and metadata |
| `test_script` | Test script with sample input |
| `create_script` | Create new primitive script |
| `delete_script` | Delete script (requires confirmation) |

**Library Operations**
| Tool | Description |
|------|-------------|
| `library_browse` | Browse library ensembles and scripts |
| `library_copy` | Copy from library to local project |
| `library_search` | Search library by keyword |
| `library_info` | Get library metadata and statistics |

**Artifact Management**
| Tool | Description |
|------|-------------|
| `delete_artifact` | Delete individual execution artifact |
| `cleanup_artifacts` | Delete old artifacts (supports dry-run) |

**Help**
| Tool | Description |
|------|-------------|
| `get_help` | Get comprehensive docs: directory structure, schemas, workflows |

#### Example Workflow

```
# 1. Check what's available
mcp__llm-orc__get_provider_status
# → Shows Ollama running with llama3, mistral models

# 2. Find an ensemble
mcp__llm-orc__library_search query="code review"
# → Found: code-analysis/security-review

# 3. Check if it can run locally
mcp__llm-orc__check_ensemble_runnable ensemble_name="security-review"
# → Shows which profiles need local alternatives

# 4. Copy and adapt
mcp__llm-orc__library_copy source="code-analysis/security-review"
mcp__llm-orc__update_ensemble ensemble_name="security-review" changes={"agents": [...]}

# 5. Run it
mcp__llm-orc__invoke ensemble_name="security-review" input_data="Review this code..."
```

#### CLI Usage

```bash
# Start MCP server (stdio transport for MCP clients)
llm-orc mcp serve

# Start with HTTP transport for debugging
llm-orc mcp serve --transport http --port 8080
```

## Ensemble Library

Looking for pre-built ensembles? Check out the [LLM Orchestra Library](https://github.com/mrilikecoding/llm-orchestra-library) - a curated collection of analytical ensembles for code review, research analysis, decision support, and more.

### Library CLI Commands

LLM Orchestra includes built-in commands to browse and copy ensembles from the library:

```bash
# Browse all available categories
llm-orc library categories
llm-orc l categories  # Using alias

# Browse ensembles in a specific category
llm-orc library browse code-analysis

# Show detailed information about an ensemble
llm-orc library show code-analysis/security-review

# Copy an ensemble to your local configuration
llm-orc library copy code-analysis/security-review

# Copy an ensemble to your global configuration
llm-orc library copy code-analysis/security-review --global
```

#### Library Source Configuration

By default, LLM Orchestra fetches library content from the remote GitHub repository. For development purposes, you can use a local copy of the library:

```bash
# Use remote GitHub library (default)
llm-orc library browse research-analysis

# Use local library for development
export LLM_ORC_LIBRARY_SOURCE=local
llm-orc library browse research-analysis  # Uses local submodule
llm-orc init                              # Copies from local submodule

# Switch back to remote
unset LLM_ORC_LIBRARY_SOURCE
```

**When to use local library:**
- Testing changes to library ensembles before publishing
- Working on feature branches of the llm-orchestra-library
- Offline development (when remote access unavailable)
- Custom ensemble development and testing

**Requirements for local library:**
- The `llm-orchestra-library` submodule must be initialized and present
- Clear error messages guide you if the local library is not found

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

#### Fan-Out (Parallel Map-Reduce)

Agents with `fan_out: true` automatically expand into N parallel instances when their upstream dependency produces an array result. This enables map-reduce style parallel processing:

```yaml
agents:
  # "Map" step: split input into chunks
  - name: chunker
    script: scripts/chunker.py
    # Returns: {"success": true, "data": ["chunk1", "chunk2", "chunk3"]}

  # "Reduce" step: process each chunk in parallel
  - name: processor
    model_profile: default-local
    depends_on: [chunker]
    fan_out: true
    system_prompt: "Analyze this text chunk..."

  # Synthesis: combine all results
  - name: synthesizer
    model_profile: default-local
    depends_on: [processor]
    system_prompt: "Synthesize the analysis results..."
```

**How it works:**

1. `chunker` runs and returns a JSON array (direct array or `{"data": [...]}` format)
2. `processor` is expanded into `processor[0]`, `processor[1]`, `processor[2]` — one per array element
3. All instances execute in parallel, each receiving their chunk plus metadata (`chunk_index`, `total_chunks`, `base_input`)
4. Results are gathered back under the original `processor` name as an ordered array
5. `synthesizer` receives the combined results and can reference them normally

**Configuration requirements:**

- `fan_out: true` requires a `depends_on` field (validated at load time)
- The upstream agent must produce a non-empty array result
- Downstream agents reference the original name — fan-out is transparent to them

**Result format for gathered fan-out agents:**

```json
{
  "response": ["result_0", "result_1", null],
  "status": "partial",
  "fan_out": true,
  "instances": [
    {"index": 0, "status": "success"},
    {"index": 1, "status": "success"},
    {"index": 2, "status": "failed", "error": "timeout"}
  ]
}
```

Status is `"success"` (all instances passed), `"partial"` (some failed), or `"failed"` (all failed). Partial results are preserved — the ensemble continues with whatever succeeded.

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

### Library Path Configuration

Control where `llm-orc init` finds primitive scripts using environment variables or project-specific configuration:

```bash
# Option 1: Custom library location via environment variable
export LLM_ORC_LIBRARY_PATH="/path/to/your/custom-library"
llm-orc init

# Option 2: Project-specific configuration via .llm-orc/.env
mkdir -p .llm-orc
echo 'LLM_ORC_LIBRARY_PATH=/path/to/your/custom-library' > .llm-orc/.env
llm-orc init

# Option 3: Use local submodule (development default)
export LLM_ORC_LIBRARY_SOURCE=local
llm-orc init

# Option 4: Auto-detect library in current directory (no configuration needed)
# Looks for: ./llm-orchestra-library/scripts/primitives/
llm-orc init
```

**Priority order:**
1. `LLM_ORC_LIBRARY_PATH` environment variable - Explicit custom location (highest priority)
2. `.llm-orc/.env` file - Project-specific configuration
3. `LLM_ORC_LIBRARY_SOURCE=local` - Package submodule
4. `./llm-orchestra-library/` - Current working directory auto-detection
5. No scripts installed (graceful fallback)

**Note**: Environment variables always take precedence over `.env` file settings, allowing temporary overrides without modifying project files.

This allows developers to maintain their own script libraries while still using llm-orc's orchestration features.

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

The short version: orchestrated multi-agent systems maintain accuracy at scale where single-agent approaches collapse. Mixture-of-Agents ensembles of open-source models have matched or exceeded frontier model performance on established benchmarks. Cascade routing strategies have replicated frontier quality at a fraction of the cost. The evidence supports the architectural bet this project is built on.

## Philosophy

**Coordination over scale. Process over generation.**

The concentrated AI buildout optimizes for one thing: making the generative phase faster. But generation was never the bottleneck. Evaluation — the human judgment that determines whether output is correct, appropriate, and worth shipping — is the binding constraint. Faster generation without proportionally better evaluation just produces more to review.

LLM Orchestra takes the opposite position. An ensemble of smaller, specialized models — each focused on a bounded analytical task — produces structured output designed for human evaluation. The system decomposes problems so that each agent owns its contribution. A security reviewer finds vulnerabilities. A performance analyst identifies bottlenecks. A synthesis agent integrates their findings. The human evaluates a structured analysis, not raw generation.

Running models locally is a practical choice, not an ideological one. No per-query billing means you can run systematic analysis across an entire codebase without watching a meter. Data stays on your hardware. And the local/cloud mix lets you put cost where it matters — expensive models for strategic insight, free local models for systematic coverage.

## License

AGPL-3.0 License - see [LICENSE](LICENSE) for details.