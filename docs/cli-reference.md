# CLI & Configuration Reference

The complete command and configuration reference for LLM Orchestra. For a
quick orientation, start with the [README](../README.md); `llm-orc --help`
is always current for flags.

## Using LLM Orchestra

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

> **Library tools require a local copy of the library.** These tools read from the local filesystem — they do not fetch from GitHub. The library is auto-detected if the `llm-orchestra-library` submodule is present in the current working directory. For Homebrew or pip installs, set `LLM_ORC_LIBRARY_PATH=/path/to/llm-orchestra-library` to point to a local clone of [llm-orchestra-library](https://github.com/mrilikecoding/llm-orchestra-library). Run `library_info` to verify the library is found.

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
# → Returns results including path "ensembles/code-analysis/security-review.yaml"

# 3. Check if it can run locally
mcp__llm-orc__check_ensemble_runnable ensemble_name="security-review"
# → Shows which profiles need local alternatives

# 4. Copy and adapt (source path is relative to library root)
mcp__llm-orc__library_copy source="ensembles/code-analysis/security-review"
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

By default, LLM Orchestra uses local filesystem detection: it checks for a `llm-orchestra-library/` directory in the current working directory, then falls back to a no-op if none is found. Remote GitHub is not used unless explicitly configured. To use a specific library source:

```bash
# Use a local library at a custom path
export LLM_ORC_LIBRARY_PATH=/path/to/llm-orchestra-library
llm-orc library browse research-analysis

# Use local package submodule explicitly
export LLM_ORC_LIBRARY_SOURCE=local
llm-orc library browse research-analysis  # Uses local submodule
llm-orc init                              # Copies from local submodule

# Use remote GitHub library
export LLM_ORC_LIBRARY_SOURCE=remote
llm-orc library browse research-analysis
```

**When to use local library:**
- Testing changes to library ensembles before publishing
- Working on feature branches of the llm-orchestra-library
- Offline development (when remote access unavailable)
- Custom ensemble development and testing

**Requirements for local library:**
- The `llm-orchestra-library` submodule must be initialized and present
- Clear error messages guide you if the local library is not found

### Contributing to the Library

The library is a separate repository at [github.com/mrilikecoding/llm-orchestra-library](https://github.com/mrilikecoding/llm-orchestra-library). To add or improve content:

1. Create and test your ensemble locally using `llm-orc` or the MCP tools
2. Copy the finished YAML into your local library clone under the appropriate category
3. Open a pull request against the library repository

The MCP `library_copy` tool copies **from** the library **to** your project. There is no reverse direction by design — contributing back goes through a PR rather than automated writes to a shared upstream.


## Configuration

### Model Profiles

Model profiles simplify ensemble configuration by providing named shortcuts for complete agent configurations including model, provider, system prompts, timeouts, and generation parameters:

```yaml
# In ~/.config/llm-orc/config.yaml or .llm-orc/config.yaml
model_profiles:
  free-local:
    model: llama3
    provider: ollama
    cost_per_token: 0.0
    system_prompt: "You are a helpful assistant that provides concise, accurate responses for local development and testing."
    timeout_seconds: 30
    temperature: 0.7
    max_tokens: 500
    options:              # Provider-specific parameters (Ollama)
      num_ctx: 4096
      top_k: 40

  default-claude:
    model: claude-sonnet-4-20250514
    provider: anthropic-api
    system_prompt: "You are an expert assistant that provides high-quality, detailed analysis and solutions."
    timeout_seconds: 60
    temperature: 0.5
    max_tokens: 2000

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

  openai-local:
    model: llama3:latest
    provider: openai-compatible       # Any OpenAI-compatible server
    base_url: http://localhost:11434/v1  # Ollama, vLLM, LM Studio, etc.
    cost_per_token: 0.0
    timeout_seconds: 30
```

**Profile Benefits:**
- **Complete Agent Configuration**: Includes model, provider, system prompts, timeout settings, and generation parameters
- **Simplified Configuration**: Use `model_profile: default-claude` instead of explicit model + provider + system_prompt + timeout
- **Consistency**: Same profile names work across all ensembles with consistent behavior
- **Cost Tracking**: Built-in cost information for budgeting
- **Generation Control**: Set `temperature`, `max_tokens`, and provider-specific `options` per profile
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
    temperature: 0.1               # Overrides profile temperature
    max_tokens: 200                # Overrides profile max_tokens
    options:                       # Merged with profile options (agent wins)
      num_ctx: 8192
```

### Ensemble Configuration
Ensemble configurations support:

- **Model profiles** for simplified, consistent model selection
- **Agent specialization** with role-specific prompts
- **Generation parameters** (`temperature`, `max_tokens`, `options`) per profile or per agent
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

#### Ensemble Agents (Composable Ensembles)

Agents can reference and execute other ensembles, enabling hierarchical composition:

```yaml
# child ensemble: topic-analysis.yaml
name: topic-analysis
agents:
  - name: analyst
    model_profile: ollama-gemma-small
    system_prompt: "Analyze the given topic in 2-3 sentences."

# parent ensemble
agents:
  - name: classifier
    script: scripts/classifier.py

  - name: topic-analyst
    ensemble: topic-analysis          # references child ensemble
    depends_on: [classifier]

  - name: synthesizer
    model_profile: default-claude
    depends_on: [topic-analyst]
```

**How it works:**

1. The `ensemble` field identifies which ensemble to execute (resolved by name from `.llm-orc/ensembles/`)
2. The child ensemble runs as a self-contained execution with its own phases and agents
3. Child executors share immutable infrastructure (config, credentials, model factory) but isolate mutable state
4. Nesting depth is limited (default: 5) to prevent unbounded recursion
5. Cross-ensemble cycles are detected at load time

#### Input Key Routing

Agents can select a specific key from upstream JSON output using `input_key`, enabling routing patterns where a classifier produces keyed output and downstream agents each consume their slice:

```yaml
agents:
  # Classifier produces: {"pdfs": ["a.pdf", "b.pdf"], "audio": ["c.mp3"]}
  - name: classifier
    script: scripts/classifier.py

  # Selects only the "pdfs" array from classifier output
  - name: pdf-processor
    ensemble: pdf-pipeline
    depends_on: [classifier]
    input_key: pdfs
    fan_out: true

  # Selects only the "audio" array
  - name: audio-processor
    ensemble: audio-pipeline
    depends_on: [classifier]
    input_key: audio
    fan_out: true

  - name: synthesizer
    model_profile: default-claude
    depends_on: [pdf-processor, audio-processor]
```

**Behavior:**

- `input_key` selects `output[key]` from the first entry in `depends_on`
- If the key is missing or the upstream output is not JSON/dict, the agent receives a runtime error
- Without `input_key`, the agent receives the full upstream output (backward compatible)
- Composes naturally with `fan_out`: `input_key` selects the array, `fan_out` expands per item
- Works with all agent types: LLM, script, and ensemble

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
🟢 quality (claude-sonnet-4 via anthropic-api)  
🟥 high-context (claude-3-5-sonnet via anthropic-api)

🌐 Available Providers: anthropic-api, ollama

=== Local Configuration Status: My Project ===
📁 Model Profiles:
🟢 security-auditor (llama3 via ollama)
🟢 senior-reviewer (claude-sonnet-4 via anthropic-api)
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

