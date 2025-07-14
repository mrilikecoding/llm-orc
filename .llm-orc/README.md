# LLM Orchestra Configuration Directory

This directory contains your LLM Orchestra project configuration, including ensembles, authentication credentials, and project settings.

## Directory Structure

```
.llm-orc/
├── README.md                    # This file - comprehensive documentation
├── config.yaml                 # Project configuration and model profiles
├── ensembles/                  # Multi-agent ensemble definitions
│   ├── startup-advisory-board.yaml      # OAuth multi-agent business analysis
│   ├── product-strategy.yaml            # Strategic product decision making
│   ├── interdisciplinary-research.yaml  # Mixed-model research analysis
│   ├── mycology-meets-technology.yaml   # Biomimicry innovation research
│   ├── sleep-and-civilization.yaml      # Cultural evolution analysis
│   └── code-review.yaml                 # Code review with security/performance analysis
└── credentials.enc             # Encrypted authentication credentials (auto-managed)
```

## Configuration Files

### config.yaml

The main project configuration file that defines model profiles and project settings. Model profiles simplify ensemble configuration by providing named shortcuts for common model + provider combinations.

#### Structure

```yaml
project:
  name: "Your Project Name"
  default_models:
    fast: llama3 # Local/fast model for quick tasks
    production: claude-4-sonnet # High-quality cloud model

model_profiles:
  # Local development models (fast, free)
  development:
    model: llama3
    provider: ollama
    cost_per_token: 0.0  # Optional: for budgeting reference

  # High-quality cloud models (subscription-based)
  production:
    model: claude-3-5-sonnet-20241022
    provider: anthropic-claude-pro-max
    # No cost_per_token: subscription-based pricing

  # API-based models (pay-per-use)
  claude-api:
    model: claude-3-5-sonnet-20241022
    provider: anthropic-api
    cost_per_token: 3.0e-06  # $3 per million tokens

  # Alternative models
  gemini-2.5-flash:
    model: gemini-2.5-flash
    provider: google-gemini
    cost_per_token: 1.0e-06

  llama3:
    model: llama3
    provider: ollama
    cost_per_token: 0.0
```

#### Configuration Options

**Project Settings:**

- `name`: Display name for your project
- `default_models`: Quick references for common model types (used by fallback logic)

**Model Profiles:**

Model profiles are named shortcuts that combine a model + provider + cost information. They simplify ensemble configuration and provide consistency across your project.

**Profile Benefits:**

- **Simplified Configuration**: Use `model_profile: production` instead of `model: claude-3-5-sonnet-20241022` + `provider: anthropic-claude-pro-max`
- **Consistency**: Same profile name works across all ensembles
- **Flexibility**: Override global profiles with local project-specific ones
- **Cost Tracking**: Built-in cost information for budgeting

**Profile Fields:**

- `model`: Model identifier (required) - the actual model name from the provider
- `provider`: Authentication provider key (required) - determines how to authenticate
- `cost_per_token`: USD cost per token (optional, for budgeting reference only)

**Profile Types:**

- **Local Development**: `development`, `llama3` - free local models via Ollama
- **Cloud Subscription**: `production`, `claude-4-sonnet` - OAuth models using existing subscriptions ($0 cost)
- **API Pay-per-use**: `claude-api`, `gemini-2.5-flash` - API models with token costs

**Note:** `cost_per_token` is purely for documentation/budgeting purposes. Actual cost calculations use hardcoded values in the model implementations. For OAuth models (subscription-based), omit `cost_per_token` entirely.

## Ensemble Examples

This directory contains diverse ensemble examples showcasing different model combinations and use cases:

### OAuth Examples (Claude Pro/Max Subscription)

#### Startup Advisory Board

**File:** `startup-advisory-board.yaml`  
**Models:** All Claude Pro/Max via OAuth  
**Use Case:** Business strategy analysis with three expert agents (VC, Tech Architect, Growth Strategist)

**Features Demonstrated:**

- ✅ OAuth authentication with automatic token refresh
- ✅ Role injection for specialized expertise
- ✅ Configurable coordinator with its own role
- ✅ Complex multi-agent coordination

#### Product Strategy Analysis

**File:** `product-strategy.yaml`  
**Models:** All Claude Pro/Max via OAuth  
**Use Case:** Product decision making with market, financial, competitive, and UX analysis

### Mixed Model Examples (Local + Cloud)

#### Interdisciplinary Research

**File:** `interdisciplinary-research.yaml`  
**Models:** 3x Ollama (llama3) + 1x Claude Pro/Max + Claude Pro/Max coordinator  
**Use Case:** Broad research analysis through anthropological, systems, philosophical, and futurist perspectives

**Model Distribution:**

- **Local (Ollama):** Anthropologist, Systems Theorist, Philosopher-Ethicist
- **Cloud (OAuth):** Futurist Analyst, Coordinator

#### Mycology Meets Technology

**File:** `mycology-meets-technology.yaml`  
**Models:** 2x Ollama + 1x Claude Pro/Max + Claude Pro/Max coordinator  
**Use Case:** Biomimicry research exploring fungal networks for technology innovation

#### Sleep and Civilization

**File:** `sleep-and-civilization.yaml`  
**Models:** 2x Ollama + 1x Claude Pro/Max + Ollama coordinator  
**Use Case:** Historical and sociological analysis of sleep's role in human development

#### Code Review

**File:** `code-review.yaml`  
**Models:** 2x Ollama + 1x Claude Pro/Max + Claude Pro/Max coordinator  
**Use Case:** Comprehensive code review with security, performance, and quality analysis

**CLI Override Examples:**
```bash
# Use default comprehensive review (security + performance + quality)
llm-orc invoke code-review < demo_code_review.py

# Override to focus only on security issues  
llm-orc invoke code-review "Focus only on security vulnerabilities and ignore other issues" < demo_code_review.py

# Override to focus only on performance problems
llm-orc invoke code-review "Analyze only performance bottlenecks and algorithmic efficiency" < demo_code_review.py

# Override for specific context
llm-orc invoke code-review "This is legacy code for a financial system. Focus on security compliance for banking regulations" < demo_code_review.py
```

**Default Task (when no CLI input):**
- Comprehensive production readiness assessment
- Security, performance, maintainability analysis  
- Actionable feedback with specific recommendations

**CLI Override (when input provided):**
- Uses your specific instructions instead
- Agents focus on your particular concerns
- Same expert perspectives, different scope

## Authentication Setup

### OAuth Authentication (Claude Pro/Max)

```bash
# Set up OAuth for Claude Pro/Max subscription
llm-orc auth add anthropic-claude-pro-max

# Verify authentication
llm-orc auth test anthropic-claude-pro-max

# Check all configured providers
llm-orc auth list
```

### API Key Authentication

```bash
# Set up Anthropic API key
llm-orc auth add anthropic-api

# Set up Claude CLI
llm-orc auth add claude-cli
```

### Local Models (Ollama)

```bash
# Install Ollama (if not already installed)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull required models
ollama pull llama3
ollama pull llama2
```

## Using Ensembles

### Running Examples

```bash
# OAuth business analysis
llm-orc invoke startup-advisory-board "Should we launch a B2B SaaS platform for restaurant inventory management?"

# Mixed-model research
llm-orc invoke interdisciplinary-research "How might virtual reality reshape human social relationships?"

# Biomimicry innovation (uses default_task)
llm-orc invoke mycology-meets-technology

# Override default_task with specific question
llm-orc invoke mycology-meets-technology "How can fungal networks inspire database sharding strategies?"

# Code review with comprehensive default scope
llm-orc invoke code-review < my-code.py

# Code review focused on specific concern
llm-orc invoke code-review "Check only for SQL injection vulnerabilities" < user-auth.py
```

### Creating Custom Ensembles

Create a new `.yaml` file in the `ensembles/` directory:

#### Task Input Priority

Ensembles can receive input in two ways with clear priority:

1. **CLI Input (Highest Priority)**: `llm-orc invoke ensemble-name "Your specific question"`
2. **Default Task (Fallback)**: `default_task` field in ensemble configuration

**Examples:**
```bash
# CLI input overrides default_task
llm-orc invoke mycology-meets-technology "How do fungi communicate?"
# Uses: "How do fungi communicate?" (ignores config default_task)

# No CLI input uses default_task  
llm-orc invoke mycology-meets-technology
# Uses: "Analyze how mycorrhizal networks..." (from config default_task)
```

#### Ensemble Configuration with Model Profiles

Create ensembles using model profiles for simplified, consistent configuration:

```yaml
name: my-custom-ensemble
description: Brief description of what this ensemble does

default_task: "Optional default task when no CLI input provided"

agents:
  - name: local-analyst
    role: quick-researcher
    model_profile: development  # Fast local analysis
    system_prompt: "You are a quick researcher for initial analysis..."
    timeout_seconds: 60

  - name: expert-reviewer
    role: senior-expert
    model_profile: production   # High-quality cloud analysis
    system_prompt: "You are a senior expert providing final analysis..."
    timeout_seconds: 90

  - name: cost-optimizer
    role: efficiency-expert
    model_profile: claude-api   # Pay-per-use for specialized tasks
    system_prompt: "You optimize for cost-effectiveness and efficiency..."
    timeout_seconds: 60

coordinator:
  model_profile: production     # Use high-quality model for synthesis
  system_prompt: "You are an executive coordinator..." # Optional: role injection
  synthesis_prompt: |
    Synthesize the research and expert analysis into actionable insights.

    Provide:
    1. Key findings from all perspectives
    2. Strategic recommendations
    3. Implementation next steps
  timeout_seconds: 120
```

### Managing Model Profiles

#### Viewing Available Profiles

Use the `list-profiles` command to see all available model profiles:

```bash
llm-orc list-profiles
```

This shows profiles organized by location:
- **📁 Local Repo**: Profiles defined in `.llm-orc/config.yaml` (project-specific)
- **🌐 Global**: Profiles defined in `~/.config/llm-orc/config.yaml` (system-wide)

Example output:
```
Available model profiles:

📁 Local Repo (.llm-orc/config.yaml):
  development:
    Model: llama3
    Provider: ollama
    Cost per token: 0.0

🌐 Global (~/.config/llm-orc/config.yaml):
  production:
    Model: claude-3-5-sonnet-20241022
    Provider: anthropic-claude-pro-max
    Cost per token: Not specified

  claude-api:
    Model: claude-3-5-sonnet-20241022
    Provider: anthropic-api
    Cost per token: 3.0e-06
```

#### Profile Override Hierarchy

- **Local profiles** (`.llm-orc/config.yaml`) override global profiles with the same name
- **Global profiles** (`~/.config/llm-orc/config.yaml`) provide system-wide defaults
- Use local profiles for project-specific model preferences
- Use global profiles for consistent defaults across all projects

#### Local Ensembles (Personal/Private)
Use special naming patterns for personal experiments that won't be committed:

```yaml
# File: my-experiments-local.yaml (automatically gitignored)
name: my-experiments-local
description: Personal ensemble for testing - not committed to git

agents:
  - name: creative-explorer
    model: llama3
    system_prompt: "Experimental role for creative exploration..."

  - name: practical-evaluator  
    model: anthropic-claude-pro-max  # Can use OAuth for personal testing
    system_prompt: "Personal evaluation approach..."

coordinator:
  model: llama3
  synthesis_prompt: "Personal synthesis style..."
```

**Local Ensemble Patterns (Auto-Gitignored):**
- `*-local.yaml` (e.g., `my-experiments-local.yaml`)
- `local-*.yaml` (e.g., `local-testing.yaml`)

**Use Local Ensembles For:**
- 🧪 **Personal experiments** and testing
- 🔒 **Sensitive configurations** with private data
- 🚧 **Work-in-progress** before sharing with team
- ⚙️ **Personal productivity** ensembles
- 🔑 **OAuth testing** without exposing credentials

## Model Profile Selection Guidelines

### Profile Types and Use Cases

#### Development Profiles (`development`, `llama3`)
**Best for:**
- **Experimentation and iteration**
- **Privacy-sensitive content**
- **High-volume/low-stakes analysis**
- **Cost-conscious projects**
- **Offline environments**
- **Quick prototyping**

#### Production Profiles (`production`, `claude-4-sonnet`)
**Best for:**
- **High-stakes decisions**
- **Complex reasoning tasks**
- **Professional/business analysis**
- **Final synthesis and coordination**
- **When you have existing Claude Pro/Max subscription**

#### API Profiles (`claude-api`, `gemini-2.5-flash`)
**Best for:**
- **Production systems with budget**
- **Specific model requirements**
- **When you need guaranteed availability**
- **Integration with other services**
- **Specialized tasks requiring specific models**

### Profile Selection Strategy

**For Mixed Ensembles:**
- Use `development` profiles for initial analysis and research
- Use `production` profiles for synthesis and final coordination
- Use `claude-api` profiles for specialized tasks requiring specific capabilities

**Example Mixed Configuration:**
```yaml
agents:
  - name: researcher
    model_profile: development    # Fast, free research
  - name: expert
    model_profile: production     # High-quality analysis
  - name: specialist
    model_profile: claude-api     # Specific model capabilities

coordinator:
  model_profile: production       # Best quality for synthesis
```

## Best Practices

### Ensemble Design with Model Profiles

1. **Mix profile types** based on task complexity and cost considerations
2. **Use development profiles** for exploration and iteration
3. **Use production profiles** for synthesis and high-stakes decisions
4. **Use API profiles** for specialized capabilities when budget allows
5. **Diverse perspectives** - avoid redundant roles
6. **Clear role definitions** for better agent performance

### Model Profile Management

- **Define global profiles** for system-wide consistency
- **Override with local profiles** for project-specific needs  
- **Use descriptive profile names** (`development`, `production` vs `profile1`, `profile2`)
- **Document profile purposes** in comments
- **Test profiles** with validation ensembles

### Cost Optimization with Profiles

- **Start with development profiles** for iteration and experimentation
- **Use production profiles** for subscription-based models (no additional cost)
- **Reserve API profiles** for specialized tasks requiring specific models
- **Monitor usage** through ensemble reports and profile cost tracking

## Security and Privacy

### Credential Protection

LLM Orchestra uses **AES encryption** to protect all stored credentials:

- **Encrypted Storage**: All API keys and OAuth tokens are encrypted with unique keys
- **File Permissions**: Credential files use `0o600` permissions (owner read/write only)  
- **Git Protection**: Credential files are automatically gitignored to prevent accidental commits
- **Local Only**: Credentials never leave your machine

**Important Security Notes:**
- ✅ **Safe to commit**: Configuration files (`config.yaml`, ensemble files)
- ❌ **Never commit**: `credentials.enc`, `.key`, or any `*.enc` files
- 🔒 **Gitignored by default**: The repository automatically ignores credential files

```bash
# Run comprehensive security check
./scripts/check-security.sh

# Manual verification
git status  # Should not show credentials.enc, .key, or *-local.yaml files

# Check file permissions (should be 600)
ls -la ~/.llm-orc/
```

### Privacy Features

- **No telemetry**: LLM Orchestra doesn't send usage data anywhere
- **Local processing**: All coordination happens on your machine
- **Provider isolation**: Different providers can't access each other's credentials
- **Automatic cleanup**: Expired tokens are automatically refreshed or removed

## Advanced Features

### Role Injection

All models support role injection through `system_prompt`, allowing specialized expertise while maintaining authentication:

```yaml
agents:
  - name: specialist
    model: anthropic-claude-pro-max
    system_prompt: "You are a domain expert in X with Y years of experience..."
```

### Configurable Coordinators

Coordinators can use any model and specialized roles:

```yaml
coordinator:
  model: anthropic-claude-pro-max
  system_prompt: "You are a senior executive..."
  synthesis_prompt: "Synthesize insights into actionable strategy..."
```

### Timeout Management

Configure timeouts at multiple levels:

```yaml
agents:
  - timeout_seconds: 60 # Agent-specific timeout

coordinator:
  timeout_seconds: 120 # Coordinator timeout
  synthesis_timeout_seconds: 90 # Synthesis-specific timeout
```

### Error Handling

Ensembles gracefully handle:

- **Agent failures** - continue with available results
- **Token expiration** - automatic refresh for OAuth
- **Network issues** - timeout and retry logic
- **Model unavailability** - intelligent fallback with user feedback

### Smart Model Fallbacks

When models fail to load, LLM Orchestra uses intelligent fallbacks:

**Fallback Priority:**
1. **Coordinator models**: `production` → `fast` → `llama3`
2. **General models**: `fast` → `production` → `llama3`
3. **Known local models**: Treated as Ollama models directly

**User Feedback:**
```bash
# Examples of fallback messages you'll see:
ℹ️  No coordinator model specified, using configured default
🔄 Using fallback model 'claude-3-5-sonnet' (from configured defaults)
⚠️  Failed to load coordinator model 'unavailable-model': Network error
🆘 Using hardcoded fallback: llama3 (consider configuring default_models)
```

**Configuration:**
```yaml
project:
  default_models:
    fast: llama3              # Used for quick/local processing
    production: claude-3-5-sonnet  # Used for high-quality tasks
```

## Troubleshooting

### Common Issues

**OAuth token expired:**

```bash
llm-orc auth refresh anthropic-claude-pro-max
```

**Local model not found:**

```bash
ollama pull llama3
```

**Seeing fallback messages:**

```bash
# If you see: "Using hardcoded fallback: llama3"
# Consider configuring default models in config.yaml:
project:
  default_models:
    fast: your-preferred-local-model
    production: your-preferred-cloud-model
```

**Ensemble not found:**

```bash
# Check available ensembles
llm-orc list-ensembles
```

**Model profile issues:**

```bash
# Check available model profiles
llm-orc list-profiles

# Verify profile configuration
cat .llm-orc/config.yaml
cat ~/.config/llm-orc/config.yaml
```

**Authentication issues:**

```bash
# Test specific provider
llm-orc auth test anthropic-claude-pro-max

# Check all configured providers
llm-orc auth list
```

**Security check before committing:**
```bash
# Run the built-in security check
./scripts/check-security.sh

# Manual verification
git status | grep -E "\.(enc|key)$|credentials\."

# If any credential files appear, remove them
git reset HEAD credentials.enc  # Example
```

### Getting Help

```bash
# Get help with commands
llm-orc --help
llm-orc invoke --help

# Check configuration
llm-orc config show

# Validate ensemble files
llm-orc validate ensemble-name
```

---

_For more information, see the [LLM Orchestra documentation](https://github.com/mrilikecoding/llm-orc) or run `llm-orc --help`._
