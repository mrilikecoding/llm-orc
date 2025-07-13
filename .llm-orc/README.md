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

The main project configuration file that defines default models, cost profiles, and project settings.

#### Structure

```yaml
project:
  name: "Your Project Name"
  default_models:
    fast: llama3 # Local/fast model for quick tasks
    production: claude-3-5-sonnet # High-quality cloud model
    oauth: claude-3-5-sonnet # OAuth-authenticated model

model_profiles:
  development:
    - model: llama3
      provider: ollama
      cost_per_token: 0.0  # Optional: for budgeting reference

  production:
    - model: claude-3-5-sonnet
      provider: anthropic
      cost_per_token: 3.0e-06  # Optional: $3 per million tokens

  oauth_test:
    - model: claude-3-5-sonnet
      provider: anthropic-claude-pro-max
      # No cost_per_token: subscription-based pricing
```

#### Configuration Options

**Project Settings:**

- `name`: Display name for your project
- `default_models`: Quick references for common model types

**Model Profiles:**

- `development`: Cost-efficient models for experimentation
- `production`: High-quality models for important tasks
- `oauth_test`: OAuth-authenticated models using subscriptions

**Model Definition Fields:**

- `model`: Model identifier (required)
- `provider`: Authentication provider key (required)
- `cost_per_token`: USD cost per token (optional, for budgeting reference only)

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

#### Shared Ensembles (Committed to Git)
Regular ensemble files are shared with your team:

```yaml
name: my-custom-ensemble
description: Brief description of what this ensemble does

default_task: "Optional default task when no CLI input provided"

agents:
  - name: agent-1
    role: descriptive-role-name
    model: llama3 # Local model
    system_prompt: "Detailed role description..."
    timeout_seconds: 60

  - name: agent-2
    role: another-role
    model: anthropic-claude-pro-max # OAuth model
    system_prompt: "Another role description..."
    timeout_seconds: 90

coordinator:
  model: llama3 # Can be any model
  system_prompt: "Coordinator role..." # Optional: role injection
  synthesis_prompt: |
    Instructions for synthesizing agent responses.

    Provide:
    1. Key insights
    2. Recommendations
    3. Next steps
  timeout_seconds: 120
```

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

## Model Selection Guidelines

### When to Use Local Models (Ollama)

- **Experimentation and iteration**
- **Privacy-sensitive content**
- **High-volume/low-stakes analysis**
- **Cost-conscious projects**
- **Offline environments**

### When to Use OAuth Models (Claude Pro/Max)

- **High-stakes decisions**
- **Complex reasoning tasks**
- **Professional/business analysis**
- **When you have existing subscription**
- **Final synthesis and coordination**

### When to Use API Models

- **Production systems**
- **Specific model requirements**
- **When you need guaranteed availability**
- **Integration with other services**

## Best Practices

### Ensemble Design

1. **Mix model types** based on task complexity
2. **Use OAuth for coordination** when you have subscription
3. **Local models for exploration**, cloud for synthesis
4. **Diverse perspectives** - avoid redundant roles
5. **Clear role definitions** for better agent performance

### Authentication Management

- **OAuth for subscriptions** - no per-token costs
- **API keys for production** - predictable billing
- **Local models for development** - zero cost
- **Test authentication** before important runs

### Cost Optimization

- **Start with local models** for iteration
- **Use OAuth when available** (subscription-based)
- **Reserve API calls** for final results
- **Monitor usage** through ensemble reports

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
llm-orc list
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
