# Model Profiles Guide

Model profiles simplify LLM Orchestra ensemble configuration by providing named shortcuts for common model + provider combinations. Instead of specifying explicit model and provider for each agent, you can use descriptive profile names that abstract away the technical details.

## Overview

Model profiles provide:
- **Simplified Configuration**: Use `model_profile: production` instead of `model: claude-3-5-sonnet-20241022` + `provider: anthropic-claude-pro-max`
- **Consistency**: Same profile names work across all ensembles
- **Cost Tracking**: Built-in cost information for budgeting and analysis
- **Flexibility**: Local profiles override global ones for project-specific needs
- **Maintainability**: Update all ensembles by changing the profile definition

## Quick Start

### 1. View Available Profiles

```bash
# List all available model profiles
llm-orc list-profiles
```

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

### 2. Use Profiles in Ensembles

```yaml
name: my-ensemble
description: Example ensemble using model profiles

agents:
  - name: researcher
    role: initial-analyst
    model_profile: development  # Fast, free local analysis
    system_prompt: "You are a researcher conducting initial analysis..."

  - name: expert
    role: senior-reviewer
    model_profile: production   # High-quality cloud analysis
    system_prompt: "You are a senior expert providing final review..."

coordinator:
  model_profile: production     # Best quality for synthesis
  synthesis_prompt: "Synthesize the research and expert analysis..."
```

## Profile Configuration

### Profile Structure

```yaml
model_profiles:
  profile-name:
    model: actual-model-identifier
    provider: authentication-provider-key
    cost_per_token: 0.0  # Optional: USD cost per token
```

### Default Profiles

LLM Orchestra includes these default profiles:

#### Development Profiles
```yaml
development:
  model: llama3
  provider: ollama
  cost_per_token: 0.0

llama3:
  model: llama3
  provider: ollama
  cost_per_token: 0.0
```

#### Production Profiles
```yaml
production:
  model: claude-3-5-sonnet-20241022
  provider: anthropic-claude-pro-max
  # No cost_per_token: subscription-based

claude-4-sonnet:
  model: claude-3-5-sonnet-20241022
  provider: anthropic-claude-pro-max
```

#### API Profiles
```yaml
claude-api:
  model: claude-3-5-sonnet-20241022
  provider: anthropic-api
  cost_per_token: 3.0e-06

gemini-2.5-flash:
  model: gemini-2.5-flash
  provider: google-gemini
  cost_per_token: 1.0e-06
```

## Configuration Hierarchy

### Global vs Local Profiles

**Global Profiles** (`~/.config/llm-orc/config.yaml`):
- System-wide defaults
- Shared across all projects
- Standard profiles for common use cases

**Local Profiles** (`.llm-orc/config.yaml`):
- Project-specific overrides
- Custom profiles for specific needs
- Override global profiles with same name

### Configuration Files

#### Global Configuration
```yaml
# ~/.config/llm-orc/config.yaml
model_profiles:
  production:
    model: claude-3-5-sonnet-20241022
    provider: anthropic-claude-pro-max

  development:
    model: llama3
    provider: ollama
    cost_per_token: 0.0

  claude-api:
    model: claude-3-5-sonnet-20241022
    provider: anthropic-api
    cost_per_token: 3.0e-06
```

#### Local Project Configuration
```yaml
# .llm-orc/config.yaml
project:
  name: "My Project"

model_profiles:
  # Override global development profile to use different model
  development:
    model: llama3.1
    provider: ollama
    cost_per_token: 0.0

  # Project-specific profile
  specialized:
    model: gemini-2.5-flash
    provider: google-gemini
    cost_per_token: 1.0e-06
```

## Profile Types and Use Cases

### Development Profiles
**Purpose**: Fast, free local models for experimentation and iteration

**Best for**:
- Rapid prototyping
- Privacy-sensitive content
- High-volume testing
- Cost-conscious development
- Offline environments

**Examples**:
```yaml
development:
  model: llama3
  provider: ollama
  cost_per_token: 0.0

fast:
  model: llama3.1
  provider: ollama
  cost_per_token: 0.0
```

### Production Profiles
**Purpose**: High-quality cloud models using subscription-based auth

**Best for**:
- High-stakes decisions
- Complex reasoning tasks
- Professional analysis
- Final synthesis and coordination
- When you have Claude Pro/Max subscription

**Examples**:
```yaml
production:
  model: claude-3-5-sonnet-20241022
  provider: anthropic-claude-pro-max

claude-4-sonnet:
  model: claude-3-5-sonnet-20241022
  provider: anthropic-claude-pro-max
```

### API Profiles
**Purpose**: Pay-per-use models for specific capabilities

**Best for**:
- Production systems with budget
- Specific model requirements
- Guaranteed availability needs
- Integration with other services
- Specialized tasks

**Examples**:
```yaml
claude-api:
  model: claude-3-5-sonnet-20241022
  provider: anthropic-api
  cost_per_token: 3.0e-06

gemini-2.5-flash:
  model: gemini-2.5-flash
  provider: google-gemini
  cost_per_token: 1.0e-06
```

## Best Practices

### Profile Naming

**Use descriptive names**:
```yaml
# Good
development:     # Clear purpose
production:      # Clear quality level
claude-api:      # Clear provider and type

# Avoid
profile1:        # No meaning
fast-model:      # Ambiguous
my-favorite:     # Personal preference
```

### Profile Organization

**Organize by use case**:
```yaml
model_profiles:
  # Local development
  development:
    model: llama3
    provider: ollama

  # Cloud subscription  
  production:
    model: claude-3-5-sonnet-20241022
    provider: anthropic-claude-pro-max

  # API pay-per-use
  claude-api:
    model: claude-3-5-sonnet-20241022
    provider: anthropic-api
    cost_per_token: 3.0e-06

  # Specialized capabilities
  multimodal:
    model: gemini-2.5-flash
    provider: google-gemini
    cost_per_token: 1.0e-06
```

### Mixed Profile Strategies

**Use different profiles for different roles**:
```yaml
agents:
  - name: initial-researcher
    model_profile: development    # Fast, free exploration
    
  - name: quality-reviewer
    model_profile: production     # High-quality analysis
    
  - name: specialist
    model_profile: claude-api     # Specific capabilities

coordinator:
  model_profile: production       # Best quality for synthesis
```

### Cost Optimization

**Profile selection by task complexity**:
- **Low complexity**: Use `development` profiles (free)
- **Medium complexity**: Use `production` profiles (subscription)
- **High complexity**: Use `claude-api` or specialized profiles (pay-per-use)

## Advanced Configuration

### Custom Provider Profiles

```yaml
model_profiles:
  custom-ollama:
    model: mistral
    provider: ollama
    cost_per_token: 0.0

  custom-anthropic:
    model: claude-3-haiku-20240307
    provider: anthropic-api
    cost_per_token: 1.0e-06
```

### Environment-Specific Profiles

```yaml
# Development environment
model_profiles:
  staging:
    model: llama3
    provider: ollama
    cost_per_token: 0.0

# Production environment  
model_profiles:
  staging:
    model: claude-3-5-sonnet-20241022
    provider: anthropic-api
    cost_per_token: 3.0e-06
```

### Profile Validation

Profiles are validated when ensembles load. Common errors:

**Missing required fields**:
```yaml
# Error: missing provider
invalid-profile:
  model: llama3
  # Missing provider field
```

**Invalid provider**:
```yaml
# Error: provider not configured for authentication
invalid-profile:
  model: gpt-4
  provider: openai  # If openai auth not configured
```

## Troubleshooting

### Profile Not Found

```bash
# Check available profiles
llm-orc list-profiles

# Verify profile exists in config
cat .llm-orc/config.yaml
cat ~/.config/llm-orc/config.yaml
```

### Authentication Issues

```bash
# Check if provider is configured
llm-orc auth list

# Test provider authentication
llm-orc auth test anthropic-claude-pro-max
```

### Profile Override Issues

Local profiles override global ones. Check both:
```bash
# Check global profiles
cat ~/.config/llm-orc/config.yaml

# Check local profiles  
cat .llm-orc/config.yaml
```

### Cost Tracking Issues

- `cost_per_token` is optional and for reference only
- Omit for subscription-based providers (OAuth)
- Use for pay-per-use providers (API keys)

## Migration from Explicit Model + Provider

### Before (Explicit Configuration)
```yaml
agents:
  - name: analyst
    model: claude-3-5-sonnet-20241022
    provider: anthropic-claude-pro-max
    
  - name: reviewer
    model: llama3
    provider: ollama
```

### After (Model Profiles)
```yaml
# Add to config.yaml
model_profiles:
  production:
    model: claude-3-5-sonnet-20241022
    provider: anthropic-claude-pro-max
    
  development:
    model: llama3
    provider: ollama

# Update ensemble
agents:
  - name: analyst
    model_profile: production
    
  - name: reviewer
    model_profile: development
```

### Migration Benefits

- **Consistency**: All ensembles use same profile names
- **Maintainability**: Update model versions in one place
- **Readability**: Profile names are more descriptive than model IDs
- **Cost Tracking**: Built-in cost information

## Integration with Ensemble Features

### Fallback Model Support

If a profile fails to load, ensembles fall back to configured defaults:

```yaml
project:
  default_models:
    fast: llama3
    production: claude-3-5-sonnet-20241022
```

### Usage Tracking

Profile information appears in ensemble results:
```json
{
  "metadata": {
    "usage": {
      "agents": {
        "researcher": {
          "model": "llama3",
          "provider": "ollama",
          "cost_usd": 0.0
        }
      }
    }
  }
}
```

### Timeout Configuration

Profiles work with all ensemble features:
```yaml
agents:
  - name: analyst
    model_profile: production
    timeout_seconds: 90
    system_prompt: "Custom role prompt..."
```

## See Also

- [Authentication Guide](authentication.md) - Setting up provider credentials
- [Ensemble Configuration](../README.md#configuration) - Complete ensemble setup
- [Cost Optimization](../README.md#cost-optimization) - Strategies for managing costs