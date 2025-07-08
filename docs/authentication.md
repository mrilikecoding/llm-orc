# Authentication System

LLM Orchestra provides a secure authentication system for managing API keys across multiple LLM providers.

## Overview

The authentication system provides:
- **Encrypted credential storage** using Fernet encryption
- **Multiple provider support** (Anthropic, Google, OpenAI, etc.)
- **CLI management commands** for easy credential administration
- **Secure file permissions** on all credential files
- **Test functionality** to verify API key validity

## Quick Setup

### Interactive Setup Wizard

The easiest way to get started:

```bash
llm-orc auth setup
```

This will prompt you for:
1. Provider name (e.g., "anthropic", "google", "openai")
2. API key (input is hidden for security)
3. Option to add additional providers

### Manual Provider Management

Add providers individually:

```bash
# Add an API key for a provider
llm-orc auth add anthropic --api-key YOUR_API_KEY

# List all configured providers
llm-orc auth list

# Test authentication for a provider
llm-orc auth test anthropic

# Remove a provider's credentials
llm-orc auth remove anthropic
```

## Security Features

### Encryption

All API keys are encrypted using the Fernet symmetric encryption algorithm before being stored to disk. The encryption key is generated automatically and stored separately from the credentials.

### File Permissions

- Credentials file: `~/.llm-orc/credentials.yaml` (permissions: 0o600)
- Encryption key: `~/.llm-orc/.encryption_key` (permissions: 0o600)

### Directory Structure

```
~/.llm-orc/
├── credentials.yaml      # Encrypted API keys
├── .encryption_key       # Encryption key (auto-generated)
└── ensembles/           # Ensemble configurations
    └── *.yaml
```

## Supported Providers

The authentication system supports any provider name, but LLM Orchestra includes built-in support for:

- **anthropic** - Claude models
- **google** - Gemini models
- **openai** - GPT models
- **ollama** - Local models (no authentication required)

## CLI Commands Reference

### `llm-orc auth add <provider>`

Add or update API key for a provider.

**Options:**
- `--api-key` (required): API key for the provider
- `--config-dir`: Custom config directory (default: ~/.llm-orc)

**Example:**
```bash
llm-orc auth add anthropic --api-key sk-ant-api03-...
```

### `llm-orc auth list`

List all configured authentication providers.

**Options:**
- `--config-dir`: Custom config directory

**Example output:**
```
Configured providers:
  anthropic: API key
  google: API key
```

### `llm-orc auth remove <provider>`

Remove authentication credentials for a provider.

**Options:**
- `--config-dir`: Custom config directory

**Example:**
```bash
llm-orc auth remove anthropic
```

### `llm-orc auth test <provider>`

Test authentication for a provider by attempting to use the stored API key.

**Options:**
- `--config-dir`: Custom config directory

**Example:**
```bash
llm-orc auth test anthropic
# Output: Authentication for anthropic is working
```

### `llm-orc auth setup`

Interactive setup wizard to configure multiple providers.

**Options:**
- `--config-dir`: Custom config directory

## Configuration Directory

By default, authentication data is stored in `~/.llm-orc/`. You can use a custom directory with the `--config-dir` option on all auth commands.

**Using custom config directory:**
```bash
llm-orc auth add anthropic --api-key YOUR_KEY --config-dir /path/to/custom/config
llm-orc auth list --config-dir /path/to/custom/config
```

## Error Handling

The authentication system provides clear error messages for common issues:

- **Missing API key**: "Error: Missing option '--api-key'"
- **Provider not found**: "No authentication found for {provider}"
- **Invalid credentials**: "Authentication for {provider} failed"
- **Storage errors**: Detailed error messages for file system issues

## Best Practices

1. **Use the setup wizard** for initial configuration
2. **Test authentication** after adding new providers
3. **Keep API keys secure** - never commit them to version control
4. **Use provider-specific names** (e.g., "anthropic" not "claude")
5. **Regularly test credentials** to catch expired keys early

## Troubleshooting

### Permission Denied Errors

If you see permission errors, check that the config directory is writable:

```bash
ls -la ~/.llm-orc/
# Should show files with 600 permissions
```

### Corrupted Credentials

If credentials become corrupted, you can reset by removing the credentials file:

```bash
rm ~/.llm-orc/credentials.yaml
llm-orc auth setup  # Reconfigure
```

### Testing Authentication

If authentication tests fail, verify:

1. API key is correct and hasn't expired
2. Network connectivity to the provider
3. Provider service status

```bash
# Test and see detailed error
llm-orc auth test anthropic -v
```

## Integration with Ensembles

Authentication is automatically handled when running ensembles. If credentials are missing for a required provider, you'll see a clear error message with instructions to run `llm-orc auth add`.

Example ensemble configuration:
```yaml
name: analysis
agents:
  - name: analyst
    model: claude-3-haiku    # Uses "anthropic" provider
    # ... rest of config
```

When you run this ensemble, LLM Orchestra will automatically use the stored Anthropic credentials.