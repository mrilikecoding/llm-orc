# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2025-01-10

### Added
- **OAuth Provider Integration** - Complete OAuth authentication support for major LLM providers
  - Google Gemini OAuth flow with `generative-language.retriever` scope
  - Anthropic OAuth flow for MCP server integration
  - Provider-specific OAuth flow factory pattern for extensibility
  - New `llm-orc auth oauth` command for OAuth configuration
  - Comprehensive test coverage using TDD methodology (Red → Green → Refactor)
  - Real authorization URLs and token exchange endpoints
  - Enhanced CLI authentication commands supporting both API keys and OAuth

### Changed
- **Authentication System** - Extended to support multiple authentication methods
  - `llm-orc auth add` now accepts both `--api-key` and OAuth credentials
  - `llm-orc auth list` shows authentication method (API key vs OAuth)
  - `llm-orc auth test` validates both API keys and OAuth tokens with expiration checking
  - `llm-orc auth setup` interactive wizard supports OAuth method selection

### Technical
- Added `GoogleGeminiOAuthFlow` class with Google-specific endpoints
- Added `AnthropicOAuthFlow` class with Anthropic console integration  
- Implemented `create_oauth_flow()` factory function for provider selection
- Updated `AuthenticationManager` to use provider-specific OAuth flows
- Added comprehensive OAuth provider integration test suite

## [0.2.2] - 2025-01-09

### Added
- **Automated Homebrew releases** - GitHub Actions workflow automatically updates Homebrew tap on release
  - Triggers on published GitHub releases
  - Calculates SHA256 hash automatically
  - Updates formula with new version and hash
  - Provides validation and error handling
  - Eliminates manual Homebrew maintenance

## [0.2.1] - 2025-01-09

### Fixed
- **CLI version command** - Fixed `--version` flag that was failing with package name detection error
  - Explicitly specify `package_name="llm-orchestra"` in Click's version_option decorator
  - Resolves RuntimeError when Click tried to auto-detect version from `llm_orc` module name
  - Package name is `llm-orchestra` but module is `llm_orc` causing the detection to fail

## [0.2.0] - 2025-01-09

### Added
- **XDG Base Directory Specification compliance** - Configuration now follows XDG standards
  - Global config moved from `~/.llm-orc` to `~/.config/llm-orc` (or `$XDG_CONFIG_HOME/llm-orc`)
  - Automatic migration from old location with user notification
  - Breadcrumb file left after migration for reference

- **Local repository configuration support** - Project-specific configuration
  - `.llm-orc` directory discovery walking up from current working directory
  - Local configuration takes precedence over global configuration
  - `llm-orc config init` command to initialize local project configuration
  - Project-specific ensembles, models, and scripts directories

- **Enhanced configuration management system**
  - New `ConfigurationManager` class for centralized configuration handling
  - Configuration hierarchy: local → global with proper precedence
  - Ensemble directory discovery in priority order
  - Project-specific configuration with model profiles and defaults

- **New CLI commands**
  - `llm-orc config init` - Initialize local project configuration
  - `llm-orc config migrate` - Manually migrate from old configuration location
  - `llm-orc config show` - Display current configuration information and paths

### Changed
- **Configuration system completely rewritten** for better maintainability
  - Authentication commands now use `ConfigurationManager` instead of direct paths
  - All configuration paths now computed dynamically based on XDG standards
  - Improved error handling and user feedback for configuration operations

- **Test suite improvements**
  - CLI authentication tests rewritten to use proper mocking
  - Configuration manager tests added with comprehensive coverage (20 test cases)
  - All tests now pass consistently with new configuration system

- **Development tooling**
  - Removed `black` dependency in favor of `ruff` for formatting
  - Updated development dependencies to use `ruff` exclusively
  - Improved type annotations throughout codebase

### Fixed
- **CLI test compatibility** with new configuration system
  - Fixed ensemble invocation tests to handle new error scenarios
  - Updated authentication command tests to work with `ConfigurationManager`
  - Resolved all CI test failures and linting issues

- **Configuration migration robustness**
  - Proper error handling when migration conditions aren't met
  - Safe directory creation with parent directory handling
  - Breadcrumb file creation for migration tracking

### Technical Details
- Issues resolved: #21 (XDG compliance), #22 (local repository support)
- 101/101 tests passing with comprehensive coverage
- All linting and type checking passes with `ruff` and `mypy`
- Configuration system now fully tested and production-ready

## [0.1.3] - Previous Release
- Basic authentication and ensemble management functionality
- Initial CLI interface with invoke and list-ensembles commands
- Multi-provider LLM support (Anthropic, Google, Ollama)
- Credential storage with encryption support