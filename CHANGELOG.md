# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.1] - 2025-07-14

### Enhanced
- **Ensemble List Command** - Enhanced `list` command to display ensembles from both local and global directories
  - Updated to use ConfigurationManager for automatic directory discovery
  - Shows ensembles from multiple configured directories with source indication
  - Automatic migration handling from legacy `~/.llm-orc` location
  - Improved user guidance for configuration setup when no ensembles found
  - Better support for mixed local/global ensemble workflows

## [0.4.0] - 2025-07-13

### Added
- **MCP Server Integration** - Model Context Protocol server implementation
  - Expose llm-orc ensembles as tools via standardized MCP protocol
  - HTTP transport on configurable port (default 3000)
  - Stdio transport for direct process communication
  - New `llm-orc serve <ensemble> --port <port>` command
  - Seamless integration with existing configuration system
  - Enables external tools (Claude Code, VS Code extensions) to leverage domain-specific workflows

- **Enhanced OAuth Authentication** - Complete Claude Pro/Max OAuth implementation
  - Anthropic Claude Pro/Max OAuth support with subscription-based access
  - Hardcoded client ID for seamless setup experience
  - PKCE (Proof Key for Code Exchange) security implementation
  - Manual token extraction flow with Cloudflare protection handling
  - Interactive OAuth setup with browser automation
  - Token refresh capabilities with automatic credential updates
  - Role injection system for OAuth token compatibility

- **Enhanced Ensemble Configuration** - CLI override and smart fallback system
  - CLI input now overrides ensemble `default_task` when provided
  - Renamed `task` to `default_task` for clearer semantics (backward compatible)
  - Smart fallback system using user-configured defaults instead of hardcoded values
  - Context-aware model fallbacks for coordinator vs general use
  - Optional `cost_per_token` field for subscription-based pricing models
  - Comprehensive user feedback and logging for fallback behavior

### Changed
- **Authentication Commands** - Enhanced CLI with OAuth-specific flows
  - `llm-orc auth add anthropic` now provides interactive setup wizard
  - Special handling for `anthropic-claude-pro-max` provider with guided OAuth
  - Improved error handling and user guidance throughout OAuth flow
  - Token storage includes client_id and refresh token management

- **Model System** - OAuth model integration and conversation handling
  - `OAuthClaudeModel` class with automatic token refresh
  - Role injection system for seamless agent role establishment
  - Conversation history management for OAuth token authentication
  - Enhanced error handling with automatic retry on token expiration

### Technical
- Added `MCPServer` class with full MCP protocol implementation
- Added `MCPServerRunner` for HTTP and stdio transport layers
- Enhanced `AnthropicOAuthFlow` with manual callback flow and token extraction
- Updated ensemble execution with CLI override logic and smart fallbacks
- Added comprehensive test coverage for MCP server and OAuth enhancements
- Pre-commit hooks with auto-fix capabilities for code quality

### Fixed
- Token expiration handling with automatic refresh and credential updates
- Ensemble configuration backward compatibility while introducing clearer semantics
- Linting and formatting issues resolved with ruff auto-fix integration

## [0.3.0] - 2025-01-10

### Added
- **OAuth Provider Integration** - Complete OAuth authentication support for major LLM providers
  - Google Gemini OAuth flow with `generative-language.retriever` scope
  - Anthropic OAuth flow for MCP server integration
  - Provider-specific OAuth flow factory pattern for extensibility
  - Comprehensive test coverage using TDD methodology (Red → Green → Refactor)
  - Real authorization URLs and token exchange endpoints
  - Enhanced CLI authentication commands supporting both API keys and OAuth

### Changed
- **Authentication System** - Extended to support multiple authentication methods
  - `llm-orc auth add` now accepts both `--api-key` and OAuth credentials
  - `llm-orc auth list` shows authentication method (API key vs OAuth)
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