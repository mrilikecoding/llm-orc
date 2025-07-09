# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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