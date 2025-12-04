# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.12.1] - 2025-12-04

### Added
- `set_project` MCP tool for multi-project support
  - Allows MCP server to operate on any project directory
  - ConfigurationManager accepts optional `project_dir` parameter
  - All subsequent tool calls use the specified project's config
  - Updated `get_help` with `context_management` category

## [0.12.0] - 2025-12-04

### Added
- **MCP Server Architecture (ADR-009)** - Model Context Protocol server for Claude Code integration
  - **25 MCP tools** organized by category: core execution, provider discovery, CRUD operations
  - **FastMCP SDK integration** with decorator-based tool and resource registration
  - **Streaming progress** via FastMCP Context for real-time execution feedback
  - **Automatic artifact storage** with JSON and Markdown output for each execution

- **Core Execution Tools**
  - `invoke` - Execute ensembles with streaming progress and artifact storage
  - `list_ensembles` - List all ensembles from local/library/global sources
  - `validate_ensemble` - Validate configuration, profiles, and dependencies
  - `update_ensemble` - Modify ensemble config with dry-run and backup support
  - `analyze_execution` - Analyze execution artifact data

- **Provider Discovery**
  - `get_provider_status` - Show available providers and Ollama models
  - `check_ensemble_runnable` - Validate ensemble can run, suggest local alternatives

- **CRUD Operations**
  - Ensemble: `create_ensemble`, `delete_ensemble`
  - Profile: `list_profiles`, `create_profile`, `update_profile`, `delete_profile`
  - Script: `list_scripts`, `get_script`, `test_script`, `create_script`, `delete_script`
  - Library: `library_browse`, `library_search`, `library_copy`, `library_info`
  - Artifact: `delete_artifact`, `cleanup_artifacts`

- **Agent Onboarding**
  - `get_help` - Comprehensive documentation for agents using the MCP server
  - Directory structure, YAML schemas, tool categories, and workflow patterns

- **MCP Resources** (read-only data access)
  - `llm-orc://ensembles` - List all available ensembles
  - `llm-orc://ensemble/{name}` - Get complete ensemble configuration
  - `llm-orc://profiles` - List available model profiles
  - `llm-orc://artifacts/{ensemble}` - List execution artifacts
  - `llm-orc://artifact/{ensemble}/{id}` - Get individual artifact details
  - `llm-orc://metrics/{ensemble}` - Get aggregated execution metrics

- **CLI Integration**
  - `llm-orc mcp serve` - Start MCP server for Claude Code
  - HTTP transport option for debugging

### Technical
- 64 BDD scenarios validating MCP server behavior
- 118 unit tests for MCP server components
- Dependency injection for testability
- Test injection points for mocking Ollama status

## [0.11.0] - 2025-11-25

### Added
- **[#24] Script Agent System** - Complete infrastructure for script-based agents in ensembles
  - **EnhancedScriptAgent**: JSON stdin/stdout contract with Pydantic schema validation
  - **ScriptResolver**: Priority-based script discovery from `.llm-orc/scripts/` directories
  - **ArtifactManager**: Timestamped execution results with JSON and Markdown output
  - **Implicit agent type detection**: Auto-detect script vs LLM agents by configuration fields
  - **Human-in-the-loop workflows**: Interactive user input primitives for research validation
  - **CLI commands**: `scripts list`, `scripts show`, `scripts test` for script management
  - **Library integration**: Automatic primitive script installation from llm-orchestra-library

- **Primitive Script Library** - Ready-to-use scripts for common operations
  - `file-ops/`: read_file.py, write_file.py for file I/O
  - `user-interaction/`: get_user_input.py, confirm_action.py for interactive workflows
  - `data-transform/`: json_extract.py for data manipulation
  - `control-flow/`: replicate_n_times.py for execution control

- **Pydantic Schema System** - Type-safe interfaces for script agents
  - `ScriptAgentInput/Output` schemas for contract validation
  - `ConversationState` for multi-turn conversation tracking
  - Event-driven architecture foundation with base Event model

- **BDD Test Suite** - 164 behavioral scenarios validating all ADRs
  - ADR-001: Pydantic script interfaces
  - ADR-002: Composable primitive system
  - ADR-003: Testable script contracts
  - ADR-005: Multi-turn conversations
  - ADR-006: Library-based primitives architecture
  - ADR-007: Progressive ensemble validation
  - ADR-008: LLM-friendly CLI and MCP design

### Changed
- **Library source configuration**: Require explicit `LLM_ORC_LIBRARY_SOURCE` for remote fetching
- **Init behavior**: Graceful fallback when no library is configured (no scripts installed by default)
- **Configuration hierarchy**: `with_scripts` parameter propagated through init chain

### Technical
- 142 commits implementing Issue #24 requirements
- 2,331 tests passing with 93.5% coverage
- Full compliance with ruff, mypy strict, and complexipy standards
- Security controls for script execution with command validation

## [0.10.1] - 2025-08-07

### Fixed
- **Critical Dependency**: Added missing `psutil>=5.9.0` dependency required for adaptive resource management
  - Resolves `ModuleNotFoundError: No module named 'psutil'` when using resource monitoring features
  - Ensures Homebrew formula includes all required dependencies for v0.10.0 features

## [0.10.0] - 2025-08-07

### Added
- **🎯 Adaptive Resource Management** - Complete system for monitoring and managing agent execution resources
  - Semaphore-based concurrency control with configurable `max_concurrent` limits
  - Real-time CPU and memory monitoring during ensemble execution
  - Per-phase performance metrics with peak and average resource tracking
  - User guidance and optimization recommendations based on system performance
  - AgentExecutor integration with comprehensive resource monitoring hooks
  - Backward compatible with existing ensemble configurations
- **🏗️ JSON-First Rendering Architecture** - Unified data transformation and presentation system
  - Schema-driven transformation from raw execution data to structured JSON
  - Consistent text/markdown renderers with single source of truth
  - API-ready structured output for all execution data
  - Eliminated scattered formatting functions across visualization modules
  - Comprehensive data schemas for execution results, metrics, and usage data
- **📊 Enhanced Monitoring & Visualization** - Improved execution feedback and display
  - Per-phase performance statistics with detailed resource breakdowns
  - Peak and average CPU/memory utilization tracking across execution phases
  - Final dependency graph display showing agent relationships after completion
  - User-friendly 1-based phase numbering for better readability
  - Model profile and actual model display in agent execution results
- **⚡ Parallel Execution Improvements** - Enhanced concurrent agent processing
  - Async parallel execution for independent agent phases
  - Improved performance monitoring for parallel workloads
  - Better dependency-aware scheduling and resource allocation
- **🔄 Enhanced Fallback Model System** - Improved model reliability and flexibility
  - Configurable fallback model chains for better resilience
  - Clear fallback status display in execution results
  - Enhanced error handling and recovery mechanisms

### Technical
- **Modular Architecture** - Broke down monolithic visualization.py into focused components
  - `visualization/dependency.py` - Agent dependency graph logic
  - `visualization/performance_metrics.py` - Resource and performance formatting
  - `visualization/results_display.py` - Execution results presentation
  - `visualization/streaming.py` - Real-time execution display and progress tracking
- **Code Quality** - Function complexity reduction and comprehensive testing
  - Extracted helper functions to reduce cyclomatic complexity violations
  - 95.19% test coverage with 1544+ passing tests
  - MyPy strict type checking compliance across all modules
  - Ruff formatting and linting compliance
- **Performance** - Optimized resource monitoring and execution coordination
  - Minimal overhead resource monitoring during execution
  - Efficient per-phase metrics collection and aggregation
  - Streamlined JSON-first architecture reducing rendering complexity

### Fixed
- Timing test reliability in CI environments with appropriate overhead allowances
- Phase numbering consistency across all output formats (now 1-based)
- Memory and CPU metric calculation accuracy using phase-based data
- Test coverage gaps in visualization and execution modules

## [0.9.1] - 2025-07-26

### Added
- **Dynamic Template Fetching** - Configuration templates now fetched dynamically from LLM Orchestra Library
  - Templates moved from local package to centralized library repository
  - Automatic fetching during `llm-orc init` and global config setup
  - Graceful fallback to local templates when library unavailable
  - Templates include: `global-config.yaml`, `local-config.yaml`, `example-local-ensemble.yaml`
  - Validation ensemble templates for testing authentication
- **Template Library Integration** - Added comprehensive template management to library repository
  - Templates available at: `https://raw.githubusercontent.com/mrilikecoding/llm-orchestra-library/main/templates/`
  - Follows same dynamic fetching pattern as ensemble commands
  - Enhanced configuration system with centralized template management

### Technical
- Enhanced `ConfigurationManager` with dynamic template content fetching
- Added `get_template_content()` function in library module with error handling
- Comprehensive test coverage for template fetching and fallback mechanisms
- Integration tests for configuration manager template usage
- Templates can be updated independently from main package releases

## [0.9.0] - 2025-07-26

### Added
- **Library CLI Commands** - Complete system for browsing and managing ensembles from the LLM Orchestra Library
  - `llm-orc library categories` - List all available ensemble categories with descriptions
  - `llm-orc library browse <category>` - Browse ensembles within a specific category
  - `llm-orc library show <ensemble>` - Display comprehensive ensemble metadata including model profiles, agent details, dependencies, and execution flow
  - `llm-orc library copy <ensemble>` - Copy ensembles from GitHub to local or global configuration with conflict handling
  - Library commands integrated into help system with 'l' alias shortcut
  - Comprehensive tab completion for category names and ensemble paths
  - Rich emoji-based UI for better user experience
  - Graceful error handling for network requests and invalid YAML
- **Enhanced Documentation** - Updated README.md with library CLI commands usage examples

### Technical
- Added `cli_library/` module with complete library management functionality
- Extended `cli_completion.py` with library ensemble path completion
- Comprehensive test coverage for all library commands (15 tests)
- Full integration with existing configuration system
- TDD implementation following project standards

## [0.8.1] - 2025-07-25

### Added
- **Tab Completion Support** - Comprehensive shell completion for improved CLI usability
  - Ensemble name completion for `invoke` and `serve` commands
  - Provider name completion for all authentication commands (`auth add`, `auth remove`, etc.)
  - Built-in `llm-orc completion` command with shell-specific setup instructions
  - Support for bash, zsh, and fish shells
  - Dynamic completion that loads data at completion time
  - Graceful error handling to prevent shell completion failures

### Technical
- Added `cli_completion.py` module with Click-based completion functions
- Comprehensive test coverage for completion functionality (7 tests, 82% coverage)
- Full type safety compliance with strict mypy checking
- TDD implementation following project standards

## [0.8.0] - 2025-07-25

### Added
- **Comprehensive Code Architecture Refactoring** - Major structural improvements for maintainability and scalability
  - Systematic file restructuring from large monolithic files to focused modules
  - New core/ directory structure for authentication, config, execution, and models
  - CLI commands organized into dedicated modules with clear separation of concerns
  - Agent types now have dedicated modules (script_agent moved to agents/)
  - Provider-specific implementations separated into individual model modules
- **Static Analysis Integration** - Enhanced code quality with automated security and dead code detection
  - Bandit security vulnerability scanning integrated into make lint pipeline
  - Vulture dead code detection for cleaner codebase maintenance
  - Individual analysis commands: `make security`, `make dead-code`
  - Complexipy complexity analysis with configurable thresholds
- **Test Quality Improvements** - Dramatically improved test reliability and coverage
  - Test warnings reduced from 22 to 4 (82% improvement)
  - Fixed AsyncMock contamination issues with better mocking strategies
  - Test coverage improved from 83% to 96%
  - Test organization now mirrors src directory structure for better navigation
- **Security Enhancements** - Critical security improvements for script execution
  - Fixed HIGH severity security issue in script_agent.py subprocess execution
  - Replaced dangerous shell=True with safer shlex.split() argument parsing
  - Added comprehensive command validation with dangerous command blocking
  - Enhanced error handling and timeout management for script agents

### Changed
- **Module Organization** - Complete restructuring of codebase architecture
  - `ensemble_execution.py` → Multiple focused execution modules in core/execution/
  - `authentication.py` → Separate core/auth/ with dedicated OAuth flows
  - `models.py` → Individual provider modules (anthropic.py, google.py, ollama.py)
  - CLI commands → Organized command modules in cli_modules/
  - Tests → Restructured to mirror src organization for better maintainability
- **Developer Experience** - Enhanced development workflow and tooling
  - `make lint` now includes 5 quality tools: mypy, ruff, complexipy, bandit, vulture
  - Pre-commit pipeline enhanced with security and dead code analysis
  - Better error messages and command validation
  - Improved test isolation and reliability

### Fixed
- Performance test timing issues in CI environments with adjusted thresholds
- Flaky tests due to AsyncMock contamination and timing sensitivities
- Security vulnerabilities in script execution with subprocess calls
- Dead code and unused imports across the codebase
- Complex function decomposition for better maintainability

### Technical Details
- 46 commits of systematic refactoring work
- 30,614 additions, 5,626 deletions
- All 1,261 tests passing with enhanced reliability
- Complexity reduction in multiple functions from 15+ to <10
- Enhanced make targets for development workflow

## [0.7.0] - 2025-07-19

### Added
- **Comprehensive CLI Visualization System** - Transform ensemble execution from black box to transparent process
  - Real-time progress tracking with Rich console streaming and live status updates
  - Professional dependency tree visualization showing execution hierarchy and flow
  - Rich symbols and colors for agent status (✓ completed, ◐ running, ✗ failed, ○ pending)
  - Markdown rendering for agent responses with automatic code block detection
  - Performance metrics display including duration, token usage, cost breakdown, and per-agent statistics
  - Cross-terminal compatibility with proper width detection and text wrapping
  - Streaming execution with live dependency tree updates during processing
  - Detailed vs simplified output modes for different use cases
- **CLI Module Refactoring** - Improved maintainability and extensibility
  - Extracted CLI code into focused modules (auth, commands, config, visualization)
  - Enhanced command organization and help display
  - Better separation of concerns for future CLI enhancements

### Fixed
- **Text Overflow Issues** - Resolved content getting cut off in various terminal environments
  - Native Rich text wrapping with proper terminal width detection
  - Consistent display across different terminal applications and sizes
  - Word boundary preservation preventing mid-word line breaks
  - Background color overflow fixes for markdown content

### Changed
- **Rich Library Integration** - Upgraded to professional terminal output
  - Replaced basic text output with Rich console formatting
  - Enhanced visual feedback for better user experience
  - Professional styling consistent with modern CLI tools

## [0.6.0] - 2025-07-17

### Added
- **Agent Dependencies** - New flexible dependency system using `depends_on` field
  - Agents can depend on specific other agents for sophisticated orchestration patterns
  - Automatic dependency validation with circular dependency detection using DFS
  - Missing dependency validation prevents configuration errors
  - Parallel execution of independent agents with sequential execution after dependencies
- **Streaming by Default** - Enhanced real-time user experience
  - Streaming enabled by default in performance configuration
  - CLI shows effective streaming setting from config
  - Real-time progress updates during ensemble execution
- **Enhanced Configuration System** - Better model profile management
  - Migration to `anthropic-claude-pro-max` with correct pricing (cost: 0.0)
  - Performance configuration section with streaming and concurrency settings
  - Improved CLI configuration display and validation

### Changed
- **BREAKING: Ensemble Configuration Format** - Migration required for existing ensembles
  - Replaced `coordinator` pattern with `depends_on` agent dependencies
  - Coordinator field removed from EnsembleConfig dataclass
  - All ensemble templates updated to use new dependency pattern
  - Legacy coordinator-based ensembles need manual migration
- **Architecture Enhancement** - Improved maintainability and performance
  - Removed centralized synthesis bottleneck for better performance
  - EnsembleExecutor handles dependency graphs instead of coordinator logic
  - Synthesis now handled by dependent agents rather than separate coordinator
- **Documentation Cleanup** - Consolidated essential information
  - Removed outdated documentation files (agent_orchestration.md, pr_review_ensemble.md, etc.)
  - Updated README with new dependency-based configuration examples
  - Added agent dependencies section with benefits and usage patterns
  - Kept only essential docs: design_philosophy.md and research analysis

### Fixed
- **Dependency Validation** - Robust configuration error prevention
  - Comprehensive circular dependency detection at configuration load time
  - Missing dependency validation with clear error messages
  - All ensemble files validated for dependency correctness
- **Test Coverage** - Complete migration to new architecture
  - Updated all 209 tests to use new dependency pattern
  - Removed coordinator-specific test files
  - Added dependency validation tests for edge cases
  - Performance tests updated to use `depends_on` instead of `dependencies`

### Performance
- **Parallel Execution** - Better resource utilization
  - Independent agents execute concurrently using asyncio
  - Dependent agents execute sequentially after dependencies complete
  - Streaming provides real-time feedback without blocking
- **Simplified Architecture** - Reduced complexity and overhead
  - Removed complex coordinator synthesis logic
  - Direct agent-to-agent dependency resolution
  - More efficient execution patterns

## [0.5.1] - 2025-07-16

### Fixed
- **Gemini Authentication** - Updated to use latest Google AI library
  - Replace deprecated `google-generativeai` with `google-genai` library
  - Update GeminiModel to use new `client.models.generate_content` API
  - Add provider-specific model instantiation in ensemble execution
  - Fix type safety for response.text handling
  - Update tests to match new API structure
  - Default to `gemini-2.5-flash` model for better performance
  - Resolves authentication failures with Google Gemini API integration

## [0.5.0] - 2025-07-16

### Added
- **Enhanced Model Profiles** - Complete agent configuration management
  - Added `system_prompt` and `timeout_seconds` fields to model profiles for complete agent configuration
  - Profiles now support complete agent defaults: model, provider, prompts, timeouts, and costs
  - Reduced configuration duplication across ensembles with centralized profile management
  - Backward compatibility maintained - explicit agent configs still override profile defaults

- **Visual Configuration Status Checking** - Real-time configuration health monitoring
  - New `llm-orc config check` command with unified status display and accessibility legend
  - New `llm-orc config check-global` command for global configuration status only
  - New `llm-orc config check-local` command for local project configuration status only
  - Visual availability indicators: 🟢 Ready to use, 🟥 Needs setup
  - Real-time provider availability detection for authenticated providers and Ollama service
  - Ensemble availability checking with dependency analysis against available providers

- **Configuration Reset Commands** - Safe configuration management with data protection
  - New `llm-orc config reset-global` command for global configuration reset
  - New `llm-orc config reset-local` command for local project configuration reset
  - Automatic backup creation with timestamped `.bak` files (default: enabled)
  - Authentication retention during reset operations (default: enabled)
  - Confirmation prompts and `--force` option for safe operation

- **Enhanced Templates and Examples**
  - Updated all validation ensembles to use proper model profiles
  - New optimized example ensembles demonstrating enhanced profile usage
  - Template consistency improvements across global and local configurations
  - Comprehensive specialist profiles for code review, startup advisory, and research scenarios

### Changed
- **Improved Fallback Model Logic** - Enhanced reliability for ensemble execution
  - Fallback models now prioritize free local models for testing reliability
  - Enhanced logging when fallback models are used
  - Error handling improvements when fallback models are unavailable
  - Simplified default model configuration to single "test" fallback for clarity

- **Configuration Display Enhancements** - Better user experience and accessibility
  - Provider availability shown with consistent emoji-based visual hierarchy
  - Default model profiles section with complete resolution chain display
  - Ensemble availability indicators with dependency analysis
  - Consistent formatting across all configuration sections with improved ordering

### Deprecated
- **`llm-orc config show`** - Replaced by comprehensive `config check` commands
  - Functionality preserved but command deprecated in favor of enhanced alternatives
  - Users should migrate to `config check`, `config check-global`, or `config check-local`

### Fixed
- **Model Profile Resolution** - Improved profile loading and error handling
  - Fixed fallback model logic to avoid mock object issues in test environments
  - Enhanced type checking for profile resolution to prevent runtime errors
  - Improved error messages for missing profiles and provider availability

- **Code Quality and Compliance** - Complete linting and formatting compliance
  - Fixed all MyPy type annotation issues for enhanced model profiles
  - Resolved Ruff formatting violations and line length compliance
  - Enhanced test coverage with 13 new comprehensive tests

### Performance
- **Provider Availability Detection** - Efficient real-time status checking
  - Optimized authentication provider detection with error handling
  - Fast Ollama service availability checking with timeout controls
  - Efficient ensemble dependency analysis for large configuration sets

### Security
- **Authentication Preservation** - Safe configuration reset with credential protection
  - Reset commands preserve API keys and OAuth tokens by default
  - Selective authentication retention with `--retain-auth` flag
  - Backup creation before reset operations for data recovery

## [0.4.3] - 2025-07-15

### Changed
- **Template-based Configuration** - Refactored configuration system to use template files
  - Replaced hardcoded default ensembles with template-based approach for better maintainability
  - Added `src/llm_orc/templates/` directory with configurable templates
  - Updated model naming from "fast/production" to "test/quality" for better clarity
  - Enhanced `init_local_config()` to use templates with project name substitution

### Fixed
- **CLI Profile Listing** - Fixed AttributeError in `llm-orc list-profiles` command
  - Added defensive error handling for malformed YAML configurations
  - Improved error messages when profile format is invalid
  - Better handling of legacy config formats

### Performance
- **Test Suite Optimization** - Improved test performance by 25% (11.86s → 8.92s)
  - Fixed synthesis model mocking in ensemble execution tests (140x faster)
  - Reduced script agent timeouts in integration tests
  - Added timeout configurations to prevent slow API calls during testing

## [0.4.2] - 2025-07-15

### Fixed
- **Security Vulnerability** - Updated aiohttp dependency to >=3.12.14 to address GHSA-9548-qrrj-x5pj
- **Authentication System** - Fixed lookup logic in ensemble execution model loading
  - Corrected authentication provider lookup to use model_name as fallback when provider not specified
  - Fixed 4 failing authentication tests by improving lookup_key handling in _load_model method
  - Enhanced OAuth model creation for anthropic-claude-pro-max provider

### Changed
- **CLI Commands** - Simplified OAuth UX by removing redundant commands (issue #35)
  - Removed `llm-orc auth test` command (functionality integrated into auth list --interactive)
  - Removed `llm-orc auth oauth` command (functionality moved to auth add)
  - Removed `llm-orc config migrate` command (automatic migration already handles this)
  - Streamlined authentication workflow with fewer, more focused commands

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