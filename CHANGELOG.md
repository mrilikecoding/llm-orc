# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.15.10] - 2026-02-25

### Added
- **Ollama options pass-through** — profiles and agents can now specify an `options` dict with provider-specific parameters (`num_ctx`, `top_k`, `top_p`, `repeat_penalty`, `seed`, etc.) that are forwarded to the Ollama API; profile and agent options are deep-merged with agent keys winning on conflict; explicit `temperature` and `max_tokens` fields always take precedence

## [0.15.9] - 2026-02-25

### Fixed
- **Profile registry unification** — `ConfigurationManager.get_model_profiles()` now scans individual YAML files in `profiles/` directories, matching what `ProfileHandler` already found; profiles created via `create_profile` are now visible at runtime
- **Silent ensemble swallowing** — `EnsembleLoader` now logs warnings for invalid ensemble YAML instead of silently skipping them
- **timeout_seconds on all agent types** — moved `timeout_seconds` from `LlmAgentConfig` to `BaseAgentConfig` so script and ensemble agents accept it (previously rejected by `extra="forbid"`)
- **Newline mangling in script agent input** — added `INPUT_TEXT` env var that preserves real newlines for text-processing scripts; `INPUT_DATA` retains the full JSON contract for schema-aware scripts

### Added
- **`--input-file` / `-f` for invoke** — CLI and MCP invoke now accept a file path whose contents become the input data; priority: positional > option > file > stdin > default

## [0.15.8] - 2026-02-24

### Refactored
- **Execution sub-packages (U3)** — reorganized flat `core/execution/` (32 files) into 5 sub-packages: `fan_out/`, `scripting/`, `runners/`, `phases/`, `monitoring/`; 21 files moved, ~170 import statements rewritten, zero cross-sub-package imports
- **OrchestraService as composition root (M1)** — CLI commands now route through `OrchestraService` instead of constructing handlers directly; both MCP and CLI share the same application service
- **ExecutorFactory required collaborators (M2)** — `EnsembleExecutor` construction pushed into factory; child executors share immutable infrastructure while isolating mutable state
- **Narrow exception handling (M4)** — replaced broad `except Exception` clauses with specific exception types across execution pipeline
- **HTTPConnectionPool.configure() (M6)** — classmethod replaces module-level mutation for connection pool configuration
- **TemplateProvider protocol (E3)** — `ConfigurationManager` now receives an injected `TemplateProvider` instead of importing template logic directly
- **ConfigurationManager split (E2/U1)** — separated construction from provision; lighter initialization path
- **Handler relocation (E1)** — moved handler files from `mcp/handlers/` to `services/handlers/`
- **set_project concurrency (E8)** — added `asyncio.Lock` to prevent concurrent `handle_set_project` calls
- **Typed agent configs (U5)** — replaced `_current_agent_configs` temporary field with typed dataclass
- **CLI test routing (U15)** — remaining CLI tests now exercise `OrchestraService` instead of bypassing it

### Removed
- **Typed event system (M3)** — deleted unused Pydantic event hierarchy (`core/events/`)
- **Completed audit documentation** — removed `codebase-audit.md` and remediation plan (all 36 findings resolved)

### Fixed
- **Intermittent test failures** — resolved timing-sensitive tests and suppressed spurious warnings
- **`invoke_streaming` wiring (E4)** — connected streaming invocation to real execution engine

### Tests
- Reduced assertion roulette (U10): removed 3 redundant `assert X is not None` tests that duplicated DI contract test
- Removed eager test (U11): deleted `test_caching.py` with inline `ResultCache` class that had no production counterpart
- Added BDD feature files (U12) for ADRs 010–014

### Docs
- Updated `architecture.md`, `script-agent-architecture.md`, and ADRs 002, 006, 007 to reflect execution sub-package paths

## [0.15.7] - 2026-02-22

### Fixed
- **Profile validation in web UI** — `ProfileHandler.get_all_profiles()` now seeds from `ConfigurationManager.get_model_profiles()` before scanning `profiles/` directories; profiles defined only in `config.yaml` (e.g. `validate-ollama`) were previously invisible to the UI and treated as unvalidated
- **Spurious `RuntimeError` on plain-text agent output** — `AgentRequestProcessor.process_script_output_with_requests()` now returns gracefully with empty `agent_requests` when the response is not JSON; callers also guard the call to script agents only, preventing the error from appearing for LLM agent responses
- **Ensemble runs reported as "Failed" in web UI** — `ExecutionHandler.invoke()` now maps internal status `"completed"` → `"success"` and `"completed_with_errors"` → `"error"` at the API boundary; the frontend was receiving `"completed"` and treating it as failure

## [0.15.6] - 2026-02-22

### Refactored
- **ProjectContext value object** — frozen dataclass groups `project_path` and `ConfigurationManager` for atomic handler propagation on `set_project`
- **OrchestraService extraction** — shared application service replaces MCPServer as the composition root; both MCP and web ports are now thin protocol adapters
- **Typed result models** — `AgentResult`, `ExecutionMetadata`, and `ExecutionResult` dataclasses replace `dict[str, Any]` in the execution pipeline
- **ExecutorFactory** — centralizes `EnsembleExecutor` construction; child executors share immutable infrastructure while isolating mutable state

### Docs
- Removed completed `codebase-audit.md` (all 8 stewardship items and 18 findings resolved)
- Updated `architecture.md`, `coding-standards.md`, and ADRs 003, 004, 009, 010 to reflect structural changes

## [0.15.5] - 2026-02-21

### Fixed
- **`set_project` config propagation gaps** — `_execution_handler` and `_promotion_handler` now receive the updated `ConfigurationManager` when `set_project` is called; previously both held a stale pre-`set_project` reference

### Refactored
- **Dead code deletion** — removed the `communication/protocol.py` cluster (`Message`, `ConversationManager`, `MessageProtocol`) and the `contracts/`+`reference/` packages (`ScriptContract`, `ContractValidator`, `FileWriteScript`, `JsonExtractScript`) — 5,323 lines across 20 files; no production code referenced these clusters

### Tests
- Reduced assertion roulette in ensemble execution tests: split two multi-concern fallback tests into focused pairs and added diagnostic messages to 20 assertions
- Added route-level unit tests for `web/api/scripts`, `web/api/artifacts`, and `web/api/profiles` update endpoint; coverage restored to 90.3%

## [0.15.4] - 2026-02-21

### Fixed
- **Library browse/search silent failures** — `_browse_ensembles` and `_search_ensembles` now log at DEBUG with full traceback when a YAML file fails to parse, rather than silently skipping it
- **Library scripts browse depth** — `library_browse`, `library_search`, and `library_info` now recurse fully into nested script directories (e.g. `scripts/specialized/network-science/`) instead of stopping one level short; category is the immediate parent directory name
- **README `library_copy` example** — source path corrected from `"code-analysis/security-review"` to `"ensembles/code-analysis/security-review"` (path is relative to library root)
- **`asyncio.get_event_loop()` deprecation** — replaced with `get_running_loop()` in `OAuthClaudeModel` and `ClaudeCLIModel`

### Changed
- **`OAuthClaudeModel.SUPPORTS_CUSTOM_ROLE_PROMPT = False`** — class-level attribute surfaces the Liskov-breaking behaviour; `ModelInterface` base now declares `SUPPORTS_CUSTOM_ROLE_PROMPT: bool = True` as the default; a `WARNING` is logged when a custom role is injected via conversation turn
- **README: MCP library tools** — added note that tools are local-filesystem-only and require `LLM_ORC_LIBRARY_PATH` for Homebrew/pip installs; added Contributing to the Library section

## [0.15.3] - 2026-02-21

### Fixed
- **`eval()` sandbox security** — execution context now passed as `locals` argument rather than merged into `restricted_globals`; prevents LLM-produced context keys (e.g. `__builtins__`, `len`) from shadowing or escaping the sandbox
- **Version fossil** — `__version__` now resolved at runtime via `importlib.metadata.version("llm-orchestra")` instead of the stale hard-coded string `"0.3.0"`
- **Silent agent erasure** — `asyncio.gather` exceptions in parallel phase execution now produce a synthetic `status: "failed"` record instead of silently dropping the agent from results
- **Silent failures logged** — seven previously silent `except Exception: pass` sites (streaming merger, artifact saving, agent request processing, model fallback loading, progress-controller UI calls) now log at `WARNING` or `DEBUG` with full traceback context
- **Deprecated `asyncio.get_event_loop()`** — replaced with `get_running_loop()` in `OAuthClaudeModel` and `ClaudeCLIModel`; `get_event_loop()` is deprecated in Python 3.10+ and unnecessary inside async context

### Changed
- **`AgentExecutor` renamed to `AgentResourceMonitor`** — class name and file now reflect actual responsibility (resource monitoring and metrics); execution lives in `LlmAgentRunner` / `ScriptAgentRunner`
- **`OAuthClaudeModel.SUPPORTS_CUSTOM_ROLE_PROMPT = False`** — class-level attribute surfaces the Liskov-breaking behaviour; `ModelInterface` base now declares `SUPPORTS_CUSTOM_ROLE_PROMPT: bool = True` as the default; a `WARNING` is logged when a custom role is injected via conversation turn
- **BDD contracts test** — `test_adr_003_testable_contracts.py` now imports `ScriptContract`, `ScriptMetadata`, `TestCase`, `ScriptCapability`, `ScriptDependency` from `llm_orc.contracts` instead of defining local shadow classes
- **Documentation corrected** — `architecture.md` marks WebSocket/batch/registration system as not yet implemented; `coding-standards.md` coverage threshold aligned to 90% (matching `pyproject.toml`); ADR-010 status changed from Proposed to Implemented

### Removed
- **Dead code constellation** — deleted orphaned Pydantic event hierarchy (`core/events/base.py`, `script_interaction.py`, `script_schemas.py`), hollow `testing/contract_validator.py` stub and CLI wrapper, and five associated test files

### Security
- Added `TestEvalSandboxSecurity` test class with five adversarial tests: `__builtins__` key injection, `__import__` unavailability, builtin shadowing via context keys, and a passing legitimate-context test

## [0.15.2] - 2026-02-21

### Fixed
- **MCP `set_project` handler propagation** — `create_ensemble`, `validate_ensemble`, and `check_ensemble_runnable` now correctly use the project directory after `set_project` is called; previously all handlers held a stale reference to the pre-`set_project` `ConfigurationManager`
- **`validate_ensemble` profile resolution** — profiles stored as individual YAML files in the profiles directories are now found by validation; previously only profiles declared under `model_profiles:` in a `config.yaml` were recognized

### Refactored
- Reduced cognitive complexity in `PromotionHandler._get_profile_tier` and `list_dependencies` (extracted `_profile_in_dir`, `_build_agent_dep_info`, `_collect_dep_sets`)
- Reduced cognitive complexity in `DependencyResolver.enhance_input_with_dependencies` (extracted `_compute_agent_input`)

## [0.15.1] - 2026-02-20

### Added
- **Ensemble Promotion Tools** — 4 new MCP tools for managing ensembles across tiers
  - `promote_ensemble` — copy ensemble + profile dependencies from local to global or library tier (with dry-run, overwrite, include_profiles options)
  - `list_dependencies` — inspect all profiles, models, and providers an ensemble requires
  - `check_promotion_readiness` — assess whether an ensemble can be promoted and what's missing
  - `demote_ensemble` — remove an ensemble from a higher tier with optional orphaned profile cleanup

### Fixed
- **Ollama detection in MCP subprocess** — `get_provider_status` now reports the actual error instead of silently returning "Ollama not running" when the check fails in the MCP server subprocess context
- **Ollama host fallback** — added `OLLAMA_HOST` env var support and automatic localhost→127.0.0.1 fallback for environments where DNS resolution differs

## [0.15.0] - 2026-02-20

### Added
- **Composable Ensemble Orchestration** — ensembles can now reference other ensembles, select specific keys from upstream output, and compose with fan-out for routing patterns
- **Ensemble Agent Type (ADR-013)** — agents can reference another ensemble via `ensemble: child-name`
  - Child ensembles execute recursively with shared infrastructure (config, credentials, models)
  - Mutable state (usage, events, streaming) is isolated per child execution
  - Configurable depth limit prevents unbounded nesting (default: 5)
  - Cross-ensemble cycle detection at load time via DFS on reference graph
  - Child executors suppress artifact saving to avoid duplicate persistence
- **Input Key Routing (ADR-014)** — agents can select a specific key from upstream JSON output via `input_key: key_name`
  - `DependencyResolver` selects `output[input_key]` from the first upstream dependency
  - Missing key or non-dict upstream produces a runtime error (per Invariant 14)
  - Composes with fan-out: `input_key` selects the array, `fan_out` expands per item
  - Works with all agent types (LLM, script, ensemble)
- **Pydantic Agent Configs (ADR-012)** — typed, validated agent configuration models
  - `LlmAgentConfig`, `ScriptAgentConfig`, `EnsembleAgentConfig` with discriminated union
  - `extra="forbid"` catches typos and invalid fields at parse time
  - `parse_agent_config()` factory discriminates by key presence (`script`, `ensemble`, `model_profile`)
  - Replaces `dict[str, Any]` throughout the execution pipeline
- **Routing Demo Ensemble** — example ensemble demonstrating the classify → route → fan-out → synthesize pattern

### Changed
- Removed conversational ensemble dead code (ADR-011): `ConversationalEnsembleManager`, `ConversationalDependencyResolver`, `ConversationalConfig`, and related modules
- Agent type routing in `EnsembleExecutor._execute_agent` now uses `isinstance` dispatch on Pydantic models instead of dict key inspection

## [0.14.4] - 2026-02-19

### Added
- Model profiles and agent configs now support `temperature` and `max_tokens` generation parameters
  - Profile YAML sets defaults; agent config in ensemble YAML can override per-agent
  - Supported across all providers: Ollama, Anthropic, Google Gemini
  - MCP `create_profile` tool accepts the new parameters

### Changed
- Merged `EnhancedScriptAgent` into `ScriptAgent`, removing the re-export shim
- Dissolved `AuthCommands`, `ConfigCommands`, and `ResultsProcessor` classes into plain module-level functions
- Extracted `FanOutCoordinator`, `PhaseMonitor`, `PhaseResultProcessor`, `AgentDispatcher` from `EnsembleExecutor`
- Extracted `EnsembleCrudHandler`, `ExecutionHandler`, `ResourceHandler` from `MCPServer`
- Unified `execute()` and `execute_with_user_input()` into `_execute_core`
- Promoted `_run_validation` and `_classify_failure_type` to module-level functions
- Removed 9 delegation stubs from `EnsembleExecutor`
- Removed dead code: `RoleManager`, `ModelManager`, `ConversationalDependencyResolver`, visualization subsystem, and other unused modules
- Removed duplicate `calculate_usage_summary` from `UsageCollector`
- Marked 3 self-free methods as `@staticmethod` in dependency resolver/analyzer
- Eliminated 11 thin delegation wrappers from `MCPServer`
- Consolidated shared helpers across MCP and web API modules

### Fixed
- Use bare `raise` instead of `raise e` in anthropic.py
- Pass `model_name` to `ClaudeModel` in `_create_api_key_model`

## [0.14.3] - 2026-02-18

### Changed
- Removed dead code: `RoleManager`, `ModelManager`, `provider_utils`, `visualization/integration`, unused methods from `cli_utils`, `agent_request_processor`, `artifact_manager`, `visualization/stream`
- Consolidated `get_mcp_server()` singleton across web API modules
- Consolidated `_get_agent_attr()` helper across MCP handler modules
- Extracted `_find_profile_file()` in profile handler to remove duplication
- Deleted 30 pass-only BDD scenarios with no test value
- Moved 5 integration tests from `tests/unit/` to `tests/integration/`
- Added shared `TestClient` fixture for web API tests
- Added `.complexipy_cache/` to `.gitignore`

## [0.14.2] - 2026-02-16

### Added
- Python 3.13 support (tested in CI on Ubuntu and macOS)
- Pinned pydantic>=2.12 for forward compatibility

### Changed
- README license text updated to AGPL-3.0

## [0.14.1] - 2026-02-16

### Fixed
- `create_ensemble` from template now preserves all agent fields (`type`, `script`, `parameters`, `cache`, `fan_out`) instead of only `name`, `model_profile`, and `depends_on`
- `validate_ensemble` now detects script agents by the `script` field, not just explicit `type: script`
- Script paths resolve relative to project directory (from `set_project`) instead of working directory, fixing MCP server usage from consuming projects
- Plexus ensemble uses relative script paths instead of absolute paths

### Added
- `llm_orc.script_utils` module with shared `unwrap_input()` for script envelope handling
  - Handles ScriptAgentInput, legacy wrapper, and direct input formats
  - Reads `AGENT_PARAMETERS` env var as fallback for parameters
  - Optional `debug=True` mode logs envelope diagnostics to stderr
- Fan-out feature documented in README and architecture docs

## [0.14.0] - 2026-02-12

### Changed
- **License changed from MIT to AGPL-3.0-or-later**
- Replaced Tailwind CSS with Pico CSS for web UI
- Collapsible sections for ensemble source and directory groups in web UI

### Added
- Runnable check endpoint and relative_path field for ensembles

### Security
- Upgraded all dependencies to fix 14 known vulnerabilities
  - aiohttp 3.12.14 → 3.13.3 (GHSA-6mq8-rvhq-8wgg)
  - cryptography 45.0.5 → 46.0.5
  - urllib3 2.6.2 → 2.6.3 (GHSA-38jv-5279-wg99)
  - python-multipart 0.0.20 → 0.0.22 (GHSA-wp53-j4wj-2cfg)
  - filelock 3.20.1 → 3.20.3, pip 25.3 → 26.0, pyasn1 0.6.1 → 0.6.2

### Fixed
- Reduced cognitive complexity in 9 functions to pass stricter linter thresholds
- Migrated `str, Enum` to `StrEnum` for ruff 0.15 compatibility
- Fixed mypy 1.19 type annotation issue in contract validator

## [0.13.0] - 2025-12-19

### Added
- **Web UI for Ensemble Management (ADR-010, Issue #74)**
  - Local web interface at `llm-orc web [--port 8765] [--host 127.0.0.1] [--open]`
  - FastAPI server with REST API endpoints for ensembles, profiles, scripts, and artifacts
  - Vite + Preact frontend with Tailwind CSS dark theme
  - Slide-out panel design for detail views and forms
  - Profile CRUD with modal forms
  - Script browser with category grouping and test runner
  - Artifact viewer with execution metrics and results
  - Eddi-lab purple gradient theme with tab navigation

- **Fan-Out Agent Pattern (Issue #73)** - Map-reduce style parallel chunk processing
  - Agents with `fan_out: true` automatically expand into N parallel instances
  - Detects JSON arrays and `{"data": [...]}` script outputs from upstream agents
  - Instance naming: `processor[0]`, `processor[1]`, `processor[2]`, etc.
  - Results gathered under original agent name with per-instance status tracking
  - Partial success support: continues with available results on instance failures
  - Config validation: `fan_out: true` requires `depends_on` field

- **Fan-Out Modules**
  - `FanOutExpander` - Detects arrays and expands agents into indexed instances
  - `FanOutGatherer` - Collects and orders instance results with status tracking

- **Fan-Out Integration Points**
  - Phase execution with automatic fan-out detection and expansion
  - Chunk-indexed input preparation in DependencyResolver
  - Instance name normalization in DependencyAnalyzer
  - Fan-out stats aggregation in ResultsProcessor
  - Artifact markdown reports with fan-out execution summaries

- **Test Ensemble**
  - `fan-out-test` ensemble with chunker + processor scripts for validation

### Security
- Upgraded filelock 3.18.0 → 3.20.1 (GHSA-w853-jp5j-5j7f TOCTOU race condition)
- Upgraded urllib3 2.5.0 → 2.6.2 (GHSA-gm62-xv2j-4w53, GHSA-2xpw-w6gg-jr37)

## [0.12.3] - 2025-12-04

### Fixed
- Script agents now receive upstream dependency results via JSON stdin
  - DependencyResolver builds proper ScriptAgentInput JSON for script agents
  - Inline scripts receive dependencies via stdin (not just environment variables)
  - File-based scripts receive dependencies via both stdin and environment variables
  - Fixes mixed script→LLM→script dependency chains in ensembles

## [0.12.2] - 2025-12-04

### Fixed
- MCP server graceful shutdown on Ctrl+C (no more threading exceptions)
- Clearer MCP server output: minimal for stdio, detailed for HTTP transport

### Changed
- HTTP transport now shows endpoint URLs for web UI integration
- Reduced Homebrew package size from 99MB to ~58MB

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