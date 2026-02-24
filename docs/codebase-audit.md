# Codebase Audit: LLM Orchestra (llm-orc)

**Date:** 2026-02-22
**Scope:** Whole codebase
**Coverage:** This analysis sampled strategically across ~121 source files (~24K lines) and ~138 test files (~53K lines). It is representative, not exhaustive. Entry points, configuration, execution engine, MCP handlers, model providers, CLI, web API, and test infrastructure were sampled. Visualization utilities and primitive scripts received lighter coverage.

## Remediation Status

Tracking resolution of findings. See `docs/plans/2026-02-23-codebase-audit-remediation.md` for the full plan.

| Finding | Description | Status | Date |
|---------|-------------|--------|------|
| E5 | Rename `adaptive_resource_manager.py` → `system_resource_monitor.py` | Resolved | 2026-02-23 |
| E6 | Remove duplicate role injection from `DependencyResolver` | Resolved | 2026-02-23 |
| E11 | Remove broken Makefile targets | Resolved | 2026-02-23 |
| E12 | Update ADR-004 status to "Superseded" | Resolved | 2026-02-23 |
| E13 | Fix reversed principle in ADR README | Resolved | 2026-02-23 |
| E14 | Update ADR 011-014 statuses to "Implemented" | Resolved | 2026-02-23 |
| E15 | Remove dead dependencies (websockets, aiohttp) | Resolved | 2026-02-23 |
| E16 | Fix README library default claim | Resolved | 2026-02-23 |
| U6 | Delete ghost directories | Resolved | 2026-02-23 |
| U7 | Remove dead `SUPPORTS_CUSTOM_ROLE_PROMPT` flag | Resolved | 2026-02-23 |
| U13 | Clean up pytest markers | Resolved | 2026-02-23 |
| E7 | Distinguish credential decryption errors from missing credentials | Resolved | 2026-02-23 |
| E9 | Emit warning when fan-out produces zero instances | Resolved | 2026-02-23 |
| E10 | Add `set_project_context` to ScriptHandler and ArtifactHandler | Resolved | 2026-02-23 |
| M5 | Align complexity thresholds | Resolved | 2026-02-23 |
| M7 | Register global exception handler on FastAPI app | Resolved | 2026-02-23 |
| U2 | Extract shared `classify_tier()` + consolidate `_dep_name` | Resolved | 2026-02-23 |
| U4 | Extract profile parsing from PromotionHandler to ProfileHandler | Resolved | 2026-02-23 |
| U8 | Remove dead EventFactory code from ScriptUserInputHandler | Resolved | 2026-02-23 |
| U9 | Replace `_test_*` escape hatches with constructor injection | Resolved | 2026-02-23 |
| U14 | Make `LLMResponseGenerator` accept `ModelInterface` | Resolved | 2026-02-23 |
| E8 | Add `asyncio.Lock` to `handle_set_project` | Resolved | 2026-02-23 |
| M3 | Delete unused typed event system | Resolved | 2026-02-23 |
| U5 | Replace `_current_agent_configs` temporary field | Resolved | 2026-02-23 |
| M4 | Narrow `except Exception` to specific types | Resolved | 2026-02-23 |
| U12 | Add BDD feature files for ADRs 010-014 | Resolved | 2026-02-23 |
| M2 | Push EnsembleExecutor construction into ExecutorFactory | Resolved | 2026-02-23 |
| M1 | Route CLI through OrchestraService | Resolved | 2026-02-23 |
| E1 | Move handlers from `mcp/handlers/` to `services/handlers/` | Resolved | 2026-02-23 |
| U15 | Rewrite CLI tests to mock OrchestraService at boundary | Resolved | 2026-02-23 |
| E2/U1 | Split ConfigurationManager construction from provisioning | Resolved | 2026-02-23 |
| E4 | Connect `invoke_streaming` to real execution | Resolved | 2026-02-23 |
| E3 | Define TemplateProvider protocol, inject into ConfigurationManager | Resolved | 2026-02-23 |
| M6 | Add `HTTPConnectionPool.configure()`, remove hidden ConfigurationManager | Resolved | 2026-02-23 |
| U3 | Reorganize flat `execution/` into sub-packages | Resolved | 2026-02-24 |
| U10/U11 | Improve test quality (assertion roulette, eager tests) | Resolved | 2026-02-24 |

## Executive Summary

LLM Orchestra is a multi-agent LLM orchestration system that coordinates ensembles of AI agents through dependency-based phase execution. The architecture follows a hexagonal (ports and adapters) pattern with `OrchestraService` as the application core. All three ports (MCP, Web, CLI) now delegate through the service layer.

**Post-remediation (2026-02-23):** Three waves of remediation resolved 34 of 36 findings. The typed event system was deleted (M3). The CLI now routes through `OrchestraService` (M1). Handlers moved from `mcp/handlers/` to `services/handlers/` (E1). `EnsembleExecutor` construction is centralized in `ExecutorFactory` with required collaborator injection (M2). `ConfigurationManager` separates construction from provisioning (E2/U1) and uses an injected `TemplateProvider` protocol instead of upward imports (E3). `HTTPConnectionPool` accepts explicit configuration (M6). `invoke_streaming` connects to real execution (E4). `set_project` is synchronized with `asyncio.Lock` (E8). Broad `except Exception` clauses were narrowed to specific types (M4). ADR statuses, markers, and documentation were corrected (E12-E16, U13). Dead code, dependencies, and ghost directories were removed (M3, E15, U6-U8).

**All 37 findings resolved.** U3 (reorganize `execution/` into sub-packages) was resolved on 2026-02-24, splitting 21 files into 5 sub-packages. U10/U11 (test quality) resolved on 2026-02-24 — removed assertion roulette init tests and deleted the eager `test_caching.py` (inline class with no production counterpart, behaviors already covered by `test_script_cache.py`).

The codebase's strongest assets are its typed agent configuration layer (Pydantic discriminated union with `extra="forbid"`), its multi-tier configuration hierarchy, and its phase-based dependency resolution with parallel execution.

## Architectural Profile

### Patterns Identified

| Pattern | Confidence | Evidence |
|---------|-----------|----------|
| Hexagonal Architecture | High | `OrchestraService` as hub; MCP, Web, and CLI all delegate through service layer |
| God Object Decomposition (complete) | High | `EnsembleExecutor` collaborators injected via `ExecutorFactory`; construction centralized |
| Strangler Fig / Incomplete Type Migration | High | `AgentResult` dataclass coexists with raw `dict[str, Any]`; `isinstance` guards at 5+ sites |
| Ambient Context | High | `ConfigurationManager` discovers config via `Path.cwd()` at construction time |
| Fan-Out / Map-Reduce | High | `FanOutExpander` + `FanOutGatherer` + `FanOutCoordinator` for parallel array processing |
| Pydantic Discriminated Union | High | `LlmAgentConfig | ScriptAgentConfig | EnsembleAgentConfig` with `extra="forbid"` |
| Global Mutable Singleton | Medium | `OrchestraService` mutated by `set_project` (now synchronized with `asyncio.Lock`) |

### Quality Attribute Fitness

| Attribute | Assessment |
|-----------|-----------|
| **Modifiability** | Strong. Pydantic configs catch errors early. `_dep_name` and tier classification consolidated (U2). All three ports route through `OrchestraService`. Handlers co-located with service layer (E1). |
| **Testability** | Strong. 90% coverage floor enforced, 2.2:1 test-to-source ratio. `_test_*` escape hatches replaced with constructor injection (U9). `ConfigurationManager` supports `provision=False` for lightweight test construction (E2/U1). `HTTPConnectionPool.configure()` enables explicit test setup (M6). CLI tests mock at service boundary (U15). |
| **Operability** | Improved. `except Exception` clauses narrowed to specific types (M4). Credential errors now distinguishable (E7). Fan-out zero-instance emits warning (E9). Global exception handler on web server (M7). |
| **Correctness** | Strong. `extra="forbid"` catches agent config typos. Fan-out emits warnings on zero instances (E9). `invoke_streaming` executes real agents (E4). `set_project` synchronized with `asyncio.Lock` (E8). |
| **Performance** | Strong. Async parallel execution, `HTTPConnectionPool` for connection reuse with explicit configuration (M6), phase-based dependency resolution enables maximum parallelism. |
| **Security** | Good. Encrypted credential storage, TLS for external calls. Credential decryption failures now distinguishable from "no credentials" (E7). Web server has global exception handler preventing traceback leakage (M7). |

### Inferred Decisions

**ID-1: BDD as Behavioral Tests, Not Architectural Enforcement** (High confidence)
ADR-004 envisioned BDD as architectural guardrails with `ADRComplianceValidator`. In practice, BDD became acceptance tests in Gherkin syntax. The compliance validator was never built. This is a better outcome than the ADR designed.

**ID-2: Primitives Thesis Reversed** (High confidence)
ADR-006 declared "primitives are content, not infrastructure." Practice revealed that core primitives receive engine-level fixes. The amendment (2026-02-18) moved 6 primitives into `src/llm_orc/primitives/`. The `priority 1.5` in `script_resolver.py` is a visible scar from inserting a new search path.

**ID-3: Dead Code from Three ADRs Lived Alongside Real Code** (High confidence)
ADR-003 (contract validation), ADR-005 (conversational ensembles), and parts of ADR-004 (hooks/agents) were implemented, marked complete, then deleted. The deletions left ghost directories, broken Makefile targets, and stale ADR status entries.

**ID-4: MCP Server Was the Canonical Interface** (High confidence)
Handlers were built inside `mcp/handlers/`. When the web UI needed the same logic, it called into `MCPServer` private methods. The `OrchestraService` extraction was a recent correction. *Post-remediation:* Handlers relocated to `services/handlers/` (E1), aligning module location with responsibility.

**ID-5: AGPL for a Local Tool** (Medium confidence)
AGPL-3.0 is chosen for a tool designed to run on localhost via stdio/MCP. The "network use" trigger that distinguishes AGPL from GPL does not apply to this deployment model. The rationale is not documented.

**ID-6: Dead Dependencies Accumulated from Deleted Features** (Medium confidence)
`websockets` and `aiohttp` appear to be holdovers from an earlier architecture. Zero imports exist across the source tree.

## Tradeoff Map

| Optimizes For | At the Expense Of | Evidence | Status |
|--------------|-------------------|----------|--------|
| Zero-config startup | Testability, determinism | `ConfigurationManager` walks `cwd` and writes to disk in constructor | **Resolved** (E2/U1): `provision=False` for lightweight use |
| Rapid feature addition | Type safety, event contract clarity | Two parallel event systems (typed `EventFactory` vs. untyped dict streams) | **Resolved** (M3): Typed event system deleted; dict streams canonical |
| CLI development velocity | Consistency between ports | CLI bypasses `OrchestraService`, duplicates ensemble lookup logic | **Resolved** (M1, U15): CLI routes through service |
| Fault tolerance (broad `except Exception`) | Debuggability, correctness | 100 catch-all clauses; credential decryption failure indistinguishable from "no credentials" | **Resolved** (M4, E7): Narrowed to specific types; credential errors distinguishable |
| Module independence | Single source of truth | `_dep_name` duplicated 4x; tier classification duplicated 5x | **Resolved** (U2): Consolidated into shared utilities |
| Connection reuse performance | Configuration dynamism, test isolation | `HTTPConnectionPool` singleton initialized once, never reset by `set_project` | **Resolved** (M6): Explicit `configure()` + reset on project switch |
| Self-contained handlers | Layering integrity | `OrchestraService` imports 11 concrete handlers from `mcp/handlers/` (dependency inversion) | **Resolved** (E1): Handlers moved to `services/handlers/` |
| Interface completeness | Truthfulness | `invoke_streaming` yields fabricated success events without running agents | **Resolved** (E4): Connected to real execution engine |
| Sequential MCP simplicity | Concurrency safety | `set_project` mutates 7 handlers without locking on a shared singleton | **Resolved** (E8): `asyncio.Lock` on `handle_set_project` |

## Findings

### Macro Level

#### M1: Hexagonal Architecture Partially Realized --- CLI Pierces the Hexagon

**Status: Resolved (2026-02-23)**

**Observation:** `OrchestraService` acts as the hexagonal core. MCP and Web are proper thin adapters. The CLI constructs `EnsembleExecutor` directly, bypassing the service entirely.
- `src/llm_orc/services/orchestra_service.py:L33-38` --- docstring declares "Both MCP and web ports delegate to this service"
- `src/llm_orc/cli_commands.py:L318` --- `executor = EnsembleExecutor()` constructed directly
- `src/llm_orc/cli_commands.py:L66-78` --- `_find_ensemble_config` reimplements logic that exists in `OrchestraService`

**Pattern:** Port Bypass. The CLI is an adapter that pierces the hexagon to talk to the domain directly.

**Tradeoff:** Optimizes for CLI delivery speed at the expense of consistency --- changes to the execution pipeline have three call sites to maintain, not one.

**Question:** What happens when the execution invocation signature changes, given that the CLI constructs its own executor while MCP/Web delegate through `OrchestraService`?

**Stewardship:** Route the CLI `invoke` command through `OrchestraService`. The CLI should reduce to: parse args, call service method, format output.

---

#### M2: EnsembleExecutor as an Accumulating Constructor

**Status: Resolved (2026-02-23)**

**Observation:** `EnsembleExecutor.__init__` spans 112 lines (L173-285) and wires 18 collaborators, including lambdas and inner functions defined inside the constructor.
- `src/llm_orc/core/execution/ensemble_execution.py:L171-285` --- constructor body
- `src/llm_orc/core/execution/ensemble_execution.py:L259-274` --- lambdas and `agent_executor_wrapper` defined inside `__init__`
- `src/llm_orc/core/execution/executor_factory.py:L13,39,74` --- three deferred imports to break circular dependency with `EnsembleExecutor`

**Pattern:** Telescoping Constructor / God Object decomposition in progress. The `ExecutorFactory` extraction was a step forward but the constructor remains the central assembly point.

**Tradeoff:** Optimizes for "everything wired in one place" at the expense of constructor comprehensibility and the circular dependency between executor and factory.

**Question:** If a new execution concern (rate limiting, request coalescing) needed to be added, how many lines would `__init__` grow?

**Stewardship:** Push construction into `ExecutorFactory` entirely. `EnsembleExecutor.__init__` receives completed collaborators and assigns them.

---

#### M3: Two Parallel Event Systems

**Status: Resolved (2026-02-23)**

**Observation:** `visualization/events.py` defines a fully typed event taxonomy (17 event types, `EventFactory` with named constructors). The actual execution pipeline emits anonymous `dict[str, Any]` events through `StreamingProgressTracker`. The typed system is never used during live execution.
- `src/llm_orc/visualization/events.py:L9-68` --- typed `ExecutionEventType` enum and `EventFactory`
- `src/llm_orc/core/execution/streaming_progress_tracker.py:L35-87` --- raw dicts with `"type"`, `"data"` keys
- `src/llm_orc/core/execution/ensemble_execution.py:L349-379` --- `_merge_streaming_events()` with "Phase 5" comment

**Pattern:** Incomplete Migration / Parallel Type Hierarchies. The typed system appears aspirational; the untyped system is live.

**Tradeoff:** Optimizes for incremental feature addition (new event types via string literals) at the expense of type safety and discoverability.

**Question:** If a consumer needed every possible event type, where would they look --- the enum, the string literals, or both?

**Stewardship:** Adopt one system and remove the other. If the typed system is canonical, wire `EventFactory` into the streaming tracker. If the dict system is canonical, delete `EventFactory`.

---

#### M4: Broad `except Exception` Clauses Erode Reliability

**Status: Resolved (2026-02-23)**

**Observation:** 100 `except Exception` clauses across 42 files. Many are in core paths where silent failure changes semantics.
- `src/llm_orc/core/auth/authentication.py:L216-217` --- `_load_credentials` returns `{}` on decryption error
- `src/llm_orc/core/config/config_manager.py:L238-240` --- `load_project_config` returns `{}` on any exception
- `src/llm_orc/core/execution/usage_collector.py:L71,L100,L138` --- three `except Exception: pass` blocks

**Pattern:** Defensive Catch-All as Resilience Proxy. The system keeps running even when monitoring fails, but callers cannot distinguish transient from permanent errors.

**Tradeoff:** Optimizes for happy-path reliability at the expense of debuggability and correctness.

**Question:** What is the user-visible difference between "no API key configured" and "credential decryption failed"?

**Stewardship:** Triage clauses into three categories: optional features (keep broad catch), boundary I/O (narrow to specific exceptions), and control-flow paths (let exceptions propagate).

---

#### M5: Complexity Thresholds Split Between CI and Local

**Status: Resolved (2026-02-23)**

**Observation:** `pyproject.toml` sets McCabe complexity at 10 (ruff C90). `Makefile` runs `complexipy --max-complexity-allowed 15`. CI runs only ruff, not complexipy.
- `pyproject.toml:L77-78` --- `max-complexity = 10`
- `Makefile:L47` --- `complexipy --max-complexity-allowed 15`

**Pattern:** Split Fitness Function. Two tools, two thresholds, only one in CI.

**Tradeoff:** Optimizes for developer flexibility at the expense of consistent enforcement.

**Question:** What is the actual complexity budget, and which metric is authoritative?

**Stewardship:** Add `complexipy` to CI. Align thresholds or document which is primary.

---

#### M6: `HTTPConnectionPool` Singleton Couples Model Performance to Process State

**Status: Resolved (2026-02-23)**

**Observation:** A process-global connection pool is lazily initialized by constructing `ConfigurationManager()` inside `models/base.py`. Configuration changes via `set_project` do not reset the pool.
- `src/llm_orc/models/base.py:L32-35` --- `ConfigurationManager()` constructed inside singleton
- `src/llm_orc/models/base.py:L7-83` --- `__new__` singleton pattern

**Pattern:** Hidden Upward Coupling / Singleton Side Effect. The model layer reaches into config infrastructure through a deferred import.

**Tradeoff:** Optimizes for self-contained HTTP pooling at the expense of transparency and testability.

**Question:** If a test mocks `ConfigurationManager` but the pool was already initialized, which configuration does the client use?

**Stewardship:** Pass connection pool configuration explicitly via constructor injection. Add teardown hooks for the pool in tests and server shutdown.

---

#### M7: Web Server Lacks Error Handling Middleware

**Status: Resolved (2026-02-23)**

**Observation:** The FastAPI app registers only `CORSMiddleware`. No global exception handler, no structured error responses.
- `src/llm_orc/web/server.py:L30-98` --- `create_app()` with only CORS middleware
- `src/llm_orc/web/api/ensembles.py:L22-75` --- only two explicit `HTTPException` raises

**Pattern:** Absent Error Boundary at the Port Adapter.

**Tradeoff:** Optimizes for development velocity at the expense of usability and security (tracebacks may expose internal paths).

**Question:** What error contract does the React frontend rely on when `execute_ensemble` fails?

**Stewardship:** Register a global exception handler that returns structured JSON errors with a stable schema.

### Meso Level

#### E1: OrchestraService Depends Inward on MCP Handlers

**Status: Resolved (2026-02-23)**

**Observation:** `OrchestraService` imports 11 concrete handler classes from `llm_orc.mcp.handlers.*`. The dependency arrow runs center to boundary, violating hexagonal architecture.
- `src/llm_orc/services/orchestra_service.py:L16-27` --- 11 `from llm_orc.mcp.handlers.*` imports

**Pattern:** Dependency Inversion Violation. The application service depends on adapter-layer implementations.

**Tradeoff:** Optimizes for rapid feature delivery at the expense of layering integrity.

**Question:** What would change if the MCP transport were replaced with gRPC?

**Stewardship:** Move handler classes to `src/llm_orc/services/` or define handler protocols in the service layer.

---

#### E2: `ConfigurationManager` Is an Omnipresent Seed

**Status: Resolved (2026-02-23)**

**Observation:** Imported in 23 files across every layer. Constructor walks the filesystem and creates directories.
- `src/llm_orc/core/config/config_manager.py:L14-40` --- constructor performs mkdir, template copying
- `src/llm_orc/models/base.py:L32-35` --- `HTTPConnectionPool` constructs one inside a lazy init

**Pattern:** Ambient Authority / Constructor Side Effects.

**Tradeoff:** Optimizes for zero-ceremony startup at the expense of testability and surprise.

**Question:** What would it cost to construct a `ConfigurationManager` in a test where `cwd` is unexpected?

**Stewardship:** Separate discovery from construction. Accept a pre-resolved path parameter; defer directory creation to explicit `provision()`.

---

#### E3: `config_manager` Imports `cli_library.library` at Call-Time

**Status: Resolved (2026-02-23)**

**Observation:** `ConfigurationManager` (core layer) defers imports of `cli_library.library` (outer/CLI layer) inside two methods.
- `src/llm_orc/core/config/config_manager.py:L126` --- `from llm_orc.cli_library.library import get_template_content`
- `src/llm_orc/core/config/config_manager.py:L455` --- `from llm_orc.cli_library.library import copy_profile_templates`

**Pattern:** Lasagna Architecture Inversion. Core reaches upward through deferred import.

**Tradeoff:** Optimizes for feature cohesion at the expense of dependency direction.

**Question:** What happens when `llm_orc` is used as a library --- does `requests` execute silently in the background?

**Stewardship:** Define a `TemplateProvider` protocol in core; inject concrete provider at composition root.

---

#### E4: `invoke_streaming` Simulates Execution Instead of Running It

**Status: Resolved (2026-02-23)**

**Observation:** `ExecutionHandler.invoke_streaming` yields hardcoded stub events (`progress: 50`, `status: "success"`) without calling any agent runner.
- `src/llm_orc/mcp/handlers/execution_handler.py:L252-303` --- fabricated events
- `src/llm_orc/services/orchestra_service.py:L204-209` --- delegates without caveat

**Pattern:** Stub-as-Production-Code. The method promises execution; the body delivers theatre.

**Tradeoff:** Optimizes for interface completeness at the expense of truthfulness.

**Question:** What decisions might a consumer make incorrectly based on fabricated `"status": "success"` events?

**Stewardship:** Connect to real execution or rename as a stub and gate behind a feature flag.

---

#### E5: `AdaptiveResourceManager` Names a Class That Does Not Exist

**Status: Resolved (2026-02-23)**

**Observation:** The file is named `adaptive_resource_manager.py` but exports only `SystemResourceMonitor` --- a passive polling monitor. `adaptive_used` is always `False`; `concurrency_decisions` is always `[]`.
- `src/llm_orc/core/execution/adaptive_resource_manager.py:L1` --- module name
- `src/llm_orc/core/execution/agent_resource_monitor.py:L28-33` --- stats always `False`/`[]`

**Pattern:** Ghost Feature. Infrastructure retains naming from a removed capability.

**Tradeoff:** Optimizes for not breaking downstream consumers at the expense of truthfulness.

**Question:** What would a user infer from `"adaptive_resource_management"` in their execution metadata?

**Stewardship:** Rename module to `system_resource_monitor.py`; rename metadata key to `resource_metrics`; remove stub fields.

---

#### E6: `DependencyResolver` Duplicates Role Injection

**Status: Resolved (2026-02-23)**

**Observation:** `DependencyResolver._build_enhanced_input_*` prepends `"You are {agent_name}"` into the user message. The role is already passed as `role_prompt` (system prompt) through `Agent.respond_to_message`. For `OAuthClaudeModel`, the role is stated three times.
- `src/llm_orc/core/execution/dependency_resolver.py:L228-265` --- role in user message
- `src/llm_orc/core/execution/orchestration.py:L19-31` --- `role_prompt=self.role.prompt`

**Pattern:** Responsibility Overlap. Two layers each establish agent identity without knowing the other does so.

**Tradeoff:** Optimizes for ensuring the LLM "remembers" its role at the expense of clean separation.

**Question:** Who owns the decision of how identity is communicated to the model?

**Stewardship:** Remove the `"You are X"` preamble from `DependencyResolver`. The role prompt is the sole place for identity.

---

#### E7: Silent Credential Failure

**Status: Resolved (2026-02-23)**

**Observation:** `_load_credentials` returns `{}` on any exception, including decryption failures. Downstream interprets this as "no credentials stored."
- `src/llm_orc/core/auth/authentication.py:L201-217` --- `except Exception: return {}`
- `src/llm_orc/core/models/model_factory.py:L233-249` --- raises "not configured" rather than "corrupt"

**Pattern:** Tolerant Reader with Silent Degradation.

**Tradeoff:** Optimizes for resilience at the expense of debuggability.

**Question:** What does a user see when their `.encryption_key` is accidentally regenerated?

**Stewardship:** Log at `WARNING` with `exc_info=True`; distinguish `InvalidToken` from `FileNotFoundError`.

---

#### E8: `set_project` Mutates Shared State Without Synchronization

**Status: Resolved (2026-02-23)**

**Observation:** `handle_set_project` writes 13 attributes on a shared singleton with no lock.
- `src/llm_orc/services/orchestra_service.py:L121-151` --- sequential writes, no atomicity

**Pattern:** Unsynchronized Shared Mutable State.

**Tradeoff:** Optimizes for implementation simplicity at the expense of concurrency safety.

**Question:** What happens to an in-flight `execute_streaming` when `set_project` replaces `config_manager` mid-execution?

**Stewardship:** Wrap with `asyncio.Lock` or make immutable-on-set-project (return new instance).

---

#### E9: Fan-Out Zero-Instance Silent No-Op

**Status: Resolved (2026-02-23)**

**Observation:** Load-time validation requires `depends_on` for fan-out agents. At runtime, if upstream fails, the coordinator silently produces zero instances --- no warning, no error.
- `src/llm_orc/core/config/ensemble_config.py:L112-128` --- load-time guard
- `src/llm_orc/core/execution/fan_out_coordinator.py:L37-61` --- silent skip

**Pattern:** Defense-in-Depth with Asymmetric Enforcement.

**Tradeoff:** Optimizes for fault tolerance at the expense of observability.

**Question:** What should a downstream agent receive when its fan-out dependency expanded to zero instances?

**Stewardship:** Emit a structured warning event when fan-out produces zero instances.

---

#### E10: ScriptHandler and ArtifactHandler Bypass Project Context

**Status: Resolved (2026-02-23)**

**Observation:** Seven of nine handlers receive `set_project_context`. `ScriptHandler` and `ArtifactHandler` always use `Path.cwd()`.
- `src/llm_orc/mcp/handlers/script_handler.py:L23` --- `Path.cwd() / ".llm-orc" / "scripts"`
- `src/llm_orc/mcp/handlers/artifact_handler.py:L18` --- `Path.cwd() / ".llm-orc" / "artifacts"`

**Pattern:** Partial Propagation.

**Tradeoff:** Optimizes for handler simplicity at the expense of behavioral consistency.

**Question:** What does a user see when they call `set_project` then `list_scripts`, and the result reflects the server's cwd?

**Stewardship:** Add `set_project_context` to both handlers; include them in the propagation sequence.

---

#### E11: Broken Makefile Targets Reference Deleted Module

**Status: Resolved (2026-02-23)**

**Observation:** Four `validate-contracts-*` targets invoke `llm_orc.testing.contract_validator`, which was deleted. The `pre-commit` target calls `validate-contracts-core`, making `make pre-commit` broken.
- `Makefile:L121-136` --- targets reference deleted module
- `Makefile:L95` --- `pre-commit` depends on `validate-contracts-core`

**Pattern:** Comment Rot at the tooling level.

**Tradeoff:** N/A --- this is an unintended artifact of incomplete cleanup.

**Question:** What does a developer trust --- the `make pre-commit` workflow or the failure they just watched?

**Stewardship:** Remove the four targets and the `pre-commit` dependency on them.

---

#### E12: ADR-004 Implementation Status Claims Deleted Artifacts

**Status: Resolved (2026-02-23)**

**Observation:** ADR-004 is marked "Implemented" with all checkboxes checked. Every referenced artifact (hooks, agents, templates) has been deleted.
- `docs/adrs/004-bdd-llm-development-guardrails.md:L8-16` --- all checked
- `.claude/` --- contains only `settings.json`

**Pattern:** Phantom Documentation.

**Stewardship:** Update status to "Superseded" or "Retired."

---

#### E13: ADR README Documents a Reversed Principle

**Status: Resolved (2026-02-23)**

**Observation:** The ADR README states "Primitives live in `llm-orchestra-library`, not in core application." ADR-006's amendment moved 6 core primitives into the package.
- `docs/adrs/README.md:L30` --- superseded principle
- `src/llm_orc/primitives/` --- 11 files confirming the amendment

**Pattern:** Version Drift.

**Stewardship:** Update the principle to reflect the amended boundary.

---

#### E14: Three ADRs Marked "Accepted" Are Fully Implemented

**Status: Resolved (2026-02-23)**

**Observation:** ADRs 012, 013, 014 are listed as "Accepted" with no completion percentage. All three are fully implemented and described in `architecture.md`.
- `docs/adrs/README.md:L17-19` --- "Accepted" with dashes
- `src/llm_orc/schemas/agent_config.py:L1` --- ADR-012 implemented

**Pattern:** Version Drift.

**Stewardship:** Update statuses to "Implemented."

---

#### E15: Dead Dependencies in pyproject.toml

**Status: Resolved (2026-02-23)**

**Observation:** `websockets` and `aiohttp>=3.13.3` are production dependencies with zero imports.
- `pyproject.toml:L28-29` --- declared
- Zero matches for `import websockets` or `import aiohttp` across `src/`

**Pattern:** Dead Dependency.

**Stewardship:** Remove both or move to optional extras.

---

#### E16: README Claims Wrong Library Default

**Status: Resolved (2026-02-23)**

**Observation:** README says "by default, fetches library content from the remote GitHub repository." The code defaults to `return "local", ""` --- a no-op.
- `src/llm_orc/cli_library/library.py:L56-58` --- default is local no-op

**Pattern:** Scope Mismatch.

**Stewardship:** Update README to match actual behavior.

### Micro Level

#### U1: ConfigurationManager Constructor Side Effects

**Status: Resolved (2026-02-23)**

**Observation:** `__init__` creates directories, writes config files, and copies templates before returning.
- `src/llm_orc/core/config/config_manager.py:L36-40` --- mkdir, template copy, default config write

**Pattern:** Constructor with Excessive Side Effects (OO Abusers).

**Tradeoff:** Optimizes for zero-ceremony startup at the expense of testability.

**Stewardship:** Split provisioning from reading. Constructor should be pure; `provision_defaults()` for setup paths.

---

#### U2: Tier Classification Duplicated Across Five Files

**Status: Resolved (2026-02-23)**

**Observation:** The decision of whether a path is "local", "global", or "library" is encoded independently in 5+ places with different algorithms.
- `src/llm_orc/mcp/handlers/promotion_handler.py:L341,L711` --- string-contains on `".llm-orc"`
- `src/llm_orc/cli_commands.py:L103-106` --- `startswith(local_config_dir)`

**Pattern:** Duplicated Code / Shotgun Surgery potential.

**Tradeoff:** Optimizes for handler self-containment at the expense of consistency.

**Stewardship:** Define `classify_tier(path)` on `ConfigurationManager` and route all sites through it.

---

#### U3: Flat 34-File Execution Package

**Status: Resolved (2026-02-24)**

**Observation:** `core/execution/` contained 32 files in a single flat directory with no sub-packages.

**Resolution:** Reorganized into 5 sub-packages: `fan_out/` (3 files), `scripting/` (6 files), `runners/` (2 files), `phases/` (6 files), `monitoring/` (4 files). 11 shared/hub files remain at root. Zero cross-sub-package imports. All 21 moves with full import rewrites across ~70 files.

---

#### U4: PromotionHandler at 2.7x Peer Size

**Status: Resolved (2026-02-23)**

**Observation:** `promotion_handler.py` is 831 lines. Next largest handler is 303 lines. It duplicates profile parsing that `ProfileHandler` already owns.
- `src/llm_orc/mcp/handlers/promotion_handler.py:L554-620` --- reimplements YAML profile parsing

**Pattern:** Large Class with Feature Envy.

**Tradeoff:** Optimizes for handler self-sufficiency at the expense of maintaining profile logic in two places.

**Stewardship:** Extend `ProfileHandler` with `get_profiles_at_tier()` and delegate from `PromotionHandler`.

---

#### U5: Temporary Fields on EnsembleExecutor

**Status: Resolved (2026-02-23)**

**Observation:** `_current_agent_configs` is assigned mid-execution and accessed via `hasattr` guard.
- `src/llm_orc/core/execution/ensemble_execution.py:L458` --- assignment in `_initialize_execution_setup`
- `src/llm_orc/core/execution/ensemble_execution.py:L782` --- `hasattr` guard

**Pattern:** Temporary Field (OO Abusers).

**Tradeoff:** Optimizes for avoiding parameter threading at the expense of making re-entrant execution semantics undefined.

**Stewardship:** Pass `agent_configs` explicitly or encapsulate in an `ExecutionContext` dataclass created per call.

---

#### U6: Ghost Directories

**Status: Resolved (2026-02-23)**

**Observation:** `core/communication/`, `contracts/`, `testing/` contain only `__pycache__`. No `.py` source files.

**Pattern:** Vestigial Structure.

**Stewardship:** Delete the directories and their `__pycache__` contents.

---

#### U7: `SUPPORTS_CUSTOM_ROLE_PROMPT` Flag Declared But Never Read

**Status: Resolved (2026-02-23)**

**Observation:** The flag is set on `ModelInterface` (True) and `OAuthClaudeModel` (False). No code reads it.
- `src/llm_orc/models/base.py:L88` --- declared
- `src/llm_orc/models/anthropic.py:L281` --- overridden
- Zero reads across `src/`

**Pattern:** Dead Interface Contract.

**Stewardship:** Either enforce at the dispatch site or remove entirely.

---

#### U8: `EventFactory` Never Triggered in Production

**Status: Resolved (2026-02-23)**

**Observation:** `ScriptUserInputHandler` emits `EventFactory` events only when `self.event_emitter` is not `None`. Every production callsite constructs with no arguments (`event_emitter=None`).
- `src/llm_orc/core/execution/script_user_input_handler.py:L157-198` --- gated on `event_emitter`
- Four construction sites: all pass no `event_emitter`

**Pattern:** Unreachable Code (conditional on a parameter never passed).

**Stewardship:** Either wire `event_emitter` from the execution pipeline or remove the gated code.

---

#### U9: Mystery Guest `_test_*` Escape Hatches in Production Code

**Status: Resolved (2026-02-23)**

**Observation:** `ScriptHandler`, `ArtifactHandler`, and `LibraryHandler` carry class-level `_test_*` attributes that tests mutate through two delegation layers.
- `src/llm_orc/mcp/handlers/script_handler.py:L12` --- `_test_scripts_dir: Path | None = None`
- `tests/unit/mcp_server/test_server.py:L330` --- `server._library_handler._test_library_dir = tmp_path`

**Pattern:** Mystery Guest. Production code carries test-only state.

**Tradeoff:** Optimizes for test setup convenience at the expense of encapsulation.

**Stewardship:** Accept `base_dir: Path` via constructor injection. Remove class-level `_test_*` attributes.

---

#### U10: Assertion Roulette in Init Tests

**Status: Resolved (2026-02-24)**

**Resolution:** Removed 3 redundant `assert X is not None` tests from `TestMCPServerInitialization`. Kept the DI contract test (`test_init_with_custom_config_manager`) which verifies the injected config_manager is the returned instance.

---

#### U11: Eager Test in `test_caching.py`

**Status: Resolved (2026-02-24)**

**Resolution:** Deleted `test_caching.py`. The inline `ResultCache` class had no production counterpart. All 5 behaviors it tested (miss, hit, key uniqueness, invalidation, reproducibility) are covered by `test_script_cache.py`'s 24 tests against the real `ScriptCache`.

---

#### U12: BDD Coverage Gaps for ADRs 010--014

**Status: Resolved (2026-02-23)**

**Observation:** BDD feature files exist for ADRs 001--009 but none for 010--014, all of which are implemented.

**Pattern:** Test-Code Correspondence Gap.

**Stewardship:** Add feature files for the implemented ADRs, prioritizing ADR-012 (typed agent configs) and ADR-014 (input_key).

---

#### U13: ADR Marker Drift

**Status: Resolved (2026-02-23)**

**Observation:** `pyproject.toml` registers markers for ADR-003 and ADR-005 (deleted/unused). No markers for ADR-010 through ADR-014 (implemented). `pytest -m adr-005` runs zero tests.

**Pattern:** Marker Drift.

**Stewardship:** Audit markers against current ADR index. Remove orphaned entries; add markers for implemented ADRs.

---

#### U14: `LLMResponseGenerator` Hardcodes `OllamaModel`

**Status: Resolved (2026-02-23)**

**Observation:** Test infrastructure class constructs `OllamaModel` unconditionally.
- `src/llm_orc/core/validation/llm_simulator.py:L33` --- `self.llm_client = OllamaModel(model_name=model)`

**Pattern:** Concrete Class Dependency in Test Infrastructure.

**Stewardship:** Accept `client: ModelInterface | None` parameter; construct `OllamaModel` only as default.

---

#### U15: CLI Tests Verify Routing, Not Behavior

**Status: Resolved (2026-02-23)**

**Observation:** `test_cli_commands.py` (1,574 lines) patches all collaborators and asserts on call signatures. Tests pass identically whether CLI uses `OrchestraService` or constructs the executor directly.
- `tests/unit/cli/test_cli_commands.py:L113-124` --- mock assertion on `execute.assert_called_once()`

**Pattern:** Implementation Testing rather than Behavior Testing.

**Stewardship:** Route CLI through `OrchestraService` first (M1), then rewrite tests to verify observable outcomes.

### Multi-Lens Observations

The following findings were independently surfaced by multiple lenses, increasing confidence:

**`_dep_name` duplicated 4x** (Pattern Recognition + Intent-Implementation + Invariant + Dependency & Coupling): All four lenses independently identified the same four file locations. Convergence confirms this is a real maintenance risk, not a stylistic preference.

**CLI bypasses OrchestraService** (Pattern Recognition + Dependency & Coupling + Structural Health + Test Quality): Four independent lenses converged on `cli_commands.py:L318`. The Pattern lens saw the hexagonal breach; Dependency saw the import graph anomaly; Structural Health saw the Stovepipe anti-pattern; Test Quality saw the implementation-testing symptom.

**`invoke_streaming` yields fabricated events** (Intent-Implementation + Dead Code): Both lenses independently identified `execution_handler.py:L252-303` as a method whose name promises execution but delivers canned output.

**`SUPPORTS_CUSTOM_ROLE_PROMPT` is dead** (Architectural Fitness + Intent-Implementation + Dead Code): Three lenses converged. Architectural Fitness saw the Liskov violation; Intent-Implementation saw the dead declaration; Dead Code confirmed zero reads.

**`mcp/handlers/` is the misplaced domain layer** (Decision Archaeology + Intent-Implementation + Dependency & Coupling): Three lenses independently concluded the handlers are application-layer business logic living under an adapter-layer namespace.

**Dead dependencies (websockets, aiohttp)** (Decision Archaeology + Documentation Integrity + Dead Code): Three lenses confirmed zero imports.

**ConfigurationManager constructor side effects** (Pattern Recognition + Dependency & Coupling + Structural Health): Three lenses converged on the constructor's filesystem mutations.

## Stewardship Guide

### What to Protect

1. **Pydantic discriminated union for agent configs** (`schemas/agent_config.py`). The `extra="forbid"` enforcement catches configuration errors at load time. This is the codebase's strongest typing boundary.

2. **Phase-based dependency resolution** (`dependency_resolver.py`, `dependency_analyzer.py`). The topological sort with parallel execution grouping is correct and well-tested. The fan-out/gather pattern is a genuine architectural contribution.

3. **`OrchestraService` as the hexagonal core**. The extraction from `MCPServer` was the right move. The service layer exists and works for two of three ports. Protect this pattern by routing the CLI through it.

4. **ADR discipline**. The project has 14 ADRs documenting decisions, reversals, and deletions. The practice of recording decisions --- including mistakes --- is valuable. The index just needs maintenance.

5. **90% coverage floor with parallel test execution**. The CI enforcement is real and the `pytest-xdist` integration works. The test infrastructure is mature even where individual test quality varies.

### What to Improve (Remaining)

All 37 audit findings have been resolved as of 2026-02-24.

### Ongoing Practices

- **Update ADR status when implementation lands.** Pair every implementation commit with a status update commit.
- **Run `make pre-commit` before pushing.** The pre-commit hook runs mypy, ruff, complexipy, bandit, and vulture.
- **Review new modules for dependency direction.** Core should not import from `mcp/` or `cli_library/`. Use `TemplateProvider`-style protocols for upward dependencies.
- **Route new CLI commands through `OrchestraService`.** The CLI is a thin display adapter — parse args, call service, format output.
- **Prefer constructor injection over class-level `_test_*` attributes.** When adding new handlers, accept `base_dir: Path` rather than adding mutable class state for tests.
- **Use `ConfigurationManager(provision=False)` for read-only access.** Only callers that need directory/template setup should use default `provision=True`.
- **Call `HTTPConnectionPool.configure()` at the composition root.** Don't construct `ConfigurationManager` inside model-layer code.
- **Log, don't swallow.** When adding new `except` blocks, narrow to specific exception types. Reserve broad `except Exception` for truly optional features with `logger.debug(exc_info=True)`.
