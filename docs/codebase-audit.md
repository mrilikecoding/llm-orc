# Codebase Audit: LLM Orchestra (llm-orc)

**Date:** 2026-02-20
**Scope:** Whole codebase — all source files under `src/llm_orc/`, all test files under `tests/`, all documentation under `docs/`
**Coverage:** This analysis sampled strategically across ~200 source files (~23,600 LOC) and ~120 test files (~58,400 LOC). Ten analytical lenses operated independently across three levels (Macro, Meso, Micro). Sampling focused on entry points, module boundaries, configuration, cross-cutting concerns, import graphs, public interfaces, validation code, test files, and documentation. Areas not covered: individual primitive script implementations in `llm-orchestra-library/`, the frontend assets served by `web/server.py`, and detailed analysis of every individual test method.

---

## Executive Summary

LLM Orchestra is a Python-based multi-agent LLM orchestration system that coordinates ensembles of AI agents through phased, dependency-resolved execution pipelines. It delivers its capabilities through three ports — a Click CLI, a FastMCP server, and a FastAPI web UI — all converging on `EnsembleExecutor` as the core execution engine.

The architecture approximates a hexagonal pattern with clear delivery-port separation at the directory level, but the boundaries are undermined by two structural problems: `MCPServer` has become the de facto application service layer for both the MCP and HTTP ports (the web API calls its private methods directly), and `dict[str, Any]` serves as the universal data contract throughout the execution subsystem (217 occurrences across 26 files), erasing the type safety that the Pydantic schemas at the script boundary were designed to provide. The system has accumulated several complete-but-unreachable features (`ConversationalEnsembleExecutor`, `communication/protocol.py`, `core/events/` Pydantic hierarchy, `ScriptContract` ABC) that create the appearance of capabilities the system does not actually deliver.

The most significant cross-cutting tension is between **execution availability** and **observability**: the system prioritizes ensuring ensembles always complete (silent exception suppression at 8+ sites, silent agent erasure, silent fallback model substitution) at the expense of callers being able to trust what they receive. An ensemble can report "success" while having lost agents, substituted models, and bypassed schema validation — all invisibly.

---

## Architectural Profile

### Patterns Identified

| Pattern | Confidence | Evidence |
|---------|-----------|----------|
| Approximate Hexagonal Architecture | High | Three delivery ports (CLI, MCP, Web) converge on core execution. Directory structure reflects port separation. |
| God Object (EnsembleExecutor) | High | 500+ line class manually wiring 18+ collaborators in `__init__`. Second instance in MCPServer (10+ handlers, dual dispatch table). |
| Stringly-Typed Protocol | High | `dict[str, Any]` as universal contract — 217 occurrences in `core/execution/` alone, 265+ across codebase. |
| Accidental Shared Service | High | MCPServer serves as application service for both MCP and Web ports via private method calls. |
| Event Sourcing (partial) | Medium | Pydantic Event hierarchy exists but is orphaned. Live events flow as raw dicts through streaming queue. |
| YAML-as-Schema | Medium | Ensemble configuration parsed from YAML with structural validation but no load-time content validation of expressions. |

### Quality Attribute Fitness

| Quality Attribute | Assessment |
|------------------|------------|
| **Execution Availability** | Strongly optimized — ensembles complete even when agents fail, models are unavailable, or schemas are violated |
| **Development Velocity** | Moderately optimized — dict-based contracts allow rapid iteration without schema definition overhead |
| **Testability** | Mixed — comprehensive test suite (90% coverage floor) but God Constructors resist injection; BDD tests validate local models not production code |
| **Modifiability** | Constrained — `dict[str, Any]` requires codebase-wide search for schema changes; MCPServer coupling means port changes ripple |
| **Observability** | Neglected — silent exception suppression, no substitution flags on fallback results, two incompatible event shapes |
| **Security** | At risk — two `eval()` paths with divergent sandboxes, context-merge vulnerability, no adversarial test coverage |
| **Correctness** | Undermined by availability bias — schema validation advisory-only at runtime, cached results bypass validation, agent erasure invisible |

### Inferred Decisions

**Inferred Decision: YAML as Configuration Language (High Confidence)**
The system chose YAML for ensemble configuration, establishing YAML files as the schema definition surface. Validation occurs at load time for structure (cycles, missing deps) but not for content (assertion expressions, condition strings). This decision appears to have been made early and never revisited as `eval()` was introduced for assertion/condition evaluation.

**Inferred Decision: Extraction Over Redesign for EnsembleExecutor (High Confidence)**
The git history shows aggressive decomposition of `EnsembleExecutor` into 15+ extracted classes over recent months. The extraction preserved the original composition pattern — all collaborators created in `__init__` — rather than introducing dependency injection. The comment "for test patchability" at L217 confirms the design is shaped by extraction constraints.

**Inferred Decision: MCP First, Web Second (High Confidence)**
MCP was the primary delivery port. When the web UI was added (ADR-010), rather than extracting a shared application service, the web routes were connected to MCPServer's private methods. The delegation stubs labeled "for web API" confirm this was recognized as a workaround.

**Inferred Decision: BDD as LLM Development Guardrail (High Confidence)**
ADR-004 established BDD scenarios as architectural enforcement. The implementation created behavioral tests but deferred structural fitness functions (explicitly marked as unimplemented in ADR-004). The BDD layer validates intent but not invariant completeness.

**Inferred Decision: Asyncio with Synchronous Bridges (Medium Confidence)**
The core execution is async, but OAuth flows use synchronous `requests` via `run_in_executor`. Three HTTP client strategies coexist (`requests`, `httpx` singleton pool, `httpx` per-call). This appears to be an accretion pattern rather than a deliberate choice.

**Inferred Decision: Silent Failure Over Loud Failure (Medium Confidence)**
At least 8 sites use `except Exception: pass` or equivalent in execution-critical paths. The pattern is consistent enough to suggest a deliberate philosophy: ensembles should always produce a result, even a degraded one. The cost is that degradation is invisible.

---

## Tradeoff Map

| Optimizes For | At the Expense Of | Evidence |
|--------------|-------------------|----------|
| Execution availability | Result fidelity and observability | Silent agent erasure, fallback model substitution without flags, advisory-only schema validation |
| Development velocity | Type safety and refactorability | `dict[str, Any]` as universal contract (217 occurrences in core/execution/) |
| Reuse without duplication | Port isolation | Web API routes through MCPServer private methods (16+ call sites) |
| Construction simplicity | Testability and replaceability | God Constructor in EnsembleExecutor (18+ collaborators, no injection) |
| Local expressiveness | Security | Two `eval()` paths with divergent sandboxes, context-merge vulnerability |
| Fast decomposition | Conceptual clarity | `AgentExecutor` doesn't execute agents; extraction preserved stale names |
| Coverage density | Diagnostic clarity | 202 assertions in 38 methods in dominant test file; Assertion Roulette |
| Design-first development | Signal clarity | Tests pass on dead code, BDD tests define local classes instead of importing production |
| OAuth authorization compliance | Interface consistency | `OAuthClaudeModel` silently discards `role_prompt`, breaking Liskov Substitution |

---

## Findings

### Macro Level

#### Finding: MCPServer Has Become an Accidental Application Service Layer

**Observation:** The web API's four routers call MCPServer's underscore-prefixed methods directly (16+ call sites), and some bypass even the delegation stubs to access private collaborators (`mcp._script_handler`, `mcp.artifact_manager`). MCPServer was designed as a protocol adapter but has accumulated application service responsibilities.
- `src/llm_orc/web/api/ensembles.py:L29,36,49,60,74` — five route handlers calling private MCPServer methods
- `src/llm_orc/web/api/scripts.py:L26,34,44` — routes reaching through to `mcp._script_handler` directly
- `src/llm_orc/web/api/artifacts.py:L20` — `mcp.artifact_manager.list_ensembles()` — two levels deep into MCPServer's composition
- `src/llm_orc/mcp/server.py:L780-806` — methods labeled "delegation stubs for web API"

**Pattern:** Accidental Shared Service / Stovepipe. A delivery-port adapter has become the shared application service for a second port, creating transitive coupling between ports that should be independently changeable.

**Tradeoff:** Optimizes for reuse without duplication at the expense of port isolation. Renaming any `MCPServer` private method requires simultaneous changes across the web layer.

**Question:** What would it cost to extract a shared `OrchestraApplicationService` that both ports route through, and would that cost be less than the ongoing friction of web developers needing to understand MCP server internals?

**Stewardship:** Extract the dozen operations both ports need into a stable public service interface. Both `MCPServer` and the web API become thin protocol adapters. The underscore-prefixed stubs are evidence the team already recognizes the problem.

---

#### Finding: `dict[str, Any]` Is the Universal Internal Protocol

**Observation:** Agent configurations, execution results, phase results, performance events, and streaming events all flow as `dict[str, Any]` — 217 occurrences across 26 files in `core/execution/` alone. Pydantic schemas exist at the script boundary (ADR-001) but have not been extended inward.
- `src/llm_orc/core/execution/ensemble_execution.py` — 24 occurrences in this file alone
- `src/llm_orc/core/execution/results_processor.py:L106-118` — returns fixed-shape dict with keys `ensemble`, `status`, `input`, `results`, `synthesis`, `metadata` — nothing enforces the shape
- `src/llm_orc/core/execution/agent_dispatcher.py:L130-134` — constructs `{"response": response, "status": "success", "model_instance": model_instance}` inline

**Pattern:** Primitive Obsession / Stringly-Typed Protocol. Raw dicts represent domain concepts that have identity, invariants, and behavior, scattering the "schema" across all read/write sites.

**Tradeoff:** Optimizes for construction velocity at the expense of correctness and refactorability. Any key misspelling is a runtime error, not a type error. Renaming a field requires codebase-wide search with no static analysis aid.

**Question:** Given that `ScriptAgentInput` and `ScriptAgentOutput` already exist as typed contracts, what prevents extending that treatment to `AgentResult`, `PhaseResult`, and `ExecutionResult`?

**Stewardship:** Define typed dataclasses or Pydantic models for the three to four most-trafficked dict shapes. The `Event` Pydantic model in `core/events/base.py` is an orphaned correct design that was never connected — it demonstrates the team already has the instinct.

> **Update (2026-02-20):** ADR-012 addresses this partially — agent configs migrate from `dict[str, Any]` to a Pydantic discriminated union (`LlmAgentConfig | ScriptAgentConfig | EnsembleAgentConfig`). This covers the agent config surface area (250 agents across 63 files). The broader execution result types (`AgentResult`, `PhaseResult`, `ExecutionResult`) remain as future work.

---

#### Finding: Silent Failure Philosophy Across Eight Sites

**Observation:** At least eight sites use `except Exception: pass` (or equivalent) in execution-critical paths, including the streaming event merger, agent request processing, artifact saving, and model fallback loading.
- `src/llm_orc/core/execution/ensemble_execution.py:L332-333` — entire streaming pipeline wrapped in `except Exception: pass`
- `src/llm_orc/core/execution/phase_result_processor.py:L155-156` — agent request processing silently swallowed
- `src/llm_orc/core/execution/script_agent_runner.py:L223-224,L246-247` — two `except Exception: pass` with `# nosec B110` suppression
- `src/llm_orc/core/execution/ensemble_execution.py:L664-670` — artifact saving failure ignored without logging
- `src/llm_orc/core/models/model_factory.py:L215-216` — fallback model loading failure silently caught

**Pattern:** Error Hiding / Fault Absorption. The system conflates "preserve execution throughput" with "preserve result integrity."

**Tradeoff:** Optimizes for surface-level resilience at the expense of observability and debuggability. An operator whose streaming output stops has no signal from which to diagnose the cause.

**Question:** If a production ensemble emits no streaming events after `execution_started`, what information is available to diagnose whether the failure is in the streaming merger, the execution, or the queue?

**Stewardship:** Apply distinct handling at each site. For truly optional operations (artifact saving), log at `WARNING`. For caller-affecting operations (streaming merger, request processing), propagate the exception or emit a structured error event.

> **Update (2026-02-20):** Domain model Invariant 13 (execution resilience) and Invariant 14 (validation vs. execution boundary) establish the intended policy: ensembles complete even with agent failures, but failures are *recorded* with a `has_errors` flag. The 8 existing silent suppression sites implement resilience without recording — conformance debt against these invariants. ADR-013's `EnsembleAgentRunner` will implement Invariant 13 correctly for child ensemble failures.

---

#### Finding: Two `eval()` Paths with Divergent Sandboxes and Leaky Context

**Observation:** Two production `eval()` sites exist with different security postures. Neither is tested for sandbox bypass.
- `src/llm_orc/core/validation/evaluator.py:L243-270` — `restricted_globals` with large builtins allowlist; `context.update()` merges execution context (including LLM output) into globals *before* `eval()`
- `src/llm_orc/schemas/conversational_agent.py:L107-119` — `safe_context = {"__builtins__": {}}` with `# nosec B307`

**Pattern:** Restricted Interpreter with Leaky Context. The sandbox restricts builtins but merges untrusted data into the global namespace before evaluation.

**Tradeoff:** Optimizes for expressiveness of YAML-level assertions at the expense of sandbox integrity. A context key named `len` or `sorted` from LLM output could shadow builtins.

**Question:** What happens when an LLM agent produces output with a key named `len`, and a behavioral assertion calls `len(results)`?

**Stewardship:** Pass execution context as locals, not merged into globals: `eval(assertion, restricted_globals, execution_context)`. Consolidate both sites to a single sandbox implementation. Add adversarial tests.

---

#### Finding: Hardcoded Fallback Silently Substitutes Models

**Observation:** When the configurable fallback chain is exhausted, `ModelFactory._try_legacy_fallback` unconditionally falls back to `OllamaModel(model_name="llama3")`. The result is tagged `"success"` with no substitution flag.
- `src/llm_orc/core/models/model_factory.py:L196-223` — falls through to hardcoded `OllamaModel(model_name="llama3")`
- `src/llm_orc/core/execution/llm_agent_runner.py:L76-137` — fallback execution emits informational events but result `status` is `"success"`

**Pattern:** Availability Invariant Override. The system maintains "agents always produce a result" by substituting whatever model is locally available.

**Tradeoff:** Optimizes for execution availability at the expense of result fidelity. A caller checking `result["status"] == "success"` cannot distinguish authentic execution from fallback.

**Question:** If a validation layer passes because `llama3` happened to produce output in the right format, what does the validation result actually certify?

**Stewardship:** Add a `model_substituted: bool` field to execution results. Make the hardcoded `llama3` fallback configurable rather than a code constant.

---

#### Finding: Version String Frozen at 0.3.0 While Package Ships as 0.14.4

**Observation:** `__init__.py` declares `__version__ = "0.3.0"` while `pyproject.toml` says `0.14.4`. The stale version is sent to Anthropic's OAuth endpoint in `User-Agent` and `X-Stainless-Package-Version` headers.
- `src/llm_orc/__init__.py:L3` — `__version__ = "0.3.0"`
- `pyproject.toml:L7` — `version = "0.14.4"`
- `src/llm_orc/core/auth/oauth_client.py:L37,39` — stale version broadcast in HTTP headers

**Pattern:** Version Drift. The runtime-observable version has been wrong for 11 minor releases.

**Tradeoff:** Optimizes for single-source versioning in pyproject.toml at the expense of runtime correctness.

**Question:** What would it take for Anthropic's servers to rely on the User-Agent version?

**Stewardship:** Replace the hardcoded string with `importlib.metadata.version("llm-orchestra")`.

---

### Meso Level

#### Finding: Web API Couples to MCPServer's Private State via `_handle_set_project`

**Observation:** `_handle_set_project` mutates four fields including reaching into `self._execution_handler._project_path` directly. It replaces `self.config_manager` but doesn't propagate to handlers holding the old reference.
- `src/llm_orc/mcp/server.py:L889-926` — four separate state mutations
- `src/llm_orc/mcp/server.py:L910` — writes directly to `_execution_handler._project_path`
- `src/llm_orc/mcp/server.py:L107-134` — handlers receive `config_manager` at construction; not updated after project switch

**Pattern:** Partial State Mutation / Shotgun Surgery. A state-change operation updates some but not all components that hold a reference.

**Tradeoff:** Optimizes for incremental implementation at the expense of internal consistency.

**Question:** What is the path a concurrent `invoke` uses if two MCP clients call `set_project` with different paths?

**Stewardship:** Replace the four-field mutation with a `ProjectContext` value object set atomically and passed explicitly to handlers.

---

#### Finding: `OAuthClaudeModel` Silently Discards `role_prompt`

**Observation:** `ModelInterface.generate_response(message, role_prompt)` establishes a contract that the model uses the caller's role. `OAuthClaudeModel` hardcodes `"You are Claude Code, Anthropic's official CLI for Claude."` and never incorporates `role_prompt`.
- `src/llm_orc/models/anthropic.py:L66-77` — `oauth_system_prompt` hardcoded; `role_prompt` ignored
- `src/llm_orc/models/base.py:L106-108` — abstract contract defines `role_prompt` as configuration

**Pattern:** Broken Liskov Substitution. Swapping implementations changes observable behavior.

**Tradeoff:** Optimizes for OAuth authorization compliance at the expense of interface consistency.

**Question:** What signal does the system give to an operator who configures a specialist `role_prompt` and routes it to an OAuth-authenticated agent?

**Stewardship:** Log a warning when `role_prompt` differs from the hardcoded prompt. Surface `SUPPORTS_CUSTOM_ROLE_PROMPT = False` as a class-level attribute.

> **Done (2026-02-21):** `OAuthClaudeModel.SUPPORTS_CUSTOM_ROLE_PROMPT = False` added. `ModelInterface` base class now declares `SUPPORTS_CUSTOM_ROLE_PROMPT: bool = True` as the default. `_inject_role_if_needed` logs a `WARNING` with the differing role_prompt the first time a custom role is established per instance.

---

#### Finding: Silent Agent Erasure in Parallel Phase Execution

**Observation:** When `asyncio.gather` runs parallel agents and a task raises `BaseException`, the result is silently dropped with `continue`. The agent disappears from `phase_results` — the ensemble can finish with `"success"` while missing an agent.
- `src/llm_orc/core/execution/agent_dispatcher.py:L52-59` — `if isinstance(result, BaseException): continue`
- `src/llm_orc/core/execution/phase_result_processor.py:L44-62` — error flag only set when `agent_result["status"] == "failed"`

**Pattern:** Fault Absorption / Silent Data Loss.

**Tradeoff:** Optimizes for execution continuity at the expense of result correctness.

**Question:** What does the caller understand about a result that reports success but is missing an agent's output?

**Stewardship:** Construct a synthetic failed-agent record when a `BaseException` escapes, so the error flag propagates and every configured agent produces a result entry.

---

#### Finding: Advisory-Only Schema Contract on the Critical Execution Path

**Observation:** `ScriptAgentRunner._validate_primitive_output` validates output against Pydantic schemas, but on failure logs a warning and returns — passing malformed output downstream unchanged.
- `src/llm_orc/core/execution/script_agent_runner.py:L177-201` — "preserves existing workflows" — warning only
- `src/llm_orc/contracts/contract_validator.py:L280-310` — the same schema enforced with hard `ValueError` in CI

**Pattern:** Contract Enforcement Asymmetry. The schema promise is meaningful only at test time.

**Tradeoff:** Optimizes for backward compatibility at the expense of contract integrity.

**Question:** What is the value of a schema contract if a downstream agent receives the invalid payload and begins processing it?

**Stewardship:** Introduce a `strict_schema_validation` flag in ensemble config, or define a timeline after which registered primitives are enforced.

---

#### Finding: `AgentExecutor` Does Not Execute Agents

**Observation:** Despite its name, `AgentExecutor` monitors resources and collects metrics. Actual execution lives in `LlmAgentRunner` and `ScriptAgentRunner`. The name was retained from a pre-extraction era.
- `src/llm_orc/core/execution/agent_executor.py:L10-78` — docstring: "Provides resource monitoring and execution metrics"
- `src/llm_orc/core/execution/ensemble_execution.py:L193-194` — only use is passing it to `PhaseMonitor`

**Pattern:** Divergent naming after extraction.

**Tradeoff:** Optimizes for fast decomposition at the expense of conceptual clarity.

**Question:** What would a new contributor assume `AgentExecutor.execute(...)` does?

**Stewardship:** Rename to `AgentResourceMonitor` or `ExecutionResourceTracker`.

---

#### Finding: Documentation Describes Features That Don't Exist

**Observation:** `docs/architecture.md` claims RESTful interface, WebSocket support, batch processing, and a provider registration system. Most don't exist or are partial. The extensibility section names `llm_orc.models.base.BaseModel` — the actual class is `ModelInterface`.
- `docs/architecture.md:L276-280` — claims WebSocket support, batch processing
- `docs/architecture.md:L305` — names `BaseModel` (wrong class name), describes "Registration System" (doesn't exist)
- `docs/adrs/010-local-web-ui.md:L3-4` — Status: "Proposed" but feature is implemented and shipping

**Pattern:** Aspirational documentation / Comment rot.

**Tradeoff:** Optimizes for communicating vision at the expense of accuracy.

**Question:** What does a developer expect when they read "RESTful Interface: HTTP API for programmatic access," and how do they discover no stable API contract exists?

**Stewardship:** Retitle aspirational sections. Update ADR-010 to "Implemented." Fix `BaseModel` to `ModelInterface`.

---

#### Finding: Coverage Threshold Documented at 95%, Enforced at 90%

**Observation:** `docs/coding-standards.md` claims 95% CI enforcement; `pyproject.toml` enforces 90%. The document also instructs "use real LLM APIs in tests — mock sparingly" while the test suite has 375+ mock imports.
- `docs/coding-standards.md:L100-101` — "95% threshold enforced in CI"
- `pyproject.toml:L89` — `--cov-fail-under=90`

**Pattern:** Threshold drift / aspirational documentation.

**Tradeoff:** Optimizes for aspirational quality signaling at the expense of accuracy.

**Question:** When a contributor reads "95% CI Enforcement" and their PR passes CI at 92%, what do they conclude?

**Stewardship:** Align the document with reality: either raise the threshold or update the document.

---

### Micro Level

#### Finding: EnsembleExecutor God Constructor — 18+ Collaborators Wired in `__init__`

**Observation:** An 85-line constructor creates and cross-wires 18+ named collaborators with no external injection. Five receive the same callback (`_emit_performance_event`). A closure at L217 exists explicitly "for test patchability."
- `src/llm_orc/core/execution/ensemble_execution.py:L158-243` — full construction site
- `src/llm_orc/core/execution/ensemble_execution.py:L217-220` — test workaround closure

**Pattern:** God Object / Large Class (Bloater).

**Tradeoff:** Optimizes for construction simplicity at the expense of testability and replaceability.

**Question:** What happens when a test needs only `ScriptAgentRunner` logic? How many objects does it instantiate?

**Stewardship:** Extract wiring into a factory. `EnsembleExecutor.__init__` should accept already-constructed collaborators.

> **Update (2026-02-20):** ADR-013 (ensemble agent type) will need to create child executors that share immutable infrastructure but isolate mutable state. This will exercise the God Constructor's limitations directly. The ADR prescribes the sharing/isolation boundary but leaves the construction mechanism to implementation. A factory method or alternate constructor path may be needed.

---

#### Finding: Three ClaudeModel Classes with Duplicated Usage Recording

**Observation:** `ClaudeModel`, `OAuthClaudeModel`, and `ClaudeCLIModel` each implement their own usage recording and cost calculation. `ClaudeModel` hardcodes per-token rates; the others use `cost_usd=0.0`. All three use deprecated `get_event_loop()`.
- `src/llm_orc/models/anthropic.py:L240-266` — `ClaudeModel` with hardcoded pricing
- `src/llm_orc/models/anthropic.py:L103-144` — `OAuthClaudeModel` duplicate with `cost_usd=0.0`
- `src/llm_orc/models/anthropic.py:L416-427` — `ClaudeCLIModel` third copy

**Pattern:** Duplicated Code (Dispensable).

**Tradeoff:** Optimizes for class-level autonomy at the expense of consistency.

**Question:** When Anthropic changes token pricing, how many files need updating?

**Stewardship:** Extract shared `_record_usage_with_cost` to the base class. Replace `get_event_loop()` with `get_running_loop()`.

> **Partial (2026-02-21):** `asyncio.get_event_loop()` replaced with `asyncio.get_running_loop()` in both `_execute_oauth_api_call` and `ClaudeCLIModel.generate_response`. The `_record_usage` method is already on the base class; per-class cost calculation logic is sufficiently distinct (real rates vs. 0.0 vs. estimation) that further extraction offers limited value.

---

#### Finding: Assertion Roulette in the Dominant Test File

**Observation:** `test_ensemble_execution.py` contains 38 test methods with 202 `assert` calls across 2,400 lines. Many tests pile multiple structural assertions without messages.
- `tests/unit/core/execution/test_ensemble_execution.py:L513-560` — one test verifies three distinct model types with 12+ assertions
- `tests/unit/core/execution/test_ensemble_execution.py:L74-93` — 11 assertions on 5 structural concerns in one test
- `tests/unit/core/execution/test_ensemble_execution.py:L948-1031` — mixes functional and performance assertions

**Pattern:** Assertion Roulette / Eager Test.

**Tradeoff:** Optimizes for coverage density at the expense of diagnostic clarity.

**Question:** What would change about maintenance cost if multi-scenario tests were split into one-scenario-per-method?

**Stewardship:** Split tests with multiple logical scenarios. Add assertion messages. Separate performance assertions into `@pytest.mark.performance`.

---

#### Finding: BDD Tests Define Local Classes Instead of Importing Production Code

**Observation:** `test_adr_003_testable_contracts.py` defines `ScriptContract`, `ScriptMetadata`, `TestCase`, and other classes inside the test file itself. The BDD scenarios pass because the test's own implementations satisfy its own ABCs.
- `tests/bdd/test_adr_003_testable_contracts.py:L19-99` — classes defined in test file, not imported from `llm_orc.*`
- `tests/bdd/features/adr-003-testable-contracts.feature:L42-50` — scenarios validate test-local models

**Pattern:** Self-referential test fixtures / Test-code correspondence gap.

**Tradeoff:** Optimizes for "green BDD" at the expense of verifying architectural constraints in shipped code.

**Question:** What would change in the CI signal if these tests deleted local definitions and imported from `llm_orc.contracts`?

**Stewardship:** Replace local definitions with production imports. This converts green-but-meaningless tests into tests that drive real integration.

---

#### Finding: Zero Adversarial Tests for `eval()` Sandbox Security

**Observation:** Neither `eval()` site has tests for sandbox bypass. The test file for `evaluator.py` covers only runtime exceptions (undefined variable, divide-by-zero) — functional paths — but contains zero tests attempting namespace escape.
- `tests/unit/core/validation/test_evaluator.py:L167-213` — tests for undefined variables and division by zero only
- `src/llm_orc/core/validation/evaluator.py:L267` — `restricted_globals.update(context)` before `eval()` — context could shadow builtins
- `src/llm_orc/schemas/conversational_agent.py:L118` — `# nosec B307` suppresses scanner without substituting test assurance

**Pattern:** Test-code correspondence gap. The risk surface tested is not the risk surface that matters.

**Tradeoff:** Optimizes for functional regression detection at the expense of security invariant verification.

**Question:** What would a test look like that proves `restricted_globals` cannot be overridden by passing `{"__builtins__": {"__import__": __import__}}` as context?

**Stewardship:** Add `TestEvalSandboxSecurity` with tests for `__builtins__` key injection, `__import__` recovery, and `().__class__.__bases__[0].__subclasses__()` traversal.

---

### Multi-Lens Observations

These findings were independently surfaced by multiple lenses, increasing confidence:

**MCPServer as Hub** (5 lenses converged: Pattern Recognition, Architectural Fitness, Dependency & Coupling, Structural Health, Dead Code)
The single strongest signal in the codebase. Five independent analyses flagged MCPServer's dual role as protocol adapter and application service. The web API's 16+ private method calls, the `_handle_set_project` partial mutation, and the 10+ handler composition root all point to the same structural problem: a missing application service layer.

**`dict[str, Any]` as Universal Protocol** (4 lenses: Pattern Recognition, Architectural Fitness, Dependency & Coupling, Structural Health)
Four lenses independently identified the untyped dict protocol as an architectural constraint. The 217 occurrences, the orphaned Pydantic `Event` class, and the contrast with the typed `ScriptAgentInput`/`ScriptAgentOutput` at the script boundary all converge on the same recommendation: extend the existing Pydantic discipline inward.

**Dead Code Constellation** (4 lenses: Pattern Recognition, Intent-Implementation Alignment, Dead Code, Test Quality)
`communication/protocol.py`, `ConversationalEnsembleExecutor`, `core/events/` Pydantic hierarchy, `ScriptContract` ABC, and `ScriptUserInputHandler.handle_input_request` form a pattern of complete-but-unreachable features. Multiple lenses flagged that tests pass on dead code, creating false confidence. The constellation tells an evolutionary story: the system has been through at least two major architectural pivots (message-passing to state-accumulator, Pydantic events to dataclass events) without cleaning up the superseded designs.

**Silent Failure / Observability Gap** (3 lenses: Architectural Fitness, Invariant Analysis, Structural Health)
Silent exception suppression, silent agent erasure, advisory-only schema validation, and invisible model fallback were independently surfaced by three lenses. The pattern is consistent enough to be a design philosophy rather than oversight — but the philosophy trades result integrity for availability without making the tradeoff visible to callers.

**`eval()` Security** (4 lenses: Architectural Fitness, Invariant Analysis, Structural Health, Test Quality)
The dual `eval()` paths, the context-merge vulnerability, and the complete absence of adversarial tests were flagged by four lenses. The `# nosec` annotations suppress tooling without substituting human judgment.

**Event System Dual Shape** (4 lenses: Pattern Recognition, Intent-Implementation Alignment, Documentation Integrity, Dead Code)
The Pydantic `Event` hierarchy and the raw-dict streaming events use different key names for the same concept (`"event_type"` vs `"type"`). The `visualization/events.py` dataclass system is a third shape. Four lenses independently identified this as dead/orphaned design that misleads readers.

---

## Stewardship Guide

### What to Protect

1. **Three-port delivery architecture.** The CLI/MCP/Web separation at the directory level is sound and worth preserving. The problem is the missing middle layer, not the port structure.

2. **Pydantic schema discipline at the script boundary.** `ScriptAgentInput`/`ScriptAgentOutput` demonstrate the team knows how to define typed contracts. This pattern is the model for extending type safety inward.

3. **Comprehensive test suite with BDD layer.** 90% coverage floor, 60+ pytest markers, 9 ADR-tagged feature files — the testing infrastructure is extensive. The problems are in what the tests verify, not in the infrastructure itself.

4. **ADR practice.** Ten documented ADRs with clear context/decision/consequences structure. The practice is valuable even where individual ADRs have drifted from implementation.

5. **Extracted execution components.** The 15+ extracted classes in `core/execution/` represent real domain boundaries (fan-out coordination, phase monitoring, dependency resolution). The extraction was valuable; the remaining problem is wiring, not decomposition.

### What to Improve (Prioritized)

1. **Extract application service layer** — Highest impact. Create an `OrchestraApplicationService` that both MCPServer and the web API route through. This dissolves the private-method coupling, enables independent port testing, and creates a stable boundary for future ports. *(Findings: MCPServer as Hub — 5 lenses converged)*

2. **Fix `eval()` context-merge vulnerability** — Highest urgency (security). Change `restricted_globals.update(context)` to pass context as `locals` parameter: `eval(assertion, restricted_globals, execution_context)`. This is a 4-character fix with meaningful security improvement. Add adversarial tests immediately. *(Findings: eval() Security — 4 lenses converged)*

   > **Done (2026-02-21):** `restricted_globals.update(context)` removed; context now passed as `locals` argument to `eval()` in `evaluator.py`. `TestEvalSandboxSecurity` class added to `tests/unit/core/validation/test_evaluator.py` with 5 adversarial tests covering `__builtins__` injection, `__import__` unavailability, and builtin shadowing.

3. **Introduce typed result models** — High impact. Define `AgentResult`, `PhaseResult`, `ExecutionResult`, `PerformanceEvent` as dataclasses or Pydantic models. Migrate the execution pipeline incrementally, starting with `AgentResult` (the most-trafficked shape). *(Findings: dict[str, Any] — 4 lenses converged)*

4. **Make silent failures visible** — High impact. At each of the 8 `except Exception: pass` sites, add appropriate handling: `WARNING` logs for optional operations, structured error events for caller-affecting operations, synthetic failure records for agent erasure. Add `model_substituted: bool` to execution results. *(Findings: Silent Failure — 3 lenses converged)*

   > **Done (2026-02-21):** All 8 silent suppression sites resolved. `agent_dispatcher.py` now constructs synthetic `status: "failed"` records for `BaseException` cases, ending silent agent erasure. Artifact saving, streaming merger, request processing, model fallback, and progress-controller UI failures all log at appropriate levels. `model_substituted` flag remains future work. Loggers added to `ensemble_execution.py`, `phase_result_processor.py`, `agent_dispatcher.py`.

5. **Delete dead code constellation** — Medium impact, high clarity. Remove `communication/protocol.py`, `core/events/base.py` + `script_interaction.py` + `script_schemas.py`, the `testing/contract_validator.py` stub, and their associated test files. Mark `ConversationalEnsembleExecutor` as experimental or delete it. *(Findings: Dead Code — 4 lenses converged)*

   > **Update (2026-02-20):** ADR-011 deletes the `ConversationalEnsembleExecutor`, `conversational_agent.py` schemas, and their tests. Remaining dead code (`communication/protocol.py`, orphaned events hierarchy, `ScriptContract` ABC) is future work.

   > **Done (2026-02-21):** `core/events/` Pydantic hierarchy (`base.py`, `script_interaction.py`, `script_schemas.py`, `__init__.py`) deleted. `testing/contract_validator.py` stub (hollow `validate_all_scripts` implementation) and `testing/__main__.py` deleted along with their 5 test files (`tests/unit/core/events/test_base.py`, `test_script_interaction_events.py`, `test_script_schemas.py`, `tests/unit/testing/test_contract_validator_cli.py`, `tests/integration/test_adr_003_integration.py`). `communication/protocol.py` was already absent.

6. **Fix version fossil** — Low effort, immediate value. Replace `__version__ = "0.3.0"` with `importlib.metadata.version("llm-orchestra")`. *(Findings: Version Drift — 2 lenses converged)*

   > **Done (2026-02-21):** `src/llm_orc/__init__.py` now uses `importlib.metadata.version("llm-orchestra")` with a `PackageNotFoundError` fallback guarded by `# pragma: no cover`.

7. **Align documentation with reality** — Medium effort. Update `architecture.md` to distinguish implemented from planned features. Update ADR-010 status. Fix `BaseModel` to `ModelInterface`. Align coverage threshold in `coding-standards.md` with `pyproject.toml`. *(Findings: Documentation — 4 findings)*

   > **Done (2026-02-21):** `architecture.md` API Integration section retitled (WebSocket/batch marked as not yet implemented); `Provider Plugins` section corrected to `ModelInterface` and Registration System marked as not yet implemented. `coding-standards.md` coverage threshold updated from 95% to 90% to match `pyproject.toml`. ADR-010 status changed from "Proposed" to "Implemented".

8. **Improve test quality** — Ongoing. Split Assertion Roulette tests. Replace BDD local class definitions with production imports. Add cache-hit validation tests. Add `eval()` sandbox adversarial tests. *(Findings: Test Quality — 7 findings)*

   > **Partial (2026-02-21):** `test_adr_003_testable_contracts.py` now imports `ScriptContract`, `ScriptMetadata`, `TestCase`, `ScriptCapability`, `ScriptDependency` from `llm_orc.contracts` instead of defining them locally. `AgentExecutor` renamed to `AgentResourceMonitor` (file, class, and test). `eval()` sandbox adversarial tests added. Assertion Roulette in `test_ensemble_execution.py` and cache-hit validation tests remain future work.

### Ongoing Practices

- **Review new modules for port coupling before merging.** Any new web route should import from the application service layer, not from MCPServer.
- **Require typed models for new dict shapes.** When adding a new data contract to `core/execution/`, define a dataclass first.
- **Add assertion messages to tests touching the execution pipeline.** The 202-assertion test file is the highest-traffic test; every new assertion in it should carry a message.
- **Run `vulture` on CI to detect dead code.** The codebase already has `vulture` in its Makefile lint target — ensure it catches the patterns identified here.
- **Test the security boundary, not just the happy path.** When `eval()` or other restricted execution is involved, adversarial tests are not optional — they are the primary verification.
- **Update ADR status when implementing.** The ADR-010 "Proposed" → implemented-but-unlabeled gap should not recur.
