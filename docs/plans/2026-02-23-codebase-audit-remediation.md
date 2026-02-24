# Codebase Audit Remediation Plan

**Date:** 2026-02-23
**Source:** docs/codebase-audit.md (2026-02-22 audit)

## Strategy

Three waves. Wave 1 is mechanical cleanup (parallel Sonnet agents). Wave 2 is targeted fixes. Wave 3 is architectural changes with dependency ordering.

Update `docs/codebase-audit.md` as each item completes.

## Wave 1 — Mechanical Cleanup

All items are deletions, renames, or status updates. No judgment calls.

- [x] **E11** Remove broken Makefile targets (`validate-contracts-*`) and `pre-commit` dependency
- [x] **E15** Remove dead dependencies (`websockets`, `aiohttp`) from `pyproject.toml`
- [x] **U6** Delete ghost directories (`core/communication/`, `contracts/`, `testing/`)
- [x] **E12** Update ADR-004 status to "Superseded"
- [x] **E13** Fix reversed principle in ADR README
- [x] **E14** Update ADR 011-014 statuses to "Implemented"; added missing ADRs 008-010 to index
- [x] **E16** Fix README library default claim
- [x] **U13** Clean up pytest markers (remove orphaned ADR-003/005, add 010-014)
- [x] **U7** Remove dead `SUPPORTS_CUSTOM_ROLE_PROMPT` flag from all model files
- [x] **E5** Rename `adaptive_resource_manager.py` → `system_resource_monitor.py`, update imports
- [x] **E6** Remove duplicate `"You are X"` preamble from `DependencyResolver`

## Wave 2 — Targeted Fixes

Self-contained, moderate judgment. Sequential but independent.

- [x] **E10** Add `set_project_context` to `ScriptHandler` and `ArtifactHandler`
- [x] **E9** Emit structured warning when fan-out produces zero instances
- [x] **E7** Distinguish credential decryption errors from missing credentials
- [x] **M5** Align complexity thresholds — already resolved (complexipy in lint target)
- [x] **M7** Register global exception handler on FastAPI app
- [x] **U2** Extract shared `classify_tier()` + consolidate `_dep_name` duplication
- [x] **U4** Extract profile parsing from `PromotionHandler` → `ProfileHandler`
- [x] **U8** Removed dead `EventFactory` code from `ScriptUserInputHandler`
- [x] **U9** Replace `_test_*` escape hatches with constructor injection
- [x] **U14** Make `LLMResponseGenerator` accept `ModelInterface` parameter

## Wave 3 — Architectural Changes

Dependency-ordered. Each change cascades.

- [x] **E8** Add `asyncio.Lock` to `set_project`
- [x] **M3** Delete unused typed event system (`visualization/events.py` + `EventFactory`)
- [x] **U5** Replace `_current_agent_configs` temporary field with typed attribute
- [x] **M4** Triage `except Exception` clauses (Category A: control-flow, Category B: boundary I/O)
- [x] **U12** Add BDD feature files for ADRs 010-014
- [x] **M2** Push `EnsembleExecutor` construction fully into `ExecutorFactory`
- [x] **M1** Route CLI through `OrchestraService`
- [x] **E1** Move handlers from `mcp/handlers/` to `services/handlers/`
- [x] **U15** Rewrite CLI tests to mock `OrchestraService` at the boundary
- [x] **E2/U1** Separate `ConfigurationManager` construction from provisioning
- [x] **E4** Connect `invoke_streaming` to real execution engine
- [x] **E3** Define `TemplateProvider` protocol, inject into ConfigurationManager
- [x] **M6** Add `HTTPConnectionPool.configure()` classmethod, remove hidden ConfigurationManager construction
- [ ] **U3** Reorganize flat `execution/` into sub-packages — **deferred** (high-churn, best as last structural change)
- [ ] **U10/U11** Improve test quality — **deferred** (wait for architecture to stabilize)
