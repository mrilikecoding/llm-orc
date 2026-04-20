# Conformance Scan Report

**Scanned against:** `docs/agentic-serving/decisions/adr-001` through `adr-011`
**Codebase:** `/Users/nathangreen/Development/eddi-lab/llm-orc/src/llm_orc/`
**Date:** 2026-04-17

---

## Summary

Two genuine structural debt items were found. Both concern ADR-006 (reference-graph validation accessible as a library routine). The cross-ensemble cycle-detection logic exists and is correct, but it is locked behind private methods on `EnsembleLoader` and is skipped entirely in the primary validation code path used by the MCP `validate_ensemble` tool and the web API. Everything else scanned clean: no Plexus coupling that makes Layer 4 mandatory (ADR-002), no hardcoded orchestrator model selection (ADR-011), no existing unsummarized result path into any LLM context (ADR-004 — no orchestrator exists yet so no violation is possible), no context-injection-at-session-start pattern (ADR-009 Phase 2 not started), and no LLM-summary ingestion path (ADR-010). The ADR numbering collision noted below is an annotation concern, not a code defect.

---

## Debt Table

| ADR | Severity | File:line | Description | Suggested Remediation |
|-----|----------|-----------|-------------|----------------------|
| ADR-006 | **High** | `src/llm_orc/services/handlers/validation_handler.py:67` | `ValidationHandler._collect_validation_errors` calls only `assert_no_cycles` (intra-ensemble dependency check). It does not call cross-ensemble reference-graph validation. A composed ensemble that introduces a cross-ensemble cycle will pass `validate_ensemble` via the MCP and web API paths but violate Invariant 5 at load time inside the executor. ADR-006 requires the composition-time validator to be the same routine as the load-time validator to prevent divergence. | Extract `_validate_cross_ensemble_cycles` and `_build_reference_graph` from `EnsembleLoader` into a public, standalone function (e.g., `validate_ensemble_reference_graph(config, search_dirs)` in `ensemble_config.py`). Call it from `ValidationHandler._collect_validation_errors`, passing the ensemble search dirs from `config_manager.get_ensembles_dirs()`. This makes the same logic reachable by the future `compose_ensemble` tool without duplication. |
| ADR-006 | **Medium** | `src/llm_orc/core/config/ensemble_config.py:218` | `EnsembleLoader.list_ensembles` calls `self.load_from_file(str(yaml_file))` with no `search_dirs` argument. Cross-ensemble cycle detection inside `load_from_file` is therefore silently skipped for all ensembles loaded through the listing path (which is also the path used by `find_ensemble` → `ValidationHandler`). The cross-ensemble guard only fires when a caller explicitly knows to pass `search_dirs`, which today only happens inside `EnsembleExecutor._setup_ensemble_agent_runner` via `_resolve_ensemble_reference`. | When calling `load_from_file` from within `list_ensembles`, pass the listing directory as a single-element `search_dirs` list so the cross-ensemble check runs at load time. Alternatively, defer to the extracted public validator (see High item above) and call it after load. Either approach closes the silent skip. |

---

## Non-Findings Worth Flagging

**ADR-002 — Plexus coupling.** Searched all `.py` files under `src/llm_orc/` for any import of Plexus or hard dependency on it. The only reference is a comment in `src/llm_orc/mcp/__init__.py:7` that cites "ADR-009: MCP Server Architecture and Plexus Integration" — this refers to the older project-level `docs/adrs/009-mcp-server-architecture.md`, not the agentic-serving ADR-009. No runtime Plexus coupling exists. Layer 4 is genuinely optional today.

**ADR-011 — Hardcoded orchestrator model selection.** The `ClaudeModel` and `ClaudeCLIModel` constructors in `src/llm_orc/models/anthropic.py:22,91` have a default `model="claude-3-5-sonnet-20241022"` parameter, but this is a constructor default for direct instantiation — `ModelFactory.load_model_from_agent_config` resolves everything through `model_profile` or explicit `model+provider` fields from config, never from a hardcoded string. No orchestrator-specific model selection exists anywhere. The profile system is general-purpose and applicable to any agent consumer.

**ADR-004 — Unsummarized results into LLM context.** `EnsembleExecutor.execute` returns `final_result.to_dict()` (full result dictionary), and `ExecutionHandler.invoke` passes that back to the MCP caller. There is no orchestrator LLM context to inject into yet, so no violation path exists today. The concern is forward-looking: the `compose_ensemble` tool and orchestrator control plane must intercept this return before passing it to the orchestrator's conversation context. Current code is neutral.

**ADR-005 — DAG engine recursion depth.** `EnsembleAgentRunner` enforces a configurable `depth_limit` (default 5, configurable via `performance_config.execution.max_ensemble_depth`). This is consistent with ADR-005's session-level budget enforcement — the two mechanisms operate at different layers (nesting depth vs. session turn/token budget) and do not conflict.

**ADR-009 — Context injection at session start (Phase 2).** No session-start context injection code exists. Phase 2 is correctly deferred.

**ADR-010 — LLM-summary ingestion.** No ingestion code of any kind exists in the codebase. No violation possible.

**Depth limit constant.** The default value of 5 in `EnsembleAgentRunner.__init__` (line 34) is also the fallback when `max_ensemble_depth` is absent from config. This is a minor robustness gap but not an ADR violation.

---

## Annotation: ADR Numbering Collision

The agentic-serving cycle produced `adr-009-plexus-integration-tool-first.md` and `adr-011-orchestrator-llm-is-a-model-profile.md` using numbers 001–011 within the `docs/agentic-serving/decisions/` namespace. The project already has `docs/adrs/009-mcp-server-architecture.md` through `014-input-key-selective-upstream-consumption.md` in a separate namespace. Code comments (e.g., `mcp/server.py:3`, `core/config/ensemble_config.py:203`) reference project-level ADR numbers like "ADR-009" and "ADR-013" that collide with the agentic-serving ADR numbering. This is not a code defect — the two ADR namespaces are in separate directories and do not overlap in meaning — but any future code comment referencing an agentic-serving ADR by number (e.g., `# ADR-006`) will be ambiguous without a namespace qualifier (e.g., `# agentic-serving/ADR-006`). Worth establishing a comment convention before agentic-serving code is written.

---

## Recommended Scenarios

The following items should become `refactor:` scenarios in `scenarios.md` before the compose_ensemble tool is built. They are structural preparations, not feature work.

1. **Extract public reference-graph validator.** Promote `EnsembleLoader._validate_cross_ensemble_cycles` and `_build_reference_graph` into a public, free function in `ensemble_config.py`. Update `ValidationHandler._collect_validation_errors` to call it with real search dirs. Update `list_ensembles` to pass search dirs at load time. Scenario: "given an ensemble that introduces a cross-ensemble cycle, `validate_ensemble` via MCP returns a validation error before the orchestrator attempts to invoke it." This is a pure behavior-correct refactor — no new capability, just closing a gap between load-time and validate-time enforcement.

2. **Verify `ValidationHandler` uses cross-ensemble graph check.** Add a test that calls `validate_ensemble` via `OrchestraService` with a pair of ensembles that reference each other cyclically. The test should fail today (validating the debt is real) and pass after the refactor above.
