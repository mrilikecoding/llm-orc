# ADR-011: Remove Conversational Ensemble System

**Status:** Accepted

**Date:** 2026-02-20

**Supersedes:** ADR-005

---

## Context

Research (Question 5 of the research log) traced the conversational ensemble system through imports, call sites, and YAML usage. The findings:

- `ConversationalEnsembleExecutor` is never imported by the CLI, MCP server, or any execution path.
- `ConversationalAgent`, `ConversationalEnsemble`, and `ConversationState` schemas are only imported by the dead executor and its tests.
- The `conversation:` key on agents appears in exactly one ensemble file (`neon-shadows-detective` in the library submodule).
- The only consumers are unit tests and one BDD test that exercise the executor in isolation.

The regular execution path already supports user-directed flows through script primitives (`get_user_input.py`, `confirm_action.py`), the `ScriptUserInputHandler`, and conditional dependencies. The conversational system duplicates this with worse integration.

The `conversation` field complicates the upcoming Pydantic agent config migration (ADR-012) by adding a dead field to the config surface area.

---

## Decision

Remove the conversational ensemble system entirely.

**Files to delete:**
- `src/llm_orc/core/execution/conversational_ensemble_executor.py`
- `src/llm_orc/schemas/conversational_agent.py`
- `tests/unit/core/execution/test_conversational_ensemble_executor.py`
- `tests/unit/schemas/test_conversation_state.py`
- `tests/bdd/test_adr_005_multi_turn_conversations.py`

**Files to keep:**
- User input primitives and `ScriptUserInputHandler` (wired into real execution)
- `neon-shadows-detective` ensemble in library submodule (`conversation:` keys are silently ignored by the regular executor)

This ADR supersedes ADR-005. ADR-005's code was never reachable from production paths.

---

## Consequences

**Positive:**
- Removes a confusing parallel system before introducing new agent types
- Eliminates the `conversation` field from the Pydantic migration scope
- Reduces test maintenance burden (dead tests removed)

**Negative:**
- If conversational ensemble patterns are needed later, they must be reimplemented. The current implementation was not integrated, so nothing of value is lost.

**Neutral:**
- The library submodule's `neon-shadows-detective` ensemble continues to work — the regular executor ignores unknown fields. After ADR-012 introduces `extra="forbid"`, that ensemble will need its `conversation:` keys removed.
