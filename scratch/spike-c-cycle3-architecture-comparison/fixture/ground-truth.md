# Spike C Fixture — Ground Truth Issue Set

**Fixture:** `diff.patch` introducing `src/llm_orc/agentic/session_budget.py` (90 lines).
**Pre-spike validation:** This fixture has 8 deliberately-injected issues across 5 categories where cheap-orchestrator-alone (Arm A) is expected to struggle on at least 4 of them. Validated by inspection 2026-05-01 before any spike trials run.

## Issue inventory

### CATEGORY 1 — Semantic / logic bug

**ISSUE-1: Off-by-one in `check_limit()` (line 56)**

```python
if self.total_used() > self.limit:
    ...
    return False
return True
```

The condition uses `>` (strictly greater), so `total_used() == limit` returns `True` (within budget). For a hard limit semantics, this should be `>=` (at-or-over the limit fails). A session at exactly `limit` tokens would be allowed to make ONE more call before the limit fires — which violates the ReAct iteration boundary contract the docstring describes ("Called at each ReAct iteration by the Orchestrator Runtime").

**Detection difficulty:** Subtle — requires reading the boundary semantics carefully. Cheap-tier-alone may miss this without prompting toward boundary-condition checks.

### CATEGORY 2 — Security / sensitive-data leak

**ISSUE-2: Logging the api_key when budget exceeded (line 53)**

```python
logger.warning(
    f"Session {self.session_id} exceeded budget: "
    f"{self.total_used()}/{self.limit} tokens. "
    f"API key: {self.api_key}"
)
```

The api_key is included in the log message. This is a sensitive credential and should NEVER appear in logs. Standard practice is to log a hash, prefix-only (first 4 chars + `***`), or omit entirely. The logger is configurable and may route to disk, monitoring systems, or external observability services — any of which expand the credential's exposure surface.

**Detection difficulty:** A security-aware reviewer should catch this immediately. Cheap-tier-alone may miss it if not prompted toward security review.

### CATEGORY 3 — Type-safety / API-contract issue

**ISSUE-3: `register()` parameter `limit: int = None` is mistyped (line 79)**

```python
def register(self, session_id: str, api_key: str, limit: int = None) -> SessionBudget:
```

The annotation says `limit: int` but the default is `None`. This should be `limit: int | None = None`. Strict mypy will flag this. Worse, the body uses `limit if limit else DEFAULT_BUDGET_LIMIT` — which fires the default ALSO for `limit=0`, which a caller might pass intentionally to mean "no usage allowed." The truthy-check fallback conflates `None` with `0`.

**Detection difficulty:** Type-checker would flag the annotation; the truthy-conflation requires careful reading.

### CATEGORY 4 — Test gap (explicit)

**ISSUE-4: PR explicitly says "tests will be added in follow-up" (commit message + module docstring)**

The diff message says: *"Tests for the new module will be added in a follow-up PR — the budget logic is straightforward and the registry is in-memory only, so test coverage is not blocking on this change."* The module docstring repeats: *"Tests for this module will be added in a follow-up PR."*

The project's `CLAUDE.md` (and standard practice) requires TDD: tests written before/with code, not after. Shipping new logic without tests — especially logic that gates budget enforcement at every ReAct iteration — is a quality regression. The "logic is straightforward" defense is exactly the kind of justification that historically precedes regressions.

**Detection difficulty:** Both the diff message and module docstring make this explicit; any reviewer reading those should flag it. **This issue is the easiest to detect — it's stated in plain text.** Useful as a "floor" check.

### CATEGORY 5 — Cross-file consistency

**ISSUE-5: `DEFAULT_BUDGET_LIMIT = 100_000` should match `DEFAULT_MAX_TOKEN_LIMIT` in `orchestrator_config.py` but doesn't**

The module's own comment says: *"Should match DEFAULT_MAX_TOKEN_LIMIT in orchestrator_config.py — if these drift, behavior across the two enforcement paths will diverge silently."*

Looking at the actual `orchestrator_config.py` in the project: `DEFAULT_MAX_TOKEN_LIMIT = 50_000_000` (50 million). The new module's `DEFAULT_BUDGET_LIMIT = 100_000` (100K) is **500× smaller**. Either the new module is wildly over-restrictive or `orchestrator_config.py` is wildly over-permissive — they cannot both be correct, and the comment explicitly flags drift as a silent-divergence risk.

**Detection difficulty:** Requires either reading the existing `orchestrator_config.py` file (cross-file reasoning) OR noticing the comment's explicit warning + checking the value. **This is the most "ensemble-architecture-favorable" issue** — a script-agent extracting cross-references and a reviewer verifying values would catch it; a single-shot LLM may not because cross-file verification requires multiple file reads and the agent may not pursue the comment's hint.

### MINOR ISSUES (lower priority; flag-worthy but not core)

**ISSUE-6 (minor): No timeout/expiry on budget tracking** — `SessionBudgetRegistry` keeps all session budgets in memory forever. Sessions that abandon (e.g., client disconnects) leak budget records. Production systems would need a TTL or explicit cleanup hook.

**ISSUE-7 (minor): `get_metadata()` returns mutable internal state** — `return self.metadata` returns a reference; callers can mutate it externally. Should `return dict(self.metadata)` to defensively copy.

**ISSUE-8 (minor): `register()` does not check for existing session_id** — calling `register()` twice with the same session_id silently overwrites. May or may not be intended; if not, should raise.

## Expected per-arm coverage (pre-spike hypothesis)

These are hypotheses to validate, not predictions of outcomes. The spike's success criterion is whether the arms produce DIFFERENT coverage patterns, not whether they match any specific prediction.

| Issue | Arm A (cheap-bare) | Arm B (cheap-with-ensemble) | Arm C (frontier-bare) |
|-------|:------------------:|:----------------------------:|:----------------------:|
| ISSUE-1 (off-by-one) | Maybe miss | Likely catch (script-agent boundary check; semantic-consistency reviewer) | Likely catch |
| ISSUE-2 (api_key in logs) | Likely miss without security framing | Likely catch (security reviewer slot) | Likely catch (frontier security awareness) |
| ISSUE-3 (type annotation) | Maybe miss | Likely catch (script-agent: mypy-style check) | Likely catch (frontier type-awareness) |
| ISSUE-4 (test gap) | Should catch (stated in plain text) | Should catch | Should catch |
| ISSUE-5 (cross-file consistency) | **Likely miss** (no cross-file reasoning) | **Likely catch** (script-agent cross-reference extraction) | Maybe catch (frontier may follow the comment's hint) |

**Pre-spike validation passes.** The fixture has at least 2 issues (ISSUE-1, ISSUE-5) where cheap-bare is expected to struggle and the architecture (Arm B) is expected to add value. ISSUE-5 specifically is the architecture-favorable case — script-agent cross-reference extraction is exactly the kind of deterministic verification the architecture's script-agent slot is designed for.

The fixture is APPROPRIATE for the spike's central question.

## Why this fixture passes the reviewer's pre-spike validation criterion

The reviewer's spec required: *"manually audit the diff for the issue categories expected to appear. If the diff has no issues in the categories where cheap-orchestrator-alone is expected to struggle, the fixture should be rejected and a more complex diff selected."*

This fixture has issues spanning 5 categories with at least 2 (ISSUE-1, ISSUE-5) where cheap-bare is expected to miss but ensemble-augmented cheap (Arm B) is expected to catch. ISSUE-2 (security) and ISSUE-3 (type-safety) are intermediate cases where heterogeneous reviewer slots may catch but single-shot cheap may miss. ISSUE-4 (test gap) is the floor — both tiers should catch it; if they don't, that's a separate finding.

This fixture is NOT in the easy regime (where Spike B sat). The fixture deliberately includes issues whose detection requires capabilities (boundary-condition awareness, security framing, type-safety scrutiny, cross-file reasoning) that are not uniformly present at all tiers and architectures.
