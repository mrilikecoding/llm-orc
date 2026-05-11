# Spike C — Arm C (frontier-bare) trial 2

**Subagent ID:** a5f0e52cd34e6ad65
**Total tokens (subagent reported):** 16346
**Duration (subagent reported):** 42404ms
**Tool uses:** 0 (single-shot facsimile)

## Code review response (verbatim)

## Code Review: `src/llm_orc/agentic/session_budget.py`

---

### 1. Logic / Semantic Issues

**Line 52 — Off-by-one in `check_limit()`**

```python
if self.total_used() > self.limit:
```

The condition uses strict greater-than, meaning a session that has consumed exactly `limit` tokens is still considered within budget and will proceed to the next ReAct iteration. Whether that is correct depends on the contract, but it is almost certainly wrong: a session at exactly the limit should not initiate another LLM call. Use `>=`.

**Line 71 — `limit=None` default with falsy guard**

```python
def register(self, session_id: str, api_key: str, limit: int = None) -> SessionBudget:
    ...
    limit=limit if limit else DEFAULT_BUDGET_LIMIT,
```

Two problems here. First, `limit=0` is a valid sentinel value meaning "no tokens allowed" and would silently be replaced with `DEFAULT_BUDGET_LIMIT` due to the falsy check. Callers passing `limit=0` to create a blocked session would get a 100k-token budget instead. Use `limit if limit is not None else DEFAULT_BUDGET_LIMIT`. Second, see type-safety section below.

**Lines 77–79 — `get()` raises `KeyError` on unknown session**

```python
def get(self, session_id: str) -> SessionBudget:
    return self._budgets[session_id]
```

`add_usage()` calls `get()` unconditionally. If the Orchestrator Runtime calls `add_usage()` before `register()` (or after a registry restart, since this is in-memory only), the unhandled `KeyError` will surface as an uncontrolled exception at the iteration boundary. The method should either return `SessionBudget | None` or raise a typed, descriptive exception with `from e` chaining.

**Lines 82–84 — `add_usage()` post-hoc check**

`add_usage()` records tokens first, then calls `check_limit()`. This means the usage is always committed even when the session is over budget. If the Orchestrator Runtime only calls `add_usage()` after a completed LLM call, that may be acceptable — but the docstring says this is called "at each ReAct iteration boundary before continuing," which implies a pre-call check. The current design cannot enforce a pre-call limit; it can only detect overage after the fact. The contract between this module and the Runtime needs to be explicit.

**Line 88 — `all_sessions()` returns a live reference**

```python
def all_sessions(self) -> dict[str, SessionBudget]:
    return self._budgets
```

This hands out a direct reference to the internal dict. Any caller can insert or delete sessions, bypassing `register()` logic. Return a shallow copy: `return dict(self._budgets)`.

**Line 60 — `remaining()` can return a negative value**

```python
def remaining(self) -> int:
    return self.limit - self.total_used()
```

After the budget is exceeded, this returns a negative integer. If any caller uses `remaining()` as a capacity signal (e.g., to size the next request), a negative value is a logic hazard. At minimum, clamp: `return max(0, self.limit - self.total_used())`. Better to document the negative-return behavior explicitly if intentional.

---

### 2. Security / Sensitive-Data Issues

**Lines 53–57 — API key written to warning log**

```python
logger.warning(
    f"Session {self.session_id} exceeded budget: "
    f"{self.total_used()}/{self.limit} tokens. "
    f"API key: {self.api_key}"
)
```

The API key is emitted verbatim into the log on every budget-exceeded event. This is a credential-leak vulnerability. Log aggregators, stdout captures, and observability pipelines will ingest and potentially index this value. Remove `api_key` from the log line entirely. If the caller needs to associate billing identity with the event, log a truncated prefix (`api_key[:8] + "..."`) or a non-reversible hash.

**Lines 28–32 — `api_key` stored as plain string in dataclass**

`SessionBudget` is a dataclass with `api_key: str` as a public field and no masking on `__repr__`. Python's default `__repr__` for dataclasses prints all fields, so any debug log, exception traceback, or repr-based serialization of a `SessionBudget` instance will expose the raw key. Add `repr=False` to the `api_key` field or implement a custom `__repr__`.

**Lines 64–66 — `get_metadata()` is redundant and documents no access control**

The `metadata: dict[str, Any]` field is public. `get_metadata()` adds no encapsulation. If metadata can carry sensitive billing or identity information, direct public access to `self.metadata` (and the mutable dict it returns) is a risk worth documenting at minimum.

---

### 3. Type-Safety / API-Contract Issues

**Line 70 — `limit: int = None` violates annotation**

```python
def register(self, session_id: str, api_key: str, limit: int = None) -> SessionBudget:
```

`None` is not an `int`. mypy strict mode will reject this. The correct signature is `limit: int | None = None`.

**Line 14 — `from typing import Any` is the only typing import**

`Any` is appropriate for `metadata`, but the module is missing `__all__` and does not export a typed interface. If the Orchestrator Runtime imports from this module, there is no enforced public surface.

**Lines 44–46 — `add_tokens()` does not validate negative inputs**

```python
def add_tokens(self, input_count: int, output_count: int) -> None:
    self.input_tokens_used += input_count
    self.output_tokens_used += output_count
```

There is no guard against negative values. A caller passing negative token counts (e.g., from a malformed provider response) would silently decrease `total_used()`, potentially un-tripping a budget that was already exceeded. Add a guard or assert non-negativity.

**Lines 82–84 — `add_usage()` return type is `bool` but callers get no error signal**

If `get()` raises `KeyError`, the `bool` return type is not honored — the caller gets an exception instead. The return type contract is incomplete without documenting the exception path.

---

### 4. Test Coverage

The commit message explicitly defers all tests to a follow-up PR, which the project's CLAUDE.md rules prohibit: "Only commit when: ALL tests pass." This is a process violation, not just a style preference.

Beyond the process issue, the deferred tests would need to cover at minimum:

- `check_limit()` at exactly the limit boundary (the off-by-one identified above would be caught here)
- `register()` with `limit=0` to catch the falsy-guard bug
- `get()` with an unknown `session_id`
- `add_usage()` called before `register()`
- `add_tokens()` with negative values
- `all_sessions()` mutation isolation
- The API key not appearing in any log output (can be asserted with `assertLogs`)

The characterization in the commit message — "budget logic is straightforward" — understates the number of edge cases present in 90 lines of code.

---

### 5. Cross-File Consistency

**Lines 20–22 — Silent drift risk with `orchestrator_config.py`**

```python
# Hard-coded budget threshold. Should match
# DEFAULT_MAX_TOKEN_LIMIT in orchestrator_config.py — if these drift,
# behavior across the two enforcement paths will diverge silently.
DEFAULT_BUDGET_LIMIT = 100_000
```

The comment acknowledges a known consistency hazard and then leaves it in place. If `orchestrator_config.py` already owns `DEFAULT_MAX_TOKEN_LIMIT`, this module should import that constant rather than re-declaring it. Two sources of truth for the same limit will diverge.

**No `__init__.py` update shown**

The diff adds a new module but does not show `src/llm_orc/agentic/__init__.py` being updated to export `SessionBudget` and `SessionBudgetRegistry`. If the package's public API is curated through `__init__.py`, this is an incomplete change.

---

### 6. Other Concerns

**Registry is not thread-safe**

`self._budgets` is a plain dict. Concurrent ReAct iterations from different sessions (or the same session if parallelism is added later) will produce data races on `add_tokens()` and `register()`. At minimum, document the single-threaded assumption. If the agentic layer runs with async concurrency (`asyncio`), a plain dict is fine for the event loop but the combined read-modify-write in `add_usage()` is still not atomic across `await` points.

**No session eviction or TTL**

The registry is in-memory with no eviction. Long-running services will accumulate `SessionBudget` objects for every session ever created. Document the expected lifecycle or add explicit `deregister(session_id)` support. The `all_sessions()` method suggests the registry is expected to be iterated, which compounds the unbounded-growth concern.

**`get_metadata()` adds no value over direct field access**

`self.metadata` is a public field. The method is dead weight and will confuse readers about whether the field is supposed to be accessed directly or through the method.
