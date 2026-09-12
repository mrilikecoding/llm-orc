# #182 slice D — live rows through real OpenCode (2026-09-11)

Serve started from this branch's worktree (tip 70ca6ecf) on :8765; the
probe's seeded package (`docs/plans/2026-09-11-daily-driver-probe/seed-repo.tgz`)
reset to its seed commit; `opencode run --format json -m llm-orc/agentic`,
fresh session per row, workspace truth captured after each.

| row | ask | events on the wire | outcome | truth |
|---|---|---|---|---|
| 08 | the probe's turn-6 ask verbatim (priority feature across storage, CLI, tests; no file named) | `glob **/*py*` → listing (todo/storage.py, todo/cli.py, todo/__init__.py, tests/test_storage.py, pyproject.toml) → prose | "Build refused: multiple files could match this ask: …/todo/storage.py, …/todo/cli.py — please name one" | zero `write`; workspace clean; 3 passed. **Turn 6's harm converted**: on main this ask minted and wrote `solution.py`. |
| 09 | "Add a remove method to TodoStore that deletes a todo by id and raises KeyError when the id is missing." (no file named) | *stale — measured against the pre-rework prefix-matching MATCH step; not yet re-driven* | **Expectation updated, not yet re-measured**: an independent review found the prefix-matching MATCH step could uniquely (and wrongly) resolve to an unrelated file on other inputs, so it was reworked to exact-equality-only matching. Under the reworked rule `TodoStore` → `{todo, store}` does not equal `storage.py`'s stem/components exactly, so this ask now refuses naming the listing (exact-match rule) rather than uniquely matching `todo/storage.py`. To be re-driven live after merge. | *pending re-drive* |

Bounds observed live (recorded, not closed):

- The refusal quotes the paths exactly as the client's glob returned them
  — absolute, under the client's own workspace root. They are the user's
  paths (not the server's, so #168's discipline holds), but relative
  paths would read better; the existing stem-match refusal has the same
  shape.
- `**/*py*` is the stem seam's pattern for the stem `py`; it also lists
  `pyproject.toml` (filtered downstream as non-`.py`) and would list any
  `*py*`-named non-Python file (`mypy.ini`, `pytest.ini`, untracked
  `.pyc`) against the 50-path cap. OpenCode's glob honoured `.gitignore`
  here (no `__pycache__` entries).

## Rows on merged main `a232f9a7` (#171 + D rework + B-1), fresh sessions, seed reset before each

| row | ask | events | outcome |
|---|---|---|---|
| 10 | the probe's turn-2 ask verbatim ("Add a remove(todo_id) method to the TodoStore class in todo/storage.py ...") | `write todo/storage.py` — NO read | **BLIND OVERWRITE**: a new in-memory `TodoStore` replaced the client's JSON-backed one; gate-accepted against its own tests; `M todo/storage.py`. B-1's guard cannot fire without a prior in context, and "add ... in F" carries no fix/update verb so no read is requested. Filed **#185**. |
| 11 | "Add a remove method to TodoStore that deletes a todo by id and raises KeyError when the id is missing." (no file named) | `glob **/*py*` → prose | "Build refused: no file in the workspace matches this ask — the workspace holds: …/todo/storage.py, …/todo/cli.py, …" — the exact-match rule; nothing written. Row 09's expectation converted. |
