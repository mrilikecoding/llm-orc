# #182 slice D — live rows through real OpenCode (2026-09-11)

Serve started from this branch's worktree (tip 70ca6ecf) on :8765; the
probe's seeded package (`docs/plans/2026-09-11-daily-driver-probe/seed-repo.tgz`)
reset to its seed commit; `opencode run --format json -m llm-orc/agentic`,
fresh session per row, workspace truth captured after each.

| row | ask | events on the wire | outcome | truth |
|---|---|---|---|---|
| 08 | the probe's turn-6 ask verbatim (priority feature across storage, CLI, tests; no file named) | `glob **/*py*` → listing (todo/storage.py, todo/cli.py, todo/__init__.py, tests/test_storage.py, pyproject.toml) → prose | "Build refused: multiple files could match this ask: …/todo/storage.py, …/todo/cli.py — please name one" | zero `write`; workspace clean; 3 passed. **Turn 6's harm converted**: on main this ask minted and wrote `solution.py`. |
| 09 | "Add a remove method to TodoStore that deletes a todo by id and raises KeyError when the id is missing." (no file named) | `glob **/*py*` → unique match `todo/storage.py` → `read` → build-gated | "Another round needed: tests did not pass" | zero `write`; workspace clean. **Discovery converted** (glob → match → read fired); the build itself fails on the edit-fragment class, which is slice B-1. |

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
