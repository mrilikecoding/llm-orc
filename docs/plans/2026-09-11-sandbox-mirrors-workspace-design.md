# The gate sandbox mirrors the workspace — arc 2 of the existing-repo shape

Lead design brief v1, 2026-09-11 (evening). Subsumes #182 slice A and
#184. Evidence: probe turns 1-3 (`docs/plans/2026-09-11-daily-driver-probe/`,
tests written as `from storage import` for a `todo/` package, destination
flattened to `test_storage.py`) and ladder turn 7 in 3/3 think-on runs
(`docs/plans/2026-09-11-ladder-runs-171-thinkoff/`: "code failed to load:
No module named 'storage'" reported on a correct shipped fix because the
re-fix sandbox holds no workspace files).

## Invariant

Every gated route judges the deliverable in a sandbox whose layout is the
client workspace's: conversation-written and client-read files sit at
their real relative paths, the deliverable sits at its real destination,
and the sandbox root is importable. A flat workspace is the subcase where
no path has a directory; the 13-turn ladder and the 4255 existing pins
are that subcase and must not move.

## Mechanism

1. **One workspace reader.** `accept_gather._workspace` keys files by
   RELATIVE PATH, not basename (the `[read <path>]` / `[wrote <path>]`
   headers already carry the path the client used; strip the client's
   absolute prefix when present — the prefix is the longest common
   directory of all headers that share it, recorded once). It becomes a
   shared helper in `_helpers.py` used by `accept_gather`, `tests_gather`
   (write-tests), and `refix_gather` (re-fix — today it materializes
   nothing, #184).
2. **Materialization at real paths.** `accept_executor._materialize`
   writes each workspace file at `tmp/<relpath>` (creating directories;
   reject any path that escapes the root or is absolute — refuse, do not
   sanitize), writes the deliverable at `tmp/<target_path>` AND keeps
   `solution.py`/`tests.py` for the runner's namespace model unchanged.
   The #171 control shadows the deliverable with empty bytes at the real
   destination too.
3. **Importable root.** The runner child's `sys.path[0]` is the sandbox
   root (it is `cwd=tmp` already; pin that `import todo.storage` resolves
   when `todo/__init__.py` was read or written; a package directory
   without `__init__.py` still imports as a namespace package).
4. **Re-fix gets the workspace** (#184): `refix_gather` emits the same
   workspace map; `accept_executor` on the re-fix route materializes it.
   A candidate importing a sibling the conversation wrote now loads.
5. **Destinations keep directories.** classify's tests destination for a
   named source `todo/storage.py` becomes `tests/test_storage.py` when
   a `tests/` directory exists in the listing, else `test_storage.py`
   beside it — deterministic; and the test-writer's import guidance names
   the module by its real dotted path when the source has a directory
   (`from todo.storage import ...`). This is the only prompt change.

## Bounds (record)

- Files the client never showed are not in the sandbox; a module
  importing an unseen sibling still fails to load, honestly.
- The listing may show a file the conversation has not read; the sandbox
  holds only read/written bodies (arc 1 decides what gets read).
- Windows separators out of scope.

## Instruments (red first; drive the real scripts; mutant per pin)

1. Probe turn-1 shape: workspace `todo/__init__.py` + `todo/storage.py`
   read, tests `from todo.storage import TodoStore`, a whole-file
   deliverable to `todo/storage.py` adding `remove` → accept; mutant:
   basename keying restored → red (ImportError).
2. Ladder turn-7 shape on the re-fix route: `storage.py` conversation-
   written, candidate `from storage import save_todos` → loads; the
   report is the real test result, not ModuleNotFoundError; mutant:
   re-fix materializes nothing → red.
3. Flat subcase: every existing gate/participation/refix pin unchanged.
4. Path safety: a header `[read ../etc/passwd]` or an absolute path that
   does not share the recorded prefix → refused, nothing materialized.
5. Destination: named source `todo/storage.py` with `tests/` in the
   listing → `tests/test_storage.py`; without → `test_storage.py`.
6. #171 control still refuses a non-participating deliverable when the
   destination is nested.
7. Reason hygiene: no sandbox path in any reason.

## Live rows (exit gate, after merge)

- Probe turn 1 (code + tests for the `todo/` package) routes as today
  (tests-primary) but the written test file lands at
  `tests/test_storage.py` importing `todo.storage`, and passes on the
  client (the code half rides #123).
- Ladder turn 7: the last text names the client's pytest result.
- Ladder 13/13 exits 0, deliverables unchanged (rerun once).

## Required self-tests in the implementer's report (agreed 2026-09-11)

Mutant evidence for every new pin; a consumer trace for the new
workspace-map shape (every script that reads `workspace`, the envelope,
the trace); a fresh-input hunt on the path-prefix stripping (two clients
with different roots in one session, a basename repeated in two
directories, a header with a trailing `(truncated)`). One independent
review round follows.
