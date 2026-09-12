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
   absolute prefix when present). It becomes a shared helper in
   `_helpers.py` used by `accept_gather`, `tests_gather` (write-tests),
   and `refix_gather` (re-fix — today it materializes nothing, #184).
   Round 1 review rework (A1, MAJOR): a naive "longest common directory,
   dirname for a lone header" rule silently lost a lone absolute
   header's own directory — the arc's own motivating shape (a client
   reads exactly the file it is about to edit). `_helpers._resolve_root`
   is a priority ladder instead, reported in the gather output as
   `workspace_root_rule` so the trace names which one fired: (a) the
   client's own `[globbed ...]` listing's common directory when a useful
   one is visible; (b) for a lone absolute header, the ask's own named
   destination or another already-relative header matched as a SUFFIX of
   it; (c) 2+ absolute headers' longest common directory; (d) a lone
   header falls back to its bare basename (named, not silent); 2+
   headers sharing no common root are left absolute and UNRESOLVED —
   mechanism 2's A2 fix is what refuses on that rather than shipping a
   silent partial workspace.
2. **Materialization at real paths.** `accept_executor._materialize`
   writes each workspace file at `tmp/<relpath>` (creating directories;
   reject any path that escapes the root or is absolute — refuse, do not
   sanitize), writes the deliverable at `tmp/<target_path>` AND keeps
   `solution.py`/`tests.py` for the runner's namespace model unchanged
   (compared by FULL path, not basename — A3). The #171 control shadows
   the deliverable with empty bytes at the real destination too. Round 1
   review rework: a workspace entry that survives root resolution still
   absolute or escaping now makes the executor refuse instead of
   silently running an incomplete workspace (A2, `workspace_unplaced`);
   a directory-shaped or write-time-colliding header can never crash the
   executor (A4, `_write_at` never raises).
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

### Round 1 review (measured, 2026-09-12)

Two MAJOR findings on the absolute-prefix path drove a rework of
mechanism 1's root resolution (a single absolute header no longer
silently loses its directory — see the design brief's mechanism section
below and issue #184's implementer report for the full ladder); five
MEDIUM findings fixed alongside (the `solution.py`/`tests.py` skip now
compares the full path, not basename; a directory-shaped or colliding
header can no longer crash the executor; the escape test pin no longer
poisons `$TMPDIR`; the workspace-import injector derives a dotted module
name). Three bounds recorded, not fixed:

- **Trailing whitespace in a header** (`[read todo/storage.py ]`)
  materializes a DISTINCT file (`todo/storage.py ` — trailing space kept)
  rather than being trimmed to match the real path. No escape; a test
  importing the real module misses honestly (`ModuleNotFoundError`), the
  same as an unseen sibling. Not fixed — trimming would mean guessing
  which whitespace is real content vs. rendering noise, and no live
  turn has shown a trailing-space header.
- **A Windows-style separator** (`todo\..\..\x.py`) materializes a file
  with that literal string as its name at the sandbox root on POSIX —
  `_safe_relative_path`'s `..`-escape check splits on `/` only, so a
  backslash-joined `..` component never fires it. No escape (the whole
  string is one inert POSIX filename); this measurement confirms the
  brief's existing "Windows separators out of scope" bound rather than
  reopening it.
- **Mechanism 5's destination half is scope, not defect, in THIS
  branch.** Only the test-writer prompt sentence (dotted-import
  guidance) landed here; classify's `tests/test_storage.py` placement
  rule and its own instrument 5 live in `classify.py`, owned by the
  sibling routing-arc worktree, and are deferred to that work's
  follow-up.

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
