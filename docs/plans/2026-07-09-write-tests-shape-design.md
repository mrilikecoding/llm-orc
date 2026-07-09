# The write-tests shape (#98, second half of the gate integrity pair)

**Problem (long-horizon drive 2026-07-09, todoapp turn 4):** a "write
tests" turn routes to build-gated, where test_writer AND code_writer both
emit test files into ONE exec namespace; same-named functions from
test_writer shadow the deliverable's, so the gate validated a shadowed
composite and ACCEPTED while the shipped test_storage.py carried a broken
test (10/11 pass in the workspace) — a wrong accept.

**Fix (issue direction (b), approved sequence):** a dedicated shape where
the deliverable IS the test file, executed against the materialized
workspace alone.

## Routing

- classify: a test-primary turn — build verb + tests as the object
  (`write/add/create/generate tests (for|of) ...`, or task-initial
  "write tests...") or a named `test_*.py` file — routes to the new
  intent `tests-seat`, `kind: python_tests`, `build: true`.
  "write is_even WITH tests" stays code-seat (tests must be the object).
- Deliverable filename, derived in classify: named `test_*.py` wins;
  else `test_<named-module>.py`; else `test_solution.py`.
- decide (guarded model): `tests-seat` joins the closed set;
  resolve `_DERIVED` gains `tests-seat: ("python_tests", True)`.
- Shape catalog: `write-tests` declares `serves: tests-seat`.

## The shape

`write-tests` (loop, until `${diagnostics.accept}`, max 2, carry
`${diagnostics.retry_input}` — fresh regeneration only; tests are the
moving side, the executor's real-workspace failure report is the retry
evidence) over `write-tests-round`:

    test_writer     existing test-writer ensemble, reused
    tests_gather    tests = seat output; code = ""; workspace from the
                    context; NO target_file (nothing shadows); reuses
                    accept_gather's extraction + workspace-import
                    injection via sibling import
    executor        accept_executor.py unchanged (empty code, workspace
                    materialized, deliverable tests run against it)
    judge           adequacy_check.py reused
    accept_gate     reused
    tests_envelope  primary = the executor-echoed tests (the exact
                    validated artifact); same diagnostics contract;
                    fresh-only retry_input

Both new ensembles are non-registry-facing except `write-tests`'s
`serves:` declaration. Top-level symlinks per the dispatch-discovery
convention. seat_contract admits via the standard success+artifact
assertions.

## What this closes

The shipped artifact IS the executed artifact: one test source, run
against the workspace alone — the shadowed-composite wrong-accept is
structurally impossible.

## Validation

- Unit: classify routing boundaries (incl. the "with tests" negative and
  test-filename derivation), tests_gather, tests_envelope.
- Wiring: catalog maps tests-seat -> write-tests; round has no
  code_writer; gate deps {executor, judge}.
- Hermetic end-to-end: echo test-writer + a real workspace module through
  the real engine; accepted envelope carries the tests as primary.
- Live: build storage.py through OpenCode, then "write tests for
  storage.py" — shipped file runs green against the workspace.
