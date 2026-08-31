# #172 + #176 — library source resolution and gate test-participants (design)

Status: review round 1 rework complete. Issues: #172, #176. Prior commit:
`fc79941b` (first author's attempt, superseded by this rework).

## The two invariants

**#176 (accept gate, `accept_executor.py`).** The gate's verdict comes only
from tests the runner actually executes. A file whose classes yield no
runnable tests is judged by its remaining tests (an empty `__cases__` child
is not itself a failure). A file with no runnable tests at all refuses.

**#172 (library source, `library.py`).** `("local", "")` never escapes as a
real path. Every local candidate — the `LLM_ORC_LIBRARY_PATH` env var, the
cwd checkout, the packaged copy — is `_is_library`-gated (a populated
`ensembles/` *directory*, not merely an existing path). When none qualifies,
the result is an explicit no-library outcome: never the caller's cwd (which
`Path("") / "ensembles"` used to silently consult as a relative path), and
never remote by implicit default (remote is reachable only through an
explicit `LLM_ORC_LIBRARY_SOURCE=remote`).

## Review round 1 outcome

The first author-independent review found the committed fix (`fc79941b`)
insufficient in both arcs and adjudicated the rework below. Findings, by
label:

**#176 — `has_cases` guessed TestCase-ness from a class's base name**
(`_is_testcase`, matching a base literally named `TestCase` / `*.TestCase`).
Demonstrated wrong in both directions across an 18-shape matrix (F1, F2):

- **W8 / W7 (wrong ACCEPT):** a tests-file class subclassing a
  project-defined TestCase subclass (`class TestBad(SharedCases): pass`,
  W8) or an aliased import (`from unittest import TestCase as TC`, W7)
  never matched the base-name heuristic — no `__cases__` child was spawned,
  and a genuinely failing inherited test never ran. The gate reported a
  passing suite that had never executed its one failing test — a wrong
  accept in the gate that decides whether a build ships.
- **W10 (wrong REJECT):** a helper `class Fixture(TestCase)` whose
  `TestCase` is a plain LOCAL class (not `unittest.TestCase`) matched the
  heuristic by name alone; the spawned `__cases__` child correctly found
  nothing runnable, but the executor treated that empty result as a
  failure — rejecting a suite whose real test had passed.
- **F2 (vacuity):** the committed pin for the over-refusal direction
  (`test_a_real_testcase_still_runs`) used a method named `test_it` — a
  nested `test_*` def, which trips `_enumerate_tests`'s legacy-fallback
  path and never consults `has_cases` at all. The pin stayed green even
  with `has_cases` hard-coded to `False`.

  Adjudicated fix: delete the guessing. `has_cases` reverts to "any
  module-level class is present" (as it was before `fc79941b`); the
  runner's real `issubclass(obj, unittest.TestCase)` decides what actually
  executes. The judgment moves to the OUTCOME: an empty `__cases__` child
  is not itself a failure (`_run_children`), and a total of zero executed
  tests still refuses (`_run_sandboxed`) with the pre-existing reason.

**#172 — the library source resolver.**

- **F3 (docstring untruth):** `fc79941b`'s commit message claimed "docstring
  fixed to describe the code" with no docstring change. `_get_library_
  source_config`'s docstring now describes the real 5-priority order,
  including that remote is reachable only via an explicit env var.
- **F4 (escape):** with an empty `llm-orchestra-library/` and no env vars,
  the branch resolved to `("local", "")`, and `Path("") / "ensembles"`
  silently consulted the caller's OWN cwd as a relative path — a new
  wrong-accept (a user's unrelated `./ensembles/` served as "the
  library"). Separately, the packaged-library check ran only inside `if
  library_source == "local"`, so a caller who set no library env var at
  all — the common case — skipped it and fell straight to the empty
  sentinel, even when the packaged library was genuinely populated.
- **F6 (`.exists()` vs `.is_dir()`):** `_is_library` accepted a FILE named
  `ensembles`; browse later died with `NotADirectoryError` iterating it.
- **F7 (citation):** `_is_library`'s docstring cited `library_handler.
  get_library_dir` as a precedent that gates on content; it doesn't
  (`get_library_dir` returns a bare directory path, no content check).
  `library_handler._browse_ensembles` is the one that actually gates.
- **F5 (no pin can fail):** two committed hunks had no regression coverage
  — the priority-3 `_is_library` gate in `library.py`, and `has_cases` in
  `accept_executor.py`'s `_excise_unbound_callable_tests`. The latter
  turned out to need its OWN fix, not just a pin (below).

Adjudicated fix: every local candidate is `_is_library`-gated (now
`.is_dir()`); the packaged-library check is unconditional (a default-path
candidate, not opt-in behind `LLM_ORC_LIBRARY_SOURCE=local`); every call
site that branches on `source_type == "local"` also requires a non-empty
`source_path`, so the empty sentinel is treated as "no library" rather than
a real (if accidentally relative) path.

**`_excise_unbound_callable_tests`'s OWN `has_cases` (F5, second pin).**
This function decides whether it's safe to delete every doomed
(guaranteed-NameError) test in a suite, on the theory that a co-present
class might cover the suite instead. Unlike the run-dispatch `has_cases`
above, this decision has NO subsequent runtime correction — once excised,
a doomed test's real NameError is gone for good. Reverting it to "any class
present" (the same fix as the run-dispatch case) would have let a harmless
helper class license destroying the only evidence of a real failure,
replacing it with the less informative "no test_* functions or TestCase
classes found". Demonstrated: an all-doomed suite with a helper class
(`class Fixture: value = 3`) alongside `def test_missing(): assert
file_exists('x')`. Fixed by dropping the class-presence carve-out
entirely — declining excision whenever every test is doomed costs nothing
(a doomed test NameErrors for real either way) and is always safe.

## F8 — pytest-style classes: fail closed, not implemented

A pytest-style class (`class TestMath: def test_add(self): ...`, no
TestCase anywhere) is invisible to the runner on both sides of the branch:
not a top-level `test_*` function, not a real `unittest.TestCase`
subclass. With any passing top-level test alongside it, the gate reported
"all passed" while the class's tests silently never ran.

**Decision (adjudicated, not relitigated):** the runner does not grow
pytest-class execution in this arc. The gate refuses instead, honestly:
when the tests file contains a class that defines at least one method of
its own starting with `test_`, and the run's executed-test count equals
exactly the count of top-level `test_*` functions (i.e. the class
contributed nothing beyond them), the gate refuses with a reason naming
the limitation — `"tests define a class-based suite the runner cannot
execute: <names>"` — rather than reporting a pass on a run that skipped
real content.

**Narrowing from the brief's literal wording, and why.** The brief's
trigger condition was "a class whose name starts with `Test` OR defines
`test_*` methods". The name-only half was dropped: W10's own fixture
(`class TestCase: pass`, a plain local double with zero methods, used
purely as a base for an unrelated helper) has a name starting with `Test`
but nothing to silently skip — flagging it would be a false refusal, not
an honest one, and the point of F8 is refusing to guess wrong. Requiring
an actual `test_*` method keeps the trigger sound against every
demonstrated shape (W3 refuses; W7, W8, W10, and the vacuity-repair pin
are unaffected) without inventing a name-based special case for the string
`TestCase` specifically.

**Bound (rule 15's kind of claim, stated plainly rather than derived from
a count):** the "contributed nothing beyond the top-level functions" check
is an aggregate signal, not a per-class one. A file with BOTH a genuine
`unittest.TestCase` (contributing real tests) AND a separate, wholly
unrelated pytest-style class (contributing nothing) would not trigger F8,
because the aggregate total is non-zero — the pytest-style class's tests
would still silently never run in that specific mixed shape. Not
demonstrated by the review's matrix (W3 is a single-class file) and not
fixed here; flagged for whoever next touches this function.

## Regression instruments

`tests/unit/serving/test_serving_accept_gate.py` (helper: `_executor`):

1. `test_a_locally_subclassed_testcase_base_still_runs` — W8, wrong ACCEPT
   on the branch (RED-verified against `fc79941b`).
2. `test_an_aliased_testcase_import_still_runs` — W7, wrong ACCEPT on the
   branch (RED-verified).
3. `test_a_locally_named_testcase_helper_does_not_wrong_reject` — W10,
   wrong REJECT on the branch (RED-verified).
4. `test_an_empty_cases_child_does_not_fail_a_suite_with_passing_tests` —
   F2 vacuity repair: a genuine TestCase with no test methods of its own,
   alongside a real passing test, so `has_cases` genuinely drives the
   `__cases__` dispatch. Mutation-verified: reverting the empty-cases skip
   in `_run_children` to an unconditional append flips this red.
5. `test_a_file_with_only_a_non_testcase_class_refuses` — the invariant's
   second half: zero runnable tests anywhere still refuses.
6. `test_all_doomed_with_a_helper_class_still_reports_the_nameerror` — F5,
   `_excise_unbound_callable_tests`. Mutation-verified: reintroducing the
   has_cases carve-out flips this red (`tests_excised` goes from 0 to 1).
7. `test_a_pytest_style_class_refuses_instead_of_silently_passing` — F8
   / W3, RED-verified against `fc79941b` (both sides of that branch
   wrongly accept this shape).

`tests/unit/serving/test_serving_shape.py`:

8. `test_a_helper_class_does_not_fail_a_passing_suite` — issue #176
   instrument 1, unchanged by this rework.
9. `test_a_real_testcase_still_runs` — issue #176 instrument 2,
   reconstructed for F2: the method is named `testAdd` (no underscore) so
   the nested-test-def legacy fallback doesn't bypass the `__cases__`
   dispatch the way the committed version did.

`tests/unit/cli/test_library_commands.py` (class `TestLibrarySourceResolution`):

10. `test_an_empty_library_directory_is_not_a_source` — pre-existing,
    unchanged; issue #172 instrument 1.
11. `test_a_populated_library_directory_is_the_source` — pre-existing,
    unchanged; over-refusal direction.
12. `test_a_file_named_ensembles_is_not_a_library` — F6, direct
    `_is_library` unit pin.
13. `test_the_packaged_library_is_a_default_candidate_when_populated` — F4
    headline instrument (the redefined issue #172 instrument 2: the
    packaged library is served by default when populated, rather than the
    branch's unconditional `("local", "")`). Hermetic: `_packaged_
    library_path` is monkeypatched to a controlled fixture rather than
    depending on this checkout's own (empty, in a bare `git worktree add`)
    submodule state.
14. `test_the_packaged_library_wins_over_an_empty_cwd_checkout_with_
    source_local` — F5, priority-3 `_is_library` gate. Mutation-verified:
    reverting to a bare `cwd_library.exists()` flips this red. Also drives
    `LLM_ORC_LIBRARY_SOURCE=local` explicitly, which the committed pin
    deleted and never exercised.
15. `test_an_unrelated_cwd_ensembles_dir_is_never_served_as_the_library` —
    F4 demonstrating pin: empty submodule dir + a cwd `ensembles/` with
    content + no env vars. The cwd ensembles must not appear.

## Issue #172's own instruments

Read via `gh issue view 172`.

1. *`_get_library_source_config` does not return a `local` source for a
   directory with no `ensembles/` inside.* — Green: `_is_library` (now
   `.is_dir()`) gates every local candidate; pinned by instruments 10, 12.
2. *With no library present anywhere and no env var set, `get_library_
   categories()` returns the remote category list rather than `[]`.* —
   Superseded by the adjudicated F4 direction: the packaged library is
   the default-path fallback instead of remote (instrument 13), and true
   nothing-resolved-anywhere still yields `[]`, never remote by implicit
   default (instrument 15, and empirically: `env -u LLM_ORC_LIBRARY_PATH
   -u LLM_ORC_LIBRARY_SOURCE uv run llm-orc library browse` in this
   worktree — itself a bare `git worktree add` with an empty submodule —
   prints zero categories, not the six remote ones).
3. *`llm-orc library browse` from a fresh clone without submodules lists
   categories.* — Conditionally true, not universally: verified
   empirically in this worktree by temporarily populating `llm-orchestra-
   library/ensembles/demo-category/` and re-running browse (listed
   `demo-category`; directory removed immediately after). This repo does
   not ship `llm-orchestra-library/` as installed package data
   (`pyproject.toml`: `packages = ["src/llm_orc"]`), so a `pip install`
   without a recursed submodule has no packaged copy to fall back to
   either — and F4's adjudication explicitly excludes a remote fallback
   from the default path. Browse is safe and honest in every environment
   (never crashes, never wrong-accepts a foreign directory) and lists
   categories whenever any of the three local candidates is genuinely
   populated; it does not universally guarantee non-empty output
   regardless of environment, which would require either shipping the
   library as package data or restoring remote-by-default — both out of
   this rework's scope.

## Verification

- `make lint`: exit 0 (mypy 419 files, ruff check + format, complexipy,
  bandit, vulture, `scripts/check_doc_drift.py` — resolves every `test_*`
  name above).
- `make test` (`uv run pytest -n auto`): 4046 passed, 92.11% coverage
  (>=90% required), no lone `-n auto` failures to re-run.
- `uv run pytest tests/unit/serving/ tests/unit/benchmarks/ -q --no-cov`:
  745 passed.
- `uv run pytest tests/unit/cli/test_library_commands.py -q --no-cov`: 51
  passed.
