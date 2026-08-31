# #172 + #176 — library source resolution and gate test-participants (design)

Status: review round 2 rework complete. Issues: #172, #176. Prior commit:
`fc79941b` (first author's attempt, superseded by round 1).

## The two invariants

**#176 (accept gate, `accept_executor.py` + `accept_executor_runner.py`).**
The gate's verdict comes only from tests the runner actually executes. A
file whose classes yield no runnable tests is judged by its remaining tests
(an empty `__cases__` child is not itself a failure, unless the runner
found a class it cannot execute at all — see F8/N-1 below). A file with no
runnable tests at all refuses. As of review round 2, the "can this class
run" judgment itself is made in the RUNNER, against the real post-exec
namespace — not guessed from source text in the executor.

**#172 (library source, `library.py`).** `("local", "")` never escapes as a
real path. The two IMPLICIT candidates — the cwd checkout and the packaged
copy — are `_is_library`-gated: `ensembles/` must be a directory AND
non-empty (review round 2, N-5 — an empty directory is never a source, no
carve-out). The one EXPLICIT candidate, `LLM_ORC_LIBRARY_PATH`, is gated on
its own existence only (review round 2, N-4 — a user's explicit
configuration is trusted, not second-guessed against ensembles content).
When nothing qualifies, the result is an explicit no-library outcome: never
the caller's cwd (which `Path("") / "ensembles"` used to silently consult
as a relative path), and never remote by implicit default (remote is
reachable only through an explicit `LLM_ORC_LIBRARY_SOURCE=remote`).

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
pytest-class execution in this arc. The gate refuses instead, honestly,
naming the class(es): `"tests define a class-based suite the runner
cannot execute: <names>"`.

**Round 1's implementation (superseded — see N-1 below).** A static AST
scan of the tests source (`_pytest_style_candidates`, requiring a
`test_*` method literally defined in the class body) plus an aggregate
equality check in the executor: refuse only when the run's executed-test
count equaled exactly the top-level function count (i.e. the candidate
class contributed nothing). This is the version W10's fail-closed
narrowing (drop the name-only trigger — see below, unchanged by round 2)
was written against.

**Round 2 finding (N-1, rule 18's third instance in this file of
guessing what runs from an AST shape — moved structural):** the
aggregate-equality check had a demonstrated wrong-accept, W19: a passing
top-level `test_ok`, a FAILING pytest-style `TestPytestStyle.test_bad`,
and a passing real `TestReal(unittest.TestCase)`. `TestReal` contributes
a test too, so the executed count (2) never equals the top-level count
(1) — the equality check never fires, and the failing class silently
never runs. The same static-scan approach also missed W21 (a pytest-style
class's `test_*` method INHERITED from a base defined in the produced
CODE file, not the tests file — no `FunctionDef` in the tests source at
all) and W23 (a `test_*` class ATTRIBUTE bound to an imported callable —
`test_it = imported_fn` — an `Assign`, not a `FunctionDef`).

**Fix:** delete the AST heuristic and the aggregate-equality check
entirely. The judgment moves to the RUNNER (`accept_executor_runner.py`,
`_leaked_test_classes`): after the code and tests modules both load, scan
the real post-exec namespace for module-level classes carrying a callable
`test_*` attribute (`getattr`/`dir`, which walk the MRO — inheritance and
assignment both count, closing W19/W21/W23 at the root: the check is on
the TRUTH, not a guess about it) that are NOT `unittest.TestCase`
subclasses. Every runner invocation (each isolated per-test child, the
`__cases__` child, and the legacy whole-run) computes this identically
from the full namespace regardless of `--only`, so the executor's
`_run_children` dedupes identical reports across children rather than
needing to single out which child is "responsible". The one
executor-side change: an empty `__cases__` child is the round-1
non-failure allowance ONLY when the runner reports no leak — a leak with
zero other executed tests must not be swallowed by that allowance
(`empty_cases_child = only == "__cases__" and n_tests == 0 and not
leaked`).

**Narrowing from the brief's literal wording, and why (unchanged by
round 2).** The brief's trigger condition was "a class whose name starts
with `Test` OR defines `test_*` methods". The name-only half was
dropped: W10's own fixture (`class TestCase: pass`, a plain local double
with zero methods, used purely as a base for an unrelated helper) has a
name starting with `Test` but nothing to silently skip — flagging it
would be a false refusal, not an honest one, and the point of F8 is
refusing to guess wrong. Requiring an actual callable `test_*` attribute
keeps the trigger sound against every demonstrated shape (W3, W19, W21,
W23, W25 refuse; W7, W8, W10, W22, and the vacuity-repair pin are
unaffected) without inventing a name-based special case for the string
`TestCase` specifically. W22 (a class with only a `runTest` method, no
`test_*` attribute, not a TestCase) confirms the boundary: nothing
anywhere in this system would ever execute it, so not flagging it is
correct, not a gap.

**Adjudicated fail-closed trade (W25):** even a pytest-style class whose
`test_*` method WOULD pass if the runner could execute it still refuses
— the runner never attempts to run it, so there is no way to know either
way, and guessing "it would have passed" is exactly the leniency the F8
decision forecloses.

**Bound (round 1's, narrowed by round 2's fix but not eliminated):** the
"contributed nothing beyond the top-level functions" framing is gone
(round 2 checks each class independently, not an aggregate count), so
W19's specific escape (a real TestCase's contribution masking a separate
leaked class via a count coincidence) is closed. What remains unfixed:
`_leaked_test_classes` itself has no known false-negative shape in the
demonstrated matrix, but a class that manufactures a `test_*` attribute
dynamically at metaclass/`__init_subclass__` time rather than through
`dir()`-visible means is not demonstrated either way and is out of this
arc's scope.

## Review round 2 outcome

Six further findings, all adjudicated (N-1 above; N-2 through N-6 here).

- **N-2 (vacuity, again):** `test_a_real_testcase_still_runs` — round 1's
  OWN repair of the F2 vacuity — did not discriminate `has_cases` either.
  With zero top-level functions, `_run_sandboxed`'s "nothing enumerable"
  check (`not names and not has_cases`) is true whenever `has_cases` is
  `False` in exactly the same way it's true when `names` is empty and
  `has_cases` legitimately doesn't matter — so a `has_cases=False` mutant
  ALSO falls to the legacy whole-run branch, which does real TestCase
  detection unconditionally and passes regardless. Verified by hand
  (`has_cases` hard-coded `False`): the pin stayed green. Fixed by adding
  a top-level `test_top`, so `names` is non-empty regardless of
  `has_cases` (the isolated path is taken either way) and only a correct
  `has_cases=True` spawns the `__cases__` child that runs `testAdd`:
  `n_tests == 2` only when both run; the `has_cases=False` mutant
  contributes just `test_top`, giving `n_tests == 1`. Mutation-verified.
- **N-3:** the same test asserted `n_tests == 1` twice — a literal
  copy-paste duplicate. Deduped (folded into N-2's `n_tests == 2` fix).
- **N-4 (decision):** an explicitly configured `LLM_ORC_LIBRARY_PATH`
  pointing at a profiles-only directory (no `ensembles/` at all) was
  silently discarded — `_is_library` gated it the same as the two
  IMPLICIT candidates — and the packaged library served in its place
  instead. Adjudication: explicit user configuration wins on its own
  existence, not on `_is_library`. Priority 1 now gates on
  `library_path.is_dir()` only; `_is_library` (content-gated) applies
  only to the cwd checkout and the packaged library. A profiles-only env
  library now serves its profiles/templates and honestly lists zero
  ensemble categories.
- **N-5 (decision):** `_is_library` still accepted an EMPTY `ensembles/`
  directory (present, zero entries — the normal post-`git worktree add`
  state, not a hypothetical) for the two IMPLICIT candidates, so that cwd
  shape still shadowed a populated packaged copy. Issue #172's own
  invariant says an empty directory is never a source, with no carve-out
  for "empty but present". `_is_library` now requires `ensembles/` to be
  a directory AND non-empty (`any(ensembles_dir.iterdir())`). Two round-1
  fixtures that created `ensembles/` with nothing inside it (nominally
  "populated" tests) needed a real entry added to stay valid.
- **N-6 (docs):** `docs/cli-reference.md`'s "Library Source
  Configuration" section (~line 270) described the pre-round-1
  filesystem-detection-then-no-op behavior; rewritten to describe the
  real 4-candidate order and the non-empty requirement. Its separate
  "Library Path Configuration" section (~line 665, for `llm-orc init`)
  claimed `LLM_ORC_LIBRARY_SOURCE=local` selects "the package submodule"
  for primitive-script installation — verified false by direct
  inspection and by running `llm-orc init` in a scratch directory with no
  environment variables set: `.llm-orc/scripts/` is created empty either
  way (`ConfigurationManager()` is constructed with no `template_provider`
  in every CLI call site, so `_copy_profile_templates` is a no-op for
  `llm-orc init`; `llm-orc scripts list` reports "No scripts found").
  Primitive scripts ship as built-in `llm_orc.primitives` package
  modules, not copied from the library; the section now says so plainly
  and points at the `library`-command-group section instead of
  duplicating a stale, disproved priority list.

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

Round 2 additions, `tests/unit/serving/test_serving_accept_gate.py`
(N-1 — the runner-side leak scan; see `## F8` above):

7a. `test_a_failing_pytest_style_class_is_not_masked_by_a_real_testcase` —
    W19, RED-verified against round 1's aggregate-equality check.
7b. `test_a_pytest_style_class_inheriting_from_the_produced_code_refuses`
    — W21 (inherited `test_*` method, base defined in the CODE file),
    RED-verified against round 1.
7c. `test_an_imported_test_attribute_still_refuses` — W23 (`test_it =
    imported_fn`, an `Assign` not a `FunctionDef`), RED-verified against
    round 1.
7d. `test_a_passing_pytest_style_class_still_refuses` — W25, the
    adjudicated fail-closed trade.
7e. `test_a_runtest_only_class_does_not_trigger_a_false_refusal` — W22,
    the non-flag boundary check.
7f. `test_a_solo_leak_names_the_class_not_the_generic_reason` — the edge
    case where `__cases__` is the ONLY child (zero top-level functions):
    the leak must name the class, not fall through to the generic "no
    test_* functions or TestCase classes found".

All six (7a–7f) are mutation-verified: gutting `_leaked_test_classes` to
always return `[]` flips 7a–7d red; reverting `_run_children`'s
`empty_cases_child` to drop `and not leaked` flips 7f red (message
degrades to the generic reason, no longer names `TestOnlyLeak`); 7e stays
green under both mutations (it does not depend on leak detection at all).

`tests/unit/serving/test_serving_shape.py`:

8. `test_a_helper_class_does_not_fail_a_passing_suite` — issue #176
   instrument 1, unchanged by this rework.
9. `test_a_real_testcase_still_runs` — issue #176 instrument 2. Round 1
   renamed the method to `testAdd` (no underscore) to dodge the
   nested-test-def legacy fallback, but with zero top-level names that
   fallback's OWN condition (`not names and not has_cases`) is met
   whenever `has_cases` is `False` too — so the pin still didn't
   discriminate (N-2). Round 2 added a top-level `test_top`:
   `n_tests == 2` only with a correct `has_cases=True`; the mutant gives
   `1`. The duplicate `assert n_tests == 1` (N-3) is gone.

`tests/unit/cli/test_library_commands.py` (class `TestLibrarySourceResolution`):

10. `test_an_empty_library_directory_is_not_a_source` — pre-existing,
    unchanged; issue #172 instrument 1.
11. `test_a_populated_library_directory_is_the_source` — pre-existing;
    the fixture's `ensembles/` dir now holds a real subdirectory (N-5
    made an empty one invalid).
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

Round 2 additions (N-4, N-5):

16. `test_a_profiles_only_env_path_is_trusted_on_its_own_existence` — N-4,
    RED-verified against round 1 (`_is_library` gated the explicit env
    path too, discarding a real profiles-only config in favor of the
    packaged library).
17. `test_an_empty_ensembles_directory_is_not_a_library` — N-5, direct
    `_is_library` unit pin. RED-verified against round 1. Mutation-
    verified: `any(ensembles_dir.iterdir())` reverted to a bare `True`
    flips this and 18 red.
18. `test_an_empty_ensembles_dir_never_shadows_a_populated_packaged_library`
    — N-5, the shadowing scenario end to end. RED-verified against
    round 1.

## Issue #172's own instruments

Read via `gh issue view 172`.

1. *`_get_library_source_config` does not return a `local` source for a
   directory with no `ensembles/` inside.* — Green: `_is_library` (now
   `.is_dir()` AND non-empty, N-5) gates the two implicit candidates
   (cwd, packaged); `LLM_ORC_LIBRARY_PATH` is existence-gated on its own
   terms (N-4). Pinned by instruments 10, 12, 16, 17, 18.
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

Round 1 close: `make test` 4046 passed. Round 2 close (below): +9 pins
(6 in `test_serving_accept_gate.py` — N-1's 7a–7f minus one already
counted; 3 in `test_library_commands.py` — N-4's 16 and N-5's 17–18).

- `make lint`: exit 0 (mypy, ruff check + format, complexipy, bandit,
  vulture, `scripts/check_doc_drift.py` — resolves every `test_*` name
  above).
- `make test` (`uv run pytest -n auto`): 4055 passed, no lone `-n auto`
  failures to re-run.
- `uv run pytest tests/unit/serving/ tests/unit/benchmarks/test_judge_adequacy_harness.py -q --no-cov`:
  751 passed.
- `uv run pytest tests/unit/cli/test_library_commands.py -q --no-cov`: 54
  passed.
