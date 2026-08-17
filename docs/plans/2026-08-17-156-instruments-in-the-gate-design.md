# #156 — the measurement instruments run in an automated gate (design)

Status: pre-flight. Issue: #156.

## Mechanism

`pyproject.toml` sets `testpaths = ["tests"]`, and no CI step runs
`benchmarks/`. So `benchmarks/agentic_serving/tests/` executes only when
someone types the path by hand.

Measured 2026-08-17: **511 tests** (the issue said 412; the #138 volume
pins landed since), all passing, ~13s serial and ~4s under `-n auto`.

`make lint` is `mypy src tests` and `ruff check src tests`, so
`benchmarks/` is ungated LOCALLY for types and style. In CI the picture
differs and an earlier draft flattened it: CI runs
`ruff check . --exclude ...`, which already covered `benchmarks/`, so
only the TYPE check was missing there. Current state: ruff check clean,
ruff format clean, **mypy 3 errors across 42 files**.

Why this matters more than a normal coverage gap: these ARE the
measurement instruments. The oracles decide shipped-correct vs
shipped-broken. `score_run` derives shipped from manifest diffs.
`stats` produces the intervals the parity table carries. A regression
here does not fail a build — it silently corrupts a run's evidence, and
the next battery inherits it. Both oracle error directions are hazards,
and the false-reject direction can fabricate a hypothesis.

## The obstacle, which is not what the issue expected

Adding `benchmarks/agentic_serving/tests` to the run does not work as-is.
`pytest tests benchmarks/agentic_serving/tests` produces **18 collection
errors**, every one `ModuleNotFoundError: No module named
'benchmarks.agentic_serving'`.

`tests/__init__.py` exists but `tests/unit/__init__.py` does not, so the
package chain breaks at `tests/unit`. Under pytest's prepend import mode
that puts `tests/unit` on `sys.path`, and `tests/unit/benchmarks/` then
resolves as top-level `benchmarks`, shadowing the real package. The
error message is misleading — it says `benchmarks` was found and
`agentic_serving` was not, because a DIFFERENT `benchmarks` was found.

This is the same class as the `parse.py` shadowing that killed pytest
inside a seeded #138 workspace.

Surveyed the whole class rather than the one instance, by walking each
test module up while `__init__.py` exists and collecting what each
resulting `sys.path` entry exposes:

- 9 directories land on `sys.path`: `.`, `tests/bdd`, `tests/fixtures`,
  `tests/unit`, `tests/unit/cli`, `tests/unit/core/validation`,
  `tests/unit/issues`, `tests/unit/schemas`, `tests/unit/serving`.
- `benchmarks` is the ONLY collision today.

So the fix is narrow, but the structure invites recurrence: any future
`tests/unit/<name>/` package silently shadows a top-level `<name>`.

## Change

Three pieces, three commits.

1. **`tests/unit/__init__.py`** (empty). Measured: with it,
   `pytest tests benchmarks/agentic_serving/tests` collects and passes
   **3933 tests** in one run (3938 at the branch tip, once this arc's
   own five guard pins land). An earlier draft said 3942, which review
   found does not reproduce — a bad number to get wrong in a document
   whose stance is "measured rather than assumed". This is structural:
   no behavior changes, only module names.

2. **A guard pin over the shadowing class**, in the shape of #138's
   fixture-name guard. Asserts that no directory pytest will insert into
   `sys.path` exposes a name that also resolves as a real top-level
   import. Computed from the tree rather than a hardcoded list, so a new
   `tests/unit/<name>/` is caught the day it lands rather than the day
   someone runs the two suites together.

3. **The instruments enter the gate**: `testpaths` gains
   `benchmarks/agentic_serving/tests`, and `make lint` gains
   `benchmarks` for mypy and ruff. The 3 mypy errors get fixed as part
   of this, since a gate that starts red is not a gate.

## The two checks the issue asked for, answered

- **Coverage gate unaffected.** `--cov=llm_orc --cov-fail-under=90`
  reports 92.2% with or without the benchmark tests. Adding tests can
  only raise `llm_orc` coverage, never lower it, and these barely touch
  `llm_orc` at all. Verified across combined runs.
- **`-n auto` is clean for the instruments themselves**: 511 tests, 3
  consecutive runs, no failures. Combined runs did surface one
  intermittent failure, but it is **pre-existing in `make test`** and
  unrelated — `test_artifact_manager_integration`, ~1 run in 6. Filed as
  #165 rather than absorbed here, because absorbing a flake into this
  arc would make this change look like the cause.

## Invariant

Every test under `benchmarks/agentic_serving/tests/` runs in the same
automated gate as `tests/`, and no test-tree package shadows an
importable top-level module.

## Regression instruments

1. **The shadowing guard itself** (piece 2), which must be RED before
   `tests/unit/__init__.py` lands and green after. That ordering is the
   whole demonstration: it fails today for a real reason.
2. **The guard catches a NEW collision**, not just the known one:
   create a `tests/unit/<name>/` package for a name that resolves as a
   top-level import and assert the guard flags it. Without this the
   guard could be a tautology that passes because the tree happens to be
   clean.
3. **A benchmark test is collected by a bare `pytest` invocation** with
   no path argument, which is what `make test` and CI actually run.
   Asserting on `testpaths` config alone would not prove collection.

## Round 2, after review

Both guards shipped in a state where they could not fail in the gate,
which on a branch arguing "a gate that starts red is not a gate" is the
wrong note to end on. Review demonstrated each.

- **The installed-module guard was inert in a full run.** It used a bare
  `find_spec`, and by the time it executes in the gate pytest has
  already prepended every basedir, so a planted
  `tests/unit/cli/vulture.py` resolved to OUR file, was judged "ours",
  and passed. It failed only when the guard file ran alone. It now
  resolves with `PathFinder.find_spec` against `sys.path` MINUS the
  directories pytest inserted. Verified: the same plant now fails the
  full `-n auto` gate.
- **The anti-tautology pin re-implemented the walk inline**, so it
  pinned nothing about the production helpers. Blanking
  `_sys_path_insertions` to return an empty dict left all three tests
  green — the exact case the commit message claimed it caught. The
  helpers now take a `root`, and the pin drives them against a planted
  tree. Verified: blanking the walk now kills two pins.
- **The walk only matched `test_*.py`.** Pytest's `python_files`
  defaults to `test_*.py *_test.py` and pyproject does not override it,
  so the identical breakage planted as `thing_test.py` broke collection
  while the guard stayed green. Both patterns are matched now.
- **Scope came from a hardcoded `("tests", "benchmarks")`**, which made
  the guard go permanently red on volume-battery arms written under
  `benchmarks/` — modules named `ledger.py`, `qty.py` and their tests,
  which pytest never collects and which this project does not delete.
  Scope is now read from `testpaths`, so the guard's reach and the
  gate's reach are the same thing by construction.
- `make format` and `make lint-fix` could not fix what `make lint` had
  started checking. Aligned.

## Known bounds

- Wall-clock: `make test` goes from ~26s to ~35s under `-n auto`
  (branch measured at 35.1 / 35.6 / 34.9s; main at 25.9 / 25.8 / 24.9s).
  Two earlier numbers were recorded, ~39s and ~50s, and they could not
  both be the measurement; review caught the disagreement. Both sat
  inside the spread of a noisy box, which is the argument for quoting
  three runs rather than one.
- The guard covers the `tests/` and `benchmarks/` trees. It does not
  cover a top-level module added elsewhere that collides with something
  in `src/`.
- Adding `__init__.py` changes module names for everything under
  `tests/unit/`. Nothing in the suite imports test modules by name, but
  that is an assertion about today's tree, not a guarantee.
- No coverage floor on the instruments themselves: `--cov` targets
  `llm_orc` only. Measured with `--cov=benchmarks`, several instruments
  sit well below the repo's 90% bar (`runner.py` 40%, `bench.py` 56%,
  `volume_fixture.py` 75%). Instrument coverage can rot with the gate
  green.
- `testpaths` names `benchmarks/agentic_serving/tests` specifically, so
  a future `benchmarks/<other>/tests/` is not gated by construction. The
  guard now inherits the same bound, deliberately: its scope tracks the
  gate's.
- The layout inconsistency that CAUSED this survives. `judge_adequacy`
  has no `tests/` of its own; its four tests live in
  `tests/unit/benchmarks/`, which is the very directory whose name did
  the shadowing. It is harmless now only because one empty
  `tests/unit/__init__.py` exists.
- The battery shell scripts are in no gate (no shellcheck).
- #165's flake did not reproduce during review: ~21 further `-n auto`
  runs, zero failures. At the observed 1-in-6 that is not a refutation
  (P(0 in 11) is about 0.13), but it is worth recording that it is
  unreproduced by a second party on a second checkout.
- This does not make the instruments CORRECT, only executed. The #84
  fixture-pinning methodology is what argues about their correctness;
  this change only ensures a regression in them fails a build.
