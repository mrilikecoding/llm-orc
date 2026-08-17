"""The instrument package must not be shadowed by the test tree (#156).

Found while getting the measurement instruments into an automated gate:
`pytest tests benchmarks/agentic_serving/tests` died with 18 collection
errors, every one `ModuleNotFoundError: No module named
'benchmarks.agentic_serving'`. The message misleads — `benchmarks` WAS
found, just the wrong one. `tests/unit/` had no `__init__.py` while
`tests/__init__.py` did, so the package chain broke at `tests/unit`,
pytest put that directory on `sys.path` under prepend import mode, and
`tests/unit/benchmarks/` resolved as the top-level `benchmarks` package.

## Why this file is two assertions and not a shadowing detector

It was a shadowing detector, twice, and both versions were wrong. The
first could not fail in a full run at all. The second fixed that, and
review then demonstrated four ways it went red on a CORRECT tree:

- an ordinary `conftest.py` added to `tests/unit/serving/` turned the
  gate red, because pytest special-cases conftest and the walk did not;
- volume-battery arms written to any path inside a testpath turned it
  red, including dot-dirs and `build/` that pytest itself skips via
  `norecursedirs` — in a project whose rule is never to delete run
  outputs;
- the verdict became test-ORDER dependent, since five suites
  `sys.path.insert` the serving scripts directory at import time, so one
  tree passed alone and failed in a full run;
- and a real namespace-package shadow (`google`, present in this venv)
  broke collection while the detector reported all clear, because a
  namespace package has no `origin`.

Three rounds, and each fix grew the false-positive surface faster than
the diagnostic was worth. The cause is structural rather than careless:
the detector reimplemented pytest's import resolution from outside, and
pytest's real rules — conftest handling, `norecursedirs`, `--ignore`,
`collect_ignore`, glob-expanded `testpaths`, namespace packages,
meta-path finders — are not reconstructible from a directory walk.

What is worth having is much smaller. A shadow of this kind ALREADY
fails the build, by breaking collection, which is exactly how #156 was
found. So a guard cannot add detection here, only a name. These two
assertions state the invariant directly, at the one place it bit, with
no walk to get wrong and no order dependence.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_the_instrument_package_resolves_to_the_real_one() -> None:
    """The #156 regression, asserted at the point it bit.

    If `tests/unit/benchmarks/` shadows the top-level package again, the
    511 instrument tests fail to import and the build goes red on 18
    ModuleNotFoundErrors naming the module pytest DID find. This turns
    that into one assertion naming the actual cause.
    """
    expected = REPO_ROOT / "benchmarks" / "agentic_serving" / "__init__.py"

    spec = importlib.util.find_spec("benchmarks.agentic_serving")

    assert spec is not None, "benchmarks.agentic_serving is not importable"
    assert spec.origin is not None, "resolved to a namespace package, not the real one"
    assert Path(spec.origin).resolve() == expected.resolve(), (
        f"benchmarks.agentic_serving resolved to {spec.origin}, not {expected}"
    )


def test_tests_unit_is_a_package() -> None:
    """The structural fix that makes the assertion above hold.

    `tests/__init__.py` exists, so a missing `tests/unit/__init__.py`
    breaks the package chain there and hands `tests/unit` to `sys.path`,
    which is the entire mechanism. Pinned on its own line because the
    file is empty and reads like an accident otherwise: someone tidying
    "unused" empty files would take the gate with it.
    """
    assert (REPO_ROOT / "tests" / "unit" / "__init__.py").exists()
