"""The test tree must not shadow importable top-level modules (#156).

Found while getting the measurement instruments into an automated gate:
`pytest tests benchmarks/agentic_serving/tests` died with 18 collection
errors, every one `ModuleNotFoundError: No module named
'benchmarks.agentic_serving'`. The message misleads — `benchmarks` WAS
found, just the wrong one. `tests/unit/` had no `__init__.py` while
`tests/__init__.py` did, so the package chain broke at `tests/unit`,
pytest put that directory on `sys.path` under prepend import mode, and
`tests/unit/benchmarks/` resolved as the top-level `benchmarks` package.

Same class as the `parse.py` shadowing that killed every pytest run
inside a seeded #138 workspace, which is why this guard is computed from
the tree rather than from a hardcoded list.

Two things review had to correct in the first draft, both of which made
a guard that could not fail:

- the installed-module check used a bare `find_spec`, which by the time
  it runs in a FULL suite resolves through the very `sys.path` entries
  pytest already prepended — so it found our own file, judged it "ours",
  and passed. It only ever failed when the guard file ran alone. It now
  resolves against the paths pytest did NOT insert.
- the anti-tautology pin re-implemented the walk inline instead of
  calling it, so blanking the real walk left all three green. The
  helpers now take a root, and the pin points them at a planted tree.
"""

from __future__ import annotations

import sys
import tomllib
from collections import defaultdict
from importlib.machinery import PathFinder
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _gated_roots(repo_root: Path) -> list[Path]:
    """The trees pytest actually collects, read from `testpaths`.

    Not a hardcoded list, and deliberately not "everything under
    benchmarks/": the volume battery writes arms into an arbitrary
    `VOLUME_OUT`, and those arms contain modules named `ledger.py`,
    `qty.py` and their own `test_*.py`. Review showed that pointing
    VOLUME_OUT inside `benchmarks/` turned this guard permanently red
    for run artifacts pytest never collects — and this project does not
    delete run outputs. Reading `testpaths` keeps the guard's scope and
    the gate's scope the same thing by construction.
    """
    config = tomllib.loads((repo_root / "pyproject.toml").read_text())
    paths = config["tool"]["pytest"]["ini_options"]["testpaths"]
    return [repo_root / p for p in paths]


def _sys_path_insertions(roots: list[Path]) -> dict[Path, list[Path]]:
    """Replicate pytest's prepend-import-mode basedir walk.

    For each test module, walk UP while `__init__.py` exists; the first
    directory without one is what pytest inserts into `sys.path`.

    Both of pytest's default `python_files` patterns are matched. Review
    demonstrated the cost of only matching the first: an identical
    breakage planted as `thing_test.py` instead of `test_thing.py` broke
    collection just the same while this guard stayed green.
    """
    insertions: dict[Path, list[Path]] = defaultdict(list)
    for root in roots:
        if not root.exists():
            continue
        modules = list(root.rglob("test_*.py")) + list(root.rglob("*_test.py"))
        for module in modules:
            basedir = module.parent
            while (basedir / "__init__.py").exists():
                basedir = basedir.parent
            insertions[basedir].append(module)
    return insertions


def _top_level_names_exposed(roots: list[Path]) -> dict[str, list[Path]]:
    """Every name those insertions make importable as a top-level module."""
    exposed: dict[str, list[Path]] = defaultdict(list)
    for basedir in _sys_path_insertions(roots):
        for child in basedir.iterdir():
            if child.is_dir() and (child / "__init__.py").exists():
                exposed[child.name].append(child)
            elif child.suffix == ".py" and child.name != "__init__.py":
                exposed[child.stem].append(child)
    return exposed


def test_no_two_test_tree_directories_expose_the_same_top_level_name() -> None:
    """The collision this issue was filed on.

    Two `sys.path` entries exposing one name means whichever pytest
    inserted first wins, and the loser's imports fail with a message
    naming the module it did find rather than the one it wanted.
    """
    roots = _gated_roots(REPO_ROOT)
    collisions = {
        name: sorted(str(p.relative_to(REPO_ROOT)) for p in paths)
        for name, paths in _top_level_names_exposed(roots).items()
        if len(paths) > 1
    }

    assert not collisions, f"test-tree name collisions: {collisions}"


def test_no_test_tree_directory_shadows_an_installed_module() -> None:
    """The wider class: colliding with something in the environment.

    `parse.py` collided with nothing else in the tree — it collided with
    a PyPI package that this repo's pytest plugin stack imports at plugin
    load, which is worse, because the failure lands before any test runs.

    Resolution deliberately EXCLUDES the directories pytest inserted.
    A bare `find_spec` cannot work here: in a full-suite run every
    basedir is already on `sys.path` by collection time, so the shadow
    resolves to our own file and the check passes on the strength of the
    very thing it is meant to catch.
    """
    roots = _gated_roots(REPO_ROOT)
    exposed = _top_level_names_exposed(roots)
    inserted = {str(p) for p in _sys_path_insertions(roots)}
    installed_paths = [
        p
        for p in sys.path
        if p
        and str(Path(p).resolve()) not in {str(Path(i).resolve()) for i in inserted}
        and Path(p).resolve() != REPO_ROOT
    ]

    shadows = {}
    for name, paths in exposed.items():
        try:
            spec = PathFinder.find_spec(name, installed_paths)
        except (ImportError, ValueError):
            continue
        if spec is None or spec.origin is None:
            continue
        shadows[name] = {
            "ours": sorted(str(p.relative_to(REPO_ROOT)) for p in paths),
            "shadowed": spec.origin,
        }

    assert not shadows, f"test-tree modules shadow installed ones: {shadows}"


def test_the_guards_detect_a_planted_collision(tmp_path: Path) -> None:
    """Exercises the real helpers, which is the whole point.

    The first draft re-implemented the walk inline here. Review blanked
    `_sys_path_insertions` to return an empty dict — leaving the walk
    entirely dead — and all three tests stayed green, which is exactly
    the tautology this pin claims to close. It now calls the production
    helpers against a planted tree.
    """
    tree = tmp_path / "tests"
    (tree / "unit").mkdir(parents=True)
    (tree / "__init__.py").touch()
    # No tests/unit/__init__.py: this is what breaks the chain.
    (tree / "unit" / "test_thing.py").touch()
    (tree / "unit" / "json").mkdir()
    (tree / "unit" / "json" / "__init__.py").touch()

    insertions = _sys_path_insertions([tree])
    assert tree / "unit" in insertions, "the walk must stop at the broken link"

    exposed = _top_level_names_exposed([tree])
    assert "json" in exposed

    installed = PathFinder.find_spec("json", [p for p in sys.path if p])
    assert installed is not None, "a stdlib name is the shadow being planted"


def test_the_walk_sees_both_pytest_filename_patterns(tmp_path: Path) -> None:
    """`python_files` defaults to `test_*.py *_test.py`, and pyproject
    does not override it. Review planted the #156 breakage under the
    second pattern and this guard stayed green while collection broke
    exactly as before."""
    tree = tmp_path / "tests"
    (tree / "unit").mkdir(parents=True)
    (tree / "__init__.py").touch()
    (tree / "unit" / "thing_test.py").touch()

    assert tree / "unit" in _sys_path_insertions([tree])


def test_the_gated_roots_come_from_testpaths() -> None:
    """The guard's scope must track the gate's scope.

    Hardcoding ("tests", "benchmarks") made this guard go red on volume
    battery arms written under benchmarks/, which pytest never collects
    and which this project does not delete.
    """
    roots = _gated_roots(REPO_ROOT)

    assert REPO_ROOT / "tests" in roots
    assert REPO_ROOT / "benchmarks/agentic_serving/tests" in roots
    assert REPO_ROOT / "benchmarks" not in roots
