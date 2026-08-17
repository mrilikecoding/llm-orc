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
inside a seeded #138 workspace, which is why this guard is written the
way that one is: computed from the tree rather than a hardcoded list, so
a new collision is caught the day it lands.
"""

from __future__ import annotations

import importlib.util
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_TREES = ("tests", "benchmarks")


def _sys_path_insertions() -> dict[Path, list[Path]]:
    """Replicate pytest's prepend-import-mode basedir walk.

    For each test module, walk UP while `__init__.py` exists; the first
    directory without one is what pytest inserts into `sys.path`.
    """
    insertions: dict[Path, list[Path]] = defaultdict(list)
    for tree in TEST_TREES:
        for module in (REPO_ROOT / tree).rglob("test_*.py"):
            basedir = module.parent
            while (basedir / "__init__.py").exists():
                basedir = basedir.parent
            insertions[basedir].append(module)
    return insertions


def _top_level_names_exposed() -> dict[str, list[Path]]:
    """Every name those insertions make importable as a top-level module."""
    exposed: dict[str, list[Path]] = defaultdict(list)
    for basedir in _sys_path_insertions():
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
    collisions = {
        name: [str(p.relative_to(REPO_ROOT)) for p in paths]
        for name, paths in _top_level_names_exposed().items()
        if len(paths) > 1
    }

    assert not collisions, f"test-tree name collisions: {collisions}"


def test_no_test_tree_directory_shadows_an_installed_module() -> None:
    """The wider class: colliding with something in the environment.

    `parse.py` collided with nothing else in the tree — it collided with
    a PyPI package that this repo's pytest plugin stack imports at plugin
    load, which is worse, because the failure lands before any test runs.
    """
    shadows = {}
    for name, paths in _top_level_names_exposed().items():
        owned = [p for p in paths if p.is_relative_to(REPO_ROOT)]
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            continue
        if spec is None or spec.origin is None:
            continue
        origin = Path(spec.origin)
        # A name that resolves to one of OUR own directories is the
        # intended import, not a shadow.
        if any(origin == p or origin.is_relative_to(p) for p in owned):
            continue
        if origin.is_relative_to(REPO_ROOT / "src"):
            continue
        shadows[name] = str(origin)

    assert not shadows, f"test-tree modules shadow installed ones: {shadows}"


def test_the_guard_detects_a_planted_collision(tmp_path: Path) -> None:
    """Without this the two guards above could be tautologies.

    They compute from the real tree, so they pass whenever the tree
    happens to be clean — including if the walk were broken and returned
    nothing at all. This plants the exact shape of the #156 collision and
    asserts the analysis reports it.
    """
    tree = tmp_path / "tests"
    (tree / "unit").mkdir(parents=True)
    (tree / "__init__.py").touch()
    # No tests/unit/__init__.py: this is what breaks the chain.
    (tree / "unit" / "test_thing.py").touch()
    (tree / "unit" / "json").mkdir()
    (tree / "unit" / "json" / "__init__.py").touch()

    basedir = (tree / "unit" / "test_thing.py").parent
    while (basedir / "__init__.py").exists():
        basedir = basedir.parent

    assert basedir == tree / "unit", "the walk must stop at the broken link"
    exposed = [
        child.name
        for child in basedir.iterdir()
        if child.is_dir() and (child / "__init__.py").exists()
    ]
    assert "json" in exposed
    assert importlib.util.find_spec("json") is not None, (
        "a planted package named after a stdlib module is exactly the shadow"
    )
