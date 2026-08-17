"""Pins for the #138 volume-ladder fixture generator.

The load-bearing invariants (design:
docs/plans/2026-08-15-138-volume-instrument-design.md):

- Per-level workspaces contain ONLY the asked modules and their tests
  (pre-flight blocker 1 — a full clone leaks every red test to the
  first pytest run, destroying the volume manipulation).
- Deterministic bytes: the same level always materializes to the same
  hashed manifest, so runs are seed-verifiable like the ladder's run-6
  baseline.
- Exactly one green and one red seeded test per module: a pytest run in
  a fresh fixture fails once per module, and a correct fix flips
  exactly that failure.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.agentic_serving.volume_fixture import (
    LEVELS,
    MODULE_ORDER,
    VOLUME_PROMPTS,
    level_modules,
    write_fixture,
)


def test_levels_and_module_order_match_the_design() -> None:
    assert LEVELS == (1, 2, 3, 5)
    assert MODULE_ORDER == ("ledger", "qty", "window", "rate", "label")


def test_level_modules_are_nested_prefixes() -> None:
    assert level_modules(1) == ("ledger",)
    assert level_modules(2) == ("ledger", "qty")
    assert level_modules(3) == ("ledger", "qty", "window")
    assert level_modules(5) == MODULE_ORDER


def test_unknown_level_is_refused() -> None:
    with pytest.raises(ValueError, match="level"):
        level_modules(4)


def test_prompts_name_exactly_the_level_modules() -> None:
    for level in LEVELS:
        prompt = VOLUME_PROMPTS[level]
        for module in level_modules(level):
            assert f"{module}.py" in prompt
        for module in set(MODULE_ORDER) - set(level_modules(level)):
            assert f"{module}.py" not in prompt


def test_no_fixture_module_shadows_an_importable_package() -> None:
    """A fixture module name must not shadow a top-level importable
    module. Found the hard way during the build: the drafted `parse.py`
    shadowed the `parse` PyPI package that this repo's pytest plugin
    stack (pytest_bdd -> parse_type) imports, so EVERY pytest run inside
    a seeded workspace died at plugin load — the truth-capture rc, the
    arm's own verification run, and the skip-rate measure with it. A
    crashed verification run is indistinguishable from a skipped one at
    the wire grain, so the confound points straight at the hypothesis.

    Bound: this checks the environment that runs truth capture (this
    repo's venv), not every arm's ambient environment; prefer names that
    are obviously not package-like regardless."""
    import importlib.util

    for module in MODULE_ORDER:
        spec = importlib.util.find_spec(module)
        assert spec is None, f"{module} shadows {getattr(spec, 'origin', spec)}"


def test_workspace_contains_only_the_asked_subset(tmp_path: Path) -> None:
    manifest = write_fixture(tmp_path / "l2", level=2)
    names = sorted(manifest)
    assert names == ["ledger.py", "qty.py", "test_ledger.py", "test_qty.py"]
    assert sorted(p.name for p in (tmp_path / "l2").iterdir() if p.name != ".git") == (
        names
    )


def test_manifests_are_deterministic_across_writes(tmp_path: Path) -> None:
    first = write_fixture(tmp_path / "a", level=5)
    second = write_fixture(tmp_path / "b", level=5)
    assert first == second


def test_workspace_is_a_git_repo_with_a_clean_seed_commit(tmp_path: Path) -> None:
    dest = tmp_path / "ws"
    write_fixture(dest, level=1)
    status = subprocess.run(
        ["git", "-C", str(dest), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert status == ""
    log = subprocess.run(
        ["git", "-C", str(dest), "log", "--oneline"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert len(log.strip().splitlines()) == 1


def test_seeded_suite_fails_exactly_once_per_module(tmp_path: Path) -> None:
    dest = tmp_path / "ws"
    write_fixture(dest, level=5)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=dest,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    tail = result.stdout.strip().splitlines()[-1]
    assert "5 failed" in tail
    assert "5 passed" in tail


def test_fixed_seeded_suite_goes_green(tmp_path: Path) -> None:
    """The canonical 1-3 line fixes flip exactly the red cases — pins that
    each seeded flaw is really the flaw the design names, and that the
    fixes the instrument scores as correct exist at the intended size."""
    dest = tmp_path / "ws"
    write_fixture(dest, level=5)
    fixes = {
        "ledger.py": (
            "def balance(entries):\n"
            '    if not entries:\n'
            '        raise ValueError("no entries")\n'
            "    total = 0\n"
            "    for amount in entries:\n"
            "        total += amount\n"
            "    return total\n"
        ),
        "qty.py": "def parse_qty(text):\n    return int(float(text))\n",
        "window.py": (
            "def last_n(items, n):\n"
            "    if n == 0:\n"
            "        return []\n"
            "    return items[-n:]\n"
        ),
        "rate.py": (
            "def per_hour(count, minutes):\n"
            "    return count / (minutes / 60)\n"
        ),
        "label.py": (
            "def slug(title):\n"
            '    return title.lower().replace(" ", "-")\n'
        ),
    }
    for name, body in fixes.items():
        (dest / name).write_text(body)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=dest,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout


def test_cli_verify_prints_the_manifest(tmp_path: Path) -> None:
    dest = tmp_path / "ws"
    module = Path(__file__).resolve().parents[1] / "volume_fixture.py"
    out = subprocess.run(
        [sys.executable, str(module), "--level", "2", "--dest", str(dest), "--verify"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    lines = [line for line in out.strip().splitlines() if line]
    assert len(lines) == 4
    for line in lines:
        path, digest = line.split("\t")
        assert path.endswith(".py")
        assert len(digest) == 64
