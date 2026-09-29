# Remote delegation, Arc 2: the serving project ships with the wheel

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A serve started in an empty directory from a wheel install runs
the agentic serving ensemble from a read-only packaged layer, writes its
state to a state dir, and still lets a checkout shadow everything, so the
mini runs `brew upgrade` instead of `git pull` (#196).

**Architecture:** The repo's `.llm-orc/` is mapped into the wheel as
`llm_orc/serving_project/` by hatchling `include` + `sources` (same file
selection rules as the package, so `.gitignore` applies). One locator
(`packaged_serving_project_dir()`) finds it in a wheel or a checkout and
honors an env override. `ConfigurationManager` gains a fourth, lowest tier
(`packaged`) in every read path: ensemble and profile dirs, runtime model
profiles (rewritten as a tier loop), the `performance:` and
`agentic_serving:` merges, `classify_tier`, and a `serving_root()` that
replaces the cwd-based resolution in the chat-completions endpoint. Writes
go through `resolve_state_dir()`: env → project dot-dir → XDG state.
The executor's child-ensemble lookup and the script resolver search every
tier. Rulings and reasons: the spec's "Arc 2 re-cut (2026-09-29)".

**Tech Stack:** Python 3.11+, hatchling (wheel), FastAPI TestClient, click
CliRunner, pytest (asyncio auto mode), ruff 88, mypy strict, complexipy
<= 15.

**Spec:** `docs/plans/2026-09-28-remote-delegation.md` (decisions,
pattern, Arc 2 card + re-cut rulings, S1/S2 findings, standing rules).
Read "Standing rules for every implementer" and the "Arc 2 re-cut" before
touching anything.

## Global Constraints

- Own git worktree off `main`: `git worktree add .claude/worktrees/arc2-packaged -b feat/packaged-serving main`.
  Never `git stash` (the working dir is shared with other sessions);
  never spawn subagents that edit or commit.
- Strict TDD. Structural and behavioral commits separate. ruff 88 +
  mypy strict from the first draft. `make lint` clean (mypy, ruff,
  format, complexipy 15, bandit, vulture, doc drift).
- Commit prefixes `feat:` `fix:` `refactor:` `test:` `docs:` `chore:`.
  No AI attribution of any kind. No session links, no scratch paths.
- Full suite: `uv run pytest -q -p no:cacheprovider`. Known local-only
  failure: `testget_available_providers_auth_only` when a router listens
  on :8080. Baseline on `main` before this arc: 4735 passed.
- Doctrine 11: pins assert outcomes through the real surface (REST
  TestClient, real `ConfigurationManager` on disk, real resolver, real
  executor, CliRunner). Each key pin is shown RED under a named mutant
  before it counts. Report mutant and the red assertion line in the
  commit body.
- Doc drift gate: `scripts/check_doc_drift.py` fails `make lint` when a
  `docs/plans/*.md` file names a `test_*` identifier in single backticks
  that exists nowhere in code. This plan names tests only inside fenced
  code blocks. Keep it that way when editing this file.
- Findings go on #196 and #191 as comments, never new issues.
- Nothing is pushed; nothing paid is run; the mini is not touched (the
  deploy step at the end is the practitioner's).
- Reads merge layers, writes target one place. The packaged tier is
  read-only: no code path may ever create a file under it. Every write
  site goes through `resolve_state_dir` or a caller-chosen scope.
- The test suite runs with `LLM_ORC_SERVING_PROJECT_DIR=""` (Task 1's
  conftest fixture), so a `ConfigurationManager` built in a temp cwd sees
  no packaged tier unless the test opts in with the
  `packaged_serving_project` fixture. Do not remove that default.

## Review Focus

Inputs the spec implies but no arc-level card names. Each gets a pin in
the owning task below.

1. **`LLM_ORC_SERVING_PROJECT_DIR` names a directory without the serving
   ensemble.** Expected: `FileNotFoundError` naming the variable and the
   marker path at construction, not a silently empty tier. Pin in Task 1.
2. **A checkout, where the packaged dir IS the project dot-dir.**
   Expected: no duplicate entries in `get_ensembles_dirs()` /
   `get_profiles_dirs()`, and runtime profiles are not merged with the
   same directory at lowest precedence (which would put the project's
   own `.local.yaml` below global). Pins in Tasks 1 and 2.
3. **First write into an absent state dir** (a fresh mini, no
   `~/.local/state/llm-orc`). Expected: created with parents, for the
   preset and for artifacts. Pins in Tasks 4 and 5.
4. **An engine write on a plain-dir serve** (the orchestrator's
   composition writer, the legacy `get_local_*_dir` helpers). Expected:
   never under the packaged or library tier: composition lands in global,
   the helpers raise naming `scope: global`. Pin in Task 7.
5. **No serving ensemble anywhere** (env override disabled, cwd has a
   `.llm-orc` without it). Expected: `serving_root()` raises naming both
   candidates; `POST /v1/chat/completions` returns a JSON error body, not
   a traceback and not a hang. Pin in Task 6.

Pre-existing, out of scope, noted for the #191 comment at the end:
`PrimitiveRegistry` has no `project_dir` and never adds the installed
package primitives (S1); `resource_handler.determine_source` and the
CRUD helpers classify tiers by substring (`".llm-orc" in str(path)`),
replaced by `classify_tier` in Task 7 where they sit on the arc's path.

---

## Task 0: the wheel carries the serving project, pinned by a checker

**Files:**
- Modify: `pyproject.toml:192-193` (`[tool.hatch.build.targets.wheel]`)
- Create: `scripts/check_wheel_contents.py`
- Modify: `Makefile` (new `wheel-check` target after `lint-check`)

**Interfaces:**
- Produces: the wheel entries `llm_orc/serving_project/{ensembles,profiles,scripts,config.yaml}`;
  `scripts/check_wheel_contents.py <wheel>` exits 0 when the packaged
  set equals `git ls-files` under those four roots, 1 with a diff.

- [ ] **Step 1: Build the baseline wheel and write the checker.** From the
  worktree root (before touching `pyproject.toml`):

```bash
mkdir -p dist/baseline && uv build --wheel -o dist/baseline -q
```

Write `scripts/check_wheel_contents.py`:

```python
"""Pin the wheel's serving project to the repo's tracked `.llm-orc/` (#196).

Usage: python scripts/check_wheel_contents.py <path/to/llm_orchestra-*.whl>

The wheel maps `.llm-orc/{ensembles,profiles,scripts,config.yaml}` to
`llm_orc/serving_project/`. This checker compares that set against
`git ls-files` for the same four roots: a wheel with no serving project,
a wheel that shipped a gitignored or untracked file, and a wheel missing
a tracked one all fail, and the diff is printed. Run from the repo root.
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

PREFIX = "llm_orc/serving_project/"
SHIPPED_ROOTS = ("ensembles/", "profiles/", "scripts/", "config.yaml")


def tracked_serving_files(repo: Path) -> set[str]:
    out = subprocess.run(
        ["git", "ls-files", "--", ".llm-orc"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    files = set()
    for line in out.splitlines():
        rel = line.removeprefix(".llm-orc/")
        if rel.startswith(SHIPPED_ROOTS):
            files.add(rel)
    return files


def packaged_files(wheel: Path) -> set[str]:
    with zipfile.ZipFile(wheel) as zf:
        return {
            name.removeprefix(PREFIX)
            for name in zf.namelist()
            if name.startswith(PREFIX)
        }


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    wheel = Path(argv[1])
    expected = tracked_serving_files(Path.cwd())
    actual = packaged_files(wheel)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if not missing and not extra:
        print(f"ok: {len(actual)} serving project files match git ls-files")
        return 0
    for rel in missing:
        print(f"missing from wheel: {PREFIX}{rel}")
    for rel in extra:
        print(f"not tracked but shipped: {PREFIX}{rel}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
```

- [ ] **Step 2: Run the checker on the baseline wheel to see it RED.**

Run: `uv run python scripts/check_wheel_contents.py dist/baseline/*.whl | tail -3; echo exit=$?`
Expected: many `missing from wheel:` lines, exit 1. This is the guard
shown failing on the pre-change system.

- [ ] **Step 3: Change the wheel selection.** Replace the two lines under
  `[tool.hatch.build.targets.wheel]` in `pyproject.toml`:

```toml
[tool.hatch.build.targets.wheel]
# The serving project (#196): the repo's own `.llm-orc/` ships as
# `llm_orc/serving_project/`, a read-only layer below project, library and
# global. Selected by the same rules as the package, so `.gitignore` keeps
# `*.local.yaml`, artifacts, the trace, the preset and `__pycache__` out.
# `scripts/check_wheel_contents.py` (make wheel-check) pins the set.
include = [
    "src/llm_orc",
    ".llm-orc/ensembles",
    ".llm-orc/profiles",
    ".llm-orc/scripts",
    ".llm-orc/config.yaml",
]

[tool.hatch.build.targets.wheel.sources]
"src/llm_orc" = "llm_orc"
".llm-orc" = "llm_orc/serving_project"
```

- [ ] **Step 4: Build and check.**

```bash
rm -rf dist/check && uv build --wheel -o dist/check -q
uv run python scripts/check_wheel_contents.py dist/check/*.whl; echo exit=$?
```

Expected: `ok: N serving project files match git ls-files`, exit 0.
If it is red with `not tracked but shipped` lines for `.local.yaml`,
`__pycache__` or `spike-*` paths, hatchling did not apply `.gitignore`
to the `include` patterns: add an explicit `exclude` list to the same
table (`"*.local.yaml"`, `"**/__pycache__"`, `".llm-orc/scripts/spike-*"`,
`".llm-orc/**/artifacts"`) and rebuild. If it is red with `missing from
wheel` for everything, hatchling ignored `include` (a `packages` key
still present wins): check the table has no `packages`. If a stray
untracked file under `.llm-orc/` is reported, the worktree is not clean:
`git status --short .llm-orc` and remove it from the worktree (the main
checkout's untracked `ensembles/research-dossier.yaml` is the
practitioner's; do not touch it there).

Then confirm nothing outside the serving project changed:

```bash
uv run python - <<'EOF'
import glob, zipfile
def names(p):
    return {n for n in zipfile.ZipFile(glob.glob(p)[0]).namelist()
            if not n.startswith("llm_orc/serving_project/") and ".dist-info/" not in n}
base, new = names("dist/baseline/*.whl"), names("dist/check/*.whl")
print("same package set:", base == new, "| only in base:", sorted(base - new)[:5], "| only in new:", sorted(new - base)[:5])
EOF
```

Expected: `same package set: True`.

- [ ] **Step 5: Makefile target.** After the `lint-check: lint` line add:

```make
wheel-check:
	rm -rf dist/wheel-check && uv build --wheel -o dist/wheel-check -q
	uv run python scripts/check_wheel_contents.py dist/wheel-check/*.whl
```

Run: `make wheel-check` → exit 0. Confirm `dist/` is gitignored
(`git status --short` shows no `dist/`).

- [ ] **Step 6: Lint and commit.**

```bash
make lint
git add pyproject.toml scripts/check_wheel_contents.py Makefile
git commit -m "feat: the wheel carries the serving project as llm_orc/serving_project

scripts/check_wheel_contents.py pins the packaged set to git ls-files
under .llm-orc/{ensembles,profiles,scripts,config.yaml}. Shown red on
the 0.21.0 wheel (every file missing) before the pyproject change."
```

---

## Task 1: the packaged tier in ConfigurationManager

**Files:**
- Create: `src/llm_orc/core/config/packaged.py`
- Modify: `src/llm_orc/core/config/config_manager.py` (`__init__`,
  `classify_tier`, `_get_library_dir`, `get_ensembles_dirs`,
  `get_profiles_dirs`)
- Modify: `tests/conftest.py` (two autouse env defaults, one opt-in fixture)
- Test: `tests/unit/core/config/test_packaged_serving.py`

**Interfaces:**
- Produces:
  - `packaged.SERVING_PROJECT_ENV = "LLM_ORC_SERVING_PROJECT_DIR"`
  - `packaged.SERVING_MARKER: Path` (`ensembles/agentic-serving/serving.yaml`)
  - `packaged.has_serving_ensemble(directory: Path) -> bool`
  - `packaged.packaged_serving_project_dir() -> Path | None`
  - `ConfigurationManager.packaged_serving_dir` property `-> Path | None`
  - `ConfigurationManager.library_dir` property `-> Path` (the library
    base; env `LLM_ORC_LIBRARY_PATH`, else `<checkout root>/llm-orchestra-library`
    where the checkout root is the local dot-dir's parent, else cwd)
  - `classify_tier` now returns `"packaged"` for paths under the packaged dir
  - `get_ensembles_dirs()` / `get_profiles_dirs()` end with the packaged
    entry when it exists and is not the local dot-dir
  - conftest fixture `packaged_serving_project(tmp_path, monkeypatch) -> Path`
    (builds a minimal serving project and points the env at it)

- [ ] **Step 1: Test isolation first.** In `tests/conftest.py`, after the
  `_isolated_global_config` fixture add:

```python
@pytest.fixture(autouse=True)
def _isolated_state_and_packaged_tier(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No packaged serving tier and a per-test state dir by default (#196).

    An editable install resolves the packaged tier to THIS checkout's
    `.llm-orc/`, so a ConfigurationManager built in a temp cwd would
    otherwise see ~60 model profiles and 100+ ensembles it did not set
    up (the #86 lesson again). Tests that pin the packaged tier opt in
    with the `packaged_serving_project` fixture below or set the env
    themselves. XDG_STATE_HOME keeps artifacts, traces and presets out
    of the developer's ~/.local/state.
    """
    monkeypatch.setenv("LLM_ORC_SERVING_PROJECT_DIR", "")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path_factory.mktemp("state")))


_PACKAGED_PROFILE = """\
name: agentic-tier-cheap-general
model: qwen3-8b
provider: llama-server
cost_per_token: 0.0
"""

_PACKAGED_CONFIG = """\
project:
  name: packaged-fixture
serving:
  self_reference: true
model_profiles:
  packaged-orch:
    model: qwen3-0.6b
    provider: llama-server
  shadowed-by-file:
    model: from-config-yaml
    provider: llama-server
agentic_serving:
  orchestrator:
    model_profile: packaged-orch
"""

_PACKAGED_SHADOW_PROFILE = """\
name: shadowed-by-file
model: from-profiles-dir
provider: llama-server
"""

_PACKAGED_SERVING = """\
name: serving
description: fixture serving ensemble (never executed by these tests)
agents:
  - name: classify
    script: scripts/agentic_serving/classify.py
"""

_PACKAGED_CHILD = """\
name: child
description: a packaged child that runs one packaged script
agents:
  - name: echo
    script: scripts/agentic_serving/classify.py
"""

_PACKAGED_SCRIPT = """\
import json, sys
data = sys.stdin.read()
print(json.dumps({"success": True, "data": {"echo": data.strip()}}))
"""


@pytest.fixture
def packaged_serving_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A minimal serving project on disk, installed as the packaged tier
    through the env override. Returns its root (the dot-dir equivalent)."""
    root = tmp_path / "serving_project"
    (root / "ensembles" / "agentic-serving").mkdir(parents=True)
    (root / "ensembles" / "agentic-serving" / "serving.yaml").write_text(
        _PACKAGED_SERVING
    )
    (root / "ensembles" / "agentic-serving" / "child.yaml").write_text(
        _PACKAGED_CHILD
    )
    (root / "profiles").mkdir()
    (root / "profiles" / "agentic-tier-cheap-general.yaml").write_text(
        _PACKAGED_PROFILE
    )
    (root / "profiles" / "shadowed-by-file.yaml").write_text(
        _PACKAGED_SHADOW_PROFILE
    )
    (root / "scripts" / "agentic_serving").mkdir(parents=True)
    (root / "scripts" / "agentic_serving" / "classify.py").write_text(
        _PACKAGED_SCRIPT
    )
    (root / "config.yaml").write_text(_PACKAGED_CONFIG)
    monkeypatch.setenv("LLM_ORC_SERVING_PROJECT_DIR", str(root))
    return root
```

`Path` is already imported in `tests/conftest.py`; check, and add
`from pathlib import Path` if not.

Run: `uv run pytest -q -p no:cacheprovider -x tests/unit/core/config tests/unit/services`
Expected: green (the env defaults change nothing yet).

- [ ] **Step 2: Write the failing tests.**

```python
"""The packaged serving tier: locator, classification, dir lists (#196)."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_orc.core.config import packaged
from llm_orc.core.config.config_manager import ConfigurationManager

REPO = Path(__file__).resolve().parents[4]


class TestLocator:
    def test_env_path_with_serving_ensemble_wins(
        self, packaged_serving_project: Path
    ) -> None:
        assert packaged.packaged_serving_project_dir() == packaged_serving_project

    def test_empty_env_disables_the_tier(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(packaged.SERVING_PROJECT_ENV, "")
        assert packaged.packaged_serving_project_dir() is None

    def test_env_path_without_serving_ensemble_is_loud(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(packaged.SERVING_PROJECT_ENV, str(tmp_path))
        with pytest.raises(FileNotFoundError, match="LLM_ORC_SERVING_PROJECT_DIR"):
            packaged.packaged_serving_project_dir()

    def test_unset_env_finds_this_checkouts_serving_project(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(packaged.SERVING_PROJECT_ENV)
        found = packaged.packaged_serving_project_dir()
        assert found == REPO / ".llm-orc"
        assert packaged.has_serving_ensemble(found)


class TestTiers:
    def test_packaged_is_the_last_ensembles_and_profiles_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, packaged_serving_project: Path
    ) -> None:
        project = tmp_path / "proj"
        (project / ".llm-orc" / "ensembles").mkdir(parents=True)
        (project / ".llm-orc" / "profiles").mkdir()
        monkeypatch.chdir(tmp_path)  # cwd is NOT the project
        cm = ConfigurationManager(project_dir=project, provision=True)

        ensembles = cm.get_ensembles_dirs()
        profiles = cm.get_profiles_dirs()

        assert ensembles[0] == project / ".llm-orc" / "ensembles"
        assert ensembles[-1] == packaged_serving_project / "ensembles"
        assert cm.global_config_dir / "ensembles" in ensembles
        assert ensembles.index(cm.global_config_dir / "ensembles") < len(ensembles) - 1
        assert profiles[-1] == packaged_serving_project / "profiles"

    def test_classify_tier_names_packaged(
        self, tmp_path: Path, packaged_serving_project: Path
    ) -> None:
        cm = ConfigurationManager(project_dir=tmp_path, provision=False)
        inside = packaged_serving_project / "ensembles" / "agentic-serving" / "serving.yaml"
        assert cm.classify_tier(inside) == "packaged"
        assert cm.classify_tier(cm.global_config_dir / "ensembles" / "x.yaml") == "global"

    def test_checkout_lists_its_dot_dir_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Review Focus 2: in a checkout the packaged dir IS the local dot-dir."""
        monkeypatch.delenv(packaged.SERVING_PROJECT_ENV)
        cm = ConfigurationManager(project_dir=REPO, provision=False)

        dirs = cm.get_ensembles_dirs()

        assert dirs[0] == REPO / ".llm-orc" / "ensembles"
        assert len(dirs) == len(set(dirs))
        assert cm.classify_tier(REPO / ".llm-orc" / "config.yaml") == "local"

    def test_library_dir_follows_the_project_not_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project = tmp_path / "proj"
        (project / ".llm-orc").mkdir(parents=True)
        (project / "llm-orchestra-library" / "ensembles").mkdir(parents=True)
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        monkeypatch.delenv("LLM_ORC_LIBRARY_PATH", raising=False)
        cm = ConfigurationManager(project_dir=project, provision=False)

        assert cm.library_dir == project / "llm-orchestra-library"
        assert project / "llm-orchestra-library" / "ensembles" in cm.get_ensembles_dirs()
        assert cm.classify_tier(project / "llm-orchestra-library" / "x") == "library"
```

`parents[4]` from `tests/unit/core/config/test_packaged_serving.py` is
the repo root; confirm with `python -c` if unsure.

- [ ] **Step 3: Run to verify they fail.**

Run: `uv run pytest tests/unit/core/config/test_packaged_serving.py -q -p no:cacheprovider --no-cov`
Expected: FAIL, `ImportError: cannot import name 'packaged'`.

- [ ] **Step 4: Write the locator module.**

```python
"""The packaged serving project: the repo's ``.llm-orc/`` shipped in the wheel (#196).

Resolution order:

1. ``LLM_ORC_SERVING_PROJECT_DIR``: a path uses that directory as the
   packaged tier and must carry the serving ensemble (a wrong path is a
   loud error, not an empty tier); an empty value disables the tier (the
   test suite's default, see ``tests/conftest.py``).
2. ``llm_orc/serving_project/`` next to this package: a wheel install.
3. The checkout's ``.llm-orc/``, two levels above ``src/llm_orc``: an
   editable install. In a checkout this is also the local project, and
   ``ConfigurationManager`` lists it once.

A candidate counts only when it carries ``ensembles/agentic-serving/serving.yaml``.
"""

from __future__ import annotations

import os
from pathlib import Path

SERVING_PROJECT_ENV = "LLM_ORC_SERVING_PROJECT_DIR"
SERVING_MARKER = Path("ensembles") / "agentic-serving" / "serving.yaml"

_PACKAGE_DIR = Path(__file__).resolve().parents[2]


def has_serving_ensemble(directory: Path) -> bool:
    """True when ``directory`` is a serving project (carries the marker)."""
    return (directory / SERVING_MARKER).is_file()


def packaged_serving_project_dir() -> Path | None:
    """The packaged serving project, or None when there is none."""
    override = os.environ.get(SERVING_PROJECT_ENV)
    if override is not None:
        if override == "":
            return None
        path = Path(override).resolve()
        if not has_serving_ensemble(path):
            raise FileNotFoundError(
                f"{SERVING_PROJECT_ENV}={override!r} has no {SERVING_MARKER}"
            )
        return path
    for candidate in (
        _PACKAGE_DIR / "serving_project",
        _PACKAGE_DIR.parent.parent / ".llm-orc",
    ):
        if has_serving_ensemble(candidate):
            return candidate
    return None
```

- [ ] **Step 5: Wire the tier into ConfigurationManager.** In
  `config_manager.py`:

Import at the top (local imports group):

```python
from llm_orc.core.config.packaged import packaged_serving_project_dir
```

In `__init__`, after `self._global_config_dir = ...` and the local-dir
resolution, add:

```python
        # The read-only tier shipped in the wheel (#196), lowest precedence.
        # In a checkout it is the local dot-dir itself; the dir lists and
        # the profile merge each skip it then, so nothing is read twice.
        self._packaged_serving_dir = packaged_serving_project_dir()
```

Add, next to the `local_config_dir` property:

```python
    @property
    def packaged_serving_dir(self) -> Path | None:
        """The packaged serving project (read-only), if this install has one."""
        return self._packaged_serving_dir

    @property
    def library_dir(self) -> Path:
        """The library base directory, whether or not it exists.

        ``LLM_ORC_LIBRARY_PATH`` wins; else the submodule location under
        the checkout root, which is the local dot-dir's parent when there
        is a project and the cwd otherwise. Derived from the project so a
        manager built with an explicit ``project_dir`` does not read a
        library off an unrelated cwd (S1 finding).
        """
        library_path_env = os.environ.get("LLM_ORC_LIBRARY_PATH")
        if library_path_env:
            return Path(library_path_env)
        root = (
            self._local_config_dir.parent
            if self._local_config_dir is not None
            else Path.cwd()
        )
        return root / "llm-orchestra-library"

    def _is_packaged_distinct(self) -> bool:
        """True when the packaged tier exists and is not the local dot-dir."""
        packaged = self._packaged_serving_dir
        if packaged is None:
            return False
        local = self._local_config_dir
        return local is None or packaged.resolve() != local.resolve()
```

Replace `classify_tier` and `_get_library_dir`:

```python
    def classify_tier(self, path: Path) -> str:
        """Classify a path as local, library, global, packaged, or unknown.

        Args:
            path: Path to classify (file or directory).

        Returns:
            One of ``"local"``, ``"library"``, ``"global"``, ``"packaged"``,
            or ``"unknown"``. A checkout's dot-dir is ``"local"`` even
            though it is also the packaged tier.
        """
        if self._local_config_dir and path.is_relative_to(self._local_config_dir):
            return "local"
        if path.is_relative_to(self.library_dir):
            return "library"
        if path.is_relative_to(self._global_config_dir):
            return "global"
        if self._packaged_serving_dir is not None and path.is_relative_to(
            self._packaged_serving_dir
        ):
            return "packaged"
        return "unknown"
```

Delete `_get_library_dir` (its only callers were `classify_tier` and the
two dir lists; grep to confirm: `grep -rn "_get_library_dir" src tests`).
If a test patches it by name, update that test to patch `library_dir`.

Rewrite `get_ensembles_dirs` and `get_profiles_dirs` (identical shape;
shown for ensembles, mirror for profiles with `"profiles"`):

```python
    def get_ensembles_dirs(self) -> list[Path]:
        """Ensemble directories in priority order.

        local → library → global → packaged. Each entry appears only when
        it exists; the packaged entry is skipped in a checkout, where it
        is the local dot-dir.
        """
        return self._tier_dirs("ensembles")

    def get_profiles_dirs(self) -> list[Path]:
        """Profile directories in priority order (same tiers as ensembles)."""
        return self._tier_dirs("profiles")

    def _tier_dirs(self, subdir: str) -> list[Path]:
        candidates: list[Path | None] = [
            self._local_config_dir,
            self.library_dir,
            self._global_config_dir,
            self._packaged_serving_dir if self._is_packaged_distinct() else None,
        ]
        return [
            base / subdir
            for base in candidates
            if base is not None and (base / subdir).exists()
        ]
```

Remove the now-unused `import os` inside the old `get_ensembles_dirs`.

- [ ] **Step 6: Run the new tests, then the config and handler suites.**

Run: `uv run pytest tests/unit/core/config/test_packaged_serving.py -q -p no:cacheprovider --no-cov`
Expected: PASS.

Run: `uv run pytest -q -p no:cacheprovider tests/unit/core/config tests/unit/services tests/unit/mcp_server tests/unit/cli tests/unit/cli_modules tests/bdd`
Expected: green. A test that mocked `_get_library_dir` or asserted the
exact `get_ensembles_dirs()` list from a cwd with a library must be
updated to the new property name only; do not weaken any assertion.

- [ ] **Step 7: Mutants (report the red line in the commit body).**
  1. In `_tier_dirs`, drop the packaged candidate → the `[-1]`
     assertions in the tiers test fail.
  2. In `classify_tier`, delete the packaged branch → the classify test
     fails with `"unknown"`.
  3. In `library_dir`, use `Path.cwd()` unconditionally → the
     library-follows-project test fails.
  Restore after each.

- [ ] **Step 8: Lint and commit (two commits).**

```bash
make lint
git add tests/conftest.py
git commit -m "test: suite runs with no packaged serving tier and a per-test state dir"
git add src/llm_orc/core/config/packaged.py src/llm_orc/core/config/config_manager.py tests/unit/core/config/test_packaged_serving.py
git commit -m "feat: packaged serving project is the lowest configuration tier

Locator honors LLM_ORC_SERVING_PROJECT_DIR (empty disables), finds
llm_orc/serving_project in a wheel and .llm-orc in a checkout.
classify_tier returns packaged; ensembles and profiles dir lists end
with it; the library dir derives from the project, not cwd.
Mutants: <list the three and the red assertion lines>."
```

---

## Task 2: runtime profiles and config merges read the packaged tier

**Files:**
- Modify: `src/llm_orc/core/config/config_manager.py` (`get_model_profiles`,
  `_get_profile_file_mtimes`, `load_performance_config`,
  `load_agentic_serving_config`)
- Test: `tests/unit/core/config/test_packaged_serving.py` (append),
  `tests/unit/web/test_api_v1_models.py` (append one REST pin)

**Interfaces:**
- Consumes: `packaged_serving_dir`, `_is_packaged_distinct()` (Task 1).
- Produces: `ConfigurationManager._profile_tiers() -> list[Path]` (config
  dirs whose profiles resolve at runtime, lowest precedence first:
  packaged, global, local; library excluded); `_load_packaged_config()
  -> dict[str, Any]`.

- [ ] **Step 1: Write the failing tests.** Append to
  `test_packaged_serving.py`:

```python
class TestRuntimeProfiles:
    def test_packaged_profile_resolves_with_no_project_and_empty_global(
        self, tmp_path: Path, packaged_serving_project: Path
    ) -> None:
        cm = ConfigurationManager(project_dir=tmp_path, provision=False)

        model, provider = cm.resolve_model_profile("agentic-tier-cheap-general")

        assert (model, provider) == ("qwen3-8b", "llama-server")
        assert cm.get_model_profiles()["packaged-orch"]["model"] == "qwen3-0.6b"

    def test_within_packaged_profiles_dir_beats_config_yaml(
        self, tmp_path: Path, packaged_serving_project: Path
    ) -> None:
        cm = ConfigurationManager(project_dir=tmp_path, provision=False)
        assert cm.get_model_profiles()["shadowed-by-file"]["model"] == "from-profiles-dir"

    def test_global_local_yaml_override_beats_packaged(
        self, tmp_path: Path, packaged_serving_project: Path
    ) -> None:
        cm = ConfigurationManager(project_dir=tmp_path, provision=False)
        override_dir = cm.global_config_dir / "profiles"
        override_dir.mkdir(parents=True)
        (override_dir / "seat.local.yaml").write_text(
            "name: agentic-tier-cheap-general\nmodel: bigger\nprovider: llama-server\n"
        )

        model, _ = cm.resolve_model_profile("agentic-tier-cheap-general")

        assert model == "bigger"

    def test_project_beats_global_beats_packaged(
        self, tmp_path: Path, packaged_serving_project: Path
    ) -> None:
        project = tmp_path / "proj"
        (project / ".llm-orc" / "profiles").mkdir(parents=True)
        (project / ".llm-orc" / "profiles" / "seat.yaml").write_text(
            "name: agentic-tier-cheap-general\nmodel: project\nprovider: llama-server\n"
        )
        cm = ConfigurationManager(project_dir=project, provision=False)
        (cm.global_config_dir / "profiles").mkdir(parents=True)
        (cm.global_config_dir / "profiles" / "seat.yaml").write_text(
            "name: agentic-tier-cheap-general\nmodel: global\nprovider: llama-server\n"
        )

        assert cm.resolve_model_profile("agentic-tier-cheap-general")[0] == "project"

    def test_library_profiles_stay_invisible_at_runtime(
        self, tmp_path: Path, packaged_serving_project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pins today's behavior (S1 step 1): listed, never resolved."""
        project = tmp_path / "proj"
        (project / ".llm-orc").mkdir(parents=True)
        lib_profiles = project / "llm-orchestra-library" / "profiles"
        lib_profiles.mkdir(parents=True)
        (lib_profiles / "lib-only.yaml").write_text(
            "name: lib-only\nmodel: m\nprovider: ollama\n"
        )
        monkeypatch.delenv("LLM_ORC_LIBRARY_PATH", raising=False)
        cm = ConfigurationManager(project_dir=project, provision=False)

        assert lib_profiles in cm.get_profiles_dirs()
        assert "lib-only" not in cm.get_model_profiles()

    def test_checkout_merges_its_dot_dir_once_at_top_precedence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Review Focus 2: packaged == local must not be merged below global."""
        monkeypatch.delenv(packaged.SERVING_PROJECT_ENV)
        cm = ConfigurationManager(project_dir=REPO, provision=False)

        tiers = cm._profile_tiers()

        assert tiers[-1] == REPO / ".llm-orc"
        assert tiers.count(REPO / ".llm-orc") == 1

    def test_cache_invalidates_when_a_packaged_profile_changes(
        self, tmp_path: Path, packaged_serving_project: Path
    ) -> None:
        cm = ConfigurationManager(project_dir=tmp_path, provision=False)
        assert cm.resolve_model_profile("agentic-tier-cheap-general")[0] == "qwen3-8b"
        target = packaged_serving_project / "profiles" / "agentic-tier-cheap-general.yaml"
        target.write_text("name: agentic-tier-cheap-general\nmodel: edited\nprovider: llama-server\n")
        import os
        os.utime(target, (target.stat().st_atime, target.stat().st_mtime + 5))

        assert cm.resolve_model_profile("agentic-tier-cheap-general")[0] == "edited"


class TestConfigMerges:
    def test_agentic_serving_orchestrator_comes_from_packaged(
        self, tmp_path: Path, packaged_serving_project: Path
    ) -> None:
        cm = ConfigurationManager(project_dir=tmp_path, provision=False)
        assert cm.load_agentic_serving_config()["orchestrator"]["model_profile"] == "packaged-orch"

    def test_global_overrides_packaged_agentic_serving(
        self, tmp_path: Path, packaged_serving_project: Path
    ) -> None:
        cm = ConfigurationManager(project_dir=tmp_path, provision=False)
        cm.global_config_dir.mkdir(parents=True, exist_ok=True)
        (cm.global_config_dir / "config.yaml").write_text(
            "agentic_serving:\n  orchestrator:\n    model_profile: mine\n"
        )
        assert cm.load_agentic_serving_config()["orchestrator"]["model_profile"] == "mine"

    def test_performance_merge_includes_packaged(
        self, tmp_path: Path, packaged_serving_project: Path
    ) -> None:
        (packaged_serving_project / "config.yaml").write_text(
            "performance:\n  execution:\n    default_timeout: 777\n"
        )
        cm = ConfigurationManager(project_dir=tmp_path, provision=False)
        assert cm.load_performance_config()["execution"]["default_timeout"] == 777
```

And in `tests/unit/web/test_api_v1_models.py` append (read its imports
first; it builds a `TestClient` around `create_app()` and monkeypatches
`get_model_profile_allowlist` in most tests; this pin must NOT patch it):

```python
def test_v1_models_from_an_empty_cwd_lists_the_packaged_orchestrator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, packaged_serving_project: Path
) -> None:
    """The real allowlist over a real ConfigurationManager: no project, no
    global config, only the packaged tier names the orchestrator seat."""
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.chdir(empty)

    with TestClient(create_app()) as client:
        response = client.get("/v1/models")

    assert response.status_code == 200
    assert "packaged-orch" in [m["id"] for m in response.json()["data"]]
```

- [ ] **Step 2: Run to verify they fail.**

Run: `uv run pytest tests/unit/core/config/test_packaged_serving.py tests/unit/web/test_api_v1_models.py -q -p no:cacheprovider --no-cov`
Expected: the runtime-profile tests fail with `ValueError: Model profile
'agentic-tier-cheap-general' not found`; the merge tests with `"default"`
/ `60`; the REST pin with `"packaged-orch"` absent.

- [ ] **Step 3: Rewrite the merge as a tier loop.** Replace
  `get_model_profiles` and `_get_profile_file_mtimes`:

```python
    def _profile_tiers(self) -> list[Path]:
        """Config dirs whose profiles resolve at runtime, lowest precedence first.

        packaged → global → local. The library tier is listed by
        ``get_profiles_dirs`` but never resolved here, as before this
        tier loop existed: a submodule profile must not shadow a global
        one by name. In a checkout the packaged dir is the local dot-dir
        and is skipped at the bottom so it merges once, at the top.
        """
        tiers: list[Path] = []
        if self._is_packaged_distinct():
            assert self._packaged_serving_dir is not None
            tiers.append(self._packaged_serving_dir)
        tiers.append(self._global_config_dir)
        if self._local_config_dir is not None:
            tiers.append(self._local_config_dir)
        return tiers

    def get_model_profiles(self) -> dict[str, dict[str, str]]:
        """Get merged model profiles from every runtime tier.

        Within a tier ``config.yaml: model_profiles`` loads first and
        ``profiles/*.yaml`` after it (``*.local.yaml`` last), so a file
        beats the config entry of the same name and a later tier beats an
        earlier one. Results are cached and invalidated when any source
        file's mtime changes.
        """
        current_mtimes = self._get_profile_file_mtimes()
        if (
            self._profiles_cache is not None
            and current_mtimes == self._profiles_cache_mtimes
        ):
            return self._profiles_cache

        merged: dict[str, dict[str, str]] = {}
        for tier in self._profile_tiers():
            config_file = tier / "config.yaml"
            if config_file.exists():
                with open(config_file) as f:
                    data = yaml.safe_load(f) or {}
                merged.update(data.get("model_profiles") or {})
            self._load_profile_yaml_files(tier / "profiles", merged)

        self._profiles_cache = merged
        self._profiles_cache_mtimes = current_mtimes
        return merged

    def _get_profile_file_mtimes(self) -> dict[str, float]:
        """Modification times of every runtime profile source, keyed by path."""
        mtimes: dict[str, float] = {}
        for tier in self._profile_tiers():
            config_file = tier / "config.yaml"
            if config_file.exists():
                mtimes[str(config_file)] = config_file.stat().st_mtime
            profiles_dir = tier / "profiles"
            if profiles_dir.exists():
                for f in profiles_dir.glob("*.yaml"):
                    mtimes[str(f)] = f.stat().st_mtime
        return mtimes
```

Add the packaged overlay to both config merges. New helper next to
`_load_global_config`:

```python
    def _load_packaged_config(self) -> dict[str, Any]:
        """The packaged serving project's ``config.yaml``, or ``{}``.

        Empty in a checkout (the file is then the local config and is
        merged as such) and when this install has no packaged tier.
        """
        if not self._is_packaged_distinct():
            return {}
        assert self._packaged_serving_dir is not None
        config_file = self._packaged_serving_dir / "config.yaml"
        if not config_file.exists():
            return {}
        with open(config_file) as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
```

In `load_performance_config`, before the global overlay:

```python
        packaged_performance = self._load_packaged_config().get("performance", {})
        ...
        # Merge configurations: defaults -> packaged -> global -> local
        merged_config = defaults.copy()
        self._deep_merge_dict(merged_config, packaged_performance)
        self._deep_merge_dict(merged_config, global_performance)
        self._deep_merge_dict(merged_config, local_performance)
```

In `load_agentic_serving_config`, the same shape:

```python
        packaged_section = self._load_packaged_config().get("agentic_serving") or {}
        if not isinstance(packaged_section, dict):
            packaged_section = {}
        ...
        self._deep_merge_dict(defaults, packaged_section)
        self._deep_merge_dict(defaults, global_section)
        self._deep_merge_dict(defaults, local_section)
```

Update the docstrings of both to name the packaged tier.

- [ ] **Step 4: Run the tests.**

Run: `uv run pytest tests/unit/core/config tests/unit/web/test_api_v1_models.py -q -p no:cacheprovider --no-cov`
Expected: PASS. The existing profile tests in `test_config.py` must
stay green untouched; if one asserted the old mtime key names
(`"global"`, `"local_prof_..."`), update only the key expectation.

- [ ] **Step 5: Mutants.**
  1. `_profile_tiers`: drop the packaged entry → the first runtime test
     fails with the `ValueError`.
  2. `_profile_tiers`: order `[global, packaged, local]` → the
     `.local.yaml` override test fails (`"qwen3-8b"`).
  3. `_profile_tiers`: append the library profiles dir → the invisibility
     test fails.
  4. `load_agentic_serving_config`: drop the packaged merge → orchestrator
     test fails with `"default"`.
  Restore after each; report in the commit body.

- [ ] **Step 6: Lint, full suite, commit.**

```bash
make lint && uv run pytest -q -p no:cacheprovider
git add src/llm_orc/core/config/config_manager.py tests/unit/core/config/test_packaged_serving.py tests/unit/web/test_api_v1_models.py
git commit -m "feat: runtime model profiles and config merges read the packaged tier

get_model_profiles is a tier loop (packaged, global, local; each tier
config.yaml then profiles/, .local.yaml last); library profiles stay
invisible at runtime, pinned. performance: and agentic_serving: merge
defaults, packaged, global, local. /v1/models from an empty cwd lists
the packaged orchestrator seat. Mutants: <four, with red lines>."
```

---

## Task 3: the script resolver searches the packaged tier

**Files:**
- Modify: `src/llm_orc/core/execution/scripting/resolver.py`
  (`_get_search_paths` ~68-113, `list_available_scripts` ~286)
- Test: `tests/unit/core/execution/scripting/test_resolver_packaged.py`

**Interfaces:**
- Consumes: `packaged_serving_project_dir()` (Task 1).
- Produces: search order project (three entries) → package primitives →
  library → global → packaged (`<pkg>/scripts`, `<pkg>`);
  `list_available_scripts()` reads `self._project_dir` when set.

- [ ] **Step 1: Write the failing tests.**

```python
"""Packaged serving scripts resolve when no project carries them (#196)."""

from __future__ import annotations

from pathlib import Path

from llm_orc.core.config.config_manager import resolve_global_config_dir
from llm_orc.core.execution.scripting.resolver import ScriptResolver


class TestResolverPackagedTier:
    def test_serving_script_resolves_from_packaged_with_empty_project(
        self, tmp_path: Path, packaged_serving_project: Path
    ) -> None:
        resolved = ScriptResolver(project_dir=tmp_path).resolve_script_path(
            "scripts/agentic_serving/classify.py"
        )
        assert Path(resolved) == packaged_serving_project / "scripts" / "agentic_serving" / "classify.py"

    def test_project_script_shadows_packaged(
        self, tmp_path: Path, packaged_serving_project: Path
    ) -> None:
        mine = tmp_path / ".llm-orc" / "scripts" / "agentic_serving" / "classify.py"
        mine.parent.mkdir(parents=True)
        mine.write_text("print('mine')\n")

        resolved = ScriptResolver(project_dir=tmp_path).resolve_script_path(
            "scripts/agentic_serving/classify.py"
        )

        assert Path(resolved) == mine

    def test_global_script_shadows_packaged(
        self, tmp_path: Path, packaged_serving_project: Path
    ) -> None:
        target = resolve_global_config_dir() / "scripts" / "agentic_serving" / "classify.py"
        target.parent.mkdir(parents=True)
        target.write_text("print('global')\n")

        resolved = ScriptResolver(project_dir=tmp_path).resolve_script_path(
            "scripts/agentic_serving/classify.py"
        )

        assert Path(resolved) == target

    def test_packaged_dirs_are_the_last_search_paths(
        self, tmp_path: Path, packaged_serving_project: Path
    ) -> None:
        paths = ScriptResolver(project_dir=tmp_path)._get_search_paths()
        assert paths[-2:] == [
            str(packaged_serving_project / "scripts"),
            str(packaged_serving_project),
        ]

    def test_no_packaged_tier_adds_nothing(self, tmp_path: Path) -> None:
        paths = ScriptResolver(project_dir=tmp_path)._get_search_paths()
        assert not any("serving_project" in p for p in paths)


class TestListAvailableScriptsHonorsProjectDir:
    def test_lists_the_project_scripts_not_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project = tmp_path / "proj"
        script = project / ".llm-orc" / "scripts" / "util" / "hello.py"
        script.parent.mkdir(parents=True)
        script.write_text("print('hi')\n")
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)

        names = {s["relative_path"] for s in ScriptResolver(project_dir=project).list_available_scripts()}

        assert "util/hello.py" in names
```

Add `import pytest` to the imports. Check `list_available_scripts`'s
dict keys by reading `_collect_local_scripts` (`relative_path` is the
path under `scripts/`; if it is stored differently, assert on the key
that carries `util/hello.py`).

- [ ] **Step 2: Run to verify they fail.**

Run: `uv run pytest tests/unit/core/execution/scripting/test_resolver_packaged.py -q -p no:cacheprovider --no-cov`
Expected: the first test fails (the resolver returns the reference
unchanged or a wrong path); the search-path and listing tests fail.

- [ ] **Step 3: Implement.** In `_get_search_paths`, after the global
  scripts block:

```python
        # Priority 4: the packaged serving project (#196), the lowest tier.
        # Skipped in a checkout, where it is the project's own dot-dir
        # and already covered by priority 1.
        packaged = packaged_serving_project_dir()
        if packaged is not None and packaged.resolve() != (
            base / self.LLM_ORC_DIR
        ).resolve():
            search_paths.extend(
                [str(packaged / self.SCRIPTS_DIR), str(packaged)]
            )
```

with `from llm_orc.core.config.packaged import packaged_serving_project_dir`
imported at the top. In `list_available_scripts` replace
`cwd = Path(os.getcwd())` / `scripts_dir = cwd / ...` with:

```python
        base = self._project_dir or Path(os.getcwd())
        scripts_dir = base / self.LLM_ORC_DIR / self.SCRIPTS_DIR
```

Update the `_get_search_paths` docstring to list all five priorities.

- [ ] **Step 4: Run.**

Run: `uv run pytest tests/unit/core/execution/scripting -q -p no:cacheprovider --no-cov`
Expected: PASS.

- [ ] **Step 5: Mutants.**
  1. Delete the priority-4 block → first test red (path mismatch).
  2. Insert the packaged entries before the global block → the
     global-shadows-packaged test red.
  3. `list_available_scripts` back to `os.getcwd()` → listing test red.

- [ ] **Step 6: Lint and commit.**

```bash
make lint
git add src/llm_orc/core/execution/scripting/resolver.py tests/unit/core/execution/scripting/test_resolver_packaged.py
git commit -m "feat: script resolver searches the packaged serving project last

list_available_scripts honors project_dir instead of cwd.
Mutants: <three, with red lines>."
```

---

## Task 4: the state dir, the serve flag, and the router preset

**Files:**
- Create: `src/llm_orc/core/config/state.py`
- Modify: `src/llm_orc/cli.py` (`serve` command, ~455-494)
- Modify: `src/llm_orc/providers/llama_server.py` (`start_router_from_config`, ~368-400)
- Test: `tests/unit/core/config/test_state.py`,
  `tests/unit/providers/test_llama_server.py` (replace the global-fallback
  test at ~439), `tests/unit/cli_modules/test_cli_serve_backend.py` (append)

**Interfaces:**
- Produces:
  - `state.STATE_DIR_ENV = "LLM_ORC_STATE_DIR"`
  - `state.ARTIFACTS_DIRNAME = "artifacts"`, `state.TRACE_DIRNAME = ".serve-trace"`,
    `state.CACHE_DIRNAME = "cache"`
  - `state.resolve_state_dir(local_config_dir: Path | None) -> Path`
  - `llm-orc serve --state-dir PATH` sets `LLM_ORC_STATE_DIR` for the process
  - the preset lands at `resolve_state_dir(config_manager.local_config_dir) / PRESET_FILENAME`

- [ ] **Step 1: Write the failing tests.** `tests/unit/core/config/test_state.py`:

```python
"""Where the serve writes: env, then the project, then XDG state (#196)."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_orc.core.config import state


class TestResolveStateDir:
    def test_env_override_wins_over_everything(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(state.STATE_DIR_ENV, str(tmp_path / "explicit"))
        assert state.resolve_state_dir(tmp_path / ".llm-orc") == tmp_path / "explicit"

    def test_project_dot_dir_when_there_is_a_project(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(state.STATE_DIR_ENV, raising=False)
        assert state.resolve_state_dir(tmp_path / ".llm-orc") == tmp_path / ".llm-orc"

    def test_xdg_state_home_without_a_project(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(state.STATE_DIR_ENV, raising=False)
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg"))
        assert state.resolve_state_dir(None) == tmp_path / "xdg" / "llm-orc"

    def test_home_local_state_when_nothing_is_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(state.STATE_DIR_ENV, raising=False)
        monkeypatch.delenv("XDG_STATE_HOME", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        assert state.resolve_state_dir(None) == tmp_path / ".local" / "state" / "llm-orc"

    def test_empty_env_is_unset(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(state.STATE_DIR_ENV, "")
        assert state.resolve_state_dir(tmp_path / ".llm-orc") == tmp_path / ".llm-orc"
```

In `tests/unit/providers/test_llama_server.py`, read the test at ~416
(`renders preset into config dir`) and ~439 (`falls back to global`).
Replace the second with:

```python
    def test_lands_in_the_state_dir_without_a_local_project(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Review Focus 3: the state dir does not exist yet and is created."""
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg"))
        config = Mock()
        config.local_config_dir = None
        config.global_config_dir = tmp_path / "global"
        config.get_model_profiles.return_value = {
            "p": {"model": "m", "provider": "llama-server", "hf_repo": "x/y"}
        }
        with patch.object(llama_server.LlamaServerSupervisor, "start"):
            supervisor = llama_server.start_router_from_config(config, binary="stub")

        expected = tmp_path / "xdg" / "llm-orc" / "llama-server.ini"
        assert supervisor.preset_path == expected
        assert expected.exists()
        assert not (tmp_path / "global" / "llama-server.ini").exists()
```

Match the surrounding tests' way of stubbing the supervisor and the
profile shape exactly (copy from the test at ~416); the assertions are
what matter. In `tests/unit/cli_modules/test_cli_serve_backend.py`
append:

```python
    def test_state_dir_flag_sets_the_env_for_the_process(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("LLM_ORC_STATE_DIR", raising=False)
        seen: dict[str, str | None] = {}

        def fake_uvicorn_run(*args: object, **kwargs: object) -> None:
            seen["state"] = os.environ.get("LLM_ORC_STATE_DIR")

        with patch("uvicorn.run", side_effect=fake_uvicorn_run):
            result = CliRunner().invoke(
                cli, ["serve", "--no-backend", "--state-dir", str(tmp_path / "st")]
            )

        assert result.exit_code == 0, result.output
        assert seen["state"] == str(tmp_path / "st")
```

- [ ] **Step 2: Run to verify they fail.**

Run: `uv run pytest tests/unit/core/config/test_state.py tests/unit/providers/test_llama_server.py tests/unit/cli_modules/test_cli_serve_backend.py -q -p no:cacheprovider --no-cov`
Expected: `ImportError` for `state`; the preset test asserts the global
path; the CLI test fails with `no such option: --state-dir`.

- [ ] **Step 3: Write the module.**

```python
"""Runtime state: where the serve writes (#196).

Artifacts, the turn trace, the script cache and the rendered router
preset are state, not configuration, and the packaged serving project is
read-only. They land, in order of precedence:

1. ``LLM_ORC_STATE_DIR`` (``llm-orc serve --state-dir`` sets it);
2. the project's ``.llm-orc/`` when there is one, today's layout, so
   instruments reading ``.llm-orc/.serve-trace/`` and
   ``.llm-orc/artifacts/`` in a checkout are unchanged;
3. ``$XDG_STATE_HOME/llm-orc``, default ``~/.local/state/llm-orc``.

Every writer and every reader of these files resolves the directory
through :func:`resolve_state_dir`; none derives it from cwd.
"""

from __future__ import annotations

import os
from pathlib import Path

STATE_DIR_ENV = "LLM_ORC_STATE_DIR"
ARTIFACTS_DIRNAME = "artifacts"
TRACE_DIRNAME = ".serve-trace"
CACHE_DIRNAME = "cache"


def resolve_state_dir(local_config_dir: Path | None) -> Path:
    """The directory runtime state lives in (not created here)."""
    override = os.environ.get(STATE_DIR_ENV)
    if override:
        return Path(override)
    if local_config_dir is not None:
        return local_config_dir
    xdg_state_home = os.environ.get("XDG_STATE_HOME")
    base = Path(xdg_state_home) if xdg_state_home else Path.home() / ".local" / "state"
    return base / "llm-orc"
```

- [ ] **Step 4: The preset.** In `start_router_from_config` replace the
  `config_dir = Path(config_manager.local_config_dir or config_manager.global_config_dir)`
  line with:

```python
    config_dir = resolve_state_dir(config_manager.local_config_dir)
```

(import `from llm_orc.core.config.state import resolve_state_dir`), keep
the `mkdir(parents=True, exist_ok=True)`, and update the docstring: "The
preset lands in the state dir (the local ``.llm-orc`` when there is one,
else the XDG state dir) so an operator can read exactly what the router
was given."

- [ ] **Step 5: The flag.** In `cli.py`'s `serve` command add, after the
  `--models-max` option:

```python
@click.option(
    "--state-dir",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    help=(
        "Where artifacts, the turn trace and the router preset are written "
        "(default: the project's .llm-orc, else $XDG_STATE_HOME/llm-orc)"
    ),
)
```

add `state_dir: Path | None` to the signature, and as the first
statement of the body:

```python
    if state_dir is not None:
        os.environ["LLM_ORC_STATE_DIR"] = str(state_dir)
```

(`import os` at module level if `cli.py` lacks it; check `from pathlib
import Path` is imported.) Mention the flag in the docstring's last
paragraph: "``--state-dir`` overrides where runtime state is written;
``LLM_ORC_STATE_DIR`` is the same setting as an environment variable."

- [ ] **Step 6: Run.**

Run: `uv run pytest tests/unit/core/config/test_state.py tests/unit/providers tests/unit/cli_modules -q -p no:cacheprovider --no-cov`
Expected: PASS.

- [ ] **Step 7: Mutants.**
  1. `resolve_state_dir`: return `local_config_dir` before checking the
     env → env-override test red.
  2. `start_router_from_config`: restore the `local or global` line →
     the state-dir preset test red (file under `global/`).
  3. `serve`: drop the `os.environ[...]` line → CLI test red (`None`).

- [ ] **Step 8: Lint and commit.**

```bash
make lint
git add src/llm_orc/core/config/state.py src/llm_orc/providers/llama_server.py src/llm_orc/cli.py tests/unit/core/config/test_state.py tests/unit/providers/test_llama_server.py tests/unit/cli_modules/test_cli_serve_backend.py
git commit -m "feat: runtime state dir; the router preset lands there

resolve_state_dir: LLM_ORC_STATE_DIR, else the project's .llm-orc,
else XDG state. llm-orc serve --state-dir sets the variable. A
global-only serve's preset moves from ~/.config/llm-orc to the state
dir. Mutants: <three, with red lines>."
```

---

## Task 5: artifacts, the script cache and the artifact handlers write and read the state dir

**Files:**
- Modify: `src/llm_orc/core/execution/artifact_manager.py` (`__init__`,
  the four `self.base_dir / ".llm-orc" / "artifacts"` sites at ~54, ~249,
  ~350, ~390)
- Modify: `src/llm_orc/core/execution/ensemble_execution.py:261`
  (`ArtifactManager()`), `_load_script_cache_config` (~359)
- Modify: `src/llm_orc/core/execution/scripting/cache.py` (`ScriptCacheConfig`,
  the three `artifact_base_dir / ".llm-orc" / "cache"` sites at ~208, ~239, ~263)
- Modify: `src/llm_orc/services/orchestra_service.py:57` and
  `handle_set_project` (~228)
- Modify: `src/llm_orc/cli_commands.py:540,598`
- Modify: `src/llm_orc/services/handlers/artifact_handler.py` (`__init__`,
  `set_project_context`, `_get_artifacts_base`)
- Modify: `src/llm_orc/services/handlers/resource_handler.py:322-340`
  (`get_artifacts_dir`)
- Test: `tests/unit/core/execution/test_artifact_manager.py` (append),
  `tests/unit/web/test_api_artifacts_state_dir.py` (new),
  `tests/unit/services/handlers/test_resource_handler.py` (append)

**Interfaces:**
- Consumes: `resolve_state_dir`, `ARTIFACTS_DIRNAME`, `CACHE_DIRNAME` (Task 4).
- Produces:
  - `ArtifactManager.__init__(self, base_dir: Path | str = ".", *, artifacts_dir: Path | None = None)`;
    attribute `artifacts_dir: Path` (the directory `<ensemble>/<ts>/` goes under).
    `base_dir` stays for callers that read it; `artifacts_dir` defaults
    to `base_dir / ".llm-orc" / "artifacts"` so every existing caller is
    unchanged.
  - `ScriptCacheConfig.cache_dir: Path` (default `Path(".") / ".llm-orc" / "cache"`,
    the same default as before); `artifact_base_dir` stays.
  - `ArtifactHandler.__init__(self, project_path: Path | None = None, config_manager: ConfigurationManager | None = None)`.

- [ ] **Step 1: Write the failing tests.** Append to
  `tests/unit/core/execution/test_artifact_manager.py` (read its imports
  and fixtures first; use its style):

```python
class TestArtifactsDir:
    def test_explicit_artifacts_dir_is_used_verbatim(self, tmp_path: Path) -> None:
        manager = ArtifactManager(artifacts_dir=tmp_path / "state" / "artifacts")

        created = manager.save_execution_results("ens", {"status": "completed"})

        assert created.parent == tmp_path / "state" / "artifacts" / "ens"
        assert manager.list_ensembles()[0]["name"] == "ens"
        assert not (Path.cwd() / ".llm-orc" / "artifacts" / "ens").exists()

    def test_default_is_base_dir_dot_llm_orc_artifacts(self, tmp_path: Path) -> None:
        manager = ArtifactManager(base_dir=tmp_path)
        assert manager.artifacts_dir == tmp_path / ".llm-orc" / "artifacts"
```

New `tests/unit/web/test_api_artifacts_state_dir.py`:

```python
"""Executing from an empty cwd writes artifacts to the state dir, not cwd (#196).

Through the REST execute route, the real OrchestraService and the real
executor: a script-only ensemble in the global tier runs, and its
artifact lands under $XDG_STATE_HOME/llm-orc/artifacts/.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import llm_orc.web.api as web_api
from llm_orc.core.config.config_manager import resolve_global_config_dir
from llm_orc.web.server import create_app

_SCRIPT = """\
import json, sys
print(json.dumps({"success": True, "data": {"ok": True}}))
"""

_ENSEMBLE = """\
name: state-probe
description: one script agent
agents:
  - name: probe
    script: probe.py
"""


@pytest.fixture
def empty_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.chdir(empty)
    monkeypatch.setattr(web_api, "_orchestra_service", None)
    return empty


def test_execute_writes_artifacts_under_the_state_dir(
    empty_cwd: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_home = Path(str(empty_cwd.parent / "state-home"))
    monkeypatch.setenv("XDG_STATE_HOME", str(state_home))
    global_dir = resolve_global_config_dir()
    (global_dir / "ensembles").mkdir(parents=True)
    (global_dir / "ensembles" / "state-probe.yaml").write_text(_ENSEMBLE)
    (global_dir / "scripts").mkdir()
    (global_dir / "scripts" / "probe.py").write_text(_SCRIPT)

    with TestClient(create_app()) as client:
        response = client.post("/api/ensembles/state-probe/execute", json={"input": "go"})

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "success", response.json()
    artifacts = state_home / "llm-orc" / "artifacts" / "state-probe"
    assert artifacts.is_dir() and any(artifacts.iterdir())
    assert not (empty_cwd / ".llm-orc").exists()
```

Append to `tests/unit/services/handlers/test_resource_handler.py`
(read how it builds a `ResourceHandler`; use a real
`ConfigurationManager(project_dir=tmp_path, provision=False)` here):

```python
class TestArtifactsDirFollowsState:
    def test_no_project_reads_the_state_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg"))
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        handler = ResourceHandler(
            ConfigurationManager(project_dir=tmp_path / "noproj", provision=False),
            EnsembleLoader(),
        )

        assert handler.get_artifacts_dir() == tmp_path / "xdg" / "llm-orc" / "artifacts"

    def test_project_reads_its_dot_dir(self, tmp_path: Path) -> None:
        (tmp_path / ".llm-orc").mkdir()
        handler = ResourceHandler(
            ConfigurationManager(project_dir=tmp_path, provision=False),
            EnsembleLoader(),
        )
        assert handler.get_artifacts_dir() == tmp_path / ".llm-orc" / "artifacts"
```

- [ ] **Step 2: Run to verify they fail.**

Run: `uv run pytest tests/unit/core/execution/test_artifact_manager.py tests/unit/web/test_api_artifacts_state_dir.py tests/unit/services/handlers/test_resource_handler.py -q -p no:cacheprovider --no-cov`
Expected: `TypeError: unexpected keyword argument 'artifacts_dir'`; the
REST pin fails on the artifacts dir assertion (artifacts under the
empty cwd) or on `.llm-orc` existing under cwd; the resource-handler
pin returns a cwd path.

- [ ] **Step 3: ArtifactManager.**

```python
    def __init__(
        self, base_dir: Path | str = ".", *, artifacts_dir: Path | None = None
    ) -> None:
        """Initialize artifact manager.

        Args:
            base_dir: Legacy root; artifacts default to
                ``base_dir/.llm-orc/artifacts``.
            artifacts_dir: The artifacts directory itself. Production
                callers pass ``resolve_state_dir(...) / "artifacts"`` (#196)
                so a serve with no project writes to the state dir, never
                under cwd or the packaged tier.
        """
        self.base_dir = Path(base_dir) if isinstance(base_dir, str) else base_dir
        self.artifacts_dir = (
            artifacts_dir
            if artifacts_dir is not None
            else self.base_dir / ".llm-orc" / "artifacts"
        )
```

Replace each `artifacts_dir = self.base_dir / ".llm-orc" / "artifacts"`
with `artifacts_dir = self.artifacts_dir`.

- [ ] **Step 4: The executor and the service.** In
  `ensemble_execution.py` line 261:

```python
        self._artifact_manager = ArtifactManager(
            artifacts_dir=resolve_state_dir(self._config_manager.local_config_dir)
            / ARTIFACTS_DIRNAME
        )
```

Check that `self._config_manager` is assigned before line 261 (it is a
constructor parameter `_config_manager`; read lines 195-262). Import
`from llm_orc.core.config.state import ARTIFACTS_DIRNAME, CACHE_DIRNAME, resolve_state_dir`.

In `_load_script_cache_config`, where the `ScriptCacheConfig(...)` is
built, pass
`cache_dir=resolve_state_dir(self._config_manager.local_config_dir) / CACHE_DIRNAME`.
In `cache.py` add to `ScriptCacheConfig`:

```python
    cache_dir: Path = field(default_factory=lambda: Path(".") / ".llm-orc" / "cache")
```

and replace the three `self.config.artifact_base_dir / ".llm-orc" / "cache"`
with `self.config.cache_dir`.

In `orchestra_service.py`, line 57 and inside `handle_set_project`
(after `self.config_manager = ctx.config_manager`):

```python
        self.artifact_manager = ArtifactManager(
            artifacts_dir=resolve_state_dir(self.config_manager.local_config_dir)
            / ARTIFACTS_DIRNAME
        )
```

`ExecutionHandler` holds the manager it was constructed with; after
`handle_set_project` rebuilds it, also pass the new one:
`self._execution_handler._artifact_manager = self.artifact_manager` is
reaching into a private; instead give `ExecutionHandler.set_project_context`
the manager: read its `set_project_context(ctx)` and add a second
method `set_artifact_manager(self, manager: ArtifactManager) -> None`
that the service calls right after rebuilding. Construct
`ArtifactHandler(config_manager=self.config_manager)` at the service's
init.

In `cli_commands.py` (both sites):

```python
    from llm_orc.core.config.config_manager import ConfigurationManager
    from llm_orc.core.config.state import ARTIFACTS_DIRNAME, resolve_state_dir
    from llm_orc.core.execution.artifact_manager import ArtifactManager

    local = ConfigurationManager(provision=False).local_config_dir
    manager = ArtifactManager(artifacts_dir=resolve_state_dir(local) / ARTIFACTS_DIRNAME)
```

- [ ] **Step 5: The handlers.** `artifact_handler.py`:

```python
    def __init__(
        self,
        project_path: Path | None = None,
        config_manager: ConfigurationManager | None = None,
    ) -> None:
        """Initialize with optional project path and configuration manager."""
        self._project_path = project_path
        self._config_manager = config_manager

    def set_project_context(self, ctx: ProjectContext) -> None:
        """Update handler to use new project context."""
        self._project_path = ctx.project_path
        self._config_manager = ctx.config_manager

    def _get_artifacts_base(self) -> Path:
        """The artifacts directory, through the state rule (#196)."""
        if self._config_manager is not None:
            local = self._config_manager.local_config_dir
        elif self._project_path is not None and (self._project_path / ".llm-orc").is_dir():
            local = self._project_path / ".llm-orc"
        else:
            local = None
        return resolve_state_dir(local) / ARTIFACTS_DIRNAME
```

(add the `ConfigurationManager` import under `TYPE_CHECKING` if the
module has that block, else a plain import; mirror `script_handler.py`.)

`resource_handler.get_artifacts_dir`: keep the first branch (a
`global_config_dir` that is itself named `artifacts`; it exists for
tests, leave it), and replace the rest with:

```python
        return resolve_state_dir(self._config_manager.local_config_dir) / ARTIFACTS_DIRNAME
```

- [ ] **Step 6: Run.**

Run: `uv run pytest -q -p no:cacheprovider tests/unit/core/execution tests/unit/services tests/unit/mcp_server tests/unit/web tests/unit/cli`
Expected: PASS. Existing tests that constructed `ArtifactManager(tmp)`
and asserted `tmp/.llm-orc/artifacts` still pass (the default is kept).
A resource-handler test that asserted the cwd fallback must move to the
state rule; keep its assertion strength.

- [ ] **Step 7: Mutants.**
  1. `ensemble_execution.py`: back to `ArtifactManager()` → the REST
     pin red (artifacts under cwd or the `.llm-orc` assertion).
  2. `resource_handler.get_artifacts_dir`: back to `Path.cwd()` → the
     no-project pin red.
  3. `ArtifactManager.save_execution_results`: use `self.base_dir /
     ".llm-orc" / "artifacts"` → the explicit-dir test red.

- [ ] **Step 8: Lint, full suite, commit (structural first).**

```bash
make lint && uv run pytest -q -p no:cacheprovider
git add src/llm_orc/core/execution/artifact_manager.py src/llm_orc/core/execution/scripting/cache.py tests/unit/core/execution/test_artifact_manager.py
git commit -m "refactor: ArtifactManager and ScriptCacheConfig take their directories explicitly

Defaults unchanged (base_dir/.llm-orc/{artifacts,cache})."
git add -A src tests
git commit -m "feat: artifacts and the script cache go through the state dir

Executor, service, CLI artifact commands, the artifact and resource
handlers all resolve artifacts via resolve_state_dir. A serve executing
from an empty cwd writes under $XDG_STATE_HOME/llm-orc/artifacts and
nothing under cwd (pinned over REST). Mutants: <three, with red lines>."
```

---

## Task 6: the serving root, the caller factory, and child ensembles across tiers

**Files:**
- Modify: `src/llm_orc/core/config/config_manager.py` (add `serving_root`)
- Modify: `src/llm_orc/web/api/v1_chat_completions.py:111-132`
- Modify: `src/llm_orc/core/execution/ensemble_execution.py`
  (`_resolve_ensemble_reference`, ~1097-1115)
- Test: `tests/unit/core/config/test_packaged_serving.py` (append),
  `tests/unit/web/test_serving_root.py` (new),
  `tests/unit/core/execution/test_child_ensembles_across_tiers.py` (new)

**Interfaces:**
- Consumes: `has_serving_ensemble`, `SERVING_MARKER` (Task 1);
  `resolve_state_dir`, `TRACE_DIRNAME` (Task 4).
- Produces:
  - `ConfigurationManager.serving_root() -> Path` (raises `FileNotFoundError`)
  - `v1_chat_completions.get_serving_ensemble_caller()` builds the caller
    on `serving_root()` with an explicit `trace_root`; `_resolve_serving_project_dir`
    is deleted.
  - `EnsembleExecutor._resolve_ensemble_reference` searches
    `project_dir/ensembles`, `project_dir`, then every entry of
    `config_manager.get_ensembles_dirs()`.

- [ ] **Step 1: Write the failing tests.** Append to
  `test_packaged_serving.py`:

```python
class TestServingRoot:
    def test_project_with_serving_ensemble_is_the_root(
        self, tmp_path: Path, packaged_serving_project: Path
    ) -> None:
        project = tmp_path / "proj"
        marker = project / ".llm-orc" / packaged.SERVING_MARKER
        marker.parent.mkdir(parents=True)
        marker.write_text("name: serving\nagents: []\n")
        cm = ConfigurationManager(project_dir=project, provision=False)
        assert cm.serving_root() == project / ".llm-orc"

    def test_project_without_it_falls_to_packaged(
        self, tmp_path: Path, packaged_serving_project: Path
    ) -> None:
        project = tmp_path / "proj"
        (project / ".llm-orc" / "ensembles").mkdir(parents=True)
        cm = ConfigurationManager(project_dir=project, provision=False)
        assert cm.serving_root() == packaged_serving_project

    def test_no_serving_ensemble_anywhere_names_both_candidates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Review Focus 5."""
        project = tmp_path / "proj"
        (project / ".llm-orc").mkdir(parents=True)
        cm = ConfigurationManager(project_dir=project, provision=False)
        with pytest.raises(FileNotFoundError, match=r"agentic-serving/serving\.yaml.*\.llm-orc.*None"):
            cm.serving_root()
```

New `tests/unit/web/test_serving_root.py`:

```python
"""The chat-completions caller is built on the serving root, not cwd (#196)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from llm_orc.web.api import v1_chat_completions
from llm_orc.web.server import create_app


@pytest.fixture
def empty_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.chdir(empty)
    monkeypatch.setattr(v1_chat_completions, "_SHARED_CALLERS", {})
    return empty


def test_factory_from_empty_cwd_uses_packaged_root_and_state_trace(
    empty_cwd: Path, packaged_serving_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(empty_cwd.parent / "xdg"))

    caller = v1_chat_completions.get_serving_ensemble_caller()

    assert caller._project_dir == packaged_serving_project
    assert caller._trace_root == empty_cwd.parent / "xdg" / "llm-orc" / ".serve-trace"
    assert caller._self_reference_enabled() is True  # packaged config.yaml says so
    assert caller._load_config().name == "serving"


def test_factory_in_a_project_with_the_ensemble_uses_the_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, packaged_serving_project: Path
) -> None:
    import shutil

    project = tmp_path / "proj"
    shutil.copytree(packaged_serving_project, project / ".llm-orc")
    monkeypatch.chdir(project)
    monkeypatch.setattr(v1_chat_completions, "_SHARED_CALLERS", {})

    caller = v1_chat_completions.get_serving_ensemble_caller()

    assert caller._project_dir == project / ".llm-orc"
    assert caller._trace_root == project / ".llm-orc" / ".serve-trace"


def test_chat_completion_with_no_serving_ensemble_is_a_json_error(
    empty_cwd: Path,
) -> None:
    """Review Focus 5: the tier is disabled (suite default) and cwd is empty."""
    with TestClient(create_app(), raise_server_exceptions=False) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "x", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 500
    assert "serving.yaml" in response.json()["detail"]
```

Read `test_serving_ensemble_endpoint.py` lines 168-200 for how
`create_app()` and the caller interact, and check the request body the
endpoint validates (`_ChatCompletionMessage`); adjust the JSON to pass
validation so the failure is the root lookup, not a 422.

New `tests/unit/core/execution/test_child_ensembles_across_tiers.py`:

```python
"""A global ensemble reaches a packaged child, and the child's packaged script runs (#196).

Through the real executor: today `_resolve_ensemble_reference` stops at
the local dot-dir, so research-dossier (global) could not reach
agentic-serving/web-searcher (packaged) on a plain-dir serve.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.core.config.ensemble_config import EnsembleLoader
from llm_orc.core.execution.executor_factory import ExecutorFactory

_PARENT = """\
name: parent
description: global parent, packaged child
agents:
  - name: kid
    ensemble: agentic-serving/child
"""


@pytest.fixture
def parent_in_global(tmp_path: Path, packaged_serving_project: Path) -> tuple[ConfigurationManager, Path]:
    cm = ConfigurationManager(project_dir=tmp_path / "noproj", provision=False)
    ensembles = cm.global_config_dir / "ensembles"
    ensembles.mkdir(parents=True)
    path = ensembles / "parent.yaml"
    path.write_text(_PARENT)
    return cm, path


def test_child_resolves_from_the_packaged_tier(parent_in_global: tuple[ConfigurationManager, Path]) -> None:
    cm, _ = parent_in_global
    executor = ExecutorFactory.create_root_executor(config_manager=cm, save_artifacts=False)

    child = executor._resolve_ensemble_reference("agentic-serving/child")

    assert child.name == "child"


async def test_parent_runs_the_packaged_child_and_its_packaged_script(
    parent_in_global: tuple[ConfigurationManager, Path],
) -> None:
    cm, path = parent_in_global
    executor = ExecutorFactory.create_root_executor(config_manager=cm, save_artifacts=False)
    config = EnsembleLoader().load_from_file(str(path))

    result = await executor.execute(config, "ping")

    assert result["status"] == "success", result
    assert "ping" in str(result["results"]["kid"])
```

Check the result shape (`status`, `results[agent]`) against
`tests/unit/core/execution/` neighbors and adjust the two assertions to
the real keys; the invariant is that the child ran and echoed the input.

- [ ] **Step 2: Run to verify they fail.**

Run: `uv run pytest tests/unit/core/config/test_packaged_serving.py tests/unit/web/test_serving_root.py tests/unit/core/execution/test_child_ensembles_across_tiers.py -q -p no:cacheprovider --no-cov`
Expected: `AttributeError: serving_root`; the factory test asserts a
cwd-derived `_project_dir`; the child test fails to resolve.

- [ ] **Step 3: `serving_root`.** In `ConfigurationManager`:

```python
    def serving_root(self) -> Path:
        """The dot-dir that carries the serving ensemble (#196).

        The project's ``.llm-orc`` when it has
        ``ensembles/agentic-serving/serving.yaml``, else the packaged
        serving project. ``ServingEnsembleCaller`` reads the ensemble,
        the serve-owned scripts and the ``serving:`` config keys from
        this one directory; other ensembles, profiles and scripts still
        merge every tier.
        """
        for candidate in (self._local_config_dir, self._packaged_serving_dir):
            if candidate is not None and has_serving_ensemble(candidate):
                return candidate
        raise FileNotFoundError(
            f"no serving ensemble ({SERVING_MARKER}) under the project "
            f"({self._local_config_dir}) or the packaged serving project "
            f"({self._packaged_serving_dir})"
        )
```

(extend the `packaged` import to `SERVING_MARKER, has_serving_ensemble,
packaged_serving_project_dir`.)

- [ ] **Step 4: The factory.** In `v1_chat_completions.py` delete
  `_resolve_serving_project_dir` and rewrite:

```python
def get_serving_ensemble_caller() -> ServingEnsembleCaller:
    """Return the Cycle-8 declarative Serving Ensemble caller (ADR-046 §1).

    Built on ``ConfigurationManager.serving_root()`` (#196): the project's
    ``.llm-orc`` when it carries the serving ensemble, else the packaged
    serving project. The turn trace goes to the state dir, never under
    the (read-only) packaged root. Shared per root so the caller's
    ensemble-config cache survives across requests (issue #93). Tests
    override this factory to point at a hermetic project dir whose
    ``code_generation`` seat is a deterministic echo (no model).
    """
    config_manager = ConfigurationManager(provision=False)
    root = config_manager.serving_root()
    caller = _SHARED_CALLERS.get(root)
    if caller is None:
        trace_root = resolve_state_dir(config_manager.local_config_dir) / TRACE_DIRNAME
        caller = ServingEnsembleCaller(project_dir=root, trace_root=trace_root)
        _SHARED_CALLERS[root] = caller
    return caller
```

Imports: `ConfigurationManager`, `from llm_orc.core.config.state import TRACE_DIRNAME, resolve_state_dir`.
`Path` may now be unused in this module; ruff will say.

- [ ] **Step 5: Child resolution.** In `_resolve_ensemble_reference`,
  after the local-dir block and before the `_find_ensemble_in_dirs`
  call:

```python
        # Every tier (#196): a global ensemble's child may be packaged,
        # a project's child may be global. Order is the tier order.
        for tier_dir in self._config_manager.get_ensembles_dirs():
            if str(tier_dir) not in search_dirs:
                search_dirs.append(str(tier_dir))
```

Update the docstring: "Searches the project directory, then every
configuration tier (local, library, global, packaged)."

- [ ] **Step 6: Run.**

Run: `uv run pytest -q -p no:cacheprovider tests/unit/core tests/unit/web`
Expected: PASS. The serving endpoint tests keep overriding the factory
and are unaffected.

- [ ] **Step 7: Mutants.**
  1. Factory: `root = Path.cwd() / ".llm-orc"` → the empty-cwd factory
     test red.
  2. Factory: omit `trace_root=` → the `_trace_root` assertion red.
  3. `_resolve_ensemble_reference`: delete the tier loop → child test red.
  4. `serving_root`: swap the candidates' order → the project-first test red.

- [ ] **Step 8: Lint, full suite, commit.**

```bash
make lint && uv run pytest -q -p no:cacheprovider
git add src/llm_orc/core/config/config_manager.py src/llm_orc/web/api/v1_chat_completions.py src/llm_orc/core/execution/ensemble_execution.py tests/unit/core/config/test_packaged_serving.py tests/unit/web/test_serving_root.py tests/unit/core/execution/test_child_ensembles_across_tiers.py
git commit -m "feat: chat completions serve from the serving root; child ensembles resolve across tiers

ConfigurationManager.serving_root() replaces the cwd-based lookup that
failed outright in an empty directory (S1). The caller gets its trace
root from the state dir. The executor's child-ensemble lookup searches
every tier after the project dir, so a global ensemble reaches a
packaged child. Mutants: <four, with red lines>."
```

---

## Task 7: writes never land in the packaged or library tier; tier classification is one function

**Files:**
- Modify: `src/llm_orc/services/handlers/ensemble_crud_handler.py:256-275`
- Modify: `src/llm_orc/services/handlers/profile_handler.py:61-74`
- Modify: `src/llm_orc/core/validation/composition_validator.py:294-326`
- Modify: `src/llm_orc/services/handlers/library_handler.py:33-43,100-112`
- Modify: `src/llm_orc/services/handlers/resource_handler.py:153-168` (`determine_source`)
- Test: `tests/unit/services/handlers/test_write_targets.py` (new),
  `tests/unit/services/handlers/test_resource_handler.py` (append)

**Interfaces:**
- Consumes: `classify_tier` with `"packaged"`, `library_dir` (Task 1).
- Produces:
  - `EnsembleCrudHandler.get_local_ensembles_dir()` and
    `ProfileHandler.get_local_profiles_dir()` return
    `local_config_dir / "ensembles" | "profiles"` or raise `ValueError`
    naming `scope: global`.
  - `ConfigManagerEnsembleWriter._resolve_local_ensembles_dir()` returns
    the project dir, else `global_config_dir / "ensembles"`.
  - `ResourceHandler.determine_source(path)` returns `classify_tier(path)`
    (`"unknown"` maps to `"global"` to keep the old contract for paths
    outside every tier).
  - `LibraryHandler.get_library_dir()` falls back to `config_manager.library_dir`.

- [ ] **Step 1: Write the failing tests.** `test_write_targets.py`:

```python
"""Engine and legacy writes with no project never touch the packaged tier (#196)."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.core.config.ensemble_config import EnsembleConfig, EnsembleLoader
from llm_orc.core.validation.composition_validator import ConfigManagerEnsembleWriter
from llm_orc.services.handlers.ensemble_crud_handler import EnsembleCrudHandler
from llm_orc.services.handlers.profile_handler import ProfileHandler


def _crud(cm: ConfigurationManager) -> EnsembleCrudHandler:
    """The handler with real config and loader; callbacks unused by these paths."""
    return EnsembleCrudHandler(
        config_manager=cm,
        ensemble_loader=EnsembleLoader(),
        find_ensemble_fn=lambda name: None,
        read_artifact_fn=lambda *args, **kwargs: None,
    )


class TestNoProject:
    def test_composition_writer_lands_in_global(
        self, tmp_path: Path, packaged_serving_project: Path
    ) -> None:
        cm = ConfigurationManager(project_dir=tmp_path / "noproj", provision=False)
        config = EnsembleConfig(name="composed", description="", agents=[])

        written = Path(ConfigManagerEnsembleWriter(cm).write(config))

        assert written == cm.global_config_dir / "ensembles" / "composed.yaml"
        assert not list(packaged_serving_project.rglob("composed.yaml"))

    def test_legacy_project_dir_helpers_raise_naming_global(
        self, tmp_path: Path, packaged_serving_project: Path
    ) -> None:
        cm = ConfigurationManager(project_dir=tmp_path / "noproj", provision=False)
        with pytest.raises(ValueError, match="scope 'global'"):
            _crud(cm).get_local_ensembles_dir()
        with pytest.raises(ValueError, match="scope 'global'"):
            ProfileHandler(cm).get_local_profiles_dir()


class TestWithProject:
    def test_helpers_return_the_project_dir_even_before_it_has_subdirs(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / ".llm-orc").mkdir()
        cm = ConfigurationManager(project_dir=tmp_path, provision=False)
        assert _crud(cm).get_local_ensembles_dir() == tmp_path / ".llm-orc" / "ensembles"
        assert ProfileHandler(cm).get_local_profiles_dir() == tmp_path / ".llm-orc" / "profiles"
```

`EnsembleConfig` requires `name` and `description`; `agents` defaults
to an empty list.

Append to `test_resource_handler.py`:

```python
class TestDetermineSourceUsesTiers:
    def test_packaged_dir_reports_packaged(
        self, tmp_path: Path, packaged_serving_project: Path
    ) -> None:
        handler = ResourceHandler(
            ConfigurationManager(project_dir=tmp_path / "noproj", provision=False),
            EnsembleLoader(),
        )
        assert handler.determine_source(packaged_serving_project / "ensembles") == "packaged"

    async def test_listing_labels_packaged_ensembles(
        self, tmp_path: Path, packaged_serving_project: Path
    ) -> None:
        handler = ResourceHandler(
            ConfigurationManager(project_dir=tmp_path / "noproj", provision=False),
            EnsembleLoader(),
        )
        entries = await handler.read_ensembles()
        sources = {e["name"]: e["source"] for e in entries}
        assert sources["serving"] == "packaged"
        assert sources["child"] == "packaged"
```

- [ ] **Step 2: Run to verify they fail.**

Run: `uv run pytest tests/unit/services/handlers/test_write_targets.py tests/unit/services/handlers/test_resource_handler.py -q -p no:cacheprovider --no-cov`
Expected: the composition writer returns a packaged path (`dirs[0]`);
the helpers return the packaged dir instead of raising; `determine_source`
returns `"global"` or `"local"` for the packaged dir (string heuristic).

- [ ] **Step 3: Implement.**

`ensemble_crud_handler.get_local_ensembles_dir`:

```python
    def get_local_ensembles_dir(self) -> Path:
        """The project's ensembles directory for writing.

        Raises:
            ValueError: when there is no project tier. Writes never fall
                through to the library or packaged tiers; ``scope: global``
                is the writable home without a project (#196).
        """
        local = self._config_manager.local_config_dir
        if local is None:
            raise ValueError(
                "No project (.llm-orc) directory to write to; "
                "use scope 'global' for the global tier"
            )
        return local / "ensembles"
```

`profile_handler.get_local_profiles_dir`: the same with `"profiles"`.

`composition_validator.ConfigManagerEnsembleWriter._resolve_local_ensembles_dir`:

```python
    def _resolve_local_ensembles_dir(self) -> Path:
        """The project's ensembles dir, else the global one.

        The orchestrator's composition write has no caller to ask for a
        scope, so it lands in the first writable tier: never the library
        or the packaged serving project (#196).
        """
        local = self._config_manager.local_config_dir
        base = local if local is not None else self._config_manager.global_config_dir
        return base / "ensembles"
```

Update its class docstring (it says it mirrors `get_local_ensembles_dir`;
it no longer does).

`resource_handler.determine_source`:

```python
    def determine_source(self, ensemble_dir: Path) -> str:
        """The tier an ensemble directory belongs to.

        Returns:
            ``'local'``, ``'library'``, ``'global'`` or ``'packaged'``;
            a directory outside every tier reports ``'global'`` as before.
        """
        tier = self._config_manager.classify_tier(Path(ensemble_dir))
        return "global" if tier == "unknown" else tier
```

Tests that build `ResourceHandler` on a `Mock()` config manager and call
`determine_source` now get a Mock back; give those mocks
`classify_tier = lambda p: "local"` (or whatever they asserted) rather
than weakening the assertion. `library_handler.get_library_dir`: replace
the `Path.cwd() / "llm-orchestra-library"` fallback with
`return self._config_manager.library_dir`; in `_resolve_copy_destination`
replace `local_dir = Path.cwd() / ".llm-orc"` with
`local_dir = self._config_manager.local_config_dir or (self._config_manager.global_config_dir)`
and the `is_local`/`is_library` string checks with
`self._config_manager.classify_tier(path) == "local"` (read the rest of
that function first; keep its shape).

- [ ] **Step 4: Run.**

Run: `uv run pytest -q -p no:cacheprovider tests/unit/services tests/unit/mcp_server tests/unit/core/validation tests/bdd`
Expected: PASS after the Mock adjustments described above. Any test
that asserted the old `dirs[0]` fallback is asserting the defect S1
named; replace it with the raise.

- [ ] **Step 5: Mutants.**
  1. Composition writer: `return ensemble_dirs[0]` → composition test
     red (packaged path).
  2. `get_local_ensembles_dir`: return `self._config_manager.get_ensembles_dirs()[0]`
     when local is None → helper test red (no raise).
  3. `determine_source`: restore the substring version → packaged label
     test red.

- [ ] **Step 6: Lint, full suite, commit (refactor then fix).**

```bash
make lint && uv run pytest -q -p no:cacheprovider
git add src/llm_orc/services/handlers/resource_handler.py src/llm_orc/services/handlers/library_handler.py tests/unit/services/handlers/test_resource_handler.py
git commit -m "refactor: tier labels and the library root come from ConfigurationManager

determine_source uses classify_tier; the library handler's fallback is
the manager's library_dir, not cwd."
git add src/llm_orc/services/handlers/ensemble_crud_handler.py src/llm_orc/services/handlers/profile_handler.py src/llm_orc/core/validation/composition_validator.py tests/unit/services/handlers/test_write_targets.py
git commit -m "fix: writes with no project never fall through to the library or packaged tier

Composition writes land in global; the legacy project-dir helpers
raise naming scope global. Mutants: <three, with red lines>."
```

---

## Task 8: docs, changelog, deploy notes

**Files:**
- Modify: `docs/serving.md` ("Where things live" table; new section
  "Layers and state" before "Operator seat configuration"; the preset
  sentence under "Local inference")
- Modify: `CHANGELOG.md` (`[Unreleased]`)
- Modify: `deploy/remote-host/README.md`, `deploy/remote-host/com.llm-orc.serve.plist`
- Modify: `docs/plans/2026-09-28-remote-delegation.md` (nothing yet; the
  live row is Task 9)

- [ ] **Step 1: `docs/serving.md`.** In "Where things live" change the
  two rows:

```markdown
| Serving ensemble + seats | `.llm-orc/ensembles/agentic-serving/` (shipped in the wheel as `llm_orc/serving_project/`) |
| Turn trace (per-turn introspection) | `src/llm_orc/web/serving/turn_trace.py` → `<state dir>/.serve-trace/turns.jsonl` |
```

Add before "## Operator seat configuration":

```markdown
## Layers and state

Configuration is read from four tiers, highest precedence first:

| Tier | Where | Written by |
|------|-------|------------|
| project | `<checkout>/.llm-orc/` (discovered walking up from cwd) | CRUD with `scope: project` (the default), the orchestrator's composition writes |
| library | `LLM_ORC_LIBRARY_PATH`, else `<checkout>/llm-orchestra-library/` | nothing (templates) |
| global | `$XDG_CONFIG_HOME/llm-orc/`, default `~/.config/llm-orc/` | CRUD with `scope: global`, operator `*.local.yaml` overrides |
| packaged | `llm_orc/serving_project/` in the wheel; this repo's `.llm-orc/` in a checkout | nothing (read-only; `brew upgrade` changes it) |

Ensembles, profile listings and scripts merge all four (first match
wins). Runtime model profiles merge packaged → global → project (the
library is listed, never resolved). The serving ensemble, its
serve-owned scripts and the `serving:` config keys are read from the
**serving root**: the project's `.llm-orc/` when it carries
`ensembles/agentic-serving/serving.yaml`, else the packaged project. So a
serve started in an empty directory runs the shipped serving ensemble,
and a checkout of this repo shadows it. `LLM_ORC_SERVING_PROJECT_DIR`
points the packaged tier at another directory (empty disables it).

Runtime state (artifacts, the turn trace, the script cache, the rendered
router preset) is written to one **state dir**: `LLM_ORC_STATE_DIR` or
`llm-orc serve --state-dir`, else the project's `.llm-orc/` when there is
one, else `$XDG_STATE_HOME/llm-orc/` (default `~/.local/state/llm-orc/`).
Nothing is ever written under the packaged tier.
```

Under "Local inference", change "At start the serve renders
`.llm-orc/llama-server.ini`" to "At start the serve renders
`llama-server.ini` into the state dir".

- [ ] **Step 2: `CHANGELOG.md`** under `[Unreleased]`:

```markdown
### Added
- The serving project (`.llm-orc/{ensembles,profiles,scripts,config.yaml}`)
  ships in the wheel as `llm_orc/serving_project/`, a read-only fourth
  configuration tier below project, library and global (#196). A serve
  started in an empty directory runs the shipped serving ensemble; a
  checkout still shadows it. Listings report `source: packaged`.
  `LLM_ORC_SERVING_PROJECT_DIR` overrides the tier (empty disables it).
  `scripts/check_wheel_contents.py` (`make wheel-check`) pins the
  packaged set to the tracked files.
- A runtime state dir for artifacts, the turn trace, the script cache and
  the rendered router preset: `LLM_ORC_STATE_DIR` / `llm-orc serve
  --state-dir`, else the project's `.llm-orc/`, else `$XDG_STATE_HOME/llm-orc`.

### Changed
- Runtime model profiles merge packaged, global and project tiers in that
  order (`.local.yaml` last within a tier); `performance:` and
  `agentic_serving:` config merge the packaged tier below global.
- Child ensemble references (`ensemble:`, `loop:`, `dispatch:`) resolve
  across every tier after the project dir; previously a global ensemble
  could not reach a child that lived elsewhere.
- A serve with no project renders `llama-server.ini` into the state dir
  (was `~/.config/llm-orc/`).
- The library submodule is located relative to the project, not cwd, when
  a `ConfigurationManager` is built with an explicit project dir.
- Engine writes with no project never land in the library or packaged
  tier: composition writes go to global; the legacy project-dir helpers
  raise naming `scope: global`.
```

Merge with the existing Arc 1 entries under the same headings (one
`### Added`, one `### Changed`).

- [ ] **Step 3: Deploy notes.** In the plist replace the
  `WorkingDirectory` string with `/Users/remoteuser/llm-orc-serve`.
  In the README replace the paragraph starting "The plist already names"
  and the "Update (checkout path)" section with:

```markdown
The plist's `WorkingDirectory` is a plain directory with no `.llm-orc`
(`mkdir -p ~/llm-orc-serve`). Since #196 the serving project ships in the
wheel: the serve runs the packaged serving ensemble, user ensembles and
profiles live in `~/.config/llm-orc/` (CRUD with `scope: global`), and
runtime state (artifacts, `.serve-trace/`, `llama-server.ini`) lives in
`~/.local/state/llm-orc/`. The checkout under `~/Development/llm-orc` is
no longer read by the serve; keep or remove it. The library submodule's
ensembles are not part of the packaged project: to keep them listed, add
`LLM_ORC_LIBRARY_PATH` = `/Users/remoteuser/Development/llm-orc/llm-orchestra-library`
to the plist's `EnvironmentVariables`.

## Update

    brew upgrade llm-orchestra
    launchctl kickstart -k gui/$(id -u)/com.llm-orc.serve
    curl https://llm-orc.remote.example/v1/models
```

Keep the "Releases are the unit of change" section's brew lines.
Move the two remotely created ensembles note: "`research-dossier` is
already global; `test-research-pipeline` still lives in the checkout's
`.llm-orc/ensembles/` and must move to `~/.config/llm-orc/ensembles/`
before the cut-over."

- [ ] **Step 4: Lint and commit.**

```bash
make lint
git add docs/serving.md CHANGELOG.md deploy/remote-host/README.md deploy/remote-host/com.llm-orc.serve.plist
git commit -m "docs: layers and state dir; the mini runs from a plain directory"
```

---

## Task 9: live row and review gate

**Files:**
- Modify: `docs/plans/2026-09-28-remote-delegation.md` (append "Arc 2
  live row (2026-09-29)" under the Arc 2 re-cut), GitHub #196 (comment).

- [ ] **Step 1: Build, install clean, serve from an empty directory.**
  From the worktree, with the laptop's `llama-server` on PATH and the
  qwen3 GGUFs already in the Hugging Face cache:

```bash
make wheel-check
W=$(pwd)/dist/wheel-check/*.whl
LIVE=$(mktemp -d)
python3 -m venv $LIVE/venv && $LIVE/venv/bin/pip install -q $W
mkdir -p $LIVE/empty $LIVE/config $LIVE/state
cd $LIVE/empty
touch $LIVE/marker
XDG_CONFIG_HOME=$LIVE/config XDG_STATE_HOME=$LIVE/state \
  $LIVE/venv/bin/llm-orc serve --port 8766 --backend-port 8790 --models-max 1 > $LIVE/serve.log 2>&1 &
sleep 20; tail -3 $LIVE/serve.log
curl -s localhost:8766/v1/models | python3 -c 'import json,sys; print([m["id"] for m in json.load(sys.stdin)["data"]])'
curl -s localhost:8766/api/ensembles | python3 -c 'import json,sys; d=json.load(sys.stdin); print(len(d), sorted({e["source"] for e in d}))'
ls $LIVE/state/llm-orc/
```

Expected: `/v1/models` lists `agentic-tier-cheap-general`; sources are
`['global', 'packaged']` (global from provisioning templates); the state
dir holds `llama-server.ini`.

- [ ] **Step 2: One serving turn and one research-dossier run.**

```bash
curl -s localhost:8766/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "agentic-tier-cheap-general",
  "messages": [{"role": "user", "content": "Create hello.py that prints hello world."}]
}' | head -c 800
ls $LIVE/state/llm-orc/.serve-trace/ && wc -l $LIVE/state/llm-orc/.serve-trace/turns.jsonl
mkdir -p $LIVE/config/llm-orc/ensembles
cp <main checkout>/.llm-orc/ensembles/research-dossier.yaml $LIVE/config/llm-orc/ensembles/
curl -s -X POST localhost:8766/api/ensembles/research-dossier/execute -H 'content-type: application/json' \
  -d '{"input":"the 1783 Laki eruption"}' | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["status"], d.get("has_errors"), str(d.get("deliverable"))[:300])'
ls $LIVE/state/llm-orc/artifacts/
find $LIVE/venv -path '*serving_project*' -newer $LIVE/marker | head
kill %1
```

Expected: the chat turn returns a completion (a tool call or refusal is
fine; the point is the packaged serving ensemble ran and the trace
landed in the state dir); `research-dossier` reports `success` with a
markdown deliverable (its `agentic-serving/web-searcher` child came
from the packaged tier); artifacts under the state dir; the `find`
prints nothing (nothing written under the packaged path); cwd
`$LIVE/empty` is still empty. If the research run takes over ten
minutes on the laptop, record the per-step status from the serve log
and the partial result instead of waiting.

- [ ] **Step 3: Record.** Append the row (commands run, observed output
  per step, wheel filename, serve version, timings) under the Arc 2
  re-cut in the spec doc. Commit:

```bash
git add docs/plans/2026-09-28-remote-delegation.md
git commit -m "docs: Arc 2 live row"
```

- [ ] **Step 4: Review gate.** Independent adversarial review (Opus, not
  the implementer) with an explicit wrong-accept hunt: for each pin in
  Tasks 0-7, can it pass while the defect it names is live? Particular
  targets: the suite-wide `LLM_ORC_SERVING_PROJECT_DIR=""` default (does
  any pin silently run without the tier?), the checkout dedupe
  (`packaged == local`), the state-dir precedence, and any test that was
  updated rather than added. Merge to local `main` only on APPROVE.
  Nothing is pushed.

- [ ] **Step 5: The mini (practitioner does or okays).** Move
  `test-research-pipeline` to `~/.config/llm-orc/ensembles/`, create
  `~/llm-orc-serve`, install the release that carries this arc, update
  the plist, kickstart, check `/v1/models` and `/api/ensembles` sources.
  Not automated here.

---

## Arcs 3-5

Held at card level in the spec until Arc 2 lands. Arc 3 (`needs_restart`
from S2; a `packaged` status is not needed: preflight classifies models,
not tiers). Arc 4 injections layer above the project tier for one run.
