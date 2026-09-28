# Remote delegation, Arc 0 + Arc 1: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the two spikes (Arc 0) and `scope: project | global` on
CRUD (Arc 1) from `docs/plans/2026-09-28-remote-delegation.md`, so user
ensembles, profiles and scripts get a home on a serve that survives
upgrades, and Arcs 2-4 can be re-cut from measured findings.

**Architecture:** Reads keep merging every configuration layer (project,
library, global). Writes now target exactly one scope, chosen by the
caller; `project` is the default and is byte-for-byte today's behavior.
A `global` write lands under `ConfigurationManager.global_config_dir`
(`$XDG_CONFIG_HOME/llm-orc` or `~/.config/llm-orc`). One small helper
module carries the scope vocabulary and the "not in this scope, lives
over there" lookup; the three CRUD handlers, two REST routers and the
MCP tool signatures thread `scope` through to it. The script resolver
gains the global scripts dir as its last search path so a global script
is runnable, not just listable.

**Tech Stack:** Python 3.11+, FastAPI (REST), FastMCP over streamable
HTTP (`/mcp`), pytest (asyncio auto mode), ruff 88, mypy strict,
complexipy <= 15.

**Spec:** `docs/plans/2026-09-28-remote-delegation.md` (decisions,
pattern, arcs, standing rules, gates). Read its "Standing rules for every
implementer" before touching anything.

## Global Constraints

- Own git worktree off `main` per arc; never `git stash`; never spawn
  subagents that edit or commit. Arc 1 branch: `feat/crud-scope`.
- Strict TDD. Structural and behavioral commits separate. ruff 88 +
  mypy strict from the first draft. `make lint` clean (mypy, ruff,
  format, complexipy 15, bandit, vulture, doc drift).
- Commit prefixes `feat:` `fix:` `refactor:` `test:` `docs:`. No AI
  attribution of any kind. No session links, no scratch paths.
- Full suite: `uv run pytest -q -p no:cacheprovider`. Known local-only
  failure: `testget_available_providers_auth_only` when a router listens
  on :8080.
- Doctrine 11: pins assert outcomes through the real surface (REST
  TestClient, MCP tools/call over `/mcp`, real resolver). Each key pin
  is shown RED under a named mutant before it counts. Report mutant and
  the red assertion line in the task's commit body or PR notes.
- Doc drift gate: `scripts/check_doc_drift.py` fails `make lint` when a
  `docs/plans/*.md` file names a `test_*` identifier in single backticks
  that exists nowhere in code. This plan therefore names tests only
  inside fenced code blocks. Keep it that way when editing this file.
- Findings go on #196 and #191 as comments, never new issues.
- Nothing is pushed; nothing paid is run. The mini step at the end of
  Arc 1 is the practitioner's.
- Reads merge layers, writes target one scope. This is the invariant
  every task below instantiates; do not add a "search all scopes and
  write wherever it was found" path anywhere.

## Review Focus

Inputs the spec implies but no arc-level card names. Each gets a pin in
the owning task below.

1. **A `scope` value outside the vocabulary** (`"local"`, `"Global"`,
   `""`). Expected: rejected with a message listing the two valid
   values, nothing written. Pin in Task 1.
2. **Delete or update with the wrong scope for a name that exists
   elsewhere.** Expected: nothing touched, error names the tier and the
   path where it lives. A silent no-op or a cross-scope delete is the
   harm. Pins in Tasks 2, 3, 4 (mutant: search all dirs).
3. **A global script that lists but cannot run.** Expected: the
   execution-path resolver finds it. Pin in Task 4 (mutant: resolver
   without the global search path).
4. **Same name in project and global.** Expected: project shadows
   global on every read (list, get, resolve), and a `scope: global`
   write never overwrites the project file. Pins in Tasks 2 and 4.
5. **Global dir absent on first global write.** Expected: created,
   including `ensembles/` `profiles/` `scripts/<category>/`. Pins in
   Tasks 2, 3, 4 use a fresh XDG temp dir with no such subdirectory.

Pre-existing defects observed while reading, out of scope here and to be
noted on #191 (one comment, no new issue): `update_ensemble` returns
`modified: True` and `changes_applied` without ever writing the changes
to the file; a `ValueError` from any CRUD handler reaches REST callers
as a 500 via the global exception handler; the `project` write-dir
fallback in `get_local_ensembles_dir` / `get_local_profiles_dir` returns
the first search dir when no `.llm-orc` dir exists, which can be the
library submodule. That fallback becomes a design question for Arc 2
(when the project layer becomes optional it should be an error, not a
fallback); record it in the S1 findings.

---

## Arc 0: two spikes

Read-mostly. Sonnet. Each spike ends by appending a dated `## Spike
findings: S<n> (2026-09-28)` section to
`docs/plans/2026-09-28-remote-delegation.md` and committing it as
`docs: S<n> spike findings`. No production code changes. Work in a
worktree off `main` (`git worktree add .claude/worktrees/spikes -b
spikes/remote-delegation main`).

### Task S1: project layering spike

**Files:**
- Read: `src/llm_orc/core/config/config_manager.py` (all of it),
  `src/llm_orc/core/execution/scripting/resolver.py:56-113`,
  `src/llm_orc/core/execution/scripting/primitive_registry.py:31`,
  `src/llm_orc/web/serving/serving_ensemble_caller.py:1059`,
  `src/llm_orc/web/api/v1_chat_completions.py:112`,
  `src/llm_orc/services/handlers/{artifact,library,script,resource}_handler.py`,
  `src/llm_orc/cli_library/library.py:125`,
  `src/llm_orc/core/session/artifact_store.py`,
  `src/llm_orc/providers/llama_server.py` (preset rendering, ~384),
  `docs/plans/2026-08-13-dot-dir-self-reference-design.md` (#144),
  `.llm-orc/config.yaml`.
- Modify: `docs/plans/2026-09-28-remote-delegation.md` (append only).

**Interfaces:**
- Consumes: nothing.
- Produces: the findings section Arc 2's task cards are re-cut from.

- [ ] **Step 1: Map the merge order today.** For each of `ensembles`,
  `profiles` (both `profiles/*.yaml` and `config.yaml: model_profiles`),
  `scripts`, and the `serving:` keys of `config.yaml`, write one line:
  which layers are consulted, in what order, which wins on a name
  collision, and the function that decides it (file:line). Note that
  `get_model_profiles` merges global then local (local wins) while
  `get_ensembles_dirs` orders local, library, global; say whether the
  two orders agree on the winner.

- [ ] **Step 2: Enumerate every `Path.cwd()` / `os.getcwd()` site.**

Run:
```bash
grep -rn "Path.cwd()\|os.getcwd()" src/llm_orc
```
For each hit, one table row: file:line, what it resolves (project root,
library root, scripts dir, artifacts dir, preset path), whether it is
already bypassed when `project_dir` is passed explicitly, and whether a
serve started in an empty directory would read or write through it.

- [ ] **Step 3: Enumerate every runtime write into the project dir.**
  Start from the known set (`.llm-orc/artifacts/`,
  `.llm-orc/agentic-sessions/` via `core/session/artifact_store.py`,
  `.serve-trace/` in `serving_ensemble_caller.py`, `llama-server.ini`
  rendered by `providers/llama_server.py`) and confirm or extend it:

```bash
grep -rn "write_text\|open(.*\"w\|mkdir" src/llm_orc | grep -v "global_config_dir" | grep -i "llm-orc\|trace\|artifact\|session\|\.ini"
```
One row each: what is written, by which code path, whether a reader
depends on it being next to the ensembles (the #144 self-reference
question), and whether it could move to a state dir unchanged.

- [ ] **Step 4: Answer #144's assumption.** Read the self-reference
  design doc and the code it governs. State in one paragraph what
  `serving.self_reference` assumes about "the project" being the repo
  checkout, and what breaks if the project is a read-only packaged layer
  plus a writable state dir.

- [ ] **Step 5: Write the answer.** Under the findings heading, answer
  the spike's question in two sentences: can the serving project be a
  read-only layer, and what must move to a state dir. Then list the
  proposed layer order for Arc 2 (project, library, packaged serving,
  global, or a corrected order with the reason), and list every site
  from Steps 2-3 that Arc 2 must route through a project context. Add
  the Arc 2 design question from Review Focus (project write-dir
  fallback when no project exists).

- [ ] **Step 6: Commit.**

```bash
git add docs/plans/2026-09-28-remote-delegation.md
git commit -m "docs: S1 spike findings on project layering and runtime writes"
```

### Task S2: router runtime models spike

**Files:**
- Read: `src/llm_orc/providers/llama_server.py` (`command()` at ~292,
  preset rendering, `/models` polling, pull endpoint),
  `src/llm_orc/web/api/models.py`.
- Modify: `docs/plans/2026-09-28-remote-delegation.md` (append only).

**Interfaces:**
- Consumes: the laptop's `llama-server` binary and the rendered preset
  at `.llm-orc/llama-server.ini` (check `git status`; it is untracked
  and rendered by the serve).
- Produces: the finding that decides Arc 3's `pullable` semantics for a
  newly shipped profile: live load, supervised router restart, or a
  `needs_restart` status.

Precondition: tests may not launch the real binary (autouse guard). This
spike runs the binary by hand, outside pytest. Stop any serve that owns a
router first, or use a different port; check with
`lsof -i :8080` (the known local-only test failure names this port).

- [ ] **Step 1: Record the binary.**

```bash
which llama-server; llama-server --version 2>&1 | head -3
llama-server --help 2>&1 | grep -n "models" 
```
Write down version, and every `--models*` flag with its help text.

- [ ] **Step 2: Start the router the way the serve does.** Build the
  argv from `LlamaServerRouter.command()` (binary, `--models-preset
  <ini>`, `--host`, `--port`, `--models-max`, `--no-webui`) with a
  port nothing else uses, and copy the rendered ini to the scratch dir
  so edits do not touch the checkout. Confirm `GET /models` lists the
  preset's sections.

- [ ] **Step 3: Probe a model outside the preset.** With the router
  running, try each and record the HTTP status and body:
  1. `POST /models/load` with `{"model": "<name not in preset>"}`.
  2. A chat completion naming a model not in the preset.
  3. Append a new section to the copied ini on disk; re-`GET /models`
     (no restart). Then send `SIGHUP` to the router; re-`GET /models`.
  4. If a `--models-dir` (or similar directory-scan) flag exists, start
     a second router with it pointing at the model cache and see whether
     a file dropped into the dir after start becomes listable.

- [ ] **Step 4: Probe the restart path.** Time a router stop and start
  with the same preset plus one added section, from SIGTERM to the
  first `GET /models` that answers. Record seconds and whether an
  in-flight completion on the old router was cut.

- [ ] **Step 5: Write the answer.** Under the findings heading: a table
  (probe, status, body excerpt, conclusion) and one sentence choosing
  between the three Arc 3 options, with the measured cost of the chosen
  one. Kill the router.

- [ ] **Step 6: Commit.**

```bash
git add docs/plans/2026-09-28-remote-delegation.md
git commit -m "docs: S2 spike findings on llama-server router runtime model loading"
```

---

## Arc 1: `scope` on CRUD

Sonnet implements, Opus reviews before merge. Worktree:
`git worktree add .claude/worktrees/arc1-scope -b feat/crud-scope main`.

### Task 1: scope vocabulary and cross-scope lookup

**Files:**
- Create: `src/llm_orc/services/handlers/scope.py`
- Test: `tests/unit/services/handlers/test_scope.py`

**Interfaces:**
- Consumes: `ConfigurationManager.global_config_dir` (property),
  `ConfigurationManager.classify_tier(path) -> "local" | "library" | "global" | "unknown"`.
- Produces:
  - `Scope = Literal["project", "global"]`
  - `parse_scope(arguments: dict[str, Any]) -> Scope` (absent means
    `"project"`; anything else raises `ValueError`)
  - `find_in_scope(*, name: str, filename: str, scope: Scope, scope_dir: Path | None, search_dirs: list[Path], classify: Callable[[Path], str], label: str) -> Path`

- [ ] **Step 1: Write the failing tests.**

```python
"""scope.py: the vocabulary and the one lookup every CRUD handler shares."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_orc.services.handlers.scope import find_in_scope, parse_scope


class TestParseScope:
    def test_absent_means_project(self) -> None:
        assert parse_scope({}) == "project"

    def test_global_is_accepted(self) -> None:
        assert parse_scope({"scope": "global"}) == "global"

    @pytest.mark.parametrize("bad", ["local", "Global", "", None])
    def test_unknown_value_is_rejected_naming_the_vocabulary(self, bad: object) -> None:
        with pytest.raises(ValueError, match=r"project.*global"):
            parse_scope({"scope": bad})


class TestFindInScope:
    def test_returns_file_in_requested_scope(self, tmp_path: Path) -> None:
        scope_dir = tmp_path / "global" / "ensembles"
        scope_dir.mkdir(parents=True)
        (scope_dir / "x.yaml").write_text("name: x\n")

        found = find_in_scope(
            name="x",
            filename="x.yaml",
            scope="global",
            scope_dir=scope_dir,
            search_dirs=[scope_dir],
            classify=lambda _p: "global",
            label="Ensemble",
        )

        assert found == scope_dir / "x.yaml"

    def test_name_in_other_tier_raises_naming_tier_and_path(
        self, tmp_path: Path
    ) -> None:
        project_dir = tmp_path / ".llm-orc" / "ensembles"
        project_dir.mkdir(parents=True)
        (project_dir / "x.yaml").write_text("name: x\n")
        global_dir = tmp_path / "global" / "ensembles"

        with pytest.raises(ValueError, match=r"not in scope 'global'.*local tier.*x\.yaml"):
            find_in_scope(
                name="x",
                filename="x.yaml",
                scope="global",
                scope_dir=global_dir,
                search_dirs=[project_dir, global_dir],
                classify=lambda _p: "local",
                label="Ensemble",
            )

    def test_missing_everywhere_raises_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Ensemble not found: x"):
            find_in_scope(
                name="x",
                filename="x.yaml",
                scope="project",
                scope_dir=tmp_path,
                search_dirs=[tmp_path],
                classify=lambda _p: "local",
                label="Ensemble",
            )

    def test_no_scope_dir_falls_through_to_search(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Ensemble not found: x"):
            find_in_scope(
                name="x",
                filename="x.yaml",
                scope="project",
                scope_dir=None,
                search_dirs=[],
                classify=lambda _p: "local",
                label="Ensemble",
            )
```

Note the `None` entry in the parametrize: with `{"scope": None}` the
call must be rejected, because a REST body can carry `"scope": null`
and silently defaulting it would hide a client bug. Absent key defaults;
explicit null does not.

- [ ] **Step 2: Run to verify they fail.**

Run: `uv run pytest tests/unit/services/handlers/test_scope.py -q -p no:cacheprovider --no-cov`
Expected: FAIL, `ModuleNotFoundError: llm_orc.services.handlers.scope`.

- [ ] **Step 3: Write the module.**

```python
"""Write scope for CRUD handlers.

Reads merge every configuration layer (project, library, global). Writes
target exactly one scope. ``project`` is the default and is today's
behavior; ``global`` writes under ``ConfigurationManager.global_config_dir``
so a serve running from a plain directory has a writable home that
survives upgrades (#196).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, cast

Scope = Literal["project", "global"]
SCOPES: tuple[str, ...] = ("project", "global")


def parse_scope(arguments: dict[str, Any]) -> Scope:
    """Read ``scope`` from tool arguments.

    An absent key means ``project``. An explicit value, including
    ``None``, must be one of the two scopes.
    """
    if "scope" not in arguments:
        return "project"
    raw = arguments["scope"]
    if raw not in SCOPES:
        raise ValueError(f"scope must be one of {list(SCOPES)}, got {raw!r}")
    return cast(Scope, raw)


def find_in_scope(
    *,
    name: str,
    filename: str,
    scope: Scope,
    scope_dir: Path | None,
    search_dirs: list[Path],
    classify: Callable[[Path], str],
    label: str,
) -> Path:
    """Locate ``filename`` in ``scope_dir`` only.

    A name that exists in another layer raises with that layer named, so
    a caller who asked for the wrong scope learns where the file lives
    instead of silently touching nothing, or the wrong file.
    """
    if scope_dir is not None:
        target = scope_dir / filename
        if target.exists():
            return target
    for directory in search_dirs:
        other = Path(directory) / filename
        if other.exists():
            raise ValueError(
                f"{label} '{name}' is not in scope '{scope}'; "
                f"it lives in the {classify(other)} tier at {other}"
            )
    raise ValueError(f"{label} not found: {name}")
```

- [ ] **Step 4: Run to verify they pass.**

Run: `uv run pytest tests/unit/services/handlers/test_scope.py -q -p no:cacheprovider --no-cov`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit.**

```bash
git add src/llm_orc/services/handlers/scope.py tests/unit/services/handlers/test_scope.py
git commit -m "feat: scope vocabulary and cross-scope lookup for CRUD handlers"
```

### Task 2: ensembles handler honors scope

**Files:**
- Modify: `src/llm_orc/services/handlers/ensemble_crud_handler.py`
  (`create_ensemble` 86-139, `delete_ensemble` 141-176,
  `update_ensemble` 178-224, `get_local_ensembles_dir` 265-284)
- Test: `tests/unit/services/handlers/test_ensemble_crud_handler.py`
  (append a class)

**Interfaces:**
- Consumes: Task 1's `parse_scope`, `find_in_scope`, `Scope`.
- Produces: `create_ensemble` result gains `"scope": scope`;
  `delete_ensemble` / `update_ensemble` accept `scope` in `arguments`
  and act only on that scope. Existing result keys unchanged.

- [ ] **Step 1: Write the failing tests.** Append to the test file. These
  use a real `ConfigurationManager` so the global dir is the XDG temp dir
  the autouse fixture in `tests/conftest.py` sets, and `classify_tier` is
  the real one.

```python
from llm_orc.core.config.config_manager import (
    ConfigurationManager,
    resolve_global_config_dir,
)


def _real_handler(project: Path) -> EnsembleCrudHandler:
    config_manager = ConfigurationManager(project_dir=project, provision=False)

    async def _read_artifact(ensemble_name: str, aid: str) -> dict[str, Any]:
        return {}

    return EnsembleCrudHandler(
        config_manager=config_manager,
        ensemble_loader=MagicMock(),
        find_ensemble_fn=lambda name: None,
        read_artifact_fn=_read_artifact,
    )


_AGENTS = [{"name": "writer", "model_profile": "local-qwen3-8b"}]


class TestEnsembleScope:
    """Writes target one scope; a wrong scope names where the file lives."""

    async def test_global_scope_creates_under_global_dir_even_when_absent(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / ".llm-orc" / "ensembles").mkdir(parents=True)
        handler = _real_handler(tmp_path)
        global_ensembles = resolve_global_config_dir() / "ensembles"
        assert not global_ensembles.exists()

        result = await handler.create_ensemble(
            {"name": "remote-made", "agents": _AGENTS, "scope": "global"}
        )

        assert result["scope"] == "global"
        assert Path(result["path"]) == global_ensembles / "remote-made.yaml"
        assert (global_ensembles / "remote-made.yaml").exists()
        assert not (tmp_path / ".llm-orc" / "ensembles" / "remote-made.yaml").exists()

    async def test_default_scope_still_writes_project(self, tmp_path: Path) -> None:
        (tmp_path / ".llm-orc" / "ensembles").mkdir(parents=True)
        handler = _real_handler(tmp_path)

        result = await handler.create_ensemble({"name": "mine", "agents": _AGENTS})

        assert result["scope"] == "project"
        assert Path(result["path"]) == tmp_path / ".llm-orc" / "ensembles" / "mine.yaml"
        assert not (resolve_global_config_dir() / "ensembles" / "mine.yaml").exists()

    async def test_global_create_does_not_overwrite_project_twin(
        self, tmp_path: Path
    ) -> None:
        project_file = tmp_path / ".llm-orc" / "ensembles" / "twin.yaml"
        project_file.parent.mkdir(parents=True)
        project_file.write_text("name: twin\ndescription: project copy\nagents: []\n")
        handler = _real_handler(tmp_path)

        await handler.create_ensemble(
            {"name": "twin", "description": "global copy", "agents": _AGENTS, "scope": "global"}
        )

        assert yaml.safe_load(project_file.read_text())["description"] == "project copy"

    async def test_delete_with_wrong_scope_touches_nothing_and_names_tier(
        self, tmp_path: Path
    ) -> None:
        project_file = tmp_path / ".llm-orc" / "ensembles" / "keep.yaml"
        project_file.parent.mkdir(parents=True)
        project_file.write_text("name: keep\nagents: []\n")
        handler = _real_handler(tmp_path)

        with pytest.raises(ValueError, match=r"not in scope 'global'.*local tier"):
            await handler.delete_ensemble(
                {"ensemble_name": "keep", "confirm": True, "scope": "global"}
            )

        assert project_file.exists()

    async def test_delete_global_scope_removes_global_file(self, tmp_path: Path) -> None:
        (tmp_path / ".llm-orc" / "ensembles").mkdir(parents=True)
        global_file = resolve_global_config_dir() / "ensembles" / "gone.yaml"
        global_file.parent.mkdir(parents=True)
        global_file.write_text("name: gone\nagents: []\n")
        handler = _real_handler(tmp_path)

        result = await handler.delete_ensemble(
            {"ensemble_name": "gone", "confirm": True, "scope": "global"}
        )

        assert result["deleted"] is True
        assert not global_file.exists()

    async def test_update_with_wrong_scope_names_tier(self, tmp_path: Path) -> None:
        project_file = tmp_path / ".llm-orc" / "ensembles" / "upd.yaml"
        project_file.parent.mkdir(parents=True)
        project_file.write_text("name: upd\nagents: []\n")
        handler = _real_handler(tmp_path)

        with pytest.raises(ValueError, match=r"not in scope 'global'"):
            await handler.update_ensemble(
                {"ensemble_name": "upd", "changes": {}, "dry_run": False, "scope": "global"}
            )

    async def test_bad_scope_writes_nothing(self, tmp_path: Path) -> None:
        (tmp_path / ".llm-orc" / "ensembles").mkdir(parents=True)
        handler = _real_handler(tmp_path)

        with pytest.raises(ValueError, match="scope must be one of"):
            await handler.create_ensemble(
                {"name": "nope", "agents": _AGENTS, "scope": "local"}
            )

        assert not list((tmp_path / ".llm-orc" / "ensembles").iterdir())
```

- [ ] **Step 2: Run to verify they fail.**

Run: `uv run pytest tests/unit/services/handlers/test_ensemble_crud_handler.py -q -p no:cacheprovider --no-cov -k Scope`
Expected: FAIL. The global-scope create lands in the project dir
(`KeyError: 'scope'` on the result, or the path assertion); the
wrong-scope delete deletes the project file instead of raising.

- [ ] **Step 3: Implement.** In `ensemble_crud_handler.py`:

Add the import:
```python
from llm_orc.services.handlers.scope import Scope, find_in_scope, parse_scope
```

Add two private methods next to `get_local_ensembles_dir`:
```python
    def _dir_for_scope(self, scope: Scope) -> Path | None:
        """Write directory for one scope; None when the project has none."""
        if scope == "global":
            return self._config_manager.global_config_dir / "ensembles"
        try:
            return self.get_local_ensembles_dir()
        except ValueError:
            return None

    def _find_in_scope(self, ensemble_name: str, scope: Scope) -> Path:
        return find_in_scope(
            name=ensemble_name,
            filename=f"{ensemble_name}.yaml",
            scope=scope,
            scope_dir=self._dir_for_scope(scope),
            search_dirs=[Path(d) for d in self._config_manager.get_ensembles_dirs()],
            classify=self._config_manager.classify_tier,
            label="Ensemble",
        )
```

In `create_ensemble`, replace the `local_dir = self.get_local_ensembles_dir()` line with:
```python
        scope = parse_scope(arguments)
        local_dir = self._dir_for_scope(scope)
        if local_dir is None:
            raise ValueError("No ensemble directory available")
```
and add `"scope": scope,` to the returned dict. Parse `scope` BEFORE
the `if not name` check is fine either way; parse it right after the
name check so a bad scope and a missing name both fail fast without
writing.

In `delete_ensemble`, replace the loop over `ensemble_dirs` (lines
159-169) with:
```python
        scope = parse_scope(arguments)
        ensemble_file = self._find_in_scope(ensemble_name, scope)
```

In `update_ensemble`, replace the loop over `ensemble_dirs` (lines
195-205) with:
```python
        scope = parse_scope(arguments)
        ensemble_path = self._find_in_scope(ensemble_name, scope)
```

- [ ] **Step 4: Run the handler file and the whole suite.**

Run: `uv run pytest tests/unit/services/handlers/test_ensemble_crud_handler.py -q -p no:cacheprovider --no-cov`
Expected: PASS, including the pre-existing tests. If a pre-existing
test fails because it used a MagicMock config manager with no
`global_config_dir`, that test never exercised global scope, so give
the mock a `global_config_dir = tmp_path / "global"`; do not weaken
the assertion.

Run: `uv run pytest -q -p no:cacheprovider`
Expected: PASS (except the known :8080 local-only failure).

- [ ] **Step 5: Mutant.** Temporarily change `_find_in_scope` to loop
  over all `get_ensembles_dirs()` and return the first hit (today's
  behavior). Run the `-k Scope` selection. Expected RED on the
  wrong-scope delete test at `assert project_file.exists()` (the file
  is gone) and on the update test (no raise). Revert. Record the mutant
  and red lines in the commit body.

- [ ] **Step 6: Commit.**

```bash
git add src/llm_orc/services/handlers/ensemble_crud_handler.py tests/unit/services/handlers/test_ensemble_crud_handler.py
git commit -m "feat: ensemble create/update/delete accept scope project|global"
```

### Task 3: profiles handler honors scope

**Files:**
- Modify: `src/llm_orc/services/handlers/profile_handler.py`
  (`get_local_profiles_dir` 60-73, `create_profile` 75-111,
  `update_profile` 113-130, `delete_profile` 132-145,
  `_find_profile_file` 147-157)
- Test: `tests/unit/services/handlers/test_profile_handler.py` (append)

**Interfaces:**
- Consumes: Task 1.
- Produces: `create_profile` result gains `"scope"`; `update_profile`
  and `delete_profile` accept `scope`. `_find_profile_file(name, scope)`
  now takes the scope (internal).

- [ ] **Step 1: Write the failing tests.** Append:

```python
from pathlib import Path

from llm_orc.core.config.config_manager import (
    ConfigurationManager,
    resolve_global_config_dir,
)


def _real_profile_handler(project: Path) -> ProfileHandler:
    return ProfileHandler(ConfigurationManager(project_dir=project, provision=False))


class TestProfileScope:
    async def test_global_scope_creates_under_global_dir(self, tmp_path: Path) -> None:
        (tmp_path / ".llm-orc" / "profiles").mkdir(parents=True)
        handler = _real_profile_handler(tmp_path)
        global_profiles = resolve_global_config_dir() / "profiles"
        assert not global_profiles.exists()

        result = await handler.create_profile(
            {"name": "remote-prof", "provider": "llama-server", "model": "qwen3-8b", "scope": "global"}
        )

        assert result["scope"] == "global"
        assert Path(result["path"]) == global_profiles / "remote-prof.yaml"
        assert not (tmp_path / ".llm-orc" / "profiles" / "remote-prof.yaml").exists()

    async def test_default_scope_writes_project(self, tmp_path: Path) -> None:
        (tmp_path / ".llm-orc" / "profiles").mkdir(parents=True)
        handler = _real_profile_handler(tmp_path)

        result = await handler.create_profile(
            {"name": "mine", "provider": "llama-server", "model": "qwen3-8b"}
        )

        assert result["scope"] == "project"
        assert Path(result["path"]) == tmp_path / ".llm-orc" / "profiles" / "mine.yaml"

    async def test_delete_with_wrong_scope_touches_nothing(self, tmp_path: Path) -> None:
        project_file = tmp_path / ".llm-orc" / "profiles" / "keep.yaml"
        project_file.parent.mkdir(parents=True)
        project_file.write_text("name: keep\nprovider: llama-server\nmodel: m\n")
        handler = _real_profile_handler(tmp_path)

        with pytest.raises(ValueError, match=r"not in scope 'global'.*local tier"):
            await handler.delete_profile({"name": "keep", "confirm": True, "scope": "global"})

        assert project_file.exists()

    async def test_update_global_scope_edits_global_file_only(self, tmp_path: Path) -> None:
        project_file = tmp_path / ".llm-orc" / "profiles" / "twin.yaml"
        project_file.parent.mkdir(parents=True)
        project_file.write_text("name: twin\nprovider: llama-server\nmodel: project\n")
        global_file = resolve_global_config_dir() / "profiles" / "twin.yaml"
        global_file.parent.mkdir(parents=True)
        global_file.write_text("name: twin\nprovider: llama-server\nmodel: global\n")
        handler = _real_profile_handler(tmp_path)

        await handler.update_profile(
            {"name": "twin", "changes": {"model": "edited"}, "scope": "global"}
        )

        assert yaml.safe_load(global_file.read_text())["model"] == "edited"
        assert yaml.safe_load(project_file.read_text())["model"] == "project"
```

- [ ] **Step 2: Run to verify they fail.**

Run: `uv run pytest tests/unit/services/handlers/test_profile_handler.py -q -p no:cacheprovider --no-cov -k Scope`
Expected: FAIL (`KeyError: 'scope'`; wrong-scope delete removes the project file; the twin update edits the project file).

- [ ] **Step 3: Implement.**

Import:
```python
from llm_orc.services.handlers.scope import Scope, find_in_scope, parse_scope
```

Add next to `get_local_profiles_dir`:
```python
    def _dir_for_scope(self, scope: Scope) -> Path | None:
        """Write directory for one scope; None when the project has none."""
        if scope == "global":
            return self._config_manager.global_config_dir / "profiles"
        try:
            return self.get_local_profiles_dir()
        except ValueError:
            return None
```

Replace `_find_profile_file`:
```python
    def _find_profile_file(self, name: str, scope: Scope) -> Path:
        """Find the profile YAML in one scope; name the tier if it lives elsewhere."""
        return find_in_scope(
            name=name,
            filename=f"{name}.yaml",
            scope=scope,
            scope_dir=self._dir_for_scope(scope),
            search_dirs=[Path(d) for d in self._config_manager.get_profiles_dirs()],
            classify=self._config_manager.classify_tier,
            label="Profile",
        )
```

In `create_profile`, after the three required-field checks:
```python
        scope = parse_scope(arguments)
        local_dir = self._dir_for_scope(scope)
        if local_dir is None:
            raise ValueError("No profiles directory configured")
```
(replacing `local_dir = self.get_local_profiles_dir()`), and return
`{"created": True, "path": str(target_file), "scope": scope}`.

In `update_profile` and `delete_profile`, replace
`self._find_profile_file(name)` with
`self._find_profile_file(name, parse_scope(arguments))`.

- [ ] **Step 4: Run the file, then the suite.** Same commands as Task 2
  Step 4 with this file. Pre-existing tests that mock
  `get_profiles_dirs` and call update/delete need
  `mock_config.global_config_dir = tmp_path / "global"` only if they use
  `scope: global`; the default path calls `get_local_profiles_dir` as
  before. `classify_tier` on a `MagicMock` returns a mock; a test that
  asserts on the cross-scope message must use the real manager as above.

- [ ] **Step 5: Mutant.** Make `_find_profile_file` ignore `scope` and
  search all dirs. Expected RED: the wrong-scope delete at
  `assert project_file.exists()`; the twin update at
  `assert ... == "project"`. Revert; record.

- [ ] **Step 6: Commit.**

```bash
git add src/llm_orc/services/handlers/profile_handler.py tests/unit/services/handlers/test_profile_handler.py
git commit -m "feat: profile create/update/delete accept scope project|global"
```

### Task 4: scripts handler honors scope and global scripts resolve

**Files:**
- Modify: `src/llm_orc/services/handlers/script_handler.py` (whole
  class), `src/llm_orc/services/orchestra_service.py:76`
  (`ScriptHandler()` construction),
  `src/llm_orc/core/execution/scripting/resolver.py:68-113`
  (`_get_search_paths`)
- Test: `tests/unit/services/handlers/test_script_handler.py` (create),
  `tests/unit/core/execution/scripting/test_resolver_global_scope.py`
  (create; if a `test_resolver*.py` already exists under that directory,
  append there instead)

**Interfaces:**
- Consumes: Task 1; `resolve_global_config_dir()` from
  `llm_orc.core.config.config_manager`.
- Produces: `ScriptHandler(project_path=None, config_manager=None)`;
  `create_script` / `delete_script` accept `scope`; `list_scripts`
  entries gain `"scope": "project" | "global"` and include global
  scripts; `get_script` / `test_script` search project then global;
  `ScriptResolver._get_search_paths()` ends with
  `<global_config_dir>/scripts` when it exists.

- [ ] **Step 1: Write the failing handler tests.** New file:

```python
"""ScriptHandler scope: writes target one dir, reads merge project then global."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_orc.core.config.config_manager import (
    ConfigurationManager,
    resolve_global_config_dir,
)
from llm_orc.services.handlers.script_handler import ScriptHandler


def _handler(project: Path) -> ScriptHandler:
    return ScriptHandler(
        project_path=project,
        config_manager=ConfigurationManager(project_dir=project, provision=False),
    )


class TestScriptScope:
    async def test_global_scope_creates_under_global_scripts_dir(
        self, tmp_path: Path
    ) -> None:
        handler = _handler(tmp_path)
        expected = resolve_global_config_dir() / "scripts" / "util" / "shout.py"
        assert not expected.parent.exists()

        result = await handler.create_script(
            {"name": "shout", "category": "util", "scope": "global"}
        )

        assert result["scope"] == "global"
        assert Path(result["path"]) == expected
        assert not (tmp_path / ".llm-orc" / "scripts" / "util" / "shout.py").exists()

    async def test_default_scope_writes_project(self, tmp_path: Path) -> None:
        handler = _handler(tmp_path)

        result = await handler.create_script({"name": "mine", "category": "util"})

        assert result["scope"] == "project"
        assert Path(result["path"]) == tmp_path / ".llm-orc" / "scripts" / "util" / "mine.py"

    async def test_list_reports_scope_and_includes_global(self, tmp_path: Path) -> None:
        handler = _handler(tmp_path)
        await handler.create_script({"name": "p", "category": "util"})
        await handler.create_script({"name": "g", "category": "util", "scope": "global"})

        listed = await handler.list_scripts({})

        by_name = {s["name"]: s["scope"] for s in listed["scripts"]}
        assert by_name == {"p": "project", "g": "global"}

    async def test_get_and_test_find_global_script(self, tmp_path: Path) -> None:
        handler = _handler(tmp_path)
        await handler.create_script({"name": "g", "category": "util", "scope": "global"})

        got = await handler.get_script({"name": "g", "category": "util"})
        ran = await handler.test_script({"name": "g", "category": "util", "input": "ping"})

        assert Path(got["path"]) == resolve_global_config_dir() / "scripts" / "util" / "g.py"
        assert ran["success"] is True
        assert ran["stdout"].strip() == "ping"

    async def test_project_shadows_global_on_read(self, tmp_path: Path) -> None:
        handler = _handler(tmp_path)
        await handler.create_script({"name": "twin", "category": "util", "scope": "global"})
        await handler.create_script({"name": "twin", "category": "util"})

        got = await handler.get_script({"name": "twin", "category": "util"})

        assert Path(got["path"]) == tmp_path / ".llm-orc" / "scripts" / "util" / "twin.py"

    async def test_delete_with_wrong_scope_touches_nothing(self, tmp_path: Path) -> None:
        handler = _handler(tmp_path)
        created = await handler.create_script({"name": "keep", "category": "util"})

        with pytest.raises(ValueError, match=r"not in scope 'global'.*project tier"):
            await handler.delete_script(
                {"name": "keep", "category": "util", "confirm": True, "scope": "global"}
            )

        assert Path(created["path"]).exists()

    async def test_global_scope_without_config_manager_is_an_error(
        self, tmp_path: Path
    ) -> None:
        handler = ScriptHandler(project_path=tmp_path)

        with pytest.raises(ValueError, match="global"):
            await handler.create_script({"name": "x", "category": "util", "scope": "global"})
```

- [ ] **Step 2: Write the failing resolver test.** New file:

```python
"""A script created with scope: global must be runnable, not only listable."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_orc.core.config.config_manager import resolve_global_config_dir
from llm_orc.core.execution.scripting.resolver import ScriptResolver


def _write_global(rel: str, body: str) -> Path:
    path = resolve_global_config_dir() / "scripts" / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return path


class TestResolverGlobalScope:
    def test_resolves_global_script_when_project_lacks_it(self, tmp_path: Path) -> None:
        target = _write_global("util/shout.py", "print('hi')\n")

        resolved = ScriptResolver(project_dir=tmp_path).resolve_script_path("util/shout.py")

        assert Path(resolved) == target

    def test_project_script_shadows_global(self, tmp_path: Path) -> None:
        _write_global("util/twin.py", "print('global')\n")
        project_file = tmp_path / ".llm-orc" / "scripts" / "util" / "twin.py"
        project_file.parent.mkdir(parents=True)
        project_file.write_text("print('project')\n")

        resolved = ScriptResolver(project_dir=tmp_path).resolve_script_path("util/twin.py")

        assert Path(resolved) == project_file

    def test_global_dir_is_last_search_path(self, tmp_path: Path) -> None:
        _write_global("util/any.py", "")

        paths = ScriptResolver(project_dir=tmp_path)._get_search_paths()

        assert Path(paths[-1]) == resolve_global_config_dir() / "scripts"
```

- [ ] **Step 3: Run both to verify they fail.**

Run: `uv run pytest tests/unit/services/handlers/test_script_handler.py tests/unit/core/execution/scripting/test_resolver_global_scope.py -q -p no:cacheprovider --no-cov`
Expected: FAIL (`TypeError: unexpected keyword argument 'config_manager'`; resolver raises its not-found error).

- [ ] **Step 4: Implement the handler.** Rewrite the top of
  `ScriptHandler` and the read/write paths:

```python
from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.services.handlers.scope import Scope, find_in_scope, parse_scope


class ScriptHandler:
    """Manages primitive script operations."""

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

    def _get_scripts_dir(self) -> Path:
        """Project scripts directory path."""
        if self._project_path is not None:
            return self._project_path / ".llm-orc" / "scripts"
        return Path.cwd() / ".llm-orc" / "scripts"

    def _global_scripts_dir(self) -> Path:
        if self._config_manager is None:
            raise ValueError("scope 'global' needs a configuration manager")
        return self._config_manager.global_config_dir / "scripts"

    def _dir_for_scope(self, scope: Scope) -> Path:
        if scope == "global":
            return self._global_scripts_dir()
        return self._get_scripts_dir()

    def _scope_dirs(self) -> list[tuple[str, Path]]:
        """Read order: project shadows global."""
        dirs: list[tuple[str, Path]] = [("project", self._get_scripts_dir())]
        if self._config_manager is not None:
            dirs.append(("global", self._global_scripts_dir()))
        return dirs

    def _tier_of(self, path: Path) -> str:
        for scope, directory in self._scope_dirs():
            if path.is_relative_to(directory):
                return scope
        return "unknown"

    def _find_script(self, category: str, name: str) -> Path:
        for _scope, scripts_dir in self._scope_dirs():
            candidate = scripts_dir / category / f"{name}.py"
            if candidate.exists():
                return candidate
        raise ValueError(f"Script '{category}/{name}' not found")
```

`list_scripts`: iterate `self._scope_dirs()`; for each existing dir
call `_collect_root_scripts(scripts_dir, scope)` and
`_collect_category_scripts(scripts_dir, category, scope)`, both of
which now take `scope: str` and put `"scope": scope` in each entry.
Return `{"scripts": []}` only when no dir exists.

`get_script` and `test_script`: replace the
`scripts_dir = ...; script_file = scripts_dir / category / f"{name}.py"; if not script_file.exists(): raise`
block with `script_file = self._find_script(category, name)`.

`create_script`: after the required-field checks,
`scope = parse_scope(arguments)`; `scripts_dir = self._dir_for_scope(scope)`;
add `"scope": scope` to the result.

`delete_script`: after the checks,
```python
        scope = parse_scope(arguments)
        script_file = find_in_scope(
            name=f"{category}/{name}",
            filename=f"{category}/{name}.py",
            scope=scope,
            scope_dir=self._dir_for_scope(scope),
            search_dirs=[d for _s, d in self._scope_dirs()],
            classify=self._tier_of,
            label="Script",
        )
```

In `orchestra_service.py:76`:
`self._script_handler = ScriptHandler(config_manager=self.config_manager)`.

If complexipy reports `list_scripts` above 15, extract the per-dir
collection into `_collect_dir(scripts_dir, category, scope)`.

- [ ] **Step 5: Implement the resolver search path.** In
  `resolver.py`, import `resolve_global_config_dir` from
  `llm_orc.core.config.config_manager` and append, after the library
  block in `_get_search_paths`:

```python
        # Priority 3: global config scripts (written by CRUD with scope: global)
        global_scripts = resolve_global_config_dir() / self.SCRIPTS_DIR
        if global_scripts.exists():
            search_paths.append(str(global_scripts))
```

- [ ] **Step 6: Run both files, then the suite.** Existing script tests
  in `tests/unit/mcp_server/test_server.py` (list/get) assert on entries
  without `scope`; if they compare whole dicts, add the key to the
  expected dict (the field is additive). Run the full suite.

- [ ] **Step 7: Mutants.** (a) Remove the resolver's Priority 3 block:
  expected RED on the resolver's first test at the path assertion.
  (b) Make `delete_script` build the path from `_find_script` instead
  of `find_in_scope`: expected RED on the wrong-scope delete at
  `assert Path(created["path"]).exists()`. Revert both; record.

- [ ] **Step 8: Commit.**

```bash
git add src/llm_orc/services/handlers/script_handler.py src/llm_orc/services/orchestra_service.py src/llm_orc/core/execution/scripting/resolver.py tests/unit/services/handlers/test_script_handler.py tests/unit/core/execution/scripting/test_resolver_global_scope.py
git commit -m "feat: script create/delete accept scope; global scripts list and resolve"
```

### Task 5: REST carries scope, pinned through the real service

**Files:**
- Modify: `src/llm_orc/web/api/ensembles.py` (request models 21-35,
  `create_ensemble` 96-115, `update_ensemble` 118-137,
  `delete_ensemble` 140-152), `src/llm_orc/web/api/profiles.py`
  (`CreateProfileRequest` 15-23, `create_profile` 32-45,
  `UpdateProfileRequest` 48-55, `update_profile` 58-63,
  `delete_profile` 66-71)
- Test: `tests/unit/web/test_api_crud_scope.py` (create)

**Interfaces:**
- Consumes: Tasks 2-3 (`scope` in handler arguments).
- Produces: `POST /api/ensembles` body `scope`; `PUT /api/ensembles/{name}`
  body `scope`; `DELETE /api/ensembles/{name}?scope=`; same three on
  `/api/profiles`. Default `project` everywhere.

- [ ] **Step 1: Write the failing tests.** These build the real
  `OrchestraService` (the module singleton is reset) against a temp
  project and the XDG temp global dir, so the assertion is the file on
  disk and the listing's `source`, not a mock's call args.

```python
"""scope over REST, through the real OrchestraService.

The assertions are files on disk and what the listing reports, so a
router that drops `scope` on the floor fails here even if every handler
unit test is green.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

import llm_orc.web.api as web_api
from llm_orc.core.config.config_manager import resolve_global_config_dir
from llm_orc.web.server import create_app

_AGENTS = [{"name": "writer", "model_profile": "local-qwen3-8b"}]


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / ".llm-orc" / "ensembles").mkdir(parents=True)
    (tmp_path / ".llm-orc" / "profiles").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(web_api, "_orchestra_service", None)
    return tmp_path


class TestEnsembleScopeOverRest:
    def test_global_scope_lands_in_global_dir_and_lists_as_global(
        self, project: Path
    ) -> None:
        with TestClient(create_app()) as client:
            created = client.post(
                "/api/ensembles",
                json={"name": "remote-made", "agents": _AGENTS, "scope": "global"},
            )
            listed = client.get("/api/ensembles")

        assert created.status_code == 200, created.text
        global_file = resolve_global_config_dir() / "ensembles" / "remote-made.yaml"
        assert Path(created.json()["path"]) == global_file
        assert global_file.exists()
        assert not (project / ".llm-orc" / "ensembles" / "remote-made.yaml").exists()
        entry = next(e for e in listed.json() if e["name"] == "remote-made")
        assert entry["source"] == "global"

    def test_default_scope_lands_in_project(self, project: Path) -> None:
        with TestClient(create_app()) as client:
            created = client.post(
                "/api/ensembles", json={"name": "mine", "agents": _AGENTS}
            )
            listed = client.get("/api/ensembles")

        assert Path(created.json()["path"]) == project / ".llm-orc" / "ensembles" / "mine.yaml"
        assert not (resolve_global_config_dir() / "ensembles" / "mine.yaml").exists()
        entry = next(e for e in listed.json() if e["name"] == "mine")
        assert entry["source"] == "local"

    def test_delete_with_scope_query_acts_on_that_scope_only(self, project: Path) -> None:
        with TestClient(create_app()) as client:
            client.post("/api/ensembles", json={"name": "keep", "agents": _AGENTS})
            wrong = client.delete("/api/ensembles/keep", params={"scope": "global"})
            right = client.delete("/api/ensembles/keep", params={"scope": "project"})

        assert wrong.status_code != 200
        assert "local tier" in wrong.text
        assert right.status_code == 200
        assert not (project / ".llm-orc" / "ensembles" / "keep.yaml").exists()

    def test_invalid_scope_is_rejected_by_the_request_model(self, project: Path) -> None:
        with TestClient(create_app()) as client:
            response = client.post(
                "/api/ensembles",
                json={"name": "nope", "agents": _AGENTS, "scope": "local"},
            )

        assert response.status_code == 422
        assert not (project / ".llm-orc" / "ensembles" / "nope.yaml").exists()


class TestProfileScopeOverRest:
    def test_global_scope_lands_in_global_dir(self, project: Path) -> None:
        with TestClient(create_app()) as client:
            created = client.post(
                "/api/profiles",
                json={"name": "remote-prof", "provider": "llama-server", "model": "qwen3-8b", "scope": "global"},
            )

        global_file = resolve_global_config_dir() / "profiles" / "remote-prof.yaml"
        assert Path(created.json()["path"]) == global_file
        assert yaml.safe_load(global_file.read_text())["model"] == "qwen3-8b"
        assert not (project / ".llm-orc" / "profiles" / "remote-prof.yaml").exists()

    def test_update_and_delete_carry_scope(self, project: Path) -> None:
        with TestClient(create_app()) as client:
            client.post(
                "/api/profiles",
                json={"name": "p", "provider": "llama-server", "model": "a", "scope": "global"},
            )
            updated = client.put("/api/profiles/p", json={"model": "b", "scope": "global"})
            wrong = client.delete("/api/profiles/p", params={"scope": "project"})
            right = client.delete("/api/profiles/p", params={"scope": "global"})

        global_file = resolve_global_config_dir() / "profiles" / "p.yaml"
        assert updated.status_code == 200
        assert wrong.status_code != 200 and "global tier" in wrong.text
        assert right.status_code == 200
        assert not global_file.exists()
```

- [ ] **Step 2: Run to verify they fail.**

Run: `uv run pytest tests/unit/web/test_api_crud_scope.py -q -p no:cacheprovider --no-cov`
Expected: FAIL. The global-scope create lands in the project dir; the
invalid-scope post returns 200 (field ignored) instead of 422.

- [ ] **Step 3: Implement.** In `ensembles.py`:

```python
from typing import Any, Literal

Scope = Literal["project", "global"]


class CreateEnsembleRequest(BaseModel):
    """Request body for ensemble creation."""

    name: str
    description: str = ""
    agents: list[dict[str, Any]] = Field(default_factory=list)
    from_template: str | None = None
    scope: Scope = "project"


class UpdateEnsembleRequest(BaseModel):
    """Request body for ensemble update."""

    changes: dict[str, Any] = Field(default_factory=dict)
    dry_run: bool = True
    backup: bool = True
    scope: Scope = "project"
```
Pass `"scope": request.scope` in `create_ensemble` and
`update_ensemble`. Change the delete signature to
`async def delete_ensemble(name: str, scope: Scope = "project") -> dict[str, Any]:`
and pass `"scope": scope`.

In `profiles.py`: add `scope: Scope = "project"` to both request models
(same `Scope` alias, defined locally or imported from
`llm_orc.services.handlers.scope`; importing is better, one vocabulary),
pass it in `create_profile`; in `update_profile` build `changes` from
`request.model_dump(exclude={"scope"})` and pass `"scope": request.scope`
alongside; `delete_profile(name: str, scope: Scope = "project")`
passes `"scope": scope`.

- [ ] **Step 4: Run the file, the existing REST tests, then the suite.**

Run: `uv run pytest tests/unit/web -q -p no:cacheprovider --no-cov`
Expected: PASS. The pre-existing mocked tests in `test_api_ensembles.py`
and `test_api_profiles.py` assert on the service being called; if they
assert exact kwargs, extend the expected dict with `"scope": "project"`.

- [ ] **Step 5: Mutant.** Drop `"scope": request.scope` from the REST
  `create_ensemble` call only. Expected RED on the first REST test at
  the `global_file` path assertion, while every Task 2 handler test
  stays green. That is the point of this pin. Revert; record.

- [ ] **Step 6: Commit.**

```bash
git add src/llm_orc/web/api/ensembles.py src/llm_orc/web/api/profiles.py tests/unit/web/test_api_crud_scope.py tests/unit/web/test_api_ensembles.py tests/unit/web/test_api_profiles.py
git commit -m "feat: REST ensemble and profile CRUD accept scope"
```

### Task 6: MCP tools carry scope, pinned over /mcp

**Files:**
- Modify: `src/llm_orc/mcp/server.py` (`update_ensemble` 274-292,
  `create_ensemble` 322-341, `delete_ensemble` 347-361,
  `create_profile` 425-456, `update_profile` 459-469,
  `delete_profile` 472-481, `create_script` 562-575,
  `delete_script` 578-591, and the hand-maintained `list_tools()`
  entry for `update_ensemble` at ~848)
- Test: `tests/unit/web/test_api_mcp.py` (append a class; reuse
  `_initialize`, `_parse_rpc_body`, `_ACCEPT_HEADERS`)

**Interfaces:**
- Consumes: Tasks 2-4.
- Produces: every listed tool has `scope: str = "project"` as its last
  parameter, forwarded as `"scope": scope`.

- [ ] **Step 1: Write the failing test.** Append:

```python
class TestMcpCrudScope:
    """scope reaches the handler through the FastMCP tool signature."""

    def test_create_profile_global_over_mcp_lands_in_global_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from llm_orc.core.config.config_manager import resolve_global_config_dir

        monkeypatch.chdir(tmp_path)
        (tmp_path / ".llm-orc" / "profiles").mkdir(parents=True)

        with TestClient(create_app()) as client:
            _, session_id = _initialize(client)
            response = client.post(
                "/mcp",
                headers={**_ACCEPT_HEADERS, "mcp-session-id": session_id},
                json={
                    "jsonrpc": "2.0",
                    "id": 4,
                    "method": "tools/call",
                    "params": {
                        "name": "create_profile",
                        "arguments": {
                            "name": "remote-prof",
                            "provider": "llama-server",
                            "model": "qwen3-8b",
                            "scope": "global",
                        },
                    },
                },
            )
            listed = client.get("/api/profiles")

        body = _parse_rpc_body(response)
        assert "error" not in body, body
        written = Path(body["result"]["structuredContent"]["path"])
        assert written == resolve_global_config_dir() / "profiles" / "remote-prof.yaml"
        assert written.exists()
        assert not (tmp_path / ".llm-orc" / "profiles" / "remote-prof.yaml").exists()
        assert "remote-prof" in {p["name"] for p in listed.json()}

    def test_every_crud_tool_advertises_scope(self) -> None:
        registered = asyncio.run(MCPServer()._mcp.list_tools())  # noqa: SLF001
        by_name = {tool.name: tool for tool in registered}
        for name in (
            "create_ensemble", "update_ensemble", "delete_ensemble",
            "create_profile", "update_profile", "delete_profile",
            "create_script", "delete_script",
        ):
            assert "scope" in by_name[name].inputSchema["properties"], name
```

- [ ] **Step 2: Run to verify it fails.**

Run: `uv run pytest tests/unit/web/test_api_mcp.py -q -p no:cacheprovider --no-cov -k Scope`
Expected: FAIL. FastMCP rejects the unknown `scope` argument (error in
the RPC body), and the schema check fails on `create_ensemble`.

- [ ] **Step 3: Implement.** For each of the eight tools add a final
  parameter `scope: str = "project",` with a docstring line
  `scope: Where to write: "project" (default) or "global"` (for delete
  and update: `scope: Which scope holds the item: "project" (default) or "global"`),
  and add `"scope": scope,` to the dict passed to the service. Example
  for `create_ensemble`:

```python
        @self._mcp.tool()
        async def create_ensemble(
            name: str,
            description: str = "",
            agents: list[dict[str, Any]] | None = None,
            from_template: str | None = None,
            scope: str = "project",
        ) -> dict[str, Any]:
            """Create a new ensemble from scratch or template.

            Args:
                name: Name of the new ensemble
                description: Optional description
                agents: List of agent configurations
                from_template: Optional template ensemble to copy from
                scope: Where to write: "project" (default) or "global"
            """
            result = await self._service.create_ensemble(
                {
                    "name": name,
                    "description": description,
                    "agents": agents or [],
                    "from_template": from_template,
                    "scope": scope,
                }
            )
            return result
```
Validation stays in `parse_scope` (one place); the tool parameter is a
plain `str` so an invalid value produces the handler's message, not a
FastMCP schema error, and the same message on all three surfaces.

In `list_tools()`, add to the `update_ensemble` entry's properties:
```python
                        "scope": {
                            "type": "string",
                            "enum": ["project", "global"],
                            "default": "project",
                        },
```

- [ ] **Step 4: Run the file, then the suite.**

- [ ] **Step 5: Mutant.** Remove `"scope": scope` from the
  `create_profile` tool body only (keep the parameter). Expected RED on
  the first MCP test at the `written ==` assertion (the file landed in
  the project). Revert; record.

- [ ] **Step 6: Commit.**

```bash
git add src/llm_orc/mcp/server.py tests/unit/web/test_api_mcp.py
git commit -m "feat: MCP CRUD tools accept scope"
```

### Task 7: docs and changelog

**Files:**
- Modify: `docs/cli-reference.md:177-190` (MCP tool table rows for the
  eight tools), `docs/architecture.md:209-216` (same tools),
  `CHANGELOG.md` (`## [Unreleased]`)

- [ ] **Step 1: Edit the two tool tables.** Append " (`scope`:
  `project` default, or `global` to write under `~/.config/llm-orc`)"
  to the create rows, and " (`scope` selects which copy)" to the
  update/delete rows. One sentence after the table in
  `docs/cli-reference.md`: "Reads merge project, library and global;
  writes go to exactly one scope. A name that exists only in another
  scope is an error naming where it lives."

- [ ] **Step 2: Changelog.** Under `## [Unreleased]`:

```markdown
### Added
- `scope: project | global` on ensemble, profile and script create/update/delete
  across REST (`POST /api/ensembles`, `PUT`/`DELETE /api/ensembles/{name}?scope=`,
  same on `/api/profiles`) and the MCP tools. Default `project` is unchanged.
  `global` writes under the XDG config dir (`~/.config/llm-orc/{ensembles,profiles,scripts}`),
  so ensembles created on a remote serve survive upgrades (#196). Scripts
  created with `scope: global` resolve at execution time (the resolver
  searches the global scripts dir last) and appear in `list_scripts` with
  their scope.

### Changed
- `delete_ensemble` / `delete_profile` / `delete_script` and the update
  tools act on one scope only. A name that lives in another tier (including
  the library) is an error naming the tier and path; previously delete
  removed the first match across all tiers, library included.
```

- [ ] **Step 3: Run lint (doc drift included) and commit.**

Run: `make lint`
Expected: clean.

```bash
git add docs/cli-reference.md docs/architecture.md CHANGELOG.md
git commit -m "docs: scope on CRUD across REST and MCP"
```

### Task 8: live row and review gate

**Files:**
- Modify: `docs/plans/2026-09-28-remote-delegation.md` (append
  "Arc 1 live row (date)" under the Arc 1 card), GitHub #196 (comment).

- [ ] **Step 1: Live row against a local serve.** From the worktree,
  with the laptop's llama-server available:

```bash
uv run llm-orc serve --port 8766 &
curl -s localhost:8766/api/profiles | python -c 'import json,sys; print([p["name"] for p in json.load(sys.stdin)][:5])'
# pick a local profile name from that list for the agent below
curl -s -X POST localhost:8766/api/ensembles -H 'content-type: application/json' \
  -d '{"name":"scope-live","agents":[{"name":"echo","model_profile":"<local profile>","system_prompt":"Reply with the single word ok."}],"scope":"global"}'
ls ~/.config/llm-orc/ensembles/scope-live.yaml
curl -s localhost:8766/api/ensembles | python -c 'import json,sys; print([e for e in json.load(sys.stdin) if e["name"]=="scope-live"])'
curl -s -X POST localhost:8766/api/ensembles/scope-live/execute -H 'content-type: application/json' -d '{"input":"hello"}' | head -c 600
curl -s -X DELETE 'localhost:8766/api/ensembles/scope-live?scope=project'
curl -s -X DELETE 'localhost:8766/api/ensembles/scope-live?scope=global'
ls ~/.config/llm-orc/ensembles/scope-live.yaml
kill %1
```
Expected: the file exists after create, the listing shows
`source: global`, execute returns `status: success`, the project-scope
delete errors naming the global tier, the global-scope delete removes
the file. If the checkout's `~/.config/llm-orc` is the developer's real
one, this writes and then deletes one file there; that is the point of
the row.

- [ ] **Step 2: Record.** Append the row (commands run, observed
  status per step, serve version) under the Arc 1 card in the spec doc.
  Comment on #196 with the row and the three pre-existing defects from
  Review Focus on #191 (one comment each, no new issues).

```bash
git add docs/plans/2026-09-28-remote-delegation.md
git commit -m "docs: Arc 1 live row"
```

- [ ] **Step 3: Review gate.** Independent adversarial review (Opus,
  not the implementer) with an explicit wrong-accept hunt: for each pin
  in Tasks 2-6, can it pass while the defect it names is live? Merge to
  local `main` only on APPROVE. Nothing is pushed.

- [ ] **Step 4: The mini (practitioner does or okays).** Move
  `research-dossier` and `test-research-pipeline` from the checkout's
  `.llm-orc/ensembles/` on remote-host into `~/.config/llm-orc/ensembles/`.
  Not automated here.

---

## Arcs 2-5

Held at card level in the spec (`docs/plans/2026-09-28-remote-delegation.md`)
until S1 and S2 findings land. Re-cut them into tasks in a new plan
file from the findings; do not expand them here speculatively.
