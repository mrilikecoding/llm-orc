# Remote delegation, Arc 3: transitive preflight

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `check_ensemble_runnable` walks an ensemble's whole closure
(child ensembles, loop bodies, dispatch targets, scripts, profiles and
their fallback chains, inline models) and classifies every dependency
with one of eleven statuses and a resolve hint, so a caller on the
tailnet learns what the remote is missing before anything runs (#196,
#191).

**Architecture:** One new core module builds the closure from typed
agent configs (`core/config/closure.py`; Arc 5 reuses it on the laptop to
ship a closure). One new handler module classifies dependencies
(`services/handlers/preflight.py`) against the router's inventory, the
profile map, the script resolver and the provider status.
`ProviderHandler.check_ensemble_runnable` composes the two and adds a
`dependencies` list to the existing result, keeping `runnable` and
`agents` for the web UI and the MCP docstring. The router's one `GET
/models` listing decides routable, downloaded and load state
(`LlamaServerClient.inventory()`); nothing else is consulted for model
presence. Rulings and the probe: the spec's "Arc 3 re-cut (2026-09-29)".

**Tech Stack:** Python 3.11+, pydantic, FastAPI TestClient, pytest
(asyncio auto mode), ruff 88, mypy strict, complexipy <= 15.

**Spec:** `docs/plans/2026-09-28-remote-delegation.md` (decisions, the
pattern, Arc 3 card, "Arc 3 re-cut (2026-09-29)", S2 findings, standing
rules). Read "Standing rules for every implementer" and the Arc 3 re-cut
before touching anything.

## Global Constraints

- Own git worktree off `main`: `git worktree add .claude/worktrees/arc3-preflight -b feat/transitive-preflight main`.
  Never `git stash` (the working dir is shared with other sessions);
  never spawn subagents that edit or commit.
- Strict TDD. Structural and behavioral commits separate. ruff 88 +
  mypy strict from the first draft. `make lint` clean (mypy, ruff,
  format, complexipy 15, bandit, vulture, doc drift).
- Commit prefixes `feat:` `fix:` `refactor:` `test:` `docs:` `chore:`.
  No AI attribution of any kind. No session links, no scratch paths.
- Full suite: `uv run pytest -q -p no:cacheprovider`. Known local-only
  failure: `testget_available_providers_auth_only` when a router listens
  on :8080. Baseline on `main` (`c85da926`) before this arc: 4786 passed
  (verify in Task 0 and record the real number).
- Doctrine 11: pins assert outcomes through the real surface (REST
  TestClient over a real `OrchestraService` on a temp project dir, the
  real `EnsembleLoader`, the real `ScriptResolver`). Each key pin is
  shown RED under a named mutant before it counts. Report mutant and the
  red assertion line in the commit body.
- Doc drift gate: `scripts/check_doc_drift.py` fails `make lint` when a
  `docs/plans/*.md` file names a `test_*` identifier in single backticks
  that exists nowhere in code. This plan names tests only inside fenced
  code blocks. Keep it that way when editing this file.
- Findings go on #196 and #191 as comments, never new issues.
- Nothing is pushed; nothing paid is run; the mini is not touched.
- Tests may not launch the real `llama-server` (autouse guard in
  `tests/conftest.py`). The router is faked at `LlamaServerClient._list`
  (Task 1 introduces it) with the exact JSON shape the probe recorded.
- The statuses, hints and the coarse agent mapping are the spec's
  rulings 2, 3 and 6 verbatim. Do not add a status.

## Review Focus

Inputs the spec implies but no card names. Each gets a pin in the owning
task.

1. **A profile that appears under two agents, one of them a child
   ensemble's.** Expected: reported once (closure is a set) with the
   first path, yet the second agent's coarse status still reflects it.
   Pin in Task 4 (the `owned` frames).
2. **A profile-level fallback chain whose last hop is missing.**
   Expected: the missing hop is `missing_profile` with the agent's path;
   the chain stops there. Pin in Task 2.
3. **A listed model whose profile has no `hf_repo`** (a `--models-dir`
   or hand-written preset entry). Expected: `ready`, not
   `missing_model_source`; the router lists it. Pin in Task 3.
4. **A bare script name that is inline content** (`script: "echo hi"`).
   Expected: `ready`; the resolver classifies it inline. A path-syntax
   reference to a file that is not there is `missing_script`. Pin in
   Task 3 (classifier) and Task 4 (through the real resolver on disk).
5. **The router listing carries `default` and a cache entry.** Expected:
   neither is a routable model; the cache entry is exactly the
   `hf_repo` string. Pin in Task 1.

Pre-existing, out of scope, for the #191 comment at the end: the web
frontend's `AgentRunnableStatus` type says `model_not_found` where the
API says `model_unavailable`; `PromotionHandler` re-derives model
presence from `providers["llama-server"]["models"]` and does not know
about the cache.

---

## Task 0: worktree and baseline

- [ ] **Step 1: Worktree.**

```bash
cd /Users/nathangreen/Development/eddi-lab/llm-orc
git worktree add .claude/worktrees/arc3-preflight -b feat/transitive-preflight main
cd .claude/worktrees/arc3-preflight
uv sync -q
```

- [ ] **Step 2: Baseline.** Record the passed count for the review.

```bash
uv run pytest -q -p no:cacheprovider 2>&1 | tail -3
```

Expected: `4786 passed` (or the real number; write it down).

---

## Task 1: the router inventory (routable models and cached sources)

**Files:**
- Modify: `src/llm_orc/providers/llama_server.py:185-235` (`_is_preset_model`, `LlamaServerClient`)
- Modify: `src/llm_orc/providers/status_types.py:18-25` (`LlamaServerProviderStatus`)
- Modify: `src/llm_orc/services/handlers/provider_handler.py:54-75` (`_get_llama_server_status`)
- Test: `tests/unit/providers/test_llama_server_client.py` (create), `tests/unit/services/handlers/test_provider_handler.py`

**Interfaces:**
- Produces: `RouterInventory(models: list[dict[str, Any]], cached: list[str])`;
  `LlamaServerClient._list() -> list[dict[str, Any]]` (the one GET; the
  test seam); `LlamaServerClient.inventory() -> RouterInventory`;
  `LlamaServerClient.models()` unchanged in behavior (delegates to
  `inventory().models`); `LlamaServerProviderStatus.cached: list[str]`;
  the provider status dict for `llama-server` carries `cached`.

- [ ] **Step 1: Write the failing tests.** The listing is the probe's
  shape (spec, Arc 3 re-cut ruling 1).

```python
"""The router's one listing decides routable, downloaded and load state."""

from __future__ import annotations

from unittest.mock import patch

from llm_orc.providers.llama_server import LlamaServerClient

PROBE_LISTING = [
    {"id": "qwen3-8b", "status": {"value": "unloaded"}, "source": "preset"},
    {"id": "qwen3-14b", "status": {"value": "unloaded"}, "source": "preset"},
    {"id": "default", "status": {"value": "unloaded"}, "source": "preset"},
    {
        "id": "unsloth/Qwen3-8B-GGUF:Q4_K_M",
        "status": {"value": "unloaded"},
        "source": "cache",
    },
]


class TestInventory:
    def test_models_hide_default_and_cache_entries(self) -> None:
        client = LlamaServerClient("http://127.0.0.1:8791")
        with patch.object(LlamaServerClient, "_list", return_value=PROBE_LISTING):
            inventory = client.inventory()
        assert [m["id"] for m in inventory.models] == ["qwen3-8b", "qwen3-14b"]

    def test_cached_is_exactly_the_hf_repo_strings(self) -> None:
        client = LlamaServerClient("http://127.0.0.1:8791")
        with patch.object(LlamaServerClient, "_list", return_value=PROBE_LISTING):
            inventory = client.inventory()
        assert inventory.cached == ["unsloth/Qwen3-8B-GGUF:Q4_K_M"]

    def test_models_and_inventory_share_one_listing(self) -> None:
        client = LlamaServerClient("http://127.0.0.1:8791")
        with patch.object(
            LlamaServerClient, "_list", return_value=PROBE_LISTING
        ) as listing:
            client.models()
            client.inventory()
        assert listing.call_count == 2  # one GET per call, never two per call
```

And in `tests/unit/services/handlers/test_provider_handler.py`, a new
class at the end:

```python
class TestLlamaServerStatusCarriesCache:
    async def test_status_reports_models_and_cached_sources(self) -> None:
        from llm_orc.providers.llama_server import LlamaServerClient

        handler = _make_handler()
        listing = [
            {"id": "qwen3-8b", "status": {"value": "loaded"}},
            {"id": "default", "status": {"value": "unloaded"}},
            {"id": "unsloth/Qwen3-8B-GGUF:Q4_K_M", "status": {"value": "unloaded"}},
        ]
        with patch.object(LlamaServerClient, "_list", return_value=listing):
            status = await handler._get_llama_server_status()

        assert status["available"] is True
        assert status["models"] == ["qwen3-8b"]
        assert status["cached"] == ["unsloth/Qwen3-8B-GGUF:Q4_K_M"]
```

- [ ] **Step 2: Run them RED.**

```bash
uv run pytest tests/unit/providers/test_llama_server_client.py tests/unit/services/handlers/test_provider_handler.py -q -p no:cacheprovider
```

Expected: FAIL, `AttributeError: ... has no attribute '_list'` /
`'inventory'`; the status test fails on `KeyError: 'cached'`.

- [ ] **Step 3: Implement.** In `llama_server.py`, replace
  `_is_preset_model` and `LlamaServerClient.models` with:

```python
def _is_preset_model(model: Mapping[str, Any]) -> bool:
    model_id = str(model.get("id", ""))
    return model_id != "default" and "/" not in model_id


def _is_cache_entry(model: Mapping[str, Any]) -> bool:
    """A raw Hugging Face cache entry: the router lists one per cached
    GGUF, its id the ``user/repo:tag`` string a profile's ``hf_repo``
    spells (probe 2026-09-29, spec Arc 3 re-cut ruling 1). The id shape
    is the signal, not the newer ``source`` field."""
    model_id = str(model.get("id", ""))
    return model_id != "default" and "/" in model_id


@dataclass(frozen=True)
class RouterInventory:
    """What one ``GET /models`` says: the models the router routes to
    (preset sections) and the sources it has on disk."""

    models: list[dict[str, Any]]
    cached: list[str]
```

and inside the class:

```python
    def _list(self) -> list[dict[str, Any]]:
        """One ``GET /models``, the raw ``data`` list."""
        with _DIRECT.open(f"{self.root_url}/models", timeout=5) as resp:
            data = json.load(resp)
        return [m for m in data.get("data", []) if isinstance(m, dict)]

    def inventory(self) -> RouterInventory:
        """Routable models and cached sources from one listing.

        Router mode lists a ``default`` entry for its own command line
        and one raw ``user/repo:tag`` entry per cached Hugging Face
        file (e2e 2026-09-16); neither is a preset model. The cache
        entries are what "downloaded" means (spec Arc 3 re-cut, ruling 1).
        """
        listing = self._list()
        return RouterInventory(
            models=[m for m in listing if _is_preset_model(m)],
            cached=sorted(str(m["id"]) for m in listing if _is_cache_entry(m)),
        )

    def models(self) -> list[dict[str, Any]]:
        """The router's model list with per-model load status."""
        return self.inventory().models
```

`dataclass` and `field` are already imported at the top of the module
(`RenderedPreset` uses them). In `status_types.py`:

```python
class LlamaServerProviderStatus(BaseModel):
    """Status of the llama-server router (#90): reachable, what it
    serves, and which sources it has downloaded (#196)."""

    available: bool
    models: list[str] = Field(default_factory=list)
    cached: list[str] = Field(default_factory=list)
    model_count: int = 0
    reason: str = ""
    base_url: str = ""
```

In `provider_handler._get_llama_server_status`, replace the `try` body:

```python
        try:
            inventory = client.inventory()
        except (OSError, ValueError) as e:
            return LlamaServerProviderStatus(
                available=False,
                reason=f"llama-server not reachable: {type(e).__name__}: {e}",
                base_url=base_url,
            ).model_dump()
        models = sorted(str(m.get("id", "")) for m in inventory.models)
        return LlamaServerProviderStatus(
            available=True,
            models=models,
            cached=inventory.cached,
            model_count=len(models),
            base_url=base_url,
        ).model_dump()
```

- [ ] **Step 4: Run GREEN, then the mutant.** Mutant M1: make
  `_is_cache_entry` return `model_id != "default"`. Expected red line:
  `assert inventory.cached == ["unsloth/Qwen3-8B-GGUF:Q4_K_M"]` (the
  preset ids leak in). Revert the mutant. Run the whole suite: the
  existing `LlamaServerSupervisor.models()` and `web/api/models.py`
  callers go through `models()` and must be untouched.

```bash
uv run pytest tests/unit/providers tests/unit/services/handlers tests/unit/web/test_api_models.py -q -p no:cacheprovider
```

- [ ] **Step 5: Commit.**

```bash
git add src/llm_orc/providers/llama_server.py src/llm_orc/providers/status_types.py src/llm_orc/services/handlers/provider_handler.py tests/unit/providers/test_llama_server_client.py tests/unit/services/handlers/test_provider_handler.py
git commit -m "feat: router inventory reports cached sources next to routable models"
```

Body: name mutant M1 and the red assertion line.

---

## Task 2: the closure walker

**Files:**
- Create: `src/llm_orc/core/config/closure.py`
- Test: `tests/unit/core/config/test_closure.py` (create; check whether
  `tests/unit/core/config/` exists and has an `__init__.py`; match the
  neighbors)

**Interfaces:**
- Consumes: `EnsembleConfig` (`llm_orc.core.config.ensemble_config`),
  the typed agent classes in `llm_orc.schemas.agent_config`.
- Produces:

```python
Kind = Literal["ensemble", "dispatch", "script", "profile", "model"]
Key = tuple[str, str]  # (kind, name)

@dataclass(frozen=True)
class Dependency:
    kind: Kind
    name: str
    via: tuple[str, ...]          # "<ensemble>.<agent>" frames, root first; () for the root
    provider: str | None = None   # inline model agents and found profiles
    found: bool = True            # ensembles resolved, profiles present

@dataclass(frozen=True)
class Closure:
    dependencies: list[Dependency]            # walk order, deduplicated on (kind, name)
    owned: dict[str, frozenset[Key]]          # frame -> every key reachable through that frame

FindEnsemble = Callable[[str], EnsembleConfig | None]

def walk_closure(
    root: EnsembleConfig,
    find_ensemble: FindEnsemble,
    profiles: Mapping[str, Mapping[str, Any]],
) -> Closure: ...
```

- [ ] **Step 1: Write the failing tests.** Real YAML on disk through the
  real loader; the finder searches the temp dir.

```python
"""The transitive closure of an ensemble (spec, Arc 3 re-cut ruling 4)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from llm_orc.core.config.closure import Dependency, walk_closure
from llm_orc.core.config.ensemble_config import EnsembleConfig, EnsembleLoader


def _write(dir_path: Path, name: str, agents: list[dict[str, Any]]) -> None:
    (dir_path / f"{name}.yaml").write_text(
        yaml.safe_dump({"name": name, "description": name, "agents": agents})
    )


def _finder(dir_path: Path):  # type: ignore[no-untyped-def]
    loader = EnsembleLoader()

    def find(name: str) -> EnsembleConfig | None:
        return loader.find_ensemble(str(dir_path), name)

    return find


@pytest.fixture
def ensembles(tmp_path: Path) -> Path:
    d = tmp_path / "ensembles"
    d.mkdir()
    return d


def _keys(deps: list[Dependency]) -> list[tuple[str, str, tuple[str, ...]]]:
    return [(d.kind, d.name, d.via) for d in deps]


class TestWalkClosure:
    def test_root_child_script_profile_and_dynamic_dispatch(
        self, ensembles: Path
    ) -> None:
        _write(
            ensembles,
            "child",
            [
                {"name": "runner", "script": "scripts/gone.py"},
                {"name": "thinker", "model_profile": "p-child"},
            ],
        )
        _write(
            ensembles,
            "top",
            [
                {"name": "writer", "model_profile": "p-top"},
                {"name": "sub", "ensemble": "child"},
                {"name": "router", "dispatch": "${sub.target}"},
                {"name": "fixed", "dispatch": "child"},
                {"name": "absent", "ensemble": "no-such"},
            ],
        )
        root = _finder(ensembles)("top")
        assert root is not None

        closure = walk_closure(
            root, _finder(ensembles), {"p-top": {}, "p-child": {}}
        )

        assert _keys(closure.dependencies) == [
            ("ensemble", "top", ()),
            ("profile", "p-top", ("top.writer",)),
            ("ensemble", "child", ("top.sub",)),
            ("script", "scripts/gone.py", ("top.sub", "child.runner")),
            ("profile", "p-child", ("top.sub", "child.thinker")),
            ("dispatch", "${sub.target}", ("top.router",)),
            ("ensemble", "no-such", ("top.absent",)),
        ]
        missing = next(d for d in closure.dependencies if d.name == "no-such")
        assert missing.found is False

    def test_second_reference_is_deduplicated_but_owned_by_both_frames(
        self, ensembles: Path
    ) -> None:
        """Review focus 1: the closure is a set; ownership is not."""
        _write(ensembles, "child", [{"name": "runner", "script": "scripts/x.py"}])
        _write(
            ensembles,
            "top",
            [
                {"name": "first", "ensemble": "child"},
                {"name": "second", "ensemble": "child"},
            ],
        )
        root = _finder(ensembles)("top")
        assert root is not None

        closure = walk_closure(root, _finder(ensembles), {})

        assert [d.name for d in closure.dependencies] == ["top", "child", "scripts/x.py"]
        assert ("script", "scripts/x.py") in closure.owned["top.first"]
        assert ("script", "scripts/x.py") in closure.owned["top.second"]

    def test_loop_body_is_an_ensemble_dependency(self, ensembles: Path) -> None:
        _write(ensembles, "body", [{"name": "step", "model_profile": "p"}])
        _write(
            ensembles,
            "top",
            [
                {
                    "name": "looper",
                    "loop": {"body": "body", "until": "${done}", "max_iterations": 3},
                }
            ],
        )
        root = _finder(ensembles)("top")
        assert root is not None

        closure = walk_closure(root, _finder(ensembles), {"p": {}})

        assert _keys(closure.dependencies)[1:] == [
            ("ensemble", "body", ("top.looper",)),
            ("profile", "p", ("top.looper", "body.step")),
        ]

    def test_fallback_chain_is_followed_until_a_missing_hop(
        self, ensembles: Path
    ) -> None:
        """Review focus 2."""
        _write(
            ensembles,
            "top",
            [{"name": "w", "model_profile": "a", "fallback_model_profile": "x"}],
        )
        profiles: dict[str, dict[str, Any]] = {
            "a": {"fallback_model_profile": "b"},
            "b": {"fallback_model_profile": "gone"},
            "x": {},
        }
        root = _finder(ensembles)("top")
        assert root is not None

        closure = walk_closure(root, _finder(ensembles), profiles)

        assert _keys(closure.dependencies)[1:] == [
            ("profile", "a", ("top.w",)),
            ("profile", "b", ("top.w",)),
            ("profile", "gone", ("top.w",)),
            ("profile", "x", ("top.w",)),
        ]
        gone = next(d for d in closure.dependencies if d.name == "gone")
        assert gone.found is False

    def test_inline_model_agent_is_a_model_dependency(self, ensembles: Path) -> None:
        _write(
            ensembles,
            "top",
            [{"name": "w", "model": "qwen3-8b", "provider": "llama-server"}],
        )
        root = _finder(ensembles)("top")
        assert root is not None

        closure = walk_closure(root, _finder(ensembles), {})

        dep = closure.dependencies[1]
        assert (dep.kind, dep.name, dep.provider) == ("model", "qwen3-8b", "llama-server")

    def test_mutual_references_terminate(self, ensembles: Path) -> None:
        """The loader rejects cycles; the walker must not depend on that."""
        from llm_orc.schemas.agent_config import EnsembleAgentConfig

        # Built directly: the loader would refuse this graph (Invariant 5).
        a = EnsembleConfig(
            name="a", description="a", agents=[EnsembleAgentConfig(name="to_b", ensemble="b")]
        )
        b = EnsembleConfig(
            name="b", description="b", agents=[EnsembleAgentConfig(name="to_a", ensemble="a")]
        )
        table = {"a": a, "b": b}

        closure = walk_closure(a, lambda n: table.get(n), {})

        assert [d.name for d in closure.dependencies] == ["a", "b"]
```

- [ ] **Step 2: Run RED.**

```bash
uv run pytest tests/unit/core/config/test_closure.py -q -p no:cacheprovider
```

Expected: `ModuleNotFoundError: llm_orc.core.config.closure`.

- [ ] **Step 3: Implement `src/llm_orc/core/config/closure.py`.**

```python
"""The transitive closure of an ensemble: everything it needs to run.

Delegation is "ship the closure, preflight transitively, resolve
explicitly, run" (docs/plans/2026-09-28-remote-delegation.md). This
module is the first two words: from a root ensemble it collects every
child ensemble (``ensemble:``, ``loop.body``, a literal ``dispatch:``),
every script, every profile with its ``fallback_model_profile`` chain,
and every inline model, each with the path of agents that reaches it.
The preflight classifier (services/handlers/preflight.py) and, later,
the CLI's closure shipper both consume it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from llm_orc.core.config.ensemble_config import EnsembleConfig
from llm_orc.schemas.agent_config import (
    DynamicDispatchAgentConfig,
    EnsembleAgentConfig,
    LlmAgentConfig,
    LoopAgentConfig,
    ScriptAgentConfig,
)

Kind = Literal["ensemble", "dispatch", "script", "profile", "model"]
Key = tuple[str, str]
FindEnsemble = Callable[[str], EnsembleConfig | None]


@dataclass(frozen=True)
class Dependency:
    """One thing an ensemble needs, and the agent path that names it."""

    kind: Kind
    name: str
    via: tuple[str, ...]
    provider: str | None = None
    found: bool = True


@dataclass(frozen=True)
class Closure:
    """The dependency set in walk order, plus which frame owns what.

    ``dependencies`` is deduplicated on ``(kind, name)``: the closure is
    a set, and the first sighting keeps its path. ``owned`` maps every
    frame (``"<ensemble>.<agent>"``) to every key reachable through it,
    duplicates included, so a per-agent verdict never misses a
    dependency that was first reported under a sibling.
    """

    dependencies: list[Dependency]
    owned: dict[str, frozenset[Key]]


def walk_closure(
    root: EnsembleConfig,
    find_ensemble: FindEnsemble,
    profiles: Mapping[str, Mapping[str, Any]],
) -> Closure:
    """Depth-first in agent order from ``root``.

    A child that does not resolve is recorded ``found=False`` and not
    walked. A ``${...}`` dispatch target cannot be followed statically
    (issue #94) and is recorded as ``dispatch``. Cycles cannot load
    (Invariant 5) but the walk keeps its own visited set regardless.
    """
    walker = _Walker(find_ensemble, profiles)
    walker.add(Dependency("ensemble", root.name, ()), ())
    walker.visit(root, ())
    return Closure(walker.deps, {f: frozenset(k) for f, k in walker.owned.items()})


class _Walker:
    def __init__(
        self, find_ensemble: FindEnsemble, profiles: Mapping[str, Mapping[str, Any]]
    ) -> None:
        self._find = find_ensemble
        self._profiles = profiles
        self.deps: list[Dependency] = []
        self.owned: dict[str, set[Key]] = {}
        self._seen: set[Key] = set()
        self._visited: set[str] = set()
        self._stack: list[str] = []  # ensembles being walked, root first
        self._members: dict[str, set[Key]] = {}  # ensemble -> keys under it

    def add(self, dep: Dependency, via: tuple[str, ...]) -> bool:
        """Record ``dep`` under every frame of ``via`` and every ensemble
        on the stack; True on first sight."""
        key: Key = (dep.kind, dep.name)
        self._own(key, via)
        if key in self._seen:
            return False
        self._seen.add(key)
        self.deps.append(dep)
        return True

    def _own(self, key: Key, via: tuple[str, ...]) -> None:
        for frame in via:
            self.owned.setdefault(frame, set()).add(key)
        for ensemble in self._stack:
            self._members.setdefault(ensemble, set()).add(key)

    def visit(self, config: EnsembleConfig, via: tuple[str, ...]) -> None:
        if config.name in self._visited:
            return
        self._visited.add(config.name)
        self._stack.append(config.name)
        try:
            for agent in config.agents:
                self._visit_agent(agent, (*via, f"{config.name}.{agent.name}"))
        finally:
            self._stack.pop()

    def _visit_agent(self, agent: Any, via: tuple[str, ...]) -> None:
        if isinstance(agent, ScriptAgentConfig):
            self.add(Dependency("script", agent.script, via), via)
        elif isinstance(agent, EnsembleAgentConfig):
            self._child(agent.ensemble, via)
        elif isinstance(agent, LoopAgentConfig):
            self._child(agent.loop.body, via)
        elif isinstance(agent, DynamicDispatchAgentConfig):
            if "${" in agent.dispatch:
                self.add(Dependency("dispatch", agent.dispatch, via), via)
            else:
                self._child(agent.dispatch, via)
        elif isinstance(agent, LlmAgentConfig):
            self._llm(agent, via)

    def _child(self, name: str, via: tuple[str, ...]) -> None:
        child = self._find(name)
        first = self.add(
            Dependency("ensemble", name, via, found=child is not None), via
        )
        if child is None:
            return
        if first:
            self.visit(child, via)
            return
        # Already walked under another frame: this frame owns its members too.
        for key in self._members.get(name, set()):
            self._own(key, via)

    def _llm(self, agent: LlmAgentConfig, via: tuple[str, ...]) -> None:
        if agent.model_profile is not None:
            self._profile_chain(agent.model_profile, via)
            if agent.fallback_model_profile:
                self._profile_chain(agent.fallback_model_profile, via)
        elif agent.model is not None:
            self.add(
                Dependency("model", agent.model, via, provider=agent.provider), via
            )

    def _profile_chain(self, name: str, via: tuple[str, ...]) -> None:
        current: str | None = name
        while current:
            profile = self._profiles.get(current)
            provider = profile.get("provider") if profile is not None else None
            dep = Dependency(
                "profile",
                current,
                via,
                provider=str(provider) if provider else None,
                found=profile is not None,
            )
            if not self.add(dep, via) or profile is None:
                return
            nxt = profile.get("fallback_model_profile")
            current = nxt if isinstance(nxt, str) and nxt else None
```

Note on `_members`: keyed by ensemble name from the walk stack (never
parsed out of a frame string), so a child walked once under `top.first`
hands its whole subtree to `top.second`, and to every ensemble above
it, on the second sighting.

- [ ] **Step 4: Run GREEN; mutant M3.** Mutant: in `_child`, delete the
  `self.visit(child, via)` line. Expected red line: the first test's
  `assert _keys(closure.dependencies) == [...]` (the child's script and
  profile vanish). Revert. `make lint` (complexipy: `_Walker` methods
  are each under 15; if `_visit_agent` trips it, split the LLM branch
  out, it already is).

- [ ] **Step 5: Commit.**

```bash
git add src/llm_orc/core/config/closure.py tests/unit/core/config/test_closure.py
git commit -m "feat: closure walker collects an ensemble's transitive dependencies"
```

---

## Task 3: the dependency classifier

**Files:**
- Create: `src/llm_orc/services/handlers/preflight.py`
- Test: `tests/unit/services/handlers/test_preflight.py` (create)

**Interfaces:**
- Consumes: `Dependency` from Task 2; the provider status dict from
  `ProviderHandler.get_provider_status` (`{"providers": {...}}`, with
  `llama-server` carrying `available`, `models`, `cached`, `reason`;
  `openai-compatible` carrying `available`, `endpoints[{base_url,
  available, models}]`; cloud providers carrying `available`).
- Produces:

```python
class DependencyStatus(StrEnum):
    READY = "ready"; PULLABLE = "pullable"; NEEDS_RESTART = "needs_restart"
    MISSING_PROFILE = "missing_profile"; MISSING_MODEL_SOURCE = "missing_model_source"
    MODEL_UNAVAILABLE = "model_unavailable"; NEEDS_CREDENTIALS = "needs_credentials"
    MISSING_SCRIPT = "missing_script"; MISSING_ENSEMBLE = "missing_ensemble"
    PROVIDER_UNAVAILABLE = "provider_unavailable"; DYNAMIC = "dynamic"

RESOLVE: dict[DependencyStatus, str]          # ruling 2's hint table
UNBLOCKING = frozenset({DependencyStatus.READY, DependencyStatus.DYNAMIC})

class DependencyReport(BaseModel):
    kind: str; name: str; via: list[str]; status: DependencyStatus
    resolve: str; detail: str; provider: str | None = None

def classify_dependencies(
    dependencies: Sequence[Dependency],
    *,
    profiles: Mapping[str, Mapping[str, Any]],
    providers: Mapping[str, Any],
    script_found: Callable[[str], bool],
) -> list[DependencyReport]: ...

def is_runnable(reports: Sequence[DependencyReport]) -> bool: ...
```

- [ ] **Step 1: Write the failing tests.** Table-driven over ruling 3.

```python
"""Dependency classification (spec, Arc 3 re-cut rulings 2, 3, 6)."""

from __future__ import annotations

from typing import Any

import pytest

from llm_orc.core.config.closure import Dependency
from llm_orc.services.handlers.preflight import (
    DependencyStatus,
    classify_dependencies,
    is_runnable,
)

ROUTER_UP: dict[str, Any] = {
    "llama-server": {
        "available": True,
        "models": ["qwen3-8b", "qwen3-14b", "handmade"],
        "cached": ["unsloth/Qwen3-8B-GGUF:Q4_K_M"],
    },
    "anthropic-api": {"available": False, "reason": "not configured"},
    "google-gemini": {"available": True, "reason": "configured"},
    "openai-compatible": {
        "available": True,
        "endpoints": [
            {"base_url": "http://oai.local/v1", "available": True, "models": ["gpt-x"]},
            {"base_url": "http://down.local/v1", "available": False, "models": []},
        ],
    },
}

PROFILES: dict[str, dict[str, Any]] = {
    "ready": {"provider": "llama-server", "model": "qwen3-8b", "hf_repo": "unsloth/Qwen3-8B-GGUF:Q4_K_M"},
    "pull": {"provider": "llama-server", "model": "qwen3-14b", "hf_repo": "unsloth/Qwen3-14B-GGUF:Q4_K_M"},
    "new": {"provider": "llama-server", "model": "qwen3-4b", "hf_repo": "unsloth/Qwen3-4B-GGUF:Q4_K_M"},
    "handmade": {"provider": "llama-server", "model": "handmade"},
    "nosrc": {"provider": "llama-server", "model": "mystery"},
    "claude": {"provider": "anthropic-api", "model": "claude-x"},
    "gemini": {"provider": "google-gemini", "model": "gemini-x"},
    "oai": {"provider": "openai-compatible", "model": "gpt-x", "base_url": "http://oai.local/v1"},
    "oai-absent": {"provider": "openai-compatible", "model": "gpt-y", "base_url": "http://oai.local/v1"},
    "oai-down": {"provider": "openai-compatible", "model": "gpt-x", "base_url": "http://down.local/v1"},
    "legacy": {"provider": "ollama", "model": "llama3"},
}


def _profile(name: str) -> Dependency:
    profile = PROFILES.get(name)
    provider = profile.get("provider") if profile else None
    return Dependency("profile", name, ("top.a",), provider=provider, found=profile is not None)


def _one(dep: Dependency, providers: dict[str, Any] = ROUTER_UP) -> tuple[DependencyStatus, str]:
    [report] = classify_dependencies(
        [dep], profiles=PROFILES, providers=providers, script_found=lambda _: False
    )
    return report.status, report.resolve


@pytest.mark.parametrize(
    ("profile", "status", "resolve"),
    [
        ("ready", DependencyStatus.READY, "none"),
        ("pull", DependencyStatus.PULLABLE, "pull"),
        ("new", DependencyStatus.NEEDS_RESTART, "restart"),
        ("handmade", DependencyStatus.READY, "none"),  # review focus 3
        ("nosrc", DependencyStatus.MISSING_MODEL_SOURCE, "add_source"),
        ("claude", DependencyStatus.NEEDS_CREDENTIALS, "add_credentials"),
        ("gemini", DependencyStatus.READY, "none"),
        ("oai", DependencyStatus.READY, "none"),
        ("oai-absent", DependencyStatus.MODEL_UNAVAILABLE, "bind"),
        ("oai-down", DependencyStatus.PROVIDER_UNAVAILABLE, "start_provider"),
        ("legacy", DependencyStatus.PROVIDER_UNAVAILABLE, "start_provider"),
        ("nope", DependencyStatus.MISSING_PROFILE, "bind"),
    ],
)
def test_profile_classification(
    profile: str, status: DependencyStatus, resolve: str
) -> None:
    assert _one(_profile(profile)) == (status, resolve)


def test_router_down_is_provider_unavailable_for_every_local_profile() -> None:
    providers = {**ROUTER_UP, "llama-server": {"available": False, "reason": "refused"}}
    status, _ = _one(_profile("ready"), providers)
    assert status == DependencyStatus.PROVIDER_UNAVAILABLE


def test_inline_model_is_classified_like_its_profile_would_be() -> None:
    dep = Dependency("model", "qwen3-14b", ("top.a",), provider="llama-server")
    assert _one(dep) == (DependencyStatus.PULLABLE, "pull")


def test_scripts_ensembles_and_dispatch() -> None:
    reports = classify_dependencies(
        [
            Dependency("ensemble", "top", ()),
            Dependency("ensemble", "no-such", ("top.x",), found=False),
            Dependency("script", "scripts/gone.py", ("top.y",)),
            Dependency("script", "echo hi", ("top.z",)),
            Dependency("dispatch", "${x.target}", ("top.w",)),
        ],
        profiles=PROFILES,
        providers=ROUTER_UP,
        script_found=lambda ref: ref == "echo hi",  # review focus 4
    )
    assert [(r.status, r.resolve) for r in reports] == [
        (DependencyStatus.READY, "none"),
        (DependencyStatus.MISSING_ENSEMBLE, "ship"),
        (DependencyStatus.MISSING_SCRIPT, "ship"),
        (DependencyStatus.READY, "none"),
        (DependencyStatus.DYNAMIC, "none"),
    ]


def test_detail_names_the_observed_fact_for_pullable() -> None:
    [report] = classify_dependencies(
        [_profile("pull")], profiles=PROFILES, providers=ROUTER_UP, script_found=lambda _: True
    )
    assert "qwen3-14b" in report.detail
    assert "unsloth/Qwen3-14B-GGUF:Q4_K_M" in report.detail
    assert "/api/models/qwen3-14b/pull" in report.detail


def test_runnable_requires_every_dependency_ready_or_dynamic() -> None:
    reports = classify_dependencies(
        [_profile("ready"), Dependency("dispatch", "${a}", ("top.b",)), _profile("pull")],
        profiles=PROFILES, providers=ROUTER_UP, script_found=lambda _: True,
    )
    assert is_runnable(reports[:2]) is True
    assert is_runnable(reports) is False
```

- [ ] **Step 2: Run RED.**

```bash
uv run pytest tests/unit/services/handlers/test_preflight.py -q -p no:cacheprovider
```

Expected: `ModuleNotFoundError: llm_orc.services.handlers.preflight`.

- [ ] **Step 3: Implement `src/llm_orc/services/handlers/preflight.py`.**

```python
"""Preflight: classify every dependency of a closure against what this
host has (spec: docs/plans/2026-09-28-remote-delegation.md, Arc 3
re-cut rulings 2, 3 and 6).

Model presence for llama-server comes from the router's one listing
(``LlamaServerClient.inventory()``): a model the router lists is
routable; a source in its cache is downloaded. Nothing else is
consulted, so preflight cannot disagree with the router that will serve
the request.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from llm_orc.core.config.closure import Dependency

_LLAMA_SERVER = "llama-server"
_OPENAI_COMPATIBLE = "openai-compatible"
_DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


class DependencyStatus(StrEnum):
    """The closed set. Adding one is a spec change, not a code change."""

    READY = "ready"
    PULLABLE = "pullable"
    NEEDS_RESTART = "needs_restart"
    MISSING_PROFILE = "missing_profile"
    MISSING_MODEL_SOURCE = "missing_model_source"
    MODEL_UNAVAILABLE = "model_unavailable"
    NEEDS_CREDENTIALS = "needs_credentials"
    MISSING_SCRIPT = "missing_script"
    MISSING_ENSEMBLE = "missing_ensemble"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    DYNAMIC = "dynamic"


RESOLVE: dict[DependencyStatus, str] = {
    DependencyStatus.READY: "none",
    DependencyStatus.DYNAMIC: "none",
    DependencyStatus.PULLABLE: "pull",
    DependencyStatus.NEEDS_RESTART: "restart",
    DependencyStatus.MISSING_PROFILE: "bind",
    DependencyStatus.MODEL_UNAVAILABLE: "bind",
    DependencyStatus.MISSING_MODEL_SOURCE: "add_source",
    DependencyStatus.NEEDS_CREDENTIALS: "add_credentials",
    DependencyStatus.MISSING_SCRIPT: "ship",
    DependencyStatus.MISSING_ENSEMBLE: "ship",
    DependencyStatus.PROVIDER_UNAVAILABLE: "start_provider",
}

UNBLOCKING = frozenset({DependencyStatus.READY, DependencyStatus.DYNAMIC})


class DependencyReport(BaseModel):
    """One dependency, its status, its hint, and the fact behind it."""

    kind: str
    name: str
    via: list[str] = Field(default_factory=list)
    status: DependencyStatus
    resolve: str
    detail: str
    provider: str | None = None


Verdict = tuple[DependencyStatus, str]


def classify_dependencies(
    dependencies: Sequence[Dependency],
    *,
    profiles: Mapping[str, Mapping[str, Any]],
    providers: Mapping[str, Any],
    script_found: Callable[[str], bool],
) -> list[DependencyReport]:
    """One report per dependency, in the closure's order."""
    reports: list[DependencyReport] = []
    for dep in dependencies:
        status, detail = _classify(dep, profiles, providers, script_found)
        reports.append(
            DependencyReport(
                kind=dep.kind,
                name=dep.name,
                via=list(dep.via),
                status=status,
                resolve=RESOLVE[status],
                detail=detail,
                provider=dep.provider,
            )
        )
    return reports


def is_runnable(reports: Sequence[DependencyReport]) -> bool:
    """Every dependency ready, or dynamic (decided at run time)."""
    return all(r.status in UNBLOCKING for r in reports)


def _classify(
    dep: Dependency,
    profiles: Mapping[str, Mapping[str, Any]],
    providers: Mapping[str, Any],
    script_found: Callable[[str], bool],
) -> Verdict:
    if dep.kind == "ensemble":
        if dep.found:
            return DependencyStatus.READY, "ensemble resolved"
        return DependencyStatus.MISSING_ENSEMBLE, f"no ensemble named {dep.name!r} in any tier"
    if dep.kind == "dispatch":
        return DependencyStatus.DYNAMIC, "dispatch target resolved at run time"
    if dep.kind == "script":
        if script_found(dep.name):
            return DependencyStatus.READY, "script resolved"
        return DependencyStatus.MISSING_SCRIPT, f"script {dep.name!r} not found on the search path"
    if dep.kind == "model":
        return _classify_model({"provider": dep.provider, "model": dep.name}, providers)
    profile = profiles.get(dep.name)
    if profile is None:
        return DependencyStatus.MISSING_PROFILE, f"no profile named {dep.name!r}"
    return _classify_model(profile, providers)


def _classify_model(profile: Mapping[str, Any], providers: Mapping[str, Any]) -> Verdict:
    provider = str(profile.get("provider") or "")
    model = str(profile.get("model") or "")
    if provider == _LLAMA_SERVER:
        return _classify_llama_server(model, profile.get("hf_repo"), providers)
    if provider == _OPENAI_COMPATIBLE or provider.startswith(_OPENAI_COMPATIBLE + "/"):
        return _classify_openai_compatible(model, profile, providers)
    info = providers.get(provider)
    if info is None:
        return DependencyStatus.PROVIDER_UNAVAILABLE, f"unknown provider {provider!r}"
    if info.get("available"):
        return DependencyStatus.READY, f"{provider} configured"
    return DependencyStatus.NEEDS_CREDENTIALS, f"{provider} has no credentials on this host"


def _classify_llama_server(
    model: str, hf_repo: Any, providers: Mapping[str, Any]
) -> Verdict:
    """Ruling 3, from the router's one listing."""
    info = providers.get(_LLAMA_SERVER, {})
    if not info.get("available"):
        return DependencyStatus.PROVIDER_UNAVAILABLE, str(info.get("reason") or "router unreachable")
    listed = model in info.get("models", [])
    source = str(hf_repo) if hf_repo else None
    if listed and source is None:
        return DependencyStatus.READY, f"model {model!r} listed by the router (no source to check)"
    if listed and source in info.get("cached", []):
        return DependencyStatus.READY, f"model {model!r} listed; {source} cached"
    if listed:
        return (
            DependencyStatus.PULLABLE,
            f"model {model!r} listed; {source} not cached (POST /api/models/{model}/pull)",
        )
    if source is not None:
        return (
            DependencyStatus.NEEDS_RESTART,
            f"model {model!r} not listed; the router scanned its preset at start",
        )
    return DependencyStatus.MISSING_MODEL_SOURCE, f"model {model!r} not listed and the profile has no hf_repo"


def _classify_openai_compatible(
    model: str, profile: Mapping[str, Any], providers: Mapping[str, Any]
) -> Verdict:
    base_url = str(profile.get("base_url") or _DEFAULT_OPENAI_BASE_URL)
    for endpoint in providers.get(_OPENAI_COMPATIBLE, {}).get("endpoints", []):
        if endpoint.get("base_url") != base_url:
            continue
        if not endpoint.get("available"):
            return DependencyStatus.PROVIDER_UNAVAILABLE, f"{base_url} unreachable"
        if model in endpoint.get("models", []):
            return DependencyStatus.READY, f"model {model!r} listed at {base_url}"
        return DependencyStatus.MODEL_UNAVAILABLE, f"model {model!r} not listed at {base_url}"
    return DependencyStatus.PROVIDER_UNAVAILABLE, f"no endpoint status for {base_url}"
```

The `_classify` function has one branch per kind; if complexipy scores
it above 15, split the profile/model tail into `_classify_named_model`.

- [ ] **Step 4: Run GREEN; mutant M2.** Mutant: in
  `_classify_llama_server`, change `if listed and source in
  info.get("cached", [])` to `if listed`. Expected red line: the
  parametrized case `("pull", DependencyStatus.PULLABLE, "pull")` fails
  with `ready` observed. Revert. `make lint`.

- [ ] **Step 5: Commit.**

```bash
git add src/llm_orc/services/handlers/preflight.py tests/unit/services/handlers/test_preflight.py
git commit -m "feat: preflight classifier maps every dependency to a status and a resolve hint"
```

---

## Task 4: `check_ensemble_runnable` walks the closure

**Files:**
- Modify: `src/llm_orc/providers/status_types.py:8-14` (`AgentStatus`)
- Modify: `src/llm_orc/services/handlers/provider_handler.py:31-38, 161-231` (constructor, `check_ensemble_runnable`)
- Modify: `src/llm_orc/services/orchestra_service.py:84-86` (construction)
- Test: `tests/unit/web/test_api_runnable_preflight.py` (create),
  `tests/unit/services/handlers/test_provider_handler.py` (existing
  four non-LLM tests must still pass unchanged)

**Interfaces:**
- Consumes: `walk_closure`/`Closure` (Task 2), `classify_dependencies`,
  `is_runnable`, `DependencyStatus`, `UNBLOCKING` (Task 3), the
  provider status with `cached` (Task 1), `ScriptResolver` and
  `ScriptNotFoundError` (`llm_orc.core.execution.scripting.resolver`).
- Produces: `ProviderHandler.__init__(profile_handler, find_ensemble,
  script_resolver_factory: Callable[[], ScriptResolver] | None = None)`;
  the result of `check_ensemble_runnable` gains `dependencies:
  list[dict]` (each a `DependencyReport.model_dump()`);
  `AgentStatus.DEPENDENCY_UNMET = "dependency_unmet"`.

- [ ] **Step 1: Write the failing REST pins.** Real service, real
  files, router faked at the listing.

```python
"""Transitive preflight over REST, through the real OrchestraService.

One closure with one dependency of every status; the report is asserted
whole, so a classifier that forgets a kind, a walker that stops at the
root, or a handler that drops `dependencies` fails here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient

import llm_orc.web.api as web_api
from llm_orc.providers.llama_server import LlamaServerClient
from llm_orc.web.server import create_app

LISTING: list[dict[str, Any]] = [
    {"id": "qwen3-8b", "status": {"value": "unloaded"}},
    {"id": "qwen3-14b", "status": {"value": "unloaded"}},
    {"id": "default", "status": {"value": "unloaded"}},
    {"id": "unsloth/Qwen3-8B-GGUF:Q4_K_M", "status": {"value": "unloaded"}},
]

PROFILES: dict[str, dict[str, Any]] = {
    "ready-prof": {"provider": "llama-server", "model": "qwen3-8b", "hf_repo": "unsloth/Qwen3-8B-GGUF:Q4_K_M"},
    "pull-prof": {"provider": "llama-server", "model": "qwen3-14b", "hf_repo": "unsloth/Qwen3-14B-GGUF:Q4_K_M"},
    "new-prof": {"provider": "llama-server", "model": "qwen3-4b", "hf_repo": "unsloth/Qwen3-4B-GGUF:Q4_K_M"},
    "nosrc-prof": {"provider": "llama-server", "model": "mystery"},
    "claude-prof": {"provider": "anthropic-api", "model": "claude-x"},
}


def _ensemble(dir_path: Path, name: str, agents: list[dict[str, Any]]) -> None:
    (dir_path / f"{name}.yaml").write_text(
        yaml.safe_dump({"name": name, "description": name, "agents": agents})
    )


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    dot = tmp_path / ".llm-orc"
    (dot / "ensembles").mkdir(parents=True)
    (dot / "scripts").mkdir()
    (dot / "config.yaml").write_text(yaml.safe_dump({"model_profiles": PROFILES}))
    (dot / "scripts" / "here.py").write_text("print('hi')\n")
    _ensemble(dot / "ensembles", "child", [
        {"name": "runner", "script": "scripts/gone.py"},
        {"name": "keeper", "script": "scripts/here.py"},
    ])
    _ensemble(dot / "ensembles", "top", [
        {"name": "writer", "model_profile": "ready-prof"},
        {"name": "big", "model_profile": "pull-prof"},
        {"name": "fresh", "model_profile": "new-prof"},
        {"name": "ghost", "model_profile": "nope"},
        {"name": "sourceless", "model_profile": "nosrc-prof"},
        {"name": "cloud", "model_profile": "claude-prof"},
        {"name": "sub", "ensemble": "child"},
        {"name": "again", "ensemble": "child"},
        {"name": "absent", "ensemble": "no-such"},
        {"name": "router", "dispatch": "${sub.target}"},
    ])
    _ensemble(dot / "ensembles", "clean", [
        {"name": "writer", "model_profile": "ready-prof"},
        {"name": "keeper", "script": "scripts/here.py"},
    ])
    _ensemble(dot / "ensembles", "only-pull", [
        {"name": "big", "model_profile": "pull-prof"},
    ])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(web_api, "_orchestra_service", None)
    monkeypatch.setattr(LlamaServerClient, "_list", lambda self: LISTING)
    return tmp_path


def _runnable(name: str) -> dict[str, Any]:
    with TestClient(create_app()) as client:
        response = client.get(f"/api/ensembles/{name}/runnable")
    assert response.status_code == 200, response.text
    return response.json()  # type: ignore[no-any-return]


class TestPreflightOverRest:
    def test_one_of_each_status_with_its_path_and_hint(self, project: Path) -> None:
        data = _runnable("top")

        assert data["runnable"] is False
        rows = [(d["kind"], d["name"], d["status"], d["resolve"], d["via"]) for d in data["dependencies"]]
        assert rows == [
            ("ensemble", "top", "ready", "none", []),
            ("profile", "ready-prof", "ready", "none", ["top.writer"]),
            ("profile", "pull-prof", "pullable", "pull", ["top.big"]),
            ("profile", "new-prof", "needs_restart", "restart", ["top.fresh"]),
            ("profile", "nope", "missing_profile", "bind", ["top.ghost"]),
            ("profile", "nosrc-prof", "missing_model_source", "add_source", ["top.sourceless"]),
            ("profile", "claude-prof", "needs_credentials", "add_credentials", ["top.cloud"]),
            ("ensemble", "child", "ready", "none", ["top.sub"]),
            ("script", "scripts/gone.py", "missing_script", "ship", ["top.sub", "child.runner"]),
            ("script", "scripts/here.py", "ready", "none", ["top.sub", "child.keeper"]),
            ("ensemble", "no-such", "missing_ensemble", "ship", ["top.absent"]),
            ("dispatch", "${sub.target}", "dynamic", "none", ["top.router"]),
        ]

    def test_agents_keep_their_coarse_status(self, project: Path) -> None:
        """Ruling 6: the web UI's view stays truthful, including for the
        second agent that names an already-reported child (review focus 1)."""
        data = _runnable("top")

        assert {a["name"]: a["status"] for a in data["agents"]} == {
            "writer": "available",
            "big": "model_unavailable",
            "fresh": "model_unavailable",
            "ghost": "missing_profile",
            "sourceless": "model_unavailable",
            "cloud": "provider_unavailable",
            "sub": "dependency_unmet",
            "again": "dependency_unmet",
            "absent": "dependency_unmet",
            "router": "available",
        }

    def test_clean_closure_is_runnable(self, project: Path) -> None:
        data = _runnable("clean")
        assert data["runnable"] is True
        assert {d["status"] for d in data["dependencies"]} == {"ready"}

    def test_pullable_alone_makes_it_not_runnable(self, project: Path) -> None:
        """Ruling 6: the caller decides about a multi-gigabyte download."""
        data = _runnable("only-pull")
        assert data["runnable"] is False
        assert [d["status"] for d in data["dependencies"]] == ["ready", "pullable"]
```

The `anthropic-api` row relies on the autouse global-config isolation in
`tests/conftest.py`: no credentials file, so `_get_cloud_provider_status`
reports `not configured`. No `openai-compatible` profile is in the
fixture, so no HTTP is attempted.

- [ ] **Step 2: Run RED.**

```bash
uv run pytest tests/unit/web/test_api_runnable_preflight.py -q -p no:cacheprovider
```

Expected: `KeyError: 'dependencies'` on the first three; the agents test
fails on `sub` being `available`.

- [ ] **Step 3: Implement.** `status_types.py`:

```python
class AgentStatus(StrEnum):
    """Status of an agent's runnability (coarse; the dependency report
    carries the fine-grained status, spec Arc 3 re-cut ruling 6)."""

    AVAILABLE = "available"
    MISSING_PROFILE = "missing_profile"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    MODEL_UNAVAILABLE = "model_unavailable"
    DEPENDENCY_UNMET = "dependency_unmet"
```

`provider_handler.py`: new imports

```python
from llm_orc.core.config.closure import Closure, Key, walk_closure
from llm_orc.core.execution.scripting.resolver import (
    ScriptNotFoundError,
    ScriptResolver,
)
from llm_orc.services.handlers.preflight import (
    UNBLOCKING,
    DependencyReport,
    DependencyStatus,
    classify_dependencies,
    is_runnable,
)
```

the constructor:

```python
    def __init__(
        self,
        profile_handler: ProfileHandler,
        find_ensemble: Callable[[str], EnsembleConfig | None],
        script_resolver_factory: Callable[[], ScriptResolver] | None = None,
    ) -> None:
        """Initialize with profile handler, ensemble finder, and the
        resolver the executor would use for scripts (ruling 5)."""
        self._profile_handler = profile_handler
        self._find_ensemble = find_ensemble
        self._script_resolver_factory = script_resolver_factory or ScriptResolver
```

and `check_ensemble_runnable` rewritten (replace the whole method body
after the two `ValueError` guards, keep the guards verbatim):

```python
        provider_status = await self.get_provider_status({})
        providers = provider_status.get("providers", {})
        all_profiles = self._profile_handler.get_all_profiles()

        closure = walk_closure(config, self._find_ensemble, all_profiles)
        reports = classify_dependencies(
            closure.dependencies,
            profiles=all_profiles,
            providers=providers,
            script_found=self._script_found,
        )
        by_key: dict[Key, DependencyReport] = {(r.kind, r.name): r for r in reports}

        agent_results = [
            self._agent_view(config.name, agent, closure, by_key, providers)
            for agent in config.agents
        ]
        result = EnsembleRunnability(
            ensemble=ensemble_name,
            runnable=is_runnable(reports),
            agents=agent_results,
        ).model_dump()
        result["dependencies"] = [r.model_dump() for r in reports]
        return result

    def _script_found(self, script_ref: str) -> bool:
        """The executor's own resolution (ruling 5): a bare name is
        inline content and resolves; only a path-syntax or absolute
        reference can be missing."""
        try:
            self._script_resolver_factory().resolve_and_classify(script_ref)
        except ScriptNotFoundError:
            return False
        return True

    def _agent_view(
        self,
        root_name: str,
        agent: Any,
        closure: Closure,
        by_key: dict[Key, DependencyReport],
        providers: dict[str, Any],
    ) -> AgentRunnability:
        """The coarse per-agent status the web UI reads, derived from the
        agent's own dependencies (ruling 6)."""
        agent_name = _get_agent_attr(agent, "name", "unknown")
        profile_name = _get_agent_attr(agent, "model_profile", None)
        result = AgentRunnability(
            name=agent_name,
            profile=profile_name if isinstance(profile_name, str) else "",
            provider=_agent_provider(agent, by_key),
        )
        owned = closure.owned.get(f"{root_name}.{agent_name}", frozenset())
        unmet = [by_key[k] for k in _in_closure_order(owned, by_key) if by_key[k].status not in UNBLOCKING]
        if not unmet:
            return result
        result.status = _COARSE[unmet[0].status]
        if result.status is AgentStatus.MODEL_UNAVAILABLE:
            result.alternatives = self._suggest_available_models(
                providers.get("llama-server", {}).get("models", [])
            )
        else:
            result.alternatives = self._suggest_local_alternatives(providers)
        return result
```

module-level helpers, after `_is_openai_compatible`:

```python
_COARSE: dict[DependencyStatus, AgentStatus] = {
    DependencyStatus.MISSING_PROFILE: AgentStatus.MISSING_PROFILE,
    DependencyStatus.PROVIDER_UNAVAILABLE: AgentStatus.PROVIDER_UNAVAILABLE,
    DependencyStatus.NEEDS_CREDENTIALS: AgentStatus.PROVIDER_UNAVAILABLE,
    DependencyStatus.PULLABLE: AgentStatus.MODEL_UNAVAILABLE,
    DependencyStatus.NEEDS_RESTART: AgentStatus.MODEL_UNAVAILABLE,
    DependencyStatus.MISSING_MODEL_SOURCE: AgentStatus.MODEL_UNAVAILABLE,
    DependencyStatus.MODEL_UNAVAILABLE: AgentStatus.MODEL_UNAVAILABLE,
    DependencyStatus.MISSING_SCRIPT: AgentStatus.DEPENDENCY_UNMET,
    DependencyStatus.MISSING_ENSEMBLE: AgentStatus.DEPENDENCY_UNMET,
}


def _in_closure_order(
    keys: frozenset[Key], by_key: dict[Key, DependencyReport]
) -> list[Key]:
    """Deterministic: the closure's own order, so the first unmet
    dependency is the same one on every call."""
    order = {k: i for i, k in enumerate(by_key)}
    return sorted(keys, key=lambda k: order[k])


def _agent_provider(agent: Any, by_key: dict[Key, DependencyReport]) -> str:
    """The label the old result carried: 'script' / 'ensemble' / 'loop'
    / 'dispatch' for non-LLM agents, the profile's provider otherwise."""
    for label in ("script", "ensemble", "loop", "dispatch"):
        if _get_agent_attr(agent, label) is not None and _get_agent_attr(agent, label) != "":
            return label
    profile_name = _get_agent_attr(agent, "model_profile", None)
    if isinstance(profile_name, str):
        report = by_key.get(("profile", profile_name))
        if report is not None and report.provider:
            return report.provider
    provider = _get_agent_attr(agent, "provider", None)
    return provider if isinstance(provider, str) else ""
```

Delete `_check_agent_runnable`, `_check_llama_server_model` and
`_check_openai_compat_model` (the classifier replaced them). Keep
`_suggest_local_alternatives` and `_suggest_available_models`. Check
`vulture` in `make lint` for anything else now unreferenced.

`orchestra_service.py:84`:

```python
        self._provider_handler = ProviderHandler(
            self._profile_handler,
            self.find_ensemble_by_name,
            lambda: ScriptResolver(project_dir=self._project_path),
        )
```

with `from llm_orc.core.execution.scripting.resolver import ScriptResolver`
added to the imports. `_project_path` is what `_get_executor` hands
`ExecutorFactory.create_root_executor`, so preflight and run resolve
scripts from the same base.

- [ ] **Step 4: Run GREEN, the old tests, and mutants M3 and M4.**

```bash
uv run pytest tests/unit/web/test_api_runnable_preflight.py tests/unit/services/handlers/test_provider_handler.py tests/unit/mcp_server/test_server.py -q -p no:cacheprovider
```

The four existing non-LLM tests in `test_provider_handler.py` still
pass without edits: `_make_handler`'s finder returns the same
`MagicMock` config for every name, the walker's visited set stops the
recursion, and a `MagicMock` agent matches no typed class. If any of
the LLM-agent tests in that file constructed profiles with fields the
classifier reads differently, fix the classifier, not the test, and say
so in the commit body.

Mutant M3 (walker, again through REST): delete `self.visit(child, via)`
in `closure.py`. Expected red line: the first pin's `assert rows == [...]`
(the two `child.*` rows vanish). Mutant M4 (runnable): change
`UNBLOCKING` to include `DependencyStatus.PULLABLE`. Expected red line:
`assert data["runnable"] is False` in the pullable-alone pin. Revert
both. `make lint`. Full suite:

```bash
uv run pytest -q -p no:cacheprovider 2>&1 | tail -3
```

- [ ] **Step 5: Commit.**

```bash
git add src/llm_orc/providers/status_types.py src/llm_orc/services/handlers/provider_handler.py src/llm_orc/services/orchestra_service.py tests/unit/web/test_api_runnable_preflight.py
git commit -m "feat: check_ensemble_runnable preflights the whole closure with a dependency report"
```

Body: M3 and M4 with their red lines; the suite count.

---

## Task 5: docs

**Files:**
- Modify: `docs/serving.md` (new section after "Layers and state")
- Modify: `docs/cli-reference.md:172` (the MCP tool table row)
- Modify: `docs/architecture.md:206`
- Modify: `src/llm_orc/mcp/server.py:665-680` and `src/llm_orc/web/api/ensembles.py:84-95` (docstrings)
- Modify: `CHANGELOG.md` (Unreleased / Added)

- [ ] **Step 1: `docs/serving.md`.** After the "Layers and state"
  section, add:

```markdown
## Preflight: what a remote is missing

`GET /api/ensembles/{name}/runnable` (MCP: `check_ensemble_runnable`)
walks the ensemble's closure: child ensembles (`ensemble:`, `loop.body`,
a literal `dispatch:`), scripts, profiles and their
`fallback_model_profile` chains, inline models. Each dependency gets one
of eleven statuses and a resolve hint:

| status | meaning | resolve |
|---|---|---|
| `ready` | present; for a local model, listed by the router and its `hf_repo` in the router's cache | `none` |
| `dynamic` | a `${...}` dispatch target, decided at run time | `none` |
| `pullable` | listed by the router, GGUF not downloaded: `POST /api/models/{model}/pull` | `pull` |
| `needs_restart` | profile has a source, the router has not scanned it (it reads the preset at start; a supervised restart costs 1.5 to 2.2 s and cuts in-flight completions) | `restart` |
| `missing_profile` | no profile of that name in any tier | `bind` |
| `model_unavailable` | an OpenAI-compatible endpoint does not list the model | `bind` |
| `missing_model_source` | not listed and the profile has no `hf_repo` | `add_source` |
| `needs_credentials` | a cloud provider with no credentials on this host | `add_credentials` |
| `missing_script` | a path-syntax script reference nothing on the search path resolves | `ship` |
| `missing_ensemble` | a child ensemble no tier has | `ship` |
| `provider_unavailable` | router or endpoint unreachable, or a provider llm-orc does not know | `start_provider` |

`runnable` is true only when every dependency is `ready` or `dynamic`; a
`pullable` model is the caller's decision, never an implicit download.
Each entry carries `via`, the `ensemble.agent` frames from the root, so
a child's missing script is reported at the top with the path to it.
Model presence comes from the router's own listing and nothing else: a
preset section is routable, a cache entry (its id is the `hf_repo`
string) is downloaded. The `agents` list keeps the coarse per-agent
status the web UI reads.
```

- [ ] **Step 2: the two docstrings, the two doc tables, the changelog.**
  MCP tool docstring "Returns runnable status with:" gains a line
  `- dependencies: every child ensemble, script, profile and model in the
  closure with status and resolve hint (docs/serving.md, Preflight)`.
  Same line in the REST docstring. `docs/cli-reference.md:172` becomes
  `| check_ensemble_runnable | Preflight the ensemble's whole closure: per-dependency status and resolve hint |`;
  `docs/architecture.md:206` the same wording. CHANGELOG under
  `## [Unreleased]` / `### Added`:

```markdown
- `check_ensemble_runnable` (REST `GET /api/ensembles/{name}/runnable`)
  preflights the whole closure: child ensembles, loop bodies, dispatch
  targets, scripts, profiles with their fallback chains and inline
  models, each reported with one of eleven statuses (`ready`, `pullable`,
  `needs_restart`, `missing_profile`, `missing_model_source`,
  `model_unavailable`, `needs_credentials`, `missing_script`,
  `missing_ensemble`, `provider_unavailable`, `dynamic`), a resolve hint
  and the agent path that names it. `runnable` now requires every
  dependency ready or dynamic; a downloadable model is `pullable`, not
  runnable. The llama-server provider status carries `cached` (the
  router's downloaded sources). Agent status gains `dependency_unmet`.
```

- [ ] **Step 3: Lint and commit.**

```bash
make lint
git add docs/serving.md docs/cli-reference.md docs/architecture.md CHANGELOG.md src/llm_orc/mcp/server.py src/llm_orc/web/api/ensembles.py
git commit -m "docs: preflight statuses, hints and the dependency report"
```

---

## Task 6: live row and review gate

**Files:**
- Modify: `docs/plans/2026-09-28-remote-delegation.md` (append "Arc 3
  live row (2026-09-29)" under the Arc 3 re-cut), GitHub #196 (comment).

- [ ] **Step 1: Serve from a plain directory on the laptop, real router.**
  The laptop's Hugging Face cache holds `Qwen3-0.6B`, `Qwen3-8B` and
  `nomic-embed-text` (`llama-server --cache-list`). From the worktree:

```bash
LIVE=$(mktemp -d); mkdir -p $LIVE/proj/.llm-orc/ensembles $LIVE/config $LIVE/state
cat > $LIVE/proj/.llm-orc/ensembles/probe.yaml <<'EOF'
name: probe
description: preflight probe
agents:
  - name: small
    model_profile: local-qwen3-1.7b
  - name: fresh
    model_profile: probe-qwen3-4b
  - name: ghost
    model_profile: no-such-profile
  - name: search
    ensemble: agentic-serving/web-searcher
  - name: gone
    script: scripts/not_here.py
EOF
cat > $LIVE/proj/.llm-orc/config.yaml <<'EOF'
model_profiles:
  probe-qwen3-4b:
    provider: llama-server
    model: qwen3-4b
    hf_repo: unsloth/Qwen3-4B-GGUF:Q4_K_M
EOF
cd $LIVE/proj
XDG_CONFIG_HOME=$LIVE/config XDG_STATE_HOME=$LIVE/state \
  uv run --project <worktree> llm-orc serve --port 8766 --backend-port 8790 --models-max 1 > $LIVE/serve.log 2>&1 &
sleep 20; tail -3 $LIVE/serve.log
curl -s localhost:8766/api/models | python3 -m json.tool | head -40
curl -s localhost:8766/api/ensembles/probe/runnable | python3 -c '
import json,sys; d=json.load(sys.stdin); print("runnable", d["runnable"])
for x in d["dependencies"]: print(x["status"].ljust(20), x["kind"].ljust(9), x["name"], x["via"], "->", x["resolve"])'
```

Expected on the first call: `local-qwen3-1.7b` → `pullable` (listed,
not cached); `probe-qwen3-4b` → `ready` or `pullable`, NOT
`needs_restart` (the serve rendered the project profile into the preset
at start; this is the point of ruling 3: `needs_restart` is for a
profile added after start); `no-such-profile` → `missing_profile`;
`agentic-serving/web-searcher` → `ready` (packaged tier) with its own
scripts and profiles listed under `probe.search`; `scripts/not_here.py`
→ `missing_script`; `runnable false`. Also
`GET /api/ensembles/research-dossier/runnable` if the ensemble is in the
global tier (copy `.llm-orc/ensembles/research-dossier.yaml` from the
main checkout into `$LIVE/config/llm-orc/ensembles/` first): expect the
packaged child and `agentic-tier-cheap-general` → `ready`.

- [ ] **Step 2: `needs_restart`, then `pullable` → `ready`.** With the
  serve still up, add a profile after start and re-check; then restart
  and re-check; then pull the ~1 GB `qwen3-1.7b` and re-check.

```bash
cat >> $LIVE/proj/.llm-orc/config.yaml <<'EOF'
  late-qwen3-1.7b-alias:
    provider: llama-server
    model: qwen3-1.7b-late
    hf_repo: unsloth/Qwen3-1.7B-GGUF:Q4_K_M
EOF
cat > $LIVE/proj/.llm-orc/ensembles/late.yaml <<'EOF'
name: late
description: profile added after the serve started
agents:
  - name: w
    model_profile: late-qwen3-1.7b-alias
EOF
curl -s localhost:8766/api/ensembles/late/runnable | python3 -c 'import json,sys; d=json.load(sys.stdin); print([(x["name"],x["status"],x["resolve"]) for x in d["dependencies"]])'
kill %1; sleep 3
XDG_CONFIG_HOME=$LIVE/config XDG_STATE_HOME=$LIVE/state \
  uv run --project <worktree> llm-orc serve --port 8766 --backend-port 8790 --models-max 1 > $LIVE/serve2.log 2>&1 &
sleep 20
curl -s localhost:8766/api/ensembles/late/runnable | python3 -c 'import json,sys; d=json.load(sys.stdin); print([(x["name"],x["status"],x["resolve"]) for x in d["dependencies"]])'
time curl -s -X POST localhost:8766/api/models/qwen3-1.7b/pull
curl -s localhost:8766/api/ensembles/probe/runnable | python3 -c 'import json,sys; d=json.load(sys.stdin); print([(x["name"],x["status"]) for x in d["dependencies"] if x["kind"]=="profile"])'
kill %1
```

Expected: `late-qwen3-1.7b-alias` → `needs_restart` before the restart,
`pullable` after (the model is now in the preset, its source is the
1.7b GGUF, not yet cached); after the pull `local-qwen3-1.7b` → `ready`
in `probe`. Record the pull's wall clock.

- [ ] **Step 3: Record.** Append the row (commands, observed output per
  step, llama-server version, timings) under the Arc 3 re-cut in the
  spec doc. Commit:

```bash
git add docs/plans/2026-09-28-remote-delegation.md
git commit -m "docs: Arc 3 live row"
```

- [ ] **Step 4: Review gate.** Independent adversarial review (Opus, not
  the implementer) with an explicit wrong-accept hunt: for each pin in
  Tasks 1-4, can it pass while the defect it names is live? Particular
  targets: the `owned` frames on a second sighting of a child (review
  focus 1); the coarse `agents` mapping when a profile is first seen
  under a different agent; `_script_found` swallowing anything wider than
  `ScriptNotFoundError`; the fixture's `anthropic-api` row depending on
  the global-config isolation (would it pass on a laptop with
  credentials? it must, because the autouse fixture isolates it; verify
  the fixture is autouse in that test path); the cache id shape against
  the mini's router build when the practitioner deploys. Merge to local
  `main` only on APPROVE. Nothing is pushed.

- [ ] **Step 5: Comment on #196 and #191.** The summary, the live row's
  numbers, the two pre-existing items from Review Focus, and the Arc 4
  entry point (the `not_equipped` error carries `dependencies`).

---

## Self-review notes (lead, 2026-09-29)

Spec coverage: card bullets 1 (walk + classify + hint) → Tasks 2-4;
bullet 2 (router presence, `pullable` needs `hf_repo`, S2's
`needs_restart`) → Tasks 1 and 3; bullet 3 (one-of-each fixture; child's
missing script at the top) → Task 4 pin 1. Rulings 1-7 each have a pin;
ruling 8 is scope. Types: `Key`, `Closure.owned`, `DependencyReport`,
`UNBLOCKING`, `is_runnable`, `_list`, `inventory`, `cached` are spelled
the same in every task. Placeholders: none; the only "check whether"
lines concern the test directory layout and `EnsembleConfig`
construction, both resolved by reading the named file.
