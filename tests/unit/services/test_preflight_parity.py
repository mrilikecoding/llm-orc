"""Preflight resolves every dependency the way the run does.

Through the real OrchestraService on a temp project, with the global
tier at ``resolve_global_config_dir()`` (a per-test temp dir, conftest).
Profiles must come from the runtime merge the run and the router preset
read (packaged < global < local, library excluded); child ensembles
from the executor's own search dirs and by-filename finder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from llm_orc.core.config.config_manager import (
    ConfigurationManager,
    resolve_global_config_dir,
)
from llm_orc.core.config.ensemble_config import EnsembleConfig, EnsembleLoader
from llm_orc.core.execution.executor_factory import ExecutorFactory
from llm_orc.providers.llama_server import LlamaServerClient
from llm_orc.services.handlers.provider_handler import Preflight
from llm_orc.services.orchestra_service import OrchestraService

LISTING: list[dict[str, Any]] = [
    {"id": "qwen3-8b", "status": {"value": "unloaded"}},
    {"id": "unsloth/Qwen3-8B-GGUF:Q4_K_M", "status": {"value": "unloaded"}},
]


def _yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data))


def _ensemble(path: Path, name: str, agents: list[dict[str, Any]]) -> None:
    _yaml(path, {"name": name, "description": name, "agents": agents})


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A project dir, with the cwd somewhere else entirely."""
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path / "elsewhere")
    monkeypatch.setattr(LlamaServerClient, "_list", lambda self: LISTING)
    root = tmp_path / "proj"
    (root / ".llm-orc" / "ensembles").mkdir(parents=True)
    return root


def _service(project: Path) -> OrchestraService:
    service = OrchestraService()
    assert service.handle_set_project(str(project))["status"] == "ok"
    return service


async def _statuses(service: OrchestraService, name: str) -> dict[str, str]:
    result = await service.check_ensemble_runnable({"ensemble_name": name})
    return {d["name"]: d["status"] for d in result["dependencies"]}


class TestProfilesResolveLikeTheRun:
    async def test_global_profile_beats_the_packaged_one_of_the_same_name(
        self, project: Path, packaged_serving_project: Path
    ) -> None:
        # packaged: agentic-tier-cheap-general -> qwen3-8b (listed, ready).
        # global: the same name -> qwen3-4b, which the router does not list.
        _yaml(
            resolve_global_config_dir() / "profiles" / "cheap.yaml",
            {
                "name": "agentic-tier-cheap-general",
                "provider": "llama-server",
                "model": "qwen3-4b",
                "hf_repo": "unsloth/Qwen3-4B-GGUF:Q4_K_M",
            },
        )
        _ensemble(
            project / ".llm-orc" / "ensembles" / "top.yaml",
            "top",
            [{"name": "w", "model_profile": "agentic-tier-cheap-general"}],
        )
        service = _service(project)
        runtime = service.config_manager.get_model_profiles()
        assert runtime["agentic-tier-cheap-general"]["model"] == "qwen3-4b"

        statuses = await _statuses(service, "top")

        assert statuses["agentic-tier-cheap-general"] == "needs_restart"

    async def test_library_only_profile_is_missing(
        self, project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = tmp_path / "library"
        (library / "profiles").mkdir(parents=True)
        monkeypatch.setenv("LLM_ORC_LIBRARY_PATH", str(library))
        _ensemble(
            project / ".llm-orc" / "ensembles" / "top.yaml",
            "top",
            [{"name": "w", "model_profile": "lib-only"}],
        )
        service = _service(project)
        # Added to the library after the service provisioned the global
        # tier (provisioning copies library profiles once, at start).
        _yaml(
            library / "profiles" / "lib-only.yaml",
            {"name": "lib-only", "provider": "llama-server", "model": "qwen3-8b"},
        )
        assert "lib-only" not in service.config_manager.get_model_profiles()

        statuses = await _statuses(service, "top")

        assert statuses["lib-only"] == "missing_profile"


class TestChildrenResolveLikeTheExecutor:
    REFS = ("child", "alias", "renamed-file", "sub/child")

    @pytest.fixture
    def service(self, project: Path) -> OrchestraService:
        ensembles = project / ".llm-orc" / "ensembles"
        _ensemble(
            ensembles / "sub" / "child.yaml",
            "child",
            [{"name": "say", "script": "echo child"}],
        )
        _ensemble(
            ensembles / "renamed-file.yaml",
            "alias",
            [{"name": "say", "script": "echo alias"}],
        )
        _ensemble(
            ensembles / "kids.yaml",
            "kids",
            [{"name": f"k{i}", "ensemble": ref} for i, ref in enumerate(self.REFS)],
        )
        return _service(project)

    async def test_found_exactly_when_the_executor_resolves(
        self, service: OrchestraService
    ) -> None:
        executor = ExecutorFactory.create_root_executor(
            project_dir=service.project_path,
            config_manager=service.config_manager,
            save_artifacts=False,
        )
        result = await service.check_ensemble_runnable({"ensemble_name": "kids"})
        found = {
            d["name"]: d["status"] == "ready"
            for d in result["dependencies"]
            if d["kind"] == "ensemble" and d["name"] in self.REFS
        }

        resolves: dict[str, bool] = {}
        for ref in self.REFS:
            try:
                executor._resolve_ensemble_reference(ref)
            except FileNotFoundError:
                resolves[ref] = False
            else:
                resolves[ref] = True

        assert found == resolves
        assert resolves == {
            "child": False,
            "alias": False,
            "renamed-file": True,
            "sub/child": True,
        }

    async def test_missing_children_report_missing_ensemble(
        self, service: OrchestraService
    ) -> None:
        statuses = await _statuses(service, "kids")

        assert statuses["child"] == "missing_ensemble"
        assert statuses["alias"] == "missing_ensemble"
        assert statuses["renamed-file"] == "ready"
        assert statuses["sub/child"] == "ready"


class TestScriptsResolveFromTheProject:
    async def test_project_script_resolves_with_the_cwd_elsewhere(
        self, project: Path
    ) -> None:
        """Ruling 5: the resolver gets the service's project path, not the
        cwd."""
        dot = project / ".llm-orc"
        (dot / "scripts").mkdir()
        (dot / "scripts" / "x.py").write_text("print('x')\n")
        _ensemble(
            dot / "ensembles" / "s.yaml", "s", [{"name": "r", "script": "scripts/x.py"}]
        )
        service = _service(project)

        statuses = await _statuses(service, "s")

        assert statuses["scripts/x.py"] == "ready"


class TestTheGateOnARunLayerView:
    """The gate called with a view reports ready exactly when the real
    executor built on that view resolves the thing (Arc 4, ruling 10)."""

    @pytest.fixture
    def run_dir(self, tmp_path: Path) -> Path:
        path = tmp_path / "runs" / "one"
        path.mkdir(parents=True)
        return path

    async def _gate(
        self,
        service: OrchestraService,
        root: EnsembleConfig,
        manager: ConfigurationManager,
    ) -> Preflight:
        return await service._provider_handler.preflight(
            root,
            root.name,
            config_manager=manager,
            project_dir=service.project_path,
        )

    @staticmethod
    def _statuses_of(outcome: Preflight) -> dict[str, str]:
        return {r.name: r.status.value for r in outcome.reports}

    def _root(self, tmp_path: Path, agents: list[dict[str, Any]]) -> EnsembleConfig:
        _ensemble(tmp_path / "root.yaml", "root", agents)
        return EnsembleLoader().load_from_file(str(tmp_path / "root.yaml"))

    async def test_a_child_only_in_the_layer(
        self, project: Path, tmp_path: Path, run_dir: Path
    ) -> None:
        _ensemble(
            run_dir / "ensembles" / "layer-child.yaml",
            "layer-child",
            [{"name": "say", "script": "echo child"}],
        )
        root = self._root(tmp_path, [{"name": "kid", "ensemble": "layer-child"}])
        service = _service(project)
        base = service.config_manager
        view = base.with_run_layer(run_dir)

        gated_view = await self._gate(service, root, view)
        gated_base = await self._gate(service, root, base)

        executors = {
            label: ExecutorFactory.create_root_executor(
                project_dir=service.project_path,
                config_manager=manager,
                save_artifacts=False,
            )
            for label, manager in (("view", view), ("base", base))
        }
        executors["view"]._resolve_ensemble_reference("layer-child")
        with pytest.raises(FileNotFoundError):
            executors["base"]._resolve_ensemble_reference("layer-child")
        assert self._statuses_of(gated_view)["layer-child"] == "ready"
        assert self._statuses_of(gated_base)["layer-child"] == "missing_ensemble"

    async def test_a_script_only_in_the_layer(
        self, project: Path, tmp_path: Path, run_dir: Path
    ) -> None:
        script = run_dir / "scripts" / "a" / "x.py"
        script.parent.mkdir(parents=True)
        script.write_text(
            "import json, sys\n"
            "sys.stdin.read()\n"
            'print(json.dumps({"success": True, "data": {"ok": True}}))\n'
        )
        root = self._root(tmp_path, [{"name": "s", "script": "a/x.py"}])
        service = _service(project)
        base = service.config_manager
        view = base.with_run_layer(run_dir)

        gated_view = await self._gate(service, root, view)
        gated_base = await self._gate(service, root, base)

        ran = {}
        for label, manager in (("view", view), ("base", base)):
            executor = ExecutorFactory.create_root_executor(
                project_dir=service.project_path,
                config_manager=manager,
                save_artifacts=False,
            )
            result = await executor.execute(root, "ping")
            ran[label] = not result["has_errors"]
        assert ran == {"view": True, "base": False}
        assert self._statuses_of(gated_view)["a/x.py"] == "ready"
        assert self._statuses_of(gated_base)["a/x.py"] == "missing_script"

    async def test_a_profile_only_in_the_layer(
        self, project: Path, tmp_path: Path, run_dir: Path
    ) -> None:
        _yaml(
            run_dir / "profiles" / "layer-prof.yaml",
            {"name": "layer-prof", "provider": "llama-server", "model": "qwen3-8b"},
        )
        root = self._root(tmp_path, [{"name": "w", "model_profile": "layer-prof"}])
        service = _service(project)
        base = service.config_manager
        view = base.with_run_layer(run_dir)

        gated_view = await self._gate(service, root, view)
        gated_base = await self._gate(service, root, base)

        agent = root.agents[0]
        loaded = {}
        for label, manager in (("view", view), ("base", base)):
            executor = ExecutorFactory.create_root_executor(
                project_dir=service.project_path,
                config_manager=manager,
                save_artifacts=False,
            )
            merged = await executor._llm_agent_runner._resolve_model_profile_to_config(
                agent
            )
            loaded[label] = merged.get("model")
        assert loaded == {"view": "qwen3-8b", "base": None}
        assert self._statuses_of(gated_view)["layer-prof"] == "ready"
        assert self._statuses_of(gated_base)["layer-prof"] == "missing_profile"

    async def test_an_inline_openai_compatible_endpoint_is_probed(
        self,
        project: Path,
        tmp_path: Path,
        run_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        probed: list[str] = []

        class _Response:
            status_code = 200

            def json(self) -> dict[str, Any]:
                return {"data": [{"id": "their-model"}]}

        class _Client:
            def __init__(self, **_: Any) -> None:
                pass

            async def __aenter__(self) -> _Client:
                return self

            async def __aexit__(self, *_: Any) -> None:
                return None

            async def get(self, url: str) -> _Response:
                probed.append(url)
                return _Response()

        monkeypatch.setattr(httpx, "AsyncClient", _Client)
        _yaml(
            run_dir / "profiles" / "theirs.yaml",
            {
                "name": "theirs",
                "provider": "openai-compatible",
                "model": "their-model",
                "base_url": "http://inline.test/v1",
            },
        )
        root = self._root(tmp_path, [{"name": "w", "model_profile": "theirs"}])
        service = _service(project)
        base = service.config_manager

        gated_view = await self._gate(service, root, base.with_run_layer(run_dir))
        gated_base = await self._gate(service, root, base)

        endpoints = gated_view.providers["openai-compatible"]["endpoints"]
        assert [e["base_url"] for e in endpoints] == ["http://inline.test/v1"]
        assert probed == ["http://inline.test/v1/models"]
        assert self._statuses_of(gated_view)["theirs"] == "ready"
        assert self._statuses_of(gated_base)["theirs"] == "missing_profile"


class TestThePairIsOneThing:
    """A view with no project dir would resolve scripts against the cwd
    while the executor uses the service's project path: gate ready, run
    failed. Half a pair is an error, not a fallback."""

    async def _root(self, project: Path) -> EnsembleConfig:
        path = project / ".llm-orc" / "ensembles" / "top.yaml"
        _ensemble(path, "top", [{"name": "w", "model": "mock-a", "provider": "mock"}])
        return EnsembleLoader().load_from_file(str(path))

    async def test_a_manager_without_a_project_dir_is_refused(
        self, project: Path
    ) -> None:
        service = _service(project)
        root = await self._root(project)

        with pytest.raises(ValueError, match="both"):
            await service._provider_handler.preflight(
                root, "top", config_manager=service.config_manager
            )

    async def test_a_project_dir_without_a_manager_is_refused(
        self, project: Path
    ) -> None:
        service = _service(project)
        root = await self._root(project)

        with pytest.raises(ValueError, match="both"):
            await service._provider_handler.preflight(
                root, "top", project_dir=service.project_path
            )

    async def test_an_explicit_no_project_is_a_whole_pair(self, project: Path) -> None:
        service = _service(project)
        root = await self._root(project)

        outcome = await service._provider_handler.preflight(
            root, "top", config_manager=service.config_manager, project_dir=None
        )

        assert outcome.closure.dependencies
