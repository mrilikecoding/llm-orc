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

import pytest
import yaml

from llm_orc.core.config.config_manager import resolve_global_config_dir
from llm_orc.core.execution.executor_factory import ExecutorFactory
from llm_orc.providers.llama_server import LlamaServerClient
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
