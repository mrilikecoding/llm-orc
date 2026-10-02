"""A run layer is a directory shaped like every other tier (Arc 4).

Through the real executor with files on disk and ``mock-*`` models, so
nothing is called out. ``ConfigurationManager.with_run_layer`` returns a
view; the executor built on it resolves child ensembles, profiles and
scripts from the layer first, and the manager it was made from resolves
none of them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.core.config.ensemble_config import EnsembleConfig, EnsembleLoader
from llm_orc.core.execution.executor_factory import ExecutorFactory
from llm_orc.core.models.model_factory import ModelFactory


def _yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data))


def _ensemble(base: Path, name: str, agents: list[dict[str, Any]]) -> EnsembleConfig:
    path = base / f"{name}.yaml"
    _yaml(path, {"name": name, "description": name, "agents": agents})
    return EnsembleLoader().load_from_file(str(path))


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A project dir, with the cwd somewhere else entirely."""
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path / "elsewhere")
    root = tmp_path / "proj"
    (root / ".llm-orc").mkdir(parents=True)
    return root


@pytest.fixture
def base(project: Path) -> ConfigurationManager:
    return ConfigurationManager(project_dir=project, provision=False)


def _run_dir(tmp_path: Path, name: str = "run") -> Path:
    path = tmp_path / "runs" / name
    path.mkdir(parents=True)
    return path


async def _run(
    cm: ConfigurationManager, project: Path, root: EnsembleConfig
) -> dict[str, Any]:
    executor = ExecutorFactory.create_root_executor(
        project_dir=project, config_manager=cm, save_artifacts=False
    )
    return await executor.execute(root, "ping")


class TestChildEnsembleInTheLayer:
    async def test_a_child_present_only_in_the_layer_runs(
        self, tmp_path: Path, project: Path, base: ConfigurationManager
    ) -> None:
        run = _run_dir(tmp_path)
        _ensemble(
            run / "ensembles",
            "layer-child",
            [{"name": "w", "model": "mock-a", "provider": "mock"}],
        )
        root = _ensemble(tmp_path, "root", [{"name": "kid", "ensemble": "layer-child"}])

        result = await _run(base.with_run_layer(run), project, root)

        assert result["status"] == "completed", result
        assert not result["has_errors"], result

    async def test_the_layer_child_beats_a_project_child_of_the_same_name(
        self, tmp_path: Path, project: Path, base: ConfigurationManager
    ) -> None:
        run = _run_dir(tmp_path)
        _ensemble(
            run / "ensembles",
            "shared",
            [{"name": "from-layer", "model": "mock-a", "provider": "mock"}],
        )
        _ensemble(
            project / "ensembles",
            "shared",
            [{"name": "from-project", "model": "mock-a", "provider": "mock"}],
        )
        root = _ensemble(tmp_path, "root", [{"name": "kid", "ensemble": "shared"}])

        result = await _run(base.with_run_layer(run), project, root)

        child = json.loads(result["results"]["kid"]["response"])
        assert list(child["results"]) == ["from-layer"]


def _profile(base: Path, name: str, model: str) -> None:
    _yaml(
        base / "profiles" / f"{name}.yaml",
        {"name": name, "model": model, "provider": "mock"},
    )


@pytest.fixture
def loaded_models(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every model name the factory is asked for, in order (calls through)."""
    seen: list[str] = []
    original = ModelFactory.load_model

    async def spy(self: ModelFactory, model_name: str, *args: Any, **kw: Any) -> Any:
        seen.append(model_name)
        return await original(self, model_name, *args, **kw)

    monkeypatch.setattr(ModelFactory, "load_model", spy)
    return seen


class TestProfileInTheLayer:
    async def test_a_profile_present_only_in_the_layer_is_the_one_loaded(
        self,
        tmp_path: Path,
        project: Path,
        base: ConfigurationManager,
        loaded_models: list[str],
    ) -> None:
        run = _run_dir(tmp_path)
        _profile(run, "layer-prof", "mock-layer")
        root = _ensemble(
            tmp_path, "root", [{"name": "w", "model_profile": "layer-prof"}]
        )

        result = await _run(base.with_run_layer(run), project, root)

        assert result["status"] == "completed", result
        assert loaded_models == ["mock-layer"]

    async def test_a_layer_profile_shadows_a_global_profile_of_the_same_name(
        self,
        tmp_path: Path,
        project: Path,
        base: ConfigurationManager,
        loaded_models: list[str],
    ) -> None:
        _profile(base.global_config_dir, "shared-prof", "mock-global")
        run = _run_dir(tmp_path)
        _profile(run, "shared-prof", "mock-layer")
        root = _ensemble(
            tmp_path, "root", [{"name": "w", "model_profile": "shared-prof"}]
        )

        await _run(base.with_run_layer(run), project, root)
        await _run(base, project, root)

        assert loaded_models == ["mock-layer", "mock-global"]


class TestTheBaseManagerIsUntouched:
    async def test_it_resolves_none_of_what_the_view_resolves(
        self, tmp_path: Path, project: Path, base: ConfigurationManager
    ) -> None:
        run = _run_dir(tmp_path)
        _ensemble(
            run / "ensembles",
            "layer-child",
            [{"name": "w", "model": "mock-a", "provider": "mock"}],
        )
        _profile(run, "layer-prof", "mock-layer")
        view = base.with_run_layer(run)
        assert "layer-prof" in view.get_model_profiles()

        executor = ExecutorFactory.create_root_executor(
            project_dir=project, config_manager=base, save_artifacts=False
        )

        assert "layer-prof" not in base.get_model_profiles()
        assert base.run_layer_dir is None
        with pytest.raises(FileNotFoundError):
            executor._resolve_ensemble_reference("layer-child")

    async def test_two_views_resolve_their_own_content_under_one_name(
        self,
        tmp_path: Path,
        project: Path,
        base: ConfigurationManager,
        loaded_models: list[str],
    ) -> None:
        runs = [_run_dir(tmp_path, "a"), _run_dir(tmp_path, "b")]
        for run in runs:
            _profile(run, "shared-prof", f"mock-{run.name}")
            _ensemble(
                run / "ensembles",
                "shared",
                [{"name": f"from-{run.name}", "model_profile": "shared-prof"}],
            )
        root = _ensemble(tmp_path, "root", [{"name": "kid", "ensemble": "shared"}])
        views = [base.with_run_layer(run) for run in runs]

        results = [await _run(view, project, root) for view in views]

        children = [json.loads(r["results"]["kid"]["response"]) for r in results]
        assert [list(c["results"]) for c in children] == [["from-a"], ["from-b"]]
        assert loaded_models == ["mock-a", "mock-b"]


def _script(path: Path, tag: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "import json, sys\n"
        "sys.stdin.read()\n"
        f'print(json.dumps({{"success": True, "data": {{"tag": "{tag}"}}}}))\n'
    )


def _tag(result: dict[str, Any], agent: str = "s") -> str:
    response = json.loads(result["results"][agent]["response"])
    return str(response["data"]["tag"])


_SCRIPT_ROOT = [{"name": "s", "script": "a/x.py"}]


class TestScriptInTheLayer:
    async def test_a_script_present_only_in_the_layer_runs(
        self, tmp_path: Path, project: Path, base: ConfigurationManager
    ) -> None:
        run = _run_dir(tmp_path)
        _script(run / "scripts" / "a" / "x.py", "layer")
        root = _ensemble(tmp_path, "root", _SCRIPT_ROOT)

        result = await _run(base.with_run_layer(run), project, root)

        assert result["status"] == "completed", result
        assert _tag(result) == "layer"

    async def test_the_layer_script_beats_a_project_and_a_test_primitive_one(
        self,
        tmp_path: Path,
        project: Path,
        base: ConfigurationManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        run = _run_dir(tmp_path)
        _script(run / "scripts" / "a" / "x.py", "layer")
        _script(project / ".llm-orc" / "scripts" / "a" / "x.py", "project")
        _script(tmp_path / "primitives" / "a" / "x.py", "test-primitive")
        monkeypatch.setenv("LLM_ORC_TEST_PRIMITIVES_DIR", str(tmp_path / "primitives"))
        root = _ensemble(tmp_path, "root", _SCRIPT_ROOT)

        result = await _run(base.with_run_layer(run), project, root)

        assert _tag(result) == "layer"

    async def test_a_script_at_the_run_dir_root_resolves_too(
        self, tmp_path: Path, project: Path, base: ConfigurationManager
    ) -> None:
        run = _run_dir(tmp_path)
        _script(run / "a" / "x.py", "layer-root")
        root = _ensemble(tmp_path, "root", _SCRIPT_ROOT)

        result = await _run(base.with_run_layer(run), project, root)

        assert _tag(result) == "layer-root"

    async def test_the_base_manager_does_not_find_a_layer_script(
        self, tmp_path: Path, project: Path, base: ConfigurationManager
    ) -> None:
        run = _run_dir(tmp_path)
        _script(run / "scripts" / "a" / "x.py", "layer")
        root = _ensemble(tmp_path, "root", _SCRIPT_ROOT)

        result = await _run(base, project, root)

        assert result["has_errors"], result


def _counting_script(path: Path, counter: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "import json, sys\n"
        "sys.stdin.read()\n"
        f'open(r"{counter}", "a").write("x")\n'
        'print(json.dumps({"success": True, "data": {"ok": True}}))\n'
    )


def _enable_script_cache(project: Path) -> None:
    _yaml(
        project / ".llm-orc" / "config.yaml",
        {"performance": {"script_cache": {"enabled": True}}},
    )


class TestRunLayerSideEffects:
    async def test_bytecode_goes_under_the_run_dir_not_the_state_dir(
        self,
        tmp_path: Path,
        project: Path,
        base: ConfigurationManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("PYTHONPYCACHEPREFIX", raising=False)
        run = _run_dir(tmp_path)
        _script(run / "scripts" / "a" / "x.py", "layer")
        (run / "scripts" / "a" / "_helpers.py").write_text("tag = 'h'\n")
        (run / "scripts" / "a" / "x.py").write_text(
            "import json, sys\n"
            "from _helpers import tag\n"
            "sys.stdin.read()\n"
            'print(json.dumps({"success": True, "data": {"tag": tag}}))\n'
        )
        root = _ensemble(tmp_path, "root", _SCRIPT_ROOT)

        result = await _run(base.with_run_layer(run), project, root)

        assert _tag(result) == "h", result
        state_pycache = project / ".llm-orc" / "pycache"
        assert not state_pycache.exists() or not list(state_pycache.rglob("*.pyc"))
        assert list((run / "pycache").rglob("*.pyc"))

    async def test_the_script_cache_is_off_for_a_run_with_a_layer(
        self, tmp_path: Path, project: Path, base: ConfigurationManager
    ) -> None:
        _enable_script_cache(project)
        counter = tmp_path / "count.txt"
        run = _run_dir(tmp_path)
        _counting_script(run / "scripts" / "a" / "x.py", counter)
        root = _ensemble(tmp_path, "root", _SCRIPT_ROOT)
        executor = ExecutorFactory.create_root_executor(
            project_dir=project,
            config_manager=base.with_run_layer(run),
            save_artifacts=False,
        )

        await executor.execute(root, "ping")
        await executor.execute(root, "ping")

        assert counter.read_text() == "xx"

    async def test_the_same_config_caches_a_script_without_a_layer(
        self, tmp_path: Path, project: Path, base: ConfigurationManager
    ) -> None:
        # the control: the cache above is off because of the layer
        _enable_script_cache(project)
        counter = tmp_path / "count.txt"
        _counting_script(project / ".llm-orc" / "scripts" / "a" / "x.py", counter)
        root = _ensemble(tmp_path, "root", _SCRIPT_ROOT)
        executor = ExecutorFactory.create_root_executor(
            project_dir=project, config_manager=base, save_artifacts=False
        )

        await executor.execute(root, "ping")
        await executor.execute(root, "ping")

        assert counter.read_text() == "x"

    async def test_a_callers_own_bytecode_prefix_still_wins(
        self,
        tmp_path: Path,
        project: Path,
        base: ConfigurationManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("PYTHONPYCACHEPREFIX", str(tmp_path / "theirs"))
        run = _run_dir(tmp_path)
        _script(run / "scripts" / "a" / "x.py", "layer")
        root = _ensemble(tmp_path, "root", _SCRIPT_ROOT)

        await _run(base.with_run_layer(run), project, root)

        assert not (run / "pycache").exists()
        assert list((tmp_path / "theirs").rglob("*.pyc"))


class TestWithRunLayer:
    def test_a_plain_manager_has_no_layer(self, base: ConfigurationManager) -> None:
        assert base.run_layer_dir is None

    def test_the_layer_is_the_first_ensemble_and_profile_dir(
        self, tmp_path: Path, project: Path, base: ConfigurationManager
    ) -> None:
        (project / ".llm-orc" / "ensembles").mkdir()
        (project / ".llm-orc" / "profiles").mkdir()
        run = _run_dir(tmp_path)
        (run / "ensembles").mkdir()
        (run / "profiles").mkdir()

        view = base.with_run_layer(run)

        assert view.run_layer_dir == run
        assert view.get_ensembles_dirs()[0] == run / "ensembles"
        assert view.get_profiles_dirs()[0] == run / "profiles"
        assert run / "ensembles" not in base.get_ensembles_dirs()

    def test_the_view_does_not_share_the_profile_cache(
        self, tmp_path: Path, base: ConfigurationManager
    ) -> None:
        base.get_model_profiles()
        run = _run_dir(tmp_path)
        _profile(run, "only-here", "mock-x")

        view = base.with_run_layer(run)

        assert "only-here" in view.get_model_profiles()
        assert "only-here" not in base.get_model_profiles()
        assert view._profiles_cache is not base._profiles_cache
        assert view._profiles_cache_mtimes is not base._profiles_cache_mtimes

    def test_a_view_over_an_empty_layer_does_not_hand_back_the_base_cache(
        self, tmp_path: Path, base: ConfigurationManager
    ) -> None:
        cached = base.get_model_profiles()

        view = base.with_run_layer(_run_dir(tmp_path))
        view.get_model_profiles()["injected"] = {"model": "mock-x"}

        assert view.get_model_profiles() is not cached
        assert "injected" not in base.get_model_profiles()
