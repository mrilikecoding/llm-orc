"""One run, gated and clean (Arc 4, Task 6).

Through a real OrchestraService on a temp project, state and global
dirs. Models are ``mock-*`` names the router listing fakes, so the gate
classifies them as a host would and nothing is called out.
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Any

import pytest
import yaml

from llm_orc.core.config.config_manager import resolve_global_config_dir
from llm_orc.core.execution.ensemble_execution import EnsembleExecutor
from llm_orc.core.execution.executor_factory import ExecutorFactory
from llm_orc.core.models.model_factory import ModelFactory
from llm_orc.providers.llama_server import LlamaServerClient
from llm_orc.services.orchestra_service import OrchestraService

LISTING: list[dict[str, Any]] = [
    {"id": name, "status": {"value": "unloaded"}}
    for name in ("mock-seat", "mock-other", "mock-a", "mock-b")
]


class Reporter:
    """A ProgressReporter that keeps what it was told."""

    def __init__(self) -> None:
        self.errors: list[str] = []

    async def info(self, message: str) -> None:
        return None

    async def warning(self, message: str) -> None:
        return None

    async def error(self, message: str) -> None:
        self.errors.append(message)

    async def report_progress(self, progress: float, total: float) -> None:
        return None


def _yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data))


def _profile(base: Path, name: str, model: str = "mock-seat", **extra: Any) -> None:
    _yaml(
        base / "profiles" / f"{name}.yaml",
        {"name": name, "provider": "llama-server", "model": model, **extra},
    )


def _ensemble(base: Path, name: str, agents: list[dict[str, Any]]) -> None:
    _yaml(
        base / "ensembles" / f"{name}.yaml",
        {"name": name, "description": name, "agents": agents},
    )


@pytest.fixture
def listing(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """The router's listing, editable by a test before the run."""
    entries = list(LISTING)
    monkeypatch.setattr(LlamaServerClient, "_list", lambda self: entries)
    return entries


@pytest.fixture
def state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The serve's state dir, apart from the project and the global dir."""
    path = tmp_path / "state"
    path.mkdir()
    monkeypatch.setenv("LLM_ORC_STATE_DIR", str(path))
    return path


@pytest.fixture
def project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, listing: Any, state_dir: Path
) -> Path:
    """A project dir, with the cwd somewhere else entirely."""
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path / "elsewhere")
    root = tmp_path / "proj"
    (root / ".llm-orc" / "ensembles").mkdir(parents=True)
    return root


@pytest.fixture
def service(project: Path) -> OrchestraService:
    svc = OrchestraService()
    assert svc.handle_set_project(str(project))["status"] == "ok"
    return svc


def _marker_script(project: Path, marker: Path) -> None:
    script = project / ".llm-orc" / "scripts" / "mark.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        "import json, sys\n"
        "sys.stdin.read()\n"
        f'open(r"{marker}", "a").write("ran")\n'
        'print(json.dumps({"success": True, "data": {"ok": True}}))\n'
    )


def _kinds(result: dict[str, Any]) -> dict[str, str]:
    return {d["name"]: d["status"] for d in result["error"]["dependencies"]}


class TestEveryRunIsGated:
    async def test_a_missing_profile_refuses_and_no_agent_ran(
        self, project: Path, service: OrchestraService, tmp_path: Path
    ) -> None:
        marker = tmp_path / "marker.txt"
        _marker_script(project, marker)
        _ensemble(
            project / ".llm-orc",
            "top",
            [
                {"name": "first", "script": "mark.py"},
                {"name": "w", "model_profile": "nope", "depends_on": ["first"]},
            ],
        )

        result = await service.invoke({"ensemble_name": "top", "input": "hi"})

        assert result["status"] == "error"
        assert result["has_errors"] is True
        assert result["results"] == {}
        assert result["deliverable"] is None
        assert result["error"]["kind"] == "not_equipped"
        assert _kinds(result)["nope"] == "missing_profile"
        assert not marker.exists()

    async def test_a_ready_named_ensemble_still_runs(
        self, project: Path, service: OrchestraService, tmp_path: Path
    ) -> None:
        _profile(project / ".llm-orc", "seat")
        _ensemble(project / ".llm-orc", "top", [{"name": "w", "model_profile": "seat"}])

        result = await service.invoke({"ensemble_name": "top", "input": "hi"})

        assert result["status"] == "success", result
        assert "error" not in result

    async def test_a_pullable_model_without_pull_is_refused(
        self, project: Path, service: OrchestraService
    ) -> None:
        _profile(project / ".llm-orc", "seat", hf_repo="x/y:Q4")
        _ensemble(project / ".llm-orc", "top", [{"name": "w", "model_profile": "seat"}])

        result = await service.invoke({"ensemble_name": "top", "input": "hi"})

        assert result["error"]["kind"] == "not_equipped"
        assert _kinds(result)["seat"] == "pullable"

    async def test_an_unmet_fallback_hop_refuses_a_ready_primary(
        self, project: Path, service: OrchestraService
    ) -> None:
        _profile(project / ".llm-orc", "seat", fallback_model_profile="gone")
        _ensemble(project / ".llm-orc", "top", [{"name": "w", "model_profile": "seat"}])

        result = await service.invoke({"ensemble_name": "top", "input": "hi"})

        assert result["error"]["kind"] == "not_equipped"
        assert _kinds(result)["gone"] == "missing_profile"
        assert _kinds(result)["seat"] == "ready"

    async def test_execute_streaming_reports_the_refusal_and_returns_it(
        self, project: Path, service: OrchestraService
    ) -> None:
        _ensemble(project / ".llm-orc", "top", [{"name": "w", "model_profile": "nope"}])
        reporter = Reporter()

        result = await service.execute_streaming("top", "hi", reporter)

        assert result["error"]["kind"] == "not_equipped"
        assert result["results"] == {}
        assert len(reporter.errors) == 1
        assert "nope" in reporter.errors[0]

    async def test_invoke_streaming_yields_one_execution_failed_event(
        self, project: Path, service: OrchestraService
    ) -> None:
        _ensemble(project / ".llm-orc", "top", [{"name": "w", "model_profile": "nope"}])

        events = [
            e
            async for e in service.invoke_streaming(
                {"ensemble_name": "top", "input": "hi"}
            )
        ]

        assert [e["type"] for e in events] == ["execution_failed"]
        assert events[0]["data"]["error"]["kind"] == "not_equipped"
        assert {
            d["name"]: d["status"] for d in events[0]["data"]["error"]["dependencies"]
        }["nope"] == "missing_profile"

    async def test_a_missing_named_ensemble_still_raises(
        self, service: OrchestraService
    ) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            await service.invoke({"ensemble_name": "ghost", "input": "hi"})


def _files(base: Path) -> dict[str, int]:
    """Every file under ``base`` with its size."""
    return {
        str(p.relative_to(base)): p.stat().st_size
        for p in sorted(base.rglob("*"))
        if p.is_file()
    }


def _trees(project: Path, state_dir: Path) -> dict[str, dict[str, int]]:
    return {
        "state": _files(state_dir),
        "global": _files(resolve_global_config_dir()),
        "project": _files(project),
    }


def _runs(state_dir: Path) -> list[Path]:
    runs = state_dir / "runs"
    return sorted(runs.iterdir()) if runs.exists() else []


def _script_source(tag: str, *, imports: str = "") -> str:
    return (
        "import json, sys\n"
        f"{imports}"
        "sys.stdin.read()\n"
        f'print(json.dumps({{"success": True, "data": {{"tag": "{tag}"}}}}))\n'
    )


def _tag(result: dict[str, Any], agent: str) -> str:
    return str(json.loads(result["results"][agent]["response"])["data"]["tag"])


INLINE: dict[str, Any] = {
    "name": "inline-top",
    "description": "inline",
    "agents": [
        {"name": "s", "script": "probe/x.py"},
        {"name": "w", "model_profile": "inline-seat"},
    ],
}


def _inline_request(**more: Any) -> dict[str, Any]:
    return {
        "ensemble": INLINE,
        "profiles": {"inline-seat": {"provider": "llama-server", "model": "mock-a"}},
        "scripts": {"probe/x.py": _script_source("injected")},
        "input": "hi",
        **more,
    }


class TestAnInlineRunLeavesNothing:
    async def test_it_runs_and_every_tree_is_the_same_afterwards(
        self, project: Path, service: OrchestraService, state_dir: Path
    ) -> None:
        # The first run of any kind makes the serve's credential storage
        # (and so its encryption key, in the global dir): serve state, not
        # the injection's. Start from a serve that has served once.
        _ensemble(project / ".llm-orc", "warm", [{"name": "s", "script": "echo hi"}])
        await service.invoke({"ensemble_name": "warm", "input": "hi"})
        before = _trees(project, state_dir)

        result = await service.invoke(_inline_request())

        assert result["status"] == "success", result
        assert _tag(result, "s") == "injected"
        assert _trees(project, state_dir) == before
        assert _runs(state_dir) == []

    async def test_an_injected_script_imports_its_injected_sibling_helper(
        self, service: OrchestraService
    ) -> None:
        request = _inline_request(
            scripts={
                "probe/x.py": _script_source(
                    "h", imports="from _helpers import tag\n"
                ).replace('"tag": "h"', '"tag": tag'),
                "probe/_helpers.py": "tag = 'from-helper'\n",
            }
        )

        result = await service.invoke(request)

        assert result["status"] == "success", result
        assert _tag(result, "s") == "from-helper"

    async def test_a_script_without_its_helper_does_not_import_the_hosts(
        self, project: Path, service: OrchestraService, tmp_path: Path
    ) -> None:
        marker = tmp_path / "host-helper-imported.txt"
        host_helper = project / ".llm-orc" / "scripts" / "probe" / "_helpers.py"
        host_helper.parent.mkdir(parents=True)
        host_helper.write_text(f'open(r"{marker}", "w").write("x")\ntag = "host"\n')
        request = _inline_request(
            scripts={
                "probe/x.py": _script_source("h", imports="from _helpers import tag\n")
            }
        )

        result = await service.invoke(request)

        assert result["has_errors"] is True
        assert not marker.exists()


class TestTheRunDirIsAlwaysRemoved:
    @pytest.fixture
    def executing(self, monkeypatch: pytest.MonkeyPatch, state_dir: Path) -> Any:
        """Replace execute: records that the run dir existed mid-run, then
        does what the test says."""
        seen: dict[str, Any] = {"mid_run": [], "started": asyncio.Event()}

        async def execute(self: Any, config: Any, input_data: str = "") -> Any:
            seen["mid_run"] = _runs(state_dir)
            seen["started"].set()
            await seen["behavior"]()

        monkeypatch.setattr(EnsembleExecutor, "execute", execute)
        return seen

    async def test_on_a_refusal(
        self, service: OrchestraService, state_dir: Path
    ) -> None:
        request = _inline_request(profiles={})

        result = await service.invoke(request)

        assert result["error"]["kind"] == "not_equipped"
        assert _runs(state_dir) == []

    async def test_on_an_executor_exception(
        self, service: OrchestraService, state_dir: Path, executing: Any
    ) -> None:
        async def boom() -> None:
            raise RuntimeError("boom")

        executing["behavior"] = boom

        with pytest.raises(RuntimeError, match="boom"):
            await service.invoke(_inline_request())

        assert len(executing["mid_run"]) == 1
        assert _runs(state_dir) == []

    async def test_on_cancellation(
        self, service: OrchestraService, state_dir: Path, executing: Any
    ) -> None:
        async def forever() -> None:
            await asyncio.Event().wait()

        executing["behavior"] = forever
        task = asyncio.create_task(service.invoke(_inline_request()))
        await executing["started"].wait()
        assert len(_runs(state_dir)) == 1

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert _runs(state_dir) == []


class TestRunsStayApart:
    async def test_two_concurrent_runs_each_see_their_own_content(
        self,
        service: OrchestraService,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        loaded: list[str] = []
        original = ModelFactory.load_model

        async def spy(self: ModelFactory, model_name: str, *a: Any, **kw: Any) -> Any:
            loaded.append(model_name)
            return await original(self, model_name, *a, **kw)

        monkeypatch.setattr(ModelFactory, "load_model", spy)
        requests = [
            _inline_request(
                profiles={"inline-seat": {"provider": "llama-server", "model": m}},
                scripts={"probe/x.py": _script_source(m)},
            )
            for m in ("mock-a", "mock-b")
        ]

        results = await asyncio.gather(*(service.invoke(r) for r in requests))

        assert [_tag(r, "s") for r in results] == ["mock-a", "mock-b"]
        assert sorted(loaded) == ["mock-a", "mock-b"]

    async def test_a_following_named_run_resolves_as_before(
        self,
        project: Path,
        service: OrchestraService,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        loaded: list[str] = []
        original = ModelFactory.load_model

        async def spy(self: ModelFactory, model_name: str, *a: Any, **kw: Any) -> Any:
            loaded.append(model_name)
            return await original(self, model_name, *a, **kw)

        monkeypatch.setattr(ModelFactory, "load_model", spy)
        layers: list[Path | None] = []
        create = ExecutorFactory.create_root_executor

        def watch(*args: Any, **kw: Any) -> Any:
            layers.append(kw["config_manager"].run_layer_dir)
            return create(*args, **kw)

        monkeypatch.setattr(ExecutorFactory, "create_root_executor", watch)
        _profile(project / ".llm-orc", "seat", model="mock-seat")
        _ensemble(project / ".llm-orc", "top", [{"name": "w", "model_profile": "seat"}])
        shadow = _inline_request(
            ensemble={
                "name": "shadow",
                "description": "s",
                "agents": [{"name": "w", "model_profile": "seat"}],
            },
            profiles={"seat": {"provider": "llama-server", "model": "mock-other"}},
            scripts={},
        )

        await service.invoke(shadow)
        await service.invoke({"ensemble_name": "top", "input": "hi"})

        assert loaded == ["mock-other", "mock-seat"]
        assert layers[-1] is None, "the named run was built on a run layer view"
        base = service.config_manager
        assert base.run_layer_dir is None
        assert base.get_model_profiles()["seat"]["model"] == "mock-seat"
        assert "inline-seat" not in base.get_model_profiles()
        assert service._executor is None or (
            service._executor._config_manager.run_layer_dir is None
        )


class TestMalformedRequestsWriteNothing:
    @pytest.mark.parametrize(
        "bad",
        [
            {"ensemble_name": "top", "ensemble": INLINE},
            {"input": "no root at all", "scripts": {"a.py": "x"}},
            {"ensemble": INLINE, "scripts": {"../x.py": "print(1)"}},
            {"ensemble": INLINE, "scripts": {"/etc/x.py": "print(1)"}},
            {"ensemble": INLINE, "scripts": {"a//b.py": "print(1)"}},
            {"ensemble": INLINE, "scripts": {"a\\b.py": "print(1)"}},
            {"ensemble": INLINE, "ensembles": {"../kid": INLINE}},
            {"ensemble": {"name": "x", "description": "no agents"}},
        ],
    )
    async def test_each_is_invalid_request_and_nothing_is_written(
        self,
        project: Path,
        service: OrchestraService,
        state_dir: Path,
        bad: dict[str, Any],
    ) -> None:
        before = _trees(project, state_dir)

        result = await service.invoke(bad)

        assert result["error"]["kind"] == "invalid_request", result
        assert result["has_errors"] is True
        assert _trees(project, state_dir) == before
        assert _runs(state_dir) == []


class TestArtifacts:
    async def test_an_inline_root_saves_none_and_a_named_root_keeps_its_own(
        self, project: Path, service: OrchestraService, state_dir: Path
    ) -> None:
        _profile(project / ".llm-orc", "seat")
        _ensemble(project / ".llm-orc", "top", [{"name": "w", "model_profile": "seat"}])

        await service.invoke(_inline_request())
        assert not (state_dir / "artifacts").exists()

        await service.invoke(
            {
                "ensemble_name": "top",
                "input": "hi",
                "profiles": {"extra": {"provider": "llama-server", "model": "mock-a"}},
            }
        )
        assert list((state_dir / "artifacts" / "top").iterdir())

    async def test_execute_streaming_saves_no_artifact_for_an_inline_root(
        self, service: OrchestraService, state_dir: Path
    ) -> None:
        injection = {k: v for k, v in _inline_request().items() if k != "input"}

        result = await service.execute_streaming(
            None, "hi", Reporter(), injection=injection
        )

        assert result["status"] == "success", result
        assert not (state_dir / "artifacts").exists()
        assert _runs(state_dir) == []


class TestInjectedKindsGateLikeTheRun:
    async def test_a_child_ensemble_only_in_the_request_is_ready_and_runs(
        self, service: OrchestraService
    ) -> None:
        root = {
            "name": "r",
            "description": "r",
            "agents": [{"name": "k", "ensemble": "kid"}],
        }
        kid = {
            "description": "kid",
            "agents": [{"name": "s", "script": "probe/x.py"}],
        }
        request = {
            "ensemble": root,
            "ensembles": {"kid": kid},
            "scripts": {"probe/x.py": _script_source("kid-ran")},
        }

        result = await service.invoke(request)

        assert result["status"] == "success", result

    async def test_the_same_root_without_its_child_is_refused_not_run(
        self, service: OrchestraService
    ) -> None:
        root = {
            "name": "r",
            "description": "r",
            "agents": [{"name": "k", "ensemble": "kid"}],
        }

        result = await service.invoke({"ensemble": root})

        assert result["error"]["kind"] == "not_equipped"
        assert _kinds(result)["kid"] == "missing_ensemble"

    async def test_a_missing_script_is_refused_not_run(
        self, service: OrchestraService
    ) -> None:
        result = await service.invoke(_inline_request(scripts={}))

        assert result["error"]["kind"] == "not_equipped"
        assert _kinds(result)["probe/x.py"] == "missing_script"


def _bound_ensemble(project: Path, tmp_path: Path) -> Path:
    """An ensemble naming profile ``a`` after a phase-0 marker script."""
    marker = tmp_path / "marker.txt"
    _marker_script(project, marker)
    _ensemble(
        project / ".llm-orc",
        "top",
        [
            {"name": "first", "script": "mark.py"},
            {"name": "w", "model_profile": "a", "depends_on": ["first"]},
        ],
    )
    return marker


@pytest.fixture
def loaded_models(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    seen: list[str] = []
    original = ModelFactory.load_model

    async def spy(self: ModelFactory, model_name: str, *a: Any, **kw: Any) -> Any:
        seen.append(model_name)
        return await original(self, model_name, *a, **kw)

    monkeypatch.setattr(ModelFactory, "load_model", spy)
    return seen


class TestBindings:
    async def test_a_missing_profile_refuses_and_the_same_call_with_bind_runs(
        self,
        project: Path,
        service: OrchestraService,
        tmp_path: Path,
        loaded_models: list[str],
    ) -> None:
        marker = _bound_ensemble(project, tmp_path)
        _profile(project / ".llm-orc", "b", model="mock-other")

        refused = await service.invoke({"ensemble_name": "top", "input": "hi"})
        assert refused["error"]["kind"] == "not_equipped"
        assert not marker.exists()

        result = await service.invoke(
            {"ensemble_name": "top", "input": "hi", "bind": {"a": "b"}}
        )

        assert result["status"] == "success", result
        assert result["bindings"] == {"a": "b"}
        assert loaded_models == ["mock-other"]
        assert marker.exists()

    async def test_a_bind_target_the_host_lacks_is_refused_even_when_it_has_the_key(
        self,
        project: Path,
        service: OrchestraService,
        tmp_path: Path,
        loaded_models: list[str],
    ) -> None:
        marker = _bound_ensemble(project, tmp_path)
        _profile(project / ".llm-orc", "a", model="mock-seat")

        result = await service.invoke(
            {"ensemble_name": "top", "input": "hi", "bind": {"a": "missing-b"}}
        )

        assert result["error"]["kind"] == "not_equipped", result
        rows = [d for d in result["error"]["dependencies"] if d["name"] == "missing-b"]
        assert [(r["status"], r["via"]) for r in rows] == [
            ("missing_profile", ["bind:a"])
        ]
        assert not marker.exists()
        assert loaded_models == []

    async def test_a_binding_beats_the_hosts_own_profile_of_that_name(
        self,
        project: Path,
        service: OrchestraService,
        tmp_path: Path,
        loaded_models: list[str],
    ) -> None:
        _bound_ensemble(project, tmp_path)
        _profile(project / ".llm-orc", "a", model="mock-seat")
        _profile(project / ".llm-orc", "b", model="mock-other")

        result = await service.invoke(
            {"ensemble_name": "top", "input": "hi", "bind": {"a": "b"}}
        )

        assert result["bindings"] == {"a": "b"}
        assert loaded_models == ["mock-other"]

    async def test_a_target_defined_only_inline_works(
        self,
        project: Path,
        service: OrchestraService,
        tmp_path: Path,
        loaded_models: list[str],
    ) -> None:
        _bound_ensemble(project, tmp_path)

        result = await service.invoke(
            {
                "ensemble_name": "top",
                "input": "hi",
                "profiles": {"b": {"provider": "llama-server", "model": "mock-b"}},
                "bind": {"a": "b"},
            }
        )

        assert result["bindings"] == {"a": "b"}
        assert loaded_models == ["mock-b"]

    async def test_the_targets_fallback_chain_is_gated_too(
        self, project: Path, service: OrchestraService, tmp_path: Path
    ) -> None:
        _bound_ensemble(project, tmp_path)
        _profile(project / ".llm-orc", "b", fallback_model_profile="gone")

        result = await service.invoke(
            {"ensemble_name": "top", "input": "hi", "bind": {"a": "b"}}
        )

        assert result["error"]["kind"] == "not_equipped"
        assert _kinds(result)["gone"] == "missing_profile"

    async def test_a_bind_key_no_profile_in_the_closure_names_is_invalid(
        self,
        project: Path,
        service: OrchestraService,
        tmp_path: Path,
        loaded_models: list[str],
    ) -> None:
        # The misspelled key must not fall through to the host's `a`.
        marker = _bound_ensemble(project, tmp_path)
        _profile(project / ".llm-orc", "a", model="mock-seat")
        _profile(project / ".llm-orc", "b", model="mock-other")

        result = await service.invoke(
            {"ensemble_name": "top", "input": "hi", "bind": {"aa": "b"}}
        )

        assert result["error"]["kind"] == "invalid_request", result
        assert "aa" in result["error"]["message"]
        assert not marker.exists()
        assert loaded_models == []

    async def test_execute_streaming_returns_the_bindings(
        self, project: Path, service: OrchestraService, tmp_path: Path
    ) -> None:
        _bound_ensemble(project, tmp_path)
        _profile(project / ".llm-orc", "b", model="mock-other")

        result = await service.execute_streaming(
            "top", "hi", Reporter(), injection={"bind": {"a": "b"}}
        )

        assert result["bindings"] == {"a": "b"}


class TestPull:
    @pytest.fixture
    def pulls(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        """Fake LlamaServerClient.pull: records each call and its thread,
        answers what the test sets. The listing is never changed, so a
        gate that re-read it would still see the model as pullable."""
        record: dict[str, Any] = {
            "calls": [],
            "threads": [],
            "answer": {"status": "loaded", "failed": False, "exit_code": None},
        }

        def pull(self: Any, model: str, *, timeout_s: float, poll_s: float) -> Any:
            record["calls"].append(model)
            record["threads"].append(threading.get_ident())
            return record["answer"]

        monkeypatch.setattr(LlamaServerClient, "pull", pull)
        return record

    def _pullable_top(self, project: Path) -> None:
        _profile(project / ".llm-orc", "seat", hf_repo="x/y:Q4")
        _profile(project / ".llm-orc", "twin", hf_repo="x/y:Q4")
        _ensemble(
            project / ".llm-orc",
            "top",
            [
                {"name": "w", "model_profile": "seat"},
                {"name": "v", "model_profile": "twin"},
            ],
        )

    async def test_pull_true_pulls_each_model_once_and_the_run_proceeds(
        self, project: Path, service: OrchestraService, pulls: dict[str, Any]
    ) -> None:
        self._pullable_top(project)

        result = await service.invoke(
            {"ensemble_name": "top", "input": "hi", "pull": True}
        )

        assert result["status"] == "success", result
        assert result["pulled"] == ["mock-seat"]
        assert pulls["calls"] == ["mock-seat"]

    async def test_the_pull_runs_off_the_event_loop_thread(
        self, project: Path, service: OrchestraService, pulls: dict[str, Any]
    ) -> None:
        self._pullable_top(project)

        await service.invoke({"ensemble_name": "top", "input": "hi", "pull": True})

        assert pulls["threads"]
        assert threading.get_ident() not in pulls["threads"]

    async def test_a_load_that_ends_unloaded_failed_keeps_the_model_pullable(
        self,
        project: Path,
        service: OrchestraService,
        pulls: dict[str, Any],
        tmp_path: Path,
        state_dir: Path,
    ) -> None:
        marker = tmp_path / "marker.txt"
        _marker_script(project, marker)
        _profile(project / ".llm-orc", "seat", hf_repo="x/y:Q4")
        _ensemble(
            project / ".llm-orc",
            "top",
            [
                {"name": "first", "script": "mark.py"},
                {"name": "w", "model_profile": "seat", "depends_on": ["first"]},
            ],
        )
        pulls["answer"] = {"status": "unloaded", "failed": True, "exit_code": 1}

        result = await service.invoke(
            {"ensemble_name": "top", "input": "hi", "pull": True}
        )

        assert result["error"]["kind"] == "not_equipped", result
        row = next(d for d in result["error"]["dependencies"] if d["name"] == "seat")
        assert row["status"] == "pullable"
        assert "unloaded" in row["detail"]
        assert "pulled" not in result
        assert not marker.exists()
        assert _runs(state_dir) == []

    async def test_without_pull_nothing_is_pulled(
        self, project: Path, service: OrchestraService, pulls: dict[str, Any]
    ) -> None:
        self._pullable_top(project)

        result = await service.invoke({"ensemble_name": "top", "input": "hi"})

        assert result["error"]["kind"] == "not_equipped"
        assert pulls["calls"] == []

    async def test_pull_does_not_resolve_a_model_the_router_does_not_list(
        self,
        project: Path,
        service: OrchestraService,
        pulls: dict[str, Any],
        listing: list[dict[str, Any]],
    ) -> None:
        _profile(project / ".llm-orc", "seat", model="mock-absent", hf_repo="x/y:Q4")
        _ensemble(project / ".llm-orc", "top", [{"name": "w", "model_profile": "seat"}])

        result = await service.invoke(
            {"ensemble_name": "top", "input": "hi", "pull": True}
        )

        assert result["error"]["kind"] == "not_equipped"
        assert _kinds(result)["seat"] == "needs_restart"
        assert pulls["calls"] == []

    async def test_execute_streaming_returns_the_pulled_models(
        self, project: Path, service: OrchestraService, pulls: dict[str, Any]
    ) -> None:
        self._pullable_top(project)

        result = await service.execute_streaming(
            "top", "hi", Reporter(), injection={"pull": True}
        )

        assert result["pulled"] == ["mock-seat"]


class TestOtherSeams:
    async def test_an_inline_profile_reusing_a_host_model_name_with_another_source(
        self,
        service: OrchestraService,
        listing: list[dict[str, Any]],
        state_dir: Path,
        loaded_models: list[str],
    ) -> None:
        listing[:] = [
            {
                "id": "mock-a",
                "status": {
                    "value": "unloaded",
                    "args": ["/bin/llama-server", "--hf-repo", "host/source:Q4"],
                },
            },
            {"id": "host/source:Q4", "status": {"value": "unloaded"}},
        ]
        request = _inline_request(
            profiles={
                "inline-seat": {
                    "provider": "llama-server",
                    "model": "mock-a",
                    "hf_repo": "caller/other:Q4",
                }
            }
        )

        result = await service.invoke(request)

        assert result["error"]["kind"] == "not_equipped", result
        assert _kinds(result)["inline-seat"] == "needs_restart"
        assert loaded_models == []
        assert _runs(state_dir) == []

    async def test_a_caller_injected_executor_is_never_used_for_a_layer_run(
        self, project: Path
    ) -> None:
        class Sentinel:
            calls = 0

            async def execute(self, config: Any, input_data: str = "") -> Any:
                Sentinel.calls += 1
                return {"status": "completed", "results": {}, "deliverable": None}

        service = OrchestraService(executor=Sentinel())  # type: ignore[arg-type]
        service.handle_set_project(str(project))
        service._executor = Sentinel()  # type: ignore[assignment]
        service._executor_injected = True

        result = await service.invoke(_inline_request())

        assert result["status"] == "success", result
        assert _tag(result, "s") == "injected"
        assert Sentinel.calls == 0
