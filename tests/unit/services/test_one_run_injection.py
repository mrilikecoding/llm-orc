"""One run, gated and clean (Arc 4, Task 6).

Through a real OrchestraService on a temp project, state and global
dirs. Models are ``mock-*`` names the router listing fakes, so the gate
classifies them as a host would and nothing is called out.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from llm_orc.providers.llama_server import LlamaServerClient
from llm_orc.services.orchestra_service import OrchestraService

LISTING: list[dict[str, Any]] = [
    {"id": "mock-seat", "status": {"value": "unloaded"}},
    {"id": "mock-other", "status": {"value": "unloaded"}},
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
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, listing: Any) -> Path:
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
