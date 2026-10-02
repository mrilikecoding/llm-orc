"""``llm-orc invoke`` runs through the preparation step REST and MCP use
(Arc 5, Task 3, ruling 1). Real ``invoke`` command through ``CliRunner``,
a real service on a temp project, the router's listing faked so the gate
classifies profiles as a host would and nothing is called out."""

# ruff: noqa: F811  (imported fixtures are redefined as test parameters)
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner, Result
from fastapi.testclient import TestClient

from llm_orc.cli import cli
from llm_orc.providers.llama_server import LlamaServerClient
from tests.unit.services.test_one_run_injection import (  # noqa: F401
    _bound_ensemble,
    _ensemble,
    _marker_script,
    _profile,
    listing,
    project,
    service,
    state_dir,
)
from tests.unit.web.test_api_run_injection import client  # noqa: F401


@pytest.fixture
def in_project(project: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The cwd is the project, as it is for a person running the CLI."""
    monkeypatch.chdir(project)
    return project


def _invoke(*args: str) -> Result:
    return CliRunner().invoke(cli, ["invoke", "top", "hi", *args])


class TestARefusedRun:
    def test_a_missing_profile_prints_the_row_and_no_agent_ran(
        self, in_project: Path, tmp_path: Path
    ) -> None:
        marker = _bound_ensemble(in_project, tmp_path)

        result = _invoke("--output-format", "text")

        assert result.exit_code == 1, result.output
        assert "missing_profile" in result.output
        assert "a" in result.output
        assert not marker.exists()

    def test_the_refusal_table_names_each_column_in_rich_mode(
        self, in_project: Path, tmp_path: Path
    ) -> None:
        _bound_ensemble(in_project, tmp_path)

        result = _invoke()

        assert result.exit_code == 1, result.output
        for column in ("kind", "name", "status", "via", "resolve"):
            assert column in result.output
        assert "missing_profile" in result.output
        assert "top.w" in result.output

    def test_json_mode_prints_the_refusal_envelope(
        self, in_project: Path, tmp_path: Path
    ) -> None:
        marker = _bound_ensemble(in_project, tmp_path)

        result = _invoke("--output-format", "json")

        assert result.exit_code == 1, result.output
        envelope = json.loads(result.output)
        assert envelope["status"] == "error"
        assert envelope["error"]["kind"] == "not_equipped"
        rows = {d["name"]: d["status"] for d in envelope["error"]["dependencies"]}
        assert rows["a"] == "missing_profile"
        assert not marker.exists()

    def test_the_cli_and_rest_give_one_verdict_on_one_fixture(
        self, in_project: Path, tmp_path: Path, client: TestClient
    ) -> None:
        """Review focus 6: the same not-runnable ensemble, refused by the
        CLI exactly when REST refuses it, with the same rows."""
        _bound_ensemble(in_project, tmp_path)

        rest = client.post("/api/ensembles/top/execute", json={"input": "hi"}).json()
        cli_run = _invoke("--output-format", "json")

        assert rest["error"]["kind"] == "not_equipped"
        assert json.loads(cli_run.output)["error"] == rest["error"]


class TestBind:
    def test_a_bind_runs_and_shows_the_binding(
        self, in_project: Path, tmp_path: Path
    ) -> None:
        marker = _bound_ensemble(in_project, tmp_path)
        _profile(in_project / ".llm-orc", "b", model="mock-other")

        result = _invoke("--bind", "a=b", "--output-format", "text")

        assert result.exit_code == 0, result.output
        assert "Bindings applied: a -> b" in result.output
        assert marker.exists()

    def test_the_json_document_carries_the_binding(
        self, in_project: Path, tmp_path: Path
    ) -> None:
        _bound_ensemble(in_project, tmp_path)
        _profile(in_project / ".llm-orc", "b", model="mock-other")

        result = _invoke("--bind", "a=b", "--output-format", "json")

        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["bindings"] == {"a": "b"}

    def test_a_misspelled_bind_key_is_refused_with_the_services_message(
        self, in_project: Path, tmp_path: Path
    ) -> None:
        marker = _bound_ensemble(in_project, tmp_path)
        _profile(in_project / ".llm-orc", "b", model="mock-other")

        result = _invoke("--bind", "typo=b", "--output-format", "text")

        assert result.exit_code == 1, result.output
        assert "bind key 'typo' names no profile in the closure" in result.output
        assert not marker.exists()

    def test_a_value_without_an_equals_sign_is_a_usage_error(
        self, in_project: Path
    ) -> None:
        result = _invoke("--bind", "a")

        assert result.exit_code == 2
        assert "NAME=TARGET" in result.output


class TestPull:
    @pytest.fixture
    def pulls(self, monkeypatch: pytest.MonkeyPatch) -> list[str]:
        calls: list[str] = []

        def pull(self: Any, model: str, *, timeout_s: float, poll_s: float) -> Any:
            calls.append(model)
            return {"status": "loaded", "failed": False, "exit_code": None}

        monkeypatch.setattr(LlamaServerClient, "pull", pull)
        return calls

    def _pullable_top(self, project: Path) -> None:
        _profile(project / ".llm-orc", "seat", hf_repo="x/y:Q4")
        _ensemble(project / ".llm-orc", "top", [{"name": "w", "model_profile": "seat"}])

    def test_a_pullable_model_is_refused_without_pull(
        self, in_project: Path, pulls: list[str]
    ) -> None:
        self._pullable_top(in_project)

        result = _invoke("--output-format", "text")

        assert result.exit_code == 1, result.output
        assert "pullable" in result.output
        assert pulls == []

    def test_pull_pulls_the_model_and_shows_it(
        self, in_project: Path, pulls: list[str]
    ) -> None:
        self._pullable_top(in_project)

        result = _invoke("--pull", "--output-format", "text")

        assert result.exit_code == 0, result.output
        assert "Models pulled: mock-seat" in result.output
        assert pulls == ["mock-seat"]


class TestARunnableEnsembleIsUnchanged:
    def test_the_json_document_keeps_its_keys(self, in_project: Path) -> None:
        _ensemble(
            in_project / ".llm-orc", "top", [{"name": "s", "script": "echo '{}'"}]
        )

        result = _invoke("--output-format", "json")

        assert result.exit_code == 0, result.output
        assert set(json.loads(result.output)) == {
            "results",
            "metadata",
            "config",
            "status",
            "has_errors",
            "deliverable",
        }
