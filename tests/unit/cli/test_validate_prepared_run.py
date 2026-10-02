"""``llm-orc validate run`` runs through the preparation step REST and MCP
use (Arc 5 follow-up to Task 3, review focus 6). Real command through
``CliRunner``, a real service on a temp project, the router's listing
faked so the gate classifies profiles as a host would."""

# ruff: noqa: F811  (imported fixtures are redefined as test parameters)
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner, Result

from llm_orc.cli import cli
from llm_orc.providers.llama_server import LlamaServerClient
from tests.unit.services.test_one_run_injection import (  # noqa: F401
    _marker_script,
    _profile,
    listing,
    project,
    state_dir,
)


@pytest.fixture
def in_project(project: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The cwd is the project, as it is for a person running the CLI."""
    monkeypatch.chdir(project)
    return project


def _validation_ensemble(
    project: Path, agents: list[dict[str, Any]], required: list[str]
) -> None:
    path = project / ".llm-orc" / "ensembles" / "top.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "name": "top",
                "description": "top",
                "agents": agents,
                "validation": {"structural": {"required_agents": required}},
            }
        )
    )


def _validate(*args: str) -> Result:
    return CliRunner().invoke(cli, ["validate", "run", "top", *args])


class TestValidateIsGated:
    def test_a_missing_profile_is_refused_and_no_agent_ran(
        self, in_project: Path, tmp_path: Path
    ) -> None:
        marker = tmp_path / "marker.txt"
        _marker_script(in_project, marker)
        _validation_ensemble(
            in_project,
            [
                {"name": "first", "script": "mark.py"},
                {"name": "w", "model_profile": "nope", "depends_on": ["first"]},
            ],
            ["first", "w"],
        )

        result = _validate()

        assert result.exit_code == 1, result.output
        assert "not_equipped" in result.output
        assert "missing_profile" in result.output
        assert not marker.exists()

    def test_a_runnable_ensemble_still_passes(
        self, in_project: Path, tmp_path: Path
    ) -> None:
        marker = tmp_path / "marker.txt"
        _marker_script(in_project, marker)
        _validation_ensemble(
            in_project, [{"name": "first", "script": "mark.py"}], ["first"]
        )

        result = _validate()

        assert result.exit_code == 0, result.output
        assert "Validation PASSED" in result.output
        assert marker.exists()

    def test_a_runnable_ensemble_still_fails_its_validation(
        self, in_project: Path, tmp_path: Path
    ) -> None:
        _marker_script(in_project, tmp_path / "marker.txt")
        _validation_ensemble(
            in_project, [{"name": "first", "script": "mark.py"}], ["first", "ghost"]
        )

        result = _validate("--verbose")

        assert result.exit_code == 1, result.output
        assert "Validation FAILED" in result.output
        assert "Missing required agents: ghost" in result.output


class TestValidateBinds:
    def _top_on_profile(self, project: Path, tmp_path: Path, profile: str) -> Path:
        marker = tmp_path / "marker.txt"
        _marker_script(project, marker)
        _validation_ensemble(
            project,
            [
                {"name": "first", "script": "mark.py"},
                {"name": "w", "model_profile": profile, "depends_on": ["first"]},
            ],
            ["first", "w"],
        )
        return marker

    def test_a_missing_profile_is_refused_without_bind(
        self, in_project: Path, tmp_path: Path
    ) -> None:
        marker = self._top_on_profile(in_project, tmp_path, "a")
        _profile(in_project / ".llm-orc", "b", model="mock-other")

        result = _validate()

        assert result.exit_code == 1, result.output
        assert "missing_profile" in result.output
        assert not marker.exists()

    def test_bind_runs_it_and_it_validates(
        self, in_project: Path, tmp_path: Path
    ) -> None:
        marker = self._top_on_profile(in_project, tmp_path, "a")
        _profile(in_project / ".llm-orc", "b", model="mock-other")

        result = _validate("--bind", "a=b")

        assert result.exit_code == 0, result.output
        assert "Validation PASSED" in result.output
        assert marker.exists()

    def test_a_value_without_an_equals_sign_is_a_usage_error(
        self, in_project: Path
    ) -> None:
        result = _validate("--bind", "a")

        assert result.exit_code == 2
        assert "NAME=TARGET" in result.output


class TestValidatePulls:
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
        _validation_ensemble(project, [{"name": "w", "model_profile": "seat"}], ["w"])

    def test_a_pullable_model_is_refused_without_pull(
        self, in_project: Path, pulls: list[str]
    ) -> None:
        self._pullable_top(in_project)

        result = _validate()

        assert result.exit_code == 1, result.output
        assert "pullable" in result.output
        assert pulls == []

    def test_pull_pulls_the_model_and_it_validates(
        self, in_project: Path, pulls: list[str]
    ) -> None:
        self._pullable_top(in_project)

        result = _validate("--pull")

        assert result.exit_code == 0, result.output
        assert "Validation PASSED" in result.output
        assert pulls == ["mock-seat"]
