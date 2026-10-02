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
from tests.unit.services.test_one_run_injection import (  # noqa: F401
    _marker_script,
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
