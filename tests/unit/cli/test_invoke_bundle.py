"""The local CLI runs a bundle by name (Arc 5, Task 9, ruling 9).

Real ``invoke`` and ``list-ensembles`` through ``CliRunner``, a real
service on a temp project. The bundle is persisted through REST on a
second service sharing the temp global config dir, as a remote's would
be.
"""

# ruff: noqa: F811  (imported fixtures are redefined as test parameters)
from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner, Result
from fastapi.testclient import TestClient

from llm_orc.cli import cli
from llm_orc.core.config.config_manager import ConfigurationManager
from tests.unit.services.test_one_run_injection import (  # noqa: F401
    _tag,
    listing,
    project,
    service,
    state_dir,
)
from tests.unit.web.test_api_bundles import _pack, _run, client  # noqa: F401


@pytest.fixture
def in_project(project: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(project)
    return project


def _invoke(name: str, *args: str) -> Result:
    return CliRunner().invoke(cli, ["invoke", name, "hi", *args])


class TestInvokeByBundleName:
    def test_a_bundle_runs_by_name(self, in_project: Path, client: TestClient) -> None:
        assert _run(client, _pack(persist="global"))["persisted"] == "pack"

        result = _invoke("pack", "--output-format", "json")

        assert result.exit_code == 0, result.output
        document = json.loads(result.output)
        assert document["status"] == "success"
        assert _tag(document, "s") == "x"

    def test_a_name_no_tier_and_no_bundle_holds_prints_where_it_looked(
        self, in_project: Path
    ) -> None:
        result = _invoke("ghost")

        assert result.exit_code == 1
        assert "Ensemble 'ghost' not found in:" in result.output
        assert str(in_project / ".llm-orc" / "ensembles") in result.output

    def test_config_dir_stays_strict_and_reads_no_bundle(
        self, in_project: Path, client: TestClient, tmp_path: Path
    ) -> None:
        _run(client, _pack(persist="global"))
        only = tmp_path / "only-here"
        only.mkdir()

        result = _invoke("pack", "--config-dir", str(only))

        assert result.exit_code == 1
        assert f"Ensemble 'pack' not found in: {only}" in result.output


@pytest.fixture
def no_tier_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    """No ensemble tier directory exists. A real manager makes the global
    one on construction, so the state is injected at the one lookup."""
    monkeypatch.setattr(ConfigurationManager, "get_ensembles_dirs", lambda self: [])


class TestListEnsembles:
    def test_a_bundle_is_listed_apart_from_the_tiers(
        self, in_project: Path, client: TestClient
    ) -> None:
        _run(client, _pack(persist="global"))

        result = CliRunner().invoke(cli, ["list-ensembles"])

        assert result.exit_code == 0, result.output
        assert "Bundles" in result.output
        assert "pack: a closure" in result.output

    def test_a_bundle_is_listed_when_no_tier_directory_exists(
        self, in_project: Path, client: TestClient, no_tier_directory: None
    ) -> None:
        _run(client, _pack(persist="global"))

        result = CliRunner().invoke(cli, ["list-ensembles"])

        assert result.exit_code == 0, result.output
        assert "No ensemble directories found" not in result.output
        assert "pack: a closure" in result.output

    def test_no_tier_directory_and_no_bundle_still_says_so(
        self, in_project: Path, no_tier_directory: None
    ) -> None:

        result = CliRunner().invoke(cli, ["list-ensembles"])

        assert "No ensemble directories found." in result.output
