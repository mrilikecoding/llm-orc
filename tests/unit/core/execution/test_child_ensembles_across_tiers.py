"""A global ensemble reaches a packaged child and its packaged script (#196).

Through the real executor: `_resolve_ensemble_reference` used to stop at
the local dot-dir, so research-dossier (global) could not reach
agentic-serving/web-searcher (packaged) on a plain-dir serve.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.core.config.ensemble_config import EnsembleLoader
from llm_orc.core.execution.executor_factory import ExecutorFactory

_PARENT = """\
name: parent
description: global parent, packaged child
agents:
  - name: kid
    ensemble: agentic-serving/child
"""


@pytest.fixture
def parent_in_global(
    tmp_path: Path,
    packaged_serving_project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ConfigurationManager, Path]:
    # an empty cwd, so no project-local scripts/ can shadow the packaged one
    (tmp_path / "noproj").mkdir()
    monkeypatch.chdir(tmp_path / "noproj")
    cm = ConfigurationManager(project_dir=tmp_path / "noproj", provision=False)
    ensembles = cm.global_config_dir / "ensembles"
    ensembles.mkdir(parents=True)
    path = ensembles / "parent.yaml"
    path.write_text(_PARENT)
    return cm, path


def test_child_resolves_from_the_packaged_tier(
    parent_in_global: tuple[ConfigurationManager, Path],
) -> None:
    cm, _ = parent_in_global
    executor = ExecutorFactory.create_root_executor(
        config_manager=cm, save_artifacts=False
    )

    child = executor._resolve_ensemble_reference("agentic-serving/child")

    assert child.name == "child"


async def test_parent_runs_the_packaged_child_and_its_packaged_script(
    parent_in_global: tuple[ConfigurationManager, Path],
    packaged_serving_project: Path,
) -> None:
    before = sorted(
        p.relative_to(packaged_serving_project)
        for p in packaged_serving_project.rglob("*")
    )
    cm, path = parent_in_global
    executor = ExecutorFactory.create_root_executor(
        config_manager=cm, save_artifacts=False
    )
    config = EnsembleLoader().load_from_file(str(path))

    result = await executor.execute(config, "ping")

    assert result["status"] == "completed", result
    assert not result["has_errors"], result
    child = json.loads(result["results"]["kid"]["response"])
    # The fixture script's own payload: only it emits {"data": {"echo": ...}},
    # and it echoes the JSON envelope it read on stdin.
    payload = json.loads(child["results"]["echo"]["response"])
    assert json.loads(payload["data"]["echo"])["input"] == "ping"
    # the script imported its sibling helper, and that import left no
    # bytecode under the read-only packaged tier
    assert payload["data"]["tag"] == "helper"
    after = sorted(
        p.relative_to(packaged_serving_project)
        for p in packaged_serving_project.rglob("*")
    )
    assert after == before
