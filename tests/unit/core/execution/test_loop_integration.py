"""Integration: a loop node re-runs a body ensemble in the real executor."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest
import yaml

from llm_orc.core.config.ensemble_config import EnsembleLoader
from llm_orc.core.execution.executor_factory import ExecutorFactory


async def _run_loop(
    body_agents: list[dict[str, Any]], loop_spec: dict[str, Any]
) -> dict[str, Any]:
    """Run a parent ensemble whose single node is a loop over a body ensemble."""
    with tempfile.TemporaryDirectory() as td:
        proj = Path(td)
        ens = proj / "ensembles"
        ens.mkdir()
        (ens / "attempt.yaml").write_text(
            yaml.dump({"name": "attempt", "description": "body", "agents": body_agents})
        )
        parent_path = ens / "parent.yaml"
        parent_path.write_text(
            yaml.dump(
                {
                    "name": "parent",
                    "description": "loop parent",
                    "agents": [{"name": "loop-node", "loop": loop_spec}],
                }
            )
        )
        config = EnsembleLoader().load_from_file(str(parent_path))
        executor = ExecutorFactory.create_root_executor(project_dir=proj)
        mock_artifact = Mock()
        mock_artifact.save_execution_results = Mock()
        with patch.object(executor, "_artifact_manager", mock_artifact):
            result: dict[str, Any] = await executor.execute(config, "go")
        outcome: dict[str, Any] = json.loads(result["results"]["loop-node"]["response"])
        return outcome


@pytest.mark.asyncio
async def test_loop_terminates_on_until() -> None:
    outcome = await _run_loop(
        body_agents=[{"name": "emit", "script": "echo '{\"ok\": true}'"}],
        loop_spec={"body": "attempt", "until": "${ok}", "max_iterations": 3},
    )
    assert outcome["terminated"] == "until"
    assert outcome["iterations"] == 1


@pytest.mark.asyncio
async def test_loop_exhausts_when_until_never_true() -> None:
    outcome = await _run_loop(
        body_agents=[{"name": "emit", "script": "echo '{\"ok\": false}'"}],
        loop_spec={"body": "attempt", "until": "${ok}", "max_iterations": 2},
    )
    assert outcome["terminated"] == "exhausted"
    assert outcome["iterations"] == 2
