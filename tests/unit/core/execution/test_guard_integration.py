"""Integration: the guard predicate skips a node in the real executor."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest
import yaml

from llm_orc.core.config.ensemble_config import EnsembleLoader
from llm_orc.core.execution.executor_factory import ExecutorFactory


async def _run(ensemble_yaml: dict[str, Any]) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(ensemble_yaml, f)
        yaml_path = f.name
    try:
        config = EnsembleLoader().load_from_file(yaml_path)
        executor = ExecutorFactory.create_root_executor()
        mock_artifact = Mock()
        mock_artifact.save_execution_results = Mock()
        with patch.object(executor, "_artifact_manager", mock_artifact):
            result: dict[str, Any] = await executor.execute(config, "go")
        return result
    finally:
        Path(yaml_path).unlink()


@pytest.mark.asyncio
async def test_guarded_node_skipped_when_predicate_false() -> None:
    """A node whose `when` is false does not execute; it is recorded skipped."""
    result = await _run(
        {
            "name": "guard-skip-test",
            "description": "guard skips a node",
            "agents": [
                {"name": "gate", "script": "echo '{\"ok\": false}'"},
                {
                    "name": "build",
                    "script": "echo built",
                    "depends_on": ["gate"],
                    "when": "${gate.ok}",
                },
            ],
        }
    )
    assert result["results"]["gate"]["status"] == "success"
    assert result["results"]["build"]["status"] == "skipped"


@pytest.mark.asyncio
async def test_guarded_node_runs_when_predicate_true() -> None:
    """A node whose `when` is true executes normally."""
    result = await _run(
        {
            "name": "guard-run-test",
            "description": "guard lets a node run",
            "agents": [
                {"name": "gate", "script": "echo '{\"ok\": true}'"},
                {
                    "name": "build",
                    "script": "echo built",
                    "depends_on": ["gate"],
                    "when": "${gate.ok}",
                },
            ],
        }
    )
    assert result["results"]["build"]["status"] == "success"


@pytest.mark.asyncio
async def test_route_and_judge_branches_then_joins() -> None:
    """Router picks a branch; the untaken branch skips; the join runs on
    whichever branch fired. This is the route-and-judge pattern end to end."""
    result = await _run(
        {
            "name": "route-and-judge",
            "description": "branch on a router choice, then join",
            "agents": [
                {"name": "router", "script": 'echo \'{"choice": "code"}\''},
                {
                    "name": "code-branch",
                    "script": "echo coded",
                    "depends_on": ["router"],
                    "when": '${router.choice} == "code"',
                },
                {
                    "name": "prose-branch",
                    "script": "echo prosed",
                    "depends_on": ["router"],
                    "when": '${router.choice} == "prose"',
                },
                {
                    "name": "join",
                    "script": "echo joined",
                    "depends_on": ["code-branch", "prose-branch"],
                },
            ],
        }
    )
    assert result["results"]["code-branch"]["status"] == "success"
    assert result["results"]["prose-branch"]["status"] == "skipped"
    # the join runs because at least one branch fired
    assert result["results"]["join"]["status"] == "success"
