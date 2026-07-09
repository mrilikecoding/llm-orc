"""Integration: a dynamic-dispatch node routes to a runtime-resolved ensemble.

Exercises the whole primitive through the real executor: a classify node emits a
target name, and the dispatch node resolves ``${classify.target}`` at the phase
layer and executes that ensemble as a child. The swap test is the spike's
swap-ability probe: the same serve skeleton routes to two structurally different
seats with zero change to the skeleton.
"""

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

# Two structurally different seats behind the same contract (produce a result).
# seat-a: a single node ("single capable model" strategy).
# seat-b: build -> verify, two dependent nodes ("verify-heavy" strategy).
_SEAT_A = [{"name": "a-out", "script": "echo from-seat-a"}]
_SEAT_B = [
    {"name": "b-build", "script": "echo built"},
    {"name": "b-verify", "script": "echo verified", "depends_on": ["b-build"]},
]


async def _run_dispatch(target: str) -> dict[str, Any]:
    """Run the serve skeleton with the classifier routing to ``target``."""
    with tempfile.TemporaryDirectory() as td:
        proj = Path(td)
        ens = proj / "ensembles"
        ens.mkdir()
        (ens / "seat-a.yaml").write_text(
            yaml.dump({"name": "seat-a", "description": "seat a", "agents": _SEAT_A})
        )
        (ens / "seat-b.yaml").write_text(
            yaml.dump({"name": "seat-b", "description": "seat b", "agents": _SEAT_B})
        )
        serve_path = ens / "serve.yaml"
        serve_path.write_text(
            yaml.dump(
                {
                    "name": "serve",
                    "description": "classify then dispatch to a seat",
                    "agents": [
                        {
                            "name": "classify",
                            "script": f"echo '{json.dumps({'target': target})}'",
                        },
                        {
                            "name": "seat",
                            "dispatch": "${classify.target}",
                            "depends_on": ["classify"],
                        },
                    ],
                }
            )
        )
        config = EnsembleLoader().load_from_file(str(serve_path))
        executor = ExecutorFactory.create_root_executor(project_dir=proj)
        mock_artifact = Mock()
        mock_artifact.save_execution_results = Mock()
        with patch.object(executor, "_artifact_manager", mock_artifact):
            result: dict[str, Any] = await executor.execute(config, "go")
        return result


def _seat_child_results(result: dict[str, Any]) -> dict[str, Any]:
    """The dispatched seat's child execution results."""
    seat = result["results"]["seat"]
    assert seat["status"] == "success", seat
    child: dict[str, Any] = json.loads(seat["response"])
    return dict(child["results"])


@pytest.mark.asyncio
async def test_dispatches_to_resolved_seat() -> None:
    result = await _run_dispatch("seat-a")
    child_results = _seat_child_results(result)
    assert "a-out" in child_results


@pytest.mark.asyncio
async def test_swap_seat_at_zero_skeleton_change() -> None:
    """Same serve skeleton; only the routed target differs. The dispatch fills
    the seat with either structurally different ensemble, no skeleton edit."""
    to_a = _seat_child_results(await _run_dispatch("seat-a"))
    to_b = _seat_child_results(await _run_dispatch("seat-b"))

    assert "a-out" in to_a
    assert "b-verify" not in to_a
    assert "b-verify" in to_b
    assert "a-out" not in to_b
