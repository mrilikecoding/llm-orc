"""A run's result carries the executor's own ``metadata`` (Arc 5, Task 4,
ruling 5) on every entry path, so the CLI can render a remote run as it
renders a local one. Additive: the other keys stay."""

# ruff: noqa: F811  (imported fixtures are redefined as test parameters)
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.unit.services.test_one_run_injection import (  # noqa: F401
    _ensemble,
    listing,
    project,
    service,
    state_dir,
)
from tests.unit.web.test_api_run_injection import (  # noqa: F401
    NAMED_SURFACES,
    _caller,
    client,
)


def _run(client: TestClient, project: Path, surface: str) -> dict[str, Any]:
    _ensemble(project / ".llm-orc", "top", [{"name": "s", "script": "echo hi"}])
    return _caller(client, surface)("top", {"input": "hi"})


@pytest.mark.parametrize("surface", NAMED_SURFACES)
def test_the_result_carries_the_usage_and_duration_a_run_produced(
    surface: str, client: TestClient, project: Path
) -> None:
    result = _run(client, project, surface)

    assert result["status"] == "success", result
    metadata = result["metadata"]
    assert {"total_tokens", "total_cost_usd"} <= set(metadata["usage"]["totals"])
    assert re.fullmatch(r"\d+\.\d+s", metadata["duration"])
    assert {"results", "deliverable", "has_errors"} <= set(result)


def test_a_refusal_envelope_is_unchanged(client: TestClient, project: Path) -> None:
    _ensemble(project / ".llm-orc", "top", [{"name": "w", "model_profile": "gone"}])

    result = client.post("/api/ensembles/top/execute", json={"input": "hi"}).json()

    assert set(result) == {"status", "has_errors", "results", "deliverable", "error"}
