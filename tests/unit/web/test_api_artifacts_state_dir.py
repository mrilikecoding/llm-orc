"""Executing from an empty cwd writes artifacts to the state dir, not cwd (#196).

Through the REST execute route, the real OrchestraService and the real
executor: a script-only ensemble in the global tier runs, and its
artifact lands under $XDG_STATE_HOME/llm-orc/artifacts/.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import llm_orc.web.api as web_api
from llm_orc.core.config.config_manager import resolve_global_config_dir
from llm_orc.web.server import create_app

_SCRIPT = """\
import json, sys
print(json.dumps({"success": True, "data": {"ok": True}}))
"""

_ENSEMBLE = """\
name: state-probe
description: one script agent
agents:
  - name: probe
    script: probe.py
"""


@pytest.fixture
def empty_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.chdir(empty)
    monkeypatch.setattr(web_api, "_orchestra_service", None)
    return empty


def test_execute_writes_artifacts_under_the_state_dir(
    empty_cwd: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_home = Path(str(empty_cwd.parent / "state-home"))
    monkeypatch.setenv("XDG_STATE_HOME", str(state_home))
    global_dir = resolve_global_config_dir()
    (global_dir / "ensembles").mkdir(parents=True)
    (global_dir / "ensembles" / "state-probe.yaml").write_text(_ENSEMBLE)
    (global_dir / "scripts").mkdir()
    (global_dir / "scripts" / "probe.py").write_text(_SCRIPT)

    with TestClient(create_app()) as client:
        response = client.post(
            "/api/ensembles/state-probe/execute", json={"input": "go"}
        )

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "success", response.json()
    artifacts = state_home / "llm-orc" / "artifacts" / "state-probe"
    assert artifacts.is_dir()
    assert any(artifacts.iterdir())
    assert not (empty_cwd / ".llm-orc").exists()
