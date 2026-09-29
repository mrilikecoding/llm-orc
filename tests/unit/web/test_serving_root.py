"""The chat-completions caller is built on the serving root, not cwd (#196)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from llm_orc.web.api import v1_chat_completions
from llm_orc.web.server import create_app


@pytest.fixture
def empty_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.chdir(empty)
    monkeypatch.setattr(v1_chat_completions, "_SHARED_CALLERS", {})
    return empty


def test_factory_from_empty_cwd_uses_packaged_root_and_state_trace(
    empty_cwd: Path,
    packaged_serving_project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(empty_cwd.parent / "xdg"))

    caller = v1_chat_completions.get_serving_ensemble_caller()

    assert caller._project_dir == packaged_serving_project
    expected = empty_cwd.parent / "xdg" / "llm-orc" / ".serve-trace"
    assert caller._trace_root == expected
    assert caller._self_reference_enabled() is True  # packaged config.yaml says so
    assert caller._load_config().name == "serving"


def test_factory_in_a_project_with_the_ensemble_uses_the_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    packaged_serving_project: Path,
) -> None:
    project = tmp_path / "proj"
    shutil.copytree(packaged_serving_project, project / ".llm-orc")
    monkeypatch.chdir(project)
    monkeypatch.setattr(v1_chat_completions, "_SHARED_CALLERS", {})

    caller = v1_chat_completions.get_serving_ensemble_caller()

    assert caller._project_dir == project / ".llm-orc"
    assert caller._trace_root == project / ".llm-orc" / ".serve-trace"


def test_chat_completion_with_no_serving_ensemble_is_a_json_error(
    empty_cwd: Path,
) -> None:
    """Review Focus 5: the tier is disabled (suite default) and cwd is empty."""
    with TestClient(create_app(), raise_server_exceptions=False) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "x", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 500
    assert "serving.yaml" in response.json()["detail"]
