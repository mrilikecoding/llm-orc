"""scope over REST, through the real OrchestraService.

The assertions are files on disk and what the listing reports, so a
router that drops `scope` on the floor fails here even if every handler
unit test is green.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

import llm_orc.web.api as web_api
from llm_orc.core.config.config_manager import resolve_global_config_dir
from llm_orc.web.server import create_app

_AGENTS = [{"name": "writer", "model_profile": "local-qwen3-8b"}]


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / ".llm-orc" / "ensembles").mkdir(parents=True)
    (tmp_path / ".llm-orc" / "profiles").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(web_api, "_orchestra_service", None)
    return tmp_path


@pytest.fixture
def bare_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A cwd with no `.llm-orc` at all -- there is no project tier."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(web_api, "_orchestra_service", None)
    return tmp_path


class TestEnsembleScopeOverRest:
    def test_global_scope_lands_in_global_dir_and_lists_as_global(
        self, project: Path
    ) -> None:
        with TestClient(create_app()) as client:
            created = client.post(
                "/api/ensembles",
                json={"name": "remote-made", "agents": _AGENTS, "scope": "global"},
            )
            listed = client.get("/api/ensembles")

        assert created.status_code == 200, created.text
        global_file = resolve_global_config_dir() / "ensembles" / "remote-made.yaml"
        assert Path(created.json()["path"]) == global_file
        assert global_file.exists()
        assert not (project / ".llm-orc" / "ensembles" / "remote-made.yaml").exists()
        entry = next(e for e in listed.json() if e["name"] == "remote-made")
        assert entry["source"] == "global"

    def test_default_scope_lands_in_project(self, project: Path) -> None:
        with TestClient(create_app()) as client:
            created = client.post(
                "/api/ensembles", json={"name": "mine", "agents": _AGENTS}
            )
            listed = client.get("/api/ensembles")

        assert (
            Path(created.json()["path"])
            == project / ".llm-orc" / "ensembles" / "mine.yaml"
        )
        assert not (resolve_global_config_dir() / "ensembles" / "mine.yaml").exists()
        entry = next(e for e in listed.json() if e["name"] == "mine")
        assert entry["source"] == "local"

    def test_delete_with_scope_query_acts_on_that_scope_only(
        self, project: Path
    ) -> None:
        # raise_server_exceptions=False: this test pins the REST 500
        # mapping for a scope-mismatch ValueError, so the client must
        # receive that response instead of Starlette re-raising it.
        with TestClient(create_app(), raise_server_exceptions=False) as client:
            client.post("/api/ensembles", json={"name": "keep", "agents": _AGENTS})
            wrong = client.delete("/api/ensembles/keep", params={"scope": "global"})
            right = client.delete("/api/ensembles/keep", params={"scope": "project"})

        assert wrong.status_code != 200
        assert "local tier" in wrong.text
        assert right.status_code == 200
        assert not (project / ".llm-orc" / "ensembles" / "keep.yaml").exists()

    def test_invalid_scope_is_rejected_by_the_request_model(
        self, project: Path
    ) -> None:
        with TestClient(create_app()) as client:
            response = client.post(
                "/api/ensembles",
                json={"name": "nope", "agents": _AGENTS, "scope": "local"},
            )

        assert response.status_code == 422
        assert not (project / ".llm-orc" / "ensembles" / "nope.yaml").exists()

    def test_default_scope_without_project_dir_is_rejected(
        self, bare_project: Path
    ) -> None:
        # raise_server_exceptions=False: the ValueError from having no
        # project tier maps to a 500 today, same as the scope-mismatch
        # delete case above.
        with TestClient(create_app(), raise_server_exceptions=False) as client:
            created = client.post(
                "/api/ensembles", json={"name": "orphan", "agents": _AGENTS}
            )

        assert created.status_code != 200
        assert not (resolve_global_config_dir() / "ensembles" / "orphan.yaml").exists()


class TestProfileScopeOverRest:
    def test_global_scope_lands_in_global_dir(self, project: Path) -> None:
        with TestClient(create_app()) as client:
            created = client.post(
                "/api/profiles",
                json={
                    "name": "remote-prof",
                    "provider": "llama-server",
                    "model": "qwen3-8b",
                    "scope": "global",
                },
            )

        global_file = resolve_global_config_dir() / "profiles" / "remote-prof.yaml"
        assert Path(created.json()["path"]) == global_file
        assert yaml.safe_load(global_file.read_text())["model"] == "qwen3-8b"
        assert not (project / ".llm-orc" / "profiles" / "remote-prof.yaml").exists()

    def test_update_and_delete_carry_scope(self, project: Path) -> None:
        # raise_server_exceptions=False: same reason as the ensemble
        # delete-scope test above.
        with TestClient(create_app(), raise_server_exceptions=False) as client:
            client.post(
                "/api/profiles",
                json={
                    "name": "p",
                    "provider": "llama-server",
                    "model": "a",
                    "scope": "global",
                },
            )
            updated = client.put(
                "/api/profiles/p", json={"model": "b", "scope": "global"}
            )
            wrong = client.delete("/api/profiles/p", params={"scope": "project"})
            right = client.delete("/api/profiles/p", params={"scope": "global"})

        global_file = resolve_global_config_dir() / "profiles" / "p.yaml"
        assert updated.status_code == 200
        assert wrong.status_code != 200
        assert "global tier" in wrong.text
        assert right.status_code == 200
        assert not global_file.exists()
