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
        with TestClient(create_app(), raise_server_exceptions=False) as client:
            client.post("/api/ensembles", json={"name": "keep", "agents": _AGENTS})
            wrong = client.delete("/api/ensembles/keep", params={"scope": "global"})
            right = client.delete("/api/ensembles/keep", params={"scope": "project"})

        assert wrong.status_code == 400
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
        with TestClient(create_app(), raise_server_exceptions=False) as client:
            created = client.post(
                "/api/ensembles", json={"name": "orphan", "agents": _AGENTS}
            )

        assert created.status_code == 400
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
            global_file = resolve_global_config_dir() / "profiles" / "p.yaml"
            updated_yaml = yaml.safe_load(global_file.read_text())

            wrong = client.delete("/api/profiles/p", params={"scope": "project"})
            right = client.delete("/api/profiles/p", params={"scope": "global"})

        assert updated.status_code == 200
        assert updated_yaml["model"] == "b"
        assert "scope" not in updated_yaml
        assert wrong.status_code != 200
        assert "global tier" in wrong.text
        assert right.status_code == 200
        assert not global_file.exists()


_SHAPE = "name: shape\ndescription: d\nagents:\n  - name: a\n    model_profile: p\n"


class TestEnsembleRefusalStatusOverRest:
    """A refusal the caller caused is a client error, with the message."""

    def _ensembles_dir(self, project: Path) -> Path:
        return project / ".llm-orc" / "ensembles"

    def test_unknown_changes_key_is_400(self, project: Path) -> None:
        (self._ensembles_dir(project) / "shape.yaml").write_text(_SHAPE)

        with TestClient(create_app(), raise_server_exceptions=False) as client:
            response = client.put(
                "/api/ensembles/shape",
                json={"changes": {"rename": "x"}, "dry_run": False},
            )

        assert response.status_code == 400
        assert "Unsupported" in response.text

    def test_invalid_result_is_400(self, project: Path) -> None:
        (self._ensembles_dir(project) / "shape.yaml").write_text(_SHAPE)

        with TestClient(create_app(), raise_server_exceptions=False) as client:
            response = client.put(
                "/api/ensembles/shape",
                json={"changes": {"remove_agents": ["nope"]}, "dry_run": False},
            )

        assert response.status_code == 400
        assert "no agent named" in response.text

    def test_symlinked_target_is_400_for_put_and_delete(self, project: Path) -> None:
        shapes = self._ensembles_dir(project) / "agentic-serving"
        shapes.mkdir()
        (shapes / "shape.yaml").write_text(_SHAPE)
        (self._ensembles_dir(project) / "shape.yaml").symlink_to(
            Path("agentic-serving") / "shape.yaml"
        )

        with TestClient(create_app(), raise_server_exceptions=False) as client:
            put = client.put(
                "/api/ensembles/shape",
                json={"changes": {"add_agents": []}, "dry_run": False},
            )
            delete = client.delete("/api/ensembles/shape")

        assert put.status_code == 400
        assert delete.status_code == 400
        assert "managed by the project" in delete.text
        assert (self._ensembles_dir(project) / "shape.yaml").is_symlink()

    def test_unknown_ensemble_is_404_for_put_and_delete(self, project: Path) -> None:
        with TestClient(create_app(), raise_server_exceptions=False) as client:
            put = client.put("/api/ensembles/ghost", json={"changes": {}})
            delete = client.delete("/api/ensembles/ghost")

        assert put.status_code == 404
        assert delete.status_code == 404
        assert "not found" in delete.text

    def test_traversal_names_are_client_errors(self, project: Path) -> None:
        outside = project / ".llm-orc" / "outside.yaml"
        outside.write_text(_SHAPE)

        with TestClient(create_app(), raise_server_exceptions=False) as client:
            put = client.put(
                "/api/ensembles/..%2Foutside",
                json={"changes": {"remove_agents": ["a"]}, "dry_run": False},
            )
            delete = client.delete("/api/ensembles/..%2Foutside")
            created = client.post(
                "/api/ensembles", json={"name": "../fresh", "agents": _AGENTS}
            )

        # An encoded slash may not even route (405/404); it must never be
        # a 500 or a success.
        assert put.status_code in (400, 404, 405)
        assert delete.status_code in (400, 404, 405)
        assert created.status_code == 400
        assert outside.read_text() == _SHAPE
        assert not (project / ".llm-orc" / "fresh.yaml").exists()

    def test_an_over_long_name_is_400_not_500(self, project: Path) -> None:
        long_name = "a" * 4000

        with TestClient(create_app(), raise_server_exceptions=False) as client:
            created = client.post(
                "/api/ensembles", json={"name": long_name, "agents": _AGENTS}
            )
            put = client.put(f"/api/ensembles/{long_name}", json={"changes": {}})
            delete = client.delete(f"/api/ensembles/{long_name}")

        assert created.status_code == 400
        assert put.status_code == 400
        assert delete.status_code == 400
        assert str(project) not in created.text + put.text + delete.text

    def test_create_into_a_new_subdirectory_makes_the_directory(
        self, project: Path
    ) -> None:
        with TestClient(create_app(), raise_server_exceptions=False) as client:
            created = client.post(
                "/api/ensembles", json={"name": "sub/inner", "agents": _AGENTS}
            )

        assert created.status_code == 200, created.text
        written = self._ensembles_dir(project) / "sub" / "inner.yaml"
        assert written.exists()
