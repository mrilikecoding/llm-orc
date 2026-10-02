"""Bundles and ``persist`` through REST, over one real OrchestraService on
temp project, state and global dirs (Arc 5, Task 9, ruling 9).

A persisted closure is a stored run request: written after the gate
passes, replayed through the same injection path on each named run.
"""

# ruff: noqa: F811  (imported fixtures are redefined as test parameters)
from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import llm_orc.web.api as web_api
from llm_orc.core.config.config_manager import resolve_global_config_dir
from llm_orc.services.orchestra_service import OrchestraService
from llm_orc.web.server import create_app
from tests.unit.services.test_one_run_injection import (  # noqa: F401
    _ensemble,
    _kinds,
    _profile,
    _runs,
    _script_source,
    _tag,
    _trees,
    listing,
    loaded_models,
    project,
    service,
    state_dir,
)

PACK: dict[str, Any] = {
    "name": "pack",
    "description": "a closure",
    "agents": [
        {"name": "s", "script": "tools/x.py"},
        {"name": "k", "ensemble": "kid"},
    ],
}
KID: dict[str, Any] = {
    "description": "kid",
    "agents": [{"name": "c", "script": "tools/y.py"}],
}


@pytest.fixture
def client(
    service: OrchestraService, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    monkeypatch.setattr(web_api, "_orchestra_service", service)
    with TestClient(create_app()) as test_client:
        yield test_client


def _pack(**more: Any) -> dict[str, Any]:
    """A run request for the closure ``pack``: a script, and a child with
    a script of its own."""
    return {
        "ensemble": PACK,
        "ensembles": {"kid": KID},
        "scripts": {
            "tools/x.py": _script_source("x"),
            "tools/y.py": _script_source("y"),
        },
        "input": "hi",
        **more,
    }


def _run(client: TestClient, body: dict[str, Any]) -> dict[str, Any]:
    response = client.post("/api/ensembles/execute", json=body)
    assert response.status_code == 200, response.text
    return dict(response.json())


def _run_named(client: TestClient, name: str, **more: Any) -> dict[str, Any]:
    return _run(client, {"ensemble_name": name, "input": "hi", **more})


def _warm(client: TestClient, project: Path) -> None:
    """The first run of any kind makes the serve's credential storage:
    serve state, not the bundle's."""
    _ensemble(project / ".llm-orc", "warm", [{"name": "s", "script": "echo hi"}])
    assert _run_named(client, "warm")["status"] == "success"


def _bundle_file(name: str = "pack") -> Path:
    return resolve_global_config_dir() / "bundles" / f"{name}.json"


class TestPersist:
    def test_a_persisting_run_stores_the_closure_keeps_its_artifact_and_says_so(
        self, client: TestClient, project: Path, state_dir: Path
    ) -> None:
        _warm(client, project)
        before = _trees(project, state_dir)

        result = _run(client, _pack(persist="global", pull=False))

        assert result["status"] == "success", result
        assert result["persisted"] == "pack"
        assert _tag(result, "s") == "x"
        stored = json.loads(_bundle_file().read_text())
        assert set(stored) == {"ensemble", "ensembles", "profiles", "scripts", "bind"}
        assert stored["ensemble"] == PACK
        assert stored["scripts"]["tools/y.py"] == _script_source("y")
        assert list((state_dir / "artifacts" / "pack").iterdir())
        assert _runs(state_dir) == []
        after = _trees(project, state_dir)
        assert after["global"].keys() - before["global"].keys() == {"bundles/pack.json"}
        assert after["project"] == before["project"]

    def test_a_run_that_does_not_persist_says_nothing_of_it(
        self, client: TestClient
    ) -> None:
        result = _run(client, _pack())

        assert "persisted" not in result
        assert not _bundle_file().exists()

    def test_the_stored_request_has_no_input_pull_or_persist(
        self, client: TestClient
    ) -> None:
        _run(client, _pack(persist="global", input="secret topic"))

        assert "secret topic" not in _bundle_file().read_text()

    def test_a_refused_gate_writes_nothing(self, client: TestClient) -> None:
        refused = _pack(persist="global")
        refused["ensemble"] = {
            **PACK,
            "agents": [*PACK["agents"], {"name": "w", "model_profile": "nope"}],
        }

        result = _run(client, refused)

        assert result["error"]["kind"] == "not_equipped"
        assert not _bundle_file().exists()
        assert not _bundle_file().parent.exists()

    def test_a_refused_gate_leaves_an_existing_bundle_byte_for_byte(
        self, client: TestClient
    ) -> None:
        _run(client, _pack(persist="global"))
        before = _bundle_file().read_bytes()
        refused = _pack(persist="global", scripts={})

        result = _run(client, refused)

        assert result["error"]["kind"] == "not_equipped"
        assert _bundle_file().read_bytes() == before

    def test_an_invalid_request_writes_nothing_and_keeps_the_bundle(
        self, client: TestClient
    ) -> None:
        _run(client, _pack(persist="global"))
        before = _bundle_file().read_bytes()
        invalid = _pack(persist="global", scripts={"date": "x"})

        result = _run(client, invalid)

        assert result["error"]["kind"] == "invalid_request"
        assert _bundle_file().read_bytes() == before

    def test_persisting_again_replaces_the_bundle(self, client: TestClient) -> None:
        _run(client, _pack(persist="global"))
        newer = _pack(
            persist="global",
            scripts={
                "tools/x.py": _script_source("x2"),
                "tools/y.py": _script_source("y"),
            },
        )

        _run(client, newer)

        stored = json.loads(_bundle_file().read_text())
        assert stored["scripts"]["tools/x.py"] == _script_source("x2")

    def test_persist_needs_an_inline_root(self, client: TestClient) -> None:
        body = {"ensemble_name": "pack", "persist": "global", "input": "hi"}
        result = _run(client, body)

        assert result["error"]["kind"] == "invalid_request"
        assert "inline root" in result["error"]["message"]

    def test_the_only_scope_is_global(self, client: TestClient) -> None:
        response = client.post("/api/ensembles/execute", json=_pack(persist="project"))

        assert response.status_code == 422

    def test_a_root_name_with_a_slash_is_invalid(self, client: TestClient) -> None:
        nested = _pack(persist="global")
        nested["ensemble"] = {**PACK, "name": "group/pack"}

        result = _run(client, nested)

        assert result["error"]["kind"] == "invalid_request"
        assert not _bundle_file("group").exists()

    def test_the_named_route_does_not_take_persist(self, client: TestClient) -> None:
        response = client.post(
            "/api/ensembles/pack/execute", json={"input": "hi", "persist": "global"}
        )

        assert response.status_code == 422

    def test_a_failure_between_the_temp_file_and_the_rename_keeps_the_bundle(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _run(client, _pack(persist="global"))
        before = _bundle_file().read_bytes()

        def no_rename(src: Any, dst: Any) -> None:
            raise OSError(28, "No space left on device")

        replacement = _pack(
            persist="global",
            scripts={
                "tools/x.py": _script_source("x2"),
                "tools/y.py": _script_source("y"),
            },
        )
        with monkeypatch.context() as failing:
            failing.setattr(os, "replace", no_rename)
            result = _run(client, replacement)

        assert result["error"]["kind"] == "invalid_request"
        assert "No space" in result["error"]["message"]
        assert [p.name for p in _bundle_file().parent.iterdir()] == ["pack.json"]
        assert _bundle_file().read_bytes() == before


class TestAPersistedNameIsNeverShadowed:
    @pytest.mark.parametrize("tier", ["local", "global"])
    def test_a_name_a_tier_resolves_is_refused_naming_the_tier(
        self, client: TestClient, project: Path, tier: str
    ) -> None:
        base = project / ".llm-orc" if tier == "local" else resolve_global_config_dir()
        _ensemble(base, "pack", [{"name": "s", "script": "echo hi"}])

        result = _run(client, _pack(persist="global"))

        assert result["error"]["kind"] == "invalid_request"
        assert tier in result["error"]["message"]
        assert not _bundle_file().exists()
