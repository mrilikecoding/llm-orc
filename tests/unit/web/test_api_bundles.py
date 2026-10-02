"""Bundles and ``persist`` through REST, over one real OrchestraService on
temp project, state and global dirs (Arc 5, Task 9, ruling 9).

A persisted closure is a stored run request: written after the gate
passes, replayed through the same injection path on each named run.
"""

# ruff: noqa: F811  (imported fixtures are redefined as test parameters)
from __future__ import annotations

import json
import os
import threading
import time
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
    _yaml,
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


def _persisted(client: TestClient, **more: Any) -> dict[str, Any]:
    """Persist ``pack`` and return the persisting run's result."""
    result = _run(client, _pack(persist="global", **more))
    assert result["persisted"] == "pack", result
    return result


class TestRunByName:
    @pytest.mark.parametrize("route", ["execute", "named"])
    def test_no_injections_gives_the_persisting_runs_result_and_writes_nothing(
        self, route: str, client: TestClient, project: Path, state_dir: Path
    ) -> None:
        _warm(client, project)
        first = _persisted(client)
        stored = _bundle_file().read_bytes()
        before = _trees(project, state_dir)

        if route == "named":
            response = client.post("/api/ensembles/pack/execute", json={"input": "hi"})
            assert response.status_code == 200, response.text
            second = dict(response.json())
        else:
            second = _run_named(client, "pack")

        assert second["status"] == "success", second
        assert second["status"] == first["status"]
        assert _tag(second, "s") == _tag(first, "s") == "x"
        assert second["results"]["s"]["response"] == first["results"]["s"]["response"]
        assert set(second["results"]) == set(first["results"])
        assert _bundle_file().read_bytes() == stored
        assert _runs(state_dir) == []
        after = _trees(project, state_dir)
        assert after["global"] == before["global"]
        assert after["project"] == before["project"]
        kept = [
            p
            for p in (state_dir / "artifacts" / "pack").iterdir()
            if p.is_dir() and not p.is_symlink()
        ]
        assert len(kept) == 2

    def test_a_same_key_injection_replaces_the_bundles_and_the_bundle_is_unchanged(
        self, client: TestClient
    ) -> None:
        _persisted(client)
        stored = _bundle_file().read_bytes()

        result = _run_named(
            client, "pack", scripts={"tools/x.py": _script_source("override")}
        )

        assert _tag(result, "s") == "override"
        assert _bundle_file().read_bytes() == stored
        assert _tag(_run_named(client, "pack"), "s") == "x"

    def test_an_injection_with_a_new_script_and_child_is_added_for_that_run_only(
        self, client: TestClient
    ) -> None:
        _persisted(client)
        extra = {
            "ensembles": {"extra": {"description": "e", "agents": []}},
            "scripts": {"tools/z.py": _script_source("z")},
        }

        result = _run_named(client, "pack", **extra)

        assert result["status"] == "success", result
        assert json.loads(_bundle_file().read_text())["scripts"].keys() == {
            "tools/x.py",
            "tools/y.py",
        }

    def test_a_script_meeting_a_bundle_script_by_another_spelling_is_invalid(
        self, client: TestClient
    ) -> None:
        root = {
            "name": "pack",
            "description": "p",
            "agents": [{"name": "s", "script": "scripts/x.py"}],
        }
        persisted = _run(
            client,
            {
                "ensemble": root,
                "scripts": {"scripts/x.py": _script_source("x")},
                "input": "hi",
                "persist": "global",
            },
        )
        assert persisted["persisted"] == "pack", persisted

        result = _run_named(client, "pack", scripts={"x.py": _script_source("mine")})

        assert result["error"]["kind"] == "invalid_request", result
        assert "x.py" in result["error"]["message"]
        assert result["results"] == {}

    def test_a_child_injection_that_meets_a_bundle_child_by_case_is_invalid(
        self, client: TestClient
    ) -> None:
        _persisted(client)

        result = _run_named(
            client, "pack", ensembles={"Kid": {"description": "k", "agents": []}}
        )

        assert result["error"]["kind"] == "invalid_request", result
        assert "Kid" in result["error"]["message"]
        assert result["results"] == {}

    def test_a_tier_file_added_later_under_the_bundles_name_wins_the_lookup(
        self, client: TestClient, project: Path
    ) -> None:
        _persisted(client)
        _ensemble(project / ".llm-orc", "pack", [{"name": "only", "script": "echo hi"}])

        result = _run_named(client, "pack")

        assert set(result["results"]) == {"only"}

    def test_a_bundle_that_does_not_parse_is_invalid_request_naming_it(
        self, client: TestClient
    ) -> None:
        _persisted(client)
        _bundle_file().write_text("{not json")

        result = _run_named(client, "pack")

        assert result["error"]["kind"] == "invalid_request"
        assert "bundle 'pack'" in result["error"]["message"]

    def test_a_bundle_that_does_not_validate_is_invalid_request_naming_it(
        self, client: TestClient
    ) -> None:
        _persisted(client)
        _bundle_file().write_text(
            json.dumps({"ensemble": PACK, "scripts": {"date": "x"}})
        )

        result = _run_named(client, "pack")

        assert result["error"]["kind"] == "invalid_request"
        assert "bundle 'pack'" in result["error"]["message"]

    def test_a_name_no_tier_and_no_bundle_holds_still_raises(
        self, client: TestClient
    ) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            _run_named(client, "ghost")

    def test_persist_is_refused_on_a_named_run_of_a_bundle(
        self, client: TestClient
    ) -> None:
        _persisted(client)

        result = _run_named(client, "pack", persist="global")

        assert result["error"]["kind"] == "invalid_request"
        assert "inline root" in result["error"]["message"]


class TestABundleDoesNotLeak:
    def test_a_host_ensemble_cannot_reach_a_child_or_script_only_a_bundle_holds(
        self, client: TestClient, project: Path
    ) -> None:
        _persisted(client)
        _ensemble(
            project / ".llm-orc",
            "host",
            [
                {"name": "k", "ensemble": "kid"},
                {"name": "s", "script": "tools/x.py"},
            ],
        )

        result = _run_named(client, "host")

        assert result["error"]["kind"] == "not_equipped", result
        kinds = _kinds(result)
        assert kinds["kid"] == "missing_ensemble"
        assert kinds["tools/x.py"] == "missing_script"

    def test_the_services_own_view_lists_no_bundle_content_after_a_bundle_run(
        self, client: TestClient, service: OrchestraService, project: Path
    ) -> None:
        _persisted(
            client,
            profiles={"pack-seat": {"provider": "llama-server", "model": "mock-a"}},
        )
        _run_named(client, "pack")
        manager = service.config_manager

        assert service.find_ensemble_by_name("pack") is None
        assert service.find_child_ensemble("kid") is None
        assert manager.run_layer_dir is None
        assert "pack-seat" not in manager.get_model_profiles()
        listed = {
            p.stem for d in manager.get_ensembles_dirs() for p in d.rglob("*.yaml")
        }
        assert not listed & {"pack", "kid"}
        assert not any(
            (d / "tools").exists() or (d / "scripts" / "tools").exists()
            for d in manager.get_ensembles_dirs()
        )
        assert [r for r in (project / ".llm-orc").rglob("*") if "pack" in r.name] == []


ROLES: dict[str, Any] = {
    "name": "roles",
    "description": "one role",
    "agents": [{"name": "w", "model_profile": "a"}],
}


class TestStoredBindings:
    @pytest.fixture
    def persisted_roles(
        self, client: TestClient, project: Path, loaded_models: list[str]
    ) -> list[str]:
        _profile(project / ".llm-orc", "b", model="mock-other")
        _profile(project / ".llm-orc", "c", model="mock-b")
        result = _run(
            client,
            {
                "ensemble": ROLES,
                "bind": {"a": "b"},
                "input": "hi",
                "persist": "global",
            },
        )
        assert result["persisted"] == "roles", result
        loaded_models.clear()
        return loaded_models

    def test_a_stored_bind_applies_on_a_named_run(
        self, client: TestClient, persisted_roles: list[str]
    ) -> None:
        result = _run_named(client, "roles")

        assert result["status"] == "success", result
        assert result["bindings"] == {"a": "b"}
        assert persisted_roles == ["mock-other"]

    def test_a_per_run_bind_on_the_same_role_overrides_it(
        self, client: TestClient, persisted_roles: list[str]
    ) -> None:
        result = _run_named(client, "roles", bind={"a": "c"})

        assert result["status"] == "success", result
        assert result["bindings"] == {"a": "c"}
        assert persisted_roles == ["mock-b"]

    def test_a_per_run_inline_profile_for_the_role_overrides_it(
        self, client: TestClient, persisted_roles: list[str]
    ) -> None:
        inline = {"a": {"provider": "llama-server", "model": "mock-a"}}

        result = _run_named(client, "roles", profiles=inline)

        assert result["status"] == "success", result
        assert "bindings" not in result
        assert persisted_roles == ["mock-a"]

    def test_the_stored_bind_is_still_there_afterwards(
        self, client: TestClient, persisted_roles: list[str]
    ) -> None:
        _run_named(client, "roles", bind={"a": "c"})

        assert json.loads(_bundle_file("roles").read_text())["bind"] == {"a": "b"}
        assert _run_named(client, "roles")["bindings"] == {"a": "b"}


def _sleeping(tag: str, started: Path, seconds: float) -> str:
    return (
        "import json, pathlib, sys, time\n"
        "sys.stdin.read()\n"
        f'pathlib.Path(r"{started}").write_text("up")\n'
        f"time.sleep({seconds})\n"
        f'print(json.dumps({{"success": True, "data": {{"tag": "{tag}"}}}}))\n'
    )


class TestAReplacedBundleDoesNotReachARunInFlight:
    def test_the_run_ends_with_the_content_it_started_with(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        started = tmp_path / "started.txt"
        _persisted(client, scripts=_both(_sleeping("v1", started, 0.3)))
        started.unlink()
        results: list[dict[str, Any]] = []
        runner = threading.Thread(
            target=lambda: results.append(_run_named(client, "pack"))
        )

        runner.start()
        deadline = time.monotonic() + 10
        while not started.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert started.exists(), "the named run never reached its script"
        _persisted(client, scripts=_both(_script_source("v2")))
        runner.join(timeout=30)

        assert [_tag(r, "s") for r in results] == ["v1"]
        assert _tag(_run_named(client, "pack"), "s") == "v2"


def _both(x_source: str) -> dict[str, str]:
    return {"tools/x.py": x_source, "tools/y.py": _script_source("y")}
