"""Bundles and ``persist`` through REST, over one real OrchestraService on
temp project, state and global dirs (Arc 5, Task 9, ruling 9).

A persisted closure is a stored run request: written after the gate
passes, replayed through the same injection path on each named run.
"""

# ruff: noqa: F811  (imported fixtures are redefined as test parameters)
from __future__ import annotations

import asyncio
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
from llm_orc.mcp.server import MCPServer
from llm_orc.services.handlers.execution_handler import ExecutionHandler
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

    def test_another_spelling_of_a_bundles_name_is_refused_naming_the_bundle(
        self, client: TestClient
    ) -> None:
        """A case-folding disk would open ``pack.json`` for ``PACK`` and
        overwrite the bundle under a root its file name does not match, so
        the refusal holds on a case-sensitive disk too: the directory may
        be copied to a folding one."""
        _persisted(client)
        before = _bundle_file().read_bytes()
        renamed = _pack(persist="global")
        renamed["ensemble"] = {**PACK, "name": "PACK"}

        result = _run(client, renamed)

        assert result["error"]["kind"] == "invalid_request"
        assert "pack" in result["error"]["message"]
        assert os.listdir(_bundle_file().parent) == ["pack.json"]
        assert _bundle_file().read_bytes() == before
        assert _run_named(client, "pack")["status"] == "success"
        deleted = client.delete("/api/ensembles/pack", params={"scope": "global"})
        assert deleted.json()["deleted"] is True

    @pytest.mark.parametrize(
        ("first", "second"),
        [("caf\u00e9", "cafe\u0301"), ("cafe\u0301", "caf\u00e9")],
        ids=["nfc-then-nfd", "nfd-then-nfc"],
    )
    def test_the_same_name_in_another_unicode_form_is_refused(
        self, client: TestClient, first: str, second: str
    ) -> None:
        """macOS stores and opens names in decomposed form, so ``café``
        spelled with one code point and with two are one file there."""
        stored = _pack(persist="global")
        stored["ensemble"] = {**PACK, "name": first}
        assert _run(client, stored)["status"] == "success"
        before = sorted(os.listdir(_bundle_file().parent))
        rival = _pack(persist="global")
        rival["ensemble"] = {**PACK, "name": second}

        result = _run(client, rival)

        assert result["error"]["kind"] == "invalid_request"
        assert "another spelling" in result["error"]["message"]
        assert sorted(os.listdir(_bundle_file().parent)) == before

    def test_a_spelling_that_lands_during_the_gate_is_refused_before_the_write(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The first check runs before the gate, which can take long (a
        model pull). A persist of ``Pack`` that finishes in that time must
        stop this one at the write."""
        real_gate = ExecutionHandler._gate

        async def gate_then_a_rival_lands(*args: Any, **more: Any) -> Any:
            outcome = await real_gate(*args, **more)
            _bundle_file("Pack").parent.mkdir(parents=True, exist_ok=True)
            _bundle_file("Pack").write_text("{}")
            return outcome

        monkeypatch.setattr(ExecutionHandler, "_gate", gate_then_a_rival_lands)

        result = _run(client, _pack(persist="global"))

        assert result["error"]["kind"] == "invalid_request"
        assert "'Pack'" in result["error"]["message"]
        assert "another spelling" in result["error"]["message"]
        assert os.listdir(_bundle_file().parent) == ["Pack.json"]

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

    def test_a_tier_file_that_lands_during_the_gate_is_refused_before_the_write(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The first check runs before the gate, which can take long (a
        model pull). A tier file written in that time must stop this
        persist at the write, or the bundle is born shadowed."""
        real_gate = ExecutionHandler._gate

        async def gate_then_a_tier_file_lands(*args: Any, **more: Any) -> Any:
            outcome = await real_gate(*args, **more)
            _ensemble(
                resolve_global_config_dir(), "pack", [{"name": "t", "script": "echo"}]
            )
            return outcome

        monkeypatch.setattr(ExecutionHandler, "_gate", gate_then_a_tier_file_lands)

        result = _run(client, _pack(persist="global"))

        assert result["error"]["kind"] == "invalid_request"
        assert "global tier" in result["error"]["message"]
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


def _reader(wait: tuple[Path, Path] | None = None) -> str:
    """A script that lists ``_data.txt``, optionally announces itself at
    ``started`` and waits for ``go``, and only then reads the file beside
    it: a run in flight that shares anything with a later persist sees
    the later content."""
    pause = ""
    if wait is not None:
        started, go = wait
        pause = (
            f'pathlib.Path(r"{started}").write_text("up")\n'
            "deadline = time.monotonic() + 30\n"
            f'while not pathlib.Path(r"{go}").exists() '
            "and time.monotonic() < deadline:\n"
            "    time.sleep(0.01)\n"
        )
    return (
        '# /// llm-orc\n# files = ["_data.txt"]\n# ///\n'
        "import json, pathlib, sys, time\n"
        "sys.stdin.read()\n"
        + pause
        + "here = pathlib.Path(__file__).resolve().parent\n"
        'data = (here / "_data.txt").read_text()\n'
        'print(json.dumps({"success": True, "data": {"tag": data}}))\n'
    )


class TestAReplacedBundleDoesNotReachARunInFlight:
    def test_the_run_ends_with_the_content_it_started_with(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        started, go = tmp_path / "started.txt", tmp_path / "go.txt"
        go.write_text("go")
        _persisted(
            client, scripts=_both(_reader((started, go)), {"tools/_data.txt": "v1"})
        )
        started.unlink()
        go.unlink()
        results: list[dict[str, Any]] = []
        runner = threading.Thread(
            target=lambda: results.append(_run_named(client, "pack"))
        )

        runner.start()
        deadline = time.monotonic() + 10
        while not started.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert started.exists(), "the named run never reached its script"
        _persisted(client, scripts=_both(_reader(), {"tools/_data.txt": "v2"}))
        go.write_text("go")
        runner.join(timeout=30)

        assert [_tag(r, "s") for r in results] == ["v1"]
        assert _tag(_run_named(client, "pack"), "s") == "v2"


def _both(x_source: str, more: dict[str, str] | None = None) -> dict[str, str]:
    return {
        "tools/x.py": x_source,
        "tools/y.py": _script_source("y"),
        **(more or {}),
    }


def _write_tier_yml(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "name: pack\ndescription: tier\nagents:\n  - name: t\n    script: echo hi\n"
    )


class TestTheOtherSurfacesKnowBundles:
    def test_the_listing_shows_a_bundle_root_with_source_bundle(
        self, client: TestClient
    ) -> None:
        _persisted(client)

        listed = {e["name"]: e for e in client.get("/api/ensembles").json()}

        assert listed["pack"]["source"] == "bundle"
        assert listed["pack"]["agent_count"] == 2
        assert listed["pack"]["description"] == "a closure"

    def test_a_listed_bundle_can_be_read_over_rest_with_source_bundle(
        self, client: TestClient
    ) -> None:
        _persisted(client)

        response = client.get("/api/ensembles/pack")

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["name"] == "pack"
        assert body["source"] == "bundle"
        assert body["description"] == "a closure"
        assert [a["name"] for a in body["agents"]] == ["s", "k"]

    def test_a_listed_bundle_can_be_read_as_the_mcp_resource(
        self, client: TestClient, service: OrchestraService
    ) -> None:
        _persisted(client)

        contents = asyncio.run(
            MCPServer(service=service)._mcp.read_resource("llm-orc://ensemble/pack")
        )

        body = json.loads(next(iter(contents)).content)
        assert body["source"] == "bundle"
        assert body["name"] == "pack"

    def test_a_tier_ensemble_of_the_same_name_is_read_first(
        self, client: TestClient, project: Path
    ) -> None:
        _persisted(client)
        _ensemble(project / ".llm-orc", "pack", [{"name": "t", "script": "echo hi"}])

        body = client.get("/api/ensembles/pack").json()

        assert "source" not in body
        assert [a["name"] for a in body["agents"]] == ["t"]

    def test_a_corrupt_bundle_is_left_out_of_the_listing(
        self, client: TestClient
    ) -> None:
        _persisted(client)
        _bundle_file().write_text("{not json")

        listed = {e["name"] for e in client.get("/api/ensembles").json()}

        assert "pack" not in listed

    def test_the_runnable_check_answers_with_the_report_and_leaves_no_layer(
        self, client: TestClient, state_dir: Path
    ) -> None:
        _persisted(client)

        response = client.get("/api/ensembles/pack/runnable")

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["ensemble"] == "pack"
        assert body["runnable"] is True
        deps = {d["name"]: d["status"] for d in body["dependencies"]}
        assert deps["tools/x.py"] == "ready"
        assert deps["kid"] == "ready"
        assert _runs(state_dir) == []

    def test_the_runnable_check_reports_what_the_host_lacks_for_a_stored_bind(
        self, client: TestClient, project: Path, state_dir: Path
    ) -> None:
        profile = project / ".llm-orc" / "profiles" / "b.yaml"
        _profile(project / ".llm-orc", "b", model="mock-other")
        _run(
            client,
            {"ensemble": ROLES, "bind": {"a": "b"}, "input": "hi", "persist": "global"},
        )
        profile.unlink()

        body = client.get("/api/ensembles/roles/runnable").json()

        assert body["runnable"] is False
        rows = [d for d in body["dependencies"] if d["name"] == "b"]
        assert [(r["status"], r["via"]) for r in rows] == [
            ("missing_profile", ["bind:a"])
        ]
        assert _runs(state_dir) == []

    def test_the_runnable_check_of_a_tier_ensemble_is_as_before(
        self, client: TestClient, project: Path
    ) -> None:
        _ensemble(project / ".llm-orc", "plain", [{"name": "s", "script": "echo hi"}])

        body = client.get("/api/ensembles/plain/runnable").json()

        assert body["runnable"] is True

    def test_delete_with_scope_global_removes_the_bundle_and_the_name_stops_resolving(
        self, client: TestClient
    ) -> None:
        _persisted(client)

        response = client.delete("/api/ensembles/pack", params={"scope": "global"})

        assert response.status_code == 200, response.text
        assert response.json()["deleted"] is True
        assert not _bundle_file().exists()
        with pytest.raises(ValueError, match="does not exist"):
            _run_named(client, "pack")

    def test_another_spelling_of_the_name_deletes_nothing_and_runs_nothing(
        self, client: TestClient
    ) -> None:
        """On a case-folding disk ``PACK`` opens ``pack.json``; the name is
        a bundle only where the directory entry is exactly ``PACK.json``."""
        _persisted(client)

        with pytest.raises(ValueError, match="not found"):
            client.delete("/api/ensembles/PACK", params={"scope": "global"})
        with pytest.raises(ValueError, match="does not exist"):
            _run_named(client, "PACK")

        assert _bundle_file().exists()
        assert _run_named(client, "pack")["status"] == "success"

    def test_a_global_tier_file_and_a_bundle_of_one_name_go_one_delete_each(
        self, client: TestClient
    ) -> None:
        """The tier file is what a delete of that name means; the bundle
        it shadows goes on the next one."""
        _persisted(client)
        tier_file = resolve_global_config_dir() / "ensembles" / "pack.yaml"
        _ensemble(
            resolve_global_config_dir(), "pack", [{"name": "t", "script": "echo hi"}]
        )

        first = client.delete("/api/ensembles/pack", params={"scope": "global"})

        assert first.status_code == 200, first.text
        assert "bundle" not in first.json()
        assert not tier_file.exists()
        assert _bundle_file().exists()

        second = client.delete("/api/ensembles/pack", params={"scope": "global"})

        assert second.json()["bundle"] == "pack"
        assert not _bundle_file().exists()

    def test_a_global_yml_file_and_a_bundle_of_one_name_go_one_delete_each(
        self, client: TestClient
    ) -> None:
        """The loader reads ``.yml`` as it reads ``.yaml``, so a tier file
        with either extension is what the delete of that name means, and
        the delete can remove it."""
        _persisted(client)
        tier_file = resolve_global_config_dir() / "ensembles" / "pack.yml"
        _write_tier_yml(tier_file)

        first = client.delete("/api/ensembles/pack", params={"scope": "global"})

        assert first.status_code == 200, first.text
        assert "bundle" not in first.json()
        assert not tier_file.exists()
        assert _bundle_file().exists()

        second = client.delete("/api/ensembles/pack", params={"scope": "global"})

        assert second.json()["bundle"] == "pack"
        assert not _bundle_file().exists()

    def test_a_tier_root_in_a_subdirectory_spares_the_bundle_on_a_delete(
        self, client: TestClient
    ) -> None:
        """A tier root the flat-file check cannot see still resolves the
        name, so the delete is the tier delete (which cannot find it by
        name, as before) and the bundle behind it stays."""
        _persisted(client)
        nested = resolve_global_config_dir() / "ensembles" / "sub" / "pack.yaml"
        _write_tier_yml(nested)

        with pytest.raises(ValueError, match="not found"):
            client.delete("/api/ensembles/pack", params={"scope": "global"})

        assert nested.exists()
        assert _bundle_file().exists()

    def test_delete_with_scope_project_does_not_touch_a_bundle(
        self, client: TestClient
    ) -> None:
        _persisted(client)

        with pytest.raises(ValueError, match="not found"):
            client.delete("/api/ensembles/pack", params={"scope": "project"})

        assert _bundle_file().exists()

    def test_validate_says_the_name_is_a_bundle(self, client: TestClient) -> None:
        _persisted(client)

        with pytest.raises(ValueError, match="bundle"):
            client.post("/api/ensembles/pack/validate")

    async def test_promote_says_the_name_is_a_bundle(
        self, client: TestClient, service: OrchestraService
    ) -> None:
        _persisted(client)

        with pytest.raises(ValueError, match="bundle"):
            await service.promote_ensemble(
                {"ensemble_name": "pack", "destination": "global", "confirm": True}
            )
        assert not (resolve_global_config_dir() / "ensembles" / "pack.yaml").exists()
