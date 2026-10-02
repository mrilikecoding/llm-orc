"""The closure shipper: a named local root as one self-contained run
request (Arc 5, Task 7).

Through a real OrchestraService on a temp project for the local side and
a second real service on its own empty dirs for the remote, reached by
REST. Nothing here opens a connection.
"""

# ruff: noqa: F811  (imported fixtures are redefined as test parameters)
from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient

import llm_orc.web.api as web_api
from llm_orc.core.config.config_manager import resolve_global_config_dir
from llm_orc.core.execution.scripting.resolver import ScriptResolver
from llm_orc.services.closure_shipper import (
    LeftOut,
    ShipError,
    ship_closure,
    ship_closure_reporting,
)
from llm_orc.services.handlers.run_request import RunRequest, materialize
from llm_orc.services.orchestra_service import OrchestraService
from llm_orc.web.server import create_app
from tests.unit.services.test_one_run_injection import (  # noqa: F401
    _ensemble,
    _profile,
    _yaml,
    listing,
    project,
    service,
    state_dir,
)


def _block(*paths: str) -> str:
    listed = ", ".join(f'"{p}"' for p in paths)
    return f"# /// llm-orc\n# files = [{listed}]\n# ///\n"


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _scripts(project: Path, ref: str, text: str = "print('x')\n") -> Path:
    return _write(project / ".llm-orc" / "scripts" / ref, text)


def _uses(project: Path, *refs: str, name: str = "top") -> None:
    agents = [{"name": f"a{i}", "script": ref} for i, ref in enumerate(refs)]
    _ensemble(project / ".llm-orc", name, agents)


def _remote_resolver(request: dict[str, Any], tmp_path: Path) -> ScriptResolver:
    """A real resolver over a run layer the request was materialized into,
    on a project that holds nothing (so only the layer can answer)."""
    run_dir = tmp_path / "layer"
    materialize(RunRequest.parse(request), run_dir)
    empty = tmp_path / "empty"
    empty.mkdir(exist_ok=True)
    return ScriptResolver(project_dir=empty, run_dir=run_dir)


def _resolved_in_the_layer(resolver: ScriptResolver, reference: str) -> Path:
    resolved, is_file = resolver.resolve_and_classify(reference)
    path = Path(resolved)
    assert is_file
    assert path.is_relative_to(resolver._run_dir or Path("/nowhere")), path
    return path


def _assert_remote_resolves(
    request: dict[str, Any], tmp_path: Path, expected: dict[str, Path]
) -> None:
    """The parity invariant: each reference resolves, on a resolver over
    the materialized request, to a file in the layer with the local
    file's bytes."""
    resolver = _remote_resolver(request, tmp_path)
    for reference, local in expected.items():
        remote = _resolved_in_the_layer(resolver, reference)
        assert remote.read_bytes() == local.read_bytes(), reference


def _ship(service: OrchestraService, root: str, **more: Any) -> dict[str, Any]:
    return ship_closure(
        root,
        find_root=service.find_ensemble_by_name,
        config_manager=service.config_manager,
        project_dir=service.project_path,
        **more,
    )


class TestEnsemblesShipAsWritten:
    def test_the_root_is_the_parsed_yaml_of_its_file_found_by_its_name(
        self, project: Path, service: OrchestraService
    ) -> None:
        path = project / ".llm-orc" / "ensembles" / "file-name.yaml"
        _yaml(
            path,
            {
                "name": "declared-name",
                "description": "d",
                "owner": "team-x",
                "agents": [{"name": "s", "script": "echo hi"}],
            },
        )

        request = _ship(service, "declared-name", input_text="hi")

        assert request["ensemble"] == yaml.safe_load(path.read_text())
        assert request["ensemble"]["owner"] == "team-x"
        assert request["input"] == "hi"

    def test_a_hierarchical_child_ships_under_its_reference_string(
        self, project: Path, service: OrchestraService
    ) -> None:
        base = project / ".llm-orc"
        _ensemble(base, "top", [{"name": "k", "ensemble": "kids/inner"}])
        path = base / "ensembles" / "kids" / "inner.yaml"
        _yaml(
            path,
            {
                "name": "inner-declared",
                "description": "d",
                "note": "kept",
                "agents": [{"name": "s", "script": "echo hi"}],
            },
        )

        request = _ship(service, "top")

        assert request["ensembles"] == {"kids/inner": yaml.safe_load(path.read_text())}

    def test_a_child_that_does_not_resolve_locally_is_left_out(
        self, project: Path, service: OrchestraService
    ) -> None:
        _ensemble(project / ".llm-orc", "top", [{"name": "k", "ensemble": "ghost"}])

        request = _ship(service, "top")

        assert request["ensembles"] == {}


class TestDataThatIsNotPlainJson:
    @pytest.mark.parametrize(
        "extra",
        [
            "created: 2026-01-01\n",
            "tags: !!set {a, b}\n",
            "1: numeric key\n",
            "ratio: .nan\n",
            "limit: .inf\n",
        ],
        ids=["date", "set", "non-string-key", "nan", "infinity"],
    )
    def test_it_is_refused_naming_the_ensemble(
        self, extra: str, project: Path, service: OrchestraService
    ) -> None:
        path = project / ".llm-orc" / "ensembles" / "odd.yaml"
        path.write_text(
            "name: odd\ndescription: d\nagents:\n"
            "  - {name: s, script: echo hi}\n" + extra
        )

        with pytest.raises(ShipError, match="'odd'.*plain data"):
            _ship(service, "odd")

    def test_a_child_is_refused_naming_its_reference(
        self, project: Path, service: OrchestraService
    ) -> None:
        base = project / ".llm-orc"
        _ensemble(base, "top", [{"name": "k", "ensemble": "kids/odd"}])
        kid = base / "ensembles" / "kids" / "odd.yaml"
        kid.parent.mkdir(parents=True)
        kid.write_text(
            "name: odd\ndescription: d\nagents:\n"
            "  - {name: s, script: echo hi}\ncreated: 2026-01-01\n"
        )

        with pytest.raises(ShipError, match="'kids/odd'.*plain data"):
            _ship(service, "top")

    def test_a_shared_anchor_ships_as_the_data_the_loader_read(
        self, project: Path, service: OrchestraService
    ) -> None:
        path = project / ".llm-orc" / "ensembles" / "anchored.yaml"
        path.write_text(
            "name: anchored\ndescription: d\nbase: &b {k: v}\ncopy: *b\n"
            "agents:\n  - {name: s, script: echo hi}\n"
        )

        request = _ship(service, "anchored")

        assert request["ensemble"]["copy"] == {"k": "v"}


class TestScriptKeys:
    def test_a_script_ships_at_a_key_the_remote_resolves_the_reference_to(
        self, project: Path, service: OrchestraService, tmp_path: Path
    ) -> None:
        local = _scripts(project, "tools/x.py", "print('one')\n")
        _uses(project, "tools/x.py")

        request = _ship(service, "top")

        assert request["scripts"] == {"scripts/tools/x.py": "print('one')\n"}
        _assert_remote_resolves(request, tmp_path, {"tools/x.py": local})

    def test_a_reference_with_and_without_the_scripts_prefix_is_one_key(
        self, project: Path, service: OrchestraService, tmp_path: Path
    ) -> None:
        local = _scripts(project, "tools/x.py")
        _uses(project, "tools/x.py", "scripts/tools/x.py")

        request = _ship(service, "top")

        assert list(request["scripts"]) == ["scripts/tools/x.py"]
        _assert_remote_resolves(
            request,
            tmp_path,
            {"tools/x.py": local, "scripts/tools/x.py": local},
        )

    def test_a_hyphenated_reference_whose_file_is_underscored(
        self, project: Path, service: OrchestraService, tmp_path: Path
    ) -> None:
        local = _scripts(project, "tools/my_x.py")
        _uses(project, "tools/my-x.py")

        request = _ship(service, "top")

        assert list(request["scripts"]) == ["scripts/tools/my_x.py"]
        _assert_remote_resolves(request, tmp_path, {"tools/my-x.py": local})

    def test_a_prefixed_and_hyphenated_reference_whose_file_is_underscored(
        self, project: Path, service: OrchestraService, tmp_path: Path
    ) -> None:
        local = _scripts(project, "tools/my_x.py")
        _uses(project, "scripts/tools/my-x.py")

        request = _ship(service, "top")

        assert list(request["scripts"]) == ["scripts/tools/my_x.py"]
        _assert_remote_resolves(request, tmp_path, {"scripts/tools/my-x.py": local})

    def test_two_spellings_of_one_file_are_one_key_and_both_resolve(
        self, project: Path, service: OrchestraService, tmp_path: Path
    ) -> None:
        local = _scripts(project, "tools/my_x.py")
        _uses(project, "tools/my-x.py", "tools/my_x.py")

        request = _ship(service, "top")

        assert list(request["scripts"]) == ["scripts/tools/my_x.py"]
        _assert_remote_resolves(
            request,
            tmp_path,
            {"tools/my-x.py": local, "tools/my_x.py": local},
        )

    def test_a_script_resolved_from_the_project_root_search_path(
        self, project: Path, service: OrchestraService, tmp_path: Path
    ) -> None:
        local = _write(project / "tools" / "x.py", "print('root')\n")
        _uses(project, "tools/x.py")

        request = _ship(service, "top")

        assert list(request["scripts"]) == ["scripts/tools/x.py"]
        _assert_remote_resolves(request, tmp_path, {"tools/x.py": local})

    def test_a_primitive_ships_from_the_package(
        self, project: Path, service: OrchestraService, tmp_path: Path
    ) -> None:
        ref = "primitives/file_ops/read_file.py"
        resolved, _ = ScriptResolver(project_dir=project).resolve_and_classify(ref)
        _uses(project, ref)

        request = _ship(service, "top")

        assert list(request["scripts"]) == [f"scripts/{ref}"]
        _assert_remote_resolves(request, tmp_path, {ref: Path(resolved)})

    def test_two_different_local_files_on_one_key_are_refused_naming_both(
        self, project: Path, service: OrchestraService
    ) -> None:
        """In the project root ``x.py`` and ``scripts/x.py`` are two files;
        on the remote they are one key."""
        _write(project / "x.py", "print('top')\n")
        _write(project / "scripts" / "x.py", "print('root')\n")
        _uses(project, "x.py", "scripts/x.py")

        with pytest.raises(ShipError) as refused:
            _ship(service, "top")

        assert "'scripts/x.py'" in str(refused.value)
        assert "'x.py'" in str(refused.value)

    def test_a_hyphen_underscore_pair_of_different_files_is_refused(
        self, project: Path, service: OrchestraService
    ) -> None:
        _scripts(project, "tools/a-b.py", "print('hyphen')\n")
        _scripts(project, "tools/a_b.py", "print('underscore')\n")
        _uses(project, "tools/a-b.py", "tools/a_b.py")

        with pytest.raises(ShipError) as refused:
            _ship(service, "top")

        assert "'tools/a-b.py'" in str(refused.value)
        assert "'tools/a_b.py'" in str(refused.value)

    def test_a_reference_with_a_dot_dot_segment_is_refused(
        self, project: Path, service: OrchestraService
    ) -> None:
        _scripts(project, "tools/x.py")
        _uses(project, "tools/../tools/x.py")

        with pytest.raises(ShipError, match="'tools/../tools/x.py'.*relative path"):
            _ship(service, "top")


class TestListedFiles:
    def test_each_listed_file_ships_beside_its_script_as_written(
        self, project: Path, service: OrchestraService, tmp_path: Path
    ) -> None:
        owner = _scripts(project, "tools/x.py", _block("_h.py", "lib/a.py"))
        helper = _scripts(project, "tools/_h.py", "H = 1\n")
        nested = _scripts(project, "tools/lib/a.py", "A = 1\n")
        _uses(project, "tools/x.py")

        request = _ship(service, "top")

        assert request["scripts"] == {
            "scripts/tools/x.py": owner.read_text(),
            "scripts/tools/_h.py": "H = 1\n",
            "scripts/tools/lib/a.py": "A = 1\n",
        }
        resolver = _remote_resolver(request, tmp_path)
        remote_owner = _resolved_in_the_layer(resolver, "tools/x.py")
        assert (remote_owner.parent / "_h.py").read_bytes() == helper.read_bytes()
        assert (remote_owner.parent / "lib/a.py").read_bytes() == nested.read_bytes()

    def test_a_listed_file_that_lists_another_ships_that_too(
        self, project: Path, service: OrchestraService
    ) -> None:
        _scripts(project, "tools/x.py", _block("lib/a.py"))
        _scripts(project, "tools/lib/a.py", _block("b.py"))
        _scripts(project, "tools/lib/b.py", "B = 1\n")
        _uses(project, "tools/x.py")

        request = _ship(service, "top")

        assert sorted(request["scripts"]) == [
            "scripts/tools/lib/a.py",
            "scripts/tools/lib/b.py",
            "scripts/tools/x.py",
        ]

    def test_the_helpers_of_a_hyphenated_reference_sit_beside_the_real_file(
        self, project: Path, service: OrchestraService, tmp_path: Path
    ) -> None:
        _scripts(project, "tools/my_x.py", _block("_h.py"))
        helper = _scripts(project, "tools/_h.py", "H = 1\n")
        _uses(project, "scripts/tools/my-x.py")

        request = _ship(service, "top")

        assert sorted(request["scripts"]) == [
            "scripts/tools/_h.py",
            "scripts/tools/my_x.py",
        ]
        resolver = _remote_resolver(request, tmp_path)
        remote = _resolved_in_the_layer(resolver, "scripts/tools/my-x.py")
        assert (remote.parent / "_h.py").read_bytes() == helper.read_bytes()

    def test_a_listed_file_absent_beside_its_script_is_left_out(
        self, project: Path, service: OrchestraService
    ) -> None:
        """A host file at the same relative path in another tier does not
        stand in for it (review focus 1); the remote reports it missing."""
        _scripts(project, "tools/x.py", _block("_h.py"))
        _write(project / "_h.py", "print('elsewhere')\n")
        _uses(project, "tools/x.py")

        request = _ship(service, "top")

        assert list(request["scripts"]) == ["scripts/tools/x.py"]

    def test_a_listed_path_that_leaves_the_directory_ships_only_the_script(
        self, project: Path, service: OrchestraService
    ) -> None:
        """Review focus 2: nothing outside the script's directory is read."""
        _scripts(project, "outside.py", "SECRET = 1\n")
        _scripts(project, "tools/x.py", _block("../outside.py"))
        _uses(project, "tools/x.py")

        request = _ship(service, "top")

        assert list(request["scripts"]) == ["scripts/tools/x.py"]
        assert "SECRET" not in str(request)

    def test_a_listed_file_that_is_not_utf8_is_refused_naming_it(
        self, project: Path, service: OrchestraService
    ) -> None:
        _scripts(project, "tools/x.py", _block("_h.py"))
        (project / ".llm-orc" / "scripts" / "tools" / "_h.py").write_bytes(b"\xff\xfe")
        _uses(project, "tools/x.py")

        with pytest.raises(
            ShipError, match="'_h.py' listed by script 'tools/x.py'.*UTF"
        ):
            _ship(service, "top")


class TestWhatDoesNotShip:
    def test_inline_shell_stays_in_the_definition_and_ships_nothing(
        self, project: Path, service: OrchestraService
    ) -> None:
        _uses(project, "echo hi")

        request = _ship(service, "top")

        assert request["scripts"] == {}
        assert request["ensemble"]["agents"] == [{"name": "a0", "script": "echo hi"}]

    def test_a_script_that_does_not_resolve_locally_is_left_out(
        self, project: Path, service: OrchestraService
    ) -> None:
        _uses(project, "tools/ghost.py")

        assert _ship(service, "top")["scripts"] == {}


class TestRefusals:
    def test_an_absolute_script_path_is_refused_naming_it(
        self, project: Path, service: OrchestraService, tmp_path: Path
    ) -> None:
        absolute = _write(tmp_path / "abs" / "x.py", "print('x')\n")
        _uses(project, str(absolute))

        with pytest.raises(ShipError, match="absolute path"):
            _ship(service, "top")

    def test_an_absolute_path_that_does_not_exist_is_refused_too(
        self, project: Path, service: OrchestraService
    ) -> None:
        _uses(project, "/nowhere/x.py")

        with pytest.raises(ShipError, match="'/nowhere/x.py'.*absolute path"):
            _ship(service, "top")

    def test_a_bare_name_that_is_a_file_in_the_cwd_is_refused(
        self,
        project: Path,
        service: OrchestraService,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Review focus 4: shipped, the remote would run it as inline
        shell (the ``date`` case)."""
        _write(tmp_path / "cwd" / "date", "#!/bin/sh\necho mine\n")
        monkeypatch.chdir(tmp_path / "cwd")
        _uses(project, "date")

        with pytest.raises(ShipError, match="'date'.*working directory"):
            _ship(service, "top")

    def test_a_script_that_is_not_utf8_is_refused_naming_it(
        self, project: Path, service: OrchestraService
    ) -> None:
        _scripts(project, "tools/x.py").write_bytes(b"\xff\xfe\x00")
        _uses(project, "tools/x.py")

        with pytest.raises(ShipError, match="'tools/x.py'.*UTF"):
            _ship(service, "top")


class TestProfilesAndTheRestOfTheRequest:
    def test_a_named_profile_ships_with_its_local_definition(
        self, project: Path, service: OrchestraService
    ) -> None:
        _profile(project / ".llm-orc", "seat", model="mock-a")
        _profile(project / ".llm-orc", "other", model="mock-b")
        _ensemble(
            project / ".llm-orc",
            "top",
            [{"name": "w", "model_profile": "seat"}],
        )

        request = _ship(service, "top", with_profiles=["seat"])

        assert list(request["profiles"]) == ["seat"]
        assert request["profiles"]["seat"]["model"] == "mock-a"

    def test_profiles_in_the_closure_are_not_shipped_unless_named(
        self, project: Path, service: OrchestraService
    ) -> None:
        _profile(project / ".llm-orc", "seat")
        _ensemble(
            project / ".llm-orc",
            "top",
            [{"name": "w", "model_profile": "seat"}],
        )

        assert _ship(service, "top")["profiles"] == {}

    def test_a_named_profile_that_does_not_exist_locally_is_an_error(
        self, project: Path, service: OrchestraService
    ) -> None:
        _uses(project, "echo hi")

        with pytest.raises(ShipError, match="profile 'ghost'"):
            _ship(service, "top", with_profiles=["ghost"])

    def test_bind_pull_and_input_ride_along_and_persist_only_when_given(
        self, project: Path, service: OrchestraService
    ) -> None:
        _uses(project, "echo hi")

        plain = _ship(service, "top", bind={"a": "b"}, pull=True, input_text="go")
        persisted = _ship(service, "top", persist="global")

        assert plain["bind"] == {"a": "b"}
        assert plain["pull"] is True
        assert plain["input"] == "go"
        assert "persist" not in plain
        assert persisted["persist"] == "global"

    def test_a_missing_root_is_an_error_naming_it(
        self, service: OrchestraService
    ) -> None:
        with pytest.raises(ShipError, match="'ghost'"):
            _ship(service, "ghost")

    def test_a_request_the_validator_would_refuse_is_refused_locally(
        self, project: Path, service: OrchestraService
    ) -> None:
        base = project / ".llm-orc"
        _yaml(
            base / "ensembles" / "rootfile.yaml",
            {
                "name": "Kid",
                "description": "d",
                "agents": [{"name": "k", "ensemble": "kid"}],
            },
        )
        _ensemble(base, "kid", [{"name": "s", "script": "echo hi"}])

        with pytest.raises(ShipError, match="would be refused.*ignoring case"):
            _ship(service, "Kid")

    def test_a_persist_the_validator_would_refuse_is_refused_locally(
        self, project: Path, service: OrchestraService
    ) -> None:
        _yaml(
            project / ".llm-orc" / "ensembles" / "pack" / "nested.yaml",
            {
                "name": "pack/nested",
                "description": "d",
                "agents": [{"name": "s", "script": "echo hi"}],
            },
        )

        with pytest.raises(ShipError, match="would be refused.*must not contain '/'"):
            _ship(service, "pack/nested", persist="global")


class TestWhatTheShipperLeftOut:
    def _reporting(self, service: OrchestraService) -> list[LeftOut]:
        request, left_out = ship_closure_reporting(
            "top",
            find_root=service.find_ensemble_by_name,
            config_manager=service.config_manager,
            project_dir=service.project_path,
        )
        assert request == _ship(service, "top")
        return left_out

    def test_a_child_a_script_and_a_listed_file_that_do_not_resolve(
        self, project: Path, service: OrchestraService
    ) -> None:
        base = project / ".llm-orc"
        _ensemble(
            base,
            "top",
            [
                {"name": "k", "ensemble": "kids/x"},
                {"name": "g", "script": "tools/y.py"},
                {"name": "o", "script": "tools/o.py"},
            ],
        )
        _scripts(project, "tools/o.py", _block("_h.py"))

        left_out = self._reporting(service)

        assert [(item.kind, item.reference) for item in left_out] == [
            ("ensemble", "kids/x"),
            ("script", "tools/y.py"),
            ("file", "_h.py"),
        ]
        assert [item.label for item in left_out] == [
            "ensemble 'kids/x'",
            "script 'tools/y.py'",
            "file '_h.py' listed by script 'tools/o.py'",
        ]

    def test_inline_shell_and_what_ships_are_not_left_out(
        self, project: Path, service: OrchestraService
    ) -> None:
        _scripts(project, "tools/x.py", _block("_h.py"))
        _scripts(project, "tools/_h.py")
        _ensemble(
            project / ".llm-orc",
            "top",
            [
                {"name": "a", "script": "echo hi"},
                {"name": "b", "script": "tools/x.py"},
            ],
        )

        assert self._reporting(service) == []


class TestErrorsOnTheShipPath:
    def test_a_child_that_does_not_load_is_a_ship_error_naming_it(
        self, project: Path, service: OrchestraService
    ) -> None:
        _ensemble(project / ".llm-orc", "top", [{"name": "k", "ensemble": "kid"}])
        _write(resolve_global_config_dir() / "ensembles" / "kid.yaml", "agents: [\n")

        with pytest.raises(ShipError, match="'kid'"):
            _ship(service, "top")

    def test_a_script_that_cannot_be_read_is_a_ship_error_naming_it(
        self, project: Path, service: OrchestraService
    ) -> None:
        script = _scripts(project, "tools/x.py", _block("_h.py"))
        _scripts(project, "tools/_h.py")
        _uses(project, "tools/x.py")
        script.chmod(0)
        try:
            if os.access(script, os.R_OK):
                pytest.skip("this user can read a file with no permissions")
            with pytest.raises(ShipError, match="'tools/x.py'"):
                _ship(service, "top")
        finally:
            script.chmod(0o644)


@pytest.fixture
def remote(tmp_path: Path, listing: Any, state_dir: Path) -> OrchestraService:
    """A second real service on its own empty project dir."""
    root = tmp_path / "remote-proj"
    (root / ".llm-orc" / "ensembles").mkdir(parents=True)
    svc = OrchestraService()
    assert svc.handle_set_project(str(root))["status"] == "ok"
    return svc


@contextmanager
def _rest(
    svc: OrchestraService, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    monkeypatch.setattr(web_api, "_orchestra_service", svc)
    with TestClient(create_app()) as client:
        yield client


def _run(
    svc: OrchestraService, monkeypatch: pytest.MonkeyPatch, body: dict[str, Any]
) -> dict[str, Any]:
    with _rest(svc, monkeypatch) as client:
        response = client.post("/api/ensembles/execute", json=body)
    assert response.status_code == 200, response.text
    return dict(response.json())


def _without_timings(text: str) -> Any:
    """A response with its run metadata (clocks, memory) dropped, at any
    depth: a child ensemble's response carries its own."""
    try:
        data = json.loads(text)
    except ValueError:
        return text
    return _strip(data)


def _strip(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: _strip(v) for k, v in data.items() if k != "metadata"}
    if isinstance(data, str):
        return _without_timings(data)
    return data


def _outcome(result: dict[str, Any]) -> dict[str, Any]:
    """What a caller sees of a run, without timings."""
    return {
        "status": result["status"],
        "has_errors": result["has_errors"],
        "deliverable": _without_timings(result["deliverable"]),
        "agents": {
            name: (agent["status"], _without_timings(agent["response"]))
            for name, agent in result["results"].items()
        },
    }


_X = (
    _block("_h.py")
    + "import json, sys\n"
    + "sys.stdin.read()\n"
    + "import _h\n"
    + 'print(json.dumps({"success": True, "data": {"tag": _h.TAG}}))\n'
)
_Y = (
    "import json, sys\n"
    "sys.stdin.read()\n"
    'print(json.dumps({"success": True, "data": {"tag": "child"}}))\n'
)


def _closure_project(project: Path) -> None:
    """A root whose ``name:`` is not its file name, a hierarchical child,
    a script with a listed helper and a second reference to it."""
    base = project / ".llm-orc"
    _yaml(
        base / "ensembles" / "root-file.yaml",
        {
            "name": "top-declared",
            "description": "d",
            "agents": [
                {"name": "first", "script": "tools/x.py"},
                {"name": "again", "script": "scripts/tools/x.py"},
                {"name": "dashed", "script": "tools/my-w.py"},
                {"name": "scored", "script": "tools/my_w.py"},
                {"name": "kid", "ensemble": "kids/inner"},
            ],
        },
    )
    _yaml(
        base / "ensembles" / "kids" / "inner.yaml",
        {
            "name": "inner-declared",
            "description": "d",
            "agents": [{"name": "leaf", "script": "tools/y.py"}],
        },
    )
    _scripts(project, "tools/x.py", _X)
    _scripts(project, "tools/_h.py", "TAG = 'helper'\n")
    _scripts(project, "tools/y.py", _Y)
    _scripts(project, "tools/my_w.py", _Y)


class TestParityWithALocalRun:
    def test_a_second_service_on_empty_dirs_returns_the_local_result(
        self,
        project: Path,
        service: OrchestraService,
        remote: OrchestraService,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _closure_project(project)
        local = _run(
            service, monkeypatch, {"ensemble_name": "top-declared", "input": "hi"}
        )
        assert local["status"] == "success", local

        request = _ship(service, "top-declared", input_text="hi")
        shipped = _run(remote, monkeypatch, request)

        assert shipped["status"] == "success", shipped
        assert _outcome(shipped) == _outcome(local)
        assert "helper" in str(_outcome(shipped)["agents"]["first"])
        assert str(tmp_path) not in json.dumps(request)
        assert str(project) not in json.dumps(request)

    def test_without_its_listed_files_the_remote_refuses_the_script(
        self,
        project: Path,
        service: OrchestraService,
        remote: OrchestraService,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _closure_project(project)
        request = _ship(service, "top-declared", input_text="hi")
        request["scripts"] = {
            key: text for key, text in request["scripts"].items() if "_h" not in key
        }

        refused = _run(remote, monkeypatch, request)

        assert refused["status"] == "error"
        assert refused["error"]["kind"] == "not_equipped"
        rows = {d["name"]: d["status"] for d in refused["error"]["dependencies"]}
        assert rows["tools/_h.py"] == "missing_script"


def _tag_script(tag: str, files: tuple[str, ...] = ()) -> str:
    return (
        (_block(*files) if files else "")
        + "import json, sys\n"
        + "sys.stdin.read()\n"
        + f'print(json.dumps({{"success": True, "data": {{"tag": "{tag}"}}}}))\n'
    )


class TestAReferenceTheRemoteWouldReadAsAnotherFile:
    def test_a_scripts_prefixed_reference_beside_a_listed_scripts_dir_is_refused(
        self,
        project: Path,
        service: OrchestraService,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Locally ``scripts/foo.py`` is the tier's ``foo.py`` (A). The
        listed file of ``scripts/y.py`` would ship at
        ``scripts/scripts/foo.py``, which the remote's resolver reaches
        first for that reference (B)."""
        _scripts(project, "foo.py", _tag_script("A"))
        _write(project / "scripts" / "y.py", _tag_script("Y", ("scripts/foo.py",)))
        _write(project / "scripts" / "scripts" / "foo.py", _tag_script("B"))
        _uses(project, "scripts/y.py", "scripts/foo.py")
        local = _run(service, monkeypatch, {"ensemble_name": "top", "input": ""})
        assert local["status"] == "success", local
        assert "A" in str(_outcome(local)["agents"]["a1"])

        with pytest.raises(ShipError, match="'scripts/foo.py'"):
            _ship(service, "top")

    def test_the_proofs_temporary_directory_is_gone_on_every_way_out(
        self,
        project: Path,
        service: OrchestraService,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(scratch))
        _scripts(project, "foo.py", _tag_script("A"))
        _uses(project, "scripts/foo.py")
        _ship(service, "top")
        assert list(scratch.iterdir()) == []

        _write(project / "scripts" / "y.py", _tag_script("Y", ("scripts/foo.py",)))
        _write(project / "scripts" / "scripts" / "foo.py", _tag_script("B"))
        _uses(project, "scripts/y.py", "scripts/foo.py")
        with pytest.raises(ShipError):
            _ship(service, "top")
        assert list(scratch.iterdir()) == []


class TestASymlinkedScript:
    def test_the_helper_beside_the_target_is_the_one_that_ships_and_runs(
        self,
        project: Path,
        service: OrchestraService,
        remote: OrchestraService,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Locally the script imports ``_h`` from beside its target (the
        interpreter puts the real directory first); the remote must get
        that file, not the one beside the link."""
        source = (
            _block("_h.py")
            + "import json, sys\n"
            + "sys.stdin.read()\n"
            + "import _h\n"
            + 'print(json.dumps({"success": True, "data": {"tag": _h.TAG}}))\n'
        )
        target = _write(project / "shared" / "x.py", source)
        _write(project / "shared" / "_h.py", "TAG = 'target'\n")
        link = project / ".llm-orc" / "scripts" / "tools" / "x.py"
        link.parent.mkdir(parents=True)
        try:
            link.symlink_to(target)
        except OSError:
            pytest.skip("this platform cannot make symlinks")
        _scripts(project, "tools/_h.py", "TAG = 'link'\n")
        _uses(project, "tools/x.py")
        local = _run(service, monkeypatch, {"ensemble_name": "top", "input": ""})
        assert local["status"] == "success", local
        assert "target" in str(_outcome(local)["agents"]["a0"])

        shipped = _run(remote, monkeypatch, _ship(service, "top"))

        assert shipped["status"] == "success", shipped
        assert _outcome(shipped) == _outcome(local)


class TestWhatTheRemoteJudges:
    def test_a_child_missing_locally_is_reported_missing_by_the_remote(
        self,
        project: Path,
        service: OrchestraService,
        remote: OrchestraService,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _ensemble(project / ".llm-orc", "top", [{"name": "k", "ensemble": "ghost"}])

        refused = _run(remote, monkeypatch, _ship(service, "top"))

        assert refused["error"]["kind"] == "not_equipped"
        rows = {d["name"]: d["status"] for d in refused["error"]["dependencies"]}
        assert rows["ghost"] == "missing_ensemble"

    def test_a_profile_the_remote_lacks_is_refused_and_shipped_when_named(
        self,
        project: Path,
        service: OrchestraService,
        remote: OrchestraService,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _profile(project / ".llm-orc", "seat", model="mock-seat")
        _ensemble(project / ".llm-orc", "top", [{"name": "w", "model_profile": "seat"}])

        refused = _run(remote, monkeypatch, _ship(service, "top"))
        shipped = _run(
            remote, monkeypatch, _ship(service, "top", with_profiles=["seat"])
        )

        rows = {d["name"]: d["status"] for d in refused["error"]["dependencies"]}
        assert rows["seat"] == "missing_profile"
        assert shipped["status"] == "success", shipped
