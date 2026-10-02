"""A run request is validated, then written into a run layer (Arc 4).

Every escape shape raises before anything is written, anywhere. Bindings
are materialized as layer profiles, read from the view as it stood before
any binding was written.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.core.config.ensemble_config import EnsembleLoader
from llm_orc.services.handlers.run_request import (
    RunRequest,
    RunRequestError,
    apply_bindings,
    materialize,
)

ROOT: dict[str, Any] = {
    "name": "root",
    "description": "root",
    "agents": [{"name": "w", "model": "mock-a", "provider": "mock"}],
}


def _tree(base: Path) -> dict[str, int]:
    """Every path under ``base`` with its size (directories are 0)."""
    return {
        str(p.relative_to(base)): (p.stat().st_size if p.is_file() else 0)
        for p in sorted(base.rglob("*"))
    }


def _parse(**fields: Any) -> RunRequest:
    return RunRequest.parse({"ensemble": ROOT, **fields})


class TestRequestShape:
    def test_both_roots_are_rejected(self) -> None:
        with pytest.raises(RunRequestError, match="exactly one"):
            RunRequest.parse({"ensemble": ROOT, "ensemble_name": "x"})

    def test_neither_root_is_rejected(self) -> None:
        with pytest.raises(RunRequestError, match="exactly one"):
            RunRequest.parse({"input": "hi"})

    def test_an_unknown_key_is_rejected(self) -> None:
        with pytest.raises(RunRequestError):
            _parse(profile={"a": {}})

    def test_a_bind_key_that_is_also_an_inline_profile_is_rejected(self) -> None:
        with pytest.raises(RunRequestError, match="twice"):
            _parse(profiles={"a": {"model": "m"}}, bind={"a": "b"})

    def test_needs_layer_only_when_something_is_injected(self) -> None:
        assert not RunRequest.parse({"ensemble_name": "x"}).needs_layer
        assert RunRequest.parse({"ensemble_name": "x", "bind": {"a": "b"}}).needs_layer
        assert _parse().needs_layer

    def test_an_inline_root_needs_a_name(self) -> None:
        with pytest.raises(RunRequestError, match="name"):
            RunRequest.parse({"ensemble": {"agents": []}})


ESCAPES = {
    "dotdot script": {"scripts": {"../x.py": "print(1)"}},
    "absolute script": {"scripts": {"/etc/x.py": "print(1)"}},
    "empty segment script": {"scripts": {"a//b.py": "print(1)"}},
    "backslash script": {"scripts": {"a\\b.py": "print(1)"}},
    "dotdot ensemble": {"ensembles": {"../kid": ROOT}},
    "absolute ensemble": {"ensembles": {"/kid": ROOT}},
    "empty segment ensemble": {"ensembles": {"a//kid": ROOT}},
    "backslash ensemble": {"ensembles": {"a\\kid": ROOT}},
    "inner dotdot": {"scripts": {"a/../../x.py": "print(1)"}},
    "profile with a slash": {"profiles": {"a/b": {"model": "m"}}},
    "profile dotdot": {"profiles": {"..": {"model": "m"}}},
    "bind key dotdot": {"bind": {"../a": "b"}},
}


class TestPersist:
    def test_global_with_an_inline_root_validates(self) -> None:
        assert _parse(persist="global").persist == "global"

    def test_the_default_is_no_persist(self) -> None:
        assert _parse().persist is None

    def test_any_other_scope_is_rejected(self) -> None:
        with pytest.raises(RunRequestError, match="persist"):
            _parse(persist="project")

    def test_a_named_root_cannot_be_persisted(self) -> None:
        with pytest.raises(RunRequestError, match="inline root"):
            RunRequest.parse({"ensemble_name": "x", "persist": "global"})

    def test_the_root_name_must_be_plain(self) -> None:
        nested = {**ROOT, "name": "group/root"}
        with pytest.raises(RunRequestError, match="'/'"):
            RunRequest.parse({"ensemble": nested, "persist": "global"})

    def test_a_nested_root_name_is_fine_without_persist(self) -> None:
        nested = {**ROOT, "name": "group/root"}
        assert RunRequest.parse({"ensemble": nested}).persist is None


class TestPathSafety:
    @pytest.mark.parametrize("fields", ESCAPES.values(), ids=ESCAPES.keys())
    def test_each_escape_raises_and_leaves_every_tree_untouched(
        self, tmp_path: Path, fields: dict[str, Any]
    ) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        run = outside / "run"
        run.mkdir()
        before = _tree(tmp_path)

        with pytest.raises(RunRequestError):
            materialize(_parse(**fields), run)

        assert _tree(tmp_path) == before

    def test_a_root_name_that_escapes_is_rejected(self, tmp_path: Path) -> None:
        run = tmp_path / "run"
        run.mkdir()

        with pytest.raises(RunRequestError):
            materialize(RunRequest.parse({"ensemble": {**ROOT, "name": "../r"}}), run)

        assert _tree(tmp_path) == {"run": 0}

    def test_a_target_resolving_outside_the_run_dir_is_refused_at_write(
        self, tmp_path: Path
    ) -> None:
        # A symlink inside the run dir pointing out: validation of the
        # key passes, the containment check on the write target does not.
        run = tmp_path / "run"
        (tmp_path / "outside").mkdir()
        run.mkdir()
        (run / "scripts").symlink_to(tmp_path / "outside")

        with pytest.raises(RunRequestError, match="outside"):
            materialize(_parse(scripts={"scripts/x.py": "print(1)"}), run)

        assert _tree(tmp_path / "outside") == {}


COLLISIONS = {
    "root and child differ in case": {"ensembles": {"ROOT": ROOT}},
    "children differ in case": {"ensembles": {"Kid": ROOT, "kid": ROOT}},
    "scripts differ in case": {"scripts": {"probe/x.py": "a", "probe/X.py": "b"}},
    "profiles differ in case": {
        "profiles": {"Seat": {"model": "m"}, "seat": {"model": "m"}}
    },
    "bind keys differ in case": {"bind": {"Seat": "b", "seat": "b"}},
    "profile and bind differ in case": {
        "profiles": {"Seat": {"model": "m"}},
        "bind": {"seat": "b"},
    },
    "scripts differ in unicode form": {
        "scripts": {"caf\u00e9.py": "a", "cafe\u0301.py": "b"}
    },
    "scripts differ in a full casefold (U+1FB7)": {
        "scripts": {"\u1fb7.py": "a", "\u1fbc\u0342.py": "b"}
    },
    "scripts differ in a full casefold (U+1FC7)": {
        "scripts": {"\u1fc7.py": "a", "\u1fcc\u0342.py": "b"}
    },
    "scripts differ in a full casefold (U+1FF7)": {
        "scripts": {"\u1ff7.py": "a", "\u1ffc\u0342.py": "b"}
    },
    "a script key is a directory of another": {
        "scripts": {"a.py": "x", "a.py/b.py": "y"}
    },
    "a script key is a directory of another, folded": {
        "scripts": {"A.py": "x", "a.py/b.py": "y"}
    },
    "one reference reaches two keys (scripts/ prefix)": {
        "scripts": {"x.py": "a", "scripts/x.py": "b"}
    },
    "one reference reaches two keys (hyphen form)": {
        "scripts": {"my-tool.py": "a", "scripts/my_tool.py": "b"}
    },
    "a script lands inside a profile file": {
        "profiles": {"a": {"model": "m"}},
        "scripts": {"profiles/a.yaml/z.py": "x"},
    },
}


class TestDistinctNames:
    @pytest.mark.parametrize("fields", COLLISIONS.values(), ids=COLLISIONS.keys())
    def test_names_that_meet_on_a_case_folding_disk_are_rejected(
        self, tmp_path: Path, fields: dict[str, Any]
    ) -> None:
        run = tmp_path / "run"
        run.mkdir()

        with pytest.raises(RunRequestError, match="distinct"):
            materialize(_parse(**fields), run)

        assert _tree(run) == {}

    def test_names_that_differ_by_more_than_case_are_accepted(self) -> None:
        request = _parse(
            ensembles={"kid": ROOT, "sub/kid": ROOT},
            profiles={"seat": {"model": "m"}, "seat2": {"model": "m"}},
            scripts={"a/x.py": "1", "a/y.py": "2", "a.py": "3"},
        )

        assert len(request.scripts) == 3


class TestScriptKeysHavePathSyntax:
    @pytest.mark.parametrize("key", ["date", "mark", "echo hi"])
    def test_a_key_with_no_path_syntax_is_rejected_naming_it(self, key: str) -> None:
        with pytest.raises(RunRequestError, match="path syntax") as raised:
            _parse(scripts={key: "echo hi"})

        assert repr(key) in str(raised.value)

    @pytest.mark.parametrize("key", ["x.py", "x.sh", "dir/x", "a/b/x.rb"])
    def test_a_key_with_a_slash_or_a_script_extension_is_accepted(
        self, key: str
    ) -> None:
        assert key in _parse(scripts={key: "echo hi"}).scripts


class TestMaterialize:
    def test_it_writes_every_kind_at_its_place(self, tmp_path: Path) -> None:
        run = tmp_path / "run"
        run.mkdir()
        request = _parse(
            ensembles={"sub/kid": {"description": "k", "agents": ROOT["agents"]}},
            profiles={"p": {"model": "m", "provider": "mock"}},
            scripts={"a/x.py": "print(1)"},
        )

        root_path = materialize(request, run)

        assert root_path == run / "ensembles" / "root.yaml"
        kid = yaml.safe_load((run / "ensembles" / "sub" / "kid.yaml").read_text())
        assert kid["name"] == "kid"
        profile = yaml.safe_load((run / "profiles" / "p.yaml").read_text())
        assert profile == {"name": "p", "model": "m", "provider": "mock"}
        assert (run / "a" / "x.py").read_text() == "print(1)"

    def test_a_named_root_returns_no_path(self, tmp_path: Path) -> None:
        run = tmp_path / "run"
        run.mkdir()

        request = RunRequest.parse(
            {"ensemble_name": "x", "scripts": {"a/x.py": "print(1)"}}
        )

        assert materialize(request, run) is None
        assert (run / "a" / "x.py").exists()


def _write_profile(base: Path, name: str, **fields: Any) -> None:
    path = base / "profiles" / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"name": name, **fields}))


@pytest.fixture
def host(tmp_path: Path) -> ConfigurationManager:
    (tmp_path / "proj" / ".llm-orc").mkdir(parents=True)
    return ConfigurationManager(project_dir=tmp_path / "proj", provision=False)


class TestApplyBindings:
    def test_the_view_resolves_the_key_to_the_targets_definition(
        self, tmp_path: Path, host: ConfigurationManager
    ) -> None:
        _write_profile(
            host.global_config_dir,
            "b",
            model="mock-b",
            provider="mock",
            fallback_model_profile="fb",
        )
        run = tmp_path / "run"
        run.mkdir()
        view = host.with_run_layer(run)

        applied, unmet = apply_bindings({"a": "b"}, view, run)

        assert (applied, unmet) == ({"a": "b"}, [])
        assert view.resolve_model_profile("a") == ("mock-b", "mock")
        profile = view.get_model_profile("a")
        assert profile is not None
        assert profile["fallback_model_profile"] == "fb"
        assert "a" not in host.get_model_profiles()

    def test_it_wins_over_a_host_profile_of_the_key_name(
        self, tmp_path: Path, host: ConfigurationManager
    ) -> None:
        _write_profile(host.global_config_dir, "a", model="mock-own", provider="mock")
        _write_profile(host.global_config_dir, "b", model="mock-b", provider="mock")
        run = tmp_path / "run"
        run.mkdir()
        view = host.with_run_layer(run)

        apply_bindings({"a": "b"}, view, run)

        assert view.resolve_model_profile("a") == ("mock-b", "mock")

    def test_a_target_defined_only_inline_works(
        self, tmp_path: Path, host: ConfigurationManager
    ) -> None:
        run = tmp_path / "run"
        run.mkdir()
        materialize(
            _parse(profiles={"b": {"model": "mock-inline", "provider": "mock"}}), run
        )
        view = host.with_run_layer(run)

        applied, unmet = apply_bindings({"a": "b"}, view, run)

        assert (applied, unmet) == ({"a": "b"}, [])
        assert view.resolve_model_profile("a") == ("mock-inline", "mock")

    def test_a_missing_target_is_unmet_and_writes_nothing(
        self, tmp_path: Path, host: ConfigurationManager
    ) -> None:
        run = tmp_path / "run"
        run.mkdir()
        view = host.with_run_layer(run)

        applied, unmet = apply_bindings({"a": "nope"}, view, run)

        assert (applied, unmet) == ({}, [("a", "nope")])
        assert _tree(run) == {}

    def test_a_target_is_read_before_any_binding_is_written(
        self, tmp_path: Path, host: ConfigurationManager
    ) -> None:
        # b is itself a bind key: one hop, so a gets b's host definition.
        _write_profile(host.global_config_dir, "b", model="mock-b", provider="mock")
        _write_profile(host.global_config_dir, "c", model="mock-c", provider="mock")
        run = tmp_path / "run"
        run.mkdir()
        view = host.with_run_layer(run)

        apply_bindings({"b": "c", "a": "b"}, view, run)

        assert view.resolve_model_profile("a") == ("mock-b", "mock")
        assert view.resolve_model_profile("b") == ("mock-c", "mock")


SERVING_DOC = Path(__file__).parents[4] / "docs" / "serving.md"


class TestTheDocumentedRequest:
    def _example(self) -> dict[str, Any]:
        """The first fenced JSON block under the request section."""
        text = SERVING_DOC.read_text()
        section = text.split("## Running a request", 1)[1]
        body = section.split("```json\n", 1)[1].split("\n```", 1)[0]
        return dict(json.loads(body))

    def test_it_validates_and_every_ensemble_in_it_loads(self, tmp_path: Path) -> None:
        request = RunRequest.parse(self._example())
        run = tmp_path / "run"
        run.mkdir()

        root_path = materialize(request, run)

        assert root_path is not None
        loader = EnsembleLoader()
        search_dirs = [str(run / "ensembles")]
        root = loader.load_from_file(str(root_path), search_dirs=search_dirs)
        child = loader.load_from_file(str(run / "ensembles" / "child.yaml"))
        assert [a.name for a in root.agents] == ["check", "sub", "review"]
        assert [a.name for a in child.agents] == ["c"]
