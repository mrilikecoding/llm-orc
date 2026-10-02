"""A run request is validated, then written into a run layer (Arc 4).

Every escape shape raises before anything is written, anywhere. Bindings
are materialized as layer profiles, read from the view as it stood before
any binding was written.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from llm_orc.core.config.config_manager import ConfigurationManager
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
