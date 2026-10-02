"""The transitive closure of an ensemble (spec, Arc 3 re-cut ruling 4)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from llm_orc.core.config.closure import (
    Dependency,
    ScriptFilesOf,
    ScriptListing,
    walk_closure,
)
from llm_orc.core.config.ensemble_config import EnsembleConfig, EnsembleLoader
from llm_orc.core.execution.scripting.files_block import ListedFiles
from llm_orc.schemas.agent_config import (
    EnsembleAgentConfig,
    LoopAgentConfig,
    LoopSpec,
)


def _write(dir_path: Path, name: str, agents: list[dict[str, Any]]) -> None:
    (dir_path / f"{name}.yaml").write_text(
        yaml.safe_dump({"name": name, "description": name, "agents": agents})
    )


def _finder(dir_path: Path) -> Callable[[str], EnsembleConfig | None]:
    loader = EnsembleLoader()

    def find(name: str) -> EnsembleConfig | None:
        return loader.find_ensemble(str(dir_path), name)

    return find


@pytest.fixture
def ensembles(tmp_path: Path) -> Path:
    d = tmp_path / "ensembles"
    d.mkdir()
    return d


def _by_file(dir_path: Path) -> Callable[[str], EnsembleConfig | None]:
    """The executor's lookup: ``<dir>/<reference>.yaml``, by filename."""
    loader = EnsembleLoader()
    return lambda ref: loader._find_ensemble_in_dirs(ref, [str(dir_path)])


def _write_file(path: Path, name: str, agents: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump({"name": name, "description": name, "agents": agents})
    )


def _keys(deps: list[Dependency]) -> list[tuple[str, str, tuple[str, ...]]]:
    return [(d.kind, d.name, d.via) for d in deps]


class TestWalkClosure:
    def test_root_child_script_profile_and_dynamic_dispatch(
        self, ensembles: Path
    ) -> None:
        _write(
            ensembles,
            "child",
            [
                {"name": "runner", "script": "scripts/gone.py"},
                {"name": "thinker", "model_profile": "p-child"},
            ],
        )
        _write(
            ensembles,
            "top",
            [
                {"name": "writer", "model_profile": "p-top"},
                {"name": "sub", "ensemble": "child"},
                {"name": "router", "dispatch": "${sub.target}"},
                {"name": "fixed", "dispatch": "child"},
                {"name": "absent", "ensemble": "no-such"},
            ],
        )
        root = _finder(ensembles)("top")
        assert root is not None

        closure = walk_closure(
            root, _finder(ensembles), {"p-top": {}, "p-child": {}}, root_ref="top"
        )

        assert _keys(closure.dependencies) == [
            ("ensemble", "top", ()),
            ("profile", "p-top", ("top.writer",)),
            ("ensemble", "child", ("top.sub",)),
            ("script", "scripts/gone.py", ("top.sub", "child.runner")),
            ("profile", "p-child", ("top.sub", "child.thinker")),
            ("dispatch", "${sub.target}", ("top.router",)),
            ("ensemble", "no-such", ("top.absent",)),
        ]
        missing = next(d for d in closure.dependencies if d.name == "no-such")
        assert missing.found is False

    def test_second_reference_is_deduplicated_but_owned_by_both_frames(
        self, ensembles: Path
    ) -> None:
        """Review focus 1: the closure is a set; ownership is not."""
        _write(ensembles, "child", [{"name": "runner", "script": "scripts/x.py"}])
        _write(
            ensembles,
            "top",
            [
                {"name": "first", "ensemble": "child"},
                {"name": "second", "ensemble": "child"},
            ],
        )
        root = _finder(ensembles)("top")
        assert root is not None

        closure = walk_closure(root, _finder(ensembles), {}, root_ref="top")

        assert [d.name for d in closure.dependencies] == [
            "top",
            "child",
            "scripts/x.py",
        ]
        assert ("script", "scripts/x.py") in closure.owned["top.first"]
        assert ("script", "scripts/x.py") in closure.owned["top.second"]

    def test_loop_body_is_an_ensemble_dependency(self, ensembles: Path) -> None:
        _write(ensembles, "body", [{"name": "step", "model_profile": "p"}])
        _write(
            ensembles,
            "top",
            [
                {
                    "name": "looper",
                    "loop": {"body": "body", "until": "${done}", "max_iterations": 3},
                }
            ],
        )
        root = _finder(ensembles)("top")
        assert root is not None

        closure = walk_closure(root, _finder(ensembles), {"p": {}}, root_ref="top")

        assert _keys(closure.dependencies)[1:] == [
            ("ensemble", "body", ("top.looper",)),
            ("profile", "p", ("top.looper", "body.step")),
        ]

    def test_fallback_chain_is_followed_until_a_missing_hop(
        self, ensembles: Path
    ) -> None:
        """Review focus 2."""
        _write(
            ensembles,
            "top",
            [{"name": "w", "model_profile": "a", "fallback_model_profile": "x"}],
        )
        profiles: dict[str, dict[str, Any]] = {
            "a": {"fallback_model_profile": "b"},
            "b": {"fallback_model_profile": "gone"},
            "x": {},
        }
        root = _finder(ensembles)("top")
        assert root is not None

        closure = walk_closure(root, _finder(ensembles), profiles, root_ref="top")

        assert _keys(closure.dependencies)[1:] == [
            ("profile", "a", ("top.w",)),
            ("profile", "b", ("top.w",)),
            ("profile", "gone", ("top.w",)),
            ("profile", "x", ("top.w",)),
        ]
        gone = next(d for d in closure.dependencies if d.name == "gone")
        assert gone.found is False

    def test_inline_model_agent_is_a_model_dependency(self, ensembles: Path) -> None:
        _write(
            ensembles,
            "top",
            [{"name": "w", "model": "qwen3-8b", "provider": "llama-server"}],
        )
        root = _finder(ensembles)("top")
        assert root is not None

        closure = walk_closure(root, _finder(ensembles), {}, root_ref="top")

        dep = closure.dependencies[1]
        assert (dep.kind, dep.name, dep.provider) == (
            "model",
            "qwen3-8b",
            "llama-server",
        )

    def test_mutual_references_terminate(self) -> None:
        """The loader rejects cycles; the walker must not depend on that."""
        # Built directly: the loader would refuse this graph (Invariant 5).
        a = EnsembleConfig(
            name="a",
            description="a",
            agents=[EnsembleAgentConfig(name="to_b", ensemble="b")],
        )
        b = EnsembleConfig(
            name="b",
            description="b",
            agents=[EnsembleAgentConfig(name="to_a", ensemble="a")],
        )
        table = {"a": a, "b": b}

        closure = walk_closure(a, lambda n: table.get(n), {}, root_ref="a")

        assert [d.name for d in closure.dependencies] == ["a", "b"]

    def test_second_agent_on_a_profile_owns_the_rest_of_its_chain(
        self, ensembles: Path
    ) -> None:
        _write(
            ensembles,
            "top",
            [
                {"name": "w1", "model_profile": "b"},
                {"name": "w2", "model_profile": "b"},
            ],
        )
        profiles: dict[str, dict[str, Any]] = {"b": {"fallback_model_profile": "c"}}
        root = _finder(ensembles)("top")
        assert root is not None

        closure = walk_closure(root, _finder(ensembles), profiles, root_ref="top")

        assert ("profile", "c") in closure.owned["top.w1"]
        assert ("profile", "c") in closure.owned["top.w2"]

    def test_profile_cycle_terminates(self, ensembles: Path) -> None:
        _write(ensembles, "top", [{"name": "w", "model_profile": "a"}])
        profiles: dict[str, dict[str, Any]] = {
            "a": {"fallback_model_profile": "b"},
            "b": {"fallback_model_profile": "a"},
        }
        root = _finder(ensembles)("top")
        assert root is not None

        closure = walk_closure(root, _finder(ensembles), profiles, root_ref="top")

        assert [d.name for d in closure.dependencies[1:]] == ["a", "b"]

    def test_inline_model_agent_keeps_its_fallback_profile(
        self, ensembles: Path
    ) -> None:
        _write(
            ensembles,
            "top",
            [
                {
                    "name": "w",
                    "model": "qwen3-8b",
                    "provider": "llama-server",
                    "fallback_model_profile": "nope",
                }
            ],
        )
        root = _finder(ensembles)("top")
        assert root is not None

        closure = walk_closure(root, _finder(ensembles), {}, root_ref="top")

        fallback = next(d for d in closure.dependencies if d.name == "nope")
        assert (fallback.kind, fallback.found) == ("profile", False)

    def test_agent_level_fallback_is_one_hop_like_runtime(
        self, ensembles: Path
    ) -> None:
        """Runtime never walks the agent-level fallback's own chain."""
        _write(
            ensembles,
            "top",
            [{"name": "w", "model_profile": "a", "fallback_model_profile": "x"}],
        )
        profiles: dict[str, dict[str, Any]] = {
            "a": {},
            "x": {"fallback_model_profile": "y"},
        }
        root = _finder(ensembles)("top")
        assert root is not None

        closure = walk_closure(root, _finder(ensembles), profiles, root_ref="top")

        names = [d.name for d in closure.dependencies]
        assert "x" in names
        assert "y" not in names

    def test_grandchild_is_owned_by_a_second_frame_on_the_shared_child(
        self, ensembles: Path
    ) -> None:
        _write(ensembles, "gc", [{"name": "runner", "script": "scripts/deep.py"}])
        _write(ensembles, "ch", [{"name": "down", "ensemble": "gc"}])
        _write(
            ensembles,
            "top",
            [
                {"name": "first", "ensemble": "ch"},
                {"name": "second", "ensemble": "ch"},
            ],
        )
        root = _finder(ensembles)("top")
        assert root is not None

        closure = walk_closure(root, _finder(ensembles), {}, root_ref="top")

        assert ("script", "scripts/deep.py") in closure.owned["top.second"]
        assert ("ensemble", "gc") in closure.owned["top.second"]


class TestWalkerKeysOnReferences:
    """Visited-state, ownership and frames follow the reference string,
    never the ``name:`` field (fix wave C3)."""

    def test_two_files_sharing_a_name_are_both_walked(self, ensembles: Path) -> None:
        _write_file(
            ensembles / "one" / "same.yaml",
            "same",
            [{"name": "r", "script": "scripts/one.py"}],
        )
        _write_file(
            ensembles / "two" / "same.yaml",
            "same",
            [{"name": "r", "script": "scripts/two.py"}],
        )
        _write(
            ensembles,
            "top",
            [
                {"name": "a", "ensemble": "one/same"},
                {"name": "b", "ensemble": "two/same"},
            ],
        )
        find = _by_file(ensembles)
        root = find("top")
        assert root is not None

        closure = walk_closure(root, find, {}, root_ref="top")

        scripts = {d.name: d.via for d in closure.dependencies if d.kind == "script"}
        assert scripts == {
            "scripts/one.py": ("top.a", "one/same.r"),
            "scripts/two.py": ("top.b", "two/same.r"),
        }

    def test_second_frame_on_a_hierarchical_child_owns_its_subtree(
        self, ensembles: Path
    ) -> None:
        _write_file(
            ensembles / "grp" / "child.yaml",
            "child",
            [{"name": "r", "script": "scripts/deep.py"}],
        )
        _write(
            ensembles,
            "top",
            [
                {"name": "first", "ensemble": "grp/child"},
                {"name": "second", "ensemble": "grp/child"},
            ],
        )
        find = _by_file(ensembles)
        root = find("top")
        assert root is not None

        closure = walk_closure(root, find, {}, root_ref="top")

        assert ("script", "scripts/deep.py") in closure.owned["top.second"]

    def test_root_frames_use_the_reference_the_caller_passed(
        self, ensembles: Path
    ) -> None:
        _write_file(
            ensembles / "grp" / "top.yaml", "top", [{"name": "w", "model_profile": "p"}]
        )
        root = _by_file(ensembles)("grp/top")
        assert root is not None

        closure = walk_closure(root, _by_file(ensembles), {}, root_ref="grp/top")

        assert _keys(closure.dependencies) == [
            ("ensemble", "grp/top", ()),
            ("profile", "p", ("grp/top.w",)),
        ]
        assert set(closure.owned) == {"grp/top.w"}

    def test_cycle_through_a_loop_body_terminates(self) -> None:
        """Built directly: the loader would refuse this graph (Invariant 5)."""
        a = EnsembleConfig(
            name="a",
            description="a",
            agents=[
                LoopAgentConfig(
                    name="spin",
                    loop=LoopSpec(body="b", until="${done}", max_iterations=2),
                )
            ],
        )
        b = EnsembleConfig(
            name="b",
            description="b",
            agents=[EnsembleAgentConfig(name="back", ensemble="a")],
        )
        table = {"a": a, "b": b}

        closure = walk_closure(a, lambda n: table.get(n), {}, root_ref="a")

        assert [d.name for d in closure.dependencies] == ["a", "b"]
        assert ("ensemble", "a") in closure.owned["a.spin"]


class TestListedScriptFiles:
    """Ruling 8: files a script lists are closure members its frames own."""

    @staticmethod
    def _listings(table: dict[str, tuple[bool, tuple[str, ...]]]) -> ScriptFilesOf:
        def files_of(dep: Dependency) -> ScriptListing:
            found, paths = table.get(dep.name, (True, ()))
            return ScriptListing(found, ListedFiles(paths=paths))

        return files_of

    def _walk(
        self, ensembles: Path, files_of: ScriptFilesOf, script: str = "tools/x.py"
    ) -> list[Dependency]:
        _write(ensembles, "top", [{"name": "run", "script": script}])
        root = _finder(ensembles)("top")
        assert root is not None
        return walk_closure(
            root, _finder(ensembles), {}, root_ref="top", script_files=files_of
        ).dependencies

    def test_a_listed_file_is_a_dependency_named_beside_its_script(
        self, ensembles: Path
    ) -> None:
        deps = self._walk(
            ensembles, self._listings({"tools/x.py": (True, ("_helpers.py",))})
        )

        assert _keys(deps)[1:] == [
            ("script", "tools/x.py", ("top.run",)),
            ("script", "tools/_helpers.py", ("top.run",)),
        ]
        listed = deps[2]
        assert (listed.beside, listed.listed) == ("tools/x.py", "_helpers.py")

    def test_a_listed_file_that_lists_files_is_followed(self, ensembles: Path) -> None:
        deps = self._walk(
            ensembles,
            self._listings(
                {
                    "tools/x.py": (True, ("a/b.py",)),
                    "tools/a/b.py": (True, ("c.py",)),
                }
            ),
        )

        assert [d.name for d in deps[1:]] == [
            "tools/x.py",
            "tools/a/b.py",
            "tools/a/c.py",
        ]

    def test_files_that_list_each_other_terminate(self, ensembles: Path) -> None:
        deps = self._walk(
            ensembles,
            self._listings(
                {"tools/x.py": (True, ("y.py",)), "tools/y.py": (True, ("x.py",))}
            ),
        )

        assert [d.name for d in deps[1:]] == ["tools/x.py", "tools/y.py"]

    def test_a_listing_that_cannot_be_read_marks_the_script(
        self, ensembles: Path
    ) -> None:
        def files_of(dep: Dependency) -> ScriptListing:
            return ScriptListing(True, ListedFiles(error="not valid TOML"))

        [_, script] = self._walk(ensembles, files_of)

        assert script.problem == "not valid TOML"

    def test_the_frame_owns_the_listed_file(self, ensembles: Path) -> None:
        _write(ensembles, "top", [{"name": "run", "script": "tools/x.py"}])
        root = _finder(ensembles)("top")
        assert root is not None

        closure = walk_closure(
            root,
            _finder(ensembles),
            {},
            root_ref="top",
            script_files=self._listings({"tools/x.py": (True, ("_h.py",))}),
        )

        assert ("script", "tools/_h.py") in closure.owned["top.run"]
