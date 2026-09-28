"""scope.py: the vocabulary and the one lookup every CRUD handler shares."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_orc.services.handlers.scope import find_in_scope, parse_scope


class TestParseScope:
    def test_absent_means_project(self) -> None:
        assert parse_scope({}) == "project"

    def test_global_is_accepted(self) -> None:
        assert parse_scope({"scope": "global"}) == "global"

    @pytest.mark.parametrize("bad", ["local", "Global", "", None])
    def test_unknown_value_is_rejected_naming_the_vocabulary(self, bad: object) -> None:
        with pytest.raises(ValueError, match=r"project.*global"):
            parse_scope({"scope": bad})


class TestFindInScope:
    def test_returns_file_in_requested_scope(self, tmp_path: Path) -> None:
        scope_dir = tmp_path / "global" / "ensembles"
        scope_dir.mkdir(parents=True)
        (scope_dir / "x.yaml").write_text("name: x\n")

        found = find_in_scope(
            name="x",
            filename="x.yaml",
            scope="global",
            scope_dir=scope_dir,
            search_dirs=[scope_dir],
            classify=lambda _p: "global",
            label="Ensemble",
        )

        assert found == scope_dir / "x.yaml"

    def test_name_in_other_tier_raises_naming_tier_and_path(
        self, tmp_path: Path
    ) -> None:
        project_dir = tmp_path / ".llm-orc" / "ensembles"
        project_dir.mkdir(parents=True)
        (project_dir / "x.yaml").write_text("name: x\n")
        global_dir = tmp_path / "global" / "ensembles"

        pattern = r"not in scope 'global'.*local tier.*x\.yaml"
        with pytest.raises(ValueError, match=pattern):
            find_in_scope(
                name="x",
                filename="x.yaml",
                scope="global",
                scope_dir=global_dir,
                search_dirs=[project_dir, global_dir],
                classify=lambda _p: "local",
                label="Ensemble",
            )

    def test_missing_everywhere_raises_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Ensemble not found: x"):
            find_in_scope(
                name="x",
                filename="x.yaml",
                scope="project",
                scope_dir=tmp_path,
                search_dirs=[tmp_path],
                classify=lambda _p: "local",
                label="Ensemble",
            )

    def test_no_scope_dir_falls_through_to_search(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Ensemble not found: x"):
            find_in_scope(
                name="x",
                filename="x.yaml",
                scope="project",
                scope_dir=None,
                search_dirs=[],
                classify=lambda _p: "local",
                label="Ensemble",
            )
