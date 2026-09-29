"""Packaged serving scripts resolve when no project carries them (#196)."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_orc.core.config.config_manager import resolve_global_config_dir
from llm_orc.core.execution.scripting.resolver import ScriptResolver

CLASSIFY = "scripts/agentic_serving/classify.py"


class TestResolverPackagedTier:
    def test_serving_script_resolves_from_packaged_with_empty_project(
        self, tmp_path: Path, packaged_serving_project: Path
    ) -> None:
        resolved = ScriptResolver(project_dir=tmp_path).resolve_script_path(CLASSIFY)

        expected = packaged_serving_project / "scripts" / "agentic_serving"
        assert Path(resolved) == expected / "classify.py"

    def test_project_script_shadows_packaged(
        self, tmp_path: Path, packaged_serving_project: Path
    ) -> None:
        mine = tmp_path / ".llm-orc" / "scripts" / "agentic_serving" / "classify.py"
        mine.parent.mkdir(parents=True)
        mine.write_text("print('mine')\n")

        resolved = ScriptResolver(project_dir=tmp_path).resolve_script_path(CLASSIFY)

        assert Path(resolved) == mine

    def test_global_script_shadows_packaged(
        self, tmp_path: Path, packaged_serving_project: Path
    ) -> None:
        target = (
            resolve_global_config_dir() / "scripts" / "agentic_serving" / "classify.py"
        )
        target.parent.mkdir(parents=True)
        target.write_text("print('global')\n")

        resolved = ScriptResolver(project_dir=tmp_path).resolve_script_path(CLASSIFY)

        assert Path(resolved) == target

    def test_packaged_dirs_are_the_last_search_paths(
        self, tmp_path: Path, packaged_serving_project: Path
    ) -> None:
        paths = ScriptResolver(project_dir=tmp_path)._get_search_paths()

        assert paths[-2:] == [
            str(packaged_serving_project / "scripts"),
            str(packaged_serving_project),
        ]

    def test_no_packaged_tier_adds_nothing(self, tmp_path: Path) -> None:
        paths = ScriptResolver(project_dir=tmp_path)._get_search_paths()

        assert not any("serving_project" in p for p in paths)


class TestListAvailableScriptsHonorsProjectDir:
    def test_lists_the_project_scripts_not_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project = tmp_path / "proj"
        script = project / ".llm-orc" / "scripts" / "util" / "hello.py"
        script.parent.mkdir(parents=True)
        script.write_text("print('hi')\n")
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)

        listed = ScriptResolver(project_dir=project).list_available_scripts()

        assert "util/hello.py" in {s["display_name"] for s in listed}
