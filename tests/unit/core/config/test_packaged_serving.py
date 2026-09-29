"""The packaged serving tier: locator, classification, dir lists (#196)."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_orc.core.config import packaged
from llm_orc.core.config.config_manager import ConfigurationManager

REPO = Path(__file__).resolve().parents[4]


class TestLocator:
    def test_env_path_with_serving_ensemble_wins(
        self, packaged_serving_project: Path
    ) -> None:
        assert packaged.packaged_serving_project_dir() == packaged_serving_project

    def test_empty_env_disables_the_tier(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(packaged.SERVING_PROJECT_ENV, "")
        assert packaged.packaged_serving_project_dir() is None

    def test_env_path_without_serving_ensemble_is_loud(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(packaged.SERVING_PROJECT_ENV, str(tmp_path))
        with pytest.raises(FileNotFoundError, match="LLM_ORC_SERVING_PROJECT_DIR"):
            packaged.packaged_serving_project_dir()

    def test_unset_env_finds_this_checkouts_serving_project(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(packaged.SERVING_PROJECT_ENV)
        found = packaged.packaged_serving_project_dir()
        assert found == REPO / ".llm-orc"
        assert packaged.has_serving_ensemble(found)


class TestTiers:
    def test_packaged_is_the_last_ensembles_and_profiles_dir(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        packaged_serving_project: Path,
    ) -> None:
        project = tmp_path / "proj"
        (project / ".llm-orc" / "ensembles").mkdir(parents=True)
        (project / ".llm-orc" / "profiles").mkdir()
        monkeypatch.chdir(tmp_path)  # cwd is NOT the project
        cm = ConfigurationManager(project_dir=project, provision=True)

        ensembles = cm.get_ensembles_dirs()
        profiles = cm.get_profiles_dirs()

        assert ensembles[0] == project / ".llm-orc" / "ensembles"
        assert ensembles[-1] == packaged_serving_project / "ensembles"
        assert cm.global_config_dir / "ensembles" in ensembles
        assert ensembles.index(cm.global_config_dir / "ensembles") < len(ensembles) - 1
        assert profiles[-1] == packaged_serving_project / "profiles"

    def test_classify_tier_names_packaged(
        self, tmp_path: Path, packaged_serving_project: Path
    ) -> None:
        cm = ConfigurationManager(project_dir=tmp_path, provision=False)
        inside = (
            packaged_serving_project / "ensembles" / "agentic-serving" / "serving.yaml"
        )
        assert cm.classify_tier(inside) == "packaged"
        assert (
            cm.classify_tier(cm.global_config_dir / "ensembles" / "x.yaml") == "global"
        )

    def test_checkout_lists_its_dot_dir_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Review Focus 2: in a checkout the packaged dir IS the local dot-dir."""
        monkeypatch.delenv(packaged.SERVING_PROJECT_ENV)
        cm = ConfigurationManager(project_dir=REPO, provision=False)

        dirs = cm.get_ensembles_dirs()

        assert dirs[0] == REPO / ".llm-orc" / "ensembles"
        assert len(dirs) == len(set(dirs))
        assert cm.classify_tier(REPO / ".llm-orc" / "config.yaml") == "local"

    def test_library_dir_follows_the_project_not_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project = tmp_path / "proj"
        (project / ".llm-orc").mkdir(parents=True)
        (project / "llm-orchestra-library" / "ensembles").mkdir(parents=True)
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        monkeypatch.delenv("LLM_ORC_LIBRARY_PATH", raising=False)
        cm = ConfigurationManager(project_dir=project, provision=False)

        assert cm.library_dir == project / "llm-orchestra-library"
        assert (
            project / "llm-orchestra-library" / "ensembles" in cm.get_ensembles_dirs()
        )
        assert cm.classify_tier(project / "llm-orchestra-library" / "x") == "library"
