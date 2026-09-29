"""The packaged serving tier: locator, classification, dir lists (#196)."""

from __future__ import annotations

import os
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


class TestRuntimeProfiles:
    def test_packaged_profile_resolves_with_no_project_and_empty_global(
        self, tmp_path: Path, packaged_serving_project: Path
    ) -> None:
        cm = ConfigurationManager(project_dir=tmp_path, provision=False)

        model, provider = cm.resolve_model_profile("agentic-tier-cheap-general")

        assert (model, provider) == ("qwen3-8b", "llama-server")
        assert cm.get_model_profiles()["packaged-orch"]["model"] == "qwen3-0.6b"

    def test_within_packaged_profiles_dir_beats_config_yaml(
        self, tmp_path: Path, packaged_serving_project: Path
    ) -> None:
        cm = ConfigurationManager(project_dir=tmp_path, provision=False)
        shadowed = cm.get_model_profiles()["shadowed-by-file"]
        assert shadowed["model"] == "from-profiles-dir"

    def test_global_local_yaml_override_beats_packaged(
        self, tmp_path: Path, packaged_serving_project: Path
    ) -> None:
        cm = ConfigurationManager(project_dir=tmp_path, provision=False)
        override_dir = cm.global_config_dir / "profiles"
        override_dir.mkdir(parents=True)
        (override_dir / "seat.local.yaml").write_text(
            "name: agentic-tier-cheap-general\nmodel: bigger\nprovider: llama-server\n"
        )

        model, _ = cm.resolve_model_profile("agentic-tier-cheap-general")

        assert model == "bigger"

    def test_project_beats_global_beats_packaged(
        self, tmp_path: Path, packaged_serving_project: Path
    ) -> None:
        project = tmp_path / "proj"
        (project / ".llm-orc" / "profiles").mkdir(parents=True)
        (project / ".llm-orc" / "profiles" / "seat.yaml").write_text(
            "name: agentic-tier-cheap-general\nmodel: project\nprovider: llama-server\n"
        )
        cm = ConfigurationManager(project_dir=project, provision=False)
        (cm.global_config_dir / "profiles").mkdir(parents=True)
        (cm.global_config_dir / "profiles" / "seat.yaml").write_text(
            "name: agentic-tier-cheap-general\nmodel: global\nprovider: llama-server\n"
        )

        assert cm.resolve_model_profile("agentic-tier-cheap-general")[0] == "project"

    def test_library_profiles_stay_invisible_at_runtime(
        self,
        tmp_path: Path,
        packaged_serving_project: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Pins today's behavior (S1 step 1): listed, never resolved."""
        project = tmp_path / "proj"
        (project / ".llm-orc").mkdir(parents=True)
        lib_profiles = project / "llm-orchestra-library" / "profiles"
        lib_profiles.mkdir(parents=True)
        (lib_profiles / "lib-only.yaml").write_text(
            "name: lib-only\nmodel: m\nprovider: ollama\n"
        )
        monkeypatch.delenv("LLM_ORC_LIBRARY_PATH", raising=False)
        cm = ConfigurationManager(project_dir=project, provision=False)

        assert lib_profiles in cm.get_profiles_dirs()
        assert "lib-only" not in cm.get_model_profiles()

    def test_checkout_merges_its_dot_dir_once_at_top_precedence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Review Focus 2: packaged == local must not be merged below global."""
        monkeypatch.delenv(packaged.SERVING_PROJECT_ENV)
        cm = ConfigurationManager(project_dir=REPO, provision=False)

        tiers = cm._profile_tiers()

        assert tiers[-1] == REPO / ".llm-orc"
        assert tiers.count(REPO / ".llm-orc") == 1

    def test_cache_invalidates_when_a_packaged_profile_changes(
        self, tmp_path: Path, packaged_serving_project: Path
    ) -> None:
        cm = ConfigurationManager(project_dir=tmp_path, provision=False)
        assert cm.resolve_model_profile("agentic-tier-cheap-general")[0] == "qwen3-8b"
        target = (
            packaged_serving_project / "profiles" / "agentic-tier-cheap-general.yaml"
        )
        target.write_text(
            "name: agentic-tier-cheap-general\nmodel: edited\nprovider: llama-server\n"
        )
        os.utime(target, (target.stat().st_atime, target.stat().st_mtime + 5))

        assert cm.resolve_model_profile("agentic-tier-cheap-general")[0] == "edited"


class TestConfigMerges:
    def test_agentic_serving_orchestrator_comes_from_packaged(
        self, tmp_path: Path, packaged_serving_project: Path
    ) -> None:
        cm = ConfigurationManager(project_dir=tmp_path, provision=False)
        orchestrator = cm.load_agentic_serving_config()["orchestrator"]
        assert orchestrator["model_profile"] == "packaged-orch"

    def test_global_overrides_packaged_agentic_serving(
        self, tmp_path: Path, packaged_serving_project: Path
    ) -> None:
        cm = ConfigurationManager(project_dir=tmp_path, provision=False)
        cm.global_config_dir.mkdir(parents=True, exist_ok=True)
        (cm.global_config_dir / "config.yaml").write_text(
            "agentic_serving:\n  orchestrator:\n    model_profile: mine\n"
        )
        orchestrator = cm.load_agentic_serving_config()["orchestrator"]
        assert orchestrator["model_profile"] == "mine"

    def test_performance_merge_includes_packaged(
        self, tmp_path: Path, packaged_serving_project: Path
    ) -> None:
        (packaged_serving_project / "config.yaml").write_text(
            "performance:\n  execution:\n    default_timeout: 777\n"
        )
        cm = ConfigurationManager(project_dir=tmp_path, provision=False)
        assert cm.load_performance_config()["execution"]["default_timeout"] == 777


class TestServingRoot:
    def test_project_with_serving_ensemble_is_the_root(
        self, tmp_path: Path, packaged_serving_project: Path
    ) -> None:
        project = tmp_path / "proj"
        marker = project / ".llm-orc" / packaged.SERVING_MARKER
        marker.parent.mkdir(parents=True)
        marker.write_text("name: serving\nagents: []\n")
        cm = ConfigurationManager(project_dir=project, provision=False)
        assert cm.serving_root() == project / ".llm-orc"

    def test_project_without_it_falls_to_packaged(
        self, tmp_path: Path, packaged_serving_project: Path
    ) -> None:
        project = tmp_path / "proj"
        (project / ".llm-orc" / "ensembles").mkdir(parents=True)
        cm = ConfigurationManager(project_dir=project, provision=False)
        assert cm.serving_root() == packaged_serving_project

    def test_no_serving_ensemble_anywhere_names_both_candidates(
        self, tmp_path: Path
    ) -> None:
        """Review Focus 5."""
        project = tmp_path / "proj"
        (project / ".llm-orc").mkdir(parents=True)
        cm = ConfigurationManager(project_dir=project, provision=False)
        with pytest.raises(
            FileNotFoundError,
            match=r"agentic-serving/serving\.yaml.*\.llm-orc.*None",
        ):
            cm.serving_root()
