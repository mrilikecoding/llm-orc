"""Engine and legacy writes with no project never touch the packaged tier (#196)."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.core.config.ensemble_config import EnsembleConfig, EnsembleLoader
from llm_orc.core.validation.composition_validator import ConfigManagerEnsembleWriter
from llm_orc.services.handlers.ensemble_crud_handler import EnsembleCrudHandler
from llm_orc.services.handlers.library_handler import LibraryHandler
from llm_orc.services.handlers.profile_handler import ProfileHandler


def _crud(cm: ConfigurationManager) -> EnsembleCrudHandler:
    """The handler with real config and loader; callbacks unused by these paths."""
    return EnsembleCrudHandler(
        config_manager=cm,
        ensemble_loader=EnsembleLoader(),
        find_ensemble_fn=lambda name: None,
        read_artifact_fn=lambda *args, **kwargs: None,
    )


class TestNoProject:
    def test_composition_writer_lands_in_global(
        self, tmp_path: Path, packaged_serving_project: Path
    ) -> None:
        cm = ConfigurationManager(project_dir=tmp_path / "noproj", provision=False)
        config = EnsembleConfig(name="composed", description="", agents=[])

        written = Path(ConfigManagerEnsembleWriter(cm).write(config))

        assert written == cm.global_config_dir / "ensembles" / "composed.yaml"
        assert not list(packaged_serving_project.rglob("composed.yaml"))

    def test_legacy_project_dir_helpers_raise_naming_global(
        self, tmp_path: Path, packaged_serving_project: Path
    ) -> None:
        cm = ConfigurationManager(project_dir=tmp_path / "noproj", provision=False)
        with pytest.raises(ValueError, match="scope 'global'"):
            _crud(cm).get_local_ensembles_dir()
        with pytest.raises(ValueError, match="scope 'global'"):
            ProfileHandler(cm).get_local_profiles_dir()

    async def test_library_copy_lands_in_global(
        self, tmp_path: Path, packaged_serving_project: Path
    ) -> None:
        library = tmp_path / "lib"
        (library / "ensembles").mkdir(parents=True)
        (library / "ensembles" / "libcopy.yaml").write_text(
            "name: libcopy\ndescription: d\nagents: []\n"
        )
        cm = ConfigurationManager(project_dir=tmp_path / "noproj", provision=False)
        handler = LibraryHandler(cm, EnsembleLoader(), library_dir=library)

        result = await handler.copy({"source": "ensembles/libcopy.yaml"})

        expected = cm.global_config_dir / "ensembles" / "libcopy.yaml"
        assert Path(result["destination"]) == expected
        assert expected.exists()
        assert not list(packaged_serving_project.rglob("libcopy.yaml"))


class TestWithProject:
    def test_helpers_return_the_project_dir_even_before_it_has_subdirs(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / ".llm-orc").mkdir()
        cm = ConfigurationManager(project_dir=tmp_path, provision=False)
        crud_dir = _crud(cm).get_local_ensembles_dir()
        assert crud_dir == tmp_path / ".llm-orc" / "ensembles"
        profiles_dir = ProfileHandler(cm).get_local_profiles_dir()
        assert profiles_dir == tmp_path / ".llm-orc" / "profiles"


class TestGroupedListing:
    def test_packaged_ensembles_get_their_own_group(
        self, tmp_path: Path, packaged_serving_project: Path
    ) -> None:
        from llm_orc.services.orchestra_service import OrchestraService

        cm = ConfigurationManager(project_dir=tmp_path / "noproj", provision=False)
        grouped = OrchestraService(config_manager=cm).list_ensembles_grouped()

        assert {e.name for e in grouped["packaged"]} >= {"serving", "child"}
        assert not [e for e in grouped["global"] if e.name == "serving"]
