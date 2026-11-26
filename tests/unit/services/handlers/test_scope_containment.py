"""A CRUD name resolves only to a file inside the scope directory.

The names come from the network (MCP tools, REST path segments), so a name
with ``../`` or an absolute path must never reach a file outside the
scope's directory. Pins go through the handlers as a caller sees them.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.core.config.ensemble_config import EnsembleConfig
from llm_orc.schemas.agent_config import LlmAgentConfig
from llm_orc.services.handlers.artifact_handler import ArtifactHandler
from llm_orc.services.handlers.ensemble_crud_handler import EnsembleCrudHandler
from llm_orc.services.handlers.library_handler import LibraryHandler
from llm_orc.services.handlers.profile_handler import ProfileHandler
from llm_orc.services.handlers.promotion_handler import PromotionHandler
from llm_orc.services.handlers.resource_handler import ResourceHandler
from llm_orc.services.handlers.script_handler import ScriptHandler

_ENSEMBLE = (
    "name: outside\ndescription: d\nagents:\n  - name: a\n    model_profile: p\n"
)
_PROFILE = "name: outside\nprovider: llama-server\nmodel: m\n"


def _config(project: Path) -> ConfigurationManager:
    return ConfigurationManager(project_dir=project, provision=False)


def _ensemble_handler(project: Path) -> EnsembleCrudHandler:
    async def _read_artifact(ensemble_name: str, aid: str) -> dict[str, Any]:
        return {}

    return EnsembleCrudHandler(
        config_manager=_config(project),
        ensemble_loader=MagicMock(),
        find_ensemble_fn=lambda name: None,
        read_artifact_fn=_read_artifact,
    )


def _escaping_name(project: Path, style: str, scope_subdir: str) -> str:
    """A name that, appended with the file suffix, lands at
    ``<project>/.llm-orc/outside.<suffix>``, one level above the scope dir."""
    if style == "relative":
        return "../outside"
    return str(project / ".llm-orc" / "outside")


@pytest.fixture
def project(tmp_path: Path) -> Path:
    for sub in ("ensembles", "profiles", "scripts"):
        (tmp_path / ".llm-orc" / sub).mkdir(parents=True)
    return tmp_path


STYLES = ["relative", "absolute"]


class TestEnsembleNamesStayInScope:
    @pytest.mark.parametrize("style", STYLES)
    @pytest.mark.parametrize("dry_run", [True, False])
    async def test_update_refuses_and_leaves_the_outside_file(
        self, project: Path, style: str, dry_run: bool
    ) -> None:
        outside = project / ".llm-orc" / "outside.yaml"
        outside.write_text(_ENSEMBLE)
        name = _escaping_name(project, style, "ensembles")

        with pytest.raises(ValueError, match="outside"):
            await _ensemble_handler(project).update_ensemble(
                {
                    "ensemble_name": name,
                    "changes": {"remove_agents": ["a"]},
                    "dry_run": dry_run,
                }
            )

        assert outside.read_text() == _ENSEMBLE
        assert not outside.with_suffix(".yaml.bak").exists()

    @pytest.mark.parametrize("style", STYLES)
    async def test_delete_refuses_and_leaves_the_outside_file(
        self, project: Path, style: str
    ) -> None:
        outside = project / ".llm-orc" / "outside.yaml"
        outside.write_text(_ENSEMBLE)
        name = _escaping_name(project, style, "ensembles")

        with pytest.raises(ValueError, match="outside"):
            await _ensemble_handler(project).delete_ensemble(
                {"ensemble_name": name, "confirm": True}
            )

        assert outside.exists()

    @pytest.mark.parametrize("style", STYLES)
    async def test_create_refuses_and_writes_nothing_outside(
        self, project: Path, style: str
    ) -> None:
        name = _escaping_name(project, style, "ensembles")

        with pytest.raises(ValueError, match="outside"):
            await _ensemble_handler(project).create_ensemble(
                {
                    "name": name,
                    "agents": [{"name": "a", "model_profile": "p"}],
                }
            )

        assert not (project / ".llm-orc" / "outside.yaml").exists()

    async def test_refusal_names_no_absolute_path(self, project: Path) -> None:
        name = _escaping_name(project, "absolute", "ensembles")

        with pytest.raises(ValueError, match="not allowed") as excinfo:
            await _ensemble_handler(project).delete_ensemble(
                {"ensemble_name": name, "confirm": True}
            )

        assert str(project) not in str(excinfo.value)

    async def test_a_name_with_a_separator_inside_the_scope_still_works(
        self, project: Path
    ) -> None:
        nested = project / ".llm-orc" / "ensembles" / "sub" / "inner.yaml"
        nested.parent.mkdir()
        nested.write_text(_ENSEMBLE)

        result = await _ensemble_handler(project).delete_ensemble(
            {"ensemble_name": "sub/inner", "confirm": True}
        )

        assert result["deleted"] is True
        assert not nested.exists()


class TestProfileNamesStayInScope:
    @pytest.mark.parametrize("style", STYLES)
    async def test_update_and_delete_refuse(self, project: Path, style: str) -> None:
        outside = project / ".llm-orc" / "outside.yaml"
        outside.write_text(_PROFILE)
        name = _escaping_name(project, style, "profiles")
        handler = ProfileHandler(_config(project))

        with pytest.raises(ValueError, match="cannot contain a directory"):
            await handler.update_profile({"name": name, "changes": {"model": "x"}})
        with pytest.raises(ValueError, match="cannot contain a directory"):
            await handler.delete_profile({"name": name, "confirm": True})

        assert outside.read_text() == _PROFILE

    @pytest.mark.parametrize("style", STYLES)
    async def test_create_refuses(self, project: Path, style: str) -> None:
        name = _escaping_name(project, style, "profiles")
        handler = ProfileHandler(_config(project))

        with pytest.raises(ValueError, match="cannot contain a directory"):
            await handler.create_profile(
                {"name": name, "provider": "llama-server", "model": "m"}
            )

        assert not (project / ".llm-orc" / "outside.yaml").exists()


class TestScriptNamesStayInScope:
    def _handler(self, project: Path) -> ScriptHandler:
        return ScriptHandler(project_path=project, config_manager=_config(project))

    @pytest.mark.parametrize("category", ["..", "ABS"])
    async def test_delete_refuses(self, project: Path, category: str) -> None:
        outside = project / ".llm-orc" / "outside.py"
        outside.write_text("print('x')\n")
        if category == "ABS":
            category = str(project / ".llm-orc")

        with pytest.raises(ValueError, match="outside"):
            await self._handler(project).delete_script(
                {"name": "outside", "category": category, "confirm": True}
            )

        assert outside.exists()

    @pytest.mark.parametrize("category", ["..", "ABS"])
    async def test_create_refuses(self, project: Path, category: str) -> None:
        if category == "ABS":
            category = str(project / ".llm-orc")

        with pytest.raises(ValueError, match="outside"):
            await self._handler(project).create_script(
                {"name": "outside", "category": category}
            )

        assert not (project / ".llm-orc" / "outside.py").exists()


class TestScriptLookupsStayInScope:
    """``get_script`` and ``test_script`` find a script by ``category/name``;
    neither may read or run a file above the scripts directory."""

    def _handler(self, project: Path) -> ScriptHandler:
        return ScriptHandler(project_path=project, config_manager=_config(project))

    def _outside_script(self, project: Path) -> tuple[Path, Path]:
        marker = project / "ran.marker"
        script = project / ".llm-orc" / "outside.py"
        script.write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")
        return script, marker

    async def test_get_script_refuses_a_category_above_the_scripts_dir(
        self, project: Path
    ) -> None:
        self._outside_script(project)

        with pytest.raises(ValueError, match="outside"):
            await self._handler(project).get_script(
                {"name": "outside", "category": ".."}
            )

    async def test_test_script_does_not_run_a_script_above_the_scripts_dir(
        self, project: Path
    ) -> None:
        _script, marker = self._outside_script(project)

        with pytest.raises(ValueError, match="outside"):
            await self._handler(project).test_script(
                {"name": "outside", "category": ".."}
            )

        assert not marker.exists()

    async def test_a_longer_category_is_refused_too(self, project: Path) -> None:
        _script, marker = self._outside_script(project)

        with pytest.raises(ValueError, match="outside"):
            await self._handler(project).test_script(
                {"name": "outside", "category": "../../.llm-orc"}
            )

        assert not marker.exists()

    async def test_a_script_inside_a_category_still_runs(self, project: Path) -> None:
        inner = project / ".llm-orc" / "scripts" / "cat" / "ok.py"
        inner.parent.mkdir()
        inner.write_text("print('fine')\n")

        result = await self._handler(project).test_script(
            {"name": "ok", "category": "cat"}
        )

        assert result["stdout"].strip() == "fine"


class TestArtifactNamesStayInArtifacts:
    """An artifact id is ``<ensemble>/<timestamp>`` under the artifacts
    directory; neither part may name anything else."""

    @pytest.fixture
    def state(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> tuple[Path, Path]:
        """The artifacts directory and a sibling directory beside it."""
        monkeypatch.setenv("LLM_ORC_STATE_DIR", str(tmp_path / "state"))
        artifacts = tmp_path / "state" / "artifacts"
        (artifacts / "ens" / "20250101-000000").mkdir(parents=True)
        victim = tmp_path / "state" / "victim"
        (victim / "20250101-000000").mkdir(parents=True)
        (victim / "20250101-000000" / "execution.json").write_text(
            '{"results": {}, "secret": "outside"}'
        )
        return artifacts, victim

    def _handler(self, tmp_path: Path) -> ArtifactHandler:
        return ArtifactHandler(project_path=tmp_path, config_manager=_config(tmp_path))

    async def test_delete_refuses_an_ensemble_part_above_the_artifacts_dir(
        self, tmp_path: Path, state: tuple[Path, Path]
    ) -> None:
        _artifacts, victim = state

        with pytest.raises(ValueError, match="not allowed"):
            await self._handler(tmp_path).delete_artifact(
                {"artifact_id": "../victim", "confirm": True}
            )

        assert victim.exists()

    @pytest.mark.parametrize("artifact_id", ["..//ens", "ens/..", "./ens", "ens/."])
    async def test_delete_refuses_a_part_that_is_not_one_plain_name(
        self, tmp_path: Path, state: tuple[Path, Path], artifact_id: str
    ) -> None:
        artifacts, _victim = state

        with pytest.raises(ValueError, match="not allowed|Invalid"):
            await self._handler(tmp_path).delete_artifact(
                {"artifact_id": artifact_id, "confirm": True}
            )

        assert (artifacts / "ens" / "20250101-000000").exists()

    async def test_delete_refuses_an_ensemble_dir_that_links_outside(
        self, tmp_path: Path, state: tuple[Path, Path]
    ) -> None:
        artifacts, victim = state
        (artifacts / "link").symlink_to(victim)

        with pytest.raises(ValueError, match="not allowed"):
            await self._handler(tmp_path).delete_artifact(
                {"artifact_id": "link/20250101-000000", "confirm": True}
            )

        assert (victim / "20250101-000000").exists()

    async def test_cleanup_refuses_an_ensemble_name_above_the_artifacts_dir(
        self, tmp_path: Path, state: tuple[Path, Path]
    ) -> None:
        _artifacts, victim = state

        with pytest.raises(ValueError, match="not allowed"):
            await self._handler(tmp_path).cleanup_artifacts(
                {"ensemble_name": "../victim", "older_than_days": -1, "dry_run": False}
            )

        assert (victim / "20250101-000000").exists()

    async def test_a_valid_artifact_id_still_deletes(
        self, tmp_path: Path, state: tuple[Path, Path]
    ) -> None:
        artifacts, _victim = state

        await self._handler(tmp_path).delete_artifact(
            {"artifact_id": "ens/20250101-000000", "confirm": True}
        )

        assert not (artifacts / "ens" / "20250101-000000").exists()

    async def test_read_artifact_refuses_a_name_above_the_artifacts_dir(
        self, tmp_path: Path, state: tuple[Path, Path]
    ) -> None:
        handler = ResourceHandler(
            config_manager=_config(tmp_path), ensemble_loader=MagicMock()
        )

        with pytest.raises(ValueError, match="not allowed"):
            await handler.read_artifact("../victim", "20250101-000000")

    async def test_read_artifacts_refuses_a_name_above_the_artifacts_dir(
        self, tmp_path: Path, state: tuple[Path, Path]
    ) -> None:
        handler = ResourceHandler(
            config_manager=_config(tmp_path), ensemble_loader=MagicMock()
        )

        with pytest.raises(ValueError, match="not allowed"):
            await handler.read_artifacts("../victim")


class TestPromotionNamesStayInTheTier:
    """Promote, check and demote turn an ensemble name (and the profile
    names an ensemble lists) into files of a tier directory."""

    @pytest.fixture
    def tiers(self, tmp_path: Path) -> tuple[PromotionHandler, Path, Path]:
        """A handler, the global tier's directory and the project's
        ensembles directory."""
        global_dir = tmp_path / "global"
        (global_dir / "ensembles").mkdir(parents=True)
        (global_dir / "profiles").mkdir()
        project_ensembles = tmp_path / "proj" / "ensembles"
        project_ensembles.mkdir(parents=True)
        config = MagicMock()
        config.global_config_dir = global_dir
        config.get_ensembles_dirs.return_value = [str(project_ensembles)]
        config.classify_tier.return_value = "local"
        provider = MagicMock()
        provider.get_provider_status = AsyncMock(return_value={"providers": {}})
        handler = PromotionHandler(
            config_manager=config,
            profile_handler=MagicMock(),
            library_handler=LibraryHandler(
                config, MagicMock(), library_dir=tmp_path / "lib"
            ),
            provider_handler=provider,
            find_ensemble=lambda name: EnsembleConfig(
                name=name, description="d", agents=[]
            ),
        )
        return handler, global_dir, project_ensembles

    async def test_demote_refuses_a_name_above_the_tier_directory(
        self, tiers: tuple[PromotionHandler, Path, Path]
    ) -> None:
        handler, global_dir, _project = tiers
        outside = global_dir / "x.yaml"
        outside.write_text(_ENSEMBLE)

        with pytest.raises(ValueError, match="not allowed"):
            await handler.demote_ensemble(
                {"ensemble_name": "../x", "tier": "global", "confirm": True}
            )

        assert outside.exists()

    async def test_promote_refuses_a_name_above_the_source_directory(
        self, tiers: tuple[PromotionHandler, Path, Path]
    ) -> None:
        handler, global_dir, project_ensembles = tiers
        (project_ensembles.parent / "x.yaml").write_text(_ENSEMBLE)

        with pytest.raises(ValueError, match="not allowed"):
            await handler.promote_ensemble(
                {
                    "ensemble_name": "../x",
                    "destination": "global",
                    "include_profiles": False,
                    "dry_run": False,
                }
            )

        assert not (global_dir / "x.yaml").exists()

    async def test_check_refuses_a_name_above_the_tier_directory(
        self, tiers: tuple[PromotionHandler, Path, Path]
    ) -> None:
        handler, _global, _project = tiers

        with pytest.raises(ValueError, match="not allowed"):
            await handler.check_promotion_readiness(
                {"ensemble_name": "../x", "destination": "global"}
            )

    async def test_demote_does_not_remove_a_profile_that_names_a_file_above(
        self, tiers: tuple[PromotionHandler, Path, Path]
    ) -> None:
        handler, global_dir, _project = tiers
        ensemble = global_dir / "ensembles" / "e.yaml"
        ensemble.write_text(
            "name: e\ndescription: d\nagents:\n"
            "  - name: a\n    model_profile: ../victim\n"
        )
        (global_dir / "profiles" / "odd.yaml").write_text(
            "name: ../victim\nprovider: llama-server\nmodel: m\n"
        )
        victim = global_dir / "victim.yaml"
        victim.write_text("keep: me\n")

        with pytest.raises(ValueError, match="not allowed"):
            await handler.demote_ensemble(
                {
                    "ensemble_name": "e",
                    "tier": "global",
                    "remove_orphaned_profiles": True,
                    "confirm": True,
                }
            )

        assert victim.exists()
        assert ensemble.exists()

    async def test_promote_does_not_write_a_profile_above_the_profiles_dir(
        self, tiers: tuple[PromotionHandler, Path, Path]
    ) -> None:
        handler, global_dir, project_ensembles = tiers
        (project_ensembles / "e.yaml").write_text(_ENSEMBLE)
        handler._find_ensemble = lambda name: EnsembleConfig(
            name=name,
            description="d",
            agents=[LlmAgentConfig(name="a", model_profile="../planted")],
        )
        cast(Any, handler._profile_handler).get_all_profiles.return_value = {
            "../planted": {"provider": "llama-server", "model": "m"}
        }

        with pytest.raises(ValueError, match="not allowed"):
            await handler.promote_ensemble(
                {"ensemble_name": "e", "destination": "global", "dry_run": False}
            )

        assert not (global_dir / "planted.yaml").exists()
        assert not (global_dir / "ensembles" / "e.yaml").exists()


class TestLibraryCopyStaysInTheLibraryAndTheTier:
    """``library_copy`` reads from the library and writes into the local
    ensembles or scripts directory, and nowhere else."""

    @pytest.fixture
    def library(self, tmp_path: Path) -> Path:
        lib = tmp_path / "lib"
        (lib / "ensembles").mkdir(parents=True)
        (lib / "ensembles" / "good.yaml").write_text(_ENSEMBLE)
        return lib

    def _handler(self, project: Path, library: Path) -> LibraryHandler:
        return LibraryHandler(_config(project), MagicMock(), library_dir=library)

    async def test_source_above_the_library_is_refused(
        self, project: Path, library: Path
    ) -> None:
        secret = library.parent / "secret.yaml"
        secret.write_text("secret: true\n")

        with pytest.raises(ValueError, match="not allowed"):
            await self._handler(project, library).copy(
                {"source": "../secret.yaml", "overwrite": True}
            )

        assert not (project / ".llm-orc" / "ensembles" / "secret.yaml").exists()

    async def test_a_source_that_links_outside_the_library_is_refused(
        self, project: Path, library: Path
    ) -> None:
        secret = library.parent / "secret.yaml"
        secret.write_text("secret: true\n")
        (library / "ensembles" / "linked.yaml").symlink_to(secret)

        with pytest.raises(ValueError, match="not allowed"):
            await self._handler(project, library).copy(
                {"source": "ensembles/linked.yaml"}
            )

        assert not (project / ".llm-orc" / "ensembles" / "linked.yaml").exists()

    async def test_an_absolute_destination_outside_the_tier_is_refused(
        self, project: Path, library: Path
    ) -> None:
        target = project / "planted.yaml"

        with pytest.raises(ValueError, match="not allowed"):
            await self._handler(project, library).copy(
                {"source": "ensembles/good.yaml", "destination": str(target)}
            )

        assert not target.exists()

    async def test_a_relative_destination_above_the_tier_is_refused(
        self, project: Path, library: Path
    ) -> None:
        with pytest.raises(ValueError, match="not allowed"):
            await self._handler(project, library).copy(
                {"source": "ensembles/good.yaml", "destination": "../planted.yaml"}
            )

        assert not (project / ".llm-orc" / "planted.yaml").exists()

    async def test_a_destination_name_inside_the_tier_is_kept(
        self, project: Path, library: Path
    ) -> None:
        result = await self._handler(project, library).copy(
            {"source": "ensembles/good.yaml", "destination": "sub/renamed.yaml"}
        )

        written = project / ".llm-orc" / "ensembles" / "sub" / "renamed.yaml"
        assert written.read_text() == _ENSEMBLE
        assert Path(result["destination"]) == written


class TestWritesDoNotFollowALinkOutside:
    """A leaf that is a symlink (dangling or not) to a file outside the
    scope directory is never written through."""

    async def test_create_ensemble_refuses_a_dangling_link(self, project: Path) -> None:
        outside = project / "elsewhere.yaml"
        (project / ".llm-orc" / "ensembles" / "evil.yaml").symlink_to(outside)

        with pytest.raises(ValueError, match="not allowed"):
            await _ensemble_handler(project).create_ensemble(
                {"name": "evil", "agents": [{"name": "a", "model_profile": "p"}]}
            )

        assert not outside.exists()

    async def test_create_profile_refuses_a_dangling_link(self, project: Path) -> None:
        outside = project / "elsewhere.yaml"
        (project / ".llm-orc" / "profiles" / "evil.yaml").symlink_to(outside)

        with pytest.raises(ValueError, match="not allowed"):
            await ProfileHandler(_config(project)).create_profile(
                {"name": "evil", "provider": "llama-server", "model": "m"}
            )

        assert not outside.exists()

    async def test_update_profile_refuses_a_link_to_a_file_outside(
        self, project: Path
    ) -> None:
        outside = project / "elsewhere.yaml"
        outside.write_text(_PROFILE)
        (project / ".llm-orc" / "profiles" / "evil.yaml").symlink_to(outside)

        with pytest.raises(ValueError, match="not allowed"):
            await ProfileHandler(_config(project)).update_profile(
                {"name": "evil", "changes": {"model": "changed"}}
            )

        assert outside.read_text() == _PROFILE

    async def test_update_profile_still_writes_through_a_link_inside(
        self, project: Path
    ) -> None:
        profiles = project / ".llm-orc" / "profiles"
        real = profiles / "real.yaml"
        real.write_text(_PROFILE)
        (profiles / "alias.yaml").symlink_to(real)

        await ProfileHandler(_config(project)).update_profile(
            {"name": "alias", "changes": {"model": "changed"}}
        )

        assert "changed" in real.read_text()

    async def test_library_copy_refuses_a_dangling_link_at_the_destination(
        self, project: Path, tmp_path: Path
    ) -> None:
        library = tmp_path / "lib"
        (library / "ensembles").mkdir(parents=True)
        (library / "ensembles" / "good.yaml").write_text(_ENSEMBLE)
        outside = project / "elsewhere.yaml"
        (project / ".llm-orc" / "ensembles" / "good.yaml").symlink_to(outside)
        handler = LibraryHandler(_config(project), MagicMock(), library_dir=library)

        with pytest.raises(ValueError, match="not allowed"):
            await handler.copy({"source": "ensembles/good.yaml", "overwrite": True})

        assert not outside.exists()

    async def test_create_script_refuses_a_dangling_link(self, project: Path) -> None:
        outside = project / "elsewhere.py"
        scripts = project / ".llm-orc" / "scripts" / "cat"
        scripts.mkdir()
        (scripts / "evil.py").symlink_to(outside)

        with pytest.raises(ValueError, match="not allowed"):
            await ScriptHandler(
                project_path=project, config_manager=_config(project)
            ).create_script({"name": "evil", "category": "cat"})

        assert not outside.exists()


class TestPromoteDoesNotFollowALinkOutside:
    async def test_promote_refuses_a_dangling_link_at_the_destination(
        self, tmp_path: Path
    ) -> None:
        global_dir = tmp_path / "global"
        (global_dir / "ensembles").mkdir(parents=True)
        outside = tmp_path / "elsewhere.yaml"
        (global_dir / "ensembles" / "e.yaml").symlink_to(outside)
        source_dir = tmp_path / "proj" / "ensembles"
        source_dir.mkdir(parents=True)
        (source_dir / "e.yaml").write_text(_ENSEMBLE)
        config = MagicMock()
        config.global_config_dir = global_dir
        config.get_ensembles_dirs.return_value = [str(source_dir)]
        config.classify_tier.return_value = "local"
        handler = PromotionHandler(
            config_manager=config,
            profile_handler=MagicMock(),
            library_handler=MagicMock(),
            provider_handler=MagicMock(),
            find_ensemble=lambda name: EnsembleConfig(
                name=name, description="d", agents=[]
            ),
        )

        with pytest.raises(ValueError, match="not allowed"):
            await handler.promote_ensemble(
                {
                    "ensemble_name": "e",
                    "destination": "global",
                    "include_profiles": False,
                    "dry_run": False,
                    "overwrite": True,
                }
            )

        assert not outside.exists()


class TestUnusableNamesAreRefusals:
    """A name the filesystem cannot take is the caller's mistake, a
    ``ValueError``, wherever the lookup happens."""

    @pytest.mark.parametrize("method", ["delete_ensemble", "update_ensemble"])
    async def test_an_over_long_ensemble_name_is_refused(
        self, project: Path, method: str
    ) -> None:
        arguments: dict[str, Any] = {
            "ensemble_name": "a" * 4000,
            "confirm": True,
            "changes": {},
        }

        with pytest.raises(ValueError, match="not allowed"):
            await getattr(_ensemble_handler(project), method)(arguments)

    async def test_an_over_long_name_is_refused_for_profiles_and_scripts(
        self, project: Path
    ) -> None:
        with pytest.raises(ValueError, match="not allowed"):
            await ProfileHandler(_config(project)).delete_profile(
                {"name": "a" * 4000, "confirm": True}
            )
        with pytest.raises(ValueError, match="not allowed"):
            await ScriptHandler(
                project_path=project, config_manager=_config(project)
            ).get_script({"name": "a" * 4000, "category": "cat"})

    async def test_create_ensemble_into_a_new_subdirectory_makes_it(
        self, project: Path
    ) -> None:
        await _ensemble_handler(project).create_ensemble(
            {"name": "sub/inner", "agents": [{"name": "a", "model_profile": "p"}]}
        )

        assert (project / ".llm-orc" / "ensembles" / "sub" / "inner.yaml").exists()

    async def test_a_profile_name_with_a_directory_is_refused(
        self, project: Path
    ) -> None:
        handler = ProfileHandler(_config(project))

        with pytest.raises(ValueError, match="cannot contain a directory"):
            await handler.create_profile(
                {"name": "sub/inner", "provider": "llama-server", "model": "m"}
            )
        with pytest.raises(ValueError, match="cannot contain a directory"):
            await handler.update_profile({"name": "sub/inner", "changes": {}})
        with pytest.raises(ValueError, match="cannot contain a directory"):
            await handler.delete_profile({"name": "sub/inner", "confirm": True})

        assert not (project / ".llm-orc" / "profiles" / "sub").exists()


class TestLinksInsideTheScopeDirectory:
    """The containment check follows links: a parent directory that is a
    link out is out, and a scope directory that is itself a link is in."""

    async def test_a_parent_directory_that_links_outside_is_refused(
        self, project: Path
    ) -> None:
        elsewhere = project / "elsewhere"
        elsewhere.mkdir()
        (elsewhere / "x.yaml").write_text(_ENSEMBLE)
        (project / ".llm-orc" / "ensembles" / "link").symlink_to(elsewhere)
        handler = _ensemble_handler(project)

        with pytest.raises(ValueError, match="outside"):
            await handler.delete_ensemble({"ensemble_name": "link/x", "confirm": True})
        with pytest.raises(ValueError, match="outside"):
            await handler.create_ensemble(
                {"name": "link/y", "agents": [{"name": "a", "model_profile": "p"}]}
            )

        assert (elsewhere / "x.yaml").exists()
        assert not (elsewhere / "y.yaml").exists()

    @pytest.fixture
    def linked_scope(self, project: Path) -> Path:
        """The project's ensembles directory as a link to a real one."""
        ensembles = project / ".llm-orc" / "ensembles"
        ensembles.rmdir()
        real = project / "real-ensembles"
        real.mkdir()
        ensembles.symlink_to(real)
        return real

    async def test_create_works_in_a_scope_directory_that_is_a_link(
        self, project: Path, linked_scope: Path
    ) -> None:
        await _ensemble_handler(project).create_ensemble(
            {"name": "made", "agents": [{"name": "a", "model_profile": "p"}]}
        )

        assert (linked_scope / "made.yaml").exists()

    async def test_update_works_in_a_scope_directory_that_is_a_link(
        self, project: Path, linked_scope: Path
    ) -> None:
        (linked_scope / "e.yaml").write_text(_ENSEMBLE)

        result = await _ensemble_handler(project).update_ensemble(
            {
                "ensemble_name": "e",
                "changes": {"add_agents": [{"name": "b", "model_profile": "p"}]},
                "dry_run": False,
            }
        )

        assert result["modified"] is True
        assert "name: b" in (linked_scope / "e.yaml").read_text()

    async def test_delete_works_in_a_scope_directory_that_is_a_link(
        self, project: Path, linked_scope: Path
    ) -> None:
        (linked_scope / "e.yaml").write_text(_ENSEMBLE)

        await _ensemble_handler(project).delete_ensemble(
            {"ensemble_name": "e", "confirm": True}
        )

        assert not (linked_scope / "e.yaml").exists()


class TestAtomicWriteStagesBesideTheTarget:
    async def test_the_temporary_file_is_made_in_the_targets_directory(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = project / ".llm-orc" / "ensembles" / "e.yaml"
        target.write_text(_ENSEMBLE)
        moves: list[tuple[Path, Path]] = []
        real_replace = os.replace

        def spy(src: Any, dst: Any) -> None:
            moves.append((Path(src), Path(dst)))
            real_replace(src, dst)

        monkeypatch.setattr(os, "replace", spy)

        await _ensemble_handler(project).update_ensemble(
            {
                "ensemble_name": "e",
                "changes": {"add_agents": [{"name": "b", "model_profile": "p"}]},
                "dry_run": False,
            }
        )

        assert moves == [(moves[0][0], target)]
        assert moves[0][0].parent == target.parent


class TestArtifactLinksAreNotFollowed:
    """A linked ensemble or timestamp directory is skipped by a walk and
    refused when named, so nothing outside the artifacts directory is
    removed or listed through it."""

    @pytest.fixture
    def tree(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> tuple[Path, Path]:
        """The artifacts directory, and an outside directory with one
        timestamp directory that is old enough to be cleaned up."""
        monkeypatch.setenv("LLM_ORC_STATE_DIR", str(tmp_path / "state"))
        artifacts = tmp_path / "state" / "artifacts"
        artifacts.mkdir(parents=True)
        outside = tmp_path / "outside"
        (outside / "ts").mkdir(parents=True)
        (outside / "ts" / "execution.json").write_text("{}")
        os.utime(outside / "ts", (1, 1))
        return artifacts, outside

    def _handler(self, tmp_path: Path) -> ArtifactHandler:
        return ArtifactHandler(project_path=tmp_path, config_manager=_config(tmp_path))

    @pytest.mark.parametrize("dry_run", [True, False])
    async def test_named_cleanup_through_a_linked_ensemble_dir_is_refused(
        self, tmp_path: Path, tree: tuple[Path, Path], dry_run: bool
    ) -> None:
        artifacts, outside = tree
        (artifacts / "lnk").symlink_to(outside)

        with pytest.raises(ValueError, match="not allowed"):
            await self._handler(tmp_path).cleanup_artifacts(
                {"ensemble_name": "lnk", "older_than_days": 1, "dry_run": dry_run}
            )

        assert (outside / "ts" / "execution.json").exists()

    @pytest.mark.parametrize("dry_run", [True, False])
    async def test_unnamed_cleanup_skips_a_linked_ensemble_dir(
        self, tmp_path: Path, tree: tuple[Path, Path], dry_run: bool
    ) -> None:
        artifacts, outside = tree
        (artifacts / "lnk").symlink_to(outside)

        result = await self._handler(tmp_path).cleanup_artifacts(
            {"older_than_days": 1, "dry_run": dry_run}
        )

        assert result["count"] == 0
        assert (outside / "ts" / "execution.json").exists()

    @pytest.mark.parametrize("dry_run", [True, False])
    async def test_cleanup_skips_a_linked_timestamp_dir(
        self, tmp_path: Path, tree: tuple[Path, Path], dry_run: bool
    ) -> None:
        artifacts, outside = tree
        (artifacts / "ens").mkdir()
        (artifacts / "ens" / "ts").symlink_to(outside / "ts")

        result = await self._handler(tmp_path).cleanup_artifacts(
            {"ensemble_name": "ens", "older_than_days": 1, "dry_run": dry_run}
        )

        assert result["count"] == 0
        assert (outside / "ts" / "execution.json").exists()

    async def test_delete_refuses_a_linked_ensemble_or_timestamp_dir(
        self, tmp_path: Path, tree: tuple[Path, Path]
    ) -> None:
        artifacts, outside = tree
        (artifacts / "lnk").symlink_to(outside)
        (artifacts / "ens").mkdir()
        (artifacts / "ens" / "ts").symlink_to(outside / "ts")
        handler = self._handler(tmp_path)

        for artifact_id in ("lnk/ts", "ens/ts"):
            with pytest.raises(ValueError, match="not allowed"):
                await handler.delete_artifact(
                    {"artifact_id": artifact_id, "confirm": True}
                )

        assert (outside / "ts" / "execution.json").exists()

    async def test_reads_refuse_a_linked_ensemble_dir(
        self, tmp_path: Path, tree: tuple[Path, Path]
    ) -> None:
        artifacts, outside = tree
        (artifacts / "lnk").symlink_to(outside)
        handler = ResourceHandler(
            config_manager=_config(tmp_path), ensemble_loader=MagicMock()
        )

        with pytest.raises(ValueError, match="not allowed"):
            await handler.read_artifacts("lnk")
        with pytest.raises(ValueError, match="not allowed"):
            await handler.read_artifact("lnk", "ts")

    async def test_read_artifact_refuses_a_timestamp_link_that_leaves(
        self, tmp_path: Path, tree: tuple[Path, Path]
    ) -> None:
        artifacts, outside = tree
        (artifacts / "ens").mkdir()
        (artifacts / "ens" / "ts").symlink_to(outside / "ts")
        handler = ResourceHandler(
            config_manager=_config(tmp_path), ensemble_loader=MagicMock()
        )

        with pytest.raises(ValueError, match="not allowed"):
            await handler.read_artifact("ens", "ts")
