"""A CRUD name resolves only to a file inside the scope directory.

The names come from the network (MCP tools, REST path segments), so a name
with ``../`` or an absolute path must never reach a file outside the
scope's directory. Pins go through the handlers as a caller sees them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.services.handlers.ensemble_crud_handler import EnsembleCrudHandler
from llm_orc.services.handlers.profile_handler import ProfileHandler
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

        with pytest.raises(ValueError, match="outside"):
            await handler.update_profile({"name": name, "changes": {"model": "x"}})
        with pytest.raises(ValueError, match="outside"):
            await handler.delete_profile({"name": name, "confirm": True})

        assert outside.read_text() == _PROFILE

    @pytest.mark.parametrize("style", STYLES)
    async def test_create_refuses(self, project: Path, style: str) -> None:
        name = _escaping_name(project, style, "profiles")
        handler = ProfileHandler(_config(project))

        with pytest.raises(ValueError, match="outside"):
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
