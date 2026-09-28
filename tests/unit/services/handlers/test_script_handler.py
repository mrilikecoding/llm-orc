"""ScriptHandler scope: writes target one dir, reads merge project then global."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_orc.core.config.config_manager import (
    ConfigurationManager,
    resolve_global_config_dir,
)
from llm_orc.services.handlers.script_handler import ScriptHandler


def _handler(project: Path) -> ScriptHandler:
    return ScriptHandler(
        project_path=project,
        config_manager=ConfigurationManager(project_dir=project, provision=False),
    )


class TestScriptScope:
    async def test_global_scope_creates_under_global_scripts_dir(
        self, tmp_path: Path
    ) -> None:
        handler = _handler(tmp_path)
        expected = resolve_global_config_dir() / "scripts" / "util" / "shout.py"
        assert not expected.parent.exists()

        result = await handler.create_script(
            {"name": "shout", "category": "util", "scope": "global"}
        )

        assert result["scope"] == "global"
        assert Path(result["path"]) == expected
        assert not (tmp_path / ".llm-orc" / "scripts" / "util" / "shout.py").exists()

    async def test_default_scope_writes_project(self, tmp_path: Path) -> None:
        (tmp_path / ".llm-orc").mkdir()
        handler = _handler(tmp_path)

        result = await handler.create_script({"name": "mine", "category": "util"})

        assert result["scope"] == "project"
        assert (
            Path(result["path"])
            == tmp_path / ".llm-orc" / "scripts" / "util" / "mine.py"
        )

    async def test_list_reports_scope_and_includes_global(self, tmp_path: Path) -> None:
        (tmp_path / ".llm-orc").mkdir()
        handler = _handler(tmp_path)
        await handler.create_script({"name": "p", "category": "util"})
        await handler.create_script(
            {"name": "g", "category": "util", "scope": "global"}
        )

        listed = await handler.list_scripts({})

        by_name = {s["name"]: s["scope"] for s in listed["scripts"]}
        assert by_name == {"p": "project", "g": "global"}

    async def test_get_and_test_find_global_script(self, tmp_path: Path) -> None:
        handler = _handler(tmp_path)
        await handler.create_script(
            {"name": "g", "category": "util", "scope": "global"}
        )

        got = await handler.get_script({"name": "g", "category": "util"})
        ran = await handler.test_script(
            {"name": "g", "category": "util", "input": "ping"}
        )

        assert (
            Path(got["path"])
            == resolve_global_config_dir() / "scripts" / "util" / "g.py"
        )
        assert ran["success"] is True
        assert ran["stdout"].strip() == "ping"

    async def test_project_shadows_global_on_read(self, tmp_path: Path) -> None:
        (tmp_path / ".llm-orc").mkdir()
        handler = _handler(tmp_path)
        await handler.create_script(
            {"name": "twin", "category": "util", "scope": "global"}
        )
        await handler.create_script({"name": "twin", "category": "util"})

        got = await handler.get_script({"name": "twin", "category": "util"})

        assert (
            Path(got["path"]) == tmp_path / ".llm-orc" / "scripts" / "util" / "twin.py"
        )

    async def test_delete_with_wrong_scope_touches_nothing(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / ".llm-orc").mkdir()
        handler = _handler(tmp_path)
        created = await handler.create_script({"name": "keep", "category": "util"})

        with pytest.raises(ValueError, match=r"not in scope 'global'.*project tier"):
            await handler.delete_script(
                {
                    "name": "keep",
                    "category": "util",
                    "confirm": True,
                    "scope": "global",
                }
            )

        assert Path(created["path"]).exists()

    async def test_omitted_scope_never_deletes_a_global_only_script(
        self, tmp_path: Path
    ) -> None:
        handler = _handler(tmp_path)
        created = await handler.create_script(
            {"name": "only-global", "category": "util", "scope": "global"}
        )

        with pytest.raises(ValueError, match=r"not in scope 'project'.*global tier"):
            await handler.delete_script(
                {"name": "only-global", "category": "util", "confirm": True}
            )

        assert Path(created["path"]).exists()

    async def test_global_scope_without_config_manager_is_an_error(
        self, tmp_path: Path
    ) -> None:
        handler = ScriptHandler(project_path=tmp_path)

        with pytest.raises(ValueError, match="global"):
            await handler.create_script(
                {"name": "x", "category": "util", "scope": "global"}
            )

    async def test_project_scope_without_project_dir_errors_and_writes_nothing(
        self, tmp_path: Path
    ) -> None:
        handler = _handler(tmp_path)  # no .llm-orc created

        with pytest.raises(ValueError, match=r"No project directory.*scope: global"):
            await handler.create_script({"name": "orphan", "category": "util"})

        expected = resolve_global_config_dir() / "scripts" / "util" / "orphan.py"
        assert not expected.exists()

    async def test_omitted_scope_without_project_dir_never_deletes_global(
        self, tmp_path: Path
    ) -> None:
        handler = _handler(tmp_path)  # no .llm-orc created
        created = await handler.create_script(
            {"name": "only-global", "category": "util", "scope": "global"}
        )

        with pytest.raises(ValueError, match=r"not in scope 'project'.*global tier"):
            await handler.delete_script(
                {"name": "only-global", "category": "util", "confirm": True}
            )

        assert Path(created["path"]).exists()
