"""A script created with scope: global must be runnable, not only listable."""

from __future__ import annotations

from pathlib import Path

from llm_orc.core.config.config_manager import resolve_global_config_dir
from llm_orc.core.execution.scripting.resolver import ScriptResolver


def _write_global(rel: str, body: str) -> Path:
    path = resolve_global_config_dir() / "scripts" / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return path


class TestResolverGlobalScope:
    def test_resolves_global_script_when_project_lacks_it(self, tmp_path: Path) -> None:
        target = _write_global("util/shout.py", "print('hi')\n")

        resolved = ScriptResolver(project_dir=tmp_path).resolve_script_path(
            "util/shout.py"
        )

        assert Path(resolved) == target

    def test_project_script_shadows_global(self, tmp_path: Path) -> None:
        _write_global("util/twin.py", "print('global')\n")
        project_file = tmp_path / ".llm-orc" / "scripts" / "util" / "twin.py"
        project_file.parent.mkdir(parents=True)
        project_file.write_text("print('project')\n")

        resolved = ScriptResolver(project_dir=tmp_path).resolve_script_path(
            "util/twin.py"
        )

        assert Path(resolved) == project_file

    def test_global_dir_is_last_search_path(self, tmp_path: Path) -> None:
        _write_global("util/any.py", "")

        paths = ScriptResolver(project_dir=tmp_path)._get_search_paths()

        assert Path(paths[-1]) == resolve_global_config_dir() / "scripts"
