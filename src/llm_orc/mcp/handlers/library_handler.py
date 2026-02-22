"""Library management handler for MCP server."""

import logging
import shutil
from pathlib import Path
from typing import Any

from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.core.config.ensemble_config import EnsembleLoader
from llm_orc.mcp.project_context import ProjectContext

logger = logging.getLogger(__name__)


class LibraryHandler:
    """Manages library browsing, copying, and search operations."""

    _test_library_dir: Path | None = None

    def __init__(
        self,
        config_manager: ConfigurationManager,
        ensemble_loader: EnsembleLoader,
    ) -> None:
        """Initialize with config manager and ensemble loader."""
        self._config_manager = config_manager
        self._ensemble_loader = ensemble_loader

    def set_project_context(self, ctx: ProjectContext) -> None:
        """Update handler to use new project context."""
        self._config_manager = ctx.config_manager

    def get_library_dir(self) -> Path:
        """Get library directory path."""
        if self._test_library_dir is not None:
            return self._test_library_dir

        for dir_path in self._config_manager.get_ensembles_dirs():
            if "library" in str(dir_path):
                return Path(dir_path).parent

        return Path.cwd() / "llm-orchestra-library"

    async def browse(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Browse library items."""
        browse_type = arguments.get("type", "all")
        library_dir = self.get_library_dir()
        result: dict[str, list[dict[str, Any]]] = {}

        if browse_type in ("all", "ensembles"):
            result["ensembles"] = self._browse_ensembles(library_dir)

        if browse_type in ("all", "scripts"):
            result["scripts"] = self._browse_scripts(library_dir)

        return result

    def _browse_ensembles(self, library_dir: Path) -> list[dict[str, Any]]:
        """Browse ensembles in library directory."""
        ensembles: list[dict[str, Any]] = []
        ensembles_dir = library_dir / "ensembles"
        if not ensembles_dir.exists():
            return ensembles

        for yaml_file in ensembles_dir.glob("**/*.yaml"):
            try:
                config = self._ensemble_loader.load_from_file(str(yaml_file))
                if config:
                    ensembles.append(
                        {
                            "name": config.name,
                            "description": config.description,
                            "path": str(yaml_file),
                        }
                    )
            except Exception:
                logger.debug(
                    "Skipping unreadable ensemble file %s", yaml_file, exc_info=True
                )
                continue
        return ensembles

    def _browse_scripts(self, library_dir: Path) -> list[dict[str, Any]]:
        """Browse scripts in library directory (recursive)."""
        scripts: list[dict[str, Any]] = []
        scripts_dir = library_dir / "scripts"
        if not scripts_dir.exists():
            return scripts

        for script_file in scripts_dir.rglob("*.py"):
            scripts.append(
                {
                    "name": script_file.stem,
                    "category": script_file.parent.name,
                    "path": str(script_file),
                }
            )
        return scripts

    def _resolve_copy_destination(self, source_path: Path) -> Path:
        """Resolve the default local destination for a library copy."""
        ensemble_dirs = self._config_manager.get_ensembles_dirs()
        local_dir = Path.cwd() / ".llm-orc"
        lib_dir = self.get_library_dir()

        for dir_path in ensemble_dirs:
            path = Path(dir_path)
            is_local = ".llm-orc" in str(path)
            is_library = str(path).startswith(str(lib_dir))
            if is_local and not is_library:
                local_dir = path.parent
                break

        subdir = "ensembles" if "ensembles" in str(source_path) else "scripts"
        return local_dir / subdir / source_path.name

    def _resolve_library_source(self, source: str) -> Path:
        """Resolve a library source string to a path."""
        library_dir = self.get_library_dir()
        source_path = library_dir / source

        if not source_path.exists() and not source.endswith(".yaml"):
            source_path = library_dir / f"{source}.yaml"

        if not source_path.exists():
            raise ValueError(f"Source not found in library: {source}")
        return source_path

    async def copy(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Copy from library to local."""
        source = arguments.get("source")
        destination = arguments.get("destination")
        overwrite = arguments.get("overwrite", False)

        if not source:
            raise ValueError("source is required")

        source_path = self._resolve_library_source(source)
        dest_path = (
            Path(destination)
            if destination
            else self._resolve_copy_destination(source_path)
        )

        if dest_path.exists() and not overwrite:
            raise ValueError(f"File already exists: {dest_path}")

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest_path)

        return {
            "copied": True,
            "source": str(source_path),
            "destination": str(dest_path),
        }

    async def search(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Search library content."""
        query = arguments.get("query", "").lower()

        if not query:
            raise ValueError("query is required")

        library_dir = self.get_library_dir()
        ensemble_results = self._search_ensembles(library_dir, query)
        script_results = self._search_scripts(library_dir, query)

        return {
            "query": query,
            "results": {
                "ensembles": ensemble_results,
                "scripts": script_results,
            },
            "total": len(ensemble_results) + len(script_results),
        }

    def _search_ensembles(self, library_dir: Path, query: str) -> list[dict[str, Any]]:
        """Search ensembles in library directory by query."""
        results: list[dict[str, Any]] = []
        ensembles_dir = library_dir / "ensembles"

        if not ensembles_dir.exists():
            return results

        for yaml_file in ensembles_dir.glob("**/*.yaml"):
            try:
                config = self._ensemble_loader.load_from_file(str(yaml_file))
                if config:
                    name_match = query in config.name.lower()
                    desc_match = query in (config.description or "").lower()
                    if name_match or desc_match:
                        results.append(
                            {
                                "name": config.name,
                                "description": config.description,
                                "path": str(yaml_file),
                                "match": ("name" if name_match else "description"),
                            }
                        )
            except Exception:
                logger.debug(
                    "Skipping unreadable ensemble file %s", yaml_file, exc_info=True
                )
                continue

        return results

    def _search_scripts(self, library_dir: Path, query: str) -> list[dict[str, Any]]:
        """Search scripts in library directory by query (recursive)."""
        results: list[dict[str, Any]] = []
        scripts_dir = library_dir / "scripts"

        if not scripts_dir.exists():
            return results

        for script_file in scripts_dir.rglob("*.py"):
            category = script_file.parent.name
            name_match = query in script_file.stem.lower()
            cat_match = query in category.lower()
            if name_match or cat_match:
                results.append(
                    {
                        "name": script_file.stem,
                        "category": category,
                        "path": str(script_file),
                        "match": ("name" if name_match else "category"),
                    }
                )

        return results

    async def info(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Get library information."""
        library_dir = self.get_library_dir()

        result: dict[str, Any] = {
            "path": str(library_dir),
            "exists": library_dir.exists(),
            "ensembles_count": 0,
            "scripts_count": 0,
            "categories": [],
        }

        if not library_dir.exists():
            return result

        ensembles_dir = library_dir / "ensembles"
        if ensembles_dir.exists():
            result["ensembles_count"] = len(list(ensembles_dir.glob("**/*.yaml")))

        scripts_dir = library_dir / "scripts"
        if scripts_dir.exists():
            all_scripts = list(scripts_dir.rglob("*.py"))
            result["scripts_count"] = len(all_scripts)
            result["categories"] = sorted(
                {script.parent.name for script in all_scripts}
            )

        return result
