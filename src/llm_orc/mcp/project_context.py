"""Project context value object for atomic handler configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from llm_orc.core.config.config_manager import ConfigurationManager


@dataclass(frozen=True)
class ProjectContext:
    """Immutable project context shared by all handlers.

    Groups the project path and its derived ConfigurationManager
    into a single value object, making propagation on set_project
    atomic and type-safe.
    """

    project_path: Path | None
    config_manager: ConfigurationManager

    @classmethod
    def create(cls, path: str | Path | None = None) -> ProjectContext:
        """Create a ProjectContext for the given project path.

        Args:
            path: Project directory path, or None for global-only config.

        Returns:
            New ProjectContext instance.
        """
        if path is not None:
            resolved = Path(path).resolve()
            return cls(
                project_path=resolved,
                config_manager=ConfigurationManager(project_dir=resolved),
            )
        return cls(
            project_path=None,
            config_manager=ConfigurationManager(),
        )
