"""Protocol for template content providers."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class TemplateProvider(Protocol):
    """Supplies template file content to the configuration layer."""

    def get_template_content(self, template_name: str) -> str:
        """Return the text content of a named template.

        Args:
            template_name: File name (with or without .yaml extension).

        Raises:
            FileNotFoundError: If the template cannot be located.
        """
        ...

    def copy_profile_templates(self, target_dir: Path) -> None:
        """Copy profile templates into *target_dir* (idempotent).

        Args:
            target_dir: Destination directory for profile YAML files.
        """
        ...
