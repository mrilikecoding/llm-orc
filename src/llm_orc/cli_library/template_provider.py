"""Concrete TemplateProvider backed by the cli_library functions."""

from __future__ import annotations

from pathlib import Path

from llm_orc.cli_library.library import (
    copy_profile_templates,
    get_template_content,
)


class LibraryTemplateProvider:
    """Wraps cli_library functions as a TemplateProvider."""

    def get_template_content(self, template_name: str) -> str:
        """Return template content from the library source.

        Args:
            template_name: File name (with or without .yaml extension).

        Raises:
            FileNotFoundError: If the template cannot be located.
        """
        return get_template_content(template_name)

    def copy_profile_templates(self, target_dir: Path) -> None:
        """Copy profile templates into *target_dir* (idempotent).

        Args:
            target_dir: Destination directory for profile YAML files.
        """
        copy_profile_templates(target_dir)
