"""Configuration utility functions for CLI operations."""

from pathlib import Path
from typing import Any

import click
import yaml


def safe_load_yaml(file_path: Path) -> dict[str, Any]:
    """Safely load YAML file with error handling.

    Args:
        file_path: Path to YAML file

    Returns:
        Loaded YAML data as dictionary, empty dict if file doesn't exist

    Raises:
        click.ClickException: If YAML parsing fails
    """
    if not file_path.exists():
        return {}

    try:
        with open(file_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise click.ClickException(
            f"Failed to parse YAML file {file_path}: {e}"
        ) from e
    except OSError as e:
        raise click.ClickException(
            f"Failed to read file {file_path}: {e}"
        ) from e


def safe_write_yaml(file_path: Path, data: dict[str, Any]) -> None:
    """Safely write YAML file with error handling.

    Args:
        file_path: Path to write YAML file
        data: Data to write as YAML

    Raises:
        click.ClickException: If YAML writing fails
    """
    try:
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
    except yaml.YAMLError as e:
        raise click.ClickException(
            f"Failed to write YAML file {file_path}: {e}"
        ) from e
    except OSError as e:
        raise click.ClickException(
            f"Failed to write file {file_path}: {e}"
        ) from e


def backup_config_file(file_path: Path) -> Path | None:
    """Create a backup of a configuration file.

    Args:
        file_path: Path to file to backup

    Returns:
        Path to backup file if created, None if original doesn't exist

    Raises:
        click.ClickException: If backup creation fails
    """
    if not file_path.exists():
        return None

    backup_path = file_path.with_suffix(f"{file_path.suffix}.backup")

    try:
        # Read original and write backup
        with open(file_path, "rb") as original:
            with open(backup_path, "wb") as backup:
                backup.write(original.read())

        return backup_path
    except OSError as e:
        raise click.ClickException(
            f"Failed to create backup of {file_path}: {e}"
        ) from e


def ensure_config_directory(config_dir: Path) -> None:
    """Ensure configuration directory exists.

    Args:
        config_dir: Path to configuration directory

    Raises:
        click.ClickException: If directory creation fails
    """
    try:
        config_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise click.ClickException(
            f"Failed to create config directory {config_dir}: {e}"
        ) from e


def remove_config_file(file_path: Path, description: str = "file") -> None:
    """Remove a configuration file with error handling.

    Args:
        file_path: Path to file to remove
        description: Description of file for error messages

    Raises:
        click.ClickException: If file removal fails
    """
    if not file_path.exists():
        return

    try:
        file_path.unlink()
    except OSError as e:
        raise click.ClickException(
            f"Failed to remove {description} {file_path}: {e}"
        ) from e
