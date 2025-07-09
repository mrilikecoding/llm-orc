"""Configuration management system for llm-orc."""

import os
import shutil
from pathlib import Path
from typing import Any

import yaml


class ConfigurationManager:
    """Manages configuration directories and file locations."""

    def __init__(self) -> None:
        """Initialize configuration manager."""
        self._global_config_dir = self._get_global_config_dir()
        self._local_config_dir = self._discover_local_config()

        # Only create global config directory if migration is not needed
        if not self.needs_migration():
            self._global_config_dir.mkdir(parents=True, exist_ok=True)

    def _get_global_config_dir(self) -> Path:
        """Get the global configuration directory following XDG spec."""
        # Check for XDG_CONFIG_HOME environment variable
        xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
        if xdg_config_home:
            return Path(xdg_config_home) / "llm-orc"

        # Default to ~/.config/llm-orc
        return Path.home() / ".config" / "llm-orc"

    def _discover_local_config(self) -> Path | None:
        """Discover local .llm-orc directory walking up from cwd."""
        current = Path.cwd()

        # Stop at root directory or when we've walked up too far
        while current != current.parent:
            llm_orc_dir = current / ".llm-orc"
            if llm_orc_dir.exists() and llm_orc_dir.is_dir():
                return llm_orc_dir
            current = current.parent

            # Stop if we've reached the file system root
            if current == current.parent:
                break

        return None

    @property
    def global_config_dir(self) -> Path:
        """Get the global configuration directory."""
        return self._global_config_dir

    def ensure_global_config_dir(self) -> None:
        """Ensure the global configuration directory exists."""
        self._global_config_dir.mkdir(parents=True, exist_ok=True)

    @property
    def local_config_dir(self) -> Path | None:
        """Get the local configuration directory if found."""
        return self._local_config_dir

    def get_ensembles_dirs(self) -> list[Path]:
        """Get ensemble directories in priority order (local first, then global)."""
        dirs = []

        # Local config takes precedence
        if self._local_config_dir:
            local_ensembles = self._local_config_dir / "ensembles"
            if local_ensembles.exists():
                dirs.append(local_ensembles)

        # Global config as fallback
        global_ensembles = self._global_config_dir / "ensembles"
        if global_ensembles.exists():
            dirs.append(global_ensembles)

        return dirs

    def get_credentials_file(self) -> Path:
        """Get the credentials file path (always in global config)."""
        return self._global_config_dir / "credentials.yaml"

    def get_encryption_key_file(self) -> Path:
        """Get the encryption key file path (always in global config)."""
        return self._global_config_dir / ".encryption_key"

    def needs_migration(self) -> bool:
        """Check if migration from old ~/.llm-orc location is needed."""
        old_config_dir = Path.home() / ".llm-orc"
        return (
            old_config_dir.exists()
            and old_config_dir.is_dir()
            and not self._global_config_dir.exists()
        )

    def migrate_from_old_location(self) -> None:
        """Migrate configuration from old ~/.llm-orc to new location."""
        old_config_dir = Path.home() / ".llm-orc"

        if not old_config_dir.exists():
            return

        if self._global_config_dir.exists():
            raise ValueError("New configuration directory already exists")

        # Create parent directories
        self._global_config_dir.parent.mkdir(parents=True, exist_ok=True)

        # Move the entire directory
        shutil.move(str(old_config_dir), str(self._global_config_dir))

        # Leave a breadcrumb file in the old location
        breadcrumb_file = old_config_dir.parent / ".llm-orc-migrated"
        with open(breadcrumb_file, "w") as f:
            f.write(f"llm-orc configuration migrated to: {self._global_config_dir}\n")

    def load_project_config(self) -> dict[str, Any]:
        """Load project-specific configuration if available."""
        if not self._local_config_dir:
            return {}

        config_file = self._local_config_dir / "config.yaml"
        if not config_file.exists():
            return {}

        try:
            with open(config_file) as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}

    def init_local_config(self, project_name: str | None = None) -> None:
        """Initialize local configuration in current directory."""
        local_dir = Path.cwd() / ".llm-orc"

        if local_dir.exists():
            raise ValueError("Local .llm-orc directory already exists")

        # Create directory structure
        local_dir.mkdir()
        (local_dir / "ensembles").mkdir()
        (local_dir / "models").mkdir()
        (local_dir / "scripts").mkdir()

        # Create basic config file
        config_data = {
            "project": {
                "name": project_name or Path.cwd().name,
                "default_models": {"fast": "llama3", "production": "claude-3-5-sonnet"},
            },
            "model_profiles": {
                "development": [
                    {"model": "llama3", "provider": "ollama", "cost_per_token": 0.0}
                ],
                "production": [
                    {
                        "model": "claude-3-5-sonnet",
                        "provider": "anthropic",
                        "cost_per_token": 0.000003,
                    }
                ],
            },
        }

        config_file = local_dir / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)

        # Create .gitignore for credentials if they are stored locally
        gitignore_file = local_dir / ".gitignore"
        with open(gitignore_file, "w") as f:
            f.write("# Local credentials (if any)\ncredentials.yaml\n.encryption_key\n")
