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
            self._setup_default_config()
            self._setup_default_ensembles()

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
        self._setup_default_config()
        self._setup_default_ensembles()

    def _setup_default_config(self) -> None:
        """Set up default global config.yaml with model profiles."""
        config_file = self._global_config_dir / "config.yaml"

        # Only create if doesn't exist (don't overwrite user configurations)
        if config_file.exists():
            return

        # Create default configuration with characteristic-based model profiles
        default_config = {
            "model_profiles": {
                # Core usage profiles based on model characteristics
                "free-local": {
                    "model": "llama3",
                    "provider": "ollama",
                    "cost_per_token": 0.0,
                },
                "default-claude": {
                    "model": "claude-sonnet-4-20250514",
                    "provider": "anthropic-claude-pro-max",
                },
                "default-gemini": {
                    "model": "gemini-2.5-flash",
                    "provider": "google-gemini",
                    "cost_per_token": 1.0e-06,
                },
                "high-context": {
                    "model": "claude-3-5-sonnet-20241022",
                    "provider": "anthropic-api",
                    "cost_per_token": 3.0e-06,
                },
                "small": {
                    "model": "claude-3-haiku-20240307",
                    "provider": "anthropic-api",
                    "cost_per_token": 1.0e-06,
                },
                
                # Validation profiles for testing authentication
                "validate-anthropic-api": {
                    "model": "claude-3-5-sonnet-20241022",
                    "provider": "anthropic-api",
                    "cost_per_token": 3.0e-06,
                },
                "validate-claude-pro-max": {
                    "model": "claude-sonnet-4-20250514",
                    "provider": "anthropic-claude-pro-max",
                },
                "validate-google-gemini": {
                    "model": "gemini-2.5-flash",
                    "provider": "google-gemini",
                    "cost_per_token": 1.0e-06,
                },
                "validate-ollama": {
                    "model": "llama3",
                    "provider": "ollama",
                    "cost_per_token": 0.0,
                },
            }
        }

        with open(config_file, "w") as f:
            yaml.dump(default_config, f, default_flow_style=False, indent=2)

    def _setup_default_ensembles(self) -> None:
        """Set up default validation ensembles in the global config."""
        ensembles_dir = self._global_config_dir / "ensembles"
        ensembles_dir.mkdir(exist_ok=True)

        # Create validation ensembles for each provider
        validation_ensembles = [
            {
                "filename": "validate-anthropic-api.yaml",
                "config": {
                    "name": "validate-anthropic-api",
                    "description": "Validate Anthropic API authentication",
                    "agents": [
                        {
                            "name": "validator",
                            "model_profile": "validate-anthropic-api",
                            "role": "Respond with 'Authentication working'",
                        }
                    ],
                    "coordinator": {
                        "model_profile": "validate-anthropic-api",
                        "synthesis_prompt": "Return the response from the validator.",
                        "output_format": "text",
                    },
                },
            },
            {
                "filename": "validate-anthropic-claude-pro-max.yaml",
                "config": {
                    "name": "validate-anthropic-claude-pro-max",
                    "description": "Validate Claude Pro/Max OAuth authentication",
                    "agents": [
                        {
                            "name": "validator",
                            "model_profile": "validate-claude-pro-max",
                            "role": "Respond with 'Authentication working'",
                        }
                    ],
                    "coordinator": {
                        "model_profile": "validate-claude-pro-max",
                        "synthesis_prompt": "Return the response from the validator.",
                        "output_format": "text",
                    },
                },
            },
            {
                "filename": "validate-google-gemini.yaml",
                "config": {
                    "name": "validate-google-gemini",
                    "description": "Validate Google Gemini API authentication",
                    "agents": [
                        {
                            "name": "validator",
                            "model_profile": "validate-google-gemini",
                            "role": "Respond with 'Authentication working'",
                        }
                    ],
                    "coordinator": {
                        "model_profile": "validate-google-gemini",
                        "synthesis_prompt": "Return the response from the validator.",
                        "output_format": "text",
                    },
                },
            },
            {
                "filename": "validate-ollama.yaml",
                "config": {
                    "name": "validate-ollama",
                    "description": "Validate Ollama local model access",
                    "agents": [
                        {
                            "name": "validator",
                            "model_profile": "validate-ollama",
                            "role": "Respond with 'Authentication working'",
                        }
                    ],
                    "coordinator": {
                        "model_profile": "validate-ollama",
                        "synthesis_prompt": "Return the response from the validator.",
                        "output_format": "text",
                    },
                },
            },
        ]

        for ensemble_info in validation_ensembles:
            filename = ensemble_info["filename"]
            assert isinstance(filename, str)
            ensemble_file = ensembles_dir / filename
            if not ensemble_file.exists():
                with open(ensemble_file, "w") as f:
                    yaml.dump(ensemble_info["config"], f, default_flow_style=False)

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
                "default_models": {"fast": "free-local", "production": "default-claude"},
            },
            "model_profiles": {
                "free-local": {
                    "model": "llama3",
                    "provider": "ollama",
                    "cost_per_token": 0.0,
                },
                "default-claude": {
                    "model": "claude-sonnet-4-20250514",
                    "provider": "anthropic-claude-pro-max",
                },
            },
        }

        config_file = local_dir / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)

        # Create .gitignore for credentials if they are stored locally
        gitignore_file = local_dir / ".gitignore"
        with open(gitignore_file, "w") as f:
            f.write("# Local credentials (if any)\ncredentials.yaml\n.encryption_key\n")

    def get_model_profiles(self) -> dict[str, dict[str, str]]:
        """Get merged model profiles from global and local configs."""
        # Start with global profiles
        global_profiles = {}
        global_config_file = self._global_config_dir / "config.yaml"
        if global_config_file.exists():
            with open(global_config_file) as f:
                global_config = yaml.safe_load(f) or {}
                global_profiles = global_config.get("model_profiles", {})

        # Merge with local profiles (local overrides global)
        local_profiles = {}
        if self._local_config_dir:
            local_config_file = self._local_config_dir / "config.yaml"
            if local_config_file.exists():
                with open(local_config_file) as f:
                    local_config = yaml.safe_load(f) or {}
                    local_profiles = local_config.get("model_profiles", {})

        # Merge profiles with local taking precedence
        merged_profiles = {**global_profiles, **local_profiles}
        return merged_profiles

    def resolve_model_profile(self, profile_name: str) -> tuple[str, str]:
        """Resolve a model profile to (model, provider) tuple."""
        profiles = self.get_model_profiles()

        if profile_name not in profiles:
            raise ValueError(f"Model profile '{profile_name}' not found")

        profile = profiles[profile_name]
        model = profile.get("model")
        provider = profile.get("provider")

        if not model or not provider:
            raise ValueError(
                f"Model profile '{profile_name}' is incomplete. "
                f"Both 'model' and 'provider' are required."
            )

        return model, provider
