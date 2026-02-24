"""Configuration management system for llm-orc."""

import os
import shutil
from pathlib import Path
from typing import Any

import yaml


class ConfigurationManager:
    """Manages configuration directories and file locations."""

    def __init__(self, project_dir: Path | None = None) -> None:
        """Initialize configuration manager.

        Args:
            project_dir: Optional project directory. If provided, uses
                project_dir/.llm-orc as local config instead of discovering
                from cwd.
        """
        self._global_config_dir = self._get_global_config_dir()
        if project_dir is not None:
            # Use explicit project directory
            llm_orc_dir = project_dir / ".llm-orc"
            self._local_config_dir = llm_orc_dir if llm_orc_dir.exists() else None
        else:
            # Discover from cwd
            self._local_config_dir = self._discover_local_config()

        # Profile cache
        self._profiles_cache: dict[str, dict[str, str]] | None = None
        self._profiles_cache_mtimes: dict[str, float] = {}

        # Create global config directory and setup defaults
        self._global_config_dir.mkdir(parents=True, exist_ok=True)
        (self._global_config_dir / "profiles").mkdir(exist_ok=True)
        self._setup_default_config()
        self._setup_default_ensembles()
        self._copy_profile_templates(self._global_config_dir / "profiles")

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
        (self._global_config_dir / "profiles").mkdir(exist_ok=True)
        self._setup_default_config()
        self._setup_default_ensembles()
        self._copy_profile_templates(self._global_config_dir / "profiles")

    def _setup_default_config(self) -> None:
        """Set up default global config.yaml by copying template content."""
        config_file = self._global_config_dir / "config.yaml"

        # Only create if doesn't exist (don't overwrite user configurations)
        if config_file.exists():
            return

        try:
            # Get the template config content from library
            template_content = self._get_template_config_content("global-config.yaml")
            with open(config_file, "w", encoding="utf-8") as f:
                f.write(template_content)
        except FileNotFoundError:
            # Fallback to empty config if template not found
            with open(config_file, "w", encoding="utf-8") as f:
                yaml.dump({"model_profiles": {}}, f, default_flow_style=False, indent=2)

    def _setup_default_ensembles(self) -> None:
        """Set up default validation ensembles by copying template files."""
        ensembles_dir = self._global_config_dir / "ensembles"
        ensembles_dir.mkdir(exist_ok=True)

        # Get the template ensembles directory
        template_dir = self._get_template_ensembles_dir()

        if not template_dir.exists():
            # Fallback to empty directory if templates not found
            return

        # Copy each template file to the ensembles directory if it doesn't exist
        for template_file in template_dir.glob("*.yaml"):
            target_file = ensembles_dir / template_file.name
            if not target_file.exists():
                shutil.copy2(template_file, target_file)

    def _get_template_ensembles_dir(self) -> Path:
        """Get the template ensembles directory path."""
        # Get the llm_orc package directory (parent of core)
        package_dir = Path(__file__).parent.parent.parent
        return package_dir / "templates" / "ensembles"

    def _get_template_config_content(self, filename: str) -> str:
        """Get template config content from library repository."""
        from llm_orc.cli_library.library import get_template_content

        try:
            return get_template_content(filename)
        except FileNotFoundError:
            # Fallback to local template if library template not found
            package_dir = Path(__file__).parent.parent.parent
            local_template_path = package_dir / "templates" / filename

            if local_template_path.exists():
                with open(local_template_path, encoding="utf-8") as f:
                    return f.read()
            else:
                raise FileNotFoundError(f"Template not found: {filename}") from None

    @property
    def local_config_dir(self) -> Path | None:
        """Get the local configuration directory if found."""
        return self._local_config_dir

    def classify_tier(self, path: Path) -> str:
        """Classify a path as belonging to the local, global, or library tier.

        Args:
            path: Path to classify (file or directory).

        Returns:
            One of ``"local"``, ``"global"``, ``"library"``, or ``"unknown"``.
        """
        library_dir = self._get_library_dir()

        if self._local_config_dir and path.is_relative_to(self._local_config_dir):
            return "local"
        if library_dir is not None and path.is_relative_to(library_dir):
            return "library"
        if path.is_relative_to(self._global_config_dir):
            return "global"
        return "unknown"

    def _get_library_dir(self) -> Path | None:
        """Resolve the library base directory (without requiring it to exist).

        Resolution order:
        1. ``LLM_ORC_LIBRARY_PATH`` environment variable
        2. ``<cwd>/llm-orchestra-library`` (default submodule location)
        """
        library_path_env = os.environ.get("LLM_ORC_LIBRARY_PATH")
        if library_path_env:
            return Path(library_path_env)
        return Path.cwd() / "llm-orchestra-library"

    def get_ensembles_dirs(self) -> list[Path]:
        """Get ensemble directories in priority order (local → library → global).

        Library path resolution:
        1. LLM_ORC_LIBRARY_PATH env var (custom location)
        2. Current directory/llm-orchestra-library (submodule)
        """
        dirs = []
        cwd = Path.cwd()

        # Priority 1: Local config takes precedence
        if self._local_config_dir:
            local_ensembles = self._local_config_dir / "ensembles"
            if local_ensembles.exists():
                dirs.append(local_ensembles)

        # Priority 2: Library ensembles
        # Check for custom library path from environment
        import os

        library_path_env = os.environ.get("LLM_ORC_LIBRARY_PATH")
        if library_path_env:
            library_ensembles = Path(library_path_env) / "ensembles"
            if library_ensembles.exists():
                dirs.append(library_ensembles)
        else:
            # Default: check for library submodule in current directory
            library_ensembles = cwd / "llm-orchestra-library" / "ensembles"
            if library_ensembles.exists():
                dirs.append(library_ensembles)

        # Priority 3: Global config as fallback
        global_ensembles = self._global_config_dir / "ensembles"
        if global_ensembles.exists():
            dirs.append(global_ensembles)

        return dirs

    def get_profiles_dirs(self) -> list[Path]:
        """Get profile directories in priority order (local → library → global).

        Library path resolution:
        1. LLM_ORC_LIBRARY_PATH env var (custom location)
        2. Current directory/llm-orchestra-library (submodule)
        """
        dirs: list[Path] = []
        cwd = Path.cwd()

        # Priority 1: Local config takes precedence
        if self._local_config_dir:
            local_profiles = self._local_config_dir / "profiles"
            if local_profiles.exists():
                dirs.append(local_profiles)

        # Priority 2: Library profiles
        library_path_env = os.environ.get("LLM_ORC_LIBRARY_PATH")
        if library_path_env:
            library_profiles = Path(library_path_env) / "profiles"
            if library_profiles.exists():
                dirs.append(library_profiles)
        else:
            # Default: check for library submodule in current directory
            library_profiles = cwd / "llm-orchestra-library" / "profiles"
            if library_profiles.exists():
                dirs.append(library_profiles)

        # Priority 3: Global config as fallback
        global_profiles = self._global_config_dir / "profiles"
        if global_profiles.exists():
            dirs.append(global_profiles)

        return dirs

    def get_credentials_file(self) -> Path:
        """Get the credentials file path (always in global config)."""
        return self._global_config_dir / "credentials.yaml"

    def get_encryption_key_file(self) -> Path:
        """Get the encryption key file path (always in global config)."""
        return self._global_config_dir / ".encryption_key"

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
        except (yaml.YAMLError, OSError):
            return {}

    def _load_global_config(self) -> dict[str, Any]:
        """Load global configuration from config.yaml file."""
        config_file = self._global_config_dir / "config.yaml"
        if not config_file.exists():
            return {}

        try:
            with open(config_file) as f:
                return yaml.safe_load(f) or {}
        except (yaml.YAMLError, OSError):
            return {}

    def load_performance_config(self) -> dict[str, Any]:
        """Load performance configuration with sensible defaults."""
        # Default performance settings
        defaults = {
            "concurrency": {
                "max_concurrent_agents": 0,  # 0 = use smart defaults
                "connection_pool": {
                    "max_connections": 100,
                    "max_keepalive": 20,
                    "keepalive_expiry": 30,
                },
            },
            "execution": {
                "default_timeout": 60,
                "monitoring_enabled": True,
                "streaming_enabled": True,
            },
            "memory": {
                "efficient_mode": False,
                "max_memory_mb": 0,  # 0 = unlimited
            },
        }

        # Try to load from global config
        global_config = self._load_global_config()
        global_performance = global_config.get("performance", {})

        # Try to load from local config
        local_config = self.load_project_config()
        local_performance = local_config.get("performance", {})

        # Merge configurations: defaults -> global -> local
        merged_config = defaults.copy()
        self._deep_merge_dict(merged_config, global_performance)
        self._deep_merge_dict(merged_config, local_performance)

        return merged_config

    def _deep_merge_dict(self, base: dict[str, Any], overlay: dict[str, Any]) -> None:
        """Deep merge overlay dict into base dict."""
        for key, value in overlay.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge_dict(base[key], value)
            else:
                base[key] = value

    def init_local_config(self, project_name: str | None = None) -> None:
        """Initialize local configuration in current directory (idempotent).

        Args:
            project_name: Optional project name (defaults to directory name)
        """
        local_dir = Path.cwd() / ".llm-orc"

        # Create directory structure (idempotent)
        local_dir.mkdir(exist_ok=True)
        (local_dir / "ensembles").mkdir(exist_ok=True)
        (local_dir / "models").mkdir(exist_ok=True)
        (local_dir / "scripts").mkdir(exist_ok=True)
        (local_dir / "profiles").mkdir(exist_ok=True)

        # Create config file from template (idempotent)
        config_file = local_dir / "config.yaml"

        if not config_file.exists():
            try:
                # Get template content from library
                template_content = self._get_template_config_content(
                    "local-config.yaml"
                )

                # Replace placeholder with actual project name
                actual_project_name = project_name or Path.cwd().name
                config_content = template_content.replace(
                    "{project_name}", actual_project_name
                )

                with open(config_file, "w", encoding="utf-8") as f:
                    f.write(config_content)
            except FileNotFoundError:
                # Fallback to minimal config if template not found
                config_data = {
                    "project": {"name": project_name or Path.cwd().name},
                    "model_profiles": {},
                }
                with open(config_file, "w", encoding="utf-8") as f:
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)

        # Copy example ensemble template to local ensembles directory (idempotent)
        local_ensemble_file = local_dir / "ensembles" / "example-local-ensemble.yaml"

        if not local_ensemble_file.exists():
            try:
                example_template_content = self._get_template_config_content(
                    "example-local-ensemble.yaml"
                )
                with open(local_ensemble_file, "w", encoding="utf-8") as f:
                    f.write(example_template_content)
            except FileNotFoundError:
                # If template not found in library, try local fallback
                template_ensemble_dir = self._get_template_ensembles_dir()
                example_template = template_ensemble_dir / "example-local-ensemble.yaml"
                if example_template.exists():
                    shutil.copy2(example_template, local_ensemble_file)

        # Core primitives now live in the installed package
        # (src/llm_orc/primitives/) and are resolved via ScriptResolver
        # priority 1.5. No longer copy from library to avoid shadowing.

        # Copy profile templates to local profiles directory (idempotent)
        self._copy_profile_templates(local_dir / "profiles")

        # Create .gitignore for credentials if they are stored locally (idempotent)
        gitignore_file = local_dir / ".gitignore"
        if not gitignore_file.exists():
            with open(gitignore_file, "w", encoding="utf-8") as f:
                f.write(
                    "# Local credentials (if any)\ncredentials.yaml\n.encryption_key\n"
                )

    def get_model_profiles(self) -> dict[str, dict[str, str]]:
        """Get merged model profiles from global and local configs.

        Results are cached and invalidated when config files change.
        """
        # Check if cache is still valid
        current_mtimes = self._get_profile_file_mtimes()
        if (
            self._profiles_cache is not None
            and current_mtimes == self._profiles_cache_mtimes
        ):
            return self._profiles_cache

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
        self._profiles_cache = merged_profiles
        self._profiles_cache_mtimes = current_mtimes
        return merged_profiles

    def _get_profile_file_mtimes(self) -> dict[str, float]:
        """Get modification times for profile config files."""
        mtimes: dict[str, float] = {}
        global_config_file = self._global_config_dir / "config.yaml"
        if global_config_file.exists():
            mtimes["global"] = global_config_file.stat().st_mtime
        if self._local_config_dir:
            local_config_file = self._local_config_dir / "config.yaml"
            if local_config_file.exists():
                mtimes["local"] = local_config_file.stat().st_mtime
        return mtimes

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

    def get_model_profile(self, profile_name: str) -> dict[str, Any] | None:
        """Get a specific model profile configuration.

        Args:
            profile_name: Name of the model profile to retrieve

        Returns:
            Model profile configuration dict or None if not found
        """
        profiles = self.get_model_profiles()
        return profiles.get(profile_name)

    def _copy_profile_templates(self, target_profiles_dir: Path) -> None:
        """Copy profile templates from llm-orchestra-library GitHub repository."""
        from llm_orc.cli_library.library import copy_profile_templates

        try:
            copy_profile_templates(target_profiles_dir)
        except (OSError, FileNotFoundError):
            # If profile copying fails, continue with init
            # This allows offline usage
            pass
