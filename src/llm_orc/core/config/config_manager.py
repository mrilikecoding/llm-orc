"""Configuration management system for llm-orc."""

import copy
import logging
import os
import shutil
from pathlib import Path
from typing import Any

import yaml

from llm_orc.core.config.packaged import (
    SERVING_MARKER,
    has_serving_ensemble,
    packaged_serving_project_dir,
)
from llm_orc.core.config.template_provider import TemplateProvider

logger = logging.getLogger(__name__)


def resolve_global_config_dir() -> Path:
    """The global configuration directory following XDG spec.

    Module-level so callers that need this one path (e.g. a local-only
    secret store) don't have to construct a full ConfigurationManager
    just to read it. ConfigurationManager._get_global_config_dir
    delegates here.
    """
    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config_home:
        return Path(xdg_config_home) / "llm-orc"

    return Path.home() / ".config" / "llm-orc"


class ConfigurationManager:
    """Manages configuration directories and file locations."""

    def __init__(
        self,
        project_dir: Path | None = None,
        *,
        provision: bool = True,
        template_provider: TemplateProvider | None = None,
    ) -> None:
        """Initialize configuration manager.

        Args:
            project_dir: Optional project directory. If provided, uses
                project_dir/.llm-orc as local config instead of discovering
                from cwd.
            provision: If True (default), call provision() to create global
                config directories and copy default templates. Pass False for
                lightweight read-only usage (e.g. tab completion, perf config).
            template_provider: Optional provider for template content. When
                None, template operations are skipped silently (no-op). Pass
                a LibraryTemplateProvider for full CLI functionality.
        """
        self._template_provider = template_provider
        self._global_config_dir = self._get_global_config_dir()
        if project_dir is not None:
            # Use explicit project directory
            llm_orc_dir = project_dir / ".llm-orc"
            self._local_config_dir = llm_orc_dir if llm_orc_dir.exists() else None
        else:
            # Discover from cwd
            self._local_config_dir = self._discover_local_config()

        # The read-only tier shipped in the wheel (#196), lowest precedence.
        # In a checkout it is the local dot-dir itself; the dir lists and
        # the profile merge each skip it then, so nothing is read twice.
        self._packaged_serving_dir = packaged_serving_project_dir()

        # The one-run layer (Arc 4), set only on a view; see with_run_layer.
        self._run_layer_dir: Path | None = None

        # Profile cache
        self._profiles_cache: dict[str, dict[str, str]] | None = None
        self._profiles_cache_mtimes: dict[str, float] = {}

        if provision:
            self.provision()

    @property
    def run_layer_dir(self) -> Path | None:
        """The run layer directory when this is a view, else ``None``."""
        return self._run_layer_dir

    def with_run_layer(self, run_dir: Path) -> "ConfigurationManager":
        """A copy whose tier lists put ``run_dir`` highest (Arc 4).

        The layer is shaped like every other tier: ``ensembles/``,
        ``profiles/``, scripts at their keys. The copy gets its own,
        empty profile cache so two views never see each other's profiles
        and neither touches this manager's.
        """
        view = copy.copy(self)
        view._run_layer_dir = run_dir
        view._profiles_cache = None
        view._profiles_cache_mtimes = {}
        return view

    def provision(self) -> None:
        """Create global config directories and copy default templates."""
        self._global_config_dir.mkdir(parents=True, exist_ok=True)
        (self._global_config_dir / "profiles").mkdir(exist_ok=True)
        self._setup_default_config()
        self._setup_default_ensembles()
        self._copy_profile_templates(self._global_config_dir / "profiles")

    def _get_global_config_dir(self) -> Path:
        """Get the global configuration directory following XDG spec."""
        return resolve_global_config_dir()

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
        """Get template config content via the injected TemplateProvider."""
        if self._template_provider is None:
            raise FileNotFoundError(
                f"Template not found: {filename} (no template provider configured)"
            )

        try:
            return self._template_provider.get_template_content(filename)
        except FileNotFoundError:
            # Fallback to local template if provider cannot locate the file
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

    @property
    def packaged_serving_dir(self) -> Path | None:
        """The packaged serving project (read-only), if this install has one."""
        return self._packaged_serving_dir

    @property
    def library_dir(self) -> Path:
        """The library base directory, whether or not it exists.

        ``LLM_ORC_LIBRARY_PATH`` wins; else the submodule location under
        the checkout root, which is the local dot-dir's parent when there
        is a project and the cwd otherwise. Derived from the project so a
        manager built with an explicit ``project_dir`` does not read a
        library off an unrelated cwd (S1 finding).
        """
        library_path_env = os.environ.get("LLM_ORC_LIBRARY_PATH")
        if library_path_env:
            return Path(library_path_env)
        root = (
            self._local_config_dir.parent
            if self._local_config_dir is not None
            else Path.cwd()
        )
        return root / "llm-orchestra-library"

    def serving_root(self) -> Path:
        """The dot-dir that carries the serving ensemble (#196).

        The project's ``.llm-orc`` when it has
        ``ensembles/agentic-serving/serving.yaml``, else the packaged
        serving project. ``ServingEnsembleCaller`` reads the ensemble,
        the serve-owned scripts and the ``serving:`` config keys from
        this one directory; other ensembles, profiles and scripts still
        merge every tier.
        """
        for candidate in (self._local_config_dir, self._packaged_serving_dir):
            if candidate is not None and has_serving_ensemble(candidate):
                return candidate
        raise FileNotFoundError(
            f"no serving ensemble ({SERVING_MARKER}) under the project "
            f"({self._local_config_dir}) or the packaged serving project "
            f"({self._packaged_serving_dir})"
        )

    def _is_packaged_distinct(self) -> bool:
        """True when the packaged tier exists and is not the local dot-dir."""
        packaged = self._packaged_serving_dir
        if packaged is None:
            return False
        local = self._local_config_dir
        return local is None or packaged.resolve() != local.resolve()

    def classify_tier(self, path: Path) -> str:
        """Classify a path as local, library, global, packaged, or unknown.

        Args:
            path: Path to classify (file or directory).

        Returns:
            One of ``"local"``, ``"library"``, ``"global"``, ``"packaged"``,
            or ``"unknown"``. A checkout's dot-dir is ``"local"`` even
            though it is also the packaged tier.
        """
        if self._local_config_dir and path.is_relative_to(self._local_config_dir):
            return "local"
        if path.is_relative_to(self.library_dir):
            return "library"
        if path.is_relative_to(self._global_config_dir):
            return "global"
        if self._packaged_serving_dir is not None and path.is_relative_to(
            self._packaged_serving_dir
        ):
            return "packaged"
        return "unknown"

    def get_ensembles_dirs(self) -> list[Path]:
        """Ensemble directories in priority order.

        local → library → global → packaged. Each entry appears only when
        it exists; the packaged entry is skipped in a checkout, where it
        is the local dot-dir.
        """
        return self._tier_dirs("ensembles")

    def get_profiles_dirs(self) -> list[Path]:
        """Profile directories in priority order (same tiers as ensembles)."""
        return self._tier_dirs("profiles")

    def _tier_dirs(self, subdir: str) -> list[Path]:
        candidates: list[Path | None] = [
            self._run_layer_dir,
            self._local_config_dir,
            self.library_dir,
            self._global_config_dir,
            self._packaged_serving_dir if self._is_packaged_distinct() else None,
        ]
        return [
            base / subdir
            for base in candidates
            if base is not None and (base / subdir).exists()
        ]

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

    def _load_packaged_config(self) -> dict[str, Any]:
        """The packaged serving project's ``config.yaml``, or ``{}``.

        Empty in a checkout (the file is then the local config and is
        merged as such) and when this install has no packaged tier.
        """
        if not self._is_packaged_distinct():
            return {}
        assert self._packaged_serving_dir is not None
        config_file = self._packaged_serving_dir / "config.yaml"
        if not config_file.exists():
            return {}
        with open(config_file) as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}

    def load_performance_config(self) -> dict[str, Any]:
        """Load performance configuration with sensible defaults.

        Defaults, then the packaged, global and local ``config.yaml``
        ``performance:`` sections, each overlaying the last.
        """
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

        packaged_performance = self._load_packaged_config().get("performance", {})

        # Try to load from global config
        global_config = self._load_global_config()
        global_performance = global_config.get("performance", {})

        # Try to load from local config
        local_config = self.load_project_config()
        local_performance = local_config.get("performance", {})

        # Merge configurations: defaults -> packaged -> global -> local
        merged_config = defaults.copy()
        self._deep_merge_dict(merged_config, packaged_performance)
        self._deep_merge_dict(merged_config, global_performance)
        self._deep_merge_dict(merged_config, local_performance)

        return merged_config

    def load_agentic_serving_config(self) -> dict[str, Any]:
        """Load agentic-serving configuration with sensible defaults.

        Defaults reflect stateless-first operation (AS-8) and llm-orc's
        core value proposition that orchestration with local-hardware
        compute trades tokens-for-quality against a single frontier-API
        call. Plexus is disabled, autonomy is ``operator-as-tool-user``,
        and Budget sizes are loose: the token ceiling is a pathology
        circuit breaker for the local-orchestration-heavy case, not a
        cost ceiling for frontier-API pricing. Frontier-mix deployments
        tighten via ``config.yaml``.

        The packaged ``config.yaml`` overlays defaults; global overlays
        packaged; local project ``config.yaml`` overlays global.
        """
        # Post-collapse only the orchestrator key is consumed (the /v1/models
        # allowlist default). Shipping defaults for budget/autonomy/plexus
        # keys nothing enforces invited operators to set ceilings that were
        # silently ignored (issue #95).
        defaults: dict[str, Any] = {
            "orchestrator": {"model_profile": "default"},
        }

        packaged_section = self._load_packaged_config().get("agentic_serving") or {}
        if not isinstance(packaged_section, dict):
            packaged_section = {}

        global_config = self._load_global_config()
        global_section = global_config.get("agentic_serving") or {}
        if not isinstance(global_section, dict):
            global_section = {}

        local_config = self.load_project_config()
        local_section = local_config.get("agentic_serving") or {}
        if not isinstance(local_section, dict):
            local_section = {}

        self._deep_merge_dict(defaults, packaged_section)
        self._deep_merge_dict(defaults, global_section)
        self._deep_merge_dict(defaults, local_section)

        return defaults

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

    def _profile_tiers(self) -> list[Path]:
        """Config dirs whose profiles resolve at runtime, lowest precedence first.

        packaged -> global -> local -> run layer. The library tier is listed by
        ``get_profiles_dirs`` but never resolved here, as before this
        tier loop existed: a submodule profile must not shadow a global
        one by name. In a checkout the packaged dir is the local dot-dir
        and is skipped at the bottom so it merges once, at the top.
        """
        tiers: list[Path] = []
        if self._is_packaged_distinct():
            assert self._packaged_serving_dir is not None
            tiers.append(self._packaged_serving_dir)
        tiers.append(self._global_config_dir)
        if self._local_config_dir is not None:
            tiers.append(self._local_config_dir)
        if self._run_layer_dir is not None:
            tiers.append(self._run_layer_dir)
        return tiers

    def get_model_profiles(self) -> dict[str, dict[str, str]]:
        """Get merged model profiles from every runtime tier.

        Within a tier ``config.yaml: model_profiles`` loads first and
        ``profiles/*.yaml`` after it (``*.local.yaml`` last), so a file
        beats the config entry of the same name and a later tier beats an
        earlier one. Results are cached and invalidated when any source
        file's mtime changes.
        """
        current_mtimes = self._get_profile_file_mtimes()
        if (
            self._profiles_cache is not None
            and current_mtimes == self._profiles_cache_mtimes
        ):
            return self._profiles_cache

        merged: dict[str, dict[str, str]] = {}
        for tier in self._profile_tiers():
            config_file = tier / "config.yaml"
            if config_file.exists():
                with open(config_file) as f:
                    data = yaml.safe_load(f) or {}
                merged.update(data.get("model_profiles") or {})
            self._load_profile_yaml_files(tier / "profiles", merged)

        self._profiles_cache = merged
        self._profiles_cache_mtimes = current_mtimes
        return merged

    @staticmethod
    def _load_profile_yaml_files(
        profiles_dir: Path,
        target: dict[str, dict[str, str]],
    ) -> None:
        """Load individual profile YAML files from a directory.

        ``*.local.yaml`` files load last (each group sorted), so an
        operator-private override of a checked-in profile name wins
        deterministically — the seam for backing a tier with a private
        provider without committing provider-specific config.
        """
        if not profiles_dir.exists():
            return
        all_files = sorted(profiles_dir.glob("*.yaml"))
        base = [f for f in all_files if not f.name.endswith(".local.yaml")]
        overrides = [f for f in all_files if f.name.endswith(".local.yaml")]
        for yaml_file in base + overrides:
            try:
                with open(yaml_file) as f:
                    data = yaml.safe_load(f) or {}
                if isinstance(data, dict) and "name" in data:
                    target[data["name"]] = data
            except Exception:
                logger.warning("Skipping invalid profile %s", yaml_file)

    def _get_profile_file_mtimes(self) -> dict[str, float]:
        """Modification times of every runtime profile source, keyed by path."""
        mtimes: dict[str, float] = {}
        for tier in self._profile_tiers():
            config_file = tier / "config.yaml"
            if config_file.exists():
                mtimes[str(config_file)] = config_file.stat().st_mtime
            profiles_dir = tier / "profiles"
            if profiles_dir.exists():
                for f in profiles_dir.glob("*.yaml"):
                    mtimes[str(f)] = f.stat().st_mtime
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
        """Copy profile templates via the injected TemplateProvider."""
        if self._template_provider is None:
            return

        try:
            self._template_provider.copy_profile_templates(target_profiles_dir)
        except (OSError, FileNotFoundError):
            # If profile copying fails, continue with init
            # This allows offline usage
            pass
