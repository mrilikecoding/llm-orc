"""Profile management handler for MCP server."""

from pathlib import Path
from typing import Any

import yaml

from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.mcp.project_context import ProjectContext
from llm_orc.services.handlers.scope import Scope, find_in_scope, parse_scope


class ProfileHandler:
    """Manages model profile CRUD operations."""

    def __init__(
        self,
        config_manager: ConfigurationManager,
    ) -> None:
        """Initialize with configuration manager."""
        self._config_manager = config_manager

    @property
    def config_manager(self) -> ConfigurationManager:
        """The manager this handler reads and writes through."""
        return self._config_manager

    def set_project_context(self, ctx: ProjectContext) -> None:
        """Update handler to use new project context."""
        self._config_manager = ctx.config_manager

    async def list_profiles(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """List model profiles."""
        provider_filter = arguments.get("provider")
        profiles: list[dict[str, Any]] = []

        profiles_dirs = self._config_manager.get_profiles_dirs()

        for dir_path in profiles_dirs:
            profiles_dir = Path(dir_path)
            if not profiles_dir.exists():
                continue

            for yaml_file in profiles_dir.glob("*.yaml"):
                try:
                    content = yaml_file.read_text()
                    data = yaml.safe_load(content) or {}
                    profile_provider = data.get("provider", "")

                    if provider_filter and profile_provider != provider_filter:
                        continue

                    profiles.append(
                        {
                            "name": data.get("name", yaml_file.stem),
                            "provider": profile_provider,
                            "model": data.get("model", ""),
                            "path": str(yaml_file),
                        }
                    )
                except Exception:
                    continue

        return {"profiles": profiles}

    def get_local_profiles_dir(self) -> Path:
        """The project's profiles directory for writing.

        Raises:
            ValueError: when there is no project tier. Writes never fall
                through to the library or packaged tiers; ``scope: global``
                is the writable home without a project (#196).
        """
        local = self._config_manager.local_config_dir
        if local is None:
            raise ValueError(
                "No project (.llm-orc) directory to write to; "
                "use scope 'global' for the global tier"
            )
        return local / "profiles"

    def _dir_for_scope(self, scope: Scope) -> Path | None:
        """Write directory for one scope; None when the project has none."""
        if scope == "global":
            return self._config_manager.global_config_dir / "profiles"
        local_config_dir = self._config_manager.local_config_dir
        if local_config_dir is None:
            return None
        return local_config_dir / "profiles"

    async def create_profile(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Create a new profile."""
        name = arguments.get("name")
        provider = arguments.get("provider")
        model = arguments.get("model")

        if not name:
            raise ValueError("name is required")
        if not provider:
            raise ValueError("provider is required")
        if not model:
            raise ValueError("model is required")

        scope = parse_scope(arguments)
        local_dir = self._dir_for_scope(scope)
        if local_dir is None:
            global_profiles = self._config_manager.global_config_dir / "profiles"
            raise ValueError(
                "No project directory (.llm-orc) here; pass scope: global to "
                f"write under {global_profiles}, or run llm-orc config init"
            )
        target_file = local_dir / f"{name}.yaml"
        if target_file.exists():
            raise ValueError(f"Profile '{name}' already exists")

        profile_data: dict[str, Any] = {
            "name": name,
            "provider": provider,
            "model": model,
        }
        if arguments.get("system_prompt"):
            profile_data["system_prompt"] = arguments["system_prompt"]
        if arguments.get("timeout_seconds"):
            profile_data["timeout_seconds"] = arguments["timeout_seconds"]
        if arguments.get("temperature") is not None:
            profile_data["temperature"] = arguments["temperature"]
        if arguments.get("max_tokens") is not None:
            profile_data["max_tokens"] = arguments["max_tokens"]

        local_dir.mkdir(parents=True, exist_ok=True)
        yaml_content = yaml.safe_dump(profile_data, default_flow_style=False)
        target_file.write_text(yaml_content)

        return {"created": True, "path": str(target_file), "scope": scope}

    async def update_profile(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Update an existing profile."""
        name = arguments.get("name")
        changes = arguments.get("changes", {})

        if not name:
            raise ValueError("name is required")

        profile_file = self._find_profile_file(name, parse_scope(arguments))

        content = profile_file.read_text()
        data = yaml.safe_load(content) or {}
        data.update(changes)

        yaml_content = yaml.safe_dump(data, default_flow_style=False)
        profile_file.write_text(yaml_content)

        return {"updated": True, "path": str(profile_file)}

    async def delete_profile(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Delete a profile."""
        name = arguments.get("name")
        confirm = arguments.get("confirm", False)

        if not name:
            raise ValueError("name is required")
        if not confirm:
            raise ValueError("Confirmation required to delete profile")

        profile_file = self._find_profile_file(name, parse_scope(arguments))
        profile_file.unlink()

        return {"deleted": True, "name": name}

    def _find_profile_file(self, name: str, scope: Scope) -> Path:
        """Find the profile YAML in one scope; name the tier if elsewhere."""
        return find_in_scope(
            name=name,
            filename=f"{name}.yaml",
            scope=scope,
            scope_dir=self._dir_for_scope(scope),
            search_dirs=[Path(d) for d in self._config_manager.get_profiles_dirs()],
            classify=self._config_manager.classify_tier,
            label="Profile",
        )

    def get_profiles_from_dir(self, directory: Path) -> dict[str, dict[str, Any]]:
        """Get profiles available in a specific directory.

        Scans only the given directory for YAML profile files and returns
        their parsed data. Does not include config.yaml model_profiles.

        Args:
            directory: Directory to scan for profile YAML files.

        Returns:
            Dict of profile_name -> profile_data for all profiles found.
        """
        profiles: dict[str, dict[str, Any]] = {}
        if not directory.exists():
            return profiles
        for yaml_file in directory.glob("*.yaml"):
            self._load_profiles_from_file(yaml_file, profiles)
        return profiles

    def get_runtime_profiles(self) -> dict[str, dict[str, str]]:
        """The profiles a run resolves: every runtime tier (packaged <
        global < local), the library excluded. The run, the router
        preset and preflight all read this one merge."""
        return self._config_manager.get_model_profiles()

    def get_all_profiles(self) -> dict[str, dict[str, Any]]:
        """Get all profiles as a dict keyed by name.

        Merges profiles from config.yaml model_profiles sections (global and
        local) with profiles from profiles/ directories. Directory-based
        profiles take precedence over config.yaml-based profiles.
        """
        profiles: dict[str, dict[str, Any]] = dict(
            self._config_manager.get_model_profiles()
        )

        for dir_path in self._config_manager.get_profiles_dirs():
            profile_dir = Path(dir_path)
            if not profile_dir.exists():
                continue

            for yaml_file in profile_dir.glob("*.yaml"):
                self._load_profiles_from_file(yaml_file, profiles)

        return profiles

    def _load_profiles_from_file(
        self,
        yaml_file: Path,
        profiles: dict[str, dict[str, Any]],
    ) -> None:
        """Load profiles from a YAML file into the profiles dict."""
        try:
            with open(yaml_file) as f:
                data = yaml.safe_load(f) or {}

            self._parse_profile_data(data, profiles)
        except Exception:
            pass

    def _parse_profile_data(
        self,
        data: dict[str, Any],
        profiles: dict[str, dict[str, Any]],
    ) -> None:
        """Parse profile data from various YAML formats."""
        if "model_profiles" in data:
            self._parse_dict_format_profiles(data["model_profiles"], profiles)
        elif "profiles" in data:
            self._parse_list_format_profiles(data["profiles"], profiles)
        elif "name" in data:
            profiles[data["name"]] = data

    def _parse_dict_format_profiles(
        self,
        model_profiles: dict[str, Any],
        profiles: dict[str, dict[str, Any]],
    ) -> None:
        """Parse dict format: model_profiles: {name: {config...}}."""
        for name, config in model_profiles.items():
            if isinstance(config, dict):
                config["name"] = name
                profiles[name] = config

    def _parse_list_format_profiles(
        self,
        profile_list: list[dict[str, Any]],
        profiles: dict[str, dict[str, Any]],
    ) -> None:
        """Parse list format: profiles: [{name: ..., ...}]."""
        for p in profile_list:
            name = p.get("name", "")
            if name:
                profiles[name] = p
