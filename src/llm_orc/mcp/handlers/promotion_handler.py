"""Promotion handler for MCP server.

Manages ensemble promotion between tiers (local → global → library),
dependency inspection, and demotion.
"""

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.core.config.ensemble_config import EnsembleConfig
from llm_orc.mcp.handlers.library_handler import LibraryHandler
from llm_orc.mcp.handlers.profile_handler import ProfileHandler
from llm_orc.mcp.handlers.provider_handler import ProviderHandler
from llm_orc.mcp.utils import get_agent_attr as _get_agent_attr


class PromotionHandler:
    """Manages ensemble promotion between tiers and dependency inspection."""

    def __init__(
        self,
        config_manager: ConfigurationManager,
        profile_handler: ProfileHandler,
        library_handler: LibraryHandler,
        provider_handler: ProviderHandler,
        find_ensemble: Callable[[str], EnsembleConfig | None],
    ) -> None:
        """Initialize with dependencies.

        Args:
            config_manager: Configuration manager instance.
            profile_handler: Profile handler for profile operations.
            library_handler: Library handler for library directory resolution.
            provider_handler: Provider handler for runnability checks.
            find_ensemble: Callback to find ensemble by name.
        """
        self._config_manager = config_manager
        self._profile_handler = profile_handler
        self._library_handler = library_handler
        self._provider_handler = provider_handler
        self._find_ensemble = find_ensemble

    # =========================================================================
    # Public tool methods
    # =========================================================================

    async def promote_ensemble(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Promote an ensemble from one tier to another, including profiles.

        Args:
            arguments: Tool arguments including ensemble_name, destination,
                include_profiles, dry_run, overwrite.

        Returns:
            Promotion result or dry-run preview.
        """
        ensemble_name = arguments.get("ensemble_name")
        destination = arguments.get("destination")
        include_profiles = arguments.get("include_profiles", True)
        dry_run = arguments.get("dry_run", True)
        overwrite = arguments.get("overwrite", False)

        if not ensemble_name:
            raise ValueError("ensemble_name is required")
        if destination not in ("global", "library"):
            raise ValueError("destination must be 'global' or 'library'")

        found = self._find_ensemble_file(ensemble_name)
        if not found:
            raise ValueError(f"Ensemble not found: {ensemble_name}")
        source_path, source_tier = found

        dest_ensembles, dest_profiles = self._get_tier_dirs(destination)
        dest_file = dest_ensembles / f"{ensemble_name}.yaml"

        if dest_file.exists() and not overwrite:
            raise ValueError(
                f"Ensemble '{ensemble_name}' already exists at {destination} tier. "
                "Use overwrite=true to replace."
            )

        # Resolve profile dependencies
        profiles_to_copy, profiles_already_present, profiles_missing = (
            self._resolve_profiles_for_promotion(
                ensemble_name, destination, include_profiles
            )
        )

        if profiles_missing:
            raise ValueError(
                f"Broken profile references: {', '.join(profiles_missing)}. "
                "These profiles do not exist in any tier."
            )

        if dry_run:
            return {
                "status": "dry_run",
                "ensemble": ensemble_name,
                "source_tier": source_tier,
                "destination": destination,
                "would_copy": {
                    "ensemble": {
                        "source": str(source_path),
                        "destination": str(dest_file),
                    },
                    "profiles": [p["name"] for p in profiles_to_copy],
                },
                "profiles_already_present": profiles_already_present,
                "overwrite": overwrite and dest_file.exists(),
            }

        # Execute the copy
        dest_ensembles.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest_file)

        copied_profiles: list[str] = []
        if include_profiles:
            copied_profiles = self._write_profiles(
                profiles_to_copy, dest_profiles, overwrite
            )

        return {
            "status": "promoted",
            "ensemble": ensemble_name,
            "source_tier": source_tier,
            "destination": destination,
            "ensemble_path": str(dest_file),
            "profiles_copied": copied_profiles,
            "profiles_already_present": profiles_already_present,
        }

    async def list_dependencies(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """List all dependencies for an ensemble.

        Args:
            arguments: Tool arguments including ensemble_name.

        Returns:
            Dependency report with agents, profiles, models, and providers.
        """
        ensemble_name = arguments.get("ensemble_name")
        if not ensemble_name:
            raise ValueError("ensemble_name is required")

        config = self._find_ensemble(ensemble_name)
        if not config:
            raise ValueError(f"Ensemble not found: {ensemble_name}")

        found = self._find_ensemble_file(ensemble_name)
        source_tier = found[1] if found else "unknown"

        all_profiles = self._profile_handler.get_all_profiles()

        provider_status = await self._provider_handler.get_provider_status({})
        providers = provider_status.get("providers", {})
        ollama_models = providers.get("ollama", {}).get("models", [])

        agents: list[dict[str, Any]] = []
        profiles_needed: set[str] = set()
        models_needed: set[str] = set()
        providers_needed: set[str] = set()

        for agent in config.agents:
            agent_name = _get_agent_attr(agent, "name", "unknown")
            script_path = _get_agent_attr(agent, "script", "")

            if script_path:
                agents.append({
                    "name": agent_name,
                    "type": "script",
                    "script": script_path,
                })
                continue

            profile_name = _get_agent_attr(agent, "model_profile", "")
            agent_info: dict[str, Any] = {
                "name": agent_name,
                "model_profile": profile_name,
                "profile_found": profile_name in all_profiles,
            }

            if profile_name in all_profiles:
                profile = all_profiles[profile_name]
                provider = profile.get("provider", "")
                model = profile.get("model", "")

                agent_info["profile_tier"] = self._get_profile_tier(profile_name)
                agent_info["provider"] = provider
                agent_info["model"] = model
                agent_info["model_available"] = self._check_model_available(
                    provider, model, providers, ollama_models
                )

                profiles_needed.add(profile_name)
                if model:
                    models_needed.add(model)
                if provider:
                    providers_needed.add(provider)
            else:
                agent_info["profile_tier"] = None
                agent_info["provider"] = None
                agent_info["model"] = None
                agent_info["model_available"] = False
                if profile_name:
                    profiles_needed.add(profile_name)

            agents.append(agent_info)

        return {
            "ensemble": ensemble_name,
            "source_tier": source_tier,
            "agents": agents,
            "profiles_needed": sorted(profiles_needed),
            "models_needed": sorted(models_needed),
            "providers_needed": sorted(providers_needed),
        }

    async def check_promotion_readiness(
        self, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Check if an ensemble is ready for promotion to a target tier.

        Args:
            arguments: Tool arguments including ensemble_name, destination.

        Returns:
            Readiness assessment with issues and required actions.
        """
        ensemble_name = arguments.get("ensemble_name")
        destination = arguments.get("destination")

        if not ensemble_name:
            raise ValueError("ensemble_name is required")
        if destination not in ("global", "library"):
            raise ValueError("destination must be 'global' or 'library'")

        config = self._find_ensemble(ensemble_name)
        if not config:
            raise ValueError(f"Ensemble not found: {ensemble_name}")

        dest_ensembles, _ = self._get_tier_dirs(destination)
        already_exists = (dest_ensembles / f"{ensemble_name}.yaml").exists()

        profiles_needed = self._get_ensemble_profiles(ensemble_name)

        provider_status = await self._provider_handler.get_provider_status({})
        providers = provider_status.get("providers", {})
        ollama_models = providers.get("ollama", {}).get("models", [])

        issues, profiles_to_copy, profiles_already_present = (
            self._assess_profiles_readiness(
                profiles_needed, destination, providers, ollama_models
            )
        )

        # Blocking issues prevent promotion
        blocking_types = {"broken_reference", "provider_unavailable"}
        ready = not any(i["type"] in blocking_types for i in issues)

        return {
            "ensemble": ensemble_name,
            "destination": destination,
            "ready": ready,
            "issues": issues,
            "already_exists": already_exists,
            "profiles_to_copy": profiles_to_copy,
            "profiles_already_present": profiles_already_present,
        }

    async def demote_ensemble(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Remove an ensemble from a higher tier.

        Args:
            arguments: Tool arguments including ensemble_name, tier,
                remove_orphaned_profiles, confirm.

        Returns:
            Demotion result or preview.
        """
        ensemble_name = arguments.get("ensemble_name")
        tier = arguments.get("tier")
        remove_orphaned_profiles = arguments.get(
            "remove_orphaned_profiles", False
        )
        confirm = arguments.get("confirm", False)

        if not ensemble_name:
            raise ValueError("ensemble_name is required")
        if tier not in ("global", "library"):
            raise ValueError("tier must be 'global' or 'library'")

        ensembles_dir, profiles_dir = self._get_tier_dirs(tier)
        ensemble_file = ensembles_dir / f"{ensemble_name}.yaml"

        if not ensemble_file.exists():
            raise ValueError(
                f"Ensemble '{ensemble_name}' not found at {tier} tier"
            )

        orphaned: list[str] = []
        if remove_orphaned_profiles:
            orphaned = self._find_orphaned_profiles(
                ensemble_name, ensembles_dir, profiles_dir
            )

        if not confirm:
            result: dict[str, Any] = {
                "status": "preview",
                "ensemble": ensemble_name,
                "tier": tier,
                "would_remove": {
                    "ensemble": str(ensemble_file),
                },
            }
            if remove_orphaned_profiles:
                result["would_remove"]["orphaned_profiles"] = orphaned
            return result

        ensemble_file.unlink()

        removed_profiles: list[str] = []
        if remove_orphaned_profiles:
            for profile_name in orphaned:
                profile_file = profiles_dir / f"{profile_name}.yaml"
                if profile_file.exists():
                    profile_file.unlink()
                    removed_profiles.append(profile_name)

        return {
            "status": "demoted",
            "ensemble": ensemble_name,
            "tier": tier,
            "removed_profiles": removed_profiles,
        }

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _get_tier_dirs(self, tier: str) -> tuple[Path, Path]:
        """Get ensembles and profiles directories for a tier.

        Args:
            tier: Target tier ('global' or 'library').

        Returns:
            Tuple of (ensembles_dir, profiles_dir).
        """
        if tier == "global":
            base = self._config_manager.global_config_dir
            return base / "ensembles", base / "profiles"
        elif tier == "library":
            base = self._library_handler.get_library_dir()
            return base / "ensembles", base / "profiles"
        raise ValueError(f"Invalid tier: {tier}")

    def _find_ensemble_file(
        self, ensemble_name: str
    ) -> tuple[Path, str] | None:
        """Find an ensemble file and identify its source tier.

        Args:
            ensemble_name: Name of the ensemble.

        Returns:
            Tuple of (file_path, tier_name) or None if not found.
        """
        library_dir = self._library_handler.get_library_dir()
        global_dir = self._config_manager.global_config_dir

        for dir_path in self._config_manager.get_ensembles_dirs():
            potential = Path(dir_path) / f"{ensemble_name}.yaml"
            if potential.exists():
                dir_str = str(dir_path)
                if ".llm-orc" in dir_str and "library" not in dir_str:
                    return potential, "local"
                elif dir_str.startswith(str(library_dir)):
                    return potential, "library"
                elif dir_str.startswith(str(global_dir)):
                    return potential, "global"
                return potential, "unknown"
        return None

    def _get_ensemble_profiles(self, ensemble_name: str) -> set[str]:
        """Extract profile names referenced by an ensemble.

        Args:
            ensemble_name: Name of the ensemble.

        Returns:
            Set of profile names.
        """
        config = self._find_ensemble(ensemble_name)
        if not config:
            raise ValueError(f"Ensemble not found: {ensemble_name}")

        profiles: set[str] = set()
        for agent in config.agents:
            profile_name = _get_agent_attr(agent, "model_profile", "")
            if profile_name:
                profiles.add(profile_name)
        return profiles

    def _resolve_profiles_for_promotion(
        self,
        ensemble_name: str,
        destination: str,
        include_profiles: bool,
    ) -> tuple[list[dict[str, Any]], list[str], list[str]]:
        """Resolve which profiles need copying for a promotion.

        Args:
            ensemble_name: Ensemble being promoted.
            destination: Target tier.
            include_profiles: Whether to include profile deps.

        Returns:
            Tuple of (to_copy, already_present, missing).
        """
        if not include_profiles:
            return [], [], []

        profiles_needed = self._get_ensemble_profiles(ensemble_name)
        dest_existing = self._profiles_at_tier(destination)
        all_profiles = self._profile_handler.get_all_profiles()

        to_copy: list[dict[str, Any]] = []
        already_present: list[str] = []
        missing: list[str] = []

        for profile_name in sorted(profiles_needed):
            if profile_name in dest_existing:
                already_present.append(profile_name)
            elif profile_name in all_profiles:
                to_copy.append({
                    "name": profile_name,
                    "data": all_profiles[profile_name],
                })
            else:
                missing.append(profile_name)

        return to_copy, already_present, missing

    def _write_profiles(
        self,
        profiles_to_copy: list[dict[str, Any]],
        dest_profiles: Path,
        overwrite: bool,
    ) -> list[str]:
        """Write profile YAML files to destination directory.

        Args:
            profiles_to_copy: Dicts with 'name' and 'data' keys.
            dest_profiles: Destination profiles directory.
            overwrite: Whether to overwrite existing files.

        Returns:
            List of profile names that were written.
        """
        dest_profiles.mkdir(parents=True, exist_ok=True)
        copied: list[str] = []
        for profile_info in profiles_to_copy:
            profile_name = profile_info["name"]
            profile_file = dest_profiles / f"{profile_name}.yaml"
            if not profile_file.exists() or overwrite:
                profile_data = dict(profile_info["data"])
                profile_data["name"] = profile_name
                content = yaml.safe_dump(
                    profile_data, default_flow_style=False
                )
                profile_file.write_text(content)
                copied.append(profile_name)
        return copied

    def _assess_profiles_readiness(
        self,
        profiles_needed: set[str],
        destination: str,
        providers: dict[str, Any],
        ollama_models: list[str],
    ) -> tuple[list[dict[str, Any]], list[str], list[str]]:
        """Assess profile dependencies for promotion readiness.

        Args:
            profiles_needed: Profile names required by ensemble.
            destination: Target tier.
            providers: Provider status dict.
            ollama_models: Available Ollama model names.

        Returns:
            Tuple of (issues, profiles_to_copy, already_present).
        """
        dest_existing = self._profiles_at_tier(destination)
        all_profiles = self._profile_handler.get_all_profiles()

        issues: list[dict[str, Any]] = []
        to_copy: list[str] = []
        already_present: list[str] = []

        for profile_name in sorted(profiles_needed):
            if profile_name in dest_existing:
                already_present.append(profile_name)
            elif profile_name in all_profiles:
                to_copy.append(profile_name)
                issues.append({
                    "type": "missing_profile",
                    "detail": (
                        f"Profile '{profile_name}' not "
                        f"found at {destination} tier"
                    ),
                    "resolution": (
                        "Will be copied during promotion "
                        "if include_profiles=true"
                    ),
                })
            else:
                issues.append({
                    "type": "broken_reference",
                    "detail": (
                        f"Profile '{profile_name}' not "
                        "found in any tier"
                    ),
                    "resolution": (
                        "Create the profile before promoting"
                    ),
                })

            if profile_name in all_profiles:
                profile = all_profiles[profile_name]
                provider = profile.get("provider", "")
                model = profile.get("model", "")
                p_info = providers.get(provider, {})

                if not p_info.get("available", False):
                    issues.append({
                        "type": "provider_unavailable",
                        "detail": (
                            f"Provider '{provider}' for "
                            f"profile '{profile_name}' "
                            "is not available"
                        ),
                        "resolution": (
                            f"Ensure {provider} is running"
                        ),
                    })
                elif provider == "ollama":
                    available = self._check_model_available(
                        provider, model,
                        providers, ollama_models,
                    )
                    if not available:
                        issues.append({
                            "type": "model_unavailable",
                            "detail": (
                                f"Model '{model}' requires "
                                "Ollama but is not installed"
                            ),
                            "resolution": (
                                "Install with: "
                                f"ollama pull {model}"
                            ),
                        })

        return issues, to_copy, already_present

    def _profiles_at_tier(self, tier: str) -> dict[str, dict[str, Any]]:
        """Get profiles available at a specific tier.

        Args:
            tier: Target tier ('global' or 'library').

        Returns:
            Dict of profile_name -> profile_data.
        """
        _, profiles_dir = self._get_tier_dirs(tier)
        profiles: dict[str, dict[str, Any]] = {}

        if not profiles_dir.exists():
            return profiles

        for yaml_file in profiles_dir.glob("*.yaml"):
            try:
                data = yaml.safe_load(yaml_file.read_text()) or {}
                self._parse_profiles_from_file(data, profiles)
            except Exception:
                continue

        return profiles

    def _parse_profiles_from_file(
        self,
        data: dict[str, Any],
        profiles: dict[str, dict[str, Any]],
    ) -> None:
        """Parse profile entries from a YAML file's data.

        Handles three formats: model_profiles dict, profiles list,
        and single-profile with top-level name key.

        Args:
            data: Parsed YAML data from a profile file.
            profiles: Dict to populate with name -> profile_data.
        """
        if "model_profiles" in data:
            for name, config in data["model_profiles"].items():
                if isinstance(config, dict):
                    config["name"] = name
                    profiles[name] = config
        elif "profiles" in data:
            for p in data["profiles"]:
                name = p.get("name", "")
                if name:
                    profiles[name] = p
        elif "name" in data:
            profiles[data["name"]] = data

    def _get_profile_tier(self, profile_name: str) -> str:
        """Determine which tier a profile lives in.

        Args:
            profile_name: Name of the profile.

        Returns:
            Tier name ('local', 'global', 'library', or 'unknown').
        """
        library_dir = self._library_handler.get_library_dir()
        global_dir = self._config_manager.global_config_dir

        for dir_path in self._config_manager.get_profiles_dirs():
            path = Path(dir_path)
            if not path.exists():
                continue
            # Check direct file match
            if (path / f"{profile_name}.yaml").exists():
                return self._classify_dir_tier(
                    str(dir_path), library_dir, global_dir
                )
            # Check multi-profile files
            for yaml_file in path.glob("*.yaml"):
                try:
                    data = yaml.safe_load(yaml_file.read_text()) or {}
                    if "model_profiles" in data and profile_name in data[
                        "model_profiles"
                    ]:
                        return self._classify_dir_tier(
                            str(dir_path), library_dir, global_dir
                        )
                    if data.get("name") == profile_name:
                        return self._classify_dir_tier(
                            str(dir_path), library_dir, global_dir
                        )
                except Exception:
                    continue
        return "unknown"

    def _classify_dir_tier(
        self, dir_str: str, library_dir: Path, global_dir: Path
    ) -> str:
        """Classify a directory path as local, global, or library.

        Args:
            dir_str: Directory path string.
            library_dir: Library base directory.
            global_dir: Global config directory.

        Returns:
            Tier name.
        """
        if ".llm-orc" in dir_str and "library" not in dir_str:
            return "local"
        if dir_str.startswith(str(library_dir)):
            return "library"
        if dir_str.startswith(str(global_dir)):
            return "global"
        return "unknown"

    def _check_model_available(
        self,
        provider: str,
        model: str,
        providers: dict[str, Any],
        ollama_models: list[str],
    ) -> bool:
        """Check if a model is available from its provider.

        Args:
            provider: Provider name.
            model: Model identifier.
            providers: Provider status dict.
            ollama_models: List of available Ollama models.

        Returns:
            True if model is available.
        """
        if provider == "ollama":
            model_base = model.split(":")[0] if ":" in model else model
            return any(
                m == model or m.startswith(f"{model_base}:")
                for m in ollama_models
            )
        provider_info = providers.get(provider, {})
        return bool(provider_info.get("available", False))

    def _find_orphaned_profiles(
        self,
        ensemble_name: str,
        ensembles_dir: Path,
        profiles_dir: Path,
    ) -> list[str]:
        """Find profiles that would be orphaned by removing an ensemble.

        Scans remaining ensembles at the tier to find profiles that are
        only referenced by the ensemble being removed.

        Args:
            ensemble_name: Ensemble being removed.
            ensembles_dir: Ensembles directory at the tier.
            profiles_dir: Profiles directory at the tier.

        Returns:
            Sorted list of orphaned profile names.
        """
        target, used_by_others = self._scan_ensemble_profiles(
            ensemble_name, ensembles_dir
        )
        orphan_candidates = target - used_by_others
        existing = self._filter_existing_at_tier(
            orphan_candidates, profiles_dir
        )
        return sorted(orphan_candidates & existing)

    def _scan_ensemble_profiles(
        self,
        ensemble_name: str,
        ensembles_dir: Path,
    ) -> tuple[set[str], set[str]]:
        """Scan ensemble files for profile references.

        Args:
            ensemble_name: Ensemble being removed.
            ensembles_dir: Directory containing ensemble YAML files.

        Returns:
            Tuple of (target_profiles, used_by_others).
        """
        target: set[str] = set()
        others: set[str] = set()
        if not ensembles_dir.exists():
            return target, others
        for yaml_file in ensembles_dir.glob("*.yaml"):
            try:
                data = yaml.safe_load(yaml_file.read_text()) or {}
                file_profiles: set[str] = set()
                for agent in data.get("agents", []):
                    profile = agent.get("model_profile", "")
                    if profile:
                        file_profiles.add(profile)
                if yaml_file.stem == ensemble_name:
                    target = file_profiles
                else:
                    others |= file_profiles
            except Exception:
                continue
        return target, others

    def _filter_existing_at_tier(
        self,
        candidates: set[str],
        profiles_dir: Path,
    ) -> set[str]:
        """Filter profile candidates to those existing at the tier.

        Args:
            candidates: Profile names to check.
            profiles_dir: Profiles directory at the tier.

        Returns:
            Set of candidate names that have actual profile files.
        """
        existing: set[str] = set()
        if not profiles_dir.exists():
            return existing
        for yaml_file in profiles_dir.glob("*.yaml"):
            try:
                data = yaml.safe_load(yaml_file.read_text()) or {}
                if "name" in data and data["name"] in candidates:
                    existing.add(data["name"])
                elif yaml_file.stem in candidates:
                    existing.add(yaml_file.stem)
            except Exception:
                continue
        return existing
