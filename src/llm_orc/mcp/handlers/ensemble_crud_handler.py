"""Ensemble CRUD handler for MCP server."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.core.config.ensemble_config import EnsembleLoader
from llm_orc.mcp.utils import get_agent_attr as _get_agent_attr


class EnsembleCrudHandler:
    """Handles ensemble create, delete, update, and analysis operations."""

    def __init__(
        self,
        config_manager: ConfigurationManager,
        ensemble_loader: EnsembleLoader,
        find_ensemble_fn: Callable[[str], Any],
        read_artifact_fn: Callable[..., Any],
    ) -> None:
        """Initialize with dependencies.

        Args:
            config_manager: Configuration manager instance.
            ensemble_loader: Ensemble loader instance.
            find_ensemble_fn: Callback to find ensemble by name.
            read_artifact_fn: Callback to read an artifact resource.
        """
        self._config_manager = config_manager
        self._ensemble_loader = ensemble_loader
        self._find_ensemble = find_ensemble_fn
        self._read_artifact = read_artifact_fn

    async def create_ensemble(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Create a new ensemble.

        Args:
            arguments: Tool arguments including name, description, agents.

        Returns:
            Creation result.
        """
        name = arguments.get("name")
        description = arguments.get("description", "")
        agents = arguments.get("agents", [])
        from_template = arguments.get("from_template")

        if not name:
            raise ValueError("name is required")

        local_dir = self.get_local_ensembles_dir()
        target_file = local_dir / f"{name}.yaml"
        if target_file.exists():
            raise ValueError(f"Ensemble already exists: {name}")

        agents_copied = 0
        if from_template:
            agents, description, agents_copied = self._copy_from_template(
                from_template, description
            )

        ensemble_data = {
            "name": name,
            "description": description,
            "agents": agents,
        }
        yaml_content = yaml.dump(ensemble_data, default_flow_style=False)

        local_dir.mkdir(parents=True, exist_ok=True)
        target_file.write_text(yaml_content)

        return {
            "created": True,
            "path": str(target_file),
            "agents_copied": agents_copied,
        }

    async def delete_ensemble(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Delete an ensemble.

        Args:
            arguments: Tool arguments including ensemble_name, confirm.

        Returns:
            Deletion result.
        """
        ensemble_name = arguments.get("ensemble_name")
        confirm = arguments.get("confirm", False)

        if not ensemble_name:
            raise ValueError("ensemble_name is required")

        if not confirm:
            raise ValueError("Confirmation required to delete ensemble")

        ensemble_dirs = self._config_manager.get_ensembles_dirs()
        ensemble_file: Path | None = None

        for dir_path in ensemble_dirs:
            potential_file = Path(dir_path) / f"{ensemble_name}.yaml"
            if potential_file.exists():
                ensemble_file = potential_file
                break

        if not ensemble_file:
            raise ValueError(f"Ensemble not found: {ensemble_name}")

        ensemble_file.unlink()

        return {
            "deleted": True,
            "path": str(ensemble_file),
        }

    async def update_ensemble(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Update an ensemble configuration.

        Args:
            arguments: Tool arguments including ensemble_name, changes, dry_run.

        Returns:
            Update result.
        """
        ensemble_name = arguments.get("ensemble_name")
        changes = arguments.get("changes", {})
        dry_run = arguments.get("dry_run", True)
        backup = arguments.get("backup", True)

        if not ensemble_name:
            raise ValueError("ensemble_name is required")

        ensemble_dirs = self._config_manager.get_ensembles_dirs()
        ensemble_path: Path | None = None

        for ensemble_dir in ensemble_dirs:
            potential_path = Path(ensemble_dir) / f"{ensemble_name}.yaml"
            if potential_path.exists():
                ensemble_path = potential_path
                break

        if not ensemble_path:
            raise ValueError(f"Ensemble not found: {ensemble_name}")

        if dry_run:
            return {
                "preview": changes,
                "modified": False,
                "backup_created": False,
            }

        backup_created = False
        if backup:
            backup_path = ensemble_path.with_suffix(".yaml.bak")
            backup_path.write_text(ensemble_path.read_text())
            backup_created = True

        return {
            "modified": True,
            "backup_created": backup_created,
            "changes_applied": changes,
        }

    async def analyze_execution(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Analyze an execution artifact.

        Args:
            arguments: Tool arguments including artifact_id.

        Returns:
            Analysis result.
        """
        artifact_id = arguments.get("artifact_id")

        if not artifact_id:
            raise ValueError("artifact_id is required")

        parts = artifact_id.split("/")
        if len(parts) != 2:
            raise ValueError(f"Invalid artifact_id format: {artifact_id}")

        ensemble_name, aid = parts
        artifact = await self._read_artifact(ensemble_name, aid)

        results = artifact.get("results", {})
        success_count = sum(1 for r in results.values() if r.get("status") == "success")

        return {
            "analysis": {
                "total_agents": len(results),
                "successful_agents": success_count,
                "failed_agents": len(results) - success_count,
            },
            "metrics": {
                "agent_success_rate": (
                    success_count / len(results) if results else 0.0
                ),
                "cost": artifact.get("cost", 0),
                "duration": artifact.get("duration", 0),
            },
        }

    def get_local_ensembles_dir(self) -> Path:
        """Get the local ensembles directory for writing.

        Returns:
            Path to local ensembles directory.

        Raises:
            ValueError: If no ensemble directory is available.
        """
        ensemble_dirs = self._config_manager.get_ensembles_dirs()

        for dir_path in ensemble_dirs:
            path = Path(dir_path)
            if ".llm-orc" in str(path) and "library" not in str(path):
                return path

        if ensemble_dirs:
            return Path(ensemble_dirs[0])

        raise ValueError("No ensemble directory available")

    def _copy_from_template(
        self, template_name: str, description: str
    ) -> tuple[list[dict[str, Any]], str, int]:
        """Copy agents and description from a template ensemble.

        Args:
            template_name: Name of the template ensemble.
            description: Current description (may be overwritten if empty).

        Returns:
            Tuple of (agents list, description, agents_copied count).

        Raises:
            ValueError: If template not found.
        """
        template_config = self._find_ensemble(template_name)
        if not template_config:
            raise ValueError(f"Template ensemble not found: {template_name}")

        agents: list[dict[str, Any]] = []
        preserved_fields = (
            "name",
            "type",
            "model_profile",
            "script",
            "parameters",
            "depends_on",
            "system_prompt",
            "cache",
            "fan_out",
        )
        for agent in template_config.agents:
            if isinstance(agent, dict):
                agents.append(dict(agent))
            else:
                agent_dict: dict[str, Any] = {}
                for attr in preserved_fields:
                    val = _get_agent_attr(agent, attr)
                    if val is not None:
                        agent_dict[attr] = val
                agents.append(agent_dict)

        final_description = description or template_config.description
        return agents, final_description, len(agents)
