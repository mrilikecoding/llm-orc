"""Ensemble CRUD handler for MCP server."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.core.config.ensemble_config import EnsembleLoader
from llm_orc.mcp.project_context import ProjectContext
from llm_orc.mcp.utils import get_agent_attr as _get_agent_attr
from llm_orc.schemas.agent_config import parse_agent_config
from llm_orc.services.handlers.scope import Scope, find_in_scope, parse_scope

_PRESERVED_FIELDS = (
    "name",
    "type",
    "model_profile",
    "ensemble",
    "script",
    "parameters",
    "depends_on",
    "system_prompt",
    "cache",
    "fan_out",
    "input_key",
    "when",
    "on_dependency_failure",
    "output_format",
    "timeout_seconds",
    "input_scope",
    "temperature",
    "max_tokens",
    "options",
    "response_format",
    "fallback_model_profile",
    "model",
    "provider",
    "loop",
    "dispatch",
)


def _agent_to_dict(agent: Any) -> dict[str, Any]:
    """Convert an AgentConfig object to a plain dict for YAML serialization."""
    agent_dict: dict[str, Any] = {}
    for attr in _PRESERVED_FIELDS:
        val = _get_agent_attr(agent, attr)
        if val is not None:
            if attr == "loop" and hasattr(val, "model_dump"):
                agent_dict[attr] = val.model_dump()
            else:
                agent_dict[attr] = val
    return agent_dict


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

    def set_project_context(self, ctx: ProjectContext) -> None:
        """Update handler to use new project context."""
        self._config_manager = ctx.config_manager

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

        scope = parse_scope(arguments)
        local_dir = self._dir_for_scope(scope)
        if local_dir is None:
            global_ensembles = self._config_manager.global_config_dir / "ensembles"
            raise ValueError(
                "No project directory (.llm-orc) here; pass scope: global to "
                f"write under {global_ensembles}, or run llm-orc config init"
            )
        target_file = local_dir / f"{name}.yaml"
        if target_file.exists():
            raise ValueError(f"Ensemble already exists: {name}")

        agents_copied = 0
        if from_template:
            agents, description, agents_copied = self._copy_from_template(
                from_template, description
            )

        normalized_agents: list[dict[str, Any]] = []
        for agent in agents:
            if isinstance(agent, dict):
                # Strip None values so extra="forbid" schemas don't reject
                # keys like model_profile=null on non-LLM agents.
                cleaned = {k: v for k, v in agent.items() if v is not None}
                cfg = parse_agent_config(cleaned)
                normalized_agents.append(cfg.model_dump(exclude_none=True))
            else:
                normalized_agents.append(agent)

        ensemble_data = {
            "name": name,
            "description": description,
            "agents": normalized_agents,
        }
        yaml_content = yaml.dump(ensemble_data, default_flow_style=False)

        local_dir.mkdir(parents=True, exist_ok=True)
        target_file.write_text(yaml_content)

        return {
            "created": True,
            "path": str(target_file),
            "agents_copied": agents_copied,
            "scope": scope,
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

        scope = parse_scope(arguments)
        ensemble_file = self._find_in_scope(ensemble_name, scope)

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

        scope = parse_scope(arguments)
        ensemble_path = self._find_in_scope(ensemble_name, scope)

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

    def _dir_for_scope(self, scope: Scope) -> Path | None:
        """Write directory for one scope; None when the project has none."""
        if scope == "global":
            return self._config_manager.global_config_dir / "ensembles"
        local_config_dir = self._config_manager.local_config_dir
        if local_config_dir is None:
            return None
        return local_config_dir / "ensembles"

    def _find_in_scope(self, ensemble_name: str, scope: Scope) -> Path:
        return find_in_scope(
            name=ensemble_name,
            filename=f"{ensemble_name}.yaml",
            scope=scope,
            scope_dir=self._dir_for_scope(scope),
            search_dirs=[Path(d) for d in self._config_manager.get_ensembles_dirs()],
            classify=self._config_manager.classify_tier,
            label="Ensemble",
        )

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

        agents = [
            dict(agent) if isinstance(agent, dict) else _agent_to_dict(agent)
            for agent in template_config.agents
        ]

        final_description = description or template_config.description
        return agents, final_description, len(agents)
