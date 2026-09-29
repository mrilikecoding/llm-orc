"""Artifact management handler for MCP server."""

import shutil
import time
from pathlib import Path
from typing import Any

from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.core.config.state import ARTIFACTS_DIRNAME, resolve_state_dir
from llm_orc.mcp.project_context import ProjectContext


class ArtifactHandler:
    """Manages execution artifact operations."""

    def __init__(
        self,
        project_path: Path | None = None,
        config_manager: ConfigurationManager | None = None,
    ) -> None:
        """Initialize with optional project path and configuration manager."""
        self._project_path = project_path
        self._config_manager = config_manager

    def set_project_context(self, ctx: ProjectContext) -> None:
        """Update handler to use new project context."""
        self._project_path = ctx.project_path
        self._config_manager = ctx.config_manager

    def _get_artifacts_base(self) -> Path:
        """The artifacts directory, through the state rule (#196)."""
        local: Path | None
        if self._config_manager is not None:
            local = self._config_manager.local_config_dir
        elif (
            self._project_path is not None
            and (self._project_path / ".llm-orc").is_dir()
        ):
            local = self._project_path / ".llm-orc"
        else:
            local = None
        return resolve_state_dir(local) / ARTIFACTS_DIRNAME

    async def delete_artifact(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Delete an artifact."""
        artifact_id = arguments.get("artifact_id")
        confirm = arguments.get("confirm", False)

        if not artifact_id:
            raise ValueError("artifact_id is required")
        if not confirm:
            raise ValueError("Confirmation required to delete artifact")

        parts = artifact_id.split("/")
        if len(parts) != 2:
            raise ValueError("Invalid artifact_id format (expected ensemble/timestamp)")

        ensemble_name, timestamp = parts

        artifacts_base = self._get_artifacts_base()
        artifact_dir = artifacts_base / ensemble_name / timestamp

        if not artifact_dir.exists():
            raise ValueError(f"Artifact '{artifact_id}' not found")

        shutil.rmtree(artifact_dir)

        return {"deleted": True, "artifact_id": artifact_id}

    async def cleanup_artifacts(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Cleanup old artifacts."""
        ensemble_name = arguments.get("ensemble_name")
        older_than_days = arguments.get("older_than_days", 30)
        dry_run = arguments.get("dry_run", True)

        artifacts_base = self._get_artifacts_base()
        cutoff_time = time.time() - (older_than_days * 24 * 60 * 60)

        ensemble_dirs = self._get_ensemble_artifact_dirs(artifacts_base, ensemble_name)
        would_delete, deleted = self._process_old_artifacts(
            ensemble_dirs, cutoff_time, dry_run
        )

        if dry_run:
            return {
                "dry_run": True,
                "would_delete": would_delete,
                "count": len(would_delete),
            }
        return {
            "dry_run": False,
            "deleted": deleted,
            "count": len(deleted),
        }

    def _get_ensemble_artifact_dirs(
        self,
        artifacts_base: Path,
        ensemble_name: str | None,
    ) -> list[Path]:
        """Get list of ensemble artifact directories to check."""
        if ensemble_name:
            return [artifacts_base / ensemble_name]
        if artifacts_base.exists():
            return [d for d in artifacts_base.iterdir() if d.is_dir()]
        return []

    def _process_old_artifacts(
        self,
        ensemble_dirs: list[Path],
        cutoff_time: float,
        dry_run: bool,
    ) -> tuple[list[str], list[str]]:
        """Process and optionally delete old artifacts."""
        would_delete: list[str] = []
        deleted: list[str] = []

        for ensemble_dir in ensemble_dirs:
            if not ensemble_dir.exists():
                continue

            for artifact_dir in ensemble_dir.iterdir():
                if not artifact_dir.is_dir():
                    continue

                mtime = artifact_dir.stat().st_mtime
                if mtime < cutoff_time:
                    artifact_id = f"{ensemble_dir.name}/{artifact_dir.name}"
                    would_delete.append(artifact_id)

                    if not dry_run:
                        shutil.rmtree(artifact_dir)
                        deleted.append(artifact_id)

        return would_delete, deleted
