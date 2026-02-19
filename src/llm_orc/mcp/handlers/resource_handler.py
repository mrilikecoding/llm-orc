"""Resource reading handler for MCP server."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.core.config.ensemble_config import EnsembleLoader


class ResourceHandler:
    """Handles resource reading operations for ensembles, profiles, artifacts."""

    def __init__(
        self,
        config_manager: ConfigurationManager,
        ensemble_loader: EnsembleLoader,
    ) -> None:
        """Initialize with configuration manager and ensemble loader."""
        self._config_manager = config_manager
        self._ensemble_loader = ensemble_loader

    async def read_resource(self, uri: str) -> Any:
        """Read an MCP resource by URI.

        Args:
            uri: Resource URI (e.g., llm-orc://ensembles)

        Returns:
            Resource content.

        Raises:
            ValueError: If resource not found.
        """
        if not uri.startswith("llm-orc://"):
            raise ValueError(f"Invalid URI scheme: {uri}")

        path = uri[len("llm-orc://") :]
        parts = path.split("/")

        if parts[0] == "ensembles":
            return await self.read_ensembles()
        elif parts[0] == "ensemble" and len(parts) > 1:
            return await self.read_ensemble(parts[1])
        elif parts[0] == "artifacts" and len(parts) > 1:
            return await self.read_artifacts(parts[1])
        elif parts[0] == "artifact" and len(parts) > 2:
            return await self.read_artifact(parts[1], parts[2])
        elif parts[0] == "metrics" and len(parts) > 1:
            return await self.read_metrics(parts[1])
        elif parts[0] == "profiles":
            return await self.read_profiles()
        else:
            raise ValueError(f"Resource not found: {uri}")

    async def read_ensembles(self) -> list[dict[str, Any]]:
        """Read all ensembles.

        Returns:
            List of ensemble metadata.
        """
        ensembles: list[dict[str, Any]] = []
        ensemble_dirs = self._config_manager.get_ensembles_dirs()

        for ensemble_dir in ensemble_dirs:
            if not Path(ensemble_dir).exists():
                continue

            source = self.determine_source(ensemble_dir)

            ensemble_dir_path = Path(ensemble_dir)
            for yaml_file in ensemble_dir_path.glob("**/*.yaml"):
                try:
                    config = self._ensemble_loader.load_from_file(str(yaml_file))
                    if config:
                        relative_path = str(yaml_file.relative_to(ensemble_dir_path))
                        ensembles.append(
                            {
                                "name": config.name,
                                "source": source,
                                "relative_path": relative_path,
                                "agent_count": len(config.agents),
                                "description": config.description,
                            }
                        )
                except Exception:
                    continue

        return ensembles

    def determine_source(self, ensemble_dir: Path) -> str:
        """Determine the source type of an ensemble directory.

        Args:
            ensemble_dir: Path to ensemble directory.

        Returns:
            Source type: 'local', 'library', or 'global'.
        """
        path = ensemble_dir
        if ".llm-orc" in str(path) and "library" not in str(path):
            return "local"
        elif "library" in str(path):
            return "library"
        else:
            return "global"

    async def read_ensemble(self, name: str) -> dict[str, Any]:
        """Read specific ensemble configuration.

        Args:
            name: Ensemble name.

        Returns:
            Ensemble configuration.

        Raises:
            ValueError: If ensemble not found.
        """
        ensemble_dirs = self._config_manager.get_ensembles_dirs()

        for ensemble_dir in ensemble_dirs:
            config = self._ensemble_loader.find_ensemble(str(ensemble_dir), name)
            if config:
                agents_list = []
                for agent in config.agents:
                    if isinstance(agent, dict):
                        agents_list.append(
                            {
                                "name": agent.get("name", ""),
                                "model_profile": agent.get("model_profile"),
                                "depends_on": agent.get("depends_on") or [],
                            }
                        )
                    else:
                        agents_list.append(
                            {
                                "name": agent.name,
                                "model_profile": agent.model_profile,
                                "depends_on": agent.depends_on or [],
                            }
                        )
                return {
                    "name": config.name,
                    "description": config.description,
                    "agents": agents_list,
                }

        raise ValueError(f"Ensemble not found: {name}")

    async def read_artifacts(self, ensemble_name: str) -> list[dict[str, Any]]:
        """Read artifacts for an ensemble.

        Args:
            ensemble_name: Name of the ensemble.

        Returns:
            List of artifact metadata.
        """
        artifacts: list[dict[str, Any]] = []
        artifacts_dir = self.get_artifacts_dir() / ensemble_name

        if not artifacts_dir.exists():
            return []

        for artifact_dir in artifacts_dir.iterdir():
            if not artifact_dir.is_dir() or artifact_dir.is_symlink():
                continue

            execution_file = artifact_dir / "execution.json"
            if not execution_file.exists():
                continue

            try:
                artifact_data = json.loads(execution_file.read_text())
                metadata = artifact_data.get("metadata", {})
                artifacts.append(
                    {
                        "id": artifact_dir.name,
                        "timestamp": metadata.get("started_at"),
                        "status": artifact_data.get("status"),
                        "duration": metadata.get("duration"),
                        "agent_count": metadata.get("agents_used"),
                    }
                )
            except Exception:
                continue

        return artifacts

    async def read_artifact(
        self, ensemble_name: str, artifact_id: str
    ) -> dict[str, Any]:
        """Read specific artifact.

        Args:
            ensemble_name: Name of the ensemble.
            artifact_id: Artifact ID (timestamp directory name).

        Returns:
            Artifact data.

        Raises:
            ValueError: If artifact not found.
        """
        artifact_dir = self.get_artifacts_dir() / ensemble_name / artifact_id
        execution_file = artifact_dir / "execution.json"

        if not execution_file.exists():
            raise ValueError(f"Artifact not found: {ensemble_name}/{artifact_id}")

        result: dict[str, Any] = json.loads(execution_file.read_text())
        return result

    async def read_metrics(self, ensemble_name: str) -> dict[str, Any]:
        """Read metrics for an ensemble.

        Args:
            ensemble_name: Name of the ensemble.

        Returns:
            Aggregated metrics.
        """
        artifacts = await self.read_artifacts(ensemble_name)

        if not artifacts:
            return {
                "success_rate": 0.0,
                "avg_cost": 0.0,
                "avg_duration": 0.0,
                "total_executions": 0,
            }

        success_count = sum(1 for a in artifacts if a.get("status") == "success")

        def parse_duration(dur: str | float | None) -> float:
            if dur is None:
                return 0.0
            if isinstance(dur, int | float):
                return float(dur)
            if isinstance(dur, str) and dur.endswith("s"):
                try:
                    return float(dur[:-1])
                except ValueError:
                    return 0.0
            return 0.0

        total_duration = sum(parse_duration(a.get("duration")) for a in artifacts)

        return {
            "success_rate": (success_count / len(artifacts) if artifacts else 0.0),
            "avg_cost": 0.0,
            "avg_duration": (total_duration / len(artifacts) if artifacts else 0.0),
            "total_executions": len(artifacts),
        }

    async def read_profiles(self) -> list[dict[str, Any]]:
        """Read model profiles.

        Returns:
            List of model profile configurations.
        """
        profiles: list[dict[str, Any]] = []
        model_profiles = self._config_manager.get_model_profiles()

        for name, config in model_profiles.items():
            profiles.append(
                {
                    "name": name,
                    "provider": config.get("provider", "unknown"),
                    "model": config.get("model", "unknown"),
                }
            )

        return profiles

    def get_artifacts_dir(self) -> Path:
        """Get artifacts directory path.

        Returns:
            Path to artifacts directory.
        """
        global_config_path = Path(self._config_manager.global_config_dir)
        if global_config_path.name == "artifacts" and global_config_path.exists():
            return global_config_path

        local_artifacts = Path.cwd() / ".llm-orc" / "artifacts"
        if local_artifacts.exists():
            return local_artifacts

        global_artifacts = global_config_path / "artifacts"
        if global_artifacts.exists():
            return global_artifacts

        return local_artifacts
