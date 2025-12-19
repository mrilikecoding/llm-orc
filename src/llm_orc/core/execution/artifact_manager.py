"""Artifact manager for saving ensemble execution results."""

import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

from llm_orc.schemas.script_agent import ScriptAgentOutput


class ArtifactManager:
    """Manages saving execution results to structured artifact directories."""

    def __init__(self, base_dir: Path | str = ".") -> None:
        """Initialize artifact manager with base directory.

        Args:
            base_dir: Base directory for artifacts (default: current directory)
        """
        self.base_dir = Path(base_dir) if isinstance(base_dir, str) else base_dir
        self._artifact_cache: dict[str, Any] = {}
        self._shared_artifacts: dict[str, dict[str, Any]] = {}

    def save_execution_results(
        self,
        ensemble_name: str,
        results: dict[str, Any],
        timestamp: str | None = None,
        relative_path: str | None = None,
    ) -> Path:
        """Save execution results to artifacts directory.

        Args:
            ensemble_name: Name of the ensemble
            results: Execution results dictionary
            timestamp: Optional timestamp string (generated if None)
            relative_path: Optional relative path for mirrored directory structure

        Returns:
            Path to the created artifact directory

        Raises:
            ValueError: If ensemble_name is invalid
            PermissionError: If directory creation fails
            TypeError: If results cannot be serialized to JSON
        """
        # Validate ensemble name
        if not ensemble_name or "\0" in ensemble_name or "\n" in ensemble_name:
            raise ValueError(f"Invalid ensemble name: {ensemble_name!r}")

        # Generate timestamp if not provided
        if timestamp is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]

        # Create directory structure
        artifacts_dir = self.base_dir / ".llm-orc" / "artifacts"

        # Use mirrored directory structure if relative_path is provided
        if relative_path:
            ensemble_dir = artifacts_dir / relative_path / ensemble_name
        else:
            ensemble_dir = artifacts_dir / ensemble_name

        timestamped_dir = ensemble_dir / timestamp

        # Create directories (parents=True, exist_ok=True for concurrency)
        try:
            timestamped_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            raise PermissionError(
                "Permission denied creating artifact directory"
            ) from e

        # Save execution.json
        json_file = timestamped_dir / "execution.json"
        try:
            with json_file.open("w") as f:
                json.dump(results, f, indent=2)
        except TypeError as e:
            raise TypeError("Results cannot be serialized to JSON") from e

        # Generate and save execution.md
        md_file = timestamped_dir / "execution.md"
        markdown_content = self._generate_markdown_report(results)
        md_file.write_text(markdown_content)

        # Update latest symlink
        self._update_latest_symlink(ensemble_dir, timestamped_dir)

        return timestamped_dir

    def _generate_markdown_report(self, results: dict[str, Any]) -> str:
        """Generate markdown report from execution results.

        Args:
            results: Execution results dictionary

        Returns:
            Formatted markdown string
        """
        lines = ["# Ensemble Execution Report", ""]

        # Add basic info
        lines.extend(self._add_basic_info(results))

        # Add fan-out summary if present
        metadata = results.get("metadata", {})
        if "fan_out" in metadata:
            lines.extend(self._add_fan_out_summary(metadata["fan_out"]))

        # Add agent results
        if "agents" in results and results["agents"]:
            lines.extend(["## Agent Results", ""])
            lines.extend(self._add_agent_results(results["agents"]))

        # Add fan-out agent results if present
        if "results" in results:
            lines.extend(self._add_fan_out_results(results["results"]))

        return "\n".join(lines)

    def _add_basic_info(self, results: dict[str, Any]) -> list[str]:
        """Add basic execution info to markdown lines."""
        lines: list[str] = []

        if "ensemble_name" in results:
            lines.extend([f"**Ensemble:** {results['ensemble_name']}", ""])

        if "timestamp" in results:
            lines.extend([f"**Executed:** {results['timestamp']}", ""])

        if "input" in results:
            lines.extend([f"**Input:** {results['input']}", ""])

        # Add duration info
        if "total_duration_ms" in results:
            duration_str = self._format_duration(results["total_duration_ms"])
            lines.extend([f"**Total Duration:** {duration_str}", ""])

        return lines

    def _add_agent_results(self, agents: list[dict[str, Any]]) -> list[str]:
        """Add agent results to markdown lines."""
        lines: list[str] = []

        for agent in agents:
            agent_name = agent.get("name", "Unknown")
            status = agent.get("status", "unknown")

            lines.append(f"### {agent_name}")
            lines.append(f"**Status:** {status}")

            if status == "completed" and "result" in agent:
                lines.extend(["", "**Output:**", "```", agent["result"], "```"])
            elif status == "failed" and "error" in agent:
                lines.extend(["", "**Error:**", "```", agent["error"], "```"])

            if "duration_ms" in agent:
                duration_str = self._format_duration(agent["duration_ms"])
                lines.append(f"**Duration:** {duration_str}")

            lines.append("")  # Empty line between agents

        return lines

    def _format_duration(self, duration_ms: int) -> str:
        """Format duration in milliseconds to human readable string."""
        if duration_ms >= 1000:
            return f"{duration_ms / 1000:.1f}s"
        return f"{duration_ms}ms"

    def _add_fan_out_summary(self, fan_out_stats: dict[str, Any]) -> list[str]:
        """Add fan-out execution summary to markdown lines."""
        lines: list[str] = ["## Fan-Out Execution Summary", ""]

        for agent_name, stats in fan_out_stats.items():
            total = stats.get("total_instances", 0)
            successful = stats.get("successful_instances", 0)
            failed = stats.get("failed_instances", 0)

            lines.append(f"**{agent_name}**: {successful}/{total} successful")
            if failed > 0:
                lines.append(f"  - {failed} failed")
            lines.append("")

        return lines

    def _add_fan_out_results(self, results: dict[str, Any]) -> list[str]:
        """Add fan-out agent results to markdown lines."""
        lines: list[str] = []
        has_fan_out = False

        for agent_name, result in results.items():
            if not isinstance(result, dict) or not result.get("fan_out"):
                continue

            if not has_fan_out:
                lines.extend(["## Fan-Out Agent Results", ""])
                has_fan_out = True

            status = result.get("status", "unknown")
            instances = result.get("instances", [])

            lines.append(f"### {agent_name}")
            lines.append(f"**Status:** {status}")

            # Count successes
            success_count = sum(
                1 for inst in instances if inst.get("status") == "success"
            )
            lines.append(f"**Instances:** {success_count}/{len(instances)} successful")

            # Show failed instances with errors
            failed_instances = [
                inst for inst in instances if inst.get("status") == "failed"
            ]
            if failed_instances:
                lines.extend(["", "**Failed Instances:**"])
                for inst in failed_instances:
                    idx = inst.get("index", "?")
                    error = inst.get("error", "Unknown error")
                    lines.append(f"- [{idx}]: {error}")

            lines.append("")

        return lines

    def _update_latest_symlink(self, ensemble_dir: Path, target_dir: Path) -> None:
        """Update the latest symlink to point to the newest execution.

        Args:
            ensemble_dir: Directory containing ensemble executions
            target_dir: Target directory to point to
        """
        latest_link = ensemble_dir / "latest"

        # Remove existing symlink if it exists
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()

        # Create new symlink (use relative path for portability)
        relative_target = target_dir.name
        latest_link.symlink_to(relative_target)

    def list_ensembles(self) -> list[dict[str, Any]]:
        """List all ensembles with artifact information.

        Returns:
            List of ensemble dictionaries with execution information
        """
        artifacts_dir = self.base_dir / ".llm-orc" / "artifacts"

        if not artifacts_dir.exists():
            return []

        ensembles = self._find_ensemble_directories(artifacts_dir)
        return sorted(ensembles, key=lambda x: x["name"])

    def _find_ensemble_directories(self, artifacts_dir: Path) -> list[dict[str, Any]]:
        """Find all ensemble directories and their execution information.

        Args:
            artifacts_dir: Base artifacts directory to search

        Returns:
            List of ensemble dictionaries with execution information
        """
        ensembles: list[dict[str, Any]] = []

        # Recursively search for ensemble directories
        for root_path in artifacts_dir.rglob("*"):
            if not root_path.is_dir():
                continue

            ensemble_info = self._extract_ensemble_info(root_path)
            if ensemble_info:
                ensembles.append(ensemble_info)

        return ensembles

    def _extract_ensemble_info(self, directory: Path) -> dict[str, Any] | None:
        """Extract ensemble information from a directory if it contains executions.

        Args:
            directory: Directory to check for ensemble executions

        Returns:
            Ensemble info dict if valid ensemble directory, None otherwise
        """
        execution_dirs = self._get_execution_directories(directory)

        if not execution_dirs:
            return None

        return {
            "name": directory.name,
            "latest_execution": max(execution_dirs),
            "executions_count": len(execution_dirs),
        }

    def _get_execution_directories(self, ensemble_dir: Path) -> list[str]:
        """Get list of valid execution directory names from ensemble directory.

        Args:
            ensemble_dir: Ensemble directory to scan

        Returns:
            List of valid execution directory names (timestamps)
        """
        execution_dirs: list[str] = []

        for item in ensemble_dir.iterdir():
            if self._is_valid_execution_directory(item):
                execution_dirs.append(item.name)

        return execution_dirs

    def _is_valid_execution_directory(self, path: Path) -> bool:
        """Check if path is a valid execution directory.

        Args:
            path: Path to check

        Returns:
            True if valid execution directory, False otherwise
        """
        return (
            path.is_dir()
            and path.name != "latest"
            and self._is_timestamp_directory(path.name)
        )

    def _is_timestamp_directory(self, name: str) -> bool:
        """Check if directory name looks like a timestamp (YYYYMMDD-HHMMSS-mmm)."""
        import re

        timestamp_pattern = r"^\d{8}-\d{6}-\d{3}$"
        return bool(re.match(timestamp_pattern, name))

    def get_latest_results(
        self, ensemble_name: str, relative_path: str | None = None
    ) -> dict[str, Any] | None:
        """Get the latest execution results for an ensemble.

        Args:
            ensemble_name: Name of the ensemble
            relative_path: Optional relative path for mirrored directory structure

        Returns:
            Latest execution results or None if not found
        """
        artifacts_dir = self.base_dir / ".llm-orc" / "artifacts"

        # Use mirrored directory structure if relative_path is provided
        if relative_path:
            ensemble_dir = artifacts_dir / relative_path / ensemble_name
        else:
            ensemble_dir = artifacts_dir / ensemble_name

        latest_link = ensemble_dir / "latest"

        if not latest_link.exists():
            return None

        # Read the execution.json file
        execution_json = latest_link / "execution.json"
        if not execution_json.exists():
            return None

        try:
            with execution_json.open("r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
                return None
        except (OSError, json.JSONDecodeError):
            return None

    def get_execution_results(
        self, ensemble_name: str, timestamp: str, relative_path: str | None = None
    ) -> dict[str, Any] | None:
        """Get specific execution results by timestamp.

        Args:
            ensemble_name: Name of the ensemble
            timestamp: Execution timestamp
            relative_path: Optional relative path for mirrored directory structure

        Returns:
            Execution results or None if not found
        """
        artifacts_dir = self.base_dir / ".llm-orc" / "artifacts"

        # Use mirrored directory structure if relative_path is provided
        if relative_path:
            execution_dir = artifacts_dir / relative_path / ensemble_name / timestamp
        else:
            execution_dir = artifacts_dir / ensemble_name / timestamp

        if not execution_dir.exists():
            return None

        execution_json = execution_dir / "execution.json"
        if not execution_json.exists():
            return None

        try:
            with execution_json.open("r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
                return None
        except (OSError, json.JSONDecodeError):
            return None

    def save_script_artifact(
        self,
        agent_name: str,
        script_name: str,
        script_output: ScriptAgentOutput,
        input_hash: str,
        metadata: dict[str, Any] | None = None,
        relative_path: str | None = None,
    ) -> Path:
        """Save script-specific artifact with metadata (ADR-001).

        Args:
            agent_name: Name of the script agent
            script_name: Name/path of the script
            script_output: Validated script output schema
            input_hash: Hash of the input data
            metadata: Additional metadata (version, environment, etc.)
            relative_path: Optional relative path for mirrored directory structure

        Returns:
            Path to the created script artifact directory

        Raises:
            ValueError: If agent_name or script_name is invalid
            TypeError: If script_output cannot be serialized
        """
        if not agent_name or not script_name:
            raise ValueError("agent_name and script_name are required")

        # Validate script output schema
        validated_output = self.validate_script_output(script_output)

        # Generate timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]

        # Create script artifacts directory structure
        artifacts_dir = self.base_dir / ".llm-orc" / "artifacts" / "scripts"

        if relative_path:
            script_dir = artifacts_dir / relative_path / agent_name
        else:
            script_dir = artifacts_dir / agent_name

        timestamped_dir = script_dir / timestamp

        # Create directories
        try:
            timestamped_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            raise PermissionError(
                "Permission denied creating script artifact directory"
            ) from e

        # Prepare artifact metadata
        artifact_metadata = {
            "agent_name": agent_name,
            "script_name": script_name,
            "input_hash": input_hash,
            "timestamp": timestamp,
            "version": metadata.get("version", "1.0") if metadata else "1.0",
            "environment": metadata.get("environment", "production")
            if metadata
            else "production",
            **(metadata or {}),
        }

        # Save script output
        output_file = timestamped_dir / "output.json"
        try:
            with output_file.open("w") as f:
                json.dump(validated_output.model_dump(), f, indent=2)
        except TypeError as e:
            raise TypeError("Script output cannot be serialized to JSON") from e

        # Save metadata
        metadata_file = timestamped_dir / "metadata.json"
        with metadata_file.open("w") as f:
            json.dump(artifact_metadata, f, indent=2)

        # Update latest symlink for this agent
        self._update_latest_symlink(script_dir, timestamped_dir)

        # Cache the artifact for performance
        cache_key = f"{agent_name}:{input_hash}"
        self._artifact_cache[cache_key] = {
            "output": validated_output,
            "metadata": artifact_metadata,
            "path": timestamped_dir,
        }

        return timestamped_dir

    def validate_script_output(
        self, script_output: ScriptAgentOutput
    ) -> ScriptAgentOutput:
        """Validate script output conforms to ScriptAgentOutput schema.

        Args:
            script_output: Script output to validate

        Returns:
            Validated script output

        Raises:
            ValueError: If validation fails
        """
        if not isinstance(script_output, ScriptAgentOutput):
            raise ValueError("script_output must be a ScriptAgentOutput instance")

        # Validate required fields are present
        if script_output.success is None:
            raise ValueError("ScriptAgentOutput.success field is required")

        # Validate agent_requests if present
        if script_output.agent_requests:
            for request in script_output.agent_requests:
                if not hasattr(request, "target_agent_type"):
                    raise ValueError("AgentRequest must have target_agent_type")
                if not hasattr(request, "parameters"):
                    raise ValueError("AgentRequest must have parameters")

        return script_output

    def get_script_artifacts(
        self, agent_name: str, relative_path: str | None = None
    ) -> list[dict[str, Any]]:
        """Get all script artifacts for an agent.

        Args:
            agent_name: Name of the script agent
            relative_path: Optional relative path for mirrored directory structure

        Returns:
            List of artifact information dictionaries
        """
        artifacts_dir = self.base_dir / ".llm-orc" / "artifacts" / "scripts"

        if relative_path:
            agent_dir = artifacts_dir / relative_path / agent_name
        else:
            agent_dir = artifacts_dir / agent_name

        if not agent_dir.exists():
            return []

        artifacts = []
        for item in agent_dir.iterdir():
            if self._is_valid_execution_directory(item):
                metadata_file = item / "metadata.json"
                if metadata_file.exists():
                    try:
                        with metadata_file.open("r") as f:
                            metadata = json.load(f)
                        artifacts.append(
                            {
                                "timestamp": item.name,
                                "path": item,
                                "metadata": metadata,
                            }
                        )
                    except (OSError, json.JSONDecodeError):
                        continue

        return sorted(artifacts, key=lambda x: x["timestamp"], reverse=True)

    def share_artifact(
        self, source_agent: str, target_agent: str, artifact_id: str
    ) -> bool:
        """Share artifact between agents in ensemble coordination.

        Args:
            source_agent: Name of the source agent
            target_agent: Name of the target agent
            artifact_id: Identifier of the artifact to share

        Returns:
            True if sharing succeeded, False otherwise
        """
        # Check if artifact exists in cache first
        cache_key = f"{source_agent}:{artifact_id}"
        if cache_key in self._artifact_cache:
            artifact_data = self._artifact_cache[cache_key]

            # Add to shared artifacts
            if target_agent not in self._shared_artifacts:
                self._shared_artifacts[target_agent] = {}

            self._shared_artifacts[target_agent][artifact_id] = {
                "source_agent": source_agent,
                "artifact_data": artifact_data,
                "shared_at": datetime.datetime.now().isoformat(),
            }
            return True

        # Check if artifact exists in file system
        artifacts = self.get_script_artifacts(source_agent)
        for artifact in artifacts:
            if artifact["metadata"].get("input_hash") == artifact_id:
                # Load artifact data
                output_file = artifact["path"] / "output.json"
                if output_file.exists():
                    try:
                        with output_file.open("r") as f:
                            output_data = json.load(f)

                        # Add to shared artifacts
                        if target_agent not in self._shared_artifacts:
                            self._shared_artifacts[target_agent] = {}

                        self._shared_artifacts[target_agent][artifact_id] = {
                            "source_agent": source_agent,
                            "artifact_data": {
                                "output": ScriptAgentOutput(**output_data),
                                "metadata": artifact["metadata"],
                                "path": artifact["path"],
                            },
                            "shared_at": datetime.datetime.now().isoformat(),
                        }
                        return True
                    except (OSError, json.JSONDecodeError):
                        continue

        return False

    def get_shared_artifacts(self, agent_name: str) -> dict[str, Any]:
        """Get artifacts shared with an agent.

        Args:
            agent_name: Name of the target agent

        Returns:
            Dictionary of shared artifacts by artifact_id
        """
        return self._shared_artifacts.get(agent_name, {})

    def get_cached_artifact(self, cache_key: str) -> dict[str, Any] | None:
        """Get cached artifact for performance optimization.

        Args:
            cache_key: Cache key for the artifact

        Returns:
            Cached artifact data or None if not found
        """
        return self._artifact_cache.get(cache_key)

    def _generate_input_hash(
        self, input_data: str, parameters: dict[str, Any] | None = None
    ) -> str:
        """Generate hash of input data and parameters for artifact caching.

        Args:
            input_data: Input data string
            parameters: Optional parameters dictionary

        Returns:
            SHA-256 hash of the input
        """
        hasher = hashlib.sha256()
        hasher.update(input_data.encode("utf-8"))

        if parameters:
            # Sort parameters for consistent hashing
            param_str = json.dumps(parameters, sort_keys=True)
            hasher.update(param_str.encode("utf-8"))

        return hasher.hexdigest()[:16]  # Use first 16 characters for readability
