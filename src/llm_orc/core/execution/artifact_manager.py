"""Artifact manager for saving ensemble execution results."""

import datetime
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

