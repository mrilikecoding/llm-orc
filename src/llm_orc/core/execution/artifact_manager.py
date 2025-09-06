"""Artifact manager for saving ensemble execution results."""

import datetime
import json
from pathlib import Path
from typing import Any


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
    ) -> Path:
        """Save execution results to artifacts directory.

        Args:
            ensemble_name: Name of the ensemble
            results: Execution results dictionary
            timestamp: Optional timestamp string (generated if None)

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

        # Add agent results
        if "agents" in results and results["agents"]:
            lines.extend(["## Agent Results", ""])
            lines.extend(self._add_agent_results(results["agents"]))

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
        ensembles: list[dict[str, Any]] = []
        artifacts_dir = self.base_dir / ".llm-orc" / "artifacts"

        if not artifacts_dir.exists():
            return ensembles

        for ensemble_dir in artifacts_dir.iterdir():
            if not ensemble_dir.is_dir():
                continue

            ensemble_name = ensemble_dir.name
            execution_dirs = []

            # Count execution directories
            for item in ensemble_dir.iterdir():
                if item.is_dir() and item.name != "latest":
                    execution_dirs.append(item.name)

            if execution_dirs:
                latest_execution = max(execution_dirs)  # Most recent timestamp
                ensembles.append(
                    {
                        "name": ensemble_name,
                        "latest_execution": latest_execution,
                        "executions_count": len(execution_dirs),
                    }
                )

        return sorted(ensembles, key=lambda x: x["name"])

    def get_latest_results(self, ensemble_name: str) -> dict[str, Any] | None:
        """Get the latest execution results for an ensemble.

        Args:
            ensemble_name: Name of the ensemble

        Returns:
            Latest execution results or None if not found
        """
        artifacts_dir = self.base_dir / ".llm-orc" / "artifacts"
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
        self, ensemble_name: str, timestamp: str
    ) -> dict[str, Any] | None:
        """Get specific execution results by timestamp.

        Args:
            ensemble_name: Name of the ensemble
            timestamp: Execution timestamp

        Returns:
            Execution results or None if not found
        """
        artifacts_dir = self.base_dir / ".llm-orc" / "artifacts"
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
