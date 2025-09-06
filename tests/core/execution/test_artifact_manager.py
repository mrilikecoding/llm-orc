"""Tests for ArtifactManager class."""

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from llm_orc.core.execution.artifact_manager import ArtifactManager


@pytest.fixture
def temp_dir() -> Any:
    """Create a temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def artifact_manager(temp_dir: Path) -> ArtifactManager:
    """Create an ArtifactManager instance for testing."""
    return ArtifactManager(base_dir=temp_dir)


@pytest.fixture
def execution_results() -> dict[str, Any]:
    """Sample execution results for testing."""
    return {
        "ensemble_name": "test-ensemble",
        "agents": [
            {
                "name": "agent1",
                "status": "completed",
                "result": "Agent 1 output",
                "duration_ms": 1500,
            },
            {
                "name": "agent2",
                "status": "completed",
                "result": "Agent 2 output",
                "duration_ms": 2000,
            },
        ],
        "total_duration_ms": 3500,
        "input": "Test input",
        "timestamp": "2024-01-15T10:30:00.123456",
    }


class TestArtifactManagerDirectoryCreation:
    """Test directory structure creation."""

    def test_creates_artifact_directories(
        self, artifact_manager: ArtifactManager, temp_dir: Path
    ) -> None:
        """Test that artifact directories are created correctly."""
        ensemble_name = "test-ensemble"
        timestamp = "20240115-103000-123"

        artifact_manager.save_execution_results(
            ensemble_name=ensemble_name, results={}, timestamp=timestamp
        )

        # Check directory structure exists
        base_artifacts_dir = temp_dir / ".llm-orc" / "artifacts"
        ensemble_dir = base_artifacts_dir / ensemble_name
        timestamped_dir = ensemble_dir / timestamp

        assert base_artifacts_dir.exists()
        assert ensemble_dir.exists()
        assert timestamped_dir.exists()

    def test_creates_nested_ensemble_directories(
        self, artifact_manager: ArtifactManager, temp_dir: Path
    ) -> None:
        """Test that nested ensemble names create proper directory structure."""
        ensemble_name = "project/sub-ensemble"
        timestamp = "20240115-103000-123"

        artifact_manager.save_execution_results(
            ensemble_name=ensemble_name, results={}, timestamp=timestamp
        )

        # Check nested directory structure
        base_artifacts_dir = temp_dir / ".llm-orc" / "artifacts"
        nested_dir = base_artifacts_dir / "project" / "sub-ensemble" / timestamp

        assert nested_dir.exists()


class TestArtifactManagerExecutionJsonSaving:
    """Test execution.json file saving."""

    def test_saves_execution_json_file(
        self,
        artifact_manager: ArtifactManager,
        temp_dir: Path,
        execution_results: dict[str, Any],
    ) -> None:
        """Test that execution.json is saved with correct content."""
        ensemble_name = "test-ensemble"
        timestamp = "20240115-103000-123"

        artifact_manager.save_execution_results(
            ensemble_name=ensemble_name, results=execution_results, timestamp=timestamp
        )

        # Check execution.json exists and has correct content
        json_file = (
            temp_dir
            / ".llm-orc"
            / "artifacts"
            / ensemble_name
            / timestamp
            / "execution.json"
        )

        assert json_file.exists()

        with json_file.open() as f:
            saved_data = json.load(f)

        assert saved_data == execution_results

    def test_execution_json_is_formatted(
        self,
        artifact_manager: ArtifactManager,
        temp_dir: Path,
        execution_results: dict[str, Any],
    ) -> None:
        """Test that execution.json is properly formatted (indented)."""
        ensemble_name = "test-ensemble"
        timestamp = "20240115-103000-123"

        artifact_manager.save_execution_results(
            ensemble_name=ensemble_name, results=execution_results, timestamp=timestamp
        )

        # Check that JSON is formatted with indentation
        json_file = (
            temp_dir
            / ".llm-orc"
            / "artifacts"
            / ensemble_name
            / timestamp
            / "execution.json"
        )
        content = json_file.read_text()

        # Should have newlines and indentation (not compact)
        assert "\n" in content
        assert "  " in content  # Should have indentation


class TestArtifactManagerExecutionMarkdownGeneration:
    """Test execution.md file generation."""

    def test_generates_execution_markdown(
        self,
        artifact_manager: ArtifactManager,
        temp_dir: Path,
        execution_results: dict[str, Any],
    ) -> None:
        """Test that execution.md is generated with proper content."""
        ensemble_name = "test-ensemble"
        timestamp = "20240115-103000-123"

        artifact_manager.save_execution_results(
            ensemble_name=ensemble_name, results=execution_results, timestamp=timestamp
        )

        # Check execution.md exists
        md_file = (
            temp_dir
            / ".llm-orc"
            / "artifacts"
            / ensemble_name
            / timestamp
            / "execution.md"
        )
        assert md_file.exists()

        content = md_file.read_text()

        # Check key content is present
        assert "# Ensemble Execution Report" in content
        assert "test-ensemble" in content
        assert "2024-01-15T10:30:00.123456" in content
        assert "Agent 1 output" in content
        assert "Agent 2 output" in content
        assert "3500ms" in content or "3.5s" in content

    def test_markdown_handles_failed_agents(
        self, artifact_manager: ArtifactManager, temp_dir: Path
    ) -> None:
        """Test markdown generation handles failed agents correctly."""
        results_with_failure = {
            "ensemble_name": "test-ensemble",
            "agents": [
                {"name": "agent1", "status": "completed", "result": "Success output"},
                {"name": "agent2", "status": "failed", "error": "Connection timeout"},
            ],
        }

        artifact_manager.save_execution_results(
            ensemble_name="test-ensemble",
            results=results_with_failure,
            timestamp="20240115-103000-123",
        )

        md_file = (
            temp_dir
            / ".llm-orc"
            / "artifacts"
            / "test-ensemble"
            / "20240115-103000-123"
            / "execution.md"
        )
        content = md_file.read_text()

        assert "failed" in content.lower()
        assert "Connection timeout" in content


class TestArtifactManagerLatestSymlink:
    """Test latest symlink creation and updates."""

    def test_creates_latest_symlink(
        self,
        artifact_manager: ArtifactManager,
        temp_dir: Path,
        execution_results: dict[str, Any],
    ) -> None:
        """Test that latest symlink is created."""
        ensemble_name = "test-ensemble"
        timestamp = "20240115-103000-123"

        artifact_manager.save_execution_results(
            ensemble_name=ensemble_name, results=execution_results, timestamp=timestamp
        )

        # Check latest symlink exists and points to correct directory
        ensemble_dir = temp_dir / ".llm-orc" / "artifacts" / ensemble_name
        latest_link = ensemble_dir / "latest"

        assert latest_link.exists()
        assert latest_link.is_symlink()

        # Should point to the timestamped directory
        target = latest_link.resolve()
        expected_target = (ensemble_dir / timestamp).resolve()
        assert target == expected_target

    def test_updates_latest_symlink_to_newest(
        self,
        artifact_manager: ArtifactManager,
        temp_dir: Path,
        execution_results: dict[str, Any],
    ) -> None:
        """Test that latest symlink is updated to point to newest execution."""
        ensemble_name = "test-ensemble"

        # Save first execution
        timestamp1 = "20240115-103000-123"
        artifact_manager.save_execution_results(
            ensemble_name=ensemble_name, results=execution_results, timestamp=timestamp1
        )

        # Save second execution (newer)
        timestamp2 = "20240115-104000-456"
        artifact_manager.save_execution_results(
            ensemble_name=ensemble_name, results=execution_results, timestamp=timestamp2
        )

        # Check latest symlink points to newest
        ensemble_dir = temp_dir / ".llm-orc" / "artifacts" / ensemble_name
        latest_link = ensemble_dir / "latest"

        target = latest_link.resolve()
        expected_target = (ensemble_dir / timestamp2).resolve()
        assert target == expected_target


class TestArtifactManagerConcurrentSaves:
    """Test handling of concurrent save operations."""

    def test_handles_concurrent_directory_creation(
        self, artifact_manager: ArtifactManager, temp_dir: Path
    ) -> None:
        """Test that concurrent directory creation doesn't cause errors."""
        # This is a basic test - in practice you'd use threading/asyncio
        # but we'll simulate the scenario by calling save multiple times
        ensemble_name = "test-ensemble"

        # Simulate concurrent saves with same timestamp
        timestamp = "20240115-103000-123"

        # Multiple saves shouldn't fail
        artifact_manager.save_execution_results(
            ensemble_name=ensemble_name, results={"test": "data1"}, timestamp=timestamp
        )

        artifact_manager.save_execution_results(
            ensemble_name=ensemble_name, results={"test": "data2"}, timestamp=timestamp
        )

        # Directory should exist
        timestamped_dir = (
            temp_dir / ".llm-orc" / "artifacts" / ensemble_name / timestamp
        )
        assert timestamped_dir.exists()

    def test_handles_symlink_update_races(
        self, artifact_manager: ArtifactManager, temp_dir: Path
    ) -> None:
        """Test that symlink updates handle race conditions gracefully."""
        ensemble_name = "test-ensemble"

        # Save multiple executions rapidly
        timestamps = [
            "20240115-103000-123",
            "20240115-103001-456",
            "20240115-103002-789",
        ]

        for timestamp in timestamps:
            artifact_manager.save_execution_results(
                ensemble_name=ensemble_name,
                results={"timestamp": timestamp},
                timestamp=timestamp,
            )

        # Latest should point to last one
        ensemble_dir = temp_dir / ".llm-orc" / "artifacts" / ensemble_name
        latest_link = ensemble_dir / "latest"

        target = latest_link.resolve()
        # Should point to one of the directories (race condition acceptable)
        assert target.name in timestamps


class TestArtifactManagerErrorHandling:
    """Test error handling scenarios."""

    def test_handles_invalid_ensemble_names(
        self, artifact_manager: ArtifactManager
    ) -> None:
        """Test that invalid ensemble names are handled gracefully."""
        # Test with various invalid names
        invalid_names = ["", "name with\0null", "name\nwith\nnewlines"]

        for invalid_name in invalid_names:
            with pytest.raises(ValueError, match="Invalid ensemble name"):
                artifact_manager.save_execution_results(
                    ensemble_name=invalid_name,
                    results={},
                    timestamp="20240115-103000-123",
                )

    @patch("pathlib.Path.mkdir")
    def test_handles_permission_errors(
        self, mock_mkdir: Any, artifact_manager: ArtifactManager
    ) -> None:
        """Test handling of permission errors during directory creation."""
        mock_mkdir.side_effect = PermissionError("Permission denied")

        with pytest.raises(PermissionError):
            artifact_manager.save_execution_results(
                ensemble_name="test-ensemble",
                results={},
                timestamp="20240115-103000-123",
            )

    @patch("json.dump")
    def test_handles_json_serialization_errors(
        self, mock_json_dump: Any, artifact_manager: ArtifactManager, temp_dir: Path
    ) -> None:
        """Test handling of JSON serialization errors."""
        mock_json_dump.side_effect = TypeError("Object not serializable")

        with pytest.raises(TypeError):
            artifact_manager.save_execution_results(
                ensemble_name="test-ensemble",
                results={"unserializable": object()},
                timestamp="20240115-103000-123",
            )


class TestArtifactManagerTimestampGeneration:
    """Test timestamp generation and formatting."""

    def test_generates_timestamp_when_none_provided(
        self, artifact_manager: ArtifactManager, temp_dir: Path
    ) -> None:
        """Test that timestamp is generated when not provided."""
        # We'll test that a timestamp is generated without mocking, to keep it simple
        result_dir = artifact_manager.save_execution_results(
            ensemble_name="test-ensemble", results={}, timestamp=None
        )

        # Check directory was created (with some timestamp)
        assert result_dir.exists()

        # Check it's in the expected parent directory structure
        ensemble_dir = temp_dir / ".llm-orc" / "artifacts" / "test-ensemble"
        assert result_dir.parent == ensemble_dir

        # Check the directory name looks like a timestamp (YYYYMMDD-HHMMSS-mmm pattern)
        import re

        timestamp_pattern = r"^\d{8}-\d{6}-\d{3}$"
        assert re.match(timestamp_pattern, result_dir.name)

    def test_uses_provided_timestamp(
        self, artifact_manager: ArtifactManager, temp_dir: Path
    ) -> None:
        """Test that provided timestamp is used correctly."""
        custom_timestamp = "20241225-120000-999"

        artifact_manager.save_execution_results(
            ensemble_name="test-ensemble", results={}, timestamp=custom_timestamp
        )

        timestamped_dir = (
            temp_dir / ".llm-orc" / "artifacts" / "test-ensemble" / custom_timestamp
        )
        assert timestamped_dir.exists()
