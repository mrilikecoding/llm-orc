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


class TestArtifactManagerMirroredDirectoryStructure:
    """Test mirrored directory structure for hierarchical ensembles."""

    def test_save_execution_results_with_relative_path(
        self, artifact_manager: ArtifactManager, temp_dir: Path
    ) -> None:
        """Test that relative_path parameter creates mirrored directory structure."""
        ensemble_name = "network-analysis"
        relative_path = "research/network-science"
        timestamp = "20240115-103000-123"
        results = {"test": "data"}

        artifact_manager.save_execution_results(
            ensemble_name=ensemble_name,
            results=results,
            timestamp=timestamp,
            relative_path=relative_path,
        )

        # Check mirrored directory structure is created
        mirrored_dir = (
            temp_dir
            / ".llm-orc"
            / "artifacts"
            / relative_path
            / ensemble_name
            / timestamp
        )
        assert mirrored_dir.exists()

        # Check execution.json exists in mirrored location
        json_file = mirrored_dir / "execution.json"
        assert json_file.exists()

        # Check content is correct
        with json_file.open() as f:
            saved_data = json.load(f)
        assert saved_data == results

    def test_save_execution_results_without_relative_path_uses_legacy_structure(
        self, artifact_manager: ArtifactManager, temp_dir: Path
    ) -> None:
        """Test that omitting relative_path maintains backward compatibility."""
        ensemble_name = "test-ensemble"
        timestamp = "20240115-103000-123"
        results = {"test": "data"}

        artifact_manager.save_execution_results(
            ensemble_name=ensemble_name,
            results=results,
            timestamp=timestamp,
        )

        # Check legacy directory structure is used
        legacy_dir = temp_dir / ".llm-orc" / "artifacts" / ensemble_name / timestamp
        assert legacy_dir.exists()

        # Check execution.json exists in legacy location
        json_file = legacy_dir / "execution.json"
        assert json_file.exists()

    def test_update_latest_symlink_with_relative_path(
        self, artifact_manager: ArtifactManager, temp_dir: Path
    ) -> None:
        """Test that latest symlink is created in correct mirrored location."""
        ensemble_name = "analysis-tool"
        relative_path = "creative/storytelling"
        timestamp = "20240115-103000-123"
        results = {"test": "data"}

        artifact_manager.save_execution_results(
            ensemble_name=ensemble_name,
            results=results,
            timestamp=timestamp,
            relative_path=relative_path,
        )

        # Check latest symlink exists in mirrored location
        ensemble_dir = (
            temp_dir / ".llm-orc" / "artifacts" / relative_path / ensemble_name
        )
        latest_link = ensemble_dir / "latest"

        assert latest_link.exists()
        assert latest_link.is_symlink()

        # Should point to the timestamped directory
        target = latest_link.resolve()
        expected_target = (ensemble_dir / timestamp).resolve()
        assert target == expected_target

    def test_list_ensembles_includes_mirrored_structure(
        self, artifact_manager: ArtifactManager, temp_dir: Path
    ) -> None:
        """Test that list_ensembles finds ensembles in mirrored directories."""
        # Create ensembles in both legacy and mirrored structures
        artifact_manager.save_execution_results(
            ensemble_name="legacy-ensemble",
            results={"test": "legacy"},
            timestamp="20240115-103000-123",
        )

        artifact_manager.save_execution_results(
            ensemble_name="research-ensemble",
            results={"test": "research"},
            timestamp="20240115-103000-124",
            relative_path="research/ai-safety",
        )

        ensembles = artifact_manager.list_ensembles()

        # Should find both ensembles
        ensemble_names = [e["name"] for e in ensembles]
        assert "legacy-ensemble" in ensemble_names
        assert "research-ensemble" in ensemble_names

    def test_get_latest_results_with_relative_path(
        self, artifact_manager: ArtifactManager, temp_dir: Path
    ) -> None:
        """Test that get_latest_results works with mirrored directory structure."""
        ensemble_name = "data-processor"
        relative_path = "testing/integration"
        results = {"output": "processed data", "status": "success"}

        artifact_manager.save_execution_results(
            ensemble_name=ensemble_name,
            results=results,
            timestamp="20240115-103000-123",
            relative_path=relative_path,
        )

        # Should be able to retrieve latest results using relative path
        latest_results = artifact_manager.get_latest_results(
            ensemble_name, relative_path=relative_path
        )

        assert latest_results is not None
        assert latest_results["output"] == "processed data"
        assert latest_results["status"] == "success"

    def test_get_execution_results_with_relative_path(
        self, artifact_manager: ArtifactManager, temp_dir: Path
    ) -> None:
        """Test that get_execution_results works with mirrored directory structure."""
        ensemble_name = "validator"
        relative_path = "research/validation"
        timestamp = "20240115-103000-123"
        results = {"validation": "passed", "score": 95}

        artifact_manager.save_execution_results(
            ensemble_name=ensemble_name,
            results=results,
            timestamp=timestamp,
            relative_path=relative_path,
        )

        # Should be able to retrieve specific execution results
        execution_results = artifact_manager.get_execution_results(
            ensemble_name, timestamp, relative_path=relative_path
        )

        assert execution_results is not None
        assert execution_results["validation"] == "passed"
        assert execution_results["score"] == 95

    def test_nested_relative_paths_create_deep_directory_structure(
        self, artifact_manager: ArtifactManager, temp_dir: Path
    ) -> None:
        """Test that deeply nested relative paths create correct directory structure."""
        ensemble_name = "deep-analyzer"
        relative_path = "research/deep-learning/nlp/transformers"
        timestamp = "20240115-103000-123"
        results = {"model": "transformer", "accuracy": 0.95}

        artifact_manager.save_execution_results(
            ensemble_name=ensemble_name,
            results=results,
            timestamp=timestamp,
            relative_path=relative_path,
        )

        # Check deeply nested directory structure
        deep_dir = (
            temp_dir
            / ".llm-orc"
            / "artifacts"
            / "research"
            / "deep-learning"
            / "nlp"
            / "transformers"
            / ensemble_name
            / timestamp
        )
        assert deep_dir.exists()

        json_file = deep_dir / "execution.json"
        assert json_file.exists()

    def test_backward_compatibility_for_existing_methods(
        self, artifact_manager: ArtifactManager, temp_dir: Path
    ) -> None:
        """Test that existing method calls without relative_path still work."""
        # This tests the current API continues to work
        ensemble_name = "compatibility-test"
        timestamp = "20240115-103000-123"
        results = {"compatibility": "maintained"}

        # This should not fail (backward compatibility)
        result_dir = artifact_manager.save_execution_results(
            ensemble_name=ensemble_name,
            results=results,
            timestamp=timestamp,
        )

        assert result_dir.exists()

        # Legacy methods should still work
        latest_results = artifact_manager.get_latest_results(ensemble_name)
        assert latest_results is not None
        assert latest_results["compatibility"] == "maintained"

        execution_results = artifact_manager.get_execution_results(
            ensemble_name, timestamp
        )
        assert execution_results is not None
        assert execution_results["compatibility"] == "maintained"


class TestArtifactManagerEndToEndIntegration:
    """End-to-end integration tests for mirrored directory structure."""

    def test_full_workflow_mirrored_structure(
        self, artifact_manager: ArtifactManager, temp_dir: Path
    ) -> None:
        """Test complete workflow with mirrored directory structure."""
        # Simulate different ensemble types in different directories
        ensembles = [
            ("research-agent", "research/ai-safety", {"task": "safety analysis"}),
            ("creative-writer", "creative/storytelling", {"task": "story generation"}),
            ("test-validator", "testing/integration", {"task": "validation"}),
            ("legacy-ensemble", None, {"task": "legacy operation"}),  # No relative_path
        ]

        # Save multiple executions for each ensemble
        for ensemble_name, relative_path, results in ensembles:
            for i in range(2):
                timestamp = f"20240115-10300{i}-123"
                artifact_manager.save_execution_results(
                    ensemble_name=ensemble_name,
                    results=results,
                    timestamp=timestamp,
                    relative_path=relative_path,
                )

        # Test directory structure exists correctly
        for ensemble_name, relative_path, _ in ensembles:
            if relative_path:
                expected_dir = (
                    temp_dir / ".llm-orc" / "artifacts" / relative_path / ensemble_name
                )
            else:
                expected_dir = temp_dir / ".llm-orc" / "artifacts" / ensemble_name

            assert expected_dir.exists()
            assert (expected_dir / "latest").exists()

        # Test list_ensembles finds all ensembles
        found_ensembles = artifact_manager.list_ensembles()
        found_names = [e["name"] for e in found_ensembles]

        for ensemble_name, _, _ in ensembles:
            assert ensemble_name in found_names

        # Test get_latest_results works for both mirrored and legacy
        for ensemble_name, relative_path, expected_data in ensembles:
            latest = artifact_manager.get_latest_results(
                ensemble_name, relative_path=relative_path
            )
            assert latest is not None
            assert latest["task"] == expected_data["task"]

        # Test get_execution_results works for specific timestamps
        for ensemble_name, relative_path, expected_data in ensembles:
            execution = artifact_manager.get_execution_results(
                ensemble_name, "20240115-103001-123", relative_path=relative_path
            )
            assert execution is not None
            assert execution["task"] == expected_data["task"]
