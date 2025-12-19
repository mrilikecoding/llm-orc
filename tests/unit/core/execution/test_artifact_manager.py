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


class TestArtifactManagerListEnsembles:
    """Test list_ensembles method comprehensive coverage."""

    def test_list_ensembles_returns_empty_list_when_no_artifacts_dir(
        self, artifact_manager: ArtifactManager, temp_dir: Path
    ) -> None:
        """Test list_ensembles returns empty list when artifacts dir doesn't exist."""
        # No artifacts created, directory shouldn't exist
        ensembles = artifact_manager.list_ensembles()
        assert ensembles == []

    def test_list_ensembles_returns_empty_list_when_artifacts_dir_empty(
        self, artifact_manager: ArtifactManager, temp_dir: Path
    ) -> None:
        """Test that list_ensembles returns empty list when artifacts dir is empty."""
        # Create empty artifacts directory
        artifacts_dir = temp_dir / ".llm-orc" / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        ensembles = artifact_manager.list_ensembles()
        assert ensembles == []

    def test_list_ensembles_ignores_non_timestamp_directories(
        self, artifact_manager: ArtifactManager, temp_dir: Path
    ) -> None:
        """Test list_ensembles ignores directories without valid timestamps."""
        artifacts_dir = temp_dir / ".llm-orc" / "artifacts"

        # Create ensemble directory with non-timestamp subdirectories
        ensemble_dir = artifacts_dir / "test-ensemble"
        ensemble_dir.mkdir(parents=True, exist_ok=True)

        # Create non-timestamp directories
        (ensemble_dir / "not-a-timestamp").mkdir()
        (ensemble_dir / "20241301-9999").mkdir()  # Wrong format (too short)
        (ensemble_dir / "latest").mkdir()  # Should be ignored
        (ensemble_dir / "some-other-dir").mkdir()

        ensembles = artifact_manager.list_ensembles()
        assert ensembles == []

    def test_list_ensembles_ignores_files_in_search(
        self, artifact_manager: ArtifactManager, temp_dir: Path
    ) -> None:
        """Test that list_ensembles ignores files during recursive search."""
        artifacts_dir = temp_dir / ".llm-orc" / "artifacts"

        # Create some files in artifacts directory
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / "some-file.txt").write_text("content")

        # Create nested files
        nested_dir = artifacts_dir / "nested"
        nested_dir.mkdir()
        (nested_dir / "another-file.json").write_text("{}")

        ensembles = artifact_manager.list_ensembles()
        assert ensembles == []

    def test_list_ensembles_finds_single_ensemble_with_single_execution(
        self, artifact_manager: ArtifactManager, temp_dir: Path
    ) -> None:
        """Test list_ensembles finds single ensemble with one execution."""
        artifact_manager.save_execution_results(
            ensemble_name="simple-ensemble",
            results={"test": "data"},
            timestamp="20240115-103000-123",
        )

        ensembles = artifact_manager.list_ensembles()

        assert len(ensembles) == 1
        assert ensembles[0]["name"] == "simple-ensemble"
        assert ensembles[0]["latest_execution"] == "20240115-103000-123"
        assert ensembles[0]["executions_count"] == 1

    def test_list_ensembles_finds_single_ensemble_with_multiple_executions(
        self, artifact_manager: ArtifactManager, temp_dir: Path
    ) -> None:
        """Test list_ensembles finds single ensemble with multiple executions."""
        timestamps = [
            "20240115-103000-123",
            "20240115-104000-456",
            "20240115-105000-789",
        ]

        for timestamp in timestamps:
            artifact_manager.save_execution_results(
                ensemble_name="multi-exec-ensemble",
                results={"timestamp": timestamp},
                timestamp=timestamp,
            )

        ensembles = artifact_manager.list_ensembles()

        assert len(ensembles) == 1
        assert ensembles[0]["name"] == "multi-exec-ensemble"
        assert ensembles[0]["latest_execution"] == "20240115-105000-789"  # Latest
        assert ensembles[0]["executions_count"] == 3

    def test_list_ensembles_finds_multiple_ensembles(
        self, artifact_manager: ArtifactManager, temp_dir: Path
    ) -> None:
        """Test list_ensembles finds multiple ensembles and sorts by name."""
        # Create ensembles in non-alphabetical order to test sorting
        ensembles_data = [
            ("zebra-ensemble", "20240115-103000-123", 1),
            ("alpha-ensemble", "20240115-104000-456", 1),
            ("beta-ensemble", "20240115-105000-789", 1),
        ]

        for name, timestamp, _ in ensembles_data:
            artifact_manager.save_execution_results(
                ensemble_name=name,
                results={"name": name},
                timestamp=timestamp,
            )

        ensembles = artifact_manager.list_ensembles()

        assert len(ensembles) == 3
        # Should be sorted alphabetically
        assert ensembles[0]["name"] == "alpha-ensemble"
        assert ensembles[1]["name"] == "beta-ensemble"
        assert ensembles[2]["name"] == "zebra-ensemble"

    def test_list_ensembles_handles_deeply_nested_structure(
        self, artifact_manager: ArtifactManager, temp_dir: Path
    ) -> None:
        """Test list_ensembles finds ensembles in deeply nested directory structures."""
        # Create ensembles with relative paths (mirrored structure)
        deep_ensembles = [
            ("deep-ensemble-1", "research/ai/nlp/transformers"),
            ("deep-ensemble-2", "creative/writing/poetry"),
            ("shallow-ensemble", "simple"),
        ]

        for name, relative_path in deep_ensembles:
            artifact_manager.save_execution_results(
                ensemble_name=name,
                results={"path": relative_path},
                timestamp="20240115-103000-123",
                relative_path=relative_path,
            )

        ensembles = artifact_manager.list_ensembles()

        assert len(ensembles) == 3
        found_names = [e["name"] for e in ensembles]
        assert "deep-ensemble-1" in found_names
        assert "deep-ensemble-2" in found_names
        assert "shallow-ensemble" in found_names


class TestArtifactManagerScriptArtifacts:
    """Test script-specific artifact features for ADR-001 requirements."""

    def test_save_script_artifact_with_metadata(
        self, artifact_manager: ArtifactManager, temp_dir: Path
    ) -> None:
        """GREEN PHASE: Test script artifact saving with metadata."""
        from llm_orc.schemas.script_agent import ScriptAgentOutput

        # Create script agent output
        script_output = ScriptAgentOutput(
            success=True,
            data={"processed": "test data", "version": "1.0"},
            error=None,
            agent_requests=[],
        )

        # Save script artifact with metadata
        artifact_path = artifact_manager.save_script_artifact(
            agent_name="test_script_agent",
            script_name="data_processor.py",
            script_output=script_output,
            input_hash="abc123",
            metadata={"version": "1.0", "environment": "test"},
        )

        # Verify artifact directory structure
        assert artifact_path.exists()
        assert artifact_path.is_dir()
        assert (artifact_path / "output.json").exists()
        assert (artifact_path / "metadata.json").exists()

        # Verify output.json content
        import json

        with (artifact_path / "output.json").open("r") as f:
            output_data = json.load(f)
        assert output_data["success"] is True
        assert output_data["data"]["processed"] == "test data"

        # Verify metadata.json content
        with (artifact_path / "metadata.json").open("r") as f:
            metadata = json.load(f)
        assert metadata["agent_name"] == "test_script_agent"
        assert metadata["script_name"] == "data_processor.py"
        assert metadata["input_hash"] == "abc123"
        assert metadata["version"] == "1.0"
        assert metadata["environment"] == "test"

    def test_validate_script_output_schema(
        self, artifact_manager: ArtifactManager
    ) -> None:
        """GREEN PHASE: Test script output validation."""
        from llm_orc.schemas.script_agent import ScriptAgentOutput

        # Valid script output
        valid_output = ScriptAgentOutput(
            success=True, data={"result": "success"}, error=None, agent_requests=[]
        )

        # Should validate successfully
        validated = artifact_manager.validate_script_output(valid_output)
        assert validated == valid_output

        # Test validation errors
        with pytest.raises(
            ValueError, match="script_output must be a ScriptAgentOutput instance"
        ):
            artifact_manager.validate_script_output({"not": "a_script_output"})  # type: ignore[arg-type]

    def test_get_script_artifacts(
        self, artifact_manager: ArtifactManager, temp_dir: Path
    ) -> None:
        """GREEN PHASE: Test getting script artifacts."""
        import time

        from llm_orc.schemas.script_agent import ScriptAgentOutput

        # Create and save multiple artifacts
        script_output1 = ScriptAgentOutput(success=True, data={"run": 1})
        script_output2 = ScriptAgentOutput(success=True, data={"run": 2})

        artifact_manager.save_script_artifact(
            agent_name="test_agent",
            script_name="test.py",
            script_output=script_output1,
            input_hash="hash1",
        )

        # Small delay to ensure different timestamps
        time.sleep(0.001)

        artifact_manager.save_script_artifact(
            agent_name="test_agent",
            script_name="test.py",
            script_output=script_output2,
            input_hash="hash2",
        )

        # Get all artifacts for agent
        artifacts = artifact_manager.get_script_artifacts("test_agent")
        assert len(artifacts) == 2
        assert all("timestamp" in artifact for artifact in artifacts)
        assert all("path" in artifact for artifact in artifacts)
        assert all("metadata" in artifact for artifact in artifacts)

        # Test non-existent agent
        empty_artifacts = artifact_manager.get_script_artifacts("non_existent")
        assert empty_artifacts == []

    def test_share_artifacts_between_agents(
        self, artifact_manager: ArtifactManager, temp_dir: Path
    ) -> None:
        """GREEN PHASE: Test artifact sharing between agents."""
        from llm_orc.schemas.script_agent import ScriptAgentOutput

        # Create and save an artifact
        script_output = ScriptAgentOutput(
            success=True, data={"shared_data": "test"}, error=None, agent_requests=[]
        )

        artifact_manager.save_script_artifact(
            agent_name="producer",
            script_name="producer.py",
            script_output=script_output,
            input_hash="shared_hash",
        )

        # Share artifact between agents
        success = artifact_manager.share_artifact(
            source_agent="producer",
            target_agent="consumer",
            artifact_id="shared_hash",
        )
        assert success is True

        # Verify shared artifacts
        shared = artifact_manager.get_shared_artifacts("consumer")
        assert "shared_hash" in shared
        assert shared["shared_hash"]["source_agent"] == "producer"

        # Test sharing non-existent artifact
        no_success = artifact_manager.share_artifact(
            source_agent="producer",
            target_agent="consumer",
            artifact_id="non_existent",
        )
        assert no_success is False

    def test_artifact_cache_performance(
        self, artifact_manager: ArtifactManager, temp_dir: Path
    ) -> None:
        """GREEN PHASE: Test artifact cache performance optimization."""
        from llm_orc.schemas.script_agent import ScriptAgentOutput

        # Create and save an artifact (should be cached)
        script_output = ScriptAgentOutput(success=True, data={"cached": "data"})

        artifact_manager.save_script_artifact(
            agent_name="cached_agent",
            script_name="cache_test.py",
            script_output=script_output,
            input_hash="cache_key",
        )

        # Get cached artifact
        cached = artifact_manager.get_cached_artifact("cached_agent:cache_key")
        assert cached is not None
        assert cached["output"].data["cached"] == "data"
        assert "metadata" in cached
        assert "path" in cached

        # Test non-existent cache key
        not_cached = artifact_manager.get_cached_artifact("non_existent_key")
        assert not_cached is None


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


class TestArtifactManagerEdgeCases:
    """Test edge cases and error paths for ArtifactManager."""

    def test_format_duration_milliseconds(
        self, artifact_manager: ArtifactManager
    ) -> None:
        """Test _format_duration with duration less than 1000ms."""
        # Duration < 1000ms should return in ms format
        result = artifact_manager._format_duration(500)
        assert result == "500ms"

        result = artifact_manager._format_duration(999)
        assert result == "999ms"

    def test_get_latest_results_no_symlink(
        self, artifact_manager: ArtifactManager, temp_dir: Path
    ) -> None:
        """Test get_latest_results when latest symlink doesn't exist."""
        # Create ensemble dir without latest symlink
        ensemble_dir = temp_dir / ".llm-orc" / "artifacts" / "test_ensemble"
        ensemble_dir.mkdir(parents=True)

        result = artifact_manager.get_latest_results("test_ensemble")
        assert result is None

    def test_get_latest_results_no_execution_json(
        self, artifact_manager: ArtifactManager, temp_dir: Path
    ) -> None:
        """Test get_latest_results when execution.json doesn't exist."""
        # Create ensemble dir with symlink but no execution.json
        ensemble_dir = temp_dir / ".llm-orc" / "artifacts" / "test_ensemble"
        ensemble_dir.mkdir(parents=True)

        # Create target directory
        target_dir = ensemble_dir / "20240101-120000-000"
        target_dir.mkdir()

        # Create symlink
        latest_link = ensemble_dir / "latest"
        latest_link.symlink_to(target_dir)

        result = artifact_manager.get_latest_results("test_ensemble")
        assert result is None

    def test_get_latest_results_invalid_json(
        self, artifact_manager: ArtifactManager, temp_dir: Path
    ) -> None:
        """Test get_latest_results with invalid JSON."""
        ensemble_dir = temp_dir / ".llm-orc" / "artifacts" / "test_ensemble"
        ensemble_dir.mkdir(parents=True)

        target_dir = ensemble_dir / "20240101-120000-000"
        target_dir.mkdir()

        # Create invalid JSON file
        execution_json = target_dir / "execution.json"
        execution_json.write_text("not valid json {")

        latest_link = ensemble_dir / "latest"
        latest_link.symlink_to(target_dir)

        result = artifact_manager.get_latest_results("test_ensemble")
        assert result is None

    def test_get_latest_results_non_dict_json(
        self, artifact_manager: ArtifactManager, temp_dir: Path
    ) -> None:
        """Test get_latest_results with JSON that isn't a dict."""
        ensemble_dir = temp_dir / ".llm-orc" / "artifacts" / "test_ensemble"
        ensemble_dir.mkdir(parents=True)

        target_dir = ensemble_dir / "20240101-120000-000"
        target_dir.mkdir()

        # Create JSON that's a list, not a dict
        execution_json = target_dir / "execution.json"
        execution_json.write_text(json.dumps(["not", "a", "dict"]))

        latest_link = ensemble_dir / "latest"
        latest_link.symlink_to(target_dir)

        result = artifact_manager.get_latest_results("test_ensemble")
        assert result is None

    def test_get_execution_results_no_directory(
        self, artifact_manager: ArtifactManager
    ) -> None:
        """Test get_execution_results when execution directory doesn't exist."""
        result = artifact_manager.get_execution_results(
            "nonexistent", "20240101-120000-000"
        )
        assert result is None

    def test_get_execution_results_no_execution_json(
        self, artifact_manager: ArtifactManager, temp_dir: Path
    ) -> None:
        """Test get_execution_results when execution.json doesn't exist."""
        ensemble_dir = temp_dir / ".llm-orc" / "artifacts" / "test_ensemble"
        execution_dir = ensemble_dir / "20240101-120000-000"
        execution_dir.mkdir(parents=True)

        result = artifact_manager.get_execution_results(
            "test_ensemble", "20240101-120000-000"
        )
        assert result is None

    def test_get_execution_results_invalid_json(
        self, artifact_manager: ArtifactManager, temp_dir: Path
    ) -> None:
        """Test get_execution_results with invalid JSON."""
        ensemble_dir = temp_dir / ".llm-orc" / "artifacts" / "test_ensemble"
        execution_dir = ensemble_dir / "20240101-120000-000"
        execution_dir.mkdir(parents=True)

        execution_json = execution_dir / "execution.json"
        execution_json.write_text("invalid json")

        result = artifact_manager.get_execution_results(
            "test_ensemble", "20240101-120000-000"
        )
        assert result is None

    def test_save_script_artifact_empty_names(
        self, artifact_manager: ArtifactManager
    ) -> None:
        """Test save_script_artifact with empty agent_name or script_name."""
        from llm_orc.schemas.script_agent import ScriptAgentOutput

        script_output = ScriptAgentOutput(success=True, data="test")

        with pytest.raises(ValueError, match="agent_name and script_name are required"):
            artifact_manager.save_script_artifact(
                agent_name="",
                script_name="test.py",
                script_output=script_output,
                input_hash="hash1",
            )

        with pytest.raises(ValueError, match="agent_name and script_name are required"):
            artifact_manager.save_script_artifact(
                agent_name="test_agent",
                script_name="",
                script_output=script_output,
                input_hash="hash1",
            )

    def test_save_script_artifact_with_relative_path(
        self, artifact_manager: ArtifactManager, temp_dir: Path
    ) -> None:
        """Test save_script_artifact with relative_path."""
        from llm_orc.schemas.script_agent import ScriptAgentOutput

        script_output = ScriptAgentOutput(success=True, data="test")

        artifact_path = artifact_manager.save_script_artifact(
            agent_name="test_agent",
            script_name="test.py",
            script_output=script_output,
            input_hash="hash1",
            relative_path="project/subdir",
        )

        # Should create in relative path
        assert "project/subdir/test_agent" in str(artifact_path)
        assert artifact_path.exists()

    def test_save_script_artifact_permission_error(
        self, artifact_manager: ArtifactManager
    ) -> None:
        """Test save_script_artifact handles permission errors."""
        from llm_orc.schemas.script_agent import ScriptAgentOutput

        script_output = ScriptAgentOutput(success=True, data="test")

        with patch(
            "pathlib.Path.mkdir", side_effect=PermissionError("Permission denied")
        ):
            with pytest.raises(
                PermissionError,
                match="Permission denied creating script artifact directory",
            ):
                artifact_manager.save_script_artifact(
                    agent_name="test_agent",
                    script_name="test.py",
                    script_output=script_output,
                    input_hash="hash1",
                )

    def test_save_script_artifact_serialization_error(
        self, artifact_manager: ArtifactManager, temp_dir: Path
    ) -> None:
        """Test save_script_artifact handles JSON serialization errors."""
        from llm_orc.schemas.script_agent import ScriptAgentOutput

        script_output = ScriptAgentOutput(success=True, data="test")

        # Mock json.dump to raise TypeError
        with patch("json.dump", side_effect=TypeError("Cannot serialize")):
            with pytest.raises(
                TypeError, match="Script output cannot be serialized to JSON"
            ):
                artifact_manager.save_script_artifact(
                    agent_name="test_agent",
                    script_name="test.py",
                    script_output=script_output,
                    input_hash="hash1",
                )

    def test_validate_script_output_success_none(
        self, artifact_manager: ArtifactManager
    ) -> None:
        """Test validate_script_output when success field is None."""
        from llm_orc.schemas.script_agent import ScriptAgentOutput

        # Create output with success=None using model_construct to bypass validation
        script_output = ScriptAgentOutput.model_construct(
            success=None, data="test", agent_requests=[]
        )

        with pytest.raises(ValueError, match="success field is required"):
            artifact_manager.validate_script_output(script_output)

    def test_validate_script_output_invalid_agent_requests(
        self, artifact_manager: ArtifactManager
    ) -> None:
        """Test validate_script_output with invalid agent_requests."""
        from llm_orc.schemas.script_agent import ScriptAgentOutput

        # Create mock request object without required fields
        class MockRequest:
            pass

        mock_request = MockRequest()

        script_output = ScriptAgentOutput.model_construct(
            success=True, data="test", agent_requests=[mock_request]
        )

        with pytest.raises(ValueError, match="must have target_agent_type"):
            artifact_manager.validate_script_output(script_output)

    def test_get_script_artifacts_with_relative_path(
        self, artifact_manager: ArtifactManager, temp_dir: Path
    ) -> None:
        """Test get_script_artifacts with relative_path."""
        from llm_orc.schemas.script_agent import ScriptAgentOutput

        script_output = ScriptAgentOutput(success=True, data="test")

        # Save artifact with relative path
        artifact_manager.save_script_artifact(
            agent_name="test_agent",
            script_name="test.py",
            script_output=script_output,
            input_hash="hash1",
            relative_path="project/subdir",
        )

        # Retrieve with relative path
        artifacts = artifact_manager.get_script_artifacts(
            "test_agent", relative_path="project/subdir"
        )

        assert len(artifacts) == 1
        assert "project/subdir/test_agent" in str(artifacts[0]["path"])

    def test_get_script_artifacts_invalid_metadata_json(
        self, artifact_manager: ArtifactManager, temp_dir: Path
    ) -> None:
        """Test get_script_artifacts skips artifacts with invalid metadata JSON."""
        from llm_orc.schemas.script_agent import ScriptAgentOutput

        script_output = ScriptAgentOutput(success=True, data="test")

        # Save normal artifact
        artifact_manager.save_script_artifact(
            agent_name="test_agent",
            script_name="test.py",
            script_output=script_output,
            input_hash="hash1",
        )

        # Create a malformed artifact directory
        artifacts_dir = temp_dir / ".llm-orc" / "artifacts" / "scripts" / "test_agent"
        bad_dir = artifacts_dir / "20240101-120000-000"
        bad_dir.mkdir(parents=True, exist_ok=True)

        # Write invalid JSON metadata
        metadata_file = bad_dir / "metadata.json"
        metadata_file.write_text("invalid json {")

        # Should skip the bad artifact and return only the good one
        artifacts = artifact_manager.get_script_artifacts("test_agent")
        assert len(artifacts) == 1

    def test_share_artifact_output_file_not_exists(
        self, artifact_manager: ArtifactManager, temp_dir: Path
    ) -> None:
        """Test share_artifact when output.json doesn't exist."""
        from llm_orc.schemas.script_agent import ScriptAgentOutput

        script_output = ScriptAgentOutput(success=True, data="test")

        # Save artifact
        artifact_manager.save_script_artifact(
            agent_name="source_agent",
            script_name="test.py",
            script_output=script_output,
            input_hash="test_hash",
        )

        # Clear the cache to force filesystem lookup
        artifact_manager._artifact_cache.clear()

        # Delete the output.json file
        artifacts_dir = temp_dir / ".llm-orc" / "artifacts" / "scripts" / "source_agent"
        for item in artifacts_dir.iterdir():
            if item.is_dir():
                output_file = item / "output.json"
                if output_file.exists():
                    output_file.unlink()

        # Should return False when output file doesn't exist
        result = artifact_manager.share_artifact(
            source_agent="source_agent",
            target_agent="target_agent",
            artifact_id="test_hash",
        )
        assert result is False

    def test_share_artifact_invalid_output_json(
        self, artifact_manager: ArtifactManager, temp_dir: Path
    ) -> None:
        """Test share_artifact with invalid output JSON."""
        from llm_orc.schemas.script_agent import ScriptAgentOutput

        script_output = ScriptAgentOutput(success=True, data="test")

        # Save artifact
        artifact_manager.save_script_artifact(
            agent_name="source_agent",
            script_name="test.py",
            script_output=script_output,
            input_hash="test_hash",
        )

        # Clear the cache to force filesystem lookup
        artifact_manager._artifact_cache.clear()

        # Corrupt the output.json file
        artifacts_dir = temp_dir / ".llm-orc" / "artifacts" / "scripts" / "source_agent"
        for item in artifacts_dir.iterdir():
            if item.is_dir():
                output_file = item / "output.json"
                if output_file.exists():
                    output_file.write_text("invalid json")

        # Should return False when JSON is invalid
        result = artifact_manager.share_artifact(
            source_agent="source_agent",
            target_agent="target_agent",
            artifact_id="test_hash",
        )
        assert result is False

    def test_generate_input_hash_with_parameters(
        self, artifact_manager: ArtifactManager
    ) -> None:
        """Test _generate_input_hash with parameters."""
        hash1 = artifact_manager._generate_input_hash(
            "test input", {"param1": "value1", "param2": "value2"}
        )

        # Same input and params should generate same hash
        hash2 = artifact_manager._generate_input_hash(
            "test input", {"param2": "value2", "param1": "value1"}
        )

        assert hash1 == hash2
        assert len(hash1) == 16  # Should be first 16 chars of SHA-256


class TestFanOutArtifacts:
    """Test fan-out execution artifact storage and markdown generation (issue #73)."""

    def test_save_fan_out_execution_results(
        self, artifact_manager: ArtifactManager, temp_dir: Path
    ) -> None:
        """Test that fan-out execution results are saved correctly."""
        results_with_fan_out = {
            "ensemble_name": "fan-out-ensemble",
            "agents": [
                {"name": "chunker", "status": "completed", "result": '["a", "b", "c"]'},
            ],
            "results": {
                "extractor": {
                    "fan_out": True,
                    "status": "partial",
                    "response": ["result_0", "result_2"],
                    "instances": [
                        {"index": 0, "status": "success"},
                        {"index": 1, "status": "failed", "error": "timeout"},
                        {"index": 2, "status": "success"},
                    ],
                },
            },
            "metadata": {
                "fan_out": {
                    "extractor": {
                        "total_instances": 3,
                        "successful_instances": 2,
                        "failed_instances": 1,
                    },
                },
            },
        }

        artifact_manager.save_execution_results(
            ensemble_name="fan-out-ensemble",
            results=results_with_fan_out,
            timestamp="20240115-103000-123",
        )

        # Verify execution.json exists and has fan-out data
        json_file = (
            temp_dir
            / ".llm-orc"
            / "artifacts"
            / "fan-out-ensemble"
            / "20240115-103000-123"
            / "execution.json"
        )

        assert json_file.exists()

        with json_file.open() as f:
            saved_data = json.load(f)

        # Verify fan-out results structure is preserved
        assert "results" in saved_data
        assert "extractor" in saved_data["results"]
        assert saved_data["results"]["extractor"]["fan_out"] is True
        assert saved_data["results"]["extractor"]["status"] == "partial"
        assert len(saved_data["results"]["extractor"]["instances"]) == 3

        # Verify fan-out metadata is preserved
        assert "metadata" in saved_data
        assert "fan_out" in saved_data["metadata"]
        assert saved_data["metadata"]["fan_out"]["extractor"]["total_instances"] == 3

    def test_markdown_includes_fan_out_summary(
        self, artifact_manager: ArtifactManager, temp_dir: Path
    ) -> None:
        """Test that markdown report includes fan-out execution summary."""
        results_with_fan_out = {
            "ensemble_name": "fan-out-ensemble",
            "agents": [
                {"name": "chunker", "status": "completed", "result": "chunks"},
            ],
            "metadata": {
                "fan_out": {
                    "extractor": {
                        "total_instances": 5,
                        "successful_instances": 4,
                        "failed_instances": 1,
                    },
                },
            },
        }

        artifact_manager.save_execution_results(
            ensemble_name="fan-out-ensemble",
            results=results_with_fan_out,
            timestamp="20240115-103000-123",
        )

        md_file = (
            temp_dir
            / ".llm-orc"
            / "artifacts"
            / "fan-out-ensemble"
            / "20240115-103000-123"
            / "execution.md"
        )

        content = md_file.read_text()

        # Should have fan-out section
        assert "Fan-Out" in content
        assert "extractor" in content
        assert "5" in content  # total instances
        assert "4" in content  # successful
        assert "1" in content  # failed

    def test_markdown_shows_fan_out_agent_instances(
        self, artifact_manager: ArtifactManager, temp_dir: Path
    ) -> None:
        """Test that markdown shows individual fan-out instance statuses."""
        results_with_fan_out = {
            "ensemble_name": "fan-out-ensemble",
            "results": {
                "extractor": {
                    "fan_out": True,
                    "status": "partial",
                    "response": ["r0", "r2"],
                    "instances": [
                        {"index": 0, "status": "success"},
                        {"index": 1, "status": "failed", "error": "Connection timeout"},
                        {"index": 2, "status": "success"},
                    ],
                },
            },
        }

        artifact_manager.save_execution_results(
            ensemble_name="fan-out-ensemble",
            results=results_with_fan_out,
            timestamp="20240115-103000-123",
        )

        md_file = (
            temp_dir
            / ".llm-orc"
            / "artifacts"
            / "fan-out-ensemble"
            / "20240115-103000-123"
            / "execution.md"
        )

        content = md_file.read_text()

        # Should show instance info
        assert "extractor" in content
        assert "partial" in content.lower() or "Partial" in content
        assert "Connection timeout" in content

    def test_markdown_handles_all_instances_success(
        self, artifact_manager: ArtifactManager, temp_dir: Path
    ) -> None:
        """Test markdown when all fan-out instances succeed."""
        results_with_fan_out = {
            "ensemble_name": "fan-out-ensemble",
            "results": {
                "processor": {
                    "fan_out": True,
                    "status": "success",
                    "response": ["r0", "r1", "r2"],
                    "instances": [
                        {"index": 0, "status": "success"},
                        {"index": 1, "status": "success"},
                        {"index": 2, "status": "success"},
                    ],
                },
            },
        }

        artifact_manager.save_execution_results(
            ensemble_name="fan-out-ensemble",
            results=results_with_fan_out,
            timestamp="20240115-103000-123",
        )

        md_file = (
            temp_dir
            / ".llm-orc"
            / "artifacts"
            / "fan-out-ensemble"
            / "20240115-103000-123"
            / "execution.md"
        )

        content = md_file.read_text()

        # Should indicate success
        assert "processor" in content
        assert "3/3" in content or "3 of 3" in content or "success" in content.lower()
