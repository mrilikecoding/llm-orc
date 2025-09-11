"""Shared test fixtures and configuration."""

import shutil
from collections.abc import Generator
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from llm_orc.core.execution.artifact_manager import ArtifactManager
from llm_orc.core.execution.ensemble_execution import EnsembleExecutor


@pytest.fixture(autouse=True)
def cleanup_test_artifacts() -> Generator[None, None, None]:
    """Automatically clean up test artifacts after each test.

    This fixture runs automatically for all tests and cleans up any artifacts
    created in .llm-orc/artifacts/ during test execution.
    """
    # Store initial top-level artifact directories before test
    artifacts_path = Path(".llm-orc/artifacts")
    initial_dirs = set()
    if artifacts_path.exists():
        initial_dirs = {d.name for d in artifacts_path.iterdir() if d.is_dir()}

    # Run the test
    yield

    # Clean up any new top-level artifact directories created during test
    if artifacts_path.exists():
        current_dirs = {d.name for d in artifacts_path.iterdir() if d.is_dir()}
        new_dirs = current_dirs - initial_dirs

        # Also check for modified directories (new timestamped subdirs)
        for dir_name in current_dirs:
            dir_path = artifacts_path / dir_name
            if dir_path.is_dir():
                # Check if this directory has new timestamped subdirectories
                if dir_name in initial_dirs:
                    # Check for new subdirectories created during test
                    subdirs = list(dir_path.iterdir())
                    # If it has new content, consider it modified
                    if any(
                        subdir.is_dir() and "202" in subdir.name for subdir in subdirs
                    ):
                        new_dirs.add(dir_name)

        # Clean up all new or modified directories
        for dir_name in new_dirs:
            dir_path = artifacts_path / dir_name
            if dir_path.exists():
                shutil.rmtree(dir_path, ignore_errors=True)


@pytest.fixture
def mock_ensemble_executor() -> Generator[EnsembleExecutor, None, None]:
    """Create an EnsembleExecutor with mocked ArtifactManager.

    This fixture ensures that tests don't create real artifacts in .llm-orc/artifacts/
    by mocking the ArtifactManager component.
    """
    executor = EnsembleExecutor()

    # Mock the ArtifactManager to prevent real artifact creation
    mock_artifact_manager = Mock(spec=ArtifactManager)
    mock_artifact_manager.save_execution_results = Mock()

    # Replace the real artifact manager with the mock
    with patch.object(executor, "_artifact_manager", mock_artifact_manager):
        yield executor
