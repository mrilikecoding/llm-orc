"""Test that verifies no artifacts are created during test runs."""

from pathlib import Path
from typing import Any

import pytest

from llm_orc.core.config.ensemble_config import EnsembleConfig
from llm_orc.schemas.agent_config import ScriptAgentConfig

_project_root = Path(__file__).resolve().parent.parent


class TestNoArtifactCreation:
    """Test that no artifacts are created during test runs."""

    def test_no_artifacts_created_during_test_instantiation(self) -> None:
        """Test that instantiating EnsembleExecutor doesn't create artifacts."""
        artifacts_dir = _project_root / ".llm-orc" / "artifacts"

        # Get initial state
        initial_dirs = set()
        if artifacts_dir.exists():
            initial_dirs = {d.name for d in artifacts_dir.iterdir() if d.is_dir()}

        # Create a simple EnsembleExecutor to simulate what tests do
        from llm_orc.core.execution.executor_factory import ExecutorFactory

        # This should NOT create artifacts during instantiation
        _ = ExecutorFactory.create_root_executor()

        # Check that no new directories were created from instantiation
        current_dirs = set()
        if artifacts_dir.exists():
            current_dirs = {d.name for d in artifacts_dir.iterdir() if d.is_dir()}

        new_dirs = current_dirs - initial_dirs

        # Instantiation should not create artifacts
        assert len(new_dirs) == 0, (
            f"New artifact directories created during instantiation: {new_dirs}"
        )

    @pytest.mark.asyncio
    async def test_mock_ensemble_executor_prevents_artifacts(
        self, mock_ensemble_executor: Any
    ) -> None:
        """Test that using mock_ensemble_executor fixture prevents artifact creation."""
        artifacts_dir = _project_root / ".llm-orc" / "artifacts"

        # Get initial state
        initial_dirs = set()
        if artifacts_dir.exists():
            initial_dirs = {d.name for d in artifacts_dir.iterdir() if d.is_dir()}

        config = EnsembleConfig(
            name="test_no_artifacts_with_mock",
            description="Test that this doesn't create artifacts with mock",
            agents=[
                ScriptAgentConfig(name="agent1", script="echo 'test'"),
            ],
        )

        # Use the mock fixture
        executor = mock_ensemble_executor

        # Mock some additional methods to avoid real execution
        from unittest.mock import AsyncMock, patch

        with (
            patch.object(
                executor._llm_agent_runner,
                "_load_role_from_config",
                new_callable=AsyncMock,
            ),
            patch.object(
                executor._script_agent_runner, "execute", new_callable=AsyncMock
            ) as mock_script,
        ):
            mock_script.return_value = ("Test output", None)

            # Execute the ensemble - this should NOT create artifacts
            result = await executor.execute(config, "test input")

        # Check that no new directories were created
        current_dirs = set()
        if artifacts_dir.exists():
            current_dirs = {d.name for d in artifacts_dir.iterdir() if d.is_dir()}

        new_dirs = current_dirs - initial_dirs

        # Using mock should not create artifacts
        assert len(new_dirs) == 0, (
            f"New artifact directories created with mock: {new_dirs}"
        )

        # Verify execution completed (proving the mock works)
        assert result is not None
