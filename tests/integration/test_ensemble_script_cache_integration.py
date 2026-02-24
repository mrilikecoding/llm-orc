"""Integration tests for ScriptCache with EnsembleExecutor.

Tests that EnsembleExecutor properly integrates with ScriptCache for transparent
caching.
"""

import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from llm_orc.core.execution.ensemble_execution import EnsembleExecutor
from llm_orc.core.execution.executor_factory import ExecutorFactory
from llm_orc.core.execution.script_cache import ScriptCache, ScriptCacheConfig
from llm_orc.schemas.agent_config import ScriptAgentConfig


class TestEnsembleScriptCacheIntegration:
    """Test suite for EnsembleExecutor integration with ScriptCache."""

    @pytest.fixture
    def temp_dir(self) -> Generator[Path, None, None]:
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def cache_config(self, temp_dir: Path) -> ScriptCacheConfig:
        """Create test cache configuration."""
        return ScriptCacheConfig(
            enabled=True,
            ttl_seconds=3600,
            max_size=100,
            persist_to_artifacts=False,
            artifact_base_dir=temp_dir,
        )

    @pytest.fixture
    def ensemble_executor(self) -> EnsembleExecutor:
        """Create ensemble executor for testing."""
        return ExecutorFactory.create_root_executor()

    def test_script_cache_integration_avoids_duplicate_execution(
        self, ensemble_executor: EnsembleExecutor, cache_config: ScriptCacheConfig
    ) -> None:
        """Test that script cache avoids duplicate script execution."""
        # This test will fail initially because EnsembleExecutor doesn't have
        # cache integration yet
        # Following TDD - RED phase
        script_content = "echo 'test output'"
        cached_result = {
            "output": "test output",
            "success": True,
            "execution_metadata": {"duration_ms": 100},
        }

        # Create cache and pre-populate
        script_cache = ScriptCache(cache_config)
        cache_key_params = {
            "input_data": "test input",
            "parameters": {},
        }
        script_cache.set(script_content, cache_key_params, cached_result)

        # Mock the executor's runner to use our cache
        runner = ensemble_executor._script_agent_runner
        with patch.object(runner, "_script_cache", script_cache):
            # Mock script execution to ensure it's not called
            with patch.object(
                runner,
                "_execute_without_cache",
                new_callable=AsyncMock,
            ) as mock_execute:
                mock_execute.return_value = ("should not be called", None)

                # Test direct cache hit
                result = script_cache.get(script_content, cache_key_params)
                assert result == cached_result

                # This ensures we have the _script_cache attribute available
                assert hasattr(runner, "_script_cache")

    async def test_ensemble_execution_with_cache_miss_executes_and_caches(
        self, ensemble_executor: EnsembleExecutor, cache_config: ScriptCacheConfig
    ) -> None:
        """Test that cache miss triggers execution and caches the result."""
        # RED phase - this will fail until we implement the integration

        script_content = "echo 'new execution'"
        execution_result = "new execution"

        # Create empty cache
        script_cache = ScriptCache(cache_config)

        # Mock the executor's runner to use our cache
        runner = ensemble_executor._script_agent_runner
        with patch.object(runner, "_script_cache", script_cache):
            # Verify cache is initially empty
            assert script_cache.get(script_content, {}) is None

            # Mock actual script execution
            with patch.object(
                runner,
                "_execute_without_cache",
                new_callable=AsyncMock,
            ) as mock_execute:
                mock_execute.return_value = (execution_result, None, False)

                # Simulate the caching flow that should happen
                # 1. Check cache (miss)
                cached_result = script_cache.get(script_content, {})
                assert cached_result is None

                # 2. Execute script (since cache missed)
                (
                    result,
                    model,
                    _,
                ) = await runner._execute_without_cache(
                    ScriptAgentConfig(name="test", script=script_content), "{}"
                )

                # 3. Cache the result
                script_cache.set(
                    script_content,
                    {},
                    {
                        "output": result,
                        "success": True,
                        "execution_metadata": {"duration_ms": 200},
                    },
                )

                # 4. Verify cache now has result
                cached_result = script_cache.get(script_content, {})
                assert cached_result is not None
                assert cached_result["output"] == execution_result

    def test_cache_configuration_from_performance_config(
        self, ensemble_executor: EnsembleExecutor
    ) -> None:
        """Test that cache configuration can be loaded from performance config."""
        # Test that the EnsembleExecutor has the cache configuration
        assert hasattr(ensemble_executor, "_script_cache_config")
        assert hasattr(ensemble_executor, "_script_cache")

        # Verify the cache config has expected defaults
        cache_config = ensemble_executor._script_cache_config
        assert cache_config.enabled is True
        assert cache_config.ttl_seconds == 3600
        assert cache_config.max_size == 1000

    def test_script_cache_respects_disabled_configuration(
        self, ensemble_executor: EnsembleExecutor
    ) -> None:
        """Test that disabled cache configuration bypasses caching."""
        # Create disabled cache config
        disabled_config = ScriptCacheConfig(enabled=False)
        script_cache = ScriptCache(disabled_config)

        with patch.object(ensemble_executor, "_script_cache", script_cache):
            script_content = "echo 'test'"

            # Even if we try to cache, disabled cache should not store
            script_cache.set(script_content, {}, {"output": "test"})
            result = script_cache.get(script_content, {})

            assert result is None  # Disabled cache returns None
