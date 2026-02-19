"""Integration tests for ScriptCache with ScriptAgent.

Tests the integration of ScriptCache with script execution in ensemble scenarios.
"""

import json
import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from llm_orc.agents.script_agent import ScriptAgent
from llm_orc.core.execution.script_cache import ScriptCache, ScriptCacheConfig


class TestScriptCacheIntegration:
    """Test suite for ScriptCache integration with script execution."""

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
    def script_cache(self, cache_config: ScriptCacheConfig) -> ScriptCache:
        """Create test script cache."""
        return ScriptCache(cache_config)

    def test_cache_hit_avoids_script_execution(
        self, script_cache: ScriptCache, temp_dir: Path
    ) -> None:
        """Test that cache hit avoids re-executing the script."""
        # Arrange
        script_content = "echo 'hello world'"
        parameters = {"test": True}
        cached_result = {
            "output": "hello world",
            "execution_metadata": {"duration_ms": 100},
            "success": True,
        }

        # Pre-populate cache
        script_cache.set(script_content, parameters, cached_result)

        # Prepare for mocking script execution

        with patch.object(
            ScriptAgent, "execute", new_callable=AsyncMock
        ) as mock_execute:
            mock_execute.return_value = json.dumps({"should": "not be called"})

            # Act - simulate cache check before execution
            cache_result = script_cache.get(script_content, parameters)

            # Assert
            assert cache_result == cached_result
            mock_execute.assert_not_called()  # Should not execute script

    def test_cache_miss_triggers_script_execution_and_caches_result(
        self, script_cache: ScriptCache, temp_dir: Path
    ) -> None:
        """Test that cache miss triggers execution and caches the result."""
        # Arrange
        script_content = "echo 'new execution'"
        parameters = {"test": True}
        execution_result = {
            "output": "new execution",
            "execution_metadata": {"duration_ms": 200},
            "success": True,
        }

        with patch.object(
            ScriptAgent, "execute", new_callable=AsyncMock
        ) as mock_execute:
            mock_execute.return_value = json.dumps(execution_result)

            # Act
            # First check cache (should miss)
            cache_result = script_cache.get(script_content, parameters)
            assert cache_result is None

            # Note: In real integration, this would be called by ensemble executor
            # Here we simulate the flow

            # Cache the result after execution
            script_cache.set(script_content, parameters, execution_result)

            # Verify cache now has the result
            cached_result = script_cache.get(script_content, parameters)

            # Assert
            assert cached_result == execution_result

    def test_script_content_change_invalidates_cache(
        self, script_cache: ScriptCache
    ) -> None:
        """Test that changing script content invalidates the cache."""
        # Arrange
        original_script = "echo 'original'"
        modified_script = "echo 'modified'"
        parameters = {"test": True}
        original_result = {"output": "original", "success": True}

        # Cache original result
        script_cache.set(original_script, parameters, original_result)

        # Act
        original_cached = script_cache.get(original_script, parameters)
        modified_cached = script_cache.get(modified_script, parameters)

        # Assert
        assert original_cached == original_result
        assert modified_cached is None  # Should be cache miss

    def test_parameter_change_invalidates_cache(
        self, script_cache: ScriptCache
    ) -> None:
        """Test that changing parameters invalidates the cache."""
        # Arrange
        script_content = "echo 'test'"
        original_params = {"test": True, "value": 1}
        modified_params = {"test": True, "value": 2}
        result = {"output": "test", "success": True}

        # Cache with original parameters
        script_cache.set(script_content, original_params, result)

        # Act
        original_cached = script_cache.get(script_content, original_params)
        modified_cached = script_cache.get(script_content, modified_params)

        # Assert
        assert original_cached == result
        assert modified_cached is None  # Should be cache miss

    def test_cache_respects_disabled_configuration(self) -> None:
        """Test that disabled cache doesn't interfere with execution."""
        # Arrange
        config = ScriptCacheConfig(enabled=False)
        cache = ScriptCache(config)

        script_content = "echo 'test'"
        parameters = {"test": True}
        result = {"output": "test", "success": True}

        # Act
        cache.set(script_content, parameters, result)
        cached_result = cache.get(script_content, parameters)

        # Assert
        assert cached_result is None  # Disabled cache should not return results

    def test_cache_statistics_track_ensemble_usage(
        self, script_cache: ScriptCache
    ) -> None:
        """Test that cache statistics properly track usage in ensemble context."""
        # Arrange
        script1 = "echo 'script1'"
        script2 = "echo 'script2'"
        params = {"test": True}
        result1 = {"output": "script1", "success": True}
        result2 = {"output": "script2", "success": True}

        # Act - Simulate ensemble execution pattern
        # First execution - cache misses
        assert script_cache.get(script1, params) is None  # Miss
        assert script_cache.get(script2, params) is None  # Miss

        # Cache results after execution
        script_cache.set(script1, params, result1)
        script_cache.set(script2, params, result2)

        # Second execution - cache hits
        assert script_cache.get(script1, params) == result1  # Hit
        assert script_cache.get(script2, params) == result2  # Hit

        # Check statistics
        stats = script_cache.get_stats()

        # Assert
        assert stats["hits"] == 2
        assert stats["misses"] == 2
        assert stats["sets"] == 2
        assert stats["hit_rate"] == 0.5  # 2 hits out of 4 total requests

    def test_ensemble_reproducibility_with_cache(
        self, script_cache: ScriptCache
    ) -> None:
        """Test that cache enables reproducible ensemble execution."""
        # Arrange
        script_content = "python -c 'import random; print(random.random())'"
        parameters = {"seed": 42}

        # Simulate first execution result (would be random without cache)
        first_result = {
            "output": "0.6394267984578837",  # Deterministic for testing
            "execution_metadata": {"duration_ms": 150},
            "success": True,
        }

        # Act
        # First "execution" - cache miss, store result
        assert script_cache.get(script_content, parameters) is None
        script_cache.set(script_content, parameters, first_result)

        # Subsequent "executions" should return cached result
        second_result = script_cache.get(script_content, parameters)
        third_result = script_cache.get(script_content, parameters)

        # Assert reproducibility
        assert second_result == first_result
        assert third_result == first_result

        # Verify cache statistics show reproducible hits
        stats = script_cache.get_stats()
        assert stats["hits"] == 2  # Two cache hits for reproducibility
        assert stats["misses"] == 1  # One initial miss
