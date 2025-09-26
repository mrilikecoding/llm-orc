"""Unit tests for ScriptCache system.

Tests for script result caching functionality that supports reproducible research
as outlined in ADR-001 architecture review.
"""

import tempfile
import time
from pathlib import Path

from llm_orc.core.execution.script_cache import ScriptCache, ScriptCacheConfig


class TestScriptCache:
    """Test suite for ScriptCache functionality."""

    def test_cache_key_generation_from_script_content_and_parameters(self) -> None:
        """Test cache key generation based on script content hash + parameters hash.

        RED PHASE: This test should fail because ScriptCache doesn't exist yet.
        """
        # Arrange
        config = ScriptCacheConfig(enabled=True, ttl_seconds=3600, max_size=100)
        cache = ScriptCache(config)

        script_content = "print('hello')"
        parameters = {"param1": "value1", "param2": 42}

        # Act
        key1 = cache._generate_cache_key(script_content, parameters)
        key2 = cache._generate_cache_key(script_content, parameters)
        key3 = cache._generate_cache_key("different script", parameters)

        # Assert
        assert key1 == key2  # Same input should generate same key
        assert key1 != key3  # Different script should generate different key
        assert isinstance(key1, str)
        assert len(key1) == 64  # SHA256 hex digest length

    def test_cache_miss_returns_none(self) -> None:
        """Test cache miss behavior."""
        # Arrange
        config = ScriptCacheConfig(enabled=True, ttl_seconds=3600, max_size=100)
        cache = ScriptCache(config)

        # Act
        result = cache.get("script_content", {"param": "value"})

        # Assert
        assert result is None

    def test_cache_hit_returns_stored_result(self) -> None:
        """Test cache hit behavior with stored result."""
        # Arrange
        config = ScriptCacheConfig(enabled=True, ttl_seconds=3600, max_size=100)
        cache = ScriptCache(config)

        script_content = "print('test')"
        parameters = {"test": True}
        expected_result = {
            "output": "test result",
            "execution_metadata": {"duration_ms": 500},
            "artifacts": [],
        }

        # Act
        cache.set(script_content, parameters, expected_result)
        result = cache.get(script_content, parameters)

        # Assert
        assert result == expected_result

    def test_cache_invalidation_on_script_content_change(self) -> None:
        """Test cache invalidation when script content changes."""
        # Arrange
        config = ScriptCacheConfig(enabled=True, ttl_seconds=3600, max_size=100)
        cache = ScriptCache(config)

        original_script = "print('original')"
        modified_script = "print('modified')"
        parameters = {"test": True}
        result_data = {"output": "test"}

        # Act
        cache.set(original_script, parameters, result_data)
        original_result = cache.get(original_script, parameters)
        modified_result = cache.get(modified_script, parameters)

        # Assert
        assert original_result == result_data
        assert modified_result is None  # Should be cache miss for modified script

    def test_ttl_expiration_removes_cached_results(self) -> None:
        """Test TTL-based cache expiration."""
        # Arrange
        config = ScriptCacheConfig(enabled=True, ttl_seconds=1, max_size=100)
        cache = ScriptCache(config)

        script_content = "print('test')"
        parameters = {"test": True}
        result_data = {"output": "test"}

        # Act
        cache.set(script_content, parameters, result_data)
        immediate_result = cache.get(script_content, parameters)

        # Wait for TTL expiration
        time.sleep(1.1)
        expired_result = cache.get(script_content, parameters)

        # Assert
        assert immediate_result == result_data
        assert expired_result is None

    def test_cache_size_limit_evicts_oldest_entries(self) -> None:
        """Test cache size limit with LRU eviction."""
        # Arrange
        config = ScriptCacheConfig(enabled=True, ttl_seconds=3600, max_size=2)
        cache = ScriptCache(config)

        # Act - Fill cache beyond capacity
        cache.set("script1", {}, {"result": "1"})
        cache.set("script2", {}, {"result": "2"})
        cache.set("script3", {}, {"result": "3"})  # Should evict script1

        # Assert
        assert cache.get("script1", {}) is None  # Evicted
        assert cache.get("script2", {}) == {"result": "2"}  # Still there
        assert cache.get("script3", {}) == {"result": "3"}  # Still there

    def test_disabled_cache_always_returns_none(self) -> None:
        """Test that disabled cache doesn't store or return results."""
        # Arrange
        config = ScriptCacheConfig(enabled=False, ttl_seconds=3600, max_size=100)
        cache = ScriptCache(config)

        # Act
        cache.set("script", {}, {"result": "test"})
        result = cache.get("script", {})

        # Assert
        assert result is None

    def test_cache_clear_removes_all_entries(self) -> None:
        """Test cache clear functionality."""
        # Arrange
        config = ScriptCacheConfig(enabled=True, ttl_seconds=3600, max_size=100)
        cache = ScriptCache(config)

        # Act
        cache.set("script1", {}, {"result": "1"})
        cache.set("script2", {}, {"result": "2"})

        cache.clear()

        # Assert
        assert cache.get("script1", {}) is None
        assert cache.get("script2", {}) is None

    def test_cache_stats_tracking(self) -> None:
        """Test cache statistics tracking."""
        # Arrange
        config = ScriptCacheConfig(enabled=True, ttl_seconds=3600, max_size=100)
        cache = ScriptCache(config)

        # Act
        cache.get("script1", {})  # Miss
        cache.set("script1", {}, {"result": "1"})
        cache.get("script1", {})  # Hit
        cache.get("script2", {})  # Miss

        stats = cache.get_stats()

        # Assert
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["sets"] == 1
        assert stats["hit_rate"] == 1 / 3

    def test_artifact_manager_integration(self) -> None:
        """Test integration with ArtifactManager for persistent caching."""
        # Arrange
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ScriptCacheConfig(
                enabled=True,
                ttl_seconds=3600,
                max_size=100,
                persist_to_artifacts=True,
                artifact_base_dir=Path(temp_dir),
            )
            cache = ScriptCache(config)

            script_content = "print('test')"
            parameters = {"test": True}
            result_data = {
                "output": "test result",
                "execution_metadata": {"duration_ms": 500},
            }

            # Act
            cache.set(script_content, parameters, result_data)

            # Create new cache instance to test persistence
            new_cache = ScriptCache(config)
            persisted_result = new_cache.get(script_content, parameters)

            # Assert
            assert persisted_result == result_data


class TestScriptCacheConfig:
    """Test suite for ScriptCacheConfig."""

    def test_default_configuration_values(self) -> None:
        """Test default configuration values."""
        # Act
        config = ScriptCacheConfig()

        # Assert
        assert config.enabled is True
        assert config.ttl_seconds == 3600
        assert config.max_size == 1000
        assert config.persist_to_artifacts is False
        assert config.artifact_base_dir == Path(".")

    def test_custom_configuration_values(self) -> None:
        """Test custom configuration values."""
        # Act
        config = ScriptCacheConfig(
            enabled=False,
            ttl_seconds=1800,
            max_size=500,
            persist_to_artifacts=True,
            artifact_base_dir=Path("/tmp/cache"),
        )

        # Assert
        assert config.enabled is False
        assert config.ttl_seconds == 1800
        assert config.max_size == 500
        assert config.persist_to_artifacts is True
        assert config.artifact_base_dir == Path("/tmp/cache")
