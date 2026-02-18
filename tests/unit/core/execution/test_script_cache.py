"""Unit tests for ScriptCache system.

Tests for script result caching functionality that supports reproducible research
as outlined in ADR-001 architecture review.
"""

import tempfile
import time
from pathlib import Path
from unittest.mock import patch

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

        # Simulate TTL expiration by advancing time
        with patch("llm_orc.core.execution.script_cache.time") as mock_time:
            mock_time.time.return_value = time.time() + 2
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


class TestScriptCacheArtifactPersistence:
    """Test suite for artifact persistence edge cases."""

    def test_load_from_artifacts_no_artifact_manager(self) -> None:
        """Test _load_from_artifacts returns None with no manager (line 196)."""
        config = ScriptCacheConfig(persist_to_artifacts=False)
        cache = ScriptCache(config)

        result = cache._load_from_artifacts("some_key")

        assert result is None

    def test_load_from_artifacts_file_not_exists(self) -> None:
        """Test _load_from_artifacts returns None when file missing (line 204)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ScriptCacheConfig(
                persist_to_artifacts=True,
                artifact_base_dir=Path(temp_dir),
            )
            cache = ScriptCache(config)

            result = cache._load_from_artifacts("nonexistent_key")

            assert result is None

    def test_load_from_artifacts_ttl_expired(self) -> None:
        """Test _load_from_artifacts handles TTL expiration (lines 211-212)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ScriptCacheConfig(
                persist_to_artifacts=True,
                artifact_base_dir=Path(temp_dir),
                ttl_seconds=1,  # 1 second TTL
            )
            cache = ScriptCache(config)

            # Save a cache entry
            cache._save_to_artifacts("test_key", {"data": "test"})

            # Simulate TTL expiration by advancing time
            with patch("llm_orc.core.execution.script_cache.time") as mock_time:
                mock_time.time.return_value = time.time() + 2
                # Should return None and delete the file
                result = cache._load_from_artifacts("test_key")

            assert result is None
            # Verify file was deleted
            cache_file = Path(temp_dir) / ".llm-orc" / "cache" / "test_key.json"
            assert not cache_file.exists()

    def test_load_from_artifacts_invalid_json(self) -> None:
        """Test _load_from_artifacts handles JSONDecodeError (lines 216-217)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ScriptCacheConfig(
                persist_to_artifacts=True,
                artifact_base_dir=Path(temp_dir),
            )
            cache = ScriptCache(config)

            # Create cache file with invalid JSON
            cache_dir = Path(temp_dir) / ".llm-orc" / "cache"
            cache_dir.mkdir(parents=True)
            cache_file = cache_dir / "bad_key.json"
            cache_file.write_text("not valid json {")

            result = cache._load_from_artifacts("bad_key")

            assert result is None

    def test_load_from_artifacts_missing_keys(self) -> None:
        """Test _load_from_artifacts handles KeyError (lines 216-217)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ScriptCacheConfig(
                persist_to_artifacts=True,
                artifact_base_dir=Path(temp_dir),
            )
            cache = ScriptCache(config)

            # Create cache file with missing required keys
            cache_dir = Path(temp_dir) / ".llm-orc" / "cache"
            cache_dir.mkdir(parents=True)
            cache_file = cache_dir / "incomplete_key.json"
            cache_file.write_text('{"incomplete": "data"}')

            result = cache._load_from_artifacts("incomplete_key")

            assert result is None

    def test_save_to_artifacts_no_artifact_manager(self) -> None:
        """Test _save_to_artifacts returns early when no artifact manager (line 227)."""
        config = ScriptCacheConfig(persist_to_artifacts=False)
        cache = ScriptCache(config)

        # Should not raise, just return early
        cache._save_to_artifacts("test_key", {"data": "test"})

    def test_save_to_artifacts_os_error(self) -> None:
        """Test _save_to_artifacts handles OSError silently (lines 245-247)."""
        config = ScriptCacheConfig(
            persist_to_artifacts=True,
            artifact_base_dir=Path("/invalid/read-only/path"),
        )
        cache = ScriptCache(config)

        # Should not raise, silently ignore error
        cache._save_to_artifacts("test_key", {"data": "test"})

    def test_save_to_artifacts_type_error(self) -> None:
        """Test _save_to_artifacts handles non-serializable data (lines 245-247)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ScriptCacheConfig(
                persist_to_artifacts=True,
                artifact_base_dir=Path(temp_dir),
            )
            cache = ScriptCache(config)

            # Try to save non-JSON-serializable data
            non_serializable = {
                "data": lambda: None
            }  # Functions can't be JSON serialized

            # Should not raise, silently ignore error
            cache._save_to_artifacts("test_key", non_serializable)

    def test_clear_artifacts_no_artifact_manager(self) -> None:
        """Test _clear_artifacts returns early when no artifact manager (line 251)."""
        config = ScriptCacheConfig(persist_to_artifacts=False)
        cache = ScriptCache(config)

        # Should not raise, just return early
        cache._clear_artifacts()

    def test_clear_artifacts_removes_cache_files(self) -> None:
        """Test _clear_artifacts removes all cache files (lines 254-258)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ScriptCacheConfig(
                persist_to_artifacts=True,
                artifact_base_dir=Path(temp_dir),
            )
            cache = ScriptCache(config)

            # Create multiple cache files
            cache._save_to_artifacts("key1", {"data": "test1"})
            cache._save_to_artifacts("key2", {"data": "test2"})

            cache_dir = Path(temp_dir) / ".llm-orc" / "cache"
            assert len(list(cache_dir.glob("*.json"))) == 2

            # Clear artifacts
            cache._clear_artifacts()

            # All cache files should be removed
            assert len(list(cache_dir.glob("*.json"))) == 0

    def test_clear_artifacts_handles_os_error(self) -> None:
        """Test _clear_artifacts handles OSError silently (lines 259-261)."""
        config = ScriptCacheConfig(
            persist_to_artifacts=True,
            artifact_base_dir=Path("/invalid/path"),
        )
        cache = ScriptCache(config)

        # Should not raise, silently ignore error
        cache._clear_artifacts()

    def test_clear_cache_clears_artifacts(self) -> None:
        """Test clear() calls _clear_artifacts with manager (line 168)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ScriptCacheConfig(
                persist_to_artifacts=True,
                artifact_base_dir=Path(temp_dir),
            )
            cache = ScriptCache(config)

            # Create cache entry
            cache._save_to_artifacts("test_key", {"data": "test"})

            cache_dir = Path(temp_dir) / ".llm-orc" / "cache"
            assert len(list(cache_dir.glob("*.json"))) == 1

            # Clear cache should clear artifacts
            cache.clear()

            # Artifact should be removed
            assert len(list(cache_dir.glob("*.json"))) == 0
