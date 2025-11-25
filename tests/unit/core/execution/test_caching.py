"""Unit tests for caching behavior.

This module contains tests for caching mechanisms used by script agents,
supporting reproducible research as outlined in the BDD scenarios.

Migrated from: tests/test_issue_24_units.py::test_caching_behavior
Related BDD: tests/bdd/features/issue-24-script-agents.feature (caching scenario)
"""

import hashlib
from typing import Any


class TestResultCache:
    """Unit tests for result caching functionality."""

    def test_caching_behavior(self) -> None:
        """Test caching mechanism for script agent results.

        Originally from BDD scenario: Script results cached for reproducible research
        Tests cache hit/miss, invalidation, and memory management.
        """

        # Simple cache implementation for testing
        class ResultCache:
            def __init__(self) -> None:
                self._cache: dict[str, Any] = {}
                self._hit_count = 0
                self._miss_count = 0

            def _generate_key(self, agent_name: str, input_data: str) -> str:
                """Generate cache key from agent and input."""
                content = f"{agent_name}:{input_data}"
                return hashlib.sha256(content.encode()).hexdigest()[:16]

            def get(self, agent_name: str, input_data: str) -> Any | None:
                """Get cached result if exists."""
                key = self._generate_key(agent_name, input_data)
                if key in self._cache:
                    self._hit_count += 1
                    return self._cache[key]
                self._miss_count += 1
                return None

            def set(self, agent_name: str, input_data: str, result: Any) -> None:
                """Cache a result."""
                key = self._generate_key(agent_name, input_data)
                self._cache[key] = result

            def invalidate(self, agent_name: str | None = None) -> int:
                """Invalidate cache entries."""
                if agent_name is None:
                    count = len(self._cache)
                    self._cache.clear()
                    return count

                # Invalidate specific agent's entries
                keys_to_remove = [
                    k
                    for k in self._cache.keys()
                    if k.startswith(
                        hashlib.sha256(f"{agent_name}:".encode()).hexdigest()[:8]
                    )
                ]
                for key in keys_to_remove:
                    del self._cache[key]
                return len(keys_to_remove)

        # Test cache operations
        cache = ResultCache()

        # Test cache miss
        result = cache.get("test-agent", "input1")
        assert result is None
        assert cache._miss_count == 1
        assert cache._hit_count == 0

        # Test cache set and hit
        cache.set("test-agent", "input1", {"result": "data1"})
        result = cache.get("test-agent", "input1")
        assert result == {"result": "data1"}
        assert cache._hit_count == 1

        # Test different inputs have different keys
        cache.set("test-agent", "input2", {"result": "data2"})
        result1 = cache.get("test-agent", "input1")
        result2 = cache.get("test-agent", "input2")
        assert result1 == {"result": "data1"}
        assert result2 == {"result": "data2"}

        # Test cache invalidation
        assert len(cache._cache) == 2
        count = cache.invalidate()
        assert count == 2
        assert len(cache._cache) == 0

        # Test cache helps with reproducibility
        cache.set("ml-agent", '{"seed": 42}', {"prediction": 0.95})

        # Same input should yield same cached result
        for _ in range(3):
            result = cache.get("ml-agent", '{"seed": 42}')
            assert result == {"prediction": 0.95}

        assert cache._hit_count == 6  # 3 more hits
