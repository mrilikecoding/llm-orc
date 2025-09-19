"""Script result caching system for reproducible research.

This module implements caching functionality for script agents to support
reproducible research as outlined in ADR-001 architecture review.
"""

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_orc.core.execution.artifact_manager import ArtifactManager


@dataclass
class ScriptCacheConfig:
    """Configuration for ScriptCache system."""

    enabled: bool = True
    ttl_seconds: int = 3600  # 1 hour default TTL
    max_size: int = 1000  # Maximum number of cached entries
    persist_to_artifacts: bool = False  # Whether to persist cache to artifacts
    artifact_base_dir: Path = field(default_factory=lambda: Path("."))


@dataclass
class CacheEntry:
    """Cache entry with timestamp and data."""

    data: Any
    timestamp: float
    access_count: int = 0


class ScriptCache:
    """Cache for script execution results with TTL and size limits.

    Provides caching based on script content hash + input parameters hash
    to support reproducible research and avoid re-executing identical operations.
    """

    def __init__(self, config: ScriptCacheConfig) -> None:
        """Initialize script cache with configuration.

        Args:
            config: Cache configuration settings
        """
        self.config = config
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "evictions": 0,
        }

        # Initialize artifact manager if persistence is enabled
        self._artifact_manager: ArtifactManager | None = None
        if config.persist_to_artifacts:
            self._artifact_manager = ArtifactManager(config.artifact_base_dir)

    def _generate_cache_key(
        self, script_content: str, parameters: dict[str, Any]
    ) -> str:
        """Generate cache key from script content and parameters.

        Args:
            script_content: The script content or path
            parameters: Input parameters dict

        Returns:
            SHA256 hash as hex string
        """
        # Create deterministic string representation
        content_hash = hashlib.sha256(script_content.encode()).hexdigest()
        params_json = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
        params_hash = hashlib.sha256(params_json.encode()).hexdigest()

        # Combine hashes
        combined = f"{content_hash}:{params_hash}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def get(self, script_content: str, parameters: dict[str, Any]) -> Any | None:
        """Get cached result if available and not expired.

        Args:
            script_content: The script content or path
            parameters: Input parameters dict

        Returns:
            Cached result or None if not found/expired
        """
        if not self.config.enabled:
            return None

        cache_key = self._generate_cache_key(script_content, parameters)

        # Check in-memory cache first
        if cache_key in self._cache:
            entry = self._cache[cache_key]

            # Check TTL expiration
            if time.time() - entry.timestamp > self.config.ttl_seconds:
                del self._cache[cache_key]
                self._stats["misses"] += 1
                return None

            # Move to end (LRU)
            self._cache.move_to_end(cache_key)
            entry.access_count += 1
            self._stats["hits"] += 1
            return entry.data

        # Try loading from artifacts if persistence enabled
        if self._artifact_manager:
            persisted_data = self._load_from_artifacts(cache_key)
            if persisted_data:
                # Add back to in-memory cache
                self._cache[cache_key] = CacheEntry(
                    data=persisted_data, timestamp=time.time(), access_count=1
                )
                self._stats["hits"] += 1
                return persisted_data

        self._stats["misses"] += 1
        return None

    def set(self, script_content: str, parameters: dict[str, Any], result: Any) -> None:
        """Cache a script execution result.

        Args:
            script_content: The script content or path
            parameters: Input parameters dict
            result: The execution result to cache
        """
        if not self.config.enabled:
            return

        cache_key = self._generate_cache_key(script_content, parameters)

        # Create cache entry
        entry = CacheEntry(data=result, timestamp=time.time(), access_count=0)

        # Add to in-memory cache
        self._cache[cache_key] = entry
        self._cache.move_to_end(cache_key)

        # Evict if over size limit
        while len(self._cache) > self.config.max_size:
            evicted_key, _ = self._cache.popitem(last=False)  # Remove oldest
            self._stats["evictions"] += 1

        # Persist to artifacts if enabled
        if self._artifact_manager:
            self._save_to_artifacts(cache_key, result)

        self._stats["sets"] += 1

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()

        # Clear persisted cache if artifacts enabled
        if self._artifact_manager:
            self._clear_artifacts()

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache hit/miss stats and hit rate
        """
        total_requests = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0.0

        return {
            **self._stats,
            "hit_rate": hit_rate,
            "size": len(self._cache),
            "max_size": self.config.max_size,
        }

    def _load_from_artifacts(self, cache_key: str) -> Any | None:
        """Load cached result from artifacts.

        Args:
            cache_key: The cache key to load

        Returns:
            Cached data or None if not found
        """
        if not self._artifact_manager:
            return None

        try:
            # Use artifact manager to load from cache directory
            cache_dir = self.config.artifact_base_dir / ".llm-orc" / "cache"
            cache_file = cache_dir / f"{cache_key}.json"

            if not cache_file.exists():
                return None

            with cache_file.open("r") as f:
                cache_data = json.load(f)

            # Check TTL
            if time.time() - cache_data["timestamp"] > self.config.ttl_seconds:
                cache_file.unlink(missing_ok=True)
                return None

            return cache_data["result"]

        except (OSError, json.JSONDecodeError, KeyError):
            return None

    def _save_to_artifacts(self, cache_key: str, result: Any) -> None:
        """Save cached result to artifacts.

        Args:
            cache_key: The cache key
            result: The result to save
        """
        if not self._artifact_manager:
            return

        try:
            # Create cache directory
            cache_dir = self.config.artifact_base_dir / ".llm-orc" / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)

            # Save cache entry
            cache_file = cache_dir / f"{cache_key}.json"
            cache_data = {
                "result": result,
                "timestamp": time.time(),
                "cache_key": cache_key,
            }

            with cache_file.open("w") as f:
                json.dump(cache_data, f, indent=2)

        except (OSError, TypeError):
            # Silently ignore persistence errors
            pass

    def _clear_artifacts(self) -> None:
        """Clear all persisted cache artifacts."""
        if not self._artifact_manager:
            return

        try:
            cache_dir = self.config.artifact_base_dir / ".llm-orc" / "cache"
            if cache_dir.exists():
                for cache_file in cache_dir.glob("*.json"):
                    cache_file.unlink(missing_ok=True)
        except OSError:
            # Silently ignore cleanup errors
            pass
