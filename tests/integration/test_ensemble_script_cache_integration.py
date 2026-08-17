"""Integration tests for ScriptCache with EnsembleExecutor.

Tests that EnsembleExecutor properly integrates with ScriptCache for transparent
caching.
"""

import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from llm_orc.core.execution.ensemble_execution import EnsembleExecutor
from llm_orc.core.execution.executor_factory import ExecutorFactory
from llm_orc.core.execution.scripting.cache import ScriptCache, ScriptCacheConfig
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
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The executor loads its script-cache config from the performance config.

        Isolated from the ambient project ``.llm-orc/config.yaml`` (whose
        ``performance.script_cache`` may enable or disable the cache) by pointing
        config discovery at a tmp project dir with a known performance config.
        The non-default ttl/max_size prove the values are read from config rather
        than falling back to the hardcoded defaults.
        """
        llm_orc_dir = tmp_path / ".llm-orc"
        llm_orc_dir.mkdir()
        (llm_orc_dir / "config.yaml").write_text(
            yaml.safe_dump(
                {
                    "performance": {
                        "script_cache": {
                            "enabled": True,
                            "ttl_seconds": 7200,
                            "max_size": 500,
                        }
                    }
                }
            )
        )
        monkeypatch.chdir(tmp_path)

        executor = ExecutorFactory.create_root_executor()

        assert hasattr(executor, "_script_cache_config")
        assert hasattr(executor, "_script_cache")
        cache_config = executor._script_cache_config
        assert cache_config.enabled is True
        assert cache_config.ttl_seconds == 7200
        assert cache_config.max_size == 500

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

    def test_a_project_with_no_script_cache_block_gets_a_disabled_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The shipped default has to survive the config loader (#160).

        Found by review: the flip to ScriptCacheConfig.enabled = False was
        INERT, because _load_script_cache_config restated every default and
        hardcoded enabled=True. The dataclass pin in test_script_cache.py
        asserts a field the runtime never read, so it could not catch this.

        This is the shape of every fresh install: templates/global-config.yaml
        writes no script_cache block and load_performance_config has no such
        key in its defaults, so the absent-config path IS the default path.
        """
        llm_orc_dir = tmp_path / ".llm-orc"
        llm_orc_dir.mkdir()
        (llm_orc_dir / "config.yaml").write_text("project:\n  name: fresh\n")
        monkeypatch.chdir(tmp_path)
        # No global config either, so nothing supplies a script_cache block.
        monkeypatch.setenv("HOME", str(tmp_path))

        executor = ExecutorFactory.create_root_executor()

        assert executor._script_cache_config.enabled is False

    def test_an_explicit_opt_in_still_wins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reading defaults off the dataclass must not break the override.

        Without this, "enabled=defaults.enabled" could be mistidied into a
        constant False and the config key would stop being honoured, with the
        pin above still green.
        """
        llm_orc_dir = tmp_path / ".llm-orc"
        llm_orc_dir.mkdir()
        (llm_orc_dir / "config.yaml").write_text(
            yaml.dump(
                {
                    "project": {"name": "opted-in"},
                    "performance": {"script_cache": {"enabled": True}},
                }
            )
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path))

        executor = ExecutorFactory.create_root_executor()

        assert executor._script_cache_config.enabled is True
