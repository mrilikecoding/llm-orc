"""Tests for EnsembleAgentRunner (ADR-013)."""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
import yaml

from llm_orc.core.config.ensemble_config import EnsembleConfig, EnsembleLoader
from llm_orc.core.execution.executor_factory import ExecutorFactory
from llm_orc.schemas.agent_config import (
    EnsembleAgentConfig,
    ScriptAgentConfig,
)


class TestChildExecutorSharesImmutableInfrastructure:
    """Scenario 5: Child executor shares immutable infrastructure."""

    def test_child_shares_config_manager_and_credentials(self) -> None:
        """Child executor uses same config manager and credentials."""
        parent = ExecutorFactory.create_root_executor()

        child = parent.create_child_executor(depth=1)

        assert child._config_manager is parent._config_manager
        assert child._credential_storage is parent._credential_storage
        assert child._model_factory is parent._model_factory


class TestChildExecutorIsolatesMutableState:
    """Scenario 6: Child executor isolates mutable state."""

    def test_child_has_own_usage_collector(self) -> None:
        """Child executor has its own usage collector."""
        parent = ExecutorFactory.create_root_executor()

        child = parent.create_child_executor(depth=1)

        assert child._usage_collector is not parent._usage_collector


class TestChildEnsembleNoArtifact:
    """Scenario 7: Child ensemble does not produce its own artifact."""

    @pytest.mark.asyncio
    async def test_child_executor_skips_artifact_saving(self) -> None:
        """Child executor does not write artifact files."""
        parent = ExecutorFactory.create_root_executor()
        child = parent.create_child_executor(depth=1)

        # Child's artifact manager should be disabled
        config = EnsembleConfig(
            name="child-ensemble",
            description="Test child",
            agents=[
                ScriptAgentConfig(
                    name="worker",
                    script='echo \'{"success": true, "data": "ok"}\'',
                ),
            ],
        )

        result = await child.execute(config, "test")

        # Should complete without error
        assert result["status"] in ["completed", "completed_with_errors"]
        # Verify no artifact was saved (artifact manager is a no-op)
        assert child._artifact_manager is None or not hasattr(
            child._artifact_manager, "_saved"
        )


class TestDepthLimitPreventsUnboundedNesting:
    """Scenario 10: Depth limit prevents unbounded nesting."""

    def test_child_executor_tracks_depth(self) -> None:
        """Child executor records its depth."""
        parent = ExecutorFactory.create_root_executor()

        child = parent.create_child_executor(depth=1)

        assert child._depth == 1

    def test_default_depth_is_zero(self) -> None:
        """Top-level executor has depth 0."""
        executor = ExecutorFactory.create_root_executor()

        assert executor._depth == 0

    @pytest.mark.asyncio
    async def test_depth_limit_error(self) -> None:
        """Execution at depth exceeding limit raises error."""
        from llm_orc.core.execution.runners.ensemble_runner import (
            EnsembleAgentRunner,
        )

        # Create runner with depth at the limit
        runner = EnsembleAgentRunner(
            ensemble_loader=lambda _n: Mock(),
            parent_executor=Mock(),
            current_depth=3,
            depth_limit=3,
        )

        config = EnsembleAgentConfig(name="deep-agent", ensemble="child-ensemble")

        with pytest.raises(RuntimeError, match="depth limit"):
            await runner.execute(config, "test input")


class TestChildEnsembleFailureIsAgentFailure:
    """Scenario 11: Child ensemble failure is an agent failure
    (fail-closed-composition B1). The ensemble agent FAILS when none of
    the child's terminal agents succeeded — its error names the child's
    terminal failures. Raising here (rather than returning a status
    field) is caught by AgentDispatcher's existing
    ``_handle_agent_execution_failure``, the same path
    ``EnsembleAgentRunner``'s own depth-limit error already uses."""

    @pytest.mark.asyncio
    async def test_sole_terminal_failure_fails_the_agent(self) -> None:
        """A single-agent child whose one terminal failed: no terminal
        succeeded, so the ensemble agent fails, naming it."""
        from llm_orc.core.execution.runners.ensemble_runner import (
            EnsembleAgentRunner,
        )

        mock_child = AsyncMock()
        mock_child.execute = AsyncMock(
            return_value={
                "status": "completed_with_errors",
                "results": {"worker": {"status": "failed", "error": "boom"}},
                "metadata": {},
            }
        )

        child_config = EnsembleConfig(
            name="child",
            description="Test child",
            agents=[
                ScriptAgentConfig(name="worker", script="echo fail"),
            ],
        )

        mock_parent = Mock()
        mock_parent.create_child_executor.return_value = mock_child

        runner = EnsembleAgentRunner(
            ensemble_loader=lambda _name: child_config,
            parent_executor=mock_parent,
            current_depth=0,
            depth_limit=5,
        )

        config = EnsembleAgentConfig(name="analysis", ensemble="child-ensemble")

        with pytest.raises(RuntimeError) as exc_info:
            await runner.execute(config, "test input")

        assert "worker" in str(exc_info.value)
        assert "boom" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_one_of_two_terminals_succeeding_is_agent_success(self) -> None:
        """A multi-terminal child: one terminal failed, the other
        succeeded. At least one terminal succeeded, so the ensemble
        agent succeeds — the full child result (including the failed
        terminal's error) still reaches the response for
        DependencyResolver's per-terminal rendering (fail-closed-
        composition D)."""
        from llm_orc.core.execution.runners.ensemble_runner import (
            EnsembleAgentRunner,
        )

        mock_child = AsyncMock()
        mock_child.execute = AsyncMock(
            return_value={
                "status": "completed_with_errors",
                "results": {
                    "search_a": {"status": "failed", "error": "timeout"},
                    "search_b": {"status": "success", "response": "found it"},
                },
                "metadata": {},
            }
        )

        child_config = EnsembleConfig(
            name="child",
            description="Test child",
            agents=[
                ScriptAgentConfig(name="search_a", script="echo a"),
                ScriptAgentConfig(name="search_b", script="echo b"),
            ],
        )

        mock_parent = Mock()
        mock_parent.create_child_executor.return_value = mock_child

        runner = EnsembleAgentRunner(
            ensemble_loader=lambda _name: child_config,
            parent_executor=mock_parent,
            current_depth=0,
            depth_limit=5,
        )

        config = EnsembleAgentConfig(name="analysis", ensemble="child-ensemble")

        response, model_instance, model_substituted = await runner.execute(
            config, "test input"
        )

        result_data = json.loads(response)
        assert result_data["results"]["search_b"]["status"] == "success"
        assert result_data["results"]["search_a"]["status"] == "failed"
        assert model_instance is None
        assert model_substituted is False

    @pytest.mark.asyncio
    async def test_intermediate_failure_with_successful_terminal_is_agent_success(
        self,
    ) -> None:
        """A single-terminal child where an INTERMEDIATE agent failed but
        the terminal still succeeded (it had another successful
        dependency, per Invariant 13): the ensemble agent succeeds."""
        from llm_orc.core.execution.runners.ensemble_runner import (
            EnsembleAgentRunner,
        )

        mock_child = AsyncMock()
        mock_child.execute = AsyncMock(
            return_value={
                "status": "completed_with_errors",
                "results": {
                    "fetch": {"status": "failed", "error": "network down"},
                    "note": {"status": "success", "response": "static note"},
                    "format": {"status": "success", "response": "formatted"},
                },
                "metadata": {},
            }
        )

        child_config = EnsembleConfig(
            name="child",
            description="Test child",
            agents=[
                ScriptAgentConfig(name="fetch", script="echo fetch"),
                ScriptAgentConfig(name="note", script="echo note"),
                ScriptAgentConfig(
                    name="format",
                    script="echo format",
                    depends_on=["fetch", "note"],
                ),
            ],
        )

        mock_parent = Mock()
        mock_parent.create_child_executor.return_value = mock_child

        runner = EnsembleAgentRunner(
            ensemble_loader=lambda _name: child_config,
            parent_executor=mock_parent,
            current_depth=0,
            depth_limit=5,
        )

        config = EnsembleAgentConfig(name="analysis", ensemble="child-ensemble")

        response, model_instance, model_substituted = await runner.execute(
            config, "test input"
        )

        result_data = json.loads(response)
        assert result_data["results"]["format"]["status"] == "success"

    @pytest.mark.asyncio
    async def test_empty_child_falls_open_to_success(self) -> None:
        """A child config with no agents has no terminals to check —
        falls open rather than manufacturing a failure the resolver
        never asked for (mirrors _terminal_agents_for_ensemble's own
        resolution-failure default)."""
        from llm_orc.core.execution.runners.ensemble_runner import (
            EnsembleAgentRunner,
        )

        mock_child = AsyncMock()
        mock_child.execute = AsyncMock(
            return_value={"status": "completed", "results": {}, "metadata": {}}
        )

        child_config = EnsembleConfig(name="child", description="Empty", agents=[])

        mock_parent = Mock()
        mock_parent.create_child_executor.return_value = mock_child

        runner = EnsembleAgentRunner(
            ensemble_loader=lambda _name: child_config,
            parent_executor=mock_parent,
            current_depth=0,
            depth_limit=5,
        )

        config = EnsembleAgentConfig(name="analysis", ensemble="child-ensemble")

        response, _model_instance, _model_substituted = await runner.execute(
            config, "test input"
        )

        assert json.loads(response)["status"] == "completed"


class TestEnsembleAgentExecutesChildEnsemble:
    """Scenario 4: Ensemble agent executes child ensemble."""

    @pytest.mark.asyncio
    async def test_ensemble_agent_end_to_end(self) -> None:
        """Parent ensemble executes child via ensemble agent."""
        # Create child ensemble YAML
        child_yaml = {
            "name": "child-review",
            "description": "Test child review",
            "agents": [
                {
                    "name": "worker",
                    "script": ('echo \'{"success": true, "data": "child result"}\''),
                },
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(child_yaml, f)
            child_path = f.name

        parent_config = EnsembleConfig(
            name="parent-pipeline",
            description="Test parent ensemble",
            agents=[
                EnsembleAgentConfig(
                    name="review",
                    ensemble="child-review",
                ),
            ],
        )

        try:
            executor = ExecutorFactory.create_root_executor()
            mock_artifact = Mock()
            mock_artifact.save_execution_results = Mock()

            # Load child config for the mock
            child_config = EnsembleLoader().load_from_file(child_path)

            # Patch the runner's resolve function and artifact manager
            runner = executor._ensemble_agent_runner
            runner._resolve = lambda _name: child_config

            with patch.object(executor, "_artifact_manager", mock_artifact):
                result = await executor.execute(parent_config, "test input")

            assert result["status"] in [
                "completed",
                "completed_with_errors",
            ]
            assert "review" in result["results"]

            review_result = result["results"]["review"]
            assert review_result["status"] == "success"

            # Response should be JSON of child's full result dict
            child_result = json.loads(review_result["response"])
            assert "results" in child_result
            assert "worker" in child_result["results"]
        finally:
            Path(child_path).unlink()


class TestEnsembleExecutorTerminalAgentLookups:
    """fail-closed-composition D: EnsembleExecutor exposes the lookups
    DependencyResolver needs to render an LLM consumer's ensemble
    dependency as terminal responses, instead of the raw execution
    record.
    """

    def test_get_agent_config_finds_dependency_by_name(self) -> None:
        """Looks up a sibling agent's own config by name."""
        executor = ExecutorFactory.create_root_executor()
        config = EnsembleConfig(
            name="parent",
            description="Test parent",
            agents=[
                EnsembleAgentConfig(name="searcher", ensemble="web-searcher"),
            ],
        )
        executor._agent_configs = config.agents

        found = executor._get_agent_config("searcher")

        assert found is config.agents[0]
        assert executor._get_agent_config("missing") is None

    def test_terminal_agents_for_ensemble_resolves_child_graph(self) -> None:
        """Terminal names come from the same reference EnsembleAgentRunner
        resolves to run the child — the agents no other agent in the
        child depends on."""
        executor = ExecutorFactory.create_root_executor()
        child_config = EnsembleConfig(
            name="web-searcher",
            description="Test child",
            agents=[
                ScriptAgentConfig(name="fetch", script="echo ok"),
                ScriptAgentConfig(
                    name="format", script="echo ok", depends_on=["fetch"]
                ),
            ],
        )

        def _resolve(_ref: str) -> EnsembleConfig:
            return child_config

        executor._resolve_ensemble_reference = _resolve  # type: ignore[assignment]

        terminals = executor._terminal_agents_for_ensemble("web-searcher")

        assert terminals == ["format"]

    def test_terminal_agents_for_ensemble_missing_ensemble_yields_empty(self) -> None:
        """A resolution failure yields an empty list, not a crash — the
        resolver falls back to the child's own results dict."""
        executor = ExecutorFactory.create_root_executor()

        def _raise(_ref: str) -> EnsembleConfig:
            raise FileNotFoundError("nope")

        executor._resolve_ensemble_reference = _raise  # type: ignore[assignment]

        assert executor._terminal_agents_for_ensemble("missing") == []
