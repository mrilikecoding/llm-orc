"""Unit tests for ProviderHandler."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from llm_orc.schemas.agent_config import (
    DynamicDispatchAgentConfig,
    EnsembleAgentConfig,
    LoopAgentConfig,
    LoopSpec,
    ScriptAgentConfig,
)
from llm_orc.services.handlers.provider_handler import ProviderHandler


def _make_handler(
    find_ensemble_return: Any = None,
    profiles: dict[str, dict[str, Any]] | None = None,
) -> ProviderHandler:
    profile_handler = MagicMock()
    profile_handler.get_all_profiles.return_value = profiles or {}
    find_ensemble = MagicMock(return_value=find_ensemble_return)
    return ProviderHandler(profile_handler, find_ensemble)


class TestCheckEnsembleRunnableNonLlmAgents:
    """Non-LLM agents should not trigger MISSING_PROFILE."""

    async def test_script_agent_runnable(self) -> None:
        """Script agents report provider='script' and status AVAILABLE."""
        from llm_orc.providers.status_types import AgentStatus

        config = MagicMock()
        config.name = "script-ens"
        config.agents = [ScriptAgentConfig(name="sc", script="echo hi")]

        handler = _make_handler(find_ensemble_return=config)

        with patch.object(
            handler, "get_provider_status", new_callable=AsyncMock
        ) as mock_status:
            mock_status.return_value = {"providers": {}}
            result = await handler.check_ensemble_runnable(
                {"ensemble_name": "script-ens"}
            )

        agent_result = result["agents"][0]
        assert agent_result["name"] == "sc"
        assert agent_result["provider"] == "script"
        assert agent_result["status"] == AgentStatus.AVAILABLE.value

    async def test_ensemble_agent_runnable(self) -> None:
        """Ensemble agents report provider='ensemble' and status AVAILABLE."""
        from llm_orc.providers.status_types import AgentStatus

        config = MagicMock()
        config.name = "composed"
        config.agents = [EnsembleAgentConfig(name="ref", ensemble="other")]

        handler = _make_handler(find_ensemble_return=config)

        with patch.object(
            handler, "get_provider_status", new_callable=AsyncMock
        ) as mock_status:
            mock_status.return_value = {"providers": {}}
            result = await handler.check_ensemble_runnable(
                {"ensemble_name": "composed"}
            )

        agent_result = result["agents"][0]
        assert agent_result["name"] == "ref"
        assert agent_result["provider"] == "ensemble"
        assert agent_result["status"] == AgentStatus.AVAILABLE.value

    async def test_loop_agent_runnable(self) -> None:
        """Loop agents report provider='loop' and status AVAILABLE."""
        from llm_orc.providers.status_types import AgentStatus

        config = MagicMock()
        config.name = "loop-ens"
        config.agents = [
            LoopAgentConfig(
                name="looper",
                loop=LoopSpec(body="body-ens", until="${done}", max_iterations=5),
            )
        ]

        handler = _make_handler(find_ensemble_return=config)

        with patch.object(
            handler, "get_provider_status", new_callable=AsyncMock
        ) as mock_status:
            mock_status.return_value = {"providers": {}}
            result = await handler.check_ensemble_runnable(
                {"ensemble_name": "loop-ens"}
            )

        agent_result = result["agents"][0]
        assert agent_result["name"] == "looper"
        assert agent_result["provider"] == "loop"
        assert agent_result["status"] == AgentStatus.AVAILABLE.value

    async def test_dynamic_dispatch_agent_runnable(self) -> None:
        """Dynamic-dispatch agents report provider='dispatch' and status AVAILABLE."""
        from llm_orc.providers.status_types import AgentStatus

        config = MagicMock()
        config.name = "dispatch-ens"
        config.agents = [
            DynamicDispatchAgentConfig(name="dispatcher", dispatch="${target}")
        ]

        handler = _make_handler(find_ensemble_return=config)

        with patch.object(
            handler, "get_provider_status", new_callable=AsyncMock
        ) as mock_status:
            mock_status.return_value = {"providers": {}}
            result = await handler.check_ensemble_runnable(
                {"ensemble_name": "dispatch-ens"}
            )

        agent_result = result["agents"][0]
        assert agent_result["name"] == "dispatcher"
        assert agent_result["provider"] == "dispatch"
        assert agent_result["status"] == AgentStatus.AVAILABLE.value


class TestLlamaServerStatusCarriesCache:
    async def test_status_reports_models_and_cached_sources(self) -> None:
        from llm_orc.providers.llama_server import LlamaServerClient

        handler = _make_handler()
        listing = [
            {"id": "qwen3-8b", "status": {"value": "loaded"}},
            {"id": "default", "status": {"value": "unloaded"}},
            {"id": "unsloth/Qwen3-8B-GGUF:Q4_K_M", "status": {"value": "unloaded"}},
        ]
        with patch.object(LlamaServerClient, "_list", return_value=listing):
            status = await handler._get_llama_server_status()

        assert status["available"] is True
        assert status["models"] == ["qwen3-8b"]
        assert status["cached"] == ["unsloth/Qwen3-8B-GGUF:Q4_K_M"]
