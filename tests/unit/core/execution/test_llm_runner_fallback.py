"""Tests for LlmAgentRunner's fallback wiring (explicit chains only).

Fallback is explicit only: a model-load or runtime failure uses the
agent-level ``fallback_model_profile`` first, then the failed profile's
own ``fallback_model_profile`` chain, and nothing else. When the chain
is empty or exhausted, the agent fails with the original error.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, Mock

import pytest

from llm_orc.core.execution.runners.llm_runner import LlmAgentRunner
from llm_orc.core.execution.usage_collector import UsageCollector
from llm_orc.core.models.model_factory import ModelConfigurationError
from llm_orc.schemas.agent_config import LlmAgentConfig


def _make_runner(
    model_factory: Mock, events: list[tuple[str, dict[str, object]]]
) -> LlmAgentRunner:
    config_manager = Mock()
    config_manager.get_model_profiles.return_value = {}
    return LlmAgentRunner(
        model_factory=model_factory,
        config_manager=config_manager,
        usage_collector=UsageCollector(),
        emit_event=lambda name, data: events.append((name, data)),
        classify_failure=lambda _e: "unknown",
    )


class TestModelLoadingFallbackWiring:
    """A model-load failure resolves through the explicit chain."""

    @pytest.mark.asyncio
    async def test_passes_agent_level_fallback_profile_to_factory(self) -> None:
        """The agent's own fallback_model_profile reaches the factory."""
        events: list[tuple[str, dict[str, object]]] = []
        model_factory = Mock()
        model_factory.load_model_from_agent_config = AsyncMock(
            side_effect=Exception("primary load failed")
        )
        fallback_model = AsyncMock()
        fallback_model.model_name = "agent-fallback-model"
        fallback_model.generate_response.return_value = "fallback response"
        model_factory.get_fallback_model = AsyncMock(
            return_value=(fallback_model, "agent-fallback")
        )

        runner = _make_runner(model_factory, events)
        agent = LlmAgentConfig(
            name="worker",
            model_profile="primary",
            fallback_model_profile="agent-fallback",
        )

        await runner.execute(agent, "input")

        model_factory.get_fallback_model.assert_called_once_with(
            context="agent_worker",
            original_profile="primary",
            agent_fallback_profile="agent-fallback",
            temperature=None,
            max_tokens=None,
            agent_options=None,
            response_format=None,
        )

    @pytest.mark.asyncio
    async def test_event_records_real_fallback_model_profile(self) -> None:
        """agent_fallback_started names the profile that actually ran,
        not a hardcoded None."""
        events: list[tuple[str, dict[str, object]]] = []
        model_factory = Mock()
        model_factory.load_model_from_agent_config = AsyncMock(
            side_effect=Exception("primary load failed")
        )
        fallback_model = AsyncMock()
        fallback_model.model_name = "micro-local-model"
        model_factory.get_fallback_model = AsyncMock(
            return_value=(fallback_model, "micro-local")
        )

        runner = _make_runner(model_factory, events)
        agent = LlmAgentConfig(name="worker", model_profile="primary")

        model, substituted = await runner._load_model_with_fallback(agent)

        assert substituted is True
        assert model is fallback_model
        fallback_events = [d for n, d in events if n == "agent_fallback_started"]
        assert len(fallback_events) == 1
        assert fallback_events[0]["fallback_model_profile"] == "micro-local"

    @pytest.mark.asyncio
    async def test_chain_exhausted_raises_original_error(self) -> None:
        """No fallback available: the agent fails with the ORIGINAL
        error, chained to the fallback-exhaustion cause."""
        events: list[tuple[str, dict[str, object]]] = []
        model_factory = Mock()
        original_error = ValueError("primary load failed")
        model_factory.load_model_from_agent_config = AsyncMock(
            side_effect=original_error
        )
        model_factory.get_fallback_model = AsyncMock(
            side_effect=ValueError("fallback chain exhausted")
        )

        runner = _make_runner(model_factory, events)
        agent = LlmAgentConfig(name="worker", model_profile="primary")

        with pytest.raises(ValueError, match="primary load failed") as exc_info:
            await runner.execute(agent, "input")

        assert exc_info.value.__cause__ is not None
        assert "fallback chain exhausted" in str(exc_info.value.__cause__)
        # No fallback event fires when there is nothing to fall back to.
        assert not [d for n, d in events if n == "agent_fallback_started"]

    @pytest.mark.asyncio
    async def test_model_configuration_error_is_never_fallback_eligible(self) -> None:
        """SF1 defense in depth: a ModelConfigurationError from the direct
        ModelFactory.load_model path (e.g. think on a non-llama-server
        provider) must propagate as-is, never trigger a substitution."""
        events: list[tuple[str, dict[str, object]]] = []
        model_factory = Mock()
        config_error = ModelConfigurationError("options.think is only supported ...")
        model_factory.load_model_from_agent_config = AsyncMock(side_effect=config_error)
        model_factory.get_fallback_model = AsyncMock()

        runner = _make_runner(model_factory, events)
        agent = LlmAgentConfig(
            name="worker",
            model_profile="hosted-thinking",
            fallback_model_profile="local-llama",
        )

        with pytest.raises(ModelConfigurationError):
            await runner.execute(agent, "input")

        model_factory.get_fallback_model.assert_not_called()
        assert not [d for n, d in events if n == "agent_fallback_started"]


class TestRuntimeFallbackWiring:
    """A runtime (post-load) failure walks the explicit fallback chain,
    trying each candidate in turn until one actually generates a
    response - not stopping at the first candidate merely offered."""

    @pytest.mark.asyncio
    async def test_runtime_failure_walks_the_chain_to_a_working_model(self) -> None:
        """OUTCOME pin: the primary fails at runtime, and so does the
        FIRST fallback hop - the response that comes back must be the
        one the SECOND hop actually generated, not just whichever
        model the chain offered first (doctrine 11: pin the harm, not
        the mechanism)."""
        events: list[tuple[str, dict[str, object]]] = []
        model_factory = Mock()

        primary_model = AsyncMock()
        primary_model.model_name = "primary-model"
        primary_model.generate_response = AsyncMock(
            side_effect=Exception("primary runtime failure")
        )
        model_factory.load_model_from_agent_config = AsyncMock(
            return_value=primary_model
        )

        hop1_model = AsyncMock()
        hop1_model.model_name = "hop1-model"
        hop1_model.generate_response = AsyncMock(
            side_effect=Exception("hop1 runtime failure")
        )
        hop2_model = AsyncMock()
        hop2_model.model_name = "hop2-model"
        hop2_model.generate_response = AsyncMock(return_value="hop2 answered")

        async def fake_chain(
            **_kwargs: object,
        ) -> AsyncIterator[tuple[object, str]]:
            yield hop1_model, "hop1-profile"
            yield hop2_model, "hop2-profile"

        model_factory.iter_fallback_chain = fake_chain

        runner = _make_runner(model_factory, events)
        agent = LlmAgentConfig(
            name="worker",
            model_profile="primary",
            fallback_model_profile="hop1-profile",
        )

        response, model, substituted = await runner.execute(agent, "input")

        assert response == "hop2 answered"
        assert model is hop2_model
        assert substituted is True
        hop1_model.generate_response.assert_awaited_once()
        hop2_model.generate_response.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_every_hop_failing_raises_the_original_chained_to_the_last(
        self,
    ) -> None:
        """Every candidate in the chain is actually tried; when all of
        them fail at runtime, the agent fails with the ORIGINAL error
        (not the last hop's), chained to the last hop's failure."""
        events: list[tuple[str, dict[str, object]]] = []
        model_factory = Mock()

        primary_model = AsyncMock()
        primary_model.model_name = "primary-model"
        primary_error = RuntimeError("primary blew up")
        primary_model.generate_response = AsyncMock(side_effect=primary_error)
        model_factory.load_model_from_agent_config = AsyncMock(
            return_value=primary_model
        )

        hop1_model = AsyncMock()
        hop1_model.model_name = "hop1-model"
        hop1_model.generate_response = AsyncMock(
            side_effect=RuntimeError("hop1 blew up")
        )
        hop2_model = AsyncMock()
        hop2_model.model_name = "hop2-model"
        hop2_error = RuntimeError("hop2 blew up")
        hop2_model.generate_response = AsyncMock(side_effect=hop2_error)

        async def fake_chain(
            **_kwargs: object,
        ) -> AsyncIterator[tuple[object, str]]:
            yield hop1_model, "hop1-profile"
            yield hop2_model, "hop2-profile"

        model_factory.iter_fallback_chain = fake_chain

        runner = _make_runner(model_factory, events)
        agent = LlmAgentConfig(
            name="worker",
            model_profile="primary",
            fallback_model_profile="hop1-profile",
        )

        with pytest.raises(RuntimeError, match="primary blew up") as exc_info:
            await runner.execute(agent, "input")

        assert exc_info.value is primary_error
        assert exc_info.value.__cause__ is hop2_error
        hop1_model.generate_response.assert_awaited_once()
        hop2_model.generate_response.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_chain_exhausted_raises_original_runtime_error(self) -> None:
        """No fallback available at runtime: the agent fails with the
        ORIGINAL error, chained to the fallback-exhaustion cause."""
        events: list[tuple[str, dict[str, object]]] = []
        model_factory = Mock()
        working_model = AsyncMock()
        working_model.model_name = "primary-model"
        runtime_error = RuntimeError("agent blew up")
        working_model.generate_response = AsyncMock(side_effect=runtime_error)
        model_factory.load_model_from_agent_config = AsyncMock(
            return_value=working_model
        )

        async def empty_chain(
            **_kwargs: object,
        ) -> AsyncIterator[tuple[object, str]]:
            return
            yield  # pragma: no cover - makes this an async generator

        model_factory.iter_fallback_chain = empty_chain

        runner = _make_runner(model_factory, events)
        agent = LlmAgentConfig(name="worker", model_profile="primary")

        with pytest.raises(RuntimeError, match="agent blew up") as exc_info:
            await runner.execute(agent, "input")

        assert exc_info.value.__cause__ is not None
        assert "fallback chain exhausted" in str(exc_info.value.__cause__)

    @pytest.mark.asyncio
    async def test_config_error_on_a_fallback_hop_is_the_recorded_error(self) -> None:
        """NIT (addendum 2026-09-23): a ModelConfigurationError raised
        while walking the chain (a fallback-only hop with a think/
        provider mismatch) must be the exception the agent fails with —
        AgentDispatcher records str(the raised exception), so a config
        error buried only in __cause__ behind the original runtime
        failure would never reach the agent's own error text."""
        events: list[tuple[str, dict[str, object]]] = []
        model_factory = Mock()

        primary_model = AsyncMock()
        primary_model.model_name = "primary-model"
        primary_error = RuntimeError("primary blew up")
        primary_model.generate_response = AsyncMock(side_effect=primary_error)
        model_factory.load_model_from_agent_config = AsyncMock(
            return_value=primary_model
        )

        config_error = ModelConfigurationError(
            "options.think is only supported for provider 'llama-server'"
        )

        async def fake_chain(
            **_kwargs: object,
        ) -> AsyncIterator[tuple[object, str]]:
            raise config_error
            yield  # pragma: no cover - makes this an async generator

        model_factory.iter_fallback_chain = fake_chain

        runner = _make_runner(model_factory, events)
        agent = LlmAgentConfig(
            name="worker",
            model_profile="primary",
            fallback_model_profile="hop1-profile",
        )

        with pytest.raises(ModelConfigurationError) as exc_info:
            await runner.execute(agent, "input")

        assert exc_info.value is config_error
        assert exc_info.value.__cause__ is primary_error
