"""Unit tests for ScriptUserInputHandler test mode with LLM simulation."""

from unittest.mock import AsyncMock

import pytest

from llm_orc.core.execution.script_user_input_handler import ScriptUserInputHandler
from llm_orc.core.validation import LLMResponseGenerator


class TestScriptUserInputHandlerTestMode:
    """Test suite for ScriptUserInputHandler test mode execution."""

    def test_initialization_with_test_mode_false(self) -> None:
        """Test handler initialization with test_mode=False (interactive mode)."""
        handler = ScriptUserInputHandler(test_mode=False)
        assert handler.test_mode is False
        assert handler.llm_simulators == {}

    def test_initialization_with_test_mode_true_no_config(self) -> None:
        """Test handler initialization with test_mode=True but no LLM config."""
        handler = ScriptUserInputHandler(test_mode=True, llm_config=None)
        assert handler.test_mode is True
        assert handler.llm_simulators == {}

    def test_initialization_with_test_mode_and_llm_config(self) -> None:
        """Test handler initialization with test_mode=True and LLM config."""
        from unittest.mock import MagicMock, patch

        llm_config = {
            "agent1": {
                "model": "qwen3:0.6b",
                "persona": "helpful_user",
            }
        }

        # Mock OllamaModel to avoid requiring Ollama in tests
        with patch("llm_orc.models.ollama.OllamaModel") as mock_ollama:
            mock_ollama.return_value = MagicMock()
            handler = ScriptUserInputHandler(test_mode=True, llm_config=llm_config)
            assert handler.test_mode is True
            assert "agent1" in handler.llm_simulators
            assert isinstance(handler.llm_simulators["agent1"], LLMResponseGenerator)

    @pytest.mark.asyncio
    async def test_get_user_input_interactive_mode(self) -> None:
        """Test get_user_input in interactive mode should raise NotImplementedError.

        Interactive mode (test_mode=False) uses real stdin which is not tested
        here.
        """
        handler = ScriptUserInputHandler(test_mode=False)

        # Should raise NotImplementedError for now since real input() needs
        # special testing
        with pytest.raises(NotImplementedError):
            await handler.get_user_input(
                agent_name="test_agent",
                prompt="Enter value: ",
                context={"previous_output": "some data"},
            )

    @pytest.mark.asyncio
    async def test_get_user_input_test_mode_with_simulator(self) -> None:
        """Test get_user_input in test mode with configured LLM simulator."""
        from unittest.mock import MagicMock, patch

        # Create simulator with cached response for deterministic testing
        llm_config = {
            "test_agent": {
                "model": "qwen3:0.6b",
                "persona": "helpful_user",
                "cached_responses": {
                    # Cache key will be generated from prompt and context
                },
            }
        }

        # Mock OllamaModel to avoid requiring Ollama in tests
        with patch("llm_orc.models.ollama.OllamaModel") as mock_ollama:
            mock_client = MagicMock()
            mock_client.generate_response = AsyncMock(return_value="simulated response")
            mock_ollama.return_value = mock_client
            handler = ScriptUserInputHandler(test_mode=True, llm_config=llm_config)

            # Should use LLM simulation
            result = await handler.get_user_input(
                agent_name="test_agent",
                prompt="Enter value: ",
                context={"previous_output": "some data"},
            )

            # Should return a string response from LLM
            assert isinstance(result, str)
            assert len(result) > 0

    @pytest.mark.asyncio
    async def test_get_user_input_test_mode_without_simulator(self) -> None:
        """Test get_user_input in test mode without simulator raises error."""
        handler = ScriptUserInputHandler(test_mode=True, llm_config=None)

        # Should raise RuntimeError for missing simulator
        with pytest.raises(
            RuntimeError, match="No LLM simulator configured for agent: test_agent"
        ):
            await handler.get_user_input(
                agent_name="test_agent",
                prompt="Enter value: ",
                context={},
            )

    @pytest.mark.asyncio
    async def test_get_user_input_test_mode_different_personas(self) -> None:
        """Test that different personas generate different response styles."""
        from unittest.mock import MagicMock, patch

        # Configure two agents with different personas
        llm_config = {
            "helpful_agent": {
                "model": "qwen3:0.6b",
                "persona": "helpful_user",
            },
            "critical_agent": {
                "model": "qwen3:0.6b",
                "persona": "critical_reviewer",
            },
        }

        # Mock OllamaModel to avoid requiring Ollama in tests
        with patch("llm_orc.models.ollama.OllamaModel") as mock_ollama:
            mock_ollama.return_value = MagicMock()
            handler = ScriptUserInputHandler(test_mode=True, llm_config=llm_config)

            # Both should return responses, but styles will differ
            # (We can't test actual style without running LLM,
            # so just verify calls work)
            assert "helpful_agent" in handler.llm_simulators
            assert "critical_agent" in handler.llm_simulators

            # Verify personas are configured correctly
            assert handler.llm_simulators["helpful_agent"].persona == "helpful_user"
            assert (
                handler.llm_simulators["critical_agent"].persona == "critical_reviewer"
            )

    def test_initialize_simulators_with_empty_config(self) -> None:
        """Test _initialize_simulators with empty config."""
        handler = ScriptUserInputHandler(test_mode=True, llm_config={})
        assert handler.llm_simulators == {}

    def test_initialize_simulators_with_cached_responses(self) -> None:
        """Test _initialize_simulators preserves cached responses."""
        from unittest.mock import MagicMock, patch

        cached_responses = {"prompt1": "response1"}
        llm_config = {
            "agent1": {
                "model": "qwen3:0.6b",
                "persona": "helpful_user",
                "cached_responses": cached_responses,
            }
        }

        # Mock OllamaModel to avoid requiring Ollama in tests
        with patch("llm_orc.models.ollama.OllamaModel") as mock_ollama:
            mock_ollama.return_value = MagicMock()
            handler = ScriptUserInputHandler(test_mode=True, llm_config=llm_config)

            # Verify cached responses are passed to generator
            assert handler.llm_simulators["agent1"].response_cache == cached_responses


class TestScriptUserInputHandlerBackwardCompatibility:
    """Test backward compatibility with existing ScriptUserInputHandler usage."""

    def test_default_initialization_unchanged(self) -> None:
        """Test that default initialization remains unchanged for backward compat."""
        # Existing code: ScriptUserInputHandler()
        handler = ScriptUserInputHandler()

        # Should work with no test mode (default)
        assert not hasattr(handler, "test_mode") or handler.test_mode is False

    def test_requires_user_input_still_works(self) -> None:
        """Test that requires_user_input detection still works."""
        handler = ScriptUserInputHandler()

        # Should still detect user input scripts
        assert handler.requires_user_input("get_user_input.py") is True
        assert handler.requires_user_input("other_script.py") is False
