"""Unit tests for ScriptUserInputHandler."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from llm_orc.core.execution.script_user_input_handler import ScriptUserInputHandler
from llm_orc.schemas.agent_config import LlmAgentConfig, ScriptAgentConfig


class TestScriptUserInputHandler:
    """Test suite for ScriptUserInputHandler."""

    def test_requires_user_input_with_get_user_input_py(self) -> None:
        """Test detection of get_user_input.py script reference."""
        handler = ScriptUserInputHandler()

        # Should detect get_user_input.py in path
        assert handler.requires_user_input("scripts/get_user_input.py") is True
        assert handler.requires_user_input("/path/to/get_user_input.py") is True
        assert handler.requires_user_input("./get_user_input.py") is True

        # Should not detect in other scripts
        assert handler.requires_user_input("scripts/process_data.py") is False
        assert handler.requires_user_input("analyze.sh") is False

    def test_requires_user_input_with_confirm_action_py(self) -> None:
        """Test detection of confirm_action.py script reference."""
        handler = ScriptUserInputHandler()

        assert handler.requires_user_input("confirm_action.py") is True
        assert (
            handler.requires_user_input("primitives/user-interaction/confirm_action.py")
            is True
        )
        assert handler.requires_user_input("/abs/path/confirm_action.py") is True

    def test_requires_user_input_with_input_function(self) -> None:
        """Test detection of input() function in script content."""
        handler = ScriptUserInputHandler()

        # Should detect input() function calls
        script_with_input = """
        name = input("Enter your name: ")
        print(f"Hello, {name}")
        """
        assert handler.requires_user_input(script_with_input) is True

        # Should detect even without parentheses context
        assert handler.requires_user_input("user_data = input(") is True

        # Currently detects input() even in comments (known limitation)
        script_with_comment = """
        # This script doesn't use input()
        print("No user interaction needed")
        """
        # This is a known behavior - it will detect input( even in comments
        assert handler.requires_user_input(script_with_comment) is True

        # Script truly without input
        script_without_input = """
        # This script doesn't need user interaction
        print("No user interaction needed")
        """
        assert handler.requires_user_input(script_without_input) is False

    def test_requires_user_input_edge_cases(self) -> None:
        """Test edge cases for user input detection."""
        handler = ScriptUserInputHandler()

        # Empty string
        assert handler.requires_user_input("") is False

        # None-like values (should handle gracefully)
        assert handler.requires_user_input("None") is False

        # Case sensitivity
        assert (
            handler.requires_user_input("INPUT(") is False
        )  # Python is case-sensitive

    def test_ensemble_requires_user_input_no_agents(self) -> None:
        """Test ensemble with no agents."""
        handler = ScriptUserInputHandler()

        # Mock ensemble config without agents attribute
        config_no_attr = Mock(spec=[])
        assert handler.ensemble_requires_user_input(config_no_attr) is False

        # Mock ensemble config with empty agents list
        config_empty = Mock(agents=[])
        assert handler.ensemble_requires_user_input(config_empty) is False

        # Mock ensemble config with None agents
        config_none = Mock(agents=None)
        assert handler.ensemble_requires_user_input(config_none) is False

    def test_ensemble_requires_user_input_non_script_agents(self) -> None:
        """Test ensemble with only non-script agents."""
        handler = ScriptUserInputHandler()

        config = Mock(
            agents=[
                LlmAgentConfig(name="analyzer", model_profile="some-profile"),
                LlmAgentConfig(name="fetcher", model_profile="some-profile"),
            ]
        )
        assert handler.ensemble_requires_user_input(config) is False

    def test_ensemble_requires_user_input_script_agents_no_input(self) -> None:
        """Test ensemble with script agents that don't require input."""
        handler = ScriptUserInputHandler()

        config = Mock(
            agents=[
                ScriptAgentConfig(name="processor", script="process_data.py"),
                ScriptAgentConfig(name="analyzer", script="analyze.sh"),
                LlmAgentConfig(name="summarizer", model_profile="some-profile"),
            ]
        )
        assert handler.ensemble_requires_user_input(config) is False

    def test_ensemble_requires_user_input_script_agents_with_input(self) -> None:
        """Test ensemble with script agents that require input."""
        handler = ScriptUserInputHandler()

        # One agent requires input
        config = Mock(
            agents=[
                ScriptAgentConfig(name="processor", script="process_data.py"),
                ScriptAgentConfig(name="input_agent", script="get_user_input.py"),
                LlmAgentConfig(name="summarizer", model_profile="some-profile"),
            ]
        )
        assert handler.ensemble_requires_user_input(config) is True

        # Script content with input()
        config2 = Mock(
            agents=[
                ScriptAgentConfig(
                    name="inline_input",
                    script='python -c "name = input()"',
                ),
            ]
        )
        assert handler.ensemble_requires_user_input(config2) is True

    def test_ensemble_requires_user_input_invalid_agent_configs(self) -> None:
        """Test ensemble with invalid agent configurations."""
        handler = ScriptUserInputHandler()

        # Non-ScriptAgentConfig agents are skipped; only ScriptAgentConfig is checked
        config = Mock(
            agents=[
                LlmAgentConfig(name="llm_agent", model_profile="some-profile"),
                ScriptAgentConfig(name="input_agent", script="get_user_input.py"),
            ]
        )
        assert handler.ensemble_requires_user_input(config) is True

    @pytest.mark.asyncio
    async def test_handle_input_request_basic(self) -> None:
        """Test basic input request handling."""
        handler = ScriptUserInputHandler()

        # Mock CLI input collector
        cli_input_collector = Mock()
        cli_input_collector.collect_input = AsyncMock(return_value="user response")

        input_request = {
            "prompt": "Enter your name: ",
            "agent_name": "test_agent",
            "script_path": "/path/to/script.py",
        }

        result = await handler.handle_input_request(
            input_request=input_request,
            conversation_id="conv123",
            cli_input_collector=cli_input_collector,
        )

        assert result == "user response"
        cli_input_collector.collect_input.assert_called_once_with("Enter your name: ")

    @pytest.mark.asyncio
    async def test_handle_input_request_default_values(self) -> None:
        """Test input request handling with default/missing values."""
        handler = ScriptUserInputHandler()

        # Mock CLI input collector
        cli_input_collector = Mock()
        cli_input_collector.collect_input = AsyncMock(
            return_value=42
        )  # Non-string return

        # Minimal input request
        input_request: dict[str, Any] = {}

        result = await handler.handle_input_request(
            input_request=input_request,
            conversation_id="conv",
            cli_input_collector=cli_input_collector,
        )

        assert result == "42"  # Should convert to string
        cli_input_collector.collect_input.assert_called_once_with("Enter input: ")

    @pytest.mark.asyncio
    async def test_handle_input_request_concurrent_calls(self) -> None:
        """Test handling multiple concurrent input requests."""
        handler = ScriptUserInputHandler()

        # Mock CLI input collector with delays
        cli_input_collector = Mock()

        async def delayed_input(prompt: str) -> str:
            await asyncio.sleep(0.01)  # Small delay
            return f"response_to_{prompt}"

        cli_input_collector.collect_input = delayed_input

        # Create multiple input requests
        requests = [
            {"prompt": f"Prompt {i}: ", "agent_name": f"agent_{i}"} for i in range(3)
        ]

        # Execute concurrently
        tasks = [
            handler.handle_input_request(
                input_request=req,
                conversation_id=f"conv_{i}",
                cli_input_collector=cli_input_collector,
            )
            for i, req in enumerate(requests)
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        for i, result in enumerate(results):
            assert result == f"response_to_Prompt {i}: "

    def test_initialization_defaults(self) -> None:
        """Test handler default initialization."""
        handler = ScriptUserInputHandler()
        assert handler.test_mode is False
        assert handler.llm_simulators == {}
