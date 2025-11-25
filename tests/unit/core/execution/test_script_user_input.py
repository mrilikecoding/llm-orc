"""Tests for script agent user input handling functionality."""

from unittest.mock import AsyncMock, Mock

import pytest

from llm_orc.core.communication.protocol import ConversationManager, MessageProtocol
from llm_orc.core.execution.script_user_input_handler import ScriptUserInputHandler


class TestScriptUserInputDetection:
    """Test detection of scripts that require user input."""

    def test_detects_get_user_input_script_reference(self) -> None:
        """Test detection when script reference points to get_user_input.py."""
        handler = ScriptUserInputHandler()

        script_ref = "primitives/user-interaction/get_user_input.py"

        result = handler.requires_user_input(script_ref)

        assert result is True

    def test_detects_script_content_with_input_function(self) -> None:
        """Test detection when script content contains input() function."""
        handler = ScriptUserInputHandler()

        script_content = """
        import json
        import sys

        user_input = input("Enter your name: ")
        print(json.dumps({"name": user_input}))
        """

        result = handler.requires_user_input(script_content)

        assert result is True

    def test_ignores_regular_scripts(self) -> None:
        """Test that regular scripts without input are not detected."""
        handler = ScriptUserInputHandler()

        script_content = """
        import json
        import sys

        print(json.dumps({"result": "success"}))
        """

        result = handler.requires_user_input(script_content)

        assert result is False

    def test_ensemble_requires_user_input_detects_interactive_agents(self) -> None:
        """Test ensemble detection of agents with interactive scripts."""
        handler = ScriptUserInputHandler()

        # Mock ensemble config with interactive agents
        mock_ensemble = Mock()
        mock_ensemble.agents = [
            {
                "name": "user_input_agent",
                "type": "script",
                "script": "primitives/user-interaction/get_user_input.py",
            },
            {
                "name": "regular_agent",
                "type": "script",
                "script": "utils/process_data.py",
            },
        ]

        result = handler.ensemble_requires_user_input(mock_ensemble)

        assert result is True

    def test_ensemble_requires_user_input_detects_no_interactive_agents(self) -> None:
        """Test ensemble detection when no interactive agents present."""
        handler = ScriptUserInputHandler()

        # Mock ensemble config with no interactive agents
        mock_ensemble = Mock()
        mock_ensemble.agents = [
            {
                "name": "regular_agent",
                "type": "script",
                "script": "utils/process_data.py",
            },
            {
                "name": "llm_agent",
                "type": "llm",
                "model_profile": "claude-3-sonnet",
            },
        ]

        result = handler.ensemble_requires_user_input(mock_ensemble)

        assert result is False

    def test_ensemble_requires_user_input_handles_empty_ensemble(self) -> None:
        """Test ensemble detection with empty or invalid config."""
        handler = ScriptUserInputHandler()

        # Test empty agents
        mock_ensemble = Mock()
        mock_ensemble.agents = []
        result = handler.ensemble_requires_user_input(mock_ensemble)
        assert result is False

        # Test no agents attribute
        mock_ensemble_no_attr = Mock()
        del mock_ensemble_no_attr.agents
        result = handler.ensemble_requires_user_input(mock_ensemble_no_attr)
        assert result is False


class TestBidirectionalCommunication:
    """Test bidirectional communication between CLI and script agents."""

    async def test_cli_collects_user_input_for_script_agent(self) -> None:
        """Test that CLI can collect user input when requested by script agent."""
        # Setup communication protocol
        conversation_manager = ConversationManager()
        protocol = MessageProtocol(conversation_manager)

        # Setup handler with communication capability
        handler = ScriptUserInputHandler()

        # Mock CLI input collector
        cli_input_collector = AsyncMock()
        cli_input_collector.collect_input.return_value = "John Doe"

        # Mock script agent that needs input
        script_agent = Mock()
        script_agent.id = "script_agent_1"

        # Start a conversation
        conversation_id = conversation_manager.start_conversation(
            participants=["cli", "script_agent_1"], topic="user_input_collection"
        )

        # Script agent requests user input
        input_request = {
            "type": "user_input_request",
            "prompt": "Enter your name: ",
            "agent_id": "script_agent_1",
        }

        # This should fail because we haven't implemented the communication flow yet
        result = await handler.handle_input_request(
            input_request, protocol, conversation_id, cli_input_collector
        )

        assert result == "John Doe"
        cli_input_collector.collect_input.assert_called_once_with("Enter your name: ")


class TestEnhancedScriptAgentIntegration:
    """Test full integration of user input handling with EnhancedScriptAgent."""

    @pytest.mark.skip(
        reason="Complex interface not yet implemented - using simpler TDD approach"
    )
    async def test_enhanced_script_agent_handles_user_input_during_execution(
        self,
    ) -> None:
        """Test EnhancedScriptAgent detects user input need and requests via handler."""
        pass  # Complex interface test - will implement later

    @pytest.mark.skip(
        reason="Complex interface not yet implemented - using simpler TDD approach"
    )
    async def test_enhanced_script_agent_fallback_for_non_interactive_scripts(
        self,
    ) -> None:
        """Test non-interactive scripts execute normally without user input handling."""
        pass  # Complex interface test - will implement later
