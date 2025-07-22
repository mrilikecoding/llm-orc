"""Tests for script-based agent execution."""

from unittest.mock import patch

import pytest

from llm_orc.agents.script_agent import ScriptAgent


class TestScriptAgent:
    """Test script-based agent functionality."""

    def test_script_agent_creation_requires_script_or_command(self) -> None:
        """Test that script agent requires either script or command."""
        # This should fail - no script or command provided
        with pytest.raises(ValueError, match="must have either 'script' or 'command'"):
            ScriptAgent("test_agent", {})

    def test_script_agent_creation_with_script(self) -> None:
        """Test script agent creation with script content."""
        config = {"script": "echo 'Hello World'"}
        agent = ScriptAgent("test_agent", config)

        assert agent.name == "test_agent"
        assert agent.script == "echo 'Hello World'"
        assert agent.command == ""
        assert agent.timeout == 60  # default timeout

    def test_script_agent_creation_with_command(self) -> None:
        """Test script agent creation with command."""
        config = {"command": "echo 'Hello World'"}
        agent = ScriptAgent("test_agent", config)

        assert agent.name == "test_agent"
        assert agent.command == "echo 'Hello World'"
        assert agent.script == ""

    def test_script_agent_creation_with_custom_timeout(self) -> None:
        """Test script agent creation with custom timeout."""
        config = {"script": "echo 'test'", "timeout_seconds": 30}
        agent = ScriptAgent("test_agent", config)

        assert agent.timeout == 30

    @pytest.mark.asyncio
    async def test_script_agent_execute_simple_script(self) -> None:
        """Test script agent execution with simple script."""
        config = {"script": "echo 'Hello World'"}
        agent = ScriptAgent("test_agent", config)

        result = await agent.execute("test input")
        assert result.strip() == "Hello World"

    @pytest.mark.asyncio
    async def test_script_agent_execute_simple_command(self) -> None:
        """Test script agent execution with simple command."""
        config = {"command": "echo 'Hello Command'"}
        agent = ScriptAgent("test_agent", config)

        result = await agent.execute("test input")
        assert result.strip() == "Hello Command"

    @pytest.mark.asyncio
    async def test_script_agent_receives_input_data(self) -> None:
        """Test that script agent receives input data via environment."""
        config = {"script": 'echo "Input: $INPUT_DATA"'}
        agent = ScriptAgent("test_agent", config)

        result = await agent.execute("test message")
        assert result.strip() == "Input: test message"

    def test_script_agent_get_agent_type(self) -> None:
        """Test script agent type identification."""
        config = {"script": "echo 'test'"}
        agent = ScriptAgent("test_agent", config)

        assert agent.get_agent_type() == "script"

    @pytest.mark.asyncio
    async def test_script_agent_with_context_variables(self) -> None:
        """Test that script agent receives context variables in environment."""
        config = {"script": 'echo "User: $CONTEXT_USER, Type: $CONTEXT_TYPE"'}
        agent = ScriptAgent("test_agent", config)

        context = {"user": "alice", "type": "admin"}
        result = await agent.execute("test input", context)
        assert result.strip() == "User: alice, Type: admin"

    @pytest.mark.asyncio
    async def test_script_agent_with_custom_environment(self) -> None:
        """Test script agent with custom environment variables."""
        config = {
            "script": 'echo "Custom: $CUSTOM_VAR"',
            "environment": {"CUSTOM_VAR": "custom_value"},
        }
        agent = ScriptAgent("test_agent", config)

        result = await agent.execute("test input")
        assert result.strip() == "Custom: custom_value"

    @pytest.mark.asyncio
    async def test_script_agent_timeout_error(self) -> None:
        """Test script agent timeout handling."""
        config = {"script": "sleep 10", "timeout_seconds": 1}
        agent = ScriptAgent("test_agent", config)

        with pytest.raises(RuntimeError, match="timed out after 1s"):
            await agent.execute("test input")

    @pytest.mark.asyncio
    async def test_script_agent_called_process_error(self) -> None:
        """Test script agent handling of command failures."""
        config = {"script": "exit 1"}  # Script that fails
        agent = ScriptAgent("test_agent", config)

        with pytest.raises(RuntimeError, match="Script agent test_agent failed"):
            await agent.execute("test input")

    @pytest.mark.asyncio
    async def test_script_agent_command_called_process_error(self) -> None:
        """Test script agent command handling of failures."""
        config = {"command": "false"}  # Command that fails
        agent = ScriptAgent("test_agent", config)

        with pytest.raises(RuntimeError, match="Script agent test_agent failed"):
            await agent.execute("test input")

    @pytest.mark.asyncio
    async def test_script_agent_general_exception_handling(self) -> None:
        """Test script agent general exception handling."""
        config = {"script": "echo 'test'"}
        agent = ScriptAgent("test_agent", config)

        # Mock subprocess.run to raise a general exception
        with patch(
            "subprocess.run", side_effect=OSError("Permission denied")
        ):
            with pytest.raises(
                RuntimeError, match="Script agent test_agent error"
            ):
                await agent.execute("test input")
