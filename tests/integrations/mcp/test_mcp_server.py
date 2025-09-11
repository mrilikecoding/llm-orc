"""Test suite for MCP server implementation."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from llm_orc.integrations.mcp.server import MCPServer


@pytest.fixture(autouse=True)
def mock_expensive_dependencies():
    """Mock expensive MCP server dependencies for all tests."""
    with patch("llm_orc.integrations.mcp.server.EnsembleExecutor"):
        with patch("llm_orc.integrations.mcp.server.ConfigurationManager"):
            with patch("llm_orc.integrations.mcp.server.EnsembleLoader"):
                yield


class TestMCPServer:
    """Test MCP server basic functionality."""

    def test_create_mcp_server_with_ensemble(self) -> None:
        """Should create MCP server with ensemble configuration."""
        ensemble_name = "test_ensemble"
        server = MCPServer(ensemble_name)
        assert server.ensemble_name == ensemble_name

    @pytest.mark.asyncio
    async def test_handle_initialize_request(self) -> None:
        """Should handle MCP initialize request and return capabilities."""
        server = MCPServer("test_ensemble")

        initialize_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0.0"},
            },
        }

        response = await server.handle_request(initialize_request)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert "result" in response
        assert "capabilities" in response["result"]
        assert "tools" in response["result"]["capabilities"]

    @pytest.mark.asyncio
    async def test_handle_tools_list_request(self) -> None:
        """Should return available tools (ensembles) when requested."""
        server = MCPServer("architecture_review")

        tools_request = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}

        response = await server.handle_request(tools_request)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 2
        assert "result" in response
        assert "tools" in response["result"]
        assert len(response["result"]["tools"]) == 1
        assert response["result"]["tools"][0]["name"] == "architecture_review"

    @pytest.mark.asyncio
    @patch("llm_orc.integrations.mcp.server.MCPServer._load_ensemble_config")
    async def test_handle_tools_call_request(self, mock_load_config: AsyncMock) -> None:
        """Should execute ensemble when tool is called."""
        # Mock the ensemble executor
        mock_executor = AsyncMock()
        mock_executor.execute.return_value = {
            "ensemble": "architecture_review",
            "status": "completed",
            "results": {"agent1": {"response": "Test response", "status": "success"}},
            "synthesis": "Synthesized result",
        }

        server = MCPServer("architecture_review")
        server.executor = mock_executor

        # Mock ensemble loading
        mock_config = Mock()
        mock_config.name = "architecture_review"
        mock_load_config.return_value = mock_config

        call_request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "architecture_review",
                "arguments": {"input": "Analyze this architecture design"},
            },
        }

        response = await server.handle_request(call_request)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 3
        assert "result" in response
        assert "content" in response["result"]

        # Verify executor was called
        mock_executor.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_invalid_method(self) -> None:
        """Should return error for invalid methods."""
        server = MCPServer("test_ensemble")

        invalid_request = {"jsonrpc": "2.0", "id": 4, "method": "invalid/method"}

        response = await server.handle_request(invalid_request)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 4
        assert "error" in response
        assert response["error"]["code"] == -32601  # Method not found

    @pytest.mark.asyncio
    async def test_handle_request_missing_id(self) -> None:
        """Should return error for request missing ID."""
        server = MCPServer("test_ensemble")

        request_without_id = {"jsonrpc": "2.0", "method": "initialize"}

        response = await server.handle_request(request_without_id)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 0
        assert "error" in response
        assert response["error"]["code"] == -32600  # Invalid request

    @pytest.mark.asyncio
    @patch("llm_orc.integrations.mcp.server.MCPServer._load_ensemble_config")
    async def test_tools_call_invalid_tool_name(
        self, mock_load_config: AsyncMock
    ) -> None:
        """Should return error for invalid tool name."""
        server = MCPServer("correct_ensemble")

        call_request = {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "wrong_ensemble",  # Different from server ensemble
                "arguments": {"input": "test"},
            },
        }

        response = await server.handle_request(call_request)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 5
        assert "error" in response
        assert response["error"]["code"] == -32602  # Invalid tool name

    @pytest.mark.asyncio
    @patch("llm_orc.integrations.mcp.server.MCPServer._load_ensemble_config")
    async def test_tools_call_ensemble_not_found(
        self, mock_load_config: AsyncMock
    ) -> None:
        """Should return error when ensemble config not found."""
        server = MCPServer("missing_ensemble")

        # Mock config loading to return None
        mock_load_config.return_value = None

        call_request = {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {
                "name": "missing_ensemble",
                "arguments": {"input": "test"},
            },
        }

        response = await server.handle_request(call_request)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 6
        assert "error" in response
        assert response["error"]["code"] == -32602
        assert "not found" in response["error"]["message"]

    @pytest.mark.asyncio
    async def test_load_ensemble_config_file_not_found(self) -> None:
        """Should raise FileNotFoundError when ensemble not found."""
        server = MCPServer("nonexistent_ensemble")

        # Mock config manager to return empty directories
        with patch.object(server.config_manager, "get_ensembles_dirs", return_value=[]):
            with pytest.raises(FileNotFoundError, match="not found in any"):
                await server._load_ensemble_config("nonexistent_ensemble")

    @pytest.mark.asyncio
    @patch("llm_orc.integrations.mcp.server.MCPServer._load_ensemble_config")
    async def test_format_ensemble_result_with_errors(
        self, mock_load_config: AsyncMock
    ) -> None:
        """Should format ensemble results including error responses."""
        mock_executor = AsyncMock()
        mock_executor.execute.return_value = {
            "ensemble": "test_ensemble",
            "status": "completed",
            "results": {
                "success_agent": {"status": "success", "response": "Success"},
                "error_agent": {"status": "error", "error": "Something failed"},
            },
        }

        server = MCPServer("test_ensemble")
        server.executor = mock_executor

        # Mock ensemble loading
        mock_config = Mock()
        mock_load_config.return_value = mock_config

        call_request = {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {
                "name": "test_ensemble",
                "arguments": {"input": "test"},
            },
        }

        response = await server.handle_request(call_request)

        content = response["result"]["content"][0]["text"]
        assert "success_agent: Success" in content
        assert "error_agent: [Error: Something failed]" in content
