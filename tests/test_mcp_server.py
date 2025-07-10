"""Test suite for MCP server implementation."""

import json
from unittest.mock import AsyncMock, Mock

import pytest

from llm_orc.mcp_server import MCPServer


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
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            }
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
        
        tools_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list"
        }
        
        response = await server.handle_request(tools_request)
        
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 2
        assert "result" in response
        assert "tools" in response["result"]
        assert len(response["result"]["tools"]) == 1
        assert response["result"]["tools"][0]["name"] == "architecture_review"

    @pytest.mark.asyncio
    async def test_handle_tools_call_request(self) -> None:
        """Should execute ensemble when tool is called."""
        # Mock the ensemble executor
        mock_executor = AsyncMock()
        mock_executor.execute.return_value = {
            "ensemble": "architecture_review",
            "status": "completed",
            "results": {"agent1": {"response": "Test response", "status": "success"}},
            "synthesis": "Synthesized result"
        }
        
        server = MCPServer("architecture_review")
        server.executor = mock_executor
        
        # Mock ensemble loading
        mock_config = Mock()
        mock_config.name = "architecture_review"
        server._load_ensemble_config = AsyncMock(return_value=mock_config)
        
        call_request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "architecture_review",
                "arguments": {
                    "input": "Analyze this architecture design"
                }
            }
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
        
        invalid_request = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "invalid/method"
        }
        
        response = await server.handle_request(invalid_request)
        
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 4
        assert "error" in response
        assert response["error"]["code"] == -32601  # Method not found