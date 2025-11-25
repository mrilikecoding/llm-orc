"""Integration tests for MCP server with ensemble loading."""

from collections.abc import Generator
from unittest.mock import AsyncMock, Mock, patch

import pytest

from llm_orc.integrations.mcp.server import MCPServer


@pytest.fixture(autouse=True)
def mock_expensive_dependencies() -> Generator[None, None, None]:
    """Mock expensive dependencies for all MCP server integration tests."""
    with patch("llm_orc.integrations.mcp.server.EnsembleExecutor"):
        with patch("llm_orc.integrations.mcp.server.ConfigurationManager"):
            with patch("llm_orc.integrations.mcp.server.EnsembleLoader"):
                yield


class TestMCPServerIntegration:
    """Test MCP server integration with real ensemble loading."""

    @pytest.mark.asyncio
    @patch("llm_orc.integrations.mcp.server.EnsembleLoader")
    @patch("llm_orc.integrations.mcp.server.ConfigurationManager")
    async def test_load_ensemble_config_from_file(
        self, mock_config_manager_class: Mock, mock_loader_class: Mock
    ) -> None:
        """Should load ensemble configuration from file."""
        # Mock configuration manager
        mock_config_manager = Mock()
        mock_config_manager.get_ensembles_dirs.return_value = ["/test/ensembles"]
        mock_config_manager_class.return_value = mock_config_manager

        # Mock ensemble loader
        mock_loader = Mock()
        mock_config = Mock()
        mock_config.name = "architecture_review"
        mock_loader.find_ensemble.return_value = mock_config
        mock_loader_class.return_value = mock_loader

        server = MCPServer("architecture_review")
        config = await server._load_ensemble_config("architecture_review")

        assert config == mock_config
        mock_loader.find_ensemble.assert_called_once_with(
            "/test/ensembles", "architecture_review"
        )

    @pytest.mark.asyncio
    @patch("llm_orc.integrations.mcp.server.MCPServer._load_ensemble_config")
    async def test_end_to_end_tools_call_with_mock_ensemble(
        self, mock_load_config: AsyncMock
    ) -> None:
        """Should execute complete tools/call flow with mocked ensemble."""
        server = MCPServer("test_ensemble")

        # Mock ensemble loading
        mock_config = Mock()
        mock_config.name = "test_ensemble"
        mock_load_config.return_value = mock_config

        # Mock executor
        mock_executor = AsyncMock()
        mock_executor.execute.return_value = {
            "ensemble": "test_ensemble",
            "status": "completed",
            "results": {"agent1": {"response": "Test response", "status": "success"}},
            "synthesis": "Test synthesis",
        }
        server.executor = mock_executor

        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "test_ensemble", "arguments": {"input": "Test input"}},
        }

        response = await server.handle_request(request)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert "result" in response
        assert "content" in response["result"]
        assert len(response["result"]["content"]) == 1
        assert response["result"]["content"][0]["type"] == "text"

        # Verify content contains expected information
        content = response["result"]["content"][0]["text"]
        assert "test_ensemble" in content
        assert "completed" in content
        assert "Test response" in content
        assert "Test synthesis" in content

        # Verify calls
        mock_load_config.assert_called_once_with("test_ensemble")
        mock_executor.execute.assert_called_once_with(mock_config, "Test input")
