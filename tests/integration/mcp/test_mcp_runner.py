"""Comprehensive tests for MCP server runners."""

import json
from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from aiohttp import web

from llm_orc.integrations.mcp.runner import MCPServerRunner, MCPStdioRunner


@pytest.fixture(autouse=True)
def mock_expensive_mcp_dependencies() -> Generator[None, None, None]:
    """Mock expensive MCP dependencies for all runner tests."""
    with patch("llm_orc.integrations.mcp.runner.MCPServer"):
        yield


class TestMCPServerRunner:
    """Test HTTP-based MCP server runner."""

    def test_init(self) -> None:
        """Test MCPServerRunner initialization."""
        # Given
        ensemble_name = "test-ensemble"
        port = 8080

        # When
        runner = MCPServerRunner(ensemble_name, port)

        # Then
        assert runner.ensemble_name == ensemble_name
        assert runner.port == port
        assert runner.mcp_server is not None

    @patch("llm_orc.integrations.mcp.runner.asyncio.run")
    def test_run(self, mock_asyncio_run: Mock) -> None:
        """Test run method starts server."""
        # Given
        runner = MCPServerRunner("test-ensemble", 8080)

        # Mock asyncio.run to close the coroutine to prevent warnings
        def close_coro(coro: Any) -> None:
            coro.close()

        mock_asyncio_run.side_effect = close_coro

        # When
        runner.run()

        # Then
        mock_asyncio_run.assert_called_once()

    @patch("llm_orc.integrations.mcp.runner.web.AppRunner")
    @patch("llm_orc.integrations.mcp.runner.web.TCPSite")
    @patch("builtins.print")
    async def test_run_server_setup(
        self, mock_print: Mock, mock_tcp_site: Mock, mock_app_runner: Mock
    ) -> None:
        """Test server setup and configuration."""
        # Given
        runner = MCPServerRunner("test-ensemble", 8080)
        mock_runner_instance = Mock()
        mock_app_runner.return_value = mock_runner_instance
        mock_runner_instance.setup = AsyncMock()
        mock_runner_instance.cleanup = AsyncMock()

        mock_site_instance = Mock()
        mock_tcp_site.return_value = mock_site_instance
        mock_site_instance.start = AsyncMock()

        # When
        with patch("asyncio.sleep", side_effect=KeyboardInterrupt):
            await runner._run_server()

        # Then
        mock_app_runner.assert_called_once()
        mock_runner_instance.setup.assert_called_once()
        mock_tcp_site.assert_called_once_with(mock_runner_instance, "localhost", 8080)
        mock_site_instance.start.assert_called_once()
        mock_runner_instance.cleanup.assert_called_once()

    async def test_handle_http_request_success(self) -> None:
        """Test successful HTTP request handling."""
        # Given
        runner = MCPServerRunner("test-ensemble", 8080)
        mock_request = Mock()
        mock_request.json = AsyncMock(
            return_value={"jsonrpc": "2.0", "id": 1, "method": "test"}
        )

        # When
        with (
            patch.object(
                runner.mcp_server,
                "handle_request",
                new=AsyncMock(
                    return_value={"jsonrpc": "2.0", "id": 1, "result": "success"}
                ),
            ) as mock_handle,
            patch(
                "llm_orc.integrations.mcp.runner.web.json_response"
            ) as mock_json_response,
        ):
            mock_json_response.return_value = web.Response()
            await runner._handle_http_request(mock_request)

            # Then
            mock_handle.assert_called_once_with(
                {"jsonrpc": "2.0", "id": 1, "method": "test"}
            )
            mock_json_response.assert_called_once_with(
                {"jsonrpc": "2.0", "id": 1, "result": "success"}
            )

    async def test_handle_http_request_json_decode_error(self) -> None:
        """Test HTTP request handling with JSON decode error."""
        # Given
        runner = MCPServerRunner("test-ensemble", 8080)

        mock_request = Mock()
        mock_request.json = AsyncMock(side_effect=json.JSONDecodeError("", "", 0))

        # When
        with patch(
            "llm_orc.integrations.mcp.runner.web.json_response"
        ) as mock_json_response:
            mock_json_response.return_value = web.Response()
            await runner._handle_http_request(mock_request)

            # Then
            expected_error = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error"},
            }
            mock_json_response.assert_called_once_with(expected_error, status=400)

    async def test_handle_http_request_internal_error(self) -> None:
        """Test HTTP request handling with internal error."""
        # Given
        runner = MCPServerRunner("test-ensemble", 8080)
        mock_request = Mock()
        mock_request.json = AsyncMock(
            return_value={"jsonrpc": "2.0", "id": 1, "method": "test"}
        )

        # When
        with (
            patch.object(
                runner.mcp_server,
                "handle_request",
                new=AsyncMock(side_effect=Exception("Internal error")),
            ),
            patch(
                "llm_orc.integrations.mcp.runner.web.json_response"
            ) as mock_json_response,
        ):
            mock_json_response.return_value = web.Response()
            await runner._handle_http_request(mock_request)

            # Then
            expected_error = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32603,
                    "message": "Internal error: Internal error",
                },
            }
            mock_json_response.assert_called_once_with(expected_error, status=500)


class TestMCPStdioRunner:
    """Test stdio-based MCP server runner."""

    def test_init(self) -> None:
        """Test MCPStdioRunner initialization."""
        # Given
        ensemble_name = "test-ensemble"

        # When
        runner = MCPStdioRunner(ensemble_name)

        # Then
        assert runner.ensemble_name == ensemble_name
        assert runner.mcp_server is not None

    @patch("llm_orc.integrations.mcp.runner.asyncio.run")
    def test_run(self, mock_asyncio_run: Mock) -> None:
        """Test run method starts stdio server."""
        # Given
        runner = MCPStdioRunner("test-ensemble")

        # Mock asyncio.run to close the coroutine to prevent warnings
        def close_coro(coro: Any) -> None:
            coro.close()

        mock_asyncio_run.side_effect = close_coro

        # When
        runner.run()

        # Then
        mock_asyncio_run.assert_called_once()

    @patch("sys.stdin")
    @patch("builtins.print")
    async def test_run_stdio_success(self, mock_print: Mock, mock_stdin: Mock) -> None:
        """Test successful stdio request processing."""
        # Given
        runner = MCPStdioRunner("test-ensemble")

        # Mock stdin to return one line then EOF
        mock_stdin.readline.side_effect = [
            '{"jsonrpc": "2.0", "id": 1, "method": "test"}\n',
            "",  # EOF
        ]

        # When
        with patch.object(
            runner.mcp_server,
            "handle_request",
            new=AsyncMock(
                return_value={"jsonrpc": "2.0", "id": 1, "result": "success"}
            ),
        ) as mock_handle:
            await runner._run_stdio()

            # Then
            mock_handle.assert_called_once_with(
                {"jsonrpc": "2.0", "id": 1, "method": "test"}
            )
            mock_print.assert_called_once_with(
                '{"jsonrpc": "2.0", "id": 1, "result": "success"}', flush=True
            )

    @patch("sys.stdin")
    @patch("builtins.print")
    async def test_run_stdio_json_decode_error(
        self, mock_print: Mock, mock_stdin: Mock
    ) -> None:
        """Test stdio handling with JSON decode error."""
        # Given
        runner = MCPStdioRunner("test-ensemble")

        # Mock stdin to return invalid JSON then EOF
        mock_stdin.readline.side_effect = [
            "invalid json\n",
            "",  # EOF
        ]

        # When
        await runner._run_stdio()

        # Then
        expected_error = {
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32700, "message": "Parse error"},
        }
        mock_print.assert_called_once_with(json.dumps(expected_error), flush=True)

    @patch("sys.stdin")
    @patch("builtins.print")
    async def test_run_stdio_internal_error(
        self, mock_print: Mock, mock_stdin: Mock
    ) -> None:
        """Test stdio handling with internal error."""
        # Given
        runner = MCPStdioRunner("test-ensemble")

        # Mock stdin to return valid JSON then EOF
        mock_stdin.readline.side_effect = [
            '{"jsonrpc": "2.0", "id": 1, "method": "test"}\n',
            "",  # EOF
        ]

        # When
        with patch.object(
            runner.mcp_server,
            "handle_request",
            new=AsyncMock(side_effect=Exception("Internal error")),
        ):
            await runner._run_stdio()

            # Then
            expected_error = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32603, "message": "Internal error: Internal error"},
            }
            mock_print.assert_called_once_with(json.dumps(expected_error), flush=True)

    @patch("sys.stdin")
    async def test_run_stdio_keyboard_interrupt(self, mock_stdin: Mock) -> None:
        """Test stdio handling with KeyboardInterrupt."""
        # Given
        runner = MCPStdioRunner("test-ensemble")

        # Mock stdin to raise KeyboardInterrupt
        mock_stdin.readline.side_effect = KeyboardInterrupt()

        # When / Then - should not raise exception
        await runner._run_stdio()

    @patch("sys.stdin")
    @patch("builtins.print")
    async def test_run_stdio_multiple_requests(
        self, mock_print: Mock, mock_stdin: Mock
    ) -> None:
        """Test stdio handling of multiple requests."""
        # Given
        runner = MCPStdioRunner("test-ensemble")

        # Mock stdin to return multiple requests then EOF
        mock_stdin.readline.side_effect = [
            '{"jsonrpc": "2.0", "id": 1, "method": "test1"}\n',
            '{"jsonrpc": "2.0", "id": 2, "method": "test2"}\n',
            "",  # EOF
        ]

        # When
        with patch.object(
            runner.mcp_server,
            "handle_request",
            new=AsyncMock(
                side_effect=[
                    {"jsonrpc": "2.0", "id": 1, "result": "first"},
                    {"jsonrpc": "2.0", "id": 2, "result": "second"},
                ]
            ),
        ) as mock_handle:
            await runner._run_stdio()

            # Then
            assert mock_handle.call_count == 2
            mock_handle.assert_any_call({"jsonrpc": "2.0", "id": 1, "method": "test1"})
            mock_handle.assert_any_call({"jsonrpc": "2.0", "id": 2, "method": "test2"})

            expected_calls = [
                (('{"jsonrpc": "2.0", "id": 1, "result": "first"}',), {"flush": True}),
                (('{"jsonrpc": "2.0", "id": 2, "result": "second"}',), {"flush": True}),
            ]
            assert mock_print.call_args_list == expected_calls


class TestMCPRunnerIntegration:
    """Fast integration tests for MCP runners using mocks instead of real server."""

    @patch("llm_orc.integrations.mcp.runner.web.json_response")
    async def test_integration_http_success(self, mock_json_response: Mock) -> None:
        """Test integration of HTTP request handling without real server."""
        # Given
        runner = MCPServerRunner("test-ensemble", 8080)

        mock_response = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": "integration_success",
        }
        mock_json_response.return_value = web.Response()

        request_data = {"jsonrpc": "2.0", "id": 1, "method": "test_integration"}
        mock_request = Mock()
        mock_request.json = AsyncMock(return_value=request_data)

        # Mock the server's handle_request method using patch.object
        with patch.object(
            runner.mcp_server, "handle_request", AsyncMock(return_value=mock_response)
        ) as mock_handle:
            # When
            await runner._handle_http_request(mock_request)

            # Then
            mock_handle.assert_called_once_with(request_data)
            mock_json_response.assert_called_once_with(mock_response)

    @patch("llm_orc.integrations.mcp.runner.web.json_response")
    async def test_integration_http_invalid_json(
        self, mock_json_response: Mock
    ) -> None:
        """Test integration with invalid JSON without real server."""
        # Given
        runner = MCPServerRunner("test-ensemble", 8080)
        mock_json_response.return_value = web.Response()

        mock_request = Mock()
        mock_request.json = AsyncMock(side_effect=json.JSONDecodeError("", "", 0))

        # When
        await runner._handle_http_request(mock_request)

        # Then
        expected_error = {
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32700, "message": "Parse error"},
        }
        mock_json_response.assert_called_once_with(expected_error, status=400)
