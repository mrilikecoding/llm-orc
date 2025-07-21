"""MCP server runner for HTTP and stdio transports."""

import asyncio
import json
import sys

from aiohttp import web

from llm_orc.integrations.mcp.server import MCPServer


class MCPServerRunner:
    """Runs MCP server with HTTP transport."""

    def __init__(self, ensemble_name: str, port: int) -> None:
        """Initialize MCP server runner."""
        self.ensemble_name = ensemble_name
        self.port = port
        self.mcp_server = MCPServer(ensemble_name)

    def run(self) -> None:
        """Start the MCP server."""
        print(
            f"Starting MCP server for ensemble '{self.ensemble_name}' "
            f"on port {self.port}"
        )
        asyncio.run(self._run_server())

    async def _run_server(self) -> None:
        """Run the HTTP server."""
        app = web.Application()
        app.router.add_post("/mcp", self._handle_http_request)

        runner = web.AppRunner(app)
        await runner.setup()

        site = web.TCPSite(runner, "localhost", self.port)
        await site.start()

        print(f"MCP server running at http://localhost:{self.port}/mcp")
        print("Press Ctrl+C to stop")

        try:
            # Keep server running
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down MCP server...")
        finally:
            await runner.cleanup()

    async def _handle_http_request(self, request: web.Request) -> web.Response:
        """Handle HTTP MCP request."""
        try:
            # Parse JSON request
            request_data = await request.json()

            # Process MCP request
            response_data = await self.mcp_server.handle_request(request_data)

            # Return JSON response
            return web.json_response(response_data)

        except json.JSONDecodeError:
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error"},
            }
            return web.json_response(error_response, status=400)
        except Exception as e:
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32603, "message": f"Internal error: {str(e)}"},
            }
            return web.json_response(error_response, status=500)


class MCPStdioRunner:
    """Runs MCP server with stdio transport."""

    def __init__(self, ensemble_name: str) -> None:
        """Initialize MCP stdio runner."""
        self.ensemble_name = ensemble_name
        self.mcp_server = MCPServer(ensemble_name)

    def run(self) -> None:
        """Start the MCP server over stdio."""
        asyncio.run(self._run_stdio())

    async def _run_stdio(self) -> None:
        """Run the stdio server."""
        try:
            while True:
                # Read request from stdin
                line = sys.stdin.readline()
                if not line:
                    break

                try:
                    request_data = json.loads(line.strip())
                    response_data = await self.mcp_server.handle_request(request_data)

                    # Write response to stdout
                    print(json.dumps(response_data), flush=True)

                except json.JSONDecodeError:
                    error_response = {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32700, "message": "Parse error"},
                    }
                    print(json.dumps(error_response), flush=True)
                except Exception as e:
                    error_response = {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {
                            "code": -32603,
                            "message": f"Internal error: {str(e)}",
                        },
                    }
                    print(json.dumps(error_response), flush=True)
        except KeyboardInterrupt:
            pass
