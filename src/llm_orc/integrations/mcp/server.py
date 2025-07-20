"""MCP server implementation for exposing llm-orc ensembles."""

from typing import Any

from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.core.config.ensemble_config import EnsembleConfig, EnsembleLoader
from llm_orc.ensemble_execution import EnsembleExecutor


class MCPServer:
    """MCP server that exposes llm-orc ensembles as tools."""

    def __init__(self, ensemble_name: str) -> None:
        """Initialize MCP server with ensemble name."""
        self.ensemble_name = ensemble_name
        self.executor = EnsembleExecutor()
        self.config_manager = ConfigurationManager()
        self.ensemble_loader = EnsembleLoader()

    async def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle incoming MCP request."""
        method = request.get("method")
        request_id = request.get("id")

        # Handle missing request_id
        if request_id is None:
            return self._error_response(0, -32600, "Invalid request - missing id")

        if method == "initialize":
            return await self._handle_initialize(request_id)
        elif method == "tools/list":
            return await self._handle_tools_list(request_id)
        elif method == "tools/call":
            return await self._handle_tools_call(request_id, request.get("params", {}))
        else:
            return self._error_response(request_id, -32601, "Method not found")

    async def _handle_initialize(self, request_id: int) -> dict[str, Any]:
        """Handle initialize request."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "llm-orc", "version": "0.3.0"},
            },
        }

    async def _handle_tools_list(self, request_id: int) -> dict[str, Any]:
        """Handle tools/list request."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "tools": [
                    {
                        "name": self.ensemble_name,
                        "description": f"Execute {self.ensemble_name} ensemble",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "input": {
                                    "type": "string",
                                    "description": "Input data for the ensemble",
                                }
                            },
                            "required": ["input"],
                        },
                    }
                ]
            },
        }

    async def _handle_tools_call(
        self, request_id: int, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle tools/call request."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if tool_name != self.ensemble_name:
            return self._error_response(request_id, -32602, "Invalid tool name")

        input_data = arguments.get("input", "")

        # Load ensemble config and execute
        config = await self._load_ensemble_config(self.ensemble_name)
        if config is None:
            return self._error_response(
                request_id, -32602, f"Ensemble '{self.ensemble_name}' not found"
            )

        result = await self.executor.execute(config, input_data)

        # Format result as MCP tool response
        content = self._format_ensemble_result(result)

        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"content": [{"type": "text", "text": content}]},
        }

    async def _load_ensemble_config(self, ensemble_name: str) -> EnsembleConfig | None:
        """Load ensemble configuration."""
        # Search in configured ensemble directories
        ensemble_dirs = self.config_manager.get_ensembles_dirs()

        for ensemble_dir in ensemble_dirs:
            config = self.ensemble_loader.find_ensemble(
                str(ensemble_dir), ensemble_name
            )
            if config:
                return config

        # If not found in any directory, raise error
        raise FileNotFoundError(
            f"Ensemble '{ensemble_name}' not found in any configured directory"
        )

    def _format_ensemble_result(self, result: dict[str, Any]) -> str:
        """Format ensemble execution result for MCP response."""
        output = f"Ensemble: {result.get('ensemble', 'unknown')}\n"
        output += f"Status: {result.get('status', 'unknown')}\n\n"

        # Add agent results
        results = result.get("results", {})
        for agent_name, agent_result in results.items():
            if agent_result.get("status") == "success":
                output += f"{agent_name}: {agent_result.get('response', '')}\n\n"
            else:
                error_msg = agent_result.get("error", "Unknown error")
                output += f"{agent_name}: [Error: {error_msg}]\n\n"

        # Add synthesis if available
        synthesis = result.get("synthesis")
        if synthesis:
            output += f"Synthesis: {synthesis}\n"

        return output

    def _error_response(
        self, request_id: int, code: int, message: str
    ) -> dict[str, Any]:
        """Create error response."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }
