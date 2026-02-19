"""API module for llm-orc web server."""

from llm_orc.mcp import MCPServer

_mcp_server: MCPServer | None = None


def get_mcp_server() -> MCPServer:
    """Get or create the shared MCP server instance."""
    global _mcp_server
    if _mcp_server is None:
        _mcp_server = MCPServer()
    return _mcp_server
