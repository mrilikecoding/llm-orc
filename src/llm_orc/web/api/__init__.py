"""API module for llm-orc web server."""

from llm_orc.mcp import MCPServer
from llm_orc.services import OrchestraService

_mcp_server: MCPServer | None = None
_orchestra_service: OrchestraService | None = None


def get_orchestra_service() -> OrchestraService:
    """Get or create the shared OrchestraService instance."""
    global _orchestra_service
    if _orchestra_service is None:
        _orchestra_service = OrchestraService()
    return _orchestra_service


def get_mcp_server() -> MCPServer:
    """Get or create the shared MCP server instance."""
    global _mcp_server
    if _mcp_server is None:
        _mcp_server = MCPServer(service=get_orchestra_service())
    return _mcp_server
