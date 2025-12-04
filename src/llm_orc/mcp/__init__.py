"""MCP Server module for llm-orc.

This module provides Model Context Protocol (MCP) server implementation
using the official FastMCP SDK. It exposes llm-orc ensembles, artifacts,
and metrics as MCP resources, and provides tools for execution and management.

Architecture follows ADR-009: MCP Server Architecture and Plexus Integration.
"""

from llm_orc.mcp.server import MCPServerV2

__all__ = ["MCPServerV2"]
