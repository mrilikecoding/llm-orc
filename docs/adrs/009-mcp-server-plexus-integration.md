# ADR-009: MCP Server Architecture and Plexus Integration

## Status
Proposed

## BDD Mapping Hints
```yaml
behavioral_capabilities:
  - capability: "Expose all ensembles as MCP resources"
    given: "An MCP client connected to llm-orc server"
    when: "Client requests llm-orc://ensembles resource"
    then: "Returns list of all available ensembles with metadata"

  - capability: "Expose individual ensemble configuration"
    given: "A valid ensemble name"
    when: "Client requests llm-orc://ensemble/{name} resource"
    then: "Returns complete ensemble YAML config as structured data"

  - capability: "Expose execution artifacts"
    given: "An ensemble with execution history"
    when: "Client requests llm-orc://artifacts/{ensemble} resource"
    then: "Returns timestamped list of execution artifacts with metrics"

  - capability: "Invoke any ensemble via MCP tool"
    given: "MCP client with invoke tool access"
    when: "Client calls invoke with ensemble_name and input"
    then: "Ensemble executes and returns structured results"

  - capability: "Stream execution progress"
    given: "Long-running ensemble execution"
    when: "Client invokes with streaming enabled"
    then: "Progressive updates sent as execution proceeds"

  - capability: "Update ensemble configuration"
    given: "Valid ensemble and change specification"
    when: "Client calls update_ensemble tool"
    then: "Ensemble config modified with backup created"

  - capability: "Post-execution Plexus integration"
    given: "Plexus context configured for ensemble"
    when: "Ensemble execution completes"
    then: "Results sent to Plexus for edge reinforcement"

test_boundaries:
  unit:
    - MCPServer.list_resources()
    - MCPServer.read_resource()
    - MCPServer.list_tools()
    - MCPServer.call_tool()
    - ResourceFormatter.ensemble_to_resource()
    - ResourceFormatter.artifacts_to_resource()
    - ToolHandler.invoke_ensemble()
    - ToolHandler.update_ensemble()

  integration:
    - mcp_server_stdio_transport
    - mcp_resource_access_flow
    - mcp_tool_execution_flow
    - plexus_post_execution_hook
    - streaming_execution_flow

validation_rules:
  - "Use official mcp Python SDK, not hand-rolled JSON-RPC"
  - "All resources use llm-orc:// URI scheme"
  - "Tools follow MCP inputSchema specification"
  - "Streaming uses MCP progress notifications"
  - "Type safety with Pydantic models for all payloads"
  - "Exception handling preserves MCP error codes"

related_adrs:
  - "ADR-008: LLM-Friendly CLI and MCP Design (supersedes partial)"

implementation_scope:
  - "src/llm_orc/mcp/"
  - "src/llm_orc/mcp/server.py"
  - "src/llm_orc/mcp/resources.py"
  - "src/llm_orc/mcp/tools.py"
  - "src/llm_orc/mcp/plexus_client.py"
```

## Context

llm-orc currently has a minimal MCP server implementation that:
- Exposes only ONE ensemble as ONE tool per server instance
- Uses hand-rolled JSON-RPC instead of official MCP SDK
- Has no resource exposure (ensembles, artifacts, metrics are not readable)
- Has no streaming support for long-running executions
- Has hardcoded version strings

The Plexus integration spec (`plexus-llm-orc-mcp-integration.md`) requires llm-orc to:
1. Expose all ensembles, artifacts, and metrics as MCP resources
2. Provide tools for invoke, update_ensemble, and analyze_execution
3. Act as MCP client to call Plexus after executions for learning
4. Support streaming for real-time execution feedback

Additionally, as ensemble usage grows, CLI-only interaction becomes cumbersome. MCP enables GUI tools (like Manza) to interact with llm-orc programmatically.

## Decision

### 1. Rewrite MCP Server Using Official SDK

Replace hand-rolled implementation with `mcp` Python package:

```python
from mcp.server import Server
from mcp.types import Resource, Tool, TextContent

server = Server("llm-orc")

@server.list_resources()
async def list_resources() -> list[Resource]:
    # Return all ensembles, artifacts as resources
    ...

@server.read_resource()
async def read_resource(uri: str) -> str:
    # Parse llm-orc:// URI and return content
    ...

@server.list_tools()
async def list_tools() -> list[Tool]:
    # Return invoke, update_ensemble, analyze_execution
    ...

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    # Execute tool and return results
    ...
```

### 2. Resource Architecture

```
llm-orc://ensembles
  → List all available ensembles with metadata

llm-orc://ensemble/{name}
  → Complete ensemble configuration (YAML as structured JSON)

llm-orc://artifacts/{ensemble}
  → List of execution artifacts for ensemble

llm-orc://artifact/{id}
  → Single artifact with full results

llm-orc://metrics/{ensemble}
  → Aggregated metrics (success rate, avg cost, avg duration)

llm-orc://profiles
  → Available model profiles
```

### 3. Tool Architecture

```yaml
tools:
  - name: invoke
    description: Execute an ensemble
    inputSchema:
      ensemble_name: string (required)
      input: string (required)
      output_format: enum[text, json] (default: json)
      streaming: boolean (default: true)
      plexus_context: string (optional)

  - name: update_ensemble
    description: Modify ensemble configuration
    inputSchema:
      ensemble_name: string (required)
      changes:
        add_agents: array
        remove_agents: array
        modify_dependencies: array
        update_model_profiles: object
      dry_run: boolean (default: true)
      backup: boolean (default: true)

  - name: analyze_execution
    description: Analyze execution results
    inputSchema:
      artifact_id: string (required)
      plexus_feedback: boolean (default: true)

  - name: validate_ensemble
    description: Validate ensemble configuration
    inputSchema:
      ensemble_name: string (required)
```

### 4. Plexus Client Integration

Add optional Plexus MCP client that calls after execution:

```python
class PlexusClient:
    """MCP client for Plexus knowledge graph."""

    async def analyze_ensemble(
        self,
        context_id: str,
        ensemble_config: dict,
        execution_metadata: dict
    ) -> dict:
        """Send execution results to Plexus for learning."""
        ...
```

Configuration in `.llm-orc/config.yaml`:
```yaml
plexus:
  enabled: true
  context_id: "my-project"
  server_command: "plexus-mcp-server"
```

### 5. Streaming Support

Use MCP progress notifications for long-running executions:

```python
@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "invoke":
        async for progress in executor.execute_streaming(...):
            await server.send_progress(progress)
        return final_result
```

### 6. CLI Integration

Add `llm-orc mcp` subcommand:

```bash
# Start MCP server (stdio transport)
llm-orc mcp serve

# Start with HTTP transport (for debugging)
llm-orc mcp serve --http --port 8080

# Check MCP server health
llm-orc mcp status
```

## Consequences

### Positive

1. **Full MCP compliance** - Works with any MCP client (Claude Desktop, Manza, etc.)
2. **Resource discoverability** - Clients can browse ensembles without prior knowledge
3. **Plexus learning loop** - Executions feed back into knowledge graph
4. **GUI enablement** - Foundation for visual ensemble management
5. **Streaming UX** - Real-time feedback for long executions

### Negative

1. **New dependency** - Requires `mcp` Python package
2. **Breaking change** - Old `serve` command behavior changes
3. **Complexity** - More code to maintain than current stub
4. **Plexus coupling** - Optional but adds integration surface

### Migration Path

1. **Phase 1**: Implement new MCP server alongside existing `serve` command
2. **Phase 2**: Deprecate old `serve`, add `mcp serve` as primary
3. **Phase 3**: Add Plexus client (optional feature)
4. **Phase 4**: Remove old implementation

## Implementation Notes

### Dependencies

```toml
[project.dependencies]
mcp = ">=1.0.0"  # Official MCP Python SDK
```

### File Structure

```
src/llm_orc/mcp/
├── __init__.py
├── server.py          # Main MCP server
├── resources.py       # Resource handlers
├── tools.py           # Tool handlers
├── plexus_client.py   # Optional Plexus integration
└── schemas.py         # Pydantic models for MCP payloads
```

### Testing Strategy

- Unit tests for each resource/tool handler
- Integration tests with mock MCP client
- End-to-end tests with real stdio transport
- Plexus integration tests (when Plexus MCP exists)

## References

- [MCP Specification](https://modelcontextprotocol.io/specification)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [Plexus Integration Spec](/Users/nathangreen/Development/manza/docs/specs/plexus/plexus-llm-orc-mcp-integration.md)
- [ADR-008: LLM-Friendly CLI and MCP Design](./008-llm-friendly-cli-mcp-design.md)
