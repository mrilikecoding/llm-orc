# ADR-009: MCP Server Architecture

## Status
Implemented

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

  - capability: "Validate ensemble before execution"
    given: "Ensemble name to validate"
    when: "Client calls validate_ensemble tool"
    then: "Returns validation status with errors for invalid profiles/dependencies"

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
    - streaming_execution_flow

validation_rules:
  - "Use official mcp Python SDK (FastMCP)"
  - "All resources use llm-orc:// URI scheme"
  - "Tools follow MCP inputSchema specification"
  - "Streaming uses MCP progress notifications"
  - "Type safety with Pydantic models for all payloads"
  - "Exception handling preserves MCP error codes"

related_adrs:
  - "ADR-008: LLM-Friendly CLI and MCP Design (MCP sections superseded by this ADR)"
  - "ADR-010: Local Web UI (can consume MCP resources)"

implementation_scope:
  - "src/llm_orc/mcp/"
  - "src/llm_orc/mcp/server.py"
```

## Context

llm-orc previously had a minimal MCP server implementation that:
- Exposed only ONE ensemble as ONE tool per server instance
- Used hand-rolled JSON-RPC instead of official MCP SDK
- Had no resource exposure (ensembles, artifacts, metrics were not readable)
- Had no streaming support for long-running executions
- Had hardcoded version strings

As ensemble usage grows, CLI-only interaction becomes limiting. MCP enables:
1. **Claude Code integration** - Direct ensemble execution from AI assistants
2. **GUI tools** - Visual ensemble management (Manza, custom UIs)
3. **Orchestration systems** - Higher-level systems (like Plexus) can invoke llm-orc
4. **Resource discovery** - Clients can browse ensembles, artifacts, and metrics

## Decision

### 1. Rewrite MCP Server Using FastMCP SDK

Replace hand-rolled implementation with FastMCP:

```python
from mcp.server.fastmcp import FastMCP, Context

mcp = FastMCP("llm-orc")

@mcp.resource("llm-orc://ensembles")
async def get_ensembles() -> str:
    """List all available ensembles."""
    ...

@mcp.tool()
async def invoke(ensemble_name: str, input_data: str, ctx: Context) -> str:
    """Execute an ensemble with streaming progress."""
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

llm-orc://artifact/{ensemble}/{id}
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
    description: Execute an ensemble with streaming progress
    inputSchema:
      ensemble_name: string (required)
      input_data: string (required)

  - name: list_ensembles
    description: List all available ensembles with metadata

  - name: validate_ensemble
    description: Validate ensemble configuration and model profiles
    inputSchema:
      ensemble_name: string (required)

  - name: update_ensemble
    description: Modify ensemble configuration
    inputSchema:
      ensemble_name: string (required)
      changes: object (required)
      dry_run: boolean (default: true)
      backup: boolean (default: true)

  - name: analyze_execution
    description: Analyze execution artifact
    inputSchema:
      artifact_id: string (required, format: ensemble/timestamp)
```

### 4. Streaming Support

Use FastMCP Context for real-time progress:

```python
@mcp.tool()
async def invoke(ensemble_name: str, input_data: str, ctx: Context) -> str:
    await ctx.info(f"Starting ensemble '{ensemble_name}'")

    async for event in executor.execute_streaming(config, input_data):
        if event["type"] == "agent_completed":
            await ctx.report_progress(completed, total)
            await ctx.info(f"Agent '{agent_name}' completed")

    return json.dumps(result)
```

### 5. Artifact Storage

The `invoke` tool automatically saves execution artifacts:
- `.llm-orc/artifacts/{ensemble}/{timestamp}/execution.json` - Full results
- `.llm-orc/artifacts/{ensemble}/{timestamp}/execution.md` - Human-readable report
- `.llm-orc/artifacts/{ensemble}/latest` - Symlink to most recent

### 6. CLI Integration

```bash
# Start MCP server (stdio transport for MCP clients)
llm-orc mcp serve

# Start with HTTP transport (for debugging/testing)
llm-orc mcp serve --transport http --port 8080

# Alias
llm-orc m serve
```

## Consequences

### Positive

1. **Full MCP compliance** - Works with any MCP client (Claude Code, Claude Desktop, etc.)
2. **Resource discoverability** - Clients can browse ensembles without prior knowledge
3. **Validation before execution** - `validate_ensemble` checks profiles and dependencies
4. **Streaming UX** - Real-time feedback for long executions
5. **Automatic artifact storage** - Every execution is recorded for analysis
6. **GUI enablement** - Foundation for visual ensemble management

### Negative

1. **New dependency** - Requires `mcp` Python package
2. **Breaking change** - Old `serve` command behavior changes
3. **Complexity** - More code to maintain than previous stub

### Neutral

1. **No built-in Plexus integration** - llm-orc is a tool, not an orchestrator. Higher-level systems like Plexus can call llm-orc via MCP if they want to track execution patterns and build knowledge graphs. This keeps llm-orc focused on ensemble execution.

## Implementation Status

### Complete (2025-12-03)

**MCPServerV2 class** (`src/llm_orc/mcp/server.py`):
- Uses FastMCP SDK with decorator-based registration
- All resources implemented and tested
- All tools implemented with streaming support
- HTTP transport option working

**Resources**:
- `llm-orc://ensembles` - Lists 68+ ensembles from local/library/global
- `llm-orc://profiles` - Lists 60+ model profiles
- `llm-orc://ensemble/{name}` - Returns ensemble YAML as JSON
- `llm-orc://artifacts/{ensemble}` - Lists execution history
- `llm-orc://artifact/{ensemble}/{id}` - Returns full artifact
- `llm-orc://metrics/{ensemble}` - Returns aggregated stats

**Tools**:
- `invoke` - Executes with streaming, saves artifacts
- `list_ensembles` - Returns all ensembles with metadata
- `validate_ensemble` - Validates config, profiles, and providers
- `update_ensemble` - Modifies config with dry-run and backup
- `analyze_execution` - Analyzes artifact data

**CLI**:
- `llm-orc mcp serve` - stdio transport
- `llm-orc mcp serve --transport http --port 8080` - HTTP/SSE transport

**Tests**:
- 23 BDD scenarios in `tests/bdd/test_adr_009_mcp_server_plexus_integration.py`

### Files

```
src/llm_orc/mcp/
├── __init__.py          # Exports MCPServerV2
└── server.py            # MCPServerV2 implementation (~1050 lines)

.mcp.json                # Claude Code MCP configuration
```

## How to Use

### Claude Code Integration

Add `.mcp.json` to project root:
```json
{
  "mcpServers": {
    "llm-orc": {
      "command": "uv",
      "args": ["run", "llm-orc", "mcp", "serve"]
    }
  }
}
```

Restart Claude Code. MCP tools become available as `mcp__llm-orc__*`.

### Validation Runbook

See the [Claude Code MCP Validation Runbook](#claude-code-mcp-validation-runbook) below.

## Potential Future Enhancements

These are NOT planned but could be added if needed:

1. **Script tools** - `scripts_list`, `scripts_test` (mentioned in ADR-008)
2. **Library tools** - `library_browse`, `library_copy` (mentioned in ADR-008)
3. **Init tool** - `init_project` (may not make sense for MCP)
4. **Artifact deletion** - `delete_artifact` tool

## Claude Code MCP Validation Runbook

This runbook validates the MCP integration is working correctly.

### Prerequisites

- `.mcp.json` configured in project root
- Claude Code session started (MCP server auto-launches)
- Ollama running locally (for `validate-ollama` ensemble)

### Step 1: List Ensembles

```
Tool: mcp__llm-orc__list_ensembles
Expected: JSON array with 68+ ensembles from local/library/global sources
```

### Step 2: Validate Good Config

```
Tool: mcp__llm-orc__validate_ensemble
Parameters: ensemble_name = "validate-ollama"
Expected: {"valid": true, "details": {"errors": [], "agent_count": 1}}
```

### Step 3: Validate Bad Config

```
Tool: mcp__llm-orc__validate_ensemble
Parameters: ensemble_name = "security-review"
Expected: {"valid": false, "details": {"errors": ["Agent '...' uses unknown profile 'default'"], ...}}
```

### Step 4: Execute Ensemble

```
Tool: mcp__llm-orc__invoke
Parameters:
  ensemble_name = "validate-ollama"
  input_data = "What is 2+2? Answer with just the number."
Expected: {"results": {"validator": {"response": "4", "status": "success"}}, "status": "completed"}
```

### Step 5: Read Resources

```
Tool: ReadMcpResourceTool
Parameters: server = "llm-orc", uri = "llm-orc://profiles"
Expected: JSON list of model profiles with provider/model info
```

### Step 6: List Resources

```
Tool: ListMcpResourcesTool
Parameters: server = "llm-orc"
Expected: 2 static resources (ensembles, profiles)
```

### Step 7: Verify Artifacts

```bash
ls -la .llm-orc/artifacts/validate-ollama/latest/
```

Expected: `execution.json` and `execution.md` exist.

### Validation Checklist

| Step | Check | Pass Criteria |
|------|-------|---------------|
| 1 | Server responds | Returns 68+ ensembles |
| 2 | Validation passes | `valid: true` for good config |
| 3 | Validation catches errors | `valid: false` with error details |
| 4 | Execution works | Returns results with `status: success` |
| 5 | Resources readable | Returns valid JSON |
| 6 | Resource list works | Returns 2 resources |
| 7 | Artifacts stored | Directory with execution.json exists |

## References

- [MCP Specification](https://modelcontextprotocol.io/specification)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [FastMCP Documentation](https://github.com/jlowin/fastmcp)
- [ADR-008: LLM-Friendly CLI and MCP Design](./008-llm-friendly-cli-mcp-design.md)
- [ADR-010: Local Web UI](./010-local-web-ui.md)
