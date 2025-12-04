# ADR-009: MCP Server Architecture and Plexus Integration

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

## Implementation Progress

### Phase 1: Core MCP Server (Complete - 2025-12-03)

**What was implemented:**

1. **MCPServerV2 class** (`src/llm_orc/mcp/server.py`)
   - Uses official FastMCP SDK with decorator-based registration
   - Resources registered via `@self._mcp.resource()` decorators
   - Tools registered via `@self._mcp.tool()` decorators
   - `run()` method exposes FastMCP's stdio transport

2. **Resources implemented:**
   - `llm-orc://ensembles` - List all ensembles with metadata
   - `llm-orc://profiles` - List model profiles
   - `llm-orc://ensemble/{name}` - Get specific ensemble config
   - `llm-orc://artifacts/{ensemble}` - List artifacts for ensemble
   - `llm-orc://artifact/{ensemble}/{id}` - Individual artifact details
   - `llm-orc://metrics/{ensemble}` - Aggregated metrics

3. **Tools implemented:**
   - `invoke` - Execute ensemble with input
   - `validate_ensemble` - Validate ensemble configuration
   - `list_ensembles` - List available ensembles

4. **CLI command:**
   - `llm-orc mcp serve` - Starts MCP server on stdio transport
   - `llm-orc mcp serve --transport http --port 8080` - HTTP option (not yet implemented)
   - Alias: `llm-orc m serve`

5. **Claude Code integration:**
   - `.mcp.json` created at project root
   - Configured to run `uv run llm-orc mcp serve`

6. **BDD tests:**
   - 23 scenarios passing in `tests/bdd/test_adr_009_mcp_server_plexus_integration.py`
   - Full coverage of resources, tools, streaming, and server lifecycle

### Phase 2: Tools and Streaming (Complete - 2025-12-03)

**Tools added:**
- `update_ensemble` - Registered with FastMCP decorator, supports dry_run and backup
- `analyze_execution` - Registered with FastMCP decorator for artifact analysis

**HTTP transport:**
- `llm-orc mcp serve --transport http --port 8080` now works
- Uses FastMCP's `sse_app()` with uvicorn

**Streaming improvements:**
- `invoke` tool now uses FastMCP Context for real-time progress
- `ctx.report_progress()` reports agent completion progress
- `ctx.info()`, `ctx.warning()`, `ctx.error()` for event logging
- Streams actual execution events from `EnsembleExecutor.execute_streaming()`

### Phase 2.5: Artifact Resource (Complete - 2025-12-03)

**Resource added:**
- `llm-orc://artifact/{ensemble}/{artifact_id}` - Individual artifact details
- Updated artifact directory structure to `{ensemble}/{id}/execution.json`
- Fixed metadata field mapping (started_at, duration, agents_used)

### Phase 3: Remaining Work (Future)

**Plexus integration:**
- `PlexusClient` class for post-execution callbacks
- Configuration in `.llm-orc/config.yaml`
- Optional feature flag

### Files Structure

```
src/llm_orc/mcp/
├── __init__.py          # Exports MCPServerV2
└── server.py            # MCPServerV2 with FastMCP integration (~1020 lines)

.mcp.json                # Claude Code MCP configuration
```

### How to Test

```bash
# Run BDD tests
uv run pytest tests/bdd/test_adr_009_mcp_server_plexus_integration.py -v

# Start MCP server manually
uv run llm-orc mcp serve

# Test with Claude Code
# Restart Claude Code to pick up .mcp.json, then tools should be available
```

### Claude Code MCP Validation Runbook

This runbook documents how Claude (via Claude Code) can validate the MCP integration
is working correctly. These steps use the actual MCP tools available in the session.

#### Prerequisites

- `.mcp.json` configured in project root
- Claude Code session started (MCP server auto-launches)
- Ollama running locally (for `validate-ollama` ensemble)

#### Step 1: Verify MCP Server Connection

List available ensembles to confirm the server is responding:

```
Tool: mcp__llm-orc__list_ensembles
Expected: JSON array of ensembles with name, source, agent_count, description
```

Example validation:
```json
[
  {"name": "validate-ollama", "source": "global", "agent_count": 1, ...},
  {"name": "security-review", "source": "global", "agent_count": 4, ...}
]
```

#### Step 2: Validate Ensemble Configuration (Passing)

Test the validation tool with a known-good ensemble:

```
Tool: mcp__llm-orc__validate_ensemble
Parameters: ensemble_name = "validate-ollama"
Expected: {"valid": true, "details": {"errors": [], "agent_count": 1}}
```

#### Step 3: Validate Ensemble Configuration (Failing)

Test the validation tool with an ensemble that has configuration errors:

```
Tool: mcp__llm-orc__validate_ensemble
Parameters: ensemble_name = "security-review"
Expected: {"valid": false, "details": {"errors": ["Agent '...' uses unknown profile 'default'"], "agent_count": 4}}
```

This verifies that validation catches real configuration issues like missing model profiles.

#### Step 4: Execute an Ensemble (With Artifact Storage)

Invoke a simple ensemble to test execution and streaming:

```
Tool: mcp__llm-orc__invoke
Parameters:
  ensemble_name = "validate-ollama"
  input_data = "What is 2+2? Answer with just the number."
Expected: {"results": {"validator": {"response": "4", "status": "success"}}, ...}
```

During execution, the MCP server reports progress via:
- `ctx.report_progress(progress, total)` - Agent completion count
- `ctx.info()` - Agent start/complete messages

**Important**: The MCP invoke tool now saves artifacts automatically. Verify in Step 7.

#### Step 5: Read MCP Resources

Test resource reading via the MCP resource tools:

```
Tool: ReadMcpResourceTool
Parameters:
  server = "llm-orc"
  uri = "llm-orc://ensembles"
Expected: JSON list of all ensembles
```

```
Tool: ReadMcpResourceTool
Parameters:
  server = "llm-orc"
  uri = "llm-orc://profiles"
Expected: JSON list of model profiles with provider/model info
```

#### Step 6: List Available Resources

Check what resources the MCP server advertises:

```
Tool: ListMcpResourcesTool
Parameters: server = "llm-orc"
Expected: Static resources (ensembles, profiles) - templated resources discovered separately
```

#### Step 7: Verify Artifact Storage (After Execution)

After running invoke in Step 4, verify artifacts were saved:

```bash
ls -la .llm-orc/artifacts/validate-ollama/
```

Expected structure:
```
validate-ollama/
├── {timestamp}/
│   ├── execution.json
│   └── execution.md
└── latest -> {timestamp}
```

You can also read the artifact content:
```bash
cat .llm-orc/artifacts/validate-ollama/latest/execution.json
```

Expected fields: `ensemble_name`, `input`, `timestamp`, `status`, `results`, `agents`

#### Validation Checklist

| Step | Check | Tool/Command | Pass Criteria |
|------|-------|--------------|---------------|
| 1 | Server responds | `list_ensembles` | Returns JSON array (68+ ensembles) |
| 2 | Validation passes | `validate_ensemble("validate-ollama")` | Returns `valid: true` |
| 3 | Validation catches errors | `validate_ensemble("security-review")` | Returns `valid: false` with errors |
| 4 | Execution works | `invoke` | Returns results with `status: success` |
| 5 | Resources readable | `ReadMcpResourceTool` | Returns valid JSON |
| 6 | Resource list works | `ListMcpResourcesTool` | Returns 2 resources |
| 7 | Artifacts stored | `ls .llm-orc/artifacts/` | Directory with execution.json exists |

#### Troubleshooting

**MCP tools not available:**
- Restart Claude Code to reload `.mcp.json`
- Check `.mcp.json` has correct command: `uv run llm-orc mcp serve`

**Invoke fails with model errors:**
- Ensure Ollama is running: `ollama list`
- Check model exists: `ollama pull llama3`

**Artifacts not found:**
- Verify working directory is project root
- Check `.llm-orc/artifacts/` exists

#### Example Validation Session

```
# 1. List ensembles
> mcp__llm-orc__list_ensembles
✓ Returns 68 ensembles from local/library/global sources

# 2. Validate good config
> mcp__llm-orc__validate_ensemble("validate-ollama")
✓ Returns valid: true, agent_count: 1

# 3. Validate bad config
> mcp__llm-orc__validate_ensemble("security-review")
✓ Returns valid: false, errors: ["Agent '...' uses unknown profile 'default'"]

# 4. Execute simple ensemble
> mcp__llm-orc__invoke("validate-ollama", "Say hello")
✓ Returns validator response with status: success

# 5. Read profiles resource
> ReadMcpResourceTool(server="llm-orc", uri="llm-orc://profiles")
✓ Returns list of model profiles

# 6. List resources
> ListMcpResourcesTool(server="llm-orc")
✓ Returns 2 resources (ensembles, profiles)

# 7. Verify artifacts
> ls .llm-orc/artifacts/validate-ollama/latest/
✓ execution.json and execution.md exist

All checks passed - MCP integration verified.
```

## References

- [MCP Specification](https://modelcontextprotocol.io/specification)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [Plexus Integration Spec](/Users/nathangreen/Development/manza/docs/specs/plexus/plexus-llm-orc-mcp-integration.md)
- [ADR-008: LLM-Friendly CLI and MCP Design](./008-llm-friendly-cli-mcp-design.md)
