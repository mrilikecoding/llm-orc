# ADR-010: Local Web UI for Ensemble Management

## Status
Proposed

## BDD Mapping Hints
```yaml
behavioral_capabilities:
  - capability: "Browse ensembles in web interface"
    given: "Web UI running at localhost"
    when: "User navigates to ensembles page"
    then: "Displays grouped list of local, library, and global ensembles"

  - capability: "View ensemble configuration"
    given: "User selects an ensemble"
    when: "Ensemble detail page loads"
    then: "Shows YAML config, dependency graph, and agent details"

  - capability: "Execute ensemble from UI"
    given: "User on ensemble detail page"
    when: "User enters input and clicks execute"
    then: "Execution runs with real-time streaming output"

  - capability: "Browse execution artifacts"
    given: "Ensemble with execution history"
    when: "User navigates to artifacts tab"
    then: "Shows timestamped list with cost, duration, status"

  - capability: "View artifact details"
    given: "User selects an artifact"
    when: "Artifact detail view loads"
    then: "Shows formatted agent outputs, synthesis, and metrics"

  - capability: "Compare artifacts"
    given: "User selects two artifacts"
    when: "User clicks compare"
    then: "Side-by-side diff view of outputs and metrics"

  - capability: "Visualize dependency graph"
    given: "Ensemble with agent dependencies"
    when: "Graph view is rendered"
    then: "Interactive DAG showing agents and data flow"

  - capability: "View execution metrics dashboard"
    given: "Ensembles with execution history"
    when: "User navigates to metrics page"
    then: "Shows aggregated stats: success rate, avg cost, trends"

test_boundaries:
  unit:
    - APIRouter.list_ensembles()
    - APIRouter.get_ensemble()
    - APIRouter.execute_ensemble()
    - APIRouter.list_artifacts()
    - ArtifactFormatter.to_html()
    - GraphBuilder.build_dependency_graph()

  integration:
    - web_server_startup_shutdown
    - ensemble_execution_streaming
    - artifact_retrieval_flow
    - websocket_progress_updates

validation_rules:
  - "API endpoints follow REST conventions"
  - "WebSocket for streaming execution updates"
  - "Static assets served efficiently (hashed filenames)"
  - "CORS disabled (localhost only)"
  - "No authentication required (local-only server)"

related_adrs:
  - "ADR-009: MCP Server Architecture (shares backend logic, provides resources)"
  - "ADR-008: LLM-Friendly CLI (consistent data formats)"

implementation_scope:
  - "src/llm_orc/web/"
  - "src/llm_orc/web/server.py"
  - "src/llm_orc/web/api.py"
  - "src/llm_orc/web/static/"
  - "src/llm_orc/cli.py (add web command)"
```

## Context

As llm-orc usage grows beyond simple CLI invocations, managing ensembles becomes cumbersome:

- **Artifact comparison**: CLI requires manual file diffing
- **Dependency visualization**: YAML doesn't convey graph structure intuitively
- **Execution monitoring**: Streaming output in terminal is hard to navigate
- **Metrics tracking**: No aggregated view of cost/performance over time
- **Configuration iteration**: Edit YAML → run → check output cycle is slow

A local web UI provides immediate value for:
- Visual ensemble management without needing external tools
- Artifact browsing and comparison
- Execution monitoring with streaming output
- Metrics tracking over time

The MCP server (ADR-009) exposes the same data as resources, enabling external tools like Manza to integrate. The local web UI serves users who want a standalone experience.

## Decision

### 1. Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                    Browser                          │
│  ┌───────────────────────────────────────────────┐  │
│  │              React/Preact SPA                 │  │
│  │  • Ensemble browser                           │  │
│  │  • Artifact viewer                            │  │
│  │  • Dependency graph (D3/Dagre)                │  │
│  │  • Execution console                          │  │
│  │  • Metrics dashboard                          │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────┬───────────────────────────────┘
                      │ HTTP/WebSocket
                      │ localhost:8765
┌─────────────────────▼───────────────────────────────┐
│                 llm-orc web server                  │
│  ┌─────────────────┐  ┌─────────────────────────┐   │
│  │   REST API      │  │   WebSocket Handler     │   │
│  │   /api/...      │  │   /ws/execute           │   │
│  └────────┬────────┘  └────────────┬────────────┘   │
│           │                        │                │
│  ┌────────▼────────────────────────▼────────────┐   │
│  │           Existing llm-orc Core              │   │
│  │  • ConfigurationManager                      │   │
│  │  • EnsembleExecutor                          │   │
│  │  • ArtifactManager                           │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

### 2. CLI Command

```bash
# Start web UI (default port 8765)
llm-orc web

# Custom port
llm-orc web --port 3000

# Open browser automatically
llm-orc web --open

# Bind to specific host (default: 127.0.0.1)
llm-orc web --host 0.0.0.0  # WARNING: exposes to network
```

### 3. REST API Design

The REST API maps directly to MCP tools from ADR-009, enabling consistent behavior across interfaces.

```yaml
# Ensembles (CRUD)
GET    /api/ensembles                    # list_ensembles tool
GET    /api/ensembles/{name}             # llm-orc://ensemble/{name} resource
POST   /api/ensembles                    # create_ensemble tool
PUT    /api/ensembles/{name}             # update_ensemble tool
DELETE /api/ensembles/{name}             # delete_ensemble tool
POST   /api/ensembles/{name}/execute     # invoke tool
POST   /api/ensembles/{name}/validate    # validate_ensemble tool

# Profiles (CRUD)
GET    /api/profiles                     # llm-orc://profiles resource
POST   /api/profiles                     # create_profile tool
PUT    /api/profiles/{name}              # update_profile tool
DELETE /api/profiles/{name}              # delete_profile tool

# Scripts (CRUD)
GET    /api/scripts                      # list_scripts tool
GET    /api/scripts/{category}/{name}    # get_script tool
POST   /api/scripts                      # create_script tool
DELETE /api/scripts/{category}/{name}    # delete_script tool
POST   /api/scripts/{category}/{name}/test  # test_script tool

# Library (Read-only)
GET    /api/library                      # library_browse tool
GET    /api/library/search?q=...         # library_search tool
GET    /api/library/{path}               # library_info tool
POST   /api/library/copy                 # library_copy tool

# Artifacts
GET    /api/artifacts                    # List all (paginated)
GET    /api/artifacts/{ensemble}         # llm-orc://artifacts/{ensemble} resource
GET    /api/artifacts/{ensemble}/{id}    # llm-orc://artifact/{ensemble}/{id} resource
DELETE /api/artifacts/{ensemble}/{id}    # delete_artifact tool
POST   /api/artifacts/cleanup            # cleanup_artifacts tool
POST   /api/artifacts/{id}/analyze       # analyze_execution tool

# Metrics
GET    /api/metrics                      # Global summary
GET    /api/metrics/{ensemble}           # llm-orc://metrics/{ensemble} resource

# Config (Read-only)
GET    /api/config                       # Current configuration
```

### 3.1 MCP Tool Mapping

The Web UI backend calls MCP tools internally, ensuring consistent behavior:

```python
from llm_orc.mcp import MCPServerV2

class EnsemblesAPI:
    def __init__(self):
        self.mcp = MCPServerV2()

    async def list_ensembles(self) -> list[dict]:
        """GET /api/ensembles"""
        return await self.mcp._read_ensembles_resource()

    async def create_ensemble(self, data: dict) -> dict:
        """POST /api/ensembles"""
        return await self.mcp._create_ensemble_tool(data)

    async def execute_ensemble(self, name: str, input_data: str) -> dict:
        """POST /api/ensembles/{name}/execute"""
        # Uses streaming internally, returns final result
        return await self.mcp._invoke_tool(name, input_data)
```

This approach:
- Reuses all MCP validation and business logic
- Ensures CLI, MCP, and Web UI behave identically
- Reduces code duplication

### 4. WebSocket for Streaming Execution

```typescript
// Client connects to /ws/execute
const ws = new WebSocket('ws://localhost:8765/ws/execute');

// Send execution request
ws.send(JSON.stringify({
  ensemble: 'code-review',
  input: 'Review this code...'
}));

// Receive streaming updates
ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  switch (msg.type) {
    case 'agent_start':
      // Agent beginning execution
      break;
    case 'agent_progress':
      // Streaming token output
      break;
    case 'agent_complete':
      // Agent finished
      break;
    case 'execution_complete':
      // Full result with artifact_id
      break;
    case 'error':
      // Execution error
      break;
  }
};
```

### 5. Frontend Pages

#### 5.1 Ensembles Page (`/`)
- Tree view grouped by source (Local, Library, Global)
- Search/filter functionality
- Quick actions: Execute, View, Copy, Delete
- **Create button**: Opens ensemble editor (new or from template)
- **Import from Library**: Browse and copy library ensembles

#### 5.2 Ensemble Detail (`/ensemble/{name}`)
- **Config tab**: Syntax-highlighted YAML with inline editing
- **Graph tab**: Interactive dependency visualization
- **Agents tab**: List with model profiles, roles - editable for local ensembles
- **Execute tab**: Input form with streaming output console
- **Artifacts tab**: Execution history for this ensemble
- **Edit/Delete buttons**: For local ensembles only (library/global are read-only)

#### 5.3 Ensemble Editor (`/ensemble/new` or `/ensemble/{name}/edit`)
- YAML editor with syntax highlighting and validation
- Visual agent builder (drag-and-drop)
- Profile selector dropdown
- Dependency graph preview
- Validate before save (uses `validate_ensemble` tool)

#### 5.4 Scripts Page (`/scripts`)
- List of primitive scripts by category
- **View**: Source code with syntax highlighting
- **Test**: Run with sample input (uses `test_script` tool)
- **Create**: New script from template
- **Delete**: Remove script (with confirmation)
- **Import from Library**: Browse and copy library scripts

#### 5.5 Library Browser (`/library`)
- Browse library ensembles and scripts
- Search functionality
- Preview before copying
- Copy to local with one click (uses `library_copy` tool)
- Shows which items are already installed locally

#### 5.6 Profiles Page (`/profiles`)
- List all model profiles with provider, model, cost info
- **Create**: New profile form
- **Edit**: Modify existing profile
- **Delete**: Remove profile (warns if in use by ensembles)
- **Test**: Quick connectivity test

#### 5.7 Artifacts Browser (`/artifacts`)
- Sortable table: timestamp, ensemble, status, cost, duration
- Filter by ensemble, date range, status
- Bulk delete capability
- Cleanup old artifacts (uses `cleanup_artifacts` tool)

#### 5.8 Artifact Detail (`/artifacts/{ensemble}/{id}`)
- **Overview**: Status, cost, duration, timestamp
- **Results**: Formatted agent outputs with syntax highlighting
- **Synthesis**: Final synthesized output
- **Raw**: JSON viewer with copy button
- **Compare**: Select another artifact for diff view
- **Delete**: Remove this artifact

#### 5.9 Metrics Dashboard (`/metrics`)
- **Summary cards**: Total executions, success rate, total cost
- **Charts**:
  - Executions over time
  - Cost trend
  - Success rate by ensemble
  - Agent performance comparison
- **Table**: Per-ensemble breakdown

#### 5.10 Settings (`/settings`)
- View current configuration
- Model profiles overview
- Link to config file locations
- llm-orc version info

### 6. Technology Choices

#### Backend
- **Framework**: FastAPI (async, WebSocket support, OpenAPI docs)
- **Static files**: Served from bundled directory
- **No database**: Uses existing artifact files

#### Frontend
- **Framework**: Preact (lightweight React alternative, ~3KB)
- **Styling**: Tailwind CSS (utility-first, small bundle)
- **Graph**: Dagre + D3 (DAG layout algorithm)
- **Charts**: Chart.js or uPlot (lightweight)
- **Build**: Vite (fast builds, good defaults)

#### Bundling Strategy
- Frontend built and bundled into `src/llm_orc/web/static/`
- Single `index.html` + hashed JS/CSS assets
- No Node.js required at runtime
- Optional: `llm-orc web --dev` for development with hot reload

### 7. Dependency Graph Visualization

```typescript
// Convert ensemble config to graph data
interface GraphNode {
  id: string;
  label: string;
  type: 'script' | 'llm';
  model_profile?: string;
}

interface GraphEdge {
  source: string;
  target: string;
  label?: string;  // e.g., "depends_on"
}

// Render with Dagre for layout, D3 for rendering
const g = new dagre.graphlib.Graph();
g.setGraph({ rankdir: 'LR' });  // Left-to-right flow

ensemble.agents.forEach(agent => {
  g.setNode(agent.name, { label: agent.name, ... });
  agent.depends_on?.forEach(dep => {
    g.setEdge(dep, agent.name);
  });
});

dagre.layout(g);
// Render nodes and edges with D3
```

### 8. Artifact Diff View

For comparing two artifacts:

```typescript
// Side-by-side comparison
interface DiffView {
  left: Artifact;
  right: Artifact;

  // Computed diffs
  metrics_diff: {
    cost: { left: number, right: number, delta: string };
    duration: { left: number, right: number, delta: string };
  };

  // Per-agent output diffs
  agent_diffs: {
    agent: string;
    left_output: string;
    right_output: string;
    diff_html: string;  // Highlighted diff
  }[];
}
```

Use `diff` library for text comparison with syntax highlighting.

### 9. Security Considerations

- **Localhost only**: Default bind to 127.0.0.1
- **No authentication**: Assumes trusted local user
- **No CORS**: Not needed for same-origin
- **Read-mostly**: Only execute and delete operations modify state
- **Warning on network bind**: Clear warning if `--host 0.0.0.0`

### 10. File Structure

```
src/llm_orc/web/
├── __init__.py
├── server.py           # FastAPI app, startup/shutdown
├── api/
│   ├── __init__.py
│   ├── ensembles.py    # Ensemble endpoints
│   ├── artifacts.py    # Artifact endpoints
│   ├── metrics.py      # Metrics endpoints
│   └── websocket.py    # WebSocket handler
├── static/             # Built frontend assets
│   ├── index.html
│   ├── assets/
│   │   ├── index-[hash].js
│   │   └── index-[hash].css
│   └── favicon.ico
└── frontend/           # Frontend source (not shipped in package)
    ├── package.json
    ├── vite.config.ts
    ├── src/
    │   ├── main.tsx
    │   ├── pages/
    │   ├── components/
    │   └── api/
    └── public/
```

## Consequences

### Positive

1. **Immediate usability**: Visual ensemble management without waiting for Manza
2. **Artifact exploration**: Easy browsing, comparison, and cleanup
3. **Dependency clarity**: Graph visualization makes structure obvious
4. **Execution monitoring**: Real-time streaming in structured UI
5. **Metrics visibility**: Aggregated cost/performance tracking
6. **Manza prototype**: Validates UX patterns before full integration
7. **Zero external dependencies**: Runs entirely locally

### Negative

1. **Maintenance burden**: Another surface to maintain
2. **Frontend complexity**: Adds JS build tooling to Python project
3. **Potential divergence**: UI patterns might not match Manza
4. **Duplication with MCP**: Some features overlap with MCP resources

### Relationship to MCP

The web UI and MCP server share the same underlying data:
- Both use `ConfigurationManager`, `EnsembleExecutor`, `ArtifactManager`
- Web UI could potentially consume MCP resources, but direct Python imports are simpler
- External tools (Manza, etc.) should use MCP; local users can use either

## Alternatives Considered

### TUI (Terminal UI)
- **Pros**: No browser needed, works over SSH
- **Cons**: Limited visualization, harder to build complex views
- **Verdict**: Could add later as `llm-orc tui`

### VS Code Extension
- **Pros**: Integrated with editor workflow
- **Cons**: Ties to specific editor, more complex distribution
- **Verdict**: Future consideration after MCP stabilizes

### Electron App
- **Pros**: Native feel, offline capable
- **Cons**: Heavy distribution, update complexity
- **Verdict**: Overkill for local tool

## Implementation Phases

### Phase 1: Core Server (1 week)
- FastAPI server with static file serving
- Basic REST API for ensembles and artifacts
- `llm-orc web` command

### Phase 2: Basic Frontend (1 week)
- Ensemble list page
- Ensemble detail with config view
- Artifact list and detail views

### Phase 3: Execution & Streaming (1 week)
- WebSocket execution handler
- Streaming output console component
- Execute from UI flow

### Phase 4: Visualization (1 week)
- Dependency graph component
- Metrics dashboard
- Artifact diff view

### Phase 5: Polish (ongoing)
- Error handling and edge cases
- Performance optimization
- Documentation

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Preact](https://preactjs.com/)
- [Dagre - Directed Graph Layout](https://github.com/dagrejs/dagre)
- [Vite](https://vitejs.dev/)
- [ADR-009: MCP Server Architecture](./009-mcp-server-architecture.md)
