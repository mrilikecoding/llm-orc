# Script Agent Architecture

## Overview

Script agents enable deterministic processing, external tool integration, and human-in-the-loop workflows within llm-orc's ensemble orchestration system. They interoperate seamlessly with LLM agents through a unified JSON I/O interface based on Pydantic schemas.

**Key Architectural Documents:**
- [ADR-001: Pydantic Script Agent Interfaces](adrs/001-pydantic-script-agent-interfaces.md)
- [ADR-002: Composable Primitive System](adrs/002-composable-primitive-agent-system.md)
- [ADR-003: Testable Script Contracts](adrs/003-testable-script-agent-contracts.md)
- [ADR-006: Library-Based Primitives](adrs/006-library-based-primitives-architecture.md)

## Core Principles

### 1. Implicit Agent Typing

Agents are identified by fields present, not explicit type declarations:

```yaml
agents:
  # Script agent - identified by 'script' field
  - name: data-extractor
    script: primitives/file-ops/read_file.py
    parameters:
      path: "data.json"

  # LLM agent - identified by 'model_profile' field
  - name: analyzer
    model_profile: gpt-4-turbo
    system_prompt: "Analyze the extracted data"
    depends_on: [data-extractor]
```

### 2. Pure Primitive-Driven Architecture

Complex behaviors emerge from primitive composition. No special modes or infrastructure:

```yaml
# Interactive game = loop + user input + LLM generation
name: cyberpunk-detective
agents:
  - name: game-loop
    script: primitives/control-flow/loop_until_quit.py

  - name: player-input
    script: primitives/user-interaction/get_user_input.py
    depends_on: [game-loop]

  - name: narrative-generator
    model_profile: creative-storyteller
    depends_on: [player-input]
```

### 3. Library-Based Content Separation

Primitives live in `llm-orchestra-library` submodule, not in core application:

```
llm-orc/                          # Orchestration infrastructure
└── src/llm_orc/
    ├── agents/script_agent.py    # Script execution engine
    ├── core/execution/
    │   └── scripting/
    │       └── resolver.py       # Script discovery
    └── schemas/script_agent.py   # Type-safe I/O schemas

llm-orchestra-library/            # Orchestrable content
└── primitives/
    ├── file-ops/
    ├── user-interaction/
    ├── data-transform/
    └── research/
```

## Agent Type Comparison

### Script Agents vs LLM Agents

| Aspect | Script Agent | LLM Agent |
|--------|-------------|-----------|
| **Identification** | Has `script` field | Has `model_profile` field |
| **Execution** | Subprocess with JSON I/O | API call with text I/O |
| **Determinism** | Fully deterministic | Non-deterministic (unless cached) |
| **Cost** | Free (local execution) | API costs per request |
| **Capabilities** | File I/O, external tools, user input | Natural language reasoning |
| **Caching** | Results cached by input hash | Optional response caching |
| **Use Cases** | Data processing, validation, external integration | Analysis, generation, reasoning |

## System Architecture

### High-Level Ensemble Execution Flow

```mermaid
graph TB
    subgraph "CLI Layer"
        CLI[llm-orc invoke ensemble]
    end

    subgraph "Configuration Layer"
        YAML[ensemble.yaml]
        CONFIG[EnsembleConfig]
        YAML --> CONFIG
    end

    subgraph "Orchestration Layer"
        EXECUTOR[EnsembleExecutor]
        RESOLVER[DependencyResolver]
        CONFIG --> EXECUTOR
        EXECUTOR --> RESOLVER
    end

    subgraph "Agent Execution Layer"
        COORD[AgentExecutionCoordinator]
        SCRIPT[ScriptAgent]
        LLM[LLMAgent]
        RESOLVER --> COORD
        COORD --> SCRIPT
        COORD --> LLM
    end

    subgraph "External Resources"
        FS[File System<br/>Scripts & Data]
        API[LLM APIs<br/>Anthropic/Google/Ollama]
        SCRIPT --> FS
        LLM --> API
    end

    CLI --> YAML
    COORD --> RESULTS[Execution Results]
    RESULTS --> CLI

    style SCRIPT fill:#e1f5e1
    style LLM fill:#e1e5f5
```

### Script Agent Execution Detail

```mermaid
sequenceDiagram
    participant Executor as EnsembleExecutor
    participant Resolver as ScriptResolver
    participant Agent as ScriptAgent
    participant Script as External Script
    participant Context as AgentContext

    Executor->>Resolver: resolve_script_path("primitives/read_file.py")
    Resolver->>Resolver: Check .llm-orc/scripts/primitives/
    Resolver->>Resolver: Check llm-orchestra-library/primitives/
    Resolver-->>Executor: /path/to/read_file.py

    Executor->>Agent: execute(input_data, context)
    Agent->>Agent: Build JSON input from parameters + context

    Note over Agent,Script: JSON I/O via stdin/stdout
    Agent->>Script: subprocess.run(stdin=json_input)
    Script->>Script: Read JSON from stdin
    Script->>Script: Execute operation
    Script->>Script: Write JSON to stdout
    Script-->>Agent: {"success": true, "data": {...}}

    Agent->>Context: Store result as context for dependent agents
    Agent-->>Executor: ScriptAgentOutput(success=true, data={...})
```

### Mixed Agent Type Ensemble

```mermaid
graph LR
    subgraph "Phase 1: Data Acquisition"
        S1[Script: read_file]
        S2[Script: http_request]
    end

    subgraph "Phase 2: LLM Analysis"
        L1[LLM: analyze_data]
        L1 -.depends_on.-> S1
        L1 -.depends_on.-> S2
    end

    subgraph "Phase 3: Human Validation"
        S3[Script: get_user_input]
        S3 -.depends_on.-> L1
    end

    subgraph "Phase 4: Final Processing"
        L2[LLM: synthesize_with_feedback]
        S4[Script: write_report]
        L2 -.depends_on.-> S3
        S4 -.depends_on.-> L2
    end

    style S1 fill:#e1f5e1
    style S2 fill:#e1f5e1
    style S3 fill:#e1f5e1
    style S4 fill:#e1f5e1
    style L1 fill:#e1e5f5
    style L2 fill:#e1e5f5
```

## JSON I/O Contract

All script agents communicate via JSON I/O following Pydantic schemas:

### Input Schema (ScriptAgentInput)

```python
class ScriptAgentInput(BaseModel):
    """Base input schema for all script agents."""
    agent_name: str
    input_data: str
    context: dict[str, Any] = Field(default_factory=dict)
    dependencies: dict[str, Any] = Field(default_factory=dict)
```

**Example Input:**
```json
{
  "agent_name": "data-processor",
  "input_data": "Process this data",
  "context": {
    "execution_id": "abc123",
    "timestamp": "2025-09-29T10:00:00Z"
  },
  "dependencies": {
    "data-extractor": {
      "success": true,
      "data": {"records": [1, 2, 3]}
    }
  }
}
```

### Output Schema (ScriptAgentOutput)

```python
class ScriptAgentOutput(BaseModel):
    """Base output schema for all script agents."""
    success: bool
    data: Any = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
```

**Example Output:**
```json
{
  "success": true,
  "data": {
    "processed_records": 3,
    "results": [...]
  },
  "metadata": {
    "execution_time_ms": 150,
    "script_version": "1.0.0"
  }
}
```

### Category-Specific Schemas

Different primitive categories have specialized schemas (see [ADR-002](adrs/002-composable-primitive-agent-system.md)):

**User Interaction:**
```python
class UserInteractionInput(BaseModel):
    agent_name: str
    prompt: str
    multiline: bool = False
    validation_pattern: str | None = None
```

**File Operations:**
```python
class FileOperationInput(BaseModel):
    agent_name: str
    operation: Literal["read", "write", "append"]
    path: str
    content: str | None = None
```

**Data Transformation:**
```python
class DataTransformInput(BaseModel):
    source_data: Any
    transformation: str
    parameters: dict[str, Any] = Field(default_factory=dict)
```

## Script Resolution

### Search Path Priority

ScriptResolver finds scripts in this order:

1. **Relative paths from project root** - `.llm-orc/scripts/`
2. **Library primitives** - `llm-orchestra-library/primitives/`
3. **Absolute paths** - `/usr/local/bin/script.py`
4. **Inline content** - Backward compatibility (deprecated)

```mermaid
graph TD
    START[Script Reference: primitives/read_file.py]

    CHECK1{Exists in<br/>.llm-orc/scripts/?}
    CHECK2{Exists in<br/>library/primitives/?}
    CHECK3{Absolute path<br/>exists?}
    CHECK4{Inline content?}

    FOUND1[Use local project script]
    FOUND2[Use library primitive]
    FOUND3[Use absolute path]
    FOUND4[Use inline content]
    ERROR[ScriptNotFoundError]

    START --> CHECK1
    CHECK1 -->|Yes| FOUND1
    CHECK1 -->|No| CHECK2
    CHECK2 -->|Yes| FOUND2
    CHECK2 -->|No| CHECK3
    CHECK3 -->|Yes| FOUND3
    CHECK3 -->|No| CHECK4
    CHECK4 -->|Yes| FOUND4
    CHECK4 -->|No| ERROR

    style FOUND1 fill:#90EE90
    style FOUND2 fill:#90EE90
    style FOUND3 fill:#90EE90
    style FOUND4 fill:#FFE4B5
    style ERROR fill:#FFB6C6
```

### Example Resolution

```yaml
agents:
  # Resolved from library
  - name: reader
    script: primitives/file-ops/read_file.py
    # → llm-orchestra-library/primitives/file-ops/read_file.py

  # Resolved from local project
  - name: custom-processor
    script: scripts/custom/process_data.py
    # → .llm-orc/scripts/custom/process_data.py

  # Absolute path
  - name: system-tool
    script: /usr/local/bin/analyzer.py
    # → /usr/local/bin/analyzer.py
```

## Primitive Categories

### Core Primitives (in llm-orchestra-library)

#### File Operations (`primitives/file-ops/`)
- `read_file.py` - Read file content with encoding support
- `write_file.py` - Write content to file
- `list_files.py` - List directory contents
- `file_exists.py` - Check file existence

#### User Interaction (`primitives/user-interaction/`)
- `get_user_input.py` - Prompt for user input
- `confirm_action.py` - Yes/no confirmation
- `select_option.py` - Multiple choice selection
- `pause_for_review.py` - Pause for human review

#### Data Transformation (`primitives/data-transform/`)
- `json_extract.py` - Extract fields from JSON
- `json_merge.py` - Merge multiple JSON objects
- `json_filter.py` - Filter JSON by criteria
- `format_markdown.py` - Convert data to markdown

#### Control Flow (`primitives/control-flow/`)
- `loop_until.py` - Conditional looping
- `replicate_n_times.py` - Repeat execution N times
- `aggregate_results.py` - Combine multiple outputs
- `conditional_execute.py` - Conditional execution

#### Research & Analytics (`primitives/research/`)
- `t_test.py` - Statistical significance testing
- `anova.py` - Analysis of variance
- `effect_size.py` - Cohen's d calculation
- `compare_populations.py` - Population comparison

#### Network Science (`primitives/network-science/`)
- `generate_topology.py` - Create network topologies
- `calculate_centrality.py` - Centrality metrics
- `detect_communities.py` - Community detection
- `measure_emergence.py` - Emergent properties

## Use Case Patterns

### Pattern 1: Human-in-the-Loop Code Review

```mermaid
sequenceDiagram
    participant LLM1 as LLM: Code Analyzer
    participant Script1 as Script: Get User Input
    participant Human as Human Reviewer
    participant LLM2 as LLM: Synthesis Agent

    LLM1->>LLM1: Analyze code for issues
    LLM1->>Script1: Pass analysis results
    Script1->>Human: Prompt: "Review this analysis?"
    Human->>Script1: "Add security concern about..."
    Script1->>LLM2: Pass analysis + human feedback
    LLM2->>LLM2: Synthesize final report
```

**Ensemble Configuration:**
```yaml
name: code-review-with-human
agents:
  - name: analyzer
    model_profile: gpt-4-turbo
    system_prompt: "Analyze code for issues"

  - name: get-human-review
    script: primitives/user-interaction/get_user_input.py
    depends_on: [analyzer]
    parameters:
      prompt: "Review AI analysis. Any additional concerns?"
      multiline: true

  - name: synthesizer
    model_profile: claude-sonnet
    system_prompt: "Synthesize automated and human feedback"
    depends_on: [get-human-review]
```

### Pattern 2: Network Science Experiment

```mermaid
graph TB
    S1[Script: Generate Topology]
    S2[Script: Calculate Metrics]
    L1[LLM: Analyze Patterns]
    S3[Script: Statistical Test]
    L2[LLM: Generate Report]

    S1 --> S2
    S2 --> L1
    L1 --> S3
    S3 --> L2

    style S1 fill:#e1f5e1
    style S2 fill:#e1f5e1
    style S3 fill:#e1f5e1
    style L1 fill:#e1e5f5
    style L2 fill:#e1e5f5
```

**Ensemble Configuration:**
```yaml
name: network-topology-study
agents:
  - name: generate-network
    script: primitives/network-science/generate_topology.py
    parameters:
      topology_type: small_world
      nodes: 100
      k: 6
      p: 0.1
    cache: true  # Deterministic results cached

  - name: calculate-metrics
    script: primitives/network-science/calculate_centrality.py
    depends_on: [generate-network]

  - name: analyze-patterns
    model_profile: network-analyst
    system_prompt: "Analyze network topology for emergence patterns"
    depends_on: [calculate-metrics]

  - name: statistical-test
    script: primitives/research/t_test.py
    depends_on: [analyze-patterns]
    parameters:
      baseline_type: random_network
      alpha: 0.05

  - name: generate-report
    model_profile: research-writer
    system_prompt: "Generate academic research report"
    depends_on: [statistical-test]
```

### Pattern 3: Interactive Narrative Game

```yaml
name: cyberpunk-detective
description: Interactive fiction with AI narration

agents:
  - name: game-state
    script: scripts/game/state_manager.py
    parameters:
      initial_state:
        location: neon_alley
        inventory: [neural_link, coffee_credits]

  - name: narrator
    model_profile: creative-storyteller
    system_prompt: "Create atmospheric cyberpunk narrative"
    depends_on: [game-state]

  - name: player-input
    script: primitives/user-interaction/get_user_input.py
    depends_on: [narrator]
    parameters:
      prompt: "> "

  - name: update-state
    script: scripts/game/process_action.py
    depends_on: [player-input]
```

## Error Handling and Exception Chaining

Script agents use proper exception chaining (ADR-003):

```python
try:
    result = subprocess.run(
        [script_path],
        input=json_input,
        capture_output=True,
        timeout=30
    )
except subprocess.TimeoutExpired as e:
    raise ScriptExecutionError(
        f"Script {script_path} timed out after 30s"
    ) from e
except subprocess.CalledProcessError as e:
    raise ScriptExecutionError(
        f"Script {script_path} failed with exit code {e.returncode}"
    ) from e
```

## Testing and Validation

### Contract Validation

All primitives implement `ScriptContract` for type safety:

```python
class ScriptContract(ABC):
    @classmethod
    @abstractmethod
    def input_schema(cls) -> type[BaseModel]:
        """Input schema for validation."""
        pass

    @classmethod
    @abstractmethod
    def output_schema(cls) -> type[BaseModel]:
        """Output schema for validation."""
        pass
```

### BDD Testing

Behavior-driven tests validate script agent functionality:

```gherkin
Scenario: Script agent executes with JSON I/O
  Given a script agent with JSON input/output contract
  When the agent executes with valid input
  Then output conforms to ScriptAgentOutput schema
  And downstream agents receive structured data
```

Run tests:
```bash
# All script agent tests
uv run pytest tests/bdd/ -k "script"

# Specific ADR tests
uv run pytest tests/bdd/ -k "adr-001"
```

## Performance Considerations

### Caching

Script agents support result caching for deterministic operations:

```yaml
agents:
  - name: expensive-calculation
    script: primitives/research/monte_carlo.py
    parameters:
      iterations: 1000000
    cache: true  # Results cached by input hash
```

### Parallel Execution

Independent script agents execute in parallel:

```yaml
agents:
  # These three run concurrently (no dependencies)
  - name: read-file-1
    script: primitives/file-ops/read_file.py
    parameters: {path: "data1.json"}

  - name: read-file-2
    script: primitives/file-ops/read_file.py
    parameters: {path: "data2.json"}

  - name: read-file-3
    script: primitives/file-ops/read_file.py
    parameters: {path: "data3.json"}
```

## Future Enhancements

See [ADR-005](adrs/005-multi-turn-agent-conversations.md) for multi-turn conversation support (70% implemented).

### Planned Features
- LLMResponseGenerator for automated conversation testing
- Visual workflow designer for ensemble composition
- Advanced conversation patterns (branching, merging)
- Performance optimization for large-scale ensembles
- Artifact management with versioned execution history

## Related Documentation

- [Coding Standards](coding-standards.md) - Code quality requirements
- [ADR Index](adrs/README.md) - Architecture decision records
- [Development Workflow](commands.md) - Make commands and TDD cycle