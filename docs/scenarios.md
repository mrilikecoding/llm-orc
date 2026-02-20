# Behavior Scenarios: Composable Ensemble Orchestration

## Feature: Conversational Ensemble System Removal (ADR-011)

### Scenario: Dead code files are absent
**Given** the cleanup increment is complete
**When** the file system is inspected
**Then** `conversational_ensemble_executor.py`, `conversational_agent.py`, `test_conversational_ensemble_executor.py`, `test_conversation_state.py`, and `test_adr_005_multi_turn_conversations.py` do not exist

### Scenario: No production imports reference conversational modules
**Given** the cleanup increment is complete
**When** all Python files under `src/llm_orc/` are scanned for import statements
**Then** no file imports from `conversational_ensemble_executor` or `conversational_agent`

### Scenario: Existing tests pass after cleanup
**Given** the conversational ensemble files have been deleted
**When** the full test suite runs
**Then** all tests pass (the deleted tests were the only consumers)

---

## Feature: Pydantic Agent Config Parsing (ADR-012)

### Scenario: LLM agent config parsed from YAML
**Given** a YAML agent dict with `name: "analyzer"` and `model_profile: "gpt4"`
**When** the dict is parsed as an `AgentConfig`
**Then** the result is an `LlmAgentConfig` instance with `name == "analyzer"` and `model_profile == "gpt4"`

### Scenario: Script agent config parsed from YAML
**Given** a YAML agent dict with `name: "scanner"` and `script: "scripts/scan.py"`
**When** the dict is parsed as an `AgentConfig`
**Then** the result is a `ScriptAgentConfig` instance with `name == "scanner"` and `script == "scripts/scan.py"`

### Scenario: Inline model requires provider
**Given** a YAML agent dict with `name: "test"`, `model: "llama3"`, and no `provider` field
**When** the dict is parsed as an `AgentConfig`
**Then** a validation error is raised indicating `provider` is required when `model` is specified

### Scenario: Profile XOR inline model enforced
**Given** a YAML agent dict with `name: "test"`, `model_profile: "gpt4"`, `model: "llama3"`, and `provider: "ollama"`
**When** the dict is parsed as an `AgentConfig`
**Then** a validation error is raised indicating `model_profile` and `model`+`provider` are mutually exclusive

### Scenario: Unknown fields rejected
**Given** a YAML agent dict with `name: "test"`, `model_profile: "gpt4"`, and `typo_field: "oops"`
**When** the dict is parsed as an `AgentConfig`
**Then** a validation error is raised indicating `typo_field` is not a recognized field

### Scenario: Agent type determined by isinstance
**Given** a parsed `LlmAgentConfig` instance
**When** the `AgentDispatcher` determines the agent type
**Then** it uses `isinstance(config, LlmAgentConfig)`, not key inspection on a dict

### Scenario: Ensemble loads with Pydantic agent configs
**Given** an ensemble YAML file with two LLM agents and one script agent
**When** `EnsembleLoader.load_from_file()` parses the file
**Then** `EnsembleConfig.agents` is a `list[AgentConfig]` containing two `LlmAgentConfig` and one `ScriptAgentConfig`

### Scenario: Agent-level overrides win over profile defaults
**Given** a model profile with `temperature: 0.7`
**And** an `LlmAgentConfig` with `model_profile` referencing that profile and `temperature: 0.2`
**When** `LlmAgentRunner` resolves the profile
**Then** the effective temperature is `0.2`

### Scenario: Integration — EnsembleExecutor runs with Pydantic agent configs
**Given** an ensemble YAML file loaded through `EnsembleLoader` (producing `list[AgentConfig]`)
**When** `EnsembleExecutor` executes the ensemble
**Then** the execution completes with results for each agent (no behavior change from dict-based configs)

---

## Feature: Ensemble Agent Type (ADR-013)

### Scenario: Ensemble agent config parsed from YAML
**Given** a YAML agent dict with `name: "review"` and `ensemble: "code-review"`
**When** the dict is parsed as an `AgentConfig`
**Then** the result is an `EnsembleAgentConfig` instance with `ensemble == "code-review"`

### Scenario: Ensemble agent mutual exclusivity
**Given** a YAML agent dict with `name: "test"`, `ensemble: "code-review"`, and `model_profile: "gpt4"`
**When** the dict is parsed as an `AgentConfig`
**Then** a validation error is raised (an agent cannot be both an ensemble agent and an LLM agent)

### Scenario: Ensemble agent participates in dependency chain
**Given** an ensemble with agents: `scanner` (script), `review` (ensemble, depends_on: [scanner]), `summarizer` (LLM, depends_on: [review])
**When** `DependencyAnalyzer` computes phases
**Then** `scanner` is in phase 0, `review` is in phase 1, `summarizer` is in phase 2

### Scenario: Ensemble agent executes child ensemble
**Given** an ensemble with an ensemble agent referencing "child-ensemble"
**And** "child-ensemble" is a valid ensemble with two LLM agents
**When** the parent ensemble is executed
**Then** the ensemble agent's response contains the child ensemble's full result dict as JSON

### Scenario: Child executor shares immutable infrastructure
**Given** a parent `EnsembleExecutor` with a config manager and credential storage
**When** an `EnsembleAgentRunner` creates a child executor
**Then** the child executor uses the same config manager and credential storage instances as the parent

### Scenario: Child executor isolates mutable state
**Given** a parent `EnsembleExecutor` with a usage collector
**When** an `EnsembleAgentRunner` creates a child executor
**Then** the child executor has its own usage collector instance, distinct from the parent's

### Scenario: Child ensemble does not produce its own artifact
**Given** a parent ensemble with an ensemble agent
**When** the parent ensemble is executed
**Then** no artifact file is written for the child ensemble execution
**And** the child's results are nested within the parent's artifact

### Scenario: Cross-ensemble cycle detected at load time
**Given** ensemble "A" contains an ensemble agent referencing "B"
**And** ensemble "B" contains an ensemble agent referencing "A"
**When** `EnsembleLoader` loads ensemble "A"
**Then** a validation error is raised indicating a cross-ensemble cycle: A -> B -> A

### Scenario: Transitive cross-ensemble cycle detected
**Given** ensemble "A" references "B", "B" references "C", "C" references "A"
**When** `EnsembleLoader` loads ensemble "A"
**Then** a validation error is raised indicating a cross-ensemble cycle: A -> B -> C -> A

### Scenario: Depth limit prevents unbounded nesting
**Given** a depth limit of 3 in performance config
**And** ensembles A -> B -> C -> D (each referencing the next via ensemble agents)
**When** ensemble "A" is executed
**Then** the execution of the ensemble agent at depth 3 (referencing D) fails with a depth limit error
**And** the parent ensemble records the failure and continues with other agents

### Scenario: Child ensemble failure is an agent failure
**Given** an ensemble with agents: `preprocess` (script), `analysis` (ensemble, depends_on: [preprocess]), `fallback` (LLM)
**And** the child ensemble referenced by `analysis` has an agent that fails
**When** the parent ensemble is executed
**Then** `analysis` reports a failure status
**And** `fallback` (which does not depend on `analysis`) executes successfully
**And** the parent ensemble completes with `has_errors: true`

### Scenario: Integration — EnsembleAgentRunner dispatched by AgentDispatcher
**Given** an `EnsembleAgentConfig` instance
**When** `AgentDispatcher` dispatches the agent for execution
**Then** the agent is routed to `EnsembleAgentRunner` (not `LlmAgentRunner` or `ScriptAgentRunner`)

---

## Feature: Input Key for Selective Upstream Consumption (ADR-014)

### Scenario: Input key selects a specific key from upstream output
**Given** an upstream agent "classifier" that produces `{"pdfs": ["a.pdf", "b.pdf"], "audio": ["c.mp3"]}`
**And** a downstream agent "pdf-processor" with `depends_on: [classifier]` and `input_key: "pdfs"`
**When** `DependencyResolver` prepares input for "pdf-processor"
**Then** "pdf-processor" receives `["a.pdf", "b.pdf"]` as its input

### Scenario: Input key with fan-out
**Given** an upstream agent "classifier" that produces `{"pdfs": ["a.pdf", "b.pdf"], "audio": ["c.mp3"]}`
**And** a downstream agent "pdf-processor" with `depends_on: [classifier]`, `input_key: "pdfs"`, and `fan_out: true`
**When** the ensemble is executed
**Then** "pdf-processor" is expanded into 2 instances — one for "a.pdf" and one for "b.pdf"

### Scenario: Missing input key is a runtime error
**Given** an upstream agent "classifier" that produces `{"pdfs": ["a.pdf"]}`
**And** a downstream agent "video-processor" with `depends_on: [classifier]` and `input_key: "videos"`
**When** `DependencyResolver` prepares input for "video-processor"
**Then** "video-processor" receives an error status (key "videos" not found in upstream output)
**And** the ensemble continues executing other agents that do not depend on "video-processor"

### Scenario: Non-dict upstream output with input key is a runtime error
**Given** an upstream agent "greeter" that produces a plain string "hello world"
**And** a downstream agent "processor" with `depends_on: [greeter]` and `input_key: "message"`
**When** `DependencyResolver` prepares input for "processor"
**Then** "processor" receives an error status (upstream output is not dict-shaped)

### Scenario: No input key passes full output (backward compatible)
**Given** an upstream agent "classifier" that produces `{"pdfs": ["a.pdf"], "audio": ["c.mp3"]}`
**And** a downstream agent "summarizer" with `depends_on: [classifier]` and no `input_key`
**When** `DependencyResolver` prepares input for "summarizer"
**Then** "summarizer" receives the full output `{"pdfs": ["a.pdf"], "audio": ["c.mp3"]}`

### Scenario: Input key works with all agent types
**Given** an LLM agent with `input_key: "text"`, a script agent with `input_key: "data"`, and an ensemble agent with `input_key: "items"`
**When** each agent's config is parsed
**Then** all three configs have `input_key` set (it is a `BaseAgentConfig` field, available to all types)

### Scenario: Integration — input key with ensemble agent in routing pattern
**Given** an ensemble with:
  - "classifier" (script) producing `{"pdfs": ["a.pdf"], "audio": ["c.mp3"]}`
  - "pdf-extractor" (ensemble agent, depends_on: [classifier], input_key: "pdfs", fan_out: true)
  - "audio-extractor" (ensemble agent, depends_on: [classifier], input_key: "audio", fan_out: true)
  - "synthesizer" (LLM, depends_on: [pdf-extractor, audio-extractor])
**When** the ensemble is executed
**Then** "pdf-extractor" fans out over `["a.pdf"]`, "audio-extractor" fans out over `["c.mp3"]`
**And** "synthesizer" receives results from both extractors
**And** the ensemble completes successfully
