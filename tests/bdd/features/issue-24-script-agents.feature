Feature: Script-Based Agent Support (Issue #24, ADR-001, ADR-002, ADR-003)
  """
  LLM Development Context:
  
  This feature implements script-based agents that integrate with LLM agents in 
  ensembles, enabling deterministic processing, external tool integration, and 
  human-in-the-loop workflows for collective intelligence research.
  
  Requirements from Issue #24:
  - Script discovery and resolution from .llm-orc/scripts/
  - Enhanced ScriptAgent with JSON input/output
  - Primitive script composition for complex behaviors
  - Artifact management and caching for reproducibility
  - Integration with existing ensemble execution
  
  Architectural Constraints:
  - ADR-001: All script I/O must use Pydantic schema validation
  - ADR-002: Scripts must be composable primitives with type-safe interfaces
  - ADR-003: All scripts must implement testable contracts with proper error handling
  
  Coding Standards Requirements:
  - Type annotations: def execute(input_data: ScriptInput) -> ScriptOutput
  - Exception chaining: raise ScriptExecutionError() from original_error
  - Async patterns: All script execution must be async-compatible
  - Line length: Maximum 88 characters with proper line breaking
  """

  Background:
    Given an llm-orc ensemble configuration
    And a script agent discovery system
    And proper Pydantic schema validation
    And existing ensemble execution infrastructure

  @core-functionality
  Scenario: Script agent executes with JSON input/output contract
    Given a script agent "read_file.py" in primitives/file-ops
    And the script expects JSON input via stdin
    And input parameters {"path": "test.txt", "encoding": "utf-8"}
    When the script agent executes within an ensemble
    Then it should receive JSON parameters via stdin
    And it should validate input using Pydantic schemas
    And it should output structured JSON with success field
    And the output should match ScriptAgentOutput schema
    And the JSON structure should be parseable by dependent agents
    And all type annotations should be preserved throughout execution

  @adr-001-compliance
  Scenario: Script agent supports dynamic parameter generation (ADR-001)
    Given a story generator script configured for "cyberpunk" theme
    And a user input agent available in the primitive registry
    And the story generator can output AgentRequest objects
    When the story generator executes with character_type "protagonist"
    Then it should generate a contextual prompt for the character
    And it should output an AgentRequest targeting "user_input" agent
    And the request should include the dynamically generated prompt
    And the prompt should contain cyberpunk-themed context
    And the user input agent should receive the generated parameters
    And all parameter passing should maintain Pydantic schema validation

  @adr-002-compliance  
  Scenario: Composable primitive scripts chain with type safety (ADR-002)
    Given primitive scripts: read_file, json_extract, write_file
    And an ensemble configuration chaining these primitives
    And each primitive has defined Pydantic input/output schemas
    When the ensemble executes with source file "config.json"
    Then read_file should execute first with file path parameter
    And read_file output should flow to json_extract as typed input
    And json_extract should validate input schema compliance
    And json_extract should transform data with specified field extraction
    And json_extract output should flow to write_file with type validation
    And write_file should persist the extracted data to target file
    And the complete chain should maintain type safety at each boundary
    And no runtime type errors should occur during execution

  @script-discovery
  Scenario: Script resolution follows priority order for discovery
    Given a script reference "primitives/network/topology.py"
    And the script exists in .llm-orc/scripts/primitives/network/topology.py
    And an alternative script exists at /usr/local/bin/topology
    When the script resolver attempts to resolve the reference
    Then it should find the script in .llm-orc/scripts/ first
    And it should return the correct absolute path
    And it should validate the script is executable
    And it should handle missing scripts gracefully with clear error messages
    And the resolution should be cached for performance

  @ensemble-integration
  Scenario: Script agents integrate seamlessly with LLM agents in ensembles
    Given an ensemble with both script and LLM agents
    And a script agent "network-analyzer" that processes topology data
    And an LLM agent "pattern-interpreter" that analyzes network patterns
    And the LLM agent depends on the script agent output
    When the ensemble executes with network data input
    Then the script agent should execute first with deterministic results
    And the script output should be structured JSON
    And the LLM agent should receive the script output as context
    And the LLM agent should process the data with AI reasoning
    And the final ensemble output should combine both deterministic and AI results
    And execution should respect dependency ordering automatically

  @error-handling
  Scenario: Script execution handles errors with proper exception chaining (ADR-003)
    Given a script agent that may encounter file system errors
    And the script is configured to read from a protected directory
    When the script executes and encounters a permission error
    Then it should catch the original PermissionError
    And it should chain the exception with ScriptExecutionError
    And the error message should be descriptive and actionable
    And the error should be properly logged for debugging
    And the ensemble should handle the failure gracefully
    And dependent agents should receive clear error information

  @caching-reproducibility
  Scenario: Script results are cached for reproducible research
    Given a script agent that generates network topology data
    And caching is enabled for deterministic operations
    And the script has been executed with specific parameters before
    When the same script is executed with identical parameters
    Then the cached result should be returned without re-execution
    And the cache key should be based on script content and parameters
    And cache invalidation should occur when script content changes
    And cached results should maintain full type safety
    And research reproducibility should be guaranteed

  @artifact-management
  Scenario: Execution results are saved to timestamped artifacts
    Given an ensemble execution with script and LLM agents
    And artifact management is configured for the ensemble
    When the ensemble completes successfully
    Then results should be saved to .llm-orc/artifacts/ensemble-name/timestamp/
    And both execution.json and execution.md should be created
    And the latest symlink should point to the newest results
    And artifact structure should support research publication requirements
    And all intermediate script outputs should be preserved

  @performance-requirements
  Scenario: Script execution maintains async performance characteristics
    Given multiple script agents configured for parallel execution
    And each script has different execution time characteristics
    When the ensemble executes with multiple independent scripts
    Then scripts should execute concurrently where possible
    And total execution time should be bounded by the slowest script
    And memory usage should remain under 50MB per script agent
    And the async execution should not block other ensemble operations
    And performance metrics should be tracked for optimization

  @primitive-library
  Scenario: Primitive scripts provide immediate value after initialization
    Given a fresh llm-orc initialization
    When the llm-orc package is installed
    Then core primitives should be available as package modules
    And file-ops primitives should include read_file, write_file operations
    And user-interaction primitives should include get_user_input
    And data-transform primitives should include json_extract, json_merge
    And control-flow primitives should include replicate_n_times
    And all primitives should follow consistent JSON I/O patterns