Feature: ADR-006 Library-Based Primitives Architecture with Multi-Language Bridge Support
  """
  LLM Development Context:

  Requirements: Implement library-based primitives architecture with Python bridge patterns
  for multi-language execution, enabling clean separation between orchestration engine
  and orchestrable components while maintaining test independence.

  Architectural constraints (ADR-006):
  - Primitives live in optional llm-orchestra-library submodule, not core application
  - Script resolution prioritizes local project scripts over library primitives
  - Bridge primitives enable cross-language execution via subprocess with JSON I/O
  - Tests must run without library submodule dependency using test fixtures
  - Graceful degradation with helpful error messages for missing primitives
  - Engine stays Python-only, complexity handled by bridge primitive pattern

  Related ADRs:
  - ADR-002: Builds on composable primitive system concepts
  - ADR-003: Script contracts apply to bridge primitives with exception chaining
  - ADR-001: Pydantic schemas used for bridge primitive I/O validation

  Implementation patterns:
  - ScriptResolver with library-aware path resolution and helpful error messages
  - TestPrimitiveFactory for library-independent test execution
  - Bridge primitives (subprocess_executor.py, node_executor.py) with JSON I/O
  - Multi-language execution patterns via Python bridges
  - Library submodule optional installation and graceful degradation

  Critical validations:
  - Test independence: All tests pass without library submodule initialization
  - Script resolution prioritization: Local scripts override library primitives
  - Bridge primitive execution with proper timeout and error handling
  - Cross-language JSON I/O compatibility and validation
  - Helpful error messages guide users to initialize library or provide alternatives
  - Exception chaining for subprocess failures (ADR-003)
  - Type safety for all bridge primitive interfaces
  """

  As a developer building orchestrated workflows
  I want library-based primitives with multi-language bridge support
  So that the engine stays focused while enabling unlimited extensibility

  Background:
    Given llm-orc is properly configured
    And the script resolution system is initialized
    And test primitive fixtures are available

  @script-resolution @adr-006
  Scenario: Script resolution prioritizes local over library primitives
    Given a script reference "primitives/user_input.py" exists in both local and library locations
    And ScriptResolver is configured with search path prioritization
    When I attempt to resolve the script path using ScriptResolver.resolve_script_path()
    Then the local project script should be returned as resolved path
    And the library primitive should not be considered
    And the resolution should complete without library dependency
    And the path should be absolute and executable

  @script-resolution @adr-006
  Scenario: ScriptResolver provides helpful error for missing library primitives
    Given a script reference "primitives/user_input.py" that only exists in library
    And the library submodule is not initialized
    When I attempt to resolve the script path using ScriptResolver.resolve_script_path()
    Then FileNotFoundError should be raised with helpful guidance
    And the error message should suggest "pip install"
    And the error message should suggest creating local implementation
    And the error message should mention test fixture usage for tests
    And the error should follow ADR-003 exception chaining patterns

  @script-resolution @adr-006
  Scenario: ScriptResolver gracefully handles missing non-primitive scripts
    Given a script reference "custom/missing_script.py" that doesn't exist anywhere
    When I attempt to resolve the script path using ScriptResolver.resolve_script_path()
    Then FileNotFoundError should be raised with basic not found message
    And the error should not include library-specific guidance
    And the error message should be clear and actionable
    And no library installation hints should be provided

  @test-independence @adr-006
  Scenario: Tests run without library submodule dependency
    Given the library submodule is not initialized or available
    And TestPrimitiveFactory is configured for test execution
    When test suite executes requiring primitive functionality
    Then all tests should pass using test fixture implementations
    And TestPrimitiveFactory should provide minimal primitive implementations
    And test primitives should follow same JSON I/O contracts as library primitives
    And no external dependencies should be required for test execution

  @test-fixtures @adr-006
  Scenario: TestPrimitiveFactory creates compatible primitive scripts
    Given a temporary directory for test primitives
    When I call TestPrimitiveFactory.create_user_input_script()
    Then a functional user_input.py test script should be created
    And the script should accept INPUT_DATA environment variable with JSON
    And the script should return structured JSON output matching library interface
    And the script should include mock_user_input parameter for test automation
    And the script should be executable with proper permissions

  @test-fixtures @adr-006
  Scenario: TestPrimitiveFactory provides complete test primitives directory
    Given a temporary directory for test setup
    When I call TestPrimitiveFactory.setup_test_primitives_dir()
    Then a primitives directory should be created with all common scripts
    And user_input.py, subprocess_executor.py, node_executor.py should exist
    And file_read.py and other core primitives should be available
    And all scripts should be executable and follow JSON I/O patterns
    And the directory structure should mirror library organization

  @bridge-primitives @adr-006
  Scenario: Subprocess executor bridge handles external command execution
    Given a subprocess_executor.py bridge primitive
    And input data with command "echo 'test output'"
    When I execute the bridge primitive with structured JSON I/O
    Then the execution should complete with structured output
    And output should include success boolean, stdout, stderr, return_code fields
    And timeout handling should be implemented for long-running commands
    And working directory and environment variables should be configurable
    And exception chaining should follow ADR-003 for subprocess failures

  @bridge-primitives @adr-006
  Scenario: Node.js executor bridge enables JavaScript execution
    Given a node_executor.py bridge primitive
    And input data with inline JavaScript script and data payload
    When I execute the bridge primitive with JSON I/O
    Then JavaScript should receive input data via BRIDGE_INPUT environment variable
    And JavaScript output should be captured and returned as structured JSON
    And both inline script content and script_path execution should be supported
    And timeout handling should prevent hanging JavaScript execution
    And temporary file cleanup should occur for inline scripts

  @bridge-primitives @adr-006
  Scenario: Bridge primitives validate JSON I/O compatibility
    Given any bridge primitive (subprocess_executor, node_executor)
    And input data containing complex nested structures
    When I execute the bridge primitive
    Then input data should be properly serialized to JSON for external process
    And output should be parsed from JSON with error handling
    And data types should be preserved through JSON round-trip
    And malformed JSON output should result in clear error messages
    And all I/O should follow consistent JSON patterns across bridges

  @multi-language-execution @adr-006
  Scenario: Multi-language ensemble execution flows work end-to-end
    Given an ensemble with agents using different language bridges
    And agents for Python, JavaScript (node_executor), and shell (subprocess_executor)
    When the ensemble executes with data flowing between language agents
    Then each agent should execute in its appropriate language environment
    And data should flow correctly between different language agents via JSON
    And execution dependencies should be respected across language boundaries
    And the final result should combine outputs from all language agents
    And error handling should work consistently across all language bridges

  @multi-language-execution @adr-006
  Scenario: JavaScript bridge executes with proper data exchange
    Given a node_executor bridge primitive
    And JavaScript code that processes input data and returns results
    And input data containing arrays and objects for processing
    When the JavaScript bridge executes the code
    Then input data should be available to JavaScript via process.env.BRIDGE_INPUT
    And JavaScript should process the data using its native capabilities
    And results should be returned via JSON output to the bridge
    And complex data structures should be preserved through the bridge
    And the execution should demonstrate language-specific processing capabilities

  @error-handling @adr-006
  Scenario: Bridge primitives handle execution failures gracefully
    Given a bridge primitive with invalid external command or script
    When the bridge primitive attempts execution
    Then execution failure should be captured and reported
    And error output should include success: false and descriptive error message
    And timeout errors should be handled separately from execution errors
    And stderr output should be captured and included in error reporting
    And exception chaining should follow ADR-003 patterns with context preservation

  @performance @adr-006
  Scenario: Bridge primitive execution completes within performance requirements
    Given a bridge primitive executing a simple external command
    When the bridge execution is performed repeatedly
    Then each execution should complete with less than 100ms overhead
    And performance should scale linearly with command complexity
    And memory usage should remain constant across multiple executions
    And timeout handling should not impact normal execution performance

  @library-optional-installation @adr-006
  Scenario: Library submodule initialization workflow works correctly
    Given a fresh clone without library submodule initialized
    When I run "git submodule update --init --recursive"
    Then the llm-orchestra-library directory should be populated
    And primitive scripts should be available in library/primitives/python/
    And script resolution should find library primitives after initialization
    And existing ensembles should work with real library primitives
    And the transition from test fixtures to library should be seamless

  @engine-simplicity @adr-006
  Scenario: Core engine remains Python-only despite multi-language support
    Given the llm-orc core orchestration engine
    When examining engine dependencies and language requirements
    Then the engine should only require Python and its dependencies
    And no Node.js, Rust, or other language runtimes should be core dependencies
    And multi-language support should be provided entirely through bridge primitives
    And engine complexity should not increase with additional language support
    And bridge pattern should handle all language-specific execution concerns

  @type-safety @adr-006
  Scenario: Bridge primitives maintain type safety with annotations
    Given bridge primitive implementations (subprocess_executor, node_executor)
    When examining the bridge primitive code structure
    Then all bridge primitives should have complete type annotations
    And input/output data structures should be properly typed
    And JSON serialization/deserialization should preserve type information
    And error handling should maintain type safety throughout execution
    And bridge primitive interfaces should be compatible with Pydantic validation (ADR-001)

  @security-validation @adr-006
  Scenario: Bridge primitives validate input safely for subprocess execution
    Given a subprocess_executor bridge primitive
    And potentially unsafe input containing shell injection attempts
    When the bridge primitive processes the input
    Then input validation should prevent command injection vulnerabilities
    And subprocess execution should use safe parameter passing
    And environment variable handling should prevent privilege escalation
    And working directory changes should be validated and contained
    And timeout enforcement should prevent resource exhaustion attacks

  @architectural-compliance @adr-006
  Scenario: Implementation aligns with ADR-006 architectural decisions
    Given the complete library-based primitives implementation
    When I review the implementation against ADR-006 requirements
    Then primitives should reside in optional llm-orchestra-library submodule
    And tests should achieve full independence from library dependencies
    And script resolution should prioritize local over library implementations
    And bridge primitives should enable multi-language execution without engine complexity
    And graceful degradation should provide helpful guidance for missing components
    And the architecture should maintain clean separation between engine and content

  @ecosystem-extensibility @adr-006
  Scenario: Architecture supports community and ecosystem development
    Given the library-based primitives architecture
    When considering community contribution and ecosystem development
    Then primitives library should be independently extensible
    And third parties should be able to contribute primitive implementations
    And new language bridges should be addable without core engine changes
    And primitive collections should be packageable as independent modules
    And the architecture should support marketplace/registry for primitive discovery

  @integration-compatibility @adr-006
  Scenario: Library-based architecture integrates with existing ensemble patterns
    Given existing ensemble configurations using primitive references
    And the library-based primitives architecture
    When ensembles execute using both local and library primitives
    Then execution should be transparent to ensemble configuration
    And primitive sourcing should not affect ensemble behavior
    And dependency resolution should work across local and library components
    And error reporting should maintain clarity about primitive sources
    And performance should remain consistent regardless of primitive location