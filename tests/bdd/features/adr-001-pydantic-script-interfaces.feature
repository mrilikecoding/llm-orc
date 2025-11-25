Feature: ADR-001 Pydantic-Based Script Agent Interface System
  """
  LLM Development Context:

  Requirements: Implement type-safe communication system for script agents using Pydantic schemas,
  enabling dynamic inter-agent communication, runtime validation, and structured parameter passing.

  Architectural constraints (ADR-001):
  - All script inputs/outputs must use Pydantic models (ScriptAgentInput/Output)
  - Dynamic parameter generation through AgentRequest objects in agent_requests field
  - Runtime validation with automatic type checking and error reporting
  - Extensible schema architecture supporting specialized interaction patterns
  - JSON serialization compatibility for inter-agent communication
  - Exception chaining for validation failures (ADR-003)

  Implementation patterns:
  - ScriptAgentInput: agent_name, input_data, context, dependencies fields
  - ScriptAgentOutput: success, data, error, agent_requests fields
  - AgentRequest: target_agent_type, parameters, priority for dynamic communication
  - UserInputRequest/Output: specialized schemas for user interaction patterns
  - FileOperationRequest/Output: specialized schemas for file operation patterns
  - Schema validation in EnhancedScriptAgent.execute_with_schema() method
  - AgentRequestProcessor for extracting and coordinating dynamic requests

  Critical validations:
  - Pydantic schema validation for all script agent interfaces
  - Type safety enforcement at runtime with clear error messages
  - JSON serialization/deserialization compatibility for agent communication
  - Dynamic parameter generation and inter-agent request coordination
  - Schema extension capability for specialized script types
  - Error handling with proper exception chaining (ADR-003)
  - Performance requirements: schema validation under 10ms per operation
  """

  As a developer building script-based agents
  I want type-safe communication using Pydantic schemas
  So that inter-agent communication is validated, dynamic, and extensible

  Background:
    Given llm-orc is properly configured
    And the Pydantic schema system is initialized
    And script agent infrastructure is available

  @core-schema-validation @adr-001
  Scenario: ScriptAgentInput validates required fields and types
    Given a script agent requiring input validation
    When I create ScriptAgentInput with agent_name "test_agent" and input_data "test input"
    Then the schema validation should succeed
    And agent_name should be validated as string type
    And input_data should be validated as string type
    And context should default to empty dictionary
    And dependencies should default to empty dictionary
    And the schema should be JSON serializable

  @core-schema-validation @adr-001
  Scenario: ScriptAgentInput rejects invalid field types
    Given a script agent requiring input validation
    When I attempt to create ScriptAgentInput with invalid field types
    Then Pydantic validation should raise TypeError or ValueError
    And the error message should clearly identify the invalid field
    And the error should include type requirements for the field
    And no partially valid object should be created

  @core-schema-validation @adr-001
  Scenario: ScriptAgentOutput provides structured response format
    Given a script agent producing output
    When I create ScriptAgentOutput with success True and data "result"
    Then the schema validation should succeed
    And success field should be boolean type
    And data field should accept Any type for flexibility
    And error field should be optional string type
    And agent_requests should default to empty list of AgentRequest objects
    And the output should be JSON serializable for inter-agent communication

  @dynamic-communication @adr-001
  Scenario: AgentRequest enables dynamic parameter generation
    Given a script agent that needs to request another agent action
    When I create AgentRequest with target_agent_type "user_input" and parameters
    Then the schema validation should succeed
    And target_agent_type should be validated as string
    And parameters should be validated as dictionary
    And priority should default to 0 as integer
    And the request should be serializable for agent coordination

  @specialized-schemas @adr-001
  Scenario: UserInputRequest provides specialized user interaction schema
    Given a script agent needing user input collection
    When I create UserInputRequest with prompt "Enter your name"
    Then the schema validation should succeed
    And prompt should be required string field
    And multiline should default to False
    And validation_pattern should be optional string
    And retry_message should be optional string
    And max_attempts should default to 3
    And the request should extend base Pydantic functionality

  @specialized-schemas @adr-001
  Scenario: UserInputOutput extends ScriptAgentOutput with user interaction data
    Given a user input script agent producing results
    When I create UserInputOutput with success True, user_input "John", and attempts_used 1
    Then the schema validation should succeed
    And it should inherit all ScriptAgentOutput fields (success, data, error, agent_requests)
    And user_input should be required string field
    And attempts_used should be required integer field
    And validation_passed should be required boolean field
    And the inheritance should maintain type safety

  @specialized-schemas @adr-001
  Scenario: FileOperationRequest provides typed file operation interface
    Given a script agent performing file operations
    When I create FileOperationRequest with operation "read" and path "test.txt"
    Then the schema validation should succeed
    And operation should be validated against Literal["read", "write", "append", "delete"]
    And path should be required string field
    And content should be optional string field
    And encoding should default to "utf-8"
    And invalid operations should be rejected with clear error

  @specialized-schemas @adr-001
  Scenario: FileOperationOutput extends ScriptAgentOutput with file operation results
    Given a file operation script agent producing results
    When I create FileOperationOutput with path, size, bytes_processed, and operation_performed
    Then the schema validation should succeed
    And it should inherit all ScriptAgentOutput fields
    And path should be required string field
    And size should be required integer field
    And bytes_processed should be required integer field
    And operation_performed should be required string field

  @enhanced-script-agent @adr-001
  Scenario: EnhancedScriptAgent validates input using ScriptAgentInput schema
    Given an EnhancedScriptAgent configured for schema validation
    And a valid ScriptAgentInput with agent_name "test" and input_data "test data"
    When I call execute_with_schema() method
    Then the input should be validated against ScriptAgentInput schema
    And validation errors should be caught with proper error messages
    And the execution should proceed with validated input data
    And the context and dependencies should be properly passed to script

  @enhanced-script-agent @adr-001
  Scenario: EnhancedScriptAgent produces ScriptAgentOutput schema
    Given an EnhancedScriptAgent that executes successfully
    And the script produces valid JSON output
    When the execute_with_schema() method completes
    Then the output should be validated as ScriptAgentOutput schema
    And success field should reflect execution status
    And data field should contain script results
    And error field should be None for successful execution
    And agent_requests should be empty list unless populated by script

  @enhanced-script-agent @adr-001
  Scenario: EnhancedScriptAgent handles schema validation failures
    Given an EnhancedScriptAgent receiving invalid input
    When schema validation fails during execute_with_schema()
    Then a ScriptAgentOutput should be returned with success False
    And error field should contain descriptive validation error message
    And data field should be None or contain raw output for debugging
    And the error should follow proper exception chaining (ADR-003)
    And no script execution should occur with invalid input

  @agent-request-processing @adr-001
  Scenario: AgentRequestProcessor extracts requests from ScriptAgentOutput
    Given a ScriptAgentOutput containing agent_requests
    And an AgentRequestProcessor instance
    When I call extract_agent_requests() method
    Then it should return a list of AgentRequest objects
    And each request should be properly validated
    And the extraction should handle empty agent_requests gracefully
    And invalid request data should be rejected with clear errors

  @agent-request-processing @adr-001
  Scenario: AgentRequestProcessor generates dynamic parameters
    Given an AgentRequest with target_agent_type "user_input"
    And parameters {"prompt": "Enter character name", "multiline": False}
    And an AgentRequestProcessor instance
    When I call generate_dynamic_parameters() method
    Then it should return the parameters dictionary
    And parameter types should be preserved during generation
    And context should be applied if provided
    And the generation should be deterministic for same inputs

  @agent-request-processing @adr-001
  Scenario: AgentRequestProcessor validates agent request schemas
    Given agent request data as dictionary
    And an AgentRequestProcessor instance
    When I call validate_agent_request_schema() method
    Then it should return True for valid AgentRequest data
    And it should return False for invalid data structure
    And it should handle missing required fields gracefully
    And it should validate parameter dictionary structure

  @agent-request-processing @adr-001
  Scenario: AgentRequestProcessor extracts requests from JSON with error handling
    Given a JSON string containing agent_requests array
    And an AgentRequestProcessor instance
    When I call extract_agent_requests_from_json() method
    Then it should return list of validated AgentRequest objects
    And JSON parsing errors should be caught and chained (ADR-003)
    And schema validation errors should be caught and chained
    And the error messages should guide debugging
    And partial extraction should be prevented on validation failure

  @json-serialization @adr-001
  Scenario: All schemas support JSON serialization for inter-agent communication
    Given instances of all schema types (ScriptAgentInput, ScriptAgentOutput, AgentRequest, etc.)
    When I serialize each schema to JSON using model_dump()
    Then serialization should succeed without data loss
    And deserialization should recreate identical objects
    And nested objects should be properly serialized
    And type information should be preserved in the JSON structure

  @json-serialization @adr-001
  Scenario: Schema JSON round-trip preserves data integrity
    Given a ScriptAgentOutput with complex nested data
    And the output contains agent_requests with various parameter types
    When I serialize to JSON and deserialize back to schema object
    Then all field values should be identical
    And type annotations should be preserved
    And nested AgentRequest objects should be properly reconstructed
    And no data corruption should occur during round-trip

  @schema-extension @adr-001
  Scenario: New specialized schemas can extend base schemas
    Given the existing ScriptAgentInput and ScriptAgentOutput base schemas
    When I define a new specialized schema inheriting from base schemas
    Then the new schema should inherit all base fields
    And new fields should be additive to base functionality
    And schema validation should work for both base and specialized fields
    And backward compatibility should be maintained with base schema usage

  @error-handling @adr-001
  Scenario: Schema validation failures use proper exception chaining (ADR-003)
    Given a script agent with invalid input data
    When schema validation fails during processing
    Then the original validation error should be preserved
    And the error should be chained with contextual information
    And the error message should identify the failing schema and field
    And debugging information should include input data causing failure
    And the error chain should follow ADR-003 requirements

  @performance @adr-001
  Scenario: Schema validation completes under performance requirements
    Given a complex ScriptAgentInput with nested context and dependencies
    When schema validation is performed repeatedly
    Then each validation should complete in under 10 milliseconds
    And validation performance should not degrade with schema complexity
    And memory usage should remain constant across multiple validations
    And performance should scale linearly with input size

  @integration @adr-001
  Scenario: Schema system integrates with existing ensemble execution
    Given an ensemble configuration using script agents
    And the ensemble includes both EnhancedScriptAgent and regular agents
    When the ensemble executes with schema validation enabled
    Then schema-validated agents should integrate seamlessly
    And execution flow should respect schema validation timing
    And error handling should work across schema and non-schema agents
    And results should be compatible with existing result processing

  @backward-compatibility @adr-001
  Scenario: Schema system maintains compatibility with existing script patterns
    Given existing script agents using legacy JSON I/O patterns
    And new schema-based script agents in the same ensemble
    When both types of agents execute in mixed configuration
    Then legacy agents should continue working without modification
    And schema agents should produce compatible output for legacy consumers
    And the transition should be gradual and non-breaking
    And both patterns should coexist during migration period

  @architectural-compliance @adr-001
  Scenario: Schema implementation aligns with ADR-001 architectural decisions
    Given the complete Pydantic schema implementation
    When I review the implementation against ADR-001 requirements
    Then all script I/O should use Pydantic schema validation
    And dynamic parameter generation should be supported through AgentRequest
    And runtime validation should be automatic with clear error reporting
    And the architecture should be extensible for new script types
    And exception chaining should follow ADR-003 patterns
    And the implementation should enable the cyberpunk game scenario described in ADR-001