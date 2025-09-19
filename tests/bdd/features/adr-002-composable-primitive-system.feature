Feature: ADR-002 Composable Primitive Agent System
  """
  LLM Development Context:

  Requirements: Implement Pydantic-based composable primitive system that transforms
  existing ad-hoc scripts into unified, type-safe, discoverable ecosystem of building blocks.

  Architectural constraints (ADR-002):
  - Universal Primitive interface with type variables (TInput, TOutput bound to BaseModel)
  - PrimitiveRegistry for discovery and schema generation (ADR-001)
  - LLM function calling integration with auto-generated schemas (ADR-003)
  - Workflow composition with dependency resolution and type checking
  - Exception chaining for primitive execution failures (ADR-003)

  Implementation patterns:
  - Primitive[TInput, TOutput] abstract base class with schema validation
  - Category-based organization (user-interaction, file-ops, data-transform, research)
  - WorkflowBuilder for declarative primitive composition with input mapping
  - LLMFunctionGenerator for OpenAI function calling schema generation
  - Type-safe primitive chaining with runtime compatibility validation

  Critical validations:
  - Schema migration from existing ad-hoc scripts to Pydantic models
  - Type compatibility validation between chained primitives
  - Dynamic primitive discovery and registration mechanisms
  - LLM agent integration with function calling schemas
  - Error propagation with proper exception chaining
  - Performance optimization for schema validation overhead (<10ms)
  """

  As a developer using llm-orc
  I want a unified composable primitive system with type safety
  So that I can build complex workflows from reusable building blocks with full validation

  Background:
    Given llm-orc is properly configured
    And the primitive system is initialized

  @priority @adr-002
  Scenario: Primitive registry discovers and registers all available primitives
    Given the primitive registry is initialized
    When primitive discovery is executed
    Then all primitives in .llm-orc/scripts/primitives should be discovered
    And each primitive should have complete metadata extracted
    And primitives should be organized by category (user-interaction, file-ops, etc.)
    And the registry should cache discovery results for performance

  @adr-002
  Scenario: Universal Primitive interface provides type-safe execution
    Given a primitive implementing the universal Primitive interface
    When the primitive is executed with typed input schema
    Then input validation should occur using Pydantic models
    And execution should return typed output schema
    And type safety should be enforced at compile and runtime
    And schema violations should raise clear validation errors

  @adr-002
  Scenario: Primitive composition validates type compatibility between chains
    Given multiple primitives with different input/output schemas
    When primitives are composed into a workflow chain
    Then type compatibility should be validated between consecutive primitives
    And incompatible type chains should be rejected with clear error messages
    And compatible chains should be marked as valid for execution
    And validation should happen before any execution begins

  @adr-002
  Scenario: Configuration-driven assembly creates executable workflows from YAML
    Given a YAML workflow configuration with primitive composition
    When the workflow assembly engine processes the configuration
    Then primitives should be validated and resolved from the registry
    And dependency resolution should determine correct execution order
    And input mapping should be validated for type compatibility
    And the resulting workflow should be executable with schema validation

  @adr-002
  Scenario: Error propagation handles primitive failures with proper chaining
    Given a workflow with multiple chained primitives
    When a primitive in the chain fails during execution
    Then the failure should be caught and chained properly (ADR-003)
    And subsequent primitives should receive error context
    And the workflow should support graceful degradation or termination
    And error details should preserve original exception context

  @adr-002
  Scenario: Dependency resolution determines automatic primitive ordering
    Given primitives with declared dependencies on other primitive outputs
    When the workflow resolver analyzes dependencies
    Then primitive execution order should be determined via topological sorting
    And circular dependencies should be detected and rejected
    And missing dependencies should be identified before execution
    And parallel execution should be optimized where dependencies allow

  @adr-002
  Scenario: Reusable components enable primitives across different compositions
    Given a set of validated primitive components
    When multiple workflows reference the same primitives
    Then primitives should be reusable across different workflow contexts
    And primitive state should be isolated between different executions
    And shared primitives should maintain consistent schema validation
    And registry should efficiently manage primitive instances

  @adr-002
  Scenario: Performance optimization executes primitives in parallel where possible
    Given a workflow with primitives having no interdependencies
    When the workflow executor analyzes the dependency graph
    Then independent primitives should be identified for parallel execution
    And execution should leverage async/await patterns for concurrency
    And overall workflow execution time should be minimized
    And parallel execution should not compromise error handling

  @adr-002
  Scenario: Validation chains ensure primitive outputs before passing to next primitive
    Given chained primitives with strict input requirements
    When one primitive completes and passes output to the next
    Then output schema validation should occur before input to next primitive
    And invalid intermediate outputs should halt the chain with clear errors
    And validation should include both schema compliance and business rules
    And validation results should be logged for debugging workflow issues

  @adr-002 @architectural-compliance
  Scenario: Primitive schemas extend existing Pydantic foundation (ADR-001)
    Given the existing Pydantic schema foundation from ADR-001
    When new primitive schemas are defined
    Then they should extend ScriptAgentInput/Output base schemas
    And maintain backward compatibility with existing script agent contracts
    And leverage existing schema validation infrastructure
    And integrate seamlessly with current agent execution patterns

  @adr-002 @architectural-compliance
  Scenario: LLM function calling integration auto-generates schemas
    Given primitives registered in the primitive registry
    When LLM agents request available function definitions
    Then function calling schemas should be auto-generated from primitive input schemas
    And LLM agents should be able to invoke primitives via function calls
    And function call arguments should be validated against primitive schemas
    And primitive execution results should be returned in LLM-compatible format

  @adr-002 @architectural-compliance
  Scenario: Category-based primitive organization maintains existing structure
    Given existing primitive categories (user-interaction, file-ops, data-transform, etc.)
    When primitives are registered and discovered
    Then category organization should be preserved and enhanced
    And primitives should be discoverable by category for targeted operations
    And category metadata should support LLM agent primitive selection
    And new categories should be extensible through configuration

  @adr-002 @error-handling
  Scenario: Primitive execution failures use proper exception chaining (ADR-003)
    Given a primitive that encounters an execution error
    When the primitive fails during schema validation or execution
    Then the error should be caught and chained with contextual information
    And the error chain should preserve original exception details
    And error messages should include primitive name and execution context
    And error handling should support debugging and recovery strategies

  @adr-002 @error-handling
  Scenario: Workflow composition handles invalid configurations gracefully
    Given a workflow configuration with schema or dependency errors
    When the workflow composer attempts to build the workflow
    Then configuration errors should be detected during validation phase
    And clear error messages should identify specific configuration problems
    And partial workflow building should be prevented with invalid configurations
    And error context should guide developers to fix configuration issues

  @adr-002 @performance
  Scenario: Schema validation overhead remains under 10ms per primitive
    Given a primitive with complex input/output schemas
    When the primitive is executed with schema validation enabled
    Then schema validation should complete in under 10 milliseconds
    And validation performance should scale linearly with schema complexity
    And caching should be used to optimize repeated schema validations
    And performance should not degrade with increasing primitive count

  @adr-002 @integration
  Scenario: Primitive system integrates with existing ensemble execution
    Given the existing EnsembleExecutor and agent coordination infrastructure
    When primitive workflows are executed within ensemble contexts
    Then primitive execution should integrate seamlessly with current agent patterns
    And ensemble-level error handling should work with primitive error chaining
    And primitive results should be compatible with existing result synthesis
    And execution tracking should include primitive-level metrics

  @adr-002 @discovery
  Scenario: Dynamic primitive discovery supports runtime registration
    Given a system with dynamically loaded primitive modules
    When new primitives are added to the system at runtime
    Then the registry should support dynamic primitive registration
    And newly registered primitives should be immediately available for workflow composition
    And discovery should handle primitive version conflicts and updates
    And registry state should remain consistent during dynamic updates

  @adr-002 @workflow-composition
  Scenario: WorkflowBuilder enables declarative primitive composition
    Given the WorkflowBuilder interface for composing primitive workflows
    When developers create workflows using the builder pattern
    Then primitive chains should be declaratively defined with clear syntax
    And input mapping between primitives should be explicitly configured
    And conditional execution should be supported for branching workflows
    And the builder should validate workflow consistency before execution

  @adr-002 @workflow-composition
  Scenario: Input mapping resolves dependencies between primitive outputs and inputs
    Given primitives with outputs that feed into subsequent primitive inputs
    When workflow input mapping is configured with reference expressions
    Then output references (${primitive.field}) should be resolved correctly
    And missing reference targets should be detected during validation
    And complex mapping transformations should be supported where needed
    And mapping resolution should preserve type safety across the chain

  @adr-002 @extensibility
  Scenario: Plugin architecture supports third-party primitive registration
    Given a plugin system for external primitive providers
    When third-party primitives are loaded into the system
    Then external primitives should register using the same universal interface
    And plugin primitives should be discoverable alongside system primitives
    And version compatibility should be managed for plugin primitives
    And plugin isolation should prevent conflicts between primitive providers