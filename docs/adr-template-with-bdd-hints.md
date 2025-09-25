# ADR Template with BDD Mapping Hints

This template enables automated ADR→BDD→TDD→Code orchestration by providing structured hints for the `llm-orc-adr-to-bdd-orchestrator` agent.

## Template Structure

```markdown
# ADR-XXX: [Title]

## Status
[Proposed|In Progress|Implemented|Validated|Accepted|Deprecated]

## BDD Mapping Hints
```yaml
# Primary behavioral capabilities this ADR enables
behavioral_capabilities:
  - capability: "Primary behavior or feature"
    given: "Initial state or precondition"
    when: "Triggering action or event"
    then: "Expected outcome or result"

  - capability: "Secondary behavior (if applicable)"
    given: "Different precondition"
    when: "Different trigger"
    then: "Different outcome"

# Test boundaries for automated test generation
test_boundaries:
  unit:
    - "Specific function or method to test"
    - "Pure logic component"
    - "Data transformation function"

  integration:
    - "Cross-component interaction"
    - "External service boundary"
    - "Database integration point"
    - "API endpoint flow"

# Validation rules for implementation
validation_rules:
  - "Type safety: All functions must have type annotations"
  - "Exception handling: Use proper exception chaining"
  - "Performance: Async operations where applicable"
  - "Coverage: Minimum 95% test coverage"

# Dependencies on other ADRs
related_adrs:
  - "ADR-001: Dependency relationship"
  - "ADR-002: Related component"

# Implementation components affected
implementation_scope:
  - "src/module/component.py"
  - "src/other/related_module.py"
```

## Context
[Original ADR context section]

## Decision
[Original ADR decision section]

## Consequences
[Original ADR consequences section]
```

## BDD Mapping Examples

### Example 1: Script Agent Dynamic Parameters

```yaml
## BDD Mapping Hints
behavioral_capabilities:
  - capability: "Script agents generate parameters for other agents"
    given: "A script with parameter_generator: true metadata"
    when: "Script executes in ensemble context"
    then: "Output becomes typed input for next agent"

  - capability: "Type validation across agent boundaries"
    given: "Script output with mixed types"
    when: "Converting to AgentRequest"
    then: "All types are validated and converted properly"

test_boundaries:
  unit:
    - ScriptResolver.discover_parameter_generators()
    - TypeConverter.json_to_agent_request()
    - AgentRequest.validate_parameter_types()

  integration:
    - script_to_agent_parameter_flow
    - ensemble_parameter_passing_pipeline
    - error_handling_across_agent_boundaries

validation_rules:
  - "Pydantic schema validation at all boundaries"
  - "Type safety with mypy strict mode"
  - "Exception chaining for parameter conversion errors"
  - "JSON serialization compatibility"

related_adrs:
  - "ADR-001: Pydantic schema system dependency"
  - "ADR-002: Composable primitive patterns"

implementation_scope:
  - "src/llm_orc/script_agent.py"
  - "src/llm_orc/ensemble/executor.py"
  - "src/llm_orc/schemas/agent_request.py"
```

### Example 2: Error Handling Patterns

```yaml
## BDD Mapping Hints
behavioral_capabilities:
  - capability: "Consistent exception chaining across components"
    given: "Any component that handles exceptions"
    when: "An error occurs during execution"
    then: "Original exception context is preserved with proper chaining"

  - capability: "Retry logic with exponential backoff"
    given: "A transient failure in external service"
    when: "Retry mechanism is triggered"
    then: "Backoff intervals increase exponentially with jitter"

test_boundaries:
  unit:
    - RetryHandler.calculate_backoff_delay()
    - ExceptionChainer.preserve_context()
    - CircuitBreaker.should_retry()

  integration:
    - end_to_end_error_propagation
    - retry_mechanism_with_real_failures
    - circuit_breaker_integration

validation_rules:
  - "All exceptions must use 'from' clause for chaining"
  - "Retry delays must include jitter to prevent thundering herd"
  - "Circuit breaker must activate after threshold failures"
  - "Error messages must be actionable and informative"

implementation_scope:
  - "src/llm_orc/error_handling/"
  - "src/llm_orc/agents/base_agent.py"
  - "src/llm_orc/ensemble/executor.py"
```

## Orchestrator Usage

When an ADR contains BDD mapping hints and is marked "In Progress":

1. **Automatic Detection**: `adr-implementation-orchestrator.sh` detects the ADR
2. **Agent Activation**: `llm-orc-adr-to-bdd-orchestrator` analyzes the hints
3. **BDD Generation**: Creates comprehensive feature files
4. **Test Decomposition**: Breaks scenarios into unit/integration specs
5. **TDD Orchestration**: Hands off to `llm-orc-tdd-specialist`
6. **Implementation**: Guides Red→Green→Refactor cycles

## Best Practices

### Writing Effective BDD Hints

1. **Be Specific**: Use concrete components and function names
2. **Cover Edge Cases**: Include error conditions and boundary cases
3. **Define Clear Boundaries**: Separate unit from integration concerns
4. **Include Dependencies**: Reference related ADRs and components

### Validation Rules Guidelines

- **Type Safety**: Always include mypy requirements
- **Error Handling**: Specify exception chaining patterns
- **Performance**: Include async/await patterns where applicable
- **Coverage**: Set specific coverage thresholds

### Integration Points

- **Related ADRs**: Explicitly list dependencies
- **Implementation Scope**: List specific files/modules
- **Test Boundaries**: Clear separation of unit vs integration

## Migration Guide

### Adding BDD Hints to Existing ADRs

1. Add `## BDD Mapping Hints` section after status
2. Analyze current implementation for behavioral capabilities
3. Identify unit vs integration test boundaries
4. Define validation rules based on coding standards
5. Update ADR status to "In Progress" to trigger orchestration

### Testing the Template

1. Use the template for a new ADR
2. Mark status as "In Progress"
3. Run `adr-implementation-orchestrator.sh` manually
4. Verify BDD scenarios are generated correctly
5. Validate test decomposition makes sense

This template bridges human architectural thinking with automated implementation, enabling efficient translation of design decisions into tested, working code.