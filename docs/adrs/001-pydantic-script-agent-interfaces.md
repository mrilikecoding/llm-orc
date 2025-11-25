# ADR-001: Pydantic-Based Script Agent Interface System

## Status
Implemented

## Implementation Status

- [x] BDD scenarios created in tests/bdd/features/adr-001-*.feature
- [x] Core implementation complete
- [x] All BDD scenarios passing
- [x] Integration tests passing
- [ ] Refactor phase complete
- [ ] Performance benchmarks met
- [ ] Documentation updated


## Implementation Progress Log
- **2025-09-19 10:54**: Status updated to Implemented

## BDD Integration
- **Scenario File**: tests/bdd/features/issue-24-script-agents.feature
- **Test Command**: `uv run pytest tests/bdd/ -k "json_inputoutput_contract"`
- **Priority Scenario**: "Script agent executes with JSON input/output contract" ✅

## Context

### Current Limitations
The existing script agent system has several limitations that prevent dynamic inter-agent communication:

1. **Static Template Substitution**: The current `${agent.output}` template system is rigid and only supports simple string replacement
2. **No Type Safety**: Script inputs/outputs lack validation and type checking
3. **Limited Inter-Agent Communication**: No structured way for agents to generate parameters or requests for other agents
4. **Inflexible User Interactions**: User prompts must be predefined in configuration, preventing dynamic prompt generation

### Use Cases Requiring Dynamic Communication
1. **Dynamic Prompt Generation**: An AI agent generates a contextual prompt that a user-input agent uses
2. **Conditional Workflows**: Script execution paths based on validated outputs from previous agents  
3. **Complex User Interactions**: Multi-step dialogs with validation and retry logic
4. **Game Development**: Dynamic character creation, procedural content generation

Example scenario:
```
Story Agent → generates "What's your character's backstory in this cyberpunk world?"
User Input Agent → uses generated prompt to collect user response
Validation Agent → checks response against story constraints
Game State Agent → updates character with validated backstory
```

## Decision

Implement a **Pydantic-based schema system** for script agent interfaces that provides:

1. **Type-Safe Communication**: All script inputs/outputs use Pydantic models
2. **Dynamic Parameter Generation**: Agents can output structured requests for other agents
3. **Runtime Validation**: Automatic validation of data flowing between agents
4. **Extensible Architecture**: Easy addition of new script types and interaction patterns

## Detailed Design

### Core Schema Architecture

```python
# Base schemas for all script agents
class ScriptAgentInput(BaseModel):
    """Base input schema for all script agents."""
    agent_name: str
    input_data: str
    context: Dict[str, Any] = Field(default_factory=dict)
    dependencies: Dict[str, Any] = Field(default_factory=dict)

class ScriptAgentOutput(BaseModel):
    """Base output schema for all script agents."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    agent_requests: List[AgentRequest] = Field(default_factory=list)

class AgentRequest(BaseModel):
    """Request for another agent to perform an action."""
    target_agent_type: str
    parameters: Dict[str, Any]
    priority: int = 0
```

### Specialized Schemas

```python
# User interaction schemas
class UserInputRequest(BaseModel):
    """Schema for requesting user input."""
    prompt: str
    multiline: bool = False
    validation_pattern: Optional[str] = None
    retry_message: Optional[str] = None
    max_attempts: int = 3

class UserInputOutput(ScriptAgentOutput):
    """Output from user input collection."""
    user_input: str
    attempts_used: int
    validation_passed: bool

# File operation schemas  
class FileOperationRequest(BaseModel):
    """Schema for file operations."""
    operation: Literal["read", "write", "append", "delete"]
    path: str
    content: Optional[str] = None
    encoding: str = "utf-8"

class FileOperationOutput(ScriptAgentOutput):
    """Output from file operations."""
    path: str
    size: int
    bytes_processed: int
    operation_performed: str
```

### Dynamic Communication Patterns

#### Pattern 1: Prompt Generation Chain
```python
# Agent 1: Generate contextual prompt
class PromptGeneratorOutput(ScriptAgentOutput):
    generated_prompt: str
    context_metadata: Dict[str, Any]
    agent_requests: List[AgentRequest] = [
        AgentRequest(
            target_agent_type="user_input",
            parameters={"prompt": generated_prompt, "multiline": False}
        )
    ]

# Agent 2: Collect user input with generated prompt
class UserInputAgent:
    def execute(self, input_data: ScriptAgentInput) -> UserInputOutput:
        # Uses dynamically generated prompt from previous agent
        prompt = input_data.dependencies["prompt_generator"].generated_prompt
        # ... collect user input
```

#### Pattern 2: Validation Chain
```python
class ValidationRequest(BaseModel):
    data_to_validate: str
    validation_rules: List[str]
    error_message: str

class ValidationOutput(ScriptAgentOutput):
    is_valid: bool
    validation_errors: List[str]
    suggestions: List[str]
```

### Implementation Strategy

1. **Phase 1**: Create base schemas and update existing user input scripts
2. **Phase 2**: Implement agent request/response system  
3. **Phase 3**: Add specialized schemas for different script types
4. **Phase 4**: Implement dynamic parameter passing and validation

## Benefits

### Type Safety & Validation
- **Runtime Validation**: Automatic validation of all agent inputs/outputs
- **IDE Support**: Full autocomplete and type checking during development
- **Error Prevention**: Catch type mismatches and invalid data early

### Dynamic Behavior
- **Agent-Generated Prompts**: Agents can create contextual prompts for user interaction
- **Conditional Execution**: Scripts can modify execution flow based on validated data
- **Complex Workflows**: Multi-step processes with branching logic

### Extensibility
- **Easy Schema Addition**: New script types just need new Pydantic models
- **Version Compatibility**: Schema evolution with backward compatibility
- **Plugin Architecture**: Third-party script types can define their own schemas

### Developer Experience
- **Clear Contracts**: Explicit interfaces between agents
- **Better Testing**: Easier to mock and test with typed interfaces
- **Documentation**: Self-documenting code via schema definitions

## Trade-offs

### Complexity
- **Learning Curve**: Developers need to understand Pydantic schema system
- **Migration Effort**: Existing scripts need to be updated to use schemas

### Performance
- **Validation Overhead**: Additional runtime validation costs
- **Memory Usage**: Schema objects may use more memory than simple dictionaries

### Flexibility vs Structure
- **Less Ad-hoc Usage**: More structured approach may feel restrictive initially
- **Schema Evolution**: Need to manage schema versioning and compatibility

## Implementation Plan

### Phase 1: Foundation (Current Sprint)
1. Create base schema classes in `src/llm_orc/schemas/`
2. Update user input scripts to use schemas
3. Implement schema validation in ensemble execution
4. Add comprehensive tests for schema system

### Phase 2: Dynamic Communication  
1. Implement agent request/response system
2. Add dynamic parameter passing between agents
3. Create examples of dynamic prompt generation

### Phase 3: Advanced Features
1. Add validation chains and conditional execution
2. Implement complex multi-step user interactions
3. Create plugin architecture for custom schemas

## Examples

### Current Static Approach
```yaml
- name: get-input
  script: primitives/user-interaction/get_user_input.py
  parameters:
    prompt: "What is your name?"

- name: echo-result  
  script: primitives/file-ops/write_file.py
  depends_on: [get-input]
  parameters:
    content: "User said: ${get-input.output}"  # Template substitution
```

### New Dynamic Approach  
```yaml
- name: story-generator
  script: primitives/ai/generate_story_prompt.py
  parameters:
    theme: "cyberpunk"
    character_type: "protagonist"

- name: collect-backstory
  script: primitives/user-interaction/get_user_input.py  
  depends_on: [story-generator]
  # Prompt comes from story-generator output schema

- name: validate-backstory
  script: primitives/validation/story_validator.py
  depends_on: [collect-backstory]
  # Validation rules from story-generator metadata
```

## Decision Rationale

This approach provides the foundation for building complex, dynamic agent workflows while maintaining type safety and clear interfaces. It enables the cyberpunk game scenario and many other advanced use cases while being backward compatible with simpler static configurations.

The Pydantic schema system strikes the right balance between structure and flexibility, providing compile-time and runtime guarantees while enabling powerful dynamic behaviors.