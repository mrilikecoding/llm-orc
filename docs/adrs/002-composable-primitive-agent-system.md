# ADR-002: Composable Primitive Agent System

## Status
Implemented

## Implementation Status

- [x] BDD scenarios created in tests/bdd/features/adr-002-*.feature
- [x] Core implementation complete
- [x] All BDD scenarios passing
- [x] Integration tests passing
- [ ] Refactor phase complete
- [ ] Performance benchmarks met
- [ ] Documentation updated

## Implementation Progress Log
- **2025-09-19 14:03**: Status updated to Implemented
- **2025-09-19 14:02**: Status updated to In_Progress

## Context

### Current Primitive Landscape
The existing `.llm-orc/scripts/primitives/` contains a rich collection of building-block operations organized into categories:

- **`user-interaction/`**: User input collection, confirmations
- **`file-ops/`**: File read/write operations  
- **`data-transform/`**: JSON extraction, data manipulation
- **`control-flow/`**: Replication, conditional logic
- **`research/`**: Statistical analysis (t-tests, performance comparison)
- **`network-science/`**: Topology generation

### Limitations of Current Approach
1. **Inconsistent Interfaces**: Each primitive has ad-hoc JSON input/output schemas
2. **No Type Safety**: Runtime errors from invalid parameter types or missing fields
3. **Limited Composability**: No formal way to chain primitives or validate compatibility
4. **Manual LLM Integration**: LLM agents must manually construct JSON for primitives
5. **No Discovery Mechanism**: No way to programmatically discover available primitives and their capabilities

### Vision: Universal Composable System
Create a **unified primitive system** where:
- Every operation is a typed, composable building block
- LLM agents can discover and invoke primitives programmatically
- Complex workflows emerge from simple primitive composition  
- Full type safety with runtime validation
- Seamless integration between script and LLM agents

## Decision

Implement a **Script-Based Composable Primitive System** where primitives are standalone executable scripts (Python, JavaScript, etc.) that conform to type-safe contracts. Primitives live in the `llm-orchestra-library` submodule as content, not infrastructure, enabling composable workflows that interoperate seamlessly with LLM agents.

## Detailed Design

### Core Architecture

#### 1. Script-Based Primitive Contracts
Primitives are executable scripts that conform to `ScriptContract` (ADR-003) and communicate via JSON I/O. They can be written in any language (Python, JavaScript, Rust, shell) as long as they:

1. Accept JSON input via stdin following `ScriptAgentInput` schema (ADR-001)
2. Return JSON output via stdout following `ScriptAgentOutput` schema (ADR-001)
3. Implement `ScriptContract` metadata for discoverability (ADR-003)
4. Use category-specific schemas for type safety

**Python Primitive Example:**
```python
#!/usr/bin/env python3
"""get_user_input.py - Collect user input primitive (user-interaction category)"""
import json
import sys
from pydantic import BaseModel, Field

# Category-specific schemas
class UserInteractionInput(BaseModel):
    """Base input for user interaction primitives."""
    agent_name: str
    prompt: str
    context: dict[str, Any] = Field(default_factory=dict)

class UserInteractionOutput(BaseModel):
    """Base output for user interaction primitives."""
    success: bool
    user_response: str | None = None
    error: str | None = None

def main() -> None:
    # Read JSON input from stdin
    input_data = UserInteractionInput(**json.load(sys.stdin))

    # Execute primitive operation
    try:
        response = input(input_data.prompt + " ")
        output = UserInteractionOutput(success=True, user_response=response)
    except Exception as e:
        output = UserInteractionOutput(success=False, error=str(e))

    # Write JSON output to stdout
    print(output.model_dump_json())

if __name__ == "__main__":
    main()
```

**JavaScript Primitive Example:**
```javascript
#!/usr/bin/env node
// transform_data.js - Data transformation primitive (data-transform category)

const readline = require('readline');

// Read JSON input from stdin
let inputBuffer = '';
process.stdin.on('data', chunk => inputBuffer += chunk);
process.stdin.on('end', () => {
    const input = JSON.parse(inputBuffer);

    try {
        // Transform data
        const transformed = { transformed: input.source_data };

        // Write JSON output to stdout
        console.log(JSON.stringify({
            success: true,
            transformed_data: transformed
        }));
    } catch (error) {
        console.log(JSON.stringify({
            success: false,
            error: error.message
        }));
    }
});
```

#### 2. Primitive Registry & Discovery
**Implementation**: `src/llm_orc/core/execution/primitive_registry.py`

```python
class PrimitiveRegistry:
    """Registry for discovering and managing primitive script agents."""

    def discover_primitives(self) -> list[dict[str, Any]]:
        """Discover available primitive scripts in .llm-orc/scripts/primitives.

        Returns:
            List of primitive metadata dictionaries with:
            - name: script filename
            - path: absolute path to script
            - relative_path: path relative to project root
            - type: "primitive"
            - category: extracted from script metadata
            - executable: whether script has execute permissions
        """
        # Scans .llm-orc/scripts/primitives/ directory
        # Extracts metadata from script docstrings/comments
        # Caches results for performance

    def get_primitive_info(self, primitive_name: str) -> dict[str, Any]:
        """Get detailed information about a specific primitive."""
        # Returns primitive metadata including schema information

    def validate_primitive(self, primitive_name: str) -> dict[str, Any]:
        """Validate that a primitive conforms to ScriptAgentInput/Output schemas."""
        # Executes primitive with test input
        # Validates JSON I/O format
        # Returns validation result
```

### Category-Specific Schemas

#### User Interaction Primitives
```python
# Base schemas for user interaction
class UserInteractionInput(BaseModel):
    """Base input for user interaction primitives."""
    agent_name: str
    context: Dict[str, Any] = Field(default_factory=dict)

class UserInteractionOutput(BaseModel):
    """Base output for user interaction primitives."""
    success: bool
    user_response: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

# Specific user input schema
class GetUserInputInput(UserInteractionInput):
    prompt: str
    multiline: bool = False
    validation_pattern: Optional[str] = None
    max_length: Optional[int] = None

class GetUserInputOutput(UserInteractionOutput):
    user_input: str
    input_length: int
    validation_passed: bool

# Confirmation schema  
class ConfirmActionInput(UserInteractionInput):
    prompt: str
    default: Literal["y", "n"] = "n"
    
class ConfirmActionOutput(UserInteractionOutput):
    confirmed: bool
    user_choice: str
```

#### Data Transformation Primitives
```python
class DataTransformInput(BaseModel):
    """Base input for data transformation primitives."""
    source_data: Any
    context: Dict[str, Any] = Field(default_factory=dict)

class DataTransformOutput(BaseModel):
    """Base output for data transformation primitives."""
    success: bool
    transformed_data: Any = None
    error: Optional[str] = None
    transformation_metadata: Dict[str, Any] = Field(default_factory=dict)

# JSON extraction specific
class JsonExtractInput(DataTransformInput):
    json_data: Union[str, Dict[str, Any]]
    fields: List[str]
    strict_mode: bool = True

class JsonExtractOutput(DataTransformOutput):
    extracted_data: Dict[str, Any]
    missing_fields: List[str]
    extraction_stats: Dict[str, int]
```

#### Research & Analytics Primitives
```python
class ResearchInput(BaseModel):
    """Base input for research primitives."""
    study_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ResearchOutput(BaseModel):
    """Base output for research primitives."""
    success: bool
    analysis_results: Dict[str, Any]
    statistical_significance: Optional[bool] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    error: Optional[str] = None

# T-test specific schema
class TTestInput(ResearchInput):
    group1: List[float]
    group2: List[float]
    alpha: float = 0.05
    test_type: Literal["welch", "student"] = "welch"

class TTestOutput(ResearchOutput):
    t_statistic: float
    p_value: float
    degrees_of_freedom: float
    effect_size_cohens_d: float
    group1_stats: GroupStatistics
    group2_stats: GroupStatistics

class GroupStatistics(BaseModel):
    mean: float
    variance: float
    n: int
    std_dev: float
```

### LLM Agent Integration

#### Function Calling Schema Generation
```python
class LLMFunctionGenerator:
    """Generate function calling schemas for LLM agents."""
    
    def __init__(self, registry: PrimitiveRegistry):
        self.registry = registry
    
    def generate_function_definitions(self) -> List[Dict[str, Any]]:
        """Generate OpenAI function calling definitions."""
        functions = []
        
        for primitive_class in self.registry.get_all():
            schema = primitive_class.input_schema().model_json_schema()
            
            function_def = {
                "name": f"execute_{primitive_class.name}",
                "description": primitive_class.description,
                "parameters": schema
            }
            functions.append(function_def)
            
        return functions
    
    def execute_primitive_from_llm_call(
        self, 
        function_name: str, 
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute primitive from LLM function call."""
        primitive_name = function_name.replace("execute_", "")
        primitive_class = self.registry.get(primitive_name)
        
        # Validate and parse input
        input_data = primitive_class.input_schema()(**arguments)
        
        # Execute primitive
        result = await primitive_class().execute(input_data)
        
        return result.model_dump()
```

#### Dynamic Prompt Generation Example
```python
class PromptGeneratorPrimitive(Primitive[PromptGeneratorInput, PromptGeneratorOutput]):
    name = "generate_story_prompt"
    description = "Generate contextual prompts for story-based interactions"
    category = "ai-generation"
    
    async def execute(self, input_data: PromptGeneratorInput) -> PromptGeneratorOutput:
        # Use LLM to generate contextual prompt based on story state
        generated_prompt = await self._generate_contextual_prompt(
            theme=input_data.theme,
            character_state=input_data.character_state
        )
        
        return PromptGeneratorOutput(
            success=True,
            generated_prompt=generated_prompt,
            next_primitive_request=UserInputRequest(
                primitive_name="get_user_input",
                parameters={"prompt": generated_prompt}
            )
        )

class PromptGeneratorInput(BaseModel):
    theme: str
    character_state: Dict[str, Any]
    story_context: Optional[str] = None

class PromptGeneratorOutput(BaseModel):
    success: bool
    generated_prompt: str
    context_metadata: Dict[str, Any]
    next_primitive_request: Optional[UserInputRequest] = None
```

### Workflow Composition

#### Primitive Chaining via Ensemble Configuration
**Implementation**: `src/llm_orc/core/execution/primitive_composer.py`

Primitives are composed declaratively via YAML ensemble configurations:

```yaml
name: cyberpunk-character-creation
description: Interactive character creation with script and LLM agent interop

agents:
  # LLM agent generates contextual prompt
  - name: story-context-generator
    model_profile: creative-writer
    system_prompt: "Generate atmospheric cyberpunk character creation prompt"

  # Script primitive collects user input
  - name: get-user-backstory
    script: primitives/user-interaction/get_user_input.py
    depends_on: [story-context-generator]
    parameters:
      prompt: "${story-context-generator.generated_prompt}"
      multiline: true

  # LLM agent analyzes input
  - name: backstory-analyzer
    model_profile: narrative-analyst
    system_prompt: "Analyze character backstory for consistency"
    depends_on: [get-user-backstory]

  # Script primitive validates analysis
  - name: validation-checker
    script: primitives/data-transform/json_extract.py
    depends_on: [backstory-analyzer]
    parameters:
      fields: ["consistency_score", "issues"]

  # Conditional execution based on validation
  - name: request-revision
    script: primitives/user-interaction/confirm_action.py
    depends_on: [validation-checker]
    condition: "validation_checker.consistency_score < 0.7"
```

**Composition Validation**:
```python
class PrimitiveComposer:
    """Engine for composing and executing chained primitive script agents."""

    def compose_primitives(self, composition_config: dict[str, Any]) -> dict[str, Any]:
        """Compose primitives into an executable chain based on configuration."""
        # Validates primitives exist via file-based discovery
        # Resolves execution order via dependency analysis
        # Returns composition metadata

    def validate_composition(self, composition_config: dict[str, Any]) -> dict[str, Any]:
        """Validate that a composition configuration is type-safe and executable."""
        # Type compatibility checked via declared input_type/output_type
        # Detects circular dependencies
        # Returns validation result with errors/warnings
```

### Implementation Strategy

#### Phase 1: Core Infrastructure
1. **Base Primitive Classes**: Implement `Primitive`, `PrimitiveRegistry`
2. **Schema Migration**: Convert existing primitives to use Pydantic schemas
3. **Registry Population**: Auto-discover and register all primitives
4. **Basic Testing**: Unit tests for each primitive with schema validation

#### Phase 2: LLM Integration  
1. **Function Schema Generation**: Auto-generate OpenAI function definitions
2. **LLM Agent Updates**: Enable agents to discover and invoke primitives
3. **Dynamic Execution**: LLM agents can call primitives via function calls
4. **Composition Helpers**: Tools for chaining primitive operations

#### Phase 3: Advanced Workflows
1. **Workflow Builder**: Declarative workflow composition
2. **Conditional Logic**: Branching based on primitive outputs
3. **Error Handling**: Robust error propagation and recovery
4. **Performance Optimization**: Parallel primitive execution

#### Phase 4: Ecosystem Expansion
1. **Plugin Architecture**: Third-party primitive registration
2. **Primitive Marketplace**: Discoverable ecosystem of specialized primitives
3. **Visual Workflow Builder**: GUI for composing complex workflows
4. **Monitoring & Analytics**: Execution tracking and optimization

## Benefits

### Universal Composability
- **Building Block Approach**: Every operation becomes a reusable component
- **Type-Safe Composition**: Pydantic ensures compatibility between primitive chains
- **Dynamic Discovery**: LLM agents can explore and use new primitives automatically

### Developer Experience
- **Consistent Interface**: All primitives follow the same patterns
- **Auto-Documentation**: Schemas serve as living documentation
- **IDE Support**: Full autocomplete and type checking
- **Easy Testing**: Mock inputs/outputs with schema validation

### AI Agent Capabilities  
- **Function Calling**: LLM agents get automatic access to all primitives
- **Dynamic Workflows**: Agents can compose complex operations at runtime
- **Self-Discovery**: Agents explore capabilities without manual integration

### Extensibility
- **Plugin System**: Easy addition of new primitive categories
- **Third-Party Integration**: External systems can provide their own primitives
- **Schema Evolution**: Backward-compatible schema updates

## Trade-offs

### Migration Complexity
- **Existing Scripts**: All current primitives need schema migration
- **Breaking Changes**: Updates to primitive interfaces
- **Learning Curve**: Developers need to understand Pydantic patterns

### Performance Considerations
- **Schema Validation**: Additional runtime overhead
- **Type Checking**: More memory usage for schema objects
- **Discovery Overhead**: Registry lookup costs

### Complexity vs Flexibility
- **Abstraction Layer**: More complex than simple scripts
- **Schema Maintenance**: Need to maintain input/output schemas
- **Versioning Challenges**: Managing schema compatibility

## Success Metrics

### Technical Metrics
- **Schema Coverage**: 100% of primitives use typed schemas
- **Type Safety**: Zero runtime type errors in primitive chains  
- **Performance**: <10ms overhead for schema validation
- **Test Coverage**: >95% coverage with schema-based tests

### Usage Metrics
- **LLM Adoption**: LLM agents using >80% of available primitives
- **Composition Depth**: Average workflow length >3 primitives
- **Error Reduction**: 90% reduction in primitive integration errors
- **Developer Velocity**: 50% faster primitive development

### Ecosystem Metrics
- **Community Primitives**: >10 third-party primitives registered
- **Workflow Sharing**: Reusable workflow templates
- **Documentation Quality**: Auto-generated docs from schemas

## Examples

### Current Ad-Hoc Approach
```python
# Manual JSON construction - error prone
user_input_config = {
    "prompt": "What's your character's name?",
    "multiline": False  # Typo: should be "multiline" 
}

# No type checking - runtime failures
result = subprocess.run([...], input=json.dumps(user_input_config))
data = json.loads(result.stdout)  # May fail if malformed
```

### New Composable Approach
```python
# Type-safe primitive construction
user_input = GetUserInputPrimitive()
input_data = GetUserInputInput(
    agent_name="character_creator",
    prompt="What's your character's name?",
    multiline=False  # Type-checked at assignment
)

# Validated execution
result = await user_input.execute(input_data)  # Returns GetUserInputOutput
assert isinstance(result.user_input, str)  # Guaranteed by schema
```

### LLM Agent Integration
```python
# LLM agent automatically gets function definitions
functions = llm_function_generator.generate_function_definitions()

# LLM can call any primitive
llm_response = await openai.ChatCompletion.acreate(
    messages=[{"role": "user", "content": "Collect the user's character backstory"}],
    functions=functions,  # All primitives available
    function_call="auto"
)

# Automatic execution from LLM function call
if llm_response.function_call:
    result = await execute_primitive_from_llm_call(
        llm_response.function_call.name,
        json.loads(llm_response.function_call.arguments)
    )
```

## Decision Rationale

This script-based composable primitive system transforms llm-orc from a collection of ad-hoc scripts into a unified, type-safe ecosystem of building blocks. It enables diverse use cases from swarm network intelligence experiments to interactive narrative experiences, demonstrating that composability is achieved through contract conformance, not inheritance.

**Key architectural principles**:

1. **Primitives are content, not infrastructure** (ADR-006): Scripts live in `llm-orchestra-library`, enabling multi-language support and independent evolution
2. **Contract-based interoperability** (ADR-003): Type safety through `ScriptContract` conformance, not class inheritance
3. **Universal I/O schemas** (ADR-001): All agents (script and LLM) communicate via `ScriptAgentInput`/`ScriptAgentOutput`
4. **Declarative composition** (ADR-005): YAML ensemble configurations compose primitives with conditional dependencies and multi-turn conversations

This approach enables seamless interoperability between deterministic scripts (network analysis, statistical tests, user input) and LLM agents (reasoning, generation, analysis) while maintaining clean separation between orchestration infrastructure and orchestrable content.