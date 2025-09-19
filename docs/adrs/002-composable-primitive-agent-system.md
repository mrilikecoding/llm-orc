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

Implement a **Pydantic-based Composable Primitive System** that transforms existing ad-hoc scripts into a unified, type-safe, discoverable ecosystem of building blocks.

## Detailed Design

### Core Architecture

#### 1. Universal Primitive Interface
```python
from abc import ABC, abstractmethod
from typing import TypeVar, Generic
from pydantic import BaseModel

# Type variables for input/output schemas
TInput = TypeVar('TInput', bound=BaseModel)
TOutput = TypeVar('TOutput', bound=BaseModel)

class Primitive(ABC, Generic[TInput, TOutput]):
    """Base class for all composable primitives."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this primitive."""
        pass
    
    @property  
    @abstractmethod
    def description(self) -> str:
        """Human-readable description."""
        pass
    
    @property
    @abstractmethod
    def category(self) -> str:
        """Category (user-interaction, file-ops, etc.)."""
        pass
    
    @abstractmethod
    async def execute(self, input_data: TInput) -> TOutput:
        """Execute the primitive operation."""
        pass
    
    @classmethod
    @abstractmethod
    def input_schema(cls) -> type[TInput]:
        """Return the input schema class."""
        pass
    
    @classmethod
    @abstractmethod  
    def output_schema(cls) -> type[TOutput]:
        """Return the output schema class."""
        pass
```

#### 2. Primitive Registry & Discovery
```python
class PrimitiveRegistry:
    """Central registry for all available primitives."""
    
    def __init__(self):
        self._primitives: Dict[str, Type[Primitive]] = {}
    
    def register(self, primitive_class: Type[Primitive]) -> None:
        """Register a primitive class."""
        self._primitives[primitive_class.name] = primitive_class
    
    def discover_by_category(self, category: str) -> List[Type[Primitive]]:
        """Find all primitives in a category."""
        return [p for p in self._primitives.values() if p.category == category]
    
    def get_schema_for_llm(self, primitive_name: str) -> Dict[str, Any]:
        """Get JSON schema for LLM function calling."""
        primitive_class = self._primitives[primitive_name]
        return primitive_class.input_schema().model_json_schema()
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

#### Primitive Chaining
```python
class WorkflowBuilder:
    """Builder for composing primitive workflows."""
    
    def __init__(self):
        self.steps: List[WorkflowStep] = []
    
    def add_primitive(
        self, 
        primitive_name: str,
        input_mapping: Dict[str, str] = None,
        condition: Optional[Callable] = None
    ) -> 'WorkflowBuilder':
        """Add a primitive to the workflow."""
        step = WorkflowStep(
            primitive_name=primitive_name,
            input_mapping=input_mapping or {},
            condition=condition
        )
        self.steps.append(step)
        return self
    
    def build(self) -> Workflow:
        """Build the complete workflow."""
        return Workflow(steps=self.steps)

# Example: Cyberpunk character creation workflow
workflow = (WorkflowBuilder()
    .add_primitive("generate_story_prompt", {
        "theme": "cyberpunk",
        "character_type": "protagonist"  
    })
    .add_primitive("get_user_input", {
        "prompt": "${generate_story_prompt.generated_prompt}"
    })
    .add_primitive("validate_story_input", {
        "user_input": "${get_user_input.user_input}",
        "validation_rules": "${generate_story_prompt.context_metadata.rules}"
    })
    .add_primitive("update_character_state", {
        "backstory": "${get_user_input.user_input}",
        "validation_result": "${validate_story_input.is_valid}"
    }, condition=lambda ctx: ctx["validate_story_input"]["is_valid"])
    .build())
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

This composable primitive system transforms llm-orc from a collection of ad-hoc scripts into a unified, type-safe ecosystem of building blocks. It enables the cyberpunk game scenario and countless other complex workflows while providing developer ergonomics and AI agent integration.

The Pydantic foundation ensures type safety and validation while maintaining the flexibility needed for dynamic AI-driven composition. This approach scales from simple single-primitive operations to complex multi-agent workflows with full type checking and runtime validation.