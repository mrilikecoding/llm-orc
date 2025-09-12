# ADR-003: Testable Script Agent Contracts & Ecosystem Interoperability

## Status
Proposed

## Context

### Current Ecosystem Challenges
With the implementation of Pydantic-based script interfaces (ADR-001) and the composable primitive system (ADR-002), we need to ensure **ecosystem-wide interoperability** across:

1. **Core Primitives** (`.llm-orc/scripts/primitives/`) - Built-in building blocks
2. **Base Script Examples** - Reference implementations and templates  
3. **User-Submitted Scripts** - Community and third-party contributions
4. **LLM Agent Interactions** - Dynamic script invocation and composition

### The Interoperability Problem
Without enforceable contracts and testing:
- User scripts may not integrate properly with the primitive system
- Breaking changes in primitives could break downstream scripts
- No way to validate that scripts conform to expected interfaces
- LLM agents can't reliably compose arbitrary script combinations
- Community contributions may introduce incompatible patterns

### Vision: Testable Contract Ecosystem
Create a **contract-driven development system** where:
- All scripts must implement testable Pydantic interfaces
- CI automatically validates contract compliance across the ecosystem
- Scripts can declare their capabilities and requirements
- LLM agents can dynamically validate script compatibility
- Community contributions are automatically tested for compliance

## Decision

Implement a **Testable Script Agent Contract System** that extends the Pydantic interfaces to provide:

1. **Contract Enforcement**: CI tests that validate all scripts implement required interfaces
2. **Capability Declaration**: Scripts declare their inputs, outputs, and dependencies
3. **Compatibility Testing**: Automated testing of script composition scenarios
4. **Extension Patterns**: Standardized patterns for arbitrary script execution, API calls, and data enrichment
5. **Community Integration**: Submission and validation pipeline for user-contributed scripts

## Detailed Design

### Core Contract System

#### 1. Universal Script Contract
```python
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field

class ScriptCapability(str, Enum):
    """Enumeration of script capabilities."""
    USER_INTERACTION = "user_interaction"
    DATA_TRANSFORMATION = "data_transformation"
    FILE_OPERATIONS = "file_operations"
    API_INTEGRATION = "api_integration"
    COMPUTATION = "computation"
    CONTROL_FLOW = "control_flow"
    EXTERNAL_EXECUTION = "external_execution"

class ScriptDependency(BaseModel):
    """Declaration of script dependencies."""
    name: str
    version: Optional[str] = None
    optional: bool = False
    pip_package: Optional[str] = None
    system_command: Optional[str] = None

class ScriptMetadata(BaseModel):
    """Comprehensive metadata for script contract."""
    name: str
    version: str
    description: str
    author: str
    category: str
    capabilities: List[ScriptCapability]
    dependencies: List[ScriptDependency] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    examples: List[Dict[str, Any]] = Field(default_factory=list)
    test_cases: List[Dict[str, Any]] = Field(default_factory=list)

class ScriptContract(ABC):
    """Universal contract that all scripts must implement."""
    
    @property
    @abstractmethod
    def metadata(self) -> ScriptMetadata:
        """Script metadata and capabilities."""
        pass
    
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
    
    @abstractmethod
    async def execute(self, input_data: BaseModel) -> BaseModel:
        """Execute the script with validated input."""
        pass
    
    @abstractmethod
    def get_test_cases(self) -> List['TestCase']:
        """Return test cases for contract validation."""
        pass

class TestCase(BaseModel):
    """Test case for script validation."""
    name: str
    description: str
    input_data: Dict[str, Any]
    expected_output: Dict[str, Any]
    should_succeed: bool = True
    setup_commands: List[str] = Field(default_factory=list)
    cleanup_commands: List[str] = Field(default_factory=list)
```

#### 2. Extension Patterns for Arbitrary Capabilities

##### Arbitrary Script Execution
```python
class ArbitraryExecutionInput(BaseModel):
    """Input for arbitrary script execution."""
    script_content: str
    language: Literal["python", "bash", "javascript", "powershell"]
    environment_variables: Dict[str, str] = Field(default_factory=dict)
    working_directory: Optional[str] = None
    timeout_seconds: int = 30
    capture_output: bool = True
    security_sandbox: bool = True

class ArbitraryExecutionOutput(BaseModel):
    """Output from arbitrary script execution."""
    success: bool
    exit_code: int
    stdout: str
    stderr: str
    execution_time_seconds: float
    error: Optional[str] = None
    security_violations: List[str] = Field(default_factory=list)

class ArbitraryExecutionScript(ScriptContract):
    """Contract for arbitrary script execution primitive."""
    
    metadata = ScriptMetadata(
        name="arbitrary_execution",
        version="1.0.0", 
        description="Execute arbitrary code in sandboxed environment",
        author="llm-orchestra",
        category="execution",
        capabilities=[ScriptCapability.EXTERNAL_EXECUTION],
        dependencies=[
            ScriptDependency(name="docker", system_command="docker"),
            ScriptDependency(name="firejail", system_command="firejail", optional=True)
        ]
    )
```

##### API Integration Pattern  
```python
class APICallInput(BaseModel):
    """Input for API integration scripts."""
    url: str = Field(..., description="API endpoint URL")
    method: Literal["GET", "POST", "PUT", "DELETE", "PATCH"] = "GET"
    headers: Dict[str, str] = Field(default_factory=dict)
    query_params: Dict[str, Any] = Field(default_factory=dict)
    body: Optional[Union[Dict[str, Any], str]] = None
    timeout_seconds: int = 30
    retry_attempts: int = 3
    auth_token: Optional[str] = None
    rate_limit_delay: float = 0.0

class APICallOutput(BaseModel):
    """Output from API calls."""
    success: bool
    status_code: int
    response_data: Any = None
    response_headers: Dict[str, str] = Field(default_factory=dict)
    response_time_seconds: float
    retry_count: int = 0
    error: Optional[str] = None
    rate_limited: bool = False

class DataEnrichmentInput(BaseModel):
    """Input for data enrichment operations."""
    source_data: Any
    enrichment_apis: List[APICallInput]
    merge_strategy: Literal["replace", "merge", "append"] = "merge"
    enrichment_fields: List[str] = Field(default_factory=list)
    parallel_requests: bool = True
    fallback_on_error: bool = True

class DataEnrichmentOutput(BaseModel):
    """Output from data enrichment."""
    success: bool
    enriched_data: Any
    enrichment_metadata: Dict[str, Any] = Field(default_factory=dict)
    api_call_results: List[APICallOutput] = Field(default_factory=list)
    error: Optional[str] = None
```

### CI Contract Enforcement System

#### 1. Automated Contract Validation
```python
class ContractValidator:
    """Validates script contracts in CI pipeline."""
    
    def __init__(self, script_directory: str):
        self.script_directory = Path(script_directory)
        self.validation_errors: List[str] = []
    
    def validate_all_scripts(self) -> bool:
        """Validate all scripts in directory tree."""
        scripts = self._discover_scripts()
        
        for script_path in scripts:
            try:
                self._validate_single_script(script_path)
            except Exception as e:
                self.validation_errors.append(f"{script_path}: {e}")
        
        return len(self.validation_errors) == 0
    
    def _validate_single_script(self, script_path: Path) -> None:
        """Validate a single script contract."""
        # 1. Import and instantiate script
        script_instance = self._load_script(script_path)
        
        # 2. Validate metadata compliance
        self._validate_metadata(script_instance.metadata)
        
        # 3. Validate input/output schemas
        self._validate_schemas(script_instance)
        
        # 4. Run test cases
        self._run_test_cases(script_instance)
        
        # 5. Test schema compatibility with existing ecosystem
        self._test_ecosystem_compatibility(script_instance)

    def _validate_schemas(self, script: ScriptContract) -> None:
        """Validate Pydantic schema compliance."""
        input_schema = script.input_schema()
        output_schema = script.output_schema()
        
        # Ensure schemas are valid Pydantic models
        assert issubclass(input_schema, BaseModel)
        assert issubclass(output_schema, BaseModel)
        
        # Validate schema can generate JSON Schema for LLM function calling
        input_json_schema = input_schema.model_json_schema()
        assert "properties" in input_json_schema
        
        # Ensure required fields are properly marked
        self._validate_required_fields(input_schema, input_json_schema)

    def _run_test_cases(self, script: ScriptContract) -> None:
        """Execute all test cases for the script."""
        test_cases = script.get_test_cases()
        
        for test_case in test_cases:
            # Setup
            self._run_setup_commands(test_case.setup_commands)
            
            try:
                # Parse input using schema
                input_data = script.input_schema()(**test_case.input_data)
                
                # Execute script
                result = await script.execute(input_data)
                
                # Validate output schema
                assert isinstance(result, script.output_schema())
                
                # Validate expected output
                if test_case.should_succeed:
                    self._validate_expected_output(result, test_case.expected_output)
                
            finally:
                # Cleanup
                self._run_cleanup_commands(test_case.cleanup_commands)

    def _test_ecosystem_compatibility(self, script: ScriptContract) -> None:
        """Test compatibility with existing ecosystem."""
        # Test that script can be composed with primitives
        for primitive in self._get_compatible_primitives(script):
            self._test_script_composition(script, primitive)
```

#### 2. CI Pipeline Integration
```yaml
# .github/workflows/script-contract-validation.yml
name: Script Contract Validation

on:
  push:
    paths: 
      - '.llm-orc/scripts/**'
      - 'src/llm_orc/schemas/**'
  pull_request:
    paths:
      - '.llm-orc/scripts/**'
      - 'src/llm_orc/schemas/**'

jobs:
  validate-contracts:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -e .
      
      - name: Validate Core Primitives
        run: |
          python -m llm_orc.testing.contract_validator \
            --directory .llm-orc/scripts/primitives \
            --level core
            
      - name: Validate Base Examples
        run: |
          python -m llm_orc.testing.contract_validator \
            --directory .llm-orc/scripts/examples \
            --level examples
            
      - name: Validate Community Scripts
        run: |
          python -m llm_orc.testing.contract_validator \
            --directory .llm-orc/scripts/community \
            --level community
            
      - name: Test Script Composition
        run: |
          python -m llm_orc.testing.composition_tester \
            --test-all-combinations \
            --max-chain-length 5
            
      - name: Generate LLM Function Schemas
        run: |
          python -m llm_orc.schemas.function_generator \
            --output schemas/llm-functions.json \
            --validate-callable
```

### Community Contribution Pipeline

#### 1. Script Submission Template
```python
# .llm-orc/scripts/community/template.py
"""
Community Script Template - Copy and modify this template for submissions
"""

from typing import Any, List
from pydantic import BaseModel, Field
from llm_orc.schemas.contracts import ScriptContract, ScriptMetadata, ScriptCapability, TestCase

class YourScriptInput(BaseModel):
    """Input schema for your script - customize as needed."""
    # Add your input fields here
    example_field: str = Field(..., description="Description for LLM agents")
    optional_field: int = Field(default=42, description="Optional parameter")

class YourScriptOutput(BaseModel):
    """Output schema for your script - customize as needed."""
    success: bool
    result: Any = None
    error: Optional[str] = None
    # Add your output fields here

class YourScript(ScriptContract):
    """Your script implementation."""
    
    metadata = ScriptMetadata(
        name="your_script_name",
        version="1.0.0",
        description="What your script does",
        author="your-github-username", 
        category="your-category",
        capabilities=[ScriptCapability.DATA_TRANSFORMATION],  # Update as appropriate
        dependencies=[],  # List any dependencies
        tags=["example", "template"]
    )
    
    @classmethod
    def input_schema(cls) -> type[BaseModel]:
        return YourScriptInput
    
    @classmethod
    def output_schema(cls) -> type[BaseModel]:
        return YourScriptOutput
    
    async def execute(self, input_data: YourScriptInput) -> YourScriptOutput:
        """Implement your script logic here."""
        try:
            # Your implementation goes here
            result = self._do_your_processing(input_data)
            
            return YourScriptOutput(
                success=True,
                result=result
            )
        except Exception as e:
            return YourScriptOutput(
                success=False, 
                error=str(e)
            )
    
    def get_test_cases(self) -> List[TestCase]:
        """Define test cases for your script."""
        return [
            TestCase(
                name="basic_functionality",
                description="Test basic script functionality", 
                input_data={"example_field": "test_value"},
                expected_output={"success": True, "result": "expected_result"}
            ),
            TestCase(
                name="error_handling",
                description="Test error handling",
                input_data={"example_field": ""},  # Invalid input
                expected_output={"success": False},
                should_succeed=False
            )
        ]
    
    def _do_your_processing(self, input_data: YourScriptInput) -> Any:
        """Your private implementation methods."""
        # Implement your logic here
        return f"Processed: {input_data.example_field}"

# For standalone execution
if __name__ == "__main__":
    import asyncio
    import json
    import sys
    
    async def main():
        # Read input from stdin
        if not sys.stdin.isatty():
            input_json = json.loads(sys.stdin.read())
        else:
            input_json = {"example_field": "test"}
        
        # Create and execute script
        script = YourScript()
        input_data = script.input_schema()(**input_json)
        result = await script.execute(input_data)
        
        # Output result
        print(result.model_dump_json(indent=2))
    
    asyncio.run(main())
```

#### 2. Submission Validation Process
```python
class CommunitySubmissionValidator:
    """Validates community script submissions."""
    
    def validate_submission(self, script_path: Path) -> SubmissionReport:
        """Comprehensive validation of community submission."""
        report = SubmissionReport(script_path=script_path)
        
        # 1. Contract compliance
        report.contract_valid = self._validate_contract(script_path)
        
        # 2. Security scanning
        report.security_issues = self._scan_security(script_path)
        
        # 3. Performance testing  
        report.performance_metrics = self._run_performance_tests(script_path)
        
        # 4. Ecosystem compatibility
        report.compatibility_score = self._test_ecosystem_fit(script_path)
        
        # 5. Code quality metrics
        report.quality_score = self._analyze_code_quality(script_path)
        
        return report

class SubmissionReport(BaseModel):
    """Report for community script submission."""
    script_path: Path
    contract_valid: bool
    security_issues: List[str] = Field(default_factory=list)
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    compatibility_score: float
    quality_score: float
    approved: bool = False
    feedback: List[str] = Field(default_factory=list)
```

### Advanced Extension Examples

#### 1. Database Integration Script
```python
class DatabaseQueryInput(BaseModel):
    """Input for database query operations."""
    connection_string: str = Field(..., description="Database connection string")
    query: str = Field(..., description="SQL query to execute")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: int = 30
    fetch_size: Optional[int] = None

class DatabaseQueryOutput(BaseModel):
    """Output from database operations."""
    success: bool
    rows_affected: int
    data: List[Dict[str, Any]] = Field(default_factory=list)
    execution_time_seconds: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

#### 2. Machine Learning Pipeline Script
```python
class MLPipelineInput(BaseModel):
    """Input for ML pipeline operations."""
    model_config: Dict[str, Any]
    training_data_path: str
    validation_split: float = 0.2
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    output_model_path: str

class MLPipelineOutput(BaseModel):
    """Output from ML pipeline."""
    success: bool
    model_metrics: Dict[str, float] = Field(default_factory=dict)
    model_path: str
    training_time_seconds: float
    validation_score: float
    error: Optional[str] = None
```

### Testing Strategy

#### 1. Contract Compliance Tests
```python
def test_all_primitives_have_contracts():
    """Ensure all primitives implement the contract interface."""
    primitives_dir = Path(".llm-orc/scripts/primitives")
    
    for script_file in primitives_dir.rglob("*.py"):
        script_module = import_module_from_path(script_file)
        
        # Find contract implementation
        contract_classes = [
            cls for name, cls in inspect.getmembers(script_module)
            if inspect.isclass(cls) and issubclass(cls, ScriptContract)
        ]
        
        assert len(contract_classes) >= 1, f"No contract found in {script_file}"

def test_schema_compatibility():
    """Test that all schemas are compatible with LLM function calling."""
    for script_class in discover_all_script_contracts():
        input_schema = script_class.input_schema()
        
        # Validate JSON schema generation
        json_schema = input_schema.model_json_schema()
        assert "properties" in json_schema
        
        # Test function calling compatibility
        function_def = generate_openai_function_definition(script_class)
        assert function_def["name"] and function_def["parameters"]
```

#### 2. Integration Tests
```python
def test_script_composition():
    """Test that scripts can be composed together."""
    # Test user input → data transform → file output
    user_input = GetUserInputScript()
    json_extract = JsonExtractScript()  
    file_write = WriteFileScript()
    
    # Create workflow
    workflow = WorkflowBuilder()
        .add_script(user_input, {"prompt": "Enter JSON data:"})
        .add_script(json_extract, {
            "json_data": "${get_user_input.user_input}",
            "fields": ["name", "age"]
        })
        .add_script(file_write, {
            "path": "output.json",
            "content": "${json_extract.extracted_data}"
        })
        .build()
    
    # Validate workflow  
    assert workflow.validate_composition()
```

## Benefits

### Ecosystem Quality
- **Contract Enforcement**: All scripts must implement testable interfaces
- **Automatic Validation**: CI catches contract violations before deployment
- **Compatibility Guarantee**: Scripts are tested for ecosystem integration

### Community Growth
- **Clear Standards**: Contributors know exactly what's required
- **Automated Onboarding**: Submission pipeline guides contributors  
- **Quality Control**: Only validated scripts enter the ecosystem

### LLM Agent Reliability
- **Predictable Interfaces**: Agents can trust script behavior
- **Dynamic Validation**: Runtime compatibility checking
- **Error Handling**: Consistent error patterns across all scripts

### Developer Experience
- **Template-Driven**: Easy script creation with provided templates
- **Comprehensive Testing**: Built-in test case framework
- **Documentation**: Self-documenting contracts and schemas

## Implementation Roadmap

### Phase 1: Foundation (Current Sprint)
1. Create `ScriptContract` base classes and testing framework
2. Migrate core primitives to contract system
3. Implement basic CI validation pipeline

### Phase 2: Extension Patterns
1. Implement arbitrary execution, API integration patterns
2. Create community submission pipeline
3. Add advanced composition testing

### Phase 3: Ecosystem Growth  
1. Launch community contribution program
2. Build visual workflow designer
3. Create script marketplace and discovery

This contract system ensures that llm-orchestra becomes a reliable, extensible platform where any script - whether core primitive, base example, or community contribution - can seamlessly interoperate with the broader ecosystem while maintaining type safety and testing compliance.