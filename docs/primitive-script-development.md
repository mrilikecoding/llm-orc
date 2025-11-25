# Primitive Script Development Guide

## Quick Start: Creating a Simple Primitive Script

A primitive script is a standalone executable that implements a specific, reusable functionality following the `ScriptContract` interface.

### Minimal Example: User Input Primitive

```python
#!/usr/bin/env python3
"""Simple user input collection primitive."""
import json
import sys
from typing import Optional
from pydantic import BaseModel, Field

class UserInputInput(BaseModel):
    """Input schema for user input script."""
    prompt: str = Field(..., description="Prompt text shown to user")
    multiline: bool = Field(default=False, description="Allow multiline input")

class UserInputOutput(BaseModel):
    """Output schema for user input script."""
    success: bool
    user_input: Optional[str] = None
    error: Optional[str] = None

def main():
    # Read input from stdin
    input_data = json.load(sys.stdin)
    parsed_input = UserInputInput(**input_data)

    try:
        # Collect user input
        response = input(parsed_input.prompt + " ")
        output = UserInputOutput(
            success=True,
            user_input=response
        )
    except Exception as e:
        output = UserInputOutput(
            success=False,
            error=str(e)
        )

    # Write output to stdout
    print(json.dumps(output.model_dump()))

if __name__ == "__main__":
    main()
```

## Script Template: Comprehensive Structure

### Structural Components

1. **Input Schema**: Define expected input structure
2. **Output Schema**: Define output structure
3. **Main Script Logic**: Implement core functionality
4. **Error Handling**: Provide robust error management
5. **Standalone Execution**: Support CLI/stdin/stdout

### Full Template Example

```python
#!/usr/bin/env python3
"""Comprehensive primitive script template."""
import asyncio
import json
import logging
import sys
from typing import Any, Optional
from pydantic import BaseModel, Field
from your_project.script_contract import ScriptContract, ScriptMetadata, ScriptCapability, TestCase

class TemplateInput(BaseModel):
    """Comprehensive input schema."""
    data: str = Field(..., description="Primary input data")
    optional_param: Optional[int] = Field(default=None, description="Optional configuration")
    context: dict[str, Any] = Field(default_factory=dict)

class TemplateOutput(BaseModel):
    """Comprehensive output schema."""
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

class TemplatePrimitive(ScriptContract):
    """Comprehensive script primitive implementation."""

    metadata = ScriptMetadata(
        name="template_primitive",
        version="1.0.0",
        description="A comprehensive primitive script template",
        author="your_username",
        category="data_transformation",
        capabilities=[
            ScriptCapability.DATA_TRANSFORMATION,
            ScriptCapability.COMPUTATION
        ],
        dependencies=[],
        tags=["template", "example"]
    )

    @classmethod
    def input_schema(cls) -> type[BaseModel]:
        return TemplateInput

    @classmethod
    def output_schema(cls) -> type[BaseModel]:
        return TemplateOutput

    async def execute(self, input_data: TemplateInput) -> TemplateOutput:
        """Core script execution method."""
        try:
            # Perform script logic
            result = await self._process_data(input_data)

            return TemplateOutput(
                success=True,
                result=result,
                metadata={"processing_time": 0.1}
            )
        except Exception as e:
            logging.error(f"Script execution failed: {e}", exc_info=True)
            return TemplateOutput(
                success=False,
                error=str(e)
            )

    async def _process_data(self, input_data: TemplateInput) -> Any:
        """Private method for data processing."""
        # Implement your core logic here
        return f"Processed: {input_data.data}"

    def get_test_cases(self) -> list[TestCase]:
        """Define test cases for validation."""
        return [
            TestCase(
                name="successful_execution",
                description="Test successful primitive execution",
                input_data={"data": "test_input"},
                expected_output={"success": True},
                should_succeed=True
            ),
            TestCase(
                name="error_handling",
                description="Test error handling scenario",
                input_data={"data": ""},
                expected_output={"success": False},
                should_succeed=False
            )
        ]

# Standalone execution support
if __name__ == "__main__":
    async def main():
        # Read JSON input from stdin
        input_json = json.load(sys.stdin) if not sys.stdin.isatty() else {}

        # Create and execute primitive
        primitive = TemplatePrimitive()
        input_data = TemplateInput(**input_json)
        result = await primitive.execute(input_data)

        # Output result
        print(result.model_dump_json(indent=2))

    asyncio.run(main())
```

## JSON I/O Specification

### Input Handling
- Always accept JSON via stdin
- Use Pydantic models for input validation
- Provide clear type hints and descriptions
- Support optional/configurable parameters

### Output Requirements
- Always produce JSON via stdout
- Include `success` boolean
- Provide `result` and optional `error` fields
- Optionally include `metadata` for additional context

## Category-Specific Schemas

### Common Categories
- **User Interaction**
- **File Operations**
- **Data Transformation**
- **Research & Analytics**
- **Network & API Integration**

### Example: User Interaction Primitive Schema

```python
class UserInteractionInput(BaseModel):
    """Base input for user interaction primitives."""
    prompt: str
    multiline: bool = False
    validation_pattern: Optional[str] = None
    max_length: Optional[int] = None

class UserInteractionOutput(BaseModel):
    """Base output for user interaction primitives."""
    success: bool
    user_response: Optional[str] = None
    validation_passed: bool = True
    error: Optional[str] = None
```

## Testing Primitives

### Local Testing Strategies
1. Unit tests for schema validation
2. Test case generation
3. Error scenario validation
4. Execution workflow testing

### Example Test Suite

```python
def test_primitive_input_validation():
    # Test input schema validation
    valid_input = {"data": "test"}
    TemplateInput(**valid_input)  # Should pass

    with pytest.raises(ValidationError):
        # Invalid input should raise exception
        TemplateInput(data=None)

def test_primitive_execution():
    primitive = TemplatePrimitive()
    input_data = TemplateInput(data="test")
    result = await primitive.execute(input_data)

    assert result.success
    assert result.result is not None
```

## CLI Testing

### Using `llm-orc scripts test`

```bash
# Test a single primitive
llm-orc scripts test primitives/user-interaction/get_user_input.py

# Test all primitives in a category
llm-orc scripts test --category user-interaction

# Run comprehensive primitive test suite
llm-orc scripts test --all
```

## Integration with Ensembles

### YAML Configuration Example

```yaml
name: user_data_collection
agents:
  - name: get-user-details
    script: primitives/user-interaction/get_user_input.py
    parameters:
      prompt: "What is your name?"

  - name: validate-input
    script: primitives/validation/input_validator.py
    depends_on: [get-user-details]
    parameters:
      input_data: "${get-user-details.user_input}"
      rules: ["non_empty", "min_length:2"]
```

## Best Practices

### Error Handling
- Always use exception chaining
- Provide meaningful error messages
- Log errors with context
- Handle timeouts and external dependencies

### Performance & Security
- Set reasonable timeout limits
- Validate all inputs
- Use security sandboxing
- Minimize external dependencies

### Example of Robust Error Handling

```python
try:
    result = await potentially_risky_operation()
except TimeoutError as timeout_err:
    logging.warning("Operation timed out")
    raise RuntimeError("Slow operation detected") from timeout_err
except Exception as e:
    logging.error(f"Unexpected error: {e}")
    raise
```

## Publishing to Library

### Contribution Workflow
1. Implement primitive following contract
2. Add comprehensive test cases
3. Document purpose and usage
4. Submit PR to `llm-orchestra-library`

### Review Criteria
- Passes all contract validations
- Clear, focused functionality
- Robust error handling
- Comprehensive test coverage
- Well-documented purpose and usage

## Conclusion

Primitive scripts in llm-orc are powerful, composable building blocks. By following these guidelines, you'll create reliable, reusable scripts that seamlessly integrate into complex workflows.