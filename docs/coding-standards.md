# LLM Orchestra Coding Standards

## Development Environment

### Package Management
- **Use `uv`**: Fast Python package manager for all dependency management
- **Virtual Environment**: `uv sync` automatically creates and manages `.venv`
- **Lock File**: `uv.lock` ensures reproducible builds across environments
- **Development Dependencies**: `uv sync --dev` for full development setup

### Python Version
- **Minimum**: Python 3.11+
- **Target**: Python 3.12 for development and CI
- **Type Checking**: Full mypy strict mode compliance required
- **Compatibility**: Support Python 3.11+ in production

## Code Quality Standards

### Write-Once Quality Code
**Critical**: All code must be written to comply with ruff (line-length=88, strict linting) and mypy (strict=true) from the first draft. Never write code that requires subsequent linting fixes.

#### Formatting Requirements
- **Line Length**: Maximum 88 characters - no exceptions
- **Break long lines using parentheses**, not backslashes
- **Multi-line function calls**: Use parentheses with parameters on new lines
- **Multi-line conditionals**: Use parentheses with conditions on new lines

#### Type Annotations (Required)
- **ALL function signatures**: `def func(param: Type) -> ReturnType:`
- **ALL class attributes** with non-obvious types
- **Complex variables**: Explicit type annotations
- **Modern syntax**: `str | None` not `Optional[str]`
- **Generic types**: `list[str]` not `list`, `dict[str, Any]` not `dict`

#### Exception Handling
- **Always use `from` clause** in exception handlers
- **Chain exceptions**: `raise RuntimeError("Failed") from e`
- **Suppress when appropriate**: `raise UserError("Invalid") from None`
- **Never bare `except:`** clauses

## Commit Discipline

### Change Type Separation
**Never mix structural and behavioral changes in the same commit**

#### Structural Changes (No behavior change)
- Renaming variables, functions, classes
- Extracting methods or functions
- Moving code between files
- Reorganizing imports
- Reformatting code

#### Behavioral Changes (Functionality change)
- New features or capabilities
- Bug fixes that change program behavior
- Modified algorithms or logic
- API changes that affect functionality

### Commit Rules
- Only commit when ALL tests pass
- ALL linter/type checker warnings resolved
- Single logical unit of work per commit
- Run tests before AND after structural changes to verify no behavior change

## Testing Framework

### Real API Testing Philosophy
**Use real LLM APIs in tests** - mock sparingly and only for unit isolation

#### Testing Approach
```python
# Preferred: Real API testing with test credentials
@pytest.mark.integration
async def test_anthropic_model_real_api():
    """Test Anthropic model with real API."""
    model = AnthropicModel(api_key=os.getenv("TEST_ANTHROPIC_KEY"))
    response = await model.generate_response("What is 2+2?")
    assert "4" in response

# Limited mocking: Only for unit isolation
@pytest.mark.unit
async def test_agent_executor_timeout_logic():
    """Test timeout logic without external dependencies."""
    mock_model = Mock(spec=BaseModel)
    mock_model.generate_response = AsyncMock(side_effect=asyncio.TimeoutError())
    
    executor = AgentExecutor(model=mock_model)
    
    with pytest.raises(AgentExecutionTimeoutError):
        await executor.execute_agent(config, "test", timeout=1)
```

### Test Organization
- **Unit Tests**: Component isolation, minimal mocking
- **Integration Tests**: Real APIs, full component interaction
- **End-to-End Tests**: Complete workflows with real providers
- **Performance Tests**: Benchmarks with real API latency

### Coverage Requirements
- **Minimum**: 95% line coverage required
- **CI Enforcement**: 95% threshold enforced in CI
- **Branch Coverage**: Critical error handling paths
- **Real Usage**: Tests reflect actual usage patterns

## Complexity Management

### Simplicity Principles
- **Minimize cognitive load**: Each function should be easily understood
- **Avoid clever code**: Prefer explicit over implicit
- **Single responsibility**: One clear purpose per class/function
- **Dependency injection**: Constructor-based dependencies only

### Complexity Limits
- **Cyclomatic complexity**: Maximum 10 per function
- **Function length**: Maximum 50 lines
- **Class length**: Maximum 500 lines
- **Nesting depth**: Maximum 4 levels

#### Complexity Example
```python
# Good: Low complexity, clear intent
async def execute_agent_with_timeout(
    self, 
    agent_config: AgentConfig, 
    input_data: str, 
    timeout_seconds: int
) -> AgentResult:
    """Execute single agent with timeout."""
    try:
        model = await self._create_model(agent_config.model_profile)
        response = await asyncio.wait_for(
            model.generate_response(input_data),
            timeout=timeout_seconds
        )
        return AgentResult(success=True, response=response)
        
    except asyncio.TimeoutError as e:
        raise AgentExecutionTimeoutError(f"Timeout after {timeout_seconds}s") from e
    except Exception as e:
        raise AgentExecutionError(f"Execution failed: {e}") from e

# Bad: High complexity, multiple responsibilities
async def complex_execute_agent(self, config, input_data, timeout, retries=3, 
                               fallback_model=None, custom_handler=None):
    # Too many parameters, responsibilities, and nested logic
    pass
```

## Async Programming Standards

### Async Patterns
- **Async all the way**: No mixing sync/async in call chains
- **Use `asyncio.gather()`** for concurrent operations
- **Proper exception handling**: `return_exceptions=True` when appropriate
- **Context managers**: `async with` for resource management

```python
# Good: Clean async concurrency
async def execute_agents_parallel(
    self, agents: list[AgentConfig]
) -> dict[str, AgentResult]:
    """Execute agents concurrently."""
    tasks = [self._execute_single_agent(agent) for agent in agents]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    return {
        agent.name: result 
        for agent, result in zip(agents, results) 
        if not isinstance(result, Exception)
    }
```

## Configuration Standards

### YAML Structure
- **Clear hierarchy**: Logical nesting and grouping
- **Schema validation**: All configs validated against schemas
- **Documentation**: Inline comments for complex options
- **Environment substitution**: `${VAR}` support where needed

```yaml
# Clear, validated structure
name: code-review
description: Multi-perspective code analysis

agents:
  - name: security-reviewer
    model_profile: efficient
    system_prompt: "Focus on security vulnerabilities and attack vectors."
    
  - name: senior-reviewer
    model_profile: default-claude
    depends_on: [security-reviewer, performance-reviewer]
    system_prompt: "Synthesize specialist findings into recommendations."
```

## Error Handling

### Exception Hierarchy
```python
class LLMOrcError(Exception):
    """Base exception for LLM Orchestra."""
    pass

class ConfigurationError(LLMOrcError):
    """Configuration validation errors."""
    pass

class AgentExecutionError(LLMOrcError):
    """Agent execution failures."""
    pass
```

### Error Patterns
- **Specific exceptions**: Use appropriate exception types
- **Chain exceptions**: Preserve error context with `from`
- **User-friendly messages**: Clear error descriptions
- **Context preservation**: Include relevant details

## Performance Standards

### Targets
- **Parallel execution**: 3-15x improvement over sequential
- **Memory efficiency**: <0.2MB overhead per agent
- **API utilization**: >95% I/O concurrency
- **Test coverage**: >95% maintained

### Benchmarking
- **Real API latency**: Test with actual provider response times
- **Concurrent execution**: Measure parallel vs sequential performance
- **Memory profiling**: Track memory usage patterns
- **Regression prevention**: Automated performance testing

## Development Workflow

### TDD Cycle
1. **Red**: Write failing test
2. **Green**: Minimum code to pass
3. **Refactor**: Improve structure while keeping tests green

### Commit Workflow
1. **Write tests first** (Red phase)
2. **Implement minimum code** (Green phase)
3. **Refactor if needed** (separate structural commits)
4. **Run full test suite** before commit
5. **Verify linting/typing** passes