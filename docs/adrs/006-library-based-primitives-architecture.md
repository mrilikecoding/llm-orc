# ADR-006: Library-Based Primitives Architecture with Multi-Language Bridge Support

## Status
Implemented

## Implementation Status
- [x] BDD scenarios created in tests/bdd/features/adr-006-library-based-primitives-architecture.feature
- [x] Library-aware ScriptResolver with search path prioritization
- [x] Graceful degradation with helpful error messages (ScriptNotFoundError)
- [x] Test independence via TestPrimitiveFactory fixtures
- [x] All 22 BDD scenarios passing
- [x] Integration tests validate orchestration with fixtures
- [x] Type safety and coding standards compliance
- [x] Refactoring complete (constants extracted, clean organization)

**Note on Library Content**: This ADR validates llm-orc's orchestration architecture. Library primitive implementations live in the `llm-orchestra-library` repository and are validated through interface contracts (ADR-003). Tests use fixtures to maintain independence from library submodule initialization.

## BDD Mapping Hints
```yaml
# Primary behavioral capabilities this ADR enables
behavioral_capabilities:
  - capability: "Script resolution prioritizes local over library primitives"
    given: "A script reference that could match in both local and library locations"
    when: "ScriptResolver attempts to resolve the script path"
    then: "Local project scripts take precedence over library primitives"

  - capability: "Bridge primitives enable cross-language execution"
    given: "A Python bridge primitive and external language script/command"
    when: "Bridge primitive executes subprocess with structured JSON I/O"
    then: "External language receives input data and returns structured output"

  - capability: "Tests run without library submodule dependency"
    given: "Test suite execution without library submodule initialized"
    when: "Tests require primitive functionality"
    then: "Test fixtures provide minimal primitive implementations"

  - capability: "Graceful degradation when primitives missing"
    given: "Script reference to missing primitive in library"
    when: "ScriptResolver attempts to resolve primitive path"
    then: "Helpful error message guides user to initialize library or provide alternative"

# Test boundaries for automated test generation
test_boundaries:
  unit:
    - "ScriptResolver.resolve_script_path()"
    - "TestPrimitiveFactory.create_user_input_script()"
    - "SubprocessExecutor.execute_with_json_io()"
    - "NodeExecutor.execute_javascript_bridge()"
    - "BridgePrimitive.validate_json_io()"

  integration:
    - "script_resolution_without_library_submodule"
    - "bridge_primitive_javascript_execution_flow"
    - "end_to_end_ensemble_with_mixed_languages"
    - "error_handling_for_missing_primitives"
    - "library_optional_installation_workflow"

# Validation rules for implementation
validation_rules:
  - "Type safety: All bridge primitives must have type annotations"
  - "Exception handling: Use proper exception chaining for subprocess failures"
  - "Performance: Bridge primitives must include timeout handling"
  - "JSON compatibility: All language bridges use consistent JSON I/O patterns"
  - "Test independence: Tests pass without library submodule initialization"
  - "Error messages: ScriptResolver provides actionable guidance for missing primitives"

# Dependencies on other ADRs
related_adrs:
  - "ADR-002: Builds on composable primitive system concepts"
  - "ADR-003: Script contracts apply to bridge primitives and validation"
  - "ADR-001: Pydantic schemas used for bridge primitive I/O validation"

# Implementation components affected
implementation_scope:
  - "tests/fixtures/test_primitives.py"
  - "src/llm_orc/core/execution/script_resolver.py"
  - "tests/unit/core/execution/test_ensemble_script_integration.py"
  - "tests/unit/core/execution/test_agent_request_integration.py"
  - "llm-orchestra-library/primitives/python/"
  - "docs/README.md"
```

## Context

### The Primitives Location Dilemma

The llm-orc orchestration system uses **primitives** - fundamental, atomic building blocks like user input collection, file I/O, HTTP requests, and data transformations. These primitives can be composed with LLM agents to create sophisticated workflows that accomplish tasks beyond what LLMs can do alone.

A key architectural question emerged: **Where should primitives live?**

#### Option A: Built-in Primitives (In Core Application)
```
src/llm_orc/
  primitives/          # ← Always available after pip install
    user_input.py
    file_read.py
    http_request.py
```

**Pros:**
- Immediately available after `pip install llm-orc`
- Tests always pass without external dependencies
- "Batteries included" philosophy
- No bootstrap problems

**Cons:**
- Blurs boundary between orchestration engine and orchestrable content
- Locks primitives to application release cycle
- Harder to extend with community contributions
- Language-specific (Python-only without major complexity)

#### Option B: Library-Based Primitives (External Submodule)
```
llm-orchestra-library/   # ← Optional but recommended
  scripts/
    primitives/          # Core primitives (Python-only engine)
      user_input.py
      file_read.py
    domain/              # Domain-specific scripts
  ensembles/            # Pre-composed workflows
```

**Pros:**
- Clean separation: engine vs. orchestrable components
- Primitives are "just scripts" like any domain-specific script
- Community can extend/contribute primitives
- Supports multiple languages naturally
- Library can evolve independently

**Cons:**
- Requires submodule initialization for full functionality
- Tests need fixtures to avoid library dependency
- More complex first-run experience

### Multi-Language Execution Requirements

Future workflows will need to execute scripts in multiple languages:
- **JavaScript**: Web scraping with Puppeteer, Node.js ecosystem
- **Shell**: System administration, command-line tools
- **Rust**: High-performance data processing
- **Python**: AI/ML, data analysis, general scripting

#### Option A: Native Multi-Language Engine
Make the orchestration engine understand JavaScript, Rust, shell, etc. directly.

**Cons:**
- Massive complexity increase in core engine
- Need interpreters, runtime management, language-specific error handling
- Tight coupling between engine and language ecosystems

#### Option B: Bridge Primitive Pattern
Provide Python primitives that execute other languages via subprocess with JSON I/O.

**Pros:**
- Engine stays Python-only and focused on orchestration
- Universal pattern for any executable (languages, containers, APIs)
- Consistent JSON interface across all languages
- Future-proof for cloud functions, microservices, etc.

### Current Test Failures

Several tests currently fail because they reference `primitives/user_input.py` but the library submodule isn't initialized:

```
AssertionError: assert 'received_dynamic_parameters' in {'error': 'Script not found: primitives/user_input.py'}
```

This revealed the architectural inconsistency: tests assume primitives exist, but they live in an optional submodule.

## Decision

**Implement library-based primitives architecture with Python bridge pattern for multi-language support.**

Specifically:

1. **Primitives remain in `llm-orchestra-library` submodule** - maintaining clean separation of concerns
2. **Multi-language support via Python bridges** - no engine complexity for other languages
3. **Tests use fixtures** - independent of library submodule initialization
4. **Graceful degradation** - helpful error messages when primitives missing

## Detailed Design

### 1. Library Structure

```
llm-orchestra-library/
  scripts/
    primitives/                  # Core primitives (Python-only)
      # Core I/O primitives
      file-ops/
        file_read.py
        file_write.py
      user-interaction/
        user_input.py
      control-flow/
        conditional.py

      # Bridge primitives (future)
      bridges/
        subprocess_executor.py   # Universal command executor
        node_executor.py        # Node.js with enhanced integration
        shell_executor.py       # Shell commands with safety

    domain/                     # Domain-specific implementations
      web_scraping/
      data_analysis/
      integrations/

  ensembles/                    # Pre-composed workflows
    data_pipeline.yaml
    web_research.yaml
```

### 2. Bridge Primitive Pattern

**Universal Subprocess Bridge** (`subprocess_executor.py`):
```python
#!/usr/bin/env python3
"""Execute external commands/scripts with JSON I/O."""

import json
import os
import subprocess
import sys
from pathlib import Path

def main():
    # Get structured input from environment
    input_data = json.loads(os.environ.get('INPUT_DATA', '{}'))

    # Required parameters
    command = input_data.get('command')
    if not command:
        print(json.dumps({"success": False, "error": "command parameter required"}))
        sys.exit(1)

    # Optional parameters
    working_dir = input_data.get('working_dir', '.')
    env_vars = input_data.get('env_vars', {})
    input_stdin = input_data.get('stdin')
    timeout = input_data.get('timeout', 30)

    try:
        # Prepare environment
        env = os.environ.copy()
        env.update(env_vars)

        # Execute with proper timeout and I/O handling
        result = subprocess.run(
            command,
            shell=True,
            cwd=working_dir,
            env=env,
            input=input_stdin,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        # Return structured output
        print(json.dumps({
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode
        }))

    except subprocess.TimeoutExpired:
        print(json.dumps({
            "success": False,
            "error": f"Command timed out after {timeout} seconds"
        }))
    except Exception as e:
        print(json.dumps({
            "success": False,
            "error": str(e)
        }))

if __name__ == "__main__":
    main()
```

**Node.js Bridge** (`node_executor.py`):
```python
#!/usr/bin/env python3
"""Execute Node.js scripts with enhanced JSON I/O bridge."""

import json
import os
import subprocess
import tempfile
from pathlib import Path

def main():
    input_data = json.loads(os.environ.get('INPUT_DATA', '{}'))

    # Support both inline script and script path
    script_content = input_data.get('script')
    script_path = input_data.get('script_path')

    if script_content:
        # Execute inline JavaScript
        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
            f.write(script_content)
            script_path = f.name

    if not script_path or not Path(script_path).exists():
        print(json.dumps({"success": False, "error": "No valid script provided"}))
        return

    try:
        # Pass structured data to Node script via environment
        env = os.environ.copy()
        env['BRIDGE_INPUT'] = json.dumps(input_data.get('data', {}))

        result = subprocess.run(
            ['node', script_path],
            env=env,
            capture_output=True,
            text=True,
            timeout=input_data.get('timeout', 30)
        )

        if result.returncode == 0:
            # Parse Node output as JSON if possible
            try:
                node_output = json.loads(result.stdout)
                print(json.dumps({"success": True, "data": node_output}))
            except json.JSONDecodeError:
                print(json.dumps({"success": True, "data": result.stdout}))
        else:
            print(json.dumps({
                "success": False,
                "error": result.stderr,
                "return_code": result.returncode
            }))

    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))
    finally:
        # Clean up temp file if created
        if script_content and Path(script_path).exists():
            Path(script_path).unlink()

if __name__ == "__main__":
    main()
```

### 3. Usage Examples

**Execute JavaScript via Bridge**:
```yaml
agents:
  - name: js_data_processor
    script: primitives/node_executor.py
    parameters:
      script: |
        const input = JSON.parse(process.env.BRIDGE_INPUT);
        const result = input.numbers.map(n => n * 2);
        console.log(JSON.stringify({processed: result}));
      data:
        numbers: [1, 2, 3, 4, 5]
      timeout: 10
```

**Execute Shell Commands**:
```yaml
agents:
  - name: system_info
    script: primitives/subprocess_executor.py
    parameters:
      command: "uname -a && df -h"
      working_dir: "/tmp"
      timeout: 15
```

**Chain Multiple Languages**:
```yaml
agents:
  - name: rust_analyzer
    script: primitives/subprocess_executor.py
    parameters:
      command: "cargo run --bin analyzer"
      working_dir: "./rust_tools"
      env_vars:
        INPUT_FILE: "data.json"
    depends_on: [data_fetcher]

  - name: js_visualizer
    script: primitives/node_executor.py
    parameters:
      script_path: "./viz/chart_generator.js"
      data: "${rust_analyzer.analysis_results}"
    depends_on: [rust_analyzer]
```

### 4. Test Strategy

**Test Primitive Fixtures** (`tests/fixtures/test_primitives.py`):
```python
"""Minimal test primitives that don't require library submodule."""

import tempfile
from pathlib import Path
from typing import Dict, Any

class TestPrimitiveFactory:
    """Creates test primitive scripts in isolated temp directories."""

    @staticmethod
    def create_user_input_script(tmp_path: Path, language: str = "python") -> Path:
        """Create a test user_input primitive."""
        script = tmp_path / "user_input.py"
        script.write_text("""#!/usr/bin/env python3
import json, os
input_data = json.loads(os.environ.get('INPUT_DATA', '{}'))
prompt = input_data.get('prompt', 'Enter value: ')
# In tests, return mock user input instead of actual input()
result = input_data.get('mock_user_input', 'test_user_input')
print(json.dumps({
    "success": True,
    "data": result,
    "received_dynamic_parameters": input_data
}))
""")
        script.chmod(0o755)
        return script

    @staticmethod
    def create_subprocess_executor(tmp_path: Path) -> Path:
        """Create a test subprocess executor."""
        script = tmp_path / "subprocess_executor.py"
        script.write_text("""#!/usr/bin/env python3
import json, os
input_data = json.loads(os.environ.get('INPUT_DATA', '{}'))
# Mock execution - return success with mocked output
print(json.dumps({
    "success": True,
    "stdout": f"Executed: {input_data.get('command', 'unknown')}",
    "stderr": "",
    "return_code": 0
}))
""")
        script.chmod(0o755)
        return script

    @classmethod
    def setup_test_primitives_dir(cls, tmp_path: Path) -> Path:
        """Setup complete test primitives directory."""
        primitives_dir = tmp_path / "primitives"
        primitives_dir.mkdir()

        cls.create_user_input_script(primitives_dir)
        cls.create_subprocess_executor(primitives_dir)

        return primitives_dir
```

**Test Fixture Usage**:
```python
@pytest.fixture
def test_primitives(tmp_path):
    """Provide test primitives without requiring library."""
    primitives_dir = TestPrimitiveFactory.setup_test_primitives_dir(tmp_path)

    # Patch ScriptResolver to check test dir first
    original_search_paths = ScriptResolver._get_search_paths
    def mock_search_paths():
        return [str(primitives_dir)] + original_search_paths()

    with mock.patch.object(ScriptResolver, '_get_search_paths', mock_search_paths):
        yield primitives_dir

def test_story_generator_to_user_input_agent_flow(test_primitives):
    """Test now uses fixture primitives, not library."""
    config = EnsembleConfig(
        agents=[
            {
                "name": "user_input_agent",
                "script": "user_input.py",
                "parameters": {"mock_user_input": "test_character_name"}
            }
        ]
    )
    # Test runs without library submodule!
```

### 5. Script Resolution Enhancement

**Enhanced ScriptResolver with Helpful Errors**:
```python
class ScriptResolver:
    """Resolves script paths with language-aware resolution."""

    def resolve_script_path(self, script: str) -> str:
        """Resolve script with helpful error messages."""
        search_paths = self._get_search_paths()

        for search_path in search_paths:
            candidate = Path(search_path) / script
            if candidate.exists():
                return str(candidate)

        # Generate helpful error message
        if script.startswith('primitives/'):
            raise ScriptNotFoundError(
                f"Primitive script not found: {script}\n\n"
                f"Hints:\n"
                f"  • Initialize the library submodule:\n"
                f"    git submodule update --init --recursive\n"
                f"  • Or provide your own implementation in:\n"
                f"    {self._get_project_root()}/{script}\n"
                f"  • Or use a test fixture in tests"
            )
        else:
            raise ScriptNotFoundError(f"Script not found: {script}")

    def _get_search_paths(self) -> list[str]:
        """Get search paths in priority order."""
        paths = []

        # 1. Current project directory (highest priority)
        project_root = self._get_project_root()
        paths.append(str(project_root))

        # 2. Library submodule (if exists)
        library_path = project_root / "llm-orchestra-library"
        if library_path.exists():
            paths.append(str(library_path))

        # 3. System-wide library installation (future)
        # paths.extend(self._get_system_library_paths())

        return paths
```

### 6. Documentation Updates

**README Enhancement**:
```markdown
## Installation

### Core Only (Orchestration Engine)
```bash
pip install llm-orc
# You can now orchestrate your own scripts and LLM agents
```

### With Primitives Library (Recommended)
```bash
git clone --recurse-submodules https://github.com/user/llm-orc.git
cd llm-orc
pip install -e .
# Now primitives like user_input.py, file_read.py are available
```

### Library Components

The optional `llm-orchestra-library` submodule provides:

- **Primitives**: Atomic building blocks (file I/O, HTTP, user input, language bridges)
- **Scripts**: Domain-specific implementations (web scraping, API integrations)
- **Ensembles**: Pre-composed workflows for common tasks

#### Multi-Language Support

Execute scripts in any language via bridge primitives:

```yaml
# JavaScript execution
- name: data_processor
  script: primitives/node_executor.py
  parameters:
    script: "const data = JSON.parse(process.env.BRIDGE_INPUT);"

# Shell commands
- name: system_check
  script: primitives/subprocess_executor.py
  parameters:
    command: "docker ps && kubectl get pods"

# Rust binary
- name: fast_analyzer
  script: primitives/subprocess_executor.py
  parameters:
    command: "cargo run --release --bin analyzer"
    working_dir: "./rust_tools"
```
```

### 7. CI/CD Strategy

**GitHub Actions Multi-Mode Testing**:
```yaml
name: Test Matrix

jobs:
  test-core-only:
    name: Test core without library
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        # No submodule initialization!
      - name: Install dependencies
        run: pip install -e .
      - name: Run core tests
        run: pytest tests/unit/ tests/integration/
        # Should pass with test fixtures

  test-with-library:
    name: Test with primitives library
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with:
          submodules: recursive
      - name: Install dependencies
        run: pip install -e .
      - name: Run full test suite
        run: pytest tests/
        # Can use real library primitives

  test-bridge-primitives:
    name: Test multi-language bridges
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with:
          submodules: recursive
      - name: Install multi-language dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y nodejs rust-all
      - name: Test bridge execution
        run: pytest tests/integration/test_bridge_primitives.py
```

## Benefits

### Architectural Clarity

**Clean Separation of Concerns**:
- **Core Application**: Orchestration, dependency resolution, execution coordination
- **Library**: Orchestrable components (primitives, scripts, ensembles)
- **Clear Boundary**: "Engine that orchestrates" vs. "things that get orchestrated"

**Consistent Philosophy**: Everything orchestrated by llm-orc is "just a script" - whether it's a primitive, domain-specific script, or ensemble definition.

### Multi-Language Support Without Complexity

**Engine Stays Simple**: Only needs to understand Python and JSON I/O
**Universal Pattern**: Bridge primitives work for any executable (languages, containers, microservices, cloud functions)
**Future-Proof**: Can easily add bridges for new languages or execution environments

### Development & Testing Benefits

**Test Independence**: Test suite runs without external dependencies
**CI Simplicity**: Can test core functionality without submodule complexity
**Development Workflow**: Core changes don't require library coordination

### Extensibility & Community

**Community Contributions**: Library can accept primitives from community
**Independent Evolution**: Library and core can have different release cycles
**Plugin Architecture**: Third parties can provide primitive collections
**Language Diversity**: Community can contribute bridges for any language

## Trade-offs

### First-Run Experience Complexity

**Additional Setup Step**: Users need `git submodule update --init` for full functionality
**Documentation Burden**: Must explain library vs. core distinction
**Support Complexity**: More potential points of failure

**Mitigation**: Clear documentation, helpful error messages, optional but recommended positioning

### Test Fixture Maintenance

**Duplicate Logic**: Test fixtures replicate primitive functionality
**Maintenance Overhead**: Need to keep fixtures in sync with real primitives
**Potential Drift**: Test behavior may diverge from real primitive behavior

**Mitigation**: Automated testing of both fixture and library primitives, shared test contracts

### Bridge Primitive Complexity

**Subprocess Overhead**: Bridge pattern adds process creation overhead
**Error Handling Complexity**: Need to handle subprocess failures gracefully
**Security Considerations**: Subprocess execution requires careful input validation

**Mitigation**: Performance benchmarks for acceptable overhead, robust error handling patterns, security review of bridge implementations

## Success Metrics

### Technical Metrics

- **Test Independence**: 100% of tests pass without library submodule
- **Bridge Performance**: <100ms overhead for bridge primitive execution
- **Error Handling**: Comprehensive error messages for all failure modes
- **Type Safety**: All bridge primitives have complete type annotations

### User Experience Metrics

- **Installation Success**: >95% of users successfully run basic ensembles after installation
- **Library Adoption**: >80% of users initialize library submodule
- **Error Resolution**: Clear error messages enable self-service problem resolution
- **Documentation Clarity**: User feedback indicates clear understanding of architecture

### Ecosystem Metrics

- **Community Contributions**: Primitives contributed by external developers
- **Language Coverage**: Bridge primitives available for top 5 languages used in workflows
- **Usage Patterns**: Evidence of multi-language ensembles in real workflows

## Implementation Plan

### Phase 1: Fix Immediate Test Failures
1. Create `tests/fixtures/test_primitives.py` with minimal implementations
2. Update failing tests to use fixtures instead of library dependencies
3. Enhance ScriptResolver with helpful error messages

### Phase 2: Document Architecture
1. Complete this ADR and get team alignment
2. Update README and documentation to reflect architecture
3. Create bridge primitive examples and usage patterns

### Phase 3: Enhance Multi-Language Support
1. Implement robust bridge primitives (`subprocess_executor.py`, `node_executor.py`)
2. Create comprehensive examples of multi-language workflows
3. Add performance benchmarks and security review

### Phase 4: Community & Ecosystem
1. Establish contribution guidelines for library primitives
2. Create plugin architecture for third-party primitive collections
3. Build marketplace/registry for discovering primitives

## Decision Rationale

This architecture establishes llm-orc as a **pure orchestration engine** that coordinates execution of external components, rather than a monolithic system that embeds specific functionality.

The bridge primitive pattern enables unlimited extensibility without engine complexity, while maintaining the consistent JSON I/O interface that makes components composable.

The library-based approach respects the philosophical principle that **primitives are content, not infrastructure** - they are building blocks to be orchestrated, not features of the orchestrator itself.

This decision positions llm-orc for long-term growth as a platform that can coordinate any executable component, in any language, using any execution environment, while maintaining architectural simplicity and clear separation of concerns.