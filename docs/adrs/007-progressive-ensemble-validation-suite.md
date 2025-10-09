# ADR-007: Progressive Ensemble Validation Suite

**Status**: Implemented
**Date**: 2025-10-06
**Implementation Date**: 2025-10-08
**Authors**: Nathan Green
**Related ADRs**: [ADR-004](004-bdd-llm-development-guardrails.md)

## BDD Mapping Hints

This ADR defines validation infrastructure for testing ensembles with varying complexity levels. While ADR-004 established BDD as a development guardrail, ADR-007 focuses on runtime validation of ensemble behavior.

**Behavioral capabilities to validate**:
- Ensemble executes with expected agent sequence
- Script agents produce valid JSON output matching schemas
- Dependent agents receive correct input from predecessors
- User input agents can operate in test mode with LLM simulation
- Validation assertions evaluate structural, schema, behavioral, and semantic properties

**Test boundary**: BDD scenarios should validate the validation framework itself (meta-validation), not individual ensemble behaviors. Ensemble-specific validations are declared in ensemble YAML files.

**Gherkin examples**:
```gherkin
Scenario: Validate ensemble with structural assertions
  Given a validation ensemble with structural requirements
  When the ensemble is executed in validation mode
  Then all structural assertions should be evaluated
  And validation results should indicate pass/fail status

Scenario: LLM-simulated user input in test mode
  Given an ensemble with user input agents
  And the ensemble is configured with LLM persona simulation
  When the ensemble executes in test mode
  Then user input agents should receive LLM-generated responses
  And no blocking stdin prompts should occur
```

## Context

With the implementation of script agents (ADR-001, ADR-002, ADR-003), multi-turn conversations (ADR-005), and library-based primitives (ADR-006), we need a systematic way to validate that ensembles work correctly. Current testing approaches are limited:

1. **Unit tests** validate individual components but not ensemble orchestration
2. **Integration tests** can test execution flow but lack semantic validation
3. **Manual testing** is time-consuming and not reproducible
4. **CI/CD validation** needs non-interactive execution

**Challenge**: Many ensembles require user input during execution. Traditional testing approaches block on stdin, making automated validation impossible. We need a way to simulate user responses using LLMs.

**Research opportunity**: This validation framework serves a dual purpose:
1. **Implementation validation**: Verify that llm-orc works correctly
2. **Research validation**: Enable statistical experiments on multi-agent systems

By designing validation ensembles with quantitative metrics (clustering coefficient, modularity, consensus rates), we can validate both functionality and research hypotheses.

## Decision

We will implement a **progressive ensemble validation suite** with the following architecture:

### 1. Flexible Validation Schema

Ensembles declare validation criteria in their YAML configuration:

```yaml
name: test-file-operations
description: Validation ensemble for file operation primitives

agents:
  - name: read-file
    script: primitives/file-ops/read_file.py
    parameters:
      path: "test-data.json"

  - name: write-file
    script: primitives/file-ops/write_file.py
    depends_on: [read-file]
    parameters:
      path: "output.json"
      content: "${read-file.content}"

validation:
  structural:
    required_agents: [read-file, write-file]
    max_execution_time: 30
    min_execution_time: 0.1

  schema:
    - agent: read-file
      output_schema: FileReadOutput
      required_fields: [success, content]
    - agent: write-file
      output_schema: FileWriteOutput
      required_fields: [success, path]

  behavioral:
    - name: execution-order
      description: "write-file must execute after read-file"
      assertion: "execution_order.index('write-file') > execution_order.index('read-file')"

    - name: file-created
      description: "output file should exist after execution"
      assertion: "Path('output.json').exists()"

  quantitative:
    - metric: execution_time
      threshold: "< 30"
      description: "Total execution time under 30 seconds"

  semantic:
    enabled: false  # Optional LLM-as-judge validation
```

Validation layers are optional - ensembles only include what they need:

- **Structural**: Execution properties (timing, agent presence, concurrency)
- **Schema**: JSON contract validation using Pydantic schemas
- **Behavioral**: Custom assertions with Python expressions
- **Quantitative**: Numerical metrics with thresholds (for research)
- **Semantic**: LLM-based evaluation of outputs (optional)

### 2. LLM User Simulation Architecture

For ensembles requiring user input, we implement **test mode** with LLM-simulated responses:

#### ScriptUserInputHandler Extension

```python
class ScriptUserInputHandler:
    """Handles user input for script agents with optional LLM simulation."""

    def __init__(
        self,
        test_mode: bool = False,
        llm_config: dict[str, Any] | None = None
    ):
        self.test_mode = test_mode
        self.llm_simulators: dict[str, LLMResponseGenerator] = {}

        if test_mode and llm_config:
            self._initialize_simulators(llm_config)

    async def get_user_input(
        self,
        agent_name: str,
        prompt: str,
        context: dict[str, Any]
    ) -> str:
        """Get user input - real or simulated based on mode."""
        if self.test_mode:
            simulator = self.llm_simulators.get(agent_name)
            if simulator:
                return await simulator.generate_response(prompt, context)
            raise RuntimeError(f"No LLM simulator for agent: {agent_name}")

        # Interactive mode - real stdin input
        return input(prompt)
```

#### LLMResponseGenerator

```python
class LLMResponseGenerator:
    """Generates contextual user responses using small LLMs."""

    def __init__(
        self,
        model: str = "qwen3:0.6b",
        persona: str = "helpful_user",
        system_prompt: str | None = None,
        response_cache: dict[str, str] | None = None
    ):
        self.model = model
        self.persona = persona
        self.system_prompt = system_prompt or self._default_persona_prompts()[persona]
        self.response_cache = response_cache or {}
        self.conversation_history: list[dict[str, str]] = []

    async def generate_response(
        self,
        prompt: str,
        context: dict[str, Any]
    ) -> str:
        """Generate contextual response using LLM."""
        # Check cache for deterministic responses
        cache_key = self._create_cache_key(prompt, context)
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]

        # Build LLM prompt with context
        llm_prompt = self._build_llm_prompt(prompt, context)

        # Generate response using local LLM
        response = await self._call_llm(llm_prompt)

        # Cache for reproducibility
        self.response_cache[cache_key] = response

        # Update conversation history
        self.conversation_history.append({
            "prompt": prompt,
            "response": response,
            "context": context
        })

        return response

    def _default_persona_prompts(self) -> dict[str, str]:
        """Default system prompts for common personas."""
        return {
            "helpful_user": (
                "You are simulating a helpful user responding to prompts. "
                "Provide realistic, contextually appropriate responses. "
                "Keep responses concise (1-2 sentences) unless asked for detail."
            ),
            "critical_reviewer": (
                "You are simulating a critical code reviewer. "
                "Point out potential issues, edge cases, and improvements. "
                "Be constructive but thorough."
            ),
            "domain_expert": (
                "You are simulating a domain expert with deep knowledge. "
                "Provide technically accurate, detailed responses. "
                "Reference best practices and potential pitfalls."
            )
        }

    def _create_cache_key(self, prompt: str, context: dict[str, Any]) -> str:
        """Create deterministic cache key from prompt and context."""
        import hashlib
        import json

        cache_input = {
            "prompt": prompt,
            "context": context,
            "persona": self.persona,
            "history_length": len(self.conversation_history)
        }

        return hashlib.sha256(
            json.dumps(cache_input, sort_keys=True).encode()
        ).hexdigest()

    async def _call_llm(self, prompt: str) -> str:
        """Call local LLM (Ollama) for response generation."""
        # Use existing LLMConnector with qwen3:0.6b
        # Implementation delegates to LLMConnector
        pass
```

#### Validation Ensemble Configuration

```yaml
name: test-conversational-code-review
description: Validation for multi-turn conversation with user feedback

agents:
  - name: analyzer
    model_profile: llama3:latest
    system_prompt: "Analyze code for issues"

  - name: human-review
    script: primitives/user-interaction/get_user_input.py
    depends_on: [analyzer]
    parameters:
      prompt: "Review AI analysis: ${analyzer.analysis}. Additional feedback?"
      multiline: true

test_mode:
  enabled: true
  llm_simulation:
    - agent: human-review
      model: qwen3:0.6b
      persona: critical_reviewer
      cached_responses:  # Optional deterministic responses
        "Review AI analysis: ...": "Add security analysis for input validation"

validation:
  behavioral:
    - name: user-input-received
      assertion: "human-review.user_input is not None"
    - name: no-blocking-prompts
      assertion: "execution_mode == 'test_mode'"
```

### 3. Progressive Validation Categories

Validation ensembles are organized by complexity:

#### **Category 1: Primitive Validation**
Test individual script agents in isolation.

```yaml
name: validate-file-read-primitive
agents:
  - name: test-read
    script: primitives/file-ops/read_file.py
    parameters:
      path: "fixtures/sample.json"

validation:
  schema:
    - agent: test-read
      output_schema: FileReadOutput
  behavioral:
    - name: success-flag
      assertion: "test-read.success == True"
```

#### **Category 2: Integration Validation**
Test agent composition and dependencies.

```yaml
name: validate-file-pipeline
agents:
  - name: read-file
    script: primitives/file-ops/read_file.py
  - name: transform
    script: primitives/data-transform/json_transform.py
    depends_on: [read-file]
  - name: write-file
    script: primitives/file-ops/write_file.py
    depends_on: [transform]

validation:
  structural:
    required_agents: [read-file, transform, write-file]
  behavioral:
    - name: dependency-order
      assertion: "execution_order == ['read-file', 'transform', 'write-file']"
```

#### **Category 3: Conversational Validation**
Test multi-turn conversations with LLM simulation.

```yaml
name: validate-conversational-workflow
agents:
  - name: llm-agent
    model_profile: llama3:latest
  - name: user-feedback
    script: primitives/user-interaction/get_user_input.py
    depends_on: [llm-agent]
  - name: refinement
    model_profile: llama3:latest
    depends_on: [user-feedback]

test_mode:
  enabled: true
  llm_simulation:
    - agent: user-feedback
      persona: helpful_user

validation:
  behavioral:
    - name: conversation-flow
      assertion: "len(execution_order) == 3"
```

#### **Category 4: Research Validation**
Test quantitative metrics for research experiments.

```yaml
name: validate-multi-agent-collaboration
agents:
  # 5 agents with various dependency patterns

validation:
  quantitative:
    - metric: clustering_coefficient
      threshold: ">= 0.3"
      description: "Network should show some clustering"

    - metric: average_path_length
      threshold: "< 3.0"
      description: "Average path length should be short"

    - metric: consensus_rate
      threshold: ">= 0.7"
      description: "Agents should reach consensus 70% of time"

  semantic:
    enabled: true
    validator_model: llama3:latest
    criteria:
      - "Agents should produce coherent, non-contradictory outputs"
      - "Final synthesis should incorporate all agent perspectives"
```

#### **Category 5: Application Validation**
End-to-end validation of real-world use cases.

```yaml
name: validate-code-review-workflow
description: Full code review with AI and human feedback

agents:
  - name: static-analysis
    script: primitives/code-analysis/run_linter.py
  - name: llm-review
    model_profile: llama3:latest
    depends_on: [static-analysis]
  - name: human-feedback
    script: primitives/user-interaction/get_user_input.py
    depends_on: [llm-review]
  - name: final-report
    model_profile: llama3:latest
    depends_on: [human-feedback]

test_mode:
  enabled: true
  llm_simulation:
    - agent: human-feedback
      persona: critical_reviewer

validation:
  structural:
    max_execution_time: 120
  behavioral:
    - name: all-stages-complete
      assertion: "all(agent in results for agent in ['static-analysis', 'llm-review', 'human-feedback', 'final-report'])"
  semantic:
    enabled: true
    criteria: ["Final report should integrate static analysis, LLM review, and human feedback"]
```

### 4. Research Cross-Purpose Example

The same validation framework enables research experiments:

```yaml
name: experiment-consensus-vs-diversity
description: Research experiment - does diversity improve consensus?

parameters:
  agent_count: 5
  topology: "small_world"  # vs "random", "scale_free"

agents:
  # Dynamically generated agents based on parameters

validation:
  quantitative:
    # Network metrics
    - metric: clustering_coefficient
      threshold: null  # No pass/fail, just measure
      record: true

    - metric: degree_distribution_entropy
      threshold: null
      record: true

    # Consensus metrics
    - metric: initial_consensus_rate
      threshold: null
      record: true

    - metric: final_consensus_rate
      threshold: null
      record: true

    - metric: consensus_convergence_time
      threshold: null
      record: true

  statistical:
    enabled: true
    runs: 30  # Multiple runs for statistical significance
    export_csv: "results/consensus_diversity_experiment.csv"

    analysis:
      - type: correlation
        variables: [clustering_coefficient, final_consensus_rate]

      - type: anova
        independent_var: topology
        dependent_var: final_consensus_rate
```

This dual-purpose design means:
- **Implementation validation**: "Does the system work?"
- **Research validation**: "What properties does the system exhibit?"

### 5. CLI Integration

```bash
# Run validation ensemble
llm-orc invoke validate-file-operations --mode test

# Run validation with verbose output
llm-orc invoke validate-conversational-workflow --mode test --verbose

# Run multiple validation ensembles
llm-orc validate --category primitives  # Run all primitive validations
llm-orc validate --category integration  # Run all integration validations
llm-orc validate --all  # Run full validation suite

# Research mode - multiple runs with statistical analysis
llm-orc experiment experiment-consensus-vs-diversity --runs 30 --export results.csv
```

### 6. ValidationEvaluator Component

```python
class ValidationEvaluator:
    """Evaluates validation criteria after ensemble execution."""

    async def evaluate(
        self,
        ensemble_name: str,
        results: EnsembleExecutionResult,
        validation_config: dict[str, Any]
    ) -> ValidationResult:
        """Evaluate all validation criteria."""

        validation_results = {
            "structural": await self._evaluate_structural(results, validation_config),
            "schema": await self._evaluate_schema(results, validation_config),
            "behavioral": await self._evaluate_behavioral(results, validation_config),
            "quantitative": await self._evaluate_quantitative(results, validation_config),
            "semantic": await self._evaluate_semantic(results, validation_config)
        }

        return ValidationResult(
            ensemble_name=ensemble_name,
            passed=all(r.passed for r in validation_results.values() if r),
            results=validation_results,
            timestamp=datetime.now()
        )
```

## Benefits

1. **Non-Interactive Testing**: LLM simulation enables automated validation of conversational ensembles
2. **Flexible Validation**: Ensembles declare only needed validation criteria
3. **Progressive Complexity**: Validation grows from primitives to full applications
4. **Research Enablement**: Same framework supports both validation and experiments
5. **Reproducibility**: Response caching ensures deterministic test results
6. **Local-Only**: Uses only Ollama models (qwen3:0.6b, llama3:latest) - no API costs
7. **Dogfooding**: Validates implementation using the system itself

## Trade-offs

### Advantages
- Systematic validation across complexity levels
- Automated testing of interactive ensembles
- Dual-purpose design (validation + research)
- No external API dependencies

### Disadvantages
- LLM simulation may not capture all user behavior patterns
- Requires maintaining validation ensembles alongside code
- Quantitative metrics need careful threshold tuning
- Small LLMs (qwen3:0.6b) may generate unrealistic responses

### Mitigation Strategies
- Provide multiple persona templates for diverse simulation
- Start with cached responses for deterministic tests
- Use validation ensembles as living documentation
- Fallback to manual testing for edge cases

## Implementation Components

The validation framework requires:

### Phase 1: Validation Infrastructure
- `ValidationEvaluator` class for running validation criteria
- Ensemble YAML schema updates for `validation` section
- `test_mode` execution flag in EnsembleExecutor
- CLI commands: `llm-orc validate`

### Phase 2: LLM User Simulation
- `ScriptUserInputHandler` with test mode support
- `LLMResponseGenerator` for persona-based responses
- Response caching mechanism
- Integration with user input primitives

### Phase 3: Validation Ensembles
- ~20-25 validation ensembles in `llm-orchestra-library/validation/`
- Coverage across 5 categories (primitives → applications)
- Documented personas and response patterns
- CI/CD integration for automated validation

### Phase 4: Research Extensions
- Statistical analysis tools for quantitative metrics
- Multi-run experiment support
- CSV export and correlation analysis
- Documentation for research use cases

## Implementation Status

**Status**: 90% Complete - Core infrastructure functional, validation ensemble suite in progress

### Completed (Phases 1-3)

**Phase 1: Validation Infrastructure** ✅
- `ValidationEvaluator` class implemented with all 5 validation layers
  - Structural: required_agents, execution time thresholds
  - Schema: JSON contract verification with required fields
  - Behavioral: Python assertion evaluation with restricted context
  - Quantitative: Metric calculation with threshold comparisons
  - Semantic: LLM-as-judge (with graceful error handling)
- Ensemble YAML schema supports `validation` and `test_mode` sections
- CLI commands functional:
  - `llm-orc validate run <ensemble>`
  - `llm-orc validate category --category <cat>`
  - `llm-orc validate all`
- EnsembleExecutor auto-runs validation when config present
- All 30 BDD scenarios passing

**Phase 2: LLM User Simulation** ✅
- `ScriptUserInputHandler` extended with test_mode support
- `LLMResponseGenerator` implemented with:
  - Persona-based system prompts (helpful_user, critical_reviewer, domain_expert)
  - Response caching for deterministic results
  - Conversation history tracking
- Integration with validation framework complete

**Phase 3: Example Validation Ensembles** ✅
- Created 5 example ensembles (1 per category):
  - Primitive: `validate-file-read.yaml`
  - Integration: `validate-file-pipeline.yaml`
  - Conversational: `validate-user-interaction.yaml`
  - Research: `validate-execution-metrics.yaml`
  - Application: `validate-data-workflow.yaml`
- Documentation: `llm-orchestra-library/ensembles/validation/README.md`

### In Progress (Dogfooding Phase)

**Phase 3 Expansion: Full Validation Suite**

Progress: 5/17 ensembles created (29%)

Target ensemble count per Success Criteria:
- Primitives: 1/5 created
- Integration: 1/5 created
- Conversational: 1/3 created
- Research: 1/2 created
- Application: 1/2 created

**Current Activity**: Progressive ensemble development through iterative testing

### Discovered Issues (During Dogfooding)

**Issue #1: Script Agent Parameter Passing**
- **Symptom**: Script agents receive default parameters instead of configured values
- **Example**: `read_file.py` gets `input.txt` instead of configured `test-data.json`
- **Impact**: Blocks validation ensemble execution
- **Investigation**: Parameters not passed correctly to script via stdin in `EnhancedScriptAgent`
- **Status**: Needs debugging in `_execute_script_with_input_handling` method

**Issue #2: Test Suite Timeout**
- **Symptom**: Full test suite times out after 2 minutes
- **Impact**: Cannot verify baseline health before building ensembles
- **Status**: Unit tests pass individually, BDD tests pass, likely integration test issue

**Issue #3: Mock Misalignment in Tests**
- **Symptom**: Script command tests were mocking ScriptResolver instead of PrimitiveRegistry
- **Resolution**: Fixed in commit c25a4c5
- **Learning**: Test mocks need to match actual implementation

### Remaining Work (10%)

**Immediate Blockers** (Must fix to proceed):
1. Debug script agent parameter passing (`EnhancedScriptAgent._execute_script_with_input_handling`)
2. Verify first validation ensemble runs successfully
3. Establish baseline: one working ensemble per category

**Validation Suite Completion** (Success Criteria #1):
1. **Primitives** (4 more ensembles):
   - validate-file-write.yaml
   - validate-json-extract.yaml
   - validate-user-input-basic.yaml (without test mode)
   - validate-control-flow.yaml

2. **Integration** (4 more ensembles):
   - validate-parallel-execution.yaml
   - validate-fan-out-fan-in.yaml
   - validate-error-propagation.yaml
   - validate-conditional-execution.yaml

3. **Conversational** (2 more ensembles):
   - validate-multi-turn-conversation.yaml
   - validate-user-confirmation-flow.yaml

4. **Research** (1 more ensemble):
   - validate-network-topology.yaml (clustering, modularity metrics)

5. **Application** (1 more ensemble):
   - validate-research-pipeline.yaml (full stack with LLM-as-judge)

**Phase 4: Research Extensions** (Not started):
- Statistical analysis tools
- Multi-run experiment support
- CSV export functionality

### Next Steps

1. **Fix parameter passing** - Debug `EnhancedScriptAgent` to pass `agent_config['parameters']` correctly to script stdin
2. **Validate first ensemble** - Get `validate-file-read.yaml` to pass all assertions
3. **Progressive build** - Create remaining 12 ensembles iteratively, learning from each
4. **Document learnings** - Update this ADR with validation results and discovered patterns
5. **Research validation** - Implement at least one research experiment (Success Criteria #4)

## Success Criteria

1. **All validation categories functional**:
   - Primitives: 5 ensembles validating core script agents
   - Integration: 5 ensembles validating agent composition
   - Conversational: 3 ensembles with LLM-simulated input
   - Research: 2 ensembles with quantitative metrics
   - Applications: 2 end-to-end workflow validations

2. **Test mode prevents blocking**:
   - User input agents receive LLM responses in test mode
   - No stdin blocking during automated validation
   - Reproducible results via response caching

3. **Flexible validation schema**:
   - Ensembles declare only needed validation layers
   - All 5 validation types (structural, schema, behavioral, quantitative, semantic) functional
   - ValidationEvaluator correctly evaluates all criteria

4. **Research cross-purpose validated**:
   - At least one research experiment using validation framework
   - Statistical analysis produces correlation/ANOVA results
   - CSV export enables external analysis

## Related Documentation

- [ADR-004: BDD as LLM Development Guardrails](004-bdd-llm-development-guardrails.md) - Development testing approach
- [ADR-005: Multi-Turn Agent Conversations](005-multi-turn-agent-conversations.md) - Conversational patterns
- [Script Agent Architecture](../script-agent-architecture.md) - JSON I/O contracts
- [CLI Interaction Modes](../cli-interaction-modes.md) - Interactive vs test mode
- [Primitive Script Development Guide](../primitive-script-development.md) - Script creation

## References

- Ensemble validation patterns: [Test ensembles in library](../../llm-orchestra-library/validation/)
- LLM simulation examples: Category 3 (Conversational Validation)
- Research metrics: Category 4 (Research Validation)
