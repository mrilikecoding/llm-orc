# ADR-005: Multi-Turn Agent Conversations

## Status
Superseded by ADR-011

> **Superseded (2026-02-20):** Research for composable ensemble orchestration (ADR-011) found that the conversational ensemble system is dead code — never imported by the CLI, MCP server, or any execution path. The regular execution path already supports user-directed flows through script primitives and the ScriptUserInputHandler. All code produced by this ADR is being removed.

## Implementation Status
- [x] BDD scenarios created and passing (18/18)
- [x] ConversationState management implemented
- [x] Conditional dependency resolution complete
- [x] ConversationalEnsembleExecutor functional
- [x] Input injection for testing operational
- [x] Safe expression evaluation implemented
- [x] Mixed agent type support complete
- [ ] Advanced conversation patterns
- [ ] Performance optimization
- [ ] Visual workflow designer

## BDD Integration
- **Scenario File**: tests/bdd/features/adr-005-multi-turn-conversations.feature
- **Test Command**: `uv run pytest tests/bdd/ -k adr-005`
- **Priority Scenario**: "Agents converse over multiple turns with accumulated context"

## Context

Current agent orchestration executes each agent (both script and LLM agents) once in a static dependency order. Many real-world scenarios require agents to interact multiple times, building context across turns:

### Use Cases Requiring Multi-Turn Conversations
1. **Iterative Problem Solving**: Analyzer → User clarification → Analyzer → Solution
2. **Content Creation Workflows**: Writer → Reviewer → Writer → Editor → Writer
3. **Interactive Debugging**: Error detector → Human input → Fixer → Validator → (repeat)
4. **Game Scenarios**: Story generator → Player input → Story continuation → (repeat)

### Current Limitations
```yaml
# Current: Static, single-execution dependency chain for all agent types
agents:
  - name: analyzer
    script: analyze.py

  - name: llm_reviewer
    model_profile: gpt-4-turbo
    depends_on: [analyzer]  # Executes once only

  - name: user_input
    script: get_input.py
    depends_on: [llm_reviewer]  # Linear, no loops

  - name: final_processor
    model_profile: claude-3-sonnet
    depends_on: [user_input]  # No multi-turn loops
```

### Business Impact
- Cannot model conversational workflows
- User interaction limited to single exchanges
- No support for iterative refinement processes
- Complex decision-making scenarios not testable

## Decision

Implement **Multi-Turn Agent Conversations** by extending the dependency system to support:

1. **Conditional Dependencies**: Agents execute based on runtime state
2. **Multiple Executions**: Same agent can run multiple times
3. **State Accumulation**: Context persists across conversation turns
4. **Conversation Limits**: Prevent infinite loops with turn limits

## Detailed Design

### Core Schema Extensions (Building on ADR-001)

```python
class ConversationalDependency(BaseModel):
    """Dependency that can trigger multiple times based on conditions."""

    agent_name: str
    condition: str | None = None         # When to execute this dependency
    max_executions: int = 1             # How many times this dep can trigger
    requires_all: bool = True           # Existing dependency logic

class ConversationalAgent(BaseModel):
    """Agent with conversation support for both script and LLM agents."""

    name: str
    type: Literal["script", "llm"] | None = None  # Optional explicit type
    script: str | None = None                    # Script agents have script path
    model_profile: str | None = None             # LLM agents have model profile
    config: dict[str, Any] = Field(default_factory=dict)
    dependencies: list[ConversationalDependency] = Field(default_factory=list)

    # Conversation-specific config
    conversation: ConversationConfig | None = None

    @validator("__root__")
    def validate_agent_type(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Validate that agent has either script or model_profile."""
        script = values.get("script")
        model_profile = values.get("model_profile")

        if not script and not model_profile:
            raise ValueError(
                "Agent must have either 'script' or 'model_profile' field"
            )
        if script and model_profile:
            raise ValueError(
                "Agent cannot have both 'script' and 'model_profile' fields"
            )
        return values

class ConversationConfig(BaseModel):
    """Configuration for agent conversation behavior."""

    max_turns: int = 1                  # Max times this agent can execute
    state_key: str | None = None        # Key to store agent's state
    triggers_conversation: bool = False  # Whether this agent can start conversations

class ConversationalEnsemble(BaseModel):
    """Ensemble supporting multi-turn conversations."""

    name: str
    agents: list[ConversationalAgent]
    conversation_limits: ConversationLimits

class ConversationLimits(BaseModel):
    """Global limits for conversation execution."""

    max_total_turns: int = 20
    max_agent_executions: dict[str, int] = Field(default_factory=dict)
    timeout_seconds: int = 300
```

### Conversation State Management

```python
class ConversationState(BaseModel):
    """Persistent state across conversation turns."""

    turn_count: int = 0
    agent_execution_count: dict[str, int] = Field(default_factory=dict)
    accumulated_context: dict[str, Any] = Field(default_factory=dict)
    conversation_history: list[ConversationTurn] = Field(default_factory=list)

    def should_execute_agent(self, agent: ConversationalAgent) -> bool:
        """Check if agent should execute based on conversation limits."""
        current_executions = self.agent_execution_count.get(agent.name, 0)
        max_executions = agent.conversation.max_turns if agent.conversation else 1

        return current_executions < max_executions

    def evaluate_condition(self, condition: str) -> bool:
        """Evaluate dependency condition against current state."""
        if not condition:
            return True

        # Safe evaluation of condition expressions
        context = {
            "turn_count": self.turn_count,
            "context": self.accumulated_context,
            "history": self.conversation_history
        }

        return eval(condition, {"__builtins__": {}}, context)

class ConversationTurn(BaseModel):
    """Record of a single conversation turn."""

    turn_number: int
    agent_name: str
    input_data: dict[str, Any]
    output_data: dict[str, Any]
    execution_time: float
    timestamp: datetime
```

### Conversational Dependency Resolution

```python
class ConversationalDependencyResolver(DependencyResolver):
    """Dependency resolver supporting multi-turn conversations."""

    def get_ready_agents_for_conversation(
        self,
        agents: list[ConversationalAgent],
        conversation_state: ConversationState
    ) -> list[ConversationalAgent]:
        """Get agents ready to execute considering conversation state."""

        ready_agents = []

        for agent in agents:
            # Check basic dependency satisfaction (existing logic)
            if not self.dependencies_satisfied(agent, conversation_state.accumulated_context):
                continue

            # Check conversation-specific conditions
            if not conversation_state.should_execute_agent(agent):
                continue

            # Evaluate conditional dependencies
            agent_ready = True
            for dep in agent.dependencies:
                if dep.condition and not conversation_state.evaluate_condition(dep.condition):
                    agent_ready = False
                    break

            if agent_ready:
                ready_agents.append(agent)

        return ready_agents
```

### Conversational Ensemble Execution

```python
class ConversationalEnsembleExecutor(EnsembleExecutor):
    """Executor supporting multi-turn agent conversations."""

    async def execute_conversation(
        self,
        ensemble: ConversationalEnsemble,
        initial_context: dict[str, Any] = None
    ) -> ConversationResult:
        """Execute ensemble as a multi-turn conversation."""

        conversation_state = ConversationState(
            accumulated_context=initial_context or {}
        )

        dependency_resolver = ConversationalDependencyResolver()

        while conversation_state.turn_count < ensemble.conversation_limits.max_total_turns:
            # Get agents ready for this turn
            ready_agents = dependency_resolver.get_ready_agents_for_conversation(
                ensemble.agents,
                conversation_state
            )

            if not ready_agents:
                break  # Conversation complete - no more agents can execute

            # Execute ready agents (reuse existing execution logic)
            turn_results = await self.execute_agents_with_context(
                ready_agents,
                conversation_state.accumulated_context
            )

            # Update conversation state
            self.update_conversation_state(conversation_state, turn_results)

            # Check for agent requests that might trigger new conversations
            self.process_agent_requests(turn_results, conversation_state)

        return ConversationResult(
            final_state=conversation_state.accumulated_context,
            conversation_history=conversation_state.conversation_history,
            turn_count=conversation_state.turn_count,
            completion_reason=self.determine_completion_reason(conversation_state, ensemble)
        )

    def update_conversation_state(
        self,
        state: ConversationState,
        turn_results: dict[str, Any]
    ) -> None:
        """Update conversation state with turn results."""

        state.turn_count += 1

        for agent_name, result in turn_results.items():
            # Update execution count
            state.agent_execution_count[agent_name] = \
                state.agent_execution_count.get(agent_name, 0) + 1

            # Accumulate context
            if result.get("success") and result.get("data"):
                state.accumulated_context[agent_name] = result["data"]

            # Record turn
            state.conversation_history.append(ConversationTurn(
                turn_number=state.turn_count,
                agent_name=agent_name,
                input_data=result.get("input", {}),
                output_data=result.get("output", {}),
                execution_time=result.get("execution_time", 0),
                timestamp=datetime.now()
            ))
```

### Input Injection for Testing

```python
class ConversationalInputHandler:
    """Handles user input in conversational ensembles."""

    def __init__(self, test_mode: bool = False):
        self.test_mode = test_mode
        self.response_generators: dict[str, ResponseGenerator] = {}

    def register_response_generator(
        self,
        agent_name: str,
        generator: ResponseGenerator
    ) -> None:
        """Register a response generator for specific agent."""
        self.response_generators[agent_name] = generator

    async def handle_input_request(
        self,
        agent_name: str,
        prompt: str,
        conversation_state: ConversationState
    ) -> str:
        """Handle input request with conversation context."""

        if self.test_mode and agent_name in self.response_generators:
            generator = self.response_generators[agent_name]
            return await generator.generate_response(prompt, conversation_state)
        else:
            # Real user input
            return await self.get_real_user_input(prompt)

class LLMResponseGenerator:
    """Generates responses using LLM agents."""

    def __init__(self, llm_config: dict[str, Any]):
        self.llm_config = llm_config
        self.response_cache: dict[str, str] = {}

    async def generate_response(
        self,
        prompt: str,
        conversation_state: ConversationState
    ) -> str:
        """Generate contextual response using LLM."""

        # Create cache key including conversation context
        context_summary = {
            "turn": conversation_state.turn_count,
            "recent_agents": [turn.agent_name for turn in conversation_state.conversation_history[-3:]],
            "key_context": {k: v for k, v in conversation_state.accumulated_context.items()
                          if isinstance(v, (str, int, bool))}
        }

        cache_key = hash((prompt, str(context_summary)))

        if cache_key in self.response_cache:
            return self.response_cache[cache_key]

        # Generate response with LLM
        llm_prompt = f"""
        Conversation context: {context_summary}
        Current prompt: {prompt}

        Respond as a user would in this conversation context. Be realistic and contextually appropriate.
        """

        response = await self.execute_llm_agent(self.llm_config, llm_prompt)

        self.response_cache[cache_key] = response
        return response
```

## Example Usage

### Configuration
```yaml
# ensemble.yaml with conversation support for mixed agent types
name: iterative-analysis
agents:
  - name: data_extractor
    script: primitives/analysis/extract_data.py
    conversation:
      max_turns: 2

  - name: llm_analyzer
    model_profile: efficient
    prompt: |
      Analyze the data provided by the data extractor agent.
      If more clarification is needed, output: {"needs_clarification": true}
    dependencies:
      - agent_name: data_extractor
    conversation:
      max_turns: 3
      triggers_conversation: true

  - name: user_clarification
    script: primitives/user-interaction/get_clarification.py
    dependencies:
      - agent_name: llm_analyzer
        condition: "context.get('needs_clarification', False)"
        max_executions: 3
    conversation:
      max_turns: 3

  - name: final_synthesis
    model_profile: micro-local
    prompt: |
      Synthesize all the analysis and clarification into a final report.
    dependencies:
      - agent_name: user_clarification
        condition: "context.get('clarification_provided', False)"

conversation_limits:
  max_total_turns: 15
  timeout_seconds: 600
```

### BDD Testing
```python
@given("a conversational ensemble with mixed agent types")
def setup_mixed_agent_conversation(bdd_context):
    ensemble = ConversationalEnsemble.from_yaml("iterative_analysis.yaml")

    # Set up input injection for testing
    input_handler = ConversationalInputHandler(test_mode=True)

    # Mock user input responses for script agents using small local models
    input_handler.register_response_generator(
        "user_clarification",
        LLMResponseGenerator({"model_profile": "llama3.2:1b"})
    )

    bdd_context["ensemble"] = ensemble
    bdd_context["input_handler"] = input_handler

@when("the mixed agent conversation executes with script and LLM agents")
async def execute_mixed_agent_conversation(bdd_context):
    executor = ConversationalEnsembleExecutor(
        input_handler=bdd_context["input_handler"]
    )

    result = await executor.execute_conversation(bdd_context["ensemble"])
    bdd_context["conversation_result"] = result

@then("script and LLM agents should collaborate across multiple turns")
def validate_mixed_agent_conversation(bdd_context):
    result = bdd_context["conversation_result"]

    # Validate conversation structure
    assert result.turn_count > 1, "Should have multiple turns"

    # Check mixed agent execution pattern
    agent_sequence = [turn.agent_name for turn in result.conversation_history]

    # Verify script agents executed
    assert "data_extractor" in agent_sequence
    assert "user_clarification" in agent_sequence

    # Verify LLM agents executed
    assert "llm_analyzer" in agent_sequence
    assert "final_synthesis" in agent_sequence

    # Validate context flows between agent types
    assert "final_analysis" in result.final_state
    assert result.final_state["synthesis_complete"] is True
```

## Benefits

### Natural Conversation Flows
- **Multi-Turn Interactions**: Agents can have extended conversations
- **Context Accumulation**: Information builds across conversation turns
- **Conditional Logic**: Dependency execution based on runtime state
- **Realistic Testing**: LLM-generated responses based on conversation context

### Architectural Consistency
- **Builds on Existing System**: Extends current dependency resolution
- **ADR-001 Compliant**: All conversation components use Pydantic schemas
- **Backward Compatible**: Regular ensembles continue to work unchanged
- **Reuses Infrastructure**: Same agent execution, error handling, etc.

## Trade-offs

### Complexity
- **Configuration Overhead**: Conversation ensembles require more setup
- **State Management**: Tracking conversation state adds complexity
- **Debugging Difficulty**: Multi-turn flows harder to debug than linear ones

### Performance
- **Longer Execution**: Conversations take more time than single-pass execution
- **Memory Usage**: Conversation history and state accumulation
- **Cache Storage**: Response caching for reproducible testing

### Reliability
- **Infinite Loops**: Need careful limits and cycle detection
- **State Consistency**: Context accumulation must be reliable
- **Test Reproducibility**: LLM responses need consistent caching

## Implementation Plan

### Phase 1: Core Conversation Support (Week 1)
- [ ] Create conversational schemas extending existing models
- [ ] Implement `ConversationState` and turn tracking
- [ ] Add conditional dependency evaluation
- [ ] Create basic conversation limits and cycle detection

### Phase 2: Conversational Execution (Week 2)
- [ ] Extend `EnsembleExecutor` with conversation support
- [ ] Implement conversational dependency resolution
- [ ] Add conversation state accumulation and persistence
- [ ] Create conversation completion detection

### Phase 3: Input Injection and Testing (Week 3)
- [ ] Implement `ConversationalInputHandler` with test mode
- [ ] Create `LLMResponseGenerator` with caching
- [ ] Add BDD testing support for conversational ensembles
- [ ] Validate conversation state accumulation in tests

### Phase 4: Integration and Documentation (Week 4)
- [ ] Update existing BDD scenarios to use conversational ensembles
- [ ] Create comprehensive conversation testing examples
- [ ] Add performance testing for multi-turn conversations
- [ ] Document conversation patterns and best practices

## Success Criteria

- [ ] Can execute agents multiple times in conditional loops
- [ ] Conversation state accumulates correctly across turns
- [ ] Input injection works non-interactively for testing
- [ ] LLM agents generate contextually appropriate responses
- [ ] Conversation limits prevent infinite loops
- [ ] All existing ensemble functionality remains unchanged
- [ ] BDD tests execute conversational flows end-to-end

## Architectural Compliance

- **Related ADRs**:
  - ADR-001 (Pydantic schemas for all conversation components)
  - ADR-002 (Agents in conversations are composable primitives)
  - ADR-003 (Exception chaining for conversation errors)
- **Validation Rules**: All conversation state must be serializable; conditional expressions must be safely evaluatable; conversation limits must be enforced
- **Breaking Changes**: None - extends existing ensemble system without breaking changes