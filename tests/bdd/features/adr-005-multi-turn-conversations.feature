Feature: ADR-005 Multi-Turn Agent Conversations
  """
  LLM Development Context:

  Requirements: Extend dependency system to support multi-turn conversations between
  mixed agent types (script + model_profile) with state accumulation across turns.

  Architectural constraints (ADR-005):
  - All conversation components must use Pydantic schemas (ADR-001)
  - ConversationalAgent extends existing Agent patterns (ADR-002)
  - Exception chaining required for conversation errors (ADR-003)
  - Small local models (llama3.2:1b, qwen2.5:1.5b) for efficient testing
  - Turn limits prevent infinite loops with graceful completion

  Implementation patterns:
  - ConversationState tracks turn_count, agent_execution_count, accumulated_context
  - ConversationalDependency supports condition evaluation and max_executions
  - Input injection uses LLMResponseGenerator with caching for reproducibility
  - Mixed agent types: script agents provide data, LLM agents provide reasoning

  Critical validations:
  - Context accumulation across script→LLM→script flows
  - Conditional dependency evaluation (runtime state-based execution)
  - Conversation limits enforcement (max_total_turns, per-agent limits)
  - AgentRequest processing within conversations
  - Data integrity across conversation turns
  """

  As a developer using llm-orc
  I want agents to have multi-turn conversations with accumulated context
  So that I can build complex workflows requiring iterative refinement

  Background:
    Given llm-orc is properly configured
    And the conversation system is initialized

  @priority @adr-005
  Scenario: Agents converse over multiple turns with accumulated context
    Given a conversational ensemble with mixed agent types
    When the mixed agent conversation executes with script and LLM agents
    Then script and LLM agents should collaborate across multiple turns
    And context should accumulate correctly between turns
    And conversation should complete within turn limits

  @adr-005
  Scenario: Script agents pass data to LLM agents in conversations
    Given a conversation with script agent followed by LLM agent
    When the script agent produces structured output
    And the LLM agent receives that output as context
    Then the LLM agent should use the script output in its reasoning
    And the conversation should maintain data integrity

  @adr-005
  Scenario: LLM agents trigger user input requests in conversations
    Given a conversation with LLM agent that needs clarification
    When the LLM agent outputs a needs_clarification signal
    Then a user input script agent should be triggered
    And input injection should provide a contextual response
    And the conversation should continue with the clarification

  @adr-005
  Scenario: Conditional dependencies control conversation flow
    Given a conversation with conditional agent dependencies
    When agents execute based on runtime conditions
    Then only agents whose conditions are met should execute
    And conversation should follow the conditional logic correctly
    And turn limits should be respected

  @adr-005
  Scenario: Input injection provides realistic user responses
    Given a conversation requiring user input
    And input injection is configured with small local models
    When user input is needed during conversation
    Then the injection system should delegate to local LLM agents
    And responses should be contextually appropriate
    And the conversation should continue naturally

  @adr-005
  Scenario: Conversation state persists across multiple turns
    Given a multi-turn conversation with state accumulation
    When agents execute across several conversation turns
    Then conversation state should persist between turns
    And agent execution counts should be tracked correctly
    And conversation history should be maintained

  @adr-005
  Scenario: Mixed agent types collaborate in conversation cycles
    Given a conversation with script→LLM→script→LLM flow
    When agents execute in conversational cycles
    Then script agents should provide data for LLM processing
    And LLM agents should generate insights for script action
    And the conversation should complete successfully
    And all agent types should participate appropriately

  @adr-005
  Scenario: Small local models enable efficient conversation testing
    Given a conversation using small local models for testing
    When the conversation executes with llama3.2:1b and qwen2.5:1.5b
    Then conversation should complete within reasonable time
    And all conversation mechanics should work correctly
    And local model responses should be contextually relevant

  @adr-005
  Scenario: Conversation limits prevent infinite loops
    Given a conversation with potential for infinite cycles
    When conversation execution begins
    Then execution should stop at max_total_turns limit
    And graceful completion should occur
    And conversation state should reflect proper termination

  @adr-005
  Scenario: Agent request generation works in conversations
    Given agents that can generate requests for other agents
    When an agent outputs AgentRequest objects
    Then the conversation system should process those requests
    And target agents should be triggered appropriately
    And request parameters should be passed correctly

  @adr-005 @architectural-compliance
  Scenario: ConversationalAgent schema validates mixed agent types
    Given a ConversationalAgent configuration with script field
    When the agent configuration is validated
    Then the agent should be classified as script type
    And model_profile field should be None
    And conversation config should be properly validated

  @adr-005 @architectural-compliance
  Scenario: ConversationalAgent schema validates LLM agent types
    Given a ConversationalAgent configuration with model_profile field
    When the agent configuration is validated
    Then the agent should be classified as LLM type
    And script field should be None
    And conversation config should be properly validated

  @adr-005 @architectural-compliance
  Scenario: ConversationalAgent schema rejects invalid dual-type agents
    Given a ConversationalAgent configuration with both script and model_profile
    When the agent configuration is validated
    Then validation should fail with clear error message
    And the error should indicate mutual exclusivity requirement

  @adr-005 @error-handling
  Scenario: Conversation execution handles script agent failures with chaining
    Given a conversation with a script agent that fails
    When the script agent raises an exception during execution
    Then the conversation should catch and chain the exception properly
    And the conversation should continue with remaining agents
    And error context should be preserved in conversation state

  @adr-005 @error-handling
  Scenario: Conversation execution handles LLM agent failures with chaining
    Given a conversation with an LLM agent that fails
    When the LLM agent raises an exception during generation
    Then the conversation should catch and chain the exception properly
    And the conversation should continue with remaining agents
    And error context should be preserved in conversation state

  @adr-005 @performance
  Scenario: Small local models provide fast conversation testing
    Given a conversation configured with llama3.2:1b and qwen2.5:1.5b
    When the conversation executes with input injection
    Then conversation should complete within 30 seconds
    And local model responses should be contextually relevant
    And conversation state should accumulate properly

  @adr-005 @state-management
  Scenario: ConversationState tracks execution counts accurately
    Given a multi-turn conversation with repeated agent executions
    When agents execute multiple times within their turn limits
    Then agent_execution_count should increment correctly
    And max_turns limits should be enforced per agent
    And conversation should stop when any agent reaches its limit

  @adr-005 @dependency-resolution
  Scenario: Conditional dependencies use safe expression evaluation
    Given conditional dependencies with complex state expressions
    When dependency conditions are evaluated against conversation state
    Then expressions should be evaluated safely without code injection
    And only whitelisted variables should be accessible
    And malformed expressions should fail gracefully with clear errors