@adr-012 @pydantic @agent-config
Feature: Pydantic Agent Config Models
  As a developer building ensemble configurations
  I want agent configs validated by Pydantic at load time
  So that typos and invalid fields are caught before execution rather than silently failing

  Background:
    Given llm-orc is properly configured
    And the Pydantic agent config models are in use

  # LLM agent config
  @llm-agent
  Scenario: LLM agent config validates required and optional fields
    Given an agent config with name "analyzer" and model_profile "claude-sonnet"
    When the config is parsed by EnsembleLoader
    Then an LlmAgentConfig should be created
    And model_profile should be set to "claude-sonnet"
    And depends_on should default to an empty list
    And fan_out should default to False

  @llm-agent
  Scenario: Inline LLM agent requires provider when model is specified
    Given an agent config with name "analyzer", model "claude-3-5-sonnet", but no provider
    When the config is parsed by EnsembleLoader
    Then a Pydantic validation error should be raised
    And the error should indicate that provider is required when model is specified

  @llm-agent
  Scenario: LLM agent cannot specify both model_profile and inline model
    Given an agent config with both model_profile "claude-sonnet" and model "claude-3-5-sonnet"
    When the config is parsed by EnsembleLoader
    Then a Pydantic validation error should be raised
    And the error should indicate that model_profile and model are mutually exclusive

  # Script agent config
  @script-agent
  Scenario: Script agent config validates script field presence
    Given an agent config with name "formatter" and script "primitives/format_output.py"
    When the config is parsed by EnsembleLoader
    Then a ScriptAgentConfig should be created
    And script should be set to "primitives/format_output.py"
    And parameters should default to an empty dict

  # Extra fields forbidden
  @validation
  Scenario: Unknown fields in agent config are rejected at load time
    Given an agent config with name "analyzer", model_profile "claude-sonnet", and an unknown field "typo_field"
    When the config is parsed by EnsembleLoader
    Then a Pydantic validation error should be raised
    And the error should identify "typo_field" as an unexpected field

  @validation
  Scenario: Removed fields cause load-time errors
    Given an ensemble YAML containing an agent with a "type:" key
    When the ensemble is loaded by EnsembleLoader
    Then a Pydantic validation error should be raised
    And the error should identify "type" as an unexpected field

  # Type dispatch
  @dispatch
  Scenario: AgentDispatcher routes LlmAgentConfig to LLM runner
    Given an LlmAgentConfig for agent "analyzer"
    When AgentDispatcher processes the config
    Then it should route to LlmAgentRunner using isinstance check
    And no key inspection should be used for type determination

  @dispatch
  Scenario: AgentDispatcher routes ScriptAgentConfig to script runner
    Given a ScriptAgentConfig for agent "formatter"
    When AgentDispatcher processes the config
    Then it should route to ScriptAgentRunner using isinstance check

  # Structural change — no behavior change
  @behavioral-equivalence
  Scenario: Ensemble with valid agents executes identically before and after migration
    Given a valid ensemble configuration with both LLM and script agents
    When the ensemble executes after the Pydantic migration
    Then all agents should produce the same outputs as before the migration
    And no logic changes should occur — only type narrowing
