@adr-013 @ensemble-agent @composition
Feature: Ensemble Agent Type
  As a developer building complex orchestration workflows
  I want an agent that can invoke another ensemble
  So that I can compose ensembles without introducing a parallel execution format

  Background:
    Given llm-orc is properly configured
    And the Pydantic agent config models are in use (ADR-012)

  # Config and dispatch
  @config
  Scenario: Agent config with ensemble key parses as EnsembleAgentConfig
    Given an agent config with name "security-scan" and ensemble "security-scanner"
    When the config is parsed by EnsembleLoader
    Then an EnsembleAgentConfig should be created
    And the ensemble field should be set to "security-scanner"

  @dispatch
  Scenario: AgentDispatcher routes EnsembleAgentConfig to EnsembleAgentRunner
    Given an EnsembleAgentConfig referencing ensemble "security-scanner"
    When AgentDispatcher processes the config
    Then it should route to EnsembleAgentRunner using isinstance check

  # Execution
  @execution
  Scenario: Ensemble agent executes the referenced child ensemble
    Given an ensemble "pipeline" with an ensemble agent referencing "security-scanner"
    And the "security-scanner" ensemble exists and is valid
    When the "pipeline" ensemble executes
    Then "security-scanner" should be executed as a child ensemble
    And the child ensemble's result should appear in the parent's agent outputs

  @execution
  Scenario: Child ensemble shares immutable infrastructure with parent
    Given a parent ensemble executing a child ensemble agent
    When the child ensemble runs
    Then the child executor should share the parent's config manager and model factory
    And the child executor should have its own isolated usage collector and event queue

  @execution
  Scenario: Child ensemble does not save its own artifact
    Given a parent ensemble executing a child ensemble agent
    When execution completes
    Then only one artifact should be saved — the parent's
    And the child's results should be nested within the parent artifact

  # Dependency integration
  @dependencies
  Scenario: Ensemble agent participates in dependency chains
    Given an ensemble with agents "classify", "scan" (depends_on classify, type ensemble), and "report" (depends_on scan)
    When the ensemble executes
    Then "classify" should run first
    And "scan" should run after "classify" completes
    And "report" should run after "scan" completes

  # Cycle detection
  @cycle-detection
  Scenario: Cross-ensemble circular reference is detected at load time
    Given ensemble "A" contains an ensemble agent referencing "B"
    And ensemble "B" contains an ensemble agent referencing "A"
    When either ensemble is loaded
    Then a load-time error should be raised
    And the error should identify the circular reference between "A" and "B"

  # Depth limiting
  @depth-limiting
  Scenario: Ensemble execution halts when depth limit is exceeded
    Given a chain of ensembles where each references the next beyond the depth limit
    When the top-level ensemble executes
    Then execution should stop with a clear error when the depth limit is reached
    And the error should indicate the depth limit was exceeded

  # Failure handling
  @failure-handling
  Scenario: Child ensemble failure is recorded as agent failure, not orchestration failure
    Given a parent ensemble with ensemble agent "scan" and independent agent "report"
    And the "scan" child ensemble is configured to fail
    When the parent ensemble executes
    Then the "scan" agent result should be marked as failed
    And the "report" agent should still execute since it does not depend on "scan"
    And the parent ensemble should record a has_errors flag without aborting
