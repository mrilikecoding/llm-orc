@adr-011 @cleanup
Feature: Remove Conversational Ensemble System
  As a developer maintaining llm-orc
  I want the conversational ensemble system removed
  So that the codebase has no unreachable execution paths and the Pydantic migration has a clean surface

  Background:
    Given llm-orc is properly configured
    And the conversational ensemble system has been removed

  # Dead code is gone
  @removal
  Scenario: ConversationalEnsembleExecutor no longer exists
    When I inspect the codebase for ConversationalEnsembleExecutor
    Then the class should not be present in any source file

  @removal
  Scenario: Conversational schemas are no longer importable
    When I attempt to import ConversationalAgent, ConversationalEnsemble, or ConversationState
    Then an ImportError should be raised

  # Production execution paths are unaffected
  @execution
  Scenario: CLI invocation does not reference conversational executor
    Given an ensemble configured without conversation keys
    When I execute the ensemble via the CLI
    Then execution should complete using the regular EnsembleExecutor
    And no conversational executor code should be involved

  @execution
  Scenario: MCP invoke tool does not reference conversational executor
    Given an ensemble named "simple-test" exists
    When I call the MCP "invoke" tool for "simple-test"
    Then execution should complete using the regular EnsembleExecutor

  # User input via script primitives still works
  @user-input
  Scenario: User input flows through script primitives after removal
    Given an ensemble using the get_user_input.py script primitive
    When the ensemble executes and requests user input
    Then the ScriptUserInputHandler should collect the input
    And execution should complete successfully without the conversational executor

  # conversation: key is silently ignored
  @backward-compatibility
  Scenario: Ensemble YAML with conversation keys loads without errors
    Given an ensemble YAML file containing "conversation:" keys on agents
    When the ensemble is loaded by EnsembleLoader
    Then the load should succeed
    And the conversation keys should be ignored by the regular executor

  # Test files removed
  @removal
  Scenario: Conversational executor test files are absent
    When I inspect the test directory for conversational ensemble tests
    Then no test file for ConversationalEnsembleExecutor should exist
    And no test file for ConversationState should exist
    And no BDD test for ADR-005 multi-turn conversations should exist
