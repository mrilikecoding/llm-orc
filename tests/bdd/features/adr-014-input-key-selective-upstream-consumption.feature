@adr-014 @input-key @routing
Feature: Input Key for Selective Upstream Consumption
  As a developer building routing workflows
  I want an agent to select a specific key from its upstream dependency's output
  So that I can route structured data to different downstream agents without modifying producers

  Background:
    Given llm-orc is properly configured
    And the Pydantic agent config models are in use (ADR-012)

  # Config
  @config
  Scenario: Agent config with input_key field is valid on all agent types
    Given an agent config with name "pdf-processor", depends_on "classify", and input_key "pdfs"
    When the config is parsed by EnsembleLoader
    Then the agent config should be created with input_key set to "pdfs"
    And input_key should be accepted on LlmAgentConfig, ScriptAgentConfig, and EnsembleAgentConfig

  @config
  Scenario: input_key defaults to None when not specified
    Given an agent config with name "analyzer" and depends_on "upstream" but no input_key
    When the config is parsed by EnsembleLoader
    Then the agent config should have input_key set to None

  # Key selection behavior
  @key-selection
  Scenario: DependencyResolver selects the specified key from upstream dict output
    Given an upstream agent "classify" that produces output {"pdfs": ["a.pdf", "b.pdf"], "audio": ["c.mp3"]}
    And a downstream agent "pdf-processor" with depends_on "classify" and input_key "pdfs"
    When the downstream agent resolves its input
    Then the agent should receive ["a.pdf", "b.pdf"] as its input
    And the "audio" key should not be included

  @key-selection
  Scenario: Agent without input_key receives the full upstream output
    Given an upstream agent "classify" that produces output {"pdfs": ["a.pdf"], "audio": ["c.mp3"]}
    And a downstream agent "reporter" with depends_on "classify" and no input_key
    When the downstream agent resolves its input
    Then the agent should receive the full dict {"pdfs": ["a.pdf"], "audio": ["c.mp3"]}

  # Error cases
  @error-handling
  Scenario: Missing key in upstream output produces a runtime error
    Given an upstream agent "classify" that produces output {"pdfs": ["a.pdf"]}
    And a downstream agent "audio-processor" with depends_on "classify" and input_key "audio"
    When the downstream agent resolves its input at execution time
    Then the agent should receive an error status
    And the error should indicate that key "audio" was not found in upstream output

  @error-handling
  Scenario: Non-dict upstream output with input_key set produces a runtime error
    Given an upstream agent "summarizer" that produces a plain string output
    And a downstream agent "router" with depends_on "summarizer" and input_key "items"
    When the downstream agent resolves its input at execution time
    Then the agent should receive an error status
    And the error should indicate that input_key requires structured dict output from the upstream

  # Composition with fan-out
  @fan-out-composition
  Scenario: input_key composes with fan_out for the routing pattern
    Given an upstream agent "classify" that produces output {"pdfs": ["a.pdf", "b.pdf"], "audio": ["c.mp3"]}
    And a downstream agent "pdf-processor" with depends_on "classify", input_key "pdfs", and fan_out true
    When the ensemble executes
    Then "pdf-processor" should expand into two instances — one for "a.pdf" and one for "b.pdf"
    And neither instance should see the "audio" key

  # Applicability across agent types
  @applicability
  Scenario: input_key works on an ensemble agent referencing a child ensemble
    Given an upstream script agent "classify" producing {"pdfs": ["a.pdf", "b.pdf"]}
    And a downstream ensemble agent "pdf-pipeline" with depends_on "classify" and input_key "pdfs"
    When the ensemble executes
    Then the "pdf-pipeline" child ensemble should receive ["a.pdf", "b.pdf"] as its input

  # First dependency only
  @dependency-selection
  Scenario: input_key selects from the first entry in depends_on
    Given an agent "consumer" with depends_on ["primary", "secondary"] and input_key "result"
    And "primary" produces output {"result": "primary-value"}
    And "secondary" produces output {"result": "secondary-value"}
    When "consumer" resolves its input
    Then "consumer" should receive "primary-value"
    And the "secondary" agent output should not be used for key selection
