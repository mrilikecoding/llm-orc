Feature: ADR-007 Progressive Ensemble Validation Suite
  """
  LLM Development Context:

  Requirements: Implement progressive ensemble validation suite with multi-layer
  validation (structural, schema, behavioral, quantitative, semantic) and LLM-simulated
  user input for non-interactive testing of conversational ensembles.

  Architectural constraints (ADR-007):
  - Flexible validation schema with 5 optional validation layers
  - LLM user simulation architecture for test mode execution
  - Progressive validation categories (primitives → applications)
  - ValidationEvaluator component for running validation criteria
  - ScriptUserInputHandler with test_mode support
  - LLMResponseGenerator for persona-based responses with caching
  - Dual-purpose design (implementation validation + research validation)

  Implementation patterns:
  - ValidationEvaluator evaluates structural, schema, behavioral, quantitative, semantic
  - ScriptUserInputHandler switches between real stdin and LLM simulation
  - LLMResponseGenerator uses qwen3:0.6b with persona prompts and response caching
  - Validation ensembles declare criteria in YAML with test_mode configuration
  - EnsembleExecutor supports test_mode flag for automated validation
  - ValidationResult models with pass/fail status and detailed results

  Critical validations:
  - ValidationEvaluator correctly evaluates all 5 validation layer types
  - LLM simulation prevents stdin blocking in test mode
  - Response caching ensures deterministic test results
  - Persona prompts generate contextually appropriate responses
  - Exception chaining for validation failures (from base framework)
  - Schema validation respects Pydantic foundation (ADR-001)
  """

  As a developer using llm-orc
  I want a validation framework that can test ensembles automatically
  So that I can validate conversational workflows without manual interaction

  Background:
    Given llm-orc is properly configured
    And the validation framework is initialized

  @priority @adr-007 @validation-framework
  Scenario: ValidationEvaluator evaluates all validation layer types
    Given a validation ensemble with all validation layers defined
    When the ValidationEvaluator evaluates the ensemble results
    Then structural validation should check execution properties
    And schema validation should verify JSON contract compliance
    And behavioral validation should evaluate custom Python assertions
    And quantitative validation should measure metrics against thresholds
    And semantic validation should use LLM-as-judge when enabled
    And the ValidationResult should indicate overall pass/fail status

  @adr-007 @validation-framework
  Scenario: Structural validation checks execution properties
    Given a validation ensemble with structural requirements
    When structural validation is performed
    Then required agents should be verified as present in execution
    And max execution time threshold should be enforced
    And min execution time threshold should be enforced
    And agent execution order should be validated
    And violations should raise clear validation errors with chaining

  @adr-007 @validation-framework
  Scenario: Schema validation verifies JSON contracts using Pydantic
    Given a validation ensemble with schema requirements
    When schema validation is performed
    Then each agent output should be validated against declared schema
    And required fields should be verified as present
    And field types should match schema definitions
    And schema violations should provide clear error messages
    And validation should use existing Pydantic infrastructure (ADR-001)

  @adr-007 @validation-framework
  Scenario: Behavioral validation evaluates custom Python assertions
    Given a validation ensemble with behavioral assertions
    When behavioral validation is performed
    Then Python assertion expressions should be evaluated safely
    And execution context should be available to assertions
    And assertion failures should report descriptive error messages
    And assertion evaluation should handle exceptions gracefully

  @adr-007 @validation-framework
  Scenario: Quantitative validation measures metrics with thresholds
    Given a validation ensemble with quantitative metrics
    When quantitative validation is performed
    Then metrics should be calculated from execution results
    And threshold comparisons should evaluate correctly (>, <, >=, <=, ==)
    And metrics without thresholds should be recorded but not validated
    And metric failures should report expected vs actual values

  @adr-007 @validation-framework
  Scenario: Semantic validation uses LLM-as-judge for quality assessment
    Given a validation ensemble with semantic validation enabled
    When semantic validation is performed
    Then validator LLM should receive agent outputs and criteria
    And LLM should evaluate outputs against semantic criteria
    And semantic validation should return pass/fail with justification
    And semantic validation should be optional and skippable

  @priority @adr-007 @llm-simulation
  Scenario: LLM simulation prevents stdin blocking in test mode
    Given an ensemble with user input agents
    And the ensemble is configured with LLM persona simulation
    When the ensemble executes in test mode
    Then user input agents should receive LLM-generated responses
    And no blocking stdin prompts should occur
    And execution should complete without manual intervention

  @adr-007 @llm-simulation
  Scenario: ScriptUserInputHandler switches between real and simulated input
    Given a ScriptUserInputHandler initialized with test mode flag
    When user input is requested from an agent
    Then in test mode the handler should use LLM simulation
    And in interactive mode the handler should use real stdin input
    And missing LLM simulator in test mode should raise clear error

  @adr-007 @llm-simulation
  Scenario: LLMResponseGenerator produces contextual responses using personas
    Given an LLMResponseGenerator with a specific persona
    When response generation is requested with prompt and context
    Then the generator should use persona-specific system prompt
    And the generator should call LLM with prompt and context
    And the generated response should be contextually appropriate
    And conversation history should be maintained for context

  @adr-007 @llm-simulation
  Scenario: Response caching ensures deterministic test results
    Given an LLMResponseGenerator with response cache
    When the same prompt and context are provided multiple times
    Then cached responses should be returned for deterministic results
    And LLM calls should only occur for uncached prompts
    And cache keys should be deterministic based on prompt and context

  @adr-007 @llm-simulation
  Scenario: Persona prompts guide LLM simulation behavior
    Given an LLMResponseGenerator with helpful_user persona
    When responses are generated for user input prompts
    Then responses should be realistic and contextually appropriate
    And responses should follow persona characteristics (helpful vs critical)
    And default personas should include helpful_user, critical_reviewer, domain_expert

  @adr-007 @progressive-validation
  Scenario: Validation ensembles declare only needed validation layers
    Given validation ensemble configurations for different categories
    When validation ensembles are loaded and parsed
    Then ensembles should be able to declare any subset of validation layers
    And ensembles without certain layers should skip those validations
    And validation layer parsing should handle missing sections gracefully

  @adr-007 @progressive-validation
  Scenario: Category 1 primitive validation tests individual script agents
    Given a primitive validation ensemble with single script agent
    When primitive validation is executed
    Then script output should be validated against declared schema
    And behavioral assertions should validate agent-specific properties
    And validation should complete for isolated script agent execution

  @adr-007 @progressive-validation
  Scenario: Category 2 integration validation tests agent composition
    Given an integration validation ensemble with multiple dependent agents
    When integration validation is executed
    Then structural validation should verify execution order
    And schema validation should verify data flow between agents
    And behavioral assertions should validate composition properties

  @adr-007 @progressive-validation
  Scenario: Category 3 conversational validation tests multi-turn workflows
    Given a conversational validation ensemble with user input agents
    And the ensemble is configured with LLM simulation
    When conversational validation is executed in test mode
    Then user input agents should receive LLM-simulated responses
    And conversation flow should be validated against behavioral assertions
    And multi-turn execution should complete without blocking

  @adr-007 @progressive-validation
  Scenario: Category 4 research validation captures quantitative metrics
    Given a research validation ensemble with quantitative metrics
    When research validation is executed
    Then network metrics should be calculated (clustering, path length)
    And consensus metrics should be measured if applicable
    And metrics without thresholds should be recorded for analysis
    And statistical analysis should be optional for multi-run experiments

  @adr-007 @progressive-validation
  Scenario: Category 5 application validation tests end-to-end workflows
    Given an application validation ensemble with full workflow
    And the ensemble includes script agents, LLM agents, and user input
    When application validation is executed in test mode
    Then all validation layers should be evaluated as declared
    And LLM simulation should handle user input requirements
    And end-to-end workflow should complete successfully

  @adr-007 @yaml-configuration
  Scenario: Validation YAML schema supports flexible validation layer declaration
    Given an ensemble YAML with validation section
    When the validation configuration is parsed
    Then structural validation config should parse with required_agents and timing
    And schema validation config should parse with agent-schema mappings
    And behavioral validation config should parse assertion expressions
    And quantitative validation config should parse metrics and thresholds
    And semantic validation config should parse criteria and validator model

  @adr-007 @yaml-configuration
  Scenario: Test mode configuration specifies LLM simulation for agents
    Given an ensemble YAML with test_mode section
    When the test mode configuration is parsed
    Then test mode enabled flag should be parsed correctly
    And llm_simulation configuration should map agents to personas
    And cached_responses should be parsed for deterministic testing
    And persona and model overrides should be supported per agent

  @adr-007 @architectural-compliance
  Scenario: Validation framework respects Pydantic foundation (ADR-001)
    Given the existing Pydantic schema infrastructure from ADR-001
    When validation framework validates schema compliance
    Then schema validation should use existing ScriptAgentInput/Output base classes
    And validation should leverage existing Pydantic validation infrastructure
    And backward compatibility should be maintained with current patterns

  @adr-007 @architectural-compliance
  Scenario: Validation framework integrates with script contracts (ADR-003)
    Given the testable script contract system from ADR-003
    When validation ensembles validate script agent outputs
    Then script contracts should be validated for compliance
    And test cases from contracts should be executable in validation mode
    And schema compatibility from contracts should inform validation

  @adr-007 @architectural-compliance
  Scenario: Validation framework supports multi-turn conversations (ADR-005)
    Given the multi-turn conversation architecture from ADR-005
    When conversational validation ensembles execute
    Then conversation state should be maintained during validation
    And turn order should be validated in behavioral assertions
    And LLM simulation should support multi-turn conversation context

  @adr-007 @architectural-compliance
  Scenario: Validation framework works with library primitives (ADR-006)
    Given the library-based primitives architecture from ADR-006
    When validation ensembles use library primitives
    Then primitive scripts should be discoverable and executable
    And category-specific schemas should be validated
    And library structure should support validation ensemble organization

  @adr-007 @error-handling
  Scenario: Validation failures use proper exception chaining
    Given a validation ensemble with failing validation criteria
    When validation is performed and criteria fail
    Then validation errors should be chained with original exception context
    And error messages should include validation-specific failure details
    And error context should guide developers to fix validation issues

  @adr-007 @error-handling
  Scenario: LLM simulation failures provide clear debugging context
    Given an ensemble in test mode with LLM simulation configured
    When LLM simulation fails due to model unavailability or errors
    Then simulation errors should be caught and chained properly
    And error messages should indicate which agent failed simulation
    And error context should suggest fallback strategies

  @adr-007 @performance
  Scenario: Response caching optimizes repeated validation runs
    Given multiple validation runs with identical prompts and context
    When validation executes repeatedly
    Then cached responses should eliminate redundant LLM calls
    And validation performance should improve with cache hits
    And cache should be persisted across validation sessions

  @adr-007 @security
  Scenario: Behavioral assertion evaluation prevents code injection
    Given a validation ensemble with behavioral assertions
    When assertions are evaluated using Python expressions
    Then assertion evaluation should occur in restricted context
    And dangerous operations should be prevented in assertions
    And assertion syntax errors should be caught and reported safely

  @adr-007 @dual-purpose
  Scenario: Validation framework supports research experiments
    Given a research validation ensemble with statistical analysis
    When the ensemble is executed multiple times for experiments
    Then metrics should be recorded across all runs
    And statistical analysis should calculate correlations and ANOVA
    And results should be exportable to CSV for external analysis
    And research mode should preserve all quantitative measurements

  @adr-007 @integration
  Scenario: CLI supports validation ensemble execution in test mode
    Given validation ensembles configured for automated testing
    When validation ensembles are invoked via CLI
    Then --mode test flag should enable test mode execution
    And --verbose flag should provide detailed validation output
    And validation results should be reported with pass/fail status
    And exit codes should indicate validation success or failure

  @adr-007 @integration
  Scenario: ValidationResult model captures comprehensive validation outcomes
    Given completed validation of an ensemble
    When ValidationResult is constructed from validation outcomes
    Then ensemble name and timestamp should be recorded
    And overall passed status should reflect all validation layers
    And individual validation layer results should be preserved
    And result serialization should support reporting and analysis
