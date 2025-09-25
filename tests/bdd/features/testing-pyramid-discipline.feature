Feature: Testing Pyramid Discipline & TDD Architectural Compliance (ADR-003)
  """
  LLM Development Context:

  This feature establishes testing pyramid discipline as an architectural requirement,
  ensuring TDD Red→Green→Refactor cycles maintain proper test structure ratios and
  architectural compliance throughout LLM-assisted development.

  Current Testing Pyramid Crisis:
  - Unit: 66% (target: 70%+) - Missing 4%
  - Integration: 5% (target: 20%+) - Missing 15%!
  - BDD: 28% (target: 10%+) - Excessive by 18%
  - Total: 38 BDD scenarios vs only 7 integration tests (5.4:1 ratio, should be ~1:2)

  Architectural Constraints:
  - ADR-003: All implementations must have testable contracts with proper unit test backing
  - Testing pyramid structure (70/20/10) enforces architectural discipline
  - Each BDD scenario requires 3:1 unit test support ratio for proper foundations

  Coding Standards Requirements:
  - Type annotations: All test functions must include return type annotations
  - Exception chaining: raise TestError("Failed validation") from original_error
  - Async patterns: Integration tests must use asyncio.gather() for concurrent validation
  - Coverage thresholds: 95% unit coverage required before BDD scenario approval
  """

  Background:
    Given an llm-orc project with testing pyramid requirements
    And ADR-003 testable contract system is active
    And TDD Red→Green→Refactor discipline is required

  @pyramid-structure
  Scenario: Testing pyramid maintains proper 70/20/10 ratio structure
    Given the current testing pyramid state
    And unit tests at 66% of total test count
    And integration tests at 5% of total test count
    And BDD scenarios at 28% of total test count
    When pyramid ratio validation is performed
    Then unit test percentage should be at least 70%
    And integration test percentage should be at least 20%
    And BDD scenario percentage should be at most 10%
    And the total test structure should follow pyramid shape
    And architectural compliance should be maintained
    And ratio violations should trigger corrective actions

  @unit-test-foundation
  Scenario: Every BDD scenario has supporting unit test foundation at 3:1 ratio
    Given a BDD scenario "Script agent executes with JSON input/output contract"
    And the scenario requires implementation of ScriptAgent.execute() method
    And ADR-003 testable contracts require unit test backing
    When unit test foundation validation is performed
    Then there should be at least 3 unit tests per BDD scenario
    And unit tests should cover ScriptAgent.execute() method
    And unit tests should cover JSON schema validation
    And unit tests should cover error handling paths
    And unit tests should test all edge cases and boundaries
    And each unit test should have proper type annotations
    And exception chaining should be validated in unit tests

  @integration-test-bridge
  Scenario: Integration tests bridge unit tests and BDD scenarios properly
    Given 217 unit tests exist in the test suite
    And 38 BDD scenarios require behavioral validation
    And only 7 integration tests currently exist
    When integration test coverage analysis is performed
    Then there should be at least 27 integration tests (target: 20% of pyramid)
    And integration tests should cover cross-component interactions
    And integration tests should validate real API integrations
    And integration tests should test ensemble execution workflows
    And integration tests should bridge unit test isolation with BDD behavioral validation
    And integration tests should use real providers with test credentials
    And async concurrent execution should be tested with real latency

  @missing-unit-detection
  Scenario: Source files without unit tests are automatically detected and flagged
    Given 26 source files currently lack corresponding unit tests
    And ADR-003 requires testable contracts for all implementations
    And coding standards mandate 95% test coverage
    When missing unit test detection is performed
    Then all source files with >10 lines of implementation should have unit tests
    And missing unit test files should be listed with expected paths
    And detection should exclude __init__.py files from requirements
    And missing tests should be prioritized by file complexity
    And automated generation suggestions should be provided
    And TDD Red phase requirements should be enforced

  @bdd-unit-relationship
  Scenario: BDD scenarios maintain proper relationship with unit test structure
    Given BDD feature file "tests/bdd/features/issue-24-script-agents.feature"
    And the feature implements Issue #24 for script agent functionality
    And BDD scenarios require unit test backing
    When BDD-Unit relationship validation is performed
    Then corresponding unit test file "tests/test_issue_24_units.py" should exist
    And unit test file should contain at least 3 tests per BDD scenario
    And unit tests should validate the underlying implementation components
    And unit tests should test the same functional boundaries as BDD scenarios
    And proper issue number cross-referencing should be maintained
    And missing unit test relationships should trigger warnings

  @tdd-cycle-compliance
  Scenario: TDD Red→Green→Refactor cycle maintains pyramid discipline
    Given a new feature implementation requiring BDD scenario
    And behavioral contracts are required before implementation
    And TDD discipline requires Red→Green→Refactor progression
    When implementing the feature following TDD cycle
    Then Red phase should write failing unit tests first (not BDD scenarios)
    And BDD scenario should define behavioral contract for implementation
    And Green phase should implement minimum code to pass unit tests
    And unit tests should achieve 95% coverage before BDD scenario execution
    And Refactor phase should maintain test passing state
    And pyramid ratios should be maintained throughout TDD cycle
    And structural changes should be separated from behavioral changes

  @architectural-drift-prevention
  Scenario: Testing pyramid prevents architectural drift through enforcement
    Given LLM-assisted development is implementing new features
    And ADR-003 establishes architectural constraints
    And testing pyramid ratios serve as architectural guardrails
    When LLM implementation attempts bypass unit test requirements
    Then pyramid validation should reject the implementation
    And missing unit test foundation should be flagged immediately
    And BDD scenarios should not pass without proper unit test backing
    And architectural compliance should be enforced through test structure
    And LLM guidance should be provided for corrective actions
    And implementation should be blocked until pyramid compliance is achieved

  @performance-regression-detection
  Scenario: Testing pyramid structure enables performance regression detection
    Given proper unit, integration, and BDD test layers
    And performance baselines established at each layer
    And TDD cycle includes performance validation
    When testing pyramid validation is performed
    Then unit tests should execute in <100ms total time
    And integration tests should complete in <30 seconds total time
    And BDD scenarios should finish in <2 minutes total time
    And performance regression should be detected at appropriate layer
    And layer-specific performance thresholds should be enforced
    And performance issues should be isolated to correct pyramid level

  @coverage-threshold-enforcement
  Scenario: Unit test coverage thresholds are enforced before BDD approval
    Given unit test coverage currently at 66% (below 70% threshold)
    And BDD scenarios require 95% unit coverage foundation
    And ADR-003 mandates testable contracts with high coverage
    When coverage validation is performed before BDD scenario execution
    Then unit test coverage must reach 95% before BDD scenarios pass
    And coverage gaps should be identified by source file
    And missing unit tests should be generated or flagged
    And BDD scenario execution should be blocked until coverage threshold met
    And coverage enforcement should maintain architectural discipline
    And TDD Red phase should address coverage gaps first

  @pyramid-ratio-alerts
  Scenario: Pyramid ratio violations trigger immediate corrective guidance
    Given current pyramid ratios are inverted (66/5/28 vs target 70/20/10)
    And proper pyramid structure guides development
    And testing discipline affects architectural quality
    When pyramid ratio analysis detects violations
    Then immediate alerts should be generated for ratio violations
    And specific corrective actions should be recommended
    And missing integration tests should be prioritized (need 15% more)
    And excessive BDD scenarios should be backed by unit tests
    And automated test generation should be triggered for missing layers
    And pyramid correction should be tracked until compliance achieved

  @commit-gate-integration
  Scenario: Testing pyramid compliance blocks commits until resolved
    Given testing pyramid validation is integrated with commit gates
    And ADR-003 establishes pyramid as architectural requirement
    And TDD discipline requires proper test structure before commits
    When attempting to commit with pyramid violations
    Then commit should be blocked until pyramid compliance is achieved
    And specific violation details should be provided to developer
    And corrective actions should be suggested (--fix flag support)
    And automated test generation should be offered
    And commit gate should enforce TDD discipline at architectural level
    And pyramid structure should be validated as pre-commit requirement

  @community-contribution-validation
  Scenario: Community contributions must maintain pyramid discipline
    Given external contributors submitting script agents or features
    And ADR-003 requires testable contracts for all components
    And pyramid discipline must be maintained across all contributions
    When validating community pull requests
    Then submitted code must include proper unit test foundation
    And integration tests should be provided for cross-component features
    And BDD scenarios should only be added with proper unit test backing
    And contribution guidelines should enforce pyramid structure
    And automated validation should check pyramid compliance
    And PR approval should require pyramid discipline compliance