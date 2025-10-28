Feature: ADR-008: LLM-Friendly CLI and MCP Design
  As an LLM assistant helping users
  I want clear, discoverable CLI commands
  So I can effectively guide llm-orc usage

  Background:
    Given I am in a directory without llm-orc configuration

  # Phase 1 & 2: Top-Level Init with Script Installation
  Scenario: Discover init command from help
    When I execute "llm-orc --help"
    Then the output should contain "init"
    And the output should contain "Initialize"

  Scenario: Initialize project with scripts by default
    Given a library directory "llm-orchestra-library/scripts/primitives" exists with sample scripts
    When I execute "llm-orc init"
    Then the command should succeed
    And the directory ".llm-orc" should exist
    And the directory ".llm-orc/scripts/primitives" should exist
    And the output should contain "Installed"
    And the output should contain "primitive scripts"

  Scenario: Initialize project without scripts
    When I execute "llm-orc init --no-scripts"
    Then the command should succeed
    And the directory ".llm-orc" should exist
    And the directory ".llm-orc/scripts/primitives" should not contain any scripts

  Scenario: Initialize with custom project name
    When I execute "llm-orc init --project-name my-test-project"
    Then the command should succeed
    And the file ".llm-orc/config.yaml" should contain "my-test-project"

  # Phase 4: Script Installation with Configurable Library Path
  Scenario: Initialize with environment variable library path
    Given the environment variable "LLM_ORC_LIBRARY_PATH" is set to a valid library location
    When I execute "llm-orc init"
    Then the command should succeed
    And scripts should be installed from the custom library location

  Scenario: Initialize with custom library path from environment
    Given a library directory "custom-lib/scripts/primitives" exists with sample scripts
    And the environment variable "LLM_ORC_LIBRARY_PATH" is set to "custom-lib"
    When I execute "llm-orc init"
    Then the command should succeed
    And scripts should be installed from the custom library location


  # Help Text and Examples
  Scenario: Init help shows examples
    When I execute "llm-orc init --help"
    Then the output should contain "Examples:"
    And the output should contain "llm-orc init"
    And the output should contain "--no-scripts"
    And the output should contain "--project-name"

  Scenario: Validate help shows examples
    When I execute "llm-orc validate --help"
    Then the output should contain "validation"
    And the output should mention how to validate ensembles

  # Command Discoverability
  Scenario: List ensembles command is discoverable
    Given I have initialized llm-orc
    When I execute "llm-orc --help"
    Then the output should contain "list-ensembles"

  Scenario: Scripts command is discoverable
    Given I have initialized llm-orc
    When I execute "llm-orc --help"
    Then the output should contain "scripts"

  # Backward Compatibility
  Scenario: Config init still works with deprecation notice
    When I execute "llm-orc config init"
    Then the command should succeed
    And the directory ".llm-orc" should exist
    And the output should contain "Use 'llm-orc init' instead"

  # Library Path Priority Order
  Scenario: Library path resolution follows priority order
    Given no environment variables are set
    And no .env file exists
    And a library directory "llm-orchestra-library/scripts/primitives" exists with sample scripts
    When I execute "llm-orc init"
    Then the command should succeed
    And the directory ".llm-orc/scripts/primitives" should exist
    And the output should contain "Installed"

  Scenario: Graceful fallback when no library found
    Given no environment variables are set
    And no .env file exists
    And the directory "./llm-orchestra-library" does not exist
    When I execute "llm-orc init"
    Then the command should succeed
    And the output should indicate "No library primitives found"
    And no scripts should be installed

  # Ensemble Discovery from Library
  Scenario: Discover ensembles from library submodule
    Given I have initialized llm-orc
    And the library directory "llm-orchestra-library/ensembles/examples/test-ensemble" exists with an ensemble.yaml
    When I execute "llm-orc list-ensembles"
    Then the output should list ensembles from the library
    And the output should contain "examples/test-ensemble"

  Scenario: Invoke ensemble from library by relative path
    Given I have initialized llm-orc
    And the library directory "llm-orchestra-library/ensembles/examples/test-ensemble" exists with a valid ensemble
    When I execute "llm-orc invoke examples/test-ensemble"
    Then the ensemble should execute successfully
    And the output should not require full path specification

  Scenario: Library ensembles have lower priority than local ensembles
    Given I have initialized llm-orc
    And a local ensemble "my-ensemble" exists in ".llm-orc/ensembles"
    And a library ensemble "my-ensemble" exists in "llm-orchestra-library/ensembles"
    When I execute "llm-orc invoke my-ensemble"
    Then the local ensemble should be executed
    And not the library ensemble

  Scenario: List ensembles shows library ensembles in separate section
    Given I have initialized llm-orc
    And local ensembles exist in ".llm-orc/ensembles"
    And library ensembles exist in "llm-orchestra-library/ensembles"
    When I execute "llm-orc list-ensembles"
    Then the output should have a "Local" section
    And the output should have a "Library" section
    And library ensembles should be listed under "Library"

  Scenario: Browse library ensembles by category
    Given I have initialized llm-orc
    And the library has ensembles in "llm-orchestra-library/ensembles/examples"
    When I execute "llm-orc library browse examples"
    Then the output should list ensembles in the examples category
    And the output should include the newly created narrative ensemble
