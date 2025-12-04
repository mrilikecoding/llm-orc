@adr-009 @mcp
Feature: MCP Server Architecture
  As an MCP client (Claude Code, Claude Desktop, etc.)
  I want to interact with llm-orc via the Model Context Protocol
  So that I can discover, execute, and manage ensembles programmatically

  Background:
    Given an MCP server instance is available

  # Resource: List all ensembles
  @resource @ensembles
  Scenario: List all available ensembles via MCP resource
    Given ensembles exist in local, library, and global directories
    When I request the "llm-orc://ensembles" resource
    Then I should receive a list of all ensembles
    And each ensemble should have name, source, and agent_count metadata

  @resource @ensembles
  Scenario: List ensembles when none exist
    Given no ensembles are configured
    When I request the "llm-orc://ensembles" resource
    Then I should receive an empty list

  # Resource: Individual ensemble configuration
  @resource @ensemble-detail
  Scenario: Get ensemble configuration via MCP resource
    Given an ensemble named "code-review" exists
    When I request the "llm-orc://ensemble/code-review" resource
    Then I should receive the complete ensemble configuration
    And the configuration should include agents and their dependencies

  @resource @ensemble-detail
  Scenario: Request non-existent ensemble
    Given no ensemble named "non-existent" exists
    When I request the "llm-orc://ensemble/non-existent" resource
    Then I should receive a resource not found error

  # Resource: Artifacts
  @resource @artifacts
  Scenario: List artifacts for an ensemble
    Given an ensemble named "code-review" has execution artifacts
    When I request the "llm-orc://artifacts/code-review" resource
    Then I should receive a list of artifacts
    And each artifact should have timestamp, status, cost, and duration

  @resource @artifacts
  Scenario: List artifacts when none exist
    Given an ensemble named "new-ensemble" has no execution artifacts
    When I request the "llm-orc://artifacts/new-ensemble" resource
    Then I should receive an empty list

  @resource @artifact-detail
  Scenario: Get individual artifact details
    Given an artifact "code-review/2025-01-15-120000" exists
    When I request the "llm-orc://artifact/code-review/2025-01-15-120000" resource
    Then I should receive the complete artifact data
    And it should include agent results and synthesis

  # Resource: Metrics
  @resource @metrics
  Scenario: Get metrics for an ensemble
    Given an ensemble "code-review" has multiple executions
    When I request the "llm-orc://metrics/code-review" resource
    Then I should receive aggregated metrics
    And metrics should include success_rate, avg_cost, and avg_duration

  # Resource: Model profiles
  @resource @profiles
  Scenario: List available model profiles
    When I request the "llm-orc://profiles" resource
    Then I should receive a list of configured model profiles
    And each profile should have name, provider, and model details

  # Tool: Invoke ensemble
  @tool @invoke
  Scenario: Invoke ensemble via MCP tool
    Given an ensemble named "simple-test" exists
    When I call the "invoke" tool with:
      | ensemble_name | simple-test          |
      | input         | Test input data      |
    Then the ensemble should execute successfully
    And I should receive structured results with agent outputs

  @tool @invoke
  Scenario: Invoke ensemble with JSON output format
    Given an ensemble named "simple-test" exists
    When I call the "invoke" tool with:
      | ensemble_name | simple-test          |
      | input         | Test input data      |
      | output_format | json                 |
    Then I should receive results in JSON format

  @tool @invoke @error
  Scenario: Invoke non-existent ensemble
    When I call the "invoke" tool with:
      | ensemble_name | non-existent         |
      | input         | Test input           |
    Then I should receive a tool error
    And the error should indicate ensemble not found

  # Tool: Validate ensemble
  @tool @validate
  Scenario: Validate ensemble configuration
    Given an ensemble named "code-review" exists with valid configuration
    When I call the "validate_ensemble" tool with:
      | ensemble_name | code-review          |
    Then validation should pass
    And I should receive validation details

  @tool @validate @error
  Scenario: Validate ensemble with invalid configuration
    Given an ensemble named "invalid-ensemble" exists with circular dependencies
    When I call the "validate_ensemble" tool with:
      | ensemble_name | invalid-ensemble     |
    Then validation should fail
    And I should receive error details about the circular dependency

  # Tool: Update ensemble (dry run)
  @tool @update
  Scenario: Dry run ensemble update
    Given an ensemble named "code-review" exists
    When I call the "update_ensemble" tool with:
      | ensemble_name | code-review                        |
      | changes       | {"remove_agents": ["style-check"]} |
      | dry_run       | true                               |
    Then I should receive a preview of changes
    And the ensemble file should not be modified

  @tool @update
  Scenario: Apply ensemble update with backup
    Given an ensemble named "code-review" exists
    When I call the "update_ensemble" tool with:
      | ensemble_name | code-review                        |
      | changes       | {"remove_agents": ["style-check"]} |
      | dry_run       | false                              |
      | backup        | true                               |
    Then the ensemble should be updated
    And a backup file should be created

  # Tool: Analyze execution
  @tool @analyze
  Scenario: Analyze execution artifact
    Given an artifact "code-review/2025-01-15-120000" exists
    When I call the "analyze_execution" tool with:
      | artifact_id | code-review/2025-01-15-120000 |
    Then I should receive execution analysis
    And analysis should include agent effectiveness metrics

  # Streaming execution
  @tool @invoke @streaming
  Scenario: Stream execution progress
    Given an ensemble named "multi-agent-test" exists with multiple agents
    When I call the "invoke" tool with streaming enabled
    Then I should receive progress notifications as agents execute
    And notifications should include agent_start, agent_progress, and agent_complete events

  # Server lifecycle
  @server
  Scenario: MCP server initialization
    When the MCP server starts
    Then it should respond to initialize request
    And capabilities should include tools and resources

  @server
  Scenario: List available tools
    When I request the tools list
    Then I should see "invoke" tool
    And I should see "validate_ensemble" tool
    And I should see "update_ensemble" tool
    And I should see "analyze_execution" tool

  @server
  Scenario: List available resources
    When I request the resources list
    Then I should see "llm-orc://ensembles" resource
    And I should see "llm-orc://profiles" resource

  # CLI integration
  @cli
  Scenario: Start MCP server via CLI
    When I run "llm-orc mcp serve" in background
    Then the server should start on stdio transport
    And it should respond to MCP requests

  @cli
  Scenario: Start MCP server with HTTP transport
    When I run "llm-orc mcp serve --http --port 8080"
    Then the server should start on HTTP transport
    And it should be accessible at "http://localhost:8080"

  # ==========================================================================
  # Phase 2: CRUD Operations (ADR-009 Phase 2)
  # ==========================================================================

  # Tool: Create ensemble
  @tool @create @phase2
  Scenario: Create ensemble from scratch
    Given a local ensembles directory exists
    When I call the "create_ensemble" tool with:
      | name        | my-new-ensemble                    |
      | description | A test ensemble                    |
      | agents      | [{"name": "agent1", "model_profile": "fast"}] |
    Then the ensemble should be created successfully
    And the ensemble file should exist at ".llm-orc/ensembles/my-new-ensemble.yaml"

  @tool @create @phase2
  Scenario: Create ensemble from template
    Given an ensemble named "code-review" exists
    And a local ensembles directory exists
    When I call the "create_ensemble" tool with:
      | name          | my-code-review                     |
      | from_template | code-review                        |
    Then the ensemble should be created successfully
    And the new ensemble should have the same agents as the template

  @tool @create @phase2 @error
  Scenario: Create ensemble with duplicate name fails
    Given an ensemble named "existing-ensemble" exists
    When I call the "create_ensemble" tool with:
      | name        | existing-ensemble                  |
      | description | Duplicate                          |
      | agents      | [{"name": "agent1", "model_profile": "fast"}] |
    Then I should receive a tool error
    And the error should indicate ensemble already exists

  # Tool: Delete ensemble
  @tool @delete @phase2
  Scenario: Delete ensemble with confirmation
    Given an ensemble named "to-delete" exists
    When I call the "delete_ensemble" tool with:
      | ensemble_name | to-delete                          |
      | confirm       | true                               |
    Then the ensemble should be deleted successfully
    And the ensemble file should no longer exist

  @tool @delete @phase2 @error
  Scenario: Delete ensemble without confirmation fails
    Given an ensemble named "protected" exists
    When I call the "delete_ensemble" tool with:
      | ensemble_name | protected                          |
      | confirm       | false                              |
    Then I should receive a tool error
    And the error should indicate confirmation required

  @tool @delete @phase2 @error
  Scenario: Delete non-existent ensemble fails
    When I call the "delete_ensemble" tool with:
      | ensemble_name | non-existent                       |
      | confirm       | true                               |
    Then I should receive a tool error
    And the error should indicate ensemble not found

  # Tool: List scripts
  @tool @scripts @phase2
  Scenario: List all available scripts
    Given scripts exist in the scripts directory
    When I call the "list_scripts" tool
    Then I should receive a list of scripts
    And each script should have name, category, and path

  @tool @scripts @phase2
  Scenario: List scripts by category
    Given scripts exist in multiple categories
    When I call the "list_scripts" tool with:
      | category | transform                          |
    Then I should receive only scripts in the "transform" category

  # Tool: Library browse
  @tool @library @phase2
  Scenario: Browse library ensembles
    Given the library contains ensembles
    When I call the "library_browse" tool with:
      | type | ensembles                          |
    Then I should receive a list of library ensembles
    And each ensemble should have name, description, and path

  @tool @library @phase2
  Scenario: Browse all library items
    Given the library contains ensembles and scripts
    When I call the "library_browse" tool
    Then I should receive both ensembles and scripts

  # Tool: Library copy
  @tool @library @phase2
  Scenario: Copy ensemble from library to local
    Given the library contains an ensemble named "library-ensemble"
    And a local ensembles directory exists
    When I call the "library_copy" tool with:
      | source | ensembles/library-ensemble         |
    Then the ensemble should be copied to local directory
    And the local ensemble file should exist

  @tool @library @phase2 @error
  Scenario: Copy from library fails if already exists
    Given the library contains an ensemble named "library-ensemble"
    And an ensemble named "library-ensemble" exists locally
    When I call the "library_copy" tool with:
      | source    | ensembles/library-ensemble         |
      | overwrite | false                              |
    Then I should receive a tool error
    And the error should indicate file already exists
