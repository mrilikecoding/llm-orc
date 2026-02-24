@adr-010 @web-ui
Feature: Local Web UI for Ensemble Management
  As a user managing ensembles locally
  I want a web interface accessible on localhost
  So that I can browse, execute, and compare ensembles without external tools

  Background:
    Given the llm-orc web server is running on localhost port 8765

  # CLI command
  @cli
  Scenario: Start web server with default settings
    Given I am in a directory with llm-orc initialized
    When I execute "llm-orc web"
    Then the web server should start on 127.0.0.1 port 8765
    And the output should indicate the server URL

  @cli
  Scenario: Start web server on a custom port
    Given I am in a directory with llm-orc initialized
    When I execute "llm-orc web --port 3000"
    Then the web server should start on port 3000

  # REST API — ensembles
  @api @ensembles
  Scenario: List ensembles via REST API
    Given ensembles exist in local, library, and global directories
    When I send GET to "/api/ensembles"
    Then the response should contain ensembles grouped by source
    And each ensemble entry should include name, source, and agent_count

  @api @ensembles
  Scenario: Get single ensemble configuration via REST API
    Given an ensemble named "code-review" exists
    When I send GET to "/api/ensembles/code-review"
    Then the response should contain the full ensemble configuration
    And the configuration should include the agents list and their dependencies

  @api @ensembles
  Scenario: Execute ensemble via REST API
    Given an ensemble named "simple-test" exists
    When I send POST to "/api/ensembles/simple-test/execute" with input "test input"
    Then the response should indicate execution started
    And the result should include agent outputs

  # REST API — artifacts
  @api @artifacts
  Scenario: List artifacts for an ensemble via REST API
    Given an ensemble named "code-review" has execution artifacts
    When I send GET to "/api/artifacts/code-review"
    Then the response should list artifacts with timestamp, status, cost, and duration

  @api @artifacts
  Scenario: Get individual artifact detail via REST API
    Given an artifact "code-review/2025-01-15-120000" exists
    When I send GET to "/api/artifacts/code-review/2025-01-15-120000"
    Then the response should include agent results and synthesis output

  # WebSocket streaming
  @websocket @streaming
  Scenario: Stream ensemble execution via WebSocket
    Given an ensemble named "multi-agent-test" exists with multiple agents
    When a client connects to "/ws/execute" and sends an execution request
    Then the server should emit "agent_start" events as each agent begins
    And the server should emit "agent_progress" events with token output
    And the server should emit "agent_complete" events as each agent finishes
    And the server should emit an "execution_complete" event with the artifact id

  # Security
  @security
  Scenario: Server binds to localhost by default
    When the web server starts without explicit host configuration
    Then it should bind to 127.0.0.1
    And it should not be accessible from external network interfaces

  @security
  Scenario: Warning is shown when binding to all interfaces
    When I execute "llm-orc web --host 0.0.0.0"
    Then the output should display a warning about network exposure

  # Shared service layer
  @service-layer
  Scenario: Web API and MCP server produce consistent data
    Given an ensemble named "code-review" exists
    When I request ensemble data from the web API at "/api/ensembles/code-review"
    And I request the same ensemble via the MCP resource "llm-orc://ensemble/code-review"
    Then both responses should reflect the same ensemble configuration

  # Metrics
  @api @metrics
  Scenario: Retrieve aggregated metrics via REST API
    Given ensembles with multiple execution artifacts exist
    When I send GET to "/api/metrics/code-review"
    Then the response should include success_rate, avg_cost, and avg_duration
