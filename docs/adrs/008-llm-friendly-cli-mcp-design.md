# ADR-008: LLM-Friendly CLI and MCP Design

**Status**: Proposed
**Date**: 2025-10-09
**Authors**: Nathan Green
**Related ADRs**: [ADR-001](001-pydantic-script-agent-interfaces.md), [ADR-007](007-progressive-ensemble-validation-suite.md)

## BDD Mapping Hints

This ADR defines design principles for CLI and MCP interfaces optimized for consumption by LLMs like Claude, enabling effective AI-driven development workflows.

**Behavioral capabilities to validate**:
- LLM agents can discover available commands through help text
- Common operations accessible at top level without subcommand navigation
- Help text includes concrete examples that LLMs can pattern-match
- Error messages suggest correct commands for recovery
- MCP tools expose llm-orc capabilities with clear schemas

**Test boundary**: BDD scenarios validate CLI output structure, help text quality, and MCP tool schemas rather than implementation details.

**Gherkin examples**:
```gherkin
Scenario: Discover initialization command
  Given an LLM assistant helping a user set up llm-orc
  When the LLM reads `llm-orc --help` output
  Then the init command should appear in the common commands section
  And help text should include example usage

Scenario: Self-correct from wrong command
  Given an LLM attempts `llm-orc run ensemble.yaml`
  When the command fails with error
  Then error message should suggest `llm-orc invoke ensemble.yaml`
  And suggest `llm-orc --help` for full command list
```

## Context

### Problem: Hidden Functionality

Current CLI structure buries important commands in submenus:
```bash
# Expected (doesn't work):
llm-orc init

# Actual (hidden in subcommand):
llm-orc config init
```

This creates discoverability problems:
1. **For humans**: First-time users can't find setup instructions
2. **For LLMs**: AI assistants can't guide users effectively without discovering commands
3. **For MCP**: Tools aren't structured to expose capabilities clearly

### The LLM-as-User Pattern

With Claude Code and MCP integration, llm-orc has two user types:
1. **Human developers**: Traditional CLI users
2. **LLM agents**: AI assistants driving development workflows

LLMs consume CLI differently than humans:
- **Pattern matching**: Learn from examples in help text
- **Self-correction**: Use error messages to retry with correct syntax
- **Discovery**: Navigate command structure through `--help` output
- **Context window**: Limited memory, need clear, concise guidance

### Current Issues

**1. Command Hierarchy Complexity**
```bash
llm-orc --help
# Shows 20+ commands alphabetically
# validate, scripts, artifacts buried at bottom
# No visual grouping or priority
```

**2. Alias Clutter**
```
Commands:
  a          Authentication management commands.
  auth       Authentication management commands.
  c          Configuration management commands.
  config     Configuration management commands.
```
Short aliases duplicate information, adding noise.

**3. Missing Examples**
```bash
llm-orc validate --help
Usage: llm-orc validate [OPTIONS] COMMAND [ARGS]...

  Validation commands for testing ensembles.

Options:
  --help  Show this message and exit.

Commands:
  all       Validate all validation ensembles.
  category  Validate ensembles by category.
  run       Validate a single ensemble.
```
No examples → LLMs don't know how to construct commands.

**4. Init Not Top-Level**
Project initialization hidden in `config` submenu contradicts Issue #24 vision of primitive scripts "initialized by default."

## Decision

We will redesign CLI and MCP interfaces using **LLM-Friendly Design Principles**:

### Design Principles

#### 1. Common Operations at Top Level

Frequently used commands accessible without subcommand navigation:
- `init` - Project initialization
- `invoke` - Ensemble execution
- `validate` - Ensemble validation
- `list-ensembles` - Ensemble discovery

**Rationale**: Reduces cognitive load and command depth for both humans and LLMs.

#### 2. Help Text with Examples

Every command includes concrete examples:
```bash
llm-orc init --help
Usage: llm-orc init [OPTIONS]

  Initialize llm-orc project with scripts and examples.

  Creates .llm-orc/ directory with:
    - Primitive scripts from library (file-ops, data-transform, etc.)
    - Example ensembles demonstrating patterns
    - Configuration templates

Examples:
  # Initialize with all defaults
  llm-orc init

  # Initialize without primitive scripts
  llm-orc init --no-scripts

  # Specify project name
  llm-orc init --project-name my-ensemble-project

Ready! After init, try:
  llm-orc scripts list          # See installed primitives
  llm-orc list-ensembles        # See example ensembles
  llm-orc invoke hello-world    # Run example
```

**Rationale**: LLMs learn command patterns from examples, enabling accurate command construction.

#### 3. Grouped Help Output

Organize commands by purpose, not alphabetically:
```bash
llm-orc --help

Usage: llm-orc [OPTIONS] COMMAND [ARGS]...

  LLM Orchestra - Multi-agent LLM communication system

Common Commands:
  init            Initialize llm-orc project
  invoke          Execute ensemble of agents
  validate        Validate ensemble execution
  list-ensembles  List available ensembles

Management:
  config          Configuration management
  auth            LLM provider authentication
  scripts         Primitive script management
  library         Browse library ensembles
  artifacts       View execution artifacts

Advanced:
  serve           Serve ensemble as MCP server
  help            Detailed help for commands
```

**Rationale**: Visual grouping helps both humans and LLMs navigate command space efficiently.

#### 4. Error Messages as Recovery Paths

Errors suggest correct commands:
```bash
$ llm-orc run ensemble.yaml
Error: No such command 'run'

Did you mean one of these?
  llm-orc invoke ensemble.yaml        # Execute ensemble
  llm-orc validate run ensemble.yaml  # Validate ensemble

See 'llm-orc --help' for all commands
```

**Rationale**: LLMs use error messages to self-correct and retry.

#### 5. Scripts by Default

`llm-orc init` installs primitive scripts by default:
```bash
llm-orc init

Initializing llm-orc project...
✓ Created .llm-orc/ directory
✓ Installed 6 primitive scripts:
  - file-ops: read_file.py, write_file.py
  - data-transform: json_extract.py
  - control-flow: replicate_n_times.py
  - user-interaction: get_user_input.py, confirm_action.py
✓ Created example ensembles
✓ Created config.yaml

Ready! Try:
  llm-orc scripts list
  llm-orc invoke hello-world
```

**Rationale**: Aligns with Issue #24 vision - users get composable primitives immediately.

### MCP Interface Design

MCP tools expose llm-orc capabilities with clear schemas:

```json
{
  "name": "llm_orc_invoke",
  "description": "Execute an ensemble of LLM and script agents",
  "inputSchema": {
    "type": "object",
    "properties": {
      "ensemble_name": {
        "type": "string",
        "description": "Name or path of ensemble YAML file"
      },
      "input_data": {
        "type": "string",
        "description": "Input data for ensemble execution"
      }
    },
    "required": ["ensemble_name"]
  }
}
```

MCP tool categories:
1. **Ensemble Operations**: invoke, validate, list-ensembles
2. **Script Management**: scripts-list, scripts-show, scripts-test
3. **Library Browsing**: library-browse, library-copy
4. **Artifact Management**: artifacts-list, artifacts-show

**Rationale**: Clear schemas enable LLMs to construct valid tool calls.

## Implementation

### Phase 1: Top-Level Commands

**Move `init` to top level:**
```python
# Before: @config.command()
# After:  @cli.command()
@cli.command()
@click.option("--no-scripts", is_flag=True, help="Skip installing primitive scripts")
@click.option("--project-name", default=None, help="Project name")
def init(no_scripts: bool, project_name: str | None) -> None:
    """Initialize llm-orc project with scripts and examples."""
    with_scripts = not no_scripts
    init_local_config(project_name, with_scripts)
```

**Keep backward compatibility:**
```python
# Keep config.init as alias for transition period
@config.command()
def init_alias(...):
    """(Deprecated) Use 'llm-orc init' instead."""
    click.echo("Note: Use 'llm-orc init' instead of 'llm-orc config init'")
    init(...)
```

### Phase 2: Help Text Enhancement

**Add examples section to all commands:**
```python
@cli.command()
def validate() -> None:
    """Validate ensemble execution with assertions.

    Runs validation ensembles that include test assertions
    to verify ensemble behavior.

    \b
    Examples:
      # Validate single ensemble
      llm-orc validate run validation/primitive/validate-file-read

      # Validate all ensembles in category
      llm-orc validate category --category primitive

      # Validate all ensembles
      llm-orc validate all --verbose

    See also: llm-orc list-ensembles --category validation
    """
```

Note: `\b` prevents Click from rewrapping example blocks.

### Phase 3: Grouped Help Output

**Use Click 8.0+ group sections:**
```python
class NaturalOrderGroup(click.Group):
    """Command group that shows commands in definition order with sections."""

    def list_commands(self, ctx):
        return list(self.commands)

    def format_help(self, ctx, formatter):
        # Custom help formatting with command sections
        self.format_usage(ctx, formatter)
        self.format_help_text(ctx, formatter)

        # Common commands section
        common_cmds = ['init', 'invoke', 'validate', 'list-ensembles']
        self.format_commands(ctx, formatter, common_cmds, "Common Commands")

        # Management commands section
        mgmt_cmds = ['config', 'auth', 'scripts', 'library', 'artifacts']
        self.format_commands(ctx, formatter, mgmt_cmds, "Management")

        # Advanced commands section
        advanced_cmds = ['serve', 'help', 'completion']
        self.format_commands(ctx, formatter, advanced_cmds, "Advanced")

@click.group(cls=NaturalOrderGroup)
def cli():
    """LLM Orchestra - Multi-agent LLM communication system."""
```

### Phase 4: Script Installation

**Implement library script copying:**
```python
def copy_library_primitives_to_local():
    """Copy primitive scripts from library to .llm-orc/scripts/."""
    library_scripts = Path(__file__).parent.parent / "llm-orchestra-library" / "scripts" / "primitives"
    local_scripts = Path(".llm-orc/scripts/primitives")

    if not library_scripts.exists():
        echo_warning("Library scripts not found. Using git submodule?")
        return

    # Copy each category
    for category_dir in library_scripts.iterdir():
        if category_dir.is_dir():
            dest = local_scripts / category_dir.name
            shutil.copytree(category_dir, dest, dirs_exist_ok=True)

            # Make scripts executable
            for script in dest.glob("**/*.py"):
                script.chmod(script.stat().st_mode | 0o111)

    script_count = len(list(local_scripts.glob("**/*.py")))
    echo_success(f"Installed {script_count} primitive scripts")
```

### Phase 5: MCP Tool Definitions

**Generate MCP tools from CLI:**
```python
class MCPToolGenerator:
    """Generate MCP tool definitions from CLI commands."""

    def generate_tools(self) -> list[dict]:
        """Generate MCP tools for common llm-orc operations."""
        return [
            {
                "name": "llm_orc_invoke",
                "description": "Execute an ensemble of LLM and script agents",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "ensemble_name": {"type": "string"},
                        "input_data": {"type": "string"}
                    },
                    "required": ["ensemble_name"]
                }
            },
            {
                "name": "llm_orc_validate",
                "description": "Validate ensemble with test assertions",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "ensemble_name": {"type": "string"},
                        "verbose": {"type": "boolean"}
                    },
                    "required": ["ensemble_name"]
                }
            },
            # ... more tools
        ]
```

## Consequences

### Positive

1. **Improved Discoverability**
   - Common operations obvious in top-level help
   - LLMs can guide users without deep command knowledge
   - Examples enable pattern matching

2. **Better First-Run Experience**
   - `llm-orc init` immediately installs working primitives
   - Users can compose ensembles without writing scripts
   - Example ensembles demonstrate patterns

3. **LLM-Driven Development**
   - Claude Code can effectively operate llm-orc
   - MCP integration exposes clear tool schemas
   - Error recovery through suggested commands

4. **Consistency Across Interfaces**
   - CLI, MCP, and documentation aligned
   - Same concepts, different interfaces
   - Learning transfers between modalities

### Negative

1. **Breaking Changes**
   - Moving `init` to top level breaks `llm-orc config init`
   - Mitigation: Keep alias with deprecation warning

2. **Help Text Verbosity**
   - Examples make help longer
   - Mitigation: Use `\b` to prevent wrapping, keep examples concise

3. **Maintenance Burden**
   - Examples in help text need updates
   - Mitigation: Include in test coverage

### Neutral

1. **Script Installation Size**
   - Adding ~10KB of primitive scripts
   - Trade-off: Immediate value vs minimal size increase
   - Decision: Value outweighs cost

## Success Criteria

1. **LLM Can Initialize Project**
   - Claude Code can read `--help` and run `init`
   - Scripts installed by default
   - Examples demonstrate usage

2. **Command Discovery**
   - All commands discoverable through `--help`
   - Common operations at top level
   - Groups clearly labeled

3. **Error Recovery**
   - Wrong commands suggest corrections
   - Error messages include examples
   - LLM can self-correct from errors

4. **MCP Integration**
   - All major operations exposed as tools
   - Schemas validate correctly
   - Tool descriptions enable discovery

## Implementation Plan

**Phase 1: Top-Level Commands** (1 day)
- Move `init` to top level
- Add `--no-scripts` flag
- Keep backward compatibility alias

**Phase 2: Script Installation** (1 day)
- Implement `copy_library_primitives_to_local()`
- Add script executable permissions
- Test across platforms

**Phase 3: Help Enhancement** (1 day)
- Add examples to all commands
- Use `\b` to preserve formatting
- Update documentation

**Phase 4: Grouped Help** (1 day)
- Implement `NaturalOrderGroup`
- Define command sections
- Test help output

**Phase 5: MCP Tools** (1 day)
- Generate tool schemas from CLI
- Test with Claude Desktop
- Document MCP integration

**Phase 6: Testing** (1 day)
- BDD scenarios for CLI output
- Test error messages
- Validate examples

## Related Documentation

- [Issue #24: Script Agent Infrastructure](https://github.com/your-org/llm-orc/issues/24)
- [ADR-001: Pydantic Script Agent Interfaces](001-pydantic-script-agent-interfaces.md)
- [ADR-007: Progressive Ensemble Validation Suite](007-progressive-ensemble-validation-suite.md)
- [CLI Interaction Modes](../cli-interaction-modes.md)

## Validation

### BDD Scenarios

```gherkin
Feature: LLM-Friendly CLI Design
  As an LLM assistant helping users
  I want clear, discoverable CLI commands
  So I can effectively guide llm-orc usage

  Scenario: Discover init command from help
    Given I am an LLM reading CLI help output
    When I execute `llm-orc --help`
    Then I should see 'init' in the Common Commands section
    And the description should mention primitive scripts

  Scenario: Initialize project with scripts
    Given I want to set up an llm-orc project
    When I execute `llm-orc init`
    Then .llm-orc/scripts/primitives/ should contain Python scripts
    And I should see a success message with next steps

  Scenario: Discover validation commands
    Given I want to validate an ensemble
    When I execute `llm-orc validate --help`
    Then I should see examples of validate subcommands
    And each example should be a complete, runnable command

  Scenario: Recover from wrong command
    Given I mistakenly try `llm-orc run ensemble.yaml`
    When the command fails
    Then the error should suggest `llm-orc invoke ensemble.yaml`
    And the error should reference `llm-orc --help`
```

## References

- Click documentation: https://click.palletsprojects.com/
- MCP specification: https://spec.modelcontextprotocol.io/
- CLI design for LLMs: [Research paper references]
