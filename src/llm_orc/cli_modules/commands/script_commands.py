"""Script management CLI commands."""

import json
import subprocess
import sys
from pathlib import Path

import click

from llm_orc.core.execution.primitive_registry import PrimitiveRegistry


def list_scripts_impl(category: str | None = None, json_output: bool = False) -> str:
    """Implementation of list_scripts command.

    Args:
        category: Optional category filter
        json_output: Whether to output JSON format

    Returns:
        Formatted output string
    """
    registry = PrimitiveRegistry()
    primitives = registry.discover_primitives()

    # Filter by category if specified
    if category:
        primitives = [p for p in primitives if p.get("category") == category]

    if json_output:
        return json.dumps(primitives, indent=2)

    # Display in table format
    if not primitives:
        return "No scripts found."

    lines = ["Available Scripts:", "-" * 80]

    # Print each script
    for primitive in primitives:
        name = primitive.get("name", "unknown")
        category_val = primitive.get("category", "uncategorized")
        description = primitive.get("description", "")
        lines.append(f"{name:<30} {category_val:<25} {description}")

    return "\n".join(lines)


@click.command()
@click.option(
    "--category",
    default=None,
    help="Filter scripts by category",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Output in JSON format",
)
def list_scripts(category: str | None, json_output: bool) -> None:
    """List available scripts."""
    output = list_scripts_impl(category, json_output)
    click.echo(output)


def show_script_impl(name: str) -> str:
    """Implementation of show_script command.

    Args:
        name: Script name to display

    Returns:
        Formatted output string

    Raises:
        FileNotFoundError: If script is not found
        KeyError: If script is not found
    """
    registry = PrimitiveRegistry()
    script_info = registry.get_primitive_info(name)

    # Build output lines
    lines = [
        f"Script: {script_info.get('name', 'unknown')}",
        f"Category: {script_info.get('category', 'uncategorized')}",
        f"Path: {script_info.get('path', 'unknown')}",
        "",
        f"Description: {script_info.get('description', 'No description available')}",
        "",
    ]

    # Show parameters if available
    parameters = script_info.get("parameters", {})
    if parameters:
        lines.append("Parameters:")
        for param_name, param_type in parameters.items():
            lines.append(f"  {param_name}: {param_type}")
        lines.append("")

    # Show return values if available
    returns = script_info.get("returns", {})
    if returns:
        lines.append("Returns:")
        for return_name, return_type in returns.items():
            lines.append(f"  {return_name}: {return_type}")

    return "\n".join(lines)


@click.command()
@click.argument("name")
def show_script(name: str) -> None:
    """Show script documentation."""
    try:
        output = show_script_impl(name)
        click.echo(output)
    except (FileNotFoundError, KeyError) as e:
        click.echo(f"Error: Script '{name}' not found.", err=True)
        raise click.ClickException(f"Script '{name}' not found.") from e


@click.command()
@click.argument("script_path")
@click.option(
    "--parameters",
    default=None,
    help="JSON parameters for the script",
)
def test_script(script_path: str, parameters: str | None) -> None:
    """Test script with parameters."""
    script_file = Path(script_path)

    if not script_file.exists():
        click.echo(f"Error: Script not found: {script_path}", err=True)
        raise click.ClickException(f"Script not found: {script_path}")

    # Parse parameters if provided
    input_data = {}
    if parameters:
        try:
            input_data = json.loads(parameters)
        except json.JSONDecodeError as e:
            click.echo(f"Error: Invalid JSON parameters: {e}", err=True)
            raise click.ClickException(f"Invalid JSON parameters: {e}") from e

    # Execute the script with Python

    try:
        result = subprocess.run(
            [sys.executable, str(script_file)],
            input=json.dumps(input_data),
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            click.echo("Error: Script execution failed", err=True)
            click.echo(f"stderr: {result.stderr}", err=True)
            raise click.ClickException("Script execution failed")

        # Validate output is JSON
        try:
            output_data = json.loads(result.stdout.strip())
            click.echo("Script executed successfully!")
            click.echo(json.dumps(output_data, indent=2))
        except json.JSONDecodeError as e:
            click.echo("Error: Script output is not valid JSON", err=True)
            click.echo(f"Output: {result.stdout}", err=True)
            raise click.ClickException("Invalid JSON output from script") from e

    except subprocess.TimeoutExpired as e:
        click.echo("Error: Script execution timed out", err=True)
        raise click.ClickException("Script execution timed out") from e
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.ClickException(f"Script execution error: {e}") from e
