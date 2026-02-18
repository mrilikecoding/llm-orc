"""Streaming execution and event handling for real-time visualization."""

import json
from typing import Any

import click
from rich.console import Console

from .dependency import create_dependency_tree
from .results_display import (
    _display_simple_results,
    display_plain_text_results,
    display_results,
)


async def run_streaming_execution(
    executor: Any,
    ensemble_config: Any,  # EnsembleConfig type
    input_data: str,
    output_format: str = "rich",
    detailed: bool = True,
) -> None:
    """Run execution with streaming progress visualization."""
    # agents = ensemble_config.agents  # Unused in this conditional path

    if output_format in ["json", "text"]:
        # Direct processing without Rich status for JSON/text output
        await _run_text_json_execution(
            executor, ensemble_config, input_data, output_format, detailed
        )
    else:
        # Rich interface for default output with real streaming
        console = Console(
            soft_wrap=True,
            width=None,
            force_terminal=True,
            no_color=False,
            legacy_windows=False,
            markup=True,
            highlight=False,
        )
        agent_statuses: dict[str, str] = {}

        # Initialize with dependency tree in status display
        initial_tree = create_dependency_tree(ensemble_config.agents, agent_statuses)

        with console.status(initial_tree, spinner="dots") as status:
            # Create progress controller for synchronous user input handling
            from llm_orc.cli_modules.utils.rich_progress_controller import (
                RichProgressController,
            )

            progress_controller = RichProgressController(
                console, status, agent_statuses, ensemble_config.agents
            )

            # Provide the executor with direct progress control for user input
            executor._progress_controller = progress_controller
            async for event in executor.execute_streaming(ensemble_config, input_data):
                event_type = event["type"]

                # Handle the event and update the display
                should_continue = _handle_streaming_event_with_status(
                    event_type,
                    event,
                    agent_statuses,
                    ensemble_config,
                    status,
                    console,
                    output_format,
                    detailed,
                )

                if not should_continue:
                    break


async def run_standard_execution(
    executor: Any,
    ensemble_config: Any,  # EnsembleConfig type
    input_data: str,
    output_format: str = "rich",
    detailed: bool = True,
) -> None:
    """Run standard execution without streaming."""
    # Execute and get the result dict with "results" and "metadata"
    result = await executor.execute(ensemble_config, input_data)

    if output_format == "json":
        # Display JSON results
        _display_json_results(result, ensemble_config)
    elif output_format == "text":
        # Use plain text output for clean piping
        display_plain_text_results(
            result["results"], result["metadata"], detailed, ensemble_config.agents
        )
    else:
        # Use Rich formatting for default output
        agents = ensemble_config.agents
        display_results(
            result["results"], result["metadata"], agents, detailed=detailed
        )


async def _run_text_json_execution(
    executor: Any,
    ensemble_config: Any,
    input_data: str,
    output_format: str,
    detailed: bool,
) -> None:
    """Run execution and output results as JSON/text in non-Rich mode."""
    try:
        if output_format == "json":
            # For JSON output, stream events as they happen
            async for event in executor.execute_streaming(ensemble_config, input_data):
                click.echo(json.dumps(event))
        else:
            # For text output, execute and display results in plain text
            result = await executor.execute(ensemble_config, input_data)
            display_plain_text_results(
                result["results"], result["metadata"], detailed, ensemble_config.agents
            )
    except Exception as e:
        if output_format == "json":
            error_event = {"type": "error", "error": str(e), "timestamp": "now"}
            click.echo(json.dumps(error_event))
        else:
            click.echo(f"Error: {e}")


def _handle_streaming_event_with_status(
    event_type: str,
    event: dict[str, Any],
    agent_statuses: dict[str, str],
    ensemble_config: Any,
    status: Any,
    console: Any,
    output_format: str = "rich",
    detailed: bool = False,
) -> bool:
    """Handle a single streaming event and update status display.

    Returns True if execution should continue, False if it should break.
    """
    if event_type == "agent_progress":
        status_changed = _handle_agent_progress_event(
            event, agent_statuses, ensemble_config
        )
    elif event_type == "execution_started":
        status_changed = False
    elif event_type == "agent_started":
        status_changed = _handle_agent_started_event(event, agent_statuses)
    elif event_type == "agent_completed":
        status_changed = _handle_agent_completed_event(event, agent_statuses)
    elif event_type == "agent_failed":
        status_changed = _handle_agent_failed_event(event, agent_statuses)
    elif event_type == "execution_completed":
        return _handle_execution_completed_event(
            event, ensemble_config, status, console, detailed
        )
    elif event_type == "user_input_required":
        status_changed = _handle_user_input_required_event(
            event, ensemble_config, status, console
        )
    elif event_type == "user_input_completed":
        status_changed = _handle_user_input_completed_event(
            event, agent_statuses, ensemble_config
        )
    else:
        status_changed = False

    if status_changed:
        current_tree = create_dependency_tree(ensemble_config.agents, agent_statuses)
        status.update(current_tree)

    return True


def _handle_agent_progress_event(
    event: dict[str, Any],
    agent_statuses: dict[str, str],
    ensemble_config: Any,
) -> bool:
    """Handle agent progress event and return True if status changed."""
    started_agent_names = event["data"].get("started_agent_names", [])
    completed_agent_names = event["data"].get("completed_agent_names", [])

    old_statuses = dict(agent_statuses)
    _update_agent_status_by_names_from_lists(
        ensemble_config.agents,
        started_agent_names,
        completed_agent_names,
        agent_statuses,
    )
    return old_statuses != agent_statuses


def _handle_agent_started_event(
    event: dict[str, Any], agent_statuses: dict[str, str]
) -> bool:
    """Handle agent started event and return True if status changed."""
    event_data = event["data"]
    agent_name = event_data["agent_name"]
    if agent_statuses.get(agent_name) != "running":
        agent_statuses[agent_name] = "running"
        return True
    return False


def _handle_agent_completed_event(
    event: dict[str, Any], agent_statuses: dict[str, str]
) -> bool:
    """Handle agent completed event and return True if status changed."""
    event_data = event["data"]
    agent_name = event_data["agent_name"]
    if agent_statuses.get(agent_name) != "completed":
        agent_statuses[agent_name] = "completed"
        return True
    return False


def _handle_agent_failed_event(
    event: dict[str, Any], agent_statuses: dict[str, str]
) -> bool:
    """Handle agent failed event and return True if status changed."""
    event_data = event["data"]
    agent_name = event_data["agent_name"]
    if agent_statuses.get(agent_name) != "failed":
        agent_statuses[agent_name] = "failed"
        return True
    return False


def _handle_execution_completed_event(
    event: dict[str, Any],
    ensemble_config: Any,
    status: Any,
    console: Any,
    detailed: bool,
) -> bool:
    """Handle execution completed event and return False to break event loop."""
    event_data = event.get("data", {})
    results = event_data.get("results", {})
    metadata = event_data.get("metadata", {})

    # Force exit status context and clear before showing results
    status.stop()
    console.print("")

    # Display final results with a completely new console to avoid interference
    from rich.console import Console as FreshConsole

    results_console = FreshConsole(force_terminal=True, width=None)

    if detailed:
        _display_detailed_execution_results(
            results, metadata, ensemble_config, results_console
        )
    else:
        _display_simple_results(
            results_console, results, metadata, ensemble_config.agents
        )

    return False


def _display_detailed_execution_results(
    results: dict[str, Any],
    metadata: dict[str, Any],
    ensemble_config: Any,
    results_console: Any,
) -> None:
    """Display detailed execution results."""
    # Display dependency graph at the top
    final_statuses = {
        name: "completed"
        for name in results.keys()
        if results[name].get("status") == "success"
    }
    final_tree = create_dependency_tree(ensemble_config.agents, final_statuses)
    results_console.print(final_tree)

    # Force display directly without Rich status interference
    results_console.print("\n[bold blue]📋 Results[/bold blue]")
    results_console.print("=" * 50)

    # Process and display agent results
    from .results_display import (
        _display_agent_result,
        _format_performance_metrics,
        _process_agent_results,
    )

    processed_results = _process_agent_results(results)
    for agent_name, result in processed_results.items():
        _display_agent_result(
            results_console,
            agent_name,
            result,
            ensemble_config.agents,
            metadata,
        )

    # Display performance metrics
    performance_lines = _format_performance_metrics(metadata)
    if performance_lines:
        results_console.print("\n" + "\n".join(performance_lines))


def _update_agent_status_by_names_from_lists(
    agents: list[dict[str, Any]],
    started_agent_names: list[str],
    completed_agent_names: list[str],
    agent_statuses: dict[str, str],
) -> None:
    """Update agent statuses based on started and completed agent name lists."""
    for agent_name in started_agent_names:
        if agent_name not in completed_agent_names:
            agent_statuses[agent_name] = "running"

    for agent_name in completed_agent_names:
        agent_statuses[agent_name] = "completed"


def _display_json_results(result: dict[str, Any], ensemble_config: Any) -> None:
    """Display results in JSON format."""
    try:
        # Safely get config dict, handling mocks/objects that aren't serializable
        try:
            config_dict = ensemble_config.to_dict()
        except (AttributeError, TypeError):
            config_dict = {"type": "mock_config"}

        output = {
            "results": result.get("results", {}),
            "metadata": result.get("metadata", {}),
            "config": config_dict,
        }

        click.echo(json.dumps(output, indent=2, default=str))
    except Exception as e:
        # Fallback error handling
        error_output = {"error": str(e), "config": {"type": "error_config"}}
        click.echo(json.dumps(error_output, indent=2))


def _handle_user_input_required_event(
    event: dict[str, Any],
    ensemble_config: Any,
    status: dict[str, Any],
    console: Any,
) -> bool:
    """Handle user input required event."""
    # Extract data from the correct nested structure
    event_data = event.get("data", {})
    agent_name = event_data.get("agent_name", "unknown")
    message = event_data.get("message", "Input required")

    console.print(f"[yellow]⏸️  {agent_name}: {message}[/yellow]")
    return False


def _handle_user_input_completed_event(
    event: dict[str, Any],
    agent_statuses: dict[str, Any],
    ensemble_config: Any,
) -> bool:
    """Handle user input completed event."""
    # Extract data from the correct nested structure
    event_data = event.get("data", {})
    agent_name = event_data.get("agent_name", "unknown")
    # Set agent status back to running (updated to completed when agent finishes)
    if agent_name in agent_statuses:
        agent_statuses[agent_name] = "running"
    return True
