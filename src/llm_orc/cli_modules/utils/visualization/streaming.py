"""Streaming execution and event handling for real-time visualization."""

import json
from collections.abc import Callable
from typing import Any

import click
from rich.console import Console
from rich.live import Live

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


async def _run_json_streaming_execution(
    input_data: str, ensemble_config: Any, execute_stream: Callable[..., Any]
) -> None:
    """Run streaming execution with JSON output."""
    try:
        # Stream events as JSON objects
        async for event in execute_stream():
            click.echo(json.dumps(event))
    except Exception as e:
        error_event = {"event_type": "error", "error": str(e), "timestamp": "now"}
        click.echo(json.dumps(error_event))


async def _run_rich_streaming_execution(
    input_data: str,
    ensemble_config: Any,
    execute_stream: Callable[..., Any],
    agents: list[dict[str, Any]],
) -> None:
    """Run streaming execution with Rich live display."""
    console = Console()
    agent_progress: dict[str, dict[str, Any]] = {}

    # Initialize agent progress
    for agent in agents:
        agent_progress[agent["name"]] = {"status": "⏳ Pending"}

    with Live(
        create_dependency_tree(agents, dict.fromkeys(agent_progress, "pending")),
        console=console,
        refresh_per_second=10,
    ) as live:
        try:
            async for event in execute_stream():
                _handle_streaming_event(event, agent_progress)

                # Update the live display
                statuses = {
                    name: info["status"] for name, info in agent_progress.items()
                }
                live.update(create_dependency_tree(agents, statuses))

                # Check if execution completed
                if event.get("event_type") == "execution_completed":
                    _process_execution_completed_event(event)
                    break

        except Exception as e:
            console.print(f"[red]Error during execution: {e}[/red]")


def _handle_streaming_event(
    event: dict[str, Any], agent_progress: dict[str, dict[str, Any]]
) -> None:
    """Handle individual streaming events."""
    event_type = event.get("event_type")
    agent_name = event.get("agent_name")

    if not agent_name:
        return

    if agent_name not in agent_progress:
        agent_progress[agent_name] = {}

    if event_type == "agent_started":
        _update_agent_progress_status(agent_name, "🔄 In Progress", agent_progress)
    elif event_type == "agent_completed":
        _update_agent_progress_status(agent_name, "✅ Completed", agent_progress)
    elif event_type == "agent_failed":
        _update_agent_progress_status(agent_name, "❌ Failed", agent_progress)
        agent_progress[agent_name]["error"] = event.get("error", "Unknown error")
    elif event_type == "fallback_started":
        _handle_fallback_started_event(event, agent_progress)
    elif event_type == "fallback_completed":
        _handle_fallback_completed_event(event, agent_progress)
    elif event_type == "fallback_failed":
        _handle_fallback_failed_event(event, agent_progress)


def _process_execution_completed_event(event: dict[str, Any]) -> None:
    """Process the execution completed event."""
    results = event.get("results", {})
    # metadata = event.get("metadata", {})  # Unused

    # Display final results
    console = Console()
    console.print("\n[bold green]✅ Execution Completed[/bold green]")

    # Show summary
    successful = len([r for r in results.values() if r.get("status") == "success"])
    total = len(results)
    console.print(f"Results: {successful}/{total} agents successful")


def _update_agent_progress_status(
    agent_name: str, status: str, agent_progress: dict[str, dict[str, Any]]
) -> None:
    """Update the status of a specific agent."""
    if agent_name not in agent_progress:
        agent_progress[agent_name] = {}
    agent_progress[agent_name]["status"] = status


def _update_agent_status_by_names(
    agent_names: list[str], status: str, agent_progress: dict[str, dict[str, Any]]
) -> None:
    """Update status for multiple agents by name."""
    for agent_name in agent_names:
        _update_agent_progress_status(agent_name, status, agent_progress)


def _handle_fallback_started_event(
    event: dict[str, Any], agent_progress: dict[str, dict[str, Any]]
) -> None:
    """Handle fallback started event."""
    agent_name = event.get("agent_name")
    original_model = event.get("original_model")
    fallback_model = event.get("fallback_model")

    if agent_name:
        _update_agent_progress_status(
            agent_name,
            f"🔄 Fallback: {original_model} → {fallback_model}",
            agent_progress,
        )


def _handle_fallback_completed_event(
    event: dict[str, Any], agent_progress: dict[str, dict[str, Any]]
) -> None:
    """Handle fallback completed event."""
    agent_name = event.get("agent_name")
    fallback_model = event.get("fallback_model")

    if agent_name:
        _update_agent_progress_status(
            agent_name, f"✅ Completed via {fallback_model}", agent_progress
        )


def _handle_fallback_failed_event(
    event: dict[str, Any], agent_progress: dict[str, dict[str, Any]]
) -> None:
    """Handle fallback failed event."""
    agent_name = event.get("agent_name")
    error = event.get("error", "Fallback failed")

    if agent_name:
        _update_agent_progress_status(agent_name, "❌ Fallback Failed", agent_progress)
        agent_progress[agent_name]["error"] = error


# Text mode streaming event handlers
def _handle_text_fallback_started(event: dict[str, Any]) -> None:
    """Handle fallback started in text mode."""
    agent_name = event.get("agent_name")
    original_model = event.get("original_model")
    fallback_model = event.get("fallback_model")

    click.echo(
        f"🔄 {agent_name}: Falling back from {original_model} to {fallback_model}"
    )


def _handle_text_fallback_completed(event: dict[str, Any]) -> None:
    """Handle fallback completed in text mode."""
    agent_name = event.get("agent_name")
    fallback_model = event.get("fallback_model")

    click.echo(f"✅ {agent_name}: Completed using fallback model {fallback_model}")


def _handle_text_fallback_failed(event: dict[str, Any]) -> None:
    """Handle fallback failed in text mode."""
    agent_name = event.get("agent_name")
    error = event.get("error", "Unknown error")

    click.echo(f"❌ {agent_name}: Fallback failed - {error}")


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
