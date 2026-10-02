"""Streaming execution and event handling for real-time visualization."""

import json
from typing import Any

import click
from rich.console import Console

from llm_orc.core.execution.results_processor import caller_status

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
) -> bool:
    """Run execution with streaming progress visualization.

    Returns:
        Whether the run's caller-facing status is "error" (fail-closed-
        composition, caller contract), for the caller's exit code.
    """
    # agents = ensemble_config.agents  # Unused in this conditional path

    if output_format in ["json", "text"]:
        # Direct processing without Rich status for JSON/text output
        return await _run_text_json_execution(
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
        outcome: dict[str, Any] = {}

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
            if hasattr(executor, "_script_agent_runner"):
                executor._script_agent_runner._progress_controller = progress_controller
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
                    outcome,
                )

                if not should_continue:
                    break

        return bool(outcome.get("has_errors", False))


async def run_standard_execution(
    executor: Any,
    ensemble_config: Any,  # EnsembleConfig type
    input_data: str,
    output_format: str = "rich",
    detailed: bool = True,
    record: dict[str, Any] | None = None,
) -> bool:
    """Run standard execution without streaming.

    Returns:
        Whether the run's caller-facing status is "error" (fail-closed-
        composition, caller contract), for the caller's exit code.
    """
    # Execute and get the result dict with "results" and "metadata"
    result = await executor.execute(ensemble_config, input_data)
    return display_result(
        _result_document(result, record),
        ensemble_config.agents,
        output_format,
        detailed,
        ensemble_config,
    )


_EXECUTOR_RESULT_KEYS = ("results", "metadata", "deliverable")


def _result_document(
    result: dict[str, Any], record: dict[str, Any] | None = None
) -> dict[str, Any]:
    """The executor's raw result in the caller vocabulary REST and MCP
    return: ``status`` is ``success`` or ``error``, with ``has_errors``,
    plus the bindings applied and models pulled when there are any. Only
    the keys the service's document has are kept: the executor's own
    ``input``, ``ensemble`` and ``execution_order`` are not part of it."""
    status, has_errors = caller_status(result.get("status"))
    kept = {k: v for k, v in result.items() if k in _EXECUTOR_RESULT_KEYS}
    applied = {k: v for k, v in (record or {}).items() if v}
    return {**kept, "status": status, "has_errors": has_errors, **applied}


def display_result(
    document: dict[str, Any],
    agents: list[Any],
    output_format: str,
    detailed: bool,
    config: Any = None,
) -> bool:
    """Print a result document in ``output_format`` and return whether its
    caller-facing status is "error".

    The document is what REST and MCP return (``results``, ``metadata``,
    ``deliverable``, ``status`` of success or error, ``has_errors``);
    ``agents`` are the ensemble's agent configs. ``config`` only adds the
    JSON document's ``config`` key. A document with no ``has_errors`` is
    an error.
    """
    has_errors = bool(document.get("has_errors", True))

    if output_format == "json":
        _display_json_results(document, config)
    elif output_format == "text":
        # Use plain text output for clean piping
        display_plain_text_results(
            document["results"], document["metadata"], detailed, agents
        )
    else:
        # Use Rich formatting for default output
        display_results(
            document["results"], document["metadata"], agents, detailed=detailed
        )

    return has_errors


async def _run_text_json_execution(
    executor: Any,
    ensemble_config: Any,
    input_data: str,
    output_format: str,
    detailed: bool,
) -> bool:
    """Run execution and output results as JSON/text in non-Rich mode.

    Returns:
        Whether the run's caller-facing status is "error", for the
        caller's exit code — an execution exception counts as an error
        too, in either output format.
    """
    has_errors = False
    try:
        if output_format == "json":
            # For JSON output, stream events as they happen
            async for event in executor.execute_streaming(ensemble_config, input_data):
                click.echo(json.dumps(event))
                if event.get("type") == "execution_completed":
                    raw_status = event.get("data", {}).get("status")
                    _, has_errors = caller_status(raw_status)
                elif event.get("type") == "execution_failed":
                    has_errors = True
        else:
            # For text output, execute and display results in plain text
            result = await executor.execute(ensemble_config, input_data)
            has_errors = display_result(
                result, ensemble_config.agents, "text", detailed
            )
    except Exception as e:
        has_errors = True
        if output_format == "json":
            error_event = {"type": "error", "error": str(e), "timestamp": "now"}
            click.echo(json.dumps(error_event))
        else:
            click.echo(f"Error: {e}")
    return has_errors


def _handle_streaming_event_with_status(
    event_type: str,
    event: dict[str, Any],
    agent_statuses: dict[str, str],
    ensemble_config: Any,
    status: Any,
    console: Any,
    output_format: str = "rich",
    detailed: bool = False,
    outcome: dict[str, Any] | None = None,
) -> bool:
    """Handle a single streaming event and update status display.

    ``outcome`` is a mutable out-param the caller reads after the loop
    ends: ``execution_completed`` sets ``outcome["has_errors"]`` to the
    run's caller-facing status (fail-closed-composition, caller
    contract), for the caller's exit code.

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
            event, ensemble_config, status, console, detailed, outcome
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
    outcome: dict[str, Any] | None = None,
) -> bool:
    """Handle execution completed event and return False to break event loop."""
    event_data = event.get("data", {})
    results = event_data.get("results", {})
    metadata = event_data.get("metadata", {})

    if outcome is not None:
        _, outcome["has_errors"] = caller_status(event_data.get("status"))

    # Force exit status context and clear before showing results
    status.stop()
    console.print("")

    # Display final results with a completely new console to avoid interference
    from rich.console import Console as FreshConsole

    results_console = FreshConsole(force_terminal=True, width=None)

    if detailed:
        _display_detailed_execution_results(
            results, metadata, ensemble_config.agents, results_console
        )
    else:
        _display_simple_results(
            results_console, results, metadata, ensemble_config.agents
        )

    return False


def _display_detailed_execution_results(
    results: dict[str, Any],
    metadata: dict[str, Any],
    agents: list[Any],
    results_console: Any,
) -> None:
    """Display detailed execution results."""
    # Display dependency graph at the top
    final_statuses = {
        name: "completed"
        for name in results.keys()
        if results[name].get("status") == "success"
    }
    final_tree = create_dependency_tree(agents, final_statuses)
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
            agents,
            metadata,
        )

    # Display performance metrics
    performance_lines = _format_performance_metrics(metadata)
    if performance_lines:
        results_console.print("\n" + "\n".join(performance_lines))


def _update_agent_status_by_names_from_lists(
    agents: list[Any],
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
    """Display results in JSON format.

    ``result`` is a result document in the caller vocabulary
    (fail-closed-composition): ``status`` ("success"/"error"),
    ``has_errors`` and ``deliverable``, as REST and MCP invoke report
    them, so a script parsing this output doesn't need a fourth,
    CLI-only shape.
    """
    try:
        # Safely get config dict, handling mocks/objects that aren't serializable
        config_dict: dict[str, Any] | None = None
        if ensemble_config is not None:
            try:
                config_dict = ensemble_config.to_dict()
            except (AttributeError, TypeError):
                config_dict = {"type": "mock_config"}

        # Every key of the document passes through, so a key the service
        # adds is never dropped here; the CLI adds only ``config``.
        output: dict[str, Any] = {
            "results": result.get("results", {}),
            "metadata": result.get("metadata", {}),
        }
        if config_dict is not None:
            output["config"] = config_dict
        output.update(
            {k: v for k, v in result.items() if k not in ("results", "metadata")}
        )
        output["status"] = result.get("status", "error")
        output["has_errors"] = bool(result.get("has_errors", True))
        output["deliverable"] = result.get("deliverable")

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
