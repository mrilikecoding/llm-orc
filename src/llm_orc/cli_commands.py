"""Main CLI command implementations."""

import asyncio
import json
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import click
import yaml

from llm_orc.cli_modules.utils.config_utils import (
    display_local_profiles,
    get_available_providers,
)
from llm_orc.cli_modules.utils.visualization import (
    run_standard_execution,
    run_streaming_execution,
)
from llm_orc.cli_modules.utils.visualization.refusal_display import (
    display_preflight,
    display_refusal,
    display_run_record,
)
from llm_orc.cli_modules.utils.visualization.streaming import display_result
from llm_orc.core.config.config_manager import (
    ConfigurationManager,
    resolve_global_config_dir,
)
from llm_orc.core.config.ensemble_config import EnsembleConfig
from llm_orc.core.config.remotes import RemoteError
from llm_orc.services.closure_shipper import LeftOut
from llm_orc.services.handlers.run_preparation import (
    RootNotFoundError,
    RunRefusedError,
)
from llm_orc.services.remote_probe import probe_remotes
from llm_orc.services.remote_run import (
    RemoteRunError,
    preflight_remote,
    run_remote,
)

WAIT_TICK_S = 1.0


def _get_service() -> Any:
    """Create an OrchestraService instance for CLI use."""
    from llm_orc.services.orchestra_service import OrchestraService

    return OrchestraService()


def _resolve_input_data(
    positional_input: str | None,
    option_input: str | None,
    file_input: str | None = None,
) -> str:
    """Resolve input data with priority: positional > option > file > stdin > default.

    Args:
        positional_input: Input data from positional argument.
        option_input: Input data from --input option.
        file_input: Path to a file whose contents become input data.

    Returns:
        Resolved input data string.

    Raises:
        FileNotFoundError: If file_input path does not exist.
    """
    final_input_data = positional_input or option_input

    if final_input_data is None and file_input is not None:
        path = Path(file_input)
        if not path.is_file():
            msg = f"Input file not found: {file_input}"
            raise FileNotFoundError(msg)
        final_input_data = path.read_text()

    if final_input_data is None:
        if not sys.stdin.isatty():
            final_input_data = sys.stdin.read().strip()
        else:
            final_input_data = "Please analyze this."

    return final_input_data


def _find_ensemble_config(
    ensemble_name: str, ensemble_dirs: list[Path], service: Any
) -> EnsembleConfig:
    """Find ensemble configuration in the provided directories.

    Args:
        ensemble_name: Name of the ensemble to find
        ensemble_dirs: List of directories to search
        service: OrchestraService instance for ensemble loading

    Returns:
        EnsembleConfig: The found ensemble configuration

    Raises:
        click.ClickException: If ensemble is not found in any directory
    """
    ensemble_config = None

    for ensemble_dir in ensemble_dirs:
        ensemble_config = cast(
            "EnsembleConfig | None",
            service.find_ensemble_in_dir(ensemble_name, str(ensemble_dir)),
        )
        if ensemble_config is not None:
            break

    if ensemble_config is None:
        searched_dirs = [str(d) for d in ensemble_dirs]
        raise click.ClickException(
            f"Ensemble '{ensemble_name}' not found in: {', '.join(searched_dirs)}"
        )

    return ensemble_config


def _format_ensemble_display_name(ensemble: EnsembleConfig) -> str:
    """Format ensemble name for display."""
    if ensemble.relative_path:
        return f"{ensemble.relative_path}/{ensemble.name}"
    return ensemble.name


def _display_ensemble_group(ensembles: Sequence[EnsembleConfig], header: str) -> None:
    """Display a group of ensembles with header."""
    if not ensembles:
        return
    click.echo(f"\n{header}")
    for ensemble in sorted(ensembles, key=lambda e: (e.relative_path or "", e.name)):
        display_name = _format_ensemble_display_name(ensemble)
        click.echo(f"  {display_name}: {ensemble.description}")


def _display_grouped_ensembles(
    config_manager: Any,
    local_ensembles: Sequence[EnsembleConfig],
    library_ensembles: Sequence[EnsembleConfig],
    global_ensembles: Sequence[EnsembleConfig],
    packaged_ensembles: Sequence[EnsembleConfig] = (),
    bundle_ensembles: Sequence[EnsembleConfig] = (),
) -> None:
    """Display grouped ensembles with proper formatting.

    Args:
        config_manager: Configuration manager instance
        local_ensembles: List of local ensemble configs
        library_ensembles: List of library ensemble configs
        global_ensembles: List of global ensemble configs
        packaged_ensembles: List of packaged (shipped with llm-orc) configs
        bundle_ensembles: Roots of persisted closures (stored run requests)
    """
    click.echo("Available ensembles:")

    _display_ensemble_group(local_ensembles, "📁 Local Repo (.llm-orc/ensembles):")
    _display_ensemble_group(
        library_ensembles, "📚 Library (llm-orchestra-library/ensembles):"
    )

    global_header = f"🌐 Global ({config_manager.global_config_dir}/ensembles):"
    _display_ensemble_group(global_ensembles, global_header)

    _display_ensemble_group(packaged_ensembles, "📦 Packaged (shipped with llm-orc):")

    bundle_header = f"🧳 Bundles ({config_manager.global_config_dir}/bundles):"
    _display_ensemble_group(bundle_ensembles, bundle_header)


def _setup_performance_display(
    config_manager: Any,
    executor: Any,
    ensemble_name: str,
    ensemble_config: "EnsembleConfig",
    streaming: bool,
    output_format: str | None,
    input_data: str,
) -> None:
    """Setup and display performance configuration for Rich interface.

    Args:
        config_manager: Configuration manager instance
        executor: Ensemble executor
        ensemble_name: Name of the ensemble
        ensemble_config: Ensemble configuration
        streaming: Streaming flag from CLI
        output_format: Output format (None for Rich interface)
        input_data: Input data for display
    """
    if output_format is not None:  # Skip for text/json output
        return

    click.echo(
        f"Executing ensemble '{ensemble_name}' with "
        f"{len(ensemble_config.agents)} agents"
    )
    click.echo("─" * 50)


def _determine_effective_streaming(
    config_manager: Any,
    output_format: str | None,
    streaming: bool,
) -> bool:
    """Determine effective streaming setting based on output format and config.

    Args:
        config_manager: Configuration manager instance
        output_format: Output format (text/json/rich)
        streaming: Streaming flag from CLI

    Returns:
        Whether to use streaming execution
    """
    # For text/JSON output, use standard execution for clean piping output
    # Only use streaming for Rich interface (default) or when explicitly requested
    if output_format in ["json", "text"]:
        return False  # Clean, non-streaming output for piping
    else:
        # Default Rich interface - use streaming
        try:
            performance_config = config_manager.load_performance_config()
            return streaming or performance_config.get("streaming_enabled", True)
        except Exception:
            # Fallback if performance config fails
            return streaming  # Use just the CLI flag


async def _execute_ensemble_with_mode(
    executor: Any,
    ensemble_config: "EnsembleConfig",
    input_data: str,
    output_format: str | None,
    detailed: bool,
    requires_user_input: bool,
    effective_streaming: bool,
    record: dict[str, Any] | None = None,
) -> bool:
    """Execute ensemble with the appropriate execution mode.

    Args:
        executor: Ensemble executor
        ensemble_config: Ensemble configuration
        input_data: Input data for execution
        output_format: Output format
        detailed: Detailed output flag
        requires_user_input: Whether ensemble requires user input
        effective_streaming: Whether to use streaming execution
        record: Bindings applied and models pulled, for the JSON document

    Returns:
        Whether the run's caller-facing status is "error" (fail-closed-
        composition, caller contract) — the caller uses this to set a
        non-zero process exit code.
    """
    # Convert None output_format to "rich" for execution functions
    execution_format = output_format or "rich"

    if requires_user_input or effective_streaming:
        # Streaming visualization; interactive scripts need its progress control
        return await run_streaming_execution(
            executor, ensemble_config, input_data, execution_format, detailed
        )
    return await run_standard_execution(
        executor, ensemble_config, input_data, execution_format, detailed, record
    )


def _root_lookup(
    service: Any, ensemble_name: str, config_dir: str | None
) -> Callable[[str], Any]:
    """The lookup the prepared run is given for the named root: the
    service's tiers, or the one ``--config-dir`` directory. A miss in the
    tiers is None, so the service can consult its bundles before the
    command reports it; ``--config-dir`` is strict and ends the command
    with the directory searched."""

    def lookup(name: str) -> Any:
        if config_dir is not None:
            return _find_ensemble_config(name, [Path(config_dir)], service)
        return service.find_ensemble_by_name(name)

    return lookup


def _not_found(service: Any, name: str) -> click.ClickException:
    searched = [str(d) for d in service.config_manager.get_ensembles_dirs()]
    return click.ClickException(
        f"Ensemble '{name}' not found in: {', '.join(searched)}"
    )


async def _run_prepared(
    service: Any,
    request: dict[str, Any],
    lookup: Callable[[str], Any],
    input_data: str,
    options: "_RunOptions",
) -> bool:
    """Run ``request`` inside the preparation step and display it. A
    refusal is displayed and counts as an error."""
    try:
        async with service.prepared_run(request, lookup) as run:
            return await _run_with_display(run, input_data, service, options)
    except RunRefusedError as refusal:
        display_refusal(refusal.envelope(), options.output_format)
        return True


async def _run_with_display(
    run: Any, input_data: str, service: Any, options: "_RunOptions"
) -> bool:
    from llm_orc.core.execution.scripting.user_input_handler import (
        ScriptUserInputHandler,
    )

    requires_user_input = ScriptUserInputHandler().ensemble_requires_user_input(
        run.config
    )

    # Override concurrency settings if provided
    if options.max_concurrent is not None:
        run.executor.set_max_concurrent_agents(options.max_concurrent)

    # Show performance configuration only for default Rich interface (not text/json)
    _setup_performance_display(
        service.config_manager,
        run.executor,
        options.ensemble_name,
        run.config,
        options.streaming,
        options.output_format,
        input_data,
    )
    display_run_record(run.bindings, run.pulled, options.output_format)

    effective_streaming = _determine_effective_streaming(
        service.config_manager, options.output_format, options.streaming
    )
    record = {"bindings": run.bindings, "pulled": run.pulled}
    return await _execute_ensemble_with_mode(
        run.executor,
        run.config,
        input_data,
        options.output_format,
        options.detailed,
        requires_user_input,
        effective_streaming,
        record,
    )


@dataclass(frozen=True)
class _RunOptions:
    ensemble_name: str
    output_format: str | None
    streaming: bool
    max_concurrent: int | None
    detailed: bool


def invoke_ensemble(
    ensemble_name: str,
    input_data: str | None,
    config_dir: str | None,
    input_data_option: str | None,
    output_format: str | None,
    streaming: bool,
    max_concurrent: int | None,
    detailed: bool,
    *,
    input_file: str | None = None,
    bind: Mapping[str, str] | None = None,
    pull: bool = False,
    remote: str | None = None,
    with_profiles: Sequence[str] = (),
    persist: str | None = None,
    preflight: bool = False,
) -> bool:
    """Invoke an ensemble of agents.

    The run is a request (``ensemble_name``, ``bind``, ``pull``) handed to
    the preparation step REST and MCP use, so an ensemble this host
    cannot run is refused before any agent starts.

    Returns:
        Whether the run's caller-facing status is "error" (fail-closed-
        composition, caller contract) — ``cli.py``'s ``invoke`` command
        uses this to set a non-zero process exit code in every output
        format (rich/text/json alike).
    """
    if preflight:
        return _preflight_invocation(
            ensemble_name,
            RemoteInvocation(
                remote=remote,
                input_data="",
                config_dir=config_dir,
                output_format=output_format,
                max_concurrent=max_concurrent,
                detailed=detailed,
                bind=bind,
                pull=pull,
                with_profiles=with_profiles,
            ),
        )
    if remote is not None:
        return _invoke_remote(
            ensemble_name,
            RemoteInvocation(
                remote=remote,
                input_data=_resolve_input_data(
                    input_data, input_data_option, file_input=input_file
                ),
                config_dir=config_dir,
                output_format=output_format,
                max_concurrent=max_concurrent,
                detailed=detailed,
                bind=bind,
                pull=pull,
                with_profiles=with_profiles,
                persist=persist,
            ),
        )
    service = _get_service()

    # Resolve input data using helper method
    input_data = _resolve_input_data(
        input_data, input_data_option, file_input=input_file
    )

    request: dict[str, Any] = {"ensemble_name": ensemble_name}
    if bind:
        request["bind"] = dict(bind)
    if pull:
        request["pull"] = True
    options = _RunOptions(
        ensemble_name, output_format, streaming, max_concurrent, detailed
    )
    lookup = _root_lookup(service, ensemble_name, config_dir)

    try:
        return asyncio.run(_run_prepared(service, request, lookup, input_data, options))
    except click.ClickException:
        raise
    except RootNotFoundError as e:
        raise _not_found(service, e.name) from e
    except Exception as e:
        raise click.ClickException(f"Ensemble execution failed: {e!s}") from e


@dataclass(frozen=True)
class RemoteInvocation:
    """What ``invoke --remote`` (or ``--preflight``, whose ``remote`` may be
    None: a local gate) was asked for."""

    remote: str | None
    input_data: str
    config_dir: str | None
    output_format: str | None
    max_concurrent: int | None
    detailed: bool
    bind: Mapping[str, str] | None = None
    pull: bool = False
    with_profiles: Sequence[str] = ()
    persist: str | None = None


def _remote_root(
    ensemble_name: str, invocation: RemoteInvocation
) -> tuple[Any, EnsembleConfig]:
    """The service and the local root a remote call ships. Refused before
    anything is sent: ``--max-concurrent`` (the request has no place for
    it) and a root that is not a tier ensemble."""
    if invocation.max_concurrent is not None:
        raise click.ClickException(
            "--max-concurrent cannot be used with --remote: "
            "the run request cannot carry it"
        )
    service = _get_service()
    root = _root_lookup(service, ensemble_name, invocation.config_dir)(ensemble_name)
    if root is None:
        # Only a tier root ships: a bundle is a stored request, not a file.
        raise _not_found(service, ensemble_name)
    return service, root


def _invoke_remote(ensemble_name: str, invocation: RemoteInvocation) -> bool:
    """Run the named local root on the remote and display its answer.

    Refused before anything is sent: ``--max-concurrent`` (the request
    has no place for it), an unknown remote, a closure that cannot ship
    and an interactive script.
    """
    remote = invocation.remote
    assert remote is not None
    service, root = _remote_root(ensemble_name, invocation)
    try:
        with _waiting_on(remote, invocation.output_format):
            document = asyncio.run(
                run_remote(
                    ensemble_name,
                    remote,
                    find_root=lambda _name: root,
                    config_manager=service.config_manager,
                    project_dir=service.project_path,
                    input_text=invocation.input_data,
                    with_profiles=invocation.with_profiles,
                    bind=invocation.bind,
                    pull=invocation.pull,
                    persist=invocation.persist,
                    on_left_out=_say_left_out,
                )
            )
    except RemoteRunError as e:
        raise click.ClickException(str(e)) from e
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    return _display_remote_document(
        document, root, invocation.output_format, invocation.detailed
    )


def _preflight_invocation(ensemble_name: str, invocation: RemoteInvocation) -> bool:
    """Gate the named root, here or on the remote, print the answer and run
    nothing. Returns whether the answer is a failure."""
    if invocation.remote is not None:
        document = _preflight_on_remote(ensemble_name, invocation, invocation.remote)
        return display_preflight(document, invocation.output_format)
    service = _get_service()
    request: dict[str, Any] = {"ensemble_name": ensemble_name}
    if invocation.bind:
        request["bind"] = dict(invocation.bind)
    if invocation.pull:
        request["pull"] = True
    lookup = _root_lookup(service, ensemble_name, invocation.config_dir)
    try:
        document = asyncio.run(service.preflight_run(request, lookup))
    except click.ClickException:
        raise
    except RootNotFoundError as e:
        raise _not_found(service, e.name) from e
    except Exception as e:
        raise click.ClickException(f"Preflight failed: {e!s}") from e
    return display_preflight(document, invocation.output_format)


def _preflight_on_remote(
    ensemble_name: str, invocation: RemoteInvocation, remote: str
) -> dict[str, Any]:
    service, root = _remote_root(ensemble_name, invocation)
    try:
        with _waiting_on(remote, invocation.output_format, "Checking"):
            return asyncio.run(
                preflight_remote(
                    ensemble_name,
                    remote,
                    find_root=lambda _name: root,
                    config_manager=service.config_manager,
                    project_dir=service.project_path,
                    with_profiles=invocation.with_profiles,
                    bind=invocation.bind,
                    pull=invocation.pull,
                    on_left_out=_say_left_out,
                )
            )
    except RemoteRunError as e:
        raise click.ClickException(str(e)) from e
    except KeyboardInterrupt:
        raise SystemExit(130) from None


def remotes_command(output_format: str | None) -> None:
    """List the configured remotes, each with a live health probe. Exit 0
    whether or not any answers; a malformed ``remotes`` key is an error."""
    try:
        rows = asyncio.run(probe_remotes(ConfigurationManager(provision=False)))
    except RemoteError as e:
        raise click.ClickException(str(e)) from e
    if output_format == "json":
        click.echo(json.dumps(rows, indent=2))
        return
    if not rows:
        config_file = resolve_global_config_dir() / "config.yaml"
        click.echo(
            "No remotes are configured. Add a remotes: block to "
            f"{config_file}:\n"
            "  remotes:\n"
            "    remote-host:\n"
            "      url: https://llm-orc.remote.example"
        )
    for row in rows:
        click.echo(_remote_line(row))


def _remote_line(row: Mapping[str, Any]) -> str:
    state = (
        f"reachable, llm-orc {row['version']}"
        if row["reachable"]
        else f"unreachable: {row['error']}"
    )
    return f"{row['name']}  {row['url']}  {state}"


def _say_left_out(left_out: list[LeftOut]) -> None:
    """One line on stderr, in every output mode, naming what the remote
    must satisfy with its own copy. stdout stays the result alone."""
    labels = ", ".join(item.label for item in left_out)
    click.echo(f"Left to the remote (not found locally): {labels}", err=True)


@contextmanager
def _waiting_on(
    remote: str, output_format: str | None, verb: str = "Running"
) -> Iterator[None]:
    """In rich mode, the remote's name and the elapsed time on stderr
    while the run is out. Text and JSON modes print nothing, so a pipe
    stays clean."""
    if output_format is not None:
        yield
        return
    done = threading.Event()
    started = time.monotonic()
    interactive = sys.stderr.isatty()

    def tick() -> None:
        while True:
            elapsed = time.monotonic() - started
            line = f"{verb} on {remote}... {elapsed:.0f}s"
            click.echo(
                f"\r{line}" if interactive else line, err=True, nl=not interactive
            )
            if done.wait(WAIT_TICK_S) or not interactive:
                return

    ticker = threading.Thread(target=tick, daemon=True)
    ticker.start()
    try:
        yield
    finally:
        done.set()
        ticker.join()
        if interactive:
            click.echo("", err=True)


def _display_remote_document(
    document: dict[str, Any],
    root: EnsembleConfig,
    output_format: str | None,
    detailed: bool,
) -> bool:
    """Show the remote's document as a local run shows its own: a refusal
    as the table, a result through the result display. Returns whether its
    status is "error"."""
    error = document.get("error")
    if isinstance(error, dict) and "kind" in error:
        display_refusal(document, output_format)
        return True
    display_run_record(
        document.get("bindings") or {},
        document.get("pulled") or [],
        output_format,
        document.get("persisted"),
    )
    return display_result(
        {"results": {}, "metadata": {}, **document},
        root.agents,
        output_format or "rich",
        detailed,
        root,
    )


def _list_ensembles_from_dir(config_dir: str, service: Any) -> None:
    """List ensembles from a specific directory."""
    ensembles = service.list_ensembles_in_dir(config_dir)

    if not ensembles:
        click.echo(f"No ensembles found in {config_dir}")
        click.echo("  (Create .yaml files with ensemble configurations)")
    else:
        click.echo(f"Available ensembles in {config_dir}:")
        for ensemble in ensembles:
            click.echo(f"  {ensemble.name}: {ensemble.description}")


def list_ensembles_command(config_dir: str | None) -> None:
    """List available ensembles."""
    if config_dir is not None:
        service = _get_service()
        _list_ensembles_from_dir(config_dir, service)
        return

    service = _get_service()
    ensemble_dirs = service.config_manager.get_ensembles_dirs()
    grouped = service.list_ensembles_grouped()
    if not ensemble_dirs and not grouped["bundle"]:
        click.echo("No ensemble directories found.")
        click.echo("Run 'llm-orc config init' to set up local configuration.")
        return

    if not any(grouped.values()):
        click.echo("No ensembles found in any configured directories:")
        for dir_path in ensemble_dirs:
            click.echo(f"  {dir_path}")
        click.echo("  (Create .yaml files with ensemble configurations)")
        return

    _display_grouped_ensembles(
        service.config_manager,
        grouped["local"],
        grouped["library"],
        grouped["global"],
        grouped["packaged"],
        grouped.get("bundle", []),
    )


def _load_profiles_from_config(config_file: Path) -> dict[str, Any]:
    """Load model profiles from a configuration file.

    Args:
        config_file: Path to the configuration file

    Returns:
        Dictionary of model profiles, empty if file doesn't exist or has no profiles
    """
    if not config_file.exists():
        return {}

    with open(config_file) as f:
        config = yaml.safe_load(f) or {}
        profiles: dict[str, Any] = config.get("model_profiles", {})
        return profiles


def _display_global_profile(profile_name: str, profile: Any) -> None:
    """Display a single global profile with validation.

    Args:
        profile_name: Name of the profile
        profile: Profile configuration (should be dict)
    """
    # Handle case where profile is not a dict (malformed YAML)
    if not isinstance(profile, dict):
        click.echo(
            f"  {profile_name}: [Invalid profile format - "
            f"expected dict, got {type(profile).__name__}]"
        )
        return

    model = profile.get("model", "Unknown")
    provider = profile.get("provider", "Unknown")
    cost = profile.get("cost_per_token", "Not specified")

    click.echo(f"  {profile_name}:")
    click.echo(f"    Model: {model}")
    click.echo(f"    Provider: {provider}")
    click.echo(f"    Cost per token: {cost}")


def list_profiles_command() -> None:
    """List available model profiles with their provider/model details."""
    service = _get_service()
    config_manager = service.config_manager

    # Get all model profiles (merged global + local)
    all_profiles = config_manager.get_model_profiles()

    if not all_profiles:
        click.echo("No model profiles found.")
        click.echo("Run 'llm-orc config init' to create default profiles.")
        return

    # Load separate global and local profiles for grouping
    global_config_file = config_manager.global_config_dir / "config.yaml"
    global_profiles = _load_profiles_from_config(global_config_file)

    local_profiles = {}
    if config_manager.local_config_dir:
        local_config_file = config_manager.local_config_dir / "config.yaml"
        local_profiles = _load_profiles_from_config(local_config_file)

    click.echo("Available model profiles:")

    # Get available providers for status indicators
    available_providers = get_available_providers(config_manager)

    # Show local profiles first (if any)
    if local_profiles:
        display_local_profiles(local_profiles, available_providers)

    # Show global profiles
    if global_profiles:
        global_config_label = f"Global ({config_manager.global_config_dir}/config.yaml)"
        click.echo(f"\n🌐 {global_config_label}:")
        for profile_name in sorted(global_profiles.keys()):
            # Skip if this profile is overridden by local
            if profile_name in local_profiles:
                click.echo(f"  {profile_name}: (overridden by local)")
                continue

            profile = global_profiles[profile_name]
            _display_global_profile(profile_name, profile)


# Script and Artifact Commands
def _format_json_output(data: Any) -> None:
    """Format and display JSON output."""
    click.echo(json.dumps(data, indent=2))


def scripts_list_command(format_type: str) -> None:
    """List available scripts."""
    from llm_orc.cli_modules.commands.script_commands import list_scripts_impl

    json_output = format_type == "json"
    output = list_scripts_impl(category=None, json_output=json_output)
    click.echo(output)


def scripts_show_command(script_name: str) -> None:
    """Show script documentation."""
    from llm_orc.cli_modules.commands.script_commands import show_script_impl

    try:
        output = show_script_impl(script_name)
        click.echo(output)
    except (FileNotFoundError, KeyError):
        click.echo(f"Script '{script_name}' not found", err=True)
        raise SystemExit(1) from None


def _parse_json_parameters(parameters_json: str | None) -> dict[str, Any]:
    """Parse JSON parameters with error handling."""
    if not parameters_json:
        return {}

    try:
        data = json.loads(parameters_json)
        if isinstance(data, dict):
            return data
        else:
            click.echo("Parameters must be a JSON object", err=True)
            raise SystemExit(1)
    except json.JSONDecodeError as e:
        click.echo("Invalid JSON in parameters", err=True)
        raise SystemExit(1) from e


def scripts_test_command(script_name: str, parameters_json: str | None) -> None:
    """Test script with parameters."""
    from llm_orc.core.execution.scripting.resolver import ScriptResolver

    resolver = ScriptResolver()
    parameters = _parse_json_parameters(parameters_json)

    # Execute script test
    result = resolver.test_script(script_name, parameters)

    if result["success"]:
        click.echo(result["output"])
        click.echo(f"Duration: {result['duration_ms']}ms")
    else:
        click.echo(result["output"], err=True)
        if "error" in result:
            click.echo(f"Error: {result['error']}", err=True)
        raise SystemExit(1)


def artifacts_list_command(format_type: str) -> None:
    """List execution artifacts."""
    from llm_orc.core.config.config_manager import ConfigurationManager
    from llm_orc.core.config.state import ARTIFACTS_DIRNAME, resolve_state_dir
    from llm_orc.core.execution.artifact_manager import ArtifactManager

    local = ConfigurationManager(provision=False).local_config_dir
    manager = ArtifactManager(
        artifacts_dir=resolve_state_dir(local) / ARTIFACTS_DIRNAME
    )
    ensembles = manager.list_ensembles()

    if format_type == "json":
        _format_json_output(ensembles)
        return

    if not ensembles:
        click.echo("No artifacts found in .llm-orc/artifacts/")
        return

    click.echo("Available artifacts:")
    for ensemble in ensembles:
        count_str = f"{ensemble['executions_count']} execution"
        if ensemble["executions_count"] != 1:
            count_str += "s"
        click.echo(
            f"  {ensemble['name']}: {count_str}, latest: {ensemble['latest_execution']}"
        )


def _display_agent_artifact(agent: dict[str, Any]) -> None:
    """Display a single agent's artifact result."""
    agent_name = agent.get("name", "Unknown")
    status = agent.get("status", "unknown")
    click.echo(f"  {agent_name}: {status}")

    if status == "completed" and "result" in agent:
        result_text = agent["result"]
        preview = result_text[:100] + ("..." if len(result_text) > 100 else "")
        click.echo(f"    → {preview}")
    elif status == "failed" and "error" in agent:
        click.echo(f"    → Error: {agent['error']}")


def _display_artifact_text_format(ensemble_name: str, results: dict[str, Any]) -> None:
    """Display artifact results in text format."""
    click.echo(f"Ensemble: {results.get('ensemble_name', ensemble_name)}")

    if "timestamp" in results:
        click.echo(f"Executed: {results['timestamp']}")

    if "total_duration_ms" in results:
        duration_s = results["total_duration_ms"] / 1000
        click.echo(f"Duration: {duration_s:.1f}s")

    if "agents" in results and results["agents"]:
        click.echo("\nAgent Results:")
        for agent in results["agents"]:
            _display_agent_artifact(agent)


def artifacts_show_command(
    ensemble_name: str, format_type: str, execution_timestamp: str | None
) -> None:
    """Show latest results for an ensemble."""
    from llm_orc.core.config.config_manager import ConfigurationManager
    from llm_orc.core.config.state import ARTIFACTS_DIRNAME, resolve_state_dir
    from llm_orc.core.execution.artifact_manager import ArtifactManager

    local = ConfigurationManager(provision=False).local_config_dir
    manager = ArtifactManager(
        artifacts_dir=resolve_state_dir(local) / ARTIFACTS_DIRNAME
    )

    if execution_timestamp:
        results = manager.get_execution_results(ensemble_name, execution_timestamp)
    else:
        results = manager.get_latest_results(ensemble_name)

    if not results:
        click.echo(f"No artifacts found for ensemble '{ensemble_name}'", err=True)
        raise SystemExit(1)

    if format_type == "json":
        _format_json_output(results)
        return

    _display_artifact_text_format(ensemble_name, results)
