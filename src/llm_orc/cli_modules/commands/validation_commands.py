"""Validation CLI commands."""

import asyncio
from pathlib import Path
from typing import Any

import click

from llm_orc.cli_commands import _find_ensemble_config
from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.core.execution.executor_factory import ExecutorFactory
from llm_orc.services.orchestra_service import OrchestraService


def _build_execution_result(result_dict: dict[str, Any], ensemble_name: str) -> Any:
    """Convert raw execution dict to EnsembleExecutionResult."""
    import json

    from llm_orc.core.validation import EnsembleExecutionResult

    metadata = result_dict.get("metadata", {})
    execution_time = metadata.get("completed_at", 0.0) - metadata.get("started_at", 0.0)

    agent_outputs: dict[str, Any] = {}
    for name, agent_result in result_dict.get("results", {}).items():
        response = agent_result.get("response", {})
        if isinstance(response, str):
            try:
                agent_outputs[name] = json.loads(response)
            except (json.JSONDecodeError, ValueError):
                agent_outputs[name] = {"output": response}
        else:
            agent_outputs[name] = response

    return EnsembleExecutionResult(
        ensemble_name=result_dict.get("ensemble_name", ensemble_name),
        execution_order=result_dict.get("execution_order", []),
        agent_outputs=agent_outputs,
        execution_time=execution_time,
    )


def _resolve_ensemble_dirs(
    config_dir: str | None,
    config_manager: ConfigurationManager,
) -> list[Path]:
    """Resolve ensemble directories from config_dir or config_manager."""
    if config_dir is not None:
        return [Path(config_dir)]
    dirs = config_manager.get_ensembles_dirs()
    if not dirs:
        click.echo(
            "No ensemble directories found. "
            "Run 'llm-orc config init' to set up local configuration.",
            err=True,
        )
        raise SystemExit(1)
    return dirs


def validate_ensemble(
    ensemble_name: str, verbose: bool, config_dir: str | None
) -> None:
    """Validate a single ensemble in test mode.

    Args:
        ensemble_name: Name of the ensemble to validate
        verbose: Show detailed validation output
        config_dir: Custom config directory path

    Raises:
        SystemExit: Exit with code 0 on pass, 1 on fail
    """
    from llm_orc.core.validation import (
        ValidationConfig,
        ValidationEvaluator,
    )

    config_manager = ConfigurationManager()
    service = OrchestraService(config_manager=config_manager)
    ensemble_dirs = _resolve_ensemble_dirs(config_dir, config_manager)
    ensemble_config = _find_ensemble_config(ensemble_name, ensemble_dirs, service)

    if not hasattr(ensemble_config, "validation") or ensemble_config.validation is None:
        click.echo(
            f"Ensemble '{ensemble_name}' does not have validation configuration.",
            err=True,
        )
        raise SystemExit(1)

    executor = ExecutorFactory.create_root_executor()

    click.echo(f"Validating ensemble: {ensemble_name}")
    click.echo("\u2500" * 50)

    try:
        result_dict = asyncio.run(
            executor.execute(ensemble_config, "validation test input")
        )
        execution_result = _build_execution_result(result_dict, ensemble_name)
        validation_config = ValidationConfig.model_validate(ensemble_config.validation)

        evaluator = ValidationEvaluator()
        validation_result = asyncio.run(
            evaluator.evaluate(
                ensemble_name=ensemble_name,
                results=execution_result,
                validation_config=validation_config,
            )
        )

        _display_validation_result(validation_result, verbose)
        raise SystemExit(0 if validation_result.passed else 1)

    except SystemExit:
        raise
    except NotImplementedError as e:
        click.echo(
            "Validation execution not yet fully implemented.",
            err=True,
        )
        raise SystemExit(1) from e
    except Exception as e:
        click.echo(f"Validation failed with error: {e!s}", err=True)
        raise SystemExit(1) from e


def validate_ensemble_category(category: str, verbose: bool) -> None:
    """Validate all ensembles in a category.

    Args:
        category: Validation category name
        verbose: Show detailed validation output

    Raises:
        SystemExit: Exit with code 0 on pass, 1 on fail
    """
    click.echo(f"Validating category: {category}")
    click.echo("Not yet implemented", err=True)
    raise SystemExit(1)


def validate_all_ensembles(verbose: bool) -> None:
    """Validate all validation ensembles.

    Args:
        verbose: Show detailed validation output

    Raises:
        SystemExit: Exit with code 0 on pass, 1 on fail
    """
    click.echo("Validating all ensembles")
    click.echo("Not yet implemented", err=True)
    raise SystemExit(1)


def _display_layer_result(layer: str, result: Any) -> None:
    """Display a single validation layer result."""
    if result is None:
        return
    status = "PASS" if result.passed else "FAIL"
    click.echo(f"  {layer}: {status}")
    for error in result.errors or []:
        click.echo(f"    - {error}")


def _display_validation_result(validation_result: Any, verbose: bool) -> None:
    """Display validation result with formatting.

    Args:
        validation_result: ValidationResult from evaluator
        verbose: Show detailed output
    """
    status = "PASSED" if validation_result.passed else "FAILED"
    click.echo(f"Validation {status}")

    if verbose:
        click.echo("\nValidation Layer Results:")
        for layer, result in validation_result.results.items():
            _display_layer_result(layer, result)
