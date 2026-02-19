"""Validation CLI commands."""

import asyncio
from typing import Any

import click

from llm_orc.cli_commands import _determine_ensemble_directories, _find_ensemble_config
from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.core.execution.ensemble_execution import EnsembleExecutor


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
        EnsembleExecutionResult,
        ValidationConfig,
        ValidationEvaluator,
    )

    # Initialize configuration manager
    config_manager = ConfigurationManager()

    # Determine ensemble directories
    ensemble_dirs = _determine_ensemble_directories(config_manager, config_dir)

    # Find ensemble configuration
    ensemble_config = _find_ensemble_config(ensemble_name, ensemble_dirs)

    # Check if ensemble has validation configuration
    if not hasattr(ensemble_config, "validation") or ensemble_config.validation is None:
        click.echo(
            f"Ensemble '{ensemble_name}' does not have validation configuration.",
            err=True,
        )
        raise SystemExit(1)

    # Create executor and run in test mode
    executor = EnsembleExecutor()

    click.echo(f"Validating ensemble: {ensemble_name}")
    click.echo("\u2500" * 50)

    try:
        # Execute ensemble in test mode
        # TODO: Implement test_mode flag in executor
        result_dict = asyncio.run(
            executor.execute(ensemble_config, "validation test input")
        )

        # Convert result dict to EnsembleExecutionResult
        # Extract execution time from metadata
        metadata = result_dict.get("metadata", {})
        execution_time = metadata.get("completed_at", 0.0) - metadata.get(
            "started_at", 0.0
        )

        # Convert agent outputs, parsing JSON responses
        agent_outputs = {}
        for agent_name, agent_result in result_dict.get("results", {}).items():
            response = agent_result.get("response", {})
            # If response is a string, try to parse as JSON first
            if isinstance(response, str):
                try:
                    import json

                    agent_outputs[agent_name] = json.loads(response)
                except (json.JSONDecodeError, ValueError):
                    # Not JSON, wrap in dict
                    agent_outputs[agent_name] = {"output": response}
            else:
                agent_outputs[agent_name] = response

        execution_result = EnsembleExecutionResult(
            ensemble_name=result_dict.get("ensemble_name", ensemble_name),
            execution_order=result_dict.get("execution_order", []),
            agent_outputs=agent_outputs,
            execution_time=execution_time,
        )

        # Parse validation config from dict to ValidationConfig
        validation_config = ValidationConfig.model_validate(ensemble_config.validation)

        # Run validation evaluation
        evaluator = ValidationEvaluator()
        validation_result = asyncio.run(
            evaluator.evaluate(
                ensemble_name=ensemble_name,
                results=execution_result,
                validation_config=validation_config,
            )
        )

        # Display results
        _display_validation_result(validation_result, verbose)

        # Set exit code based on validation result
        if validation_result.passed:
            raise SystemExit(0)
        else:
            raise SystemExit(1)

    except NotImplementedError as e:
        click.echo("Validation execution not yet fully implemented.", err=True)
        raise SystemExit(1) from e
    except Exception as e:
        click.echo(f"Validation failed with error: {str(e)}", err=True)
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
