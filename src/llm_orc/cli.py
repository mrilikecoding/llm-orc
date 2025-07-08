"""Command line interface for llm-orc."""

import asyncio
import json
import os
import sys
from pathlib import Path

import click

from llm_orc.authentication import AuthenticationManager, CredentialStorage
from llm_orc.ensemble_config import EnsembleLoader
from llm_orc.ensemble_execution import EnsembleExecutor


@click.group()
@click.version_option()
def cli():
    """LLM Orchestra - Multi-agent LLM communication system."""
    pass


@cli.command()
@click.argument("ensemble_name")
@click.option(
    "--config-dir",
    default=None,
    help="Directory containing ensemble configurations",
)
@click.option(
    "--input-data",
    default=None,
    help="Input data for the ensemble (if not provided, reads from stdin)",
)
@click.option(
    "--output-format",
    type=click.Choice(["json", "text"]),
    default="text",
    help="Output format for results",
)
def invoke(ensemble_name: str, config_dir: str, input_data: str, output_format: str):
    """Invoke an ensemble of agents."""
    if config_dir is None:
        # Default to ~/.llm-orc/ensembles if no config dir specified
        config_dir = os.path.expanduser("~/.llm-orc/ensembles")

    # Handle input from stdin if not provided via --input
    if input_data is None:
        if not sys.stdin.isatty():
            # Read from stdin (piped input)
            input_data = sys.stdin.read().strip()
        else:
            # No input provided and not piped, use default
            input_data = "Please analyze this."

    loader = EnsembleLoader()
    ensemble_config = loader.find_ensemble(config_dir, ensemble_name)

    if ensemble_config is None:
        raise click.ClickException(
            f"Ensemble '{ensemble_name}' not found in {config_dir}"
        )

    if output_format == "text":
        click.echo(f"Invoking ensemble: {ensemble_name}")
        click.echo(f"Description: {ensemble_config.description}")
        click.echo(f"Agents: {len(ensemble_config.agents)}")
        click.echo(f"Input: {input_data}")
        click.echo("---")

    # Execute the ensemble
    async def run_ensemble():
        executor = EnsembleExecutor()
        return await executor.execute(ensemble_config, input_data)

    try:
        result = asyncio.run(run_ensemble())

        if output_format == "json":
            click.echo(json.dumps(result, indent=2))
        else:
            # Text format - show readable output
            click.echo(f"Status: {result['status']}")
            click.echo(f"Duration: {result['metadata']['duration']}")

            # Show usage summary
            if "usage" in result["metadata"]:
                usage = result["metadata"]["usage"]
                totals = usage.get("totals", {})
                click.echo("\nUsage Summary:")
                click.echo(f"  Total Tokens: {totals.get('total_tokens', 0):,}")
                click.echo(f"  Total Cost: ${totals.get('total_cost_usd', 0.0):.4f}")
                click.echo(f"  Agents: {totals.get('agents_count', 0)}")

                # Show per-agent usage
                agents_usage = usage.get("agents", {})
                if agents_usage:
                    click.echo("\nPer-Agent Usage:")
                    for agent_name, agent_usage in agents_usage.items():
                        tokens = agent_usage.get("total_tokens", 0)
                        cost = agent_usage.get("cost_usd", 0.0)
                        duration = agent_usage.get("duration_ms", 0)
                        model = agent_usage.get("model", "unknown")
                        click.echo(
                            f"  {agent_name} ({model}): {tokens:,} tokens, "
                            f"${cost:.4f}, {duration}ms"
                        )

                # Show synthesis usage if present
                synthesis_usage = usage.get("synthesis", {})
                if synthesis_usage:
                    tokens = synthesis_usage.get("total_tokens", 0)
                    cost = synthesis_usage.get("cost_usd", 0.0)
                    duration = synthesis_usage.get("duration_ms", 0)
                    model = synthesis_usage.get("model", "unknown")
                    click.echo(
                        f"  synthesis ({model}): {tokens:,} tokens, "
                        f"${cost:.4f}, {duration}ms"
                    )

            click.echo("\nAgent Results:")
            for agent_name, agent_result in result["results"].items():
                if agent_result["status"] == "success":
                    click.echo(f"  {agent_name}: {agent_result['response']}")
                else:
                    click.echo(f"  {agent_name}: ERROR - {agent_result['error']}")

            if result.get("synthesis"):
                click.echo(f"\nSynthesis: {result['synthesis']}")

    except Exception as e:
        raise click.ClickException(f"Ensemble execution failed: {str(e)}") from e


@cli.command("list-ensembles")
@click.option(
    "--config-dir",
    default=None,
    help="Directory containing ensemble configurations",
)
def list_ensembles(config_dir: str):
    """List available ensembles."""
    if config_dir is None:
        # Default to ~/.llm-orc/ensembles if no config dir specified
        config_dir = os.path.expanduser("~/.llm-orc/ensembles")

    loader = EnsembleLoader()
    ensembles = loader.list_ensembles(config_dir)

    if not ensembles:
        click.echo(f"No ensembles found in {config_dir}")
        click.echo("  (Create .yaml files with ensemble configurations)")
    else:
        click.echo(f"Available ensembles in {config_dir}:")
        for ensemble in ensembles:
            click.echo(f"  {ensemble.name}: {ensemble.description}")


@cli.group()
def auth():
    """Authentication management commands."""
    pass


@auth.command("add")
@click.argument("provider")
@click.option("--api-key", required=True, help="API key for the provider")
@click.option("--config-dir", default=None, help="Config directory path")
def auth_add(provider: str, api_key: str, config_dir: str):
    """Add API key authentication for a provider."""
    config_path = Path(config_dir) if config_dir else None
    storage = CredentialStorage(config_path)

    try:
        storage.store_api_key(provider, api_key)
        click.echo(f"API key for {provider} added successfully")
    except Exception as e:
        raise click.ClickException(f"Failed to store API key: {str(e)}") from e


@auth.command("list")
@click.option("--config-dir", default=None, help="Config directory path")
def auth_list(config_dir: str):
    """List configured authentication providers."""
    config_path = Path(config_dir) if config_dir else None
    storage = CredentialStorage(config_path)

    try:
        providers = storage.list_providers()
        if not providers:
            click.echo("No authentication providers configured")
        else:
            click.echo("Configured providers:")
            for provider in providers:
                click.echo(f"  {provider}: API key")
    except Exception as e:
        raise click.ClickException(f"Failed to list providers: {str(e)}") from e


@auth.command("remove")
@click.argument("provider")
@click.option("--config-dir", default=None, help="Config directory path")
def auth_remove(provider: str, config_dir: str):
    """Remove authentication for a provider."""
    config_path = Path(config_dir) if config_dir else None
    storage = CredentialStorage(config_path)

    try:
        # Check if provider exists
        if provider not in storage.list_providers():
            raise click.ClickException(f"No authentication found for {provider}")

        storage.remove_provider(provider)
        click.echo(f"Authentication for {provider} removed")
    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(f"Failed to remove provider: {str(e)}") from e


@auth.command("test")
@click.argument("provider")
@click.option("--config-dir", default=None, help="Config directory path")
def auth_test(provider: str, config_dir: str):
    """Test authentication for a provider."""
    config_path = Path(config_dir) if config_dir else None
    auth_manager = AuthenticationManager(config_path)
    storage = CredentialStorage(config_path)

    try:
        api_key = storage.get_api_key(provider)
        if not api_key:
            raise click.ClickException(f"No authentication found for {provider}")

        if auth_manager.authenticate(provider, api_key):
            click.echo(f"Authentication for {provider} is working")
        else:
            raise click.ClickException(f"Authentication for {provider} failed")
    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(f"Failed to test authentication: {str(e)}") from e


@auth.command("setup")
@click.option("--config-dir", default=None, help="Config directory path")
def auth_setup(config_dir: str):
    """Interactive setup wizard for authentication."""
    config_path = Path(config_dir) if config_dir else None
    storage = CredentialStorage(config_path)

    click.echo("Welcome to LLM Orchestra setup!")
    click.echo("This wizard will help you configure authentication for LLM providers.")
    click.echo()

    while True:
        provider = click.prompt("Provider name (e.g., anthropic, google, openai)")
        api_key = click.prompt("API key", hide_input=True)

        try:
            storage.store_api_key(provider, api_key)
            click.echo(f"✓ {provider} configured successfully")
        except Exception as e:
            click.echo(f"✗ Failed to configure {provider}: {str(e)}")

        if not click.confirm("Add another provider?"):
            break

    click.echo()
    click.echo(
        "Setup complete! You can now use 'llm-orc auth list' to see your "
        "configured providers."
    )


if __name__ == "__main__":
    cli()
