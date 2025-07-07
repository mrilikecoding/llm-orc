"""Command line interface for llm-orc."""

import os
from pathlib import Path

import click

from llm_orc.ensemble_config import EnsembleLoader


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
def invoke(ensemble_name: str, config_dir: str):
    """Invoke an ensemble of agents."""
    if config_dir is None:
        # Default to ~/.llm-orc/ensembles if no config dir specified
        config_dir = os.path.expanduser("~/.llm-orc/ensembles")
    
    click.echo(f"Invoking ensemble: {ensemble_name}")
    click.echo(f"Looking in config directory: {config_dir}")
    
    loader = EnsembleLoader()
    ensemble_config = loader.find_ensemble(config_dir, ensemble_name)
    
    if ensemble_config is None:
        raise click.ClickException(f"Ensemble '{ensemble_name}' not found in {config_dir}")
    
    click.echo(f"Found ensemble: {ensemble_config.description}")
    click.echo(f"Agents: {len(ensemble_config.agents)}")
    
    # For now, just show what we would do
    # TODO: Implement actual ensemble execution
    raise click.ClickException("Ensemble execution not implemented yet")


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


if __name__ == "__main__":
    cli()
