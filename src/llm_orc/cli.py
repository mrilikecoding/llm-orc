"""Command line interface for llm-orc."""

import click


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
    click.echo(f"Invoking ensemble: {ensemble_name}")
    if config_dir:
        click.echo(f"Looking in config directory: {config_dir}")
    # For now, just fail with a message about ensemble not existing
    raise click.ClickException(f"Ensemble '{ensemble_name}' not found")


@cli.command("list-ensembles")
def list_ensembles():
    """List available ensembles."""
    click.echo("Available ensembles:")
    click.echo("  (no ensembles configured yet)")


if __name__ == "__main__":
    cli()
