"""What the CLI prints when a run is refused before any agent starts.

The refusal envelope is the one REST and MCP return (Arc 4, ruling 10):
``error.kind``, ``error.message`` and, for ``not_equipped``, the
dependency report. JSON mode prints the envelope; the other modes print
the message and the report as a table.
"""

import json
from collections.abc import Mapping, Sequence
from typing import Any

import click
from rich.console import Console
from rich.table import Table

_COLUMNS = ("kind", "name", "status", "via", "resolve")


def display_refusal(envelope: Mapping[str, Any], output_format: str | None) -> None:
    """Print a refused run's envelope in the caller's output format."""
    if output_format == "json":
        click.echo(json.dumps(envelope, indent=2, default=str))
        return
    error = envelope.get("error") or {}
    click.echo(f"Run refused ({error.get('kind')}): {error.get('message')}")
    rows = _rows(error.get("dependencies") or [])
    if not rows:
        return
    if output_format == "text":
        _plain_table(rows)
    else:
        _rich_table(rows)


def display_preflight(document: Mapping[str, Any], output_format: str | None) -> bool:
    """Print a preflight answer: the whole dependency table, met rows
    included, or the JSON answer. A refusal envelope is shown as a refusal.
    Returns whether the answer is a failure (not runnable, or a refusal)."""
    error = document.get("error")
    if isinstance(error, Mapping) and "kind" in error:
        display_refusal(document, output_format)
        return True
    runnable = bool(document.get("runnable"))
    if output_format == "json":
        click.echo(json.dumps(document, indent=2, default=str))
        return not runnable
    click.echo(f"Preflight: {'runnable' if runnable else 'not runnable'}")
    bindings = document.get("bindings") or {}
    if bindings:
        applied = ", ".join(f"{key} -> {target}" for key, target in bindings.items())
        click.echo(f"Bindings that would apply: {applied}")
    rows = _rows(document.get("dependencies") or [])
    if rows and output_format == "text":
        _plain_table(rows)
    elif rows:
        _rich_table(rows)
    if document.get("pull_requested"):
        click.echo(
            "A run with --pull would download the pullable models; nothing was pulled"
        )
    return not runnable


def display_run_record(
    bindings: Mapping[str, str],
    pulled: Sequence[str],
    output_format: str | None,
    persisted: str | None = None,
) -> None:
    """Name the bindings applied, the models pulled and the bundle stored,
    when there are any."""
    if output_format == "json":
        return
    if bindings:
        applied = ", ".join(f"{key} -> {target}" for key, target in bindings.items())
        click.echo(f"Bindings applied: {applied}")
    if pulled:
        click.echo(f"Models pulled: {', '.join(pulled)}")
    if persisted:
        click.echo(f"Bundle persisted: {persisted}")


def _rows(dependencies: Sequence[Mapping[str, Any]]) -> list[tuple[str, ...]]:
    return [
        (
            str(d.get("kind", "")),
            str(d.get("name", "")),
            str(d.get("status", "")),
            " > ".join(str(v) for v in d.get("via") or []),
            str(d.get("resolve", "")),
        )
        for d in dependencies
    ]


def _plain_table(rows: Sequence[tuple[str, ...]]) -> None:
    widths = [
        max(len(cell) for cell in col) for col in zip(_COLUMNS, *rows, strict=True)
    ]
    for row in (_COLUMNS, *rows):
        line = "  ".join(
            cell.ljust(width) for cell, width in zip(row, widths, strict=True)
        )
        click.echo(line.rstrip())


def _rich_table(rows: Sequence[tuple[str, ...]]) -> None:
    table = Table()
    for column in _COLUMNS:
        table.add_column(column, overflow="fold")
    for row in rows:
        table.add_row(*row)
    Console(width=max(Console().width, 120)).print(table)
