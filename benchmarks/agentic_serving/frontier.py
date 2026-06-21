"""The frontier arm — a Claude Sonnet subagent comparator (§7).

The tier comparison (the benchmark's centerpiece) sets the cheap orchestrated
stack against an expensive frontier model on the same cells. The frontier model
is a Claude Sonnet subagent given the cell task, writing the deliverables into a
per-cell workspace — ``[cheap + framework]`` vs ``[frontier, no orchestration]``,
the value-proposition reading of the cycle's central question.

This module owns the *pure* parts: the prompt the subagent is given, and scoring
a populated workspace into a :class:`CellResult`. The live dispatch is an
in-session step (the Agent tool, ``model=sonnet``) — ``python -m bench`` is a
plain process with no Agent tool, so the frontier arm is gathered in-session and
scored here, not run autonomously by the CLI.

Scoring uses :func:`scorer.score_frontier`: the three file-derived hard signals
(form / convergence / coherence) score identically to the cheap arm;
loop-termination is N/A to a one-shot model (§4).
"""

from __future__ import annotations

from pathlib import Path

from benchmarks.agentic_serving.model import Cell, CellResult
from benchmarks.agentic_serving.scorer import score_frontier


def frontier_prompt(cell: Cell) -> str:
    """The task for a Sonnet frontier subagent (§7).

    The cell's task plus an explicit instruction to write each expected file into
    the working directory, so the subagent's output lands where
    :func:`score_cell` reads it. The deliverable names are listed explicitly so
    the produced set matches the scorer's expectations.
    """
    files = ", ".join(cell.expected_deliverables)
    return (
        f"{cell.prompt}\n\n"
        f"Write each of these files into the current working directory: {files}. "
        "Produce the files only — no commentary."
    )


def score_cell(workspace: Path, cell: Cell) -> CellResult:
    """Score a Sonnet-populated frontier workspace into a CellResult (n=1, §7)."""
    record = score_frontier(workspace, cell)
    return CellResult(cell=cell, records=(record,))
