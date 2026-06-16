"""Value types for the agentic-serving benchmark (pure data).

Per ``docs/agentic-serving/benchmark-design.md`` §3 (grid), §4 (metrics).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Cell:
    """One benchmark task — a (horizon × complexity) grid cell or a probe cell.

    ``expected_deliverables`` are the filenames the task asks for (the convergence
    + form targets); the validation is derived from each name's extension. The
    content-coherence check infers cross-references from the produced files
    themselves (no dependency graph is declared here — §4).
    """

    name: str
    horizon: int  # rung, 1..N (deliverable count / dependency depth — §3)
    complexity: int  # rung, 1..N (per-deliverable difficulty — §3)
    prompt: str
    expected_deliverables: tuple[str, ...]
    kind: str = "grid"  # "grid" | "probe"


@dataclass(frozen=True)
class MetricRecord:
    """The deterministic metric record for one cell run (§4).

    The four hard-pass signals gate :attr:`passed`; the rest are reported.
    """

    # hard-pass signals
    form_valid: bool
    converged: bool
    content_coherent: bool
    terminated_clean: bool
    # reported-not-gating
    delegation_rate: float | None
    escalated: bool
    churn: int | None
    # observability
    produced: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        """A cell passes iff all four hard-pass signals hold (§4)."""
        return (
            self.form_valid
            and self.converged
            and self.content_coherent
            and self.terminated_clean
        )


@dataclass(frozen=True)
class CellResult:
    """A cell + the records of its runs (n≥1) + the resolved pass verdict.

    ``passed`` applies the pre-registered k/n threshold (§5): a cell at n>1
    passes iff at least ``ceil(2/3 · n)`` of its runs passed.
    """

    cell: Cell
    records: tuple[MetricRecord, ...]
    degraded: bool = False

    @property
    def n(self) -> int:
        return len(self.records)

    @property
    def passes_count(self) -> int:
        return sum(1 for r in self.records if r.passed)

    @property
    def passed(self) -> bool:
        """Pre-registered threshold (§5, P2-B): ≥2/3 of runs pass."""
        if not self.records:
            return False
        # ceil(2/3 · n) — n=1→1, n=3→2, n=5→4
        needed = -(-2 * self.n // 3)
        return self.passes_count >= needed
