"""Agentic-serving benchmark — horizon × complexity ladder.

Per ``docs/agentic-serving/benchmark-design.md``. A robust, repeatable,
$0-local-default benchmark for the tool-driven multi-turn serving surface, read
three ways: regression (per-cell pass/fail), axis-2 ceiling (highest passing
cell), north-star tier comparison (cheap-local vs frontier).

Four units, clean boundaries:

* :mod:`benchmarks.agentic_serving.model` — value types (``Cell``,
  ``MetricRecord``, ``CellResult``). Pure data.
* :mod:`benchmarks.agentic_serving.scorer` — ``(workspace, log-slice, cell) →
  MetricRecord``. Pure function of artifacts; deterministic; unit-tested.
* :mod:`benchmarks.agentic_serving.scorecard` — ``results → heatmap / ceiling /
  boundary cells / match verdict``. Pure; unit-tested.
* :mod:`benchmarks.agentic_serving.runner` — drives one cell live (workspace,
  unique session marker, serve, slice capture). Exercised live, not in CI.

The corpus (the grid + the §6 bleed-injection probe cells) lives in
:mod:`benchmarks.agentic_serving.corpus`.
"""
