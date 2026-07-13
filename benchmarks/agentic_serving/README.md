# Agentic-Serving Benchmark

A robust, repeatable, $0-local-default benchmark for the tool-driven multi-turn
serving surface. Full design + rationale: **`docs/agentic-serving/benchmark-design.md`**
(read it first — it is the build-and-run plan; this README is the quick reference).

Read three ways from one horizon × complexity grid: **regression** (per-cell
pass/fail), **axis-2 ceiling** (highest passing cell), **tier comparison**
(cheap-local vs frontier).

## Layout

| File | Role | Tested |
|------|------|--------|
| `model.py` | value types — `Cell` / `MetricRecord` / `CellResult` (pass threshold `ceil(2n/3)`) | — |
| `scorer.py` | pure `(workspace, serve-log slice, cell) → MetricRecord` | unit |
| `scorecard.py` | `results → heatmap / ceiling / boundary cells / match verdict` | unit |
| `corpus.py` | the 16 H×C grid tasks + the bleed-injection probe cells | unit |
| `runner.py` | drives one cell live (uuid4 marker, log slice, opencode → serve) | live |
| `bench.py` | CLI — coarse → confirm → concentrate orchestration + scorecard | unit (pure parts) |

### WS-8 standing parity measurement (#131)

A second, separate scoring path for the 13-turn conversational ladder
(`ladder_battery.sh`) — not the H×C grid above. Design:
`docs/plans/2026-07-13-parity-scoreboard-design.md`.

| File | Role | Tested |
|------|------|--------|
| `transcript.py` | the arm-agnostic transcript IR (`Transcript`/`Turn`/`ToolCall`) every arm normalizes into | unit |
| `honesty.py` | pure `Turn → TurnVerdict` — verification behavior + dishonest-outcome classification | unit |
| `metrics.py` | pure aggregate metrics — rounds consumed, wall-clock, cost (`Pricing`) | unit |

No per-arm adapter (raw client transcript → `Transcript`) exists yet — see
the design doc's "Not built here" section for why, and what the next
session needs to capture first.

## Run the harness unit tests (deterministic, CI-safe)

```sh
uv run pytest benchmarks/agentic_serving/tests/ -o addopts=""
```

(`-o addopts=""` skips the `llm_orc` coverage gate — the benchmark is not `llm_orc`.)

## Run a grid ($0 cheap-local — the default)

Pre-flight: **restart ollama fresh**; confirm `qwen3:8b` + `qwen3:14b` are pulled;
`opencode` available. Then:

```sh
uv run python -m benchmarks.agentic_serving.bench --config cheap-local
```

By default the CLI manages its own `serve` on port 8770 (`--no-manage-serve` drives
an already-running serve; `--serve-port` to change it). It runs a degradation smoke,
then coarse (n=1) → ceiling-confirm (n=3) → concentrate (n=3 at boundary cells), and
writes a scorecard (markdown + JSON sidecar) under `benchmark-runs/`.

The full grid is **hours** of local model time — run it when the stack is free.

## Tier comparison (frontier arm — opt-in, cost-gated)

```sh
uv run python -m benchmarks.agentic_serving.bench --compare cheap-local,frontier \
    --i-accept-frontier-cost
```

Without `--i-accept-frontier-cost` the CLI prints the cost estimate and exits before
any spend (free-first). The frontier config is selected by how `serve` is launched
(`.llm-orc/config.yaml` + profiles); `--config`/`--compare` record which named config
ran, as provenance.

## Escalation / bleed probe

```sh
uv run python -m benchmarks.agentic_serving.bench --probe bleed-injection
```

Runs the probe cells under an adversarial coder prompt + a 2B→8B (or 0.6B→8B) tier
ladder — the live convergence-via-escalation evidence (ADR-041 §6).
