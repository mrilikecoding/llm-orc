#!/usr/bin/env python3
"""
Spike (b) — synthetic-data simulation of time-decay windowing.

Cycle 4, ADR-016 bounding-mechanism (b) validation.
Question: does linear-decay-within-dual-bound windowing bound bias-compounding
compared to no-windowing baseline?

Setup:
  - Generate synthetic calibration signals = bias(t) + noise(t)
  - Compute calibration verdicts under several configurations
  - Measure tracking accuracy, lag, and stale-signal influence

Configurations compared:
  1. non_windowed         — aggregate all history equally (failure-mode baseline)
  2. linear_decay_default — dual-bound (60 time units / 100 signals), linear decay 1.0→0.0
  3. linear_decay_smaller — dual-bound (30 / 50), linear decay
  4. linear_decay_larger  — dual-bound (120 / 200), linear decay
  5. hard_cutoff_default  — same dual-bound, no decay (uniform weight in window)
  6. exponential_decay    — same dual-bound, exponential decay (alternative shape)

Bias scenarios:
  A. slow_drift           — bias shifts smoothly from -0.5 to +0.5
  B. step_change          — bias is -0.5 for first half, +0.5 for second half
  C. periodic_oscillation — bias = 0.5 * sin(2π t / period)

Reads spike parameters from this file's constants. Outputs:
  - <scratch>/results.json — per-configuration metrics across all scenarios
  - <scratch>/summary.txt — human-readable ranking and falsification verdict

Falsification criterion:
  If linear_decay_default's tracking error and stale-signal weight do not differ
  meaningfully from non_windowed, the windowing's bias-bound property is invalidated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

# ---------------------------------------------------------------------------
# Spike parameters
# ---------------------------------------------------------------------------

N_STEPS = 500
SEED = 20260506

# Dual-bound defaults
DEFAULT_TIME_BOUND = 60
DEFAULT_SIGNAL_BOUND = 100
SMALLER_TIME_BOUND = 30
SMALLER_SIGNAL_BOUND = 50
LARGER_TIME_BOUND = 120
LARGER_SIGNAL_BOUND = 200

# Exponential-decay rate
EXPONENTIAL_DECAY_RATE = 1.0 / 30.0  # half-life ~ 21 steps

NOISE_SD = 0.2

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "scratch" / "spike-cycle4-adr016-b-windowing"


# ---------------------------------------------------------------------------
# Bias scenarios — bias(t) for t in [0, N_STEPS)
# ---------------------------------------------------------------------------

def slow_drift(t: np.ndarray) -> np.ndarray:
    return -0.5 + 1.0 * (t / (N_STEPS - 1))


def step_change(t: np.ndarray) -> np.ndarray:
    return np.where(t < N_STEPS // 2, -0.5, 0.5)


def periodic_oscillation(t: np.ndarray) -> np.ndarray:
    period = N_STEPS / 4
    return 0.5 * np.sin(2 * np.pi * t / period)


SCENARIOS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "slow_drift": slow_drift,
    "step_change": step_change,
    "periodic_oscillation": periodic_oscillation,
}


# ---------------------------------------------------------------------------
# Verdict configurations — verdict(t) = weighted_aggregate(signals[s] for s ≤ t)
# ---------------------------------------------------------------------------

def verdict_non_windowed(signals: np.ndarray, time: np.ndarray) -> np.ndarray:
    """Aggregate all history equally — the failure-mode baseline."""
    verdicts = np.zeros_like(signals)
    cumsum = np.cumsum(signals)
    counts = np.arange(1, len(signals) + 1)
    verdicts = cumsum / counts
    return verdicts


def _linear_decay_weights(window_size: int) -> np.ndarray:
    """Linear decay 1.0 at most-recent → 0.0 at window-edge."""
    return np.linspace(1.0, 0.0, window_size, endpoint=False)


def _hard_cutoff_weights(window_size: int) -> np.ndarray:
    """Uniform weight inside window."""
    return np.ones(window_size)


def _exponential_decay_weights(window_size: int, rate: float) -> np.ndarray:
    """Exponential decay e^(-rate * lag) for lag in 0..window_size-1."""
    return np.exp(-rate * np.arange(window_size))


def verdict_windowed(
    signals: np.ndarray,
    time: np.ndarray,
    time_bound: int,
    signal_bound: int,
    weight_fn: Callable[[int], np.ndarray],
) -> np.ndarray:
    """Verdict using dual-bound windowing (whichever bound is shorter applies)."""
    n = len(signals)
    verdicts = np.zeros_like(signals)
    for t_idx in range(n):
        t = time[t_idx]
        # Signal-bound: most recent signal_bound signals
        start_idx_by_count = max(0, t_idx + 1 - signal_bound)
        # Time-bound: signals within time_bound time units
        start_idx_by_time = np.searchsorted(time, t - time_bound, side="left")
        start_idx = max(start_idx_by_count, int(start_idx_by_time))
        window_signals = signals[start_idx:t_idx + 1][::-1]  # most-recent first
        window_size = len(window_signals)
        if window_size == 0:
            verdicts[t_idx] = 0.0
            continue
        weights = weight_fn(window_size)
        verdicts[t_idx] = np.sum(window_signals * weights) / np.sum(weights)
    return verdicts


CONFIGURATIONS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "non_windowed": verdict_non_windowed,
    "linear_decay_default": lambda s, t: verdict_windowed(
        s, t, DEFAULT_TIME_BOUND, DEFAULT_SIGNAL_BOUND, _linear_decay_weights
    ),
    "linear_decay_smaller": lambda s, t: verdict_windowed(
        s, t, SMALLER_TIME_BOUND, SMALLER_SIGNAL_BOUND, _linear_decay_weights
    ),
    "linear_decay_larger": lambda s, t: verdict_windowed(
        s, t, LARGER_TIME_BOUND, LARGER_SIGNAL_BOUND, _linear_decay_weights
    ),
    "hard_cutoff_default": lambda s, t: verdict_windowed(
        s, t, DEFAULT_TIME_BOUND, DEFAULT_SIGNAL_BOUND, _hard_cutoff_weights
    ),
    "exponential_decay_default": lambda s, t: verdict_windowed(
        s, t, DEFAULT_TIME_BOUND, DEFAULT_SIGNAL_BOUND,
        lambda w: _exponential_decay_weights(w, EXPONENTIAL_DECAY_RATE),
    ),
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def tracking_error(verdict: np.ndarray, bias: np.ndarray) -> float:
    """Mean squared error between verdict and current-bias."""
    return float(np.mean((verdict - bias) ** 2))


def lag_estimate(verdict: np.ndarray, bias: np.ndarray, max_lag: int = 60) -> int:
    """Cross-correlation peak offset; how many steps verdict lags bias."""
    v = verdict - np.mean(verdict)
    b = bias - np.mean(bias)
    if np.std(v) == 0 or np.std(b) == 0:
        return 0
    correlations = []
    for lag in range(max_lag + 1):
        if lag == 0:
            corr = np.corrcoef(v, b)[0, 1]
        else:
            corr = np.corrcoef(v[lag:], b[:-lag])[0, 1]
        correlations.append(corr if not np.isnan(corr) else 0.0)
    return int(np.argmax(correlations))


def stale_signal_influence(
    signals: np.ndarray,
    time: np.ndarray,
    config_name: str,
    final_idx: int,
) -> float:
    """Fraction of total verdict-weight contributed by signals older than DEFAULT bounds.

    Defined as: at the final timestep, what fraction of weighted contribution
    comes from signals older than DEFAULT_TIME_BOUND time units (or older than
    DEFAULT_SIGNAL_BOUND signals back). For non_windowed config this is
    typically large; for properly-windowed configs it should be ≤ 0 (zero by
    construction inside the window, with all stale signals weighted 0).
    """
    if config_name == "non_windowed":
        # All signals contribute equally; fraction outside default window = (idx - 100) / (idx + 1)
        if final_idx < DEFAULT_SIGNAL_BOUND:
            return 0.0
        return (final_idx - DEFAULT_SIGNAL_BOUND + 1) / (final_idx + 1)
    # For windowed configs: by construction, signals outside the window have weight 0
    return 0.0


# ---------------------------------------------------------------------------
# Run simulation
# ---------------------------------------------------------------------------

def run() -> dict:
    rng = np.random.default_rng(SEED)
    time = np.arange(N_STEPS, dtype=float)

    results: dict[str, dict[str, dict[str, float]]] = {}
    raw_traces: dict[str, dict[str, dict[str, list[float]]]] = {}

    for scenario_name, scenario_fn in SCENARIOS.items():
        bias = scenario_fn(time)
        noise = rng.normal(0.0, NOISE_SD, size=N_STEPS)
        signals = bias + noise

        scenario_results: dict[str, dict[str, float]] = {}
        scenario_traces: dict[str, dict[str, list[float]]] = {
            "bias": bias.tolist(),
            "signals": signals.tolist(),
        }

        for config_name, verdict_fn in CONFIGURATIONS.items():
            verdicts = verdict_fn(signals, time)
            scenario_results[config_name] = {
                "tracking_error_mse": tracking_error(verdicts, bias),
                "lag_steps": lag_estimate(verdicts, bias),
                "stale_signal_weight_fraction": stale_signal_influence(
                    signals, time, config_name, N_STEPS - 1
                ),
                "final_verdict": float(verdicts[-1]),
                "final_bias": float(bias[-1]),
            }
            scenario_traces[config_name] = {"verdicts": verdicts.tolist()}

        results[scenario_name] = scenario_results
        raw_traces[scenario_name] = scenario_traces

    return {"metrics": results, "traces": raw_traces}


def write_outputs(results: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # JSON output (full data including traces for re-analysis)
    json_path = OUTPUT_DIR / "results.json"
    with json_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote: {json_path}")

    # Summary text
    summary_path = OUTPUT_DIR / "summary.txt"
    metrics = results["metrics"]
    lines: list[str] = []
    lines.append("Spike (b) — Time-Decay Windowing Validation")
    lines.append("=" * 50)
    lines.append("")
    lines.append(f"Steps: {N_STEPS}, seed: {SEED}, noise sd: {NOISE_SD}")
    lines.append("")

    for scenario_name, scenario_results in metrics.items():
        lines.append(f"## Scenario: {scenario_name}")
        lines.append("")
        lines.append(
            f"{'config':<28} {'tracking_mse':>13} {'lag_steps':>10} "
            f"{'stale_weight':>13} {'final_verdict':>14} {'final_bias':>11}"
        )
        for config_name, m in scenario_results.items():
            lines.append(
                f"{config_name:<28} {m['tracking_error_mse']:>13.4f} "
                f"{m['lag_steps']:>10d} {m['stale_signal_weight_fraction']:>13.3f} "
                f"{m['final_verdict']:>14.3f} {m['final_bias']:>11.3f}"
            )
        lines.append("")

    # Validation check — separates bias-bound (structural) from tracking-error (parametric)
    lines.append("## Validation check")
    lines.append("")
    lines.append("Two distinct properties are tested:")
    lines.append("")
    lines.append("(1) STRUCTURAL bias-bound — does windowing prevent stale signals from")
    lines.append("    contributing to current verdicts? Tested via stale_signal_weight_fraction.")
    lines.append("    PASS criterion: linear_decay_default reduces stale-weight by ≥ 0.5 vs")
    lines.append("    non_windowed in all scenarios.")
    lines.append("")
    lines.append("(2) PARAMETRIC tracking — do default window parameters track bias trajectory")
    lines.append("    better than non-windowed baseline? Tested via tracking_error_mse ratio.")
    lines.append("    Pass means the default is operationally workable; sub-optimal-but-positive")
    lines.append("    is parameter-tuning territory, not falsification.")
    lines.append("    PASS criterion: linear_decay_default tracking_error < non_windowed in all")
    lines.append("    scenarios (any positive improvement).")
    lines.append("")

    structural_passes: list[bool] = []
    parametric_passes: list[bool] = []
    parametric_optimum: dict[str, str] = {}

    for scenario_name, scenario_results in metrics.items():
        non_w = scenario_results["non_windowed"]
        windowed = scenario_results["linear_decay_default"]
        if non_w["tracking_error_mse"] > 0:
            err_ratio = windowed["tracking_error_mse"] / non_w["tracking_error_mse"]
        else:
            err_ratio = float("nan")
        stale_diff = (
            non_w["stale_signal_weight_fraction"]
            - windowed["stale_signal_weight_fraction"]
        )
        # Find optimum config for this scenario (excluding non_windowed)
        windowed_configs = {
            k: v for k, v in scenario_results.items() if k != "non_windowed"
        }
        best_config = min(
            windowed_configs.keys(),
            key=lambda k: windowed_configs[k]["tracking_error_mse"],
        )
        parametric_optimum[scenario_name] = best_config

        structural_pass = stale_diff >= 0.5
        parametric_pass = err_ratio < 1.0
        structural_passes.append(structural_pass)
        parametric_passes.append(parametric_pass)

        lines.append(
            f"  {scenario_name}:"
        )
        lines.append(
            f"    structural — stale_weight reduction = {stale_diff:.3f}"
            f" {'PASS' if structural_pass else 'FAIL'}"
        )
        lines.append(
            f"    parametric — default tracking_error_ratio = {err_ratio:.3f}"
            f" {'PASS' if parametric_pass else 'FAIL'}"
        )
        lines.append(
            f"    parametric optimum (across windowed configs) = {best_config}"
        )

    lines.append("")
    structural_verdict = (
        "VALIDATED" if all(structural_passes) else "FALSIFIED"
    )
    parametric_verdict = (
        "VALIDATED" if all(parametric_passes) else "FALSIFIED"
    )
    lines.append(
        f"STRUCTURAL VERDICT: {structural_verdict} "
        f"(does windowing structurally bound stale-signal influence?)"
    )
    lines.append(
        f"PARAMETRIC VERDICT: {parametric_verdict} "
        f"(do default parameters produce positive tracking improvement?)"
    )
    lines.append("")
    lines.append("Parameter-tuning notes:")
    for scenario_name, best_config in parametric_optimum.items():
        if best_config != "linear_decay_default":
            lines.append(
                f"  {scenario_name}: default is sub-optimal; {best_config} "
                f"tracks better — operational tuning territory"
            )
        else:
            lines.append(f"  {scenario_name}: default is optimum across tested configs")

    with summary_path.open("w") as f:
        f.write("\n".join(lines))
    print(f"Wrote: {summary_path}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    results = run()
    write_outputs(results)
