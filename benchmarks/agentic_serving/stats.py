"""Interval estimates for the parity table (#63's parity-v2 slice).

Parity table v2 cites interval estimates instead of bare small-n counts
(roadmap ws8). Pure python — the rig carries no scipy, and at battery
sizes (n <= 39) exact combinatorics are trivial.

Method choices, deliberate and minimal:
- **Wilson score interval** for a per-column proportion: well-behaved at
  the boundary scores the batteries actually produce (39/39, 0/39),
  where the Wald interval degenerates; the closed form at p-hat = 1 is
  lower = n / (n + z^2), which the tests pin.
- **Fisher's exact test** (two-sided, minimum-likelihood convention) for
  comparing two columns: exact at these n, no asymptotics.
Anything beyond this (power analysis, corrections for many comparisons)
waits until the table actually makes many comparisons — the #63 wishlist
stays open; this is the slice v2 needs.
"""

from __future__ import annotations

from math import comb, sqrt

_Z_95 = 1.959964


def wilson_interval(
    successes: int, trials: int, z: float = _Z_95
) -> tuple[float, float]:
    """The Wilson score interval for ``successes``/``trials``.

    Raises ``ValueError`` on impossible counts — a scoring bug must fail
    loudly, never render as a plausible interval.
    """
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError(f"impossible counts: {successes}/{trials}")
    p_hat = successes / trials
    z_sq = z * z
    denominator = 1 + z_sq / trials
    center = p_hat + z_sq / (2 * trials)
    half = z * sqrt(p_hat * (1 - p_hat) / trials + z_sq / (4 * trials * trials))
    return ((center - half) / denominator, (center + half) / denominator)


def fisher_exact(
    successes_a: int, trials_a: int, successes_b: int, trials_b: int
) -> float:
    """Two-sided Fisher's exact p-value for the 2x2 table
    [[successes_a, failures_a], [successes_b, failures_b]] under the
    minimum-likelihood convention (sum the probabilities of every table
    with the same margins whose probability does not exceed the observed
    table's)."""
    if not (0 <= successes_a <= trials_a and 0 <= successes_b <= trials_b):
        raise ValueError("impossible counts")
    total = trials_a + trials_b
    total_successes = successes_a + successes_b

    def table_probability(k: int) -> float:
        return (
            comb(trials_a, k)
            * comb(trials_b, total_successes - k)
            / comb(total, total_successes)
        )

    k_min = max(0, total_successes - trials_b)
    k_max = min(trials_a, total_successes)
    observed = table_probability(successes_a)
    tolerance = observed * 1e-9
    return sum(
        probability
        for k in range(k_min, k_max + 1)
        if (probability := table_probability(k)) <= observed + tolerance
    )
