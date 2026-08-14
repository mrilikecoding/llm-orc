"""Interval estimates for the parity table (#63's v2 slice).

Parity table v2 cites interval estimates instead of bare small-n counts
(roadmap ws8: "realism rows; interval estimates"). Pure-python (no scipy
on the rig): Wilson score intervals for per-column proportions, Fisher's
exact test for two-column comparisons. Known-value pins: the Wilson
closed form at p̂=1 (lower = n/(n+z²)) and Fisher's classic
lady-tasting-tea table from the literature.
"""

from __future__ import annotations

import pytest

from benchmarks.agentic_serving.stats import fisher_exact, wilson_interval


def test_wilson_at_perfect_score_matches_the_closed_form() -> None:
    # p-hat = 1: lower bound collapses to n / (n + z^2); upper is 1.
    low, high = wilson_interval(39, 39)
    assert high == pytest.approx(1.0)
    assert low == pytest.approx(39 / (39 + 1.959964**2), abs=1e-6)


def test_wilson_interval_known_value() -> None:
    # 11/13 at 95%: computed from the Wilson formula independently.
    low, high = wilson_interval(11, 13)
    assert low == pytest.approx(0.5777, abs=2e-4)
    assert high == pytest.approx(0.9567, abs=2e-4)


def test_wilson_at_zero_is_the_mirror_of_perfect() -> None:
    low, high = wilson_interval(0, 39)
    assert low == pytest.approx(0.0)
    assert high == pytest.approx(1 - 39 / (39 + 1.959964**2), abs=1e-6)


def test_wilson_rejects_impossible_counts() -> None:
    with pytest.raises(ValueError):
        wilson_interval(14, 13)
    with pytest.raises(ValueError):
        wilson_interval(-1, 13)
    with pytest.raises(ValueError):
        wilson_interval(0, 0)


def test_fisher_exact_lady_tasting_tea() -> None:
    # The classic [[3,1],[1,3]] table: one-sided p = 0.242857...,
    # two-sided p = 0.485714... (literature values).
    two_sided = fisher_exact(3, 4, 1, 4)
    assert two_sided == pytest.approx(0.485714, abs=1e-5)


def test_fisher_exact_identical_columns_is_one() -> None:
    assert fisher_exact(13, 13, 13, 13) == pytest.approx(1.0)


def test_fisher_exact_extreme_difference_is_small() -> None:
    # 0/39 vs 39/39 — the most extreme 2x2 at this n.
    p = fisher_exact(0, 39, 39, 39)
    assert p < 1e-20
