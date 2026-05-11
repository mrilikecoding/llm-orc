# Research Log 005e — Spike on ADR-016 Bounding Mechanism (b): Time-Decay Windowing

**Cycle:** 4
**Phase:** DECIDE (Tranche-C ADR-016 validation, per practitioner Path-2 authorization 2026-05-06)
**Date:** 2026-05-06
**Type:** Synthetic-data spike — logic-validation of the windowing specification
**Mechanism under test:** ADR-016 bounding mechanism (b), time-decay windowing for the bias-compounding horizon

---

## Spike question

Does the linear-decay-within-dual-bound-window specification — provisional in ADR-016 as 60-time-units / 100-signals dual-bound, linear decay from 1.0 at most-recent to 0.0 at window-edge — actually bound bias compounding compared to a no-windowing baseline?

The cycle's load-bearing literature evidence on feedback-bias compounding is Khanal et al.'s universal non-improvement finding (arXiv:2603.29231): episodic memory augmentation produced no improvement across 10 tested models at long horizons, with six models showing negative effects. The mechanism Khanal et al. identify is feedback-bias compounding when stale signals influence current decisions. ADR-016's mechanism (b) is the proposed bound.

The spike does not validate operational fit (would require real-deployment evidence). It validates the *logic* of the windowing specification — whether the structural mechanism produces the bias-bound property the literature evidence motivates.

---

## Method

**Setup.** A 500-step synthetic calibration data flow. Each signal at step t is the sum of an underlying bias trajectory `bias(t)` and i.i.d. Gaussian noise (sd 0.2). Three bias scenarios: slow drift (smooth trajectory from −0.5 to 0.5); step change (−0.5 for first half, 0.5 for second half); periodic oscillation (period 125, amplitude 0.5).

**Configurations compared.** Six aggregation strategies for computing calibration verdicts at each step:

1. `non_windowed` — equal-weighted aggregation of all history (failure-mode baseline)
2. `linear_decay_default` — dual-bound 60 time-units / 100 signals, linear decay 1.0 → 0.0 (the ADR-016 specification under test)
3. `linear_decay_smaller` — dual-bound 30 / 50, linear decay
4. `linear_decay_larger` — dual-bound 120 / 200, linear decay
5. `hard_cutoff_default` — same dual-bound as default, uniform weight inside window (no decay)
6. `exponential_decay_default` — same dual-bound as default, exponential decay (rate 1/30, half-life ≈ 21 steps)

**Two distinct validation tests:**

- **Structural bias-bound.** Does the windowing prevent stale signals from contributing to current verdicts? Measured via the fraction of weighted contribution coming from signals older than the default window at the final timestep. Pass criterion: stale-weight reduction ≥ 0.5 vs non-windowed.

- **Parametric tracking.** Do the default window parameters produce a positive tracking-error reduction vs non-windowed baseline? Pass criterion: default `linear_decay_default` tracking_error_mse < non_windowed tracking_error_mse in all scenarios.

The two tests separate the structural property (does the mechanism work?) from the parametric property (are the default values right?).

---

## Findings

### Structural bias-bound: VALIDATED

| Scenario | non_windowed stale-weight | windowed stale-weight | Reduction |
|----------|---------------------------|----------------------|-----------|
| slow_drift | 0.800 | 0.000 | 0.800 |
| step_change | 0.800 | 0.000 | 0.800 |
| periodic_oscillation | 0.800 | 0.000 | 0.800 |

The windowing **structurally eliminates** stale-signal contribution. By construction (and as the metric confirms), signals outside the window have weight 0 in the verdict; non-windowed aggregation gives them equal weight to recent signals. The 80% reduction is uniform across all three bias scenarios — the structural property does not depend on the bias trajectory shape.

This is the load-bearing finding for ADR-016. The mechanism's bias-bound property is validated at the logical level. Whether the bound is operationally sufficient to prevent compounding under real-deployment conditions is empirical territory beyond synthetic-data validation, but the *logical* bound is established.

### Parametric tracking: VALIDATED

| Scenario | non_windowed tracking_mse | linear_decay_default tracking_mse | Ratio (windowed/non) |
|----------|--------------------------|-----------------------------------|---------------------|
| slow_drift | 0.0862 | 0.0036 | 0.042 |
| step_change | 0.2588 | 0.0253 | 0.098 |
| periodic_oscillation | 0.1221 | 0.0763 | 0.625 |

The default parameters produce positive tracking-error reduction in all scenarios. The reduction is large for slow drift (96%) and step change (90%), modest for periodic oscillation (37%). The pattern is consistent with the windowing specification's purpose: bounded windows respond faster to bias trajectory changes, and the response speed dominates tracking quality when the bias changes are large or sudden.

### Parameter sensitivity (operational tuning territory)

Across all three bias scenarios, `linear_decay_smaller` (dual-bound 30 / 50, linear decay) is the parametric optimum:

| Scenario | linear_decay_smaller tracking_mse | linear_decay_default tracking_mse | Smaller / Default |
|----------|-----------------------------------|-----------------------------------|-------------------|
| slow_drift | 0.0027 | 0.0036 | 0.75 |
| step_change | 0.0147 | 0.0253 | 0.58 |
| periodic_oscillation | 0.0247 | 0.0763 | 0.32 |

The smaller-window configuration consistently tracks better. The largest gap is in periodic oscillation, where the period (125 steps) exceeds the default window (100 signals); the smaller window (50 signals) tracks the periodic bias closely because it does not aggregate across multiple periods.

`hard_cutoff_default` (no decay inside window) tracks worse than `linear_decay_default` in slow_drift (0.0056 vs 0.0036) and periodic (0.1472 vs 0.0763). Linear decay's performance advantage over hard cutoff is meaningful: weighting recent signals more heavily produces better tracking than uniform weighting inside the window.

`exponential_decay_default` performs comparably to `linear_decay_default` across all scenarios (within 5% on tracking_error_mse). The decay-shape choice is secondary to the windowing's existence.

### What the spike does *not* establish

- **Operational window-shape correctness for real deployments.** The synthetic bias trajectories are not calibrated to any actual ensemble-output distribution; they are stress-tests of the structural mechanism. The 100-signal default may or may not be appropriate for real ensemble dispatch frequencies.

- **Multi-iteration scale behavior.** The simulation runs for 500 steps; cycle's North-Star benchmark involves multi-session work where the bias dynamics may differ qualitatively from synthetic single-trajectory drift.

- **Interaction with mechanisms (a), (c), (d), (e).** The spike isolates mechanism (b). The full bounding-mechanism set may compose in ways the isolated test does not capture (e.g., out-of-band audit (d) may detect parameter drift that windowing alone cannot).

- **Whether the calibration *consumer's* response to windowed signals is correct.** The verdict (Proceed / Reflect / Abstain per ADR-014) is the consumer; the spike measures verdict-tracking-bias correlation, not whether the verdict produces correct dispatch decisions in real ensemble work.

These limitations are honest scope conditions for the spike, not falsifications of its findings.

---

## Implications for ADR-016

### Mechanism (b) status: provisionally validated at the logical level

The structural bias-bound property is validated. The mechanism (b) specification — linear-decay-within-dual-bound — produces the bias-compounding bound the literature evidence motivates. Mechanism (b)'s conditional-acceptance status in ADR-016 can be relaxed at the logical-validation level: the mechanism is structurally sound rather than provisional-without-evidence.

The conditional-acceptance status remains appropriate for *operational* validation — whether the mechanism's bias-bound property holds in real deployments, whether the default parameters fit deployment-realistic bias dynamics, and whether the full bounding-mechanism set composes correctly in practice are all questions the synthetic-data spike does not address. BUILD-time and first-deployment evidence remain the natural validation surface for those questions.

### Operational tuning notes for BUILD

The parametric optimum across all tested scenarios is `linear_decay_smaller` (30 time-units / 50 signals). The default specification in ADR-016 (60 / 100) tracks worse than smaller in all tested scenarios. Two implications:

1. **The default may be too large for deployment-realistic bias dynamics.** Empirical tuning during BUILD against deployment-realistic dispatch frequencies should explore smaller window sizes as candidates.

2. **The default's choice of 60/100 was drafting-time synthesis** without empirical anchor. Updating the default to a more empirically-supported value (or simply documenting that operational tuning is required at deployment) is a low-cost ADR-016 revision.

### Decay-shape note for BUILD

Linear decay outperforms hard cutoff across scenarios. Exponential decay is comparable to linear; the decay-shape choice between linear and exponential is operationally low-stakes. Hard cutoff (uniform weight in window) is structurally adequate but parametrically inferior; the linear-decay specification in ADR-016 should hold rather than be relaxed to "any window-shape inside the dual-bound."

### Composition with mechanism (d)

The spike's most consequential gap is mechanism (b) does not on its own detect *parameter drift* — i.e., when the bias dynamics shift such that the current window-shape parameters no longer track well. Mechanism (d), the periodic out-of-band audit dispatch, is the natural detector for this case. The spike's parameter-sensitivity finding (smaller is better across all tested scenarios) reinforces the value of mechanism (d) — without periodic audit, the deployment cannot detect when its window parameters need adjustment.

This suggests mechanisms (b) and (d) are more tightly coupled than ADR-016's drafting suggested: (b) is the bias-bound mechanism, (d) is the *meta*-bound mechanism that detects when (b)'s parameters need tuning. The composition is not "two independent bounding mechanisms" but "a primary bounding mechanism plus a parameter-drift detector."

### Falsification trigger status

The synthetic-data spike does not trigger the falsification clause in ADR-016. The mechanism produces the bias-bound property; the elaboration-by-evidence framing commitment from research-gate Grounding Action 2 remains in force. Operational validation in BUILD or first-deployment evidence remains the next validation gate.

---

## Spike artifacts retained per corpus retention policy

Per cycle-status §"Conformance Notes" — Spike artifacts retention (Cycle 3 directive, applies to corpus until close):

- `.llm-orc/scripts/spike_adr016_b_time_decay_windowing.py` (the simulation script)
- `scratch/spike-cycle4-adr016-b-windowing/results.json` (full per-scenario metrics + traces)
- `scratch/spike-cycle4-adr016-b-windowing/summary.txt` (human-readable structural and parametric verdict)
