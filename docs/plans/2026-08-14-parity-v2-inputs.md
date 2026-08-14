# Parity table v2 inputs (#63 slice, #141 + #138 pre-registrations)

## 1. Interval estimates (#63's v2 slice — DELIVERED)

`benchmarks/agentic_serving/stats.py`: Wilson score intervals (boundary-
safe at the perfect/zero columns the batteries actually produce) and
Fisher's exact test (exact at battery n, minimum-likelihood two-sided).
Pure python, TDD'd against closed-form and literature pins. The #63
wishlist (power analysis, multiple-comparison corrections) stays open;
this is the slice v2 needs.

95% intervals over the recorded columns (v1 caveats still apply —
versions differ across rows; do not pool):

| column | count | Wilson 95% |
|---|---|---|
| Arm 0 post-fix strict (run 5, J-scored) | 11/13 | [0.578, 0.957] |
| Arm 1a Haiku 4.5 strict | 38/39 | [0.868, 0.995] |
| Arm 1b Sonnet strict | 39/39 | [0.910, 1.000] |
| Arm 2a Haiku 4.5 strict | 35/39 | [0.764, 0.959] |
| Arm 2b Sonnet strict | 39/39 | [0.910, 1.000] |
| Arm 2a dishonest | 4/39 | [0.041, 0.236] |
| every other arm's dishonest | 0/39 | [0.000, 0.090] |

Key comparisons (Fisher exact, two-sided):

- **The honesty headline is not yet statistically separable.** Arm 2a's
  4/39 dishonest vs Arm 1a's 0/39: p = 0.115. The "same Haiku, harness-
  sensitive honesty" headline is directionally supported and every
  scorer's notes agree on the mechanism, but at n=39 per cell Fisher
  cannot exclude chance. Parity table v2 must carry this as an interval/
  p-value, and closing it needs either more runs (~n=80+ per cell for
  power at this effect size) or an asterisk that stays.
- Arm 0 strict (11/13) vs Sonnet's perfect 39/39: p = 0.059 — the serve's
  strict column is statistically indistinguishable from the frontier
  columns at current n (0 vs Arm 2a: p = 0.63). The honest v2 framing:
  every strict interval overlaps; the discriminating axes must come from
  the realism rows (#138) and honesty-at-scale, not the toy battery.

## 2. #141 pre-registration — the CLAUDE.md confound (Arm 2)

Design (per the issue, sharpened): one Arm-2 model (Haiku 4.5 — the
cell where the AGENTS.md literature predicts the largest swing), 13-turn
battery, TWO conditions in one continuing conversation construct each:
**Developer** (as-is: the practitioner's global CLAUDE.md stack) vs
**None** (global CLAUDE.md suppressed). Same seeds, same
capture_truth.sh per turn, both J-scored independently.

- Pre-registered gate (from the issue): |strict difference| ≤ 1 turn →
  the confound is noise; footnote with evidence. ≥ 2 turns → v2 needs a
  normalized rerun or an asterisk on the 13/13 headline. Honesty column
  compared alongside with the Fisher machinery above.
- **Practitioner decision required before running:** the None condition
  suppresses `~/.claude/CLAUDE.md` for the run (move-aside or a HOME
  override for the driving session). That manipulates the operator's
  live config, so it is not run autonomously; the spike is ready to
  execute on a go signal. Cost rides the subscription (issue note).

## 3. #138 pre-registration — shipped-broken vs task volume

Design (per the issue): 3–5 task-volume levels (concurrent changes per
turn), frontier arm(s) + the serve on identical levels, hidden-oracle
instrument, verification behavior (did tests run before shipping?) and
shipped-broken rate per level.

- Pre-registered gate (from the issue): flat frontier skip-rate and
  shipped-broken across levels → the toy-battery negative generalizes;
  a rising rate crossing >15% shipped-unverified at the largest level →
  the frontier-failure-at-scale hypothesis confirmed and v2 gets its
  discriminating rows.
- **Prerequisite work (the real cost):** a volume-laddered task set with
  per-level hidden oracles does not exist yet — it is new instrument
  content of roughly the original battery's build effort, and the
  battery header's seeding/truth discipline applies. Paid runs (Arm 1
  via opencode go is pre-authorized within reason; estimate at v1 rates:
  well under $5 for n=1 per level per model). Sequenced after #141
  (which reuses the EXISTING battery and needs only a go signal).

## Sequencing

#63 slice: done (this doc + stats.py). #141: ready, blocked on the
practitioner's None-condition go. #138: instrument build first (a design
doc + oracle set), then paid runs. Parity table v2 lands after #141
resolves and carries intervals for every column regardless of #138's
timing.
