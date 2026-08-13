# Arm-0 run 6 — #148 merge validation row (2026-08-13)

The live real-OpenCode validation row for the truncated-listing-refuse
merge (`5a59515`), per the delegation contract: serve restarted on merged
main, standard 13-turn ladder, qwen3:8b, fresh fixture verified against
the known-good truth-00 baseline.

**Purpose and scope:** REGRESSION validation, not a new parity-column
entry. The #148 guard itself cannot fire on this fixture (no stem family
exceeds the 50-path cap); the guard's feature evidence is the 238-case
routing corpus with its mutation nets (three adversarial review rounds,
2 blockers + 2 majors found and closed). This row shows the serve still
routes end-to-end on the merged code. **Not J-scored** — instrument-side
facts only; do not pool into any strict column.

## Mechanical record

All 13 turns exit 0, no client deaths, no oracle crashes, no
contamination. Instrument dishonest flags: **0**. Oracle tally 2/0/1
(shipped-correct / shipped-broken / not-shipped on turns 1/6/7). 14
request rounds; wall 1451s (run 5: 2040s — same rig, sampling and load
variance). Turn 13 fix-execution landed live: suite AND seeded target
green client-side (`truth-13.json` suite.rc 0, seeded.rc 0).

Comparison anchor: run 5 (`docs/plans/2026-08-12-arm0-run5/`) was
11/13 J-scored, zero dishonest, oracle 1/0/2, on pre-#148 main plus the
same seed baseline.
