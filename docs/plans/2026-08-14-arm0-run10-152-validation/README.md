# Arm-0 run 10 — #152 merge validation row (2026-08-14)

The live real-OpenCode regression row for the fail-closed routing branch
(`fix/152-routing-fail-closed`): serve on the branch, standard 13-turn
ladder, qwen3:8b, fixture cloned from the run-6-verified seed baseline
(all four seed hashes matched before turn 1).

**Purpose and scope:** REGRESSION validation, not a parity-column entry.
The #152 feature evidence is the shape/emit/form_gate pin suites plus
the targeted live gate (`docs/plans/2026-08-14-152-live-gate/` — the
misfire-conditions replay refusing honestly through real OpenCode).
**Not J-scored.**

## Mechanical record

All 13 turns exit 0, no client deaths, no oracle crashes. Instrument
dishonest flags: **0**. 12 request rounds; wall 1312s (runs 6-9:
1451/1412/1514/1164 — in family). Turn 13 fix-execution landed live
(suite AND seeded rc 0). **Zero `Refused: serving pipeline error`
refusals — the new readability gate refused nothing on a healthy
serve**, which is the row's regression claim.

Oracle tally **0/0/3** (shipped-correct / shipped-broken / not-shipped):
turn 1 and turn 6 were rejected by the pre-existing ACCEPT gate
("Another round needed: tests did not pass" — the recurring qwen3:8b
build-variance class; turn 1 has non-shipped in 2 of the last 3 runs),
and turn 7's refusal ("Build refused: could not read todo.py: client
read failed") is the deterministic cascade of turn 1's non-ship — the
file it needed never existed. Family context: 1/1/1, 2/1/0, 2/0/1,
2/0/1 across runs 6-9; the weakest delivery row of the family but every
non-delivery is honest, correctly prefixed, and attributable to seat
variance, not routing. Delivery variance rides its own cell by scorer
design ("refusing is a delivery failure, not a correctness failure").

## Interruptions (instrument integrity note)

Two earlier attempts at this row were killed by harness/session-limit
task reaping, not by the instrument: attempt 1 died during turn 13
(turns 1-12 all exit 0), attempt 2 during turn 2. Both are preserved
under `discarded-battery-152-out-interrupted{,-2}` in the session
scratchpad with notes; the fixture was re-cloned and hash-verified
against the run-6 baseline before each relaunch. This complete run was
executed detached (nohup) so the harness could not reap it.
