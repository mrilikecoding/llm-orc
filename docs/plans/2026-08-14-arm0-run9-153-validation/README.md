# Arm-0 run 9 — #153 merge validation row (2026-08-14)

The live real-OpenCode regression row for the offset-continuation reads
branch (`feat/153-offset-reads`), per the delegation contract: serve on
the branch, standard 13-turn ladder, qwen3:8b, fixture cloned from the
run-6-verified seed baseline.

**Purpose and scope:** REGRESSION validation, not a parity-column entry.
The #153 feature evidence is the read_stitch suite (18 tests incl. the
real captured trailer and exact-body stitch equality) plus the targeted
live gate (`docs/plans/2026-08-14-153-live-gate/` — the 80KB caller
grounded via a 2-part stitch, converting the #121 recorded bound). **Not
J-scored.**

## Mechanical record

All 13 turns exit 0, no client deaths, no oracle crashes. Instrument
dishonest flags: **0**. 15 request rounds; wall 1164s (runs 6/7/8:
1451/1412/1514 — fastest of the family). Turn 13 fix-execution landed
live (suite AND seeded rc 0). Oracle tally **2/0/1** (turns 6,7 correct;
turn 1 not shipped — the recurring turn-1 n=1 class, this time as a
non-ship rather than a broken ship).

**Continuation non-interference:** the serve trace ledger holds exactly
ONE `read_continuation` row — the targeted live gate's stitch of the
80KB caller — and none from the battery (no ladder file exceeds the
client's 50KB cap, so the seam is inert on the ladder as designed).
