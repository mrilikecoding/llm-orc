# Arm-0 run 8 — #121 merge validation row (2026-08-13)

The live real-OpenCode regression row for the content-grep branch
(`feat/121-content-grep`, self_reference ON), per the delegation
contract: serve on the branch, standard 13-turn ladder, qwen3:8b,
fixture cloned from the run-7 seed repo (all four seed sha256 match the
run-6 known-good baseline).

**Purpose and scope:** REGRESSION validation, not a parity-column entry.
The #121 feature evidence is the corpus/caller/endpoint suites (three
design review rounds + implementation review round 1 fixed) plus the
targeted live gate (`docs/plans/2026-08-13-121-live-gate/`). **Not
J-scored** — instrument-side facts only; do not pool into any strict
column.

## Mechanical record

All 13 turns exit 0, no client deaths, no oracle crashes. Instrument
dishonest flags: **0**. 16 request rounds; wall 1514s (run 7: 1412s,
run 6: 1451s — same rig class). Turn 13 fix-execution landed live:
suite AND seeded target green (`truth-13.json` rc 0/0).

Oracle tally **2/1/0** (shipped-correct / shipped-broken / not-shipped
on turns 6,7 / 1 / —): the best arm-0 oracle row across runs 5–8 (run
5: 1/0/2, run 6: 2/0/1, run 7: 1/1/1). Turn 7's compose-with-storage
probe passed for the first time in this run family. Turn 1's todo.py
again missed the add-to-existing-list probe (the recurring n=1 class;
recorded, not diagnosed per doctrine 6).

**Grep/self-path non-interference (the regression claim this row exists
for):** the serve trace ledger holds exactly 5 `need-grep` routings and
2 `need-self-files` routings — all from the targeted live gates, none
from the battery. No ladder turn's explain fall-through carried a
complete listing into the grep trigger; the rung is inert on the ladder
exactly as designed.

Timing note: battery turns ran while review-round-1 fixes (`cbea33c`)
landed mid-run (scripts re-read per turn). Every surface those fixes
touched is grep-phase-only, and no battery turn routed through the grep
phase (above), so the row is valid for the merged result.
