# Arm-0 run 7 — #144 merge validation row (2026-08-13)

The live real-OpenCode regression row for the serve-native dot-dir
self-reference branch (`feat/144-dot-dir-self-reference`), per the
delegation contract: serve on the branch (self_reference flag ON in this
repo's committed config), standard 13-turn ladder, qwen3:8b, fixture
seeds verified byte-identical to the run-6 known-good truth-00 baseline
(all four sha256 match; seeds reconstructed from run-6's captured reads).

**Purpose and scope:** REGRESSION validation, not a parity-column entry.
The #144 feature evidence is the corpus/caller/endpoint suites (design
pre-flight with 3 blockers closed + adversarial review round 1) plus the
targeted live gate (`docs/plans/2026-08-13-144-live-gate/`). This row
shows the ladder still routes end-to-end with the flag ON. **Not
J-scored** — instrument-side facts only; do not pool into any strict
column.

## Mechanical record

All 13 turns exit 0, no client deaths, no oracle crashes. Instrument
dishonest flags: **0**. 15 request rounds; wall 1412s (run 6: 1451s, 14
rounds — same rig class). Turn 13 fix-execution landed live: suite AND
seeded target green client-side (`truth-13.json` suite.rc 0, seeded.rc 0).

Oracle tally 1/1/1 (shipped-correct / shipped-broken / not-shipped on
turns 6/1/7). Turn 1's `todo.py` shipped but its API missed the oracle's
add-to-existing-list probe — the first shipped-broken on an arm-0 oracled
turn across runs 5–7 (run 5: 1/0/2, run 6: 2/0/1). Turn 1 is a pure
build-seat rung untouched by this branch; per doctrine 6 this is a
per-turn sample at n=1, recorded, not diagnosed.

**Self-path non-interference (the regression claim this row exists
for):** the serve trace ledger shows exactly 2 `need-self-files` routings
and 2 `self_read_round` re-entries across its whole history — both from
the targeted live gate, none from the battery. With the flag ON, no
ladder turn's stems subset-match a serve-owned script; the union is inert
on the ladder exactly as designed.

Timing note: battery turns 1–12 ran against the branch at the live-gate
commit; the round-1 review fixes (`ac419af`) landed mid-run (scripts are
re-read per turn). Every surface those fixes touched is self-path-only,
and no battery turn routed through the self path (above), so the row is
valid for the merged result.
