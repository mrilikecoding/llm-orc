# Compound ladder rung A×B results — depth × carry-side (2026-06-08)

$0 local, qwen3:14b via Ollama /v1. Real production composition (judge call 1 →
strip_verdict → `_seat_filler_messages(remaining_anchor)` → seat) through the real
`decide()`. Task: **read config.py, then write settings_loader.py +
test_settings_loader.py + validator.py** (one read [setup] + three write
deliverables — axis B's read-then-write, deepened from 2 writes to 3).

Pre-registered (probe.py docstring) + methods-reviewed twice: the A×C plan was
redirected to A×B (`research-methods-compound-axc.md`, P3), and the A×B redesign
was confirmed trustworthy before the run (`research-methods-compound-axb.md`,
no P1; prior P1/P2 findings discharged).

## Result

| State (done) | write-advance | read events | premature-finish (no_tool) | read-churn | judge clean-naming | verdict | turn_shape |
|--------------|---------------|-------------|----------------------------|------------|--------------------|---------|------------|
| R0 (nothing) | read-first **9/10** (1 jumped to a write) | — | 0/10 | — | — | (first turn) | carry 9 / gen 1 |
| R1 (read) | **14/15** delegated writes | 1 re-read config | 0/15 | 1/15 | 14/15 | REMAINING 10/10 | gen 14 / carry 1 |
| R2 (read+module) | **12/15** (11 delegated + 1 inline) | 1 read-of-next-file | 2/15 | 0/15 | **15/15** | REMAINING 10/10 | gen 12 / carry 3 |
| R3 (read+module+test, the deep state) | **13/15** delegated | 0 | 2/15 | 0/15 | **15/15** | REMAINING 10/10 | gen 13 / carry 2 |
| RC (read+3 writes) | finish 10/10 | — | (correct finish) | 0/10 | n/a | COMPLETE 10/10 | carry 10 |

n: R0/RC = 10, R1/R2/R3 = 15 (the discriminating states, powered per the P1 fix).

## Verdict — PASS. The carry-side holds under depth; no interaction-effect limit.

The methods review's open question #1 (do failures surface only when axes
compound?) is answered for A×B on this surface: **no.** Compounding depth (axis A)
with the leading read (axis B) does not break what either single axis passed.

1. **Read-first holds: 9/10 (R0).** The seat-filler reads config.py before
   writing — at the pre-registered ≥9/10 boundary; a 1-sample dip from axis-B's
   10/10 at depth-2 (one sample jumped straight to a delegated write). Within
   boundary, plausibly noise at n=10.
2. **Advance holds near the 0.9 baseline at depth-3-after-read:** R1 14/15 (0.93),
   R2 12/15 (0.80), R3 13/15 (0.87) — all ≥ the pre-registered 12/15 (0.80) holds
   line. R1/R3 are clear holds; R2 sits exactly at the boundary (see the §middle-band
   caveat). Advance does NOT degrade monotonically with depth.
3. **Carry-side integrity holds across THREE remaining states (the headline).** The
   read is never re-pulled (read-churn: 1/15 at R1 — the same minor re-read axis B
   saw at depth-2; 0 at R2/R3) and the judge keeps treating the read as context,
   never a deliverable: clean-naming 14/15 (R1) then **15/15 (R2 and R3)** — it
   names only unproduced WRITES, never the read or a produced file. FC-61 holds
   under depth.
4. **Convergence clean with a read + 3 writes in the record:** COMPLETE 10/10 at
   RC, zero false-continue. The deeper record does not confuse the judge.
5. **Meter (WP-LB-M) validated live across all 65 decisions** — the real-model
   acceptance of the just-committed outcome-derived stamping, and of the
   bidirectional bug axis B documented: R0 reads → `carry` (was `generation`
   pre-WP-LB-M), R1–R3 delegated/inline writes → `generation` (was `carry`), RC
   finish → `carry`. Zero anchor-phrasing fragility (no `boundary_excluded`
   leakage — no repair on this rung).

## The one elevated signal — premature finish on deep REMAINING states

no_tool rose from 0/15 (R1) to **2/15 (R2) and 2/15 (R3)** (~13%): the seat-filler
occasionally finishes despite the REMAINING anchor "produce the next file." This is
consistent with axis-B's ~10% no-tool baseline and does NOT worsen monotonically
with depth (R2 = R3). It is the recorded premature-finish risk, backstopped by the
next re-judgment (re-judges REMAINING, re-anchors) + the AS-3 cap. Not a limit, but
the closest thing the compound surfaced — worth watching on deeper / live rungs.

## Middle-band caveat (pre-registered, methods-review P2)

R2 write-advance = 12/15 (0.80) sits in the 0.73–0.90 ambiguous band: at n=15 the
holds/degradation split at 12/15 cannot cleanly separate p=0.80 from p=0.73. R1
(14/15) and R3 (13/15) are unambiguous holds; R2 is "holds, at the boundary." If a
sharper read of R2 is wanted, n=25–30 on R2 alone would resolve it — but R3 (the
deeper state) holding at 13/15 makes a genuine depth-driven degradation unlikely
(degradation would deepen with depth, not dip at R2 and recover at R3).

## Harness note (measurement imperfection, corrected in interpretation)

`_classify_action` counts advance as "target is an unproduced deliverable,"
regardless of whether the action was a write or a read. R2 row 4 was a *read* of
the not-yet-written test file (a read-then-write step mid-sequence, on-theme for
this rung, not a failure) — over-counted as an advance, so the raw advance=13/15
is 12/15 on a production-only definition. Recorded transparently; R1/R3 are
unaffected (their non-write carries were the config re-read [not a deliverable]
and finishes). For future compound rungs: gate advance on a write action.

## Disposition

A×B passes — the first compound rung holds. Carry-side-under-depth is now
characterized: no interaction-effect limit on qwen3:14b for the file-action subset.
The cloud-contrast trigger (any REMAINING-state advance ≤ 0.73) did NOT fire → no
cloud spend. Candidate next rungs (practitioner-directed): A×C (depth + repair —
the repair-churn thread, now with the powered n + pre-registered boundary the
methods review requires); deeper depth (4–5 writes); axis D (mid-session intent
refinement); or the live multi-turn trajectory run (the state-injection limitation
shared by all rungs — a BUILD-phase OpenCode validation per the A×B review's P3).
