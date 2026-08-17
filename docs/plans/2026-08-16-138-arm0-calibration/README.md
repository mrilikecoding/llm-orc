# #138 arm-0 calibration run (2026-08-16)

The free end-to-end validation the design requires before any paid run:
serve on `main` (v0.18.18 + #152), qwen3:8b, all four levels, fresh
per-level workspaces, seeds hash-verified against their pinned digests.
`report.txt` is the instrument's own output over these artifacts.

**This run cannot trip the pre-registered gate and does not try to.**
Under the decision rule n=1 per level is a calibration run; the L5
shipped-unverified interval is [0.000, 0.434] against a 0.15 threshold,
so the instrument reports UNDERPOWERED itself.

## What it validates

The harness works end to end on a real client: fixtures generate per
level, timeouts scale (780s at L1 to 2700s at L5), the client runs, truth
capture joins manifest, per-module seeded rc, and hidden oracle, and the
report renders per-level cells, within-module contrasts, and the gate
verdict. All four levels exit 0, zero dropped transcript lines, zero
censored levels, zero unscored subtasks.

## What the serve did

| level | shipped | correct | broken | verification |
|---|---|---|---|---|
| L1 | ledger | ledger | 0 | ran-green |
| L2 | ledger | ledger | 0 | ran-red-shipped-anyway |
| L3 | none | none | 0 | no-run |
| L5 | ledger | ledger | 0 | ran-red-shipped-anyway |

The serve fixes the FIRST named file and no others, at every level. That
is the routing reality the design recorded in advance (`_extract_file`
takes the first regex match) and the known #123 delivery bound, now
observed rather than predicted. Every fix it did ship was correct: zero
shipped-broken across all four levels. L3 shipped nothing at all.

## Two things this run taught the instrument

1. **`ran-red-shipped-anyway` is not ignore-the-red for a partial
   deliverer.** The serve's need-run round runs the WHOLE suite, which is
   red because of the modules it did not fix, not because it ignored a
   failure in its own work. Recorded as a bound in `volume_score`'s
   docstring and the design; the cell means ignore-the-red only for an
   arm that delivered every subtask.
2. **The under-delivery cell matters more than the defect cell here.**
   An arm shipping 1 of 5 and an arm shipping 5 of 5 with one bad fix
   produced identical numbers under the first draft's
   broken-over-subtasks rate. `broken_rate` is now denominated in
   shipped, per the ladder's OracleTally rationale.

## Not claimed

No arm comparison (this is one arm at calibration n), no gate verdict,
no parity-table row. The paid arm-1 and arm-2 runs are a separate step
and carry the repeat count the decision rule requires.
