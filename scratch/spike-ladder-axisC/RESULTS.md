# Ladder axis C results — repair-shaped (2026-06-08)

$0 local, qwen3:14b via Ollama /v1, n=10/state. Real production composition
(judge → anchor → seat) through the real `decide()`. Task: fix string_utils.py
(count_vowels misses uppercase, buggy module inlined) + write test_string_utils.py.
Two deliverables: the repaired module (delivered as a write) + the test.

## Result

| State (done) | next action | advance | churn | delegated | judge verdict | turn_shape |
|--------------|-------------|---------|-------|-----------|---------------|------------|
| RC0 (nothing) | repair-write module 10/10 | 10/10 | 0 | 10/10 | (first turn) | **boundary_excluded 10/10** |
| RC1 (module fixed) | write test 8/10, re-module 2/10 | 8/10 | **2/10** | 10/10 | REMAINING 10/10 | carry 8 / boundary_excluded 2 |
| RCc (fixed+test) | finish 10/10 | — | 0 | 0 | COMPLETE 10/10 | carry 10/10 |

## Verdict — PASS (repair mechanism + P1 accounting hold), one minor churn tail

1. **The repair flow engages and delivers.** RC0: the seat-filler writes the
   fixed module 10/10, delegated 10/10. `boundary_excluded` 10/10 — the repair
   instruction ("fix the bug …") triggers it; the third turn shape, completing
   the A/B/C meter evidence.
2. **The P1 repair-turn accounting resolution holds empirically.** A repair
   delivered as a write of the target file is counted as that deliverable
   produced: RC1 judge REMAINING 10/10 (module fix done → the test remains);
   RCc judge COMPLETE 10/10 (both done) → finish. The existing accounting
   standard ("a successful write counts") recognizes the repair under the
   current write-only mapping — no judge-prompt change needed this rung.
3. **Convergence clean** (COMPLETE 10/10).

## Minor finding — the first churn in the ladder (repair-specific)

RC1 re-targets the already-fixed module **2/10** (vs 0/10 churn across axes A
and B). The "fix the bug in string_utils.py" framing keeps the module salient
even after the fix, occasionally re-pulling next-action selection back to it.
Advance is still 8/10 and the churn is backstopped by re-judgment (the next
trailing turn re-judges REMAINING and re-anchors) + the AS-3 cap. A real
repair-specific limitation, not a blocker — the remaining-work anchor (ADR-038)
is slightly less decisive on repair tasks than on clean writes (8/10 vs 9/10).

## Meter finding — the third shape, and direct evidence of anchor-phrasing fragility

`boundary_excluded` appears correctly on the repair turn (RC0 10/10). But RC1 is
the clincher for the redesign: the *same state* stamps `carry` 8/10 and
`boundary_excluded` 2/10 — the difference is only whether the judge's REMAINING
statement happened to contain "fix". The turn shape is being read off the
judge's surface phrasing, which is non-deterministic. The shape must come from
the framework's own per-turn knowledge (the action taken + the instruction's
boundary/observed-carry nature), not the anchor text.

## Disposition

Axis C passes; all three escalation axes (A deliverable-count, B mixed
read-write, C repair) hold on qwen3:14b. The ladder's mechanism question is
answered for this surface: delegate + advance + converge holds across depth,
mixed, and repair shapes, with only a minor repair churn tail. The meter
`turn_shape` redesign is now fully grounded across all three shapes — a
WP-LB-J follow-up (hybrid action+instruction classification) with its own
TDD cycle.
