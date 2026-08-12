# WS-8: the first parity table (2026-07-15)

The number #131 exists to produce, published with every caveat that binds it.
Rubric: `2026-07-14-strict-per-turn-table-design.md` (frozen; amendment log
current). Evidence: `2026-07-1{4,5}-arm0-runs/`, `2026-07-15-arm2-runs/`.
All J-tier calls by author-independent scorers, quoted-transcript records
committed per run. Arm 1 (same models via OpenCode Go) is NOT yet run — this
table isolates the serve-vs-frontier-harness comparison, not harness-vs-
harness.

## The headline 2x2 (oracled build turns 1/6/7, aggregated per arm)

| arm | n runs | shipped-correct | shipped-broken | not-shipped | broken_rate (primary) | delivery_rate |
|---|---|---|---|---|---|---|
| Arm 0: serve (qwen3:8b + structural gate, via OpenCode) | 3 | 4 | 1 | 4 | **0.20** (1/5) | 0.44 (4/9) |
| Arm 2a: Claude Code subagent, Haiku 4.5 | 1 | 3 | 0 | 0 | **0.00** (0/3) | 1.00 |
| Arm 2b: Claude Code subagent, Sonnet | 1 | 3 | 0 | 0 | **0.00** (0/3) | 1.00 |

## Strict per-turn score and the honesty column

| arm | strict | dishonest turns | dishonest mechanisms |
|---|---|---|---|
| Arm 0 (runs 2/3/4) | 25/39 (~64%) | **3/39 — one per run, never zero** | recall substitution ×2 (undisclosed rejected first ask, #133); recap fabrication ×1 (phantom function) |
| Arm 2a Haiku | 13/13 | 0/13 | — |
| Arm 2b Sonnet | 13/13 | 0/13 | — |

Cost: Arm 0 is $0 marginal (local). Arm 2 rode the subscription (unmetered
here; Arm 1 via API will carry real per-token cost). Wall-clock: Arm 0
~22–28 min/run on the rig; Arm-2 agent time roughly ~8 min (Haiku) and
~23 min (Sonnet), excluding the driver's capture overhead.

## The honest reading

- **The predicted frontier failure did not materialize at this task
  difficulty.** The §3 thesis was that discretionary verification lets
  plausible-but-wrong code ship; on this ladder both frontier arms shipped
  correct code on every oracled turn (and the shipped-broken artifact in the
  whole table belongs to the SERVE — run 2 turn 7, the #110 class).
- **The serve's differentiator here is not correctness-when-shipping; it is
  cost, and its liability is delivery and honesty.** The gate's honest
  rejects account for the entire strict-score gap, and the serve produced
  exactly one dishonest outcome per run in the recall/recap surface while
  both frontier arms disclosed unprompted.
- **The serve's dishonesty is structurally fixable; the frontier's honesty is
  a rented model property.** #133 is a deterministic disclosure fix. That
  asymmetry — fixable-by-construction vs good-by-disposition — is the
  defensible form of the original thesis, and it is a different claim from
  the one §3 set out to test.
- **A 13/13 ceiling means the ladder no longer discriminates at the top.**
  The battery was built as the serve's dev-loop regression suite (§8.3, OPEN
  threat); the frontier arms saturate it. The next informative comparison
  needs the meta-task realism axis (real-repo work), not more toy runs.

## Addendum 2026-08-12 — the honesty column's fix validated

The table's Arm-0 dishonesty finding (one per run, never zero; #133/#134)
drove a deterministic fix arc (merged 2026-08-12 after five adversarial
review rounds). The post-fix validation run — a different serve version,
NOT poolable into the frozen rows above — scored **11/13 strict, ZERO
dishonest** under the same independent-scorer protocol
(`docs/plans/2026-08-12-arm0-run5/`): turn 10 disclosed a rejected first
ask instead of substituting (runs 2/3's class), turn 5 answered from the
write ledger instead of fabricating (run 4's class). The "fixable by
construction" claim in the honest reading above now has its data point.

## Caveats that bind every row (do not quote the table without them)

1. n=1 per Arm-2 model; n=3 for Arm 0. No variance estimate for Arm 2.
2. Arm-2 DECLARED CONFOUNDS: the subagents inherit the practitioner's
   CLAUDE.md stack (visibly active — unprompted red-green TDD, N+M+1
   citations), the agent sandbox, no permission prompts, and a "coordinator"
   framing absent from Arm 0's client.
3. Scorer affinity: Sonnet subagents scored all arms, including the Sonnet
   arm (that scorer itself recommends a differently-sourced second look
   before its 6/6 is treated as comparative data). The Arm-0 scores carry no
   such caveat.
4. Arm 0 ran behind a DIFFERENT client (OpenCode) than Arm 2 (Claude Code
   subagent) — model, gate, and harness all vary between the arms shown;
   Arm 1 exists to hold the harness constant against Arm 0 and the models
   constant against Arm 2. Until it runs, "harness effect" and "model
   effect" are not separable in this table.
5. §8.3 item-selection threat OPEN: the rungs were added as the serve
   acquired capabilities. Rung provenance is in the battery header.
6. The verification-rate column is ABSENT BY DESIGN (§4: withdrawn as a
   construct asymmetry; the pytest-shim instrument was named but not built).
7. Turn 5 is counted per the rubric's letter for all arms; its
   measures-nothing note was falsified on Arm-0 run 4 (§11 log).
