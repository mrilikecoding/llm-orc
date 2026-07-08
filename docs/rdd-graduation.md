# RDD Graduation Record — Agentic Serving

**Graduated:** 2026-07-08
**RDD plugin version(s) used:** v0.8.7 at Cycle-8 open (corpus migration version 0.8.5; earlier cycles ran earlier 0.8.x versions)
**Scope:** the scoped `docs/agentic-serving/` corpus — 8 cycles, 2026-04 → 2026-07
**Cycle topic at close:** Cycle 8, the declarative-ensemble collapse (the agent *is* a declarative ensemble; the bespoke imperative serving layer deleted)

The full corpus survives on the **`research/agentic-serving-corpus`** branch
(cut at the final Cycle-8 working state). Nothing was destroyed; this branch
carries only the distilled, native-format knowledge.

## What was migrated

| Knowledge | Source | Destination |
|-----------|--------|-------------|
| Serving architecture: pipeline, constraints, invariants, module→code map | `ORIENTATION.md`, `system-design.md` §Cycle 8, `field-guide.md` | `docs/serving.md` |
| Surviving architectural decisions (12 ADRs, numbers preserved) | `docs/agentic-serving/decisions/` | `docs/adrs/serving/` |
| ADR namespace rule (serving vs project-level numbering) | — | `docs/adrs/serving/README.md` |

## What was archived (research branch only)

Everything else under `docs/agentic-serving/`: essays (6) + reflections +
research logs (~60 spike/lit-review records), superseded ADRs (36 of 48),
domain model, scenarios, interaction specs, product discovery, roadmap,
architecture map, benchmark design, field guide, field notes (5 PLAY
sessions), housekeeping (cycle status, ~150 audits, gates), proposals, and
references. Also archived: the repo-root `.rdd/` process state,
`docs/housekeeping/`, `scratch/` spike evidence, and the `spike-omega*`
ensembles under `.llm-orc/ensembles/` (spike-retention boundary = corpus
close, reached here).

## Dangling-reference policy

Code comments citing corpus artifacts were re-pointed where the target
migrated (`system-design.md` → `docs/serving.md`). References to ADR numbers
not present in `docs/adrs/serving/`, and any remaining
`docs/agentic-serving/...` path, resolve on the research branch — they are
historical record, kept deliberately (the pre-graduation scan found ~513
corpus-identifier references; inlining or rewriting all of them was judged
churn without payoff).

## Known open items carried forward (not scaffolding — real work)

- **Conversation memory**: the serve is single-turn by construction (Cycle-8
  PLAY field note #3); threading session-substrate state into the ensemble
  input is a named design question (ADR-046 §3 points at the substrate;
  a Plexus-backed lens is a candidate direction).
- **Capability frontier**: fix / edit-existing / run-tests seats (full
  model-parity target).
- **Accept-gate independence vs a live non-cooperative builder** (ADR-048
  Conditional-Acceptance target) — the adversarial harness is unrun.
