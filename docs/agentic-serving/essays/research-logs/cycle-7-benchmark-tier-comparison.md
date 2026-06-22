# Cycle 7 — Agentic-Serving Benchmark: tier comparison + horizon ceiling

**Date:** 2026-06-22
**Phase:** BUILD/PLAY-prep (the benchmark is the experiential-evidence harness PLAY consumes)
**Spec:** `docs/agentic-serving/benchmark-design.md`
**Scorecard:** `benchmark-runs/tier-comparison/scorecard-final.md`

## Purpose

Answer the cycle's load-bearing question empirically: does a cheap-orchestrator
stack plus the framework's orchestration (`[qwen3.6-plus seat + local 8b coder +
escalation] + loop/delegation/termination`) match an expensive frontier model
with no orchestration (`[Claude Sonnet, one-shot]`) on structural reliability of
multi-file code generation? The construct is the value-proposition reading
named in the central question — not single-vs-orchestration generically.

## Setup

- **Cheap arm:** the Spike-τ working config — hosted `qwen3.6-plus` seat (now via
  the **OpenCode Go** subscription endpoint `/zen/go/v1`, not pay-as-you-go), local
  `qwen3:8b` coder, `8b→14b→MiniMax` coder-tier escalation. Driven through the
  real **OpenCode** client (1.17.9) against `llm-orc serve`. Cents/session.
- **Frontier arm:** one Claude Sonnet subagent per cell, one-shot, no framework,
  writing deliverables into a per-cell workspace; scored on the file-derived
  signals only (termination is N/A to a one-shot model).
- **Scoring (deterministic, $0):** form validity (`ast.parse` / `json.loads`),
  convergence (all expected files produced), content coherence (dependent files
  reference real sibling APIs — AST cross-reference), termination (the loop
  reaches a clean COMPLETE finish — cheap arm only).
- **Harness deltas this run:** per-cell ollama restart (graduated from the
  `scratch/benchmark-grid-run` phased driver — the Spike-τ flat-latency method),
  §3 sweep-cell selection, scaled per-cell timeout, per-cell logging.

## Results

### Complexity axis (H3 = 5 files, C1 → C4): cheap MATCHES frontier

All four complexity rungs converge form-valid, complete, and coherent on both
arms. The cheap arm is **reliable** here: `h3c3` and `h3c4` each passed **3/3** on
a fresh-environment re-run.

An earlier reading flagged `h3c4` as "flaky at the C4 edge" after it timed out
0/3 in a confirmation pass. That was **refuted**: the timeouts were a *late-run
degradation artifact* (they occurred ~2.5 h into a continuous run, where cumulative
memory pressure slowed the rig). On a fresh environment `h3c4` converges in ~6 min
(3/3) and `h3c3` in ~16–22 min (3/3). `h3c3`'s original confirm timeout was the
23-min cap cutting off a genuinely slow-but-converging run, not non-convergence.

### Horizon axis (file count, fixed low complexity): cheap DIVERGES at ~15 files

| cell | files | cheap convergence | cheap coherence | frontier |
|------|-------|-------------------|-----------------|----------|
| l12  | 12    | ✓                 | ✓ clean         | ✓ |
| l15  | 15    | ✓ (15/15)         | ✗ (0/3)         | ✓ |
| l20  | 20    | ✓ (20/20, 37–52 min) | ✗ (0/3)      | ✓ (~56 s) |

The cheap arm's **convergence holds to 20 files** — given a generous cap it
produces every file and terminates cleanly. Its **coherence ceiling is ~12 files**:
at 15+ it loses cross-reference fidelity. The specific, reproducible failure is a
wrong import *source* — e.g. `step6.py` writes `from step1 import f5` instead of
`from step5 import f5`; the function name is right but the module is wrong, so the
reference is to a function that does not exist there. Consistent at 0/3 for both
l15 and l20, on a fresh environment — a genuine **capability** ceiling, not
degradation, sampling, or cap. The frontier holds the full import graph straight
to 20 files.

### Two-arm verdict

**Complexity gap 0, horizon gap 2** → MATCH on complexity, NO MATCH on horizon.

The cheap orchestrated stack matches the frontier on **bounded multi-file tasks**
(every complexity rung at 5 files; clean up to ~12 files) at cents/Go-subscription
cost. Beyond ~12 files it diverges on two separable axes: **cross-reference
coherence** (a capability ceiling at ~15 files) and **speed** (l20 at 37–52 min vs
the frontier's ~56 s). Convergence and termination — the framework's reliability
machinery — hold throughout; the cheap *model's* coherence and speed are the
ceiling, not the framework.

## Methodological notes — real-client validation earned three fixes

Each of these was invisible to the harness tests and surfaced only against the
real OpenCode client (the recurring "validate against the real client" lesson):

1. **Marathon degradation.** A continuous run degrades after ~95 min on the 32 GB
   rig (memory-thrashing the 8b↔14b swap). Per-cell ollama restart holds latency
   flat and was what exonerated the `h3c4`/`h3c3` "flakiness".
2. **OpenCode CLI/app version skew.** Mid-session the OpenCode desktop app (1.17.9)
   migrated the shared session SQLite DB to a schema with a `NOT NULL session_message.seq`
   column; the CLI (1.15.5) then failed every run with a constraint violation
   surfaced as an opaque "Unexpected server error". Resolved by upgrading the CLI
   to 1.17.9.
3. **Scorer termination fragility.** OpenCode 1.17.9 fires a trailing *toolless* aux
   request (the title generator) whose own turn-decision lands last in the serve-log
   slice. The scorer's `_terminated_clean` had read the *last* decision, so it
   mis-scored a cleanly-completed task as a zombie. Fixed to detect a clean COMPLETE
   finish *anywhere* in the slice (robust to trailing aux requests; capped/zombie
   loops, which never reach COMPLETE, still score False).

The seat was also moved from pay-as-you-go Zen (`/zen/v1`) to the OpenCode Go
subscription endpoint (`/zen/go/v1`) — the same `qwen3.6-plus` model, flat
`$12/5h · $30/wk · $60/mo` limits instead of per-request billing.

## Artifacts

- Scorecard: `benchmark-runs/tier-comparison/scorecard-final.{md,json}`
- Frontier workspaces (8 Sonnet cells): `benchmark-runs/frontier/`
- Cheap flaky-cell retry (h3c4/h3c3 → 3/3): `scratch/benchmark-flaky-retry/` +
  `benchmark-runs/flaky-retry/`
- Cheap horizon retry (l15/l20 → 0/3 coherence): `scratch/benchmark-horizon-retry/`
  + `benchmark-runs/horizon-retry/`
