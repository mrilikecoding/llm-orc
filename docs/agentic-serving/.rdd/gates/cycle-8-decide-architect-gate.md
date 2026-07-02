# Gate Reflection: Cycle 8 — The declarative-ensemble collapse DECIDE → ARCHITECT

**Date:** 2026-07-02
**Phase boundary:** DECIDE → ARCHITECT
**Cycle:** Cycle 8 — The declarative-ensemble collapse

## Belief-mapping question composed for this gate

The DECIDE deliverables this session were the Cycle-8 scenarios + interaction-specs (the five ADRs were accepted in prior sessions). The honest-limits scenarios encode a tension ADR-048 names but does not resolve: under thin acceptance criteria the accept gate degrades to "runs and is non-trivially tested," the ordinary daily-driver turn carries thin criteria, and each gate round adds latency against the Q3 interactive-speed constraint. Question posed (belief-mapping, handed to ARCHITECT): *what would you need to believe for the composed accept gate to belong in the DEFAULT serving shape ARCHITECT designs, rather than as an opt-in that engages only when a turn carries real acceptance criteria?*

## User's response

First pass: "Hm - say more. I would think for thin-criteria we'll have lower-latency models anyway, so we still want the same process." (Position: gate default-on; latency handled by cheaper / thinking-off seats.)

After the agent surfaced two surviving tensions (the deterministic executor is inert on non-build turns; thinking-off cuts per-inference latency but not the round multiplier) and reframed the fork as "gate in the default build-turn shape, criteria scale the ceiling not the presence": "Makes sense. If rounds are short (a few seconds) and well scoped, then I'm fine to go multiple rounds."

On the susceptibility-snapshot finding (that the agent's supporting argument over-read ADR-048 §2 as settling gate *placement* when it settles gate *power*, and conflated a richness-judged opt-in with a deterministic presence-gated opt-in) and the corrected, spike-grounded argument + Grounding Reframe: "Makes sense." Plus a forward note: the eventual clean-main-vs-research-branch reconciliation of the long RDD Cycle-8 corpus is a graduate concern, parked for later.

## Pedagogical move selected

Challenge (belief-mapping), followed by a Grounding Reframe (ADR-059; FF1 pattern per ADR-098) after the isolated susceptibility snapshot returned a specific, actionable, in-cycle finding on the gate-placement synthesis.

## Commitment gating outputs

**Settled premises (building on going into ARCHITECT):**
- The five Cycle-8 ADRs (044 AS-11; 045 clean-slate collapse; 046 target architecture / orchestrator-actor dissolution / classify→seat→marshal; 047 Topaz-keyed registry + shape catalog; 048 composed accept gate) — accepted, argument-audit-converged.
- Full model-parity via composition (DISCOVER): a general per-turn handler, not a build-new-files pipeline.
- Accept gate default-on in the build-turn shape; build-vs-non-build split follows classify's executable-deliverable routing (non-build turns run no gate); multiple accept rounds acceptable when each is short (~seconds) and well-scoped.
- Deferral discipline: current-state doc sweep + `agentic/` code deletion land at BUILD.

**Open questions (holding open going into ARCHITECT):**
- Q5 — removal-execution sequencing + supersession of Cycle-7 serving ADRs 033/036/037/039/041/043 (roadmap/BUILD-facing).
- Round-multiplier bounding (per-round latency budget + ADR-048 §5 open composition rule + judge false-reject rate) — ARCHITECT/BUILD fitness.
- ADR-046 §Open: decider-as-seat swap (unspiked); seat-contract wiring (`core/validation/` as the seat pass/fail gate, designed not wired); AS-3/AS-7 concern re-homing.
- ADR-048 Conditional-Acceptance targets (BUILD/PLAY): live-builder independence (+ artifact-injection channel, builder/judge model-weight correlation); judge reliability; sandboxed execution; unstated-input oracle rung.
- ADR-047 deferred: composer-ensemble path + compose-at-runtime primitive.
- Graduate-time: reconcile a clean `main` merge against a research branch preserving the full RDD Cycle-8 corpus (parked; graduate concern).

**Specific commitments carried forward to ARCHITECT:**
- The gate-placement synthesis (default-on in the build-turn shape) is recorded in `scenarios.md` as gate-conversation synthesis / validation-pending, NOT as ADR-derived content — the corrected note rests on the spike-grounded triviality-floor argument (the judge's `assert True` rejection is builder- and criteria-independent, so a presence-gated opt-in reduces to executor-only, which ADR-048 §Rejected-alternatives rejects).
- **Grounding Reframe logged:** rerun ADR-048's grounding-spike fixtures with criteria withheld/thinned, to measure what the judge-half adds over the executor on thin-criteria turns — owed at BUILD-entry before the gate is wired unconditional (validation-pending). Practitioner accepted the corrected reasoning; the spike is deferred to BUILD-entry unless run sooner.
