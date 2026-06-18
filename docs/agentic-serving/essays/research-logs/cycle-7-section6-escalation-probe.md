# Cycle 7 — §6 Bleed-Injection Escalation Probe (ADR-041 convergence CA)

**Date:** 2026-06-18
**Cost:** $0 (all local; qwen3 family, 32GB rig; opencode 1.15.5)
**Status:** Complete. Live protection + recovery validated under deliberate
adversarial injection; coder-tier escalation did NOT fire (recovery robustness
defeats injection). Live escalate-and-converge stays the ADR-041 convergence-CA
PLAY / organic-deployment item, now with earned justification.
**Harness:** `scratch/spike-pi-escalation-probe/` (probe.py driver + run outputs
+ serve.log), retained per spike-artifact retention. Drives
`benchmarks/agentic_serving/{runner,scorer,corpus}` directly against a dedicated
serve on :8772 (the `bench --probe` path also runs the rig-blocked full grid).

## Purpose

Produce the live escalate-and-converge evidence the ADR-041 convergence
Conditional Acceptance names: the coder-tier escalation lever (ADR-041 §Decision
5) closing a persistent form bleed in a real session. The benchmark §6 design
specifies an adversarial coder + a tier ladder to force the bleed, but the
harness only *marks* the probe cells (`kind="probe"`) and defers the adversarial
coder + tier ladder to operator config — `runner.py` and `bench.py` apply
neither (they run probe cells identically to grid cells). This probe supplies
that config.

## Method

Free-first, local, adversarial coder (practitioner-directed). Temporary,
git-reverted config:

- **Adversarial coder.** `code-generator.yaml` temporarily replaced by a single
  adversarial agent (terminal node = the deliverable per `resolve_deliverable`),
  system prompt: always wrap code in a markdown fence + always append a trailing
  prose sentence, fighting the ADR-035 bare-output dispatch directive.
  Single-agent removes the injection-point and escalation-scope ambiguity of the
  3-agent production coder (whose terminal is the synthesizer, and whose tier
  swap scope was unverified).
- **Seat + ladder.** Local 14b seat (`agentic-orchestrator-offline-tools`); paid
  frontier rung off (free-first). Two ladder configs:
  - Config A — production ladder: cheap 8b -> escalated 14b.
  - Config B — wide §6 gap: cheap 0.6b (temp `agentic-tier-tiny-coder`) ->
    escalated 8b. The §6 design's `0.6B/2B -> 8B` ladder, intended to widen the
    capability gap so the cheap rung bleeds persistently.
- **Cell.** `probe-cli` (single-file argparse CLI; the documented σ/π form-bleed
  zone).
- **Signal.** Loop-driver log (`form recovery:`, `form escalation:`,
  `completeness:`), the scorer MetricRecord, and `ast.parse` of the delivered
  file.

## Runs

| Run | Cheap coder | Bleeds before recovery | Recovery | Escalation | Outcome |
|-----|-------------|------------------------|----------|------------|---------|
| A1  | 8b   | 1 (dispatch)          | recovered, redispatches=1     | none | converged, COMPLETE, valid |
| B1  | 0.6b | 1 (dispatch)          | recovered, redispatches=1     | none | converged, COMPLETE, valid |
| B2  | 0.6b | 2 (dispatch + retry1) | recovered, redispatches=2 (cap) | none | converged, COMPLETE, valid |

All three runs: `form_valid` / `converged` / `content_coherent` /
`terminated_clean` all True; `delegation_rate` 1.0; `churn` 0; **0 invalid files
reached the client**. Wall times 341s (A1) / 481s (B1) / 806s (B2) — the rising
trend is rig load, not a system result.

## Finding

The §6 probe's premise — adversarial injection produces a *persistent* bleed
that exhausts cheap recovery and triggers coder-tier escalation — does not hold
on this stack, because the form-recovery mechanism is too effective. The
adversarial bleed is *intermittent* at every model size tested (8b and 0.6b):
re-sampling at the model's default temperature lands a clean, parseable output
within the 2-retry cap (`_FORM_REDISPATCH_CAP = 2`). Run B2 reached the last
cheap retry (one re-sample short of cap exhaustion) but still recovered.

A stronger adversarial prompt would raise the per-sample bleed rate and could
cross the cap-exhaustion threshold. But the same prompt would also pressure the
escalated rung, so whether escalation *closes* the bleed (convergence) versus
merely fires and exhausts the ladder (protection + a degraded short session) is
then a separate, unresolved question — the tier swap changes the model, not the
ensemble's system prompt.

The load-bearing result: under deliberate adversarial injection, the
deterministic destination-validity gate (protection) and the bounded cheap-tier
recovery (the convergence helper) handled every bleed live, across the 8b–0.6b
range. Coder-tier escalation is a deeper backstop than the natural failure
distribution reaches.

## Disposition (practitioner-directed)

The coder-tier escalation lever stays validated by:

- the `TestLoopDriverFormEscalation` unit tests + the real-gate-through-terminal
  integration test, which inject a deterministic in-process gate to drive the
  escalation code path directly, and
- Spike π Arm E — the MiniMax frontier rung closed a *forced* persistent bleed in
  isolation (n=6).

This probe adds live protection + recovery-robustness evidence under adversarial
conditions. Live escalate-and-converge (a real persistent bleed, escalated to a
more capable rung that closes it) remains the ADR-041 convergence-CA PLAY /
organic-deployment item, now with earned justification: it could not be
manufactured by prompt/model injection because recovery is robust. The §6 probe
as designed cannot produce the escalation evidence; exercising the live
escalation path would require in-process gate injection (what the unit tests do)
or a recovery-cap reduction, not adversarial prompting.
