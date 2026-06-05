# Gate Reflection: Cycle 7 loop-back #4 DECIDE → BUILD

**Date:** 2026-06-04
**Phase boundary:** decide (loop-back #4 — swappability fork) → build (WP-LB-I resume)
**Cycle:** Cycle 7 — Framework-driven orchestration: routing as code

## Belief-mapping question composed for this gate

The gate's primary composed question was commitment-gating, referenced to
the spike verdict and the practitioner's own scoping principle: a proposed
settled/open split — settled: ADR-036 stands / WP-LB-I resumes / no
ADR-037; paid-in-bounded-slots-with-local-path as a recorded design
constraint; the ω.3b finding (prompt-position demotion is not a structural
guarantee) — versus open: swappability (Cycle 8 candidate);
verifier-on-disagreement deferral; fully-data composition variant
(unspiked); hosted slots as future DECIDE material — with the invitation
to move items between lists.

Alternatives engagement during the gate was practitioner-initiated: the
hosted-viability question ("in the version where we use a MiniMax for
example, is this path viable?") received a measured-data answer (solves
capability/latency/residency; does not solve ω.3b's authority leak; kills
the $0-offline property; best entry shape is the verifier slot). The
susceptibility snapshot flagged that the verifier-slot framing was
agent-originated and the hosted-broker-as-primary alternative did not
receive full criterion-level treatment — carried as BUILD Advisory B.

## User's response

To the hosted-variant analysis and recording proposal: "Yes we could
record that. I'm not opposed to certain paid strategies if they are
limited in scope and make the rest of the ensemble pipeline viable. But
having a local option is important as well in case I don't want to pay."

To the settled/open split: "Makes sense"

Earlier in the phase the practitioner directed the tier widening ("We
could also consider next-tier (7-9B param models)"), authorized the paid
reference arm ("I would also authorize a small budget for testing against
Zen OpenCode paid models"), and raised then self-deferred the
verifier-on-disagreement design question ("Probably a test for a
different time").

## Pedagogical move selected

Commitment gating (with the practitioner's own hosted-variant question
serving as the gate's alternatives-engagement). The agent presented the
six-criteria evaluation with criterion 2 (swappability — the fork's
motivation) explicitly unresolved for both proposals rather than scored
for the incumbent.

## Commitment gating outputs

**Settled premises (building on these going into BUILD):**
- ADR-036 stands; WP-LB-I resumes; no candidate ADR-037 (Spike ω refuted
  Proposal B as specified on four independent pre-registered grounds)
- Paid components acceptable in bounded, value-concentrated slots that
  enable the pipeline; every paid slot must have a local degradation
  path; $0-local operation remains first-class (practitioner principle,
  recorded in the spike log as a DECIDE-artifact design constraint)
- ω.3b finding: fenced/labeled data position does not remove instruction
  authority; "structural" means deterministic framework code, not prompt
  position; data-position discipline anti-correlated with post-training
  recency on this evidence (H1 inverted)

**Open questions (held open going into BUILD):**
- Swappability — unresolved for BOTH proposals; lives in the candidate
  Cycle 8 (transferability); BUILD evidence feeds it without resolving it
  for A (snapshot Advisory A)
- Verifier-on-disagreement — named deferral with trigger sizing from
  recorded ω data (≈20% fire rate, ≈86% error coverage on qwen3.5:9b)
- Fully-data composition variant — unspiked; the natural probe if a
  B-shape returns
- Hosted slots (verifier / fallback / hosted-broker-as-primary) — future
  DECIDE material; the primary-architecture alternative requires full
  criterion-level treatment before adoption (snapshot Advisory B)

**Specific commitments carried forward to BUILD:**
- Resume WP-LB-I (V3 composition + F-4 docstring) per roadmap; WP-LB-J
  fork-neutral, open-choice ordering with I; TS-15 after both; WP-LB-E
  resumes after
- Acceptance gates unchanged: suite green (baseline 2914) + $0
  real-OpenCode run with delegation VERIFIED fired (serve-log `dispatch
  start` / `TurnDecision`)
- Loop-back #3 architect-snapshot advisories remain active (A: meter
  co-required for TS-15; B: transferability stays modal; C: verify the
  F-fold preserves FC-51 axis-2 intent at WP-LB-J entry)
- ω.3b finding is a structural input to any future design placing
  instruction-shaped content in a data region (snapshot Advisory C)
