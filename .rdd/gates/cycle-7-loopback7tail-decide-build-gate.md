# Gate Reflection: Cycle 7 (Framework-driven orchestration) decide → build (loop-back #7 tail)

**Date:** 2026-06-10
**Phase boundary:** decide → build (loop-back #7 tail — the ADR-040 completeness-gate scope decision)
**Cycle:** Cycle 7 — Framework-driven orchestration: routing as code

## Belief-mapping question composed for this gate

At the prior ADR-040 DECIDE gate the practitioner rejected the named-file scope as a thin slice
("if it's an essential part of the framework it should be deterministic; I don't want to carve
out thin slices when a real solution is a clear choice"). The belief-mapping question this tail
gate had to answer: **"What would you need to believe for the named-file-only deterministic scope
to be the *right* boundary rather than an arbitrary thin slice?"** The answer was made empirical
through Spike η — you would need to believe the stochastic judge-fallback adequately handles the
unnamed/described path; if it does not, the gate should generalize via a deliverable-enumerator
(arm B), and if it does, the named-file boundary is the principled edge of determinism.

## User's response

Resolved across the session as the evidence arrived, with the practitioner directing the rigor at
each fork:
1. Authorized running Spike η (arm C first — the cheap decisive recall gate).
2. After arm C passed and the isolated probe + clean live baseline showed the judge tracking
   correctly, directed **"try to break the judge on a harder task first"** rather than accept the
   apparent adequacy on one task shape — the decisive rigor call.
3. After the break-the-judge probe located the limits (judge holds on explicit deliverables;
   breaks softly on implicit-margins; breaks hard on compaction, which is structurally
   non-manifest given the sha256-first-message session identity), chose **"accept the judge-fallback,
   close ADR-040 named-file-grounded."**

Earlier in the gate the practitioner also probed the arm-B design directly — asking whether the
relocated stochasticity could be *evaluated and retried* — which sharpened the arm-B record
(the concentrated turn-1 chokepoint is verifiable via consensus, structural rules, and
reconciliation-against-produced, and retriable cheaply and reversibly, unlike the per-turn judge).

## Pedagogical move selected

Probe (belief-mapping on the scope boundary), resolved empirically rather than rhetorically. The
agent surfaced the determinism-principle-versus-evidence tension at each fork and presented the
spike results (including correcting a session-bleed test-harness artifact that briefly looked like
the judge failing), and the practitioner drove the break-the-judge rigor and made the call.

## Commitment gating outputs

**Settled premises (building on these going into BUILD):**
- The bare termination judge is adequate for *described* tasks (isolated 0/12 false-COMPLETE;
  clean live 4/4 converge; held on 8 explicit deliverables). σ's false-COMPLETE was a named-task /
  live-context result that J-3 already takes off the judge.
- The named-file deterministic gate (J-3 + persist-once, ADR-040) is the principled edge of
  determinism: determinism applies exactly where the task names its deliverables (mechanically
  recoverable); the measured-adequate judge covers the described case beyond it.
- ADR-040 is **Accepted, named-file-grounded** — not Conditional, not a provisional thin slice.
- The compaction failure mode is structurally non-manifest here (session identity pins the first
  user message, so the full task persists each turn).

**Open questions (held open going into BUILD):**
- The enumerator's advantage on *vague-implicit* tasks ("production-ready X") is **unproven** —
  arm C validated enumeration on *described* deliverables only; the one place the judge is weakest
  (implicit margins) is the one place arm B's advantage was not measured.
- The form/adequacy gap (ADR-035): existence-completeness is the gate's scope, not coherence — the
  η live runs surfaced a JavaScript CLI for a Python library (intent-divergence) and the σ 8b
  `cli.py` prose-bleed. Coder/form quality, separately owned.
- persist-once under a *real* client compaction event is validated only in simulation.
- The coverage fraction (how much of a real deployment's task space names files) is uncharacterized.
- Latent: the sha256-first-message session identity means two clients sending an identical first
  message would share a session — a real multi-tenancy concern in llm-orc, out of scope here.

**Specific commitments carried forward to BUILD:**
- Arm B (enumerate → deterministic gate, persist-once for compaction immunity) is a recorded,
  buildable option for a future cycle that finds an implicit/compaction-heavy task space where the
  judge demonstrably fails and the enumerator demonstrably beats it. Arm C and the eval/retry
  analysis are its prior art.
- Spike η artifacts (research log + `scratch/spike-eta-deliverable-enumerator/`) retained per the
  corpus-close retention policy.
