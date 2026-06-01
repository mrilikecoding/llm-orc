# Gate Reflection: Cycle 7 (BUILD → RESEARCH → DISCOVER loop-back) discover → model

**Date:** 2026-06-01
**Phase boundary:** discover (update mode, loop from RESEARCH re-entry) → model
**Cycle:** Cycle 7 — Framework-driven orchestration: routing as code (loop-back from BUILD; client-tool-action terminal)

## Belief-mapping question composed for this gate

> We deliberately left the wrapper-vs-callee fork open, and that is what DECIDE inherits, so I want to test whether it is genuinely open or quietly settled. **What would you need to believe for the *callee* reading to be the right one** — that the layer-A loop-driver is the center and ADR-027's plan→dispatch→synthesize pipeline is just something the loop calls (or, per σ.2, not the per-turn primitive at all)? And the sharper half: **what evidence at DECIDE or ARCHITECT would actually tip you toward callee over wrapper?** If the honest answer is "I'd want it to be wrapper because that's less rework," that's worth saying plainly — that's the pull the loop-back snapshot flagged.

Composed against Tension 21 (the wrapper-vs-callee fork) and the load-bearing commitment #3 (test the "incomplete, not wrong" framing independently rather than inherit it). The question targets the specific artifact content (the two dispositions in Tension 21 + the Skill Orchestration User loop-back refinement), not the practitioner's prior engagement.

## User's response

Multi-exchange gate. Key responses, verbatim:

1. On the agent's independent "middle" framing (ADR-027 sound for layer-B generation but mis-identified as *the* architecture; wrapper-vs-callee is the real open question): *"That frame makes sense. Perhaps I made it in terms too binary - but the spike seemed to yield good results which is what is important to me."*
2. On whether to run a long-horizon spike now vs. proceed: *"I'm reasonably convinced about the grounded-loop, so let's proceed."*
3. To the wrapper-vs-callee belief-mapping question: *"At this point I have not preconceived notion. I am focused on outcome and I want the effective path to win, which we need to ground with evidence."*
4. To the solution-scoping + settled/open commitment-gating split: *"Looks good."*

## Pedagogical move selected

Challenge — belief-mapping referenced to specific Tension 21 content (the wrapper and callee dispositions; the "what evidence would tip you" second half). Followed by an agent-surfaced **evidence-availability asymmetry** (the only multi-turn evidence — σ.2 — is callee-shaped; no wrapper-as-multi-turn-loop was ever spiked), which converted the practitioner's "let evidence decide" into a concrete DECIDE-entry deliverable (a wrapper-shaped probe alongside the OQ #19-style comparison) rather than a slogan that would settle by callee-skew default.

## Commitment gating outputs

**Settled premises (the practitioner is building on these going into MODEL):**

- Text-only terminal is rejected; the endpoint must emit `tool_calls` and participate in the client's multi-turn loop (Spike π; settled rejection, not held open). Justification is the client's *execution model*, not filesystem geography.
- Parity is the bar — behavioral / experiential, NOT latency (local models are inherently slower; accepted tradeoff).
- Layer A (per-turn agentic-loop-driver) is a real role no ADR-027 component holds; layer B (per-turn generation) is what the pipeline does.
- The artifact-bridge (F-ρ.1) is a designed-in requirement (production ensembles route deliverables server-side per ADR-025); it survives co-location.
- The loop-back spikes validated the **mechanisms at n=1**, not the architecture (wrapper-vs-callee) and not sustained long-horizon driving.

**Open questions (held open into DECIDE / ARCHITECT):**

- Wrapper vs. callee (no preconceived notion; to be decided by effectiveness grounded in evidence — and the evidence plan must cover *both* shapes, not only the callee-shaped one σ.2 produced).
- Layer-A seat-filler (three candidates: model in OpenCode's "model" seat; routing-planner extended to per-turn driving; dedicated component).
- Grounded-loop hypothesis (practitioner prior: reasonably solid; falsification probe queued for DECIDE entry; ADR-097 Conditional Acceptance as backstop).
- Sustained long-horizon driving (BUILD-phase validation target, against the real built terminal).

**Specific commitments carried forward to MODEL / DECIDE:**

- MODEL: fold the six new vocabulary terms into `domain-model.md`; do NOT promote the three DECIDE-pending terms (layer-A loop-driver, layer-B generation, grounded loop) to settled domain vocabulary; carry the grounded-loop hypothesis's conditional reopening of AS-9's scope (grounded-vs-ungrounded driving) as an explicit AS-9 scope note rather than silently absorbing it.
- DECIDE-entry: a wrapper-vs-callee comparison parallel to OQ #19 (build-complexity), with named discriminating criteria (reliability / grounded-loop probe; build-complexity; latency-over-turns; operator-mental-model fit) and an evidence plan that includes at least one **wrapper-shaped probe** to avoid settling by callee-skew. The grounded-loop falsification probe runs at DECIDE entry against a harder/longer task than σ.2's.

## Audit / snapshot summary

- Susceptibility snapshot (`housekeeping/audits/susceptibility-snapshot-cycle-7-loopback-discover.md`): **No Grounding Reframe.** All three loop-back-RESEARCH carry-forward advisories honored substantively. Rapid-compounding signature (from the first DISCOVER snapshot) absent. Four carry-forward advisories: (1) MODEL vocabulary disposition; (2) DECIDE wrapper-shaped-probe evidence plan; (3) DECIDE σ.2 callee-shape weight; (4) DECIDE grounded-loop falsification probe on a harder task. Advisories 2+3 were partially addressed in-cycle by rebalancing Tension 21 (callee disposition expanded to b1/b2; σ.2 pulled from parenthetical to named evidentiary weight; evidence-availability asymmetry + wrapper-shaped-probe note added; default-pull made symmetric).
- The most significant epistemic event was the practitioner holding the wrapper-vs-callee fork genuinely open ("no preconceived notion... the effective path to win, grounded with evidence"), which the agent then protected from a hidden callee-skew by surfacing the evidence-availability asymmetry.
