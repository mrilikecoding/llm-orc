# Gate Reflection: Cycle 7 (BUILD → RESEARCH loop-back) research → discover

**Date:** 2026-05-30
**Phase boundary:** research (loop from BUILD) → discover (update mode)
**Cycle:** Cycle 7 — Framework-driven orchestration: routing as code (loop-back from BUILD)

## Belief-mapping question composed for this gate

> What would you need to believe for the `tool_calls` terminal to be the right thing for ARCHITECT to center — rather than the layer-A loop-driver? σ showed layer A is a distinct role the pipeline has no component for, and that the terminal is one action *within* a driven loop. If the headline is "add a terminal stage to ADR-027's pipeline," that's a bolt-on. If it's "ADR-027 has no agentic-loop-driver, and that's the gap," that's structural — and it touches whether the single-turn plan→dispatch→synthesize shape survives at all.

Surfaced alongside three framing observations from the argument audit that the audit deliberately did not auto-correct: (A) ADR-027 "incomplete" vs "under-specified for a different problem"; (B) layer A as the primary finding rather than the terminal (the focused question); (C) the scope gap between mechanism-level validation and the long-horizon north-star.

## User's response

Multi-turn exchange. Key responses, verbatim:

1. **Clarifying-question + endorsement:** *"An agentic loop driver seems important, no? What is 'terminal' in this context?"*
2. **Pushback against unflagged extrapolation:** after the agent answered the terminal-vs-loop-driver question and introduced "orchestrator-LLM returns in a grounded seat" as a discussion-shaping framing — *"I'm not sure — what do the spikes we've done indicate?"*
3. **Invitation to judgment:** *"What do you think?"*
4. **Approval of refinement:** *"Yes"* — to the proposed C8 edits (W8.6 + three candidate seat-fillers + grounded-loop hypothesis explicitly flagged as working inference).

## Pedagogical move selected

Challenge — using belief-mapping referenced to specific C8 content (layer-A vs terminal; the single-turn pipeline shape).

## Commitment gating outputs

**Settled premises (the practitioner is building on these going into DISCOVER):**

- C8 added: a client-tool-action terminal is necessary for parity; the justification is the client's *execution model*, not filesystem *geography* (which dissolves under co-location).
- Layer A — a per-turn agentic-loop-driver — is a *role* no current ADR-027 component holds. ADR-027 is *structurally under-specified for the multi-turn agentic behavior the north-star requires*, not merely missing a stage.
- The integrated north-star pattern (cheap layer-A loop-driver + per-turn ensemble delegation) composes at the mechanism level (σ.2; an ensemble-generated test ran and passed).
- F-ρ.1: capability ensembles route deliverables to the server-side artifact store by design (ADR-025); the terminal must read the artifact and marshal it into the client `write` call.
- F-ρ.2 RETRACTED — verification error caught by the isolated citation auditor; the production `code-generator` resolves its cheap tier on the free tier.

**Open questions (held open into DISCOVER):**

- Which seat-filler for layer A — three named candidates: (1) a model in OpenCode's "model" seat (σ-tested); (2) the routing planner extended to per-turn driving; (3) a new dedicated layer-A component. Candidates, not findings. Selection deferred to DECIDE/ARCHITECT.
- Grounded-loop hypothesis (working inference, NOT a spike finding) — the load-bearing discriminating question for candidate (1)'s consistency with AS-9. AS-9's scope reopens specifically around the grounded-vs-ungrounded framing (not the bounded-role pattern generally) if the hypothesis fails.
- Long-horizon driving (dozens of turns, harder tasks, latency compounding) — BUILD-phase validation target, against the real built terminal.
- (Per snapshot advisory) Whether the north-star requires amending ADR-027 or a different architectural model at its center. The "incomplete, not wrong / fix aligned with ADR-027's philosophy" framing entered via cycle-status and was not independently tested against the alternative; the framing also happens to minimize upstream design change. DISCOVER's update to the Skill Orchestration User mental model is the natural place to test this.

**Specific commitments carried forward to DISCOVER (update mode):**

- Make the "delegate work, apply locally" loop explicit in the Tool User and Skill Orchestration User mental models.
- Assumption-inversion on "endpoint-returns-text is sufficient" — settled rejection given C8.
- **Test the architectural framing independently** (snapshot advisory #1): does the north-star require amending ADR-027 or a different architectural model? Do not inherit the "incomplete, not wrong / aligned with ADR-027's philosophy" framing without examination.
- Surface the F-ρ.1 interaction shape (ensemble produces → server reads artifact → terminal marshals → client executes) in the stakeholder mental model update, not only in ARCHITECT's module decomposition (snapshot advisory #3).
- Name an observable discriminating failure for the grounded-loop hypothesis before BUILD runs (snapshot advisory #2).

## Audit / snapshot summary

- Argument audit: 4 rounds. R1: 0 P1, 3 P2, 2 P3 + 3 framing observations (A/B/C surfaced to gate). R2: 0 P1, 0 P2 (all R1 corrections resolved). R3 (post-gate W8.6 addition): 0 P1, 3 P2, 2 P3. R4: **0 P1, 0 P2** (all R3 corrections resolved).
- Citation audit: 2 rounds. R1: 0 P1, 1 P2 (F-ρ.2 false claim — RETRACTED), 2 P3. R2: 0 P1, 0 P2 (retraction verified; one P3 leftover consistency in Amendment B — also fixed).
- Susceptibility snapshot: **No Grounding Reframe.** Three advisories carried forward (architectural-framing test; grounded-loop discriminating failure; F-ρ.1 in stakeholder mental model).
- The most significant epistemic event was practitioner reflection catching agent over-extrapolation at gate-time (the "orchestrator-LLM returns in a grounded seat" framing), resulting in the post-gate W8.6 refinement that records seat-filler candidates and the grounded-loop hypothesis with load-bearing scope flags. Captured in reflections file `essays/reflections/006b-client-tool-action-terminal.md`.
